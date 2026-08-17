"""Worker-side LP / latency-benefit logic for SCULPTOR.

`Path_Distribution_Computer` holds the per-worker LP machinery: a
persistent Gurobi model (built once at init via `init_persistent_lp`),
an LP cache, monte-carlo routing probabilities, and the `latency_benefit`
function the driver calls thousands of times per gradient probe.

In production this class is constructed as a Ray actor via
`path_distribution_computer_ray._LocalPathDistributionComputer`
(see that file for the Ray dispatch surface). The base `__init__` here
raises in non-debug mode — direct subprocess instantiation is gone with
the ZMQ removal (May 2026).

Key methods used by the driver:
  - `latency_benefit(adv, **kw)`  monte-carlo estimate, hot path during
                                   gradient probes
  - `solve_generic_lp_persistent` reuse the Gurobi shell for one LP
                                   evaluation (gradient probes)
  - `update_deployment(dep)`      re-shard when the deployment changes
                                   (per-dpsize boundary or adaptive resize)
  - `dump_mem_log()`              return this worker's [mem-worker] file
                                   content (driver collects at end of run)

Unit-test instantiation: pass `debug=True` to skip the run guard.
"""
import numpy as np, pickle, copy, time, random, os
from collections import defaultdict
random.seed(31415)
from constants import *
from helpers import *
from test_polyphase import *
from optimal_adv_wrapper import Optimal_Adv_Wrapper

from solve_lp_assignment import solve_generic_lp_with_failure_catch, get_paths_by_ug, NO_PATH_INGRESS

import gpshim as gp  # gurobipy-subset facade; SCULPTOR_LP_BACKEND=gurobi(default)|highs
from scipy.sparse import csr_matrix

gp.setParam("OutputFlag", 0)


# Per-worker mem-log file path cache. Each worker's _log_mem_worker
# appends here in addition to stdout. The driver collects via the
# dump_mem_log RPC at end of run -- needed because Ray's log_to_driver
# silently drops worker stdout in some configurations (observed empirically:
# 0 [mem-worker] lines reached the boost2532 sweep log despite hooks firing
# in path_distribution_computer.update_deployment).
_WORKER_MEM_LOG_PATHS = {}


def _get_worker_mem_log_path(worker_i):
	if worker_i not in _WORKER_MEM_LOG_PATHS:
		log_dir = os.environ.get('SCULPTOR_WORKER_MEM_LOG_DIR', '/tmp')
		_WORKER_MEM_LOG_PATHS[worker_i] = os.path.join(
			log_dir, 'sculptor_worker_{}_{}.log'.format(worker_i, os.getpid()))
	return _WORKER_MEM_LOG_PATHS[worker_i]


def _log_mem_worker(worker_i, tag, **extra):
	"""Worker-side memory snapshot. Writes to a per-worker file (collected
	by the driver via dump_mem_log RPC at end of run) AND prints to stdout
	(best-effort; under Ray, stdout often doesn't reach the driver -- the
	file is the authoritative copy).

	Output format matches helpers.log_mem so parse_mem.py treats driver
	and worker lines uniformly: `[mem-worker idx=N] tag=TAG rss_mb=R ...`.
	"""
	if os.environ.get('SCULPTOR_LOG_MEM', '1') == '0':
		return
	rss_kb = vms_kb = peak_kb = sys_avail_kb = -1
	try:
		with open('/proc/self/status', 'r') as f:
			for line in f:
				if line.startswith('VmRSS:'):    rss_kb  = int(line.split()[1])
				elif line.startswith('VmSize:'): vms_kb  = int(line.split()[1])
				elif line.startswith('VmPeak:'): peak_kb = int(line.split()[1])
		with open('/proc/meminfo', 'r') as f:
			for line in f:
				if line.startswith('MemAvailable:'):
					sys_avail_kb = int(line.split()[1]); break
	except (FileNotFoundError, PermissionError):
		return
	bits = ['tag={}'.format(tag),
	        'rss_mb={}'.format(rss_kb//1024),
	        'vms_mb={}'.format(vms_kb//1024),
	        'peak_mb={}'.format(peak_kb//1024),
	        'sys_avail_mb={}'.format(sys_avail_kb//1024),
	        'pid={}'.format(os.getpid()),
	        't={:.2f}'.format(time.time())]
	for k, v in extra.items():
		bits.append('{}={}'.format(k, v))
	line = '[mem-worker idx={}] '.format(worker_i) + ' '.join(bits)
	print(line, flush=True)
	try:
		with open(_get_worker_mem_log_path(worker_i), 'a') as f:
			f.write(line + '\n')
	except Exception:
		pass


remeasure_a = None
try:
	remeasure_a = pickle.load(open('remeasure_a.pkl','rb'))
except:
	pass

USER_OF_INTEREST = None

## TODO -- just remove the old version once we're more confident it works well
TEST_BETTER_VERSION = True

# dont make this too big or you'll break the VM
# 2000 for lots of workers / smaller VMs. 15000 for fewer workers / large VMs
MAX_CACHE_SIZE = 8000

PROB_TOLERANCE = .05 ## if probabilities differ by more than this much, we have to recalculate things

def get_a_cache_rep(a):
	tups = []
	a = threshold_a(a)
	for x,y in zip(*np.where(a)):
		tups.append((x,y))
	return tuple(sorted(tups))

class Path_Distribution_Computer(Optimal_Adv_Wrapper):
	"""Base class holding the per-worker LP / latency-benefit logic.

	Constructed in production only via the Ray actor subclass
	(path_distribution_computer_ray._LocalPathDistributionComputer), which
	calls Optimal_Adv_Wrapper.__init__ directly with the subdeployment +
	init_kwa it was handed by Worker_Manager. The base __init__ below
	only sets the cheap per-instance state and is callable in debug mode
	for unit tests that exercise the LP methods in isolation.
	"""
	def __init__(self, worker_i, base_port, **kwargs):
		self.worker_i = worker_i
		self.port = base_port
		self.logging_iter = 0
		self.timing = { k:0 for k in ['solve_unified_lp_not_optimize', 'optimize', 'get_paths_by_ug','organizing_results',
		'get_ingress_probabilities_by_dict_generic', 'sim_rti', 'total_rti_calc', 'pmat_organize',
		'solve_generic_lp_persistent', 'solve_generic_lp_not_persistent']}
		self.rti_data = {}

		# SCULPTOR_MC_NUM: monte carlo simulations to determine distributions
		# (default 5, the original hardcoded value; 1 = single-draw noisy
		# estimator, for model-uncertainty experiments)
		self.MC_NUM = int(os.environ.get('SCULPTOR_MC_NUM', '5'))

		if kwargs.get('debug', False):
			self.n_prefixes = None
			return
		# Non-debug instantiation of this class was historically the worker
		# subprocess's entry point (ZMQ socket bind + handshake + run-loop).
		# That path is gone. Production callers go through
		# path_distribution_computer_ray.Path_Distribution_Computer.remote(...)
		# via worker_comms.Worker_Manager; the Ray subclass calls
		# Optimal_Adv_Wrapper.__init__ directly and never hits this branch.
		raise RuntimeError(
			"Path_Distribution_Computer must be constructed via the Ray actor "
			"(use worker_comms.Worker_Manager). Pass debug=True for unit tests.")

	def summarize_timing(self):
		# Print per-key cumulative LP-solve timings (optimize / get_paths_by_ug /
		# organizing_results / solve_unified_lp_not_optimize, etc.). Called every
		# 50 latency_benefit calls in the worker batch loop. Useful for
		# identifying which sub-step inside the LP solve dominates at scale.
		total_time = sum(list(self.timing.values()))
		if total_time < 1e-6:
			return
		print("\n===============\nWorker {} timing summary (cumulative)".format(self.worker_i))
		for k in sorted(list(self.timing), key=lambda el: -self.timing[el]):
			pct = round(self.timing[k] * 100.0 / total_time, 2)
			print("  {:<40s} {:>6.2f}%  ({:.1f} ms)".format(k, pct, self.timing[k] * 1000))
		print("==================\n")

	def init_persistent_lp(self):
		"""Sets up the persistent Gurobi shell with static Volumes and Capacities."""
		self.model = gp.Model(f"Worker_{self.worker_i}_Persistent")
		self.model.Params.LogToConsole = 0
		self.model.Params.Method = 1
		self.model.Params.Threads = 1
		# SCULPTOR_GRB_DUMP=<dir>: emit a Gurobi log per worker and dump the
		# first ~3 solves as .mps for Gurobi support. Only worker 0 dumps to
		# keep artifact count small.
		self._grb_dump_dir = os.environ.get('SCULPTOR_GRB_DUMP')
		self._grb_dump_count = 0
		if self._grb_dump_dir and self.worker_i == 0:
			os.makedirs(self._grb_dump_dir, exist_ok=True)
			self.model.Params.LogFile = os.path.join(
				self._grb_dump_dir, f'sculptor_{self.dpsize}_w{self.worker_i}.log')

		# 1. Permanent Dummy Variable for MLU (Y)
		self.mlu_dummy = self.model.addVar(lb=0.0, obj=0.0, name="mlu_Y")

		# 2. Permanent Rows (Constraints)
		self.vol_constrs = {}
		for ugi, ug in enumerate(self.whole_deployment_ugs):
			# IMPORTANT: Gurobi addConstr needs an expression. 
			# Since we haven't added columns yet, we set LHS to 0.0 explicitly
			target_vol = float(self.whole_deployment_ug_vols[ugi])
			
			# We use a Linear Expression placeholder (0.0) 
			# so Gurobi knows this is a constraint to be filled later.
			self.vol_constrs[ug] = self.model.addLConstr(0.0, gp.GRB.EQUAL, target_vol, name=f"vol_{ug}")

		# SCULPTOR_CAPACITY_HEADROOM is gated on self._in_training (set via the
		# 'set_training_mode' RPC from sas.solve()), so the eval phase sees full
		# capacities. Cache the unscaled caps and start in eval mode (in_training
		# = False); the driver flips us into training mode before its gradient
		# loop.
		self._in_training = False
		self._link_capacities_full = np.concatenate([self.link_capacities_arr.flatten(), [1000000.0]])
		self.static_caps = self._compute_static_caps()
		self.cap_constrs = {}
		for pi in range(len(self.static_caps)):
			target_cap = float(self.static_caps[pi])
			# sum(paths) <= cap
			self.cap_constrs[pi] = self.model.addLConstr(0.0, gp.GRB.LESS_EQUAL, target_cap, name=f"cap_{pi}")

		self.var_pool = {} # Key: (ug, poppi) -> Gurobi Var Object

	def _compute_static_caps(self):
		# Headroom is gated on _in_training so the eval phase always sees full
		# capacities. Pin the [..., 1000000.0] "no-route" sentinel; it's an
		# overflow sink, not a real link.
		if self._in_training:
			h = float(os.environ.get('SCULPTOR_CAPACITY_HEADROOM', '0'))
			scaled = self._link_capacities_full[:-1] * (1.0 - h)
			return np.concatenate([scaled, self._link_capacities_full[-1:]])
		return self._link_capacities_full.copy()

	def set_training_mode(self, in_training):
		# RPC entrypoint: driver flips this around the gradient loop. When mode
		# changes we rebuild static_caps and push new RHS values into the
		# persistent Gurobi cap constraints; subsequent solve_unified_lp calls
		# read self.static_caps directly when restoring RHS, so this is the only
		# place capacity values get refreshed.
		if bool(in_training) == bool(self._in_training):
			return
		self._in_training = bool(in_training)
		if not hasattr(self, 'model'):
			return  # persistent LP not built yet; init_persistent_lp will pick up the flag
		self.static_caps = self._compute_static_caps()
		for pi, constr in self.cap_constrs.items():
			constr.RHS = float(self.static_caps[pi])

	def solve_unified_lp(self, available_paths, obj_coeffs, using_mlu=False):
		"""Core solve logic. Toggles between Standard and MLU."""
		# 1. Deactivate all path variables
		ts = time.time()
		all_vars = list(self.var_pool.values())
		self.model.setAttr("UB", all_vars, [0.0] * len(all_vars))

		# 2. Configure MLU variable and Capacity RHS
		if using_mlu:
			self.mlu_dummy.Obj = 1.0 / ALPHA
			self.mlu_dummy.UB = gp.GRB.INFINITY
			for pi, constr in self.cap_constrs.items():
				self.model.chgCoeff(constr, self.mlu_dummy, -1.0 * self.static_caps[pi])
				constr.RHS = 0.0 # MLU mode uses RHS of 0
		else:
			self.mlu_dummy.Obj = 0.0
			self.mlu_dummy.UB = 0.0
			for pi, constr in self.cap_constrs.items():
				self.model.chgCoeff(constr, self.mlu_dummy, 0.0)
				constr.RHS = self.static_caps[pi] # Restore static capacity

		# 3. Activate/Discover Columns (Volume RHS is already static)
		for (ug, poppi), latency in zip(available_paths, obj_coeffs):
			key = (ug, poppi)
			if key not in self.var_pool:
				col = gp.Column()
				col.addTerms(1.0, self.vol_constrs[ug])
				col.addTerms(1.0, self.cap_constrs[poppi])
				self.var_pool[key] = self.model.addVar(lb=0.0, obj=latency, column=col)

		active_vars = [self.var_pool[ug, poppi] for ug, poppi in available_paths]
		self.model.setAttr("UB", active_vars, [gp.GRB.INFINITY] * len(active_vars))
		self.model.setAttr("Obj", active_vars, obj_coeffs)
		# Stash for solve_generic_lp_persistent's raw_x extraction:
		# iterating self.var_pool there (which can hit 100k-1M entries
		# in production) and doing a Gurobi `.X` API call per var costs
		# tens of ms per LP solve at scale. Instead, batch-read X from
		# the small active set using getAttr("X", active_vars) — one
		# C-call, results in a numpy/list of len(active_vars).
		self._last_active_paths = available_paths
		self._last_active_vars = active_vars

		self.timing['solve_unified_lp_not_optimize'] += time.time() - ts
		ts = time.time()
		if self._grb_dump_dir and self.worker_i == 0 and self._grb_dump_count < 3:
			mps_path = os.path.join(
				self._grb_dump_dir,
				f'sculptor_{self.dpsize}_w{self.worker_i}_solve{self._grb_dump_count}.mps')
			self.model.write(mps_path)
			self._grb_dump_count += 1
		self.model.optimize()
		self.timing['optimize'] += time.time() - ts
		if self.model.status == 2:
			return self.model
		return None

	def _path_obj_coeffs(self, available_paths, obj, site_cost_alpha):
		"""Per-path LP objective coefficients (latencies). Named sub-step of
		solve_generic_lp_persistent so subclasses can override path pricing."""
		obj_coeffs = []
		for ug, poppi in available_paths:
			if poppi == NO_PATH_INGRESS(self):
				obj_coeffs.append(NO_ROUTE_LATENCY)
			else:
				if obj == "avg_latency":
					obj_coeffs.append(self.whole_deployment_ug_perfs[ug][self.popps[poppi]])
				elif obj == "per_site_cost":
					pop, _ = self.popps[poppi]
					site_cost = self.site_costs[pop]
					obj_coeffs.append(self.whole_deployment_ug_perfs[ug][self.popps[poppi]] + site_cost_alpha * site_cost)
				else:
					raise ValueError("obj {} not supported in solve_generic_lp_persistent".format(obj))
		return obj_coeffs

	def solve_generic_lp_persistent(self, routed_through_ingress, obj, **kwargs):
		"""The high-level wrapper that tries Standard first, then MLU."""
		ts = time.time()
		available_paths, _ = get_paths_by_ug(self, routed_through_ingress)
		self.timing['get_paths_by_ug'] += time.time() - ts

		# Pre-calculate objective (latencies)
		site_cost_alpha = kwargs.get('site_cost_alpha', DEFAULT_SITE_COST)
		obj_coeffs = self._path_obj_coeffs(available_paths, obj, site_cost_alpha)

		# 1. Try Standard Solve
		model_res = self.solve_unified_lp(available_paths, obj_coeffs, using_mlu=False)

		# 2. Fallback to MLU if Standard is Infeasible
		if model_res is None:
			model_res = self.solve_unified_lp(available_paths, obj_coeffs, using_mlu=True)

		if model_res is None:
			# LP is infeasible (e.g. failure scenario removed the last route
			# for some users). Used to call exit(0), which killed the Ray
			# actor and propagated as RayActorError -> caught silently by
			# the per-strategy try/except in eval_latency_failure ->
			# strategy's failure-mode fields stayed empty for every
			# strategy. Suspected root cause of the actual-32 failure-eval
			# data loss called out in HANDOFF_SESSION_6.md.
			#
			# Return a "no-route" sentinel: every UG gets NO_ROUTE_LATENCY,
			# fraction_congested_volume=1.0, empty path mappings. Downstream
			# consumers (assess_failure_resilience, flash-crowd, diurnal)
			# already treat NO_ROUTE_LATENCY as the infeasible signal.
			print("Infeasible problem, returning no-route sentinel (was exit(0))")
			n_ug = self.whole_deployment_n_ug
			return {
				"objective": NO_ROUTE_LATENCY,
				"raw_solution": {},
				"paths_by_ug": {},
				"lats_by_ug": np.full(n_ug, NO_ROUTE_LATENCY, dtype=float),
				"available_paths": [],
				"solved": False,
				"vols_by_poppi": {pi: 0.0 for pi in range(len(self.static_caps))},
				"fraction_congested_volume": 1.0,
			}

		## Distribution is the amount of volume (not percent) placed on each path
		## a path is specified by a <user, popp>
		# Two implementations, gated by SCULPTOR_DISABLE_RAW_X_BATCH so the
		# bench harness can compare them in one pytest invocation. Default
		# (unset): batched getAttr("X", active_vars) — one Gurobi C-call
		# on the small active set. Set: legacy per-var .X loop over the
		# entire var_pool (which grows to 100k-2M entries at scale).
		if os.environ.get('SCULPTOR_DISABLE_RAW_X_BATCH'):
			raw_x = {key: var.X for key, var in self.var_pool.items() if var.X > 1e-7}
		else:
			x_vals = self.model.getAttr("X", self._last_active_vars)
			raw_x = {key: x for key, x in zip(self._last_active_paths, x_vals) if x > 1e-7}
	
		# Initialize result containers
		lats_by_ug_arr = np.zeros(self.whole_deployment_n_ug)
		vols_by_poppi = {pi: 0.0 for pi in range(len(self.static_caps))}
		paths_by_ug_res = {}
		congested_vol, total_vol = 0.0, 0.0

		ts = time.time()
		# First pass: link loads (congestion is a property of the LINK total,
		# not of any single path's volume -- the old per-path check missed
		# links inundated by many small allocations).
		for (ug, poppi), vol_amt in raw_x.items():
			vols_by_poppi[poppi] += vol_amt
			total_vol += vol_amt
		_no_path_pi = NO_PATH_INGRESS(self)
		inundated = {pi for pi, v in vols_by_poppi.items()
					 if pi != _no_path_pi and v > self.static_caps[pi] + 1e-6}

		# Congestion-aware belief (Tom, 2026-08-13). In MLU-fallback solves
		# (standard solve infeasible -> elastic caps), volume on inundated
		# links used to be priced at its REAL path latency in BOTH the
		# returned scalar and lats_by_ug -- so the belief/gradient machinery
		# never felt congestion, and the optimizer learned to strand traffic
		# (georand collapses; see solve_lp_assignment.py's matching fix).
		# Congested volume is now priced exactly like no-route volume: the
		# NO_ROUTE_LATENCY sentinel, which training already scales down via
		# SCULPTOR_NO_ROUTE_LATENCY (documented choice: 1000ms for training,
		# canonical 30000 for eval). In standard (feasible) solves nothing
		# is inundated and this is a no-op by construction.
		# SCULPTOR_CONGESTION_AWARE_OBJ=0 restores legacy pricing.
		_cong_aware = os.environ.get(
			'SCULPTOR_CONGESTION_AWARE_OBJ', '1') != '0'

		_soft_routed_wsum = 0.0
		_soft_routed_vol = 0.0
		_soft_bad_vol = 0.0
		for (ug, poppi), vol_amt in raw_x.items():
			ugi = self.whole_deployment_ug_to_ind[ug]

			if poppi in inundated:
				congested_vol += vol_amt

			if ugi not in paths_by_ug_res:
				paths_by_ug_res[ugi] = []
			paths_by_ug_res[ugi].append((poppi, vol_amt / self.whole_deployment_ug_to_vol[ug]))

			# Calculate latency for this specific <user, path> allocation
			if poppi == _no_path_pi:
				path_lat = NO_ROUTE_LATENCY
				_soft_bad_vol += vol_amt
			elif _cong_aware and poppi in inundated:
				path_lat = NO_ROUTE_LATENCY
				_soft_bad_vol += vol_amt
			else:
				path_lat = self.whole_deployment_ug_perfs[ug][self.popps[poppi]]
				_soft_routed_wsum += path_lat * vol_amt
				_soft_routed_vol += vol_amt

			# Weighted average latency contribution
			lats_by_ug_arr[ugi] += path_lat * (vol_amt / self.whole_deployment_ug_to_vol[ug])

		obj_norm = np.sum(self.whole_deployment_ug_vols)
		if _cong_aware:
			# SOFT BOUNDED objective (Tom, 2026-08-14): congested/no-route
			# volume contributes a BOUNDED penalty (SCULPTOR_SOFT_CONG_PENALTY
			# ms-equivalent per unit bad-fraction, default 50) instead of
			# sentinel-scale latency inside the average. Sentinel pricing made
			# a single adjacent flip move the believed objective by 5-10
			# units -> oversized gradients -> rescale pathologies/instability
			# (dep3 freeze, both ladder eras). lats_by_ug keeps the sentinel
			# marking so eval-side identification is unchanged.
			_soft_P = float(os.environ.get('SCULPTOR_SOFT_CONG_PENALTY', '50'))
			_total_v = _soft_routed_vol + _soft_bad_vol
			_avg_routed = (_soft_routed_wsum / _soft_routed_vol
						   if _soft_routed_vol > 0 else 0.0)
			_frac_bad = _soft_bad_vol / _total_v if _total_v > 0 else 1.0
			_objective = -1 * (_avg_routed + _soft_P * _frac_bad)
		else:
			_objective = -1 * model_res.objVal / obj_norm
		self.timing['organizing_results'] += time.time()-ts
		return {
			"objective": _objective, # Framing 'benefit' as positive
			"legacy_objective": -1 * model_res.objVal / obj_norm,
			"raw_solution": raw_x,
			"paths_by_ug": paths_by_ug_res,
			"lats_by_ug": lats_by_ug_arr,
			"available_paths": available_paths,
			"solved": True,
			"vols_by_poppi": vols_by_poppi,
			"fraction_congested_volume": congested_vol / (total_vol + 1e-9)
		}

	def init_all_vars(self):
		## Latency benefit for each user is -1 * MAX_LATENCY -> -1 MIN_LATENCY
		## divided by their contribution to the total volume (i.e., multiplied by a weight)
		## important that every worker has the same lbx
		min_vol,max_vol = np.min(self.ug_vols), np.max(self.ug_vols)
		total_deployment_volume = np.sum(self.ug_vols)
		if self.simulated:
			min_lbx = np.maximum(-.1,-1 * NO_ROUTE_LATENCY * max_vol / total_deployment_volume)
		else:
			min_lbx = np.maximum(-.1,-1 * NO_ROUTE_LATENCY * max_vol / total_deployment_volume)

		max_lbx = 0

		self.lbx = np.linspace(min_lbx, max_lbx,num=LBX_DENSITY)
		self.big_lbx = np.zeros((LBX_DENSITY, self.n_ug))
		for i in range(self.n_ug):
			self.big_lbx[:,i] = copy.copy(self.lbx)
		self.lb_range_trackers = {ui: [min_lbx,max_lbx] for ui in range(self.n_ug)}
		self.lb_range_alpha = .005 ## EWMA for update LB range definitions

		self.stop = False
		self.calc_cache = Calc_Cache()
		self.this_time_ip_cache = {}

		self.iter = 0
		self.init_persistent_lp() # Setup the Gurobi shell

	def increment_iter(self):
		self.iter += 1

	def clear_caches(self):
		self.this_time_ip_cache = {}
		self.calc_cache.clear_all_caches()

	def summarize_cache_size(self):
		## 
		for obj,nm in zip([self.user_ip_cache, self.calc_cache, self.this_time_ip_cache], 
			['user ip', 'calc cache', 'this time ip']):
			self.print("{} cache -- {} size".format(nm,round(len(pickle.dumps(obj))/1e6)))

	def lfs_to_penalty(self, lfs):
		"""LFS is the greedy-allocation volume divided by the link capacity. Want to compute a
			multiplicative latency penalty to encourage people to not inundate links. 
			But can't be too rough so that it's unstable."""
		return np.power(lfs, .1)

	def sim_rti(self):
		### Randomly simulates routes and returns them according to our model of ingress probabilities
		## routed_through_ingress: prefix -> ug -> popp

		## helpful object to precompute
		ts = time.time()
		self.pmat_by_prefix = {}
		for ui in self.ingress_probabilities:
			self.pmat_by_prefix[ui] = {}
			for (poppi, pref_i), p in self.ingress_probabilities[ui].items():
				try:
					self.pmat_by_prefix[ui][pref_i] 
				except KeyError:
					self.pmat_by_prefix[ui][pref_i] = [[],[]]
				self.pmat_by_prefix[ui][pref_i][0].append(poppi)
				self.pmat_by_prefix[ui][pref_i][1].append(p)
		self.timing['pmat_organize'] += time.time() - ts

		## Aggregate by prefix (since we're simulating routes)
		routed_through_ingress = {}
		for ui in self.ingress_probabilities:
			## randomly simulate routing
			choices_by_simi = {}
			for pref_i in self.pmat_by_prefix[ui]:
				poppis, probs = self.pmat_by_prefix[ui][pref_i]
				random_poppi = np.random.choice(poppis, size=self.MC_NUM, replace=True, p=probs)
				for mci in range(self.MC_NUM):
					try:
						routed_through_ingress[mci]
					except KeyError:
						routed_through_ingress[mci] = {}
					try:
						routed_through_ingress[mci][pref_i][self.whole_deployment_ugs[ui]] = self.popps[random_poppi[mci]]
					except KeyError:
						routed_through_ingress[mci][pref_i] = {self.whole_deployment_ugs[ui]: self.popps[random_poppi[mci]]}

		return routed_through_ingress

	def sim_rti_better(self):
		ts = time.time()
		
		# --- Step 1: Flatten Data & Prepare Matrices ---
		# We need to map the nested dict structure into linear arrays for vectorization.
		# We will track metadata to reconstruct the dictionary later.
		self.rti_data["meta_data"] = [] # List of tuples: (user_index, prefix_index, user_group_name)
		self.rti_data["all_probs"] = [] # List of probability arrays
		self.rti_data["all_poppis"] = [] # List of ingress index arrays
		
		# Iterate through your existing structure to flatten it
		for ui, entries in self.ingress_probabilities.items():
			# Temporary storage to group by prefix for this user
			# (Your original code grouped by prefix, so we must too)
			temp_group = {} 
			
			for (poppi, pref_i), p in entries.items():
				if pref_i not in temp_group:
					temp_group[pref_i] = {'pops': [], 'probs': []}
				temp_group[pref_i]['pops'].append(poppi)
				temp_group[pref_i]['probs'].append(p)
				
			# Add these grouped entries to our master lists
			ug_name = self.whole_deployment_ugs[ui]
			for pref_i, data in temp_group.items():
				self.rti_data["meta_data"].append((ui, pref_i, ug_name))
				self.rti_data["all_probs"].append(data['probs'])
				self.rti_data["all_poppis"].append(data['pops'])

		self.timing['pmat_organize'] += time.time() - ts

		# --- Step 2: Memory-Efficient Vectorized Selection ---
		
		self.rti_data["num_scenarios"] = len(self.rti_data["all_probs"])
		self.rti_data["max_choices"] = max(len(p) for p in self.rti_data["all_probs"])
		
		# Create Padded Matrix and CDF as before
		# Memory: O(N_scenarios * Max_Choices) - Very small
		P_matrix = np.zeros((self.rti_data["num_scenarios"], self.rti_data["max_choices"]))
		self.rti_data["choices_matrix"] = np.full((self.rti_data["num_scenarios"], self.rti_data["max_choices"]), -1, dtype=int)
		
		for i, (probs, pops) in enumerate(zip(self.rti_data["all_probs"], self.rti_data["all_poppis"])):
			n = len(probs)
			P_matrix[i, :n] = probs
			self.rti_data["choices_matrix"][i, :n] = pops

		cdf = np.cumsum(P_matrix, axis=1)
		cdf[:, -1] = 1.0
		# 1. Create offsets. shape: (N_scenarios,)
		# Each row 'i' is shifted by i. 
		# This ensures values in row 0 are in range [0, 1], row 1 in [1, 2], etc.
		self.rti_data["offsets"] = np.arange(self.rti_data["num_scenarios"])
		
		# 2. Add offsets to the CDF
		# shape: (N_scenarios, Max_Choices)
		self.rti_data["cdf_offset"] = cdf + self.rti_data["offsets"][:, None]
		# Generate Random numbers
		# Memory: O(N_scenarios * MC_NUM)
		rand_vals = np.random.rand(self.rti_data["num_scenarios"], self.MC_NUM)
		
		# 3. Add offsets to the random values
		# shape: (N_scenarios, MC_NUM)
		rand_offset = rand_vals + self.rti_data["offsets"][:, None]

		# 4. Flatten both
		# cdf_flat size: N_scenarios * Max_Choices
		# rand_flat size: N_scenarios * MC_NUM
		cdf_flat = self.rti_data["cdf_offset"].ravel()
		rand_flat = rand_offset.ravel()

		# 5. Perform one giant binary search
		# This returns the insertion index in the flattened array
		insert_indices = np.searchsorted(cdf_flat, rand_flat)

		# 6. Map back to 2D indices
		# The 'insert_indices' are indices into the FLATTENED cdf.
		# We need to know which column (which choice) that corresponds to.
		# Since cdf_flat is row-major, modulo Max_Choices gives us the column index.
		# However, because we offset the VALUES, not the indices, simply taking modulo
		# of the result index works because 'cdf_flat' is effectively sorted globally.
		idx_selections_flat = insert_indices % self.rti_data["max_choices"]
		# Reshape back to (N_scenarios, MC_NUM)
		idx_selections = idx_selections_flat.reshape(self.rti_data["num_scenarios"], self.MC_NUM)
		# Map indices back to actual POPPIs
		row_indices = np.arange(self.rti_data["num_scenarios"])[:, None]
		selected_poppis = self.rti_data["choices_matrix"][row_indices, idx_selections]

		# --- Construct Output Dictionary ---
		routed_through_ingress = {}
		# We iterate over the scenarios (rows) and their generated results
		for i, (ui, pref_i, ug_name) in enumerate(self.rti_data["meta_data"]):
			simulated_routes = selected_poppis[i] # Array of size MC_NUM
			
			for mci, poppi in enumerate(simulated_routes):
				# Access dictionary structure only once per MC index if possible
				if mci not in routed_through_ingress:
					routed_through_ingress[mci] = {}
				
				mc_dict = routed_through_ingress[mci]
				
				# Map the integer index back to the real object (self.popps)
				real_pop_obj = self.popps[poppi]
				
				# Assign deep in the structure
				if pref_i not in mc_dict:
					mc_dict[pref_i] = {}
				
				mc_dict[pref_i][ug_name] = real_pop_obj
				try:
					if not (self.ingress_probabilities[ui][poppi,pref_i] > 0):
						print("{},{},{} has route but {} probability".format(ui,poppi,pref_i,self.ingress_probabilities[ui][poppi,pref_i]))
				except KeyError:
					if ui not in self.ingress_probabilities:
						print("UI in not ingress probs")
					elif (poppi,pref_i) not in self.ingress_probabilities[ui]:
						print("{},{} not in inress probs for {}".format(poppi,pref_i,ui))
					exit(0)

		return routed_through_ingress	

	def get_ingress_probabilities_by_dict_generic(self, a, verb=False, **kwargs):
		## Uses dictionaries to do the job
		a_log = threshold_a(a).astype(bool)

		sum_a = np.sum(a,axis=0)

		timers = {
			'cache_hits': 0,
			'cache_lookups': 0,
			'api': 0,
			'dpi': 0,
			'apugi': 0,
			'vpugi': 0,
			'sort_calc': 0,
			'final_calc': 0,
			'total': 0,
		}

		self.ingress_probabilities = {ui:{} for ui in range(self.whole_deployment_n_ug)}
		for pref_i in np.where(sum_a)[0]:
			ts_loop = time.time()
			tloga = tuple(a_log[:,pref_i].flatten())
			if np.sum(a[:,pref_i]) == 0:
				continue
			try:
				for (poppi,ui), prob in self.this_time_ip_cache[tloga].items():
					# will need a more complicated caching mechanism if ever non-uniform
					self.ingress_probabilities[ui][poppi,pref_i] = 1.0/prob 
				timers['cache_hits'] += 1
				timers['cache_lookups'] += time.time() - ts_loop
				continue
			except KeyError:
				pass

			## i.e, for each user and for each popp. compute whether a parent of that popp is currently active
			active_parent_indicator = {}
			poppis_active = {poppi:None for poppi in np.where(a_log[:,pref_i])[0]}
			for ug,child,parent in self.parent_tracker: ### we should modify parent tracker to map parent to children
				ui = self.whole_deployment_ug_to_ind[ug]
				parenti = self.popp_to_ind[parent]
				childi = self.popp_to_ind[child]
				try:
					poppis_active[parenti]
					active_parent_indicator[ui,childi] = 1
				except KeyError:
					continue
			timers['api'] += time.time() - ts_loop; ts_loop=time.time()

			## For active poppi in active_poppis, for user in poppi to users, if not parent active for poppi -> tabulate

			## Group by user
			self.this_time_ip_cache[tloga] = {}
			cacheref = self.this_time_ip_cache[tloga]
			for ui in range(self.whole_deployment_n_ug):
				these_poppis = []
				ref = self.whole_deployment_ui_to_poppi[ui]
				for poppi in poppis_active:
					try:
						ref[poppi]
					except KeyError:
						continue ### user doesn't have this popp
					try:
						active_parent_indicator[ui,poppi] ### We have an active parent, ignore
						continue
					except KeyError:
						these_poppis.append(poppi)

				if len(these_poppis) == 0:
					continue
				npoppis = len(these_poppis)
				likelihood = 1.0 / npoppis
				for poppi in these_poppis:
					self.ingress_probabilities[ui][poppi,pref_i] = likelihood
					### Cache the entries that have non-zero probability
					cacheref[poppi,ui] = npoppis
			timers['final_calc'] += time.time() - ts_loop; ts_loop=time.time()
	
	def get_ingress_probabilities_and_sim(self, a, verb=False, **kwargs):
		"""
	    Combined and optimized version of get_ingress_probabilities + sim_rti_better.
	    Directly produces the routed_through_ingress dictionary using pattern caching.

	    Orchestrator over two named sub-steps so subclasses can override the
	    sampling stage independently of the (deterministic) option/probability
	    computation:
	      _compute_scenario_options(a)     populates self.rti_data
	      _sample_scenario_realizations()  MC-draws routed_through_ingress
	    """
		ts_total = time.time()
		self._compute_scenario_options(a, verb=verb, **kwargs)
		routed_through_ingress = self._sample_scenario_realizations()
		self.timing['total_rti_calc'] += time.time() - ts_total
		return routed_through_ingress

	def _compute_scenario_options(self, a, verb=False, **kwargs):
		"""Populate self.rti_data (per-(ug,prefix) ingress options + probabilities)
		for advertisement `a`. Deterministic; pattern-cached."""
		ts_total = time.time()

		# --- 1. Initialize Containers ---
		# Instead of nested dicts, we build the flat lists required for vectorization directly.
		self.rti_data = {
			"meta_data": [],  # List of tuples: (ui, pref_i, ug_name)
			"all_probs": [],  # List of probability lists: [0.5, 0.5]
			"all_poppis": []  # List of choice lists: [pop_A, pop_B]
		}

		# Ensure persistent cache exists (persist this across function calls)
		if not hasattr(self, 'pattern_cache'):
			self.pattern_cache = {}

		# Local variable speedups to avoid self lookups in loop
		ugs = self.whole_deployment_ugs
		# Assumed to be {ui: [poppi, poppi...]} or {ui: {poppi: data}}
		ui_to_poppi = self.whole_deployment_ui_to_poppi 

		# --- 2. Process Availability Matrix (a) ---
		# Assuming threshold_a logic is effectively: > 1e-6 means active
		a_log = (a > 1e-6) 

		# Iterate over prefixes (columns of a)
		for pref_i in range(a.shape[1]):
			col = a_log[:, pref_i]
			
			# Optimization: If no POPs are active for this prefix, skip entirely
			if not np.any(col):
				continue

			# Create a hashable signature for this availability state
			tloga = tuple(col)

			# --- CACHE CHECK ---
			if tloga in self.pattern_cache:
				# HIT: We have seen this network state before.
				# cached_entries is a list of: (ui, valid_pops_list, probs_list)
				cached_entries = self.pattern_cache[tloga]
				
				# Fast append to master lists
				# We reuse the logic (pops/probs), but update the prefix index (pref_i)
				for ui, pops, probs in cached_entries:
					self.rti_data["meta_data"].append((ui, pref_i, ugs[ui]))
					self.rti_data["all_probs"].append(probs)
					self.rti_data["all_poppis"].append(pops)
				continue

			# --- CACHE MISS: Calculate Logic ---
			# This block only runs when we encounter a UNIQUE network failure state
			
			# 1. Identify active POPs indices
			active_poppis = np.where(col)[0]
			active_poppis_set = set(active_poppis)

			# 2. Identify Blocked (User, Child) pairs due to Active Parents
			# blocked_user_child stores (ui, child_poppi) that are FORBIDDEN
			blocked_user_child = set()
			for ug, child, parent in self.parent_tracker:
				parenti = self.popp_to_ind[parent]
				# If the parent is active in this specific state 'tloga', the child is blocked
				if parenti in active_poppis_set:
					ui = self.whole_deployment_ug_to_ind[ug]
					childi = self.popp_to_ind[child]
					blocked_user_child.add((ui, childi))

			# 3. Build Routing for this State
			entries_for_cache = [] # To store (ui, pops, probs) for future reuse

			for ui in range(self.whole_deployment_n_ug):
				valid_pops = []
				
				# Get potentially available POPs for this user (static config)
				potential_pops = ui_to_poppi[ui]
				
				for poppi in potential_pops:
					# Condition 1: POP must be physically UP
					if poppi not in active_poppis_set:
						continue
					
					# Condition 2: POP must not be blocked by an active parent
					if (ui, poppi) in blocked_user_child:
						continue
					
					valid_pops.append(poppi)

				if not valid_pops:
					continue

				# Compute Uniform Probability
				n = len(valid_pops)
				probs = [1.0 / n] * n
				
				# Append to current run
				self.rti_data["meta_data"].append((ui, pref_i, ugs[ui]))
				self.rti_data["all_probs"].append(probs)
				self.rti_data["all_poppis"].append(valid_pops)

				# Append to Cache
				entries_for_cache.append((ui, valid_pops, probs))

			# Save this state's logic to cache so we never calculate it again for this pattern
			self.pattern_cache[tloga] = entries_for_cache

		self.timing['pmat_organize'] += time.time() - ts_total

	def _sample_scenario_realizations(self):
		"""Monte-carlo draw of self.MC_NUM joint route realizations from the
		scenario options in self.rti_data (populated by
		_compute_scenario_options). Returns routed_through_ingress:
		{mc_index: {prefix: {ug: popp}}}."""
		# --- 3. Vectorized Simulation (Previously sim_rti_better) ---
		# Now self.rti_data is fully populated. We proceed with the vectorized selection.

		self.rti_data["num_scenarios"] = len(self.rti_data["all_probs"])
		if self.rti_data["num_scenarios"] == 0:
			return {}

		self.rti_data["max_choices"] = max(len(p) for p in self.rti_data["all_probs"])

		# Create Padded Matrix
		P_matrix = np.zeros((self.rti_data["num_scenarios"], self.rti_data["max_choices"]))
		self.rti_data["choices_matrix"] = np.full((self.rti_data["num_scenarios"], self.rti_data["max_choices"]), -1, dtype=int)

		for i, (probs, pops) in enumerate(zip(self.rti_data["all_probs"], self.rti_data["all_poppis"])):
			n = len(probs)
			P_matrix[i, :n] = probs
			self.rti_data["choices_matrix"][i, :n] = pops

		# CDF Construction
		cdf = np.cumsum(P_matrix, axis=1)
		cdf[:, -1] = 1.0 # Force sum to 1.0 to avoid float precision issues

		# Offset Trick for Vectorized Search
		# Shifts the values of every row so we can search a single flattened array
		offsets = np.arange(self.rti_data["num_scenarios"])
		cdf_offset = cdf + offsets[:, None]

		# Generate Random Numbers
		rand_vals = np.random.rand(self.rti_data["num_scenarios"], self.MC_NUM)
		rand_offset = rand_vals + offsets[:, None]

		# Flatten for searchsorted
		cdf_flat = cdf_offset.ravel()
		rand_flat = rand_offset.ravel()

		# Binary Search (Finds insertion point in flattened CDF)
		insert_indices = np.searchsorted(cdf_flat, rand_flat)

		# Map back to 2D indices
		idx_selections_flat = insert_indices % self.rti_data["max_choices"]
		idx_selections = idx_selections_flat.reshape(self.rti_data["num_scenarios"], self.MC_NUM)

		# Retrieve selected POP indices
		row_indices = np.arange(self.rti_data["num_scenarios"])[:, None]
		selected_poppis = self.rti_data["choices_matrix"][row_indices, idx_selections]

		# --- 4. Construct Final Output Dictionary ---
		routed_through_ingress = {}

		for i, (ui, pref_i, ug_name) in enumerate(self.rti_data["meta_data"]):
			simulated_routes = selected_poppis[i] # Array of size MC_NUM
			
			for mci, poppi in enumerate(simulated_routes):
				if mci not in routed_through_ingress:
					routed_through_ingress[mci] = {}
				
				# Ensure structure exists
				if pref_i not in routed_through_ingress[mci]:
					routed_through_ingress[mci][pref_i] = {}
				
				# Assuming self.popps is a list/dict of actual POP objects
				routed_through_ingress[mci][pref_i][ug_name] = self.popps[poppi]

		return routed_through_ingress

	def generic_objective_pdf(self, obj, a, **kwargs):
		"""
			Solves self.MC_NUM traffic assignment problems, assuming that user routes are distributed
			according to distribution self.ingress_probabilities.
		"""

		### TODO -- maybe implement subset of users, but not really essential
		ts = time.time()
		if not TEST_BETTER_VERSION:
			all_routed_through_ingress = self.sim_rti_better()
		else:
			all_routed_through_ingress = self.get_ingress_probabilities_and_sim(a)
		self.timing['sim_rti'] += time.time() - ts
		objs = np.zeros(self.MC_NUM)
		for i in range(self.MC_NUM):
			routed_through_ingress = all_routed_through_ingress[i]
			if obj == "avg_latency" or obj == "per_site_cost":
				ts = time.time()
				total_obj = self.solve_generic_lp_persistent(routed_through_ingress, obj)["objective"]
				self.timing['solve_generic_lp_persistent'] += time.time() - ts
			else:
				ts = time.time()
				# Pass the adv matrix `a` through so multi-LP objectives
				# (static_failure, backup_capacity) can recover it. Plain
				# objectives ignore the kwarg.
				total_obj = solve_generic_lp_with_failure_catch(self, routed_through_ingress, obj, adv=a)['objective']
				self.timing['solve_generic_lp_not_persistent'] += time.time() - ts
			objs[i] = total_obj
		### return x and distribution of x
		## numpy histogram returns all bin edges which is of length len(x) + 1
		## so cut off the last edge
		if max(objs) - min(objs) < .001:
			## trivial distribution
			x = np.linspace(objs[0],objs[0]+1, num=LBX_DENSITY)
			pdfx = np.zeros(LBX_DENSITY)
			pdfx[0] = 1.0
		else:
			pdfx, x = np.histogram(objs, bins=LBX_DENSITY, density=True)
			x = x[:-1]
			pdfx = pdfx / np.sum(pdfx)
		return x, pdfx

	def generic_benefit(self, a, f_w, **kwargs):
		"""

		Calculates average and distributional estimate of benefit, where benefit
		is a function of the (joint) routing distribution.
		Works by MC-sampling routing distribution and computing a histogram of benefits.

		"""

		a_effective = threshold_a(a)
		verb = kwargs.get('verbose_workers')

		### We may choose to compute expected benefit over a subset of all users
		### when we do this, the key thing is to remember to turn off caching
		subset_ugs = False
		which_ugs = kwargs.get('ugs', None)
		if which_ugs is not None:
			subset_ugs = True

		# DEFAULT ON -- Tom-ratified 2026-08-16 after a full paired A/B
		# (210-cell ladder grid x cache on/off, identical deployments/seeds,
		# 167 matched cells): quality delta ON-OFF median +0.000 / mean
		# +0.114 ms vs opp (worst arm median +0.32), wall-clock 3.1x FASTER
		# and immune to the late-run belief-support cost blowup that stalls
		# cache-off cells. Caching does not meaningfully hurt performance.
		# SCULPTOR_LB_CACHE=0: never RETURN memoized latency-benefit results;
		# re-run the MC fresh on every call (the store below still happens,
		# harmlessly). Rationale: the cache freezes benefit(A) AND its pdf at
		# the first evaluation's random draws, invalidated only by a real
		# measurement (clear_new_measurement_caches). With measurements every
		# step (stock/fixed mode) that's harmless; under gated/starved probing
		# nothing clears it, so beliefs -- including the uncertainty the probe
		# gate consumes -- become stale frozen snapshots.
		_use_lb_cache = os.environ.get('SCULPTOR_LB_CACHE', '1') != '0'
		if not verb and not subset_ugs and _use_lb_cache:
			## don't rely on caching if we want to log / print statistics
			try:
				cache_rep = get_a_cache_rep(a_effective)
				benefit, (xsumx_cache_rep, psumx_cache_rep) = self.calc_cache.all_caches['lb'][cache_rep]
				xsumx = np.linspace(xsumx_cache_rep[0], xsumx_cache_rep[1], num=LBX_DENSITY)
				psumx = np.zeros(LBX_DENSITY)
				for i,d in psumx_cache_rep.items():
					psumx[i] = d
				ret = (benefit, (xsumx,psumx))

				return ret
			except KeyError:
				pass

		## Dims are path, prefix, user
		if not TEST_BETTER_VERSION:
			ts = time.time()
			self.get_ingress_probabilities_by_dict_generic(a_effective, **kwargs) ## populates self.ingress_probabilities
			self.timing['get_ingress_probabilities_by_dict_generic'] += time.time() - ts

		if subset_ugs: ##### REVISIT
			which_ugs_this_worker = get_intersection(which_ugs, self.whole_deployment_ugs)
			if len(which_ugs_this_worker) == 0:
				pdf = np.zeros(self.lbx.shape)
				pdf[-1] = 1
				return 0, (self.lbx.flatten(), pdf.flatten())
			which_ugs_i = np.array([self.whole_deployment_ug_to_ind[ug] for ug in which_ugs_this_worker])


		## Calculate pdf of the generic objective
		if subset_ugs:
			xsumx, psumx = self.generic_objective_pdf(f_w, a_effective, which_ugs_i=which_ugs_i)
		else:
			xsumx, psumx = self.generic_objective_pdf(f_w, a_effective)

		xsumx = xsumx.flatten(); psumx = psumx.flatten()
		benefit = np.sum(xsumx * psumx)

		if not subset_ugs:
			### Store compressed versions of these variables
			cache_rep = get_a_cache_rep(a_effective)
			xsumx_cache_rep = (xsumx[0], xsumx[-1])
			psumx_cache_rep = {}
			for i in np.where(psumx)[0]:
				psumx_cache_rep[i] = psumx[i]

			self.calc_cache.all_caches['lb'][cache_rep] = (benefit, (xsumx_cache_rep, psumx_cache_rep))
		# print("Returning benefit : {}".format(benefit))
		return benefit, (xsumx, psumx)  

	def latency_benefit(self, a, **kwargs):
		"""Calculates distribution of latency benefit at a given advertisement. Benefit is the sum of 
			benefits across all users. Closed form calculation."""
		return self.generic_benefit(a, kwargs.get('generic_obj'))

	def log(self,s):
		self.log_ptr = open(os.path.join(LOG_DIR, 'worker_{}_log-{}.txt'.format(self.worker_i, self.dpsize)),'a')
		self.log_ptr.write(s)
		self.log_ptr.close()

	def print(self, s):
		print("Worker {} -- {}".format(self.worker_i, s))

	def check_clear_cache(self):
		cache_to_clear = self.calc_cache.all_caches['lb']

		if len(cache_to_clear) > MAX_CACHE_SIZE:
			# order of lbx_density + n_popps*n_prefixes per entry
			# self.print("Clearing calc cache, currently size {}".format(
			#   len(pickle.dumps(self.calc_cache))/1e6))
			# self.print("Clearing calc cache, current len {}".format(len(cache_to_clear)))
			self.this_time_ip_cache = {}
			self.calc_cache.all_caches['lb'] = {}
		if not self.simulated:
			if np.random.random() > .9999:
				## Just randomly clear it since we don't measure often
				self.this_time_ip_cache = {}
				self.calc_cache.all_caches['lb'] = {}


	def get_node_mem_avail_mb(self):
		"""Return this Ray-actor node's /proc/meminfo MemAvailable in MB.
		Used by the driver-side autoscale policy to monitor worker-node
		memory pressure (the existing policy only knew about the HEAD
		node, but at 64 workers x ~2 GB the worker node can pressure
		first). None if /proc isn't readable.
		"""
		try:
			with open('/proc/meminfo', 'r') as f:
				for line in f:
					if line.startswith('MemAvailable:'):
						return int(line.split()[1]) // 1024
		except (FileNotFoundError, PermissionError):
			return None
		return None

	def dump_mem_log(self):
		"""Return the contents of this worker's [mem-worker] log file.
		Driver calls this via send_receive_workers (ZMQ) or Ray RPC at the
		end of a run to collect per-worker memory data, since Ray's stdout
		forwarding to the driver is unreliable in our setup.
		"""
		try:
			with open(_get_worker_mem_log_path(self.worker_i), 'r') as f:
				return f.read()
		except (FileNotFoundError, PermissionError):
			return ''

	def clear_new_meas_caches(self):
		# print("Clearing caches in worker {}".format(self.worker_i))
		self.this_time_ip_cache = {}
		self.pattern_cache = {}
		self.calc_cache.clear_new_measurement_caches()

	# The ZMQ command dispatcher (`check_for_commands`), the worker run loop
	# (`run`), and the `if __name__ == "__main__":` script entrypoint that
	# used to live here were removed when the project went Ray-only. The
	# equivalent dispatch for the Ray actor lives in
	# path_distribution_computer_ray._LocalPathDistributionComputer as a set
	# of `_cmd_*` methods routed via `handle_msg`. See git history for the
	# old ZMQ implementation.