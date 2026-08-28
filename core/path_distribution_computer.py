"""Worker-side LP / latency-benefit logic for SCULPTOR.

`Path_Distribution_Computer` holds the per-worker LP machinery: a
persistent Gurobi model (built once at init via `init_persistent_lp`),
an LP cache, monte-carlo routing probabilities, and the `latency_benefit`
function the driver calls thousands of times per gradient probe.

In production this class is constructed as a Ray actor via
`path_distribution_computer._LocalPathDistributionComputer`
(see that file for the Ray dispatch surface). The base `__init__` here
raises in non-debug mode — direct subprocess instantiation is gone with
the ZMQ removal (May 2026).

Key methods used by the driver:
  - `latency_benefit(adv, **kw)`  monte-carlo estimate, hot path during
                                   gradient probes
  - `solve_generic_lp_persistent` reuse the Gurobi shell for one LP
                                   evaluation (gradient probes)
  - `update_deployment(dep)`      refresh worker state when the deployment changes
                                   (per-dpsize boundary or adaptive resize)
  - `dump_mem_log()`              return this worker's [mem-worker] file
                                   content (driver collects at end of run)

Unit-test instantiation: pass `debug=True` to skip the run guard.
"""

# run-as-script bootstrap: this module lives in a package now,
# so put the repo root on sys.path before importing siblings.
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

import numpy as np, pickle, copy, time, random, os
import ray
from collections import defaultdict
random.seed(31415)
from helpers.constants import *
from helpers.helpers import *
from core.test_polyphase import *
from core.optimal_adv_wrapper import Optimal_Adv_Wrapper

from core.solve_lp_assignment import solve_generic_lp_with_failure_catch, get_paths_by_ug, NO_PATH_INGRESS

import core.gpshim as gp  # gurobipy-subset facade; SCULPTOR_LP_BACKEND=gurobi(default)|highs
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


def _malloc_trim():
	"""Return freed allocator arenas to the OS (Linux). RSS is a high-water
	mark: pymalloc/glibc rarely release pages after big temporaries (the
	pmat_organize/mega-batch buffers), so worker RSS ratchets up even with
	zero live garbage -- the 'unattributed' 1.6GB in the 2026-08-24 census.
	OPT-IN (SCULPTOR_MALLOC_TRIM=1) as of 2026-08-25: per-message trim
	is harmless with headroom but under memory pressure the freed pages
	re-fault on the next allocation -- a plausible amplifier of the
	actual-32 endgame degradation (LB grads 80s->1188s as free memory
	hit 3GB). Prefer-underfit: off unless explicitly enabled."""
	if os.environ.get('SCULPTOR_MALLOC_TRIM', '0') != '1':
		return
	try:
		import ctypes
		ctypes.CDLL('libc.so.6').malloc_trim(0)
	except Exception:
		pass


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


def _deep_size(obj, _seen=None, _budget=None, _stats=None):
	"""Approximate bytes held by `obj`, following containers.

	numpy arrays report `.nbytes` (getsizeof sees only a ~112-byte header,
	which is how a multi-GB shard looked tiny).

	LARGE CONTAINERS ARE SAMPLED, not truncated. The previous version
	walked up to a fixed node budget and then simply stopped, which
	silently halved every big dict: a 3173-ug x 40-popp ug_perfs measured
	11.3 MB against a true 22.1 MB, and because the cut-off is
	deterministic every worker reported the identical total -- 158 MB
	against a real 1074 MB RSS (2026-08-21). Sampling K entries and
	scaling by len() keeps the cost bounded AND the estimate unbiased for
	the homogeneous containers this code holds.
	"""
	import sys as _s
	if _seen is None:
		_seen = set()
		_budget = [int(os.environ.get('SCULPTOR_OBJSIZE_BUDGET', '20000000'))]
		_stats = {'sampled': 0, 'truncated': 0}
	if _budget[0] <= 0:
		if _stats is not None:
			_stats['truncated'] += 1
		return 0
	oid = id(obj)
	if oid in _seen:
		return 0
	_seen.add(oid)
	_budget[0] -= 1
	try:
		if hasattr(obj, 'nbytes'):          # numpy / anything array-like
			return int(obj.nbytes)
		total = _s.getsizeof(obj)
	except (TypeError, AttributeError):
		return 0

	sample_min = int(os.environ.get('SCULPTOR_OBJSIZE_SAMPLE_MIN', '5000'))
	sample_k = int(os.environ.get('SCULPTOR_OBJSIZE_SAMPLE_K', '1000'))
	try:
		if isinstance(obj, dict):
			n = len(obj)
			if n > sample_min:
				# STRIDE across the container, never the first K. A head
				# sample is catastrophically wrong when large entries are
				# clustered early: a 40k dict with its big values first
				# measured 417 MB against a true 16.5 MB (+2418%). Walking
				# the keys is a cheap C-level loop; only the sampled
				# entries pay for a deep walk.
				stride = max(1, n // sample_k)
				acc, seen_n = 0, 0
				for j, (k, v) in enumerate(obj.items()):
					if j % stride:
						continue
					acc += _deep_size(k, _seen, _budget, _stats)
					acc += _deep_size(v, _seen, _budget, _stats)
					seen_n += 1
				if seen_n:
					total += int(acc * (float(n) / seen_n))
					if _stats is not None:
						_stats['sampled'] += 1
			else:
				for k, v in obj.items():
					total += _deep_size(k, _seen, _budget, _stats)
					total += _deep_size(v, _seen, _budget, _stats)
		elif isinstance(obj, (list, tuple, set, frozenset)):
			n = len(obj)
			if n > sample_min:
				stride = max(1, n // sample_k)
				acc, seen_n = 0, 0
				for j, x in enumerate(obj):
					if j % stride:
						continue
					acc += _deep_size(x, _seen, _budget, _stats)
					seen_n += 1
				if seen_n:
					total += int(acc * (float(n) / seen_n))
					if _stats is not None:
						_stats['sampled'] += 1
			else:
				for x in obj:
					total += _deep_size(x, _seen, _budget, _stats)
	except RuntimeError:                    # mutated while we walked it
		pass
	return total


def _proc_rss_mb():
	try:
		with open('/proc/self/status') as f:
			for line in f:
				if line.startswith('VmRSS:'):
					return int(line.split()[1]) // 1024
	except (FileNotFoundError, PermissionError):
		pass
	return -1


def _log_objsize_worker(worker_i, tag, obj, top_n=12):
	"""Per-worker object-size census: which attributes hold the memory.

	`[mem-worker]` already answers "how much RSS"; this answers "held by
	what", which is the question you actually need when deciding whether a
	deployment size fits in a given instance family. Off unless
	SCULPTOR_LOG_OBJSIZE=1, because a full census walks the shard.

	Emits one line per big attribute into the same per-worker log file the
	driver already collects via the `dump_mem_log` RPC, so nothing new is
	needed on the collection side:
	    [objsize idx=N] tag=TAG attr=ATTR mb=M n=LEN
	"""
	if os.environ.get('SCULPTOR_LOG_OBJSIZE', '0') != '1':
		return
	t0 = time.time()
	rows = []
	try:
		for name in dir(obj):
			if name.startswith('__'):
				continue
			try:
				val = getattr(obj, name)
			except Exception:
				continue
			if callable(val):
				continue
			nbytes = _deep_size(val)
			if nbytes < 1024 * 1024:        # skip anything under 1 MB
				continue
			try:
				n = len(val)
			except (TypeError, AttributeError):
				n = ''
			rows.append((nbytes, name, n))
	except Exception:
		return
	rows.sort(reverse=True)
	lines = ['[objsize idx={}] tag={} attr={} mb={:.1f} n={} t={:.2f}'.format(
		worker_i, tag, name, nbytes / 1048576.0, n, time.time())
		for nbytes, name, n in rows[:top_n]]
	census_mb = sum(r[0] for r in rows) / 1048576.0
	rss_mb = _proc_rss_mb()
	lines.append('[objsize idx={}] tag={} TOTAL_mb={:.1f} attrs={} '
	             'rss_mb={} unattributed_mb={:.1f} coverage_pct={:.1f} '
	             'census_s={:.2f} t={:.2f}'.format(
	                 worker_i, tag, census_mb, len(rows), rss_mb,
	                 (rss_mb - census_mb) if rss_mb > 0 else -1,
	                 (100.0 * census_mb / rss_mb) if rss_mb > 0 else -1,
	                 time.time() - t0, time.time()))
	for line in lines:
		print(line, flush=True)
	try:
		with open(_get_worker_mem_log_path(worker_i), 'a') as f:
			f.write('\n'.join(lines) + '\n')
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

# SCULPTOR_LP_SOLVE_DEBUG=1 (Tom 2026-08-20 'literally put print
# statements in path_distribution_computer, stop guessing'): per-solve
# anatomy printed by worker 0 (and driver) only —
#   [lpdbg] w=N k=<solve#> backend=<x> paths=<active> act=+A/-D new=<vars
#   created> simplex_iters=<I> t_prep=<s> t_opt=<s>
# plus per-generic_benefit pattern-cache hit/miss and rti timing.
LP_SOLVE_DEBUG = os.environ.get('SCULPTOR_LP_SOLVE_DEBUG', '0') == '1'

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
	(path_distribution_computer._LocalPathDistributionComputer), which
	calls Optimal_Adv_Wrapper.__init__ directly with the deployment +
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
		self.MC_NUM = int(os.environ.get('SCULPTOR_MC_NUM', DEFAULT_MC_NUM))

		if kwargs.get('debug', False):
			self.n_prefixes = None
			return
		# Non-debug instantiation of this class was historically the worker
		# subprocess's entry point (ZMQ socket bind + handshake + run-loop).
		# That path is gone. Production callers go through
		# path_distribution_computer.Path_Distribution_Computer_Actor.remote(...)
		# via worker_comms.Worker_Manager; the Ray subclass calls
		# Optimal_Adv_Wrapper.__init__ directly and never hits this branch.
		raise RuntimeError(
			"Path_Distribution_Computer must be constructed via the Ray actor "
			"(use worker_comms.Worker_Manager). Pass debug=True for unit tests.")

	# Which timing keys are measured INSIDE which others. The counters are
	# nested, not siblings: solve_generic_lp_persistent wraps
	# get_paths_by_ug and solve_unified_lp, and solve_unified_lp wraps
	# lp_feed / optimize / organizing_results. Summing them all as one
	# denominator double-counts the inner work, so "lp_persistent=38%" was
	# never 38% of wall clock (Tom spotted the inconsistency 2026-08-21).
	_TIMING_CHILDREN = {
		'solve_generic_lp_persistent': ['get_paths_by_ug',
										'solve_unified_lp_not_optimize',
										'optimize', 'organizing_results'],
		'sim_rti': ['total_rti_calc'],
		'total_rti_calc': ['pmat_organize'],
	}

	def _mark_init(self, key, seconds):
		"""Accumulate ONE-TIME startup cost, reported separately from the
		per-batch [wt] line.

		The first gradient batch is several times more expensive than every
		later one, and until now that spike was a single opaque bar: the
		per-batch counters are reset each batch, so whatever the worker
		does once -- receiving the deployment, building rb_backups,
		standing up the persistent LP, the first path enumeration -- was
		folded into batch #1 with no way to separate it. This makes the
		startup decomposable instead of merely visible.
		"""
		try:
			self.init_timing[key] = self.init_timing.get(key, 0.0) + seconds
		except AttributeError:
			self.init_timing = {key: seconds}

	def summarize_init_timing(self, tag=''):
		it = getattr(self, 'init_timing', None)
		if not it:
			return
		total = sum(it.values())
		if total < 1e-6:
			return
		parts = ' '.join('{}={:.2f}s'.format(k, v)
						 for k, v in sorted(it.items(), key=lambda kv: -kv[1]))
		print('[{}] [wt-init] w={} tag={} total={:.2f}s {}'.format(
			time.strftime('%H:%M:%SZ', time.gmtime()),
			self.worker_i, tag or 'startup', total, parts), flush=True)

	def _self_timing(self):
		"""Exclusive (self) time per key: total minus its children.

		Values can go slightly negative from clock jitter across nested
		counters; clamped at zero rather than hidden, since a large
		negative would mean the nesting map is wrong.
		"""
		self_t = {}
		for k, v in self.timing.items():
			child = sum(self.timing.get(c, 0.0)
						for c in self._TIMING_CHILDREN.get(k, []))
			self_t[k] = max(0.0, v - child)
		return self_t

	def summarize_timing(self):
		# Print per-key cumulative LP-solve timings (optimize / get_paths_by_ug /
		# organizing_results / solve_unified_lp_not_optimize, etc.). Called every
		# 50 latency_benefit calls in the worker batch loop. Useful for
		# identifying which sub-step inside the LP solve dominates at scale.
		self_t = self._self_timing()
		total_time = sum(self_t.values())      # exclusive: real wall, no
		                                       # double counting
		if total_time < 1e-6:
			return
		# ONE compact parseable line per batch (Tom 2026-08-19: informative,
		# not spammy). Format: [wt] w=<i> total=<s> k1=pct k2=pct ...
		# (categories under 1% omitted). The dash time-share panel parses
		# this; humans can read it in the raw tail.
		parts = ' '.join(
			'{}={:.0f}%'.format(k.replace('solve_generic_lp_', 'lp_')
								 .replace('solve_unified_lp_not_optimize', 'lp_feed')
								 .replace('get_ingress_probabilities_by_dict_generic', 'ing_prob')
								 .replace('get_paths_by_ug', 'paths')
								 .replace('organizing_results', 'organize'),
							   100.0 * self_t[k] / total_time)
			for k in sorted(self_t, key=lambda el: -self_t[el])
			if self_t[k] / total_time >= 0.01)
		incl = self.timing.get('solve_generic_lp_persistent', 0.0)
		n_lp = getattr(self, 'n_lp_solves', 0)
		print("[{}] [wt] w={} total={:.1f}s mc={} lp_solves={} lp_incl={:.0f}% {}"
			  .format(time.strftime('%H:%M:%SZ', time.gmtime()),
					  self.worker_i, total_time, self.MC_NUM, n_lp,
					  100.0 * incl / total_time if total_time else 0,
					  parts), flush=True)

	def init_persistent_lp(self):
		"""Sets up the persistent Gurobi shell with static Volumes and Capacities."""
		_t_init0 = time.time()
		self.model = gp.Model(f"Worker_{self.worker_i}_Persistent")
		# fresh model => no vars are active; incremental trackers
		self._last_active_vars = None
		self._active_keys = None
		self._last_mlu = False
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
		self._mark_init('init_persistent_lp', time.time() - _t_init0)

	def _mlu_lp_weight(self):
		"""LP-units MLU dominance weight: SCULPTOR_MLU_WEIGHT_MULT (10)
		x sum(vol_u * best_lat_u) -- the LP latency term is a
		vol-weighted SUM, so the floor-sum puts W*MLU ~10x its dynamic
		range (mean-scale twin of the retired hard_objectives A)."""
		w = getattr(self, '_mlu_lp_weight_cached', None)
		if w is None:
			best = self.lat_matrix.min(axis=0)
			vols = np.asarray(
				[self.whole_deployment_ug_to_vol[ug]
				 for ug in self.whole_deployment_ugs], dtype=np.float64)
			mult = float(os.environ.get('SCULPTOR_MLU_WEIGHT_MULT', '10'))
			self._mlu_floor_mean = float(np.average(best, weights=vols))
			w = mult * float(np.sum(best * vols))
			self._mlu_lp_weight_cached = w
		return w

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

	def solve_unified_lp(self, available_paths, obj_coeffs, using_mlu=False,
						 mlu_weight=None):
		"""Core solve logic. Toggles between Standard and MLU.
		mlu_weight: objective weight on the MLU variable in MLU mode
		(default 1/ALPHA, the historical infeasibility-fallback weight).
		max_util training passes its dominance weight here so ONE LP
		minimizes latency + W*MLU (Tom 2026-08-25)."""
		# 1. Deactivate ONLY the previously-active path variables (Tom
		# 2026-08-19 hot-loop optimization: the old deactivate-ALL swept
		# O(var_pool)=O(170k at actual-25) per solve — ~46% of lb-solve
		# time — although consecutive candidates differ in ~tens of
		# paths. Invariant: vars outside _last_active_vars already have
		# UB=0 (newly-discovered vars get UB set right after creation
		# via the active-set setAttr below).
		ts = time.time()
		# Deactivation is handled in section 3: incrementally (diff vs
		# the previous active set) on the hot path, or a full sweep of
		# the previously-active vars on the fallback path.
		prev_active = getattr(self, '_last_active_vars', None)

		# 2. Configure MLU variable and Capacity RHS
		if using_mlu:
			self.mlu_dummy.Obj = (1.0 / ALPHA if mlu_weight is None
								  else float(mlu_weight))
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

		# 3. Activate/Discover columns INCREMENTALLY (Tom 2026-08-19 hot
		# loop). Consecutive candidates share ~99% of active paths; we
		# diff with C-level set ops on the path keys and touch ONLY the
		# changed columns. Minimal perturbation also preserves HiGHS's
		# basis, so the subsequent optimize() runs warm (the measured
		# dominant win: within-call MC repeats were near-free).
		# Assumption guarded by _incr_obj_ok: objective coefficients for
		# a persisting active path are call-invariant (true for the
		# static-latency objectives; MLU/site-cost variants pass
		# obj-coeff arrays that can vary -> fall back to full Obj push).
		new_keys = set(available_paths)
		# lpdbg probe (HANDOFF_EODS32 open mystery): var_pool grows past the
		# full (ug,popp) matrix at actual-32 (268k -> 473k); sample the
		# newly-minted keys to identify what mints them (pseudo-ug split?).
		_minted = [] if LP_SOLVE_DEBUG else None
		for (ug, poppi), latency in zip(available_paths, obj_coeffs):
			if (ug, poppi) not in self.var_pool:
				if _minted is not None:
					_minted.append((ug, poppi))
				col = gp.Column()
				col.addTerms(1.0, self.vol_constrs[ug])
				col.addTerms(1.0, self.cap_constrs[poppi])
				self.var_pool[ug, poppi] = self.model.addVar(
					lb=0.0, obj=latency, column=col)
		prev_keys = getattr(self, '_active_keys', None)
		# SCULPTOR_LP_INCR_MLU=1 (Tom 2026-08-20, lpdbg finding): the
		# standard->MLU alternation in congested regimes (belief phase)
		# permanently disabled incrementality via the _last_mlu guard —
		# every solve did a full-pool UB sweep + cold simplex on an
		# ever-growing var_pool. The UB activation state and the path
		# objective coefficients (static latencies for avg_latency) are
		# mode-invariant; section 2 already switches the MLU dummy/cap
		# state per call. So incremental diffs stay valid across modes.
		_incr_mlu_ok = os.environ.get('SCULPTOR_LP_INCR_MLU', '0') == '1'
		# same-MODE consecutive solves are always incrementally safe
		# (UB state and path coeffs are call-invariant); only a mode
		# SWITCH stays gated behind SCULPTOR_LP_INCR_MLU. Without this,
		# max_util training (always-MLU) would full-sweep every solve.
		_incr = (prev_keys is not None
				 and os.environ.get('SCULPTOR_LP_INCREMENTAL', '1') != '0'
				 and (_incr_mlu_ok or
					  using_mlu == getattr(self, '_last_mlu', False)))
		active_vars = [self.var_pool[k] for k in available_paths]
		if _incr:
			to_deact = [self.var_pool[k] for k in prev_keys - new_keys]
			act_idx = [i for i, k in enumerate(available_paths)
					   if k not in prev_keys]
			if to_deact:
				self.model.setAttr("UB", to_deact, [0.0] * len(to_deact))
			if act_idx:
				to_act = [active_vars[i] for i in act_idx]
				self.model.setAttr("UB", to_act,
								   [gp.GRB.INFINITY] * len(to_act))
				self.model.setAttr("Obj", to_act,
								   [obj_coeffs[i] for i in act_idx])
		else:
			# fallback: full deactivate of whatever was active, then
			# full activate of the new set (first call, MLU switches,
			# or SCULPTOR_LP_INCREMENTAL=0)
			if prev_active is None:
				prev_active = list(self.var_pool.values())
			if prev_active:
				self.model.setAttr("UB", prev_active,
								   [0.0] * len(prev_active))
			self.model.setAttr("UB", active_vars,
							   [gp.GRB.INFINITY] * len(active_vars))
			self.model.setAttr("Obj", active_vars, obj_coeffs)
		self._active_keys = new_keys
		self._last_mlu = using_mlu
		# Stash for solve_generic_lp_persistent's raw_x extraction:
		# iterating self.var_pool there (which can hit 100k-1M entries
		# in production) and doing a Gurobi `.X` API call per var costs
		# tens of ms per LP solve at scale. Instead, batch-read X from
		# the small active set using getAttr("X", active_vars) — one
		# C-call, results in a numpy/list of len(active_vars).
		self._last_active_paths = available_paths
		self._last_active_vars = active_vars

		self.timing['solve_unified_lp_not_optimize'] += time.time() - ts
		_dbg_prep_t = time.time() - ts
		ts = time.time()
		if self._grb_dump_dir and self.worker_i == 0 and self._grb_dump_count < 3:
			mps_path = os.path.join(
				self._grb_dump_dir,
				f'sculptor_{self.dpsize}_w{self.worker_i}_solve{self._grb_dump_count}.mps')
			self.model.write(mps_path)
			self._grb_dump_count += 1
		self.model.optimize()
		self.timing['optimize'] += time.time() - ts
		if LP_SOLVE_DEBUG and getattr(self, 'worker_i', -1) in (0, 'drv'):
			self._dbg_solve_k = getattr(self, '_dbg_solve_k', 0) + 1
			_n_act = len(act_idx) if _incr else len(active_vars)
			_n_deact = len(to_deact) if _incr else -1
			print('[lpdbg] w={} k={} backend={} paths={} act=+{}/-{} '
				  'pool={} mint={} mint_sample={} simplex_iters={} '
				  't_prep={:.3f}s t_opt={:.3f}s'.format(
					  getattr(self, 'worker_i', 'drv'), self._dbg_solve_k,
					  gp.BACKEND, len(available_paths), _n_act, _n_deact,
					  len(self.var_pool), len(_minted or ()),
					  (_minted or [])[:3],
					  getattr(self.model, '_last_iter_count', -1),
					  _dbg_prep_t, time.time() - ts), flush=True)
		if self.model.status == 2:
			return self.model
		return None


	def _ug_max_lat(self):
		"""Per-ug max REAL latency from lat_matrix (marker entries
		excluded; pathless ugs fall back to MAX_LATENCY). Base of the
		user-specific penalty prices; cached per lat_matrix identity."""
		key = id(self.lat_matrix)
		c = getattr(self, '_ugmax_c', None)
		if c is None or c[0] != key:
			lm = self.lat_matrix
			mx = np.where(lm < NO_ROUTE_LATENCY, lm, -np.inf).max(axis=0)
			mx = np.where(np.isfinite(mx), mx, float(MAX_LATENCY))
			self._ugmax_c = (key, mx.astype(np.float64))
		return self._ugmax_c[1]

	def _path_obj_coeffs(self, available_paths, obj, site_cost_alpha):
		"""Per-path LP objective coefficients (latencies). Named sub-step of
		solve_generic_lp_persistent so subclasses can override path pricing."""
		# VECTORIZED pricing off the arrays get_paths_by_ug's batched
		# path stashed (same order as available_paths; length-checked).
		# lat_matrix already encodes missing-perf as NO_ROUTE_LATENCY --
		# identical values to the dict chain incl. the stale-path
		# fallback (Tom 2026-08-25 loop elimination).
		_arrs = getattr(self, '_paths_arrays', None)
		if (_arrs is not None and _arrs[2] == len(available_paths)
				and obj in ("avg_latency", "per_site_cost")
				and getattr(self, 'lat_matrix', None) is not None):
			_uu, _pp, _ = _arrs
			self._paths_arrays = None   # single-shot; guards staleness
			_np_ing = NO_PATH_INGRESS(self)
			_is_np = _pp == _np_ing
			_pc = np.where(_is_np, 0, _pp)
			coeffs = self.lat_matrix[_pc, _uu].astype(np.float64, copy=True)
			if obj == "per_site_cost":
				_scbp = getattr(self, '_site_cost_by_poppi', None)
				if _scbp is None:
					_scbp = np.asarray(
						[self.site_costs[pop] for pop, _ in self.popps],
						dtype=np.float64)
					self._site_cost_by_poppi = _scbp
				coeffs = coeffs + site_cost_alpha * _scbp[_pc]
			# USER-SPECIFIC no-route price (Tom 2026-08-28 v2): stranding
			# a user costs 2x that user's own max path latency in every
			# gradient LP -- scale-matched pressure, no marker cliff.
			coeffs[_is_np] = (NO_ROUTE_PENALTY_MULT
							  * self._ug_max_lat()[_uu[_is_np]])
			# stale-path forensics survive: a real path priced NO_ROUTE
			# means the ug had no perf entry for it -- log loudly (rare)
			_stale = np.where((~_is_np)
							  & (self.lat_matrix[_pc, _uu]
								 >= NO_ROUTE_LATENCY))[0]
			for _i in _stale[:5]:
				ug = available_paths[_i][0]
				if self.popps[int(_pp[_i])] not in \
						self.whole_deployment_ug_perfs.get(ug, {}):
					self._log_stale_path(ug, int(_pp[_i]), available_paths)
			return coeffs.tolist()
		obj_coeffs = []
		# hoisted: NO_PATH_INGRESS(self) was re-evaluated per PATH --
		# 1.68M calls per 13-job batch in the 2026-08-24 profile
		_no_path = NO_PATH_INGRESS(self)
		_ugmax = self._ug_max_lat()
		_ug2i = {u: i for i, u in enumerate(self.whole_deployment_ugs)}
		for ug, poppi in available_paths:
			if poppi == _no_path:
				obj_coeffs.append(NO_ROUTE_PENALTY_MULT * _ugmax[_ug2i[ug]])
			else:
				if obj == "avg_latency":
					try:
						obj_coeffs.append(self.whole_deployment_ug_perfs[ug][self.popps[poppi]])
					except KeyError:
						# RARE STALE-PATH EVENT (2026-08-24, Tom): a path was
						# offered for a (ug, popp) the ug has no perf entry
						# for -- observed once per multi-hour solve at size
						# 32 (KeyError ('vtrwarsaw','9009'), iter 71). Log a
						# full forensic dump LOUDLY, then price the path as
						# unroutable instead of killing a 5h solve: one
						# NO_ROUTE-priced path among ~100k biases one LP
						# call, aborting loses the whole strategy.
						self._log_stale_path(ug, poppi, available_paths)
						obj_coeffs.append(NO_ROUTE_PENALTY_MULT
										  * _ugmax[_ug2i[ug]])
				elif obj == "per_site_cost":
					pop, _ = self.popps[poppi]
					site_cost = self.site_costs[pop]
					obj_coeffs.append(self.whole_deployment_ug_perfs[ug][self.popps[poppi]] + site_cost_alpha * site_cost)
				else:
					raise ValueError("obj {} not supported in solve_generic_lp_persistent".format(obj))
		return obj_coeffs


	def _log_stale_path(self, ug, poppi, available_paths):
		"""Forensics for the rare stale-path KeyError. Everything a future
		debugging session needs, deduped per (ug, poppi) per process."""
		seen = getattr(self, '_stale_path_seen', None)
		if seen is None:
			seen = self._stale_path_seen = set()
		count = getattr(self, '_stale_path_count', 0)
		self._stale_path_count = count + 1
		if (ug, poppi) in seen:
			return
		seen.add((ug, poppi))
		popp = self.popps[poppi] if poppi < len(self.popps) else None
		prios = getattr(self,
			'whole_deployment_ground_truth_ingress_priorities', {})
		perfs_ug = self.whole_deployment_ug_perfs.get(ug, {})
		ug_paths = [pi for (u, pi) in available_paths if u == ug]
		import json as _json
		print('[stale-path] w={} n_seen={} FORENSICS {}'.format(
			self.worker_i, self._stale_path_count,
			_json.dumps({
				'ug': str(ug), 'poppi': poppi, 'popp': str(popp),
				'n_popps': self.n_popp, 'dpsize': str(self.dpsize),
				'popp_in_popp_to_ind': popp in self.popp_to_ind,
				'ug_in_perfs': ug in self.whole_deployment_ug_perfs,
				'n_perfs_ug': len(perfs_ug),
				'popp_in_prios_ug': popp in prios.get(ug, {}),
				'n_prios_ug': len(prios.get(ug, {})),
				'ug_available_poppis': ug_paths[:20],
				'perf_popp_sample': [str(k) for k in list(perfs_ug)[:5]],
				'prio_popp_sample': [str(k) for k in
									 list(prios.get(ug, {}))[:5]],
				'ugs_total': len(self.whole_deployment_ug_perfs),
				'ug_in_ug_to_ind': ug in getattr(self, 'ug_to_ind', {}),
			})), flush=True)

	def solve_generic_lp_persistent(self, routed_through_ingress, obj, **kwargs):
		"""The high-level wrapper that tries Standard first, then MLU."""
		ts = time.time()
		available_paths, _ = get_paths_by_ug(self, routed_through_ingress,
											 want_paths_by_ug=False)
		self.timing['get_paths_by_ug'] += time.time() - ts

		# Pre-calculate objective (latencies)
		site_cost_alpha = kwargs.get('site_cost_alpha', DEFAULT_SITE_COST)
		# max_util (Tom 2026-08-25): ONE persistent LP minimizing
		# latency + W*MLU -- latency path coefficients, MLU mode, W
		# sized so the MLU term dominates (SCULPTOR_MLU_WEIGHT_MULT x
		# the vol-weighted per-UG optimal-latency floor, matching the
		# retired two-LP hard_objectives form). Replaces fresh
		# steady-LP + fresh solve_min_mlu (15s-timeout burns) per call.
		_is_mlu_obj = obj == 'max_util'
		if _is_mlu_obj:
			kwargs['is_mlu_obj'] = True
		obj_coeffs = self._path_obj_coeffs(
			available_paths, 'avg_latency' if _is_mlu_obj else obj,
			site_cost_alpha)
		if _is_mlu_obj:
			model_res = self.solve_unified_lp(
				available_paths, obj_coeffs, using_mlu=True,
				mlu_weight=self._mlu_lp_weight())
			# falls through to the shared extraction below; the MLU
			# term joins the soft objective there

		# 1. Try Standard Solve — unless the standard LP has been
		# infeasible on a streak (congested/belief regime): then skip the
		# doomed solve and go straight to MLU, retrying standard every
		# 25th call so we notice when the regime decongests.
		# (SCULPTOR_LP_ADAPTIVE_MLU=1, Tom 2026-08-20 lpdbg finding —
		# halves belief-phase solves and keeps the solve mode stable so
		# incrementality can engage.)
		_adaptive = os.environ.get('SCULPTOR_LP_ADAPTIVE_MLU', '0') == '1'
		_streak = getattr(self, '_std_infeas_streak', 0)
		_skip_standard = (_adaptive and _streak >= 3
						  and (_streak % 25) != 0)
		if not _is_mlu_obj:
			model_res = None
			if not _skip_standard:
				model_res = self.solve_unified_lp(available_paths, obj_coeffs, using_mlu=False)
				if model_res is None:
					self._std_infeas_streak = _streak + 1
				else:
					self._std_infeas_streak = 0
			else:
				self._std_infeas_streak = _streak + 1

			# 2. Fallback to MLU if Standard is Infeasible (or skipped)
			if model_res is None:
				model_res = self.solve_unified_lp(available_paths, obj_coeffs, using_mlu=True)

		if model_res is None:
			# LP is infeasible (e.g. failure scenario removed the last route
			# for some users). Used to call exit(0), which killed the Ray
			# actor and propagated as RayActorError -> caught silently by
			# the per-strategy try/except in eval_all_solution_types ->
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
			# objective is the bounded PRICE, lats keep the MARKER (Tom
			# 2026-08-28): the flat 30000 objective here zeroed gradients
			# for any adv stuck in an infeasible region (maxhard joint,
			# stuck-at-iter-2 forensics).
			from core.solve_lp_assignment import _all_stranded_price
			return {
				"objective": _all_stranded_price(self),
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
		_sentinel_tainted = set()
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
			if path_lat == NO_ROUTE_LATENCY:
				_sentinel_tainted.add(ugi)

		# _ug_sentinel_pricing (SCULPTOR_EVAL_VOLSCEN eval solves, Tom
		# 2026-08-20): degenerate optima split a congestion-touched ug's
		# volume differently across vertices, so its lats_by_ug value is a
		# vertex-dependent MIXTURE of sentinel and real latency. Downstream
		# eval aggregations filter on lat == NO_ROUTE_LATENCY exactly, so
		# mixtures leak sentinel-scale garbage into averages. Under this
		# flag any congestion-touched ug is priced at the sentinel exactly
		# — vertex-stable and matches the aggregation's intent ("skip ugs
		# touching congestion"). Training paths never set the flag.
		if getattr(self, '_ug_sentinel_pricing', False) and _sentinel_tainted:
			for _ti in _sentinel_tainted:
				lats_by_ug_arr[_ti] = NO_ROUTE_LATENCY

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
			if kwargs.get('is_mlu_obj'):
				# max_util training scalar: soft latency + A_mean * MLU
				# (mean-scale, soft-bounded -- Tom's gradient-stability
				# ruling; MLU from the solution's link loads)
				_mlu = 0.0
				_npi = NO_PATH_INGRESS(self)
				for pi, v in vols_by_poppi.items():
					if pi != _npi and self.static_caps[pi] > 0:
						_mlu = max(_mlu, v / self.static_caps[pi])
				self._mlu_lp_weight()   # ensures _mlu_floor_mean
				_A = (float(os.environ.get('SCULPTOR_MLU_WEIGHT_MULT',
										   '10'))
					  * self._mlu_floor_mean)
				_objective -= _A * _mlu
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
			min_lbx = np.maximum(-.1,-1 * NO_ROUTE_PENALTY_MS * max_vol / total_deployment_volume)
		else:
			min_lbx = np.maximum(-.1,-1 * NO_ROUTE_PENALTY_MS * max_vol / total_deployment_volume)

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
		_dbg_pc0 = len(getattr(self, 'pattern_cache', {}))

		# --- 1. Initialize Containers ---
		# Instead of nested dicts, we build the flat lists required for vectorization directly.
		self.rti_data = {
			"meta_data": [],  # legacy (sim_rti_better); combined path uses block_meta
			"all_probs": [],  # legacy per-scenario lists (sim_rti_better)
			"all_poppis": [],
			"blocks": [],     # [(lengths:int16[], choices_pad:int16[nxm])]
			"block_meta": [], # [(pref_i, [ug names in block-row order])]
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

			# --- CACHE CHECK ---
			key = np.packbits(col).tobytes()
			if key in self.pattern_cache:
				# HIT: compact padded entry (uis:int32, lengths:int16,
				# choices_pad:int16[n_scen x max_n], -1 padded, ug-name
				# list). Appended as a BLOCK; the per-ui meta_data tuple
				# loop that used to run here (one append per ug per
				# active prefix per call) is gone -- names are cached
				# with the pattern, prefix ids ride per-block
				# (Tom 2026-08-25 loop elimination).
				uis_e, lens_e, pad_e, names_e = self.pattern_cache[key]
				self.rti_data["blocks"].append((lens_e, pad_e))
				self.rti_data["block_meta"].append((pref_i, names_e, uis_e))
				continue

			# --- CACHE MISS: Calculate Logic (VECTORIZED, Tom 2026-08-19
			# startup_optimizations: this per-UG python loop was 23% of
			# every cold solve at actual-25 per the line-level flamegraph)
			# One-time CSR of ui -> potential popps; per miss the valid
			# set is a boolean fancy-index + reduceat, no per-UG python.
			active_poppis = np.where(col)[0]
			active_poppis_set = set(active_poppis)

			if not hasattr(self, '_uipop_csr'):
				_flat, _offs = [], [0]
				for _ui in range(self.whole_deployment_n_ug):
					_flat.extend(ui_to_poppi[_ui])
					_offs.append(len(_flat))
				self._uipop_csr = (
					np.asarray(_flat, dtype=np.int32),
					np.asarray(_offs, dtype=np.int64))
			_flat, _offs = self._uipop_csr

			# Blocked (ui, child) pairs from active parents. Compact path
			# (SCULPTOR_COMPACT_PT, Tom 2026-08-20): _pt_csr groups rows by
			# parent, so we touch only ACTIVE parents' rows — the legacy
			# python scan walked EVERY (ug,child,parent) entry per miss,
			# which scales ~n_ug*n_popp^2 with measurements (354MB and
			# millions of entries at actual-25 late-run).
			# blocked (ui, child) pairs as ARRAYS -- the per-row python
			# set-build was the top consumer of pmat_organize on misses
			# (Tom 2026-08-25 loop elimination); duplicates are fine, the
			# mask clear below is idempotent.
			blocked_rows = None
			_csr = getattr(self, '_pt_csr', None)
			if _csr is not None:
				_pt_parents, _pt_offs, _pt_rows = _csr
				if _pt_parents.shape[0]:
					_j_act = np.nonzero(col[_pt_parents])[0]
					if _j_act.size:
						blocked_rows = np.concatenate(
							[_pt_rows[_pt_offs[_j]:_pt_offs[_j + 1]]
							 for _j in _j_act])
			else:
				# legacy dict path (SCULPTOR_COMPACT_PT=0 or no update yet)
				_pairs = [
					(self.whole_deployment_ug_to_ind[ug],
					 self.popp_to_ind[child])
					for ug, child, parent in self.parent_tracker
					if self.popp_to_ind[parent] in active_poppis_set]
				if _pairs:
					blocked_rows = np.asarray(_pairs, dtype=np.int64)

			# 3. Build routing for this state — fully vectorized.
			mask = col[_flat]                     # popp physically up
			if blocked_rows is not None and blocked_rows.shape[0]:
				# one searchsorted against a once-per-deployment sorted
				# (ui, child) -> flat-position index replaces the
				# per-pair python segment scan
				if not hasattr(self, '_uipop_poskey'):
					_ui_of_pos = np.repeat(
						np.arange(self.whole_deployment_n_ug,
								  dtype=np.int64),
						np.diff(_offs))
					_poskey = _ui_of_pos * self.n_popp + _flat
					_order = np.argsort(_poskey, kind='stable')
					self._uipop_poskey = (_poskey[_order], _order)
				_sk, _order = self._uipop_poskey
				_bk = (blocked_rows[:, 0].astype(np.int64) * self.n_popp
					   + blocked_rows[:, 1])
				_idx = np.minimum(np.searchsorted(_sk, _bk),
								  max(len(_sk) - 1, 0))
				_valid = _sk[_idx] == _bk if len(_sk) else _idx[:0].astype(bool)
				mask[_order[_idx[_valid]]] = False
			lens_all = np.add.reduceat(
				mask.astype(np.int16),
				_offs[:-1].clip(max=max(len(_flat) - 1, 0)))
			# reduceat quirk: empty segments (offs[i]==offs[i+1]) copy the
			# next element — zero them explicitly
			_empty = (_offs[1:] - _offs[:-1]) == 0
			if _empty.any():
				lens_all[_empty] = 0
			keep = lens_all > 0
			uis_e = np.where(keep)[0].astype(np.int32)
			lens_e = lens_all[keep].astype(np.int16)
			n_scen = int(uis_e.shape[0])
			valid_flat = _flat[mask]
			max_n = int(lens_e.max()) if n_scen else 0
			pad_e = np.full((n_scen, max_n), -1, dtype=np.int16)
			if n_scen:
				_colidx = np.arange(max_n)[None, :] < lens_e[:, None]
				pad_e[_colidx] = valid_flat
			# materialize ug names ONCE per pattern (cached); no per-call
			# meta_data tuple loop
			names_e = [ugs[_ui] for _ui in uis_e]
			self.pattern_cache[np.packbits(col).tobytes()] = (
				uis_e, lens_e, pad_e, names_e)
			self.rti_data["blocks"].append((lens_e, pad_e))
			self.rti_data["block_meta"].append((pref_i, names_e, uis_e))

		self.timing['pmat_organize'] += time.time() - ts_total
		if LP_SOLVE_DEBUG and getattr(self, 'worker_i', -1) in (0, 'drv'):
			print('[lpdbg-pc] w={} pattern_misses={} pattern_total={} '
				  't_scen={:.3f}s'.format(
					  getattr(self, 'worker_i', 'drv'),
					  len(self.pattern_cache) - _dbg_pc0,
					  len(self.pattern_cache), time.time() - ts_total),
				  flush=True)

	def _sample_scenario_realizations(self):
		"""Monte-carlo draw of self.MC_NUM joint route realizations from the
		scenario options in self.rti_data (populated by
		_compute_scenario_options). Returns routed_through_ingress:
		{mc_index: {prefix: {ug: popp}}}."""
		# --- 3. Vectorized Simulation (Previously sim_rti_better) ---
		# Now self.rti_data is fully populated. We proceed with the vectorized selection.

		blocks = self.rti_data.get("blocks") or []
		if blocks:
			# Compact-block path (pattern-cache producer, Tom 2026-08-19):
			# assemble padded matrices with vectorized block copies.
			# Arithmetic matches the legacy path exactly — P rows are the
			# same uniform [1/n]*n float64 values, cumsum'd identically.
			lens_all = np.concatenate([b[0] for b in blocks]).astype(np.int64)
			self.rti_data["num_scenarios"] = int(lens_all.shape[0])
			if self.rti_data["num_scenarios"] == 0:
				return {}
			self.rti_data["max_choices"] = int(lens_all.max())
			P_matrix = np.zeros((self.rti_data["num_scenarios"], self.rti_data["max_choices"]))
			self.rti_data["choices_matrix"] = np.full(
				(self.rti_data["num_scenarios"], self.rti_data["max_choices"]), -1, dtype=int)
			row = 0
			for lens_e, pad_e in blocks:
				nrow, ncol = pad_e.shape
				self.rti_data["choices_matrix"][row:row + nrow, :ncol] = pad_e
				row += nrow
			mask = (np.arange(self.rti_data["max_choices"])[None, :]
					< lens_all[:, None])
			P_matrix[mask] = np.repeat(1.0 / lens_all, lens_all)
		else:
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
		# Per-BLOCK dict(zip(names, popps)) at C speed replaces the
		# per-(scenario x MC) python loop (Tom 2026-08-25 loop
		# elimination). Same nested shape, same insertion order.
		if not hasattr(self, '_popps_arr'):
			self._popps_arr = np.empty(len(self.popps), dtype=object)
			self._popps_arr[:] = self.popps
		block_meta = self.rti_data.get("block_meta")
		routed_through_ingress = {}
		if block_meta:
			for mci in range(self.MC_NUM):
				per_pref = {}
				row = 0
				for pref_i, names_e, _uis_e in block_meta:
					nrow = len(names_e)
					per_pref[pref_i] = dict(zip(
						names_e,
						self._popps_arr[
							selected_poppis[row:row + nrow, mci]].tolist()))
					row += nrow
				routed_through_ingress[mci] = per_pref
			# ARRAY HANDOFF (Tom 2026-08-25): the (ug_idx, popp_idx)
			# pairs the dicts were built FROM, stashed per-realization.
			# get_paths_by_ug's batched path recognizes these dicts by
			# identity and skips the name->index reconversion entirely.
			# Strong dict refs pin object ids until the next sample.
			_u_flat = (np.concatenate([m[2] for m in block_meta])
					   .astype(np.int64))
			self._rti_flat = [
				(routed_through_ingress[mci], _u_flat,
				 selected_poppis[:, mci].astype(np.int64))
				for mci in range(self.MC_NUM)]
		else:
			# legacy meta_data path (sim_rti_better producer)
			for i, (ui, pref_i, ug_name) in enumerate(self.rti_data["meta_data"]):
				simulated_routes = selected_poppis[i]
				for mci, poppi in enumerate(simulated_routes):
					if mci not in routed_through_ingress:
						routed_through_ingress[mci] = {}
					if pref_i not in routed_through_ingress[mci]:
						routed_through_ingress[mci][pref_i] = {}
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
			if ((obj == "avg_latency" or obj == "per_site_cost"
					or obj == "max_util")
					and os.environ.get('SCULPTOR_LP_FORCE_NONPERSISTENT',
									   '0') != '1'):
				# SCULPTOR_LP_FORCE_NONPERSISTENT=1 (2026-08-24 bench knob):
				# route to the fresh-model path below -- a small model built
				# from only the ACTIVE columns, vs the persistent pool that
				# drags every ever-minted column (215k at actual-20) through
				# each factorization. Diagnostic for the optimize-dominates
				# finding; harmless in production (default off).
				ts = time.time()
				total_obj = self.solve_generic_lp_persistent(routed_through_ingress, obj)["objective"]
				self.timing['solve_generic_lp_persistent'] += time.time() - ts
				# COUNT, not just time. MC_NUM being set on the actor proves
				# only that an attribute holds a number; this proves the MC
				# loop actually runs that many LP solves per benefit call.
				self.n_lp_solves = getattr(self, 'n_lp_solves', 0) + 1
			else:
				ts = time.time()
				# Pass the adv matrix `a` through so multi-LP objectives
				# (static_failure, backup_capacity) can recover it. Plain
				# objectives ignore the kwarg.
				total_obj = solve_generic_lp_with_failure_catch(self, routed_through_ingress, obj, adv=a)['objective']
				self.timing['solve_generic_lp_not_persistent'] += time.time() - ts
			# Non-finite objective (e.g. maxhard prio: stage-2 bulk LP
			# infeasible in-worker returns objective=None) previously
			# poisoned the histogram (autodetected range [nan, nan]) ->
			# worker 'ERROR' -> driver concat TypeError -> cell death
			# (2026-08-18, 162/180 maxhard-v2 prio cells). Infeasible
			# realizations get a pessimistic finite benefit instead.
			objs[i] = total_obj if (total_obj is not None
					and np.isfinite(total_obj)) else np.nan
		if np.isnan(objs).any():
			_finite = objs[~np.isnan(objs)]
			_worst = (float(_finite.min()) - 1.0) if len(_finite) else -1e4
			objs[np.isnan(objs)] = _worst
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

	def dump_mem_components(self, tag='manual'):
		"""SCULPTOR_WORKER_MEMPROF=1: log deep sizes of the worker's big
		attributes (Tom 2026-08-19 RAM-attribution ask). One `[memprof]`
		line per component >50MB, same sink as _log_mem_worker. Cheap
		enough to call at update_deployment_done / cache-clear points
		only — NOT per lb call."""
		if os.environ.get('SCULPTOR_WORKER_MEMPROF', '0') != '1':
			return
		import sys as _sys
		def _sz(o, seen):
			if id(o) in seen:
				return 0
			seen.add(id(o))
			if isinstance(o, np.ndarray):
				return o.nbytes
			s = _sys.getsizeof(o, 0)
			if isinstance(o, dict):
				s += sum(_sz(k, seen) + _sz(v, seen) for k, v in o.items())
			elif isinstance(o, (list, tuple, set, frozenset)):
				s += sum(_sz(x, seen) for x in o)
			return s
		comps = {}
		for name in ('calc_cache', 'ingress_probabilities', 'ug_perfs',
					 'whole_deployment_ug_perfs', 'measured_prefs',
					 'parent_tracker', 'pattern_cache',
					 'this_time_ip_cache', 'ingress_priorities',
					 'popp_by_ug_indicator', 'linear_prog_soln_cache'):
			try:
				v = getattr(self, name)
			except AttributeError:
				continue
			if name == 'calc_cache':
				for ck, cv in getattr(v, 'all_caches', {}).items():
					b = _sz(cv, set())
					if b > 5e7:
						comps['calc_cache.' + str(ck)] = b
				continue
			b = _sz(v, set())
			if b > 5e7:
				comps[name] = b
		for name, b in sorted(comps.items(), key=lambda kv: -kv[1]):
			_log_mem_worker(self.worker_i, 'memprof_' + tag,
							component=name,
							size_mb=int(b / 1048576))

	def clear_new_meas_caches(self):
		# print("Clearing caches in worker {}".format(self.worker_i))
		self.dump_mem_components('pre_clear')
		self.this_time_ip_cache = {}
		self.pattern_cache = {}
		self.calc_cache.clear_new_measurement_caches()

	# The ZMQ command dispatcher (`check_for_commands`), the worker run loop
	# (`run`), and the `if __name__ == "__main__":` script entrypoint that
	# used to live here were removed when the project went Ray-only. The
	# equivalent dispatch for the Ray actor lives in
	# path_distribution_computer._LocalPathDistributionComputer as a set
	# of `_cmd_*` methods routed via `handle_msg`. See git history for the
	# old ZMQ implementation.

########################################################################
# Ray actor layer (merged in from path_distribution_computer.py,
# 2026-08-21). It was always a subclass of the class above plus the
# _cmd_* dispatch surface, so the second module only bought a second
# import path. Three names live here now:
#
#   Path_Distribution_Computer        the plain compute class (above)
#   _LocalPathDistributionComputer    same, plus the _cmd_* handlers,
#                                     runnable in-process (tests use this)
#   Path_Distribution_Computer_Actor  the @ray.remote wrapper
#
# The actor used to ALSO be called Path_Distribution_Computer, which only
# worked because it lived in a separate module. Renamed on merge.
########################################################################
class _LocalPathDistributionComputer(Path_Distribution_Computer):
	"""Same body as the Ray actor below, but without the @ray.remote decorator.

	Exposed so unit tests can instantiate a worker directly (no Ray, no ZMQ)
	and call methods like a normal Python object. The decorated Ray version
	(`Path_Distribution_Computer`) is defined at the bottom of this file by
	wrapping this class.
	"""

	def __init__(self, worker_i, deployment, init_kwargs):
		# Replicate the non-ZMQ portion of the original __init__ (lines 42-62
		# of path_distribution_computer.py). Skip start_connection / run.
		#
		# `deployment` is the FULL deployment (one shared plasma entry,
		# ray.put once by worker_comms.start_workers and
		# auto-dereferenced by Ray at actor-init time). Every worker
		# computes gradient jobs over the entire deployment.
		self.worker_i = worker_i
		self.port = 0  # unused under Ray
		self.logging_iter = 0
		self.timing = {k: 0 for k in [
			'solve_unified_lp_not_optimize', 'optimize', 'get_paths_by_ug',
			'organizing_results', 'get_ingress_probabilities_by_dict_generic',
			'sim_rti', 'total_rti_calc', 'pmat_organize',
			'solve_generic_lp_persistent',
			'solve_generic_lp_not_persistent']}
		self.rti_data = {}
		# SCULPTOR_MC_NUM was DEAD on this path until 2026-08-21: this class
		# replicates the base __init__ rather than calling it (it goes
		# straight to Optimal_Adv_Wrapper.__init__), so the env read at the
		# top of Path_Distribution_Computer.__init__ never ran for the
		# production Ray actor. cluster/chain_v3.sh has been passing
		# SCULPTOR_MC_NUM=1 into runs that silently used 5. Same shape as
		# the GRAD_SCALE dead seam.
		#
		# Default is now 1 (Tom, 2026-08-21). NOTE this is not only a speed
		# knob: MC_NUM is the number of Monte-Carlo draws of the routing
		# distribution, so 1 is a single-draw noisy estimator, not a
		# cheaper way to compute the same number.
		# init_kwargs wins over the env -- see the note in get_init_kwa:
		# the env does not survive the driver -> Ray actor boundary.
		# Saved for the full-rebirth path in _cmd_update_deployment (before
		# the pop below mutates it). _cmd_update_kwa refreshes it, so a
		# rebirth always uses the newest kwa the driver has sent.
		self._saved_init_kwa = copy.deepcopy(init_kwargs)
		self.MC_NUM = int(init_kwargs.pop(
			'mc_num', os.environ.get('SCULPTOR_MC_NUM', DEFAULT_MC_NUM)))
		if worker_i == 0:
			# Announce the value the ACTOR actually uses. The env read used
			# to live only in the base __init__, which this class bypasses,
			# so SCULPTOR_MC_NUM silently had no effect -- printing it is
			# how an A/B can prove the knob is live rather than assuming it.
			print('[mc] worker {} MC_NUM={} (env SCULPTOR_MC_NUM={!r})'.format(
				worker_i, self.MC_NUM,
				os.environ.get('SCULPTOR_MC_NUM')), flush=True)

		# Construct the optimization wrapper directly with the deployment
		# and init kwargs supplied by Worker_Manager (no ZMQ handshake).
		Optimal_Adv_Wrapper.__init__(self, deployment, **init_kwargs)

		# Open the per-worker log file the same way the original did.
		log_path = os.path.join(
			LOG_DIR, 'worker_{}_log-{}.txt'.format(self.worker_i, self.dpsize))
		with open(log_path, 'w'):
			pass

		if os.environ.get('SCULPTOR_STARTUP_TIMELOG') == '1':
			import helpers.timelog as timelog
			timelog.activate()
		self.init_all_vars()
		# Note: no run() loop and no main_socket. Ray dispatches method calls.
		# Equivalent of the ZMQ version's `worker_proc_start` mem-snapshot,
		# fired at the end of actor __init__ (post init_all_vars so the
		# Gurobi shell + Optimal_Adv_Wrapper state are loaded).
		_log_mem_worker(self.worker_i, 'worker_proc_start',
		                dpsize=getattr(self, 'dpsize', '?'))

	# ------------------------------------------------------------------ #
	# Pickled-tuple dispatcher. Worker_Manager calls this so the
	# existing send_receive_* / send_messages_workers API stays unchanged.
	# ------------------------------------------------------------------ #
	def handle_msg(self, msg_bytes):
		_malloc_trim()   # ~ms; keeps the arena ratchet from compounding
		try:
			cmd, data = pickle.loads(msg_bytes)
		except Exception as e:
			print("Worker {} failed to unpickle msg: {}".format(self.worker_i, e))
			return "ERROR"
		method = getattr(self, '_cmd_' + cmd, None)
		if method is None:
			print("Invalid CMD in worker {} : {}".format(self.worker_i, cmd))
			return "ERROR"
		try:
			return method(data)
		except Exception as e:
			# Diagnosability contract (Tom 2026-08-21, after the actual-32
			# smoke burned a VM-hour on an un-diagnosable
			# "KeyError: ('vtrwarsaw','9009')"): the FIRST occurrence of a
			# given (cmd, type, message) signature on this worker always
			# prints its full stack -- the 2026-08-19 log cleanup made the
			# stack opt-in via SCULPTOR_VERBOSE_ERRORS, which meant the one
			# copy of the traceback we needed was never written. Repeats of
			# the same signature collapse to a counted one-liner, so a
			# per-iteration error still can't flood the log.
			# SCULPTOR_VERBOSE_ERRORS=1 forces a stack on EVERY occurrence.
			import traceback
			tb = traceback.format_exc()
			sig = (cmd, type(e).__name__, str(e))
			try:
				self._err_seen
			except AttributeError:
				self._err_seen = {}
			n = self._err_seen.get(sig, 0) + 1
			self._err_seen[sig] = n
			stamp = time.strftime('%H:%M:%SZ', time.gmtime())
			if n == 1 or os.environ.get('SCULPTOR_VERBOSE_ERRORS') == '1':
				print("[{}] Worker {} ERROR cmd '{}': {}: {}\n{}".format(
					stamp, self.worker_i, cmd, type(e).__name__, e, tb),
					flush=True)
			else:
				print("[{}] Worker {} ERROR cmd '{}': {}: {} (x{}, stack above)".format(
					stamp, self.worker_i, cmd, type(e).__name__, e, n), flush=True)
			# Ship the stack back to the driver too. Ray dedups worker stdout
			# across the cluster ("[repeated 8x]"), so the driver-side
			# RuntimeError may be the only copy that survives into the log.
			# Still a str, so the driver's isinstance(ret, str) check holds.
			return "ERROR\n" + tb

	# ------------------------------------------------------------------ #
	# Command handlers. One per branch of the original if/elif dispatch
	# in check_for_commands(). Each takes the `data` payload and returns
	# the value that the original sent back over ZMQ.
	# ------------------------------------------------------------------ #

	def _cmd_calc_lb(self, data):
		ret = []
		self.this_time_ip_cache = {}
		for (args, kwargs) in data:
			ret.append(self.latency_benefit(*args, **kwargs))
		# Original did `del self.this_time_ip_cache`; reset to {} is equivalent
		# and avoids AttributeError if anything still references it.
		self.this_time_ip_cache = {}
		return ret

	def _cmd_solve_lp(self, data):
		ret = []
		ts = time.time()
		self.check_load_rw_measure_wrapper()
		n_iters = 0
		for fields in sorted(data, key=lambda el: el[0]):
			if len(fields) == 4:
				adv_i, adv, deployment, update_dep = fields
			else:
				adv_i, adv, opt_adv, deployment, update_dep = fields
			if update_dep:
				deployment_save = self.output_deployment()
				self.clear_caches()
				self.update_deployment(deployment, quick_update=True,
					verb=False, exit_on_impossible=False)
			if os.environ.get('SCULPTOR_VOLDEBUG') == '1':
				_ug0 = self.whole_deployment_ugs[0]
				print('[voldebug] adv_i={} update_dep={} attr_vol={:.4f} rhs={:.4f} dep_vol={:.4f}'.format(
					adv_i if len(fields) == 4 else fields[0], update_dep,
					float(self.whole_deployment_ug_to_vol[_ug0]),
					float(self.vol_constrs[_ug0].RHS) if hasattr(self, 'vol_constrs') else -1,
					float(deployment['whole_deployment_ug_to_vol'][_ug0])), flush=True)
			self.check_load_rw_measure_wrapper()

			rti, _ = self.calculate_ground_truth_ingress(adv, do_cache=False)
			this_ret = solve_generic_lp_with_failure_catch(
				self, rti, deployment.get('generic_objective'), adv=adv)
			if update_dep:
				self.update_deployment(deployment_save, quick_update=True,
					verb=False, exit_on_impossible=False)
				self.check_load_rw_measure_wrapper()
			ret.append((adv_i, this_ret))
			n_iters += 1
		return ret

	def _cmd_calc_compressed_lb(self, data):
		# Reset self.timing accumulators at the top of each batch so the
		# end-of-batch summarize_timing() shows per-batch cumulative LP-
		# solve breakdown (not lifetime-cumulative or last-call-only). The
		# inner solve_unified_lp etc. now += into self.timing.
		for k in self.timing:
			self.timing[k] = 0.0
		ts = time.time()
		tlp = time.time()
		ret = []
		base_args, base_kwa = data[0]
		base_adv, = base_args
		base_adv = base_adv.astype(bool)
		ret.append({'ans': self.latency_benefit(base_adv, **base_kwa),
			'job_id': base_kwa.get('job_id', -1)})
		i = 0
		last_timing_summary = 0
		for diff, kwa in data[1:]:
			kwa['verbose_workers'] = (base_kwa.get('verbose_workers', False)
				or kwa.get('verbose_workers', False))
			for ind in zip(*diff):
				base_adv[ind] = not base_adv[ind]
			ret.append({'ans': self.latency_benefit(base_adv, **kwa),
				'job_id': kwa.get('job_id', -1)})
			for ind in zip(*diff):
				base_adv[ind] = not base_adv[ind]
			i += 1
			if time.time() - tlp > 100:
				self.print("[{}] {} pct. done calcing latency benefits, {}ms per iter".format(
					time.strftime("%H:%M:%SZ", time.gmtime()),
					round(i * 100.0 / len(data), 1),
					round(1000 * (time.time() - ts) / i)))
				tlp = time.time()
			# (intermediate i%50 summarize_timing trigger removed — the
			# end-of-batch summary below now always fires, giving one clean
			# per-batch breakdown instead of accumulator-growing intermediate
			# snapshots.)
			self.check_clear_cache()
		# Always emit one cumulative summary at end of each batch so that
		# fine-grained parallelism (e.g. 132 perms ÷ 64 actors = 2 per actor)
		# doesn't silently skip the i%50 trigger above. Worker 0 alone (the
		# Ray-dedup'd "[repeated Nx]" message handles the rest).
		if self.worker_i == 0 and len(data) > 1:
			self.summarize_timing()
		return ret

	def _cmd_reset_new_meas_cache(self, _data=None):
		self.clear_new_meas_caches()
		return "ACK"

	def _cmd_solve_lp_volscen(self, data):
		# SCULPTOR_EVAL_VOLSCEN fast path (Tom 2026-08-20): a batch of
		# eval scenarios that differ from the loaded deployment ONLY in
		# ug volumes and/or link capacities (diurnal / flash-crowd /
		# volume-multiplier phases). Volumes don't affect ingress
		# selection, so rti is computed ONCE per adv; each scenario then
		# swaps in its vol/cap vectors directly — no per-scenario
		# update_deployment + clear_caches, no full-deployment pickles.
		# Uses the SAME LP builder as _cmd_solve_lp, so per-scenario LP
		# results are exactly identical to the legacy path.
		adv, obj, scenarios = data
		self.check_load_rw_measure_wrapper()
		rti, _ = self.calculate_ground_truth_ingress(adv, do_cache=False)
		save_vols = self.whole_deployment_ug_vols
		save_map = self.whole_deployment_ug_to_vol
		save_caps = self.link_capacities_arr
		save_caps_full = getattr(self, '_link_capacities_full', None)
		save_static = getattr(self, 'static_caps', None)
		# scenario volumes must ALSO reach the persistent LP model's
		# per-ug volume constraints — solve_generic_lp_with_failure_catch
		# prefers the persistent model, whose RHS was baked at build time
		# (missed on the first pass; caught by the eval exactness A/B:
		# arm B solved with stale volumes).
		self._ug_sentinel_pricing = True
		_have_model = hasattr(self, 'model') and hasattr(self, 'vol_constrs')
		if _have_model:
			_ordered_constrs = [self.vol_constrs[ug]
								for ug in self.whole_deployment_ugs]
		ret = []
		try:
			for adv_i, vol_vec, cap_arr in scenarios:
				if vol_vec is not None:
					self.whole_deployment_ug_vols = vol_vec
					self.whole_deployment_ug_to_vol = {
						ug: v for ug, v in zip(self.whole_deployment_ugs, vol_vec)}
					if _have_model:
						for _c, _v in zip(_ordered_constrs, vol_vec):
							_c.RHS = float(_v)
				if cap_arr is not None:
					self.link_capacities_arr = cap_arr
					self._link_capacities_full = np.concatenate(
						[cap_arr.flatten(), [1000000.0]])
					self.static_caps = self._compute_static_caps()
				if os.environ.get('SCULPTOR_VOLDEBUG') == '1':
					_ug0 = self.whole_deployment_ugs[0]
					print('[voldebug-vs] adv_i={} vol={:.4f} rhs={:.4f}'.format(
						adv_i, float(self.whole_deployment_ug_to_vol[_ug0]),
						float(self.vol_constrs[_ug0].RHS)), flush=True)
				this_ret = solve_generic_lp_with_failure_catch(
					self, rti, obj, adv=adv)
				ret.append((adv_i, this_ret))
		finally:
			self._ug_sentinel_pricing = False
			self.whole_deployment_ug_vols = save_vols
			self.whole_deployment_ug_to_vol = save_map
			self.link_capacities_arr = save_caps
			if save_caps_full is not None:
				self._link_capacities_full = save_caps_full
			if save_static is not None:
				self.static_caps = save_static
			if _have_model:
				for _c, _v in zip(_ordered_constrs, save_vols):
					_c.RHS = float(_v)
		return ret

	def _cmd_update_parent_tracker(self, parents_on):
		for ug in parents_on:
			for beaten_ingress, routed_ingress in parents_on[ug]:
				self.parent_tracker[ug, beaten_ingress, routed_ingress] = True
		if len(parents_on) > 0:
			self.clear_new_meas_caches()
		return "ACK"

	def _cmd_update_parent_tracker_csr(self, payload):
		# SCULPTOR_COMPACT_PT path: payload is either an ObjectRef to (or
		# directly) the (parents, offsets, rows, nonempty) CSR from
		# _encode_parents_on_csr. Zero-copy per node via plasma; replaces
		# the per-worker string-tuple dict (354MB/worker at actual-25).
		parents, offsets, rows, nonempty = payload
		self._pt_csr = (parents, offsets, rows)
		if nonempty:
			self.clear_new_meas_caches()
		return "ACK"

	def _cmd_update_deployment(self, data):
		# The base-class update_deployment short-circuits its worker_manager
		# fan-out branch when self.worker_manager is unset (AttributeError
		# guard), which is the case inside a worker actor.
		deployment, kwargs = data
		_log_mem_worker(self.worker_i, 'update_deployment_enter',
		                dpsize=deployment.get('dpsize', '?'))
		_t_ud = time.time()
		if kwargs.get('quick_update', False):
			self.update_deployment(deployment, **kwargs)
		else:
			# FULL REBIRTH. The base update_deployment refreshes the
			# deployment dicts but not the derived state: the persistent
			# Gurobi LP (vol_constrs keyed by the OLD ug set, var_pool by
			# the OLD popp indices), the lbx grids sized to the OLD n_ug,
			# and every hasattr-guarded lazy cache (_uipop_csr, _pt_csr,
			# rti_data, parent_tracker, ...). Sim 0 of a multi-sim eval is
			# consistent because the actor is CONSTRUCTED with its
			# deployment; sims 1+ came through this path and dereferenced
			# the new deployment through old structures (KeyError in
			# _path_obj_coeffs / IndexError in _compute_scenario_options ->
			# 'worker N returned error' -> strategy sparse dropped, 19/20
			# sims on 2026-08-22; nsim=1 everywhere is why no smoke caught
			# it). Rebuilding piecemeal is how the bug regenerates -- the
			# only future-proof update is the same code path as
			# construction: wipe the instance and re-run __init__.
			wi = self.worker_i
			ikw = copy.deepcopy(self._saved_init_kwa)
			try:
				self.model.dispose()  # else one leaked Gurobi env per sim
			except Exception:
				pass
			self.__dict__.clear()
			self.__init__(wi, deployment, ikw)
		self._mark_init('update_deployment', time.time() - _t_ud)
		_log_mem_worker(self.worker_i, 'update_deployment_done',
		                dpsize=deployment.get('dpsize', '?'))
		# Right after the shard lands is the moment worth measuring: the
		# per-size state is fully built and nothing transient is inflating
		# it. No-op unless SCULPTOR_LOG_OBJSIZE=1.
		_log_objsize_worker(self.worker_i, 'post_update_deployment_dp{}'.format(
			deployment.get('dpsize', '?')), self)
		self.dump_mem_components('post_update_deployment')
		# Emit the one-time accounting here: by this point the deployment
		# has landed and any lazily-built state (rb_backups, persistent LP)
		# has been charged to init_timing rather than to batch #1.
		self.summarize_init_timing('post_update_deployment')
		return "ACK"

	def _cmd_dump_mem_log(self, _data=None):
		# Exposes the inherited dump_mem_log() to the cmd dispatcher so
		# Worker_Manager._collect_and_emit_worker_mem_logs can pull
		# per-worker mem files at end of run via the standard
		# send_receive_workers API.
		return self.dump_mem_log()

	def _cmd_get_node_mem_avail_mb(self, _data=None):
		# Exposes the inherited get_node_mem_avail_mb() so the driver
		# can probe worker-node memory via the standard cmd dispatch.
		# Used by Worker_Manager._maybe_autoscale to monitor worker-node
		# memory pressure (head and worker live on different boxes).
		return self.get_node_mem_avail_mb()

	def _cmd_update_kwa(self, new_kwa):
		# Keep the rebirth snapshot current: _cmd_update_deployment re-runs
		# __init__ with _saved_init_kwa, and the driver sends update_kwa
		# immediately before update_deployment each sim.
		try:
			self._saved_init_kwa.update(copy.deepcopy(new_kwa))
		except AttributeError:
			self._saved_init_kwa = copy.deepcopy(new_kwa)
		if new_kwa.get('n_prefixes') is not None:
			self.n_prefixes = new_kwa.get('n_prefixes')
		if new_kwa.get('gamma') is not None:
			self.gamma = new_kwa.get('gamma')
		if new_kwa.get('with_capacity') is not None:
			self.with_capacity = new_kwa.get('with_capacity')
		return "ACK"

	def _cmd_increment_iter(self, _data=None):
		self.increment_iter()
		return "ACK"

	def _cmd_set_iter(self, data):
		self.iter = data
		return "ACK"

	def _cmd_set_training_mode(self, data):
		# Inherited from Path_Distribution_Computer: toggles self._in_training
		# and pushes new RHS into the persistent Gurobi cap constraints so
		# SCULPTOR_CAPACITY_HEADROOM only applies during the gradient loop.
		self.set_training_mode(data)
		return "ACK"

	def _cmd_set_mc_num(self, data):
		# Explore-time MC override (Tom, 2026-08-14): the driver bumps
		# MC_NUM while evaluating explore candidates (entropy needs a real
		# distribution; training may run MC_NUM=1) and restores it after.
		self.MC_NUM = int(data)
		return "ACK"

	def _cmd_reset_cache(self, _data=None):
		self.clear_caches()
		return "ACK"

	def _cmd_init(self, _data=None):
		# Original called start_connection() to (re)bind a ZMQ socket. No-op for Ray.
		return "ACK"

	def _cmd_end(self, _data=None):
		# Cleanup hook. The Worker_Manager will follow up with ray.kill(actor).
		return "ACK"

	# Convenience: lets the master sanity-check that an actor is alive.
	def ping(self):
		return "ACK"


# Public Ray actor: the same class above wrapped with @ray.remote.
# worker_comms.Worker_Manager imports this (renamed on the
# 2026-08-21 merge; it used to be Path_Distribution_Computer).
#
# max_restarts=-1: if the actor's node is lost (e.g. AWS reclaims the spot
# worker), Ray may restart the actor on another live node instead of leaving
# it permanently dead. This is defence-in-depth — the Worker_Manager also does
# an app-level rebuild+retry on RayActorError (worker_comms._with_recovery),
# which is the primary recovery path and re-applies deployment state. We do NOT
# set max_task_retries: a transparently-retried task would run against a
# restarted actor that lost its post-__init__ state (update_deployment is not
# replayed by Ray), so we drive the retry at the app level instead.
Path_Distribution_Computer_Actor = ray.remote(
    num_cpus=1, max_restarts=-1)(_LocalPathDistributionComputer)


# ---------------------------------------------------------------- CLI --
# python core/path_distribution_computer.py --replay_example_load
# (Tom 2026-08-24): replay a realistic hot-loop workload against ONE
# in-process worker (no Ray) on a harvested deployment; print the phase
# latency table per scenario, then the per-attribute object-size census.
# The scenario engine lives in unit_tests/bench_path_distribution.py.
if __name__ == '__main__':
	import argparse as _ap
	_p = _ap.ArgumentParser()
	_p.add_argument('--replay_example_load', action='store_true')
	_p.add_argument('--replay_batch_load', action='store_true',
					help='drive _cmd_calc_compressed_lb like a VM RB/LB '
						 'flush; add --profile for a cProile split')
	_p.add_argument('--replay_realistic_load', action='store_true',
					help="Tom's A/B/C protocol: init-shaped base adv, "
						 'production single-flip pairs, repeated rounds '
						 'with --pct-new fresh perturbations + base '
						 'drift; rounds 2+ are the steady-state numbers')
	_p.add_argument('--indices', type=int, default=12,
					help='probed entries per batch (jobs = 1 + 2x this '
						 '+ --rb-rows)')
	_p.add_argument('--rounds', type=int, default=6)
	_p.add_argument('--pct-new', type=float, default=0.2)
	_p.add_argument('--drift', type=int, default=3,
					help='base entries toggled between rounds (the '
						 'applied gradient step); 0 = frozen base, '
						 'repeats become memo hits')
	_p.add_argument('--rb-rows', type=int, default=0,
					help='popp-row-zero jobs per batch (RB fan-out mix)')
	_p.add_argument('--obj', default='avg_latency')
	_p.add_argument('--no-clear-meas', action='store_true',
					help='skip the per-round measurement-cache clear '
						 '(production clears every iteration)')
	_p.add_argument('--no-seed-pt', action='store_true',
					help='skip seeding the parent tracker from the init '
						 'measurement (fresh-worker regime)')
	_p.add_argument('--jobs', type=int, default=24)
	_p.add_argument('--profile', action='store_true')
	_p.add_argument('--pickle', default='cache/popp_failure_latency_'
					'comparison_testing_feature-actual-20_dep_sweep_20.pkl')
	_p.add_argument('--steps', type=int, default=4)
	_p.add_argument('--flips-per-step', type=int, default=12)
	_p.add_argument('--scenarios', default='warm,cold,mlu_off,mlu_on,rebuild')
	_p.add_argument('--rebuild-every', type=int, default=15)
	_a = _p.parse_args()
	if not (_a.replay_example_load or _a.replay_batch_load
			or _a.replay_realistic_load):
		_p.error('pass --replay_example_load, --replay_batch_load or '
				 '--replay_realistic_load')
	import sys as _sys, os as _os
	_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
		_os.path.abspath(__file__))))
	_sys.argv = ['bench', '--pickle', _a.pickle,
				 '--scenarios', _a.scenarios,
				 '--rebuild-every', str(_a.rebuild_every)]
	import unit_tests.bench_path_distribution as _b
	_w = _b.build_worker(_a.pickle)
	if _a.replay_realistic_load:
		_b.realistic_rounds(_w, n_indices=_a.indices, n_rounds=_a.rounds,
							pct_new=_a.pct_new, drift=_a.drift,
							obj=_a.obj, rb_rows=_a.rb_rows,
							seed_pt=not _a.no_seed_pt,
							clear_meas=not _a.no_clear_meas,
							profile=_a.profile)
		print('\n== object-size census (top attributes) ==')
		_log_objsize_worker(0, 'replay_realistic_load', _w, top_n=15)
		raise SystemExit(0)
	if _a.replay_batch_load:
		_b.replay_batch(_w, n_jobs=_a.jobs, profile=False)      # warm pool
		_b.replay_batch(_w, n_jobs=_a.jobs, profile=_a.profile) # measured
		print('\n== object-size census (top attributes) ==')
		_log_objsize_worker(0, 'replay_batch_load', _w, top_n=15)
		raise SystemExit(0)
	_S = _b.scenario_table(_a)   # single source of truth for scenarios
	_res = {}
	for _n in _a.scenarios.split(','):
		_env, _objs, _reb = _S[_n.strip()]
		_res[_n] = _b.run_scenario(_w, _n, (_a.steps, _a.flips_per_step),
								   _env, _objs, _reb)
	print('\n== object-size census (top attributes) ==')
	_log_objsize_worker(0, 'replay_example_load', _w, top_n=15)
	_base = _res.get('warm')
	print('\n== summary (ms/solve) ==')
	for _k, _v in _res.items():
		print('  {:10s} {:7.0f}{}'.format(_k, _v * 1000,
			  '   ({:+.0f}% vs warm)'.format(100 * (_v - _base) / _base)
			  if _base and _k != 'warm' else ''))
