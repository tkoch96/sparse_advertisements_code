"""SCULPTOR: SGD-based BGP advertisement optimization.

This is the core of the codebase. The class hierarchy:

    Optimal_Adv_Wrapper                 (optimal_adv_wrapper.py)
      └─ Sparse_Advertisement_Wrapper   (this file, ~line 100)
            ├─ Sparse_Advertisement_Solver  (this file, ~line 920) — SGD
            └─ Sparse_Advertisement_Eval    (this file, near end) — eval entry

`Sparse_Advertisement_Eval` is the public entry-point used by drivers
(`eval_latency_failure.evaluate_all_metrics`, `experiments.run_objective`).
It instantiates a `Sparse_Advertisement_Solver`, calls `compare_different_solutions`
to run sparse + the baseline strategies (painter, anyopt, anycast, etc.),
and exposes the solved advertisement matrices to downstream eval phases.

`compare_different_solutions` (line ~760) is the orchestrator: forks
non-sparse strategies into a `ProcessPoolExecutor` (so they run
concurrently with sparse on the head), then runs sparse in the main
process backed by the Ray actor pool managed by `Worker_Manager`.

`Sparse_Advertisement_Solver.solve()` is the training loop (line ~2230).
Each iter: gradient probe → SGD step (with momentum, optional proximal
L1) → measure ingresses on changed prefixes → optional max-info probe →
stop-tracker check. Saves `state-N.pkl` checkpoints every 5 iters under
`runs/<ts>-<dpsize>-sparse/` so a crashed run can hot-start.

Hot loops to know about:
  - Gradient probing fan-out: `latency_benefit_fn` (line ~360) builds
    per-worker message lists and calls `wm.send_receive_messages_workers`
  - Adaptive worker resize: `Worker_Manager.process_pending_resize` is
    called at the top of each training iter (line ~2300); a watcher
    thread armed in `compare_different_solutions` triggers ramp-up
    when concurrent parallel strategies finish

See README.md "Architecture overview" for the bigger picture.
"""
import matplotlib.pyplot as plt, copy, time, numpy as np, itertools, pickle, warnings, tqdm, glob
import gpshim as gp  # gurobipy-subset facade; SCULPTOR_LP_BACKEND=gurobi(default)|highs
from subprocess import call, check_output
import concurrent.futures
import multiprocessing as _mp
import os as _os
import threading as _threading
np.setbufsize(262144*8)

# Memory instrumentation. Linux-only (reads /proc); silent on macOS. Gated by
# SCULPTOR_LOG_MEM (default ON) so this is opt-out for local debugging.
def _log_mem(tag, **extra):
	if _os.environ.get('SCULPTOR_LOG_MEM', '1') == '0':
		return
	rss_kb = vms_kb = peak_kb = sys_avail_kb = -1
	try:
		with open('/proc/self/status', 'r') as f:
			for line in f:
				if line.startswith('VmRSS:'):    rss_kb    = int(line.split()[1])
				elif line.startswith('VmSize:'): vms_kb    = int(line.split()[1])
				elif line.startswith('VmPeak:'): peak_kb   = int(line.split()[1])
		with open('/proc/meminfo', 'r') as f:
			for line in f:
				if line.startswith('MemAvailable:'):
					sys_avail_kb = int(line.split()[1]); break
	except (FileNotFoundError, PermissionError):
		return
	bits = [f'tag={tag}',
	        f'rss_mb={rss_kb//1024}',
	        f'vms_mb={vms_kb//1024}',
	        f'peak_mb={peak_kb//1024}',
	        f'sys_avail_mb={sys_avail_kb//1024}',
	        f'pid={_os.getpid()}',
	        f't={time.time():.2f}']
	for k, v in extra.items():
		bits.append(f'{k}={v}')
	print('[mem] ' + ' '.join(bits), flush=True)
np.set_printoptions(precision=3)
# np.random.seed(31415)
# import random
# random.seed(31416)
import scipy.stats
import sys
# from sklearn.mixture import GaussianMixture
# from sklearn.exceptions import ConvergenceWarning
# import warnings
np.set_printoptions(threshold=sys.maxsize)
from helpers import *
from constants import *
from painter import Painter_Adv_Solver
from anyopt import Anyopt_Adv_Solver
from optimal_adv_wrapper import Optimal_Adv_Wrapper
from worker_comms import Worker_Manager
from generic_objective import Generic_Objective

try:
	from eval_latency_failure import plot_lats_from_adv
except ImportError:
	pass

from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning

from deployment_setup import *


def compare_estimated_actual_per_user(dpsize):
	modeled_user_lats = {}
	for worker_log in glob.glob(os.path.join(LOG_DIR, 'worker*log-{}.txt'.format(dpsize))):
		for row in open(worker_log,'r'):
			if 'benefit_estimate' not in row: continue
			_,_,ui,bi,lb,p,popps_str,itr = row.strip().split(',')
			itr,ui,bi,lb,p = int(itr),int(ui),int(bi),float(lb),float(p)
			try:
				modeled_user_lats[itr]
			except KeyError:
				modeled_user_lats[itr] = {}
			try:
				modeled_user_lats[itr][ui].append((bi,lb,p))
			except KeyError:
				modeled_user_lats[itr][ui] = [(bi,lb,p)]
	actual_user_lats = {}
	for row in open(os.path.join(LOG_DIR, 'main_thread_log-{}.txt'.format(dpsize)),'r'):
		if 'benefit_estimate' not in row: continue
		_,itr,ui,poppi,pct,b = row.strip().split(',')
		itr,ui,poppi,pct,b = int(itr),int(ui),int(poppi),float(pct),float(b)
		try:
			actual_user_lats[itr]
		except KeyError:
			actual_user_lats[itr] = {}
		try:
			actual_user_lats[itr][ui].append((poppi,b,pct))
		except KeyError:
			actual_user_lats[itr][ui] = [(poppi,b,pct)]

	itrs = list(sorted(list(actual_user_lats)))
	uis = list(sorted(list(actual_user_lats[itrs[0]])))

	plt.rcParams["figure.figsize"] = (8,6)
	f,ax = plt.subplots()
	current_itr = itrs[-1]
	all_deltas = np.zeros((len(uis), len(itrs)))
	for ui in uis:
		for itr in itrs:
			modeled_user_lats[itr][ui] = sum(lb*p for _,lb,p in set(modeled_user_lats[itr][ui]))
			actual_user_lats[itr][ui] = sum(lb*p for _,lb,p in set(actual_user_lats[itr][ui]))
			if np.abs(modeled_user_lats[itr][ui] - actual_user_lats[itr][ui]) > .3 and itr == current_itr:
				print("ITR: {} User {} has modeled benefit {} but actual {}".format(itr, ui, modeled_user_lats[itr][ui], actual_user_lats[itr][ui]))
		these_modeled_lats = np.array([modeled_user_lats[itr][ui] for itr in itrs])
		these_actual_lats = np.array([actual_user_lats[itr][ui] for itr in itrs])
		deltas = np.abs(these_actual_lats - these_modeled_lats)
		all_deltas[ui,:] = deltas


	ax.plot(itrs, np.min(all_deltas, axis=0),label='Min')
	ax.plot(itrs, np.median(all_deltas, axis=0),label='Median')
	ax.plot(itrs, np.max(all_deltas, axis=0),label='Max')
	ax.legend()

	ax.set_xlabel("Iteration")
	ax.set_ylabel("Actual - Modeled Benefit")
	plt.savefig("figures/benefit_modeling_error.pdf")
	plt.clf()
	plt.close()

def investigate_congestion_events():
	return
	import glob
	link_failure_events = {}
	for worker_log in glob.glob(os.path.join(CACHE_DIR, 'worker*log.txt')):
		for row in open(worker_log,'r'):
			if 'link_fail_report' not in row: continue
			_,itr,ingress_i,failing_poppi,link_cap,vol_users,uis,p_fails = row.strip().split(',')
			if failing_poppi != 'none':
				failing_poppi = int(failing_poppi)
			itr,ingress_i,link_cap,vol_users,p_fails = int(itr),int(ingress_i),float(link_cap),float(vol_users),float(p_fails)
			uis = [int(el) for el in uis.split('-')]

			uid = (ingress_i,failing_poppi)

			try:
				link_failure_events[uid].append((itr, vol_users - link_cap))
			except KeyError:
				link_failure_events[uid] = [(itr,vol_users - link_cap)]
			max_itr = itr
	# for row in open(os.path.join(CACHE_DIR, 'main_thread_log.txt'),'r'):
	# 	if 'link_fail_report' not in row: continue
	# 	_,itr,ingress_i,failing_poppi,link_cap,vol_users,uis,p_fails = row.strip().split(',')
	# 	if failing_poppi != 'none':
	# 		failing_poppi = int(failing_poppi)
	# 	itr,ingress_i,link_cap,vol_users,p_fails = int(itr),int(ingress_i),float(link_cap),float(vol_users),float(p_fails)
	# 	uis = [int(el) for el in uis.split('-')]

	# 	uid = (ingress_i,failing_poppi)

	# 	try:
	# 		link_failure_events[uid].append((itr, vol_users - link_cap))
	# 	except KeyError:
	# 		link_failure_events[uid] = [(itr,vol_users - link_cap)]
	# 	max_itr = itr

	plt.rcParams["figure.figsize"] = (8,6)
	f,ax = plt.subplots()
	for uid in link_failure_events:
		x,y = [el[0] for el in link_failure_events[uid]],\
			[el[1] for el in link_failure_events[uid]]
		if max_itr - x[-1] < 5:
			# only label recent issues
			ax.plot(x,y,label='{} over, {} fails'.format(uid[0],uid[1]))
		else:
			ax.plot(x,y)

	ax.set_xlabel("Iteration")
	ax.set_ylabel("Excess Link Load")
	# ax.legend(fontsize=6)
	plt.savefig("figures/reported_failure_events_during_training.pdf")
	plt.clf()
	plt.close()


# Strategies that can run in an isolated subprocess (no Ray workers needed).
# Their solve_X methods only use local single-LP via solve_lp_assignment +
# pure-Python heuristics, so they parallelize with sparse cleanly.
#
# 'anyopt' is included even though its monte-carlo phase normally uses
# solve_lp_with_failure_catch_mp (which needs Ray workers): the mp method
# falls back to serial single-LP when worker_manager is None. Slower than
# the Ray path but lets anyopt overlap with sparse. At actual-N the 100-MC
# serial cost is ~100s of single-LP -- still a win against sparse running
# for hours in the main process.
_PARALLEL_STRATEGY_NAMES = frozenset({
	'painter', 'anyopt', 'one_per_pop', 'anycast', 'random', 'one_per_peering',
})


def _read_sys_avail_mb_local():
	"""Linux-only sys-avail read for the subprocess concurrency cap.
	Returns int MB or None when /proc isn't readable (macOS, restricted
	containers)."""
	try:
		with open('/proc/meminfo', 'r') as f:
			for line in f:
				if line.startswith('MemAvailable:'):
					return int(line.split()[1]) // 1024
	except (FileNotFoundError, PermissionError):
		return None
	return None


def _max_concurrent_strategies(n_parallel_types):
	"""Cap how many fork()'d strategy subprocesses run concurrently, based
	on current head free memory and a per-subprocess RSS estimate.

	Returns int >= 1. Falls back to n_parallel_types (no cap) when
	/proc/meminfo isn't readable (macOS) so local dev smoke runs aren't
	throttled artificially.
	"""
	avail_mb = _read_sys_avail_mb_local()
	if avail_mb is None:
		return n_parallel_types
	try:
		headroom_mb = int(_os.environ.get('SCULPTOR_HEAD_RAM_HEADROOM_MB', '8000'))
	except ValueError:
		headroom_mb = 8000
	try:
		per_est_mb = int(_os.environ.get('SCULPTOR_STRATEGY_RSS_ESTIMATE_MB', '8000'))
	except ValueError:
		per_est_mb = 8000
	usable = avail_mb - headroom_mb
	if usable <= 0 or per_est_mb <= 0:
		return 1
	return max(1, min(n_parallel_types, usable // per_est_mb))


def _solve_one_strategy_in_subprocess(strategy_name, deployment, init_kwa,
                                      kwargs_for_solve):
	"""Module-level worker for ProcessPoolExecutor.

	Each subprocess constructs its OWN Sparse_Advertisement_Eval without
	a worker_manager. The cheap strategies only need local LP (which uses
	in-process Gurobi via solve_lp_assignment, no Ray), so we sidestep the
	whole worker pool. Returns (strategy_name, solution_dict).
	"""
	# Defensive: never recurse into parallel mode inside a subprocess.
	_os.environ['SCULPTOR_DISABLE_PARALLEL_STRATEGIES'] = '1'

	# Re-import inside the fork-child to make sure we hit the module-state-
	# fresh code path. With fork start method this is essentially a no-op
	# since the imports are inherited, but it shields against spawn fallback.
	from sparse_advertisements_v3 import Sparse_Advertisement_Eval

	sas = Sparse_Advertisement_Eval(deployment, **init_kwa)
	sas.update_deployment(deployment)
	# solve_X methods write into sas.solutions, which is normally initialised
	# inside compare_different_solutions; we're bypassing that here so do it
	# manually.
	sas.solutions = {}

	solve_fns = {
		'painter': sas.solve_painter,
		'anyopt': sas.solve_anyopt,
		'one_per_pop': sas.solve_one_per_pop,
		'anycast': sas.solve_anycast,
		'random': sas.solve_random,
		'one_per_peering': sas.solve_one_per_peering,
	}
	if strategy_name not in solve_fns:
		raise ValueError("strategy {} not parallelizable".format(strategy_name))

	t0 = time.time()
	solve_fns[strategy_name](**kwargs_for_solve)
	elapsed = time.time() - t0
	soln = sas.solutions[strategy_name]
	soln['_subprocess_wall'] = elapsed
	return strategy_name, soln


class Sparse_Advertisement_Wrapper(Optimal_Adv_Wrapper):
	def __init__(self, *args, init={'type':'using_objective'}, explore='bimodality',
			using_resilience_benefit=False, **kwargs):
		super().__init__(*args, **kwargs)
		# (hyper-) parameters

		self.iter = 0
		# Whether we're inside the gradient / training loop. Gated by solve()
		# below; downstream LPs (driver-side _apply_capacity_headroom and the
		# worker-side persistent Gurobi) check this so SCULPTOR_CAPACITY_HEADROOM
		# only affects training, not the eval phase.
		self._in_training = False
		self.initialization = init
		self.explore = explore
		# Stop conditions: (a) hit max_n_iter, OR (b) all rolling deltas small.
		# el = [iter, rolling_delta, rolling_delta_eff, rolling_adv_delta]
		#
		# SCULPTOR_STOP_DROP_ADV_DELTA=1 drops the el[3] (rolling_adv_delta)
		# clause. The MAX over per-element adv changes is dominated by a few
		# bits oscillating near the L1-proximal threshold even at converged
		# state, so the EWMA stays above rolling_adv_eps=0.01 indefinitely.
		# Without that clause we tighten epsilon by 10x (default 0.005 -> 5e-4)
		# to compensate -- otherwise sparse stops at the edge of the epsilon
		# bound with measurable residual objective improvement still possible.
		# Empirical with the tightened threshold: sparse stops at iter ~70-100
		# (still vs MAX_ITER=200) with normal-LP quality matching the full
		# 200-iter baseline.
		# SCULPTOR_MIN_ITER: floor on training iters. The convergence clause
		# cannot fire before this iter, so a hot-started run that loaded an
		# already-converged state still trains forward to the floor. The hard
		# max_n_iter cap is unaffected. Default 0 == original behaviour.
		self._min_n_iter = int(_os.environ.get('SCULPTOR_MIN_ITER', '0') or 0)
		if _os.environ.get('SCULPTOR_STOP_DROP_ADV_DELTA', '0') == '1':
			_tight = 0.1
			self.stopping_condition = lambda el : el[0] > self.max_n_iter or (
				el[0] >= self._min_n_iter
				and el[1] < self.epsilon * _tight and np.abs(el[2]) < self.epsilon * _tight
			)
		else:
			self.stopping_condition = lambda el : el[0] > self.max_n_iter or (
				el[0] >= self._min_n_iter
				and el[3] < self.rolling_adv_eps and el[1] < self.epsilon and np.abs(el[2]) < self.epsilon)
		# SCULPTOR_STOP_RULE (default v2; merged from the ablation fork
		# 2026-08-16). The legacy rule above is dead code at georand scale:
		# absolute epsilon=.005 vs ~150-scale objectives leaves rolling_delta
		# 10-150x above threshold forever, and the adv-delta clause never
		# drops below .01 under churn. v2 fires when ALL of: grounded (a
		# recent real measurement backs the belief), rolling_delta < REL x
		# its own initial value (scale-free), and the believed best is
		# unimproved for PATIENCE iters. Replay+live validated (regret
		# <= 0.11ms; 500-cap smoke exited at 167 with full budget spent).
		# SCULPTOR_STOP_RULE=stock restores the legacy lambda.
		if _os.environ.get('SCULPTOR_STOP_RULE', 'v2') == 'v2':
			_rel = float(_os.environ.get('SCULPTOR_STOP_V2_REL', '0.03'))
			_pat = int(_os.environ.get('SCULPTOR_STOP_V2_PATIENCE', '20'))
			def _stop_v2(el, _self=self, _rel=_rel, _pat=_pat):
				it, rd = el[0], el[1]
				if it > _self.max_n_iter:
					return True
				if it < _self._min_n_iter:
					return False
				init = getattr(_self, '_rolling_delta_init', None)
				bi = getattr(_self, '_stopv2_best_iter', None)
				if not init or bi is None:
					return False
				pm = getattr(_self, 'abl_probe_mode', None)
				if pm == 'smart':
					grounded = it >= int(getattr(_self, 'abl_probe_tconv', 0))
				elif pm in ('gated', 'scheduled', 'adaptive', 'slotted'):
					grounded = (getattr(_self, 'abl_probes_spent', 0)
					            >= int(getattr(_self, 'abl_probe_n', 0)))
				else:
					grounded = True  # stock solver measures every iteration
				# patience also counts from the LAST MEASUREMENT (2026-08-16:
				# finishing the probe budget early must not hasten exit -- the
				# run must stay flat for a full window AFTER its final
				# verification, not merely after its last best)
				_lp = getattr(_self, '_abl_last_probe_iter', None)
				_post_ok = (_lp is None) or ((it - _lp) >= _pat)
				fire = grounded and _post_ok and rd < _rel * init and (it - bi) >= _pat
				if fire:
					_self.abl_exit_reason = 'stop_v2'
					print('[stop-v2] iter={} rd={:.4g} rd_init={:.4g} best_iter={} -> EARLY EXIT'.format(it, rd, init, bi), flush=True)
				return fire
			self.stopping_condition = _stop_v2
		## Whether to incorporate capacity into the objective function
		self.with_capacity = kwargs.get('with_capacity', False)
		### We might vary these functions depending on settings from time to time
		### but always aim to unify them after dev
		self.latency_benefit_fn = self.latency_benefit
		self.gradient_fn = self.gradients
		## Whether to incorporate resilience into the objective function
		# (Note if gamma = 0, this won't matter anyway)
		self.using_resilience_benefit = using_resilience_benefit
		if using_resilience_benefit:
			assert self.gamma > 0
			self.resilience_benefit_fn = self.resilience_benefit
			self.gradients_resilience_benefit_fn = self.gradients_resilience_benefit
		else:
			self.resilience_benefit_fn = lambda a : 0
			self.gradients_resilience_benefit_fn = lambda a : np.zeros(a.shape)

		self.proximal = True

		self.reset_metrics()
		if self.verbose:
			print("Creating problem with {} peers, {} prefixes, {} UGs.".format(self.n_popp, self.n_prefixes, len(self.ugs)))

		## Queue up calls to individual workers
		self.lb_args_queue = []

	def get_init_kwa(self):
		kwa =  {
			'lambduh': self.lambduh, 
			'gamma': self.gamma, 
			'with_capacity': self.with_capacity,
			'verbose': False,
			'init': self.initialization,
			'explore': self.explore,
			'using_resilience_benefit': self.using_resilience_benefit,
			'n_prefixes': self.n_prefixes,
			'save_run_dir': self.save_run_dir,
		}
		kwa['generic_objective'] = self.generic_objective.obj
		return kwa

	def reset_metrics(self):
		# For analysis
		self.metrics = {}
		for k in ['actual_nonconvex_objective', 'advertisements', 'effective_objectives', 
			'pseudo_objectives', 'grads', 'cost_grads', 'l_benefit_grads', 'res_benefit_grads',
			'path_likelihoods', 'EL_difference', 'resilience_benefit', 'latency_benefit',
			'gt_latency_benefit','gt_resilience_benefit', 'effective_gammas', 'link_utilizations', 'frac_latency_benefit_calls', 'frac_resilience_benefit_calls',
				'popp_rb_sample_probabilities']:
			self.metrics[k] = []

	def gradients(self, *args, **kwargs):
		pass

	def gradients_resilience_benefit(self,*args, **kwargs):
		pass

	def get_n_most_likely_peers_justsort(self, ug, available_peers, n=5, verb=False):
		sorted_available_peers = sorted(available_peers, key = rank_f)
		return sorted_available_peers[0:n]

	def compress_lb_args_queue(self, **kwargs):
		### Idea: first adv is the base, rest are deltas from the base
		### transmit the base and the deltas
		ugs = kwargs.get('ugs', None)
		is_verb = kwargs.get('verbose_workers', False)
		base_args, base_kwa = self.lb_args_queue[0]
		if ugs is not None:
			base_kwa['ugs'] = ugs
		base_kwa['verbose_workers'] = is_verb or base_kwa.get('verbose_workers',False)
		base_kwa['generic_obj'] = self.generic_objective.obj
		base_adv, = base_args

		base_adv = threshold_a(base_adv)
		base_args = (base_adv,)

		self.compressed_lb_args_queue = [(base_args, base_kwa)]
		for other_args, kwa in self.lb_args_queue[1:]:
			other_adv, = other_args
			other_adv = threshold_a(other_adv)
			if ugs is not None:
				kwa['ugs'] = ugs
			kwa['verbose_workers'] = is_verb or kwa.get('verbose_workers',False)
			kwa['generic_obj'] = self.generic_objective.obj
			self.compressed_lb_args_queue.append((np.where(base_adv!=other_adv), kwa))

	def flush_latency_benefit_queue_generic(self, **kwargs):
		"""
			For the generic objective, it makes more sense to split jobs among workers rather than
			create smaller jobs for each worker. So, easier splitting process here.
			But we need to do a slightly custom args creation process.
		"""

		### Idea: first adv is the base, rest are deltas from the base
		### transmit the base and the deltas
		n_workers = min(self.get_n_workers(), len(self.lb_args_queue))

		for i,(a,kwa) in enumerate(self.lb_args_queue):
			kwa['job_id'] = i

		ugs = kwargs.get('ugs', None)
		is_verb = kwargs.get('verbose_workers', False)
		base_args, base_kwa = self.lb_args_queue[0]
		if ugs is not None:
			base_kwa['ugs'] = ugs
		base_kwa['verbose_workers'] = is_verb or base_kwa.get('verbose_workers',False)
		base_kwa['generic_obj'] = self.generic_objective.obj
		base_adv, = base_args

		base_adv = threshold_a(base_adv)
		base_args = (base_adv,)

		all_worker_jobs_seq = split_seq(self.lb_args_queue[1:], n_workers)
		
		all_workers_jobs = [[(base_args, base_kwa)] for _ in range(n_workers)]

		for i,job_set in enumerate(all_worker_jobs_seq):
			for other_args, kwa in job_set:
				other_adv, = other_args
				other_adv = threshold_a(other_adv)
				if ugs is not None:
					kwa['ugs'] = ugs
				kwa['verbose_workers'] = is_verb or kwa.get('verbose_workers',False)
				kwa['generic_obj'] = self.generic_objective.obj
				all_workers_jobs[i].append((np.where(base_adv!=other_adv), kwa))

		msgs = list([pickle.dumps(['calc_compressed_lb', subset]) for subset in all_workers_jobs])
		# print(list([['calc_compressed_lb', subset] for subset in all_workers_jobs]))
		rets = self.worker_manager.send_receive_messages_workers(msgs, n_workers=n_workers)
		
		### just append all the jobs, in order. it's important that these things happen in order
		### since that's how we ID the job
		n_to_flush = len(self.lb_args_queue)
		ret_to_call = [None for _ in range(n_to_flush)]

		all_rets = []
		for worker_i in range(n_workers):
			if worker_i > 0:
				all_rets = all_rets + rets[worker_i][1:]
			else: ## get the base answer from worker 0
				all_rets = all_rets + rets[worker_i]
		for adv_ret_i,ret in enumerate(all_rets):
			mean,(x,px) = ret['ans']
			ret_to_call[adv_ret_i] = (mean, (x.flatten(), px.flatten()))

		self.lb_args_queue = []
		self.get_cache()
		return ret_to_call

	def flush_latency_benefit_queue(self, **kwargs):
		return self.flush_latency_benefit_queue_generic(**kwargs)

	def latency_benefit(self, *args, **kwargs):
		self.lb_args_queue.append((copy.deepcopy(args), copy.deepcopy(kwargs)))
		if kwargs.get('retnow', False): # we want an immediate calculation
			return self.flush_latency_benefit_queue(**kwargs)[0]

	def get_gamma(self):
		return self.gamma

	def resilience_benefit(self, a, **kwargs):
		""" sum over peers of E(delta benefit when that peer is knocked out)."""
		# want to maximize resilience beneift, so want to maximize new benefits
		# when peers are knocked out
		if not self.simulated or self.generic_objective.obj not in ["avg_latency"] or self.gamma == 0:
			return 0
		# Under headroom mode (SCULPTOR_CAPACITY_HEADROOM>0), resilience is
		# absorbed into the LP via reserved capacity, so we don't use the
		# RB-grad for optimization. The N_popps+1 LP flush here is purely to
		# populate the "Believed: RB" print and pseudo_objective stopping
		# signal — both can run RB-free without harming convergence. Skipping
		# saves ~18s/iter at actual-10.
		#
		# Gated on _in_training so a future caller that wants the real RB
		# value for reporting (e.g. paper-figure stats) gets it back. Headroom
		# is a training-time approximation only.
		if self._in_training and float(os.environ.get('SCULPTOR_CAPACITY_HEADROOM', '0')) > 0:
			return 0
		tmp = np.ones(a.shape)
		cpkwargs = copy.deepcopy(kwargs)
		cpkwargs['retnow'] = False
		self.latency_benefit_fn(a, **cpkwargs)
		for popp in self.popps:
			# we don't know for sure where users are going
			# so we have to compute over all users
			tmp[self.popp_to_ind[popp],:] = 0
			cpkwargs['failing_popp'] = popp
			self.latency_benefit_fn(a * tmp, **cpkwargs)
			tmp[self.popp_to_ind[popp],:] = 1
		rets = self.flush_latency_benefit_queue()

		benefit = 0
		for b,_ in rets[1:]:
			benefit += b

		return benefit

	def init_advertisement(self):
		print("Initializing advertisement...")
		# SCULPTOR_DEPLOYMENT_SEED also pins the initial advertisement so an A/B
		# pair starts from the same point. Offset by 1 to decorrelate from the
		# deployment-build RNG state without exposing a second env var.
		_seed = os.environ.get('SCULPTOR_DEPLOYMENT_SEED')
		if _seed is not None:
			np.random.seed(int(_seed) + 1)
		mode = self.initialization['type']
		if mode == 'random_binary':
			return np.random.randint(0,2,size=(self.n_popp, self.n_prefixes)) * 1.0
		elif mode == 'normal':
			return ADVERTISEMENT_THRESHOLD+ np.sqrt(self.initialization['var']) \
				* np.random.normal(size=(self.n_popp, self.n_prefixes))
		elif mode == 'ones':
			return np.ones((self.n_popp, self.n_prefixes))
		elif mode == 'zeros':
			return np.zeros((self.n_popp, self.n_prefixes))
		elif mode == 'uniform':
			return np.random.random(size=(self.n_popp, self.n_prefixes))
		elif mode == 'pseudo_anycast':
			a = np.zeros((self.n_popp, self.n_prefixes))
			a[:,0] = 1
			return a
		if mode == 'using_objective':
			### idea ~ 1 anycast prefix
			### 1 prefix motivated by objective
			### rest set completely randomly near .5, but with aim of lambduh * norm penalty = LB
			### expected latency benefit is roughly -1 * (MAX_LATENCY - MIN_LATENCY) / 2
			### so number of entries on should be (MAX_LATENCY - MIN_LATENCY) / 2 out of lambduh * n_popp * (n_pref - 2)
			### max of .05


			# everything off, to start, with some jitter
			a = .35 * np.ones((self.n_popp, self.n_prefixes)) + (.2 * (np.random.uniform(size=(self.n_popp, self.n_prefixes)) - .5 ))
			a[:,0] = .55 # anycast on the first prefix
			for i in range(self.n_pops):
				these_popps = np.array([self.popp_to_ind[popp] for popp in self.popps if popp[0] == self.pops[i]])
				a[these_popps,i+1] = .55
			## linear decrease to the end
			start_ind = self.n_pops + 1
			prob_ons = np.linspace(.05,.005,num=(self.n_prefixes-start_ind))
			for i in range(self.n_prefixes-start_ind):
				prob_on = prob_ons[i]
				is_on = np.random.random(size=(self.n_popp)) < prob_on
				a[is_on,start_ind+i] = .55
			a += .02 * (np.random.uniform(size=a.shape) - .5) # noise
			print("Done Initializing")
			print("Initial numbers of popps on per prefix.")
			print(np.sum(threshold_a(a),axis=0))

			# self.solve_lp_assignment(threshold_a(a))
			return a
		else:
			raise ValueError("Adv init {} not recognized.".format(mode))

	def modeled_objective(self, a, **kwargs):
		"""Approx actual objective with our belief."""
		if self.verbose:
			print("Calculating modeled objective")
		norm_penalty = self.advertisement_cost(a)
		kwargs['retnow'] = True
		latency_benefit, u = self.latency_benefit_fn(a, **kwargs)

		if self.using_resilience_benefit:
			resilience_benefit = self.resilience_benefit_fn(a, **kwargs)
		else:
			resilience_benefit = 0

		if self.verbose:
			benefits,probs = u
			ex = np.average(benefits,weights=probs+1e-8)
			exsq = np.average(np.power(benefits,2),weights=probs+1e-8)
			var = exsq - np.power(ex,2)
			std = np.sqrt(var)
			print("Believed: NP: {}, LB: {} ({} std dev), RB: {}".format(round(norm_penalty,3),
				round(latency_benefit,3), round(std,3), round(resilience_benefit,3)))

		# gamma = self.get_gamma()
		gamma = self.gamma
		if gamma <= 1:
			benefit = latency_benefit + gamma * resilience_benefit
		else:
			benefit = 1 / gamma * latency_benefit + resilience_benefit

		return self.lambduh * norm_penalty - (benefit)

class Sparse_Advertisement_Eval(Sparse_Advertisement_Wrapper):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def solve_anycast(self, verbose=True, **kwargs):

		self.solution_type = 'sparse' ## just use sparse measurements
		self.get_realworld_measure_wrapper()

		## Simple anycast
		anycast_advertisement = np.zeros((self.n_popp, self.n_prefixes))
		anycast_advertisement[:,0] = 1

		optimization_advertisement_representation = {}
		for poppi,prefi in zip(*np.where(threshold_a(anycast_advertisement))):
			optimization_advertisement_representation[self.popps[poppi], prefi] = None

		self.solutions['anycast'] = {
			'objective': self.measured_objective(anycast_advertisement),
			'advertisement': anycast_advertisement,
			'advertisement_representation': optimization_advertisement_representation,
			'latency_benefit':  self.get_ground_truth_latency_benefit(anycast_advertisement),
			'prefix_cost': self.prefix_cost(anycast_advertisement),
			'norm_penalty': self.advertisement_cost(anycast_advertisement),
			'n_advs': 1,
		}

		self.solution_type = "None"
		self.get_realworld_measure_wrapper()

		return "ok"

	def solve_one_per_peering(self, **kwargs):
		## One per peering
		one_per_peering_adv = np.eye(self.n_popps)
		optimization_advertisement_representation = {}
		for poppi,prefi in zip(*np.where(threshold_a(one_per_peering_adv))):
			optimization_advertisement_representation[self.popps[poppi], prefi] = None
		self.solutions['one_per_peering'] = {
			'objective': self.measured_objective(one_per_peering_adv),
			'advertisement': one_per_peering_adv,
			'latency_benefit':  self.get_ground_truth_latency_benefit(one_per_peering_adv),
			'prefix_cost': self.prefix_cost(one_per_peering_adv),
			'norm_penalty': self.advertisement_cost(one_per_peering_adv),
			'advertisement_representation': optimization_advertisement_representation,
			'n_advs': 1,
		}

		return "ok"

	def solve_random(self, verbose=True, **kwargs):
		# To get the random solution, try every possible combination of advertisements
		# Not possible for problems that are too large
		n_arr = self.n_popp * self.n_prefixes
		logn_possibilities = n_arr
		all_as = []
		
		n_possibilities = int(10)
		objs = np.zeros((n_possibilities,))
		actual_objs = np.zeros((n_possibilities,))
		vcpy = copy.copy(self.verbose)
		self.verbose = False

		for i in tqdm.tqdm(range(n_possibilities), desc="Finding a decent random solution."):
			a = np.random.randint(0,high=2,size=(self.n_popp,self.n_prefixes))
			objs[i] = self.measured_objective(a)
			actual_objs[i] = self.actual_nonconvex_objective(a)
			all_as.append(copy.deepcopy(a))
		self.verbose = vcpy

		# Approx according to L1 norm
		random_objective = np.min(objs)
		approx_random_adv = all_as[np.argmin(objs)].reshape(self.n_popp, self.n_prefixes)

		# Actual
		actual_random_objective = np.min(actual_objs)
		l0_random_adv = all_as[np.argmin(actual_objs)].reshape(self.n_popp, self.n_prefixes)

		self.solutions['random'] = {
			'objective': random_objective,
			'advertisement': approx_random_adv,
			'latency_benefit':  self.get_ground_truth_latency_benefit(approx_random_adv),
			'prefix_cost': self.prefix_cost(approx_random_adv),
			'norm_penalty': self.advertisement_cost(approx_random_adv),
			'n_advs': 1,
		}

		return "ok"

	def solve_anyopt(self, **kwargs):
		deployment = self.output_deployment()
		self.anyopt = Anyopt_Adv_Solver(deployment, **self.get_init_kwa())
		self.anyopt.set_worker_manager(self.get_worker_manager())
		self.anyopt.solve()

		anyopt_adv = self.anyopt.advs
		anyopt_obj = self.measured_objective(anyopt_adv)

		self.solutions['anyopt'] = {
			'objective': anyopt_obj,
			'advertisement': anyopt_adv,
			'latency_benefit':  self.get_ground_truth_latency_benefit(anyopt_adv),
			'prefix_cost': self.prefix_cost(anyopt_adv),
			'norm_penalty': self.advertisement_cost(anyopt_adv),
			'advertisement_representation': self.anyopt.optimization_advertisement_representation,
			'n_advs': self.anyopt.path_measures,
		}
		self.clear_caches()

	def solve_sparse(self, **kwargs):
		deployment = self.output_deployment()
		self.sas = Sparse_Advertisement_Solver(deployment, **self.get_init_kwa())
		self.sas.set_worker_manager(self.get_worker_manager())
		self.sas.compute_one_per_peering_solution()

		try:
			self.sas.painter_solution = self.solutions['painter']
		except KeyError:
			pass
		self.sas.verbose=True
		self.sas.solve(**kwargs)
		try:
			self.sas.make_plots()
		except:
			pass
		final_adv = self.sas.optimization_advertisement
		self.sas.reset_metrics()
		self.sas.metrics['advertisements'].append(final_adv)


		sparse_adv = threshold_a(final_adv)
		sparse_objective = self.sas.measured_objective(sparse_adv)
		print("In outer LB {}".format(self.sas.get_ground_truth_latency_benefit(sparse_adv)))
		# pickle.dump(self.sas.output_deployment(), open('saving_working_sparse_deployment.pkl','wb'))
		self.solutions['sparse'] = {
			'objective': sparse_objective,
			'latency_benefit':  self.sas.get_ground_truth_latency_benefit(sparse_adv),
			'norm_penalty': self.sas.advertisement_cost(sparse_adv),
			'prefix_cost': self.sas.prefix_cost(sparse_adv),
			'advertisement': sparse_adv,
			'advertisement_representation': self.sas.optimization_advertisement_representation,
			'n_advs': self.sas.path_measures,
		}

		self.clear_caches()

	def solve_one_per_pop(self, **kwargs):
		# Solve for the one per pop solution
		deployment = self.output_deployment()
		self.one_per_pop = Painter_Adv_Solver(deployment, **self.get_init_kwa())
		self.one_per_pop.set_worker_manager(self.get_worker_manager())

		self.one_per_pop.one_per_pop()
		one_per_pop_adv = self.one_per_pop.painter_advs_to_sparse_advs(self.one_per_pop.advs)
		one_per_pop_obj = self.one_per_pop.measured_objective(one_per_pop_adv)
		self.solutions['one_per_pop'] = {
			'objective': one_per_pop_obj,
			'latency_benefit':  self.one_per_pop.get_ground_truth_latency_benefit(one_per_pop_adv),
			'norm_penalty': self.one_per_pop.advertisement_cost(one_per_pop_adv),
			'prefix_cost': self.one_per_pop.prefix_cost(one_per_pop_adv),
			'advertisement': one_per_pop_adv,
			'advertisement_representation': self.one_per_pop.optimization_advertisement_representation,
			'n_advs': 1,
		}

		self.clear_caches()

	def solve_painter(self, **kwargs):
		## Solve for the painter solution
		# painter is an improvement over anycast, so it has one less prefix to work with
		# and its assumed the first column will be anycast
		deployment = self.output_deployment()
		self.painter = Painter_Adv_Solver(deployment, **self.get_init_kwa())
		self.painter.set_worker_manager(self.get_worker_manager())

		self.painter.painter_v5(cd=5000)
		painter_adv = self.painter.painter_advs_to_sparse_advs(self.painter.advs)
		print('doing painter')
		painter_obj = self.painter.measured_objective(painter_adv)

		# print("Painter Adv, obj: {} {}".format(painter_adv, painter_obj))
		self.solutions['painter'] = {
			'objective': painter_obj,
			'latency_benefit':  self.painter.get_ground_truth_latency_benefit(painter_adv),
			'norm_penalty': self.painter.advertisement_cost(painter_adv),
			'prefix_cost': self.painter.prefix_cost(painter_adv),
			'advertisement': painter_adv,
			'advertisement_representation': self.painter.optimization_advertisement_representation,
			'n_advs': self.painter.path_measures,
		}

		self.clear_caches()

	def painter_objective(self, a, **kwargs):
		## Improvement over anycast
		user_latencies = self.get_ground_truth_user_latencies(a, **kwargs)
		improves = np.array([self.ug_anycast_perfs[ug] - user_latencies[self.ug_to_ind[ug]] for \
			ug in self.ugs])
		mean_improve = np.sum(improves * self.ug_vols) / np.sum(self.ug_vols)
		return -1 * mean_improve

	def anyopt_objective(self, a):
		## Latency benefit
		return -1 * self.get_ground_truth_latency_benefit(a)

	def compare_different_solutions(self, **kwargs):
		verbose = kwargs.get('verbose', True)

		if kwargs.get('soln_types') is not None:
			solution_types = kwargs.get('soln_types')
		else:
			solution_types = ['sparse', 'anyopt', 'painter', 'one_per_pop', 'anycast', 'random', 'one_per_peering']
		metrics = {
			'sparse_objective_vals': {k:[] for k in solution_types},
			'painter_objective_vals': {k:[] for k in solution_types},
			'anyopt_objective_vals': {k:[] for k in solution_types},
			'normalized_sparse_benefit': {k:[] for k in solution_types},
			'latency_benefits': {k: [] for k in solution_types},
			'norm_penalties': {k: [] for k in solution_types},
			'prefix_cost': {k: [] for k in solution_types},
			'objective_diffs': {k:[] for k in solution_types},
			'latency_benefit_diffs': {k:[]for k in solution_types},
			'n_advs': {k:[] for k in solution_types},
			'adv_solns': {k:[] for k in solution_types},
			'adv_representation_solns': {k:[] for k in solution_types},
			'max_sparse_benefits': [],
			'max_painter_benefits': [],
		}
		solve_fns = {'sparse': self.solve_sparse, 'painter': self.solve_painter, 'anyopt': self.solve_anyopt,
			'one_per_pop': self.solve_one_per_pop, 'anycast': self.solve_anycast, 'random': self.solve_random,
			'one_per_peering': self.solve_one_per_peering,}
		self.solutions = {}
		# Strategies that raised even after worker-pool recovery. Recorded so
		# the run never silently claims success while dropping comparison/eval
		# data (the spot-reclaim incident: dp32 finished with no baselines but
		# the sweep reported "ALL DONE").
		self._failed_strategies = []
		if not self.simulated:
			self.get_realworld_measure_wrapper()
		# Optional callback to checkpoint partial results after each strategy
		# completes (e.g. so a crash during painter doesn't lose the already-
		# computed SCULPTOR advertisement). Receives (solution_type, metrics).
		on_strategy_complete = kwargs.get('on_strategy_complete', None)

		# Parallelism gate. When enabled (default), each non-sparse strategy
		# runs in an isolated fork()'d subprocess concurrently with sparse.
		# Cheap strategies don't need Ray workers -- they only use local
		# single-LP solves via solve_lp_assignment. Disable via env var or
		# kwargs for serial behavior (e.g. for debugging).
		parallel_enabled = (
			self.simulated  # subprocess path is only validated for simulated mode
			and not _os.environ.get('SCULPTOR_DISABLE_PARALLEL_STRATEGIES')
			and not kwargs.get('disable_parallel_strategies', False)
		)

		def _record_simulated(solution_type, soln):
			"""Pull the per-strategy fields from a solution dict into metrics."""
			metrics['sparse_objective_vals'][solution_type].append(soln['objective'])
			metrics['n_advs'][solution_type].append(soln.get('n_advs', 1))
			metrics['adv_solns'][solution_type].append(soln['advertisement'])
			metrics['latency_benefits'][solution_type].append(soln['latency_benefit'])
			metrics['norm_penalties'][solution_type].append(soln['norm_penalty'])
			metrics['prefix_cost'][solution_type].append(soln['prefix_cost'])
			adv = soln['advertisement']
			# These two metrics need a working SAS to compute (they call
			# get_ground_truth_user_latencies / get_ground_truth_latency_benefit
			# on the *main-process* SAS, which has the real deployment loaded).
			metrics['painter_objective_vals'][solution_type].append(self.painter_objective(adv))
			metrics['anyopt_objective_vals'][solution_type].append(self.anyopt_objective(adv))

		def _record_actual(solution_type, soln):
			metrics['sparse_objective_vals'][solution_type].append(soln['objective'])
			metrics['n_advs'][solution_type].append(soln.get('n_advs', 1))
			metrics['adv_solns'][solution_type].append(soln['advertisement'])
			metrics['adv_representation_solns'][solution_type].append(
				soln.get('advertisement_representation', {}))
			metrics['latency_benefits'][solution_type].append(soln['latency_benefit'])
			metrics['norm_penalties'][solution_type].append(soln['norm_penalty'])
			metrics['prefix_cost'][solution_type].append(soln['prefix_cost'])

		_record = _record_simulated if self.simulated else _record_actual

		def _fire_callback(solution_type):
			if on_strategy_complete is None: return
			try:
				on_strategy_complete(solution_type, metrics)
			except Exception:
				import traceback
				traceback.print_exc()

		for i in range(kwargs.get('n_run', 50)):
			if verbose:
				print("Comparing different solutions iteration {}".format(i))

			# Partition: serial-in-main runs anything that needs Ray workers
			# (sparse), plus everything in non-simulated mode (which we haven't
			# validated subprocess-safe yet). Parallel-in-subprocess gets the
			# cheap strategies in simulated mode.
			if parallel_enabled:
				serial_types = [s for s in solution_types if s not in _PARALLEL_STRATEGY_NAMES]
				parallel_types = [s for s in solution_types if s in _PARALLEL_STRATEGY_NAMES]
			else:
				serial_types = list(solution_types)
				parallel_types = []

			# Launch parallel subprocesses BEFORE sparse so they run truly
			# concurrently with the main-process sparse solve.
			executor = None
			futures = {}
			if parallel_types:
				ctx = _mp.get_context('fork')
				# Concurrency cap based on current head free memory. Each
				# forked strategy subprocess inherits the driver's memory
				# at fork time (copy-on-write); as it touches pages they
				# diverge and count toward its RSS. Limit how many can run
				# concurrently so we don't overrun the head.
				#
				# `max_concurrent` = floor((avail - headroom) / per_strategy_est).
				# Knobs:
				#   SCULPTOR_HEAD_RAM_HEADROOM_MB     reserve this much (default 8 GB)
				#   SCULPTOR_STRATEGY_RSS_ESTIMATE_MB per-subprocess budget (default 8 GB)
				# When either is unset and /proc unreadable (macOS), falls
				# back to launching all strategies concurrently (legacy
				# behaviour) so this never breaks local dev smoke runs.
				max_concurrent = _max_concurrent_strategies(len(parallel_types))
				executor = concurrent.futures.ProcessPoolExecutor(
					max_workers=max_concurrent, mp_context=ctx)
				deployment_for_sub = self.output_deployment()
				init_kwa_for_sub = self.get_init_kwa()
				# Strip kwargs that don't apply (the subprocess builds its own
				# SAS so things like on_strategy_complete shouldn't propagate).
				sub_kwargs = {k: v for k, v in kwargs.items()
					if k not in {'on_strategy_complete', 'soln_types', 'verbose'}}
				if verbose:
					if max_concurrent < len(parallel_types):
						print("[parallel] launching {} non-sparse strategies with concurrency cap {}: {}".format(
							len(parallel_types), max_concurrent, parallel_types))
					else:
						print("[parallel] launching {} non-sparse strategies in subprocesses: {}".format(
							len(parallel_types), parallel_types))
				for s in parallel_types:
					fut = executor.submit(
						_solve_one_strategy_in_subprocess,
						s, deployment_for_sub, init_kwa_for_sub, sub_kwargs)
					futures[fut] = s

				# Adaptive worker resize: if the Worker_Manager was started
				# with a reduced pool (SCULPTOR_N_WORKERS_DURING_PARALLEL),
				# spawn a watcher thread that asks it to grow to the full
				# SCULPTOR_N_WORKERS target as soon as the parallel-strategy
				# subprocesses finish. The Worker_Manager queues the request;
				# actual add_workers runs at the next sparse iter boundary
				# via wm.process_pending_resize (no concurrent fanouts).
				wm = getattr(self, 'worker_manager', None)
				if wm is not None and hasattr(wm, 'request_add_workers'):
					try:
						target = wm._target_n_workers()
						current = len(getattr(wm, 'worker_sockets', {}) or {})
						n_to_add = max(0, target - current)
					except Exception:
						n_to_add = 0
					if n_to_add > 0:
						future_list = list(futures.keys())
						def _watch_parallel_done(_futs, _wm, _n):
							concurrent.futures.wait(_futs,
								return_when=concurrent.futures.ALL_COMPLETED)
							_wm.request_add_workers(_n)
						t = _threading.Thread(
							target=_watch_parallel_done,
							args=(future_list, wm, n_to_add),
							daemon=True, name='sculptor-worker-ramp')
						t.start()
						if verbose:
							print("[adaptive-workers] watcher armed: will request +{} "
								"workers when {} parallel strategies finish".format(
									n_to_add, len(future_list)))

			# Run serial strategies (sparse) in main process with full Ray pool.
			for solution_type in serial_types:
				if verbose:
					print("\n---solving {} (main process)---\n".format(solution_type))
				try:
					solve_fns[solution_type](**kwargs)
					_record(solution_type, self.solutions[solution_type])
				except Exception:
					import traceback
					traceback.print_exc()
					self._failed_strategies.append(solution_type)
					print("Strategy {} failed; continuing with remaining strategies.".format(solution_type))
				_fire_callback(solution_type)

			# Collect parallel results as they complete.
			for fut in concurrent.futures.as_completed(futures):
				solution_type = futures[fut]
				try:
					_, soln = fut.result()
					self.solutions[solution_type] = soln
					if verbose:
						sub_wall = soln.get('_subprocess_wall')
						if sub_wall is not None:
							print("[parallel] {} finished in {:.2f}s (subprocess)".format(
								solution_type, sub_wall))
					_record(solution_type, soln)
				except Exception:
					import traceback
					traceback.print_exc()
					self._failed_strategies.append(solution_type)
					print("Strategy {} failed; continuing with remaining strategies.".format(solution_type))
				_fire_callback(solution_type)

			if executor is not None:
				executor.shutdown(wait=True)

			if not kwargs.get('dont_update_deployment', False):
				## Update to new random deployment
				new_deployment = get_random_deployment(self.dpsize)
				self.update_deployment(new_deployment)
			if verbose:
				print(metrics['sparse_objective_vals'])

		# Loudly surface any strategy that failed even after pool recovery, and
		# drop a marker file in the run dir so a sweep / downstream eval can
		# tell that this dpsize's comparison data is incomplete.
		metrics['failed_strategies'] = list(self._failed_strategies)
		if self._failed_strategies:
			banner = "!" * 72
			print("\n{0}\n[INCOMPLETE] {1} strategy/strategies failed after recovery: {2}\n"
				  "Comparison/eval data for dpsize={3} is INCOMPLETE.\n{0}\n".format(
					  banner, len(self._failed_strategies),
					  sorted(set(self._failed_strategies)), getattr(self, 'dpsize', '?')),
				  flush=True)
			save_dir = getattr(self, 'save_run_dir', None)
			if save_dir:
				try:
					with open(os.path.join(save_dir, 'BASELINES_INCOMPLETE.txt'), 'w') as _f:
						_f.write("dpsize={}\nfailed_strategies={}\n".format(
							getattr(self, 'dpsize', '?'),
							sorted(set(self._failed_strategies))))
				except Exception:
					pass

		return metrics

class Sparse_Advertisement_Solver(Sparse_Advertisement_Wrapper):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.solution_type = 'sparse'
		self.beta = .3 # gradient momentum
		self.sigmoid_k = 5.0 # heavisside gradient parameter

		self.gradient_support = [(a_i,a_j) for a_i in range(self.n_popp) for a_j in range(self.n_prefixes)]
		if self.simulated:
			self.gradient_support_settings = {
				'lb_support_size': 20*self.n_pops,
				'popp_rb_support_size': 60*self.n_pops,
				'pop_rb_support_size': 30*self.n_pops,  # smaller search space than popp; modest budget
				'info_support_size': 5*self.n_pops,
			}
			if self.gamma == 0:
				self.gradient_support_settings['lb_support_size'] *= 4
		else:
			## we are severely rate limited by measurement speed, so we should aim to compute as much as possible
			self.gradient_support_settings = {
				'lb_support_size': int(.3*(self.n_popps * self.n_prefixes)),
				'popp_rb_support_size': int(.5*(self.n_popps * self.n_prefixes)),
				'pop_rb_support_size': int(.25*(self.n_popps * self.n_prefixes)),
				'info_support_size': 10*self.n_pops,
			}

		self.uncertainty_factor = 10
		self.n_max_info_iter = 1

		self.optimization_var_names = ['rolling_delta', 'rolling_delta_eff', 'rolling_adv_delta', 'rolling_adv_eps', 'last_objective',
			'current_pseudo_objective', 'iter', 'uncertainty_factor', 'stop', 'alpha', 'path_measures', 'current_effective_objective',
			'current_objective', 'calc_times', 'current_latency_benefit', 'current_resilience_benefit']
		if self.simulated:
			self.save_state_every = 5 # how often to save our optimization state
		else:
			self.save_state_every = 1

	def apply_prox_l1(self, w_k):
		"""Applies proximal gradient method to updated variable. Proximal gradient
			for L1 norm is a soft-threshold at the learning rate."""
		return np.sign(w_k) * np.maximum(np.abs(w_k) - self.lambduh * self.alpha, np.zeros(w_k.shape))

	def heaviside_gradient(self, before, after, a_ij):
		# Gradient of sigmoid function
		# when a_ij goes from zero to one, latency benefit value goes from before to after
		# we approx. that as the continuous function before + (after - before) / (1 + exp(-k * a_ij))
		# return the derivative of this function evaluated at a_ij
		x = a_ij - ADVERTISEMENT_THRESHOLD
		return (after - before) * self.sigmoid_k * np.exp(-self.sigmoid_k * x) / (1 + np.exp(-self.sigmoid_k * x))**2

	def heaviside_gradient_sigmoid(self, a):
		x = a - ADVERTISEMENT_THRESHOLD
		grad = self.sigmoid_cost_k * np.exp(-self.sigmoid_cost_k*x) / (1 + np.exp(-self.sigmoid_cost_k*x))**2
		return grad

	def get_last_advertisement(self):
		return self.metrics['advertisements'][-1]

	def get_last_objective(self, effective=False):
		if effective:
			return self.measured_objective(threshold_a(self.get_last_advertisement()))
		else:
			return self.measured_objective(self.get_last_advertisement())
	
	def gradients_latency_benefit(self, a):
		L_grad = np.zeros(a.shape)
		a_effective = threshold_a(a).astype(bool)


		total_n_grad_calc = self.gradient_support_settings['lb_support_size']

		# SCULPTOR_ADAPTIVE_PROBE_BUDGET=1: scale the probe budget down as the
		# algorithm converges. Once rolling_delta drops far below its initial
		# value, most probes return ~0 gradient -- wasted LP work. Floor at
		# max(10, 5 x n_pop) so we still get a meaningful sample every iter.
		# Off by default; ratio_floor=0.01 means budget never drops below 1%
		# of nominal.
		if _os.environ.get('SCULPTOR_ADAPTIVE_PROBE_BUDGET', '0') == '1':
			rd_now = getattr(self, 'rolling_delta', None)
			rd_init = getattr(self, '_rolling_delta_init', None)
			if rd_now is not None and rd_init and rd_init > 0:
				# rolling_delta starts at ~10 (init_optimization_vars). Most
				# of the gradient signal is in the first few orders of magnitude
				# of decay, so use a square-root-ish schedule to avoid
				# collapsing the budget too aggressively.
				ratio = max(0.01, min(1.0, (rd_now / rd_init) ** 0.5))
				floor = max(10, 5 * self.n_pops)
				new_budget = max(floor, int(total_n_grad_calc * ratio))
				if new_budget < total_n_grad_calc:
					print(f"[adaptive-budget] iter={self.iter} rd={rd_now:.2e} ratio={ratio:.3f} "
					      f"budget {total_n_grad_calc} -> {new_budget}")
				total_n_grad_calc = new_budget

		pct_explore = 60 # pct of gradient calculation budget dedicated to exploring
		N_EXPLORE = int(total_n_grad_calc * pct_explore/100)
		# number of gradient calcs that re-calc previously high gradients
		N_REMEASURE = total_n_grad_calc - N_EXPLORE

		calls = []
		n_significant = 0
		try:
			best_from_last_time = sorted(self.last_lb_calls_results.items(), key = lambda el :
				-1 * np.abs(el[1]))
			# SCULPTOR_SIG_CUTOFF (Tom 2026-08-16): the remeasure-significance
			# cutoff was ABSOLUTE (.01) -- calibrated for ~20ms latency-scale
			# objectives, it silently discarded ALL remeasure signal on
			# fraction-scale objectives (fracb range ~0.25: 134/200 iters had
			# zero significant remeasures; caught on fracb_smart_full-dep2-N50).
			# 'p5' (default): cutoff = 5th percentile of the previous
			# iteration's |gradient| distribution -- only the bottom 5% of
			# signals are dropped, at any objective scale. 'abs' restores .01.
			if _os.environ.get('SCULPTOR_SIG_CUTOFF', 'p5') == 'p5':
				_prev_mags = np.abs(np.array([v for _, v in best_from_last_time]))
				_sig_cut = (max(1e-12, float(np.percentile(_prev_mags, 5)))
							if len(_prev_mags) else 1e-12)
			else:
				_sig_cut = .01
			for ind,val in best_from_last_time:
				if (ind,'ba') in calls or (ind,'ab') in calls:
					continue
				if np.abs(val) < self.lambduh or np.abs(val) < _sig_cut:
					# if it's not important enough to warrant the cost, don't bother
					continue
				if np.abs(ADVERTISEMENT_THRESHOLD - a[ind]) > \
					ADVERTISEMENT_THRESHOLD * 7 / 10: 
					# advertismeent is almost completely on or completely off
					continue
				a_ij = a_effective[ind]
				tmpsave = a[ind]
				if not a_ij: # off
					a[ind] = 0.0
					self.latency_benefit(a)
					a[ind] = 1.0
					self.latency_benefit(a)
					calls.append((ind, 'ba'))
				else: # on
					a[ind] = 1.0
					self.latency_benefit(a)
					a[ind] = 0.0
					self.latency_benefit(a)
					calls.append((ind, 'ab'))
				a[ind] = tmpsave
				n_significant += 1
				if n_significant >= N_REMEASURE:
					break
			print("Last LB call, {} were significant".format(n_significant))
		except AttributeError: # there are no last calls on the first iteration
			pass

		N_REMEASURE = len(calls)
		N_EXPLORE = total_n_grad_calc - N_REMEASURE

		self.last_lb_calls_results = {}

		all_inds = [(a_i,a_j) for a_i in range(self.n_popps) for a_j in range(self.n_prefixes)]
		already_in_calls = [ind for ind,_ in calls]
		possible_choices = get_difference(all_inds, already_in_calls)
		if len(possible_choices) > 0:
			possible_choice_inds = np.arange(len(possible_choices))

			N_EXPLORE = np.minimum(N_EXPLORE, len(possible_choices))
			choice_probs = np.array([ADVERTISEMENT_THRESHOLD - np.abs(a[ind] - ADVERTISEMENT_THRESHOLD) + .01 \
				for ind in possible_choices])
			choice_probs = choice_probs / np.sum(choice_probs)
			explore_inds = np.random.choice(possible_choice_inds, size = N_EXPLORE, 
				replace = False, p = choice_probs)
			explore_inds = [possible_choices[i] for i in explore_inds]

			for ind in explore_inds:
				if (ind,'ba') in calls or (ind,'ab') in calls: 
					continue
				a_ij = a_effective[ind]
				tmpsave = a[ind]
				if not a_ij: # off
					a[ind] = 0.0
					self.latency_benefit(a)
					a[ind] = 1.0
					self.latency_benefit(a)
					calls.append((ind, 'ba'))
				else: # on
					a[ind] = 1.0
					self.latency_benefit(a)
					a[ind] = 0.0
					self.latency_benefit(a)
					calls.append((ind, 'ab'))
				a[ind] = tmpsave

		all_lb_rets = self.flush_latency_benefit_queue()
		return self._assemble_lb_gradients(calls, all_lb_rets, a, L_grad)

	def _assemble_lb_gradients(self, calls, all_lb_rets, a, L_grad):
		"""Turn the flushed (benefit, pdf) pairs into the LB gradient. Named
		sub-step of gradients_latency_benefit so subclasses can intercept the
		per-call return values (each entry of all_lb_rets is (mean, (x, pdf)))."""
		# Per-coordinate (raw flip-delta, sigma) capture (merged from the
		# ablation fork 2026-08-16): the workers already return full benefit
		# pdfs; stock code discarded them. heaviside_gradient scales the
		# delta by a sigmoid slope, so sign-error math must use RAW delta vs
		# RAW sigma (same units). delta is a difference of two means over
		# MC_NUM draws -> estimation noise = (var_b + var_a) / MC_NUM.
		try:
			_mc = int(os.environ.get(
				'SCULPTOR_MC_NUM_EXPLORE' if getattr(self, '_abl_sigma_refresh_iter', False)
				else 'SCULPTOR_MC_NUM', '5'))
			_stats = {}
			for _i, (_ind, _bta) in enumerate(calls):
				_var, _means = 0.0, []
				for _j in (2 * _i, 2 * _i + 1):
					_mean_j, (_x, _p) = all_lb_rets[_j]
					_means.append(float(_mean_j))
					_var += self._abl_pdf_var(_x, _p)
				_delta = (_means[1] - _means[0]) if _bta == 'ba' else (_means[0] - _means[1])
				_var_se = self._abl_var_smooth(('lb', _ind), _var / max(_mc, 1))
				_stats[_ind] = (_delta, _var_se ** 0.5)
			self._abl_grad_sigma = _stats
		except Exception as _e:
			print('lb sigma capture failed (non-fatal): {}'.format(_e))
		for i, call_ind in enumerate(calls):
			ind, before_then_after = call_ind

			if before_then_after == 'ba':
				before,_ = all_lb_rets[2*i]
				after, _ = all_lb_rets[2*i+1]
			else:
				after,_ = all_lb_rets[2*i]
				before, _ = all_lb_rets[2*i+1]
			this_grad = self.heaviside_gradient(before, after, a[ind])

			self.last_lb_calls_results[ind] = this_grad
			L_grad[ind] = this_grad


		L_grad = L_grad.clip(-GRAD_CLIP_VAL,GRAD_CLIP_VAL)

		self.all_lb_calls_results.append(L_grad)

		for (popp,pref), _ in calls:
			try:
				self.n_latency_benefit_calls[popp,pref] += 1
			except KeyError:
				self.n_latency_benefit_calls[popp,pref] = 1

		return L_grad

	def _abl_var_smooth(self, key, var_se):
		"""EWMA-smoothed sigma^2 (merged from the ablation fork): refresh
		iterations update the EWMA with the fresh (real-MC) estimate; other
		iterations floor the (often zero at MC=1) instantaneous estimate."""
		if not hasattr(self, '_abl_var_ewma'):
			self._abl_var_ewma = {}
		ew = self._abl_var_ewma
		if getattr(self, '_abl_sigma_refresh_iter', False):
			ew[key] = var_se if key not in ew else 0.5 * ew[key] + 0.5 * var_se
			return ew[key]
		return max(var_se, ew.get(key, 0.0))

	@staticmethod
	def _abl_pdf_var(x, p):
		"""Variance of a worker-returned benefit histogram (the (x, pdf)
		pair latency_benefit computes)."""
		x = np.asarray(x, dtype=float).flatten()
		p = np.asarray(p, dtype=float).flatten()
		if x.size == 0 or p.sum() <= 0:
			return 0.0
		p = p / p.sum()
		m = float((x * p).sum())
		return float(((x - m) ** 2 * p).sum())

	def _abl_capture_rb(self, store_attr, calls_advs, all_lb_rets):
		"""(signed raw delta, variance) per coordinate for RB probes (merged
		from the ablation fork). Each call is a (failed_off, failed_on) ret
		pair; assembly uses heaviside(before=failed_on, after=failed_off) so
		signed raw = failed_off - failed_on."""
		MC_NUM = int(os.environ.get(
			'SCULPTOR_MC_NUM_EXPLORE' if getattr(self, '_abl_sigma_refresh_iter', False)
			else 'SCULPTOR_MC_NUM', '5'))
		store = {}
		ind = 0
		for coord in calls_advs:
			off_mean, (ox, op) = all_lb_rets[ind]
			on_mean, (nx, np_) = all_lb_rets[ind + 1]
			raw = float(off_mean) - float(on_mean)
			var = self._abl_var_smooth(
				(store_attr, coord),
				(self._abl_pdf_var(ox, op) + self._abl_pdf_var(nx, np_)) / max(MC_NUM, 1))
			sl = store.setdefault(coord, [0.0, 0.0])
			sl[0] += raw
			sl[1] += var
			ind += 2
		setattr(self, store_attr, store)

	def gradients(self, a, add_metrics=True):
		# gradient is the proximal gradient of the L1 norm
		# minus lambduh times gradient of L 
		# gradient of L is calculated via a continuous approximation
		if self.verbose:
			ts = time.time()
		L_grad = self.gradients_latency_benefit(a)
		if self.verbose:
			print("Calcing latency benefit grad took {}s".format(int(time.time() - ts)))
		if self.verbose:
			ts = time.time()
		res_grad = self.gradients_resilience_benefit_fn(a)
		if self.verbose:
			print("Calcing resilience benefit grad took {}s".format(int(time.time() - ts)))
		
		gamma = self.get_gamma()
		# gamma specifies a tradeoff between LB and RB, so shouldn't really be > 1
		# to encourage stability
		if gamma <= 1: 
			net_grad = L_grad + gamma * res_grad
			if add_metrics:
				self.metrics['l_benefit_grads'].append(L_grad)
				self.metrics['res_benefit_grads'].append(gamma * res_grad)
				self.metrics['cost_grads'].append(self.lambduh * self.alpha * np.ones(L_grad.shape))
		else:
			net_grad = 1 / gamma * L_grad + res_grad
			if add_metrics:
				self.metrics['l_benefit_grads'].append(1 / gamma * L_grad)
				self.metrics['res_benefit_grads'].append(res_grad)
				self.metrics['cost_grads'].append(self.lambduh * self.alpha * np.ones(L_grad.shape))

		net_grad = self._rescale_gradient(net_grad, a)

		# ---- merged from the ablation fork (2026-08-16) ----
		# NaN guard (repo bug: failure-scenario LPs with zero routable
		# volume propagate NaN and collapse the advertisement)
		_bad = ~np.isfinite(net_grad)
		if _bad.any():
			self.abl_nan_grad_iters = 1 + getattr(self, 'abl_nan_grad_iters', 0)
			print('WARNING: {} non-finite gradient entries zeroed (occurrence {})'.format(int(_bad.sum()), self.abl_nan_grad_iters), flush=True)
			net_grad = np.nan_to_num(net_grad, nan=0.0, posinf=0.0, neginf=0.0)
		_g_ret = -1 * net_grad
		self._abl_last_grads = _g_ret
		# belief trace + stop-v2 best tracking (SCULPTOR_STOP_V2_IMP
		# debounces churn-level noise)
		_b = getattr(self, 'current_pseudo_objective', None)
		if _b is not None and np.isfinite(_b):
			self.metrics.setdefault('abl_belief_objective', []).append(
				(int(getattr(self, 'iter', -1)), float(_b)))
			if not hasattr(self, '_stopv2_b0'):
				self._stopv2_b0 = float(_b)
				self._stopv2_best = float(_b)
				self._stopv2_best_iter = int(getattr(self, 'iter', 0))
			else:
				_imp = float(os.environ.get('SCULPTOR_STOP_V2_IMP', '0.02'))
				_span = max(self._stopv2_b0 - self._stopv2_best, 1e-9)
				if float(_b) < self._stopv2_best - _imp * _span:
					self._stopv2_best = float(_b)
					self._stopv2_best_iter = int(getattr(self, 'iter', 0))
		# probe-diagnostic resolution (decision-targeted probing): fill the
		# AFTER state once the measurement has been folded into beliefs
		_diag = getattr(self, '_abl_pending_probe_diag', None)
		if _diag is not None:
			self._abl_pending_probe_diag = None
			_ind = _diag.get('coord')
			_after = getattr(self, '_abl_grad_sigma', {}).get(_ind)
			_diag['sigma_after'] = float(_after[1]) if _after else None
			_diag['delta_after'] = float(_after[0]) if _after else None
			try:
				_diag['U_after'] = (float(self._abl_probe_uncertainty(_g_ret)[0])
				                    if hasattr(self, '_abl_probe_uncertainty') else None)
			except Exception:
				_diag['U_after'] = None
			_diag['belief_after'] = (float(_b) if _b is not None and np.isfinite(_b) else None)
			if not hasattr(self, '_abl_probe_log'):
				self._abl_probe_log = []
			self._abl_probe_log.append(_diag)
		return _g_ret

	def _rescale_gradient(self, net_grad, a):
		"""Scale the combined gradient toward ~one advertisement flip per
		step: amplify small gradients up to the nearest threshold crossing
		(capped at DESIRED_MAX_VAL), damp very large ones. Extracted from
		gradients() verbatim so variants can override the step policy."""
		# SCULPTOR_GRAD_SCALE (merged from the ablation fork 2026-08-16,
		# Tom-ratified): step-size policy. DEFAULT = 'adagrad' (AdaGrad-Norm,
		# alpha0 via SCULPTOR_ALPHA0/SCULPTOR_ABLATION_ALPHA0, default 1) --
		# 5-seed validated vs the legacy auto-scaler (+10.1 vs +13.7
		# composite vs opp), horizon-free, no per-size retuning. 'fixed' =
		# vanilla alpha*grad; 'dog' = DoG (parameter-free, Ivgi et al. 2023);
		# 'auto' = the legacy amplify/damp policy (kept for reproduction).
		_mode = os.environ.get('SCULPTOR_GRAD_SCALE',
			os.environ.get('SCULPTOR_ABLATION_GRAD_SCALE', 'adagrad'))
		if _mode == 'fixed':
			return net_grad
		if _mode == 'adagrad':
			g2 = float(np.sum(np.asarray(net_grad) ** 2))
			self._adagrad_G = getattr(self, '_adagrad_G', 0.0) + g2
			if not hasattr(self, '_adagrad_alpha0'):
				_a0 = os.environ.get('SCULPTOR_ALPHA0',
					os.environ.get('SCULPTOR_ABLATION_ALPHA0', '1'))
				if _a0 == 'auto':
					_gninf = float(np.max(np.abs(net_grad)))
					self._adagrad_alpha0 = ADVERTISEMENT_THRESHOLD * (g2 ** 0.5) / max(_gninf, 1e-12)
					print('[adagrad] alpha0=auto -> {:.4g}'.format(self._adagrad_alpha0), flush=True)
				else:
					self._adagrad_alpha0 = float(_a0)
			alpha_t = self._adagrad_alpha0 / np.sqrt(1e-12 + self._adagrad_G)
			return net_grad * (alpha_t / self.alpha)
		if _mode == 'dog':
			aa = np.asarray(a, dtype=float)
			if not hasattr(self, '_dog_a0'):
				self._dog_a0 = np.copy(aa)
				self._dog_rbar = float(os.environ.get('SCULPTOR_ABLATION_DOG_EPS', '0.05'))
			self._dog_rbar = max(self._dog_rbar, float(np.linalg.norm(aa - self._dog_a0)))
			g2 = float(np.sum(np.asarray(net_grad) ** 2))
			self._dog_G = getattr(self, '_dog_G', 0.0) + g2
			alpha_t = self._dog_rbar / np.sqrt(1e-12 + self._dog_G)
			return net_grad * (alpha_t / self.alpha)
		# ---- legacy 'auto' amplify/damp policy ----
		DESIRED_MAX_VAL = 5.0
		max_val = np.max(np.abs(net_grad.flatten()))
		if max_val < DESIRED_MAX_VAL and max_val > 0:
			## try to flip at least one index
			## check to make sure this rescale wouldn't flip multiple advertisement indices at once
			## we might not flip any, or flip multiple because of momentum, however
			
			inds = np.abs(net_grad)>1e-3

			alphas = (ADVERTISEMENT_THRESHOLD - a[inds]) / (self.alpha * net_grad[inds])
			alphas = alphas[alphas>0]
			if len(alphas) > 0:
				limiting_alpha = np.min(alphas)
				mult = np.minimum(limiting_alpha, (DESIRED_MAX_VAL / max_val)) * 1.0001
			else:
				mult = 1.0

			net_grad = net_grad * mult
			print("Modified gradient by a factor of {} to ensure approximately one flip".format(mult))
		else:
			# Damp to the SAME bound the amplify branch targets (2026-08-14):
			# the historical 0.1 cap made steps ~50x smaller than the
			# one-flip design target, permanently freezing deployments whose
			# raw flip-gradients exceed DESIRED_MAX_VAL (dep3: frozen at init
			# in BOTH ladder eras and under soft congestion pricing -- its
			# ~7-unit gradients are REAL routed-latency impacts of
			# high-volume coordinates, not pricing artifacts).
			print("WARNING -- gradient is very large, max val is {}".format(max_val))
			net_grad = net_grad * DESIRED_MAX_VAL / max_val

		return net_grad

	def gradients_resilience_benefit_popp(self, advertisement):

		## want to test popp,pref 
		## turn it off, fail a popp. measure LB (a)
		## turn it on, fail same popp. measure LB (b)
		## should turn popp,pref on if (b) > (a)



		### Positive resilience benefit gradient means turning a popp
		## on will increase resilience
		### increasing resilience means maximizing benefit under popp failures


		grad_rb = np.zeros(advertisement.shape)
		calls = []


		### We monte-carlo sample the full space
		total_n_grad_calc = self.gradient_support_settings['popp_rb_support_size']
		
		pct_explore = 80 # pct of gradient calculation budget dedicated to exploring
		N_EXPLORE = int(total_n_grad_calc * pct_explore/100)
		# number of gradient calcs that re-calc previously high gradients
		N_REMEASURE = total_n_grad_calc - N_EXPLORE
		gamma = self.get_gamma()
		try:
			best_from_last_time = sorted(self.last_rb_calls_results_popp.items(), key = lambda el :
				-1 * np.abs(el[1]))
			# same objective-scale-relative significance cutoff as the LB
			# remeasure filter (SCULPTOR_SIG_CUTOFF, Tom 2026-08-16)
			if _os.environ.get('SCULPTOR_SIG_CUTOFF', 'p5') == 'p5':
				_prev_mags = np.abs(np.array([v for _, v in best_from_last_time]))
				_sig_cut = (max(1e-12, float(np.percentile(_prev_mags, 5)))
							if len(_prev_mags) else 1e-12)
			else:
				_sig_cut = .01
			n_significant = 0
			for (popp,rand_kill_popp,rand_outer_prefix),val in best_from_last_time:
				if (popp,rand_kill_popp,rand_outer_prefix) in calls:
					continue
				if gamma * np.abs(val) < self.lambduh or np.abs(val) < _sig_cut:
					# if it's not important enough to warrant the cost, don't bother
					continue
				if np.abs(ADVERTISEMENT_THRESHOLD - advertisement[self.popp_to_ind[popp], rand_outer_prefix]) > \
					ADVERTISEMENT_THRESHOLD * 7 / 10: 
					# advertisment is almost completely on or completely off
					continue

				tmp_a = copy.copy(advertisement)

				this_popp_random_kill = self.popp_to_ind[rand_kill_popp]
				tmp_a[this_popp_random_kill,:] = 0.0 # kill this random popp
				this_killed_popp_ugs = self.popp_to_users.get(this_popp_random_kill, [])
				if len(this_killed_popp_ugs) == 0:
					continue

				poppi = self.popp_to_ind[popp]
				tmp_a[poppi,rand_outer_prefix] = 1.0 # Turn this popp on
				self.latency_benefit(tmp_a, ugs=this_killed_popp_ugs)
				tmp_a[poppi,rand_outer_prefix] = 0.0 # turn this popp off
				self.latency_benefit(tmp_a, ugs=this_killed_popp_ugs)

				calls.append((popp, rand_kill_popp, rand_outer_prefix, this_killed_popp_ugs))

				n_significant += 1
				if n_significant >= N_REMEASURE:
					break
			print("Last RB call, {} were significant".format(n_significant))

		except AttributeError: # there are no last calls on the first iteration
			pass

		N_REMEASURE = len(calls)
		N_EXPLORE = total_n_grad_calc - N_REMEASURE


		all_popps = np.arange(self.n_popp)


		try:
			raise AttributeError
			## Sample popps that need more help, more
			rand_popp_choices = np.random.choice(all_popps, p=self.popp_rb_sample_probabilities, 
				size=N_EXPLORE)
		except AttributeError:
			rand_popp_choices = np.random.randint(low=0,high=self.n_popps,
				size=N_EXPLORE)

		# associated prefix distribution should be biased towards prefixes that are far from 1 and 0
		possible_prefix_choices = np.arange(self.n_prefixes)
		prob_each_pref = np.ones(self.n_prefixes) / self.n_prefixes

		for rand_kill_poppi in rand_popp_choices:
			rand_kill_popp = self.popps[rand_kill_poppi]
			
			poppi_helper = np.random.choice(all_popps,
				 p=self.popp_backup_sample_probs[rand_kill_poppi,:]) 
			popp_helper = self.popps[poppi_helper] # popp ij testing gradient is poppi,rand_outer_prefix (should we turn this on/off to help out?)

			rand_outer_prefix = int(np.random.choice(possible_prefix_choices, p=prob_each_pref))

			if (popp_helper, rand_kill_popp, rand_outer_prefix) in calls: continue
			
			tmp_a = copy.copy(advertisement)
			tmp_a[rand_kill_poppi,:] = 0.0 # kill this random popp
			this_killed_popp_ugs = self.popp_to_users.get(rand_kill_poppi, [])
			if len(this_killed_popp_ugs) == 0:
				continue

			tmp_a[poppi_helper,rand_outer_prefix] = 1.0 # Turn this popp on
			self.latency_benefit(tmp_a, ugs=this_killed_popp_ugs)
			tmp_a[poppi_helper,rand_outer_prefix] = 0.0 # turn this popp off
			self.latency_benefit(tmp_a, ugs=this_killed_popp_ugs)
			calls.append((popp_helper, rand_kill_popp, rand_outer_prefix, this_killed_popp_ugs))

		all_lb_rets = self.flush_latency_benefit_queue()
		grad_rb = self._assemble_rb_popp_gradients(calls, all_lb_rets, advertisement, grad_rb)

		### Track which calls are being made
		for poppi,poppj,pref,_ in calls:
			try:
				self.n_resilience_benefit_popp_calls[poppi,poppj,pref] += 1
			except KeyError:
				self.n_resilience_benefit_popp_calls[poppi,poppj,pref] = 1

		if not self.simulated:
			max_val = np.max(np.abs(grad_rb.flatten()))
			if max_val > 1:
				grad_rb = grad_rb / max_val

		grad_rb = grad_rb.clip(-GRAD_CLIP_VAL,GRAD_CLIP_VAL)

		return grad_rb

	def _assemble_rb_popp_gradients(self, calls, all_lb_rets, advertisement, grad_rb):
		# RB sigma capture (merged from the ablation fork 2026-08-16)
		try:
			_coords = [(self.popp_to_ind[c[0]], c[2]) for c in calls]
			self._abl_capture_rb('_abl_rb_stats_popp', _coords, all_lb_rets)
		except Exception as _e:
			print('rb sigma capture failed (non-fatal): {}'.format(_e))
		"""Turn the flushed (benefit, pdf) pairs into the popp-failure
		resilience gradient. Named sub-step of gradients_resilience_benefit_popp
		so subclasses can intercept the per-call return values."""
		self.last_rb_calls_results_popp = {}
		ind = 0

		for call_popp, killed_popp, rand_outer_prefix, this_killed_popp_ugs in calls:
			poppi = self.popp_to_ind[call_popp]

			failed_off,_ = all_lb_rets[ind] ## popp failed, random popp,prefix under consideration off
			failed_on,_ = all_lb_rets[ind+1] ## popp failed, random popp,prefix under consideration on


			this_grad = self.heaviside_gradient(
				failed_on, failed_off,
				advertisement[poppi,rand_outer_prefix])

			grad_rb[poppi,rand_outer_prefix] += this_grad

			self.last_rb_calls_results_popp[call_popp,killed_popp,rand_outer_prefix] = this_grad
			self.all_rb_calls_results_popps[self.popp_to_ind[killed_popp]].append((self.iter, poppi, rand_outer_prefix, this_grad))

			ind += 2
		return grad_rb

	def gradients_resilience_benefit_pop(self, advertisement):

		## want to test popp,pref 
		## turn it off, fail a PoP. measure LB (a)
		## turn it on, fail same PoP. measure LB (b)
		## should turn popp,pref on if (b) > (a)



		### Positive resilience benefit gradient means turning a popp
		## on will increase resilience
		### increasing resilience means maximizing benefit under PoP failures


		grad_rb = np.zeros(advertisement.shape)
		#### Previously disabled ("unused, too noisy"). Re-enabled May-30 to test
		#### whether the pop-failure gradient term closes the site-failure
		#### painter-beats-sparse gap. Gated externally via SCULPTOR_ALPHA_POP
		#### (called only when alpha > 0 in gradients_resilience_benefit).
		a_effective = threshold_a(advertisement).astype(bool)
		calls = []


		total_n_grad_calc = self.gradient_support_settings['pop_rb_support_size']
		
		pct_explore = 80 # pct of gradient calculation budget dedicated to exploring
		N_EXPLORE = int(total_n_grad_calc * pct_explore/100)
		# number of gradient calcs that re-calc previously high gradients
		N_REMEASURE = total_n_grad_calc - N_EXPLORE
		gamma = self.get_gamma()
		try:
			best_from_last_time = sorted(self.last_rb_calls_results_pop.items(), key = lambda el : 
				-1 * np.abs(el[1]))
			n_significant = 0
			for (popp,rand_kill_pop,rand_outer_prefix),val in best_from_last_time:
				if (popp,rand_kill_pop,rand_outer_prefix) in calls: 
					continue
				if gamma * np.abs(val) < self.lambduh:
					# if it's not important enough to warrant the cost, don't bother
					continue
				if np.abs(ADVERTISEMENT_THRESHOLD - advertisement[self.popp_to_ind[popp],rand_outer_prefix]) > \
					ADVERTISEMENT_THRESHOLD * 7 / 10: 
					# advertisment is almost completely on or completely off
					continue

				tmp_a = copy.copy(a_effective)
				tmp_a[self.pop_to_popp_inds[rand_kill_pop],:] = False # kill this random pop

				poppi = self.popp_to_ind[popp]
				tmp_a[poppi,rand_outer_prefix] = True # Turn this popp on
				self.latency_benefit(tmp_a)
				tmp_a[poppi,rand_outer_prefix] = False # turn this popp off
				self.latency_benefit(tmp_a)

				calls.append((popp, rand_kill_pop, rand_outer_prefix))

				n_significant += 1
				if n_significant >= N_REMEASURE:
					break
			print("Last RB call, {} were significant".format(n_significant))

		except AttributeError: # there are no last calls on the first iteration
			pass

		N_REMEASURE = len(calls)
		N_EXPLORE = total_n_grad_calc - N_REMEASURE


		### Popps for which we're testing if we want to turn them on/off
		rand_popp_choices = np.random.randint(low=0,high=self.n_popps,
			size=N_EXPLORE) 
		### associated prefixes for the rand_popp_choices
		# associated prefix distribution should be biased towards prefixes that are far from 1 and 0
		random_prefix_choices = np.zeros(N_EXPLORE,dtype=np.int32)
		possible_choices = np.arange(self.n_prefixes)
		for i in range(N_EXPLORE):
			prob_each_pref = ADVERTISEMENT_THRESHOLD - np.abs(advertisement[rand_popp_choices[i],:] - ADVERTISEMENT_THRESHOLD) + .1
			prob_each_pref = prob_each_pref / np.sum(prob_each_pref)
			prob_each_pref = np.ones(self.n_prefixes) / self.n_prefixes
			random_prefix_choices[i] = int(np.random.choice(possible_choices, p=prob_each_pref))


		for poppi, rand_outer_prefix in zip(rand_popp_choices,random_prefix_choices):
			popp = self.popps[poppi] # popp ij testing gradient is poppi,rand_outer_prefix

			## random kill PoP
			this_popp_random_kill = np.random.choice(np.arange(self.n_popp),
				 p=self.popp_backup_sample_probs[poppi,:]) 
			rand_kill_pop = self.popps[this_popp_random_kill][0]
			if (popp, rand_kill_pop, rand_outer_prefix) in calls: continue
			
			tmp_a = copy.copy(a_effective)
			tmp_a[self.pop_to_popp_inds[rand_kill_pop],:] = False # kill this random pop

			tmp_a[poppi,rand_outer_prefix] = True # Turn this popp on
			self.latency_benefit(tmp_a)
			tmp_a[poppi,rand_outer_prefix] = False # turn this popp off
			self.latency_benefit(tmp_a)
			calls.append((popp, rand_kill_pop, rand_outer_prefix))

		all_lb_rets = self.flush_latency_benefit_queue()
		grad_rb = self._assemble_rb_pop_gradients(calls, all_lb_rets, advertisement, grad_rb)

		grad_rb = grad_rb.clip(-GRAD_CLIP_VAL,GRAD_CLIP_VAL)

		return grad_rb

	def _assemble_rb_pop_gradients(self, calls, all_lb_rets, advertisement, grad_rb):
		# RB sigma capture (merged from the ablation fork 2026-08-16)
		try:
			_coords = [(self.popp_to_ind[c[0]], c[2]) for c in calls]
			self._abl_capture_rb('_abl_rb_stats_pop', _coords, all_lb_rets)
		except Exception as _e:
			print('rb sigma capture failed (non-fatal): {}'.format(_e))
		"""Turn the flushed (benefit, pdf) pairs into the pop-failure
		resilience gradient. Named sub-step of gradients_resilience_benefit_pop
		so subclasses can intercept the per-call return values."""
		self.last_rb_calls_results_pop = {}
		ind = 0
		for call_popp, killed_pop, rand_outer_prefix in calls:
			poppi = self.popp_to_ind[call_popp]

			failed_off,_ = all_lb_rets[ind] ## popp failed, random popp,prefix under consideration off
			failed_on,_ = all_lb_rets[ind+1] ## popp failed, random popp,prefix under consideration on


			this_grad = self.heaviside_gradient(
				failed_on, failed_off,
				advertisement[poppi,rand_outer_prefix])

			grad_rb[poppi,rand_outer_prefix] += this_grad

			self.last_rb_calls_results_pop[call_popp,killed_pop,rand_outer_prefix] = this_grad
			self.all_rb_calls_results_pops[self.pop_to_ind[killed_pop]].append((self.iter, poppi, rand_outer_prefix, this_grad))

			ind += 2
		return grad_rb

	def gradients_resilience_benefit(self, advertisement):
		# Under SCULPTOR_CAPACITY_HEADROOM>0 the inner LP already reserves
		# capacity for failures, so the SGD-RB gradient is unnecessary.
		# Symmetric with resilience_benefit() (the value), which also
		# short-circuits under headroom. Gated on _in_training so this is
		# strictly a training-time approximation -- non-training callers
		# get the real gradient.
		if self._in_training and float(os.environ.get('SCULPTOR_CAPACITY_HEADROOM', '0')) > 0:
			return np.zeros(advertisement.shape)

		grad_link_failure = self.gradients_resilience_benefit_popp(advertisement)
		# SCULPTOR_ALPHA_POP weights the pop-failure resilience gradient.
		# Default 0 reproduces the prior "popp only" behaviour (the pop
		# gradient was historically disabled with the note "hurts convergence").
		# Test setting alpha=0.05..0.5 to see if it closes the painter-beats-
		# sparse gap on site-failure latency that shows up at dp15 / dp25.
		#
		# SCULPTOR_ALPHA_POP_ANNEAL_END_ITER (int, default 0):
		#   If >0, linearly ramp alpha from 0 -> SCULPTOR_ALPHA_POP across the
		#   first N iters of sparse training. Lets the latency-benefit term
		#   converge first before the noisier pop-failure gradient kicks in.
		alpha_max = float(os.environ.get('SCULPTOR_ALPHA_POP', '0'))
		anneal_end = int(os.environ.get('SCULPTOR_ALPHA_POP_ANNEAL_END_ITER', '0'))
		if anneal_end > 0:
			it = max(0, int(getattr(self, 'iter', 0)))
			alpha = alpha_max * min(1.0, it / float(anneal_end))
		else:
			alpha = alpha_max
		if alpha > 0:
			grad_pop_failure = self.gradients_resilience_benefit_pop(advertisement)
		else:
			grad_pop_failure = 0
		return grad_link_failure + alpha * grad_pop_failure

	def impose_advertisement_constraint(self, a):
		"""The convex constraint 0 <= a_ij <= 1 has the simple solution to clip."""
		a = np.clip(a,0,1.0)
		return a

	def _plot_model_error(self):
		"""Companion figure per run (merged from the ablation fork
		2026-08-16): believed vs ground-truth objective per iteration + the
		absolute gap -- the model-drift diagnostic for grounding-cadence
		questions. Writes model_error_over_iterations.pdf into the run dir."""
		bel = self.metrics.get('abl_belief_objective') or []
		gt = self.metrics.get('actual_nonconvex_objective') or []
		rd = getattr(self, 'save_run_dir', None)
		if not bel or not gt or not rd or not os.path.isdir(rd):
			return
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
		bel_by_iter = dict(bel)
		n = min(len(gt), (max(bel_by_iter) + 1) if bel_by_iter else 0)
		its = [i for i in range(n) if i in bel_by_iter]
		if not its:
			return
		g = [float(gt[i]) for i in its]
		b = [bel_by_iter[i] for i in its]
		err = [abs(x - y) for x, y in zip(b, g)]
		probes = [r['iter'] for r in (getattr(self, '_abl_gate_hist', []) or [])
		          if r.get('probe') and not r.get('skipped')]
		# per-probe WHY annotations (Tom 2026-08-16): what was measured and
		# why -- picked popp + on/off, P(sign error) at selection time, and
		# the realized belief surprise the measurement delivered
		_plog = {int(d.get('iter', -1)): d
		         for d in (getattr(self, '_abl_probe_log', None) or [])}
		fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
		ax[0].plot(its, g, color='#333333', lw=1.6, label='ground truth')
		ax[0].plot(its, b, color='#2a78d6', lw=1.4, label='belief')
		ax[1].plot(its, err, color='#c02f4e', lw=1.5)
		for a_ in ax:
			for p in probes:
				a_.axvline(p, color='#2f9e6e', alpha=.45, lw=1)
		_ymax = max(err) if err else 1.0
		for p in probes:
			d = _plog.get(int(p))
			if not d:
				continue
			_surp = (None if d.get('belief_after') is None
			         or d.get('belief_before') is None
			         else d['belief_after'] - d['belief_before'])
			txt = '{} {}\nPerr={:.2f}'.format(
				d.get('popp', '?'), d.get('turning', ''),
				d.get('p_err', float('nan')))
			if _surp is not None:
				txt += '\nsurp={:+.2f}'.format(_surp)
			ax[1].annotate(txt, xy=(p, _ymax * 0.95), fontsize=6,
			               rotation=90, va='top', ha='right',
			               color='#2f9e6e', alpha=0.9)
		if _plog:
			_surps = [abs(d['belief_after'] - d['belief_before'])
			          for d in _plog.values()
			          if d.get('belief_after') is not None
			          and d.get('belief_before') is not None]
			ax[1].set_title(
				'{} targeted probes; mean P(sign err) at pick = {:.2f}; '
				'mean |belief surprise| = {:.2f}'.format(
					len(_plog),
					float(np.mean([d.get('p_err', 0) for d in _plog.values()])),
					float(np.mean(_surps)) if _surps else float('nan')),
				fontsize=8)
		ax[0].set_ylabel('objective (cost)')
		ax[0].legend(fontsize=8, frameon=False)
		ax[0].set_title('model error over iterations (green = probe iterations)', fontsize=10)
		ax[1].set_ylabel('|belief - ground truth|')
		ax[1].set_xlabel('iteration')
		for a_ in ax:
			a_.grid(alpha=.25)
		fig.tight_layout()
		fig.savefig(os.path.join(rd, 'model_error_over_iterations.pdf'))
		plt.close(fig)

	def make_plots(self, *args, **kwargs):
		try:
			self._plot_model_error()
		except Exception as _e:
			print('model-error plot failed (non-fatal): {}'.format(_e), flush=True)

		## Takes a while (plots from logs). These plot helpers fail when the
		## per-iter log file is sparse (e.g. MAX_ITER=10 / small / fresh cache):
		## the exception is harmless but spams stderr every SCULPTOR iter.
		## Silence unless SCULPTOR_VERBOSE_PLOT_ERRORS is set.
		try:
			compare_estimated_actual_per_user(self.dpsize)
			investigate_congestion_events()
		except Exception:
			if os.environ.get('SCULPTOR_VERBOSE_PLOT_ERRORS'):
				import traceback
				traceback.print_exc()
		

		n_sp = 10  # row 9: adaptive-WHEN metrics (K + surprise)
		plt.rcParams["figure.figsize"] = (10,4*n_sp)
		plt.rcParams.update({'font.size': 14})
		f,ax = plt.subplots(n_sp,2)

		soln = self.get_last_advertisement()

		# General convergence metrics plot
		i=0
		while True:
			try:
				all_as = np.array(self.metrics['advertisements'][i:])
				all_grads = np.array(self.metrics['grads'][i:])
				all_cost_grads = np.array(self.metrics['cost_grads'][i:])
				all_l_benefit_grads = np.array(self.metrics['l_benefit_grads'][i:])
				all_res_benefit_grads = np.array(self.metrics['res_benefit_grads'][i:])
				linestyles = ['-','*','^','>','v']
				colors = ['orange','brown','aqua','deeppink','peru','grey','k','tan']
				for pref_i in range(self.n_prefixes):
					pref_sty = linestyles[pref_i%len(linestyles)]
					for popp_i in range(self.n_popp):
						if self.dpsize == 'small':
							ax[0,0].plot(all_as[:,popp_i,pref_i], 
								c=colors[popp_i%len(colors)])	
						ax[1,0].plot(all_grads[:,popp_i,pref_i], 
							c=colors[popp_i%len(colors)])
						ax[2,0].plot(all_cost_grads[:,popp_i,pref_i], 
							c=colors[popp_i%len(colors)])
						ax[3,0].plot(all_l_benefit_grads[:,popp_i,pref_i], 
							c=colors[popp_i%len(colors)])
						ax[4,0].plot(all_res_benefit_grads[:,popp_i,pref_i], 
							c=colors[popp_i%len(colors)])
				ax[0,0].set_ylabel("a")
				ax[1,0].set_ylabel("Net Grad")
				ax[2,0].set_ylabel("Cost Grad")
				ax[3,0].set_ylabel("LB Grad")
				ax[4,0].set_ylabel("Res Grad")
			except:
				import traceback
				traceback.print_exc()
				i += 1
				if i >= len(self.metrics['grads']):
					break
				continue
			break

		start_iter = 0
		if self.iter > 10:
			start_iter = 10

		all_objectives = self.metrics['actual_nonconvex_objective']
		all_pseudo_objectives = self.metrics['pseudo_objectives']
		all_effective_ojectives = self.metrics['effective_objectives']
		all_resilience_benefits = self.metrics['resilience_benefit']
		all_latency_benefits = self.metrics['latency_benefit']
		all_gt_latency_benefits = self.metrics['gt_latency_benefit']
		all_gt_resilience_benefits = self.metrics['gt_resilience_benefit']
		all_gammas = self.metrics['effective_gammas']
		ax[1,1].plot(list(range(start_iter,len(all_pseudo_objectives))), all_pseudo_objectives[start_iter:])
		ax[1,1].set_ylabel("Believed Objective")
		ax[0,1].plot(all_objectives)
		ax[0,1].set_ylabel("GT Objective")
		ax[2,1].plot(all_effective_ojectives)
		ax[2,1].set_ylabel("GT Effective Objective")
		ax[3,1].plot(list(range(start_iter,len(all_resilience_benefits))), all_resilience_benefits[start_iter:])
		ax[3,1].set_ylabel("Res Ben")
		ax[4,1].plot(list(range(start_iter,len(all_latency_benefits))), all_latency_benefits[start_iter:])
		ax[4,1].set_ylabel("Lat Ben")

		ax[5,0].plot(all_gt_latency_benefits)
		ax[5,0].set_ylabel("GT Lat Ben")
		ax[5,1].plot(all_gt_resilience_benefits)
		ax[5,1].set_ylabel("GT Res Ben")
		
		#### Add in optimal lines
		####### 
		try:
			ax[5,0].hlines(y=self.optimal_expensive_solution['latency'], xmin=0, xmax=self.iter, linewidth=2, color='k')
			ax[5,0].text(0,self.optimal_expensive_solution['latency'],"One per Peering")
		except AttributeError:
			pass
		try:
			ax[5,1].hlines(y=self.optimal_expensive_solution['resilience'], xmin=0, xmax=self.iter, linewidth=2, color='k')
			ax[5,1].text(0,self.optimal_expensive_solution['resilience'],"One per Peering")
		except AttributeError:
			pass
		try:
			ax[0,1].hlines(y=self.optimal_expensive_solution['overall'], xmin=0, xmax=self.iter, linewidth=2, color='k')
			ax[0,1].text(0,self.optimal_expensive_solution['overall'],"One per Peering")
		except AttributeError:
			pass

		#### ADD IN PAINTER LINES IF APPROPRIATE
		try:
			ax[5,0].hlines(y=self.painter_solution['latency_benefit'], xmin=0, xmax=self.iter, linewidth=2, color='r')
		except AttributeError:
			pass
		try:
			self.painter_gt_resilience_benefit
		except AttributeError:
			try:
				self.painter_gt_resilience_benefit = self.get_ground_truth_resilience_benefit(self.painter_solution['advertisement'])
			except AttributeError:
				pass
		try:
			ax[5,1].hlines(y=self.painter_gt_resilience_benefit, xmin=0, xmax=self.iter, linewidth=2, color='r')
		except AttributeError:
			pass
		try:
			ax[0,1].hlines(y=self.painter_solution['objective'], xmin=0, xmax=self.iter, linewidth=2, color='r')
			ax[0,1].text(0,self.painter_solution['objective'], "PAINTER")
		except AttributeError:
			pass


		ax[6,0].plot(all_gammas)
		ax[6,0].set_ylabel("Effective Gamma")

		try:
			all_link_utilizations = np.array(self.metrics['link_utilizations'])
			for poppi in range(self.n_popps):
				ax[7,0].plot(all_link_utilizations[:,poppi])
			ax[7,0].set_ylabel("Link Utilizations")
		except:
			pass

		# Probe-gate / exploration panels (2026-08-14, Tom): rendered only
		# when the ablation fork recorded gate history; stock runs unaffected.
		try:
			gh = self._abl_gate_hist
			its = [g['iter'] for g in gh]
			us = [g.get('U') for g in gh]
			cs = [g.get('c') for g in gh]
			if any(u is not None for u in us):
				ax[6,1].plot([i for i,u in zip(its,us) if u is not None],
							 [u for u in us if u is not None], label='U')
			if any(c is not None for c in cs):
				ax[6,1].plot([i for i,c in zip(its,cs) if c is not None],
							 [c for c in cs if c is not None], '--', label='c')
			# U components (2026-08-14: U = U_sigma + w*entropy ratio)
			for key, style in (('U_sig', ':'), ('U_ent', '-.')):
				vs = [(g['iter'], g.get(key)) for g in gh
					  if g.get(key) is not None]
				if vs:
					ax[6,1].plot([v[0] for v in vs], [v[1] for v in vs],
								 style, lw=.9, label=key)
			for g in gh:
				if g.get('probe'):
					ax[6,1].axvline(g['iter'], color='g', alpha=.25, lw=1)
			ax[6,1].set_yscale('log')
			ax[6,1].set_ylabel('Gate: U vs c (green=probe)')
			ax[6,1].legend(fontsize=5)
			ax[7,1].step(its, [g['spent'] for g in gh], where='post')
			ax[7,1].set_ylabel('Probes spent')
			ax[8,0].plot(its, [g['uf'] for g in gh], label='unc factor')
			ms = [(g['iter'], g.get('med_sigma')) for g in gh
				  if g.get('med_sigma') is not None]
			if ms:
				ax[8,0].plot([m[0] for m in ms], [m[1] for m in ms], ':',
							 label='med sigma')
				for g in gh:
					if g.get('refresh'):
						ax[8,0].axvline(g['iter'], color='b', alpha=.1, lw=.8)
				ax[8,0].legend(fontsize=5)
			ax[8,0].set_yscale('log')
			ax[8,0].set_ylabel('Unc factor / med sigma (blue=MC refresh)')
			ev = [(g['iter'], g['explore_val']) for g in gh
				  if g.get('explore_val') is not None]
			if ev:
				ax[8,1].scatter([e[0] for e in ev], [e[1] for e in ev], s=10,
								label='chosen')
			ea = [(g['iter'], g.get('ent_anchor')) for g in gh
				  if g.get('ent_anchor') is not None]
			if ea:
				ax[8,1].plot([e[0] for e in ea], [e[1] for e in ea], '--',
							 lw=.9, label='ent anchor')
				ax[8,1].legend(fontsize=5)
			if ev or ea:
				ax[8,1].set_yscale('symlog')
			ax[8,1].set_ylabel('Explore value / entropy anchor')
		except (AttributeError, IndexError, KeyError, TypeError):
			pass

		# ---- adaptive-WHEN metrics (Tom 2026-08-16): the surprise-AIMD
		# grounding gate's state over iterations. Left: probe interval K
		# (log2) with fired/skipped probe markers. Right: realized belief
		# surprise per grounding vs theta (the K-adaptation input).
		try:
			gh = getattr(self, '_abl_gate_hist', None) or []
			ks = [(g['iter'], g['K']) for g in gh if g.get('K') is not None]
			if ks:
				ax[9,0].step([k[0] for k in ks], [k[1] for k in ks],
				             where='post', color='#4a3aa7', lw=1.4)
				_fired = [g['iter'] for g in gh
				          if g.get('probe') and not g.get('skipped')]
				_skip = [g['iter'] for g in gh
				         if g.get('probe') and g.get('skipped')]
				for _x in _fired:
					ax[9,0].axvline(_x, color='#2f9e6e', alpha=.5, lw=1)
				for _x in _skip:
					ax[9,0].axvline(_x, color='#eda100', alpha=.5, lw=1,
					                linestyle=':')
				ax[9,0].set_yscale('log', base=2)
				ax[9,0].set_ylabel('probe interval K\n(green=probe, orange=skip)')
				_sp = [(g['iter'], g['surprise']) for g in gh
				       if g.get('surprise') is not None]
				if _sp:
					ax[9,1].scatter([p[0] for p in _sp], [p[1] for p in _sp],
					                s=18, color='#c02f4e')
					_theta = float(os.environ.get(
						'SCULPTOR_ABLATION_SURPRISE_THETA', '0.02'))
					ax[9,1].axhline(_theta, color='#888', lw=1, linestyle='--')
					ax[9,1].set_yscale('symlog', linthresh=0.01)
					ax[9,1].set_ylabel('grounding surprise\n(dashed = theta)')
		except (AttributeError, IndexError, KeyError, TypeError):
			pass

		save_fig(os.path.join(self.save_run_dir, 'convergence_over_iterations.pdf'), abs_path=True)

	def print_adv(self, a):
		for popp_i in range(self.n_popp):
			for pref_i in range(self.n_prefixes):
				print("PoPP {} Prefix {}: {}".format(self.popps[popp_i], pref_i, a[popp_i,pref_i]))

	def set_alpha(self):
		assert self.lambduh < 10
		if self.lambduh < 10 and self.lambduh > 1:
			self.alpha = .00005
		elif self.lambduh <= 1 and self.lambduh > .1:
			self.alpha = .0005
		elif self.lambduh <= .1 and self.lambduh > .01:
			self.alpha = .001
		elif self.lambduh <= .01:
			self.alpha = .01
		# base-alpha env override (merged from the ablation fork 2026-08-16;
		# under 'adagrad'/'dog' policies alpha folds out and this is inert)
		_a = os.environ.get('SCULPTOR_ALPHA', os.environ.get('SCULPTOR_ABLATION_ALPHA'))
		if _a:
			self.alpha = float(_a)
			print('base alpha override: {}'.format(self.alpha), flush=True)

	def get_gamma(self):
		### Idea is to increase gamma to our desired value as we become more confident about adjacent strategies
		if self.simulated:
			uncertainty_factor = np.maximum(1,np.abs(self.uncertainty_factor))
			divider = uncertainty_factor * (1 / (1 + 3 / np.sqrt((self.iter+1))))
		else:
			## no uncertainty factor since we don't do max info (for now)
			divider = (1 + 5 / np.sqrt((self.iter+1)))

		return self.gamma / divider

	def _broadcast_mc_num(self, n):
		# Mirror of _broadcast_training_mode: fan a new MC_NUM out to the
		# persistent workers (handled by _cmd_set_mc_num on the actor).
		wm = getattr(self, 'worker_manager', None)
		if wm is None:
			return
		try:
			wm.send_receive_workers(pickle.dumps(('set_mc_num', int(n))))
		except Exception as e:
			print('[mc-explore] set_mc_num broadcast failed: {}'.format(e), flush=True)

	def _decision_probe_target(self):
		"""Decision-aware measurement targeting (merged from the ablation
		fork 2026-08-16, Tom-ratified L6\' status quo): among the one-flip
		adjacency, propose the coordinate with the largest EXPECTED REGRET
		of deciding unprobed: score = Phi(-|delta|/sigma) * |g| --
		P(sign error on the raw flip-delta) x magnitude of the step the
		solver would take on it. Max-entropy targeting optimizes the MAP;
		this optimizes the next DECISION. Returns an adjacent advertisement
		to measure, or None (callers fall back to the entropy proposal)."""
		from math import erfc, sqrt
		g = getattr(self, '_abl_last_grads', None)
		sig = getattr(self, '_abl_grad_sigma', None)
		if g is None or not sig:
			return None
		g = np.asarray(g)
		cur = threshold_a(np.asarray(self.optimization_advertisement, dtype=float))
		scored = []
		for ind, (delta, sg) in sig.items():
			gv = float(g[ind])
			if gv == 0.0:
				continue
			p_err = 0.5 * erfc(abs(delta) / (sg * sqrt(2.0))) if sg > 0 else 0.0
			if p_err <= 0.0:
				continue
			scored.append((p_err * abs(gv), p_err, delta, sg, ind))
		for score, p_err, delta, sg, ind in sorted(scored, reverse=True):
			aa = np.copy(cur)
			aa[ind] = 1.0 - aa[ind]
			if aa.sum() == 0:
				continue
			if tuple(aa.flatten()) in getattr(self, 'measured', {}):
				continue
			print('[probe-decision] iter={} coord={} score={:.4g} p_err={:.3f} delta={:.4g} sigma={:.4g} ({} scored)'.format(
				getattr(self, 'iter', -1), ind, score, p_err, delta, sg, len(scored)), flush=True)
			self._abl_decision_choice = {
				'coord': ind, 'score': float(score), 'p_err': float(p_err),
				'delta_before': float(delta), 'sigma_before': float(sg),
				'grad': float(g[ind]), 'n_scored': len(scored),
				'popp': str(self.popps[ind[0]]),
				'turning': 'on' if cur[ind] <= ADVERTISEMENT_THRESHOLD else 'off',
				'rank_gap': (float(score - max(t[0] for t in scored)) if scored else 0.0)}
			return aa
		return None

	def solve_max_information(self, current_advertisement):
		"""Wrapper (Tom, 2026-08-14): entropy-based explore needs a real
		belief DISTRIBUTION, but training may run SCULPTOR_MC_NUM=1 (e.g.
		the ablation ladder), which collapses every candidate pdf to a
		single draw -- info value degenerates and explore either picks
		nothing (probe falls back to re-measuring the current adv) or
		re-picks measured advs. Evaluate explore candidates under
		SCULPTOR_MC_NUM_EXPLORE draws (default 5), restoring the training
		MC_NUM afterwards."""
		# SCULPTOR_MAXINFO_TARGET (default 'decision', merged 2026-08-16):
		# expected-regret targeting first; the entropy proposal remains the
		# fallback when nothing scores ('entropy' restores stock behavior).
		if os.environ.get('SCULPTOR_MAXINFO_TARGET', 'decision') == 'decision':
			_cand = self._decision_probe_target()
			if _cand is not None:
				return _cand
		explore_mc = int(os.environ.get('SCULPTOR_MC_NUM_EXPLORE', '5'))
		base_mc = int(os.environ.get('SCULPTOR_MC_NUM', '5'))
		if explore_mc == base_mc:
			return self._solve_max_information_body(current_advertisement)
		self._broadcast_mc_num(explore_mc)
		try:
			return self._solve_max_information_body(current_advertisement)
		finally:
			self._broadcast_mc_num(base_mc)

	def _solve_max_information_body(self, current_advertisement):
		"""Search through neighbors of a, calculate maximum uncertainty."""
		uncertainties = {}
		_info_t0 = time.time()

		a = np.copy(threshold_a(current_advertisement))
		current_benefit,_ = self.latency_benefit_fn(a, retnow=True)
		_info_t_after_curbenefit = time.time()
		awful_benefit = -1000000
		uncertainty_alpha = .25
		# f,ax = plt.subplots(5)
		# self.plti=0

		def get_range(u):
			benefits,probs = u
			significant_prob = np.where(probs>.01)[0]
			if len(significant_prob) == 0:
				return 0
			min_benefit, max_benefit = benefits[significant_prob[0]], benefits[significant_prob[-1]]
			range_benefit = np.abs(max_benefit - min_benefit)
			return range_benefit

		def value_func(u,**kwargs):
			benefits,probs = u
			if len(probs) == 1:
				return awful_benefit
			setting_up = kwargs.get('setting_up',False)

			range_benefit = get_range(u)
			if range_benefit == 0 and not setting_up:
				return awful_benefit
			explore = kwargs.get('force', self.explore)
			if explore == 'positive_benefit':
				if np.sum(probs[benefits>=current_benefit]) > .99: return awful_benefit
				v = np.abs(np.sum(benefits[benefits>current_benefit] * probs[benefits>current_benefit]))
			elif explore == 'entropy':
				v = scipy.stats.entropy(probs+1e-8)
			elif explore == 'bimodality':
				# discussion here https://digitalcommons.wayne.edu/cgi/viewcontent.cgi?article=1120&context=jmasm
				benefits = benefits.flatten()
				probs = probs.flatten()
				ex = np.average(benefits,weights=probs+1e-8)
				exsq = np.average(np.power(benefits,2),weights=probs+1e-8)
				var = exsq - np.power(ex,2)
				std = np.sqrt(var)
				skew = np.average(np.power((benefits - ex) / (std+1e-8), 3), weights = probs+1e-8)
				kurt = np.average(np.power((benefits - ex) / (std+1e-8) , 4), weights = probs+1e-8)
				# maximizing v is maximizing bimodality
				v = -1 * (kurt - np.power(skew,2))
			elif explore == 'other_bimodality':
				negative_part = np.where(benefits <= current_benefit)[0]

				positive_part = np.where(benefits > current_benefit)[0]
				positive_mass = np.sum(probs[positive_part] * (benefits[positive_part] - current_benefit))
				negative_mass = np.sum(probs[negative_part] * (current_benefit - benefits[negative_part]))
				v = positive_mass * negative_mass
			elif explore == 'gmm':
				### idea is maximize distance between bimodal peaks
				### we find bimodal peaks by fitting gmm
				probs = np.array(probs).flatten()
				x_samp = np.random.choice(benefits,size=(1000,1),p=probs/np.sum(probs))
				with warnings.catch_warnings():
					warnings.filterwarnings('error')
					try:
						gmm_model = GaussianMixture(n_components=2).fit(x_samp)
					except ConvergenceWarning:
						return awful_benefit
				gm_means = gmm_model.means_
				v = np.abs(np.diff(gm_means.flatten()))[0]

			significant_prob = np.where(probs>.01)[0]
			min_benefit, max_benefit = benefits[significant_prob[0]], benefits[significant_prob[-1]]
			range_benefit = np.abs(max_benefit - min_benefit)

			if not setting_up:
				if range_benefit == 0:
					return v
				else:
					return v * range_benefit
			else:
				return v
		# if any of these gives a decent signal, measure that
		ranked_explore_methodologies = ['entropy']#['entropy', 'bimodality', 'other_bimodality']

		try:
			self.min_explore_value
		except AttributeError:
			# maybe not the best way?
			x = np.linspace(-1*MAX_LATENCY,-1*MIN_LATENCY,num=LBX_DENSITY)
			methods = ['positive_benefit','entropy','bimodality','other_bimodality','gmm']
			max_min_vals = {m:awful_benefit for m in methods}
			for i in range(len(x)):
				px = np.zeros(x.shape)
				px[i] = 1.0
				px = px + .0001*np.random.uniform(size=px.shape)
				px = px/np.sum(px)
				for method in methods:
					max_min_vals[method] = np.maximum(value_func((x,px), setting_up=True,
						force=method) * 3, max_min_vals[method])
			self.min_explore_value = max_min_vals
			print("Min explore values!")
			print(self.min_explore_value)
			print("\n")

		n_flips = 1
		max_time = 2 # seconds
		t_start = time.time()
		while True:
			# dedicate some percent to exploring permutations specific to transit providers
			pct_resilience = 50
			pct_transit = 30

			n_total = self.gradient_support_settings['info_support_size']
			n_rb = int(n_total * pct_resilience / 100)
			n_lb = n_total - n_rb
			
			perms,perm_labs = [],[]

			## Latency benefit exploration
			all_inds = [(i,j) for i in range(self.n_popp) for j in range(self.n_prefixes)]
			all_perms = sorted(list(itertools.permutations(all_inds, n_flips)))
			np.random.shuffle(all_perms)

			max_n_transit_lb = int(n_lb * pct_transit / 100)
			max_n_nontransit_lb = n_lb - max_n_transit_lb

			transit_perms, nontransit_perms = [], []
			for perm in all_perms:
				if any(poppi in self.provider_popp_inds for poppi,prefi in perm):
					if len(transit_perms) < max_n_transit_lb:
						transit_perms.append(perm)
				else:
					if len(nontransit_perms) < max_n_nontransit_lb:
						nontransit_perms.append(perm)
				if len(nontransit_perms) >= max_n_nontransit_lb and len(transit_perms) >= max_n_transit_lb:
					break
			perms += (transit_perms + nontransit_perms)
			perm_labs += (['LBtransit' for _ in range(len(transit_perms))] + ['LBnontransit' for 
				_ in range(len(nontransit_perms))])

			## Resilience benefit exploration
			max_n_transit_rb = int(n_rb * pct_transit / 100)
			max_n_nontransit_rb = n_rb - max_n_transit_rb

			popp_inds = np.arange(self.n_popps)
			np.random.shuffle(popp_inds)
			transit_perms, nontransit_perms = [], []
			for popp_ind in popp_inds:
				# try to acquire confidence in predictions in cases where popps fail
				perm = tuple([(popp_ind,prefi) for prefi in range(self.n_prefixes) if a[popp_ind,prefi]])
				if len(perm) == 0:
					continue
				if any(poppi in self.provider_popp_inds for poppi,prefi in perm):
					if len(transit_perms) < max_n_transit_rb:
						transit_perms.append(perm)
				else:
					if len(nontransit_perms) < max_n_nontransit_rb:
						nontransit_perms.append(perm)
				if len(nontransit_perms) >= max_n_nontransit_rb and len(transit_perms) >= max_n_transit_rb:
					break
			perms += (transit_perms + nontransit_perms)
			perm_labs += (['RBtransit' for _ in range(len(transit_perms))] + ['RBnontransit' for 
				_ in range(len(nontransit_perms))])

			## add in a couple pop failures for good measure
			pops = np.random.choice(self.pops, size=2)
			for pop in pops:
				corresponding_popps = list([self.popp_to_ind[popp] for popp in self.popps if popp[0] == pop])
				# check to see this PoP is on at all
				perm = tuple([(popp_ind,prefi) for prefi in range(self.n_prefixes) \
					for popp_ind in corresponding_popps if a[popp_ind,prefi]])
				if len(perm) == 0:
					continue
				perms.append(perm)
				perm_labs.append("RBPoP")


			## done searching for perms
			if len(perms) < n_total:
				n_left = self.gradient_support_settings['info_support_size'] - len(perms)
				not_in = get_difference(all_perms, perms)
				n_left = np.minimum(len(not_in), n_left)
				perms = perms + not_in[0:n_left]
				perm_labs += ["random" for _ in range(n_left)]

			_info_t_before_flush = time.time()
			print(f"[InfoTiming] build_perms: {_info_t_before_flush - _info_t_after_curbenefit:.3f}s, n_perms={len(perms)}")
			ts = time.time()
			print("Starting to measure {} perms".format(len(perms)))
			for flips in perms:
				for flip in flips:
					a[flip] = 1 - a[flip]
				self.latency_benefit_fn(a)
				for flip in flips:
					a[flip] = 1 - a[flip]
			all_lb_rets = self.flush_latency_benefit_queue()
			_info_t_after_flush = time.time()
			print(f"[InfoTiming] flush_latency_benefit_queue: {_info_t_after_flush - ts:.3f}s ({len(perms)} perms)")
			print("Measured perms {}s elapsed".format(time.time() - ts))

			for flipi, flips in enumerate(perms):
				_,u = all_lb_rets[flipi]
				uncertainties[flips] = {
					'distribution': u,
					'label': perm_labs[flipi],
				}

			_info_t_before_vf = time.time()
			potential_value_measure = {m:{} for m in ranked_explore_methodologies}
			max_benefit = {m:-1 * np.inf for m in ranked_explore_methodologies}
			best_flips = {m: None for m in ranked_explore_methodologies}
			for flips,vals in uncertainties.items():
				u = vals['distribution']
				inds = np.where(u[1]>.01)[0]
				for m in ranked_explore_methodologies:
					potential_value_measure[m][flips] = value_func(u, force=m)
					if potential_value_measure[m][flips] >= max_benefit[m]:
						best_flips[m] = flips
						max_benefit[m] = potential_value_measure[m][flips]
			print(f"[InfoTiming] value_func loop: {time.time() - _info_t_before_vf:.3f}s")
			print(f"[InfoTiming] solve_max_information TOTAL: {time.time() - _info_t0:.3f}s")
			for m in ranked_explore_methodologies:
				if best_flips[m] is not None:
					print("Best explore value was {} for {}".format(potential_value_measure[m][best_flips[m]],m))
					self._last_explore_value = potential_value_measure[m][best_flips[m]]
					if potential_value_measure[m][best_flips[m]] > self.min_explore_value[m]:
						for flip in best_flips[m]:
							a[flip] = 1 - a[flip]
						_,u = self.latency_benefit_fn(a, retnow=True)
						inds = np.where(u[1]>.01)[0]
						# print("explore methodology best flips: {}".format(m))
						# for i in inds:
						# 	print("LB {} with prob {}".format(round(u[0][i],2), round(u[1][i],2)))
						if tuple(a.flatten()) in self.measured:
							# Explore re-selected an ALREADY-MEASURED advertisement.
							# Default (Tom, 2026-08-14): SKIP the measurement and let
							# training continue -- a re-pick means explore had nothing
							# NEW worth measuring (with MC_NUM=1 its pdfs were
							# degenerate anyway), NOT that training converged.
							# SCULPTOR_REMEASURE_STOP=1 restores the 08-13 graceful
							# training stop; =0 restores stock (hard exit(0)).
							_rm_mode = os.environ.get('SCULPTOR_REMEASURE_STOP', 'skip')
							if _rm_mode == 'skip':
								self._explore_remeasure_skips = 1 + getattr(
									self, '_explore_remeasure_skips', 0)
								print('[REMEASURE-SKIP] iter={} explore re-picked a '
									'measured adv (value={:.4g}); skipping measurement, '
									'training continues ({} skips so far)'.format(
									self.iter, potential_value_measure[m][best_flips[m]],
									self._explore_remeasure_skips), flush=True)
								return None
							if _rm_mode != '0':
								if getattr(self, '_explore_remeasure_stop', None) is None:
									print('=' * 72, flush=True)
									print('[REMEASURE-STOP] Explore selected an ALREADY-MEASURED advertisement', flush=True)
									print('[REMEASURE-STOP] iter={} methodology={} flips(coords)={} value={}'.format(
										self.iter, m, list(best_flips[m]),
										potential_value_measure[m][best_flips[m]]), flush=True)
									print('[REMEASURE-STOP] Beliefs are resolved; STOPPING TRAINING gracefully.', flush=True)
									print('=' * 72, flush=True)
									self._explore_remeasure_stop = {
										'iter': int(self.iter), 'methodology': m,
										'flips': [list(np.atleast_1d(f)) for f in best_flips[m]],
									}
								return None
							print("Re-measuring {}".format(a))
							print(potential_value_measure[m][best_flips[m]])
							pickle.dump(a,open('remeasure_a.pkl','wb'))
							print('woops')
							_,u = self.latency_benefit_fn(a, verbose_workers=True,retnow=True)
							print("This flips had value: {}".format(value_func(u,force=m)))
							print(u)
							exit(0)
						# print("Best flips was: {}".format(best_flips[m]))

						# Running anchor (2026-08-14): set-once anchoring caused a
						# scale race -- if the first explore fired before beliefs
						# contained congestion-priced (NO_ROUTE-scale) mass, the tiny
						# anchor made later sentinel-scale ranges blow uncertainty_factor
						# up by ~4 orders of magnitude (dep3 forensics: 7.5 -> 16386),
						# which collapsed effective gamma and blinded the probe gate.
						try:
							self.typical_high_uncertainty = (
								0.75 * self.typical_high_uncertainty
								+ 0.25 * get_range(u) / 2)
						except AttributeError:
							self.typical_high_uncertainty = get_range(u) / 2
							print("Typical High Uncertainty is {}".format(self.typical_high_uncertainty))

						if 'RB' in uncertainties[best_flips[m]]['label']:
							uncertainty_measure = 1 + 10/self.typical_high_uncertainty*get_range(u) * (value_func(u,setting_up=True,force=m) - self.min_explore_value[m])
							self.uncertainty_factor = (1 - uncertainty_alpha) * \
								self.uncertainty_factor + uncertainty_alpha * uncertainty_measure
						else:
							self.uncertainty_factor *= (1 - uncertainty_alpha)
						self.uncertainty_factor = np.maximum(1, self.uncertainty_factor)
						print("New uncertainty factor is {}".format(self.uncertainty_factor))

						return a
					# else:
					# 	tmpa = copy.copy(a)
					# 	for flip in best_flips[m]:
					# 		tmpa[flip] = 1 - tmpa[flip]
					# 	_,u = self.latency_benefit_fn(tmpa,retnow=True)
					# 	inds = np.where(u[1]>.01)[0]
					# 	print("explore methodology: {}".format(m))
					# 	for i in inds:
					# 		print("uniformative LB {} with prob {}".format(round(u[0][i],2), round(u[1][i],2)))
			n_flips += 1
			if n_flips == 2:
				break
		self.uncertainty_factor *= (1 - uncertainty_alpha)
		print("New uncertainty factor is {}".format(self.uncertainty_factor))
		return None		
		# plt.close()



	def stop_tracker(self, advertisement, skip_measuring=False):
		# --- Timing Setup ---
		print(f"\n--- TIMING START Iteration {self.iter} ---")
		perf_t = time.time()
		
		# Stop when the objective doesn't change, 
		# but use an EWMA to track the change so that we don't spuriously exit
		delta_alpha = .2
		delta_eff_alpha = .2

		ts = time.time()

		if not self.simulated:
			if self.iter == 0:
				### Save optimization state, just in case 
				self.output_optimization_state()
				print(f"[Timing] output_optimization_state (init): {time.time() - perf_t:.5f}s")
				perf_t = time.time()

		# re-calculate objective
		self.last_objective = self.current_pseudo_objective
		self.last_effective_objective = self.current_effective_objective
		
		self.metrics['effective_gammas'].append(self.get_gamma())
		print(f"[Timing] get_gamma: {time.time() - perf_t:.5f}s")
		perf_t = time.time()

		if not skip_measuring or len(self.metrics['gt_latency_benefit']) == 0:
			#### This takes the most time, probably because we always step to a new advertisement and so reset our caches
			
			self.metrics['actual_nonconvex_objective'].append(self.measured_objective(advertisement, verb=True, save_metrics=True))
			print(f"[Timing] measured_objective (1st): {time.time() - perf_t:.5f}s")
			perf_t = time.time()

			self.metrics['gt_latency_benefit'].append(self.get_ground_truth_latency_benefit(advertisement, verb=True, save_ug_ingress_decisions=True))
			print(f"[Timing] get_ground_truth_latency_benefit: {time.time() - perf_t:.5f}s")
			perf_t = time.time()

			self.metrics['gt_resilience_benefit'].append(self.get_ground_truth_resilience_benefit(advertisement, store_metrics=True))
			print(f"[Timing] get_ground_truth_resilience_benefit: {time.time() - perf_t:.5f}s")
			perf_t = time.time()

			self.metrics['effective_objectives'].append(self.measured_objective(copy.copy(threshold_a(advertisement))))
			print(f"[Timing] measured_objective (2nd - effective): {time.time() - perf_t:.5f}s")
			perf_t = time.time()

		else:
			for k in ['actual_nonconvex_objective', 'gt_latency_benefit', 'gt_resilience_benefit', 'effective_objectives']:
				self.metrics[k].append(self.metrics[k][-1])
			print(f"[Timing] Skipping measurement (appending last metrics): {time.time() - perf_t:.5f}s")
			perf_t = time.time()

		self.current_objective = self.metrics['actual_nonconvex_objective'][-1]
		self.current_latency_benefit = self.metrics['gt_latency_benefit'][-1]
		self.current_resilience_benefit = self.metrics['gt_resilience_benefit'][-1]

		self.current_pseudo_objective = self.modeled_objective(advertisement, verbose=True)
		print(f"[Timing] modeled_objective (pseudo): {time.time() - perf_t:.5f}s")
		perf_t = time.time()

		self.current_effective_objective = self.modeled_objective(threshold_a(advertisement))
		print(f"[Timing] modeled_objective (effective): {time.time() - perf_t:.5f}s")
		perf_t = time.time()

		self.metrics['pseudo_objectives'].append(self.current_pseudo_objective)
		
		rb = self.resilience_benefit(advertisement)
		self.metrics['resilience_benefit'].append(rb)
		print(f"[Timing] resilience_benefit: {time.time() - perf_t:.5f}s")
		perf_t = time.time()

		lb_model = self.latency_benefit_fn(advertisement,retnow=True)
		self.metrics['latency_benefit'].append(lb_model[0])
		print(f"[Timing] latency_benefit_fn: {time.time() - perf_t:.5f}s")
		perf_t = time.time()

		## Add to metrics
		self.metrics['frac_latency_benefit_calls'].append(len(self.n_latency_benefit_calls) / (self.n_popps * self.n_prefixes))
		self.metrics['frac_resilience_benefit_calls'].append(len(self.n_resilience_benefit_popp_calls) / (self.n_popps * self.n_popps * self.n_prefixes))

		### Notify workers of new training iteration (recovery-wrapped: a spot
		### reclaim mid-training rebuilds the pool and retries instead of dying)
		self.worker_manager.send_receive_workers(pickle.dumps(('increment_iter', "meep")))
		print(f"[Timing] Worker notification loop: {time.time() - perf_t:.5f}s")
		perf_t = time.time()

		self.rolling_delta = (1 - delta_alpha) * self.rolling_delta + delta_alpha * np.abs(self.current_pseudo_objective - self.last_objective)
		# Capture the first non-default rolling_delta as the "init" reference
		# for SCULPTOR_ADAPTIVE_PROBE_BUDGET's ratio computation.
		if not hasattr(self, '_rolling_delta_init') or self._rolling_delta_init is None:
			# Set after the first real EWMA update (skip the initial 10.0 sentinel).
			if self.rolling_delta < 10.0:
				self._rolling_delta_init = float(self.rolling_delta)
		self.rolling_delta_eff = (1 - delta_eff_alpha) * self.rolling_delta_eff + \
			delta_eff_alpha * np.abs(self.current_effective_objective - self.last_effective_objective)
		adv_delta = np.max(np.abs((advertisement - self.last_advertisement).flatten()))
		self.rolling_adv_delta = (1 - delta_alpha) * self.rolling_adv_delta + delta_alpha * adv_delta
		
		# Original print logic kept intact
		print("RAD: {} {}s".format(self.rolling_adv_delta, time.time() - ts))
		
		self.stop = self.stopping_condition([self.iter,self.rolling_delta,self.rolling_delta_eff,self.rolling_adv_delta])
		print(f"[Timing] stopping_condition: {time.time() - perf_t:.5f}s")
		perf_t = time.time()

		if self.iter % self.save_state_every == 0:
			### Save optimization state 
			self.output_optimization_state()
			print(f"[Timing] output_optimization_state (save): {time.time() - perf_t:.5f}s")
			perf_t = time.time()

		self.output_small_stats()
		print(f"[Timing] output_small_stats: {time.time() - perf_t:.5f}s")

	def get_init_kwa(self):
		kwa = {
			'lambduh': self.lambduh, 
			'gamma': self.gamma, 
			'with_capacity': self.with_capacity,
			'verbose': False,
			'init': self.initialization,
			'explore': self.explore,
			'using_resilience_benefit': self.using_resilience_benefit,
			'n_prefixes': self.n_prefixes,
			'save_run_dir': self.save_run_dir,
		}
		kwa['generic_objective'] = self.generic_objective.obj
		return kwa

	def output_small_stats(self):
		print("Saving smaller stats every iteration, dont exit...")
		self.optimization_vars = {}
		for k in self.optimization_var_names:
			self.optimization_vars[k] = getattr(self, k)
		out_fn = os.path.join(self.save_run_dir, 'small-stats-{}.pkl'.format(self.iter))
		save_state = {
			'optimization_advertisement_representation': self.optimization_advertisement_representation,
			'optimization_vars': self.optimization_vars, # related to when we're going to stop
		}
		pickle.dump(save_state, open(out_fn, 'wb'))
		print("Done saving")

	def output_optimization_state(self):
		print("Saving optimization state, dont exit...")
		_log_mem('save_state_enter', iter=self.iter)
		self.optimization_vars = {}
		for k in self.optimization_var_names:
			self.optimization_vars[k] = getattr(self, k)
		out_fn = os.path.join(self.save_run_dir, 'state-{}.pkl'.format(self.iter))
		save_state = {
			'deployment': self.og_deployment, # link caps, user performance, etc.
			'optimization_advertisement_representation': self.optimization_advertisement_representation,
			'ug_modified_deployment': self.output_deployment(), # link caps, user performance, etc.
			'all_rb_calls_results_popps': self.all_rb_calls_results_popps,
			'last_gti': self.last_gti,
			'advertisement': self.get_last_advertisement(), # the optimization variable
			'last_advertisement': self.last_advertisement,
			'optimization_vars': self.optimization_vars, # related to when we're going to stop
			'parent_tracker': self.parent_tracker, # measured ingress preferences
			'measured': self.measured,
			'measured_prefs': self.measured_prefs,
			'metrics': self.metrics,
		}
		pickle.dump(save_state, open(out_fn, 'wb'))
		_log_mem('save_state_done', iter=self.iter, pkl_mb=_os.path.getsize(out_fn)//(1024*1024))
		print("Done saving")

	def load_optimization_state(self, specific_iter=None):
		self.clear_caches()
		save_port = copy.deepcopy(self.port)
		if specific_iter is None:
			import glob
			all_states = glob.glob(os.path.join(self.save_run_dir, '*'))
			all_iters = [int(re.search("state\-(.+)\.pkl", fn).group(1)) for fn in all_states if "state" in fn]
			specific_iter = np.max(all_iters)
		print("Loading save state {}".format(specific_iter))
		save_state = pickle.load(open(os.path.join(self.save_run_dir, 'state-{}.pkl'.format(specific_iter)),'rb'))

		## update advertisement
		self.metrics = save_state['metrics']
		self.optimization_advertisement = save_state['advertisement']
		self.optimization_advertisement_representation = save_state.get('optimization_advertisement_representation', {})
		self.last_gti = save_state.get('last_gti',{})
		self.last_advertisement = save_state['last_advertisement']
		self.n_prefixes = self.optimization_advertisement.shape[1]
		print(np.sum(self.optimization_advertisement>.5,axis=0))

		## update deployment from the OG deployment (currently loaded) to the pseudo-UG deployment
		self.og_deployment = save_state['deployment']
		new_deployment_with_pseudo_users = save_state['ug_modified_deployment']
		self.og_deployment['port'] = save_port
		new_deployment_with_pseudo_users['port'] = save_port
		self.update_deployment(new_deployment_with_pseudo_users)
		print(np.sum(self.optimization_advertisement>.5,axis=0))

		## various optimization variables
		for k,v in save_state['optimization_vars'].items():
			print("{} {}".format(k,v))
			setattr(self, k, v)

		## information about learned preferneces
		self.measured_prefs = save_state['measured_prefs']
		self.parent_tracker = save_state['parent_tracker']
		for (ui,beaten_ingress,routed_ingress), tf in self.parent_tracker.items():
			if not tf: continue
			try:
				self.calc_cache.all_caches['parents_on'][ui][beaten_ingress,routed_ingress] = None
			except KeyError:
				self.calc_cache.all_caches['parents_on'][ui] = {(beaten_ingress,routed_ingress): None}
		self.update_parent_tracker_workers()
		print(np.sum(self.optimization_advertisement>.5,axis=0))

		self.measured = save_state['measured']
		self.all_rb_calls_results_popps = save_state['all_rb_calls_results_popps']

		if self.simulated:
			self.calculate_user_choice(self.optimization_advertisement, get_ug_catchments=True)
			self.get_ground_truth_latency_benefit(self.optimization_advertisement)
		else:
			self.update_ug_ingress_decisions()
		print(np.sum(self.optimization_advertisement>.5,axis=0))

		### Notify workers of current training iteration (recovery-wrapped)
		self.worker_manager.send_receive_workers(pickle.dumps(('set_iter', self.iter)))

		self.stop = self.stopping_condition([self.iter,self.rolling_delta,self.rolling_delta_eff,self.rolling_adv_delta])
		print(np.sum(self.optimization_advertisement>.5,axis=0))

	def init_optimization_vars(self):
		_log_mem('iov_enter')
		self.clear_caches()
		_log_mem('iov_post_clear_caches')

		self.set_alpha() # momentum parameter

		self.calc_times = []
		self.measured = {}
		self.path_measures = 0
		self.last_gti = None

		## Track which popps are on/off, helpful for use in the actual deployment
		opt_adv_on_off = threshold_a(self.optimization_advertisement)
		self.optimization_advertisement_representation = {}
		for poppi,prefi in zip(*np.where(opt_adv_on_off)):
			self.optimization_advertisement_representation[self.popps[poppi], prefi] = None

		if self.verbose:
			# self.print_adv(advertisement)
			print("Optimizing over {} peers and {} ugs".format(self.n_popp, self.n_ug))

		self.iter = 0

		self.reset_metrics()
		self.stop = False

		# Add to metrics / init vars
		self.current_objective = self.measured_objective(self.optimization_advertisement, save_ug_ingress_decisions=True)
		_log_mem('iov_post_measured_objective')
		self.current_latency_benefit = self.get_ground_truth_latency_benefit(self.optimization_advertisement, verb=True, save_ug_ingress_decisions=True)
		_log_mem('iov_post_gt_latency_benefit')
		self.current_resilience_benefit = self.get_ground_truth_resilience_benefit(self.optimization_advertisement, store_metrics=True)
		_log_mem('iov_post_gt_resilience_benefit')

		self.current_pseudo_objective = self.modeled_objective(self.optimization_advertisement)
		self.current_effective_objective = self.modeled_objective(threshold_a(self.optimization_advertisement))
		self.last_objective = self.current_pseudo_objective
		self.last_effective_objective = self.current_effective_objective
		self.rolling_delta = 10
		self.rolling_delta_eff = 10
		self.rolling_adv_delta = 10
		self.rolling_adv_eps = .01

		self.metrics['pseudo_objectives'].append(self.current_pseudo_objective)
		self.metrics['actual_nonconvex_objective'].append(self.current_objective)
		self.metrics['effective_objectives'].append(self.measured_objective(threshold_a(self.optimization_advertisement)))
		self.metrics['advertisements'].append(copy.copy(self.optimization_advertisement))

	def modify_ugs(self):
		##### DEPRECATED
		try:
			### See if we've already computed the modified deployment
			self.og_deployment
			return
		except AttributeError:
			pass
		## create a pseudo deployment modeled after the optimal solution
		## make a user's optimally assigned popp their lowest-latency popp
		## split users by volume
		self.og_deployment = self.output_deployment()

		print("Not modifying deployment to use pseudo-UGs because not using heuristic approximations.")
		return

	def reset_ugs(self):
		self.update_deployment(self.og_deployment)
		print(np.sum(self.optimization_advertisement>.5,axis=0))
		# if not self.simulated:
		# 	self.get_realworld_measure_wrapper()

	def _broadcast_training_mode(self, in_training):
		# Set local flag (read by driver-side _apply_capacity_headroom) and
		# fan out to workers (which rebuild their persistent Gurobi caps).
		# Mirrors the existing set_iter / increment_iter broadcast pattern.
		self._in_training = bool(in_training)
		wm = getattr(self, 'worker_manager', None)
		if wm is None:
			return
		wm.send_receive_workers(pickle.dumps(('set_training_mode', self._in_training)))

	def solve(self, **kwargs):
		## Orchestrator only: each phase below is a named sub-step so
		## variants (e.g. the ablation fork) can override individual
		## capabilities. Pure code motion from the original inline body.
		# Setup: hot-start from saved state if available, else initialize the
		# advertisement + optimization vars and take the first measurement.
		# Returns False when a hot-started run is already past max iters.
		if not self._solve_setup(**kwargs):
			return
		self._solve_t_start = time.time()
		self.t_per_iter = 0
		self._probe_framework_init()

		if not self.simulated:
			self.last_measured_advertisement = self.optimization_advertisement

		# Enter training mode: gates SCULPTOR_CAPACITY_HEADROOM in
		# solve_lp_assignment._apply_capacity_headroom (driver) and in the
		# workers' persistent Gurobi (Path_Distribution_Computer.set_training_mode).
		# Eval phase runs after solve() returns, with this flag cleared in
		# the finally below so an exception in the SGD loop can't leak
		# training-mode caps into subsequent eval LPs.
		self._broadcast_training_mode(True)

		try:
			while not self.stop:

				timers = []
				t_last = time.time()

				# Prologue: iteration banner, per-iter timers/mem logging, and
				# any pending Ray worker-pool resize.
				self._solve_iter_begin()

				# Gradient phase: one gradient_fn evaluation (latency-benefit
				# probes via the workers' MC model + optional resilience term,
				# clipped and rescaled toward ~one advertisement flip).
				grads = self._solve_compute_gradients()

				## grads
				timers.append(time.time() - t_last)
				t_last = time.time()

				if self.probe_mode in ('scheduled', 'slotted'):
					# WHEN-probing (merged from the ablation fork L2/L6,
					# Tom 2026-08-17): measure-XOR-step under a TOTAL
					# budget of SCULPTOR_PROBE_N groundings over a
					# SCULPTOR_PROBE_TCONV horizon. 'scheduled' fires
					# every ~TCONV/N iterations; 'slotted' (L6, the
					# production WHEN) gives probe k the slot
					# k*period +- period/2 and, within it, fires early
					# when the last grounding's realized SURPRISE was
					# hot, center when quiet, slot-end as backstop.
					# Probing is pure grounding at the CURRENT
					# advertisement (WHAT targeting retired 2026-08-17:
					# the current point is the finite-difference hub).
					# Budget exhaustion stops MEASURING, never TRAINING.
					probe = (self._probe_slotted_decision()
							 if self.probe_mode == 'slotted'
							 else self._probe_scheduled_decision())
					if not (probe and self._probe_ground_current()):
						# step iteration. Preserve stock's
						# uncertainty_factor decay invariant: stock
						# decays inside solve_max_information every
						# iteration; under probe-XOR-step that code only
						# runs on probe iterations (the ~16k-factor
						# deadlock, 2026-08-14).
						self.uncertainty_factor = max(
							1.0, self.uncertainty_factor * (1 - .25))
						self._solve_apply_step(grads)

					## measure
					timers.append(time.time() - t_last)
					t_last = time.time()
					_log_mem('iter_post_measure', iter=self.iter)

					## info (no separate exploration phase under WHEN)
					timers.append(time.time() - t_last)
					t_last = time.time()
				else:
					# Step phase: momentum update w = a - alpha*g + beta*(a - a_last),
					# optional proximal L1, then clip to [0,1] via
					# impose_advertisement_constraint.
					self._solve_apply_step(grads)

					# Measurement phase: if the thresholded advertisement changed,
					# measure ground-truth ingresses (real deployments batch changes
					# before advertising).
					self._solve_post_step_measure()

					## measure
					timers.append(time.time() - t_last)
					t_last = time.time()
					_log_mem('iter_post_measure', iter=self.iter)

					# Exploration phase: pick and measure up to n_max_info_iter
					# maximally-informative advertisements (entropy/bimodality of the
					# predicted benefit distribution) to shrink model uncertainty.
					self._solve_max_info_phase()

					## info
					timers.append(time.time() - t_last)
					t_last = time.time()

				# Stopping phase: update rolling objective/advertisement deltas
				# and evaluate the stopping condition (respects
				# SCULPTOR_MIN_ITER / max_n_iter).
				self._solve_check_stop()

				_log_mem('iter_post_stop_tracker', iter=self.iter)
				self.iter += 1

				## stop
				timers.append(time.time() - t_last)
				t_last = time.time()

				# Epilogue: t_per_iter accounting, periodic progress prints +
				# convergence plots, per-phase timer summary.
				self._solve_iter_end(timers)

		finally:
			# Exit training mode before the eval phase touches the same LPs.
			self._broadcast_training_mode(False)

		# Finalize: persist optimization state to the run dir, restore the
		# full UG set, final summary prints.
		self._solve_finalize()

	# ---- WHEN-probing framework (merged from the ablation fork, Tom ----
	# 2026-08-17). L6 'slotted' = the production probe-timing method;
	# 'scheduled' = its even-spacing backstop (L2). Env knobs (each falls
	# back to its SCULPTOR_ABLATION_* twin so ladder cells keep working):
	#   SCULPTOR_PROBE_MODE   post_step (stock default) | scheduled | slotted
	#   SCULPTOR_PROBE_N      total grounding budget for the run
	#   SCULPTOR_PROBE_TCONV  assumed convergence horizon (slot tiling)
	#   SCULPTOR_SURPRISE_THETA  hot-surprise threshold (default 0.02)

	@staticmethod
	def _probe_env(name, default):
		return os.environ.get('SCULPTOR_' + name,
							  os.environ.get('SCULPTOR_ABLATION_' + name,
											 default))

	def _probe_framework_init(self):
		self.probe_mode = self._probe_env('PROBE_MODE', 'post_step')
		if self.probe_mode not in ('post_step', 'scheduled', 'slotted'):
			# other modes (gated/smart/adaptive/fixed) are ablation-fork
			# experiments, not production capabilities
			self.probe_mode = 'post_step'
		self.probe_n = int(self._probe_env('PROBE_N', '10'))
		self.probe_tconv = int(self._probe_env(
			'PROBE_TCONV', str(getattr(self, 'max_n_iter', 100) or 100)))
		self.probes_spent = 0
		self._probe_last_iter = -10 ** 9
		self._probe_last_attempt = -10 ** 9
		self._probe_surprise_pending = None
		self._probe_last_surprise_val = None
		self._probe_last_surprise = None

	def _probe_resolve_surprise(self):
		"""Realized belief surprise of the LAST grounding: how much the
		measurement moved the belief, relative to the achieved belief span
		(the one bias-immune error signal -- a biased model never
		volunteers that it needs checking, L7 autopsy)."""
		if self._probe_surprise_pending is None:
			return
		pre, probe_iter = self._probe_surprise_pending
		b = getattr(self, 'current_pseudo_objective', None)
		if b is None or not np.isfinite(b) or self.iter <= probe_iter:
			return
		span = max(abs(getattr(self, '_stopv2_b0', float(b))
					   - getattr(self, '_stopv2_best', float(b))), 1e-9)
		surprise = abs(float(b) - pre) / span
		self._probe_last_surprise = float(surprise)
		self._probe_last_surprise_val = float(surprise)
		self._probe_surprise_pending = None
		print('[probe-gate] {} surprise={:.4f}'.format(
			self.probe_mode, surprise), flush=True)

	def _probe_arm_surprise(self):
		b = getattr(self, 'current_pseudo_objective', None)
		self._probe_surprise_pending = (
			(float(b) if b is not None and np.isfinite(b) else 0.0),
			int(self.iter))

	def _probe_slotted_decision(self):
		"""Slotted WHEN (Tom 2026-08-16: "mean measurement rate stays
		evenly spaced; bias measurements to where they're needed WITHIN
		their expected interval"). Probe k owns slot k*period +- w
		(w = period/2, slots tile TCONV exactly), so the budget is always
		fully spent and the long-run rate IS the schedule. Within a slot:
		fire from the slot START when the last grounding surprise was hot
		(the model demonstrably drifting), from the CENTER when quiet;
		the slot END force-fires (schedule = backstop). Skipped probes
		retry every iteration until the slot closes -- no budget leak."""
		period = max(1, int(round(float(self.probe_tconv)
								  / max(1, self.probe_n))))
		w = max(1, period // 2)
		k = self.probes_spent + 1          # next probe, 1-indexed
		can = k <= self.probe_n
		self._probe_resolve_surprise()
		theta = float(self._probe_env('SURPRISE_THETA', '0.02'))
		hot = (self._probe_last_surprise_val or 0.0) > theta
		center = k * period
		earliest, latest = center - w, center + w
		due = self.iter >= (earliest if hot else center)
		force = self.iter >= latest
		decision = can and (due or force)
		if decision:
			self._probe_last_attempt = self.iter
			self._probe_arm_surprise()
		print('[probe-gate] iter={} mode=slotted k={}/{} slot=[{},{}] '
			  'hot={} spent={} -> {}'.format(
				  self.iter, k, self.probe_n, earliest, latest, hot,
				  self.probes_spent,
				  'PROBE' if decision else 'step'), flush=True)
		return decision

	def _probe_scheduled_decision(self):
		"""scheduled mode: unconditional probe every ~TCONV/N iterations --
		no self-assessment, spends exactly N over the horizon."""
		period = max(1, int(round(float(self.probe_tconv)
								  / max(1, self.probe_n))))
		due = (self.iter - self._probe_last_iter) >= period
		can = self.probes_spent < self.probe_n
		retry_ok = (self.iter - self._probe_last_attempt >= min(3, period))
		decision = due and can and retry_ok
		if decision:
			self._probe_last_attempt = self.iter
			self._probe_arm_surprise()
		print('[probe-gate] iter={} mode=scheduled period={} since_last={} '
			  'spent={}/{} -> {}'.format(
				  self.iter, period, self.iter - self._probe_last_iter,
				  self.probes_spent, self.probe_n,
				  'PROBE' if decision else 'step'), flush=True)
		return decision

	def _probe_ground_current(self):
		"""Grounding measurement at the CURRENT advertisement. Returns
		True iff a measurement actually happened; when the current adv is
		already measured the probe request is IGNORED (spend nothing,
		stop nothing -- Tom 2026-08-14) and the caller steps instead."""
		pm_before = int(getattr(self, 'path_measures', 0))
		cur = tuple(threshold_a(np.asarray(
			self.optimization_advertisement, dtype=float)).flatten())
		if cur in getattr(self, 'measured', {}):
			self._probe_skips = 1 + getattr(self, '_probe_skips', 0)
			print('[probe-gate] iter={} probe SKIPPED (current already '
				  'measured; {} skips)'.format(
					  self.iter, self._probe_skips), flush=True)
			return False
		self._solve_post_step_measure()
		if int(getattr(self, 'path_measures', 0)) == pm_before:
			return False
		self.probes_spent += 1
		self._probe_last_iter = self.iter
		return True

	def _solve_setup(self, **kwargs):
		"""Hot-start or cold-start initialization. Returns False when a hot-started run is already past max iters (solve() returns immediately)."""
		_log_mem('solve_enter', dpsize=getattr(self, 'dpsize', '?'))
		try:
			## If we're hot-starting, load the optimization state. But this will throw an error if we're not
			self.load_optimization_state()
			_log_mem('solve_post_load_state', iter=self.iter, mode='hotstart')
			if self.iter >= self.max_n_iter:
				self.reset_ugs()
				return False
		except ValueError:
			print("\n=====NOT HOT STARTING======\n")
			_log_mem('solve_cold_start')
			self.modify_ugs()
			_log_mem('solve_post_modify_ugs')
			self.optimization_advertisement = self.init_advertisement()
			self.last_advertisement = copy.copy(self.optimization_advertisement)
			if not self.simulated:
				## This is our first measurement
				self.calculate_ground_truth_ingress(self.optimization_advertisement)

			self.init_optimization_vars()
			_log_mem('solve_post_init_optim_vars')

			# Measure where we start, update model of path probabilities
			self.measure_ingresses(self.optimization_advertisement)
			_log_mem('solve_post_first_measure_ingresses')
		return True

	def _solve_iter_begin(self):
		"""Per-iteration prologue: prints, timers, mem log, pending worker resize."""
		if self.verbose:
			print("\n\n")
			print("LEARNING ITERATION : {}".format(self.iter))
			print("\n\n")
		self.ts_loop = time.time()
		_log_mem('iter_start', iter=self.iter)
		# Adaptive worker resize hook: if the watcher thread set up
		# in compare_different_solutions has flagged a ramp-up
		# (parallel-strategy subprocesses finished), grow the Ray
		# actor pool now. Runs synchronously on the main thread, so
		# no concurrent fanouts are in flight. No-op when no ramp
		# is pending (the common case).
		_wm = getattr(self, 'worker_manager', None)
		if _wm is not None and hasattr(_wm, 'process_pending_resize'):
			_wm.process_pending_resize()

	def _solve_compute_gradients(self):
		"""Gradient phase: one gradient_fn evaluation for this iteration.
		Periodic sigma refresh (merged from the ablation fork 2026-08-16):
		every SCULPTOR_SIGMA_REFRESH iters (default 10) evaluate this
		iteration's gradient flips under MC_NUM_EXPLORE draws so the
		captured sigmas come from a real distribution (at MC_NUM=1 the
		instantaneous estimates are point masses)."""
		if self.verbose:
			print("calcing grads")
		_refresh_every = int(os.environ.get('SCULPTOR_SIGMA_REFRESH',
			os.environ.get('SCULPTOR_ABLATION_SIGMA_REFRESH', '10')))
		# mc-off runs (SCULPTOR_ABLATION_MC=0, the no_mc rung's deterministic
		# pseudo-path worker) support exactly ONE realization -- broadcasting
		# MC_NUM_EXPLORE at refresh KeyError'd every L1/L2 cell of the v4
		# grid (469 cells, 2026-08-16 night). No refresh when MC is off.
		_mc_off = os.environ.get('SCULPTOR_ABLATION_MC', '1') == '0'
		self._abl_sigma_refresh_iter = ((not _mc_off)
			and (self.iter % max(1, _refresh_every) == 0))
		_explore_mc = int(os.environ.get('SCULPTOR_MC_NUM_EXPLORE', '5'))
		_base_mc = int(os.environ.get('SCULPTOR_MC_NUM', '5'))
		if self._abl_sigma_refresh_iter and _explore_mc != _base_mc:
			self._broadcast_mc_num(_explore_mc)
			try:
				grads = self.gradient_fn(self.optimization_advertisement)
			finally:
				self._broadcast_mc_num(_base_mc)
		else:
			grads = self.gradient_fn(self.optimization_advertisement)
		_log_mem('iter_post_grad', iter=self.iter)
		return grads

	def _solve_apply_step(self, grads):
		"""Momentum step + optional prox-L1 + advertisement constraint + metrics."""
		self.recent_grads = grads
		# update advertisement by taking a gradient step with momentum and then applying the proximal gradient for L1
		a_k = self.optimization_advertisement
		w_k = a_k - self.alpha * grads + self.beta * (a_k - self.last_advertisement)
		if self.proximal:
			self.optimization_advertisement = self.apply_prox_l1(w_k)
		else:
			self.optimization_advertisement = w_k
		self.last_advertisement = copy.copy(a_k)

		# another constraint we may want is 0 <= a_ij <= 1
		# the solution is just clipping to be in the set
		# clipping can mess with gradient descent
		self.optimization_advertisement = self.impose_advertisement_constraint(self.optimization_advertisement)
 
		self.metrics['advertisements'].append(copy.copy(self.optimization_advertisement))
		self.metrics['grads'].append(self.optimization_advertisement - a_k)

	def _solve_post_step_measure(self):
		"""Measure ground truth after the step when the (thresholded) advertisement changed; real-deployment variant batches changes."""
		measured_this_round = False
		if self.simulated:
			# Take a gradient step and update measured paths + probabilities
			if not np.array_equal(threshold_a(self.optimization_advertisement), threshold_a(self.last_advertisement)):
				if self.verbose:
					print("Gradient stepped to a new advertisement, issuing measurement.")
					print("Changed Indices: {}".format(np.where(np.abs(threshold_a(self.optimization_advertisement) - threshold_a(self.last_advertisement)))))
				self.measure_ingresses(self.optimization_advertisement)
				opt_adv_on_off = threshold_a(self.optimization_advertisement)
				self.optimization_advertisement_representation = {}
				for poppi,prefi in zip(*np.where(opt_adv_on_off)):
					self.optimization_advertisement_representation[self.popps[poppi], prefi] = None
		else:
			NUM_PREFS_TRIGGER_CHANGE = 2 # N prefixes change
			NUM_TOTAL_TRIGGER_CHANGE = 4 # N popps change
			measured_this_round = False
			differences = np.where(np.abs(threshold_a(self.optimization_advertisement) - threshold_a(self.last_measured_advertisement)))
			print("Differences in advertisement so far : {}".format(differences))
			prefs_changed = {}
			if len(differences) > 0:
				## Change in the advertisement, update the actual-deployment-specific tracker
				opt_adv_on_off = threshold_a(self.optimization_advertisement)
				self.optimization_advertisement_representation = {}
				for poppi,prefi in zip(*np.where(opt_adv_on_off)):
					self.optimization_advertisement_representation[self.popps[poppi], prefi] = None

			for poppi,prefi in zip(*differences):
				prefs_changed[prefi] = None
			if len(prefs_changed) >= NUM_PREFS_TRIGGER_CHANGE or len(differences[0]) >= NUM_TOTAL_TRIGGER_CHANGE:
				print("Indices: {}, Prefixes : {} changed, so measuring now...".format(differences, list(prefs_changed)))
				self.measure_ingresses(self.optimization_advertisement)
				self.last_gti = self.calculate_ground_truth_ingress(self.optimization_advertisement)
				measured_this_round = True
				self.last_measured_advertisement = self.optimization_advertisement.copy()
		self._measured_this_round = measured_this_round

	def _solve_max_info_phase(self):
		"""Exploration phase: measure up to n_max_info_iter maximally-informative advertisements."""
		# Calculate, advertise & measure information about the prefix that would
		# give us the most new information
		if self.verbose:
			tsmaxinfo = time.time()
		if self.simulated: ## maybe tmp
			for maxinfoi in range(self.n_max_info_iter):
				maximally_informative_advertisement = self.solve_max_information(self.optimization_advertisement)
				if maximally_informative_advertisement is not None:
					print("Found an interesting advertisement on iteration {}, so measuring...".format(maxinfoi))
					self.measure_ingresses(maximally_informative_advertisement)
				else:
					if self.verbose:
						print("No further maximally informative advertisement to measure.")
					break
			if self.verbose:
				print("finding max info took {}s ".format(round(time.time() - tsmaxinfo,2)))

	def _solve_check_stop(self):
		"""Stopping-condition update via stop_tracker."""
		if self.simulated:
			## Check stopping conditions
			self.stop_tracker(self.optimization_advertisement)
		else:
			## Check stopping conditions if we measured this round, to avoid excessive measurements
			if self._measured_this_round:
				self.stop_tracker(self.optimization_advertisement)
			else:
				self.stop_tracker(self.optimization_advertisement, skip_measuring=True)

	def _solve_iter_end(self, timers):
		"""Per-iteration epilogue: t_per_iter, periodic prints/plots, timer summary."""
		self.t_per_iter = (time.time() - self._solve_t_start) / self.iter
		if self.iter % PRINT_FREQUENCY(self.dpsize) == 0 and self.verbose:
			print("Optimizing, iter: {}, t_per_iter : {}s, GTO: {}, RD: {}, RDE: {}, {} path measures".format(
				self.iter, round(self.t_per_iter,2), 
				self.metrics['actual_nonconvex_objective'][-1],self.rolling_delta, self.rolling_delta_eff,
				self.path_measures))

			try:
				self.make_plots()
			except:
				import traceback
				traceback.print_exc()

		for t,lab in zip(timers, ['grads','measure','info','stop']):
			print("Timer: {} -- {} s".format(lab, round(t,2)))
		self.calc_times = list(zip(timers, ['grads','measure','info','stop']))

		print("Updated numbers of popps on per prefix.")
		print(np.sum(threshold_a(self.optimization_advertisement),axis=0))

	def _solve_finalize(self):
		"""Post-loop: persist optimization state, restore UGs, final prints."""
		# After finishing, end the optimization
		self.output_optimization_state()

		self.reset_ugs()

		if self.verbose:
			print("Stopped train loop on {}, t per iter: {}s, {} path measures, O:{}, RD: {}, RDE: {}".format(
				self.iter, round(self.t_per_iter,2), self.path_measures, 
				self.current_pseudo_objective, self.rolling_delta, self.rolling_delta_eff))
		self.metrics['t_per_iter'] = self.t_per_iter


def main():
	try:
		import sys
		np.random.seed(31415)
		dpsize = sys.argv[1]
		deployment = get_random_deployment(dpsize)

		## useful for fixing the deployment between testing various settings
		# deployment = pickle.load(open('runs/1710776224-small-sparse/state-0.pkl','rb'))['deployment']

		lambduh = 0
		gamma = 2.0
		n_prefixes = deployment_to_prefixes(deployment)
		sas = Sparse_Advertisement_Solver(deployment, verbose=True,
				lambduh=lambduh,with_capacity=True,explore=DEFAULT_EXPLORE, 
				using_resilience_benefit=True, gamma=gamma, n_prefixes=n_prefixes)
		wm = Worker_Manager(sas.get_init_kwa(), deployment)
		wm.start_workers()
		sas.set_worker_manager(wm)
		sas.solve()
		print(sas.get_ground_truth_latency_benefit(sas.optimization_advertisement))
		soln = sas.get_last_advertisement()
		sas.make_plots()
		plot_lats_from_adv(sas, soln, 'basic_run_demo_{}.pdf'.format(sas.dpsize))

		compare_estimated_actual_per_user()

	except:
		import traceback
		traceback.print_exc()
	finally:
		wm.stop_workers()


if __name__ == "__main__":
	main()
