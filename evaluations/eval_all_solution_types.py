"""Primary driver: `evaluate_all_metrics(dpsize, port, **kw)`.

This is the main entry point that nearly every sweep / experiment driver
ultimately calls. It:

  1. Loads (or initialises) the per-dpsize metrics pickle.
  2. For each `random_iter`, builds a fresh `Sparse_Advertisement_Eval`,
     attaches a Ray `Worker_Manager`, calls `compare_different_solutions`
     to run sparse + parallel-strategy baselines, and records the
     per-strategy advertisement matrices.
  3. Runs the post-training eval phases (latency, pct-volume-within-latency,
     failure resilience, flash-crowd, diurnal) on each random_iter's
     solutions. The phases are implemented in `wrapper_eval.py`.
  4. Periodically pickles the metrics dict to disk after each phase so a
     crash mid-eval doesn't lose work.
  5. Calls `wm.stop_workers()` in a finally block so the Ray actor pool
     is always torn down (also pulls per-worker mem logs to the driver
     log on the way out).

Adaptive-workers integration: the env var
`SCULPTOR_N_WORKERS_DURING_PARALLEL` is read at `wm.start_workers()`
time and triggers the watcher-thread + ramp-up flow in
`compare_different_solutions`. See README.md "Environment variables".

`python eval_all_solution_types.py --dpsize <name> --port <p>` invokes it
directly from the shell. Sweep drivers (`experiments/deployment_sizes_full_timing_investigation/run_deployment_sweep.py`,
`evaluate_over_deployment_sizes.py`, etc.) call it from a loop over
dpsizes.
"""

# run-as-script bootstrap: this module lives in a package now,
# so put the repo root on sys.path before importing siblings.
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from helpers.constants import *
from helpers.helpers import *
# eval phase RAM/timing markers (Tom 2026-08-20). LAZY import: a
# top-level 'from sparse_advertisements_v3 import _log_mem' here fires
# the circular-import chain early and the later star-import then sees a
# partially-initialized module (get_random_deployment NameError, found
# on the instrumented 32 pre-flight).

def _fit_x_to_data(a, floor):
	"""Scale a CDF panel to the data instead of the no-route sentinel.

	These panels used to hardcode set_xlim([-NO_ROUTE_LATENCY/2, 0]) -- i.e.
	-15000 ms. Real latency deltas are single-digit ms, so every curve
	collapsed onto x=0 and the panel read as empty (Tom, 2026-08-21). Fit to
	the plotted data, never showing more range than the sentinel floor.
	"""
	import numpy as _np
	xs = [l.get_xdata() for l in a.get_lines() if len(l.get_xdata())]
	if not xs:
		a.set_xlim([floor, 0]); return
	v = _np.concatenate([_np.asarray(x, dtype=float) for x in xs])
	v = v[_np.isfinite(v)]
	if not len(v):
		a.set_xlim([floor, 0]); return
	lo = float(_np.percentile(v, 0.5))
	lo = max(lo, floor)
	if lo > -1e-9:
		lo = min(-1.0, float(v.min()))
	a.set_xlim([lo * 1.05, 0])


def _log_mem(*a, **k):
	from core.sparse_advertisements_v3 import _log_mem as _f
	return _f(*a, **k)
from evaluations.wrapper_eval import *
from evaluations.objectives import objective_hooks
from core.solve_lp_assignment import *

import pickle, numpy as np, matplotlib.pyplot as plt, copy, itertools, time
from core.sparse_advertisements_v3 import *


def _release_memory(label=""):
	"""Force GC + return freed heap pages back to the OS.

	Python's per-strategy peak memory in the failure-eval phase is ~2 GB
	even with light_result stripping, but glibc malloc doesn't return
	freed pages to the OS by default. Across 6 strategies that's enough
	monotonic RSS growth to OOM the 64 GB cluster head -- which is exactly
	what we observed on the first actual-32 eval-resume attempt (RSS
	climbed 11 -> 53 GB before we aborted).
	"""
	import gc
	gc.collect()
	try:
		import ctypes
		ctypes.CDLL("libc.so.6").malloc_trim(0)
	except (OSError, AttributeError):
		# macOS / non-glibc: malloc_trim isn't available. gc.collect()
		# alone is still useful (it forces refcount cycles to be reaped).
		pass
	if label:
		try:
			import resource
			import sys as _sys
			raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
			# Linux: ru_maxrss is KB. macOS: ru_maxrss is bytes.
			if _sys.platform == 'darwin':
				rss_gb = raw / (1024.0 ** 3)
			else:
				rss_gb = raw / (1024.0 ** 2)
			print(f"[mem-release] {label}: peak RSS ~ {rss_gb:.2f} GB")
		except Exception:
			pass

def calc_pct_volume_within_latency(sas, adv):
	routed_through_ingress, _ = sas.calculate_ground_truth_ingress(adv)
	## want latency -> amt of volume that can reach popp within that latency / amt of volume that could possibly reach it
	all_lats = list([l for ug in sas.ug_perfs for l in sas.ug_perfs[ug].values()])
	min_overall_lat = np.min(all_lats)
	max_overall_lat = np.max(all_lats)
	lats = np.linspace(min_overall_lat, max_overall_lat, num=100)
	ret = np.zeros((len(lats)))
	for i,lat in tqdm.tqdm(enumerate(lats), desc="Finding volume within each latency threshold..."):
		vol_this_lat_possible = 0
		for ug in sas.ug_perfs:
			for popp,l in sas.ug_perfs[ug].items():
				if l <= lat:
					vol_this_lat_possible += sas.ug_to_vol[ug]
		vol_this_lat_achieved = 0
		counted={}
		for prefix_i in routed_through_ingress:
			for ug,popp in routed_through_ingress[prefix_i].items():
				l = sas.ug_perfs[ug][popp]
				if l <= lat:
					try:
						counted[ug,popp]
					except KeyError:
						vol_this_lat_achieved += sas.ug_to_vol[ug]
						counted[ug,popp] = None
		ret[i] = vol_this_lat_achieved / vol_this_lat_possible
	return {'latencies':lats, 'volume_fractions':ret}

def evaluate_all_metrics(dpsize, port, save_run_dir=None, **kwargs):
	# for each deployment, get different advertisement strategies
	# look at latency under popp and link failures compared to optimal
	# sparse should be simultaneously better in both normal and failure scenarios

	### save_run_dir can be None (just start from scratch), single directory (in which case nsim must be 1), or list of directories of length
	### nsim, some of which may be None 

	# np.random.seed(31413)
	metrics = {}

	### scale individual metro/thing volume
	X_vals = np.linspace(10,500,num=20)#[10,40,80,100,130,150,180,200,210,220,250]#[10,20,30,40,50,60,70,80,90,100]
	### overprovisioning factor (1.3 = 30%)
	Y_vals = [1.3]

	wm = None
	sas = None

	performance_metrics_fn = kwargs.get('use_performance_metrics_fn', global_performance_metrics_fn(dpsize))
	# SCULPTOR_SOLN_TYPES: comma list overriding the default strategy set
	# (same convention as experiments/eods/run_eods_cell.py). Lets an A/B
	# run sparse-only without paying for the baselines + their evals.
	_env_solns = [t for t in os.environ.get('SCULPTOR_SOLN_TYPES', '').split(',') if t]
	soln_types = kwargs.get('soln_types', _env_solns or global_soln_types)
	if 'soln_types' in kwargs:
		del kwargs['soln_types']

	## Format save_run_dirs
	N_TO_SIM = kwargs.get('nsim',1)
	if N_TO_SIM > 1:
		if save_run_dir is not None:
			assert type(save_run_dir) == list and len(save_run_dir) == N_TO_SIM
			save_run_dirs = save_run_dir
		else:
			save_run_dirs = [None for _ in range(N_TO_SIM)]
	else:
		save_run_dirs = [save_run_dir]

	metrics = copy.deepcopy(default_metrics)
	if os.path.exists(performance_metrics_fn):
		metrics = pickle.load(open(performance_metrics_fn,'rb'))
	for k in list(metrics):
		if k not in default_metrics:
			del metrics[k]
	for k in default_metrics:
		if k not in metrics:
			print(k)
			metrics[k] = copy.deepcopy(default_metrics[k])
		for i in range(N_TO_SIM):
			for k in metrics:
				if i not in metrics[k]:# and i in default_metrics[k]:
					print("{} {}".format(k, i))
					metrics[k][i] = copy.deepcopy(default_metrics[k][0])
	try:
		for random_iter in range(N_TO_SIM):
			try:
				if save_run_dirs[random_iter] is not None: ## we want to hotstart on a save run dir and continue training
					raise TypeError
				metrics['compare_rets'][random_iter]['n_advs'] ## if this field is populated, we've already computed this iteration's solution
				continue
			except (TypeError, KeyError):
				# TypeError: hotstart requested, or compare_rets slot is None.
				# KeyError: n_advs absent (e.g. cleared to force a re-solve,
				# 2026-08-24) -- either way this sim is NOT computed: solve it.
				pass
			print("-----Deployment number = {} -------".format(random_iter))
			_log_mem('eval_solve_strategies', ri=random_iter)
			if save_run_dirs[random_iter] is not None:
				print("Loading from hotstart dir")
				deployment = pickle.load(open(os.path.join(RUN_DIR, save_run_dirs[random_iter], 'state-0.pkl'), 'rb'))['deployment']
				deployment['port'] = port
			elif kwargs.get('prefix_deployment') is not None:
				print("Prefixing deployment")
				deployment = kwargs.get('prefix_deployment')
				deployment['port'] = port
			else:
				# SCULPTOR_EVAL_SEED: pair A/B arms on identical per-sim
				# deployments by driving SCULPTOR_DEPLOYMENT_SEED (which
				# reseeds INSIDE get_random_deployment, immune to upstream
				# RNG drift -- a main()-level np.random.seed was not: arms
				# consume different amounts of RNG before each draw). Seed
				# varies per sim AND per attempt: the <20-popps filter
				# below redraws, and a fixed seed would redraw the same
				# rejected deployment forever.
				_es = os.environ.get('SCULPTOR_EVAL_SEED')
				_attempt = 0
				while True:
					try:
						if _es:
							os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(
								int(_es) + 1000 * random_iter + _attempt)
						_attempt += 1
						deployment = get_random_deployment(dpsize, **kwargs)
						deployment['port'] = port
						if len(deployment['popps']) < 20:
							continue
						break
					except:
						## It could be that this function fails because our random PoP selection isn't great
						## Just keep trying and it will eventually work
						import traceback
						traceback.print_exc()
			metrics['deployment'][random_iter] = deployment

			# ---- depstore lookup (SCULPTOR_DEPSTORE=1, Tom 2026-08-30):
			# a full compare_ret for THIS deployment (content-id keyed, so
			# random deployments can never collide) under the current
			# semantic fingerprint skips solving entirely.
			_ds_on = os.environ.get('SCULPTOR_DEPSTORE', '0') == '1'
			if _ds_on:
				from core import depstore as _dstore
				_ds = _dstore.Depstore()
				# kwargs-borne SEMANTIC config must enter the key (Tom
				# 2026-08-30 review caught it): n_prefixes etc. arrive as
				# function args, not env -- without them, two prefix
				# budgets on the SAME deployment share a fingerprint and
				# cross-hit. Resolved values, over-keyed.
				_dcfg = {'dpsize': str(dpsize),
						 'dep_id': _dstore.deployment_id(deployment),
						 'n_prefixes': str(kwargs.get('n_prefixes',
													  'auto')),
						 'generic_objective': str(kwargs.get(
							 'generic_objective',
							 os.environ.get('SCULPTOR_GENERIC_OBJECTIVE',
											'avg_latency'))),
						 'gamma': str(gamma),
						 'lambduh': str(lambduh),
						 'capacity': str(capacity)}
				_want_iters = int(os.environ.get('SCULPTOR_MAX_ITER') or 0)
				_art = _ds.get_training(min_iters=_want_iters, config=_dcfg)
				_blob = (_ds.get_eval(_art.fp, 'compare_ret')
						 if _art is not None else None)
				if _blob is not None and 'latencies' not in _blob:
					print('[depstore] stale blob (pre-latencies era) for '
						  'sim {} -- treating as MISS'.format(random_iter),
						  flush=True)
					_blob = None
				if _blob is not None:
					print('[depstore] HIT sim {}: fp={} n_iters={} -- '
						  'skipping solve'.format(random_iter, _art.fp,
												  _art.n_iters), flush=True)
					metrics['compare_rets'][random_iter] = _blob['ret']
					metrics['save_run_dir'][random_iter] = None
					metrics['ug_to_vol'][random_iter] = _blob['ug_to_vol']
					metrics['best_latencies'][random_iter] = _blob['best_latencies']
					metrics['settings'][random_iter] = _blob.get('settings')
					metrics['adv'][random_iter] = _blob['adv']
					metrics['latencies'][random_iter] = _blob['latencies']
					if (os.environ.get('SCULPTOR_DEPSTORE_MODE') ==
							'eval_only' and _ds.get_eval(
							_art.fp, 'paper_evals_done') is None):
						print('[depstore] eval_only: training cached but '
							  'paper evals never ran for fp {} -- run the '
							  'eval pass (full pipeline) first'.format(
								  _art.fp), flush=True)
						os._exit(3)
					continue
				print('[depstore] MISS sim {}: {}'.format(
					random_iter,
					_ds.why_miss(min_iters=_want_iters, config=_dcfg)),
					flush=True)
				# SCULPTOR_DEPSTORE_MODE=eval_only (Tom 2026-08-30
				# skeleton): evals must consume CACHED training only --
				# a miss means the training script has not run; fail
				# loudly instead of silently retraining.
				if os.environ.get('SCULPTOR_DEPSTORE_MODE') == 'eval_only':
					# os._exit: the surrounding bare except swallows
					# SystemExit (caught by the pipeline test step (d))
					print('[depstore] eval_only: training cache MISS for '
						  'sim {} -- run the training pass first '
						  '(SCULPTOR_DEPSTORE_MODE=train_only)'.format(
							  random_iter), flush=True)
					os._exit(3)

			n_prefixes = kwargs.get('n_prefixes', deployment_to_prefixes(deployment))
	
			sas = Sparse_Advertisement_Eval(deployment, verbose=True,
				lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
				using_resilience_benefit=(os.environ.get('SCULPTOR_USE_RESILIENCE','1')=='1'), gamma=gamma, n_prefixes=n_prefixes,
				save_run_dir=save_run_dirs[random_iter],
				generic_objective=kwargs.get('generic_objective',
					os.environ.get('SCULPTOR_GENERIC_OBJECTIVE', 'avg_latency')))

			metrics['settings'][random_iter] = sas.get_init_kwa()
			if wm is None:
				wm = Worker_Manager(sas.get_init_kwa(), deployment)
				# Adaptive resize: start with SCULPTOR_N_WORKERS_DURING_PARALLEL
				# workers if set, so the concurrent parallel-strategy
				# subprocesses (painter etc.) launched by
				# compare_different_solutions face less Ray-side contention.
				# After those subprocesses finish, a watcher thread requests
				# ramp-up to the SCULPTOR_N_WORKERS target via
				# wm.request_add_workers, applied at the next sparse iter
				# boundary via wm.process_pending_resize.
				_dp_env = os.environ.get('SCULPTOR_N_WORKERS_DURING_PARALLEL')
				_dp_initial = None
				if _dp_env is not None:
					try:
						_dp_initial = int(_dp_env)
					except ValueError:
						print("WARNING: SCULPTOR_N_WORKERS_DURING_PARALLEL={!r} is not an int; ignoring".format(_dp_env))
				wm.start_workers(n_workers_override=_dp_initial)
			sas.set_worker_manager(wm)
			sas.update_deployment(deployment)
			### Solve the problem for each type of solution (sparse, painter, etc...)
			# Per-strategy checkpoint: pickle the metrics-so-far after every
			# solution_type completes, so a crash in (e.g.) painter doesn't
			# lose the SCULPTOR advertisement we already computed. The
			# closure captures `metrics` from the outer scope; we stash the
			# partial compare_rets onto it so downstream eval phases that
			# read metrics['compare_rets'][random_iter] see what's been
			# done so far.
			def _on_strategy_complete(solution_type, partial_metrics):
				metrics['compare_rets'][random_iter] = partial_metrics
				try:
					pickle.dump(metrics, open(performance_metrics_fn, 'wb'))
					print("[ckpt] saved metrics after strategy={} → {}".format(
						solution_type, performance_metrics_fn))
				except Exception:
					import traceback
					traceback.print_exc()
			ret = sas.compare_different_solutions(n_run=1, verbose=True,
				 dont_update_deployment=True, soln_types=soln_types,
				 on_strategy_complete=_on_strategy_complete, **kwargs)
			metrics['compare_rets'][random_iter] = ret
			metrics['save_run_dir'][random_iter] = sas.sas.save_run_dir # sparse's save run dir
			ug_vols = sas.ug_to_vol
			metrics['ug_to_vol'][random_iter] = sas.ug_vols
			metrics['best_latencies'][random_iter] = copy.copy(sas.best_lats_by_ug)
			if _ds_on:
				try:
					_adv = np.asarray(ret['adv_solns']['sparse'][0])
					_trained_iters = int(getattr(sas.sas, 'iter', 0) or
										 os.environ.get('SCULPTOR_MAX_ITER') or 0)
					_fp = _ds.put_training(
						_adv, _trained_iters, deployment=deployment,
						config=_dcfg,
						provenance={'src': 'eval_all_solution_types',
									'dpsize': str(dpsize)})
					if _fp:
						_ds.put_eval(_fp, 'compare_ret', {
							'ret': ret, 'ug_to_vol': sas.ug_vols,
							'best_latencies': copy.copy(sas.best_lats_by_ug),
							'settings': metrics['settings'][random_iter]})
						print('[depstore] PUT sim {}: fp={} it={}'.format(
							random_iter, _fp, _trained_iters), flush=True)
				except Exception as _e:
					print('[depstore] put failed (non-fatal): {}'.format(_e),
						  flush=True)
			for solution in soln_types:
				try:
					adv = ret['adv_solns'][solution][0]
				except:
					print("No solution for {}".format(solution))
					continue
				pre_lats_by_ug = sas.solve_lp_with_failure_catch(adv)['lats_by_ug']

				metrics['adv'][random_iter][solution] = adv
				metrics['latencies'][random_iter][solution] = pre_lats_by_ug

			if _ds_on and '_fp' in dir() and _fp:
				try:
					_ds.put_eval(_fp, 'compare_ret', {
						'ret': metrics['compare_rets'][random_iter],
						'ug_to_vol': metrics['ug_to_vol'][random_iter],
						'best_latencies': metrics['best_latencies'][random_iter],
						'settings': metrics['settings'][random_iter],
						'adv': metrics['adv'][random_iter],
						'latencies': metrics['latencies'][random_iter]})
					print('[depstore] PUT complete blob for sim {} '
						  '(incl. adv/latencies)'.format(random_iter),
						  flush=True)
				except Exception as _e:
					print('[depstore] blob update failed (non-fatal): '
						  '{}'.format(_e), flush=True)

			pickle.dump(metrics, open(performance_metrics_fn,'wb'))

	except:
		import traceback
		traceback.print_exc()

	# SCULPTOR_DEPSTORE_MODE=train_only (Tom 2026-08-30 skeleton): the
	# training pass ends here -- solutions are trained + PUT into the
	# depstore, family evals are the eval pass's job.
	if os.environ.get('SCULPTOR_DEPSTORE_MODE') == 'train_only':
		print('[depstore] train_only: stopping before family evals',
			  flush=True)
		return metrics

	# RECALC_LATENCY_WITH_PENALTY = False
	# try:
	# 	changed=False
	# 	for random_iter in range(N_TO_SIM):
	# 		k_of_interest = 'latencies_penalty'
	# 		havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)
	# 		if RECALC_LATENCY_WITH_PENALTY or havent_calced_everything:
	# 			print("-----Latency with penalty calc for deployment number = {} -------".format(random_iter))
	# 			if sas is None:
	# 				deployment = metrics['deployment'][random_iter]
	# 				deployment['port'] = port

	# 				n_prefixes = kwargs.get('n_prefixes', deployment_to_prefixes(deployment))
	# 				sas = Sparse_Advertisement_Eval(deployment, verbose=True,
	# 					lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
	# 					using_resilience_benefit=(os.environ.get('SCULPTOR_USE_RESILIENCE','1')=='1'), gamma=gamma, n_prefixes=n_prefixes)
	# 				if wm is None:
	# 					wm = Worker_Manager(sas.get_init_kwa(), deployment)
	# 					wm.start_workers()
	# 				sas.set_worker_manager(wm)
	# 				sas.update_deployment(deployment)
	# 			else:
	# 				deployment = metrics['deployment'][random_iter]
	# 				deployment['port'] = port
	# 				sas.update_deployment(deployment)
	# 			ug_vols = sas.ug_to_vol
	# 			ret = metrics['compare_rets'][random_iter]
	# 			for solution in soln_types:
	# 				try:
	# 					adv = ret['adv_solns'][solution][0]
	# 				except:
	# 					print("No solution for {}".format(solution))
	# 					continue

	# 				print("Assessing latency with penalty for {}".format(solution))
	# 				one_per_peer_adv = np.eye(sas.n_popps)
	# 				penalty_lats_by_ug = sas.solve_lp_with_failure_catch_weighted_penalty(adv, one_per_peer_adv)['lats_by_ug']
	# 				metrics['latencies_penalty'][random_iter][solution] = penalty_lats_by_ug
	# 				changed=True

	# 	if changed:
	# 		pickle.dump(metrics, open(performance_metrics_fn,'wb'))
	# except:
	# 	import traceback
	# 	traceback.print_exc()

	# Drop the driver-side LP solution cache between phases. The eval LPs
	# don't reuse keys across phases (each scenario has a unique cache_rep),
	# so the cache only accumulates memory -- enough to OOM the head at
	# actual-32 if left to grow across all 6 eval phases. See clear_lp_caches
	# docstring in optimal_adv_wrapper.
	if sas is not None:
		sas.clear_lp_caches()

	# ---- objective-dependent half -------------------------------------
	# Everything from here used to be inlined: the failure/diurnal/flash-crowd
	# phases and all 11 comparison panels, all written against latency in ms
	# weighted by UG volume. Run after an MLU or priority optimisation they
	# produced numbers that looked valid and meant nothing. Split out
	# 2026-08-21; objective_hooks routes to the right suite.
	# SCULPTOR_REQUIRE_SOLNS, enforced at EVAL time too (2026-08-24): the
	# solve-time guard is bypassed when a stale pickle resume-skips or a
	# swallowed exception jumps the strategy loop -- the per_site_cost cell
	# "passed" its banner with every sparse metric None. A required
	# solution missing HERE is the same lie one phase later.
	_req = [x for x in os.environ.get('SCULPTOR_REQUIRE_SOLNS', '').split(',')
			if x]
	if _req:
		for _ri in range(N_TO_SIM):
			_cr = (metrics.get('compare_rets') or {}).get(_ri) or {}
			_missing = [s for s in _req
						if not (_cr.get('adv_solns') or {}).get(s)]
			if _missing:
				print('!' * 72)
				print('[FATAL] required solution(s) {} missing at eval time '
					  '(sim {}) -- aborting instead of emitting a sparse-less '
					  'comparison.'.format(_missing, _ri))
				print('!' * 72)
				raise SystemExit(43)
	objective = objective_hooks.resolve_objective(kwargs.get('generic_objective'))
	ctx = objective_hooks.EvalContext(
		sas=sas, wm=wm, metrics=metrics, soln_types=soln_types, dpsize=dpsize,
		port=port, kwargs=kwargs, N_TO_SIM=N_TO_SIM,
		performance_metrics_fn=performance_metrics_fn,
		default_metrics=default_metrics, lambduh=lambduh, gamma=gamma,
		capacity=capacity, X_vals=X_vals, Y_vals=Y_vals, objective=objective)
	# valid_iters and save_fig_fn are derived inside the suite (from
	# SIM_INDS_TO_PLOT and kwargs respectively), so they are not passed here.
	hooks = objective_hooks.for_objective(objective)
	print('[eval] objective={} -> {}'.format(objective, hooks.__name__))
	metrics = hooks.run(ctx) or metrics
	# Persist what the hook scored. The latency suite checkpoints
	# internally, but the thin objective modules (mlu, site-cost,
	# frac-beyond) only mutate ctx.metrics in memory -- without this dump
	# their *_by_strategy keys never reached the pickle and the paper
	# table read '-' for every cell (2026-08-22).
	try:
		pickle.dump(metrics, open(performance_metrics_fn, 'wb'))
		print("[ckpt] saved metrics after objective hook -> {}".format(
			performance_metrics_fn))
	except Exception:
		import traceback
		traceback.print_exc()

	# depstore: stamp 'paper evals ran' per trained fp so eval_only can
	# distinguish trained-but-never-evaluated from fully done (skeleton;
	# the marker payload will grow into a real eval-artifact index).
	if os.environ.get('SCULPTOR_DEPSTORE', '0') == '1':
		try:
			from core import depstore as _dstore
			_ds2 = _dstore.Depstore()
			_n_stamped = 0
			for _ri, _dep in (metrics.get('deployment') or {}).items():
				if not _dep:
					continue
				_cfg2 = {'dpsize': str(dpsize),
						 'dep_id': _dstore.deployment_id(_dep)}
				_a2 = _ds2.get_training(min_iters=0, config=_cfg2)
				if _a2 is not None:
					_ds2.put_eval(_a2.fp, 'paper_evals_done',
								  {'sim': _ri, 'done': True})
					_n_stamped += 1
			print('[depstore] paper_evals_done stamped for {} sim(s)'
				  .format(_n_stamped), flush=True)
		except Exception as _e:
			print('[depstore] marker stamp failed (non-fatal): {}'.format(
				_e), flush=True)

	return metrics

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_run_dir", default=None)
	parser.add_argument("--use_cache_deployment", action='store_true', default=False)
	parser.add_argument("--dpsize", default=None, required=True)
	# --port is vestigial under Ray -- nothing binds it. Optional since
	# 2026-08-21; kept only because deployment dicts carry the field.
	parser.add_argument("--port", default=DEFAULT_PORT, type=int)
	# 2026-08-21: the objective used to be hardcoded to avg_latency here, so
	# there was no way to evaluate site-cost / MLU / frac-beyond without
	# editing the file. Env var honoured too, for worker subprocesses.
	parser.add_argument("--objective", default=None,
		help="generic objective (avg_latency, per_site_cost, max_util, "
			 "frac_beyond_optimal, ...). Extension objectives need "
			 "SCULPTOR_XOBJS=1.")
	args = parser.parse_args()
	if args.objective:
		os.environ['SCULPTOR_GENERIC_OBJECTIVE'] = args.objective

	port = int(args.port)

	np.random.seed(31415)
	if args.save_run_dir is not None:
		## we could specify an array of hotstart dirs otherwise, but that's a task for another day
		assert N_TO_SIM == 1
		evaluate_all_metrics(args.dpsize, int(port), save_run_dir=args.save_run_dir)
	elif args.use_cache_deployment:
		deployment = pickle.load(open(global_performance_metrics_fn(dpsize), 'rb'))['deployment'][0]
		evaluate_all_metrics(args.dpsize, int(port), prefix_deployment=deployment)
	else:
		evaluate_all_metrics(args.dpsize, int(port))



		
