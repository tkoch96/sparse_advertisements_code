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
# Mirror the driver's imports: this module is the objective-dependent half of
# eval_all_solution_types and the moved code resolves the same names
# (check_calced_everything, assess_failure_resilience, get_cdf_xy, ...).
from evaluations.wrapper_eval import *
from core.solve_lp_assignment import *
from core.sparse_advertisements_v3 import *
import pickle, numpy as np, matplotlib.pyplot as plt, copy, itertools, time
# Helpers that stayed with the driver (they are used by both halves). Safe
# despite the apparent cycle: objective_hooks imports this module lazily inside
# for_objective(), so nothing is mid-initialisation when this resolves.
from evaluations.eval_all_solution_types import (
	_log_mem, _release_memory, _fit_x_to_data, calc_pct_volume_within_latency)
# eval phase RAM/timing markers (Tom 2026-08-20). LAZY import: a
# top-level 'from sparse_advertisements_v3 import _log_mem' here fires
# the circular-import chain early and the later star-import then sees a
# partially-initialized module (get_random_deployment NameError, found
# on the instrumented 32 pre-flight).


"""Evaluation phases and figures for the latency + gamma*resilience objective.

This is the objective SCULPTOR is usually trained on, and until 2026-08-21 it
was the ONLY evaluation `eval_all_solution_types` could run -- these phases
were inlined there, so an MLU or priority run silently got latency-shaped
numbers. They now live behind `objective_hooks`.

Phases, in order (each is staleness-gated and re-pickles metrics as it goes,
so a crash in one keeps the earlier ones):

  1. pct-volume-within-latency multipliers
  2. failure resilience -- single-link and single-PoP, mutable + sticky
  3. diurnal cycle
  4. flash-crowd / DDoS resilience

then the comparison figure: 11 panels, titled, x-limits fitted to the data.

Everything here assumes latency in milliseconds weighted by UG volume. If you
are adding an objective, add a sibling module rather than a branch in here --
see evaluations/objective_hooks.py.
"""

OBJECTIVES = ('avg_latency',)


def run(ctx):
	# uniform objective-value column (Tom 2026-08-23): one LP evaluation
	# of THE objective per stored advertisement, same function for every
	# strategy, per-sim deployment (see _objective_eval_base).
	try:
		from evaluations.objectives._objective_eval_base import (
			score_all_strategies, objective_value_scorer)
		score_all_strategies(ctx, objective_value_scorer('avg_latency'),
							 'objective_value_by_strategy')
	except Exception:
		import traceback
		traceback.print_exc()
	"""Run the latency+resilience phases and draw the comparison figure.

	The body below moved verbatim out of evaluate_all_metrics on 2026-08-21;
	these unpacks exist so it still reads against the same local names.
	"""
	sas = ctx.sas
	wm = ctx.wm
	metrics = ctx.metrics
	soln_types = ctx.soln_types
	dpsize = ctx.dpsize
	port = ctx.port
	kwargs = ctx.kwargs
	N_TO_SIM = ctx.N_TO_SIM
	performance_metrics_fn = ctx.performance_metrics_fn
	valid_iters = ctx.valid_iters
	default_metrics = ctx.default_metrics
	lambduh = ctx.lambduh
	gamma = ctx.gamma
	capacity = ctx.capacity
	save_fig_fn = ctx.save_fig_fn
	X_vals = ctx.X_vals
	Y_vals = ctx.Y_vals

	# SCULPTOR_RECALC: comma list of metric families to force-recompute
	# against the CACHED solutions (Tom 2026-08-23: "precisely target
	# which thing you want to recompute without custom scripts").
	# Families: pct_vol, failure, volume, diurnal, flash,
	# diurnal_bisect, flash_bisect (bisections only, grids untouched) (or 'all').
	# Everything else keeps the load->skip-if-computed contract.
	_recalc = {x.strip() for x in os.environ.get('SCULPTOR_RECALC', '').split(',') if x.strip()}
	_force = lambda fam: 'all' in _recalc or fam in _recalc
	RECALC_PCT_VOL_IN_LAT_MULTIPLIERS = _force('pct_vol')
	try:
		for random_iter in range(N_TO_SIM):
			k_of_interest = 'pct_volume_within_latency'
			havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)
			if RECALC_PCT_VOL_IN_LAT_MULTIPLIERS or havent_calced_everything:
				print("-----Volume calc for deployment number = {} -------".format(random_iter))
				_log_mem('eval_volume_calc', ri=random_iter)
				# fresh eval object per sim (2026-08-22): reusing one sas via
				# update_deployment left stale driver-side derived state across
				# differently-sized deployments (IndexError inside
				# compute_one_per_peering_solution; the stats TypeErrors were
				# downstream of the unfilled defaults). Construction is the only
				# safe cross-deployment update; the worker pool alone is reused.
				sas = None
				if sas is None:
					deployment = metrics['deployment'][random_iter]
					deployment['port'] = port

					n_prefixes = kwargs.get('n_prefixes', deployment_to_prefixes(deployment))
					sas = Sparse_Advertisement_Eval(deployment, verbose=True,
						lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
						using_resilience_benefit=(os.environ.get('SCULPTOR_USE_RESILIENCE','1')=='1'), gamma=gamma, n_prefixes=n_prefixes)
					if wm is None:
						wm = Worker_Manager(sas.get_init_kwa(), deployment)
						wm.start_workers()
					sas.set_worker_manager(wm)
					sas.update_deployment(deployment)
				else:
					deployment = metrics['deployment'][random_iter]
					deployment['port'] = port
					sas.update_deployment(deployment)
				ug_vols = sas.ug_to_vol
				ret = metrics['compare_rets'][random_iter]
				print(ret)
				for solution in soln_types:
					try:
						adv = ret['adv_solns'][solution][0]
					except:
						print("No solution for {}".format(solution))
						continue
					print("Assessing pct volume within latency for {}".format(solution))
					m = calc_pct_volume_within_latency(sas, adv)
					metrics['pct_volume_within_latency'][random_iter][solution] = m

					
				pickle.dump(metrics, open(performance_metrics_fn,'wb'))
	except:
		import traceback
		traceback.print_exc()

	if sas is not None: sas.clear_lp_caches()

	RECALC_FAILURE_METRICS = _force('failure')
	try:
		changed=False
		for random_iter in range(N_TO_SIM):
			k_of_interest = 'popp_failures_latency_optimal_specific'
			havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)

			if RECALC_FAILURE_METRICS or havent_calced_everything:
				print("-----Failure calc for deployment number = {} -------".format(random_iter))
				_log_mem('eval_failure_calc', ri=random_iter)
				# FRESH driver-side eval object per sim (2026-08-22).
				# Reusing one sas across sims via update_deployment left
				# driver-side derived state keyed to the PREVIOUS sim's
				# deployment -- compute_one_per_peering_solution inside
				# update_deployment then indexed a 122-popp array with a
				# 128-popp index (IndexError) and the phase silently
				# produced empty failure metrics for every sim. Same
				# disease as the worker-side nsim>1 rebirth bug, driver
				# edition: the only safe update across differently-sized
				# deployments is construction. The worker pool IS reused
				# (workers rebirth correctly on the update push).
				deployment = metrics['deployment'][random_iter]
				deployment['port'] = port
				n_prefixes = kwargs.get('n_prefixes', deployment_to_prefixes(deployment))
				sas = Sparse_Advertisement_Eval(deployment, verbose=True,
					lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
					using_resilience_benefit=(os.environ.get('SCULPTOR_USE_RESILIENCE','1')=='1'), gamma=gamma, n_prefixes=n_prefixes)
				if wm is None:
					wm = Worker_Manager(sas.get_init_kwa(), deployment)
					wm.start_workers()
				sas.set_worker_manager(wm)
				sas.update_deployment(deployment)
				changed=True

				# Compute one-per-peering reference LPs once and share across
				# all strategies. The OPP-ref LP depends only on
				# (which, failed_popp_or_pop), not on the strategy -- so the
				# legacy assess_failure_resilience was solving the same set
				# of 811 reference LPs 6 times (once per strategy). At
				# actual-32 that's ~3900 redundant LP solves per phase.
				# Skip the precompute if every remaining strategy is already
				# in the metrics pickle (resume case).
				strats_needing_work = [
					s for s in soln_types
					if RECALC_FAILURE_METRICS
					or metrics[k_of_interest][random_iter][s] == default_metrics[k_of_interest][0][s]
				]
				opp_refs_popps = None
				opp_refs_pops = None
				if len(strats_needing_work) > 0:
					print("[opp-ref] precomputing one-per-peering reference LPs for {} strategies".format(
						len(strats_needing_work)))
					t0 = time.time()
					opp_refs_popps = precompute_one_per_peering_failure_lps(sas, which='popps')
					print("[opp-ref] popps precompute: {:.1f}s ({} scenarios)".format(
						time.time() - t0, len(opp_refs_popps)))
					t0 = time.time()
					opp_refs_pops = precompute_one_per_peering_failure_lps(sas, which='pops')
					print("[opp-ref] pops precompute: {:.1f}s ({} scenarios)".format(
						time.time() - t0, len(opp_refs_pops)))

				for solution in soln_types:
					if not RECALC_FAILURE_METRICS:
						if metrics[k_of_interest][random_iter][solution]  != \
							default_metrics[k_of_interest][0][solution]:
							print("Already calced {}".format(solution))
							continue
					print(solution)
					adv = metrics['adv'][random_iter][solution]
					if len(adv) == 0:
						print("No solution for {}".format(solution))
						continue
					try:
						ret = assess_failure_resilience(sas, adv, which='popps', opp_ref_results=opp_refs_popps)
						metrics['popp_failures_congestion'][random_iter][solution] = ret['mutable']['congestion_delta']
						metrics['popp_failures_latency_optimal'][random_iter][solution] = ret['mutable']['latency_delta_optimal']
						metrics['popp_failures_latency_optimal_specific'][random_iter][solution] = ret['mutable']['latency_delta_specific']
						metrics['popp_failures_latency_before'][random_iter][solution] = ret['mutable']['latency_delta_before']

						metrics['popp_failures_sticky_congestion'][random_iter][solution] = ret['sticky']['congestion_delta']
						metrics['popp_failures_sticky_latency_optimal'][random_iter][solution] = ret['sticky']['latency_delta_optimal']
						metrics['popp_failures_sticky_latency_optimal_specific'][random_iter][solution] = ret['sticky']['latency_delta_specific']
						metrics['popp_failures_sticky_latency_before'][random_iter][solution] = ret['sticky']['latency_delta_before']

						# Free the popps ret dict before allocating the pops one, so
						# both don't sit simultaneously in memory.
						del ret

						ret = assess_failure_resilience(sas, adv, which='pops', opp_ref_results=opp_refs_pops)
						metrics['pop_failures_congestion'][random_iter][solution] = ret['mutable']['congestion_delta']
						metrics['pop_failures_latency_optimal'][random_iter][solution] = ret['mutable']['latency_delta_optimal']
						metrics['pop_failures_latency_optimal_specific'][random_iter][solution] = ret['mutable']['latency_delta_specific']
						metrics['pop_failures_latency_before'][random_iter][solution] = ret['mutable']['latency_delta_before']

						metrics['pop_failures_sticky_congestion'][random_iter][solution] = ret['sticky']['congestion_delta']
						metrics['pop_failures_sticky_latency_optimal'][random_iter][solution] = ret['sticky']['latency_delta_optimal']
						metrics['pop_failures_sticky_latency_optimal_specific'][random_iter][solution] = ret['sticky']['latency_delta_specific']
						metrics['pop_failures_sticky_latency_before'][random_iter][solution] = ret['sticky']['latency_delta_before']
						del ret
					except:
						import traceback
						traceback.print_exc()
						continue
					finally:
						# Return per-strategy LP heap back to OS so RSS doesn't
						# monotonically grow across the 6 strategies. Also
						# checkpoint the metrics pickle so a later OOM doesn't
						# wipe out completed strategies (mirrors the
						# on_strategy_complete pattern from
						# compare_different_solutions).
						_release_memory(label=f"after failure-eval strategy={solution}")
						try:
							pickle.dump(metrics, open(performance_metrics_fn, 'wb'))
							print(f"[ckpt] saved metrics after failure-eval strategy={solution}")
						except Exception:
							import traceback
							traceback.print_exc()


		if changed:
			pickle.dump(metrics, open(performance_metrics_fn,'wb'))

	except:
		import traceback
		traceback.print_exc()

	if sas is not None: sas.clear_lp_caches()

	RECALC_VOL_MULTIPLIERS = _force('volume')

	if sas is not None: sas.clear_lp_caches()

	RECALC_DIURNAL = _force('diurnal')
	RECALC_DIURNAL_BISECT = _force('diurnal_bisect')
	diurnal_multipliers = [25,50,65,70,75,85,95,105,115,125,150]
	try:
		changed=False
		for random_iter in range(N_TO_SIM):
			k_of_interest = 'diurnal'
			havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)
			if RECALC_DIURNAL or RECALC_DIURNAL_BISECT or havent_calced_everything:
				print("-----Diurnal calc for deployment number = {} -------".format(random_iter))
				_log_mem('eval_diurnal', ri=random_iter)
				# fresh eval object per sim (2026-08-22): reusing one sas via
				# update_deployment left stale driver-side derived state across
				# differently-sized deployments (IndexError inside
				# compute_one_per_peering_solution; the stats TypeErrors were
				# downstream of the unfilled defaults). Construction is the only
				# safe cross-deployment update; the worker pool alone is reused.
				sas = None
				if sas is None:
					deployment = metrics['deployment'][random_iter]
					deployment['port'] = port

					n_prefixes = kwargs.get('n_prefixes', deployment_to_prefixes(deployment))
					sas = Sparse_Advertisement_Eval(deployment, verbose=True,
						lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
						using_resilience_benefit=(os.environ.get('SCULPTOR_USE_RESILIENCE','1')=='1'), gamma=gamma, n_prefixes=n_prefixes)
					if wm is None:
						wm = Worker_Manager(sas.get_init_kwa(), deployment)
						wm.start_workers()
					sas.set_worker_manager(wm)
					sas.update_deployment(deployment)
				else:
					deployment = metrics['deployment'][random_iter]
					deployment['port'] = port
					sas.update_deployment(deployment)
				diurnal_deployments = get_diurnal_deployments(sas, diurnal_multipliers)
				ret = metrics['compare_rets'][random_iter]
				for solution in soln_types:
					if not RECALC_DIURNAL:
						if metrics[k_of_interest][random_iter][solution]  != \
							default_metrics[k_of_interest][0][solution]:
							print("Already calced {}".format(solution))
							continue
					try:
						adv = ret['adv_solns'][solution][0]
					except:
						print("No solution for {}".format(solution))
						continue
					try:
						print("Assessing diurnal effect for {}, sim number {}".format(solution,random_iter))
						## can reuse the flash crowd function since its the same
						## "X_vals ~ hours of day 0 - 23"
						## "Y_vals" ~ intensities
						metrics[k_of_interest][random_iter][solution] = assess_resilience_to_flash_crowds_mp(sas, adv, solution, list(range(24)), diurnal_multipliers, diurnal_deployments)
						changed=True
					except:
						import traceback
						traceback.print_exc()

				# Per-sim, ALL-solutions batched bisection of the critical
				# diurnal multiplier (same rationale as the flash version;
				# the Y grid floor-clipped everyone to its lowest point).
				try:
					_have_bisect = (metrics.get('bisect_critical_diurnal') or {}).get(random_iter)
					if _have_bisect and not (RECALC_DIURNAL or RECALC_DIURNAL_BISECT):
						raise StopIteration   # already bisected this sim; caught below
					_advs = {}
					for _sol in soln_types:
						try:
							_advs[_sol] = ret['adv_solns'][_sol][0]
						except (KeyError, IndexError):
							pass
					if _advs:
						def _mk_diur(v, _sas=sas):
							# hour entries are wrapped one level deeper
							# ({hour: {'None': deployment}}) -- flatten, or
							# the worker receives the wrapper dict and dies
							# with KeyError 'dpsize' (caught by the small
							# nsim=3 smoke, 2026-08-23)
							_d = get_diurnal_deployments(_sas, [v])
							return [dep for h in sorted(_d[v])
									for dep in _d[v][h].values()]
						_lo = 1.0
						_hi = float(max(diurnal_multipliers)) if len(diurnal_multipliers) else 100.0
						_res = bisect_critical_intensities(
							sas, _advs, _mk_diur, _lo, _hi, rel_tol=0.05,
							label='bisect-diurnal sim={}'.format(random_iter))
						metrics.setdefault('bisect_critical_diurnal', {})[
							random_iter] = {k: float(v) for k, v in _res.items()}
						print('[bisect-diurnal] sim={} {}'.format(random_iter,
							{k: round(v, 2) for k, v in _res.items()}), flush=True)
						changed=True
						# checkpoint per sim: the family-end dump lost sims 0-1
						# when the 2026-08-23 GCS death killed sim 2 mid-flight
						pickle.dump(metrics, open(performance_metrics_fn,'wb'))
				except StopIteration:
					pass
				except Exception:
					import traceback
					traceback.print_exc()

		if changed:
			pickle.dump(metrics, open(performance_metrics_fn,'wb'))
	except:
		import traceback
		traceback.print_exc()


	if sas is not None: sas.clear_lp_caches()

	### Calculates some measure of practical resilience for each strategy
	### current resilience measure is flash crowd / DDoS attack in a region
	RECALC_RESILIENCE = _force('flash')
	RECALC_FLASH_BISECT = _force('flash_bisect')
	try:
		changed=False
		for random_iter in range(N_TO_SIM):
			k_of_interest = 'resilience_to_congestion'
			havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)
			if RECALC_RESILIENCE or RECALC_FLASH_BISECT or havent_calced_everything:
				print("-----Flash crowd calc for deployment number = {} -------".format(random_iter))
				_log_mem('eval_flash_crowd', ri=random_iter)
				# fresh eval object per sim (2026-08-22): reusing one sas via
				# update_deployment left stale driver-side derived state across
				# differently-sized deployments (IndexError inside
				# compute_one_per_peering_solution; the stats TypeErrors were
				# downstream of the unfilled defaults). Construction is the only
				# safe cross-deployment update; the worker pool alone is reused.
				sas = None
				if sas is None:
					deployment = metrics['deployment'][random_iter]
					deployment['port'] = port

					n_prefixes = kwargs.get('n_prefixes', deployment_to_prefixes(deployment))
					sas = Sparse_Advertisement_Eval(deployment, verbose=True,
						lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
						using_resilience_benefit=(os.environ.get('SCULPTOR_USE_RESILIENCE','1')=='1'), gamma=gamma, n_prefixes=n_prefixes)
					if wm is None:
						wm = Worker_Manager(sas.get_init_kwa(), deployment)
						wm.start_workers()
					sas.set_worker_manager(wm)
					sas.update_deployment(deployment)
				else:
					deployment = metrics['deployment'][random_iter]
					deployment['port'] = port
					sas.update_deployment(deployment)
				inflated_deployments = get_inflated_metro_deployments(sas, X_vals, Y_vals)
				ug_vols = sas.ug_to_vol
				ret = metrics['compare_rets'][random_iter]
				for solution in soln_types:
					if not RECALC_RESILIENCE:
						if metrics[k_of_interest][random_iter][solution]  != \
							default_metrics[k_of_interest][0][solution]:
							print("Already calced {}".format(solution))
							continue
					try:
						adv = ret['adv_solns'][solution][0]
					except:
						print("No solution for {}".format(solution))
						continue
					try:
						print("Assessing resilience to congestion for {}, sim number {}".format(solution,random_iter))
						print("Baseline congestion is {}".format(solve_lp_with_failure_catch(sas, adv)['fraction_congested_volume']))
						# m = assess_resilience_to_congestion(sas, adv, solution, X_vals)['metrics']
						m = assess_resilience_to_flash_crowds_mp(sas, adv, solution, X_vals, Y_vals, inflated_deployments)
						metrics['resilience_to_congestion'][random_iter][solution] = m['metrics']
						metrics['prefix_withdrawals'][random_iter][solution] = m['prefix_withdrawals']
						metrics['fraction_congested_volume'][random_iter][solution] = m['fraction_congested_volume']
						changed=True
					except:
						import traceback
						traceback.print_exc()

				# Per-sim, ALL-solutions batched bisection of the critical
				# surge (Tom 2026-08-23: each deployment/solution has its
				# OWN critical value; the fixed grid snapped them all to
				# one point). Runs at the COMPUTE site -- correct sas and
				# deployment for THIS sim -- batching every solution's
				# midpoint into one LP fan-out per round.
				try:
					_have_bisect = (metrics.get('bisect_critical_flash') or {}).get(random_iter)
					if _have_bisect and not (RECALC_RESILIENCE or RECALC_FLASH_BISECT):
						raise StopIteration
					_advs = {}
					for _sol in soln_types:
						try:
							_advs[_sol] = ret['adv_solns'][_sol][0]
						except (KeyError, IndexError):
							pass
					if _advs:
						_Y0 = Y_vals[len(Y_vals) // 2] if Y_vals else 1.3
						def _mk_flash(v, _sas=sas, _Y=_Y0):
							_d = get_inflated_metro_deployments(_sas, [v], [_Y])
							return list(_d[_Y][v].values())
						_lo = float(min(X_vals)) if len(X_vals) else 1.0
						_hi = float(max(X_vals)) if len(X_vals) else 100.0
						_res = bisect_critical_intensities(
							sas, _advs, _mk_flash, _lo, _hi, rel_tol=0.05,
							label='bisect-flash sim={}'.format(random_iter))
						metrics.setdefault('bisect_critical_flash', {})[
							random_iter] = {k: float(v) for k, v in _res.items()}
						print('[bisect-flash] sim={} {}'.format(random_iter,
							{k: round(v, 2) for k, v in _res.items()}), flush=True)
						changed=True
						pickle.dump(metrics, open(performance_metrics_fn,'wb'))
				except StopIteration:
					pass
				except Exception:
					import traceback
					traceback.print_exc()

		if changed:
			pickle.dump(metrics, open(performance_metrics_fn,'wb'))
	except:
		import traceback
		traceback.print_exc()
	finally:
		if wm is not None:
			wm.stop_workers()

	################################
	### PLOTTING
	################################
	
	i=0
	LATENCY_I = i;i+=1
	PCT_VOL_WITHIN_LATENCY_I = i; i+= 1
	## Mutable
	POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I = i;i+=1
	POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I = i;i+=1


	FLASH_CROWD_LATENCY_VARY_X_I = i;i+=1
	FLASH_CROWD_CONGESTION_VARY_X_I = i;i+=1
	FLASH_CROWD_LIMITING_FACTOR_LATENCY_I = i;i+=1
	FLASH_CROWD_LIMITING_FACTOR_CONGESTION_I = i;i+=1
	

	DIURNAL_LATENCY_VARY_I = i;i+=1
	DIURNAL_CONGESTION_VARY_I = i;i+=1
	DIURNAL_ASSIGNMENT_DELTA_VARY_I = i;i+=1


	n_subs = i
	f,ax=plt.subplots(n_subs,1)
	f.set_size_inches(6,4*n_subs)

	## we plot the performance changes for a single flash crowd volume increase
	single_X_of_interest = X_vals[len(X_vals)//2]
	single_Y_of_interest = Y_vals[len(Y_vals)//2]
	SIM_INDS_TO_PLOT = list(range(N_TO_SIM))

	#### Plotting everything
	for k in list(metrics):
		if 'latency' in k or 'latencies' in k:
			metrics['stats_' + k] = {}
	interesting_latency_suboptimalities = [-10,-50,-100]
	for add_str in ["", "_penalty"]:
		for k in ['stats_latency{}_thresholds_normal'.format(add_str), 'stats_latency{}_thresholds_fail_popp'.format(add_str), 
			'stats_latency{}_thresholds_fail_pop'.format(add_str)]:
			metrics[k] = {solution: {i:{} for i in SIM_INDS_TO_PLOT} for solution in soln_types}
	metrics['stats_resilience_to_congestion'] = {solution: {i:{} for i in SIM_INDS_TO_PLOT} for solution in soln_types}
	_log_mem('eval_stats_assembly')
	metrics['stats_diurnal'] = {solution: {i:{} for i in SIM_INDS_TO_PLOT} for solution in soln_types}

	def get_failure_metric_arr(k, solution, verb=False):
		ret = []
		avg_ret = []
		mind,maxd = np.inf,-1*np.inf
		all_vol = 0
		actually_all_vol = 0
		vol_congested = 0
		vol_best_case_congested = 0
		vol_high_latency = 0
		vol_no_route = 0
		with_high_extra = []
		## storing latency threshold statistics
		threshold_stats = {i:{} for i in SIM_INDS_TO_PLOT}

		for ri in SIM_INDS_TO_PLOT:
			# Skip random_iters where this solution produced no data (e.g. a
			# strategy that failed for that iter). Averaging over the iters it
			# DID succeed on is correct; the original KeyError here bubbled up to
			# the caller's try/except and silently dropped the whole solution.
			if solution not in metrics[k][ri] or not metrics[k][ri][solution]:
				continue
			these_metrics = metrics[k][ri][solution]
			ugs={}

			summaries_by_element = {}
			this_diffs,this_vols = [],[]
			this_sim_total_volume_congested = 0
			this_sim_total_volume = 0
			for fields in these_metrics:
				if len(fields) == 6:
					diff,vol,ug,element,perf1,perf2 = fields
				else:
					diff,vol,ug,element,perf1,perf2,_ = fields
				ugs[ug] = None
				actually_all_vol += vol
				if perf1 == NO_ROUTE_LATENCY: ## the best-case scenario is congested
					vol_best_case_congested += vol
					perf2 = perf1
				else:
					this_sim_total_volume += vol
					# HIGH-LATENCY classification by THRESHOLD, not marker
					# equality (Tom 2026-08-29): a ug with a fraction of its
					# volume stranded carries a marker-BLENDED latency (e.g.
					# half stranded ~ 15015ms) that passed the old equality
					# filter and skewed the mean by 1000x at small sizes.
					# Any >=1% marker blend classifies as high-latency; those
					# users feed the *_with_high aggregate + the frac fields,
					# never the headline mean.
					_hi_cut = 200.0 + 0.01 * (NO_ROUTE_LATENCY - 200.0)
					_is_high = (perf2 >= _hi_cut)
					if not _is_high:
						avg_ret.append((perf1-perf2,vol))
						this_diffs.append(perf1-perf2)
						this_vols.append(vol)
					else:
						vol_high_latency += vol
						with_high_extra.append((perf1-perf2, vol))
						if perf2 == NO_ROUTE_LATENCY:
							vol_no_route += vol
					if perf2 == NO_ROUTE_LATENCY:
						vol_congested += vol
						this_sim_total_volume_congested += vol
						perf2=perf2*100
					all_vol += vol
					if diff > maxd:
						maxd=diff
					if diff < mind:
						mind=diff
				ret.append((perf1-perf2, vol))

			### Store the fraction of users that DONT satisfy a latency objective
			this_sim_fraction_volume_congested = this_sim_total_volume_congested / (this_sim_total_volume + .0000001)
			try:
				this_x,this_cdf_x = get_cdf_xy(list(zip(this_diffs,this_vols)), weighted=True)
				for lat_threshold in interesting_latency_suboptimalities:
					xi = np.argmin(np.abs(this_x-lat_threshold))
					threshold_stats[ri][lat_threshold] = (1-this_sim_fraction_volume_congested) * this_cdf_x[xi] + this_sim_fraction_volume_congested
			except IndexError: # no good data
				for lat_threshold in interesting_latency_suboptimalities:
					threshold_stats[ri][lat_threshold] = this_sim_fraction_volume_congested ## all users are within the latency

			


		x=np.linspace(mind,maxd,num=200)
		if vol_congested > 0:
			x[0] = -1*100*NO_ROUTE_LATENCY

		try:
			avg_latency_difference = np.average([el[0] for el in avg_ret], weights=[el[1] for el in avg_ret])
		except ZeroDivisionError:
			print("Problem doing {} {}".format(k,solution))
			# was NO_ROUTE_LATENCY -- injecting the marker as a STAT put
			# 30000 into downstream averages/plots. NaN renders as a gap.
			avg_latency_difference = float('nan')
		try:
			avg_latency_difference_with_high = np.average(
				[el[0] for el in avg_ret + with_high_extra],
				weights=[el[1] for el in avg_ret + with_high_extra])
		except ZeroDivisionError:
			avg_latency_difference_with_high = float('nan')
		print("Average latency difference {},{}: {}".format(solution, k, avg_latency_difference))
		print("{} pct. volume congested".format(round(100 * vol_congested / (actually_all_vol + .00001), 2)))
		print("{} pct. optimally congested, all volume: {}".format(round(100 * vol_best_case_congested / (actually_all_vol+.00001), 2), actually_all_vol))

		return ret, x, {
			'avg_latency_difference': avg_latency_difference, 
			'avg_latency_difference_with_high': avg_latency_difference_with_high,
			'frac_vol_congested': vol_congested / (all_vol+.0000001), 
			'frac_vol_high_latency': vol_high_latency / (all_vol+.0000001),
			'frac_vol_no_route': vol_no_route / (all_vol+.0000001),
			'frac_vol_bestcase_congested': vol_best_case_congested / (actually_all_vol+.0000001),
		}, threshold_stats


	for solution in soln_types:
		print(solution)
		# Random_iters where this solution actually has results. A strategy that
		# failed for some iters (empty data, e.g. the popp_to_users hot-start
		# crash) is now aggregated over the iters it succeeded on instead of
		# being dropped from the stats entirely.
		valid_iters = [ri for ri in SIM_INDS_TO_PLOT
			if solution in metrics['latencies'][ri]
			and len(metrics['latencies'][ri][solution]) > 0]
		try:
			#### Changes in latency
			diffs = []
			wts = []
			for random_iter in valid_iters:
				this_diffs = []
				this_wts = []
				for i in range(len(metrics['best_latencies'][random_iter])):
					diffs.append(metrics['best_latencies'][random_iter][i] - metrics['latencies'][random_iter][solution][i])
					this_diffs.append(metrics['best_latencies'][random_iter][i] - metrics['latencies'][random_iter][solution][i])
					wts.append(metrics['ug_to_vol'][random_iter][i])
					this_wts.append(metrics['ug_to_vol'][random_iter][i])

				this_x,this_cdf_x = get_cdf_xy(list(zip(this_diffs,this_wts)), weighted=True)
				for lat_threshold in interesting_latency_suboptimalities:
					xi = np.argmin(np.abs(this_x-lat_threshold))
					metrics['stats_latency_thresholds_normal'][solution][random_iter][lat_threshold] = this_cdf_x[xi]
			for lat_threshold in interesting_latency_suboptimalities:
				avg_suboptimality = np.mean(list([metrics['stats_latency_thresholds_normal'][solution][random_iter][lat_threshold] for random_iter in valid_iters]))
				print("({}) {} pct of traffic within {} ms of optimal for normal LP".format(solution, 100*round(1-avg_suboptimality,4), lat_threshold))
			x,cdf_x = get_cdf_xy(list(zip(diffs,wts)), weighted=True)
			ax[LATENCY_I].plot(x,cdf_x,label=solution)
			avg_latency_diff = np.average(diffs, weights=wts)
			print("Average latency compared to optimal : {}".format(avg_latency_diff))
			metrics['stats_best_latencies'][solution] = avg_latency_diff



			# #### Changes in latency (with weighted penalty)
			# diffs = []
			# wts = []
			# for random_iter in SIM_INDS_TO_PLOT:
			# 	this_diffs = []
			# 	this_wts = []
			# 	for i in range(len(metrics['best_latencies'][random_iter])):
			# 		diffs.append(metrics['best_latencies'][random_iter][i] - metrics['latencies_penalty'][random_iter][solution][i])
			# 		this_diffs.append(metrics['best_latencies'][random_iter][i] - metrics['latencies_penalty'][random_iter][solution][i])
			# 		wts.append(metrics['ug_to_vol'][random_iter][i])
			# 		this_wts.append(metrics['ug_to_vol'][random_iter][i])

			# 	this_x,this_cdf_x = get_cdf_xy(list(zip(this_diffs,this_wts)), weighted=True)
			# 	for lat_threshold in interesting_latency_suboptimalities:
			# 		xi = np.argmin(np.abs(this_x-lat_threshold))
			# 		metrics['stats_latency_penalty_thresholds_normal'][solution][random_iter][lat_threshold] = this_cdf_x[xi]
			# for lat_threshold in interesting_latency_suboptimalities:
			# 	avg_suboptimality = np.mean(list([metrics['stats_latency_penalty_thresholds_normal'][solution][random_iter][lat_threshold] for random_iter in SIM_INDS_TO_PLOT]))
			# 	print("({}) {} pct of traffic within {} ms of optimal for latency penalty LP".format(solution, 100*round(1-avg_suboptimality,4), lat_threshold))
			
			# x,cdf_x = get_cdf_xy(list(zip(diffs,wts)), weighted=True)
			# avg_latency_diff = np.average(diffs, weights=wts)
			# print("Average latency compared to optimal with penalty : {}".format(avg_latency_diff))
			# metrics['stats_latencies_penalty'][solution] = avg_latency_diff			


			#### PCT of Volume within a Certainty Latency Threshold
			for random_iter in valid_iters:
				m = metrics['pct_volume_within_latency'][random_iter][solution]
				ax[PCT_VOL_WITHIN_LATENCY_I].plot(m['latencies'], m['volume_fractions'], label="{} -- Sim {}".format(solution, random_iter))
			
			#### Resilience to PoP and PoPP failures
			
			all_differences, x, stats, threshold_stats = get_failure_metric_arr('popp_failures_latency_optimal_specific', solution)
			metrics['stats_' + 'popp_failures_latency_optimal_specific'][solution] = stats
			metrics['stats_latency_thresholds_fail_popp'][solution] = threshold_stats
			x, cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)

			# all_differences, x, stats, threshold_stats = get_failure_metric_arr('popp_failures_latency_penalty_optimal_specific', solution)
			# metrics['stats_' + 'popp_failures_latency_penalty_optimal_specific'][solution] = stats
			# metrics['stats_latency_penalty_thresholds_fail_popp'][solution] = threshold_stats
			# x, cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# x, cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)

			all_differences, x, stats, threshold_stats = get_failure_metric_arr('pop_failures_latency_optimal_specific', solution)
			metrics['stats_' + 'pop_failures_latency_optimal_specific'][solution] = stats
			metrics['stats_latency_thresholds_fail_pop'][solution] = threshold_stats
			x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)

			# all_differences, x, stats, threshold_stats = get_failure_metric_arr('pop_failures_latency_penalty_optimal_specific', solution)
			# metrics['stats_' + 'pop_failures_latency_penalty_optimal_specific'][solution] = stats
			# metrics['stats_latency_penalty_thresholds_fail_pop'][solution] = threshold_stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)



			#### Resilience to flash crowds
			m = metrics['resilience_to_congestion']
			cm = metrics['fraction_congested_volume']
			

			## Want to track where latency and congestion start to impact performance
			latency_limits_over_Y = []
			congestion_limits_over_Y = []

			for Y_val in Y_vals:
				latency_delta_meds = []			
				congestion_meds = []
				avg_congestion_by_X_val_sim = {ri:[] for ri in SIM_INDS_TO_PLOT}
				for X_val in X_vals:
					all_lats, all_congestions = [], []
					for ri in SIM_INDS_TO_PLOT:
						try:
							this_all_congestions = []
							for avg_lat_deltas in m[ri][solution][Y_val][X_val]:
								all_lats.append(avg_lat_deltas)
							for congestion in cm[ri][solution][Y_val][X_val]:
								all_congestions.append(congestion)
								this_all_congestions.append(congestion)

							avg_congestion_by_X_val_sim[ri].append(np.average(this_all_congestions))
						except:
							continue

					# if X_val == single_X_of_interest and Y_val == single_Y_of_interest:
					# 	## Plot CDF for this specific X val and Y val
					# 	x,cdf_x = get_cdf_xy(list(zip(all_lats,all_vols)), weighted=True)
						
					lat_med = np.average(all_lats)
					con_med = np.average(all_congestions)

					latency_delta_meds.append(lat_med)
					congestion_meds.append(con_med)
				
				## High level stats
				for ri in SIM_INDS_TO_PLOT:
					try:
						critical_X = X_vals[np.where(np.array(avg_congestion_by_X_val_sim[ri]) > 0)[0][0]]
					except IndexError:
						## either never or always is congested
						critical_X = X_vals[0]
					_b = (metrics.get('bisect_critical_flash') or {}).get(ri, {}).get(solution)
					metrics['stats_resilience_to_congestion'][solution][ri] = \
						float(_b) if _b is not None else critical_X


				latency_delta_meds = np.array(latency_delta_meds)
				congestion_meds = np.array(congestion_meds)
				if Y_val == single_Y_of_interest:
					## Plot variance over X for this specific Y val
					ax[FLASH_CROWD_LATENCY_VARY_X_I].plot(X_vals, latency_delta_meds, label=solution)
					ax[FLASH_CROWD_CONGESTION_VARY_X_I].plot(X_vals, congestion_meds, label=solution)
				try:
					latency_limits_over_Y.append(X_vals[np.where(latency_delta_meds > 1)[0][0]])
				except IndexError:
					latency_limits_over_Y.append(X_vals[-1])
				try:
					congestion_limits_over_Y.append(X_vals[np.where(congestion_meds > 0)[0][0]])
				except IndexError:
					congestion_limits_over_Y.append(X_vals[-1])
			ax[FLASH_CROWD_LIMITING_FACTOR_LATENCY_I].plot(Y_vals, latency_limits_over_Y, label=solution)
			ax[FLASH_CROWD_LIMITING_FACTOR_CONGESTION_I].plot(Y_vals, congestion_limits_over_Y, label=solution)





			#### Diurnal Resilience
			hours_of_day = np.array(list(range(24)))
			## Want to track what diurnal multiplier causes congestion
			avg_latency_by_sim_Y_val = {ri: [] for ri in SIM_INDS_TO_PLOT}
			avg_congestion_by_sim_Y_val = {ri: [] for ri in SIM_INDS_TO_PLOT}

			avg_latency_by_Y_val_sim = {dm: [] for dm in diurnal_multipliers}
			avg_congestion_by_Y_val_sim = {dm: [] for dm in diurnal_multipliers}
			avg_churn_by_Y_val_sim = {dm: [] for dm in diurnal_multipliers}

			for Y_val in diurnal_multipliers:
				all_lats, all_congestions = {ri:[] for ri in SIM_INDS_TO_PLOT}, {ri:[] for ri in SIM_INDS_TO_PLOT}
				all_churns = {ri:[] for ri in SIM_INDS_TO_PLOT}
				for ri in SIM_INDS_TO_PLOT:
					for X_val in hours_of_day:
						try:
							all_lats[ri].append(metrics['diurnal'][ri][solution]['metrics'][Y_val][X_val][0][0])
							all_churns[ri].append(metrics['diurnal'][ri][solution]['metrics'][Y_val][X_val][0][1])
							all_congestions[ri].append(metrics['diurnal'][ri][solution]['fraction_congested_volume'][Y_val][X_val])
						except (KeyError, TypeError, IndexError):
							# TypeError fires when the diurnal phase populated an
							# empty default (a list) instead of the eval dict --
							# don't let the plotting kill the whole driver
							continue

					lat_med = np.average(all_lats[ri])
					con_med = np.average(all_congestions[ri])
					churn_med = 100 * np.average(all_churns[ri])


					avg_latency_by_sim_Y_val[ri].append(lat_med)
					avg_congestion_by_sim_Y_val[ri].append(con_med)

					avg_latency_by_Y_val_sim[Y_val].append(lat_med)
					avg_congestion_by_Y_val_sim[Y_val].append(con_med)
					avg_churn_by_Y_val_sim[Y_val].append(churn_med)

			ax[DIURNAL_LATENCY_VARY_I].plot(diurnal_multipliers, list([np.mean(avg_latency_by_Y_val_sim[dm]) for dm in diurnal_multipliers]), label=solution)
			ax[DIURNAL_CONGESTION_VARY_I].plot(diurnal_multipliers, list([np.mean(avg_congestion_by_Y_val_sim[dm]) for dm in diurnal_multipliers]), label=solution)
			ax[DIURNAL_ASSIGNMENT_DELTA_VARY_I].plot(diurnal_multipliers, list([np.mean(avg_churn_by_Y_val_sim[dm]) for dm in diurnal_multipliers]), label=solution)


			## High level stats
			for ri in SIM_INDS_TO_PLOT:
				try:
					critical_Y = diurnal_multipliers[np.where(np.array(avg_congestion_by_sim_Y_val[ri]) > 0)[0][0]]
				except IndexError:
					## either never or always is congested
					critical_Y = diurnal_multipliers[0]
				_b = (metrics.get('bisect_critical_diurnal') or {}).get(ri, {}).get(solution)
				metrics['stats_diurnal'][solution][ri] = \
					float(_b) if _b is not None else critical_Y
		except:
			import traceback
			traceback.print_exc()
			continue

	ax[LATENCY_I].set_title('Normal operation: per-UG latency', fontsize=9)
	ax[LATENCY_I].legend(fontsize=8)
	ax[LATENCY_I].grid(True)
	ax[LATENCY_I].set_xlabel("Best - Actual Latency (ms)")
	ax[LATENCY_I].set_ylabel("CDF of Traffic")
	ax[LATENCY_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	_fit_x_to_data(ax[LATENCY_I], -1*NO_ROUTE_LATENCY/2)



	ax[PCT_VOL_WITHIN_LATENCY_I].set_title('Normal operation: volume within a latency target', fontsize=9)
	ax[PCT_VOL_WITHIN_LATENCY_I].legend(fontsize=8)
	ax[PCT_VOL_WITHIN_LATENCY_I].grid(True)
	ax[PCT_VOL_WITHIN_LATENCY_I].set_xlabel("Latency (ms)")
	ax[PCT_VOL_WITHIN_LATENCY_I].set_ylabel("Fraction Ingresses Reachable")
	ax[PCT_VOL_WITHIN_LATENCY_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	### FAILURE PLOTS
	## MUTABLE

	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_title('Single-link failure: latency vs. optimal', fontsize=9)
	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].legend(fontsize=8)
	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].grid(True)
	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_xlabel("Latency Change Under Single-Link Failure (best - actual) (ms)")
	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_ylabel("CDF of Affected Traffic,Links")
	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	_fit_x_to_data(ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I], -1*NO_ROUTE_LATENCY/2)








	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_title('Single-PoP failure: latency vs. optimal', fontsize=9)
	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].legend(fontsize=8)
	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].grid(True)
	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_xlabel("Latency Change Under Single-PoP Failure (best - actual) (ms)")
	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_ylabel("CDF of Affected Traffic,PoPs")
	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	_fit_x_to_data(ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I], -1*NO_ROUTE_LATENCY/2)






	### FLASH CROWDS




	ax[FLASH_CROWD_LATENCY_VARY_X_I].set_xlabel("Increase in Traffic per Metro for Flash Crowd (pct.)")
	ax[FLASH_CROWD_LATENCY_VARY_X_I].set_ylabel("Average Latency Change \nunder Flash Crowd (new - old) (ms)")
	ax[FLASH_CROWD_LATENCY_VARY_X_I].grid(True)
	ax[FLASH_CROWD_LATENCY_VARY_X_I].set_title('Flash crowd: latency vs. traffic increase', fontsize=9)
	ax[FLASH_CROWD_LATENCY_VARY_X_I].legend(fontsize=8)

	ax[FLASH_CROWD_CONGESTION_VARY_X_I].set_xlabel("Increase in Traffic per Metro for Flash Crowd (pct.)")
	ax[FLASH_CROWD_CONGESTION_VARY_X_I].set_ylabel("Average Delta Fraction Congested \n Traffic under Flash Crowd (new - old)")
	ax[FLASH_CROWD_CONGESTION_VARY_X_I].grid(True)
	ax[FLASH_CROWD_CONGESTION_VARY_X_I].set_title('Flash crowd: congestion vs. traffic increase', fontsize=9)
	ax[FLASH_CROWD_CONGESTION_VARY_X_I].legend(fontsize=8)

	ax[FLASH_CROWD_LIMITING_FACTOR_LATENCY_I].set_xlabel("Link Capacity Overprovisioning (pct)")
	ax[FLASH_CROWD_LIMITING_FACTOR_LATENCY_I].set_ylabel("Flash Crowd Latency Resilience")
	ax[FLASH_CROWD_LIMITING_FACTOR_LATENCY_I].grid(True)
	ax[FLASH_CROWD_LIMITING_FACTOR_LATENCY_I].set_title('Flash crowd: latency vs. overprovisioning', fontsize=9)
	ax[FLASH_CROWD_LIMITING_FACTOR_LATENCY_I].legend(fontsize=8)

	ax[FLASH_CROWD_LIMITING_FACTOR_CONGESTION_I].set_xlabel("Link Capacity Overprovisioning (pct)")
	ax[FLASH_CROWD_LIMITING_FACTOR_CONGESTION_I].set_ylabel("Flash Crowd Congestion Resilience")
	ax[FLASH_CROWD_LIMITING_FACTOR_CONGESTION_I].grid(True)
	ax[FLASH_CROWD_LIMITING_FACTOR_CONGESTION_I].set_title('Flash crowd: congestion vs. overprovisioning', fontsize=9)
	ax[FLASH_CROWD_LIMITING_FACTOR_CONGESTION_I].legend(fontsize=8)


	ax[DIURNAL_LATENCY_VARY_I].set_title('Diurnal cycle: latency vs. multiplier', fontsize=9)
	ax[DIURNAL_LATENCY_VARY_I].legend(fontsize=8)
	ax[DIURNAL_LATENCY_VARY_I].grid(True)
	ax[DIURNAL_LATENCY_VARY_I].set_xlabel("Diurnal Multiplier Amount (pct)")
	ax[DIURNAL_LATENCY_VARY_I].set_ylabel("Average Latency Increase (ms)")

	ax[DIURNAL_CONGESTION_VARY_I].set_title('Diurnal cycle: congestion vs. multiplier', fontsize=9)
	ax[DIURNAL_CONGESTION_VARY_I].legend(fontsize=8)
	ax[DIURNAL_CONGESTION_VARY_I].grid(True)
	ax[DIURNAL_CONGESTION_VARY_I].set_xlabel("Diurnal Multiplier Amount (pct)")
	ax[DIURNAL_CONGESTION_VARY_I].set_ylabel("Average Congestion (ms)")

	ax[DIURNAL_ASSIGNMENT_DELTA_VARY_I].set_title('Diurnal cycle: assignment churn vs. multiplier', fontsize=9)
	ax[DIURNAL_ASSIGNMENT_DELTA_VARY_I].legend(fontsize=8)
	ax[DIURNAL_ASSIGNMENT_DELTA_VARY_I].grid(True)
	ax[DIURNAL_ASSIGNMENT_DELTA_VARY_I].set_xlabel("Diurnal Multiplier Amount (pct)")
	ax[DIURNAL_ASSIGNMENT_DELTA_VARY_I].set_ylabel("Average Daily Traffic Churn (pct)")

	save_fig_fn = kwargs.get('save_fig_fn', "popp_latency_failure_comparison_{}.pdf".format(dpsize))

	f.tight_layout()
	save_fig(save_fig_fn)

	return metrics
