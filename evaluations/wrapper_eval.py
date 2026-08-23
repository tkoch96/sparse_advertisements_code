"""Post-training evaluation phase implementations.

After `compare_different_solutions` produces per-strategy advertisement
matrices, the eval phases simulate each strategy under various scenarios
and record the result in the `metrics` dict that `evaluate_all_metrics`
serializes to disk.

Phases implemented here:
  - `assess_failure_resilience`     popp-failure and pop-failure latency,
                                     congestion, no-route fractions
  - flash-crowd assessment           per-traffic-multiplier latency
  - diurnal scenarios                latency under time-of-day load shifts
  - capacity-modification scenarios  modeling-assumption sweeps (mostly
                                     unused)

Result shape: each phase writes into
`metrics[phase_name][random_iter][solution_type]`. See the relevant
function for the per-phase nested dict shape.

The phases use `sas.solve_lp_with_failure_catch(adv, ...)` which is the
public LP-solving entry point on Sparse_Advertisement_Eval. That call
goes through `solve_lp_assignment.solve_generic_lp_with_failure_catch`.
"""
import tqdm, numpy as np, os, copy
from helpers.constants import *
gamma = 4
capacity = True
N_TO_SIM = 1
#### NOTE -- need to make sure lambduh decreases with the problem size
#### or else the latency gains won't be significant enough to get a signal through
lambduh = 0
global_soln_types = ['sparse', 'anyopt', 'painter', 'anycast', 'one_per_pop', 'one_per_peering']

def _run_tag_suffix():
	# SCULPTOR_RUN_TAG namespaces the metrics pickle so parallel A/B runs at
	# the same dpsize don't cross-contaminate (e.g. one run loading another
	# run's compare_rets and skipping its own training loop). Empty by default
	# so existing single-run workflows are unchanged.
	tag = os.environ.get('SCULPTOR_RUN_TAG', '')
	return '_' + tag if tag else ''

global_performance_metrics_fn =  lambda dps : os.path.join(CACHE_DIR, 'popp_failure_latency_comparison_{}{}.pkl'.format(dps, _run_tag_suffix()))

### Default metrics for performance evaluations
default_metrics = {
	# bisected critical intensities (per sim: {solution: value}). MUST be in
	# this schema: the loader in eval_all_solution_types deletes any pickle
	# key not listed here, which silently ate the 2026-08-23 bisect results.
	'bisect_critical_flash': {i:{} for i in range(N_TO_SIM)},
	'bisect_critical_diurnal': {i:{} for i in range(N_TO_SIM)},
	'save_run_dir': {i:None for i in range(N_TO_SIM)},
	'compare_rets': {i:None for i in range(N_TO_SIM)},
	'adv': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'adv_representation': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'deployment': {i:None for i in range(N_TO_SIM)},
	'ug_to_vol': {i:None for i in range(N_TO_SIM)},
	'settings': {i:None for i in range(N_TO_SIM)},
	'pct_volume_within_latency': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'popp_failures_latency_optimal': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'popp_failures_latency_before': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'popp_failures_latency_optimal_specific': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'popp_failures_latency_penalty_optimal_specific': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'popp_failures_congestion': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'popp_failures_congestion_penalty': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_latency_optimal': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_latency_optimal_specific': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_latency_penalty_optimal_specific': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_latency_before': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_congestion': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_congestion_penalty': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},

	'popp_failures_high_cap_latency_optimal': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'popp_failures_high_cap_latency_optimal_specific': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_high_cap_latency_optimal': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_high_cap_latency_optimal_specific': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},


	'popp_failures_sticky_latency_optimal': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'popp_failures_sticky_latency_before': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'popp_failures_sticky_latency_optimal_specific': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'popp_failures_sticky_congestion': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_sticky_latency_optimal': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_sticky_latency_optimal_specific': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_sticky_latency_before': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_sticky_congestion': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},

	'latencies': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'latencies_penalty': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'best_latencies': {i:None for i in range(N_TO_SIM)},
	'resilience_to_congestion': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'prefix_withdrawals': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'fraction_congested_volume': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'diurnal': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
}

def check_calced_everything(metrics, random_iter, k_of_interest):
	havent_calced_everything = False
	soln_types = list(metrics['latencies'][random_iter])
	for solution in soln_types:
		if metrics[k_of_interest][random_iter][solution]  == \
			default_metrics[k_of_interest][0][solution]:
			havent_calced_everything = True
			break
	return havent_calced_everything

def assess_failure_resilience_one_per_peering(sas, adv, which='popps'):
	ret = {redirection_mode: {'congestion_delta': [], 'latency_delta_optimal': [], 'latency_delta_before': [], 'latency_delta_specific': []}
		for redirection_mode in ['mutable']}
	if which == 'popps':
		iterover = sas.popps
	else: # pops
		iterover = sas.pops
	call_args = []
	dep = sas.output_deployment()
	_, ug_catchments = sas.calculate_user_choice(adv)
	iteri_to_ugs = {}
	for ugi in ug_catchments:
		ug = sas.ugs[ugi]
		for iteri,v in ug_catchments[ugi]:
			iteri = sas.popps[iteri]
			if which == 'popps':
				try:
					iteri_to_ugs[iteri].append((ug,v))
				except KeyError:
					iteri_to_ugs[iteri] = [(ug,v)]
			else:
				try:
					iteri_to_ugs[iteri[0]].append((ug,v))
				except KeyError:
					iteri_to_ugs[iteri[0]] = [(ug,v)]

	for i,iteri in enumerate(iterover):
		adv_cpy = np.copy(adv)
		if which == 'popps':
			adv_cpy[sas.popp_to_ind[iteri],:] = 0
		else:
			these_popps = np.array([sas.popp_to_ind[popp] for popp in sas.popps if popp[0] == iteri])
			adv_cpy = np.copy(adv)
			adv_cpy[these_popps,:] = 0
		## q: what is latency experienced for these ugs compared to optimal?
		call_args.append((adv_cpy, dep, i%20==0))

	base_soln = sas.solve_lp_with_failure_catch(adv)
	base_user_latencies = base_soln['lats_by_ug']
	# base_ug_routes = base_soln['routed_through_ingress']

	## find base UG -> prefix mappings. these mappings might be ambiguous, so we choose the best-case scenario
	# first get ug -> popps -> prefixes
	# ug_to_popp_to_prefix = {}
	# for prefix_i in base_ug_routes:
	# 	for ug,poppi in base_ug_routes[prefix_i].items():
	# 		try:
	# 			ug_to_popp_to_prefix[ug]
	# 		except KeyError:
	# 			ug_to_popp_to_prefix[ug] = {}
	# 		popp = sas.popps[poppi]
	# 		try:
	# 			ug_to_popp_to_prefix[ug][popp].append(prefix_i)
	# 		except KeyError:
	# 			ug_to_popp_to_prefix[ug][popp] = [prefix_i]
	# Then, given LP solution, calculate the UG to prefix assignments
	# base_user_prefix_assignments = {ug: [] for ug in sas.ugs}
	# for ugi, pathvols in base_soln['paths_by_ug'].items():
	# 	ug = sas.ugs[ugi]
	# 	for poppi, vol in pathvols:
	# 		popp = sas.popps[poppi]
	# 		possible_prefixes = ug_to_popp_to_prefix[ug][popp]

	# 		base_user_prefix_assignments[ug].append((tuple(possible_prefixes), vol))


	lp_rets = sas.solve_lp_with_failure_catch_mp(call_args, cache_res=False)
	for i,iteri in enumerate(iterover):	

		## q: what is latency experienced for these ugs compared to optimal?
		this_soln = lp_rets[i]
		user_latencies = this_soln['lats_by_ug']
		## best user latencies is not necessarily just lowest latency
		## need to factor in capacity
		best_soln = lp_rets[i]
		best_user_latencies = best_soln['lats_by_ug']

		ret['mutable']['congestion_delta'].append(this_soln['fraction_congested_volume'] - best_soln['fraction_congested_volume'])

		these_ugs = {ug: fracv for ug,fracv in iteri_to_ugs.get(iteri,[])}
		# routed_through_ingress = this_soln['routed_through_ingress']
		for ug in sas.ugs:
			#### Mutable decisions
			old_perf = base_user_latencies[sas.ug_to_ind[ug]]
			new_perf = user_latencies[sas.ug_to_ind[ug]]
			best_perf = best_user_latencies[sas.ug_to_ind[ug]]
			ret['mutable']['latency_delta_optimal'].append((best_perf - new_perf, 
				sas.ug_to_vol[ug], ug, iteri, best_perf, new_perf))
			ret['mutable']['latency_delta_before'].append((old_perf - new_perf, 
				sas.ug_to_vol[ug], ug, iteri, best_perf, new_perf))
			
			try:
				fracv = these_ugs[ug]
				ret['mutable']['latency_delta_specific'].append((best_perf - new_perf,
					fracv*sas.ug_to_vol[ug], ug, iteri, best_perf, new_perf,
					this_soln['paths_by_ug'][sas.ug_to_ind[ug]] ))
			except KeyError:
				pass


	return ret

def precompute_one_per_peering_failure_lps(sas, which='popps', **kwargs):
	"""LP-solve every failure scenario under a one-per-peering advertisement.

	The result depends ONLY on (which, failed_popp_or_pop), not on the
	strategy being evaluated. So we compute these reference LPs once per
	failure-eval session and share them across all 6 strategies, instead
	of recomputing inside every assess_failure_resilience call.

	At actual-32 this saves ~50% of failure-eval LP work: each of the ~811
	failure scenarios was being solved 6 extra times (once per strategy)
	for the OPP reference, in addition to the strategy-specific LP. With
	this dedup, OPP-ref LPs run once and the per-strategy work is half.

	Returns dict {iteri: lp_result} where lp_result is the light_result
	form (just lats_by_ug + fraction_congested_volume + paths_by_ug;
	paths_by_ug is unused for OPP-refs but kept by the worker default).
	"""
	if which == 'popps':
		iterover = sas.popps
	else: # pops
		iterover = sas.pops
	dep = sas.output_deployment()
	call_args = []
	for iteri in iterover:
		one_per_peer_adv = np.eye(sas.n_popps)
		if which == 'popps':
			one_per_peer_adv[sas.popp_to_ind[iteri],:] = 0
		else:
			for popp in sas.popps:
				if popp[0] == iteri:
					one_per_peer_adv[sas.popp_to_ind[popp],:] = 0
		call_args.append((one_per_peer_adv, dep, False))
	# Use the same light_result + cache_res=False discipline as
	# assess_failure_resilience.
	lp_rets = sas.solve_lp_with_failure_catch_mp(
		call_args, cache_res=False, light_result=True, **kwargs)
	return {iteri: lp_rets[i] for i, iteri in enumerate(iterover)}


def assess_failure_resilience(sas, adv, which='popps', opp_ref_results=None, **kwargs):
	ret = {redirection_mode: {'congestion_delta': [], 'latency_delta_optimal': [], 'latency_delta_before': [], 'latency_delta_specific': []}
		for redirection_mode in ['sticky', 'mutable']}
	if which == 'popps':
		iterover = sas.popps
	else: # pops
		iterover = sas.pops
	call_args = []
	dep = sas.output_deployment()
	_, ug_catchments = sas.calculate_user_choice(adv)
	iteri_to_ugs = {}

	use_penalty = kwargs.get('penalty', False)

	# If the caller didn't precompute one-per-peering reference LPs (the
	# fast path used by the failure-eval phase loop), compute them inline
	# so this function remains correct for one-off callers. Strategy-loop
	# callers pass opp_ref_results=precompute_one_per_peering_failure_lps(...)
	# to amortize the work across all 6 strategies.
	have_precomputed_opp = opp_ref_results is not None
	if not have_precomputed_opp and not use_penalty:
		opp_ref_results = precompute_one_per_peering_failure_lps(
			sas, which=which, **{k: v for k, v in kwargs.items() if k != 'penalty'})

	for ugi in ug_catchments:
		ug = sas.ugs[ugi]
		for iteri,v in ug_catchments[ugi]:
			iteri = sas.popps[iteri]
			if which == 'popps':
				try:
					iteri_to_ugs[iteri].append((ug,v))
				except KeyError:
					iteri_to_ugs[iteri] = [(ug,v)]
			else:
				try:
					iteri_to_ugs[iteri[0]].append((ug,v))
				except KeyError:
					iteri_to_ugs[iteri[0]] = [(ug,v)]

	# Per-scenario strategy-specific call_args. With opp_ref_results, we
	# emit ONE call per scenario (the failed-strategy adv). The legacy
	# path (use_penalty + no precompute) emits two interleaved calls.
	for i,iteri in enumerate(iterover):
		adv_cpy = np.copy(adv)
		if which == 'popps':
			adv_cpy[sas.popp_to_ind[iteri],:] = 0
		else:
			these_popps = np.array([sas.popp_to_ind[popp] for popp in sas.popps if popp[0] == iteri])
			adv_cpy = np.copy(adv)
			adv_cpy[these_popps,:] = 0
		## q: what is latency experienced for these ugs compared to optimal?
		if use_penalty:
			# Legacy two-call-per-scenario penalty path. Not used by the
			# failure-eval phase loop in eval_all_solution_types.py.
			opt_adv = np.eye(sas.n_popps)
			if which == 'popps':
				opt_adv[sas.popp_to_ind[iteri],:] = 0
			else:
				these_popps = np.array([sas.popp_to_ind[popp] for popp in sas.popps if popp[0] == iteri])
				opt_adv[these_popps,:] = 0
			call_args.append((adv_cpy, opt_adv, dep, False))
			one_per_peer_adv = np.eye(sas.n_popps)
			if which == 'popps':
				one_per_peer_adv[sas.popp_to_ind[iteri],:] = 0
			else:
				for popp in sas.popps:
					if popp[0] == iteri:
						one_per_peer_adv[sas.popp_to_ind[popp],:] = 0
			call_args.append((one_per_peer_adv, dep, False))
		else:
			# Fast path: only failed-strategy adv per scenario. The OPP-ref
			# result comes from opp_ref_results (precomputed once, shared
			# across all strategies in the phase loop).
			call_args.append((adv_cpy, dep, False))

	base_soln = sas.solve_lp_with_failure_catch(adv)
	base_user_latencies = base_soln['lats_by_ug']

	# See assess_failure_resilience header / clear_lp_caches docstring for
	# why cache_res=False and light_result=True here.
	lp_rets = sas.solve_lp_with_failure_catch_mp(call_args, cache_res=False, light_result=True, **kwargs)

	for i,iteri in enumerate(iterover):

		## q: what is latency experienced for these ugs compared to optimal?
		if use_penalty:
			this_soln = lp_rets[i*2]
			best_soln = lp_rets[i*2+1]
		else:
			this_soln = lp_rets[i]
			best_soln = opp_ref_results[iteri]
		user_latencies = this_soln['lats_by_ug']
		## best user latencies is not necessarily just lowest latency
		## need to factor in capacity
		best_user_latencies = best_soln['lats_by_ug']

		ret['mutable']['congestion_delta'].append(this_soln['fraction_congested_volume'] - best_soln['fraction_congested_volume'])
		ret['sticky']['congestion_delta'].append(this_soln['fraction_congested_volume'] - best_soln['fraction_congested_volume']) # tmp

		these_ugs = {ug: fracv for ug,fracv in iteri_to_ugs.get(iteri,[])}
		# routed_through_ingress = this_soln['routed_through_ingress']
		for ug in sas.ugs:
			#### Mutable decisions
			old_perf = base_user_latencies[sas.ug_to_ind[ug]]
			new_perf = user_latencies[sas.ug_to_ind[ug]]
			best_perf = best_user_latencies[sas.ug_to_ind[ug]]
			### Too much data to store
			# ret['mutable']['latency_delta_optimal'].append((best_perf - new_perf, 
			# 	sas.ug_to_vol[ug], ug, iteri, best_perf, new_perf))
			# ret['mutable']['latency_delta_before'].append((old_perf - new_perf, 
			# 	sas.ug_to_vol[ug], ug, iteri, best_perf, new_perf))
			
			try:
				fracv = these_ugs[ug]
				ret['mutable']['latency_delta_specific'].append((best_perf - new_perf,
					fracv*sas.ug_to_vol[ug], ug, iteri, best_perf, new_perf,
					this_soln['paths_by_ug'][sas.ug_to_ind[ug]] ))
			except KeyError:
				pass


			# #### Sticky (DNS) decisions
			
			# old_perf = base_user_latencies[sas.ug_to_ind[ug]]
			# old_prefixes_volumes = base_user_prefix_assignments[ug]
			# this_ug_new_perfs = []
			# total_volume = 0
			# for old_prefixes, volume in old_prefixes_volumes:
			# 	possible_new_latencies = []
			# 	for prefix in old_prefixes:
			# 		poppi = routed_through_ingress[prefix].get(ug)
			# 		if poppi is not None:
			# 			possible_new_latencies.append(sas.ug_perfs[ug][sas.popps[poppi]])
			# 		else:
			# 			possible_new_latencies.append(NO_ROUTE_LATENCY)
			# 	new_latency = np.min(possible_new_latencies) # optimistically assume the lucky prefix was assigned
			# 	this_ug_new_perfs.append((new_latency, volume))
			# 	total_volume += volume

			# new_perf = np.sum(l*v for l,v in this_ug_new_perfs) / total_volume
			# best_perf = best_user_latencies[sas.ug_to_ind[ug]]
			# ret['sticky']['latency_delta_optimal'].append((best_perf - new_perf, 
			# 	sas.ug_to_vol[ug], ug, iteri, best_perf, new_perf))
			# ret['sticky']['latency_delta_before'].append((old_perf - new_perf, 
			# 	sas.ug_to_vol[ug], ug, iteri, best_perf, new_perf))
			# try:
			# 	fracv = these_ugs[ug]
			# 	ret['sticky']['latency_delta_specific'].append((best_perf - new_perf,
			# 		fracv*sas.ug_to_vol[ug], ug, iteri, best_perf, new_perf))
			# except KeyError:
			# 	pass


	return ret

def assess_failure_resilience_actual_deployment(sas, adv_rep, solution, which='popps'):
	ret = {'congestion_delta': [], 'latency_delta_specific': []}

	plain_deployment = sas.output_deployment()
	sas.load_solution_realworld_measure_wrapper(solution, match_file_patterns=['tmp_ripe_results*.pkl', 'painter*ripe_results.pkl'])

	adv = sas.adv_rep_to_adv(adv_rep)
	dep = sas.output_deployment()
	if which == 'popps':
		iterover = sas.popps
	else: # pops
		iterover = sas.pops
	_, ug_catchments = sas.calculate_user_choice(adv)
	iteri_to_ugs = {}
	for ugi in ug_catchments:
		ug = sas.ugs[ugi]
		for iteri,v in ug_catchments[ugi]:
			iteri = sas.popps[iteri]
			if which == 'popps':
				try:
					iteri_to_ugs[iteri].append((ug,v))
				except KeyError:
					iteri_to_ugs[iteri] = [(ug,v)]
			else:
				try:
					iteri_to_ugs[iteri[0]].append((ug,v))
				except KeyError:
					iteri_to_ugs[iteri[0]] = [(ug,v)]

	base_soln = sas.solve_lp_with_failure_catch(adv)
	base_user_latencies = base_soln['lats_by_ug']

	## Solve for the optimal allocations
	call_args = []
	for i,iteri in enumerate(iterover):
		## best user latencies is not necessarily just lowest latency
		## need to factor in capacity
		one_per_peer_adv = np.eye(sas.n_popps)
		if which == 'popps':
			one_per_peer_adv[sas.popp_to_ind[iteri],:] = 0
		else:
			for popp in sas.popps:
				if popp[0] == iteri:
					one_per_peer_adv[sas.popp_to_ind[popp],:] = 0
		call_args.append((one_per_peer_adv, dep, False))
	# Same reasoning as the cache_res=False above in assess_failure_resilience:
	# each failure scenario yields a unique cache key; caching wastes ~MB per
	# call with no hit-rate benefit.
	lp_rets = sas.solve_lp_with_failure_catch_mp(call_args, cache_res=False)

	best_solutions = {}
	for i,iteri in enumerate(iterover):	
		## best user latencies is not necessarily just lowest latency
		## need to factor in capacity
		best_solutions[iteri] = lp_rets[i]
	
	## Measure everything we need to measure

	def get_needs_measuring(sub_iterover):
		all_cols_to_measure = []
		for i,iteri in enumerate(sub_iterover): ## only measure up to a certain point
			adv_cpy = np.copy(adv)
			if which == 'popps':
				adv_cpy[sas.popp_to_ind[iteri],:] = 0
			else:
				these_popps = np.array([sas.popp_to_ind[popp] for popp in sas.popps if popp[0] == iteri])
				adv_cpy = np.copy(adv)
				adv_cpy[these_popps,:] = 0
			## get the subset of columns that need measuring
			all_cols_to_measure = all_cols_to_measure + sas.check_need_measure_actual_deployment(adv_cpy)
		## Remove dups
		from core.realworld_measure_wrapper import popps_to_hash
		filtered_all_cols_to_measure = []
		already_in = {}
		for _adv in all_cols_to_measure:
			hash_adv = popps_to_hash(list([sas.popps[poppi] for poppi in np.where(_adv)[0]]))
			try:
				already_in[hash_adv]
				continue
			except KeyError:
				already_in[hash_adv] = None
			filtered_all_cols_to_measure.append(_adv)

		return filtered_all_cols_to_measure

	if which == 'popps':
		popp_to_vol = {}
		for popp, ugvols in iteri_to_ugs.items():
			for ug,volfrac in ugvols:
				try:
					popp_to_vol[popp] += volfrac * sas.ug_to_vol[ug]
				except KeyError:
					popp_to_vol[popp] = volfrac * sas.ug_to_vol[ug]
		sorted_popps = sorted(iterover, key = lambda popp : -1 * popp_to_vol.get(popp,0))
		sorted_popp_vols = np.array([popp_to_vol.get(popp,0) for popp in sorted_popps])
		csum_sorted_popp_vols = np.cumsum(sorted_popp_vols)

		## Maybe automate this -- i.e., keep increasing the cutoff fraction until some critical threshold of num advertisements
		for cutoff_frac in [.9, .95, .97, .99]:
			cutoff_index = np.where(csum_sorted_popp_vols >= cutoff_frac*csum_sorted_popp_vols[-1])[0][0]

			updated_iterover = iterover[0:cutoff_index]
			print("Cutoff frac: {} Updated {} popps to {} popps".format(cutoff_frac, len(iterover), len(updated_iterover)))
			n_measure = len(get_needs_measuring(updated_iterover))
			print("{} columns to measure".format(n_measure))
			if n_measure >= 0:
				break
		iterover = updated_iterover

	## populate call args
	call_args = []
	for i,iteri in enumerate(iterover): ## only measure up to a certain point
		adv_cpy = np.copy(adv)
		if which == 'popps':
			adv_cpy[sas.popp_to_ind[iteri],:] = 0
		else:
			these_popps = np.array([sas.popp_to_ind[popp] for popp in sas.popps if popp[0] == iteri])
			adv_cpy = np.copy(adv)
			adv_cpy[these_popps,:] = 0
		call_args.append((adv_cpy,dep,False))

	all_cols_to_measure = get_needs_measuring(iterover)
	adv_round_i = 0
	while len(all_cols_to_measure) > 0:
		## Measure everything in the real Internet
		print("{} advertisement columns left to measure in round {}".format(len(all_cols_to_measure), adv_round_i))
		super_adv = np.concatenate(all_cols_to_measure, axis=1)
		n_adv_batches = int(np.ceil(super_adv.shape[1] / N_PREFIXES))
		for i in range(n_adv_batches):
			sas.calculate_ground_truth_ingress(super_adv[:,i*N_PREFIXES:(i+1)*N_PREFIXES])
			break
		all_cols_to_measure = get_needs_measuring(iterover)
		adv_round_i += 1


	## Now that we've measured everything, this should return without needing to measure
	# lp_rets = sas.solve_lp_with_failure_catch_mp(call_args, cache_res=True) ## maybe get working if I care
	lp_rets = []
	for adv,_,_ in tqdm.tqdm(call_args,desc="Solving linear programs..."):
		lp_rets.append(sas.solve_lp_with_failure_catch(adv, verb=True))

	for i,iteri in enumerate(iterover):	

		## q: what is latency experienced for these ugs compared to optimal?
		this_soln = lp_rets[i]
		user_latencies = this_soln['lats_by_ug']

		## best user latencies is not necessarily just lowest latency
		## need to factor in capacity
		best_soln = best_solutions[iteri]
		best_user_latencies = best_soln['lats_by_ug']

		ret['congestion_delta'].append(this_soln['fraction_congested_volume'] - best_soln['fraction_congested_volume'])

		these_ugs = {ug: fracv for ug,fracv in iteri_to_ugs.get(iteri,[])}
		# routed_through_ingress = this_soln['routed_through_ingress']
		for ug in sas.ugs:
			#### Mutable decisions
			old_perf = base_user_latencies[sas.ug_to_ind[ug]]
			new_perf = user_latencies[sas.ug_to_ind[ug]]
			best_perf = best_user_latencies[sas.ug_to_ind[ug]]

			try:
				fracv = these_ugs[ug]
				ret['latency_delta_specific'].append((best_perf - new_perf,
					fracv*sas.ug_to_vol[ug], ug, iteri, best_perf, new_perf,
					this_soln['paths_by_ug'][sas.ug_to_ind[ug]] ))
			except KeyError:
				pass

	sas.deload_realworld_measure_wrapper(plain_deployment)

	return ret

def get_inflated_metro_deployments(sas, X_vals, Y_vals):
	""" Gets deployments with modified user volumes and/or link capacities."""
	#### X_vals: how much to inflate each metro by
	#### Y_vals: overprovisioning rate for the links
	#### we want to see how our ability to withstand flash crowds varies as we increase the global capacity of the deployment
	from core.deployment_setup import get_link_capacities
	ret = {Y_val: {X_val: {} for X_val in X_vals} for Y_val in Y_vals}
	deployment = sas.output_deployment(copykeys=None)
	vol_by_metro = {}
	for metro,asn in sas.ugs:
		try:
			vol_by_metro[metro] += sas.ug_to_vol[(metro,asn)]
		except KeyError:
			vol_by_metro[metro] = sas.ug_to_vol[(metro,asn)]
	for Y in tqdm.tqdm(Y_vals, desc='populating multiprocessing call args...'):
		for X in X_vals:
			for metro in vol_by_metro:
				# minimally copy the deployment to not cause memory errors
				quick_deployment = sas.output_deployment(copykeys=['ug_to_vol','whole_deployment_ug_to_vol','link_capacities'])
				## modify global link capacities
				new_link_capacities = get_link_capacities(quick_deployment, scale_factor=Y, verb=False)
				quick_deployment['link_capacities'] = new_link_capacities

				## modify volume in a specific metro
				for ug,v in quick_deployment['ug_to_vol'].items():
					if ug[0] == metro:
						quick_deployment['ug_to_vol'][ug] = v * (1 + X/100)
						quick_deployment['whole_deployment_ug_to_vol'][ug] = v * (1 + X/100)
				ret[Y][X][metro] = quick_deployment
	return ret

def get_inflated_metro_deployments_actual_deployment(sas, X_vals, Y_vals):
	""" Gets deployments with modified user volumes and/or link capacities."""
	#### X_vals: how much to inflate each metro by
	#### we want to see how our ability to withstand flash crowds varies as we increase the global capacity of the deployment
	ret = {Y_val: {X_val: {} for X_val in X_vals} for Y_val in Y_vals}
	vol_by_metro = {}
	for metro,asn in sas.ugs:
		try:
			vol_by_metro[metro] += sas.ug_to_vol[(metro,asn)]
		except KeyError:
			vol_by_metro[metro] = sas.ug_to_vol[(metro,asn)]
	for Y in tqdm.tqdm(Y_vals, desc='populating multiprocessing call args...'):
		for X in X_vals:
			for metro in vol_by_metro:
				# minimally copy the deployment to not cause memory errors
				quick_deployment = sas.output_deployment(copykeys=['ug_to_vol','whole_deployment_ug_to_vol','link_capacities'])

				## modify volume in a specific metro
				for ug,v in quick_deployment['ug_to_vol'].items():
					if ug[0] == metro:
						quick_deployment['ug_to_vol'][ug] = v * (1 + X/100)
						quick_deployment['whole_deployment_ug_to_vol'][ug] = v * (1 + X/100)
				ret[Y][X][metro] = quick_deployment
	return ret


def _volscen_extract(sas, flat_deployments, base_dep):
	"""SCULPTOR_EVAL_VOLSCEN conformance check + extraction. Returns
	[(vol_vec|None, cap_arr|None), ...] iff every scenario deployment
	differs from the CURRENT sas deployment only in ug volumes and/or
	link capacities (the shape produced by get_diurnal_deployments /
	get_inflated_metro_deployments / get_inflated_total_deployments via
	output_deployment(copykeys=...)); None => caller uses legacy path.
	Identity comparison on the non-copied fields is the guard: any
	structural difference fails closed."""
	VOL_KEYS = {'ug_to_vol', 'whole_deployment_ug_to_vol', 'link_capacities'}
	out = []
	for d in flat_deployments:
		for k, v in d.items():
			if k in VOL_KEYS or k == 'port':
				continue
			if k not in base_dep or base_dep[k] is not v:
				return None
		vol_vec = None
		if d.get('whole_deployment_ug_to_vol') is not base_dep.get('whole_deployment_ug_to_vol'):
			m = d['whole_deployment_ug_to_vol']
			vol_vec = np.array([m[ug] for ug in sas.whole_deployment_ugs], dtype=np.float64)
		cap_arr = None
		if d.get('link_capacities') is not base_dep.get('link_capacities'):
			lc = d['link_capacities']
			cap_arr = np.array([lc[popp] for popp in sas.popps], dtype=np.float64)
		out.append((vol_vec, cap_arr))
	return out

def assess_resilience_to_flash_crowds_mp(sas, adv, solution, X_vals, Y_vals, inflated_deployments):
	## X vals is flash crowd volume surge
	## Y vals is link capacity multiplier

	## assume each metro's volume increases by X times on average
	## see if there's a solution
	## if there's a solution, do it and note the latency penalty compared to optimal

	# return cdf of latency penalties, possibly as a function of X
	metrics = {Y:{X:[] for X in X_vals} for Y in Y_vals}
	prefix_withdrawals = {Y:{X:[] for X in X_vals} for Y in Y_vals}
	fraction_congested_volumes = {Y:{X:[] for X in X_vals} for Y in Y_vals}

	adv = threshold_a(adv)

	base_soln = sas.solve_lp_with_failure_catch(adv)

	flat_deps = []
	for Y_val in Y_vals:
		for X_val in X_vals:
			for metro in inflated_deployments[Y_val][X_val]:
				flat_deps.append(inflated_deployments[Y_val][X_val][metro])

	scen = None
	if os.environ.get('SCULPTOR_EVAL_VOLSCEN', '0') == '1':
		scen = _volscen_extract(sas, flat_deps, sas.output_deployment(copykeys=None))
	if scen is not None:
		_obj = getattr(getattr(sas, 'generic_objective', None), 'obj', 'avg_latency')
		all_rets = sas.solve_lp_volscen_mp(threshold_a(adv), _obj, scen)
	else:
		call_args = [(adv, d, True) for d in flat_deps]
		### Call all the solutions with multiprocessing
		all_rets = sas.solve_lp_with_failure_catch_mp(call_args)
	i=0
	print("Done, parsing return values from workers")
	if os.environ.get('SCULPTOR_VOLDEBUG') == '1' and len(all_rets) > 258:
		_l = np.asarray(all_rets[258]['lats_by_ug'], dtype=float)
		print('[voldebug-agg] i=258 mean={:.1f} max={:.1f} n_exact30000={} n_ge15000={}'.format(
			float(np.mean(_l)), float(np.max(_l)),
			int(np.sum(_l == 30000.0)), int(np.sum(_l >= 15000))), flush=True)
	for Y in Y_vals:
		previous_hour_solution = None
		for X in X_vals:
			for metro in inflated_deployments[Y][X]:
				prefix_withdrawals[Y][X].append([]) ## unused
				
				soln_adv = all_rets[i]
				i += 1

				latency_deltas = []
				vols = []
				for old_lat, new_lat, vol in zip(base_soln['lats_by_ug'], soln_adv['lats_by_ug'], sas.ug_vols):
					if old_lat == NO_ROUTE_LATENCY or new_lat == NO_ROUTE_LATENCY: continue
					latency_deltas.append(new_lat - old_lat)
					vols.append(vol)
				### structure this as ug,poppi -> val
				## The persistent-LP path returns raw_solution as a dict
				## {(ug,poppi): vol_amt}; the older non-persistent path returns
				## it as a numpy array aligned with available_paths. Handle both.
				raw = soln_adv['raw_solution']
				if isinstance(raw, dict):
					path_to_path_val = dict(raw)
				else:
					path_to_path_val = {}
					for path_val, (ug, poppi) in zip(raw, soln_adv['available_paths']):
						path_to_path_val[ug, poppi] = path_val
				if previous_hour_solution is not None:
					## sqrt mean squared difference in traffic assignments
					total_diff = 0
					all_paths = set(list(path_to_path_val)).union(set(list(previous_hour_solution)))
					for path in all_paths:
						total_diff += (previous_hour_solution.get(path,0) - path_to_path_val.get(path,0))**2
					assignment_delta = np.sqrt(total_diff / len(all_paths))
				else:
					assignment_delta = 0
				previous_hour_solution = path_to_path_val

				fraction_congested_volumes[Y][X].append(soln_adv['fraction_congested_volume'])
				if len(latency_deltas) > 0:
					metrics[Y][X].append((np.average(latency_deltas, weights=vols), assignment_delta))
				else:
					metrics[Y][X].append((NO_ROUTE_LATENCY, assignment_delta))
	return {
		'metrics': metrics,
		'prefix_withdrawals':prefix_withdrawals, 
		'fraction_congested_volume': fraction_congested_volumes,
	}


def assess_resilience_to_flash_crowds_actual_deployment(sas, adv_rep, solution, X_vals, Y_vals, which):
	## X vals is flash crowd volume surge
	## Y vals is link capacity multiplier

	## !!!!!!for painter/TIPSY!!!!!!
	## assume each metro's volume increases by X times on average
	## see if there's a solution
	## if there's a solution, do it and note the latency penalty compared to optimal

	plain_deployment = sas.output_deployment()
	sas.load_solution_realworld_measure_wrapper(solution, match_file_patterns=['tmp_ripe_results*.pkl', 'painter*ripe_results.pkl'])

	## Need to get these separately for each solution type in actual_deployments
	if which == 'diurnal':
		inflated_deployments = get_diurnal_deployments(sas, Y_vals)
	elif which == 'flash_crowd':
		inflated_deployments = get_inflated_metro_deployments_actual_deployment(sas, X_vals, Y_vals)

	adv = sas.adv_rep_to_adv(adv_rep)

	# return cdf of latency penalties, possibly as a function of X
	metrics = {Y:{X:[] for X in X_vals} for Y in Y_vals}
	prefix_withdrawals = {Y:{X:[] for X in X_vals} for Y in Y_vals}
	fraction_congested_volumes = {Y:{X:[] for X in X_vals} for Y in Y_vals}

	adv = threshold_a(adv)

	base_soln = sas.solve_lp_with_failure_catch(adv)

	call_args = []
	for Y_val in Y_vals:
		for X_val in X_vals:
			for metro in sorted(inflated_deployments[Y_val][X_val]):
				d = inflated_deployments[Y_val][X_val][metro]
				## always clear the deployment cache (True on third arg)
				call_args.append((adv, d, True))

	### Maybe multiprocess this one day if I care enough
	dep = sas.output_deployment()
	all_rets = []
	for _adv,d,_ in tqdm.tqdm(call_args,desc="Evaluating linear programs..."):
		# link_capacities_arr
		# whole_deployment_ug_vols
		# whole_deployment_ug_to_vol

		#### COPYING the part of update_deployment that modifies these components, updating the entire deployment takes too long
		sas.ug_to_vol = d['ug_to_vol']
		sas.whole_deployment_ug_to_vol = d['whole_deployment_ug_to_vol']
		sas.ug_vols = np.zeros(sas.n_ug)
		sas.whole_deployment_ug_vols = np.zeros(sas.whole_deployment_n_ug)
		for ug, v in sas.ug_to_vol.items():
			sas.ug_vols[sas.ug_to_ind[ug]] = v
		for ug, v in sas.whole_deployment_ug_to_vol.items():
			sas.whole_deployment_ug_vols[sas.whole_deployment_ug_to_ind[ug]] = v

		# use verb to ignore cache
		all_rets.append(sas.solve_lp_with_failure_catch(_adv,verb=True,dont_update_deployment=True))
	
	sas.ug_to_vol = dep['ug_to_vol']
	sas.whole_deployment_ug_to_vol = dep['whole_deployment_ug_to_vol']
	sas.ug_vols = np.zeros(sas.n_ug)
	sas.whole_deployment_ug_vols = np.zeros(sas.whole_deployment_n_ug)
	for ug, v in sas.ug_to_vol.items():
		sas.ug_vols[sas.ug_to_ind[ug]] = v
	for ug, v in sas.whole_deployment_ug_to_vol.items():
		sas.whole_deployment_ug_vols[sas.whole_deployment_ug_to_ind[ug]] = v

	i=0
	for Y in Y_vals:
		for X in X_vals:
			for metro in sorted(inflated_deployments[Y][X]):
				prefix_withdrawals[Y][X].append([]) ## unused
				
				soln_adv = all_rets[i]
				i += 1

				latency_deltas = []
				vols = []
				for old_lat, new_lat, vol in zip(base_soln['lats_by_ug'], soln_adv['lats_by_ug'], sas.ug_vols):
					if old_lat == NO_ROUTE_LATENCY or new_lat == NO_ROUTE_LATENCY: continue
					old_lat = 0
					latency_deltas.append(new_lat - old_lat)
					vols.append(vol)
				fraction_congested_volumes[Y][X].append(soln_adv['fraction_congested_volume'])
				if len(latency_deltas) > 0:
					metrics[Y][X].append(np.average(latency_deltas, weights=vols))
				else:
					metrics[Y][X].append(NO_ROUTE_LATENCY)
	sas.deload_realworld_measure_wrapper(plain_deployment)
	return {
		'metrics': metrics,
		'prefix_withdrawals':prefix_withdrawals, 
		'fraction_congested_volume': fraction_congested_volumes,
	}

def get_inflated_total_deployments(sas, X_vals):
	""" Gets deployments with modified user volumes and/or link capacities."""
	#### X_vals: how much to inflate deployment by
	#### we want to see how our ability to withstand increased overall volume
	ret = {X_val: None for X_val in X_vals}
	deployment = sas.output_deployment(copykeys=None)
	for X in X_vals:
		# minimally copy the deployment to not cause memory errors
		quick_deployment = sas.output_deployment(copykeys=['ug_to_vol','whole_deployment_ug_to_vol'])
		## modify volume globally
		for ug,v in quick_deployment['ug_to_vol'].items():
			quick_deployment['ug_to_vol'][ug] = v * (1 + X/100)
			quick_deployment['whole_deployment_ug_to_vol'][ug] = v * (1 + X/100)
		ret[X] = quick_deployment
	return ret


def metro_to_diurnal_factor(metro, hour):
	def diurnal_factor(hour_of_day):
		## From https://dl.acm.org/doi/pdf/10.1145/3341301.3359655
		## linear interpolation of purple line in figure 1
		if hour_of_day < 2:
			return 0.6
		elif hour_of_day < 6:
			return 0.1 * hour_of_day + 0.4
		elif hour_of_day < 10:
			return -0.225 * hour_of_day + 2.35
		elif hour_of_day < 14:
			return 0.1
		elif hour_of_day < 20:
			return 0.5/6 * hour_of_day - 16/15
		else:
			return 0.6

	## Synthetic deployments (dpsize='small'/'decent'/'med') use integer metro
	## indices that aren't in POP2TIMEZONE. Fall back to a deterministic, varied
	## timezone derived from the metro identifier so each synthetic metro still
	## peaks at a different time of day -- which is the whole point of this
	## metric. At actual-N scale, real metro names hit POP2TIMEZONE directly
	## and behavior is unchanged.
	try:
		tz = POP2TIMEZONE[metro]
	except KeyError:
		import hashlib
		# stable across runs/python sessions (Python's builtin hash() isn't)
		h = int(hashlib.md5(str(metro).encode()).hexdigest()[:8], 16)
		tz = (h % 24) - 12
	hour_of_day = (tz + 12 + hour) % 24
	return diurnal_factor(hour_of_day)

def get_diurnal_deployments(sas, diurnal_intensities):
	"""Gets deployments with modified user volumes and/or link capacities modeling a diurnal pattern."""
	ret = {intensity: {hour:{} for hour in range(24)} for intensity in diurnal_intensities}
	metros = list(sorted(list(set(metro for metro,asn in sas.ugs))))
	for intensity in diurnal_intensities:
		for hour in range(24): ## for each hour of the day
			# minimally copy the deployment to not cause memory errors
			quick_deployment = sas.output_deployment(copykeys=['ug_to_vol','whole_deployment_ug_to_vol'])
			for metro in metros:
				multiplier = (1 + intensity/100) * metro_to_diurnal_factor(metro, hour)
				## modify volume in this metro according to a diurnal pattern
				for ug,v in quick_deployment['ug_to_vol'].items():
					if ug[0] == metro:
						quick_deployment['ug_to_vol'][ug] = v * multiplier
						quick_deployment['whole_deployment_ug_to_vol'][ug] = v * multiplier
			ret[intensity][hour]['None'] = quick_deployment
	return ret



def bisect_critical_intensities(sas, adv_by_key, make_deps, lo, hi,
                                rel_tol=0.05, abs_tol=0.1, max_rounds=12,
                                cong_eps=1e-4,
                                cong_key='fraction_congested_volume',
                                reference_key='one_per_peering',
                                label='bisect'):
    """Reference-first parallel bisection for critical congestion
    intensity (Tom 2026-08-23, v2 of his 2026-08-23 grid complaint):

      (a) bisect the REFERENCE (one-per-peering, the optimal routing)
          alone over [lo, hi] -- each round is one midpoint, i.e. one
          batch of n_scenarios LPs (24 hours for diurnal);
      (b) opp's critical is an upper bound for every other solution, so
          the rest bisect over [lo, opp_crit] -- first midpoint is
          ~opp_crit/2 -- with every unresolved solution's midpoint
          batched into one solve_lp_with_failure_catch_mp call per
          round. The halved range plus the shared bound lands the whole
          thing in ~10 rounds.

    adv_by_key: {key: advertisement}
    make_deps(v): list of scenario deployments at intensity v (metros
        for a flash surge, hours for a diurnal multiplier)
    Returns {key: critical_intensity}. Floor/ceiling semantics:
    congested already at lo -> lo; never congested at hi -> hi; a
    non-reference solution that survives opp_crit returns opp_crit
    (a tie with optimal at this tolerance).
    """
    def _batch(points):
        # points: [(key, v)] -> {(key, v): congested_bool}
        call_args = []
        spans = []
        for k, v in points:
            deps = make_deps(v)
            a = threshold_a(adv_by_key[k])
            start = len(call_args)
            call_args.extend((a, d, True) for d in deps)
            spans.append((k, v, start, len(call_args)))
        rets = sas.solve_lp_with_failure_catch_mp(call_args)
        out = {}
        for k, v, i0, i1 in spans:
            fr = 0.0
            for r in rets[i0:i1]:
                try:
                    fr = max(fr, float(r.get(cong_key) or 0.0))
                except (TypeError, AttributeError):
                    pass
            out[(k, v)] = fr > cong_eps
        for k, v, i0, i1 in spans:
            frs = []
            for r in rets[i0:i1]:
                try:
                    frs.append(float(r.get(cong_key) or 0.0))
                except (TypeError, AttributeError):
                    frs.append(-1.0)
            print('[{}] eval {} @ {:.3f}: max_cong={:.4f} over {} scenarios -> {}'.format(
                label, k, v, max(frs) if frs else -1.0, i1 - i0,
                'CONGESTED' if out[(k, v)] else 'ok'), flush=True)
        return out

    def _run(keys, k_lo, k_hi, probe_hi=True):
        print('[{}] search over [{:.3f}, {:.3f}] for {}'.format(
            label, float(k_lo), float(k_hi), keys), flush=True)
        bracket = {k: [float(k_lo), float(k_hi)] for k in keys}
        resolved = {}
        # probe_hi=False in phase B: congested-at-opp_crit is the trivial
        # expected result for a suboptimal solution, so skip that batch and
        # go straight to the opp_crit/2 midpoint (Tom 2026-08-23). A
        # solution that in fact never congests converges to ~k_hi anyway.
        probes = [(k, float(k_lo)) for k in keys]
        if probe_hi:
            probes += [(k, float(k_hi)) for k in keys]
        ends = _batch(probes)
        for k in keys:
            if ends[(k, float(k_lo))]:
                resolved[k] = float(k_lo)   # congested at the floor
                print('[{}] {} congested at floor {:.3f} -> floor'.format(
                    label, k, float(k_lo)), flush=True)
            elif probe_hi and not ends[(k, float(k_hi))]:
                resolved[k] = float(k_hi)   # never congests in range
                print('[{}] {} never congests up to {:.3f} -> ceiling'.format(
                    label, k, float(k_hi)), flush=True)
        for _round in range(max_rounds):
            # stop width: max(abs_tol, rel_tol*lo) -- no point resolving
            # an intensity multiplier below +/-0.25 absolute (Tom 2026-08-23)
            todo = [k for k in keys if k not in resolved
                    and bracket[k][1] - bracket[k][0] >
                    max(abs_tol, rel_tol * max(abs(bracket[k][0]), 1e-9))]
            if not todo:
                break
            mids = [(k, 0.5 * (bracket[k][0] + bracket[k][1]))
                    for k in todo]
            print('[{}] round {}: midpoints {}'.format(label, _round + 1,
                {k: round(v, 3) for k, v in mids}), flush=True)
            res = _batch(mids)
            for k, v in mids:
                if res[(k, v)]:
                    bracket[k][1] = v
                else:
                    bracket[k][0] = v
                print('[{}]   {} bracket -> [{:.3f}, {:.3f}]'.format(
                    label, k, bracket[k][0], bracket[k][1]), flush=True)
        for k in keys:
            if k not in resolved:
                resolved[k] = 0.5 * (bracket[k][0] + bracket[k][1])
        return resolved

    keys = list(adv_by_key)
    if reference_key not in adv_by_key or len(keys) == 1:
        return _run(keys, lo, hi)
    print('[{}] PHASE A: bisect reference {} alone'.format(
        label, reference_key), flush=True)
    ref_crit = _run([reference_key], lo, hi)[reference_key]
    print('[{}] PHASE A done: {} critical = {:.3f}; PHASE B caps everyone at it'.format(
        label, reference_key, ref_crit), flush=True)
    others = [k for k in keys if k != reference_key]
    out = _run(others, lo, max(ref_crit, float(lo) * (1 + rel_tol)),
               probe_hi=False)
    out[reference_key] = ref_crit
    return out
