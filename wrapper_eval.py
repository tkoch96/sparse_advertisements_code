import tqdm, numpy as np, os, copy
from constants import *
gamma = 4
capacity = True
N_TO_SIM = 1
#### NOTE -- need to make sure lambduh decreases with the problem size
#### or else the latency gains won't be significant enough to get a signal through
lambduh = .00001
# soln_types = ['sparse', 'anycast', 'one_per_pop', 'painter', 'anyopt', 'random', 'one_per_peering']
global_soln_types = ['anyopt', 'painter', 'sparse', 'anycast', 'one_per_pop', 'one_per_peering']
# soln_types = ['sparse', 'painter', 'anyopt', 'oracle']

global_performance_metrics_fn =  lambda dps : os.path.join(CACHE_DIR, 'popp_failure_latency_comparison_{}.pkl'.format(dps))

### Default metrics for performance evaluations
default_metrics = {
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
	'popp_failures_latency_lagrange_optimal_specific': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'popp_failures_congestion': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'popp_failures_congestion_penalty': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'popp_failures_congestion_lagrange': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_latency_optimal': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_latency_optimal_specific': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_latency_penalty_optimal_specific': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_latency_lagrange_optimal_specific': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_latency_before': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_congestion': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_congestion_penalty': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
	'pop_failures_congestion_lagrange': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},

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
	'volume_multipliers': {i:{k:[] for k in global_soln_types} for i in range(N_TO_SIM)},
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

def assess_failure_resilience(sas, adv, which='popps', **kwargs):
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
	use_lagrange = kwargs.get('lagrange', False)

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
		if use_penalty:
			opt_adv = np.eye(sas.n_popps)
			if which == 'popps':
				opt_adv[sas.popp_to_ind[iteri],:] = 0
			else:
				these_popps = np.array([sas.popp_to_ind[popp] for popp in sas.popps if popp[0] == iteri])
				opt_adv[these_popps,:] = 0
			call_args.append((adv_cpy, opt_adv, dep, i%20==0))
		else:
			call_args.append((adv_cpy, dep, i%20==0))

		## best user latencies is not necessarily just lowest latency
		## need to factor in capacity
		one_per_peer_adv = np.eye(sas.n_popps)
		if which == 'popps':
			one_per_peer_adv[sas.popp_to_ind[iteri],:] = 0
		else:
			for popp in sas.popps:
				if popp[0] == iteri:
					one_per_peer_adv[sas.popp_to_ind[popp],:] = 0
		call_args.append((one_per_peer_adv, dep, i%20==0))
	
	base_soln = sas.solve_lp_with_failure_catch(adv)
	base_user_latencies = base_soln['lats_by_ug']

	if use_lagrange:
		lp_rets = sas.solve_lp_with_failure_catch_mp(call_args, worker_cmd='solve_lp_lagrange', cache_key='lagrange', cache_res=True, **kwargs)
	else:
		lp_rets = sas.solve_lp_with_failure_catch_mp(call_args, cache_res=True, **kwargs)

	for i,iteri in enumerate(iterover):	

		## q: what is latency experienced for these ugs compared to optimal?
		this_soln = lp_rets[i*2]
		user_latencies = this_soln['lats_by_ug']
		## best user latencies is not necessarily just lowest latency
		## need to factor in capacity
		best_soln = lp_rets[i*2+1]
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
	lp_rets = sas.solve_lp_with_failure_catch_mp(call_args, cache_res=True)

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
		from realworld_measure_wrapper import popps_to_hash
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
	from deployment_setup import get_link_capacities
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

	call_args = []
	for Y_val in Y_vals:
		for X_val in X_vals:
			for metro, d in inflated_deployments[Y_val][X_val].items():
				## always clear the deployment cache (True on third arg)
				call_args.append((adv, d, True))

	### Call all the solutions with multiprocessing
	all_rets = sas.solve_lp_with_failure_catch_mp(call_args)
	i=0
	print("Done, parsing return values from workers")
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
				path_to_path_val = {}
				for path_val, (ug,poppi) in zip(soln_adv['raw_solution'], soln_adv['available_paths']):
					path_to_path_val[ug,poppi] = path_val
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
		
	hour_of_day = (POP2TIMEZONE[metro] + 12 + hour) % 24
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

def assess_volume_multipliers(sas, adv, solution, inflated_deployments):
	# return cdf of latency penalties, possibly as a function of X
	X_vals = list(sorted(list(inflated_deployments)))
	metrics = {X: None for X in X_vals}

	adv = threshold_a(adv)
	base_soln = sas.solve_lp_with_failure_catch(adv)
	call_args = []
	for X_val in X_vals:
		## always clear the deployment cache (True on third arg)
		d = inflated_deployments[X_val]
		call_args.append((adv, d, True))

	### Call all the solutions with multiprocessing
	all_rets = sas.solve_lp_with_failure_catch_mp(call_args)
	i=0
	print("Done, parsing return values from workers")
	for X in X_vals:
		soln_adv = all_rets[i]
		i += 1

		latency_deltas = []
		vols = []
		for old_lat, new_lat, vol in zip(base_soln['lats_by_ug'], soln_adv['lats_by_ug'], sas.ug_vols):
			if old_lat == NO_ROUTE_LATENCY or new_lat == NO_ROUTE_LATENCY: 
				print("Congestion in assess volume multipliers, which shouldn't happen")
				print("{} {} {}".format(X, old_lat, new_lat))
				raise ValueError
			# latency_deltas.append(new_lat - old_lat)
			latency_deltas.append(new_lat)
			vols.append(vol)
		metrics[X] = np.average(latency_deltas, weights=vols)

	return {
		'metrics': metrics,
	}


