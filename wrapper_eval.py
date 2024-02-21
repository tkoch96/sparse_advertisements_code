import tqdm, numpy as np, os
from constants import *
gamma = 10
capacity = True
N_TO_SIM = 2 if DPSIZE not in ['really_friggin_small','small'] else 1
#### NOTE -- need to make sure lambduh decreases with the problem size
#### or else the latency gains won't be significant enough to get a signal through
lambduh = .00001
soln_types = ['sparse', 'anycast', 'one_per_pop', 'painter', 'anyopt', 'random', 'one_per_peering']
# soln_types = ['sparse', 'painter', 'anyopt', 'oracle']

performance_metrics_fn = os.path.join(CACHE_DIR, 'popp_failure_latency_comparison_{}.pkl'.format(DPSIZE))

def assess_failure_resilience(sas, adv, which='popps'):
	ret = {'congestion_delta': [], 'latency_delta_optimal': [], 'latency_delta_before': []}
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
					iteri_to_ugs[iteri].append(ug)
				except KeyError:
					iteri_to_ugs[iteri] = [ug]
			else:
				try:
					iteri_to_ugs[iteri[0]].append(ug)
				except KeyError:
					iteri_to_ugs[iteri[0]] = [ug]

	for iteri in iterover:
		adv_cpy = np.copy(adv)
		if which == 'popps':
			adv_cpy[sas.popp_to_ind[iteri],:] = 0
		else:
			these_popps = np.array([sas.popp_to_ind[popp] for popp in sas.popps if popp[0] == iteri])
			adv_cpy = np.copy(adv)
			adv_cpy[these_popps,:] = 0
		## q: what is latency experienced for these ugs compared to optimal?
		call_args.append((adv_cpy, dep))

		

		## best user latencies is not necessarily just lowest latency
		## need to factor in capacity
		one_per_peer_adv = np.eye(sas.n_popps)
		if which == 'popps':
			one_per_peer_adv[sas.popp_to_ind[iteri],:] = 0
		else:
			for popp in sas.popps:
				if popp[0] == iteri:
					one_per_peer_adv[sas.popp_to_ind[popp],:] = 0
		call_args.append((one_per_peer_adv, dep))

	old_user_latencies = sas.solve_lp_with_failure_catch(adv)['lats_by_ug']
	lp_rets = sas.solve_lp_with_failure_catch_mp(call_args)
	for i,iteri in enumerate(iterover):	

		## q: what is latency experienced for these ugs compared to optimal?
		this_soln = lp_rets[i*2]
		try:
			user_latencies = this_soln['lats_by_ug']
		except KeyError:
			print(this_soln)
			continue
		## best user latencies is not necessarily just lowest latency
		## need to factor in capacity
		best_soln = lp_rets[i*2+1]
		best_user_latencies = best_soln['lats_by_ug']

		ret['congestion_delta'].append(this_soln['fraction_congested_volume'] - best_soln['fraction_congested_volume'])

		# these_ugs = iteri_to_ugs.get(iteri,[])
		# for ug in these_ugs:

		for ug in sas.ugs:
			old_perf = old_user_latencies[sas.ug_to_ind[ug]]
			new_perf = user_latencies[sas.ug_to_ind[ug]]
			best_perf = best_user_latencies[sas.ug_to_ind[ug]]
			ret['latency_delta_optimal'].append((best_perf - new_perf, 
				sas.ug_to_vol[ug], ug, iteri))
			ret['latency_delta_before'].append((old_perf - new_perf, 
				sas.ug_to_vol[ug], ug, iteri))
	return ret


def assess_resilience_to_flash_crowds_mp(sas, adv, solution, X_vals):
	## !!!!!!for painter/TIPSY!!!!!!
	## assume each metro's volume increases by X times on average
	## see if there's a solution
	## if there's a solution, do it and note the latency penalty compared to optimal

	# return cdf of latency penalties, possibly as a function of X
	metrics = {X:[] for X in X_vals}
	prefix_withdrawals = {X:[] for X in X_vals}
	fraction_congested_volumes = {X:[] for X in X_vals}

	adv = threshold_a(adv)

	base_soln = sas.solve_lp_with_failure_catch(adv)

	print("Populating multiprocessing call args...")
	call_args = []
	for X in X_vals:
		vol_by_metro = {}
		for metro,asn in sas.ugs:
			try:
				vol_by_metro[metro] += sas.ug_to_vol[(metro,asn)]
			except KeyError:
				vol_by_metro[metro] = sas.ug_to_vol[(metro,asn)]

		for metro,vol in vol_by_metro.items():
			deployment = sas.output_deployment()
			for ug,v in deployment['ug_to_vol'].items():
				if ug[0] == metro:
					deployment['ug_to_vol'][ug] = v * (1 + X/100)
			call_args.append((adv, deployment))

	### Call all the solutions with multiprocessing
	all_rets = sas.solve_lp_with_failure_catch_mp(call_args)
	i=0
	print("Done, parsing return values from workers")
	for X in X_vals:
		vol_by_metro = {}
		for metro,asn in sas.ugs:
			try:
				vol_by_metro[metro] += sas.ug_to_vol[(metro,asn)]
			except KeyError:
				vol_by_metro[metro] = sas.ug_to_vol[(metro,asn)]

		for metro,vol in vol_by_metro.items():
			prefix_withdrawals[X].append([])
			
			soln_adv = all_rets[i]
			i += 1

			latency_deltas = []
			vols = []
			ugi=0
			for old_lat, new_lat, vol in zip(base_soln['lats_by_ug'], soln_adv['lats_by_ug'], sas.ug_vols):
				latency_deltas.append(new_lat - old_lat)
				vols.append(vol)
				ugi += 1
			fraction_congested_volumes[X].append(soln_adv['fraction_congested_volume'] - base_soln['fraction_congested_volume'])
			metrics[X].append((latency_deltas, vols))

	return {
		'metrics': metrics,
		'prefix_withdrawals':prefix_withdrawals, 
		'fraction_congested_volume': fraction_congested_volumes,
	}