import numpy as np,  scipy, time
from helpers import *

def solve_lp_with_failure_catch(sas, adv, **kwargs):
	### minimizes average latency, but if that fails it instead 
	### minimizes MLU
	ts = time.time()
	verb = False
	computing_best_lats = kwargs.get('computing_best_lats', False)
	ret_min_latency = solve_lp_assignment(sas, adv, **kwargs)
	if verb:
		print("Trying first without MLU took {} s".format(round(time.time() - ts,2)))
		ts = time.time()
	if ret_min_latency['solved']:
		if kwargs.get('smallverb'):
			print("Solved LP just minimizing latency")
		return ret_min_latency
	elif kwargs.get('smallverb'):
		print("Failed to solve min latency problem.")

	if np.sum(adv.flatten()) == 0:
		print("No solution because no active advertisements")
		return {'solved': False}
	ugs = sas.whole_deployment_ugs
	## get vector of latency of all users to all (available) ingresses
	# First, get winning ingresses from available prefixes and the priority model
	routed_through_ingress,_ = sas.calculate_ground_truth_ingress(adv)

	all_ug_lat_ingresses = {}
	for prefixi in routed_through_ingress:
		for ug in routed_through_ingress[prefixi]:
			poppi = routed_through_ingress[prefixi][ug]
			all_ug_lat_ingresses[ug,poppi] = None

	available_paths = sorted(list(all_ug_lat_ingresses), key = lambda el : el[0])
	n_paths = len(available_paths)

	caps = np.reshape(sas.link_capacities_arr, (sas.n_popps,1))

	## optimization variable is [Y,v]
	## Y is dummy upper bound variable, v is percent of volume UG places on path
	## 1s -> upper bound on utilization, small numbers -> path distributions
	# weighted_available_latencies = np.array([sas.whole_deployment_ug_perfs[ug][sas.popps[poppi]] * sas.whole_deployment_ug_to_vol[ug] \
	# 	for ug,poppi in available_paths])
	weighted_available_latencies = np.array([sas.whole_deployment_ug_perfs[ug][sas.popps[poppi]] \
		for ug,poppi in available_paths])
	dummy_minimizer = np.concatenate([np.ones((sas.n_popps)),ALPHA * weighted_available_latencies])

	## upper bound A for enforcing utilization
	A_util = np.zeros((sas.n_popps, sas.n_popps + n_paths))
	for linkj in range(sas.n_popps):
		A_util[linkj,linkj] = -1
	for i,(ug,poppi) in enumerate(available_paths):
		A_util[poppi, sas.n_popps + i] = 1 / caps[poppi]
	b_ub = np.zeros((sas.n_popps))		

	## conservation of user volume
	volume_conservation_A = np.zeros((sas.whole_deployment_n_ug, sas.n_popps + n_paths))
	conservation_b = sas.whole_deployment_ug_vols

	for pli in range(n_paths):
		poppi = available_paths[pli][1]
		ugi = sas.whole_deployment_ug_to_ind[available_paths[pli][0]]
		volume_conservation_A[ugi,sas.n_popps + pli] = 1

	if verb:
		print("Setup took {} s".format(round(time.time() - ts,2)))
		ts = time.time()

	res = scipy.optimize.linprog(dummy_minimizer, 
		A_eq = volume_conservation_A, b_eq = conservation_b,
		A_ub = A_util, b_ub = b_ub,
		method='highs-ds')
	if verb:
		print("Solving took {} s".format(round(time.time() - ts,2)))
		ts = time.time()

	if res.status != 0:
		### infeasible problem, likely no route for some users
		# print("Infeasible problem")
		return {
			'solved': False,
		}

	## Distribution is the amount of volume (not percent) placed on each path
	## a path is specified by a <user, popp>
	distribution = res.x
	path_distribution = distribution[sas.n_popps:]

	## Compute paths by ug
	paths_by_ug = {}
	for (ug,poppi),vol_amt in zip(available_paths, path_distribution):
		if vol_amt > 0:
			if poppi == 5 and kwargs.get('verb'):
				print("{} {} {}".format(ug, poppi, vol_amt))
			ugi = sas.whole_deployment_ug_to_ind[ug]
			vol_pct = vol_amt / sas.whole_deployment_ug_to_vol[ug]
			try:
				paths_by_ug[ugi].append((poppi, vol_pct))
			except KeyError:
				paths_by_ug[ugi] = [(poppi, vol_pct)]

	## Compute utils so that we can penalize users with no option
	vols_by_poppi = {poppi:0 for poppi in range(sas.n_popps)}
	for ugi, poppivols in paths_by_ug.items():
		ug = sas.whole_deployment_ugs[ugi]
		for poppi,volpct in poppivols:
			vols_by_poppi[poppi] += (volpct * sas.whole_deployment_ug_to_vol[ug])
	vols_by_poppi = {poppi:v/float(caps[poppi]) for poppi,v in vols_by_poppi.items()}
	inundated_popps = {poppi:None for poppi,v in vols_by_poppi.items() if v > 1}
	if not res.success:
		print("\n\n\nNo solution in minimizing MLU, here are inundated popps : {}\n\n".format(
			inundated_popps))
	if kwargs.get('verb'):
		print("Inundated popps {}, \n vols by poppi: {}".format(inundated_popps, vols_by_poppi))

	lats_by_ug = {}
	all_volume, congested_volume = 0, 0
	for ugi, pathvols in paths_by_ug.items():
		ug = sas.whole_deployment_ugs[ugi]
		# if kwargs.get('verb'):
		# 	print("{} ({}) -- {}".format(ugi,ug,pathvols))
		these_lats = []
		cum_vol = 0
		for poppi,vol in pathvols:
			try:
				inundated_popps[poppi]
				if kwargs.get('really_bad_fail',False):
					these_lats.append((NO_ROUTE_LATENCY*100, vol))
				else:
					these_lats.append((NO_ROUTE_LATENCY, vol))
				congested_volume += sas.whole_deployment_ug_vols[ugi] * vol
			except KeyError:
				popp = sas.popps[poppi]
				these_lats.append((sas.whole_deployment_ug_perfs[ug][popp], vol))
			cum_vol += vol
			all_volume += sas.whole_deployment_ug_vols[ugi]
		avg_lat = np.sum([el[0] * el[1] for el in these_lats]) / cum_vol
		lats_by_ug[ug] = avg_lat
	lats_by_ug_arr = np.zeros((sas.whole_deployment_n_ug))
	for ug,lat in lats_by_ug.items():
		lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = lat

	fraction_congested_volume = congested_volume / all_volume

	if verb:
		print("postprocess took {} s".format(round(time.time() - ts,2)))
		ts = time.time()
		print(lats_by_ug_arr[0])

	return {
		"objective": res.fun,
		"raw_solution": res.x,
		"paths_by_ug": paths_by_ug,
		"lats_by_ug" : lats_by_ug_arr,
		"solved": res.success,
		"vols_by_poppi": vols_by_poppi,
		"fraction_congested_volume": fraction_congested_volume,
	}

def solve_lp_assignment(sas, adv, ugs=None, verb=False, **kwargs):
	### Minimizes average latency subject to not inundating a link,
	### but could fail if there's not enough aggregate capacity
	computing_best_lats = kwargs.get('computing_best_lats', False)
	if np.sum(adv.flatten()) == 0:
		return {'solved': False}


	ugs = sas.whole_deployment_ugs
	## get vector of latency of all users to all (available) ingresses
	# First, get winning ingresses from available prefixes and the priority model
	routed_through_ingress,_ = sas.calculate_ground_truth_ingress(adv)
	all_ug_lat_ingresses = {}
	for prefixi in routed_through_ingress:
		for ug in routed_through_ingress[prefixi]:
			poppi = routed_through_ingress[prefixi][ug]
			all_ug_lat_ingresses[ug,poppi] = None

	available_paths = sorted(list(all_ug_lat_ingresses), key = lambda el : el[0])
	n_paths = len(available_paths)

	# weighted_available_latencies = np.array([sas.whole_deployment_ug_perfs[ug][sas.popps[poppi]] * 1 * sas.whole_deployment_ug_to_vol[ug] \
	# 	for ug,poppi in available_paths])
	weighted_available_latencies = np.array([sas.whole_deployment_ug_perfs[ug][sas.popps[poppi]] * 1 \
		for ug,poppi in available_paths])
	# if verb:
	# 	available_latencies = np.array([sas.ug_perfs[ug][sas.popps[poppi]] for 
	# 		ug,poppi in available_paths])
	# 	print(adv)
	# 	print(available_paths)
	# 	print(available_latencies)

	cap_constraint_A = np.zeros((sas.n_popps, n_paths))
	caps = np.reshape(sas.link_capacities_arr, (sas.n_popps,1))

	volume_conservation_A = np.zeros((sas.whole_deployment_n_ug, n_paths))
	conservation_b = sas.whole_deployment_ug_vols
	# conservation_b = np.ones((sas.n_ug,1))

	for pli in range(n_paths):
		poppi = available_paths[pli][1]
		ugi = sas.whole_deployment_ug_to_ind[available_paths[pli][0]]
		
		cap_constraint_A[poppi,pli] = 1

		volume_conservation_A[ugi,pli] = 1


	### Solve for volume on each popp,user
	ts = time.time()
	res = scipy.optimize.linprog(weighted_available_latencies, 
		A_ub = cap_constraint_A, b_ub = caps,
		A_eq = volume_conservation_A, b_eq = conservation_b)
	# print("B {} s, {} iter, res: {}".format(round(time.time() - ts,2), res.nit,
	# 	np.sum(np.abs(res.con))))

	##### !!!!!!!!!!!!!!!!!!
	## Distribution is the AMOUNT OF VOLUME (NOT PERCENT) placed on each path
	## a path is specified by a <user, popp>
	##### !!!!!!!!!!!!!!!!!!

	if res.status != 0:
		return {'solved': False}
	distribution = res.x
	if verb:
		print("Solved distribution without any congestion")

	paths_by_ug = {}
	for (ug,poppi),vol_amt in zip(available_paths, distribution):
		if vol_amt > 0:
			ugi = sas.whole_deployment_ug_to_ind[ug]
			vol_pct = vol_amt / sas.whole_deployment_ug_to_vol[ug]
			try:
				paths_by_ug[ugi].append((poppi, vol_pct))
			except KeyError:
				paths_by_ug[ugi] = [(poppi, vol_pct)]

	## Compute utils so that we can penalize users with no option
	vols_by_poppi = {poppi:0 for poppi in range(sas.n_popps)}
	for ugi, poppivols in paths_by_ug.items():
		ug = sas.whole_deployment_ugs[ugi]
		for poppi,volpct in poppivols:
			vols_by_poppi[poppi] += (volpct * sas.whole_deployment_ug_to_vol[ug])
	# Convert to poppi utilizations
	vols_by_poppi = {poppi:round(v/float(caps[poppi]),2) for poppi,v in vols_by_poppi.items()}
	inundated_popps = {poppi:v for poppi,v in vols_by_poppi.items() if v > 1}
	if not res.success:
		print("\n\n\nNo solution in minimizing MLU, here are inundated popps : {}\n\n".format(
			inundated_popps))

	lats_by_ug = {}
	all_volume, congested_volume = 0, 0
	for ugi, pathvols in paths_by_ug.items():
		ug = sas.whole_deployment_ugs[ugi]
		these_lats = []
		cum_vol = 0
		for poppi,vol in pathvols:
			try:
				inundated_popps[poppi]
				if kwargs.get('really_bad_fail',False):
					these_lats.append((NO_ROUTE_LATENCY*100, vol))
				else:
					these_lats.append((NO_ROUTE_LATENCY, vol))
				congested_volume += sas.whole_deployment_ug_vols[ugi] * vol
			except KeyError:
				popp = sas.popps[poppi]
				these_lats.append((sas.whole_deployment_ug_perfs[ug][popp], vol))
			cum_vol += vol
			all_volume += sas.whole_deployment_ug_vols[ugi]
		avg_lat = np.sum([el[0] * el[1] for el in these_lats]) / cum_vol
		lats_by_ug[ug] = avg_lat
	lats_by_ug_arr = np.zeros((sas.whole_deployment_n_ug))
	for ug,lat in lats_by_ug.items():
		lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = lat

	fraction_congested_volume = congested_volume / all_volume
	return {
		"objective": res.fun,
		"raw_solution": res.x,
		"paths_by_ug": paths_by_ug,
		"lats_by_ug" : lats_by_ug_arr,
		"solved": res.success,
		"vols_by_poppi": vols_by_poppi,
		"fraction_congested_volume": fraction_congested_volume,
	}
	