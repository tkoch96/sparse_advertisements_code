import numpy as np,  scipy, time
from helpers import *

TOL = .001

def solve_lp_with_failure_catch(sas, adv, **kwargs):
	### minimizes average latency, but if that fails it instead 
	### minimizes MLU
	computing_best_lats = kwargs.get('computing_best_lats', False)
	ret_min_latency = solve_lp_assignment(sas, adv, **kwargs)
	if ret_min_latency['solved']:
		return ret_min_latency

	if np.sum(adv.flatten()) == 0:
		return {'solved': False}
	ugs = sas.ugs
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
	## 1s -> upper bound on utilization, 0s -> path distributions
	dummy_minimizer = np.concatenate([np.ones((sas.n_popps)),np.zeros((n_paths))])

	## upper bound A for enforcing utilization
	A_util = np.zeros((sas.n_popps, sas.n_popps + n_paths))
	for linkj in range(sas.n_popps):
		A_util[linkj,linkj] = -1
	for i,(ug,poppi) in enumerate(available_paths):
		A_util[poppi, sas.n_popps + i] = 1 / caps[poppi]
	b_ub = np.zeros((sas.n_popps))		

	## conservation of user volume
	volume_conservation_A = np.zeros((sas.n_ug, sas.n_popps + n_paths))
	conservation_b = sas.ug_vols
	# conservation_b = np.ones((sas.n_ug,1))

	for pli in range(n_paths):
		poppi = available_paths[pli][1]
		ugi = sas.ug_to_ind[available_paths[pli][0]]
		volume_conservation_A[ugi,sas.n_popps + pli] = 1


	ts = time.time()
	res = scipy.optimize.linprog(dummy_minimizer, 
		A_eq = volume_conservation_A, b_eq = conservation_b,
		A_ub = A_util, b_ub = b_ub,
		options={'tol':TOL})
	# print("A {} s, {} iter, res: {}".format(round(time.time() - ts,2), res.nit,
	# 	np.sum(np.abs(res.con))))

	## Distribution is the amount of volume (not percent) placed on each path
	## a path is specified by a <user, popp>
	distribution = res.x.round(2)
	path_distribution = distribution[sas.n_popps:]

	## Compute paths by ug
	paths_by_ug = {}
	for (ug,poppi),vol_amt in zip(available_paths, path_distribution):
		if vol_amt > 0:
			ugi = sas.ug_to_ind[ug]
			vol_pct = vol_amt / sas.ug_to_vol[ug]
			try:
				paths_by_ug[ugi].append((poppi, vol_pct))
			except KeyError:
				paths_by_ug[ugi] = [(poppi, vol_pct)]

	## Compute utils so that we can penalize users with no option
	vols_by_poppi = {poppi:0 for poppi in range(sas.n_popps)}
	for ugi, poppivols in paths_by_ug.items():
		ug = sas.ugs[ugi]
		for poppi,volpct in poppivols:
			vols_by_poppi[poppi] += (volpct * sas.ug_to_vol[ug])
	vols_by_poppi = {poppi:v/caps[poppi] for poppi,v in vols_by_poppi.items()}
	inundated_popps = {poppi:None for poppi,v in vols_by_poppi.items() if v > 1}
	if not res.success:
		print("\n\n\nNo solution in minimizing MLU, here are inundated popps : {}\n\n".format(
			inundated_popps))

	inflations, lats_by_ug = [], {}
	for ugi, pathvols in paths_by_ug.items():
		ug = sas.ugs[ugi]
		if not computing_best_lats:
			best_lat = sas.best_lats_by_ug[sas.ug_to_ind[ug]]
		else:
			best_lat = 0
		these_lats = []
		cum_vol = 0
		for poppi,vol in pathvols:
			try:
				inundated_popps[poppi]
				these_lats.append((NO_ROUTE_LATENCY, vol))
			except KeyError:
				popp = sas.popps[poppi]
				these_lats.append((sas.ug_perfs[ug][popp], vol))
			cum_vol += vol
		avg_lat = np.sum([el[0] * el[1] for el in these_lats]) / cum_vol
		lats_by_ug[ug] = avg_lat
		avg_inflation = avg_lat - best_lat
		inflations.append((avg_inflation, sas.ug_to_vol[ug]))
	lats_by_ug_arr = np.zeros((sas.n_ug))
	for ug,lat in lats_by_ug.items():
		lats_by_ug_arr[sas.ug_to_ind[ug]] = lat

	return {
		"paths_by_ug": paths_by_ug,
		"lats_by_ug" : lats_by_ug_arr,
		"solved": res.success,
	}

def solve_lp_assignment(sas, adv, ugs=None, verb=False, **kwargs):
	### Minimizes average latency subject to not inundating a link,
	### but could fail if there's not enough aggregate capacity
	computing_best_lats = kwargs.get('computing_best_lats', False)
	if np.sum(adv.flatten()) == 0:
		return {'solved': False}


	ugs = sas.ugs
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

	weighted_available_latencies = np.array([sas.ug_perfs[ug][sas.popps[poppi]] * sas.ug_to_vol[ug] \
		for ug,poppi in available_paths])
	if verb:
		available_latencies = np.array([sas.ug_perfs[ug][sas.popps[poppi]] for 
			ug,poppi in available_paths])
		print(adv)
		print(available_paths)
		print(available_latencies)

	cap_constraint_A = np.zeros((sas.n_popps, n_paths))
	caps = np.reshape(sas.link_capacities_arr, (sas.n_popps,1))

	volume_conservation_A = np.zeros((sas.n_ug, n_paths))
	conservation_b = sas.ug_vols
	# conservation_b = np.ones((sas.n_ug,1))

	for pli in range(n_paths):
		poppi = available_paths[pli][1]
		ugi = sas.ug_to_ind[available_paths[pli][0]]
		cap_constraint_A[poppi,pli] = 1
		volume_conservation_A[ugi,pli] = 1


	### Solve for volume on each popp,user
	ts = time.time()
	res = scipy.optimize.linprog(weighted_available_latencies, 
		A_ub = cap_constraint_A, b_ub = caps,
		A_eq = volume_conservation_A, b_eq = conservation_b,
		options={'tol':TOL})
	# print("B {} s, {} iter, res: {}".format(round(time.time() - ts,2), res.nit,
	# 	np.sum(np.abs(res.con))))

	## Distribution is the amount of volume (not percent) placed on each path
	## a path is specified by a <user, popp>
	distribution = res.x.round(4)
	if verb:
		print(distribution)

	paths_by_ug = {}
	for (ug,poppi),vol_amt in zip(available_paths, distribution):
		if vol_amt > 0:
			ugi = sas.ug_to_ind[ug]
			vol_pct = vol_amt / sas.ug_to_vol[ug]
			try:
				paths_by_ug[ugi].append((poppi, vol_pct))
			except KeyError:
				paths_by_ug[ugi] = [(poppi, vol_pct)]

	if verb:
		print(paths_by_ug)
		## Compute utils so that we can penalize users with no option
		vols_by_poppi = {poppi:0 for poppi in range(sas.n_popps)}
		for ugi, poppivols in paths_by_ug.items():
			ug = sas.ugs[ugi]
			for poppi,volpct in poppivols:
				vols_by_poppi[poppi] += (volpct * sas.ug_to_vol[ug])
		vols_by_poppi = {poppi:v/caps[poppi] for poppi,v in vols_by_poppi.items()}
		inundated_popps = {poppi:None for poppi,v in vols_by_poppi.items() if v > 1}
		if not res.success:
			print("\n\n\nNo solution in minimizing MLU, here are inundated popps : {}\n\n".format(
				inundated_popps))

	inflations, lats_by_ug = [], {}
	for ugi, pathvols in paths_by_ug.items():
		ug = sas.ugs[ugi]
		if not computing_best_lats:
			best_lat = sas.best_lats_by_ug[ugi]
		else:
			best_lat = 0
		these_lats = []
		cum_vol = 0
		for poppi,vol in pathvols:
			popp = sas.popps[poppi]
			these_lats.append((sas.ug_perfs[ug][popp], vol))
			cum_vol += vol
		avg_lat = np.sum([el[0] * el[1] for el in these_lats]) / cum_vol
		lats_by_ug[ug] = avg_lat
		avg_inflation = avg_lat - best_lat
		inflations.append((avg_inflation, sas.ug_to_vol[ug]))
	lats_by_ug_arr = np.zeros((sas.n_ug))
	for ug,lat in lats_by_ug.items():
		lats_by_ug_arr[sas.ug_to_ind[ug]] = lat

	
	return {
		"paths_by_ug": paths_by_ug,
		"lats_by_ug" : lats_by_ug_arr,
		"solved": res.success,
	}