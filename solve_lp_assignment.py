import numpy as np,  scipy, time
from helpers import *


def solve_lp_across_prefixes(sas, adv):
	###### PROBABLY DOESN'T WORK
	###### question is how to go from assignment of ug,poppi,prefix to just poppi,prefix

	### simultaneously solves for user allocation to prefixes and 
	### prefix allocation to popps

	# compute valid (popp, prefix, client pairs)
	n_popp_per_client_prefix = 5 ## consider this many per client, to limit problem size
	max_n_prefixes = sas.n_prefixes
	n_paths = n_popp_per_client * max_n_prefixes * sas.n_ug
	available_paths = {}
	itr = 0
	for ug in sas.ugs:
		most_likely_popps = sas.get_popp_pdf(ug, n_popp_per_client_prefix)
		for popp,prob in most_likely_popps:
			if popp is not None:
				for prefix_i in range(max_n_prefixes):
					available_paths[sas.ug_to_ind[ug],sas.popp_to_ind[popp],prefix_i] = None

	available_paths = sorted(list(all_ug_lat_ingresses), key = lambda el : str(el[0]) + str(el[1]) + str(el[2]))

	caps = np.reshape(sas.link_capacities_arr, (sas.n_popps,1))

	## optimization variable is [Y,v]
	## Y is dummy upper bound variable, v is percent of volume UG places on path
	## 1s -> upper bound on utilization, small numbers -> path distributions
	
	ALPHA = .01  # alpha controls tradeoff between minimizing MLU and minimizing latency
	
	weighted_available_latencies = np.array([sas.ug_perfs[ug][sas.popps[poppi]] * sas.ug_to_vol[ug] \
		for ug,poppi,_ in available_paths])
	dummy_minimizer = np.concatenate([np.ones((sas.n_popps)), ALPHA * weighted_available_latencies])

	## upper bound A for enforcing utilization
	A_util = np.zeros((sas.n_popps, sas.n_popps + n_paths))
	for linkj in range(sas.n_popps):
		A_util[linkj,linkj] = -1
	for i,(ug,poppi,prefixi) in enumerate(available_paths):
		A_util[poppi, sas.n_popps + i] = 1 / caps[poppi]
	b_ub = np.zeros((sas.n_popps))		

	## conservation of user volume
	volume_conservation_A = np.zeros((sas.n_ug, sas.n_popps + n_paths))
	conservation_b = sas.ug_vols

	for pli in range(n_paths):
		poppi = available_paths[pli][1]
		ugi = sas.ug_to_ind[available_paths[pli][0]]
		volume_conservation_A[ugi,sas.n_popps + pli] = 1


	ts = time.time()
	res = scipy.optimize.linprog(dummy_minimizer, 
		A_eq = volume_conservation_A, b_eq = conservation_b,
		A_ub = A_util, b_ub = b_ub,
		method='highs-ds')

	if res.status != 0:
		### infeasible problem, likely no route for some users
		print("Infeasible problem")
		return {
			'solved': False,
		}

	## Distribution is the amount of volume (not percent) placed on each path
	## a path is specified by a <user, popp>
	distribution = res.x
	path_distribution = distribution[sas.n_popps:]

	### now we want to convert this distribution into an advertisement
	dist_by_ug = {ug: {prefix_i:{} for prefix_i in range(max_n_prefixes)} for ug in sas.ugs}
	for (ug,poppi,prefix_i), vol_amt in zip(available_paths, path_distribution):
		dist_by_ug[ug][prefix_i][poppi] = vol_amt
	for ug in dist_by_ug:
		pass


def solve_lp_with_failure_catch(sas, adv, **kwargs):
	### minimizes average latency, but if that fails it instead 
	### minimizes MLU
	ts = time.time()
	verb = True
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
	## 1s -> upper bound on utilization, small numbers -> path distributions
	ALPHA = .001
	weighted_available_latencies = np.array([sas.ug_perfs[ug][sas.popps[poppi]] * sas.ug_to_vol[ug] \
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
	volume_conservation_A = np.zeros((sas.n_ug, sas.n_popps + n_paths))
	conservation_b = sas.ug_vols

	for pli in range(n_paths):
		poppi = available_paths[pli][1]
		ugi = sas.ug_to_ind[available_paths[pli][0]]
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
		print("Infeasible problem")
		return {
			'solved': False,
		}


	# print("A {} s, {} iter, res: {}".format(round(time.time() - ts,2), res.nit,
	# 	np.sum(np.abs(res.con))))

	## Distribution is the amount of volume (not percent) placed on each path
	## a path is specified by a <user, popp>
	distribution = res.x
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
	vols_by_poppi = {poppi:v/float(caps[poppi]) for poppi,v in vols_by_poppi.items()}
	inundated_popps = {poppi:None for poppi,v in vols_by_poppi.items() if v > 1}
	if not res.success:
		print("\n\n\nNo solution in minimizing MLU, here are inundated popps : {}\n\n".format(
			inundated_popps))
	if kwargs.get('verb'):
		print("Inundated popps {}, \n vols by poppi: {}".foramt(inundated_popps, vols_by_poppi))

	inflations, lats_by_ug = [], {}
	all_volume, congested_volume = 0, 0
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
				congested_volume += sas.ug_vols[ugi] * vol
			except KeyError:
				popp = sas.popps[poppi]
				these_lats.append((sas.ug_perfs[ug][popp], vol))
			cum_vol += vol
			all_volume += sas.ug_vols[ugi]
		avg_lat = np.sum([el[0] * el[1] for el in these_lats]) / cum_vol
		lats_by_ug[ug] = avg_lat
		avg_inflation = avg_lat - best_lat
		inflations.append((avg_inflation, sas.ug_to_vol[ug]))
	lats_by_ug_arr = np.zeros((sas.n_ug))
	for ug,lat in lats_by_ug.items():
		lats_by_ug_arr[sas.ug_to_ind[ug]] = lat

	fraction_congested_volume = congested_volume / all_volume

	if verb:
		print("postprocess took {} s".format(round(time.time() - ts,2)))
		ts = time.time()

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


	ugs = sas.ugs
	## get vector of latency of all users to all (available) ingresses
	# First, get winning ingresses from available prefixes and the priority model
	routed_through_ingress,_ = sas.calculate_ground_truth_ingress(adv)
	all_ug_lat_ingresses = {}
	for prefixi in routed_through_ingress:
		for ug in routed_through_ingress[prefixi]:
			poppi = routed_through_ingress[prefixi][ug]
			all_ug_lat_ingresses[ug,poppi] = None
			if kwargs.get('smallverb'):
				if poppi in [3,5] and sas.ug_to_ind[ug] in [51,59]:
					print("UGI {} path to {} available".format(sas.ug_to_ind[ug], poppi))
	available_paths = sorted(list(all_ug_lat_ingresses), key = lambda el : el[0])
	n_paths = len(available_paths)

	weighted_available_latencies = np.array([sas.ug_perfs[ug][sas.popps[poppi]] * sas.ug_to_vol[ug] \
		for ug,poppi in available_paths])
	# if verb:
	# 	available_latencies = np.array([sas.ug_perfs[ug][sas.popps[poppi]] for 
	# 		ug,poppi in available_paths])
	# 	print(adv)
	# 	print(available_paths)
	# 	print(available_latencies)

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
		method='highs-ds')
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
		print(distribution.round(4))

	paths_by_ug = {}
	for (ug,poppi),vol_amt in zip(available_paths, distribution):
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
	# Convert to poppi utilizations
	vols_by_poppi = {poppi:round(v/float(caps[poppi]),2) for poppi,v in vols_by_poppi.items()}
	inundated_popps = {poppi:v for poppi,v in vols_by_poppi.items() if v > 1}
	if not res.success:
		print("\n\n\nNo solution in minimizing MLU, here are inundated popps : {}\n\n".format(
			inundated_popps))

	inflations, lats_by_ug = [], {}
	all_volume, congested_volume = 0, 0
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
				congested_volume += sas.ug_vols[ugi] * vol
			except KeyError:
				popp = sas.popps[poppi]
				these_lats.append((sas.ug_perfs[ug][popp], vol))
			cum_vol += vol
			all_volume += sas.ug_vols[ugi]
		avg_lat = np.sum([el[0] * el[1] for el in these_lats]) / cum_vol
		lats_by_ug[ug] = avg_lat
		avg_inflation = avg_lat - best_lat
		inflations.append((avg_inflation, sas.ug_to_vol[ug]))
	lats_by_ug_arr = np.zeros((sas.n_ug))
	for ug,lat in lats_by_ug.items():
		lats_by_ug_arr[sas.ug_to_ind[ug]] = lat

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
	