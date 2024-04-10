import numpy as np,  scipy, time
from helpers import *
from scipy.sparse import csr_matrix
import gurobipy as gp

def NO_PATH_INGRESS(sas):
	return sas.n_popps

def get_paths_by_ug(sas, adv):
	# First, get winning ingresses from available prefixes and the priority model
	routed_through_ingress, _ = sas.calculate_ground_truth_ingress(adv)

	paths_by_ug = {}
	for prefixi in sorted(routed_through_ingress):
		for ug in sas.whole_deployment_ugs:
			if routed_through_ingress[prefixi].get(ug) is None: continue
			poppi = sas.popp_to_ind[routed_through_ingress[prefixi][ug]]
			try:
				paths_by_ug[ug].append(poppi)
			except KeyError:
				paths_by_ug[ug] = [poppi]
	ugs_with_no_path = get_difference(list(sas.whole_deployment_ugs), list(paths_by_ug))
	for ug in ugs_with_no_path:
		if not sas.simulated:
			print("UG {} has no path, clients: {}".format(ug, sas.ug_to_ip.get(ug)))
	# if len(ugs_with_no_path) > 0:
	# 	print(ugs_with_no_path)
	# 	print(routed_through_ingress)
	# 	exit(0)
	### As an approximation, only consider the best N paths per UG. Otherwise computation is too expensive
	all_ug_lat_ingresses = {}
	N_KEEP = -1
	for ug in sorted(paths_by_ug):
		sorted_options = sorted(set(paths_by_ug[ug]), key = lambda el : sas.whole_deployment_ug_perfs[ug][sas.popps[el]])
		if N_KEEP >= 0:
			keep_options = sorted_options[0:N_KEEP]
		else:
			keep_options = sorted_options
		for poppi in keep_options:
			all_ug_lat_ingresses[ug,poppi] = None
	available_paths = sorted(list(all_ug_lat_ingresses), key = lambda el : el[0])

	for ug in ugs_with_no_path:
		available_paths.append((ug, NO_PATH_INGRESS(sas)))

	return routed_through_ingress, available_paths, paths_by_ug

def solve_lp_with_failure_catch(sas, adv, **kwargs):
	### minimizes average latency, but if that fails it instead 
	### minimizes MLU
	ts = time.time()
	verb = False
	ret_min_latency = solve_lp_assignment(sas, adv, **kwargs)
	if verb:
		print("Trying first without MLU took {} s".format(round(time.time() - ts,2)))
		ts = time.time()
	if ret_min_latency['solved']:
		if kwargs.get('smallverb'):
			print("Solved LP just minimizing latency")
		return ret_min_latency
	elif kwargs.get('smallverb') or verb:
		print("Failed to solve min latency problem.")

	if np.sum(adv.flatten()) == 0:
		print("No solution because no active advertisements")
		return {'solved': False}
	ugs = sas.whole_deployment_ugs

	## get vector of latency of all users to all (available) ingresses
	routed_through_ingress, available_paths, paths_by_ug = get_paths_by_ug(sas, adv)

	n_paths = len(available_paths)
	n_popps = sas.n_popps + 1 ### number of popps + 1 representing a "no route" ingress

	## caps is usualy link capaciites, but then very "large" for users with no route
	caps = np.concatenate([sas.link_capacities_arr.flatten(), np.array([100000])])

	## optimization variable is [Y,v]
	## Y is dummy upper bound variable, v is percent of volume UG places on path
	## 1s -> upper bound on utilization, small numbers -> path distributions
	weighted_available_latencies = np.ones(n_paths)
	for i,(ug,poppi) in enumerate(available_paths):
		if poppi == NO_PATH_INGRESS(sas):
			weighted_available_latencies[i] = NO_ROUTE_LATENCY
		else:
			weighted_available_latencies[i] = sas.whole_deployment_ug_perfs[ug][sas.popps[poppi]]
	dummy_minimizer = np.concatenate([np.ones((n_popps)),ALPHA * weighted_available_latencies])

	### upper bound A for enforcing utilization
	n_entries_util = n_paths + n_popps
	util_data = np.zeros((n_entries_util))
	util_row = np.zeros((n_entries_util))
	util_col = np.zeros((n_entries_util))
	for linkj in range(n_popps):
		util_data[linkj] = -1
		util_row[linkj] = linkj
		util_col[linkj] = linkj
	for i,(ug,poppi) in enumerate(available_paths):
		if poppi == NO_PATH_INGRESS(sas):
			util_data[n_popps+i] = 1 / 1000000 ## very high "capacity" for no path
		else:
			util_data[n_popps+i] = 1 / caps[poppi]
		util_row[n_popps+i] = poppi
		util_col[n_popps+i] = n_popps + i
	A_util = csr_matrix((util_data, (util_row, util_col)), shape=(n_popps, n_popps + n_paths))
	b_ub = np.zeros((n_popps)).flatten()	


	### Set up volume conservation matrix
	n_entries_vol_conservation = n_paths
	vol_conservation_data = np.ones((n_entries_vol_conservation))
	vol_conservation_row = np.zeros((n_entries_vol_conservation))
	vol_conservation_col = np.zeros((n_entries_vol_conservation))

	for pli in range(n_paths):
		poppi = available_paths[pli][1]
		ugi = sas.whole_deployment_ug_to_ind[available_paths[pli][0]]
		vol_conservation_row[pli] = ugi
		vol_conservation_col[pli] = n_popps + pli

	volume_conservation_A = csr_matrix((vol_conservation_data, (vol_conservation_row, vol_conservation_col)), shape=(sas.whole_deployment_n_ug, n_popps + n_paths))
	conservation_b = sas.whole_deployment_ug_vols

	if verb:
		print("Setup took {} s".format(round(time.time() - ts,2)))
		ts = time.time()

	# res = scipy.optimize.linprog(dummy_minimizer, 
	# 	A_eq = volume_conservation_A, b_eq = conservation_b,
	# 	A_ub = A_util, b_ub = b_ub,
	# 	method='highs-ds')

	### Gurobi solve
	model = gp.Model()
	model.Params.LogToConsole = 0
	model.Params.Threads = 1
	x = model.addMVar(n_popps + n_paths, name='volume_each_path', lb=0)
	model.setObjective(x @ dummy_minimizer)
	model.addConstr(A_util @ x <= b_ub)
	model.addConstr(volume_conservation_A @ x == conservation_b)
	model.optimize()


	if verb:
		print("Solving took {} s".format(round(time.time() - ts,2)))
		ts = time.time()

	if model.status != 2:
		### infeasible problem, likely no route for some users
		print("Infeasible problem, exiting")
		exit(0)
		return {
			'solved': False,
		}

	## Distribution is the amount of volume (not percent) placed on each path
	## a path is specified by a <user, popp>
	distribution = x.X
	path_distribution = distribution[n_popps:]

	## Compute paths by ug
	lats_by_ug_arr = np.zeros((sas.whole_deployment_n_ug))
	paths_by_ug = {}
	for (ug,poppi),vol_amt in zip(available_paths, path_distribution):
		if poppi == NO_PATH_INGRESS(sas): 
			lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = NO_ROUTE_LATENCY
			continue # no path
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
	vols_by_poppi = {poppi:v/float(caps[poppi]) for poppi,v in vols_by_poppi.items()}
	inundated_popps = {poppi:None for poppi,v in vols_by_poppi.items() if v > 1}
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
				# print("UG {} experiencing congestion".format(sas.whole_deployment_ugs[ugi]))
				congested_volume += sas.whole_deployment_ug_vols[ugi] * vol
			except KeyError:
				popp = sas.popps[poppi]
				these_lats.append((sas.whole_deployment_ug_perfs[ug][popp], vol))
			cum_vol += vol
			all_volume += sas.whole_deployment_ug_vols[ugi]
		avg_lat = np.sum([el[0] * el[1] for el in these_lats]) / cum_vol
		lats_by_ug[ug] = avg_lat
	for ug,lat in lats_by_ug.items():
		lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = lat


	fraction_congested_volume = congested_volume / all_volume

	if verb:
		print("postprocess took {} s".format(round(time.time() - ts,2)))
		ts = time.time()
		print(lats_by_ug_arr[0])

	return {
		"objective": model.objVal,
		"raw_solution": x.X,
		"paths_by_ug": paths_by_ug,
		"lats_by_ug" : lats_by_ug_arr,
		"solved": model.status,
		"vols_by_poppi": vols_by_poppi,
		"fraction_congested_volume": fraction_congested_volume,
		# "routed_through_ingress": routed_through_ingress,
	}

def solve_lp_assignment(sas, adv, verb=False, **kwargs):
	### Minimizes average latency subject to not inundating a link,
	### but could fail if there's not enough aggregate capacity
	if np.sum(adv.flatten()) == 0:
		return {'solved': False}


	ugs = sas.whole_deployment_ugs

	routed_through_ingress, available_paths, paths_by_ug = get_paths_by_ug(sas, adv)
	n_paths = len(available_paths)
	n_popps = sas.n_popps + 1 ### number of popps + 1 representing a "no route" ingress

	available_latencies = np.ones(n_paths)
	for i,(ug,poppi) in enumerate(available_paths):
		if poppi == NO_PATH_INGRESS(sas):
			available_latencies[i] = NO_ROUTE_LATENCY
		else:
			available_latencies[i] = sas.whole_deployment_ug_perfs[ug][sas.popps[poppi]]
	
	### Set up capacity constraint matrix
	n_entries_cap_constraint = n_paths
	cap_constraint_data = np.ones((n_entries_cap_constraint))
	cap_constraint_row = np.zeros((n_entries_cap_constraint))
	cap_constraint_col = np.zeros((n_entries_cap_constraint))

	### Set up volume conservation matrix
	n_entries_vol_conservation = n_paths
	vol_conservation_data = np.ones((n_entries_vol_conservation))
	vol_conservation_row = np.zeros((n_entries_vol_conservation))
	vol_conservation_col = np.zeros((n_entries_vol_conservation))

	## caps is usualy link capaciites, but then very "large" for users with no route
	caps = np.concatenate([sas.link_capacities_arr.flatten(), np.array([100000])])

	conservation_b = sas.whole_deployment_ug_vols

	for pli in range(n_paths):
		poppi = available_paths[pli][1]
		ugi = sas.whole_deployment_ug_to_ind[available_paths[pli][0]]
		
		cap_constraint_row[pli] = poppi
		cap_constraint_col[pli] = pli

		vol_conservation_row[pli] = ugi
		vol_conservation_col[pli] = pli

	cap_constraint_A = csr_matrix((cap_constraint_data, (cap_constraint_row, cap_constraint_col)), shape=(n_popps, n_paths))
	volume_conservation_A = csr_matrix((vol_conservation_data, (vol_conservation_row, vol_conservation_col)), shape=(sas.whole_deployment_n_ug, n_paths))

	### Solve for volume on each popp,user
	ts = time.time()

	### Gurobi solve
	model = gp.Model()
	model.Params.LogToConsole = 0
	model.Params.Threads = 1
	x = model.addMVar(n_paths, name='volume_each_path', lb=0)
	model.setObjective(x @ available_latencies)
	model.addConstr(cap_constraint_A @ x <= caps)
	model.addConstr(volume_conservation_A @ x == conservation_b)
	model.optimize()

	##### !!!!!!!!!!!!!!!!!!
	## Distribution is the AMOUNT OF VOLUME (NOT PERCENT) placed on each path
	## a path is specified by a <user, popp>
	##### !!!!!!!!!!!!!!!!!!

	if model.status != 2: ## 2 is optimal
		# print("Didnt solve")
		return {'solved': False}
	path_distribution = x.X
	if verb:
		print("Solved distribution without any congestion")

	lats_by_ug_arr = np.zeros((sas.whole_deployment_n_ug))
	paths_by_ug = {}
	for (ug,poppi),vol_amt in zip(available_paths, path_distribution):
		if poppi == NO_PATH_INGRESS(sas): 
			lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = NO_ROUTE_LATENCY
			continue # no path
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
				# print("UG {} experiencing congestion".format(sas.whole_deployment_ugs[ugi]))
				congested_volume += sas.whole_deployment_ug_vols[ugi] * vol
			except KeyError:
				popp = sas.popps[poppi]
				these_lats.append((sas.whole_deployment_ug_perfs[ug][popp], vol))
			cum_vol += vol
			all_volume += sas.whole_deployment_ug_vols[ugi]
		avg_lat = np.sum([el[0] * el[1] for el in these_lats]) / cum_vol
		lats_by_ug[ug] = avg_lat
	for ug,lat in lats_by_ug.items():
		lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = lat

	fraction_congested_volume = congested_volume / all_volume

	return {
		"objective": model.objVal,
		"raw_solution": x.X,
		"paths_by_ug": paths_by_ug,
		"lats_by_ug" : lats_by_ug_arr,
		"solved": model.status,
		"vols_by_poppi": vols_by_poppi,
		"fraction_congested_volume": fraction_congested_volume,
		# "routed_through_ingress": routed_through_ingress,
	}
	