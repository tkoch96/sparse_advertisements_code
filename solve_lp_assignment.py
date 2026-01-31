import numpy as np,  scipy, time
from helpers import *
from scipy.sparse import csr_matrix
import gurobipy as gp
gp.setParam("OutputFlag", 0)

def NO_PATH_INGRESS(sas):
	return sas.n_popps

def get_paths_by_ug(sas, routed_through_ingress):
	## sas is Sparse_Advertisement_Solver (i.e., deployment) object
	## routed_through_ingress is one possible realization of routes. it is a dictionary maping prefixes -> user -> ingress
	## returns available paths which is a list of all (users, ingresses)
	## returns paths_by_ug which is a dictionary mapping ug -> [list of ingresses]

	### Returns structured paths for downstream use
	## availa
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

	return available_paths, paths_by_ug

def _get_paths_by_ug(sas, adv, **kwargs):
	## First, get winning ingresses from available prefixes and the priority model
	## (i.e., assumes that we're doing actual routing)
	routed_through_ingress, _ = sas.calculate_ground_truth_ingress(adv, **kwargs)
	## subroutine to structure these outputs
	available_paths, paths_by_ug = get_paths_by_ug(sas, routed_through_ingress)
	return routed_through_ingress, available_paths, paths_by_ug

def get_obj_fn(model, minimizer_weight, opt_var, obj, n_paths, sas, using_mlu=False):
	# model -> gurobi model
	# minimizer weight -> usually latencies
	# obj -> specify which objective you want
	if obj == 'avg_latency':
		obj_fn = minimizer_weight @ opt_var
		obj_norm = np.sum(sas.whole_deployment_ug_vols)
		model.Params.MIPGap = .01 ## allow a little slack
	elif obj == 'squaring':
		if using_mlu:
			obj_fn = minimizer_weight[0] * opt_var[0] + minimizer_weight[1:] @ (opt_var[1:] * opt_var[1:])
		else:
			obj_fn = minimizer_weight @ (opt_var * opt_var)
		obj_norm = np.sum(sas.whole_deployment_ug_vols * sas.whole_deployment_ug_vols)
	elif obj == 'square_rooting':
		sqrtx = model.addMVar(n_paths, name='sqrtx', lb=0)
		if using_mlu:
			model.addConstr(sqrtx * sqrtx == opt_var[1:])
			obj_fn = minimizer_weight[0] * opt_var[0] + minimizer_weight[1:] @ sqrtx
		else:
			model.addConstr(sqrtx * sqrtx == opt_var)
			obj_fn = minimizer_weight @ sqrtx
		obj_norm = np.sum(np.sqrt(sas.whole_deployment_ug_vols))
		### We need to be lax in our convergence criteria or this just takes
		### way too long 
		model.Params.MIPGap = .05 
	else:
		raise ValueError("Objective {} not implemented in solve_lp_assignment".format(obj))
	return model, opt_var, obj_fn, obj_norm

def solve_joint_latency_bulk_download(sas, routed_through_ingress, obj, **kwargs):
	## minimizes average latency for low latency traffic and (sorta) amount of congested low latency traffic


	avg_latency_ret = solve_generic_lp_with_failure_catch(sas, routed_through_ingress, 'avg_latency')
	if not avg_latency_ret['solved']:
		print("Didn't even solve low latency allocation ... ")
		exit(0)


	available_paths, paths_by_ug = get_paths_by_ug(sas, routed_through_ingress)
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
	model.Params.TimeLimit = 15.0 # seconds, should be approx. double what it takes for a LP
	model.Params.Threads = N_WORKERS_GENERIC
		
	#### Low latency 
	x = avg_latency_ret['raw_solution']
	if len(x) > n_paths: ## cut off the MLU index
		x = x[1:]


	#### Bulk traffic
	b = model.addMVar(n_paths, name='bulk_traffic_volume_each_path', lb=0)
	oversubscribe = cap_constraint_A @ (b + x) - caps
	significances = cap_constraint_A @ x


	bulk_conservation_b = sas.whole_deployment_ug_bulk_vols
	model.addConstr(volume_conservation_A @ b == bulk_conservation_b)
	## another constraint could be like bulk oversubscription is at most N X normal capacity, where N can be 10 or something
	BULK_CAP_LIMIT = 100.0
	model.addConstr(cap_constraint_A @ (b + x) <= BULK_CAP_LIMIT * caps)

	obj_fn = oversubscribe @ significances #+ 100 * oversubscribe @ np.ones(n_popps)
	
	model.setObjective(obj_fn)
	model.optimize()

	##### !!!!!!!!!!!!!!!!!!
	## Distribution is the AMOUNT OF VOLUME (NOT PERCENT) placed on each path
	## a path is specified by a <user, popp>
	##### !!!!!!!!!!!!!!!!!!

	if model.status != 2: ## 2 is optimal
		print("Didnt solve")
		exit(0)
		return {'solved': False}
	low_latency_path_distribution = x
	bulk_path_distribution = b.X
	# print("Solved!")
	# print(x)
	# print(b.X)
	# exit(0)

	# if verb:
	# 	print("Solved distribution without any congestion")

	lats_by_ug_arr = np.zeros((sas.whole_deployment_n_ug))
	paths_by_ug = {}
	vols_by_poppi = {poppi:0 for poppi in range(sas.n_popps)}
	for (ug,poppi),vol_amt in zip(available_paths, low_latency_path_distribution):
		if poppi == NO_PATH_INGRESS(sas): 
			lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = NO_ROUTE_LATENCY
			continue # no path
		if vol_amt > 0:
			ugi = sas.whole_deployment_ug_to_ind[ug]
			vol_pct = vol_amt / sas.whole_deployment_ug_to_vol[ug]
			vols_by_poppi[poppi] += vol_amt
			try:
				paths_by_ug[ugi].append((poppi, vol_pct))
			except KeyError:
				paths_by_ug[ugi] = [(poppi, vol_pct)]

	bulk_lats_by_ug_arr = np.zeros((sas.whole_deployment_n_ug)) ## assumes we're assigning bulk traffic && low latency traffic
	bulk_paths_by_ug = {}
	bulk_vols_by_poppi = {poppi:vols_by_poppi[poppi] for poppi in range(sas.n_popps)} ## start with the low-latency allocation
	for (ug,poppi),vol_amt in zip(available_paths, bulk_path_distribution):
		if poppi == NO_PATH_INGRESS(sas): 
			bulk_lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = NO_ROUTE_LATENCY
			continue # no path
		if vol_amt > 0:
			ugi = sas.whole_deployment_ug_to_ind[ug]
			vol_pct = vol_amt / sas.whole_deployment_ug_to_bulk_vol[ug]
			bulk_vols_by_poppi[poppi] += vol_amt
			try:
				bulk_paths_by_ug[ugi].append((poppi, vol_pct))
			except KeyError:
				bulk_paths_by_ug[ugi] = [(poppi, vol_pct)]

	# Convert to poppi utilizations (without bulk traffic)
	vols_by_poppi = {poppi:round(v/float(caps[poppi]),2) for poppi,v in vols_by_poppi.items()}
	# Convert to poppi utilizations (with bulk traffic)
	bulk_vols_by_poppi = {poppi:round(v/float(caps[poppi]),2) for poppi,v in bulk_vols_by_poppi.items()}

	lats_by_ug = {}
	all_volume, congested_volume = 0, 0
	for ugi, pathvols in paths_by_ug.items():
		ug = sas.whole_deployment_ugs[ugi]
		these_lats = []
		cum_vol = 0
		for poppi,vol in pathvols:
			popp = sas.popps[poppi]
			if vols_by_poppi.get(poppi, 0) > 1:
				these_lats.append((NO_ROUTE_LATENCY, vol))
				congested_volume += vol
			else:
				these_lats.append((sas.whole_deployment_ug_perfs[ug][popp], vol))
			cum_vol += vol
			all_volume += sas.whole_deployment_ug_vols[ugi]
		avg_lat = np.sum([el[0] * el[1] for el in these_lats]) / cum_vol
		lats_by_ug[ug] = avg_lat
	for ug,lat in lats_by_ug.items():
		lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = lat

	bulk_lats_by_ug = {} ## latency-sensitive latency, except preload all the links with bulk traffic
	all_volume_with_bulk, congested_volume_with_bulk = 0, 0
	for ugi, pathvols in paths_by_ug.items():
		ug = sas.whole_deployment_ugs[ugi]
		these_lats = []
		cum_vol = 0
		for poppi,vol in pathvols:
			popp = sas.popps[poppi]
			if bulk_vols_by_poppi.get(poppi,0) > 1:
				these_lats.append((NO_ROUTE_LATENCY, sas.whole_deployment_ug_vols[ugi] * vol))
				congested_volume_with_bulk += (sas.whole_deployment_ug_vols[ugi] * vol)
			else:				
				these_lats.append((sas.whole_deployment_ug_perfs[ug][popp], sas.whole_deployment_ug_vols[ugi] * vol))
			cum_vol += (sas.whole_deployment_ug_vols[ugi] * vol)
			all_volume_with_bulk += (sas.whole_deployment_ug_vols[ugi] * vol)
		avg_lat = np.sum([el[0] * el[1] for el in these_lats]) / cum_vol
		bulk_lats_by_ug[ug] = avg_lat
	for ug,lat in bulk_lats_by_ug.items():
		bulk_lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = lat

	fraction_congested_volume = congested_volume / all_volume
	fraction_congested_volume_with_bulk = congested_volume_with_bulk / all_volume_with_bulk


	## Actual objective value incorporates both average latency and bulk traffic
	# multiply by -1 because of the way I legacy did the code
	# objective_val = -1 * (np.dot(available_latencies,x) / np.sum(sas.whole_deployment_ug_vols) + ALPHA_BULK * model.objVal / (np.sum(sas.whole_deployment_ug_vols) * (1 + BULK_MULTIPLIER)))

	## we can return whatever nonsense we want to
	# print("{} {}".format(np.dot(available_latencies,x) / np.sum(sas.whole_deployment_ug_vols), congested_volume_with_bulk,ALPHA_BULK * congested_volume_with_bulk / np.sum(sas.whole_deployment_ug_vols)))

	random_ug = sas.whole_deployment_ugs[0]
	# print("{} {} {} {} ".format(ug,sas.whole_deployment_ug_to_vol[random_ug],
	# 	sas.whole_deployment_ug_to_bulk_vol[random_ug] / sas.whole_deployment_ug_to_vol[random_ug],
	# 	fraction_congested_volume_with_bulk))


	if ALPHA_BULK > 1:
		objective_val = -1 * (1.0 / ALPHA_BULK * np.dot(available_latencies,x) / np.sum(sas.whole_deployment_ug_vols) + congested_volume_with_bulk / np.sum(sas.whole_deployment_ug_vols))
	else:
		objective_val = -1 * (np.dot(available_latencies,x) / np.sum(sas.whole_deployment_ug_vols) + ALPHA_BULK * congested_volume_with_bulk / np.sum(sas.whole_deployment_ug_vols))

	return {
		"objective": objective_val,
		"solved": model.status,
		"raw_low_latency_solution": x,
		"raw_bulk_traffic_solution": b.X,
		"paths_by_ug": paths_by_ug,
		"lats_by_ug" : lats_by_ug_arr,
		"vols_by_poppi": vols_by_poppi,
		"fraction_congested_volume": fraction_congested_volume,
		"bulk_paths_by_ug": bulk_paths_by_ug,
		"bulk_lats_by_ug" : bulk_lats_by_ug_arr,
		"bulk_vols_by_poppi": bulk_vols_by_poppi,
		"fraction_congested_volume_with_bulk": fraction_congested_volume_with_bulk,
	}

generic_lp_functions = {
	'joint_latency_bulk_download': solve_joint_latency_bulk_download,
}

def solve_generic_lp_with_failure_catch(sas, routed_through_ingress, obj, **kwargs):
	### Minmizes f(w) subject to capacity and volume constraints
	### w is the amount of volume to place on each path where a path is a <user, routed ingress>

	try:
		return generic_lp_functions[obj](sas, routed_through_ingress, obj, **kwargs)
	except KeyError:
		pass


	verb = False
	ret = solve_generic_lp(sas, routed_through_ingress, obj, **kwargs)
	if ret['solved']:
		if kwargs.get('smallverb') or verb:
			print("Solved Generic LP without MLU")
		return ret
	elif kwargs.get('smallverb') or verb:
		print("Failed to solve non-MLU problem")

	ugs = sas.whole_deployment_ugs
	available_paths, paths_by_ug = get_paths_by_ug(sas, routed_through_ingress)

	n_paths = len(available_paths)
	n_popps = sas.n_popps + 1 ### number of popps + 1 representing a "no route" ingress

	## caps is usualy link capaciites, but then very "large" for users with no route
	caps = np.concatenate([sas.link_capacities_arr.flatten(), np.array([100000])]).flatten()

	### upper bound A for enforcing utilization
	n_entries_util = n_popps + n_paths
	util_data = np.zeros((n_entries_util))
	util_row = np.zeros((n_entries_util))
	util_col = np.zeros((n_entries_util))
	# ### Indicator for mapping paths to links
	#### (For max link usage, maybe useful in the future but maybe not)
	# vol_to_link_data = np.zeros((n_entries_util))
	# vol_to_link_row = np.zeros((n_entries_util))
	# vol_to_link_col = np.zeros((n_entries_util))
	for i in range(n_popps): ## set the entire first column to -1
		util_data[i] = -1 
		util_row[i] = i
		util_col[i] = 0

	for i,(ug,poppi) in enumerate(available_paths):
		if poppi == NO_PATH_INGRESS(sas):
			util_data[n_popps+i] = 1 / 1000000.0 ## very high "capacity" for no path
		else:
			util_data[n_popps+i] = 1 / caps[poppi]
		util_row[n_popps+i] = poppi
		util_col[n_popps+i] = 1 + i

		# # indicator that path i is for ingress poppi
		# vol_to_link_data[i] = 1.0
		# vol_to_link_row[i] = poppi
		# vol_to_link_col[i] = i

	A_util = csr_matrix((util_data, (util_row, util_col)), shape=(n_popps, 1+n_paths))
	b_ub = np.zeros((n_popps)).flatten()	
	
	# A_volume_to_link = csr_matrix((vol_to_link_data, (vol_to_link_row, vol_to_link_col)), shape=(n_popps, n_paths))

	### Set up volume conservation matrix
	n_entries_vol_conservation = 1+n_paths
	vol_conservation_data = np.zeros((n_entries_vol_conservation))
	vol_conservation_row = np.zeros((n_entries_vol_conservation))
	vol_conservation_col = np.zeros((n_entries_vol_conservation))

	for pli in range(n_paths):
		ugi = sas.whole_deployment_ug_to_ind[available_paths[pli][0]]
		vol_conservation_row[1+pli] = ugi
		vol_conservation_col[1+pli] = 1 + pli
		vol_conservation_data[1+pli] = 1

	volume_conservation_A = csr_matrix((vol_conservation_data, (vol_conservation_row, vol_conservation_col)), shape=(sas.whole_deployment_n_ug, n_entries_vol_conservation))
	conservation_b = sas.whole_deployment_ug_vols.flatten()


	## optimization variable is [Y,v]
	## Y is dummy upper bound variable, v is percent of volume UG places on path
	## 1 -> upper bound on utilization, small numbers -> path distributions
	weighted_available_latencies = np.ones(n_paths)
	for i,(ug,poppi) in enumerate(available_paths):
		if poppi == NO_PATH_INGRESS(sas):
			weighted_available_latencies[i] = NO_ROUTE_LATENCY
		else:
			weighted_available_latencies[i] = sas.whole_deployment_ug_perfs[ug][sas.popps[poppi]]
	## ALPHA defined in constants ;; tradeoff between minimizing MLU and minimizing latency
	dummy_minimizer = np.concatenate([np.array([1.0 / ALPHA]),weighted_available_latencies]).flatten()


	### Gurobi solve
	model = gp.Model()
	model.Params.LogToConsole = 0
	model.Params.Threads = N_WORKERS_GENERIC
	model.Params.TimeLimit = 3.0 # seconds, should be approx. double what it takes for a LP
	x = model.addMVar(1 + n_paths, name='volume_each_path', lb=0)

	model, x, obj_fn, obj_norm = get_obj_fn(model, dummy_minimizer, x, obj, n_paths, sas, using_mlu=True)

	model.setObjective(obj_fn)
	model.addConstr(A_util @ x <= b_ub)
	model.addConstr(volume_conservation_A @ x == conservation_b)
	model.optimize()

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
	path_distribution = distribution[1:]

	## Compute paths by ug
	lats_by_ug_arr = np.zeros((sas.whole_deployment_n_ug))
	paths_by_ug = {}
	vols_by_poppi = {poppi:0 for poppi in range(sas.n_popps)}
	for (ug,poppi),vol_amt in zip(available_paths, path_distribution):
		if poppi == NO_PATH_INGRESS(sas): 
			lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = NO_ROUTE_LATENCY
			continue # no path
		if vol_amt > 0:
			ugi = sas.whole_deployment_ug_to_ind[ug]
			vol_pct = vol_amt / sas.whole_deployment_ug_to_vol[ug]
			vols_by_poppi[poppi] += vol_amt
			try:
				paths_by_ug[ugi].append((poppi, vol_pct))
			except KeyError:
				paths_by_ug[ugi] = [(poppi, vol_pct)]

	vols_by_poppi = {poppi:v/float(caps[poppi]) for poppi,v in vols_by_poppi.items()}
	inundated_popps = {poppi:None for poppi,v in vols_by_poppi.items() if v > 1}
	# print("Inundated popps {} ({}), \n vols by poppi: {}".format(inundated_popps, list([sas.popps[poppi] for poppi in inundated_popps]), vols_by_poppi))

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
				# if not sas.simulated:
				# 	print("In min MLU, UG {} experiencing congestion".format(sas.whole_deployment_ugs[ugi]))
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
		"objective": -1 * model.objVal / obj_norm,
		"raw_solution": x.X,
		"paths_by_ug": paths_by_ug,
		"lats_by_ug" : lats_by_ug_arr,
		"solved": model.status,
		"vols_by_poppi": vols_by_poppi,
		"fraction_congested_volume": fraction_congested_volume,
		# "routed_through_ingress": routed_through_ingress,
	}

def solve_generic_lp(sas, routed_through_ingress, obj, **kwargs):
	### Minimizes average latency subject to not inundating a link,
	### but could fail if there's not enough aggregate capacity

	available_paths, paths_by_ug = get_paths_by_ug(sas, routed_through_ingress)
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
	model.Params.TimeLimit = 15.0 # seconds, should be approx. double what it takes for a LP
	model.Params.Threads = N_WORKERS_GENERIC
	
	x = model.addMVar(n_paths, name='volume_each_path', lb=0)
	model, x, obj_fn, obj_norm = get_obj_fn(model, available_latencies, x, obj, n_paths, sas, using_mlu=False)

	model.addConstr(cap_constraint_A @ x <= caps)
	model.addConstr(volume_conservation_A @ x == conservation_b)
	model.setObjective(obj_fn)
	model.optimize()


	##### !!!!!!!!!!!!!!!!!!
	## Distribution is the AMOUNT OF VOLUME (NOT PERCENT) placed on each path
	## a path is specified by a <user, popp>
	##### !!!!!!!!!!!!!!!!!!

	if model.status != 2: ## 2 is optimal
		# print("Didnt solve")
		return {'solved': False}
	path_distribution = x.X
	# if verb:
	# 	print("Solved distribution without any congestion")

	lats_by_ug_arr = np.zeros((sas.whole_deployment_n_ug))
	paths_by_ug = {}
	vols_by_poppi = {poppi:0 for poppi in range(sas.n_popps)}
	for (ug,poppi),vol_amt in zip(available_paths, path_distribution):
		if poppi == NO_PATH_INGRESS(sas): 
			lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = NO_ROUTE_LATENCY
			continue # no path
		if vol_amt > 0:
			ugi = sas.whole_deployment_ug_to_ind[ug]
			vol_pct = vol_amt / sas.whole_deployment_ug_to_vol[ug]
			vols_by_poppi[poppi] += vol_amt
			try:
				paths_by_ug[ugi].append((poppi, vol_pct))
			except KeyError:
				paths_by_ug[ugi] = [(poppi, vol_pct)]

	# Convert to poppi utilizations
	vols_by_poppi = {poppi:round(v/float(caps[poppi]),2) for poppi,v in vols_by_poppi.items()}

	lats_by_ug = {}
	all_volume, congested_volume = 0, 0
	for ugi, pathvols in paths_by_ug.items():
		ug = sas.whole_deployment_ugs[ugi]
		these_lats = []
		cum_vol = 0
		for poppi,vol in pathvols:
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
		"objective": -1 * model.objVal / obj_norm,
		"raw_solution": x.X,
		"paths_by_ug": paths_by_ug,
		"lats_by_ug" : lats_by_ug_arr,
		"available_paths": available_paths,
		"solved": model.status,
		"vols_by_poppi": vols_by_poppi,
		"fraction_congested_volume": fraction_congested_volume,
	}

def solve_lp_with_failure_catch(sas, adv, **kwargs):
	### minimizes average latency, but if that fails it instead 
	### minimizes MLU
	verb = False
	ret_min_latency = solve_lp_assignment(sas, adv, **kwargs)
	if ret_min_latency['solved']:
		if kwargs.get('smallverb') or verb:
			print("Solved LP just minimizing latency")
		return ret_min_latency
	elif kwargs.get('smallverb') or verb:
		print("Failed to solve min latency problem.")

	if np.sum(adv.flatten()) == 0:
		print("No solution because no active advertisements")
		return {'solved': False}
	ugs = sas.whole_deployment_ugs

	## get vector of latency of all users to all (available) ingresses
	routed_through_ingress, available_paths, paths_by_ug = _get_paths_by_ug(sas, adv, **kwargs)
	# if not sas.simulated:
	# 	print(paths_by_ug)

	n_paths = len(available_paths)
	n_popps = sas.n_popps + 1 ### number of popps + 1 representing a "no route" ingress

	## caps is usualy link capaciites, but then very "large" for users with no route
	caps = np.concatenate([sas.link_capacities_arr.flatten(), np.array([100000])]).flatten()

	## optimization variable is [Y,v]
	## Y is dummy upper bound variable, v is percent of volume UG places on path
	## 1 -> upper bound on utilization, small numbers -> path distributions
	weighted_available_latencies = np.ones(n_paths)
	for i,(ug,poppi) in enumerate(available_paths):
		if poppi == NO_PATH_INGRESS(sas):
			weighted_available_latencies[i] = NO_ROUTE_LATENCY
		else:
			weighted_available_latencies[i] = sas.whole_deployment_ug_perfs[ug][sas.popps[poppi]]
	## ALPHA defined in constants ;; tradeoff between minimizing MLU and minimizing latency
	dummy_minimizer = np.concatenate([np.array([1]), ALPHA * weighted_available_latencies]).flatten()

	### upper bound A for enforcing utilization
	n_entries_util = n_popps + n_paths
	util_data = np.zeros((n_entries_util))
	util_row = np.zeros((n_entries_util))
	util_col = np.zeros((n_entries_util))
	for i in range(n_popps): ## set the entire first column to -1
		util_data[i] = -1 
		util_row[i] = i
		util_col[i] = 0
	for i,(ug,poppi) in enumerate(available_paths):
		if poppi == NO_PATH_INGRESS(sas):
			util_data[n_popps+i] = 1 / 1000000.0 ## very high "capacity" for no path
		else:
			util_data[n_popps+i] = 1 / caps[poppi]
		util_row[n_popps+i] = poppi
		util_col[n_popps+i] = 1 + i
	A_util = csr_matrix((util_data, (util_row, util_col)), shape=(n_popps, 1+n_paths))
	b_ub = np.zeros((n_popps)).flatten()	


	### Set up volume conservation matrix
	n_entries_vol_conservation = 1+n_paths
	vol_conservation_data = np.zeros((n_entries_vol_conservation))
	vol_conservation_row = np.zeros((n_entries_vol_conservation))
	vol_conservation_col = np.zeros((n_entries_vol_conservation))

	for pli in range(n_paths):
		ugi = sas.whole_deployment_ug_to_ind[available_paths[pli][0]]
		vol_conservation_row[1+pli] = ugi
		vol_conservation_col[1+pli] = 1 + pli
		vol_conservation_data[1+pli] = 1

	volume_conservation_A = csr_matrix((vol_conservation_data, (vol_conservation_row, vol_conservation_col)), shape=(sas.whole_deployment_n_ug, n_entries_vol_conservation))
	conservation_b = sas.whole_deployment_ug_vols.flatten()

	# res = scipy.optimize.linprog(dummy_minimizer, 
	# 	A_eq = volume_conservation_A, b_eq = conservation_b,
	# 	A_ub = A_util, b_ub = b_ub,
	# 	method='highs-ds')

	### Gurobi solve
	model = gp.Model()
	model.Params.LogToConsole = 0
	model.Params.Threads = 1
	x = model.addMVar(1 + n_paths, name='volume_each_path', lb=0)
	model.setObjective(x @ dummy_minimizer)
	model.addConstr(A_util @ x <= b_ub)
	model.addConstr(volume_conservation_A @ x == conservation_b)
	model.optimize()


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
	path_distribution = distribution[1:]

	## Compute paths by ug
	lats_by_ug_arr = np.zeros((sas.whole_deployment_n_ug))
	paths_by_ug = {}
	vols_by_poppi = {poppi:0 for poppi in range(sas.n_popps)}
	for (ug,poppi),vol_amt in zip(available_paths, path_distribution):
		if poppi == NO_PATH_INGRESS(sas): 
			lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = NO_ROUTE_LATENCY
			continue # no path
		if vol_amt > 0:
			ugi = sas.whole_deployment_ug_to_ind[ug]
			vol_pct = vol_amt / sas.whole_deployment_ug_to_vol[ug]
			vols_by_poppi[poppi] += vol_amt
			try:
				paths_by_ug[ugi].append((poppi, vol_pct))
			except KeyError:
				paths_by_ug[ugi] = [(poppi, vol_pct)]

	vols_by_poppi = {poppi:v/float(caps[poppi]) for poppi,v in vols_by_poppi.items()}
	inundated_popps = {poppi:None for poppi,v in vols_by_poppi.items() if v > 1}
	# print("Inundated popps {} ({}), \n vols by poppi: {}".format(inundated_popps, list([sas.popps[poppi] for poppi in inundated_popps]), vols_by_poppi))

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
				# if not sas.simulated:
				# 	print("In min MLU, UG {} experiencing congestion".format(sas.whole_deployment_ugs[ugi]))
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

	obj_norm = np.sum(np.sqrt(sas.whole_deployment_ug_vols))

	return {
		"objective": -1 * model.objVal / obj_norm,
		"raw_solution": x.X,
		"paths_by_ug": paths_by_ug,
		"available_paths": available_paths,
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

	routed_through_ingress, available_paths, paths_by_ug = _get_paths_by_ug(sas, adv, **kwargs)
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
	# if verb:
	# 	print("Solved distribution without any congestion")

	lats_by_ug_arr = np.zeros((sas.whole_deployment_n_ug))
	paths_by_ug = {}
	vols_by_poppi = {poppi:0 for poppi in range(sas.n_popps)}
	for (ug,poppi),vol_amt in zip(available_paths, path_distribution):
		if poppi == NO_PATH_INGRESS(sas): 
			lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = NO_ROUTE_LATENCY
			continue # no path
		if vol_amt > 0:
			ugi = sas.whole_deployment_ug_to_ind[ug]
			vol_pct = vol_amt / sas.whole_deployment_ug_to_vol[ug]
			vols_by_poppi[poppi] += vol_amt
			try:
				paths_by_ug[ugi].append((poppi, vol_pct))
			except KeyError:
				paths_by_ug[ugi] = [(poppi, vol_pct)]

	# Convert to poppi utilizations
	vols_by_poppi = {poppi:round(v/float(caps[poppi]),2) for poppi,v in vols_by_poppi.items()}

	#### Inundated popps here happen due to numerical precision errors, not a big deal
	# inundated_popps = {poppi:v for poppi,v in vols_by_poppi.items() if v > 1}
	# if len(inundated_popps) > 0:
	# 	print("Weird that this is happening. Inundated popps {} ({}), \n caps/vols by poppi: {} // {}".format(inundated_popps, list([sas.popps[poppi] for poppi in inundated_popps]),
	# 		list([caps[poppi] for poppi in inundated_popps]), vols_by_poppi))
	# 	for (ug,poppi),vol_amt in zip(available_paths, path_distribution):
	# 		if poppi == NO_PATH_INGRESS(sas): 
	# 			continue # no path
	# 		if poppi not in inundated_popps: continue
	# 		if vol_amt > 0:
	# 			ugi = sas.whole_deployment_ug_to_ind[ug]
	# 			vol_pct = vol_amt / sas.whole_deployment_ug_to_vol[ug]
	# 			try:
	# 				paths_by_ug[ugi].append((poppi, vol_pct))
	# 			except KeyError:
	# 				paths_by_ug[ugi] = [(poppi, vol_pct)]
	# 		print("{} {} {}".format(ugi, sas.whole_deployment_ug_to_vol[ug], paths_by_ug[ugi]))

	lats_by_ug = {}
	all_volume, congested_volume = 0, 0
	for ugi, pathvols in paths_by_ug.items():
		ug = sas.whole_deployment_ugs[ugi]
		these_lats = []
		cum_vol = 0
		for poppi,vol in pathvols:
			popp = sas.popps[poppi]
			these_lats.append((sas.whole_deployment_ug_perfs[ug][popp], vol))
			cum_vol += vol
			all_volume += sas.whole_deployment_ug_vols[ugi]
		avg_lat = np.sum([el[0] * el[1] for el in these_lats]) / cum_vol
		lats_by_ug[ug] = avg_lat
	for ug,lat in lats_by_ug.items():
		lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = lat

	fraction_congested_volume = congested_volume / all_volume

	obj_norm = np.sum(sas.whole_deployment_ug_vols)

	return {
		"objective": -1 * model.objVal / obj_norm,
		"raw_solution": x.X,
		"available_latencies": available_latencies,
		"available_paths": available_paths,
		"paths_by_ug": paths_by_ug,
		"lats_by_ug" : lats_by_ug_arr,
		"solved": model.status,
		"vols_by_poppi": vols_by_poppi,
		"fraction_congested_volume": fraction_congested_volume,
		# "routed_through_ingress": routed_through_ingress,
	}
