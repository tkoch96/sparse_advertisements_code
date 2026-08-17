"""LP objective implementations (Gurobi).

All objectives SCULPTOR optimizes against are implemented here as
`solve_lp_assignment_<name>(sas, routed_through_ingress, obj, **kwargs)`
functions. The active set is registered in `generic_lp_functions` at the
bottom of the file:

  - avg_latency        baseline; minimize traffic-weighted avg user latency
  - per_site_cost      avg_latency + alpha * sum(active site cost)
  - joint_priority     HPrio LP solved first (strict-priority semantics),
                       bulk traffic fills in around it
  - site_failure       (1-β)·steady avg_latency + β·mean over per-PoP
                       failures with user→prefix mapping frozen to
                       steady-state (no-DNS-update semantic)

Each function returns a dict with at least:
  objective     final LP value (Gurobi's ObjVal)
  solved        Gurobi status string ('optimal', 'infeasible', ...)
  paths_by_ug   {ug_index: [(poppi, vol_pct), ...]} — how each UG routes
  lats_by_ug    np.ndarray of per-UG latencies (use NO_ROUTE_LATENCY for
                users with no feasible path)

`solve_generic_lp_with_failure_catch` (top of file) is the public wrapper
that all callers use. It dispatches via `generic_lp_functions[obj]` and
catches `GurobiError` to return a no-route sentinel rather than crashing
the training loop on transient infeasibility.

`solve_generic_lp_persistent` (in path_distribution_computer.py) is the
hot-path version used by the gradient probes — it reuses a persistent
Gurobi model rather than rebuilding from scratch for each call.

Adding a new objective: see README.md "Adding new things" → "A new
objective function".
"""
import os as _os
import numpy as np,  scipy, time, math
from helpers import *
from scipy.sparse import csr_matrix
import gpshim as gp  # gurobipy-subset facade; SCULPTOR_LP_BACKEND=gurobi(default)|highs
gp.setParam("OutputFlag", 0)

def _apply_capacity_headroom(arr, sas=None):
	"""Multiply capacities by (1 - SCULPTOR_CAPACITY_HEADROOM env var, default 0).
	Used to leave headroom in LP capacity constraints so that a single popp
	failure can be absorbed without re-running the LP. When set >0 this replaces
	the SGD-based resilience benefit gradient entirely (both the gradient and
	the value short-circuit automatically; see sparse_advertisements_v3.py).

	Only applies when sas._in_training is True (set/cleared in
	Sparse_Advertisement_Solver.solve around the gradient loop) -- otherwise the
	eval phase would solve every solution's LP under reduced caps, shifting the
	"optimal" reference for ALL solutions (sparse, painter, anyopt, ...).
	"""
	if not getattr(sas, '_in_training', False):
		return arr
	h = float(os.environ.get('SCULPTOR_CAPACITY_HEADROOM', '0'))
	return arr * (1.0 - h)

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
	if obj == 'avg_latency' or obj == "per_site_cost":
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

def obj_round(v):
	"""SCULPTOR_OBJ_ROUND=<d>: quantize objective scalars to d decimals
	(Tom 2026-08-17). Purpose: solver-jitter falsification test -- LP
	solutions among degenerate optima differ between engines by <1e-9 in
	objective but perturb solution-derived scalars; rounding every
	objective the algorithm consumes bounds that channel below the
	quantum so tie-break jitter cannot flip decisions. Unset = off."""
	d = os.environ.get('SCULPTOR_OBJ_ROUND')
	if d is None or v is None:
		return v
	try:
		return round(float(v), int(d))
	except (TypeError, ValueError):
		return v


def _soft_bounded_objective(sas, lats_by_ug_arr, fraction_congested_volume,
		legacy_objective):
	"""Congestion-aware SOFT BOUNDED scalar (Tom 2026-08-14; see
	path_distribution_computer): -(avg routed latency + SOFT_CONG_PENALTY *
	frac(congested+noroute)). ONE helper for every objective return site --
	the 08-13 fix patched two MLU fallbacks inline and MISSED the primary
	min-latency path (74%-congested adv scored ~4ms, below the opp floor).

	BAD volume is derived FROM lats_by_ug_arr, not from the caller's
	congestion accounting: the sentinel marks BOTH congested and NO-ROUTE
	volume, and fraction_congested_volume misses the no-route class --
	stranded volume then leaks sentinel-scale into the "routed" average
	(opp-under-failure scored ~300 on scenarios stranding 0.1% of volume;
	caught by the dash refs 2026-08-14 late). Per-UG bad weight =
	clip((lat-200)/(NO_ROUTE-200), 0, 1): exact 1 for pure-sentinel UGs,
	~0 for real latencies, proportional for blended-average UGs.
	fraction_congested_volume is kept in the signature for reference only.
	SCULPTOR_CONGESTION_AWARE_OBJ=0 restores legacy."""
	if _os.environ.get('SCULPTOR_CONGESTION_AWARE_OBJ', '1') == '0':
		return obj_round(legacy_objective)
	_soft_P = float(_os.environ.get('SCULPTOR_SOFT_CONG_PENALTY', '50'))
	vols = np.asarray(sas.whole_deployment_ug_vols, dtype=float)
	lats = np.asarray(lats_by_ug_arr, dtype=float).flatten()
	_tv = float(np.sum(vols))
	_S = float(np.sum(lats * vols))
	_bad_w = np.clip((lats - 200.0) / max(NO_ROUTE_LATENCY - 200.0, 1.0),
					 0.0, 1.0)
	_B = float(np.sum(_bad_w * vols))
	_R = max(0.0, _S - NO_ROUTE_LATENCY * _B)
	_routed_v = max(_tv - _B, 1e-9)
	return obj_round(-1 * (_R / _routed_v + _soft_P * _B / _tv))


def _is_avg_latency_obj(obj):
	# True when the LP is solving plain avg_latency -- the ONLY case where
	# the soft-bounded latency scalar may REPLACE the model objective.
	# Hard objectives (frac_beyond, MLU, priority, ...) keep their own
	# model.objVal: overriding them with a latency scalar destroys their
	# semantics (caught on the hardB3v2 smoke, 2026-08-14 late).
	name = getattr(obj, 'obj', obj)
	return name == 'avg_latency'


def solve_joint_latency_bulk_download(sas, routed_through_ingress, obj, **kwargs):
	## minimizes average latency for low latency traffic and (sorta) amount of congested low latency traffic


	# no_persistent: this function consumes avg_latency_ret['raw_solution'],
	# a path-aligned array only the NON-persistent LP returns; the worker's
	# persistent ret has a different shape (MLinExpr+dict TypeError,
	# 2026-08-14 -- first time this objective ran inside a Ray worker).
	avg_latency_ret = solve_generic_lp_with_failure_catch(sas, routed_through_ingress, 'avg_latency', no_persistent=True)
	if not avg_latency_ret['solved']:
		print("Didn't even solve low latency allocation ... ")
		# was exit(0): process death with rc=0 (the silent-death pattern);
		# return unsolved so callers can handle it (2026-08-14)
		return {'solved': False, 'objective': None}


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
	caps = np.concatenate([_apply_capacity_headroom(sas.link_capacities_arr.flatten(), sas), np.array([100000])])


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
	## BULK_CAP_LIMIT = 3.0 # sigcomm 2025 value
	## BULK_CAP_LIMIT = 100.0 # historical default in this codebase (labeled "temporary to debug" but
	## existing priority experiments produced their results under this value, so it is preserved as
	## the default to avoid invalidating prior results. Override via kwargs.get('bulk_cap_limit', ...)
	## from the objective spec.
	bulk_cap_limit = kwargs.get('bulk_cap_limit', 100.0)
	model.addConstr(cap_constraint_A @ (b + x) <= bulk_cap_limit * caps)

	obj_fn = oversubscribe @ significances #+ 100 * oversubscribe @ np.ones(n_popps)
	
	model.setObjective(obj_fn)
	model.optimize()

	##### !!!!!!!!!!!!!!!!!!
	## Distribution is the AMOUNT OF VOLUME (NOT PERCENT) placed on each path
	## a path is specified by a <user, popp>
	##### !!!!!!!!!!!!!!!!!!

	if model.status != 2: ## 2 is optimal
		print("Didnt solve")
		# was exit(0) followed by an unreachable return (2026-08-14)
		return {'solved': False, 'objective': None}
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


	if (_os.environ.get('SCULPTOR_CONGESTION_AWARE_OBJ', '1') != '0'
			and _is_avg_latency_obj(obj)):
		objective_val = _soft_bounded_objective(
			sas, lats_by_ug_arr, fraction_congested_volume,
			-1 * (np.dot(available_latencies,x) / np.sum(sas.whole_deployment_ug_vols) + ALPHA_BULK * congested_volume_with_bulk / np.sum(sas.whole_deployment_ug_vols)))
	elif ALPHA_BULK > 1:
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

def solve_lp_assignment_with_site_cost_with_failure_catch(sas, routed_through_ingress, obj, site_cost_alpha=DEFAULT_SITE_COST, **kwargs):
	### Minimizes f(w) subject to capacity and volume constraints
	### w is the amount of volume to place on each path where a path is a <user, routed ingress>
	### Incorporates Site Cost into the minimization objective.

	verb = False
	# Try solving strictly first
	ret = solve_lp_assignment_with_site_cost(sas, routed_through_ingress, obj, site_cost_alpha=site_cost_alpha, **kwargs)
	if ret['solved']:
		if kwargs.get('smallverb') or verb:
			print("Solved Generic LP without MLU")
		return ret
	elif kwargs.get('smallverb') or verb:
		print("Failed to solve non-MLU problem")

	# --- Fallback: MLU Minimization ---
	
	ugs = sas.whole_deployment_ugs
	available_paths, paths_by_ug = get_paths_by_ug(sas, routed_through_ingress)

	n_paths = len(available_paths)
	n_popps = sas.n_popps + 1 

	## caps is usually link capacities, but then very "large" for users with no route
	caps = np.concatenate([_apply_capacity_headroom(sas.link_capacities_arr.flatten(), sas), np.array([100000])]).flatten()

	### upper bound A for enforcing utilization
	n_entries_util = n_popps + n_paths
	util_data = np.zeros((n_entries_util))
	util_row = np.zeros((n_entries_util))
	util_col = np.zeros((n_entries_util))

	for i in range(n_popps): ## set the entire first column to -1 (Variable Y)
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

	## optimization variable is [Y,v]
	## Y is dummy upper bound variable, v is percent of volume UG places on path
	
	# --- COMBINED OBJECTIVE CALCULATION ---
	# Weight = Latency + (site_cost_alpha * Site_Cost)
	weighted_available_metrics = np.ones(n_paths)
	for i,(ug,poppi) in enumerate(available_paths):
		if poppi == NO_PATH_INGRESS(sas):
			# No route: penalty latency, 0 site cost
			weighted_available_metrics[i] = NO_ROUTE_LATENCY
		else:
			lat = sas.whole_deployment_ug_perfs[ug][sas.popps[poppi]]
			pop, _ = sas.popps[poppi]
			site_cost = sas.site_costs[pop]
			weighted_available_metrics[i] = lat + (site_cost_alpha * site_cost)
			
	## ALPHA defined in constants ;; tradeoff between minimizing MLU and minimizing (latency + cost)
	dummy_minimizer = np.concatenate([np.array([1.0 / ALPHA]), weighted_available_metrics]).flatten()

	### Gurobi solve
	model = gp.Model()
	model.Params.LogToConsole = 0
	model.Params.Threads = N_WORKERS_GENERIC
	model.Params.TimeLimit = 3.0 
	x = model.addMVar(1 + n_paths, name='volume_each_path', lb=0)

	# Pass using_mlu=True so get_obj_fn handles the Y variable correctly
	model, x, obj_fn, obj_norm = get_obj_fn(model, dummy_minimizer, x, obj, n_paths, sas, using_mlu=True)

	model.setObjective(obj_fn)
	model.addConstr(A_util @ x <= b_ub)
	model.addConstr(volume_conservation_A @ x == conservation_b)
	model.optimize()

	if model.status != 2:
		print("Infeasible problem, exiting")
		exit(0)
		return {'solved': False}

	## Distribution is the amount of volume (not percent) placed on each path
	distribution = x.X
	path_distribution = distribution[1:]

	## Compute paths by ug
	lats_by_ug_arr = np.zeros((sas.whole_deployment_n_ug))
	paths_by_ug = {}
	vols_by_poppi = {poppi:0 for poppi in range(sas.n_popps)}
	
	for (ug,poppi),vol_amt in zip(available_paths, path_distribution):
		if poppi == NO_PATH_INGRESS(sas): 
			lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = NO_ROUTE_LATENCY
			continue 
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
	for ug,lat in lats_by_ug.items():
		lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = lat


	fraction_congested_volume = congested_volume / all_volume

	# Congestion-aware objective (Tom, 2026-08-13). This MLU fallback used to
	# report -model.objVal/obj_norm, in which over-capacity volume is priced
	# at its REAL (small) latency -- while the lats_by_ug array the same call
	# returns charges NO_ROUTE_LATENCY for exactly that volume. Same LP, two
	# incompatible summaries, and the optimizer exploited the gap: shedding
	# users IMPROVED the scalar it optimizes (georand, 2026-08-12 -- a run
	# stranding 74% of volume reported 19.6ms against a 21ms reference while
	# evaluation read 22,387ms). The scalar now comes from lats_by_ug_arr,
	# which already has the intended semantics: true latency for uncongested
	# volume, the no-route penalty (NO_ROUTE_LATENCY, env-tunable via
	# SCULPTOR_NO_ROUTE_LATENCY for training) for congested volume -- so
	# training and evaluation are the same number by construction.
	# SCULPTOR_CONGESTION_AWARE_OBJ=0 restores the legacy scalar for
	# reproducing pre-fix datasets.
	# soft-bounded scalar via the ONE helper (bad volume derived from
	# lats: congested AND no-route; see _soft_bounded_objective)
	_objective = _soft_bounded_objective(
		sas, lats_by_ug_arr, fraction_congested_volume,
		-1 * model.objVal / obj_norm)

	return {
		"objective": _objective,
		"legacy_objective": -1 * model.objVal / obj_norm,
		"raw_solution": x.X,
		"paths_by_ug": paths_by_ug,
		"lats_by_ug" : lats_by_ug_arr,
		"solved": model.status,
		"vols_by_poppi": vols_by_poppi,
		"fraction_congested_volume": fraction_congested_volume,
	}

def solve_lp_assignment_with_site_cost(sas, routed_through_ingress, obj, site_cost_alpha=DEFAULT_SITE_COST, **kwargs):
	### Minimizes (alpha * site_cost + average latency) subject to not inundating a link,
	### but could fail if there's not enough aggregate capacity
	available_paths, paths_by_ug = get_paths_by_ug(sas, routed_through_ingress)
	n_paths = len(available_paths)
	n_popps = sas.n_popps + 1 

	# --- COMBINED OBJECTIVE CALCULATION ---
	# We construct 'available_metrics' to include both latency and site cost
	available_metrics = np.ones(n_paths)
	for i,(ug,poppi) in enumerate(available_paths):
		if poppi == NO_PATH_INGRESS(sas):
			# NO_ROUTE_LATENCY + 0 cost
			available_metrics[i] = NO_ROUTE_LATENCY
		else:
			lat = sas.whole_deployment_ug_perfs[ug][sas.popps[poppi]]
			pop, _ = sas.popps[poppi]
			site_cost = sas.site_costs[pop]
			available_metrics[i] = lat + (site_cost_alpha * site_cost)
	
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

	## caps is usually link capacities, but then very "large" for users with no route
	caps = np.concatenate([_apply_capacity_headroom(sas.link_capacities_arr.flatten(), sas), np.array([100000])])

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
	model.Params.TimeLimit = 15.0 
	model.Params.Threads = N_WORKERS_GENERIC
	
	x = model.addMVar(n_paths, name='volume_each_path', lb=0)
	
	# We pass the combined metrics (Latency + Cost) to get_obj_fn
	model, x, obj_fn, obj_norm = get_obj_fn(model, available_metrics, x, obj, n_paths, sas, using_mlu=False)

	model.addConstr(cap_constraint_A @ x <= caps)
	model.addConstr(volume_conservation_A @ x == conservation_b)
	model.setObjective(obj_fn)
	model.optimize()

	if model.status != 2: ## 2 is optimal
		return {'solved': False}
	path_distribution = x.X

	lats_by_ug_arr = np.zeros((sas.whole_deployment_n_ug))
	paths_by_ug = {}
	vols_by_poppi = {poppi:0 for poppi in range(sas.n_popps)}
	
	for (ug,poppi),vol_amt in zip(available_paths, path_distribution):
		if poppi == NO_PATH_INGRESS(sas): 
			lats_by_ug_arr[sas.whole_deployment_ug_to_ind[ug]] = NO_ROUTE_LATENCY
			continue 
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


def _failure_obj_split(sas, fail_ret, no_route_penalty, congestion_penalty):
	"""Two-component failure-scenario objective:
	  soft = -avg_lat_routed
	         - no_route_penalty * frac_true_no_route
	         - congestion_penalty * frac_congested

	Where the LP marks BOTH true-no-route (user has no surviving popp on their
	pinned prefix) AND congested-popp (popp inundated) as lat=NO_ROUTE_LATENCY,
	but we separate them via the LP's `fraction_congested_volume`:
	  frac_no_route_total = users with lat >= NO_ROUTE_LATENCY (counts both)
	  frac_congested      = LP's fraction_congested_volume (only inundated popps)
	  frac_true_no_route  = frac_no_route_total - frac_congested

	no_route_penalty is "bad" (user has literally no route -- DNS pin + no
	surviving popp on that prefix). congestion_penalty is "sort of bad"
	(user routed to popp, popp overloaded -- a constraint violation in the
	Lagrangian sense). Pick no_route_penalty >> congestion_penalty so the
	optimizer prioritizes avoiding the unrecoverable case.
	"""
	if not fail_ret.get('solved'):
		# LP infeasible -> worst case, full true-no-route.
		return -no_route_penalty
	lats = np.asarray(fail_ret.get('lats_by_ug', []), dtype=float)
	vols = np.asarray(sas.whole_deployment_ug_vols, dtype=float).flatten()
	if lats.size == 0 or vols.size == 0:
		return -no_route_penalty
	n = min(lats.size, vols.size)
	lats, vols = lats[:n], vols[:n]
	total_vol = vols.sum()
	if total_vol <= 0:
		return -no_route_penalty

	no_route_mask = lats >= NO_ROUTE_LATENCY - 1e-9
	frac_no_route_total = float(vols[no_route_mask].sum() / total_vol)
	frac_congested = float(fail_ret.get('fraction_congested_volume', 0.0) or 0.0)
	# Cap at the total, since fraction_congested_volume can occasionally
	# overshoot in MLU-mode reporting (handled by the LP's loose bounds).
	frac_congested = min(frac_congested, frac_no_route_total)
	frac_true_no_route = max(0.0, frac_no_route_total - frac_congested)

	routed_vols = vols[~no_route_mask]
	if routed_vols.sum() > 0:
		avg_lat_routed = float(np.average(lats[~no_route_mask], weights=routed_vols))
	else:
		avg_lat_routed = 0.0
	return (-avg_lat_routed
			- no_route_penalty * frac_true_no_route
			- congestion_penalty * frac_congested)







def solve_lp_assignment_site_failure(sas, routed_through_ingress, obj, **kwargs):
	"""Steady-state avg-latency LP + EXHAUSTIVE mean over per-PoP (site) failures,
	with user->prefix frozen to the steady-state assignment.

	Same shape as static_failure but:
	  - Site failures (entire PoP down), not popp failures.
	  - Exhaustive over all n_pops sites -> no sampling noise.
	  - Deterministic gradient: same set of scenarios every call.

	n_pops is typically ~3-30 for the SCULPTOR deployments, so exhaustive
	enumeration is tractable (1 + n_pops LP solves per call).

	Required kwargs (from Generic_Objective):
	  adv: (n_popp, n_prefix) advertisement matrix.

	Optional kwargs (from ObjectiveSpec.lp_kwargs):
	  site_failure_beta:               weight on the failure term. Default 0.5.
	  site_failure_no_route_penalty:   per-unit no-route penalty in
	                                    `_failure_obj_split`. Default 20.0.
	"""
	adv = kwargs.get('adv')
	beta = float(kwargs.get('site_failure_beta', 0.5))
	# Two-component failure penalty. no_route is the unrecoverable case
	# (user's pinned prefix has zero surviving popps) -- make it heavy.
	# congestion is the soft-failure case (user routed but popp overloaded)
	# -- moderately bad, scales linearly with violation. Pick
	# no_route_penalty >> congestion_penalty so the optimizer prioritises
	# avoiding unrecoverable failures.
	no_route_penalty = float(kwargs.get('site_failure_no_route_penalty', 50.0))
	congestion_penalty = float(kwargs.get('site_failure_congestion_penalty', 10.0))

	inner_kwargs = {k: v for k, v in kwargs.items() if k not in (
		'adv', 'site_failure_beta', 'site_failure_no_route_penalty',
		'site_failure_congestion_penalty')}

	# 1. Steady-state solve.
	steady = solve_generic_lp_with_failure_catch(
		sas, routed_through_ingress, 'avg_latency', **inner_kwargs)
	if not steady.get('solved'):
		return steady
	if adv is None:
		# No adv to evaluate -> return steady as-is. Worker callers must
		# pass adv via the **kwargs path (see path_distribution_computer.py).
		return steady

	# 2. Derive user->prefix pinning from the steady solution. Primary popp =
	#    largest-volume popp in the steady LP routing; pinned prefix = the
	#    prefix that routes the user to that primary.
	user_to_prefix = {}
	for ugi, pathvols in steady.get('paths_by_ug', {}).items():
		if not pathvols:
			continue
		ug = sas.whole_deployment_ugs[ugi]
		best_poppi = max(pathvols, key=lambda pv: pv[1])[0]
		for prefix_i, ug_to_popp in routed_through_ingress.items():
			popp_tuple = ug_to_popp.get(ug)
			if popp_tuple is not None and sas.popp_to_ind[popp_tuple] == best_poppi:
				user_to_prefix[ug] = prefix_i
				break

	# 3. Build the {pop -> [popp_indices]} map ONCE.
	pop_to_popp_inds = {}
	for popp_i, (pop_name, _) in enumerate(sas.popps):
		pop_to_popp_inds.setdefault(pop_name, []).append(popp_i)

	# 4. For each PoP, fail all its popps simultaneously and solve a
	#    constrained avg-latency LP (paths restricted to each user's pinned
	#    prefix, surviving popps only).
	failure_objs = []
	for pop_name, failed_popp_inds in pop_to_popp_inds.items():
		a_fail = adv.copy()
		for popp_i in failed_popp_inds:
			a_fail[popp_i, :] = 0
		fail_routed, _ = sas.calculate_ground_truth_ingress(a_fail)

		constrained = {pref_i: {} for pref_i in fail_routed}
		for ug, pinned in user_to_prefix.items():
			if pinned in fail_routed:
				popp = fail_routed[pinned].get(ug)
				if popp is not None:
					constrained[pinned][ug] = popp

		fail_ret = solve_generic_lp_with_failure_catch(
			sas, constrained, 'avg_latency', **inner_kwargs)
		soft = _failure_obj_split(sas, fail_ret, no_route_penalty, congestion_penalty)
		failure_objs.append(soft)

	steady_obj = steady['objective']
	mean_failure = float(np.mean(failure_objs)) if failure_objs else steady_obj
	combined = (1.0 - beta) * steady_obj + beta * mean_failure
	steady['objective'] = combined
	steady['site_failure_steady_obj'] = float(steady_obj)
	steady['site_failure_mean_failure_obj'] = mean_failure
	steady['site_failure_n_sites'] = len(failure_objs)
	steady['site_failure_beta'] = beta
	steady['site_failure_no_route_penalty'] = no_route_penalty
	steady['site_failure_congestion_penalty'] = congestion_penalty
	return steady



generic_lp_functions = {
	'joint_latency_bulk_download': solve_joint_latency_bulk_download,
	'per_site_cost': solve_lp_assignment_with_site_cost_with_failure_catch,
	'site_failure': solve_lp_assignment_site_failure,
}

## Objectives that solve_generic_lp_persistent (in path_distribution_computer.py)
## knows how to handle. Routing these through the persistent Gurobi model warm-
## starts the simplex basis across consecutive calls; this is the main reason
## the persistent model exists, so we should actually use it.
_PERSISTENT_GUROBI_OBJECTIVES = ('avg_latency', 'per_site_cost')


def solve_min_mlu(sas, routed_through_ingress):
	"""THE canonical 'MLU of an advertisement' (Tom 2026-08-15: one
	implementation so bugs get fixed in one place): best-achievable peak
	link utilization given the advertisement's ingress options. Gurobi
	LP: min Y s.t. every routable UG's volume splits across its
	per-prefix winner popps and every link load <= Y*cap. Monotone in
	the option sets, so one-per-peering is a hard floor (<= 1/scale by
	the anycast-provisioning argument: caps = anycast_load*scale makes
	the anycast split feasible under opp). Deliberately NOT built on
	get_paths_by_ug -- the min-Y fallback's path construction drops most
	paths (open bug 2026-08-15); this reads routed_through_ingress
	({prefix: {ug: popp}}) directly. UGs with no ingress under any
	prefix are excluded here; callers charge them separately (bounded
	stranding penalty, never the 30s sentinel).
	Returns (mlu, routable_vol_frac); mlu is None if nothing routes.
	Consumers: experiments/model_error/objectives.py lat_plus_max_util."""
	from scipy.sparse import lil_matrix
	cand = {}
	for _pref, ug_to_popp in routed_through_ingress.items():
		for ug, popp in ug_to_popp.items():
			cand.setdefault(ug, set()).add(sas.popp_to_ind[popp])
	vols_arr = np.asarray(sas.whole_deployment_ug_vols, dtype=float).flatten()
	vols = {u: float(v) for u, v in zip(sas.whole_deployment_ugs, vols_arr)}
	ugs = [u for u in sas.whole_deployment_ugs if cand.get(u)]
	total_v = float(vols_arr.sum())
	if not ugs or total_v <= 0:
		return None, 0.0
	routable_v = float(sum(vols[u] for u in ugs))
	caps = np.asarray(sas.link_capacities_arr, dtype=float).flatten()[:sas.n_popps]
	idx = [(u, p) for u in ugs for p in sorted(cand[u])]
	nx = len(idx)
	A_eq = lil_matrix((len(ugs), nx + 1))
	b_eq = np.array([vols[u] for u in ugs])
	urow = {u: r for r, u in enumerate(ugs)}
	A_ub = lil_matrix((sas.n_popps, nx + 1))
	for j, (u, p) in enumerate(idx):
		A_eq[urow[u], j] = 1.0
		A_ub[p, j] = 1.0
	for i in range(sas.n_popps):
		A_ub[i, nx] = -max(float(caps[i]), 1e-9)
	model = gp.Model()
	model.Params.LogToConsole = 0
	model.Params.TimeLimit = 15.0
	model.Params.Threads = N_WORKERS_GENERIC
	z = model.addMVar(nx + 1, lb=0)
	model.addConstr(A_eq.tocsr() @ z == b_eq)
	model.addConstr(A_ub.tocsr() @ z <= np.zeros(sas.n_popps))
	model.setObjective(z[nx].sum(), gp.GRB.MINIMIZE)
	model.optimize()
	if model.status != gp.GRB.OPTIMAL:
		raise RuntimeError('solve_min_mlu: gurobi status {}'.format(model.status))
	return float(z.X[nx]), routable_v / total_v


def solve_min_hinge_excess(sas, routed_through_ingress, best_lats_by_ug,
		x_ms=10.0, over_penalty=1000.0):
	"""Capability twin of solve_min_mlu for the fracb lane (2026-08-16,
	Tom's invariant: NOTHING may display better than one-per-peering).
	The old fracb eval reported frac-beyond of the LATENCY-optimal split,
	an assignment-derived metric a constrained arm can legitimately beat
	opp on (opp's richer options let the latency LP push more marginal
	users past the threshold). This LP optimizes the metric itself:
	min sum(vol * max(0, path_lat - (best_u + x_ms))) -- per-path hinge
	costs are CONSTANTS, so it is a plain LP -- with soft capacity
	(elastic overflow at a bounded over_penalty per unit volume, never
	the 30s sentinel). Monotone in the option set => opp is an exact
	floor. Returns (excess_per_unit_vol, lats_by_ug_proxy, frac_beyond)
	where frac_beyond is the fraction of volume routed on beyond-
	threshold paths in the optimal split (display component)."""
	from scipy.sparse import lil_matrix
	cand = {}
	for _pref, ug_to_popp in routed_through_ingress.items():
		for ug, popp in ug_to_popp.items():
			cand.setdefault(ug, set()).add(sas.popp_to_ind[popp])
	vols_arr = np.asarray(sas.whole_deployment_ug_vols, dtype=float).flatten()
	ug_list = list(sas.whole_deployment_ugs)
	vols = {u: float(v) for u, v in zip(ug_list, vols_arr)}
	best = {u: float(b) for u, b in zip(
		ug_list, np.asarray(best_lats_by_ug, dtype=float).flatten())}
	ugs = [u for u in ug_list if cand.get(u)]
	total_v = float(vols_arr.sum())
	if not ugs or total_v <= 0:
		return None, None, None
	caps = np.asarray(sas.link_capacities_arr, dtype=float).flatten()[:sas.n_popps]
	idx, cost = [], []
	for u in ugs:
		for p in sorted(cand[u]):
			idx.append((u, p))
			lat = float(sas.whole_deployment_ug_perfs[u][sas.popps[p]])
			cost.append(max(0.0, lat - (best[u] + float(x_ms))))
	nx = len(idx)
	# vars: x paths, then per-popp overflow o
	nv = nx + sas.n_popps
	A_eq = lil_matrix((len(ugs), nv))
	b_eq = np.array([vols[u] for u in ugs])
	urow = {u: r for r, u in enumerate(ugs)}
	A_ub = lil_matrix((sas.n_popps, nv))
	for j, (u, p) in enumerate(idx):
		A_eq[urow[u], j] = 1.0
		A_ub[p, j] = 1.0
	for i in range(sas.n_popps):
		A_ub[i, nx + i] = -1.0
	b_ub = np.maximum(caps, 1e-9)
	c = np.concatenate([np.asarray(cost), np.full(sas.n_popps,
												  float(over_penalty))])
	model = gp.Model()
	model.Params.LogToConsole = 0
	model.Params.TimeLimit = 15.0
	model.Params.Threads = N_WORKERS_GENERIC
	z = model.addMVar(nv, lb=0)
	model.addConstr(A_eq.tocsr() @ z == b_eq)
	model.addConstr(A_ub.tocsr() @ z <= b_ub)
	model.setObjective(c @ z, gp.GRB.MINIMIZE)
	model.optimize()
	if model.status != gp.GRB.OPTIMAL:
		raise RuntimeError('solve_min_hinge_excess: gurobi status {}'.format(
			model.status))
	x = z.X[:nx]
	# FULL objective per unit volume — hinge PLUS the overflow penalty
	# (Tom 2026-08-17): reporting only dot(cost, x) dropped the elastic-
	# overflow term the LP actually minimizes, so an arm that crams all
	# volume onto few overloaded links showed near-zero "excess" (hinge 0,
	# overflow hidden) and beat opp by -1.29 on the georand dash — while
	# opp honestly spread onto worse-but-uncongested paths. Monotonicity
	# (opp = exact floor) holds for c@z, not for its hinge half.
	overflow = z.X[nx:]
	excess = (float(np.dot(np.asarray(cost), x))
			  + float(over_penalty) * float(np.sum(overflow))) / total_v
	beyond_v = float(np.sum([x[j] for j, cst in enumerate(cost) if cst > 0]))
	# UNROUTED volume is charged, not excused (Tom 2026-08-17): UGs with
	# no candidate route under the adv used to be silently dropped from
	# the LP while total_v stayed in the denominator, so a blackholing
	# advertisement scored BELOW one-per-peering (georand fracb arms at
	# -1.29 vs opp; trained rungs learned to exploit the hole). Charge
	# each unrouted UG the sentinel-latency hinge — the same formula a
	# 30s path would pay — restoring opp as an exact floor.
	unrouted = [u for u in ug_list if not cand.get(u)]
	if unrouted:
		sentinel = float(os.environ.get(
			'SCULPTOR_HINGE_NOROUTE_MS', str(NO_ROUTE_LATENCY)))
		excess += float(sum(
			vols[u] * max(0.0, sentinel - (best[u] + float(x_ms)))
			for u in unrouted)) / total_v
		beyond_v += float(sum(vols[u] for u in unrouted))
	return excess, None, beyond_v / total_v


def _can_use_persistent_gurobi(sas):
	"""True iff `sas` is a worker with an initialized persistent Gurobi model.

	When sas is the main-thread Optimal_Adv_Wrapper, it has neither the method
	nor `self.model` / `self.var_pool` -- so we have to fall back to scipy.
	"""
	return (hasattr(sas, 'solve_generic_lp_persistent')
			and hasattr(sas, 'model')
			and hasattr(sas, 'var_pool'))


def solve_generic_lp_with_failure_catch(sas, routed_through_ingress, obj, **kwargs):
	### Minmizes f(w) subject to capacity and volume constraints
	### w is the amount of volume to place on each path where a path is a <user, routed ingress>

	## Prefer the worker's persistent Gurobi model for objectives it supports.
	## This is the path that warm-starts across calls and is dramatically
	## faster on the MC inner loop (same adv, varying routing realization).
	## Falls through to scipy if (a) sas isn't a worker, (b) the persistent
	## solve returns unsolved, or (c) it raises.
	# force_mlu=True (2026-08-15, pure-MLU objective): skip the primary
	# formulations and go straight to the MLU-MINIMIZING fallback LP --
	# callers that want the best achievable peak utilization (not the
	# utilization of a latency-optimal split, which pins at ~1.0).
	if (obj in _PERSISTENT_GUROBI_OBJECTIVES and _can_use_persistent_gurobi(sas)
		and not kwargs.get('no_persistent') and not kwargs.get('force_mlu')):
		try:
			ret = sas.solve_generic_lp_persistent(
				routed_through_ingress, obj, **kwargs)
			if ret.get('solved'):
				if kwargs.get('smallverb'):
					print("Solved {} via persistent Gurobi".format(obj))
				return ret
			# Unsolved -> fall through to the scipy MLU path below.
		except Exception as e:
			print("Persistent Gurobi solve raised ({}); "
				  "falling back to scipy".format(e))

	if not kwargs.get('force_mlu'):
		try:
			return generic_lp_functions[obj](sas, routed_through_ingress, obj, **kwargs)
		except KeyError:
			pass

	verb = False
	if not kwargs.get('force_mlu'):
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
	caps = np.concatenate([_apply_capacity_headroom(sas.link_capacities_arr.flatten(), sas), np.array([100000])]).flatten()

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

	# Congestion-aware objective (Tom, 2026-08-13). This MLU fallback used to
	# report -model.objVal/obj_norm, in which over-capacity volume is priced
	# at its REAL (small) latency -- while the lats_by_ug array the same call
	# returns charges NO_ROUTE_LATENCY for exactly that volume. Same LP, two
	# incompatible summaries, and the optimizer exploited the gap: shedding
	# users IMPROVED the scalar it optimizes (georand, 2026-08-12 -- a run
	# stranding 74% of volume reported 19.6ms against a 21ms reference while
	# evaluation read 22,387ms). The scalar now comes from lats_by_ug_arr,
	# which already has the intended semantics: true latency for uncongested
	# volume, the no-route penalty (NO_ROUTE_LATENCY, env-tunable via
	# SCULPTOR_NO_ROUTE_LATENCY for training) for congested volume -- so
	# training and evaluation are the same number by construction.
	# SCULPTOR_CONGESTION_AWARE_OBJ=0 restores the legacy scalar for
	# reproducing pre-fix datasets.
	# soft-bounded scalar via the ONE helper (bad volume derived from
	# lats: congested AND no-route; see _soft_bounded_objective)
	_objective = _soft_bounded_objective(
		sas, lats_by_ug_arr, fraction_congested_volume,
		-1 * model.objVal / obj_norm)

	return {
		"objective": _objective,
		"legacy_objective": -1 * model.objVal / obj_norm,
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
	caps = np.concatenate([_apply_capacity_headroom(sas.link_capacities_arr.flatten(), sas), np.array([100000])])

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
		"objective": (_soft_bounded_objective(
			sas, lats_by_ug_arr, fraction_congested_volume,
			-1 * model.objVal / obj_norm)
			if _is_avg_latency_obj(obj) else -1 * model.objVal / obj_norm),
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
	caps = np.concatenate([_apply_capacity_headroom(sas.link_capacities_arr.flatten(), sas), np.array([100000])]).flatten()

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
		"objective": _soft_bounded_objective(sas, lats_by_ug_arr, fraction_congested_volume, -1 * model.objVal / obj_norm),
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
	caps = np.concatenate([_apply_capacity_headroom(sas.link_capacities_arr.flatten(), sas), np.array([100000])])

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
		"objective": _soft_bounded_objective(sas, lats_by_ug_arr, fraction_congested_volume, -1 * model.objVal / obj_norm),
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

# Optional extension objectives (experiments/model_error/objectives.py):
# SCULPTOR_XOBJS=1 registers them into generic_lp_functions at import
# time -- in EVERY process importing this module (driver AND Ray
# workers), which the runtime register() alone cannot guarantee. Module
# tail so the circular import (objectives.py imports from here)
# resolves against a fully-initialized namespace.
if _os.environ.get('SCULPTOR_XOBJS', '0') == '1':
	try:
		from experiments.model_error import objectives as _xobjs
		_xobjs.register()
		print('[xobjs] extension objectives registered: {}'.format(
			sorted(_xobjs.REGISTERED_OBJECTIVES)))
	except Exception as _xe:
		print('[xobjs] registration FAILED: {}'.format(_xe))

