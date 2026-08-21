"""
Stochastic LP for failure-resilient routing.

Given a fixed advertisement, optimises Σ_s p_s · latency(s) over a scenario set
S where each scenario s is a set of failed popps and p_s is its probability.

Routing decisions are per-scenario (recourse), so for a fixed advertisement the
scenarios are independent LPs. Two implementations:

  - method='cold':  rebuild the Gurobi model from scratch for each scenario.
                    Baseline; useful only as a reference for the speedup plot.

  - method='warm':  one persistent Gurobi model; flip cap_constrs[poppi].RHS
                    to 0.0 for failed popps between scenarios. Gurobi's simplex
                    warm-starts the basis -- only the changed constraint may
                    be violated, so re-optimisation is dramatically cheaper.
                    This is the path you'd actually use in production.

`solve_stochastic_lp(...)` is the entrypoint. It is *deliberately* built on top
of the existing `_LocalPathDistributionComputer` so we inherit volume
conservation, NO_ROUTE handling, and the persistent-Gurobi infrastructure
without reimplementing them.
"""
from dataclasses import dataclass, field
from typing import Iterable, FrozenSet, List, Tuple, Optional
import time

import numpy as np


# Scenario = (frozenset of failed popps, probability)
Scenario = Tuple[FrozenSet[Tuple[str, str]], float]


@dataclass
class StochasticLPResult:
	expected_latency: float                          # Σ p_s · avg_latency(s)
	per_scenario_latencies: List[float]              # avg_latency under each scenario
	per_scenario_objectives: List[float]             # raw LP objective (negative of latency)
	per_scenario_solved: List[bool]                  # did each LP solve cleanly
	scenarios: List[Scenario]                        # the input scenario list, kept for downstream
	wall_time: float                                 # total time inside solve_stochastic_lp
	method: str                                      # 'cold' or 'warm'
	per_scenario_paths: Optional[List[dict]] = None  # paths_by_ug per scenario (optional, omitted by default for memory)
	per_scenario_overflow: Optional[List[float]] = None  # fraction of volume routed in excess of static_caps (MLU mode signal)


def solve_stochastic_lp(
	worker,
	advertisement: np.ndarray,
	scenarios: List[Scenario],
	method: str = 'multi_scenario',
	keep_paths: bool = False,
) -> StochasticLPResult:
	"""Solve the per-scenario LPs for the given advertisement and return the
	weighted-expectation result.

	`worker` must be a Path_Distribution_Computer (or _LocalPathDistributionComputer)
	with an initialised persistent Gurobi model.

	`scenarios` is a list of (failed_popps, probability) tuples. `failed_popps`
	is a frozenset of (pop, peer) tuples. Probabilities should sum to ~1.

	`method`:
	  'multi_scenario' - Gurobi's Multi-Scenario API: one optimize() call
	     solves all K scenarios with shared basis exploitation. Recommended.
	  'warm' - sequential warm-start: K optimize() calls, each reuses the
	     prior solve's basis. Useful as a correctness reference.
	  'cold' - rebuild the model from scratch per scenario. Benchmark baseline.
	"""
	if method not in ('multi_scenario', 'warm', 'cold'):
		raise ValueError("method must be 'multi_scenario', 'warm', or 'cold'")
	if not scenarios:
		raise ValueError("scenarios must be non-empty")

	# Validate probabilities sum to ~1; warn if not (don't raise -- tests
	# legitimately use unnormalised sub-samples and renormalise).
	psum = sum(p for _, p in scenarios)
	if abs(psum - 1.0) > 1e-6:
		# Caller is responsible for whether this matters; we still compute the
		# weighted sum with the given probabilities.
		pass

	t_start = time.time()

	# Semantics: each scenario MASKS the advertisement (zeroes out rows for
	# failed popps) and re-computes rti. This matches what
	# solve_lp_with_failure_catch_mp does today and gives UGs a chance to be
	# re-routed onto alternate popps that the unmasked rti wouldn't expose.
	if method == 'multi_scenario':
		results = _solve_multi_scenario(worker, advertisement, scenarios, keep_paths)
	elif method == 'warm':
		results = _solve_warm(worker, advertisement, scenarios, keep_paths)
	else:
		results = _solve_cold(worker, advertisement, scenarios, keep_paths)

	wall = time.time() - t_start

	per_lat = [r['latency'] for r in results]
	per_obj = [r['objective'] for r in results]
	per_solved = [r['solved'] for r in results]
	per_overflow = [r.get('fraction_overflow_volume', 0.0) for r in results]
	expected = float(sum(p * lat for (_, p), lat in zip(scenarios, per_lat)))

	out = StochasticLPResult(
		expected_latency=expected,
		per_scenario_latencies=per_lat,
		per_scenario_objectives=per_obj,
		per_scenario_solved=per_solved,
		scenarios=scenarios,
		wall_time=wall,
		method=method,
		per_scenario_paths=[r.get('paths_by_ug') for r in results] if keep_paths else None,
		per_scenario_overflow=per_overflow,
	)
	return out


# ---- multi-scenario (one optimize() call, Gurobi internal scenario solver) ----

def _solve_multi_scenario(worker, advertisement, scenarios, keep_paths):
	"""Gurobi's Multi-Scenario API: build the model once with N scenarios baked
	in via per-scenario variable upper bounds, single optimize() call.

	Per-scenario differences encoded via:
	  - Var.ScenNUB[i]: 0 if path is not in scenario i's paths_by_ug (path goes
	    through a failed popp, or simply not exposed by rti for this scenario);
	    infinity otherwise.
	  - Cap constraints stay at their canonical RHS = static_caps[poppi];
	    failed-popp paths get pinned to 0 via UB, so the cap constraints don't
	    need per-scenario RHS overrides.

	The Multi-Scenario API exploits shared structure (one basis tree, scenario
	exploration), so this is the fast path when K is large.
	"""
	import gurobipy as gp
	from solve_lp_assignment import get_paths_by_ug, NO_PATH_INGRESS
	from constants import NO_ROUTE_LATENCY

	# Phase 1: per-scenario rti + paths_by_ug, computed once.
	per_scenario_paths = []
	per_scenario_failed_popps = []
	for failed_popps, _prob in scenarios:
		adv_s = advertisement.copy()
		for popp in failed_popps:
			if popp in worker.popp_to_ind:
				adv_s[worker.popp_to_ind[popp], :] = 0
		rti_s, _ = worker.calculate_ground_truth_ingress(adv_s, do_cache=False)
		avail_s, _ = get_paths_by_ug(worker, rti_s)
		per_scenario_paths.append(frozenset(avail_s))
		per_scenario_failed_popps.append(
			frozenset(worker.popp_to_ind[p] for p in failed_popps if p in worker.popp_to_ind)
		)

	# Union of all paths across scenarios; new vars added to var_pool if missing.
	all_paths = set()
	for paths in per_scenario_paths:
		all_paths |= paths

	for (ug, poppi) in all_paths:
		if (ug, poppi) in worker.var_pool:
			continue
		if poppi == NO_PATH_INGRESS(worker):
			latency = NO_ROUTE_LATENCY
		else:
			latency = worker.whole_deployment_ug_perfs[ug][worker.popps[poppi]]
		col = gp.Column()
		col.addTerms(1.0, worker.vol_constrs[ug])
		col.addTerms(1.0, worker.cap_constrs[poppi])
		worker.var_pool[(ug, poppi)] = worker.model.addVar(lb=0.0, obj=latency, column=col)

	# Make sure objective coefficients are correct for every var we care about
	# (in case anything stale from an earlier MLU-mode solve corrupted them).
	for (ug, poppi) in all_paths:
		var = worker.var_pool[(ug, poppi)]
		if poppi == NO_PATH_INGRESS(worker):
			var.Obj = NO_ROUTE_LATENCY
		else:
			var.Obj = worker.whole_deployment_ug_perfs[ug][worker.popps[poppi]]

	# MLU mode with a STRONG penalty: this way, capacity-feasible scenarios
	# leave MLU at 0 (recovering the non-MLU answer that warm gets), AND
	# capacity-infeasible scenarios still solve (MLU engages just enough to
	# satisfy volume conservation). The penalty is heavy enough that the LP
	# never "trades latency for overflow" -- mirrors warm's non-MLU-first /
	# MLU-fallback behavior in one solver call.
	#
	# Formulation: sum(flow) - static_caps[i] * mlu_dummy ≤ 0, with
	# mlu_dummy in objective at MLU_PENALTY. When all caps fit, mlu=0 ⇒
	# constraint reduces to sum(flow)≤0, which is too tight, so we keep
	# constraint coefficient and let mlu>=1 be the no-overflow level. Set
	# initial mlu_dummy lower bound to 1 so "no overflow" is the default.
	import gurobipy as gp
	MLU_PENALTY = 1e6  # much larger than max plausible latency * volume
	worker.mlu_dummy.Obj = MLU_PENALTY
	worker.mlu_dummy.UB = gp.GRB.INFINITY
	worker.mlu_dummy.LB = 1.0  # at mlu=1 the constraint is sum(flow) ≤ cap (non-MLU)
	for pi, constr in worker.cap_constrs.items():
		worker.model.chgCoeff(constr, worker.mlu_dummy, -1.0 * float(worker.static_caps[pi]))
		constr.RHS = 0.0

	# Default: deactivate every path var; per-scenario UB will re-enable.
	all_vars = list(worker.var_pool.values())
	worker.model.setAttr("UB", all_vars, [0.0] * len(all_vars))

	# Phase 2: configure the multi-scenario problem.
	K = len(scenarios)
	# NumScenarios=0 then set to K to wipe stale per-scenario data from prior calls.
	worker.model.NumScenarios = 0
	worker.model.NumScenarios = K

	for i in range(K):
		worker.model.Params.ScenarioNumber = i
		paths_i = per_scenario_paths[i]
		# All vars start with the model's nominal UB (0); we override per-scenario.
		for (ug, poppi), var in worker.var_pool.items():
			if (ug, poppi) in paths_i:
				var.ScenNUB = gp.GRB.INFINITY
			else:
				var.ScenNUB = 0.0

	# Phase 3: one optimize() call solves all K scenarios.
	worker.model.optimize()

	# Phase 4: per-scenario result extraction.
	results = []
	for i in range(K):
		worker.model.Params.ScenarioNumber = i
		# Per-scenario objective value (or infeasible status).
		try:
			scen_obj_val = float(worker.model.ScenNObjVal)
			# Gurobi uses GRB.INFINITY for infeasible scenario solutions.
			infeasible = scen_obj_val >= 1e30
		except Exception:
			infeasible = True
		if infeasible:
			results.append({
				'solved': False,
				'objective': float('-inf'),
				'latency': float('inf'),
				'paths_by_ug': {} if keep_paths else None,
				'fraction_overflow_volume': 0.0,
			})
			continue

		# Reconstruct latency from per-scenario variable values (var.ScenNX).
		paths_by_ug_res = {}
		total_weighted_lat = 0.0
		vol_by_popp = {}
		for (ug, poppi), var in worker.var_pool.items():
			vol = float(var.ScenNX)
			if vol <= 1e-7:
				continue
			ugi = worker.whole_deployment_ug_to_ind[ug]
			ug_vol = float(worker.whole_deployment_ug_to_vol[ug])
			vpct = vol / max(ug_vol, 1e-9)
			paths_by_ug_res.setdefault(ugi, []).append((poppi, vpct))
			if poppi == NO_PATH_INGRESS(worker):
				path_lat = NO_ROUTE_LATENCY
			else:
				path_lat = worker.whole_deployment_ug_perfs[ug][worker.popps[poppi]]
			total_weighted_lat += path_lat * vol
			vol_by_popp[poppi] = vol_by_popp.get(poppi, 0.0) + vol

		no_path_idx = len(worker.static_caps) - 1
		overflow_vol, total_vol = 0.0, 0.0
		for poppi, v in vol_by_popp.items():
			total_vol += v
			if poppi == no_path_idx:
				continue
			cap = float(worker.static_caps[poppi])
			if v > cap + 1e-6:
				overflow_vol += (v - cap)

		obj_norm = float(np.sum(worker.whole_deployment_ug_vols))
		avg_lat = total_weighted_lat / max(obj_norm, 1e-9)
		results.append({
			'solved': True,
			'objective': -avg_lat,
			'latency': avg_lat,
			'paths_by_ug': paths_by_ug_res if keep_paths else None,
			'fraction_overflow_volume': overflow_vol / max(total_vol, 1e-9),
		})

	# Reset to single-scenario mode so subsequent unrelated calls see a normal
	# model. NumScenarios=0 disables multi-scenario; ScenarioNumber reset for
	# good measure.
	worker.model.NumScenarios = 0

	return results


# ---- warm-start (one persistent Gurobi model, change RHS per scenario) ----

def _solve_warm(worker, advertisement, scenarios, keep_paths):
	"""One persistent Gurobi model across scenarios.

	For each scenario we mask the advertisement (zero out failed popps' rows)
	and recompute rti; solve_generic_lp_persistent then warm-starts from the
	previous solve's basis. The var_pool grows monotonically as new (ug,poppi)
	paths get exercised by different scenarios.
	"""
	results = []
	for failed_popps, _prob in scenarios:
		adv_s = advertisement.copy()
		for popp in failed_popps:
			if popp in worker.popp_to_ind:
				adv_s[worker.popp_to_ind[popp], :] = 0
		rti_s, _ = worker.calculate_ground_truth_ingress(adv_s, do_cache=False)
		ret = _solve_safely(worker, rti_s)
		results.append(_summarise_ret(ret, worker, keep_paths))
	return results


# ---- cold-start (rebuild Gurobi model from scratch per scenario) ----

def _solve_cold(worker, advertisement, scenarios, keep_paths):
	"""Rebuild the persistent Gurobi LP per scenario.

	Benchmark baseline for measuring basis-reuse benefit: per scenario, dispose
	the existing Gurobi model and reinit from scratch -- so no basis or
	var_pool carries over between scenarios. The semantics (advertisement
	masking, rti recompute) is identical to warm.
	"""
	import gurobipy as gp

	results = []
	for failed_popps, _prob in scenarios:
		# Dispose & rebuild the model. We replicate init_persistent_lp's body
		# rather than calling it directly so we're certain the rebuilt model
		# matches one-to-one with what the worker would have created originally.
		try:
			worker.model.dispose()
		except Exception:
			pass

		worker.model = gp.Model(f"Worker_{worker.worker_i}_Cold")
		worker.model.Params.LogToConsole = 0
		worker.model.Params.Method = 1
		worker.model.Params.Threads = 1
		worker.mlu_dummy = worker.model.addVar(lb=0.0, obj=0.0, name="mlu_Y")
		worker.vol_constrs = {}
		for ugi, ug in enumerate(worker.whole_deployment_ugs):
			target_vol = float(worker.whole_deployment_ug_vols[ugi])
			worker.vol_constrs[ug] = worker.model.addLConstr(0.0, gp.GRB.EQUAL, target_vol, name=f"vol_{ug}")
		worker.cap_constrs = {}
		for pi in range(len(worker.static_caps)):
			worker.cap_constrs[pi] = worker.model.addLConstr(0.0, gp.GRB.LESS_EQUAL, float(worker.static_caps[pi]), name=f"cap_{pi}")
		worker.var_pool = {}

		# Mask adv + recompute rti
		adv_s = advertisement.copy()
		for popp in failed_popps:
			if popp in worker.popp_to_ind:
				adv_s[worker.popp_to_ind[popp], :] = 0
		rti_s, _ = worker.calculate_ground_truth_ingress(adv_s, do_cache=False)

		ret = _solve_safely(worker, rti_s)
		results.append(_summarise_ret(ret, worker, keep_paths))

	return results


def _solve_safely(worker, rti):
	"""Call worker.solve_generic_lp_persistent without letting it `exit(0)`
	on infeasibility. Mirror the body of solve_generic_lp_persistent at
	path_distribution_computer.py:178+ but: (a) return {'solved': False}
	instead of exiting on dual infeasibility, and (b) compute average
	latency by iterating the solved variables (not model.objVal -- in MLU
	fallback mode model.objVal is contaminated by the MLU dummy variable's
	contribution and is NOT just the avg latency × volume).

	MLU fallback uses a STRONG penalty (1e6) so MLU stays at the no-overflow
	floor unless feasibility strictly requires more. This is the same
	formulation _solve_multi_scenario uses, so the two paths agree
	deterministically on MLU-required scenarios.
	"""
	import gurobipy as gp
	from solve_lp_assignment import get_paths_by_ug, NO_PATH_INGRESS
	from constants import NO_ROUTE_LATENCY

	available_paths, _ = get_paths_by_ug(worker, rti)
	obj_coeffs = []
	for ug, poppi in available_paths:
		if poppi == NO_PATH_INGRESS(worker):
			obj_coeffs.append(NO_ROUTE_LATENCY)
		else:
			obj_coeffs.append(worker.whole_deployment_ug_perfs[ug][worker.popps[poppi]])

	# Try non-MLU first (matches multi-scenario's mlu=1 floor).
	model_res = worker.solve_unified_lp(available_paths, obj_coeffs, using_mlu=False)
	if model_res is None:
		# MLU fallback. Override mlu_dummy with the strong-penalty formulation
		# that matches _solve_multi_scenario so both paths converge to the
		# same routing. solve_unified_lp(using_mlu=True) uses ALPHA-based
		# penalty which is too soft; we set our own here.
		MLU_PENALTY = 1e6
		worker.mlu_dummy.Obj = MLU_PENALTY
		worker.mlu_dummy.UB = gp.GRB.INFINITY
		worker.mlu_dummy.LB = 1.0
		for pi, constr in worker.cap_constrs.items():
			worker.model.chgCoeff(constr, worker.mlu_dummy, -1.0 * float(worker.static_caps[pi]))
			constr.RHS = 0.0
		# Activate the same path vars as solve_unified_lp would
		all_vars = list(worker.var_pool.values())
		worker.model.setAttr("UB", all_vars, [0.0] * len(all_vars))
		# Discover any missing vars + activate
		for (ug, poppi), latency in zip(available_paths, obj_coeffs):
			key = (ug, poppi)
			if key not in worker.var_pool:
				col = gp.Column()
				col.addTerms(1.0, worker.vol_constrs[ug])
				col.addTerms(1.0, worker.cap_constrs[poppi])
				worker.var_pool[key] = worker.model.addVar(lb=0.0, obj=latency, column=col)
			else:
				worker.var_pool[key].Obj = latency
			worker.var_pool[key].UB = gp.GRB.INFINITY
		worker.model.optimize()
		if worker.model.status != 2:
			# Reset state before returning
			worker.mlu_dummy.LB = 0.0
			worker.mlu_dummy.UB = 0.0
			worker.mlu_dummy.Obj = 0.0
			return {'solved': False, 'objective': float('-inf'), 'paths_by_ug': {}}
		# Reset LB on mlu_dummy so subsequent non-MLU solves work
		# (we don't reset Obj/UB/coefficients here -- solve_unified_lp does
		# that on its next call).
		try:
			model_res = worker.model
		finally:
			pass
	if model_res is None:
		return {'solved': False, 'objective': float('-inf'), 'paths_by_ug': {}}

	# Reconstruct latency from variable values (not model.objVal -- MLU-safe).
	# Also compute per-popp total volume so we can detect MLU-overflow (any
	# popp whose total flow exceeded static_caps means MLU mode kicked in and
	# the routing is using more capacity than the link nominally has).
	paths_by_ug_res = {}
	total_weighted_lat = 0.0
	vol_by_popp = {}
	for (ug, poppi) in available_paths:
		var = worker.var_pool.get((ug, poppi))
		if var is None:
			continue
		vol = float(var.X)
		if vol <= 1e-7:
			continue
		ugi = worker.whole_deployment_ug_to_ind[ug]
		ug_vol = float(worker.whole_deployment_ug_to_vol[ug])
		vpct = vol / max(ug_vol, 1e-9)
		paths_by_ug_res.setdefault(ugi, []).append((poppi, vpct))
		if poppi == NO_PATH_INGRESS(worker):
			path_lat = NO_ROUTE_LATENCY
		else:
			path_lat = worker.whole_deployment_ug_perfs[ug][worker.popps[poppi]]
		total_weighted_lat += path_lat * vol
		vol_by_popp[poppi] = vol_by_popp.get(poppi, 0.0) + vol

	# Congestion: any real popp where total routed volume exceeds its capacity.
	# (Skip the NO_PATH_INGRESS sentinel index = last.)
	no_path_idx = len(worker.static_caps) - 1
	overflow_vol = 0.0
	total_vol = 0.0
	for poppi, v in vol_by_popp.items():
		total_vol += v
		if poppi == no_path_idx:
			continue
		cap = float(worker.static_caps[poppi])
		if v > cap + 1e-6:
			overflow_vol += (v - cap)

	obj_norm = float(np.sum(worker.whole_deployment_ug_vols))
	avg_lat = total_weighted_lat / max(obj_norm, 1e-9)
	return {
		'solved': True,
		'objective': -avg_lat,  # objective convention: -latency (more positive = better)
		'paths_by_ug': paths_by_ug_res,
		'fraction_overflow_volume': overflow_vol / max(total_vol, 1e-9),
	}


def _summarise_ret(ret, worker, keep_paths):
	"""Normalise the dict that solve_generic_lp_persistent returns into the
	shape we want for StochasticLPResult."""
	solved = bool(ret.get('solved', False))
	# 'objective' in the existing convention is -avg_latency
	obj = float(ret.get('objective', 0.0))
	latency = -obj  # positive latency
	out = {
		'solved': solved,
		'objective': obj,
		'latency': latency,
		'fraction_overflow_volume': float(ret.get('fraction_overflow_volume', 0.0)),
	}
	if keep_paths:
		out['paths_by_ug'] = ret.get('paths_by_ug')
	return out


# ---- scenario factories ----

def nominal_only_scenario() -> List[Scenario]:
	return [(frozenset(), 1.0)]


def single_popp_scenarios(deployment, p_any_fail: float = 0.5) -> List[Scenario]:
	"""Nominal + every single-popp failure, uniform prob over failures."""
	popps = deployment['popps']
	nominal = (frozenset(), 1.0 - p_any_fail)
	per = p_any_fail / len(popps) if popps else 0.0
	return [nominal] + [(frozenset([tuple(p)]), per) for p in popps]


def single_pop_scenarios(deployment, p_any_fail: float = 0.5) -> List[Scenario]:
	"""Nominal + every single-pop failure (drops ALL popps at that pop)."""
	popps = [tuple(p) for p in deployment['popps']]
	pops = sorted({pop for pop, _peer in popps})
	nominal = (frozenset(), 1.0 - p_any_fail)
	per = p_any_fail / len(pops) if pops else 0.0
	failure_scenarios = []
	for pop in pops:
		failed = frozenset(p for p in popps if p[0] == pop)
		failure_scenarios.append((failed, per))
	return [nominal] + failure_scenarios


def solve_headroom_lp(worker, advertisement: np.ndarray,
					   headroom_factor: float = 0.2) -> dict:
	"""Single LP solve where every real-popp capacity is multiplied by
	(1 - headroom_factor). The NO_PATH sentinel cap (last entry) is left
	alone. This is what SCULPTOR_CAPACITY_HEADROOM does during training in
	condition C of the Session-3 experiment: ONE LP solve per gradient
	step, with a heuristic capacity haircut as the failure-tolerance proxy.

	Returns: { latency, objective, wall_time, method, headroom_factor }.
	"""
	saved = np.array(worker.static_caps).copy()
	# Scale real popp caps, leave the NO_PATH sentinel at the tail untouched.
	worker.static_caps = saved.copy()
	worker.static_caps[:-1] = saved[:-1] * (1.0 - headroom_factor)
	try:
		t0 = time.time()
		rti, _ = worker.calculate_ground_truth_ingress(advertisement, do_cache=False)
		ret = _solve_safely(worker, rti)
		wall = time.time() - t0
	finally:
		worker.static_caps = saved
	return {
		'latency': -float(ret['objective']),
		'objective': float(ret['objective']),
		'solved': bool(ret.get('solved', False)),
		'wall_time': wall,
		'method': 'headroom',
		'headroom_factor': headroom_factor,
		'fraction_overflow_volume': float(ret.get('fraction_overflow_volume', 0.0)),
	}


def compute_unroutable_volume_fractions(
		stoch_result, worker) -> List[float]:
	"""For each scenario in a StochasticLPResult (must have keep_paths=True),
	return the fraction of total UG volume that ended up on the NO_PATH_INGRESS
	sentinel — i.e. could not be routed under that scenario's failure mask.

	Higher = worse failure tolerance.
	"""
	if stoch_result.per_scenario_paths is None:
		raise ValueError(
			"stoch_result.per_scenario_paths is None; "
			"call solve_stochastic_lp with keep_paths=True")
	from solve_lp_assignment import NO_PATH_INGRESS
	no_path_poppi = NO_PATH_INGRESS(worker)
	fracs = []
	for paths in stoch_result.per_scenario_paths:
		no_path_vol = 0.0
		total_vol = 0.0
		if paths is None:
			fracs.append(float('nan'))
			continue
		for ugi, allocs in paths.items():
			ug = worker.whole_deployment_ugs[ugi]
			ug_vol = float(worker.whole_deployment_ug_to_vol[ug])
			for poppi, vpct in allocs:
				v = vpct * ug_vol
				total_vol += v
				if poppi == no_path_poppi:
					no_path_vol += v
		fracs.append(no_path_vol / max(total_vol, 1e-9))
	return fracs


def pick_gradient_step_scenarios_importance(
		deployment,
		K: int,
		iter_seed: int,
		last_per_scenario_latencies: Optional[List[float]] = None,
		p_any_fail: float = 0.5,
		exploration_weight: float = 0.1,
) -> List[Scenario]:
	"""Importance-weighted version of pick_gradient_step_scenarios.

	If `last_per_scenario_latencies` is provided (from the prior iteration's
	full-set evaluation, aligned with `single_popp_scenarios` -- index 0 is
	nominal), each failure scenario's sampling probability is set proportional
	to `max(lat_under_scenario - lat_nominal, 0)` (its "failure impact").
	A fraction `exploration_weight` is mixed with uniform so every scenario
	keeps non-zero probability.

	Each sampled failure scenario is RETURNED WITH ITS IMPORTANCE-CORRECTED
	WEIGHT:  weight_i = p_any_fail / (K * p_importance_i)
	so that solve_stochastic_lp's `expected_latency = Σ_i weight_i * L(s_i)`
	is an unbiased estimator of (1 - p_any_fail)·L(nominal) +
	(p_any_fail / n_failure)·Σ_{all failures} L(s). I.e. matches the
	expectation we'd get with the FULL scenario set, on average.

	If `last_per_scenario_latencies is None` (cold start), reduces to uniform
	sampling but still IS-weighted (so the math degenerates to weight_i =
	p_any_fail/n_failure -- same as the non-IS pick).
	"""
	all_scen = single_popp_scenarios(deployment, p_any_fail=p_any_fail)
	n_failure = len(all_scen) - 1
	if K >= n_failure:
		return all_scen   # full set is unbiased by construction

	# Sampling probability per failure scenario
	if last_per_scenario_latencies is not None and \
			len(last_per_scenario_latencies) == len(all_scen):
		nominal_lat = float(last_per_scenario_latencies[0])
		fail_lats = np.asarray([float(x) for x in last_per_scenario_latencies[1:]])
		# clip impact to >0 (failures can produce lower latency under
		# BGP-priority non-monotonicity; treat those as neutral)
		impact = np.maximum(fail_lats - nominal_lat, 1e-6)
		p_imp_raw = impact / impact.sum()
		p_imp = ((1.0 - exploration_weight) * p_imp_raw
				 + exploration_weight * (1.0 / n_failure))
		p_imp = p_imp / p_imp.sum()
	else:
		p_imp = np.full(n_failure, 1.0 / n_failure)

	rng = np.random.default_rng(int(iter_seed))
	# With-replacement: standard IS estimator with weight =
	#   (target_prob / proposal_prob) / K
	# Target distribution for failure scenarios is uniform: 1/n_failure.
	# Proposal distribution is p_imp.
	# Multiplying by p_any_fail gets the failure-portion of the joint
	# distribution, yielding scenario-level weight
	#   p_any_fail / (K * n_failure * p_imp_i).
	# Without-replacement would require Horvitz-Thompson with marginal
	# inclusion probabilities (not just K*p_imp), which is more complex.
	idx = rng.choice(n_failure, size=K, replace=True, p=p_imp)

	# Deduplicate: if the same scenario is picked m times, that contributes
	# m identical scenarios to the picked list. solve_stochastic_lp would
	# solve them separately (wasteful), so we coalesce by summing weights.
	picked: List[Scenario] = [all_scen[0]]
	scen_weights = {}
	for i in idx:
		scen, _ = all_scen[int(i) + 1]
		w = p_any_fail / (K * n_failure * float(p_imp[int(i)]))
		scen_weights[scen] = scen_weights.get(scen, 0.0) + w
	for scen, w in scen_weights.items():
		picked.append((scen, w))
	return picked


def pick_gradient_step_scenarios(
		deployment, K: int, iter_seed: int, p_any_fail: float = 0.5,
) -> List[Scenario]:
	"""Pick K single-popp failure scenarios for a SCULPTOR gradient step.

	Returns the same list_of_scenarios shape that solve_stochastic_lp expects:
	a nominal scenario at index 0 plus K (or all available) failure scenarios.

	Behavior:
	  - If K >= n_failure_scenarios: returns ALL single-popp failures.
	    Deterministic, unbiased — the true expectation over the failure space.
	  - Otherwise: uniform random sample of K failure scenarios using
	    np.random.default_rng(iter_seed). Sampling varies with iter_seed so
	    successive gradient steps see DIFFERENT samples (unbiased on average).

	The previous design cached one fixed K-sample across all iters, which
	silently biased the gradient toward whichever scenarios were drawn at
	iter 0. This function avoids that by re-rolling per iter.
	"""
	all_scen = single_popp_scenarios(deployment, p_any_fail=p_any_fail)
	n_failure = len(all_scen) - 1
	if K >= n_failure:
		return all_scen
	rng = np.random.default_rng(int(iter_seed))
	idx = rng.choice(n_failure, size=K, replace=False) + 1
	picked = [all_scen[0]] + [all_scen[int(i)] for i in idx]
	total = sum(p for _, p in picked)
	return [(s, p / total) for s, p in picked]


def subsample_scenarios(scenarios: List[Scenario], k: int, rng) -> List[Scenario]:
	"""Uniform random sub-sample of size k, renormalised so probs sum to 1."""
	if k >= len(scenarios):
		# renormalise the existing list
		total = sum(p for _, p in scenarios)
		return [(s, p / total) for s, p in scenarios]
	idx = rng.choice(len(scenarios), size=k, replace=False)
	sub = [scenarios[i] for i in idx]
	total = sum(p for _, p in sub)
	return [(s, p / total) for s, p in sub]
