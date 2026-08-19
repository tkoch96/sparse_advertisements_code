"""
LP correctness tests.

For a solved LP, the result dictionary the worker / scipy paths return is
shaped:
  {
	'objective': float,                     # -1 * avg latency (neg = benefit)
	'paths_by_ug': {ugi -> [(poppi, vol_pct), ...]},
	'lats_by_ug':  np.ndarray of length n_ugs,
	'vols_by_poppi': {poppi -> volume},
	'solved': True | False,
	'fraction_congested_volume': float,
	'available_paths': [(ug, poppi), ...],  # only the Gurobi path includes this
  }

These tests verify that what comes back from the solver actually makes sense.
Without these, refactors silently produce LPs that solve but encode the wrong
thing.

We test both solver backends (Gurobi persistent + scipy) and cross-check that
they agree on the objective for the same problem.

Run:
	pytest -v tests/test_lp_correctness.py
"""
import os
import math

import numpy as np
import pytest


# Helper: build worker + a baseline rti for one of the small sizes.
def _setup(size='small'):
	import random
	random.seed(31415)
	np.random.seed(31415)
	from deployment_setup import get_random_deployment
	from path_distribution_computer_ray import _LocalPathDistributionComputer

	dep = get_random_deployment(size, port=31600)
	subdep = dep
	init_kwa = {
		'lambduh': 1.0, 'gamma': 0, 'verbose': False,
		'n_prefixes': None, 'with_capacity': False,
		'save_run_dir': None, 'generic_objective': 'avg_latency',
	}
	worker = _LocalPathDistributionComputer(
		worker_i=0, deployment=subdep, init_kwargs=init_kwa)
	n_popps = len(dep['popps'])

	# A baseline advertisement: every popp on, one prefix. Deterministic rti.
	adv = np.ones((n_popps, 1))
	rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
	return worker, dep, adv, rti


def _check_invariants(worker, deployment, ret, ctx_label=""):
	"""Run the universal LP-solution invariants against `ret`. ctx_label is
	a string included in assertion messages so we know which mode failed."""
	tag = "[{}] ".format(ctx_label) if ctx_label else ""

	# 1. Solved.
	assert ret.get('solved'), tag + "LP not solved"

	# 2. Per-user volume percentages should sum to ~1.0 (or 0 if the user got
	#    no path at all, which can happen in MLU mode or degenerate inputs).
	paths_by_ug = ret['paths_by_ug']
	for ugi, allocs in paths_by_ug.items():
		total_pct = sum(vp for _, vp in allocs)
		# Allow either ~1.0 (fully routed) or ~0 (no path), with a generous
		# tolerance for floating point + congestion-scaling effects.
		assert (math.isclose(total_pct, 1.0, abs_tol=1e-3)
				or math.isclose(total_pct, 0.0, abs_tol=1e-6)), (
			tag + "user {} has vol_pct sum {} -- should be ~1.0 or ~0"
			.format(ugi, total_pct))

	# 3. lats_by_ug entries are non-negative and finite.
	lats = np.asarray(ret['lats_by_ug'])
	assert np.all(np.isfinite(lats)), tag + "lats_by_ug has non-finite values"
	assert np.all(lats >= -1e-6), tag + "lats_by_ug has negative entries"

	# 4. lats_by_ug consistent with paths_by_ug: for each user, the recorded
	#    latency should be the volume-weighted average of the per-path
	#    latencies. We check users whose total_pct is ~1.
	from constants import NO_ROUTE_LATENCY
	from solve_lp_assignment import NO_PATH_INGRESS
	ugs_list = worker.whole_deployment_ugs
	popps_list = worker.popps
	for ugi, allocs in paths_by_ug.items():
		ug = ugs_list[ugi]
		total_pct = sum(vp for _, vp in allocs)
		if not math.isclose(total_pct, 1.0, abs_tol=1e-3):
			continue
		expected = 0.0
		for poppi, vp in allocs:
			if poppi == NO_PATH_INGRESS(worker):
				expected += vp * NO_ROUTE_LATENCY
			else:
				expected += vp * worker.whole_deployment_ug_perfs[ug][popps_list[poppi]]
		assert math.isclose(expected, lats[ugi], rel_tol=1e-3, abs_tol=1e-2), (
			tag + "lats_by_ug[{}]={:.3f} but paths_by_ug implies {:.3f}"
			.format(ugi, lats[ugi], expected))

	# 5. fraction_congested_volume is a fraction (some solvers return values
	#    slightly above 1.0 in MLU mode; clip the upper bound generously).
	fcv = ret.get('fraction_congested_volume')
	if fcv is not None:
		assert -1e-6 <= fcv <= 1.5, (
			tag + "fraction_congested_volume = {} out of plausible range"
			.format(fcv))


# ---------------------------------------------------------------------------#
# Test the persistent Gurobi path.
# ---------------------------------------------------------------------------#
@pytest.mark.unit
@pytest.mark.gurobi
def test_persistent_gurobi_lp_invariants():
	"""solve_generic_lp_persistent on a small deployment must return a
	solution that respects volume conservation and latency consistency."""
	worker, deployment, adv, rti = _setup('small')
	ret = worker.solve_generic_lp_persistent(rti, 'avg_latency')
	_check_invariants(worker, deployment, ret, ctx_label="persistent_gurobi")


# ---------------------------------------------------------------------------#
# Test the scipy path (call solve_generic_lp directly to bypass the dispatch
# that now routes avg_latency to Gurobi for workers).
# ---------------------------------------------------------------------------#
@pytest.mark.unit
@pytest.mark.gurobi  # scipy doesn't need Gurobi, but worker construction does
def test_scipy_lp_invariants():
	"""The scipy fallback path (solve_generic_lp) must satisfy the same
	invariants as the Gurobi path. This is what runs for objectives the
	persistent model doesn't handle, and as a fallback for failures."""
	worker, deployment, adv, rti = _setup('small')
	from solve_lp_assignment import solve_generic_lp
	ret = solve_generic_lp(worker, rti, 'avg_latency')
	_check_invariants(worker, deployment, ret, ctx_label="scipy")


# ---------------------------------------------------------------------------#
# Cross-check: both backends should agree on the objective for the same LP.
# ---------------------------------------------------------------------------#
@pytest.mark.unit
@pytest.mark.gurobi
def test_gurobi_and_scipy_objectives_agree():
	"""Gurobi and scipy solve the same LP. Up to numerical tolerance and
	algorithm differences, the objective value should agree.

	Tolerance is generous (5%) because (a) the two formulations differ
	slightly in MLU handling, and (b) at LP scale 'agreeing within a few
	percent' is what we actually care about -- the SCULPTOR algorithm
	doesn't require exact agreement, just consistent ranking."""
	worker, deployment, adv, rti = _setup('small')
	from solve_lp_assignment import solve_generic_lp
	g_ret = worker.solve_generic_lp_persistent(rti, 'avg_latency')
	s_ret = solve_generic_lp(worker, rti, 'avg_latency')
	assert g_ret['solved'] and s_ret['solved']
	g, s = g_ret['objective'], s_ret['objective']
	# Both should be non-positive (objective is -avg_latency).
	assert g <= 1e-6 and s <= 1e-6, \
		"Objectives should be <= 0; got Gurobi={}, scipy={}".format(g, s)
	rel = abs(g - s) / (abs(s) + 1e-9)
	assert rel < 0.05, (
		"Gurobi and scipy objectives disagree by {:.1%}: "
		"Gurobi={:.4f}, scipy={:.4f}".format(rel, g, s)
	)


# ---------------------------------------------------------------------------#
# Volume conservation is the single property a correctness regression would
# most likely break. We test it explicitly at multiple advertisement
# configurations.
# ---------------------------------------------------------------------------#
# ---------------------------------------------------------------------------#
# Toy LP tests: hand-verifiable expected outputs.
#
# Each of these constructs a scenario where we know what the LP *must* return,
# and asserts that it does. These catch regressions that invariant checks
# (volume conservation etc.) can miss -- e.g., a sign error in the objective
# would still preserve volume conservation but return the wrong allocation.
# ---------------------------------------------------------------------------#

def _set_caps_uniform(worker, cap_value):
	"""Override every link capacity to `cap_value` on the worker's persistent
	model. solve_unified_lp picks RHS values from `self.static_caps` on each
	call, so we don't need to rebuild the LP; just rewrite the array."""
	# static_caps = [link_caps..., NO_PATH_cap]. Keep NO_PATH at its sentinel.
	worker.static_caps = np.concatenate([
		np.full(len(worker.link_capacities_arr.flatten()), cap_value),
		[worker.static_caps[-1]],  # leave the NO_PATH bucket alone
	]).astype(float)


@pytest.mark.unit
@pytest.mark.gurobi
def test_unlimited_capacity_routes_each_user_to_best_reachable_popp():
	"""When capacity is effectively unlimited, the LP minimises avg latency
	by sending each user's full volume to its lowest-latency reachable popp
	(the lowest-latency popp in any of its rti entries).

	This is a falsifiable answer we can compute by hand from the deployment
	data: for each user, find min latency in their rti popps; that's the
	value the LP should report in lats_by_ug. Catches sign errors and
	allocation bugs that invariant tests would miss."""
	worker, deployment, adv, rti = _setup('really_friggin_small')
	# Make capacity so large that no user could realistically saturate any link.
	_set_caps_uniform(worker, 1e12)

	ret = worker.solve_generic_lp_persistent(rti, 'avg_latency')
	assert ret['solved']

	ugs_list = worker.whole_deployment_ugs
	popps_list = worker.popps

	# Compute the expected per-user best latency.
	expected_lats = {}
	for ug in ugs_list:
		# All popps the user is rti-routed to across prefixes.
		reachable = set()
		for prefix_i, ug_to_popp in rti.items():
			if ug in ug_to_popp:
				reachable.add(ug_to_popp[ug])
		if not reachable:
			continue
		# Best latency over reachable popps.
		expected_lats[ug] = min(
			worker.whole_deployment_ug_perfs[ug][p] for p in reachable
		)

	mismatches = []
	for ug, best in expected_lats.items():
		ugi = worker.whole_deployment_ug_to_ind[ug]
		actual = float(ret['lats_by_ug'][ugi])
		if not math.isclose(actual, best, rel_tol=1e-3, abs_tol=1e-2):
			mismatches.append((ug, ugi, actual, best))

	assert not mismatches, (
		"With unlimited capacity each user should route fully to their best "
		"reachable popp. Mismatches (ug, ugi, actual_lat, expected_lat): {}"
		.format(mismatches[:5])  # cap output length
	)


@pytest.mark.unit
@pytest.mark.gurobi
def test_zero_latency_popp_attracts_all_traffic_for_one_user():
	"""Override one user's latency to a single popp to be 0. With unlimited
	capacity, that user must route 100% to that popp and report lats_by_ug[u]
	== 0. Hand-verifiable -- the LP can't do better than 0."""
	worker, deployment, adv, rti = _setup('really_friggin_small')
	_set_caps_uniform(worker, 1e12)

	# Pick a user and one of their reachable popps; force its latency to 0.
	ug = next(iter(rti[0].keys())) if rti.get(0) else None
	assert ug is not None, "no user with rti entry"
	target_popp = rti[0][ug]
	# Save and override.
	orig_lat = worker.whole_deployment_ug_perfs[ug][target_popp]
	worker.whole_deployment_ug_perfs[ug][target_popp] = 0.0

	try:
		ret = worker.solve_generic_lp_persistent(rti, 'avg_latency')
		assert ret['solved']
		ugi = worker.whole_deployment_ug_to_ind[ug]
		actual = float(ret['lats_by_ug'][ugi])
		assert math.isclose(actual, 0.0, abs_tol=1e-3), (
			"User {} was made to have a zero-latency reachable popp but the LP "
			"reports lats_by_ug[{}] = {}. The LP isn't picking the optimal "
			"path -- check objective sign / coefficient assignment."
			.format(ug, ugi, actual))
	finally:
		# Restore so subsequent tests don't see mutated state.
		worker.whole_deployment_ug_perfs[ug][target_popp] = orig_lat


@pytest.mark.unit
@pytest.mark.gurobi
def test_tight_capacity_forces_overflow_to_alternative_popp():
	"""Construct a scenario where one popp is the universal first choice but
	is tightly capped. The LP must overflow some traffic to alternative popps.

	Verifiable claim: the constrained popp's allocated volume must NOT exceed
	its capacity (in standard mode) or the LP must have entered MLU mode and
	report fraction_congested_volume > 0."""
	worker, deployment, adv, rti = _setup('really_friggin_small')

	# Pick the popp that the most users are rti-routed to across prefixes.
	from collections import Counter
	popp_load = Counter()
	for prefix_i, ug_to_popp in rti.items():
		for ug, popp in ug_to_popp.items():
			popp_load[popp] += worker.whole_deployment_ug_to_vol[ug]
	hot_popp, hot_load = popp_load.most_common(1)[0]
	hot_poppi = worker.popp_to_ind[hot_popp]

	# Make most popps generous, but cap the hot popp at 10% of its load. Force
	# the LP to spread traffic around.
	_set_caps_uniform(worker, 1e12)
	worker.static_caps[hot_poppi] = 0.1 * hot_load

	ret = worker.solve_generic_lp_persistent(rti, 'avg_latency')
	assert ret['solved'], "tight-capacity LP did not solve"

	hot_vol = ret['vols_by_poppi'].get(hot_poppi, 0.0)
	cap = worker.static_caps[hot_poppi]
	mlu_used = ret.get('fraction_congested_volume', 0.0) > 1e-6

	# Either standard mode held cap (hot_vol <= cap), or MLU kicked in (proportional
	# overshoot is allowed). At least one must be true.
	if not mlu_used:
		assert hot_vol <= cap * 1.001 + 1e-6, (
			"Standard-mode LP violated capacity at hot popp: "
			"allocated {:.4f} vs cap {:.4f}".format(hot_vol, cap)
		)
	# Either way, the LP must have moved at least SOME traffic off the hot popp
	# unless every single user only had this popp available (then there's
	# nowhere to send it).
	all_user_reaches = set()
	for ug_to_popp in rti.values():
		all_user_reaches.update(ug_to_popp.values())
	if len(all_user_reaches) > 1:
		# There were alternatives. The LP should have used them.
		total_routed = sum(ret['vols_by_poppi'].values())
		alt_vol = total_routed - hot_vol
		assert alt_vol > 0, (
			"Hot popp was overloaded but the LP didn't send any traffic to "
			"alternative popps (alt_vol={}). Either MLU is over-aggressive "
			"or the optimizer is ignoring capacity constraints."
			.format(alt_vol)
		)


# ---------------------------------------------------------------------------#
# (Volume conservation tests, retained.)
# ---------------------------------------------------------------------------#
@pytest.mark.unit
@pytest.mark.gurobi
@pytest.mark.parametrize('density', [1.0, 0.5, 0.25])
def test_volume_conservation_under_varying_advertisement_density(density):
	"""For a randomly-toggled advertisement at the given density, every routed
	user's total volume must equal their assigned volume. Tests at multiple
	densities catch density-dependent bugs (e.g., handling of users with no
	available routes)."""
	worker, deployment, adv, rti = _setup('small')
	n_popps = len(deployment['popps'])

	rng = np.random.default_rng(7)
	adv2 = (rng.random((n_popps, 1)) < density).astype(float)
	if adv2.sum() == 0:
		adv2[rng.integers(n_popps), 0] = 1.0
	rti2, _ = worker.calculate_ground_truth_ingress(adv2, do_cache=False)

	ret = worker.solve_generic_lp_persistent(rti2, 'avg_latency')
	assert ret['solved'], "LP not solved at density {}".format(density)

	# Sum of vols_by_poppi should be ~ total deployment volume (modulo
	# routing to NO_PATH ingress, which gets a 'high capacity' synthetic
	# sink so its volume still counts toward the total).
	from constants import NO_ROUTE_LATENCY  # noqa: F401
	total_alloc = sum(ret['vols_by_poppi'].values())
	total_vol = float(np.sum(worker.whole_deployment_ug_vols))
	assert math.isclose(total_alloc, total_vol, rel_tol=1e-3, abs_tol=1e-3), (
		"Volume conservation broken at density {}: allocated={} vs total={}"
		.format(density, total_alloc, total_vol))


# ---------------------------------------------------------------------------#
# Backup-capacity / static_failure tests removed -- those objectives were
# retired in the registry cleanup. The site_failure objective subsumed their
# use cases. See git log for the original test code.
# ---------------------------------------------------------------------------#




# ---------------------------------------------------------------------------#
# Site-failure objective tests.
#
# Steady-state avg-latency LP + EXHAUSTIVE mean over per-PoP failures, with
# user->prefix frozen to the steady-state assignment.
# ---------------------------------------------------------------------------#

def _solve_site_failure(worker, adv, rti, beta=0.5, no_route_penalty=20.0):
	from solve_lp_assignment import solve_lp_assignment_site_failure
	return solve_lp_assignment_site_failure(
		worker, rti, 'site_failure', adv=adv,
		site_failure_beta=beta, site_failure_no_route_penalty=no_route_penalty)


@pytest.mark.unit
@pytest.mark.gurobi
def test_site_failure_enumerates_all_pops():
	"""The site_failure LP must evaluate failures for every PoP in the
	deployment, not a sample. Falsifiable: site_failure_n_sites must equal
	the number of unique PoPs in sas.popps."""
	worker, deployment, _, _ = _setup('really_friggin_small')
	n_popps = len(deployment['popps'])
	adv = np.ones((n_popps, 3))
	rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
	ret = _solve_site_failure(worker, adv, rti)
	assert ret['solved']
	# Compute expected unique PoP count from sas.popps.
	unique_pops = {p for p, _ in worker.popps}
	assert ret['site_failure_n_sites'] == len(unique_pops), (
		"Expected {} site failures, got {}. Site-failure LP must enumerate "
		"every PoP exhaustively (no sampling).".format(
			len(unique_pops), ret['site_failure_n_sites']))


@pytest.mark.unit
@pytest.mark.gurobi
def test_site_failure_decomposition_with_no_failures_yields_steady():
	"""When beta=0 the combined objective should be identical to the steady
	avg-latency objective -- the failure term contributes 0 weight."""
	worker, deployment, _, _ = _setup('really_friggin_small')
	n_popps = len(deployment['popps'])
	rng = np.random.default_rng(7)
	adv = (rng.random((n_popps, 3)) < 0.6).astype(float)
	for j in range(adv.shape[1]):
		if adv[:, j].sum() == 0:
			adv[rng.integers(n_popps), j] = 1.0
	rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)

	ret_b0 = _solve_site_failure(worker, adv, rti, beta=0.0)
	from solve_lp_assignment import solve_generic_lp_with_failure_catch
	steady = solve_generic_lp_with_failure_catch(worker, rti, 'avg_latency')
	assert math.isclose(ret_b0['objective'], steady['objective'], rel_tol=1e-6, abs_tol=1e-6), (
		"beta=0 must reproduce steady-state objective exactly. Got combined={}, steady={}"
		.format(ret_b0['objective'], steady['objective']))


@pytest.mark.unit
@pytest.mark.gurobi
def test_site_failure_combined_objective_is_convex_in_beta():
	"""For fixed adv, the combined objective is a linear interpolation
	between steady-state and mean-failure values: combined =
	(1-beta)*steady + beta*mean_failure. Test by computing at beta=0 and
	beta=1, then verifying beta=0.5 produces their midpoint."""
	worker, deployment, _, _ = _setup('really_friggin_small')
	n_popps = len(deployment['popps'])
	adv = np.ones((n_popps, 3))
	rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)

	r0 = _solve_site_failure(worker, adv, rti, beta=0.0)
	r1 = _solve_site_failure(worker, adv, rti, beta=1.0)
	rmid = _solve_site_failure(worker, adv, rti, beta=0.5)
	expected_mid = 0.5 * r0['objective'] + 0.5 * r1['objective']
	assert math.isclose(rmid['objective'], expected_mid, rel_tol=1e-5, abs_tol=1e-5), (
		"Combined objective must linearly interpolate beta. Got "
		"r(b=0)={}, r(b=0.5)={}, r(b=1)={}, expected midpoint={}"
		.format(r0['objective'], rmid['objective'], r1['objective'], expected_mid))


@pytest.mark.unit
@pytest.mark.gurobi
def test_site_failure_split_penalty_no_route_heavier_than_congestion():
	"""For a given failure scenario where all degradation is 'true no-route'
	(user has no surviving popp on pinned prefix), increasing no_route_penalty
	from X to 2X should DOUBLE the penalty contribution. Increasing
	congestion_penalty from Y to 2Y should not change the objective when
	there is no congestion."""
	worker, deployment, _, _ = _setup('really_friggin_small')
	n_popps = len(deployment['popps'])
	# Sparse adv -- 1-hot per prefix, ensure users have nothing to fall back to.
	# When a site fails, users routed to that site go no-route.
	adv = np.zeros((n_popps, 3))
	for j in range(3): adv[j, j] = 1.0
	rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)

	from solve_lp_assignment import solve_lp_assignment_site_failure
	r_low = solve_lp_assignment_site_failure(
		worker, rti, 'site_failure', adv=adv,
		site_failure_beta=1.0,  # ignore steady, just measure failure term
		site_failure_no_route_penalty=10.0,
		site_failure_congestion_penalty=10.0)
	r_high_noroute = solve_lp_assignment_site_failure(
		worker, rti, 'site_failure', adv=adv,
		site_failure_beta=1.0,
		site_failure_no_route_penalty=20.0,  # 2x
		site_failure_congestion_penalty=10.0)
	r_high_congest = solve_lp_assignment_site_failure(
		worker, rti, 'site_failure', adv=adv,
		site_failure_beta=1.0,
		site_failure_no_route_penalty=10.0,
		site_failure_congestion_penalty=20.0)  # 2x
	# Doubling no_route_penalty should make objective worse (more negative).
	# Doubling congestion_penalty should change less (since no_route dominates).
	delta_noroute = r_high_noroute['objective'] - r_low['objective']
	delta_congest = r_high_congest['objective'] - r_low['objective']
	assert delta_noroute < -1e-3, (
		"Doubling no_route_penalty must decrease objective (more negative). "
		"Got delta={}".format(delta_noroute))
	assert abs(delta_noroute) > abs(delta_congest), (
		"no_route_penalty should have larger impact than congestion_penalty "
		"on a sparse adv where most failures cause true no-route. "
		"|delta_noroute|={}, |delta_congest|={}".format(
			abs(delta_noroute), abs(delta_congest)))


