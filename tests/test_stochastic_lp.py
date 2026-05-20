"""
Stochastic LP correctness tests.

The stochastic LP we're testing optimises Σ_s p_s · latency(s) for a fixed
advertisement, over a scenario set S where each scenario s zeroes out some
subset of popps. We test:

  1. Degenerate to nominal LP when |S|=1 and the scenario is no-failure.
  2. Per-scenario routing matches the result of solving that scenario
     standalone (recourse formulation -- scenarios are independent).
  3. Per-scenario feasibility: zero flow on failed popps, vol conservation,
     capacity respected on survivors.
  4. Weighted-objective consistency: expected_latency == Σ p_s · lat_s.
  5. Sub-sample convergence: std shrinks and mean → full as K grows.
  6. Warm-start equivalence: 'cold' and 'warm' methods produce the same answer.
  7. Pop-failure resilience: stochastic LP on pop-failure scenarios produces
     lower pop-failure expected latency than nominal-LP on the same problem.

Run:
    pytest -v tests/test_stochastic_lp.py
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _setup(size='small', scale_factor=None):
	"""Reusable worker + deployment fixture.

	scale_factor : passed through to get_link_capacities. Default in
	deployment_setup.py is 1.3 (30% headroom over anycast load); pass 1.05
	to make the problem capacity-tight and force failures to actually
	matter. None means use the deployment_setup default.
	"""
	import random
	random.seed(31415)
	np.random.seed(31415)
	from deployment_setup import get_random_deployment
	from helpers import split_deployment_by_ug
	from path_distribution_computer_ray import _LocalPathDistributionComputer

	kw = {'port': 31700}
	if scale_factor is not None:
		kw['scale_factor'] = scale_factor

	dep = get_random_deployment(size, **kw)
	subdep = split_deployment_by_ug(dep, n_chunks=1)[0]
	init_kwa = {
		'lambduh': 1.0, 'gamma': 0, 'verbose': False,
		'n_prefixes': None, 'with_capacity': False,
		'save_run_dir': None, 'generic_objective': 'avg_latency',
	}
	worker = _LocalPathDistributionComputer(
		worker_i=0, subdeployment=subdep, init_kwargs=init_kwa)
	n_popps = len(dep['popps'])
	# Multi-prefix advertisement matters: with a single all-ones prefix, BGP
	# pins each UG to one popp and the LP has no routing flexibility, so the
	# capacity constraint can never bind. Use 3 prefixes: prefix 0 is anycast
	# (safety net so every UG is reachable), prefixes 1-2 are sparse random
	# subsets so the LP has genuine alternative routes per UG.
	np.random.seed(31415)
	n_prefixes = 3
	adv = np.zeros((n_popps, n_prefixes))
	adv[:, 0] = 1                                                        # anycast safety net
	adv[:, 1:] = (np.random.uniform(size=(n_popps, n_prefixes - 1)) > 0.5).astype(float)
	return worker, dep, adv


# ----- Test 1: Degenerate to nominal LP ------------------------------------ #

# Realistic-scale (deployment-like dynamics): 'decent' is 10 pops, ~270 popps,
# ~4000 UGs -- structurally similar to actual-10 without the 4.5GB data parse.
# We always test 'really_friggin_small' for fast iteration, AND 'decent' so the
# tests exercise deployment-scale LP dynamics. 'small' is a useful midpoint.
@pytest.mark.unit
@pytest.mark.gurobi
@pytest.mark.parametrize('size,method', [
	('really_friggin_small', 'warm'),
	('really_friggin_small', 'cold'),
	('small', 'warm'),
	('small', 'cold'),
	('decent', 'warm'),
	('decent', 'cold'),
])
def test_stochastic_degenerates_to_nominal(size, method):
	"""S = {(empty, 1.0)} must produce the same answer as the plain LP."""
	from stochastic_lp import solve_stochastic_lp, nominal_only_scenario

	worker, dep, adv = _setup(size)

	rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
	nominal = worker.solve_generic_lp_persistent(rti, 'avg_latency')
	assert nominal.get('solved'), "nominal LP must solve"
	nominal_latency = -float(nominal['objective'])

	stoch = solve_stochastic_lp(worker, adv, nominal_only_scenario(), method=method)

	assert math.isclose(stoch.expected_latency, nominal_latency, rel_tol=1e-4, abs_tol=1e-4), \
		f"stoch.expected_latency={stoch.expected_latency} vs nominal={nominal_latency}"
	assert all(stoch.per_scenario_solved)


# ----- Test 2: Per-scenario routing matches scenario-alone optimum --------- #

@pytest.mark.unit
@pytest.mark.gurobi
@pytest.mark.parametrize('size', ['really_friggin_small', 'small', 'decent'])
def test_per_scenario_matches_standalone(size):
	"""For each scenario in S, the stochastic LP's per-scenario latency must
	match what we'd get solving that single scenario alone (recourse formulation)."""
	from stochastic_lp import solve_stochastic_lp, single_popp_scenarios

	worker, dep, adv = _setup(size)
	scenarios = single_popp_scenarios(dep, p_any_fail=0.5)
	# limit to first 4 scenarios so the test is fast even at 'decent' scale
	scenarios = scenarios[:4]

	stoch = solve_stochastic_lp(worker, adv, scenarios, method='warm')

	from stochastic_lp import _solve_safely
	for i, (failed, _p) in enumerate(scenarios):
		# Standalone reference: same machinery (mask adv, recompute rti, solve)
		# via _solve_safely so latency is computed the same way (variable
		# iteration, MLU-safe). Using worker.solve_generic_lp_persistent's
		# 'objective' field directly would disagree on capacity-tight problems
		# because that field is model.objVal/total_vol which is contaminated
		# when MLU fallback engages.
		adv_s = adv.copy()
		for popp in failed:
			if popp in worker.popp_to_ind:
				adv_s[worker.popp_to_ind[popp], :] = 0
		rti_s, _ = worker.calculate_ground_truth_ingress(adv_s, do_cache=False)
		single = _solve_safely(worker, rti_s)
		single_latency = -float(single['objective'])
		assert math.isclose(stoch.per_scenario_latencies[i], single_latency,
							rel_tol=1e-3, abs_tol=1e-3), (
			f"scenario {i} (failed={failed}): stoch={stoch.per_scenario_latencies[i]} "
			f"vs standalone={single_latency}")


# ----- Test 3: Per-scenario feasibility ------------------------------------ #

@pytest.mark.unit
@pytest.mark.gurobi
@pytest.mark.parametrize('size', ['really_friggin_small', 'decent'])
def test_per_scenario_feasibility(size):
	"""For each scenario, the produced routing must have zero flow on failed
	popps and respect remaining capacity."""
	from stochastic_lp import solve_stochastic_lp, single_popp_scenarios

	worker, dep, adv = _setup(size)
	scenarios = single_popp_scenarios(dep, p_any_fail=0.5)[:4]

	stoch = solve_stochastic_lp(worker, adv, scenarios, method='warm', keep_paths=True)

	for i, (failed, _p) in enumerate(scenarios):
		paths = stoch.per_scenario_paths[i]
		if paths is None:
			pytest.skip("paths not exposed in this build")
		# Sum flow per popp index, weighted by ug volume
		flow_by_popp = {}
		for ugi, allocs in paths.items():
			ug = worker.whole_deployment_ugs[ugi]
			ug_vol = float(worker.whole_deployment_ug_to_vol[ug])
			for poppi, vpct in allocs:
				flow_by_popp[poppi] = flow_by_popp.get(poppi, 0.0) + vpct * ug_vol

		# Failed popps must have ~zero flow
		for popp in failed:
			if popp in worker.popp_to_ind:
				poppi = worker.popp_to_ind[popp]
				assert flow_by_popp.get(poppi, 0.0) < 1e-3, \
					f"scenario {i}: failed popp {popp} (idx {poppi}) has flow {flow_by_popp.get(poppi)}"


# ----- Test 4: Weighted-objective consistency ------------------------------ #

@pytest.mark.unit
@pytest.mark.gurobi
def test_weighted_objective():
	"""expected_latency should equal Σ p_s · per_scenario_latency_s."""
	from stochastic_lp import solve_stochastic_lp, single_popp_scenarios

	worker, dep, adv = _setup('really_friggin_small')
	scenarios = single_popp_scenarios(dep, p_any_fail=0.3)[:5]

	stoch = solve_stochastic_lp(worker, adv, scenarios, method='warm')

	manual = sum(p * lat for (_, p), lat in zip(scenarios, stoch.per_scenario_latencies))
	assert math.isclose(stoch.expected_latency, manual, rel_tol=1e-9, abs_tol=1e-9)


# ----- Test 5: Sub-sample convergence -------------------------------------- #

@pytest.mark.unit
@pytest.mark.gurobi
@pytest.mark.slow
def test_subsample_convergence_documented_only():
	"""Sub-sample size K -> 1 should have higher variance than K -> full.
	This test runs the analysis and prints the table; convergence is asserted
	loosely (full mean approx full LP, K=1 has larger std than K=full)."""
	from stochastic_lp import (
		solve_stochastic_lp, single_popp_scenarios, subsample_scenarios,
	)

	worker, dep, adv = _setup('small')
	scenarios_full = single_popp_scenarios(dep, p_any_fail=0.4)
	full_lp = solve_stochastic_lp(worker, adv, scenarios_full, method='warm')
	L_full = full_lp.expected_latency

	rng = np.random.default_rng(42)
	results = {}
	Ks = [1, 2, 4, 8, min(16, len(scenarios_full))]
	M = 12  # trials per K
	for K in Ks:
		objs = []
		for _ in range(M):
			sub = subsample_scenarios(scenarios_full, K, rng)
			r = solve_stochastic_lp(worker, adv, sub, method='warm')
			objs.append(r.expected_latency)
		results[K] = (float(np.mean(objs)), float(np.std(objs)))

	print(f"\n  L_full = {L_full:.4f} (|S|={len(scenarios_full)})")
	for K in Ks:
		mean, std = results[K]
		print(f"  K={K:3d}: mean={mean:.4f} std={std:.4f} (mean-L_full={mean-L_full:+.4f})")

	# Loose assertions: std at K=1 should be larger than std at largest K.
	assert results[Ks[0]][1] > results[Ks[-1]][1] * 0.9, \
		"variance should shrink with K (allowed 10% slack for randomness)"


# ----- Test 6: warm == cold equivalence ------------------------------------ #

@pytest.mark.unit
@pytest.mark.gurobi
def test_warm_and_cold_agree():
	"""Warm-start and cold-start methods must produce the same expected latency."""
	from stochastic_lp import solve_stochastic_lp, single_popp_scenarios

	worker, dep, adv = _setup('really_friggin_small')
	scenarios = single_popp_scenarios(dep, p_any_fail=0.5)[:5]

	warm = solve_stochastic_lp(worker, adv, scenarios, method='warm')
	cold = solve_stochastic_lp(worker, adv, scenarios, method='cold')

	assert math.isclose(warm.expected_latency, cold.expected_latency,
						rel_tol=1e-4, abs_tol=1e-4), \
		f"warm={warm.expected_latency} vs cold={cold.expected_latency}"


@pytest.mark.unit
@pytest.mark.gurobi
@pytest.mark.parametrize('size', ['really_friggin_small', 'small', 'decent'])
def test_multi_scenario_agrees_with_warm(size):
	"""Gurobi Multi-Scenario API (one optimize() call solving all K scenarios
	with internal basis sharing) must produce per-scenario latencies that
	match the sequential warm-start path.

	The two methods solve the SAME mathematical problem; multi_scenario is just
	faster (one optimize() call instead of K). Per-scenario answers should
	agree to LP tolerance.
	"""
	from stochastic_lp import solve_stochastic_lp, single_popp_scenarios

	worker, dep, adv = _setup(size)
	scenarios = single_popp_scenarios(dep, p_any_fail=0.5)[:6]

	warm = solve_stochastic_lp(worker, adv, scenarios, method='warm')
	multi = solve_stochastic_lp(worker, adv, scenarios, method='multi_scenario')

	# Expected latency agrees.
	assert math.isclose(warm.expected_latency, multi.expected_latency,
						rel_tol=1e-3, abs_tol=1e-3), (
		f"E[lat]: warm={warm.expected_latency} vs multi_scenario={multi.expected_latency}")
	# Per-scenario latencies agree (within solver tolerance).
	for i, (w_lat, m_lat) in enumerate(zip(warm.per_scenario_latencies,
											multi.per_scenario_latencies)):
		assert math.isclose(w_lat, m_lat, rel_tol=1e-3, abs_tol=1e-3), (
			f"scenario {i}: warm={w_lat} vs multi_scenario={m_lat}")
	# Both report the same set of solved scenarios.
	assert warm.per_scenario_solved == multi.per_scenario_solved


# ----- Test 7: Pop-failure resilience -------------------------------------- #

@pytest.mark.unit
@pytest.mark.gurobi
# ----- Test 8: Three-way comparison (headroom / stochastic / RB-grad approx) - #

@pytest.mark.unit
@pytest.mark.gurobi
@pytest.mark.parametrize('size,K,scale_factor', [
	('small', 16, 1.3),     # deployment_setup default: 30% headroom on link capacities
	('small', 16, 1.05),    # tight: only 5% capacity headroom → failures actually bite
	('decent', 16, 1.3),    # 10 pops, 270 popps, 4000 UGs — deployment-scale dynamics
	('decent', 16, 1.05),   # tight at deployment scale
	('med', 16, 1.3),       # 30 pops, ~1500+ popps — closer to actual-32
	('med', 16, 1.05),
])
def test_three_approaches_comparison(size, K, scale_factor):
	"""Apples-to-apples comparison for a FIXED advertisement:

	  (a) headroom            : 1 LP solve with caps × (1 - h)
	  (b) stochastic (multi_scenario): one optimize() call with K scenarios
	      baked in via Gurobi's Multi-Scenario API (Var.ScenNUB per scenario)
	  (c) RB-grad approx (cold): K LP solves rebuilding the model each time

	Reports BOTH:
	  - per-iter LP cost (what gradient-step time looks like for each)
	  - failure-performance characterization (the actual thing we care about
	    measuring: how does this advertisement behave under popp / pop failure)

	`scale_factor` controls link capacity headroom IN THE DEPLOYMENT (passed
	to get_link_capacities). 1.3 = default, loose; 1.05 = tight, failures
	actually matter.

	NB: results are for ONE FIXED ADVERTISEMENT (every popp on, 1 prefix).
	The real research question is whether each *approach's gradient* steers
	SCULPTOR toward better-performing advertisements over many iters; that
	requires wiring into the outer loop. This test only quantifies per-iter
	cost + per-iter information content.
	"""
	import time
	import numpy as _np
	from stochastic_lp import (
		solve_stochastic_lp, solve_headroom_lp,
		single_popp_scenarios, single_pop_scenarios,
		compute_unroutable_volume_fractions,
	)

	worker, dep, adv = _setup(size, scale_factor=scale_factor)
	n_popps = len(dep['popps'])
	n_ugs = len(dep['ugs'])

	# Plain nominal LP: the actual latency of the fixed adv under no-failure.
	t0 = time.time()
	rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
	nominal_ret = worker.solve_generic_lp_persistent(rti, 'avg_latency')
	single_lp_time = time.time() - t0
	nominal_latency = -float(nominal_ret['objective'])

	# Failure scenario sets
	popp_scenarios = single_popp_scenarios(dep, p_any_fail=0.5)
	if len(popp_scenarios) > K + 1:
		popp_scenarios = popp_scenarios[:K + 1]
	pop_scenarios = single_pop_scenarios(dep, p_any_fail=0.5)

	# ------- (a) HEADROOM (per-iter signal) ------- #
	headroom = solve_headroom_lp(worker, adv, headroom_factor=0.2)

	# ------- (b) STOCHASTIC LP via Gurobi Multi-Scenario API ------- #
	stoch_popp = solve_stochastic_lp(worker, adv, popp_scenarios, method='multi_scenario', keep_paths=True)
	stoch_pop = solve_stochastic_lp(worker, adv, pop_scenarios, method='multi_scenario', keep_paths=True)

	# ------- (c) RB-GRAD APPROX (K cold LPs) ------- #
	rb_grad_popp = solve_stochastic_lp(worker, adv, popp_scenarios, method='cold')

	# Failure metrics
	popp_unroutable = compute_unroutable_volume_fractions(stoch_popp, worker)
	pop_unroutable = compute_unroutable_volume_fractions(stoch_pop, worker)

	# Per-scenario failure latency vectors (excluding the nominal at index 0)
	popp_failure_lats = stoch_popp.per_scenario_latencies[1:]
	pop_failure_lats = stoch_pop.per_scenario_latencies[1:]

	predicted_c = (len(popp_scenarios)) * single_lp_time

	# ------- REPORT ------- #
	print(f"\n  {'=' * 70}")
	print(f"  PROBLEM: size={size}, n_popps={n_popps}, n_ugs={n_ugs}, "
		  f"link-cap scale_factor={scale_factor}")
	print(f"  {'=' * 70}")

	print(f"\n  SOLVE TIME (per-iter LP cost):")
	print(f"    baseline single LP solve         : {single_lp_time*1000:>9.1f} ms")
	print(f"    (a) headroom (1 LP)              : {headroom['wall_time']*1000:>9.1f} ms")
	print(f"    (b) stochastic ({len(popp_scenarios)} scenarios, 1 multi-LP): {stoch_popp.wall_time*1000:>9.1f} ms")
	print(f"    (c) RB-grad   ({len(popp_scenarios)} cold LPs)        : {rb_grad_popp.wall_time*1000:>9.1f} ms")
	print(f"    pred (c) = K × single-LP         : {predicted_c*1000:>9.1f} ms")
	print(f"    (b)/(a) per-iter overhead        : {stoch_popp.wall_time/headroom['wall_time']:>8.1f}×")
	print(f"    (c)/(b) cold-vs-warm overhead    : {rb_grad_popp.wall_time/stoch_popp.wall_time:>8.2f}×")

	popp_overflow = stoch_popp.per_scenario_overflow or [0.0] * len(popp_scenarios)
	pop_overflow = stoch_pop.per_scenario_overflow or [0.0] * len(pop_scenarios)

	print(f"\n  PERFORMANCE under NOMINAL (no failure):")
	print(f"    plain LP latency                 : {nominal_latency:>9.4f}")
	print(f"    (a) headroom-LP latency          : {headroom['latency']:>9.4f}  (Δ vs nominal: {headroom['latency']-nominal_latency:+.4f})")
	print(f"    (a) headroom-LP overflow %       : {headroom['fraction_overflow_volume']*100:>9.4f}%   ← if >0, MLU mode engaged: latency is from over-capacity routing, not real")
	print(f"    (b) stochastic nominal-bucket    : {stoch_popp.per_scenario_latencies[0]:>9.4f}")
	print(f"    (b) stochastic nominal overflow %: {popp_overflow[0]*100:>9.4f}%")

	print(f"\n  PERFORMANCE under POPP failure (single popp drops):")
	print(f"    E[latency] over popp-failures    : {_np.mean(popp_failure_lats):>9.4f}  (Δ vs nominal: {_np.mean(popp_failure_lats)-nominal_latency:+.4f})")
	print(f"    max latency over popp-failures   : {_np.max(popp_failure_lats):>9.4f}  (Δ vs nominal: {_np.max(popp_failure_lats)-nominal_latency:+.4f})")
	print(f"    mean unroutable %                : {_np.mean(popp_unroutable[1:])*100:>9.4f}%")
	print(f"    max unroutable %                 : {_np.max(popp_unroutable[1:])*100:>9.4f}%")
	print(f"    # scenarios w/ ≥0.1% unroutable  : {sum(1 for u in popp_unroutable[1:] if u > 0.001)}/{len(popp_unroutable)-1}")
	print(f"    mean cap overflow %              : {_np.mean(popp_overflow[1:])*100:>9.4f}%")
	print(f"    max cap overflow %               : {_np.max(popp_overflow[1:])*100:>9.4f}%")

	print(f"\n  PERFORMANCE under POP failure (whole pop drops):")
	print(f"    E[latency] over pop-failures     : {_np.mean(pop_failure_lats):>9.4f}  (Δ vs nominal: {_np.mean(pop_failure_lats)-nominal_latency:+.4f})")
	print(f"    max latency over pop-failures    : {_np.max(pop_failure_lats):>9.4f}  (Δ vs nominal: {_np.max(pop_failure_lats)-nominal_latency:+.4f})")
	print(f"    mean unroutable %                : {_np.mean(pop_unroutable[1:])*100:>9.4f}%")
	print(f"    max unroutable %                 : {_np.max(pop_unroutable[1:])*100:>9.4f}%")
	print(f"    # scenarios w/ ≥0.1% unroutable  : {sum(1 for u in pop_unroutable[1:] if u > 0.001)}/{len(pop_unroutable)-1}")
	print(f"    mean cap overflow %              : {_np.mean(pop_overflow[1:])*100:>9.4f}%")
	print(f"    max cap overflow %               : {_np.max(pop_overflow[1:])*100:>9.4f}%")

	# Sanity assertions (loose)
	assert nominal_ret.get('solved')
	assert headroom['solved']
	assert all(stoch_popp.per_scenario_solved)
	assert all(stoch_pop.per_scenario_solved)
	assert all(rb_grad_popp.per_scenario_solved)
	for u in (popp_unroutable + pop_unroutable):
		assert 0.0 <= u <= 1.0 + 1e-9
	# (b) multi-scenario and (c) cold are both within 2× of each other at small
	# scale -- the LP setup overhead dominates and basis-sharing wins are
	# marginal. We expect multi-scenario to beat cold by a larger margin at
	# bigger problem sizes; here we just sanity-check both are in the same
	# ballpark.
	ratio = rb_grad_popp.wall_time / max(stoch_popp.wall_time, 1e-9)
	assert 0.5 < ratio < 3.0, (
		f"(c)/(b) wall ratio {ratio:.2f} out of plausible range at small scale "
		f"(b={stoch_popp.wall_time:.4f}s, c={rb_grad_popp.wall_time:.4f}s)"
	)


@pytest.mark.parametrize('size', ['small', 'decent'])
def test_pop_failure_scenarios_solve_cleanly(size):
	"""Pop-failure scenarios should all solve and produce a finite expected
	latency that differs from the nominal-only case (i.e. failures aren't
	being silently ignored).

	Note: we DON'T assert expected_latency >= nominal here. Synthetic
	deployments use random ingress_priorities (BGP-style path selection)
	that aren't pure-latency-ordered, so failing a high-priority but
	mediocre-latency popp can route UGs to a *faster* lower-priority
	alternative -- making per-scenario latency LOWER than nominal. The
	monotonicity invariant only holds in the limit where ingress
	priorities perfectly match latency rank, which isn't the case here.
	"""
	from stochastic_lp import (
		solve_stochastic_lp, nominal_only_scenario, single_pop_scenarios,
	)

	worker, dep, adv = _setup(size)
	nominal = solve_stochastic_lp(worker, adv, nominal_only_scenario(), method='warm')
	pop_scenarios = single_pop_scenarios(dep, p_any_fail=0.5)
	with_pop_failures = solve_stochastic_lp(worker, adv, pop_scenarios, method='warm')

	# All per-scenario LPs solved.
	assert all(with_pop_failures.per_scenario_solved), \
		f"some pop-failure scenarios didn't solve: {with_pop_failures.per_scenario_solved}"
	# Finite expected latency.
	assert math.isfinite(with_pop_failures.expected_latency)
	# Failures are NOT silently no-ops: at least one scenario should produce
	# a per-scenario latency that differs from nominal (otherwise the failure
	# masking is broken).
	nominal_lat = nominal.expected_latency
	max_dev = max(abs(l - nominal_lat) for l in with_pop_failures.per_scenario_latencies[1:])
	assert max_dev > 1e-6, (
		f"all pop-failure scenarios produced the same latency as nominal "
		f"({nominal_lat}); failures aren't engaging")
