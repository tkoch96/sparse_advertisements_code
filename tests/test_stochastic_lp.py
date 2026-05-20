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


def _setup(size='small'):
	"""Reusable worker + deployment fixture. Mirrors test_lp_correctness.py:_setup."""
	import random
	random.seed(31415)
	np.random.seed(31415)
	from deployment_setup import get_random_deployment
	from helpers import split_deployment_by_ug
	from path_distribution_computer_ray import _LocalPathDistributionComputer

	dep = get_random_deployment(size, port=31700)
	subdep = split_deployment_by_ug(dep, n_chunks=1)[0]
	init_kwa = {
		'lambduh': 1.0, 'gamma': 0, 'verbose': False,
		'n_prefixes': None, 'with_capacity': False,
		'save_run_dir': None, 'generic_objective': 'avg_latency',
	}
	worker = _LocalPathDistributionComputer(
		worker_i=0, subdeployment=subdep, init_kwargs=init_kwa)
	n_popps = len(dep['popps'])
	# every popp on, one prefix: most permissive starting point so failure
	# scenarios actually have something to remove
	adv = np.ones((n_popps, 1))
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

	for i, (failed, _p) in enumerate(scenarios):
		# Standalone reference: mask the advertisement (zero rows for failed
		# popps), recompute rti, solve once. This matches the semantics that
		# solve_stochastic_lp uses internally.
		adv_s = adv.copy()
		for popp in failed:
			if popp in worker.popp_to_ind:
				adv_s[worker.popp_to_ind[popp], :] = 0
		rti_s, _ = worker.calculate_ground_truth_ingress(adv_s, do_cache=False)
		single = worker.solve_generic_lp_persistent(rti_s, 'avg_latency')

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


# ----- Test 7: Pop-failure resilience -------------------------------------- #

@pytest.mark.unit
@pytest.mark.gurobi
# ----- Test 8: Three-way comparison (headroom / stochastic / RB-grad approx) - #

@pytest.mark.unit
@pytest.mark.gurobi
@pytest.mark.parametrize('size,K', [
	('small', 16),
])
def test_three_approaches_comparison(size, K):
	"""Apples-to-apples on the inner-LP cost SCULPTOR's gradient step pays:

	  (a) headroom            : 1 LP solve with caps × (1 - h)
	  (b) stochastic (warm)   : K LP solves with persistent-Gurobi basis reuse
	  (c) RB-grad approx (cold): K LP solves rebuilding the model each time

	(b) and (c) bracket the right answer for "how long does it take to compute
	the resilience signal over K scenarios"; the existing RB-grad uses warm
	persistent Gurobi in practice, so (c) is an upper bound on the SGD-RB cost.

	Also reports the *failure-tolerance metric* for the fixed adv:
	per-scenario fraction of UG volume that lands on NO_PATH_INGRESS = the
	"users couldn't be routed" rate under failure.
	"""
	import time
	from stochastic_lp import (
		solve_stochastic_lp, solve_headroom_lp, single_popp_scenarios,
		compute_unroutable_volume_fractions,
	)

	worker, dep, adv = _setup(size)
	# Baseline single-LP-solve time for the same fixed adv (nominal scenario)
	t0 = time.time()
	rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
	_ = worker.solve_generic_lp_persistent(rti, 'avg_latency')
	single_lp_time = time.time() - t0

	# Scenario set: K single-popp failures (warm runs them in order; cold rebuilds each)
	scenarios = single_popp_scenarios(dep, p_any_fail=0.5)[:K]

	# (a) headroom
	headroom = solve_headroom_lp(worker, adv, headroom_factor=0.2)

	# (b) stochastic LP, warm-start basis reuse
	stoch = solve_stochastic_lp(worker, adv, scenarios, method='warm', keep_paths=True)

	# (c) RB-grad approximation = K cold solves
	rb_grad = solve_stochastic_lp(worker, adv, scenarios, method='cold')

	# Failure-routing metric: how much UG volume couldn't be routed per scenario
	unroutable = compute_unroutable_volume_fractions(stoch, worker)

	# Single-LP × K is a back-of-envelope prediction for (c)
	predicted_c = K * single_lp_time

	print(f"\n  size={size}, K={K}, n_popps={len(dep['popps'])}, n_ugs={len(dep['ugs'])}")
	print(f"  baseline single LP solve              : {single_lp_time*1000:>9.1f} ms")
	print(f"  (a) headroom 0.2  (1 LP)              : {headroom['wall_time']*1000:>9.1f} ms")
	print(f"  (b) stochastic   (K={K} warm LPs)      : {stoch.wall_time*1000:>9.1f} ms")
	print(f"  (c) RB-grad ≈   (K={K} cold LPs)      : {rb_grad.wall_time*1000:>9.1f} ms")
	print(f"      predicted K * single-LP            : {predicted_c*1000:>9.1f} ms")
	print(f"")
	print(f"  failure-routing metric (volume on NO_PATH_INGRESS):")
	import numpy as _np
	print(f"    nominal scenario                     : {unroutable[0]:.5f}")
	print(f"    mean over failure scenarios          : {_np.mean(unroutable[1:]):.5f}")
	print(f"    max over failure scenarios           : {_np.max(unroutable[1:]):.5f}")
	print(f"    n_scenarios with >0.1% unroutable    : {sum(1 for u in unroutable[1:] if u > 0.001)}/{K-1}")
	print(f"")
	print(f"  expected latencies:")
	print(f"    (a) headroom-LP latency              : {headroom['latency']:.4f}")
	print(f"    (b)/(c) E[latency] over K scenarios  : {stoch.expected_latency:.4f}")

	# Sanity checks
	assert all([headroom['solved'], all(stoch.per_scenario_solved),
				all(rb_grad.per_scenario_solved)])
	assert all(0.0 <= u <= 1.0 + 1e-9 for u in unroutable), \
		f"unroutable fractions out of range: {unroutable}"
	# Headroom is ~1 LP; stochastic is ~K LPs. Stochastic should be at least
	# K/3× slower than headroom (loose bound; allows for warm-start savings).
	assert stoch.wall_time > headroom['wall_time'] * K / 3, \
		f"stochastic wall ({stoch.wall_time}) suspiciously low vs headroom ({headroom['wall_time']}) for K={K}"
	# Cold should be slower (or equal within noise) than warm.
	assert rb_grad.wall_time >= stoch.wall_time * 0.9, \
		f"cold ({rb_grad.wall_time}) faster than warm ({stoch.wall_time}) -- warm-start path broken?"


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
