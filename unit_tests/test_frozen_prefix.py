"""frozen_prefix LP correctness tests (see core/frozen_prefix.py).

The frozen_prefix objective is a single joint LP over w(ug, prefix) shared
across the normal scenario and sampled popp-failure scenarios. These tests
verify the semantics the paper will lean on:
  - standard LP-return invariants (conservation, lats consistency)
  - gamma=0 degenerates to soft steady latency (no failure influence)
  - the objective scalar is BOUNDED (never NO_ROUTE_LATENCY marker scale)
  - failure scenarios price no-route/overflow with the soft two-component
    penalties, no-route > congestion
  - determinism: identical inputs -> identical objective
  - the driver-side kill sampler rotates across iterations, is stable
    within one, and respects the sample-size knob

Run:
	SCULPTOR_XOBJS=1 pytest -v unit_tests/test_frozen_prefix.py
"""
import math
import os

import numpy as np
import pytest

os.environ.setdefault('SCULPTOR_XOBJS', '1')

from unit_tests.test_lp_correctness import _setup, _check_invariants


def _solve(worker, adv, rti, **kw):
	from core.frozen_prefix import solve_lp_frozen_prefix
	return solve_lp_frozen_prefix(worker, rti, 'frozen_prefix', adv=adv, **kw)


@pytest.mark.unit
def test_frozen_prefix_registered():
	"""SCULPTOR_XOBJS=1 must make 'frozen_prefix' dispatchable via the
	generic LP registry (the paper cells select objectives by this name)."""
	import core.hard_objectives as ho
	ho.register()
	from core.solve_lp_assignment import generic_lp_functions
	assert 'frozen_prefix' in generic_lp_functions


@pytest.mark.unit
def test_frozen_prefix_lp_invariants():
	worker, dep, adv, rti = _setup()
	ret = _solve(worker, adv, rti)
	_check_invariants(worker, dep, ret, ctx_label='frozen_prefix')
	# frozen-specific contract fields
	assert ret['frozen_prefix_n_fail'] > 0
	assert 0.0 <= ret['frozen_prefix_fail_no_route_frac'] <= 1.0
	assert ret['frozen_prefix_no_route_penalty'] > ret['frozen_prefix_congestion_penalty'], \
		'no-route must price above congestion (Tom 2026-09-05)'


@pytest.mark.unit
def test_frozen_prefix_objective_bounded():
	"""Objective scalar must never be marker scale (gradient-stability
	rule): even killing EVERY popp only prices the soft penalties."""
	from helpers.constants import NO_ROUTE_LATENCY
	worker, dep, adv, rti = _setup()
	ret = _solve(worker, adv, rti, frozen_kill_popps=list(range(worker.n_popps)))
	assert abs(ret['objective']) < 0.5 * NO_ROUTE_LATENCY
	# single prefix advertised on all popps: killing any ONE popp still
	# leaves survivors, so nothing should be no-route
	ret_one = _solve(worker, adv, rti, frozen_kill_popps=[0])
	assert ret_one['frozen_prefix_fail_no_route_frac'] < 1e-6


@pytest.mark.unit
def test_frozen_prefix_gamma_zero_is_steady_latency():
	"""With gamma=0 the scalar must equal -(vol-weighted normal latency)
	minus any normal-scenario overflow penalty -- failure scenarios must
	contribute nothing."""
	worker, dep, adv, rti = _setup()
	ret = _solve(worker, adv, rti, frozen_gamma=0.0)
	vols = np.array([worker.whole_deployment_ug_to_vol[ug]
					 for ug in worker.whole_deployment_ugs])
	lats = np.asarray(ret['lats_by_ug'])
	from helpers.constants import NO_ROUTE_LATENCY
	routed = lats < NO_ROUTE_LATENCY - 1e-9
	avg_lat = float(np.average(lats[routed], weights=vols[routed]))
	expected = -(avg_lat
				 + ret['frozen_prefix_congestion_penalty']
				 * ret['frozen_prefix_normal_overflow_frac'])
	assert math.isclose(ret['objective'], expected, rel_tol=1e-3, abs_tol=1e-2), \
		'gamma=0 objective {} != steady expectation {}'.format(
			ret['objective'], expected)


@pytest.mark.unit
def test_frozen_prefix_no_route_pricing():
	"""Kill the ONLY popp of a user's only prefix -> that user's volume is
	no-route in the failure scenario and priced at the no-route penalty."""
	worker, dep, adv, rti = _setup()
	# Advertise each popp on its own prefix (one-per-peering shape): every
	# user's per-prefix fallback set is a single popp, so killing popp j
	# makes prefix j dead for everyone using it.
	n_popps = worker.n_popps
	opp = np.eye(n_popps)
	opp_rti, _ = worker.calculate_ground_truth_ingress(opp, do_cache=False)
	ret = _solve(worker, opp, opp_rti, frozen_kill_popps=[0],
				 frozen_gamma=1.0)
	assert ret.get('solved'), 'one-per-peering frozen LP unsolved'
	# users the LP placed on prefix 0 (winner popp 0) lose their route
	# when popp 0 dies; if anyone was placed there, no-route frac > 0
	placed_on_0 = any(poppi == 0 for allocs in ret['paths_by_ug'].values()
					  for poppi, _ in allocs)
	if placed_on_0:
		assert ret['frozen_prefix_fail_no_route_frac'] > 0
	# and the LP had the option to avoid prefix 0: with a heavy no-route
	# penalty, raising P_nr must not IMPROVE the objective
	ret_heavy = _solve(worker, opp, opp_rti, frozen_kill_popps=[0],
					   frozen_gamma=1.0, frozen_no_route_penalty=200.0)
	assert ret_heavy['objective'] <= ret['objective'] + 1e-6


@pytest.mark.unit
def test_frozen_prefix_deterministic():
	worker, dep, adv, rti = _setup()
	r1 = _solve(worker, adv, rti, frozen_kill_popps=[0, 1])
	r2 = _solve(worker, adv, rti, frozen_kill_popps=[0, 1])
	assert r1['objective'] == r2['objective']


@pytest.mark.unit
def test_frozen_prefix_no_adv_context_survives():
	"""Without adv (some driver call sites), the LP must not crash --
	affected pairs are conservatively dead in failure scenarios."""
	worker, dep, adv, rti = _setup()
	ret = _solve(worker, None, rti, frozen_kill_popps=[0])
	assert ret.get('solved')


@pytest.mark.unit
def test_kill_sampler_rotation_and_stability():
	"""Driver-side FrozenPrefixObjective: kill set fixed within an
	iteration, rotates across iterations, honors N_FAIL."""
	worker, dep, adv, rti = _setup()
	from core.generic_objective import FrozenPrefixObjective

	class _FakeSAS:
		pass

	fake = _FakeSAS()
	fake.n_popps = worker.n_popps
	fake.popp_to_ind = worker.popp_to_ind
	fake.whole_deployment_ug_perfs = worker.whole_deployment_ug_perfs
	fake.whole_deployment_ug_to_vol = worker.whole_deployment_ug_to_vol
	fake.calculate_ground_truth_ingress = worker.calculate_ground_truth_ingress
	fake.iter = 0

	# levers resolve at construction from the plugin defaults + env overrides
	os.environ['SCULPTOR_FROZEN_PREFIX_N_FAIL'] = '3'
	os.environ['SCULPTOR_FROZEN_PREFIX_TOP_LOAD'] = '0'   # rotation needs free slots
	try:
		gobj = FrozenPrefixObjective(fake, 'frozen_prefix')
		assert gobj.lp_kwargs['frozen_n_fail'] == 3 and gobj.lp_kwargs['frozen_top_load'] == 0
		k0a = gobj.per_call_lp_kwargs(adv)['frozen_kill_popps']
		k0b = gobj.per_call_lp_kwargs(adv)['frozen_kill_popps']
		assert k0a == k0b, 'kill set must be stable within an iteration'
		assert len(k0a) == min(3, worker.n_popps)
		# the objective's levers ride along to the workers with the kill list
		assert gobj.per_call_lp_kwargs(adv)['frozen_lat_scale'] == gobj.lp_kwargs['frozen_lat_scale']
		sets = {tuple(k0a)}
		for it in range(1, 6):
			fake.iter = it
			sets.add(tuple(gobj.per_call_lp_kwargs(adv)['frozen_kill_popps']))
		assert len(sets) > 1, 'kill set must rotate across iterations'
		# top-load slots are FIXED: with top_load >= n_fail the set is the
		# heaviest popps every iteration (deliberate, not a bug)
		os.environ['SCULPTOR_FROZEN_PREFIX_TOP_LOAD'] = '3'
		gtop = FrozenPrefixObjective(fake, 'frozen_prefix')
		fixed = {tuple(gtop.per_call_lp_kwargs(adv)['frozen_kill_popps'])}
		for it in range(1, 4):
			fake.iter = it
			fixed.add(tuple(gtop.per_call_lp_kwargs(adv)['frozen_kill_popps']))
		assert len(fixed) == 1
	finally:
		del os.environ['SCULPTOR_FROZEN_PREFIX_N_FAIL']
		del os.environ['SCULPTOR_FROZEN_PREFIX_TOP_LOAD']


@pytest.mark.unit
def test_default_kill_set_is_adv_independent():
	"""The in-LP default stride must not depend on the advertisement --
	otherwise probe pairs (one bit flipped) would score against different
	scenario sets."""
	from core.frozen_prefix import default_kill_popps
	assert default_kill_popps(100, 20) == default_kill_popps(100, 20)
	assert len(default_kill_popps(100, 20)) == 20
	assert default_kill_popps(5, 20) == [0, 1, 2, 3, 4]
