"""Lifted frozen_prefix formulation == stacked reference (core/frozen_prefix.py).

The lifted LP writes each popp's normal load once and adds a row only where
a failure scenario changes it. These tests prove it is an EXACT
reformulation on the small deployment (the actual-5 / actual-10 / actual-32
rungs are recorded in SCRAPBOOK_FROZEN_PREFIX.md):
  - vectorized per-pair fallback == per-scenario ground-truth-ingress lookup
  - identical objective for sampled and exhaustive kill sets, several advs
  - identical per-scenario diagnostics (no-route / overflow fractions)
  - formulation switch (kwarg > env) and the default is lifted

Run:
	SCULPTOR_XOBJS=1 pytest -v unit_tests/test_frozen_prefix_lifted.py
"""
import math
import os

import numpy as np
import pytest

os.environ.setdefault('SCULPTOR_XOBJS', '1')

from unit_tests.test_lp_correctness import _setup


def _advs(n_popps):
	rs = np.random.RandomState(7)
	rand = (rs.rand(n_popps, 6) < 0.35).astype(float)
	rand[:, 0] = 1.0
	return {
		'allon': np.ones((n_popps, 1)),
		'random6': rand,
		'one_per_peering': np.eye(n_popps),
	}


def _pairs(worker, rti):
	from helpers.constants import NO_ROUTE_LATENCY
	pp, pu, bw = [], [], []
	for prefix_i, u2p in sorted(rti.items()):
		for ug, pt in u2p.items():
			pi = worker.popp_to_ind.get(pt)
			if pi is None:
				continue
			lat = worker.whole_deployment_ug_perfs.get(ug, {}).get(pt)
			if lat is None or lat >= NO_ROUTE_LATENCY:
				continue
			pp.append(prefix_i); pu.append(worker.whole_deployment_ug_to_ind[ug]); bw.append(pi)
	return np.asarray(pp), np.asarray(pu), np.asarray(bw)


def _solve(worker, adv, rti, form, kill, **kw):
	from core.frozen_prefix import solve_lp_frozen_prefix
	from core.objective_registry import lp_kwargs_for
	kwa = lp_kwargs_for('frozen_prefix')
	kwa.update(kw)
	return solve_lp_frozen_prefix(worker, rti, 'frozen_prefix', adv=adv,
								  frozen_kill_popps=kill, frozen_formulation=form, **kwa)


@pytest.mark.unit
def test_vectorized_fallbacks_match_ground_truth_ingress():
	from core.frozen_prefix import frozen_fallbacks, _frozen_fallbacks_via_gti
	from helpers.helpers import threshold_a
	worker, dep, _, _ = _setup()
	for name, adv in _advs(worker.n_popps).items():
		rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
		pp, pu, bw = _pairs(worker, rti)
		fast = frozen_fallbacks(worker, adv, pp, pu, bw)
		slow = _frozen_fallbacks_via_gti(worker, threshold_a(adv), pp, pu, bw)
		assert np.array_equal(fast, slow), '{}: {} fallback mismatches'.format(
			name, int(np.sum(fast != slow)))
		# a pair's fallback is never its own winner
		assert not np.any((fast >= 0) & (fast == bw))
		if name == 'one_per_peering':
			assert np.all(fast == -1), 'single-popp prefixes have no fallback'


@pytest.mark.unit
def test_lifted_equals_stacked_objective_and_diagnostics():
	from core.frozen_prefix import default_kill_popps
	worker, dep, _, _ = _setup()
	n = worker.n_popps
	kills = {'stride': default_kill_popps(n, 7), 'exhaustive': list(range(n)),
			 'single': [0]}
	for name, adv in _advs(n).items():
		rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
		for kname, kill in kills.items():
			a = _solve(worker, adv, rti, 'stacked', kill)
			b = _solve(worker, adv, rti, 'lifted', kill)
			assert a.get('solved') and b.get('solved')
			assert math.isclose(a['objective'], b['objective'], rel_tol=1e-7, abs_tol=1e-7), \
				'{}/{}: stacked {} != lifted {}'.format(name, kname, a['objective'], b['objective'])
			for key in ('frozen_prefix_fail_no_route_frac', 'frozen_prefix_fail_overflow_frac',
						'frozen_prefix_normal_overflow_frac', 'frozen_prefix_unroutable_frac'):
				assert math.isclose(a[key], b[key], abs_tol=1e-7), (name, kname, key, a[key], b[key])
			assert b['frozen_prefix_formulation'] == 'lifted'
			assert b['frozen_prefix_n_fail'] == len(kill)
			# the lifted model never grows with K the way the stacked one does
			# (stacked: pairs + (K+1)*n_popps vars)
			if len(kill) >= 3:
				assert b['frozen_prefix_n_vars'] < len(_pairs(worker, rti)[0]) + (len(kill) + 1) * n


@pytest.mark.unit
def test_lifted_matches_under_levers():
	"""gamma / penalties / lat_scale / headroom all flow through both."""
	worker, dep, adv, rti = _setup()
	adv = _advs(worker.n_popps)['random6']
	rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
	kill = list(range(worker.n_popps))
	levers = dict(frozen_gamma=2.5, frozen_no_route_penalty=80.0,
				  frozen_congestion_penalty=30.0, frozen_lat_scale=0.1,
				  frozen_cap_headroom=0.9)
	a = _solve(worker, adv, rti, 'stacked', kill, **levers)
	b = _solve(worker, adv, rti, 'lifted', kill, **levers)
	assert math.isclose(a['objective'], b['objective'], rel_tol=1e-7, abs_tol=1e-7)
	assert b['frozen_prefix_gamma'] == 2.5 and b['frozen_prefix_cap_headroom'] == 0.9


@pytest.mark.unit
def test_formulation_switch_and_default():
	from core.frozen_prefix import solve_lp_frozen_prefix
	worker, dep, adv, rti = _setup()
	ret = solve_lp_frozen_prefix(worker, rti, 'frozen_prefix', adv=adv)
	assert ret['frozen_prefix_formulation'] == 'lifted', 'default must be lifted'
	ret = solve_lp_frozen_prefix(worker, rti, 'frozen_prefix', adv=adv,
								 frozen_formulation='stacked')
	assert 'frozen_prefix_formulation' not in ret     # reference returns no tag
	os.environ['SCULPTOR_FROZEN_PREFIX_FORMULATION'] = 'stacked'
	try:
		ret = solve_lp_frozen_prefix(worker, rti, 'frozen_prefix', adv=adv)
		assert 'frozen_prefix_formulation' not in ret
		ret = solve_lp_frozen_prefix(worker, rti, 'frozen_prefix', adv=adv,
									 frozen_formulation='lifted')
		assert ret['frozen_prefix_formulation'] == 'lifted', 'kwarg beats env'
	finally:
		del os.environ['SCULPTOR_FROZEN_PREFIX_FORMULATION']


@pytest.mark.unit
def test_eval_pin_is_exhaustive_by_default():
	"""The paper-table pin hedges against EVERY popp failure unless
	SCULPTOR_FROZEN_PREFIX_PIN_N_FAIL samples (2026-09-07 finding: a 20-popp
	stride pin left 2.5-2.8% congested volume that the exhaustive pin removes)."""
	from core.frozen_prefix_eval import pin_pairs
	import core.frozen_prefix as fp
	worker, dep, _, _ = _setup()
	adv = _advs(worker.n_popps)['random6']
	rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
	seen = {}
	orig = fp.solve_lp_frozen_prefix
	def spy(sas, r, obj, **kw):
		seen['kill'] = list(kw['frozen_kill_popps']); seen['tl'] = kw.get('frozen_time_limit')
		return orig(sas, r, obj, **kw)
	fp.solve_lp_frozen_prefix = spy     # pin_pairs imports it at call time
	try:
		pin_pairs(worker, adv, rti)
		assert seen['kill'] == list(range(worker.n_popps))
		assert seen['tl'] and seen['tl'] > 30
		os.environ['SCULPTOR_FROZEN_PREFIX_PIN_N_FAIL'] = '7'
		pin_pairs(worker, adv, rti)
		assert len(seen['kill']) == 7
	finally:
		fp.solve_lp_frozen_prefix = orig
		os.environ.pop('SCULPTOR_FROZEN_PREFIX_PIN_N_FAIL', None)
