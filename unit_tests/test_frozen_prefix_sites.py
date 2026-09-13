"""Site (group) failure scenarios in the frozen_prefix objective (Tom 2026-09-11):
a kill-set entry may be a whole site -- every peering at it fails together --
and the LP prices it like any single-peering scenario (fallback latency,
P_nr for stranded pairs, P_c on the overflow the fallback volume causes).

  - lifted == stacked reference with mixed single + site kill sets
  - group fallbacks: singleton group == frozen_fallbacks; vectorized == gti
  - default kill set spends round(site_frac * n_fail) slots on sites
  - site_frac = 0 is the pre-09-11 objective (checked by the lifted tests +
    the scratch regression on 2026-09-11: 24/24 values identical)
  - the driver-side sampler emits site tuples at the requested share

Run:
	SCULPTOR_XOBJS=1 pytest -v unit_tests/test_frozen_prefix_sites.py
"""
import math
import os

import numpy as np
import pytest

os.environ.setdefault('SCULPTOR_XOBJS', '1')

from unit_tests.test_lp_correctness import _setup
from unit_tests.test_frozen_prefix_lifted import _advs, _pairs, _solve


def _mixed_kills(worker):
	from core.frozen_prefix import default_kill_popps, site_groups
	n = worker.n_popps
	groups = site_groups(worker)
	assert len(groups) >= 2
	return {
		'sites_only': list(groups),
		'stride+site': default_kill_popps(n, 7) + [groups[0]],
		'stride+2sites': default_kill_popps(n, 7) + [groups[0], groups[-1]],
		'exhaustive+sites': list(range(n)) + list(groups),
		'overlap': [groups[1][0]] + [groups[1]],     # a peering AND its site
	}


@pytest.mark.unit
def test_group_fallbacks_singleton_equals_single_and_matches_gti():
	from core.frozen_prefix import frozen_fallbacks, frozen_fallbacks_killed, site_groups
	from helpers.helpers import threshold_a
	worker, dep, _, _ = _setup()
	ind = worker.popp_by_ug_indicator
	for name, adv in _advs(worker.n_popps).items():
		rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
		pp, pu, bw = _pairs(worker, rti)
		fb1 = frozen_fallbacks(worker, adv, pp, pu, bw)
		for k in np.unique(bw)[:5]:
			idx, fb = frozen_fallbacks_killed(worker, adv, pp, pu, bw, [int(k)])
			assert np.array_equal(idx, np.where(bw == k)[0])
			assert np.array_equal(fb, fb1[idx]), (name, int(k))
		for g in site_groups(worker):
			idx, fast = frozen_fallbacks_killed(worker, adv, pp, pu, bw, g)
			worker.popp_by_ug_indicator = None       # force the gti path
			try:
				idx2, slow = frozen_fallbacks_killed(worker, threshold_a(adv), pp, pu, bw, g)
			finally:
				worker.popp_by_ug_indicator = ind
			assert np.array_equal(idx, idx2) and np.array_equal(fast, slow), (name, g)
			assert not np.any(np.isin(fast[fast >= 0], list(g))), 'fallback inside the dead site'
			if name == 'one_per_peering':
				assert np.all(fast == -1)


@pytest.mark.unit
def test_lifted_equals_stacked_with_site_scenarios():
	worker, dep, _, _ = _setup()
	n = worker.n_popps
	kills = _mixed_kills(worker)
	for name, adv in _advs(n).items():
		rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
		for kname, kill in kills.items():
			a = _solve(worker, adv, rti, 'stacked', kill)
			b = _solve(worker, adv, rti, 'lifted', kill)
			assert a.get('solved') and b.get('solved'), (name, kname)
			assert math.isclose(a['objective'], b['objective'], rel_tol=1e-6, abs_tol=1e-6), \
				'{}/{}: stacked {} != lifted {}'.format(name, kname, a['objective'], b['objective'])
			for key in ('frozen_prefix_fail_no_route_frac', 'frozen_prefix_fail_overflow_frac',
						'frozen_prefix_normal_overflow_frac', 'frozen_prefix_unroutable_frac'):
				assert math.isclose(a[key], b[key], abs_tol=1e-7), (name, kname, key, a[key], b[key])
			n_groups = sum(1 for k in kill if not isinstance(k, (int, np.integer)))
			assert b['frozen_prefix_n_site_fail'] == n_groups
			assert b['frozen_prefix_n_fail'] == len(kill)
			assert a['frozen_prefix_kill_popps'] == b['frozen_prefix_kill_popps']


@pytest.mark.unit
def test_site_scenarios_change_the_objective_where_they_should():
	"""A site scenario strands every pair of a single-popp prefix at that
	site and displaces the others; the objective must move vs. no-sites,
	and a site kill must cost at least its own peerings' single kills."""
	from core.frozen_prefix import site_groups
	worker, dep, _, _ = _setup()
	adv = _advs(worker.n_popps)['random6']
	rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
	g = site_groups(worker)[0]
	base = _solve(worker, adv, rti, 'lifted', [])
	singles = _solve(worker, adv, rti, 'lifted', [int(k) for k in g])
	site = _solve(worker, adv, rti, 'lifted', [g])
	assert base['objective'] != site['objective']
	# the site scenario is one scenario (weight gamma), each single kill is
	# gamma/|g|: pooled failure cost of the site >= mean of its singles
	assert -site['objective'] >= -singles['objective'] - 1e-9


@pytest.mark.unit
def test_default_kill_set_site_share_and_lever():
	from core.frozen_prefix import default_kill_scenarios, default_kill_popps, site_groups, \
		normalize_kill_scenarios, kill_scenarios_from_kwargs
	from core.objective_registry import lp_kwargs_for
	worker, dep, _, _ = _setup()
	n = worker.n_popps
	assert default_kill_scenarios(worker, 20, 0.0) == default_kill_popps(n, 20)
	ks = default_kill_scenarios(worker, 20, 0.1)
	groups = [k for k in ks if isinstance(k, tuple)]
	assert len(groups) == 2 and len(ks) == 20, ks
	assert all(g in site_groups(worker) for g in groups)
	assert normalize_kill_scenarios([3, (5, 4, 5), 3, ()]) == [(3,), (4, 5)]
	assert lp_kwargs_for('frozen_prefix')['frozen_site_fail_frac'] == 0.1   # default since 2026-09-11
	assert lp_kwargs_for('frozen_prefix', env={'SCULPTOR_FROZEN_PREFIX_SITE_FAIL_FRAC': '0'})['frozen_site_fail_frac'] == 0.0
	assert lp_kwargs_for('frozen_prefix', env={'SCULPTOR_FROZEN_PREFIX_SITE_FAIL_FRAC': '0.1'})['frozen_site_fail_frac'] == 0.1
	assert len([s for s in kill_scenarios_from_kwargs(worker, {'frozen_site_fail_frac': 0.1}, 20) if len(s) > 1]) == 2
	assert kill_scenarios_from_kwargs(worker, {'frozen_kill_popps': [1, (2, 3)]}, 20) == [(1,), (2, 3)]


@pytest.mark.unit
def test_sampler_emits_sites_at_the_requested_share():
	from core.generic_objective import FrozenPrefixObjective
	from core.frozen_prefix import site_groups
	worker, dep, adv, rti = _setup()
	obj = FrozenPrefixObjective.__new__(FrozenPrefixObjective)
	obj.sas = worker
	obj.lp_kwargs = {'frozen_n_fail': 20, 'frozen_explore_frac': 0.5, 'frozen_top_load': 0,
					 'frozen_site_fail_frac': 0.1}
	obj._volume_reach_weights = lambda: np.ones(worker.n_popps)
	obj._load_weights = lambda a: None
	kill = obj._sample_kill_set(3, adv)
	sites = [k for k in kill if isinstance(k, tuple)]
	singles = [k for k in kill if not isinstance(k, tuple)]
	assert len(sites) == 2 and len(singles) == 18, (len(sites), len(singles))
	assert all(s in site_groups(worker) for s in sites)
	obj.lp_kwargs['frozen_site_fail_frac'] = 0.0
	assert all(not isinstance(k, tuple) for k in obj._sample_kill_set(3, adv))


@pytest.mark.unit
def test_column_generation_is_exact():
	"""Column generation (solve on a small candidate set, price every excluded
	pair with the row duals, add violators, repeat) must reproduce the full
	LP's optimum and diagnostics for every adv and kill set -- and converge
	in a few rounds with far fewer active pairs."""
	worker, dep, _, _ = _setup()
	n = worker.n_popps
	kills = _mixed_kills(worker)
	kills['stride7'] = list(range(0, n, max(1, n // 7)))[:7]
	for name, adv in _advs(n).items():
		rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
		for kname, kill in kills.items():
			full = _solve(worker, adv, rti, 'lifted', kill, frozen_colgen=0, frozen_persistent=0)
			cg = _solve(worker, adv, rti, 'lifted', kill, frozen_colgen=1, frozen_colgen_min_pairs=0, frozen_colgen_k=2, frozen_persistent=0)
			assert full.get('solved') and cg.get('solved'), (name, kname)
			assert math.isclose(full['objective'], cg['objective'], rel_tol=1e-6, abs_tol=1e-6), \
				'{}/{}: full {} != colgen {}'.format(name, kname, full['objective'], cg['objective'])
			# the optimum can be degenerate (several x with the same objective), so
			# the per-scenario diagnostics may differ in the 5th digit
			for key in ('frozen_prefix_fail_no_route_frac', 'frozen_prefix_fail_overflow_frac',
						'frozen_prefix_normal_overflow_frac', 'frozen_prefix_normal_lat'):
				assert math.isclose(full[key], cg[key], rel_tol=1e-3, abs_tol=1e-4), (name, kname, key, full[key], cg[key])
			assert cg['frozen_prefix_colgen_rounds'] >= 1
			assert cg['frozen_prefix_active_pairs'] <= len(full['frozen_prefix_pairs']) + 2 * worker.whole_deployment_n_ug + 20 \
				or cg['frozen_prefix_active_pairs'] < full['frozen_prefix_n_vars']
			assert full['frozen_prefix_colgen_rounds'] == 0


@pytest.mark.unit
def test_persistent_warm_start_matches_lifted_over_probe_sequence():
	"""The persistent model (one HiGHS model per process, edited between
	probes, re-solved from the incumbent basis) must reproduce the lifted
	solve's objective and diagnostics on a sequence of one-entry flips, and
	must actually be re-used (one model, few blocks rebuilt per probe)."""
	import core.frozen_prefix_persistent as fpp
	from core.frozen_prefix import default_kill_scenarios
	worker, dep, _, _ = _setup()
	n = worker.n_popps
	fpp._MODELS.clear()
	rs = np.random.RandomState(11)
	adv = _advs(n)['random6'].copy()
	kill = default_kill_scenarios(worker, 7, 0.3)
	solves = 0
	for step in range(8):
		if step:
			i, j = rs.randint(n), rs.randint(adv.shape[1])
			adv[i, j] = 1.0 - adv[i, j]           # a probe: flip one entry
		rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
		ref = _solve(worker, adv, rti, 'lifted', kill, frozen_persistent=0, frozen_colgen=0)
		per = _solve(worker, adv, rti, 'lifted', kill, frozen_persistent=1)
		assert ref.get('solved') and per.get('solved'), step
		assert per['frozen_prefix_formulation'] == 'persistent'
		assert math.isclose(ref['objective'], per['objective'], rel_tol=1e-6, abs_tol=1e-6), \
			'step {}: lifted {} != persistent {}'.format(step, ref['objective'], per['objective'])
		for key in ('frozen_prefix_fail_no_route_frac', 'frozen_prefix_fail_overflow_frac',
					'frozen_prefix_normal_overflow_frac', 'frozen_prefix_unroutable_frac', 'frozen_prefix_normal_lat'):
			assert math.isclose(ref[key], per[key], rel_tol=1e-3, abs_tol=1e-4), (step, key, ref[key], per[key])
		if step:
			assert per['frozen_prefix_blocks_rebuilt'] <= 3, per['frozen_prefix_blocks_rebuilt']
		solves = per['frozen_prefix_persistent_solves']
	assert solves == 8 and len(fpp._MODELS) == 1
	# a kill set above the persistent cap (the exhaustive eval pin at size 32)
	# must NOT go persistent
	big = _solve(worker, adv, rti, 'lifted', list(range(n)), frozen_persistent=1, frozen_persistent_max_k=10)
	assert big['frozen_prefix_formulation'] == 'lifted'
	ref = _solve(worker, adv, rti, 'lifted', list(range(n)), frozen_persistent=0, frozen_colgen=0)
	assert math.isclose(ref['objective'], big['objective'], rel_tol=1e-6, abs_tol=1e-6)


@pytest.mark.unit
def test_persistent_survives_kill_set_rotation_and_eviction():
	"""Training rotates the kill set every iteration and a worker keeps at
	most _MAX_MODELS models: three kill sets x eight probes, eviction in
	between, every solve must still match the lifted builder (2026-09-11
	size-32 reproduction: the third kill set returned a stale, slightly
	better-than-optimal objective)."""
	import core.frozen_prefix_persistent as fpp
	from core.frozen_prefix import site_groups
	worker, dep, _, _ = _setup()
	n = worker.n_popps
	fpp._MODELS.clear()
	rs = np.random.RandomState(5)
	groups = site_groups(worker)
	adv0 = _advs(n)['random6'].copy()
	for it in range(4):
		singles = sorted(rs.choice(n, size=7, replace=False).tolist())
		kill = [int(k) for k in singles] + [groups[int(rs.randint(len(groups)))]]
		adv = adv0.copy()
		for pr in range(8):
			i, j = rs.randint(n), rs.randint(adv.shape[1])
			adv[i, j] = 1.0 - adv[i, j]
			rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
			ref = _solve(worker, adv, rti, 'lifted', kill, frozen_persistent=0, frozen_colgen=0)
			per = _solve(worker, adv, rti, 'lifted', kill, frozen_persistent=1)
			assert per['frozen_prefix_formulation'] == 'persistent'
			assert math.isclose(ref['objective'], per['objective'], rel_tol=1e-7, abs_tol=1e-7), \
				'it {} probe {}: lifted {} != persistent {}'.format(it, pr, ref['objective'], per['objective'])
	assert len(fpp._MODELS) <= fpp._MAX_MODELS
