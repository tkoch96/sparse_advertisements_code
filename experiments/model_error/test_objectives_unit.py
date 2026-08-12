"""No-Gurobi unit tests for experiments/model_error/objectives.py.

Covers the pure math helpers AND the generic-LP-contract layer (with
solve_generic_lp_with_failure_catch monkeypatched to canned returns), so
the objectives can be trusted before any Gurobi/worker wiring.

    ~/Documents/venv312/bin/python -m pytest experiments/model_error/test_objectives_unit.py -q
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.model_error import objectives as O


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSas:
    """Minimal stand-in for the attributes the objectives touch."""

    def __init__(self, n_ug=4, n_popps=3):
        self.n_popps = n_popps
        self.ugs = [('m', i) for i in range(n_ug)]
        self.whole_deployment_ugs = self.ugs
        self.ug_vols = np.ones(n_ug)
        self.whole_deployment_ug_vols = self.ug_vols
        # each UG can reach every popp; popp p costs 10*(p+1) ms for
        # everyone -> per-UG floor is 10.0
        self.ug_perfs = {ug: {('P', p): 10.0 * (p + 1)
                              for p in range(n_popps)} for ug in self.ugs}
        self.whole_deployment_ug_perfs = self.ug_perfs
        self.link_capacities_arr = np.array([10.0, 10.0, 5.0])[:n_popps]
        self.popps = [('P', p) for p in range(n_popps)]
        self.popp_to_ind = {popp: i for i, popp in enumerate(self.popps)}
        self.ingress_calls = []

    def calculate_ground_truth_ingress(self, a):
        self.ingress_calls.append(np.copy(a))
        return {'routed': 'fake'}, None


def steady_ret(lats, vols_by_poppi=None, frac_congested=0.0, solved=True):
    lats = np.asarray(lats, dtype=float)
    return {
        'solved': solved,
        # real generic-LP convention: 'objective' is a benefit ~ -avg_lat
        'objective': -float(np.mean(lats)) if len(lats) else 0.0,
        'lats_by_ug': lats,
        'paths_by_ug': {},
        'vols_by_poppi': vols_by_poppi if vols_by_poppi is not None
        else np.zeros(3),
        'fraction_congested_volume': frac_congested,
    }


@pytest.fixture
def patched_lp(monkeypatch):
    """Patch the LP entry point objectives.py calls; returns a dict the
    test mutates to control per-call canned results."""
    state = {'queue': [], 'calls': []}

    def fake_lp(sas, routed, obj, **kwargs):
        state['calls'].append((obj, kwargs))
        assert obj == 'avg_latency', 'objectives must compose the steady LP'
        return state['queue'].pop(0) if state['queue'] else steady_ret([1, 2, 3, 4])

    import solve_lp_assignment
    monkeypatch.setattr(
        solve_lp_assignment, 'solve_generic_lp_with_failure_catch', fake_lp)
    return state


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_frac_beyond_basic():
    lats = [10, 25, 50, 200]
    best = [10, 10, 10, 10]
    vols = [1, 1, 1, 1]
    # gaps: 0, 15, 40, 190
    assert O._frac_beyond(lats, best, 10, vols) == pytest.approx(0.75)
    assert O._frac_beyond(lats, best, 50, vols) == pytest.approx(0.25)
    assert O._frac_beyond(lats, best, 500, vols) == 0.0


def test_frac_beyond_volume_weighted():
    lats = [10, 100]
    best = [10, 10]
    assert O._frac_beyond(lats, best, 10, [3, 1]) == pytest.approx(0.25)
    assert O._frac_beyond(lats, best, 10, [1, 3]) == pytest.approx(0.75)


def test_max_util_array_and_dict_forms():
    caps = [10.0, 10.0, 5.0]
    # array form; 4th entry is the no-route pseudo link and must be ignored
    ret = {'vols_by_poppi': np.array([5.0, 2.0, 4.0, 999.0])}
    assert O._max_util_from_ret(ret, caps, 3) == pytest.approx(0.8)
    # dict form, sparse, pseudo-index dropped
    ret = {'vols_by_poppi': {0: 5.0, 2: [2.0, 2.0], 3: 999.0}}
    assert O._max_util_from_ret(ret, caps, 3) == pytest.approx(0.8)


def test_max_util_zero_capacity_guard():
    ret = {'vols_by_poppi': np.array([1.0, 1.0])}
    assert O._max_util_from_ret(ret, [0.0, 2.0], 2) == pytest.approx(0.5)


def test_perf_floor_best_lats():
    sas = FakeSas()
    best = O._perf_floor_best_lats(sas, 4)
    assert best.shape == (4,)
    assert np.allclose(best, 10.0)


def test_ug_vols_length_dispatch():
    sas = FakeSas(n_ug=4)
    assert len(O._ug_vols_for(sas, 4)) == 4
    # mismatched n -> safe uniform fallback
    assert np.allclose(O._ug_vols_for(sas, 7), 1.0)


# ---------------------------------------------------------------------------
# Generic-LP-contract layer (canned LP)
# ---------------------------------------------------------------------------

def test_frac_beyond_objective_contract(patched_lp):
    sas = FakeSas()
    patched_lp['queue'] = [steady_ret([10, 25, 50, 200])]
    out = O.solve_lp_frac_beyond_optimal(sas, {}, 'frac_beyond_optimal',
                                         frac_beyond_x=10)
    # floor best is 10 everywhere -> gaps 0,15,40,190 -> 3/4 beyond 10
    assert out['objective'] == pytest.approx(-0.75)
    assert out['frac_beyond'] == pytest.approx(0.75)
    assert out['solved']
    assert out['steady_avg_lat'] == pytest.approx(np.mean([10, 25, 50, 200]))
    assert out['frac_beyond_x'] == 10


def test_frac_beyond_respects_best_lats_kwarg(patched_lp):
    sas = FakeSas()
    patched_lp['queue'] = [steady_ret([10, 25, 50, 200])]
    out = O.solve_lp_frac_beyond_optimal(
        sas, {}, 'frac_beyond_optimal', frac_beyond_x=10,
        best_lats=np.array([10, 25, 50, 200], dtype=float))
    assert out['objective'] == 0.0
    assert out['frac_beyond'] == 0.0


def test_frac_beyond_unsolved_passthrough(patched_lp):
    sas = FakeSas()
    patched_lp['queue'] = [{'solved': False}]
    out = O.solve_lp_frac_beyond_optimal(sas, {}, 'frac_beyond_optimal')
    assert out == {'solved': False}


def test_lat_plus_mlu_objective_and_alpha_default(patched_lp):
    sas = FakeSas()
    patched_lp['queue'] = [steady_ret(
        [10, 20, 30, 40], vols_by_poppi=np.array([5.0, 0.0, 4.0]))]
    out = O.solve_lp_lat_plus_max_util(sas, {}, 'lat_plus_max_util')
    # mlu = 4/5 = 0.8 ; alpha default = mean floor = 10.0
    assert out['max_util'] == pytest.approx(0.8)
    assert out['mlu_alpha'] == pytest.approx(10.0)
    assert out['objective'] == pytest.approx(-(25.0 + 10.0 * 0.8))


def test_lat_plus_mlu_alpha_override(patched_lp):
    sas = FakeSas()
    patched_lp['queue'] = [steady_ret(
        [10, 20, 30, 40], vols_by_poppi=np.array([5.0, 0.0, 4.0]))]
    out = O.solve_lp_lat_plus_max_util(sas, {}, 'lat_plus_max_util',
                                       mlu_alpha=100.0)
    assert out['objective'] == pytest.approx(-(25.0 + 100.0 * 0.8))


def test_popp_failure_congestion_mean_and_max(patched_lp):
    sas = FakeSas()
    adv = np.ones((3, 2))
    # 1 steady + 3 failure scenarios
    patched_lp['queue'] = [
        steady_ret([10, 20, 30, 40]),
        steady_ret([1], frac_congested=0.0),
        steady_ret([1], frac_congested=0.5),
        steady_ret([1], frac_congested=0.1),
    ]
    out = O.solve_lp_popp_failure_congestion(
        sas, {}, 'popp_failure_congestion', adv=adv)
    assert out['objective'] == pytest.approx(-(0.0 + 0.5 + 0.1) / 3)
    assert out['popp_failure_mean_frac'] == pytest.approx((0.0 + 0.5 + 0.1) / 3)
    assert out['popp_failure_max_frac'] == pytest.approx(0.5)
    assert out['popp_failure_n_scenarios'] == 3
    # each scenario re-derived ingress for the failed adv
    assert len(sas.ingress_calls) == 3
    for i, a_fail in enumerate(sas.ingress_calls):
        assert a_fail[i].sum() == 0, 'scenario %d must zero popp %d' % (i, i)


def test_popp_failure_unsolved_scenario_counts_fully_congested(patched_lp):
    sas = FakeSas()
    adv = np.ones((3, 2))
    patched_lp['queue'] = [
        steady_ret([10, 20, 30, 40]),
        {'solved': False},
        steady_ret([1], frac_congested=0.0),
        steady_ret([1], frac_congested=0.0),
    ]
    out = O.solve_lp_popp_failure_congestion(
        sas, {}, 'popp_failure_congestion', adv=adv)
    assert out['objective'] == pytest.approx(-1.0 / 3)


def test_popp_failure_no_adv_returns_steady(patched_lp):
    sas = FakeSas()
    patched_lp['queue'] = [steady_ret([10, 20, 30, 40])]
    out = O.solve_lp_popp_failure_congestion(sas, {}, 'popp_failure_congestion')
    assert out['objective'] == pytest.approx(-25.0)
    assert 'popp_failure_n_scenarios' not in out


def test_popp_failure_empty_adv_scenario_is_total_congestion(patched_lp):
    sas = FakeSas(n_popps=1)
    sas.link_capacities_arr = np.array([10.0])
    adv = np.ones((1, 2))
    patched_lp['queue'] = [steady_ret([10, 20, 30, 40])]
    out = O.solve_lp_popp_failure_congestion(
        sas, {}, 'popp_failure_congestion', adv=adv)
    # failing the only popp leaves an all-zero adv -> counted as 1.0
    assert out['objective'] == pytest.approx(-1.0)


def test_popp_failure_sample_k_stride(patched_lp):
    sas = FakeSas(n_popps=3)
    adv = np.ones((3, 2))
    patched_lp['queue'] = [steady_ret([10, 20, 30, 40])] + [
        steady_ret([1], frac_congested=0.2)] * 2
    out = O.solve_lp_popp_failure_congestion(
        sas, {}, 'popp_failure_congestion', adv=adv, popp_failure_sample_k=2)
    assert out['popp_failure_n_scenarios'] == 2


def test_register_inserts_into_registry():
    import solve_lp_assignment
    names = O.register()
    assert names == sorted(O.REGISTERED_OBJECTIVES)
    for name, fn in O.REGISTERED_OBJECTIVES.items():
        assert solve_lp_assignment.generic_lp_functions[name] is fn
    # registered names must not collide with existing production objectives
    assert 'avg_latency' not in O.REGISTERED_OBJECTIVES


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
