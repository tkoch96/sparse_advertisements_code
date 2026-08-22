"""No-Gurobi unit checks for the monte-carlo OFF rung plumbing.

  A. The visual refactor of get_ingress_probabilities_and_sim (split into
     _compute_scenario_options + _sample_scenario_realizations) is
     behavior-identical to the pre-refactor implementation (git HEAD copy),
     bit-for-bit under a fixed RNG seed.
  B. Abl_MC_Off_Worker._sample_scenario_realizations is deterministic,
     consumes NO RNG, returns exactly one realization, and its price map
     holds probability-weighted expected latencies keyed by the min-latency
     representative option; _path_obj_coeffs applies the prices and leaves
     unknown paths (e.g. NO_PATH sentinel) at stock pricing.

Run: python -m experiments.ablation.test_mc_off_unit
"""
import importlib.util
import os
import subprocess
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.path_distribution_computer import Path_Distribution_Computer  # noqa: E402
from experiments.ablation.mc_off_worker import Abl_MC_Off_Worker  # noqa: E402


POPPS = [('popA', 'peer1'), ('popA', 'peer2'), ('popB', 'peer1')]
UGS = ['ug0', 'ug1', 'ug2', 'ug3']
# ug -> {popp: latency}
PERFS = {
    'ug0': {POPPS[0]: 10.0, POPPS[1]: 30.0, POPPS[2]: 50.0},
    'ug1': {POPPS[0]: 20.0, POPPS[2]: 40.0},
    'ug2': {POPPS[1]: 15.0, POPPS[2]: 25.0},
    'ug3': {POPPS[2]: 60.0},
}


def _synth_state(obj):
    obj.whole_deployment_n_ug = len(UGS)
    obj.whole_deployment_ugs = list(UGS)
    obj.whole_deployment_ug_to_ind = {u: i for i, u in enumerate(UGS)}
    obj.popps = list(POPPS)
    obj.n_popps = len(POPPS)
    obj.popp_to_ind = {p: i for i, p in enumerate(POPPS)}
    obj.parent_tracker = []
    obj.whole_deployment_ug_perfs = dict(PERFS)
    obj.whole_deployment_ui_to_poppi = {
        i: {obj.popp_to_ind[p]: None for p in PERFS[u]} for i, u in enumerate(UGS)}
    return obj


def _load_old_module():
    src = subprocess.check_output(
        ['git', 'show', 'HEAD:path_distribution_computer.py'], cwd=_REPO_ROOT).decode()
    path = os.path.join(_REPO_ROOT, 'experiments', 'ablation', '_pdc_head_snapshot.py')
    with open(path, 'w') as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location('_pdc_head_snapshot', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def test_refactor_equivalence():
    old_mod, snap_path = _load_old_module()
    try:
        a = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        new = _synth_state(Path_Distribution_Computer(0, 0, debug=True))
        old = _synth_state(old_mod.Path_Distribution_Computer(0, 0, debug=True))
        for trial_seed in (0, 7, 1234):
            np.random.seed(trial_seed)
            rti_new = new.get_ingress_probabilities_and_sim(a)
            np.random.seed(trial_seed)
            rti_old = old.get_ingress_probabilities_and_sim(a)
            assert rti_new == rti_old, (trial_seed, rti_new, rti_old)
            assert set(rti_new) == set(range(new.MC_NUM))
        print('[unit] A. refactor equivalence: stock sampling bit-identical to '
              'HEAD across 3 seeds OK')
    finally:
        os.remove(snap_path)


class _BareMCOff(Abl_MC_Off_Worker):
    def __init__(self):  # skip the heavy production __init__
        self.MC_NUM = 1
        self.rti_data = {}
        self.timing = {'get_paths_by_ug': 0, 'pmat_organize': 0, 'total_rti_calc': 0}
        self._abl_pseudo_price = {}
        self._abl_mc = {'pseudo_calls': 0, 'stock_sample_calls': 0,
                        'point_mass_violations': 0, 'pdf_calls': 0}


def test_mc_off_pseudo_path():
    a = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    w = _synth_state(_BareMCOff())
    w._compute_scenario_options(a)
    rng_before = np.random.get_state()[1].copy()
    out1 = w._sample_scenario_realizations()
    price1 = dict(w._abl_pseudo_price)
    out2 = w._sample_scenario_realizations()
    rng_after = np.random.get_state()[1]
    assert np.array_equal(rng_before, rng_after), 'mc-off sampling consumed RNG'
    assert out1 == out2, 'mc-off sampling not deterministic'
    assert list(out1) == [0], 'mc-off must return exactly one realization'
    assert w._abl_mc['pseudo_calls'] == 2

    # prefix 0 active popps = {0,1}; prefix 1 active = {1,2}
    # ug0 options pref0 = popps {0,1} lats {10,30} -> E=20, rep=0
    #             pref1 = popps {1,2} lats {30,50} -> E=40, rep=1
    assert out1[0][0]['ug0'] == POPPS[0] and out1[0][1]['ug0'] == POPPS[1]
    assert abs(price1[('ug0', 0)] - 20.0) < 1e-9
    assert abs(price1[('ug0', 1)] - 40.0) < 1e-9
    # ug3 only has popp 2 (active in pref1 only): E = 60, rep = 2
    assert out1[0][1]['ug3'] == POPPS[2] and abs(price1[('ug3', 2)] - 60.0) < 1e-9
    # ug1: pref0 options {0} -> E=20 rep=0 ; pref1 options {2} -> E=40 rep=2
    assert abs(price1[('ug1', 0)] - 20.0) < 1e-9 and abs(price1[('ug1', 2)] - 40.0) < 1e-9
    # ug2: pref0 options {1} -> E=15 rep=1 ; pref1 options {1,2} lats {15,25} -> E=20 rep=1
    # collision on (ug2, 1): min(15, 20) = 15 must win
    assert abs(price1[('ug2', 1)] - 15.0) < 1e-9

    # pricing override: pseudo-priced paths use E, unknown paths use stock
    w._abl_pseudo_price = dict(price1)
    paths = [('ug0', 0), ('ug0', 1), ('ug2', 1), ('ug1', 2)]
    coeffs = w._path_obj_coeffs(paths, 'avg_latency', 0)
    assert coeffs == [20.0, 40.0, 15.0, 40.0], coeffs
    w._abl_pseudo_price = {}
    coeffs_stock = w._path_obj_coeffs(paths, 'avg_latency', 0)
    assert coeffs_stock == [10.0, 30.0, 15.0, 40.0], coeffs_stock  # true latencies
    print('[unit] B. mc-off pseudo-path: deterministic, RNG-free, expected-'
          'latency pricing + collision min + stock fallback OK')


if __name__ == '__main__':
    test_refactor_equivalence()
    test_mc_off_pseudo_path()
    print('[unit] ALL OK')
