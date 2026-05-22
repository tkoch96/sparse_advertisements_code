"""Sweep: how does Step 1's win depend on var_pool size?

For a fixed dpsize/n_perms, vary N_WARMUP ∈ {0, 1, 3, 5, 10, 20} to grow
var_pool to different sizes. Measure step1-on vs step1-off at each.

Theory: the legacy code iterates var_pool, so its cost scales with
var_pool size. Step 1 iterates only active_vars (size ~independent of
var_pool). So the absolute savings should grow with var_pool size.
"""
import os, sys, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from tests.test_worker_perf_sweep import (
    _build_worker, _build_advertisement, _measure, _reset_transient
)
from tests.test_worker_perf import make_calc_compressed_lb_batch


WARMUP_LEVELS = [0, 1, 3, 5, 10]
DPSIZE = 'small'
N_PERMS = 132
SEED = 1


@pytest.mark.slow
@pytest.mark.parametrize('n_warmup', WARMUP_LEVELS)
def test_warmup_curve(n_warmup):
    print(f"\n{'='*60}")
    print(f"=== n_warmup={n_warmup}  (dpsize={DPSIZE}, n_perms={N_PERMS}) ===")
    print(f"{'='*60}")

    worker = _build_worker(DPSIZE, SEED, n_prefixes=6)

    t = time.time()
    for w_i in range(n_warmup):
        w_adv = _build_advertisement(worker, n_prefixes=6, seed=SEED * 100 + w_i)
        w_rng = np.random.default_rng(SEED * 100 + w_i)
        w_data = make_calc_compressed_lb_batch(w_adv, n_perms=32, perm_rng=w_rng)
        np.random.seed(42 + w_i)
        worker._cmd_calc_compressed_lb(w_data)
    print(f"  warmup wall: {time.time()-t:.2f}s, var_pool={len(worker.var_pool)}")

    adv = _build_advertisement(worker, n_prefixes=6, seed=SEED)
    perm_rng = np.random.default_rng(SEED)
    data = make_calc_compressed_lb_batch(adv, N_PERMS, perm_rng)

    os.environ.pop('SCULPTOR_DISABLE_RAW_X_BATCH', None)
    _reset_transient(worker)
    on = _measure(worker, data, label='STEP1_ON')

    os.environ['SCULPTOR_DISABLE_RAW_X_BATCH'] = '1'
    _reset_transient(worker)
    off = _measure(worker, data, label='STEP1_OFF')
    os.environ.pop('SCULPTOR_DISABLE_RAW_X_BATCH', None)

    print(f"  fingerprint:  on={on['fp']}  off={off['fp']}  match={on['fp']==off['fp']}")
    print(f"  wall (s):     on={on['wall']:.3f}    off={off['wall']:.3f}")
    if off['wall'] > 0:
        d = on['wall'] - off['wall']
        pct = 100.0 * d / off['wall']
        print(f"  Δ:            {d:+.3f}s ({pct:+.1f}%)")
    print(f"  var_pool:     {on['var_pool']}")
    # Print the targeted timer
    sgl_on = on['breakdown'].get('solve_generic_lp_persistent', 0) * 1000
    sgl_off = off['breakdown'].get('solve_generic_lp_persistent', 0) * 1000
    print(f"  solve_generic_lp_persistent: on={sgl_on:.0f}ms  off={sgl_off:.0f}ms  Δ={sgl_on-sgl_off:+.0f}ms")

    assert on['fp'] == off['fp']
