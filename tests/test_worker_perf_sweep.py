"""Sweep bench: quantify Step 1 (batched getAttr X) across dpsize × n_perms.

For each (dpsize, n_perms) combination:
  1. Build worker, warm up var_pool with N_WARMUP prep batches
  2. Reset transient caches + timing
  3. Measure with Step 1 ENABLED (default code path): batched getAttr
  4. Reset transient caches + timing
  5. Measure with Step 1 DISABLED (SCULPTOR_DISABLE_RAW_X_BATCH=1): legacy per-var .X loop
  6. Record both timings + fingerprints + var_pool size

Both measurements share the SAME warmed worker, so Gurobi state is
identical and the comparison is apples-to-apples. The only variable is
which raw_x extraction path runs.

Note: both calls hit cache misses (we reset calc_cache between them), so
both do real LP work. The Gurobi warm-start basis carries over between
them but that's true for both modes equally.

Run: `PYTHONHASHSEED=0 ~/Documents/venv312/bin/python -m pytest tests/test_worker_perf_sweep.py -v -s`
"""
import os, sys, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import deployment_setup
from path_distribution_computer_ray import _LocalPathDistributionComputer
from helpers import split_deployment_by_ug_separated
from constants import ADVERTISEMENT_THRESHOLD

# Reuse fingerprint + batch builder from the other test file
from tests.test_worker_perf import (
    make_calc_compressed_lb_batch,
    fingerprint_results,
)


# Test matrix: (dpsize, n_perms). Tune for wall-time budget.
SWEEP = [
    ('small',  1),
    ('small',  32),
    ('small',  132),
    ('decent', 32),
    ('decent', 132),
]
SEED = 1
N_WARMUP = 5


def _build_worker(dpsize, seed, n_prefixes=6):
    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(seed)
    np.random.seed(seed)
    deployment = deployment_setup.get_random_deployment(dpsize)
    static_dep, slices = split_deployment_by_ug_separated(deployment, n_chunks=1)
    init_kwargs = {
        'lambduh': 0.1, 'gamma': 1.0, 'with_capacity': False, 'verbose': False,
        'init': {'type': 'normal', 'var': 0.01}, 'explore': 'entropy',
        'using_resilience_benefit': True, 'n_prefixes': n_prefixes,
        'save_run_dir': '/tmp/test_sweep_save_run', 'generic_objective': 'avg_latency',
    }
    os.makedirs(init_kwargs['save_run_dir'], exist_ok=True)
    return _LocalPathDistributionComputer(
        worker_i=0, subdeployment=slices[0],
        init_kwargs=init_kwargs, static_dep=static_dep)


def _build_advertisement(worker, n_prefixes, seed):
    np.random.seed(seed + 1)
    n_popp = len(worker.popps)
    return ADVERTISEMENT_THRESHOLD + 0.1 * np.random.normal(size=(n_popp, n_prefixes))


def _warmup(worker, dpsize, seed, n_prefixes, n_warmup=N_WARMUP):
    for w_i in range(n_warmup):
        w_adv = _build_advertisement(worker, n_prefixes, seed * 100 + w_i)
        w_rng = np.random.default_rng(seed * 100 + w_i)
        # Cap warmup-batch size at 32 to keep warmup wall reasonable
        w_data = make_calc_compressed_lb_batch(w_adv, n_perms=32, perm_rng=w_rng)
        np.random.seed(42 + w_i)
        worker._cmd_calc_compressed_lb(w_data)


def _reset_transient(worker):
    # Clear ALL caches that affect timing — otherwise the second
    # measurement gets a free ride on the first's populated state:
    #   - pattern_cache: per-prefix routing pattern cache (in sim_rti)
    #   - calc_cache.all_caches['lb']: latency-benefit cache (in generic_benefit)
    #   - this_time_ip_cache: ingress-probabilities cache (gates the sim_rti cache_hits)
    # The first three were the issue in v1 of this test — second
    # measurement looked artificially fast because sim_rti / generic_benefit
    # hit caches populated by the first measurement.
    if hasattr(worker, 'pattern_cache'):
        worker.pattern_cache = {}
    if hasattr(worker, 'this_time_ip_cache'):
        worker.this_time_ip_cache = {}
    if hasattr(worker, 'calc_cache'):
        worker.calc_cache.clear_all_caches()
    for k in worker.timing:
        worker.timing[k] = 0.0


def _measure(worker, data, label):
    np.random.seed(12345)
    t0 = time.time()
    ret = worker._cmd_calc_compressed_lb(data)
    wall = time.time() - t0
    breakdown = dict(worker.timing)
    fp = fingerprint_results(ret)
    return {'label': label, 'wall': wall, 'breakdown': breakdown, 'fp': fp,
            'var_pool': len(worker.var_pool)}


@pytest.mark.slow
@pytest.mark.parametrize('dpsize,n_perms', SWEEP)
def test_sweep_step1(dpsize, n_perms):
    print(f"\n{'='*70}")
    print(f"=== SWEEP: dpsize={dpsize}, n_perms={n_perms}, seed={SEED} ===")
    print(f"{'='*70}")

    worker = _build_worker(dpsize, SEED, n_prefixes=6)
    print(f"  built worker: n_ugs={worker.whole_deployment_n_ug}, n_popps={worker.n_popps}")

    t = time.time()
    _warmup(worker, dpsize, SEED, n_prefixes=6)
    print(f"  warmup ({N_WARMUP} batches @ 32 perms): {time.time()-t:.1f}s")
    print(f"  var_pool after warmup: {len(worker.var_pool)}")

    adv = _build_advertisement(worker, n_prefixes=6, seed=SEED)
    perm_rng = np.random.default_rng(SEED)
    data = make_calc_compressed_lb_batch(adv, n_perms, perm_rng)

    # --- Measure with Step 1 ENABLED (default code path) ---
    os.environ.pop('SCULPTOR_DISABLE_RAW_X_BATCH', None)
    _reset_transient(worker)
    step1 = _measure(worker, data, label='STEP1_ON')

    # --- Measure with Step 1 DISABLED ---
    os.environ['SCULPTOR_DISABLE_RAW_X_BATCH'] = '1'
    _reset_transient(worker)
    legacy = _measure(worker, data, label='STEP1_OFF')
    os.environ.pop('SCULPTOR_DISABLE_RAW_X_BATCH', None)

    # --- Report ---
    print(f"\n  var_pool size at measurement: {step1['var_pool']}")
    print(f"  fingerprint:  step1_on={step1['fp']}  step1_off={legacy['fp']}")
    if step1['fp'] != legacy['fp']:
        print(f"  ❌ FINGERPRINT MISMATCH — step 1 introduces a result change")
    else:
        print(f"  ✅ fingerprint match — correctness preserved")
    print(f"\n  wall:       STEP1_ON = {step1['wall']:.3f}s     STEP1_OFF = {legacy['wall']:.3f}s")
    if legacy['wall'] > 0:
        delta = step1['wall'] - legacy['wall']
        pct = 100.0 * delta / legacy['wall']
        print(f"              Δ = {delta:+.3f}s  ({pct:+.1f}%)")

    print(f"\n  per-key (ms):")
    all_keys = sorted(set(step1['breakdown']) | set(legacy['breakdown']),
                      key=lambda k: -legacy['breakdown'].get(k, 0))
    for k in all_keys:
        on_ms = step1['breakdown'].get(k, 0) * 1000
        off_ms = legacy['breakdown'].get(k, 0) * 1000
        if on_ms < 1 and off_ms < 1:
            continue
        d = on_ms - off_ms
        print(f"    {k:<42s} ON={on_ms:>9.1f}  OFF={off_ms:>9.1f}  Δ={d:+8.1f}")

    # Assert correctness (don't make perf an assertion — could vary by host)
    assert step1['fp'] == legacy['fp'], (
        f"Step 1 changed result: ON={step1['fp']} OFF={legacy['fp']}")
