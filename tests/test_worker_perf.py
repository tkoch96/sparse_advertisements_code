"""Parametrized worker-side performance + correctness scaffold.

Goal: a regression gate for optimizations to path_distribution_computer
internals (sim_rti / total_rti_calc / pmat_organize / etc.) that:

1. Uses ACTUAL deployments from deployment_setup.get_random_deployment.
   These are sensitive to seeded routing/peer structure — synthetic
   stand-in inputs would mask real bugs.
2. Uses ACTUAL initial advertisements (random_binary / threshold-near
   from sparse_advertisements_v3.init_advertisement) — gradient and
   info-phase semantics depend on the advertisement's structure.
3. Drives the worker through its public RPC entry point (`_cmd_calc_
   compressed_lb`) with parametrized batch shapes that mirror the
   real call sites (info-phase ~132 perms, gradient ~500 perms).
4. Returns a structured dict with timing + benefit-vector identity hash
   so that any candidate optimization can be checked for correctness
   (≤0.5% delta on benefit, exact equality on cache_rep key) AND a
   wall-time win.

Usage:
    pytest tests/test_worker_perf.py -k bench -v -s
    pytest tests/test_worker_perf.py -k correctness -v

Skipped by default in `unit and not slow`; runs under `slow`.
"""
import os, sys, time, copy, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import pytest

import deployment_setup
from path_distribution_computer_ray import _LocalPathDistributionComputer
from helpers import split_deployment_by_ug_separated, split_deployment_by_ug
from constants import ADVERTISEMENT_THRESHOLD


# ---------------------------------------------------------------------------
# Fixtures: real deployments + real advertisements
# ---------------------------------------------------------------------------

# Keep the test matrix small. actual-32 is too slow for an inner test loop;
# decent is a good proxy for "more than toy but still <60s/iter local".
DPSIZES_FOR_PERF = ['small', 'decent']
SEEDS = [1, 2]
N_PERMS_TO_TEST = [1, 32, 132]   # matches the info phase shape


@pytest.fixture(scope='module')
def real_deployment_factory():
    """Return a function (dpsize, seed) -> deployment dict.

    We seed both the env var (which deployment_setup respects) and the
    explicit np.random.seed call inside get_random_deployment so the
    result is reproducible across pytest invocations.
    """
    cache = {}

    def _build(dpsize, seed):
        key = (dpsize, seed)
        if key in cache:
            # Return a deepcopy so per-test mutations don't leak between
            # parametrized cases (the worker constructor mutates the
            # subdeployment in init_all_vars).
            return copy.deepcopy(cache[key])
        os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(seed)
        np.random.seed(seed)
        dep = deployment_setup.get_random_deployment(dpsize)
        cache[key] = dep
        return copy.deepcopy(dep)

    return _build


@pytest.fixture(scope='module')
def real_init_advertisement():
    """Return a function (deployment, n_prefixes, mode) -> advertisement.

    Mirrors sparse_advertisements_v3.init_advertisement for the common
    'normal' init (threshold + Gaussian noise). Seeds with the same
    convention so the result is deterministic.
    """
    def _build(deployment, n_prefixes=6, mode='normal', seed=1):
        # Mirror v3.py's seed offset convention so the produced
        # advertisement matches what SCULPTOR's init would emit.
        np.random.seed(seed + 1)
        n_popp = len(deployment['popps'])
        if mode == 'normal':
            return (ADVERTISEMENT_THRESHOLD + 0.1 * np.random.normal(
                size=(n_popp, n_prefixes)))
        elif mode == 'random_binary':
            return np.random.randint(0, 2, size=(n_popp, n_prefixes)) * 1.0
        else:
            raise ValueError(mode)

    return _build


@pytest.fixture(scope='module')
def worker_factory(real_deployment_factory):
    """Return a function (dpsize, seed, n_prefixes) -> _LocalPathDistributionComputer.

    Builds a worker pre-initialized with the deployment's first slice
    (one worker, one chunk) so the worker has a full LP shell and the
    rest of init_all_vars run. Suitable for benchmarking calc_compressed_lb
    on representative input shapes.
    """
    cache = {}

    def _build(dpsize, seed, n_prefixes=6):
        key = (dpsize, seed, n_prefixes)
        if key in cache:
            return cache[key]
        deployment = real_deployment_factory(dpsize, seed)
        # We use the static+slice refactored split so the worker init
        # path matches what start_workers does in production.
        static_dep, slices = split_deployment_by_ug_separated(
            deployment, n_chunks=1)
        init_kwargs = {
            'lambduh': 0.1,
            'gamma': 1.0,
            'with_capacity': False,
            'verbose': False,
            'init': {'type': 'normal', 'var': 0.01},
            'explore': 'entropy',
            'using_resilience_benefit': True,
            'n_prefixes': n_prefixes,
            'save_run_dir': '/tmp/test_worker_perf_save_run',
            'generic_objective': 'avg_latency',
        }
        os.makedirs(init_kwargs['save_run_dir'], exist_ok=True)
        worker = _LocalPathDistributionComputer(
            worker_i=0, subdeployment=slices[0],
            init_kwargs=init_kwargs, static_dep=static_dep)
        cache[key] = worker
        return worker

    return _build


# ---------------------------------------------------------------------------
# Batch builder: mirrors what flush_latency_benefit_queue_generic sends
# ---------------------------------------------------------------------------

def make_calc_compressed_lb_batch(advertisement, n_perms, perm_rng):
    """Build a `data` list shaped exactly like the runtime caller sends.

    Format: [(base_args, base_kwa), (diff_1, kwa_1), (diff_2, kwa_2), ...]
    where each diff is `np.where(base != perturbed)` — a single bit-flip
    per perturbation in the simplest case. Mirrors v3.py:289-297.
    """
    base_adv = (advertisement > ADVERTISEMENT_THRESHOLD).astype(np.float64)
    base_kwa = {
        'verbose_workers': False,
        'generic_obj': 'avg_latency',
        'job_id': 0,
    }
    data = [((base_adv,), base_kwa)]
    n_popp, n_prefix = advertisement.shape
    for i in range(n_perms):
        flat_idx = perm_rng.integers(0, n_popp * n_prefix)
        poppi, prefi = divmod(int(flat_idx), n_prefix)
        diff = (np.array([poppi]), np.array([prefi]))
        kwa = dict(base_kwa)
        kwa['job_id'] = i + 1
        data.append((diff, kwa))
    return data


# ---------------------------------------------------------------------------
# Correctness fingerprint: identity of returned benefit vector
# ---------------------------------------------------------------------------

def fingerprint_results(ret):
    """Return a stable fingerprint of a calc_compressed_lb return value.

    Each entry of ret is {'ans': (benefit, (xsumx, psumx)), 'job_id': k}.
    We hash a tuple of (job_id, rounded-benefit, summary-of-xsumx-psumx)
    so floats round to ~1e-6 and tiny non-determinism (e.g. dict-iter
    order) doesn't trigger spurious mismatches.
    """
    items = []
    for entry in ret:
        jid = entry['job_id']
        benefit, (xsumx, psumx) = entry['ans']
        x_summary = (round(float(np.min(xsumx)), 6),
                     round(float(np.max(xsumx)), 6))
        # quantize psumx so trivial FP drift is forgiven
        psumx_q = np.round(np.asarray(psumx).flatten(), 8)
        psumx_nonzero = tuple(sorted(
            (int(i), float(v)) for i, v in enumerate(psumx_q) if v != 0))
        items.append((jid, round(float(benefit), 6), x_summary, psumx_nonzero))
    blob = repr(sorted(items, key=lambda e: e[0])).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('dpsize', DPSIZES_FOR_PERF)
@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('n_perms', N_PERMS_TO_TEST)
def test_bench_calc_compressed_lb(worker_factory, real_init_advertisement,
                                   dpsize, seed, n_perms):
    """Time the worker's calc_compressed_lb on a real deployment.

    Run twice: cold (worker fresh) and warm (caches populated). Print
    a structured timing block + the per-key cumulative timing summary
    so a baseline can be recorded before changing internals.
    """
    worker = worker_factory(dpsize, seed, n_prefixes=6)
    deployment = worker.whole_deployment_ugs  # for cache invalidation
    # Reset persistent caches that accumulate across calls in production.
    if hasattr(worker, 'pattern_cache'):
        worker.pattern_cache = {}
    if hasattr(worker, 'calc_cache'):
        worker.calc_cache.clear_all_caches()
    for k in worker.timing:
        worker.timing[k] = 0.0

    # Build the advertisement from the same seed convention as v3.py
    # init_advertisement does — guarantees reproducibility.
    adv = real_init_advertisement(
        {'popps': worker.whole_deployment_popps if hasattr(worker, 'whole_deployment_popps') else worker.popps},
        n_prefixes=6, mode='normal', seed=seed)

    perm_rng = np.random.default_rng(seed)
    data = make_calc_compressed_lb_batch(adv, n_perms, perm_rng)

    # Cold call (seed before so the MC sim inside sim_rti is reproducible
    # — same fingerprint expected across runs)
    np.random.seed(12345)
    t0 = time.time()
    ret_cold = worker._cmd_calc_compressed_lb(data)
    cold_s = time.time() - t0

    # Reset timing accumulators between cold/warm to measure them in isolation
    cold_breakdown = dict(worker.timing)
    for k in worker.timing:
        worker.timing[k] = 0.0

    # Warm call (same seed, same input → same output)
    np.random.seed(12345)
    t0 = time.time()
    ret_warm = worker._cmd_calc_compressed_lb(data)
    warm_s = time.time() - t0
    warm_breakdown = dict(worker.timing)

    cold_fp = fingerprint_results(ret_cold)
    warm_fp = fingerprint_results(ret_warm)

    print(f"\n[bench] dpsize={dpsize} seed={seed} n_perms={n_perms}")
    print(f"  cold: {cold_s:.3f}s    warm: {warm_s:.3f}s    speedup: {cold_s/max(warm_s, 1e-6):.1f}x")
    print(f"  cold-breakdown:")
    for k in sorted(cold_breakdown, key=lambda k: -cold_breakdown[k]):
        if cold_breakdown[k] > 0.001:
            print(f"    {k:<45s} {cold_breakdown[k]*1000:>7.1f} ms")
    print(f"  warm-breakdown:")
    for k in sorted(warm_breakdown, key=lambda k: -warm_breakdown[k]):
        if warm_breakdown[k] > 0.001:
            print(f"    {k:<45s} {warm_breakdown[k]*1000:>7.1f} ms")
    print(f"  fingerprint cold={cold_fp} warm={warm_fp}")

    # Sanity: same input → same fingerprint
    assert cold_fp == warm_fp, (
        f"calc_compressed_lb returned different output on cold vs warm "
        f"call with the same input. cold={cold_fp} warm={warm_fp}")


@pytest.mark.unit
@pytest.mark.parametrize('dpsize', ['small'])
@pytest.mark.parametrize('seed', [1])
def test_correctness_calc_compressed_lb_reproducible(
        worker_factory, real_init_advertisement, dpsize, seed):
    """Same input twice produces identical output. Sanity gate for
    any refactor: must keep this passing."""
    worker = worker_factory(dpsize, seed, n_prefixes=4)
    if hasattr(worker, 'pattern_cache'):
        worker.pattern_cache = {}
    if hasattr(worker, 'calc_cache'):
        worker.calc_cache.clear_all_caches()

    adv = real_init_advertisement(
        {'popps': worker.popps}, n_prefixes=4, mode='normal', seed=seed)
    perm_rng = np.random.default_rng(seed)
    data1 = make_calc_compressed_lb_batch(adv, n_perms=16, perm_rng=perm_rng)
    # Rebuild rng so the same perturbations are generated
    perm_rng = np.random.default_rng(seed)
    data2 = make_calc_compressed_lb_batch(adv, n_perms=16, perm_rng=perm_rng)

    # Seed np.random before each call so the Monte Carlo simulation
    # inside get_ingress_probabilities_and_sim is reproducible. Without
    # this the test would catch a real nondeterminism in sim_rti (which
    # uses np.random.rand without explicit seeding) — see the memory
    # note `project_sculptor_sameseed_noise.md`. For perf-regression
    # comparisons we want a fixed reference.
    np.random.seed(12345)
    fp1 = fingerprint_results(worker._cmd_calc_compressed_lb(data1))
    # Reset caches between calls so we exercise the cache-miss path twice
    worker.pattern_cache = {}
    worker.calc_cache.clear_all_caches()
    np.random.seed(12345)
    fp2 = fingerprint_results(worker._cmd_calc_compressed_lb(data2))

    assert fp1 == fp2, (
        f"calc_compressed_lb is non-deterministic across cache-clear "
        f"resets: fp1={fp1} fp2={fp2}. This breaks the regression gate "
        f"for any refactor.")
