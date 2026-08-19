"""Driver-side startup profiler (Tom 2026-08-19: 'which operations in
startup contribute which time, which objects contribute which ram').

Runs ONLY the driver deployment build (get_random_deployment) — no Ray,
no workers, no training — so it can run alongside a live campaign cell
(single-core, ~10-20G). Produces cache/eods/startup_profile_<seed>.json:

  phases    wall + driver-RSS delta + tracemalloc top-10 allocation
            sites for the build
  objects   deep sizes of every deployment component (numpy nbytes +
            recursive container accounting via pympler)

Run under py-spy for the function-level TIME flamegraph twin:
  ~/venv312/bin/py-spy record -r 50 -o startup_prof.svg -- \
      ~/venv312/bin/python -m experiments.eods.profile_startup --seed 2
"""
import argparse
import json
import os
import resource
import sys
import time
import tracemalloc

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0


def deep_size_gb(obj, seen=None):
    """numpy-aware recursive sizeof (pympler asizeof chokes on big
    ndarray graphs; nbytes is exact for arrays)."""
    import numpy as np
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0.0
    seen.add(oid)
    if isinstance(obj, np.ndarray):
        return obj.nbytes / 1073741824.0
    s = sys.getsizeof(obj, 0) / 1073741824.0
    if isinstance(obj, dict):
        s += sum(deep_size_gb(k, seen) + deep_size_gb(v, seen)
                 for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        s += sum(deep_size_gb(x, seen) for x in obj)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=2)
    ap.add_argument('--dpsize', default='testing_feature-actual-25')
    ap.add_argument('--out', default=None)
    ap.add_argument('--deep', action='store_true',
                    help='also construct the driver Sparse_Advertisement_'
                         'Eval object and size its attributes')
    args = ap.parse_args()

    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(args.seed)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    out_fn = args.out or os.path.join(
        _REPO_ROOT, 'cache', 'eods',
        'startup_profile_seed{}.json'.format(args.seed))

    report = {'seed': args.seed, 'dpsize': args.dpsize, 'phases': []}

    tracemalloc.start(10)
    t0 = time.time()
    r0 = rss_gb()
    snap0 = tracemalloc.take_snapshot()

    from deployment_setup import get_random_deployment
    report['phases'].append({'phase': 'imports',
                             'wall_s': round(time.time() - t0, 1),
                             'rss_delta_gb': round(rss_gb() - r0, 2)})

    t1 = time.time()
    r1 = rss_gb()
    deployment = get_random_deployment(args.dpsize)
    snap1 = tracemalloc.take_snapshot()
    top = snap1.compare_to(snap0, 'lineno')[:10]
    report['phases'].append({
        'phase': 'get_random_deployment',
        'wall_s': round(time.time() - t1, 1),
        'rss_delta_gb': round(rss_gb() - r1, 2),
        'top_alloc_sites': [
            {'site': str(st.traceback), 'size_gb': round(
                st.size_diff / 1073741824.0, 3)} for st in top],
    })
    tracemalloc.stop()

    if args.deep:
        # Driver-object construction: the s1v2 trace shows ~28 of the
        # 33 pre-worker minutes are NOT get_random_deployment — they
        # live in Sparse_Advertisement_Eval init / update_deployment.
        # Build it (no worker manager, no ray) and size its attributes.
        deployment['port'] = 39999
        tracemalloc.start(10)
        snap2 = tracemalloc.take_snapshot()
        t3 = time.time()
        r3 = rss_gb()
        from wrapper_eval import gamma, capacity, lambduh
        from constants import DEFAULT_EXPLORE
        from sparse_advertisements_v3 import Sparse_Advertisement_Eval
        sas = Sparse_Advertisement_Eval(
            deployment, verbose=False, lambduh=lambduh,
            with_capacity=capacity, explore=DEFAULT_EXPLORE,
            using_resilience_benefit=True, gamma=gamma,
            n_prefixes=None, generic_objective='avg_latency')
        snap3 = tracemalloc.take_snapshot()
        top = snap3.compare_to(snap2, 'lineno')[:12]
        report['phases'].append({
            'phase': 'Sparse_Advertisement_Eval_init',
            'wall_s': round(time.time() - t3, 1),
            'rss_delta_gb': round(rss_gb() - r3, 2),
            'top_alloc_sites': [
                {'site': str(st.traceback), 'size_gb': round(
                    st.size_diff / 1073741824.0, 3)} for st in top],
        })
        tracemalloc.stop()
        big = {}
        for name in dir(sas):
            if name.startswith('__'):
                continue
            try:
                v = getattr(sas, name)
            except Exception:
                continue
            if callable(v):
                continue
            try:
                g = deep_size_gb(v)
            except Exception:
                continue
            if g > 0.05:
                big[name] = round(g, 3)
        report['sas_attributes_gb'] = dict(
            sorted(big.items(), key=lambda kv: -kv[1]))

    t2 = time.time()
    objs = {}
    for k, v in deployment.items():
        try:
            objs[k] = round(deep_size_gb(v), 3)
        except Exception as e:
            objs[k] = 'err: {}'.format(e)
    report['deployment_components_gb'] = dict(
        sorted(((k, v) for k, v in objs.items()
                if not isinstance(v, str)),
               key=lambda kv: -kv[1]))
    report['sizeof_wall_s'] = round(time.time() - t2, 1)
    report['total_rss_gb'] = round(rss_gb(), 2)
    report['total_wall_s'] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(out_fn), exist_ok=True)
    with open(out_fn, 'w') as f:
        json.dump(report, f, indent=1)
    print('[profile_startup] wrote', out_fn)
    print(json.dumps({k: report[k] for k in
                      ('total_wall_s', 'total_rss_gb')}, indent=1))


if __name__ == '__main__':
    main()
