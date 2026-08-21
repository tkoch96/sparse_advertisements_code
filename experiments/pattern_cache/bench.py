"""pattern_cache representation bench (Tom 2026-08-19: 'vital but
implemented inefficiently — find much more efficient representations').

Populates a REAL worker's pattern_cache through the production code path
(random advertisement columns -> get_ingress_probabilities... miss path),
then re-encodes the SAME content into candidate schemes and reports:

  bytes/entry, bytes/(ug,pops) pair (the scale-free constant that
  extrapolates to actual-25/32), and hit-path reconstruction wall time
  (the price of compactness — the current repr's hit path is a plain
  list iteration).

Schemes:
  A current    list[(ui, [poppi...], [1/n...])]      (baseline)
  B noprobs    list[(ui, [poppi...])]                probs always 1/n
  C csr        (uis:int32[], offs:int32[], pops:int32[])  one entry
  D csr+key    C plus np.packbits key instead of tuple-of-bool
  E arena      global CSR arena: one growing pops/uis/offs pool, dict
               maps packed-key -> (row_start, row_end)

    PYTHONHASHSEED=0 SCULPTOR_LP_BACKEND=highs \
        python -m experiments.pattern_cache.bench --patterns 200
"""
import argparse
import os
import random
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault('SCULPTOR_LP_BACKEND', 'highs')
os.environ.setdefault('MPLBACKEND', 'Agg')


def deep_bytes(o, seen=None):
    if seen is None:
        seen = set()
    if id(o) in seen:
        return 0
    seen.add(id(o))
    if isinstance(o, np.ndarray):
        return o.nbytes
    s = sys.getsizeof(o, 0)
    if isinstance(o, dict):
        s += sum(deep_bytes(k, seen) + deep_bytes(v, seen)
                 for k, v in o.items())
    elif isinstance(o, (list, tuple, set, frozenset)):
        s += sum(deep_bytes(x, seen) for x in o)
    return s


def encode_noprobs(cache):
    return {k: [(ui, pops) for ui, pops, _ in v] for k, v in cache.items()}


def encode_csr(cache, pack_key=False, pop_dtype=np.int32):
    out = {}
    for k, v in cache.items():
        uis = np.fromiter((e[0] for e in v), dtype=np.int32, count=len(v))
        offs = np.zeros(len(v) + 1, dtype=np.int32)
        pops_cat = []
        for i, (_, pops, _) in enumerate(v):
            pops_cat.extend(pops)
            offs[i + 1] = len(pops_cat)
        pops_arr = np.asarray(pops_cat, dtype=pop_dtype)
        key = (np.packbits(np.array(k, dtype=bool)).tobytes()
               if pack_key else k)
        out[key] = (uis, offs, pops_arr)
    return out


def encode_arena(cache):
    """One global pool; per-key just (uis_slice, row bounds)."""
    all_uis, all_offs, all_pops = [], [0], []
    index = {}
    for k, v in cache.items():
        row_start = len(all_uis)
        for ui, pops, _ in v:
            all_uis.append(ui)
            all_pops.extend(pops)
            all_offs.append(len(all_pops))
        key = np.packbits(np.array(k, dtype=bool)).tobytes()
        index[key] = (row_start, len(all_uis))
    return {'uis': np.asarray(all_uis, np.int32),
            'offs': np.asarray(all_offs, np.int32),
            'pops': np.asarray(all_pops, np.int32),
            'index': index}


def hit_reconstruct_current(entry):
    out = []
    for ui, pops, probs in entry:
        out.append((ui, pops, probs))
    return out


def hit_reconstruct_csr(entry):
    uis, offs, pops = entry
    out = []
    for i in range(len(uis)):
        p = pops[offs[i]:offs[i + 1]].tolist()
        n = len(p)
        out.append((int(uis[i]), p, [1.0 / n] * n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patterns', type=int, default=200)
    ap.add_argument('--dpsize', default='really_friggin_small')
    args = ap.parse_args()

    random.seed(31415)
    np.random.seed(31415)
    from core.deployment_setup import get_random_deployment
    from core.path_distribution_computer import _LocalPathDistributionComputer
    dep = get_random_deployment(args.dpsize, port=31600)
    # positional: compatible with both the pre-desharding signature
    # (worker_i, subdeployment, init_kwargs, static_dep=None) and the
    # current (worker_i, deployment, init_kwargs)
    w = _LocalPathDistributionComputer(
        0, dep, {
            'lambduh': 1.0, 'gamma': 0, 'verbose': False,
            'n_prefixes': None, 'with_capacity': False,
            'save_run_dir': None, 'generic_objective': 'avg_latency'})

    # populate the REAL cache via the production miss path
    n_popp, n_pref = len(w.popps), w.n_prefixes
    rng = np.random.default_rng(3)
    t0 = time.time()
    i = 0
    while len(getattr(w, 'pattern_cache', {})) < args.patterns:
        a = (rng.random((n_popp, n_pref)) > .5).astype(np.float64)
        np.random.seed(3000 + i)
        w.latency_benefit(a, retnow=True, generic_obj='avg_latency')
        i += 1
        if i > args.patterns * 3:
            break
    cache = w.pattern_cache
    n_entries = len(cache)
    n_pairs = sum(len(v) for v in cache.values())
    print('populated: {} distinct patterns, {} (ug,pops) pairs, '
          '{:.1f}s'.format(n_entries, n_pairs, time.time() - t0))

    reprs = {
        'A current (list of (ui,pops,probs))': cache,
        'B noprobs': encode_noprobs(cache),
        'C csr per entry': encode_csr(cache, pack_key=False),
        'D csr + packed key': encode_csr(cache, pack_key=True),
        'D2 csr + packed key + uint16 pops': encode_csr(
            cache, pack_key=True, pop_dtype=np.uint16),
        'E global arena + packed key': encode_arena(cache),
    }
    print('\n{:<36s} {:>10s} {:>10s} {:>12s}'.format(
        'scheme', 'total', 'per entry', 'per pair'))
    base = None
    for name, r in reprs.items():
        b = deep_bytes(r)
        if base is None:
            base = b
        print('{:<36s} {:>9.2f}M {:>9.1f}K {:>10.1f}B  ({:.1%})'.format(
            name, b / 1048576, b / n_entries / 1024, b / n_pairs,
            b / base))

    # hit-path cost: reconstruct every entry once
    t0 = time.time()
    for v in cache.values():
        hit_reconstruct_current(v)
    t_cur = time.time() - t0
    csr = reprs['C csr per entry']
    t0 = time.time()
    for v in csr.values():
        hit_reconstruct_csr(v)
    t_csr = time.time() - t0
    print('\nhit-path reconstruct (all {} entries): current {:.1f}ms, '
          'csr {:.1f}ms ({:.1f}x)'.format(
              n_entries, t_cur * 1000, t_csr * 1000,
              t_csr / max(t_cur, 1e-9)))


if __name__ == '__main__':
    main()
