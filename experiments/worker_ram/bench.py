"""Isolated RAM/caching bench for path_distribution_computer (Tom 2026-08-20).

Measures, OUTSIDE the iteration loop, on a real deployment:
  1. lb-cache key cost: get_a_cache_rep tuple keys vs packbits keys
     (size + build time), on realistic gradient-probe advertisements.
  2. parent_tracker repr: tuple-key dict vs compact int32 ndarray
     (size + per-pattern-miss scan time, current python loop vs vectorized).
  3. rti_data transient churn per latency_benefit call:
     choices_matrix int64->int16, P_matrix float64->float32.
  4. var_pool growth over a probe-shaped workload.

Usage: python -m experiments.worker_ram.bench [dpsize] [n_calls]
"""
import os, sys, time, copy, pickle
import numpy as np

os.environ.setdefault('SCULPTOR_DEPLOYMENT_SEED', '1')
os.environ.setdefault('SCULPTOR_LP_BACKEND', 'highs')
os.environ.setdefault('SCULPTOR_MC_NUM', '1')

import core.deployment_setup as deployment_setup
from core.path_distribution_computer import _LocalPathDistributionComputer
from core.path_distribution_computer import get_a_cache_rep
from helpers.constants import ADVERTISEMENT_THRESHOLD
from helpers.helpers import threshold_a


def deep_sz(o, seen=None):
    if seen is None:
        seen = set()
    if id(o) in seen:
        return 0
    seen.add(id(o))
    if isinstance(o, np.ndarray):
        return o.nbytes
    s = sys.getsizeof(o, 0)
    if isinstance(o, dict):
        s += sum(deep_sz(k, seen) + deep_sz(v, seen) for k, v in o.items())
    elif isinstance(o, (list, tuple, set, frozenset)):
        s += sum(deep_sz(x, seen) for x in o)
    return s


def main():
    dpsize = sys.argv[1] if len(sys.argv) > 1 else 'decent'
    n_calls = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    n_prefixes = 6

    np.random.seed(1)
    dep = deployment_setup.get_random_deployment(dpsize)
    n_popp = len(dep['popps'])
    print(f"[bench] dpsize={dpsize} n_popp={n_popp} n_ug={len(dep['ugs'])}")

    init_kwargs = {
        'lambduh': 0.1, 'gamma': 1.0, 'with_capacity': False,
        'verbose': False, 'init': {'type': 'normal', 'var': 0.01},
        'explore': 'entropy', 'using_resilience_benefit': True,
        'n_prefixes': n_prefixes,
        'save_run_dir': '/tmp/worker_ram_bench_save_run',
        'generic_objective': 'avg_latency',
    }
    os.makedirs(init_kwargs['save_run_dir'], exist_ok=True)
    w = _LocalPathDistributionComputer(worker_i=0, deployment=dep,
                                       init_kwargs=init_kwargs)

    # ---- workload: gradient-probe shaped (single-entry flips around a base) ----
    np.random.seed(2)
    base = (ADVERTISEMENT_THRESHOLD + 0.1 *
            np.random.normal(size=(n_popp, n_prefixes)))
    t0 = time.time()
    for i in range(n_calls):
        a = copy.copy(base)
        # flip 1-2 random entries: the driver's probe batches look like this
        for _ in range(np.random.randint(1, 3)):
            a[np.random.randint(n_popp), np.random.randint(n_prefixes)] += \
                np.random.choice([-.4, .4])
        w.latency_benefit(a, generic_obj='avg_latency')
        if i % 10 == 0:
            base = a  # drift the base like SGD steps do
    wall = time.time() - t0
    print(f"[bench] {n_calls} latency_benefit calls in {wall:.1f}s "
          f"({1000*wall/n_calls:.0f} ms/call)")

    # ---- 1. lb cache keys ----
    lb = w.calc_cache.all_caches['lb']
    keys = list(lb.keys())
    key_bytes = deep_sz(keys)
    val_bytes = deep_sz(list(lb.values()))
    # packbits alternative
    ts = time.time()
    for _ in range(200):
        get_a_cache_rep(base)
    t_tuple = (time.time() - ts) / 200
    ts = time.time()
    for _ in range(200):
        np.packbits(threshold_a(base).astype(bool)).tobytes()
    t_pack = (time.time() - ts) / 200
    pack_key_len = len(np.packbits(threshold_a(base).astype(bool)).tobytes())
    n_on = int(np.sum(threshold_a(base)))
    print(f"\n[lb-cache] entries={len(lb)} keys={key_bytes/1e6:.1f}MB "
          f"values={val_bytes/1e6:.1f}MB")
    print(f"[lb-key] n_on={n_on} tuple_key={key_bytes/max(1,len(lb))/1024:.1f}KB/entry "
          f"packbits_key={pack_key_len}B/entry "
          f"(x{key_bytes/max(1,len(lb))/max(1,pack_key_len):.0f} smaller)")
    print(f"[lb-key] build time: tuple {1e6*t_tuple:.0f}us vs packbits {1e6*t_pack:.0f}us "
          f"(x{t_tuple/t_pack:.0f} faster)")

    # ---- 2. parent_tracker repr ----
    # synthesize realistic entries: for each ug, pairs among its popps
    pt = {}
    rng = np.random.RandomState(3)
    for ug in dep['ugs']:
        pps = list(dep['ug_perfs'][ug])
        if len(pps) < 2:
            continue
        for _ in range(min(30, len(pps) * 2)):
            c, p = rng.randint(len(pps)), rng.randint(len(pps))
            if c != p:
                pt[(ug, pps[c], pps[p])] = True
    n_ent = len(pt)
    dict_bytes = deep_sz(pt)
    # compact: int32 array [ui, childi, parenti]
    arr = np.zeros((n_ent, 3), dtype=np.int32)
    for i, (ug, c, p) in enumerate(pt):
        arr[i] = (w.whole_deployment_ug_to_ind[ug],
                  w.popp_to_ind[c], w.popp_to_ind[p])
    print(f"\n[parent_tracker] synthetic n={n_ent} "
          f"tuple-dict={dict_bytes/1e6:.1f}MB int32-array={arr.nbytes/1e6:.2f}MB "
          f"(x{dict_bytes/arr.nbytes:.0f} smaller)")
    # scan cost: current python loop vs vectorized active-parent check
    active = rng.rand(n_popp) > 0.5
    ts = time.time()
    blocked = set()
    for (ug, c, p) in pt:
        if active[w.popp_to_ind[p]]:
            blocked.add((w.whole_deployment_ug_to_ind[ug], w.popp_to_ind[c]))
    t_loop = time.time() - ts
    ts = time.time()
    m = active[arr[:, 2]]
    blocked_v = np.unique(arr[m][:, :2], axis=0)
    t_vec = time.time() - ts
    print(f"[parent_tracker] per-pattern-miss scan: python {1000*t_loop:.1f}ms "
          f"vs vectorized {1000*t_vec:.2f}ms (x{t_loop/max(t_vec,1e-9):.0f} faster), "
          f"blocked sets equal={blocked == set(map(tuple, blocked_v))}")

    # ---- 3. rti transient churn ----
    rd = w.rti_data
    n_scen = rd.get('num_scenarios', 0)
    maxc = rd.get('max_choices', 0)
    cm = rd.get('choices_matrix')
    cm_bytes = cm.nbytes if cm is not None else 0
    print(f"\n[rti-transient] per-call: n_scen={n_scen} max_choices={maxc} "
          f"choices_matrix({cm.dtype if cm is not None else '-'})={cm_bytes/1e6:.1f}MB "
          f"as int16={n_scen*maxc*2/1e6:.1f}MB; P_matrix f64={n_scen*maxc*8/1e6:.1f}MB "
          f"as f32={n_scen*maxc*4/1e6:.1f}MB")
    md = rd.get('meta_data', [])
    print(f"[rti-transient] meta_data list: {len(md)} tuples = {deep_sz(md)/1e6:.1f}MB rebuilt/call")

    # ---- 4. var_pool / pattern cache ----
    print(f"\n[var_pool] entries={len(w.var_pool)} "
          f"active_now={len(getattr(w, '_active_keys', []) or [])}")
    pc = getattr(w, 'pattern_cache', {})
    print(f"[pattern_cache] entries={len(pc)} bytes={deep_sz(pc)/1e6:.1f}MB")

    import resource
    print(f"\n[rss] peak={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.0f}MB (mac: bytes/1e6)")


if __name__ == '__main__':
    main()
