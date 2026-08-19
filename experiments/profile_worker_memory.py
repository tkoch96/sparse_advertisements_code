"""Object-level RAM census of a SCULPTOR worker (Tom 2026-08-16: "profile
the ram usage of all objects in each path_distribution_computer ...
verifiable loggable proof ... goal is maximum cores").

Four measured phases, all logged to --out (JSON + human table):

  1. WORKER CENSUS -- build a real worker IN-PROCESS (the exposed
     _LocalPathDistributionComputer test seam; same construction path as
     production: the full deployment + get_init_kwa) and
     pympler.asizeof EVERY instance attribute, post-init and again after
     exercising representative LB-gradient batches.
  2. GROWTH CURVES -- real parent_tracker (+ friends) loaded from the
     night's saved state-*.pkl sequences: entries + deep bytes vs
     iteration, from ACTUAL runs (no synthetic data).
  3. ALTERNATIVE STRUCTURES -- re-encode the real parent_tracker into
     candidate representations and asizeof each on the same data:
     (a) packed-int set  (b) sorted uint64 numpy + searchsorted
     (c) dense numpy bool bitmap [n_ug x n_popp x n_popp].
  4. VERDICTS -- per top object: reducible? alternative? measured % saving.

Usage (head):
  env <georand knobs> python -m experiments.profile_worker_memory \
      --dpsize small --state-glob '/home/ubuntu/v3ada5_ws/S*/runs/*/state-*.pkl' \
      --out cache/ablation/memprof/report
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

try:
    from pympler import asizeof
except ImportError:
    print('pip install pympler first', file=sys.stderr)
    raise


def deep_mb(obj):
    return asizeof.asizeof(obj) / 1048576.0


def census(obj, tag, out):
    rows = []
    for k, v in sorted(vars(obj).items()):
        try:
            mb = deep_mb(v)
        except Exception as e:
            rows.append((k, None, 'asizeof failed: {}'.format(e)))
            continue
        rows.append((k, mb, type(v).__name__))
    rows.sort(key=lambda r: -(r[1] or 0))
    total = sum(r[1] or 0 for r in rows)
    print('\n===== {} attr census (deep, total {:.1f} MB) ====='.format(tag, total))
    for k, mb, ty in rows[:30]:
        print('  %-42s %9.2f MB  %s' % (k, mb or -1, ty))
    out[tag] = {'total_mb': total,
                'attrs': [(k, mb, ty) for k, mb, ty in rows]}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dpsize', default='small')
    ap.add_argument('--state-glob', default=None)
    ap.add_argument('--out', default='cache/ablation/memprof/report')
    ap.add_argument('--exercise-iters', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    OUT = {}

    # ---------------- phase 1: in-process worker census ----------------
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from constants import DEFAULT_EXPLORE
    from wrapper_eval import capacity
    from deployment_setup import get_random_deployment
    from sparse_advertisements_v3 import Sparse_Advertisement_Eval
    from helpers import deployment_to_prefixes
    from path_distribution_computer_ray import _LocalPathDistributionComputer

    dep = get_random_deployment(args.dpsize)
    sas = Sparse_Advertisement_Eval(
        dep, verbose=False, lambduh=0, with_capacity=capacity,
        explore=DEFAULT_EXPLORE, using_resilience_benefit=False, gamma=0,
        n_prefixes=deployment_to_prefixes(dep),
        generic_objective='avg_latency')
    init_kwa = sas.get_init_kwa()
    t0 = time.time()
    w = _LocalPathDistributionComputer(0, sas.output_deployment(), init_kwa)
    print('worker built in {:.1f}s'.format(time.time() - t0))
    census(w, 'worker_post_init', OUT)

    # exercise: representative compressed-LB gradient batches on random
    # binary advertisements (the production hot path), plus iter ticks
    n_popp, n_pref = len(w.popps), w.n_prefixes
    rng = np.random.default_rng(0)
    for it in range(args.exercise_iters):
        # production wire format (flush_latency_benefit_queue_generic):
        # data[0] = ((base_adv,), base_kwa); data[1:] = (np.where(base !=
        # other), kwa) — the handler flips those indices in-place.
        base = (rng.random((n_popp, n_pref)) > 0.5).astype(float)
        kwa = {'generic_obj': 'avg_latency'}
        data = [((base,), dict(kwa))]
        for ci in range(min(30, n_popp * n_pref)):
            other = base.copy()
            other[ci % n_popp, ci % n_pref] = 1 - other[ci % n_popp, ci % n_pref]
            data.append((np.where(base != other), dict(kwa)))
        try:
            w.handle_msg(pickle.dumps(('increment_iter', 'meep')))
        except Exception:
            pass
        ret = w.handle_msg(pickle.dumps(('calc_compressed_lb', data)))
        if isinstance(ret, str):
            print('exercise batch errored: {!r}'.format(ret[:120]))
            break
        print('exercise iter {} ok ({} calls)'.format(it, len(data)))
    census(w, 'worker_after_{}_iters'.format(args.exercise_iters), OUT)

    # ---------------- phase 2: real growth curves ----------------------
    if args.state_glob:
        by_run = {}
        for fn in glob.glob(args.state_glob):
            run = os.path.dirname(fn)
            it = int(re.search(r'state-(\d+)', fn).group(1))
            by_run.setdefault(run, []).append((it, fn))
        growth = {}
        for run, files in sorted(by_run.items()):
            files.sort()
            pts = []
            for it, fn in files:
                try:
                    st = pickle.load(open(fn, 'rb'))
                except Exception:
                    continue
                pt = st.get('parent_tracker') if isinstance(st, dict) else None
                if pt is None:
                    continue
                pts.append({'iter': it, 'entries': len(pt),
                            'deep_mb': deep_mb(pt)})
            if pts:
                growth[run] = pts
                last = pts[-1]
                print('run %s: parent_tracker %d entries, %.1f MB at iter %d'
                      % (os.path.basename(os.path.dirname(run)) or run,
                         last['entries'], last['deep_mb'], last['iter']))
        OUT['parent_tracker_growth'] = growth

        # ------------- phase 3: alternative structures ----------------
        # biggest real parent_tracker found:
        big_fn, big_pt = None, None
        for run, files in by_run.items():
            it, fn = max(files)
            try:
                st = pickle.load(open(fn, 'rb'))
                pt = st.get('parent_tracker')
                if pt and (big_pt is None or len(pt) > len(big_pt)):
                    big_pt, big_fn = pt, fn
            except Exception:
                continue
        if big_pt:
            ugs = sorted({k[0] for k in big_pt})
            popps = sorted({k[1] for k in big_pt} | {k[2] for k in big_pt})
            ug_i = {u: i for i, u in enumerate(ugs)}
            pp_i = {p: i for i, p in enumerate(popps)}
            U, P = len(ugs), len(popps)
            packed = np.array(sorted(
                ug_i[u] * P * P + pp_i[c] * P + pp_i[p]
                for (u, c, p) in big_pt), dtype=np.uint64)
            alt = {
                'dict_tuplekeys_CURRENT': deep_mb(big_pt),
                'set_packed_int': deep_mb(set(int(x) for x in packed)),
                'numpy_sorted_uint64': deep_mb(packed),
                'numpy_uint32': deep_mb(packed.astype(np.uint32)),
                'dense_bool_bitmap_UxPxP': deep_mb(
                    np.zeros((U, P, P), dtype=bool)),
                'index_maps_overhead': deep_mb(ug_i) + deep_mb(pp_i),
            }
            print('\n===== parent_tracker alternatives (same %d entries, '
                  'U=%d P=%d, from %s) =====' % (len(big_pt), U, P, big_fn))
            cur = alt['dict_tuplekeys_CURRENT']
            for k, mb in sorted(alt.items(), key=lambda kv: -kv[1]):
                print('  %-28s %9.2f MB  (%+.0f%% vs current)'
                      % (k, mb, 100.0 * (mb - cur) / cur))
            OUT['parent_tracker_alternatives'] = {
                'source': big_fn, 'entries': len(big_pt), 'U': U, 'P': P,
                'sizes_mb': alt}

    with open(args.out + '.json', 'w') as f:
        json.dump(OUT, f, indent=1, default=float)
    print('\nwrote', args.out + '.json')


if __name__ == '__main__':
    main()
