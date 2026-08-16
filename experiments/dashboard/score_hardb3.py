"""Incremental own-objective scoring for the hardB3 grid (Mac-side,
while the VM trains). Idempotent: caches per-cell scores keyed by file
path+mtime in cache/model_error/hardB3_scores.json; each pass scores
only new arrivals. Parallel child per (world, seed).

Child mode: --child <world> <seed>  scores all unscored cells of that
world+seed and prints one SCORE line per cell (parent merges).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

REPO = '/Users/tomkoch/Documents/sparse_advertisements_code'
STORE = os.path.join(REPO, 'cache/model_error/hardB3_scores.json')
ROOT = 'cache/ablation/hardB3'  # overridable via --root (hardB3v2 etc.)
MARKER = 'SCORE '

# 'mlu' = lat_plus_max_util as of 2026-08-15 LATE (Tom): pure max_util
# is GAMEABLE BY STRANDING -- the min-MLU LP parks no-route volume on a
# 100000-capacity pseudo-ingress, so sparse advs that strand users beat
# one-per-peering (impossible under a correct MLU: opp's feasible set is
# a superset). The latency term charges stranded volume NO_ROUTE_LATENCY,
# closing the exploit; MLU rides as alpha*Y (alpha self-scales to the
# vol-weighted optimal-latency floor). The pure-MLU era is quarantined in
# PUREMLU_STRANDING_ERA_mlu on the head.
# 'mlupure' = the STANDALONE MLU objective v2 (Tom 2026-08-16): dominant
# minMLU weight + routability charge, latency as tie-break only.
OBJ_OF = {'fracb': 'frac_beyond_optimal', 'mlu': 'lat_plus_max_util',
          'mlupure': 'max_util',
          'prio': 'joint_latency_bulk_download', 'popfail': 'site_failure'}
WORLD_OF = {'fracb': 'geo', 'mlu': 'geo', 'mlupure': 'geo', 'prio': 'geo', 'popfail': 'stock'}
# painter reference sets, scored under every objective of their world
PAINTER_ROOT = {'geo': 'painter_georand', 'stock': 'painter_stock'}


def cells(world=None, seed=None):
    """Yield (key, path, objective_dirname, rung_label) for grid cells and
    painter refs. key = repo-relative path."""
    for f in glob.glob(os.path.join(
            REPO, ROOT, '*/*/N*/seed_*_*.json')):
        rel = os.path.relpath(f, REPO)
        parts = rel.split(os.sep)
        o = parts[3]
        if o not in OBJ_OF:
            continue
        s = int(os.path.basename(f).split('_')[1])
        if world and WORLD_OF[o] != world:
            continue
        if seed and s != seed:
            continue
        yield rel, f, o, s
    for w, proot in PAINTER_ROOT.items():
        if world and w != world:
            continue
        for f in glob.glob(os.path.join(
                REPO, ROOT, proot, 'N*', 'seed_*_*.json')):
            s = int(os.path.basename(f).split('_')[1])
            if seed and s != seed:
                continue
            for o, ow in WORLD_OF.items():
                if ow == w:
                    yield ('painter:{}:'.format(o)
                           + os.path.relpath(f, REPO)), f, o, s


def child(world, seed, todo_keys):
    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(seed)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('RAY_ADDRESS', 'local')
    os.environ.setdefault('RAY_TMPDIR', '/tmp/ray_h3s_{}_{}'.format(
        world, seed))
    os.environ['SCULPTOR_XOBJS'] = '1'
    os.environ.pop('SCULPTOR_FRAC_BEYOND_REL', None)
    sys.path.insert(0, REPO)
    if world == 'geo':
        from experiments.model_error import worlds
        worlds.apply('georand')
    import numpy as np
    from constants import DEFAULT_EXPLORE
    from wrapper_eval import capacity
    from deployment_setup import get_random_deployment
    from sparse_advertisements_v3 import Sparse_Advertisement_Eval
    from helpers import deployment_to_prefixes
    from solve_lp_assignment import solve_generic_lp_with_failure_catch

    dep = get_random_deployment('small')
    dep['generic_objective'] = 'avg_latency'
    sas = Sparse_Advertisement_Eval(
        dep, verbose=False, lambduh=0, with_capacity=capacity,
        explore=DEFAULT_EXPLORE, using_resilience_benefit=False, gamma=0,
        n_prefixes=deployment_to_prefixes(dep),
        generic_objective='avg_latency')

    todo = set(todo_keys)
    opp_cache = {}
    for key, f, o, s in cells(world, seed):
        if key not in todo:
            continue
        obj = OBJ_OF[o]
        adv = np.asarray(json.load(open(f))['adv'], dtype=float)
        try:
            rti, _ = sas.calculate_ground_truth_ingress(adv)
            ret = solve_generic_lp_with_failure_catch(sas, rti, obj, adv=adv)
            val = (float(ret['objective'])
                   if ret.get('solved') and ret.get('objective') is not None
                   else None)
            # component capture (Tom 2026-08-16: the joint-objective
            # scalar REPLACED utilization on the dash — the MLU tab must
            # show real utilization, so persist the ret components)
            comp = {c: float(ret[c]) for c in
                    ('max_util', 'steady_avg_lat', 'bad_frac', 'mlu_alpha')
                    if ret.get(c) is not None}
        except Exception as e:
            val, comp = None, {}
        if o not in opp_cache:
            a0 = np.eye(sas.n_popps)
            r0, _ = sas.calculate_ground_truth_ingress(a0)
            try:
                q = solve_generic_lp_with_failure_catch(sas, r0, obj, adv=a0)
                opp_cache[o] = ((float(q['objective'])
                                 if q.get('solved') else None),
                                {c: float(q[c]) for c in
                                 ('max_util', 'steady_avg_lat', 'bad_frac')
                                 if q.get(c) is not None})
            except Exception:
                opp_cache[o] = (None, {})
        print(MARKER + json.dumps(
            {'key': key, 'mtime': os.path.getmtime(f), 'obj_val': val,
             'opp_val': opp_cache[o][0], 'components': comp or None,
             'opp_components': opp_cache[o][1] or None}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--child', nargs=2, metavar=('WORLD', 'SEED'))
    ap.add_argument('--todo-file')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--root', default=None,
                    help='repo-rel data root (default cache/ablation/hardB3)')
    ap.add_argument('--store', default=None,
                    help='scores JSON path (default hardB3_scores.json)')
    args = ap.parse_args()
    global ROOT, STORE
    if args.root:
        ROOT = args.root
    if args.store:
        STORE = os.path.join(REPO, args.store)
    if args.child:
        child(args.child[0], int(args.child[1]),
              json.load(open(args.todo_file)))
        return

    store = json.load(open(STORE)) if os.path.exists(STORE) else {}
    # PRUNE entries whose result file no longer exists (quarantines/
    # deletions): without this the tab mixes eras — 87 pure-MLU-era rows
    # displayed alongside the redo (caught by Tom 2026-08-16 ~03:20Z).
    def _key_file(k):
        return k.split(':', 2)[2] if k.startswith('painter:') else k
    n0 = len(store)
    store = {k: v for k, v in store.items()
             if os.path.exists(os.path.join(REPO, _key_file(k)))}
    if len(store) != n0:
        print('[prune] dropped {} stale entries'.format(n0 - len(store)))
        json.dump(store, open(STORE, 'w'))
    work = {}
    for key, f, o, s in cells():
        rec = store.get(key)
        if rec and abs(rec['mtime'] - os.path.getmtime(f)) < 1:
            continue
        work.setdefault((WORLD_OF[o], s), []).append(key)
    print('{} cells to score across {} children'.format(
        sum(len(v) for v in work.values()), len(work)))

    def run(ws):
        (w, s), keys = ws
        tf = '/tmp/h3s_todo_{}_{}.json'.format(w, s)
        json.dump(keys, open(tf, 'w'))
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), '--child', w, str(s),
             '--todo-file', tf, '--root', ROOT],
            capture_output=True, text=True,
            env=dict(os.environ, PYTHONPATH=REPO))
        out = []
        for line in p.stdout.splitlines():
            if line.startswith(MARKER):
                out.append(json.loads(line[len(MARKER):]))
        if not out and keys:
            print('[score] child {}/{} produced nothing: {}'.format(
                w, s, (p.stdout + p.stderr)[-200:]))
        return out

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for recs in ex.map(run, sorted(work.items())):
            for r in recs:
                store[r['key']] = r
    json.dump(store, open(STORE, 'w'))
    print('store now has {} scored cells'.format(len(store)))


if __name__ == '__main__':
    main()
