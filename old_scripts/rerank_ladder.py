"""Re-rank existing ladder final advertisements under the new hard
objectives (hard-vs-easy objective evidence).

For each seed in a rescored ladder dataset (default
cache/ablation/fork_small_20x200_v3): build the stock world (env knobs
apply as usual), compute capacity-aware per-UG optimal latencies, then
score every rung's final advertisement under:
  - frac_beyond {10,50,100}  (threshold objective)
  - avg_lat + alpha*MLU      (nonlinear max objective)
  - popp-failure congestion  (joint/rare-event objective; n_popps LPs)
plus steady avg_lat for reference. Per-seed subprocess isolation
(rescore recipe). Output: one JSON per seed under
cache/model_error/rerank/<tag>/seed_<s>.json + an aggregate table
printed at the end.

    python -m experiments.model_error.rerank_ladder --seeds 1-20 \
        --in-dir cache/ablation/fork_small_20x200_v3 --tag fixedmode_v3
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MARKER = 'RERANK_RESULT '


def rerank_seed(seed, in_dir, dpsize, out_dir):
    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(seed)
    os.environ.setdefault('MPLBACKEND', 'Agg')

    from helpers.constants import DEFAULT_EXPLORE
    from evaluations.wrapper_eval import capacity
    from core.deployment_setup import get_random_deployment
    from core.sparse_advertisements_v3 import Sparse_Advertisement_Eval
    from helpers.helpers import deployment_to_prefixes
    from core import hard_objectives as O

    dep = get_random_deployment(dpsize)
    dep['generic_objective'] = 'avg_latency'
    sas = Sparse_Advertisement_Eval(
        dep, verbose=False, lambduh=0, with_capacity=capacity,
        explore=DEFAULT_EXPLORE, using_resilience_benefit=False, gamma=0,
        n_prefixes=deployment_to_prefixes(dep),
        generic_objective='avg_latency')
    vols = np.asarray(sas.ug_vols, dtype=float)
    best = O.best_lats_by_ug(sas)

    out = {'seed': seed, 'dpsize': dpsize, 'rungs': {}}
    for fn in sorted(glob.glob(os.path.join(
            in_dir, 'seed_{}_*.json'.format(seed)))):
        with open(fn) as f:
            r = json.load(f)
        if 'adv' not in r:
            continue
        adv = np.asarray(r['adv'], dtype=float)
        ret = O._steady_ret(sas, adv)
        if ret is None:
            out['rungs'][r['rung']] = {'solved': False}
            continue
        lats = np.asarray(ret['lats_by_ug'], dtype=float)
        entry = {
            'avg_lat': float(np.average(lats, weights=vols)),
            'frac_beyond': O.frac_users_beyond(
                sas, adv, xs=(10, 50, 100), best_lats=best, ret=ret),
            'lat_plus_mlu': O.latency_plus_max_util(
                sas, adv, best_lats=best),
            'popp_fail': O.frac_congested_under_popp_failures(sas, adv),
        }
        entry['popp_fail'].pop('per_scenario', None)
        entry['lat_plus_mlu'].pop('per_scenario', None)
        out['rungs'][r['rung']] = entry
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'seed_{}.json'.format(seed)), 'w') as f:
        json.dump(out, f, indent=2)
    print(MARKER + json.dumps({'seed': seed, 'n_rungs': len(out['rungs'])}),
          flush=True)


def parse_seeds(spec):
    if '-' in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(',')]


def aggregate(out_dir):
    import statistics as st
    per_rung = {}
    for fn in glob.glob(os.path.join(out_dir, 'seed_*.json')):
        with open(fn) as f:
            r = json.load(f)
        for rung, e in r['rungs'].items():
            if not e.get('avg_lat'):
                continue
            per_rung.setdefault(rung, []).append(e)
    print('{:>12} {:>3} | {:>8} | {:>7} {:>7} {:>7} | {:>9} {:>8} | {:>9} {:>9}'.format(
        'rung', 'n', 'avg_lat', 'fb10', 'fb50', 'fb100',
        'lat+aMLU', 'MLU', 'pfail_mu', 'pfail_mx'))
    for rung, es in sorted(per_rung.items(),
                           key=lambda kv: st.median(
                               e['avg_lat'] for e in kv[1])):
        print('{:>12} {:>3} | {:8.3f} | {:7.3f} {:7.3f} {:7.3f} | {:9.3f} {:8.3f} | {:9.4f} {:9.4f}'.format(
            rung, len(es),
            st.median(e['avg_lat'] for e in es),
            st.median(e['frac_beyond']['10'] for e in es),
            st.median(e['frac_beyond']['50'] for e in es),
            st.median(e['frac_beyond']['100'] for e in es),
            st.median(e['lat_plus_mlu']['objective'] for e in es),
            st.median(e['lat_plus_mlu']['max_util'] for e in es),
            st.median(e['popp_fail']['mean'] for e in es),
            st.median(e['popp_fail']['max'] for e in es)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='1-20')
    ap.add_argument('--in-dir', default='cache/ablation/fork_small_20x200_v3')
    ap.add_argument('--dpsize', default='small')
    ap.add_argument('--tag', required=False, default='fixedmode_v3')
    ap.add_argument('--child-seed', type=int, default=None)
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()

    out_dir = os.path.join('cache', 'model_error', 'rerank', args.tag)
    if args.child_seed is not None:
        rerank_seed(args.child_seed, args.in_dir, args.dpsize, out_dir)
        return
    if args.aggregate_only:
        aggregate(out_dir)
        return
    for s in parse_seeds(args.seeds):
        if os.path.exists(os.path.join(out_dir, 'seed_{}.json'.format(s))):
            print('[rerank] seed {} exists, skipping'.format(s), flush=True)
            continue
        env = dict(os.environ)
        env.update({'RAY_ADDRESS': 'local',
                    'RAY_TMPDIR': '/tmp/ray_rerank_{}_{}'.format(
                        os.getpid(), s)})
        p = subprocess.run(
            [sys.executable, '-m', 'experiments.model_error.rerank_ladder',
             '--child-seed', str(s), '--in-dir', args.in_dir,
             '--dpsize', args.dpsize, '--tag', args.tag],
            env=env, capture_output=True, text=True)
        ok = MARKER in p.stdout
        print('[rerank] seed {} {}'.format(s, 'ok' if ok else
              'FAILED: ' + (p.stdout + p.stderr)[-300:]), flush=True)
    aggregate(out_dir)


if __name__ == '__main__':
    main()
