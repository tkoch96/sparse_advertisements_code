"""Steady-state congestion + clean-latency scoring for wave advertisements.

For each (seed, rung, N-dir): one steady LP in-world, recording
  steady_congested_frac : the LP's fraction_congested_volume (steady state)
  clean_avg_lat         : vol-weighted avg latency over ROUTED traffic only
                          (UGs whose latency < NO_ROUTE/2, i.e. excluding
                          the 30s sentinel for stranded volume)
  routed_frac           : volume fraction included in clean_avg_lat
World knobs come from env as usual. Child-per-seed isolation.

    SCULPTOR_LAT_MODEL=geo ... python -m dashboard.steady_metrics \
        --dirs cache/ablation/nsweep_v2_georand/N1,cache/ablation/nsweep_v2_georand/N50 \
        --tag georand --seeds 1-5
Output: cache/model_error/steady/<tag>.json
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


MARKER = 'STEADY_METRICS '
SENTINEL_CUT = 15000.0


def child(seed, dirs, dpsize):
    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(seed)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from helpers.constants import DEFAULT_EXPLORE
    from evaluations.wrapper_eval import capacity
    from core.deployment_setup import get_random_deployment
    from core.sparse_advertisements_v3 import Sparse_Advertisement_Eval
    from helpers.helpers import deployment_to_prefixes

    dep = get_random_deployment(dpsize)
    dep['generic_objective'] = 'avg_latency'
    sas = Sparse_Advertisement_Eval(
        dep, verbose=False, lambduh=0, with_capacity=capacity,
        explore=DEFAULT_EXPLORE, using_resilience_benefit=False, gamma=0,
        n_prefixes=deployment_to_prefixes(dep),
        generic_objective='avg_latency')
    vols = np.asarray(sas.ug_vols, dtype=float)

    def eval_adv(base, adv):
        ret = sas.solve_lp_with_failure_catch(
            np.asarray(adv, dtype=float))
        if not ret.get('solved', True):
            base['solved'] = False
            return base
        lats = np.asarray(ret['lats_by_ug'], dtype=float).flatten()
        routed = lats < SENTINEL_CUT
        routed_vol = float(vols[routed].sum())
        clean = (float(np.average(lats[routed], weights=vols[routed]))
                 if routed_vol > 0 else None)
        _obj = ret.get('objective')
        base.update({
            'solved': True,
            'steady_congested_frac':
                float(ret.get('fraction_congested_volume', 0.0)),
            'clean_avg_lat': clean,
            'routed_frac': routed_vol / float(vols.sum()),
            # cost form of the soft-bounded scalar training descends:
            # avg routed lat + SOFT_CONG_PENALTY * frac(cong+noroute)
            'train_obj': (-float(_obj) if _obj is not None else None),
        })
        return base

    out = []
    # Reference advertisements FIRST, through the SAME eval path (Tom
    # 2026-08-14: refs must flow through the one pipeline so objective
    # changes never desync them). FIRST because shared-instance eval
    # state drifts with call count — refs evaluated last came back
    # inflated (opp per-failure cost pinned at the soft bound while a
    # fresh process said 21-23ms; same hazard as the 2026-08-11
    # in-process contamination incident). dir='REFS' entries; plots
    # route them separately, never through the (arm, N) key parser.
    out.append(eval_adv(
        {'dir': 'REFS', 'seed': seed, 'rung': 'opp'},
        np.eye(len(sas.popps))))
    _pg = os.environ.get(
        'SCULPTOR_EVAL_PAINTER_GLOB',
        'cache/ablation/hardB3/painter_georand/N1/seed_{}_painter.json')
    _pfn = _pg.format(seed)
    if os.path.exists(_pfn):
        with open(_pfn) as f:
            out.append(eval_adv(
                {'dir': 'REFS', 'seed': seed, 'rung': 'painter'},
                json.load(f)['adv']))
    for d in dirs:
        for fn in sorted(glob.glob(os.path.join(
                d, 'seed_{}_*.json'.format(seed)))):
            with open(fn) as f:
                r = json.load(f)
            if 'adv' not in r:
                continue
            out.append(eval_adv(
                {'dir': d, 'seed': seed, 'rung': r['rung'],
                 'n_iters': r.get('n_iters'),
                 'probes_spent': r.get('probes_spent'),
                 'exit_reason': r.get('exit_reason')}, r['adv']))
    # CANARY: opp re-evaluated LAST. Drift vs the first eval = the
    # shared-instance contamination is back; downstream loudly warns.
    out.append(eval_adv(
        {'dir': 'REFS', 'seed': seed, 'rung': 'opp_canary'},
        np.eye(len(sas.popps))))
    # SANITY (Tom 2026-08-17): one-per-peering exposes every path, so in
    # THIS evaluator no advertisement can beat opp on the bounded
    # composite (train_obj). clean_avg_lat is conditional-on-routed and
    # can legitimately dip below opp when routed_frac < 1 (the 79%%-
    # congested L5N5s1 incident) — the invariant is train_obj ONLY.
    _opp_train = out[0].get('train_obj') if out[0].get('solved') else None
    if _opp_train is not None:
        _tol = max(0.05, 1e-3 * abs(_opp_train))
        for rec in out[1:]:
            t = rec.get('train_obj')
            if (rec.get('solved') and t is not None
                    and t < _opp_train - _tol):
                rec['sanity_violation'] = True
                print('[SANITY] seed {} {} {}: train_obj {:.3f} < opp '
                      '{:.3f} — impossible in-evaluator; eval bug'.format(
                          seed, rec.get('dir'), rec.get('rung'),
                          t, _opp_train), flush=True)
    print(MARKER + json.dumps(out), flush=True)


def parse_seeds(spec):
    if '-' in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(',')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seeds', default='1-5')
    ap.add_argument('--dpsize', default='small')
    ap.add_argument('--child-seed', type=int, default=None)
    args = ap.parse_args()
    dirs = args.dirs.split(',')

    if args.child_seed is not None:
        child(args.child_seed, dirs, args.dpsize)
        return

    results = []
    for s in parse_seeds(args.seeds):
        env = dict(os.environ)
        env.update({'RAY_ADDRESS': 'local',
                    'RAY_TMPDIR': '/tmp/ray_steady_{}_{}'.format(
                        os.getpid(), s)})
        p = subprocess.run(
            [sys.executable, '-m', 'dashboard.steady_metrics',
             '--child-seed', str(s), '--dirs', args.dirs,
             '--dpsize', args.dpsize, '--tag', args.tag],
            env=env, capture_output=True, text=True)
        lines = [l for l in p.stdout.splitlines() if l.startswith(MARKER)]
        if not lines:
            print('[steady] seed {} FAILED: {}'.format(
                s, (p.stdout + p.stderr)[-300:]), flush=True)
            continue
        results.extend(json.loads(lines[-1][len(MARKER):]))
        print('[steady] seed {} ok'.format(s), flush=True)

    if not results:
        print('[{}] NO results (eval children all failed -- e.g. license); '
              'NOT clobbering the existing store'.format(__name__), flush=True)
        return
    out_dir = os.path.join('cache', 'model_error', 'steady')
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, '{}.json'.format(args.tag))
    with open(fn, 'w') as f:
        json.dump(results, f, indent=2)
    for e in results:
        if e.get('solved'):
            print('{} seed{} {:>10}: steady_cong={:>6.1%} routed={:>6.1%} clean_lat={}'.format(
                e['dir'].split('/')[-1], e['seed'], e['rung'],
                e['steady_congested_frac'], e['routed_frac'],
                'n/a' if e['clean_avg_lat'] is None
                else '{:.1f}ms'.format(e['clean_avg_lat'])), flush=True)
    # THROW after persisting (Tom 2026-08-17): flagged records stay in
    # the store for forensics, but the run fails loudly so no dashboard
    # step quietly plots an impossible below-opp composite.
    bad = [e for e in results if e.get('sanity_violation')]
    assert not bad, (
        'SANITY: {} record(s) beat one-per-peering on train_obj — '
        'impossible in-evaluator; see [SANITY] lines / store {}'.format(
            len(bad), fn))


if __name__ == '__main__':
    main()
