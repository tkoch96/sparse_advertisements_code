"""Failure-scenario metrics for wave advertisements: popp AND pop
failures, congestion plus the latency of AFFECTED users (Tom, 2026-08-13).

For each (dir, seed, rung) and each single-entity failure scenario
(every popp; every pop = all popps at that site, rescore_fork's exact
enumeration):
  cong                 : LP fraction_congested_volume under the failure
  affected users       : UGs carrying volume on the failed entity in the
                         PRE-failure LP assignment (paths_by_ug)
  affected_routed_lat  : vol-weighted mean POST-failure latency of the
                         affected users that still route (lat < 15000);
                         stranded affected volume is reported separately
                         (affected_stranded_frac) so the latency panel
                         is not sentinel-polluted
Aggregates per cell: mean congestion over all scenarios; mean
affected_routed_lat / affected_stranded_frac over scenarios that have
affected users. World knobs from env as usual. Child-per-seed isolation,
seeds run in parallel (5 Gurobi sessions).

    SCULPTOR_LAT_MODEL=geo SCULPTOR_PREF_MODEL=random \
        python -m dashboard.failure_metrics \
        --dirs <d1,d2,...> --tag policy_failure_fixed --seeds 1-5
Output: cache/model_error/failure/<tag>.json
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


MARKER = 'FAILURE_METRICS '
SENTINEL_CUT = 15000.0


def _ug_poppis(ret, ug_to_ind):
    """Normalize ret['paths_by_ug'] into {ug_index: set(poppi)} across the
    key/value shapes the LP layers produce (ug tuple or index; bare poppi
    or (poppi, vol_pct))."""
    out = {}
    for ug, paths in (ret.get('paths_by_ug') or {}).items():
        if isinstance(ug, (int, np.integer)):
            ui = int(ug)
        else:
            ui = ug_to_ind.get(tuple(ug) if isinstance(ug, list) else ug)
        if ui is None:
            continue
        s = out.setdefault(ui, set())
        for p in (paths if isinstance(paths, (list, tuple)) else []):
            poppi = p[0] if isinstance(p, (list, tuple)) else p
            s.add(int(poppi))
    return out


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
    ug_to_ind = {ug: i for i, ug in enumerate(sas.ugs)}

    def scenarios(which):
        if which == 'popp':
            for popp in sas.popps:
                yield [sas.popp_to_ind[popp]]
        else:
            for pop in sas.pops:
                yield [sas.popp_to_ind[p] for p in sas.popps if p[0] == pop]

    def solve(a):
        ret = sas.solve_lp_with_failure_catch(np.asarray(a, dtype=float))
        return ret if ret.get('solved', True) else None

    def eval_adv(entry, a):
            a = np.asarray(a, dtype=float)
            pre = solve(a)
            if pre is None:
                entry['solved'] = False
                return entry
            pre_up = _ug_poppis(pre, ug_to_ind)
            entry['solved'] = True
            _soft_P = float(os.environ.get(
                'SCULPTOR_SOFT_CONG_PENALTY', '50'))
            for which in ('popp', 'pop'):
                congs, aff_lats, aff_strand, obj_costs = [], [], [], []
                for failed in scenarios(which):
                    fs = set(failed)
                    aff = [ui for ui, ps in pre_up.items() if ps & fs]
                    a2 = np.copy(a)
                    a2[list(fs), :] = 0
                    ret2 = solve(a2) if a2.sum() > 0 else None
                    if ret2 is None:
                        congs.append(1.0)
                        # everything unrouted: soft objective = P * 1
                        obj_costs.append(_soft_P)
                        if aff:
                            aff_strand.append(1.0)
                        continue
                    congs.append(
                        float(ret2.get('fraction_congested_volume', 0.0)))
                    _o = ret2.get('objective')
                    obj_costs.append(-float(_o) if _o is not None
                                     else _soft_P)
                    if not aff:
                        continue
                    lats = np.asarray(
                        ret2['lats_by_ug'], dtype=float).flatten()
                    al, av = lats[aff], vols[aff]
                    routed = al < SENTINEL_CUT
                    aff_strand.append(
                        float(av[~routed].sum() / av.sum()))
                    if routed.any():
                        aff_lats.append(float(
                            np.average(al[routed], weights=av[routed])))
                entry[which] = {
                    'cong_mean': float(np.mean(congs)),
                    'cong_max': float(np.max(congs)),
                    # sum over scenarios of the soft-bounded LP cost —
                    # for popp this is training's resilience term
                    # Sum_popps cost(a with popp failed) (gamma applied
                    # at plot time)
                    'obj_cost_sum': float(np.sum(obj_costs)),
                    'affected_routed_lat_mean':
                        float(np.mean(aff_lats)) if aff_lats else None,
                    'affected_stranded_frac_mean':
                        float(np.mean(aff_strand)) if aff_strand else None,
                    'n_scenarios': len(congs),
                    'n_scenarios_with_affected': len(aff_strand),
                }
            return entry

    out = []
    # REFS FIRST + trailing canary (see steady_metrics: shared-instance
    # eval state drifts with call count; refs evaluated last were
    # inflated to the soft bound).
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
                {'dir': d, 'seed': seed, 'rung': r['rung']}, r['adv']))
    out.append(eval_adv(
        {'dir': 'REFS', 'seed': seed, 'rung': 'opp_canary'},
        np.eye(len(sas.popps))))
    print(MARKER + json.dumps(out), flush=True)


def parse_seeds(spec):
    if '-' in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(',')]


def run_seed(s, args):
    env = dict(os.environ)
    env.update({'RAY_ADDRESS': 'local',
                'RAY_TMPDIR': '/tmp/ray_failm_{}_{}'.format(
                    os.getpid(), s)})
    p = subprocess.run(
        [sys.executable, '-m', 'dashboard.failure_metrics',
         '--child-seed', str(s), '--dirs', args.dirs,
         '--dpsize', args.dpsize, '--tag', args.tag],
        env=env, capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith(MARKER)]
    if not lines:
        print('[failm] seed {} FAILED: {}'.format(
            s, (p.stdout + p.stderr)[-300:]), flush=True)
        return []
    print('[failm] seed {} ok'.format(s), flush=True)
    return json.loads(lines[-1][len(MARKER):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seeds', default='1-5')
    ap.add_argument('--dpsize', default='small')
    ap.add_argument('--child-seed', type=int, default=None)
    ap.add_argument('--jobs', type=int, default=5)
    args = ap.parse_args()

    if args.child_seed is not None:
        child(args.child_seed, args.dirs.split(','), args.dpsize)
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for res in ex.map(lambda s: run_seed(s, args),
                          parse_seeds(args.seeds)):
            results.extend(res)

    if not results:
        print('[{}] NO results (eval children all failed -- e.g. license); '
              'NOT clobbering the existing store'.format(__name__), flush=True)
        return
    out_dir = os.path.join('cache', 'model_error', 'failure')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, '{}.json'.format(args.tag)), 'w') as f:
        json.dump(results, f, indent=2)
    for e in results:
        if e.get('solved'):
            print('{} seed{} {:>10}: popp cong={:.3f} aff_lat={} | pop cong={:.3f} aff_lat={}'.format(
                '/'.join(e['dir'].split('/')[-2:]),
                e['seed'], e['rung'],
                e['popp']['cong_mean'],
                'n/a' if e['popp']['affected_routed_lat_mean'] is None
                else '{:.1f}'.format(e['popp']['affected_routed_lat_mean']),
                e['pop']['cong_mean'],
                'n/a' if e['pop']['affected_routed_lat_mean'] is None
                else '{:.1f}'.format(e['pop']['affected_routed_lat_mean'])),
                flush=True)


if __name__ == '__main__':
    main()
