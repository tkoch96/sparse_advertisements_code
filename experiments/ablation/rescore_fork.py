"""Authoritative re-scoring of fork-ladder results. Deliberately boring:

  * one fresh subprocess per seed (never shares a process with a solver)
  * RAY_ADDRESS=local + private RAY_TMPDIR: never attaches to a running
    cluster (attaching to the sweep's busy cluster crashed evals)
  * NO Worker_Manager / no _mp fan-out: every LP is a driver-side
    sas.solve_lp_with_failure_catch call in a plain loop

Metrics per advertisement (all repo LP code, NO_ROUTE volume charged
NO_ROUTE_LATENCY):
  avg_lat                      : steady volume-weighted avg latency
  fail_popp/fail_pop:
    avg_lat_under_failure_abs  : mean over single popp/pop failures of the
                                 vol-weighted avg latency (LP re-assignment)
    opp_avg_lat_under_failure_abs : same for one_per_peering
The pipeline objective is avg_lat + gamma * avg_lat_under_failure_abs
(combined at plot time so gamma is a display choice).

    python -m experiments.ablation.rescore_fork --in-dir cache/ablation/fork_full_res --all
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


MARKER = 'lp_driver_v2'


def rescore_seed(seed, in_dir, dpsize):
    fns = sorted(glob.glob(os.path.join(in_dir, 'seed_{}_*.json'.format(seed))))
    todo = []
    for fn in fns:
        with open(fn) as f:
            r = json.load(f)
        if 'adv' in r and r.get('fail_eval') != MARKER:
            todo.append((fn, r))
    if not todo:
        print('[rescore seed {}] nothing to do'.format(seed), flush=True)
        return

    # isolation: fresh local Ray, never attach to a running cluster
    os.environ['RAY_ADDRESS'] = 'local'
    os.environ['RAY_TMPDIR'] = '/tmp/ray_rescore_{}'.format(os.getpid())
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
        n_prefixes=deployment_to_prefixes(dep), generic_objective='avg_latency')
    vols = np.asarray(sas.ug_vols)

    def steady(adv):
        ret = sas.solve_lp_with_failure_catch(np.asarray(adv, dtype=float))
        return float(np.average(np.asarray(ret['lats_by_ug']), weights=vols))

    def scenarios(which):
        if which == 'popps':
            for popp in sas.popps:
                yield [sas.popp_to_ind[popp]]
        else:
            for pop in sas.pops:
                yield [sas.popp_to_ind[p] for p in sas.popps if p[0] == pop]

    # SCULPTOR_RESCORE_STORE_SCENARIOS=1: additionally persist the
    # per-failure-scenario latencies (for scenario-level CDFs, cdf_fork.py).
    # Default (unset) keeps the original aggregate-only JSON schema.
    store_scen = os.environ.get('SCULPTOR_RESCORE_STORE_SCENARIOS', '0') == '1'

    def fail_abs(adv, which):
        a = np.asarray(adv, dtype=float)
        per_s = []
        for failed in scenarios(which):
            a2 = np.copy(a)
            a2[failed, :] = 0
            if a2.sum() == 0:
                per_s.append(float(np.average(
                    np.full(len(vols), 30000.0), weights=vols)))
                continue
            per_s.append(steady(a2))
        return float(np.mean(per_s)), per_s

    opp_adv = np.eye(sas.n_popps)
    opp_steady = steady(opp_adv)
    opp_fail, opp_fail_scen = {}, {}
    for w in ('popps', 'pops'):
        opp_fail[w], opp_fail_scen[w] = fail_abs(opp_adv, w)

    for fn, r in todo:
        old = r.get('diff_vs_opp')
        r['avg_lat'] = steady(r['adv'])
        r['opp_avg_lat'] = opp_steady
        r['diff_vs_opp'] = r['avg_lat'] - opp_steady
        for which, key in (('popps', 'fail_popp'), ('pops', 'fail_pop')):
            mean_abs, per_scen = fail_abs(r['adv'], which)
            r[key] = {
                'avg_lat_under_failure_abs': mean_abs,
                'opp_avg_lat_under_failure_abs': opp_fail[which],
            }
            if store_scen:
                r[key]['per_scenario_lats'] = per_scen
                r[key]['opp_per_scenario_lats'] = opp_fail_scen[which]
        r.pop('opp_fail', None)
        r['rescored'] = True
        r['fail_eval'] = MARKER
        r['driver_diff_vs_opp'] = old
        with open(fn, 'w') as f:
            json.dump(r, f, indent=2, default=float)
        comb = r['diff_vs_opp'] + 4 * (r['fail_popp']['avg_lat_under_failure_abs']
                                       - r['fail_popp']['opp_avg_lat_under_failure_abs'])
        print('[rescore seed {} {}] steady={:+.3f} combined(g4)={:+.3f}'.format(
            seed, r['rung'], r['diff_vs_opp'], comb), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', required=True)
    p.add_argument('--dpsize', default='small')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--all', action='store_true')
    p.add_argument('--port-base', type=int, default=0, help='unused; kept for CLI compat')
    args = p.parse_args()

    if args.seed is not None:
        rescore_seed(args.seed, args.in_dir, args.dpsize)
        return
    assert args.all
    seeds = sorted({int(os.path.basename(fn).split('_')[1])
                    for fn in glob.glob(os.path.join(args.in_dir, 'seed_*_*.json'))})
    for seed in seeds:
        subprocess.run([sys.executable, '-m', 'experiments.ablation.rescore_fork',
                        '--in-dir', args.in_dir, '--dpsize', args.dpsize,
                        '--seed', str(seed)], cwd=_REPO_ROOT)


if __name__ == '__main__':
    main()
