"""Paired A/B: does SCULPTOR_MC_NUM=1 cost us convergence quality?

    python experiments/mc_ab/mc_ab.py --seeds 10 --iters 200 --dpsize really_friggin_small

Background
----------
`MC_NUM` is the number of Monte-Carlo draws of the joint routing
distribution taken per latency-benefit evaluation
(`_sample_scenario_realizations`, MC_NUM x solve_generic_lp_persistent).
It was hardcoded to 5 on the production Ray actor -- the env read lived in
a base `__init__` that `_LocalPathDistributionComputer` bypasses, so
`SCULPTOR_MC_NUM` had no effect on any real run (fixed 2026-08-21).

Dropping to 1 is a ~5x cut on the dominant per-job cost, but it is NOT a
cheaper route to the same number: 1 draw is a single-sample noisy
estimator of the benefit distribution. This harness measures what that
noise costs.

Design
------
PAIRED, because single-trial comparisons at a fixed seed are
noise-dominated in this codebase (`SCULPTOR_DEPLOYMENT_SEED` does not pin
the gradient-step RNG). For each of N deployment seeds we run BOTH arms on
the SAME deployment from the SAME initial advertisement, then compare
within-pair. Across-pair variance -- which is large -- cancels.

The arms cannot share an RNG stream by construction: MC=5 draws five times
as many random numbers per evaluation, so the trajectories diverge after
the first benefit call no matter what seed we set. That divergence IS the
effect under test; the paired design is what makes it measurable.

Reported per pair: final objective (lower is better -- this is the
minimised objective, not a benefit), iterations run, wall time. Then a
paired comparison across seeds: mean/median delta, a sign test, and a
paired t-statistic. With 10 pairs the sign test is the honest headline;
the t-statistic assumes normality we have not earned.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def run_one(dpsize, seed, mc_num, iters, port, out_dir):
    """One arm. Returns a dict of convergence stats, or {'error': ...}."""
    # MUST be set before Worker_Manager.start_workers -- the actors read it
    # at construction.
    os.environ['SCULPTOR_MC_NUM'] = str(mc_num)
    os.environ['SCULPTOR_MAX_ITER'] = str(iters)
    os.environ['SCULPTOR_N_WORKERS'] = os.environ.get('MC_AB_WORKERS', '4')

    import numpy as np
    from helpers.constants import DEFAULT_EXPLORE
    from core.deployment_setup import get_random_deployment
    from core.sparse_advertisements_v3 import Sparse_Advertisement_Solver
    from core.worker_comms import Worker_Manager
    from helpers.helpers import deployment_to_prefixes
    from evaluations.wrapper_eval import capacity

    # Same deployment for both arms of a pair.
    np.random.seed(seed)
    deployment = get_random_deployment(dpsize)
    deployment['port'] = port
    n_prefixes = deployment_to_prefixes(deployment)

    # Same INIT for both arms: re-seed immediately before constructing the
    # solver so the initial advertisement is drawn from the same state
    # regardless of how many draws the deployment build consumed.
    np.random.seed(seed * 1000 + 7)

    sas = Sparse_Advertisement_Solver(
        deployment, verbose=False, lambduh=0, with_capacity=capacity,
        explore=DEFAULT_EXPLORE, using_resilience_benefit=True, gamma=1.0,
        n_prefixes=n_prefixes)
    wm = Worker_Manager(sas.get_init_kwa(), deployment)
    wm.start_workers()
    sas.set_worker_manager(wm)

    t0 = time.time()
    err = None
    try:
        sas.solve()
    except Exception as e:                        # noqa: BLE001
        import traceback
        traceback.print_exc()
        err = '{}: {}'.format(type(e).__name__, e)
    wall = time.time() - t0

    m = getattr(sas, 'metrics', {}) or {}

    def _traj(key):
        v = m.get(key) or []
        return [float(x) for x in v if isinstance(x, (int, float))]

    pseudo = _traj('pseudo_objectives')
    actual = _traj('actual_nonconvex_objective')
    effective = _traj('effective_objectives')

    try:
        wm.stop_workers()
    except Exception:                             # noqa: BLE001
        pass

    return {
        'dpsize': dpsize, 'seed': seed, 'mc_num': mc_num,
        'requested_iters': iters, 'wall_s': wall, 'error': err,
        'n_iters_run': len(pseudo),
        'final_pseudo': pseudo[-1] if pseudo else None,
        'final_actual': actual[-1] if actual else None,
        'final_effective': effective[-1] if effective else None,
        'best_actual': min(actual) if actual else None,
        'best_effective': min(effective) if effective else None,
        'traj_pseudo': pseudo, 'traj_effective': effective,
    }


def _sign_test(deltas):
    """Two-sided sign test p-value; exact binomial, no scipy needed."""
    from math import comb
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0, pos, neg
    k = min(pos, neg)
    tail = sum(comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail / (2 ** n)), pos, neg


def analyse(rows, metric='final_effective'):
    pairs = {}
    for r in rows:
        if r.get('error'):
            continue
        pairs.setdefault(r['seed'], {})[r['mc_num']] = r
    deltas, table = [], []
    for seed in sorted(pairs):
        p = pairs[seed]
        if 1 not in p or 5 not in p:
            continue
        a, b = p[5].get(metric), p[1].get(metric)
        if a is None or b is None:
            continue
        d = b - a                       # MC1 - MC5; objective is MINIMISED,
        deltas.append(d)                # so positive = MC1 is WORSE
        table.append((seed, a, b, d, p[5]['wall_s'], p[1]['wall_s'],
                      p[5]['n_iters_run'], p[1]['n_iters_run']))
    return deltas, table


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--dpsize', default='really_friggin_small')
    ap.add_argument('--mc', default='5,1')
    ap.add_argument('--port', type=int, default=31780)
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed-start', type=int, default=1)
    a = ap.parse_args()

    out_dir = a.out or os.path.join(_REPO, 'cache', 'mc_ab')
    os.makedirs(out_dir, exist_ok=True)
    out_fn = os.path.join(out_dir, 'mc_ab_{}_{}iters.json'.format(
        a.dpsize, a.iters))
    mcs = [int(x) for x in a.mc.split(',')]

    rows = []
    if os.path.exists(out_fn):
        try:
            rows = json.load(open(out_fn))
            print('resuming: {} rows already done'.format(len(rows)))
        except ValueError:
            rows = []
    done = {(r['seed'], r['mc_num']) for r in rows}

    port = a.port
    for seed in range(a.seed_start, a.seed_start + a.seeds):
        for mc in mcs:
            if (seed, mc) in done:
                continue
            print('\n' + '=' * 68)
            print('seed {}  MC_NUM={}  dpsize={}  iters={}'.format(
                seed, mc, a.dpsize, a.iters))
            print('=' * 68, flush=True)
            port += 1
            r = run_one(a.dpsize, seed, mc, a.iters, port, out_dir)
            rows.append(r)
            json.dump(rows, open(out_fn, 'w'), indent=1)
            print('  -> iters={} final_actual={} wall={:.1f}s {}'.format(
                r['n_iters_run'], r['final_actual'], r['wall_s'],
                'ERROR: ' + r['error'] if r['error'] else ''), flush=True)

    print('\n\n' + '=' * 68)
    print('PAIRED RESULTS  ({})'.format(out_fn))
    print('=' * 68)
    for metric in ('final_actual', 'final_pseudo', 'best_actual'):
        deltas, table = analyse(rows, metric)
        if not deltas:
            continue
        print('\n--- metric: {} (lower = better) ---'.format(metric))
        print('{:>5} {:>14} {:>14} {:>12} {:>9} {:>9}'.format(
            'seed', 'MC=5', 'MC=1', 'delta', 'w5(s)', 'w1(s)'))
        for seed, a5, b1, d, w5, w1, i5, i1 in table:
            print('{:>5} {:>14.5f} {:>14.5f} {:>+12.5f} {:>9.0f} {:>9.0f}'
                  .format(seed, a5, b1, d, w5, w1))
        n = len(deltas)
        mean = sum(deltas) / n
        srt = sorted(deltas)
        med = (srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2]))
        p, pos, neg = _sign_test(deltas)
        print('  n pairs      : {}'.format(n))
        print('  mean delta   : {:+.5f}   (positive = MC=1 WORSE)'.format(mean))
        print('  median delta : {:+.5f}'.format(med))
        print('  sign test    : {} worse / {} better, p = {:.4f}'.format(
            pos, neg, p))
        if n > 1:
            var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
            se = (var / n) ** 0.5
            print('  paired t     : {:.3f}  (se {:.5f})'.format(
                mean / se if se else float('nan'), se))
    ok = [r for r in rows if not r.get('error')]
    for mc in mcs:
        ws = [r['wall_s'] for r in ok if r['mc_num'] == mc]
        if ws:
            print('\nMC={}: mean wall {:.1f}s over {} runs'.format(
                mc, sum(ws) / len(ws), len(ws)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
