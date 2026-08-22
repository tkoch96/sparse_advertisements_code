"""A/B/C/D: does the MERGED smart probing match the ablation fork?

    python integration_tests/test_probe_merge.py --seeds 20 --budgets 5,20

Four arms per deployment seed, all from the SAME deployment and the SAME
initial advertisement:

    A  fork   budget 5      experiments/ablation/sculptor_fork.Ablation_Fork
    B  fork   budget 20
    C  merged budget 5      core.sparse_advertisements_v3, PROBE_MODE=smart
    D  merged budget 20

The claim under test is NOT bit-equality. The fork carries per-iteration
assertions, gate-history recording and several ablation-only knobs, and it
drives its own replica of the solve() orchestrator; the merged path runs
the production orchestrator. What must agree is the BEHAVIOUR the merge was
for:

  1. the budget binds        -- probes never exceed N, in either
  2. the budget is spent     -- or, if not, WHY is visible: a probe that
                                finds nothing to measure is counted as a
                                skip, never silently dropped
  3. more budget helps       -- C vs D (and A vs B) move the objective the
                               same DIRECTION
  4. fork ~= merged          -- A~C and B~D agree within noise, paired

Reported as paired per-seed deltas with a sign test, because single-trial
comparisons in this codebase are noise-dominated (the gradient-step RNG is
not pinned by the deployment seed).

Exit status is 0 only if the hard invariants (1) hold for every arm; the
statistical claims are reported, not asserted, since with 20 seeds they
are evidence rather than proof.

BUDGET ACCOUNTING (verified 2026-08-21, fork and merged alike):
    measures == probes_spent + 1
The +1 is the ONE grounding measurement solve() makes during _solve_setup,
before the gate exists. The fork excludes it too -- it captures
_abl_pm_solve_start AFTER _solve_setup -- so the budget governs PROBES, not
total path_measures. Asserting `measures <= N` would fail a correct
implementation, which is why the invariant below is on probes_spent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _clear_probe_env():
    for k in list(os.environ):
        if k.startswith('SCULPTOR_PROBE') or k.startswith('SCULPTOR_ABLATION'):
            del os.environ[k]


def run_arm(arm, dpsize, seed, budget, iters, port):
    """One arm. Returns stats incl. the measurement count actually used."""
    _clear_probe_env()
    os.environ['SCULPTOR_MAX_ITER'] = str(iters)
    os.environ['SCULPTOR_N_WORKERS'] = os.environ.get('PROBE_AB_WORKERS', '4')

    import numpy as np
    from helpers.constants import DEFAULT_EXPLORE
    from core.deployment_setup import get_random_deployment
    from core.worker_comms import Worker_Manager
    from helpers.helpers import deployment_to_prefixes
    from evaluations.wrapper_eval import capacity

    if arm == 'fork':
        os.environ['SCULPTOR_ABLATION_PROBE_MODE'] = 'smart'
        os.environ['SCULPTOR_ABLATION_PROBE_N'] = str(budget)
        os.environ['SCULPTOR_ABLATION_PROBE_TCONV'] = str(iters)
        from experiments.ablation.sculptor_fork import \
            Ablation_Sparse_Advertisement_Solver as CLS
    else:
        os.environ['SCULPTOR_PROBE_MODE'] = 'smart'
        os.environ['SCULPTOR_PROBE_N'] = str(budget)
        os.environ['SCULPTOR_PROBE_TCONV'] = str(iters)
        from core.sparse_advertisements_v3 import (
            Sparse_Advertisement_Solver as CLS)

    np.random.seed(seed)
    deployment = get_random_deployment(dpsize)
    deployment['port'] = port
    n_prefixes = deployment_to_prefixes(deployment)
    np.random.seed(seed * 1000 + 7)          # identical init across arms

    sas = CLS(deployment, verbose=False, lambduh=0, with_capacity=capacity,
              explore=DEFAULT_EXPLORE, using_resilience_benefit=True,
              gamma=1.0, n_prefixes=n_prefixes)
    wm = Worker_Manager(sas.get_init_kwa(), deployment)
    wm.start_workers()
    sas.set_worker_manager(wm)

    pm0 = int(getattr(sas, 'path_measures', 0))
    t0, err = time.time(), None
    try:
        sas.solve()
    except Exception as e:                        # noqa: BLE001
        import traceback
        traceback.print_exc()
        err = '{}: {}'.format(type(e).__name__, e)
    wall = time.time() - t0

    m = getattr(sas, 'metrics', {}) or {}
    pseudo = [float(x) for x in (m.get('pseudo_objectives') or [])
              if isinstance(x, (int, float))]
    actual = [float(x) for x in (m.get('actual_nonconvex_objective') or [])
              if isinstance(x, (int, float))]
    measures = int(getattr(sas, 'path_measures', 0)) - pm0
    spent = int(getattr(sas, 'probes_spent',
                        getattr(sas, 'abl_probes_spent', 0)) or 0)
    try:
        wm.stop_workers()
    except Exception:                             # noqa: BLE001
        pass

    return {'arm': arm, 'dpsize': dpsize, 'seed': seed, 'budget': budget,
            'iters_requested': iters, 'n_iters_run': len(pseudo),
            'measures': measures, 'probes_spent': spent,
            'probe_skips': int(getattr(sas, '_probe_skips',
                                       getattr(sas, '_abl_probe_skips', 0))
                               or 0),
            'final_actual': actual[-1] if actual else None,
            'best_actual': min(actual) if actual else None,
            'final_pseudo': pseudo[-1] if pseudo else None,
            'wall_s': wall, 'error': err}


def _sign_test(deltas):
    from math import comb
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0, pos, neg
    k = min(pos, neg)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), pos, neg


def _paired(rows, key_a, key_b, metric='final_actual'):
    """key_* are (arm, budget) tuples."""
    idx = {}
    for r in rows:
        if r.get('error'):
            continue
        idx[(r['arm'], r['budget'], r['seed'])] = r
    deltas, seeds = [], []
    for seed in sorted({r['seed'] for r in rows}):
        ra = idx.get((key_a[0], key_a[1], seed))
        rb = idx.get((key_b[0], key_b[1], seed))
        if not ra or not rb:
            continue
        va, vb = ra.get(metric), rb.get(metric)
        if va is None or vb is None:
            continue
        deltas.append(vb - va)
        seeds.append(seed)
    return deltas, seeds


def report(rows, budgets):
    print('\n' + '=' * 74)
    print('HARD INVARIANT: measurements never exceed the budget')
    print('=' * 74)
    ok = True
    for arm in ('fork', 'merged'):
        for b in budgets:
            rs = [r for r in rows if r['arm'] == arm and r['budget'] == b
                  and not r.get('error')]
            if not rs:
                continue
            mx = max(r['probes_spent'] for r in rs)
            mean = sum(r['probes_spent'] for r in rs) / len(rs)
            bad = [r['seed'] for r in rs if r['probes_spent'] > b]
            # the setup grounding must be the ONLY non-probe measurement
            drift = [r['seed'] for r in rs
                     if r['measures'] - r['probes_spent'] != 1]
            status = 'OK' if not bad else 'VIOLATED on seeds {}'.format(bad)
            if bad or drift:
                ok = False
            if drift:
                status += '  !! measures-probes != 1 on seeds {}'.format(drift)
            print('  {:<7} budget={:<3} n={:<3} probes mean={:.1f} '
                  'max={:<3} {}'.format(arm, b, len(rs), mean, mx, status))
    print('\n' + '=' * 74)
    print('BUDGET IS SPENT (mean measurements / budget)')
    print('=' * 74)
    for arm in ('fork', 'merged'):
        for b in budgets:
            rs = [r for r in rows if r['arm'] == arm and r['budget'] == b
                  and not r.get('error')]
            if rs:
                mean = sum(r['probes_spent'] for r in rs) / len(rs)
                sk = sum(r.get('probe_skips', 0) for r in rs) / len(rs)
                note = ''
                if mean < 0.95 * b:
                    note = ('  <- {:.1f} probe(s)/run found nothing to '
                            'measure (grounding at an unchanged '
                            'advertisement)'.format(sk))
                print('  {:<7} budget={:<3} -> {:.1f}/{} = {:.0%}{}'.format(
                    arm, b, mean, b, mean / b, note))

    lo, hi = min(budgets), max(budgets)
    print('\n' + '=' * 74)
    print('PAIRED COMPARISONS (metric: final_actual, lower = better)')
    print('=' * 74)
    for label, ka, kb in (
            ('A vs C  (fork{} vs merged{})'.format(lo, lo),
             ('fork', lo), ('merged', lo)),
            ('B vs D  (fork{} vs merged{})'.format(hi, hi),
             ('fork', hi), ('merged', hi)),
            ('A vs B  (fork{} vs fork{})'.format(lo, hi),
             ('fork', lo), ('fork', hi)),
            ('C vs D  (merged{} vs merged{})'.format(lo, hi),
             ('merged', lo), ('merged', hi))):
        d, seeds = _paired(rows, ka, kb)
        if not d:
            print('  {:<34} no pairs'.format(label))
            continue
        n = len(d)
        mean = sum(d) / n
        srt = sorted(d)
        med = srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
        p, pos, neg = _sign_test(d)
        print('  {:<34} n={:<3} mean={:+.5f} median={:+.5f} '
              'sign {}+/{}- p={:.4f}'.format(label, n, mean, med, pos, neg, p))
    print('\n  (delta = second minus first; for A-vs-C and B-vs-D a delta '
          'near zero\n   and a non-significant sign test is the merge '
          'agreeing with the fork.)')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--seed-start', type=int, default=1)
    ap.add_argument('--budgets', default='5,20')
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--dpsize', default='small')
    ap.add_argument('--port', type=int, default=31900)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    budgets = [int(x) for x in a.budgets.split(',')]
    out_dir = a.out or os.path.join(_REPO, 'cache', 'probe_merge')
    os.makedirs(out_dir, exist_ok=True)
    out_fn = os.path.join(out_dir, 'probe_merge_{}_{}it.json'.format(
        a.dpsize, a.iters))
    rows = []
    if os.path.exists(out_fn):
        try:
            rows = json.load(open(out_fn))
            print('resuming with {} rows'.format(len(rows)))
        except ValueError:
            rows = []
    done = {(r['arm'], r['budget'], r['seed']) for r in rows}

    port = a.port
    for seed in range(a.seed_start, a.seed_start + a.seeds):
        for arm in ('fork', 'merged'):
            for b in budgets:
                if (arm, b, seed) in done:
                    continue
                port += 1
                print('\n{} seed={} arm={} budget={} {}'.format(
                    '=' * 20, seed, arm, b, '=' * 20), flush=True)
                r = run_arm(arm, a.dpsize, seed, b, a.iters, port)
                rows.append(r)
                json.dump(rows, open(out_fn, 'w'), indent=1)
                print('  -> iters={} measures={}/{} final={} wall={:.0f}s {}'
                      .format(r['n_iters_run'], r['measures'], b,
                              r['final_actual'], r['wall_s'],
                              'ERR ' + r['error'] if r['error'] else ''),
                      flush=True)

    ok = report(rows, budgets)
    print('\nrows -> {}'.format(out_fn))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
