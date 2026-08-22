"""Convergence as a function of measurement budget (dpsize=small).

    python integration_tests/test_convergence_vs_budget.py --seeds 5 \
        --budgets 2,5,10,20,40 --iters 60

Sweeps SCULPTOR_PROBE_N over the merged smart-probing path and plots what
the budget buys. Every budget runs on the SAME set of deployments from the
SAME initial advertisements, so the curves are paired across budgets and
the (large) across-deployment variance cancels.

Output: figures/integration_tests/convergence_vs_budget_<dpsize>.png
  left   objective trajectory vs iteration, one line per budget
         (median across seeds, IQR band)
  right  final objective vs budget, per seed and median -- the
         diminishing-returns curve

Why this is an integration test and not a unit test: it exercises the real
solver, real workers and the real probe gate end-to-end, and its value is
the artifact plus the two invariants it asserts:

  * probes never exceed the budget, for every budget and seed
  * the budget is actually spent (>= half of it), so a flat curve means
    "more measurements do not help" rather than "the gate never fired"

Both are checked; the exit status reflects them. The SHAPE of the curve is
reported, not asserted -- convergence quality is noisy at this scale and a
hard threshold would be a flaky test rather than a meaningful one.

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

FIG_DIR = os.path.join(_REPO, 'figures', 'integration_tests')


def run_one(dpsize, seed, budget, iters, port):
    for k in list(os.environ):
        if k.startswith('SCULPTOR_PROBE') or k.startswith('SCULPTOR_ABLATION'):
            del os.environ[k]
    os.environ['SCULPTOR_MAX_ITER'] = str(iters)
    os.environ['SCULPTOR_N_WORKERS'] = os.environ.get('CVB_WORKERS', '4')
    os.environ['SCULPTOR_PROBE_MODE'] = 'smart'
    os.environ['SCULPTOR_PROBE_N'] = str(budget)
    os.environ['SCULPTOR_PROBE_TCONV'] = str(iters)

    import numpy as np
    from helpers.constants import DEFAULT_EXPLORE
    from core.deployment_setup import get_random_deployment
    from core.sparse_advertisements_v3 import Sparse_Advertisement_Solver
    from core.worker_comms import Worker_Manager
    from helpers.helpers import deployment_to_prefixes
    from evaluations.wrapper_eval import capacity

    np.random.seed(seed)
    deployment = get_random_deployment(dpsize)
    deployment['port'] = port
    n_prefixes = deployment_to_prefixes(deployment)
    np.random.seed(seed * 1000 + 7)

    sas = Sparse_Advertisement_Solver(
        deployment, verbose=False, lambduh=0, with_capacity=capacity,
        explore=DEFAULT_EXPLORE, using_resilience_benefit=True, gamma=1.0,
        n_prefixes=n_prefixes)
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
    actual = [float(x) for x in (m.get('actual_nonconvex_objective') or [])
              if isinstance(x, (int, float))]
    measures = int(getattr(sas, 'path_measures', 0)) - pm0
    try:
        wm.stop_workers()
    except Exception:                             # noqa: BLE001
        pass
    return {'dpsize': dpsize, 'seed': seed, 'budget': budget,
            'iters_requested': iters, 'n_iters_run': len(actual),
            'measures': measures,
            'probes_spent': int(getattr(sas, 'probes_spent', 0) or 0),
            'wall_s': wall, 'error': err,
            'final_actual': actual[-1] if actual else None,
            'best_actual': min(actual) if actual else None,
            'traj': actual}


def plot(rows, dpsize, out_png):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ok = [r for r in rows if not r.get('error') and r['traj']]
    budgets = sorted({r['budget'] for r in ok})
    if not budgets:
        return None
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    cmap = plt.get_cmap('viridis')

    for i, b in enumerate(budgets):
        trajs = [r['traj'] for r in ok if r['budget'] == b]
        if not trajs:
            continue
        n = min(len(t) for t in trajs)
        arr = np.array([t[:n] for t in trajs], dtype=float)
        med = np.median(arr, axis=0)
        lo = np.percentile(arr, 25, axis=0)
        hi = np.percentile(arr, 75, axis=0)
        col = cmap(i / max(1, len(budgets) - 1))
        ax.plot(range(n), med, color=col, label='N={}'.format(b), lw=1.6)
        ax.fill_between(range(n), lo, hi, color=col, alpha=.13, lw=0)
    ax.set_xlabel('iteration')
    ax.set_ylabel('objective (lower = better)')
    ax.set_title('{}: convergence by measurement budget'.format(dpsize),
                 fontsize=9)
    ax.legend(fontsize=7, title='budget', title_fontsize=7)
    ax.grid(alpha=.25, lw=.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    seeds = sorted({r['seed'] for r in ok})
    for s in seeds:
        xs = [r['budget'] for r in sorted(
            (r for r in ok if r['seed'] == s), key=lambda r: r['budget'])]
        ys = [r['final_actual'] for r in sorted(
            (r for r in ok if r['seed'] == s), key=lambda r: r['budget'])]
        if len(xs) > 1:
            ax2.plot(xs, ys, '-', color='#8b9296', lw=.7, alpha=.55)
    meds = []
    for b in budgets:
        v = [r['final_actual'] for r in ok
             if r['budget'] == b and r['final_actual'] is not None]
        meds.append(float(np.median(v)) if v else float('nan'))
    ax2.plot(budgets, meds, 'o-', color='#31647f', lw=2, label='median')
    ax2.set_xscale('log')
    ax2.set_xticks(budgets)
    ax2.set_xticklabels([str(b) for b in budgets])
    ax2.set_xlabel('measurement budget (SCULPTOR_PROBE_N)')
    ax2.set_ylabel('final objective')
    ax2.set_title('what the budget buys ({} seeds, paired)'.format(len(seeds)),
                  fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=.25, lw=.6)
    for sp in ('top', 'right'):
        ax2.spines[sp].set_visible(False)
    fig.text(0.5, -0.02,
             'grey = individual deployments (same deployment + init across '
             'budgets); band = IQR across seeds',
             ha='center', fontsize=6.5, color='#6d7478')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out_png


def check(rows, budgets):
    """Hard invariants. Returns (ok, lines)."""
    out, ok = [], True
    for b in budgets:
        rs = [r for r in rows if r['budget'] == b and not r.get('error')]
        if not rs:
            continue
        mx = max(r['probes_spent'] for r in rs)
        mean = sum(r['probes_spent'] for r in rs) / len(rs)
        over = [r['seed'] for r in rs if r['probes_spent'] > b]
        if over:
            ok = False
            out.append('  budget={:<3} FAIL: exceeded on seeds {}'.format(
                b, over))
        elif mean < 0.5 * b:
            ok = False
            out.append('  budget={:<3} FAIL: only {:.1f}/{} spent -- a flat '
                       'curve here would mean the gate never fired, not '
                       'that measurements do not help'.format(b, mean, b))
        else:
            out.append('  budget={:<3} ok: mean {:.1f}/{} spent, max {}'
                       .format(b, mean, b, mx))
    return ok, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed-start', type=int, default=101)
    ap.add_argument('--budgets', default='2,5,10,20,40')
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--dpsize', default='small')
    ap.add_argument('--port', type=int, default=32100)
    ap.add_argument('--out', default=None)
    ap.add_argument('--plot-only', action='store_true')
    a = ap.parse_args()

    budgets = [int(x) for x in a.budgets.split(',')]
    out_dir = a.out or os.path.join(_REPO, 'cache', 'convergence_vs_budget')
    os.makedirs(out_dir, exist_ok=True)
    out_fn = os.path.join(out_dir, 'cvb_{}_{}it.json'.format(
        a.dpsize, a.iters))
    rows = []
    if os.path.exists(out_fn):
        try:
            rows = json.load(open(out_fn))
        except ValueError:
            rows = []
    done = {(r['seed'], r['budget']) for r in rows}

    if not a.plot_only:
        port = a.port
        for seed in range(a.seed_start, a.seed_start + a.seeds):
            for b in budgets:
                if (seed, b) in done:
                    continue
                port += 1
                print('\n{} seed={} budget={} {}'.format(
                    '=' * 18, seed, b, '=' * 18), flush=True)
                r = run_one(a.dpsize, seed, b, a.iters, port)
                rows.append(r)
                json.dump(rows, open(out_fn, 'w'), indent=1)
                print('  -> iters={} measures={}/{} final={} wall={:.0f}s {}'
                      .format(r['n_iters_run'], r['measures'], b,
                              r['final_actual'], r['wall_s'],
                              'ERR ' + r['error'] if r['error'] else ''),
                      flush=True)

    png = plot(rows, a.dpsize, os.path.join(
        FIG_DIR, 'convergence_vs_budget_{}.png'.format(a.dpsize)))
    print('\n' + '=' * 68)
    print('INVARIANTS')
    print('=' * 68)
    ok, lines = check(rows, budgets)
    for l in lines:
        print(l)
    print('\nfigure -> {}'.format(png))
    print('rows   -> {}'.format(out_fn))
    print('\nVERDICT: {}'.format('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
