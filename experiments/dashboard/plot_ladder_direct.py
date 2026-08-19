"""License-independent ladder figure: mean in-run diff_vs_opp per arm/N
straight from the cell JSONs (per-seed opp-normalized by construction).
Used when the Gurobi-backed steady/failure stores are unavailable; the
title says which scorer produced it. Writes the classic tab's figure
slot + the a10 tab's."""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
NS = [1, 2, 5, 10, 20, 50]
ARMS = [('L1_nomc_fixed', 'L1 no_mc+fixed', '#2a78d6'),
        ('L2_nomc_sched', 'L2 no_mc+sched', '#eb6834'),
        ('L3_nomem_sched', 'L3 no_mem+sched', '#1baf7a'),
        ('L4_nodir_sched', 'L4 no_dir+sched', '#eda100'),
        ('L5_full_sched', 'L5 full+sched', '#e87ba4'),
        ('L6_full_slotted', 'L6 slotted WHEN', '#4a3aa7')]


def render(root, out, title, field='diff_vs_opp',
           ylabel='steady avg_lat - same-seed opp (ms)'):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    drawn = False
    for arm, label, color in ARMS:
        xs, ys, ns_ = [], [], []
        for n in NS:
            vals = []
            for fn in glob.glob(os.path.join(
                    REPO, root, arm, 'N{}'.format(n), 'seed_*_*.json')):
                try:
                    d = json.load(open(fn))
                except (OSError, ValueError):
                    continue
                if d.get('solve_error'):
                    continue
                if field == 'objective':
                    if d.get('repo_objective') is None \
                            or d.get('opp_objective') is None:
                        continue
                    vals.append(d['repo_objective'] - d['opp_objective'])
                else:
                    if d.get(field) is None:
                        continue
                    vals.append(d[field])
            if vals:
                xs.append(n); ys.append(float(np.mean(vals)))
                ns_.append(len(vals))
        if xs:
            ax.plot(xs, ys, 'o-', color=color, lw=1.6, ms=5,
                    label='{} (n={})'.format(label, min(ns_)))
            drawn = True
    if not drawn:
        plt.close(fig)
        return False
    ax.axhline(0, color='k', lw=1, linestyle='--')
    ax.set_xscale('log'); ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlabel('measurement budget N')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=.25); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, out), dpi=150)
    plt.close(fig)
    print('wrote', out)
    return True


def _store_has_data(tag_prefix):
    import glob as g
    for fn in g.glob(os.path.join(REPO, 'cache/model_error/steady',
                                  tag_prefix + '_steady*.json')):
        try:
            if json.load(open(fn)):
                return True
        except (OSError, ValueError):
            pass
    return False


def main():
    # Delegate to the fresh-eval renderer when its stores hold data;
    # otherwise render the license-independent direct figures. Prevents
    # plot_policy5 from painting empty axes over real data (the blank-dash
    # incident, 2026-08-17).
    import subprocess, sys
    if _store_has_data('policy'):
        env = dict(os.environ, POLICY_PLOT_STAT='mean',
                   POLICY_PLOT_OUT='policy_ladder_v3_5panel',
                   PYTHONPATH=REPO, MPLBACKEND='Agg')
        subprocess.run([sys.executable, '-m',
                        'experiments.model_error.plot_policy5'],
                       env=env, cwd=REPO)
    else:
        render('cache/ablation/policy_ladder_v3',
               'figures/policy_ladder_v3_5panel_objective.png',
               'Policy ladder v3 (10 deployments) — IN-RUN steady scores '
               '(fresh-eval composite pending Gurobi license renewal)')
    # Solver-fork (HiGHS) ladder tab (Tom 2026-08-17): same
    # delegate-or-direct logic against the highs stores.
    if _store_has_data('policy_highs'):
        env = dict(os.environ, POLICY_PLOT_STAT='mean',
                   POLICY_PLOT_TAG_PREFIX='policy_highs',
                   POLICY_PLOT_OUT='policy_ladder_highs_5panel',
                   PYTHONPATH=REPO, MPLBACKEND='Agg')
        subprocess.run([sys.executable, '-m',
                        'experiments.model_error.plot_policy5'],
                       env=env, cwd=REPO)
    else:
        render('cache/ablation/policy_ladder_highs',
               'figures/policy_ladder_highs_5panel_objective.png',
               'Policy ladder — NEW SOLVER (HiGHS via solver_fork) — '
               'IN-RUN steady scores (fresh-eval composite pending)')
    if _store_has_data('a10'):
        env = dict(os.environ, POLICY_PLOT_STAT='mean',
                   POLICY_PLOT_TAG_PREFIX='a10',
                   POLICY_PLOT_OUT='policy_ladder_a10_5panel',
                   PYTHONPATH=REPO, MPLBACKEND='Agg')
        subprocess.run([sys.executable, '-m',
                        'experiments.model_error.plot_policy5'],
                       env=env, cwd=REPO)
    else:
        render('cache/ablation/policy_ladder_a10',
               'figures/policy_ladder_a10_5panel_objective.png',
               'actual-10 ladder — IN-RUN steady scores (partial: '
               'license-paused)')
    # a10x10 grid (Tom 2026-08-18): direct render only — no fresh-eval
    # steady store; cells carry their own diff_vs_opp.
    # y-axis = the TRAINED objective (lat + 0.1*resilience), not just
    # its steady-latency component (Tom's catch 2026-08-19)
    render('cache/ablation/policy_ladder_a10x10',
           'figures/policy_ladder_a10x10_5panel_objective.png',
           'actual-10 x 10 deployments — L1-L6, N=10, TRAINED objective '
           '(lat + 0.1*resilience) vs opp (mean over landed seeds)',
           field='objective',
           ylabel='objective (lat + 0.1*resilience) - same-seed opp')


if __name__ == '__main__':
    main()
