"""Step-policy program figures for the dashboard (2026-08-16 session):

  figures/step_alpha_h_trainobj.png   5-seed dash training objective bar
                                      (auto / fixed 0.5 / adagrad a0=1)
  figures/step_alpha_h2_trainobj.png  7-arm 2-seed bar (+ alpha0 grid, DoG)
  figures/adastep_ladder.png          ADASTEP ladder smoke vs the original
                                      v3 smoke (N=5 seed 1, steady diff)

Inputs: cache/model_error/{steady,failure}/step_alpha_h*_*.json (scored on
the head, rsynced by the dash refresh spec) + the two smoke JSON trees.
Run via the dashboard registry ('always' step) -- not by hand-rolled
scripts (dashboard steps contract).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from experiments.ablation.compare_paradigms import dash_train_obj  # noqa: E402

FIGS = os.path.join(REPO, 'figures')
os.makedirs(FIGS, exist_ok=True)
COLORS = ['#4a3aa7', '#2a78d6', '#eda100', '#eb6834', '#2f9e6e',
          '#b03060', '#607d8b']


def trainobj_bar(steady_tag, failure_tag, arms, seeds, out, title):
    if not (os.path.exists(os.path.join(
                REPO, 'cache/model_error/steady', steady_tag + '.json'))
            and os.path.exists(os.path.join(
                REPO, 'cache/model_error/failure', failure_tag + '.json'))):
        return
    totals, opp = dash_train_obj(steady_tag, failure_tag, arms, seeds, 0.1)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, (name, label) in enumerate(arms):
        b = totals.get(name, {})
        common = [s for s in seeds if s in b and s in opp]
        if not common:
            continue
        d = [b[s] - opp[s] for s in common]
        ax.bar(i, np.mean(d), color=COLORS[i % len(COLORS)], width=.62)
        ax.errorbar(i, np.mean(d),
                    yerr=np.std(d) / max(1, len(d) ** .5),
                    color='k', capsize=4, lw=1)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([l for _, l in arms], fontsize=8, rotation=14)
    ax.axhline(0, color='k', lw=1)
    ax.set_ylabel('mean training objective - same-seed opp')
    ax.set_title(title + '\n(lower = better; 0 = one-per-peering)',
                 fontsize=10)
    ax.grid(alpha=.25, axis='y')
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print('wrote', out)


def ladder_compare(out):
    rungs = ['L1_nomc_fixed', 'L2_nomc_sched', 'L3_nomem_sched',
             'L4_nodir_sched', 'L5_full_sched', 'L6_full_schedwhat',
             'L6p_full_scheddecision', 'L7_full_smartcons']
    stores = [('stock auto-scale', 'cache/ablation/policy_ladder_v3',
               '#9a9a9a'),
              ('adagrad a0=1', 'cache/ablation/policy_ladder_v3_ADASTEP',
               '#4a3aa7')]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    width = 0.38
    labels = [r.split('_')[0] for r in rungs]
    any_data = False
    for si, (label, root, color) in enumerate(stores):
        vals = []
        for r in rungs:
            fns = glob.glob(os.path.join(
                REPO, root, r, 'N5', 'seed_1_*.json'))
            v = None
            if fns:
                d = json.load(open(fns[0]))
                v = d.get('diff_vs_opp')
            vals.append(v)
        xs = [i + (si - 0.5) * width for i in range(len(rungs))]
        for x, v in zip(xs, vals):
            if v is not None:
                ax.bar(x, v, width=width, color=color)
                any_data = True
        ax.bar(-10, 0, color=color, label=label)  # legend proxy
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlim(-0.7, len(rungs) - 0.3)
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('steady avg_lat - opp (ms)')
    ax.set_title('WHAT/WHEN ladder smoke, N=5 seed 1, 100 iters: stock vs '
                 'adagrad step (L1-L4 ignore the seam; their deltas = '
                 'single-trial noise floor)', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.25, axis='y')
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print('wrote', out)


def main():
    # h3 tags (once landed) supersede h: same cells rescored PLUS the
    # L4/no_direction arm run through the identical harness (Tom
    # 2026-08-16: "loop in L4 into the smoke test")
    if os.path.exists(os.path.join(
            REPO, 'cache/model_error/steady/step_alpha_h3_steady.json')):
        trainobj_bar(
            'step_alpha_h3_steady', 'step_alpha_h3_failure',
            [('auto', 'auto-scale (stock)'), ('a050', 'fixed alpha=0.5'),
             ('adagrad', 'adagrad a0=1'),
             ('L4nodir', 'L4 one-flip (no_dir)')],
            [1, 2, 3, 4, 5],
            os.path.join(FIGS, 'step_alpha_h_trainobj.png'),
            'L5 step policies vs the L4 one-flip rung, seeds 1-5 '
            '(georand, 50 iters)')
    else:
        trainobj_bar(
            'step_alpha_h_steady', 'step_alpha_h_failure',
            [('auto', 'auto-scale (stock)'), ('a050', 'fixed alpha=0.5'),
             ('adagrad', 'adagrad a0=1')],
            [1, 2, 3, 4, 5],
            os.path.join(FIGS, 'step_alpha_h_trainobj.png'),
            'L5 step-policy A/B, seeds 1-5 (georand, 50 iters)')
    trainobj_bar(
        'step_alpha_h2_steady', 'step_alpha_h2_failure',
        [('auto', 'auto (stock)'), ('a050', 'fixed 0.5'),
         ('adagrad03', 'ada a0=.3'), ('adagrad', 'ada a0=1'),
         ('adagrad3', 'ada a0=3'), ('adaauto', 'ada a0=auto'),
         ('dog', 'DoG')],
        [1, 2],
        os.path.join(FIGS, 'step_alpha_h2_trainobj.png'),
        'alpha0 sensitivity + parameter-free rows, seeds 1-2')
    ladder_compare(os.path.join(FIGS, 'adastep_ladder.png'))


if __name__ == '__main__':
    main()
