"""v5 scout sweep dash figure (Tom 2026-08-18: L3 / L5-adagrad /
L5-rmsprop x every objective x seeds 201-205 at N=10 — 'a sense of
what i want to measure before the absolute full panel').

Pulls store + harvested convergence figs from the sweep VM
(SCULPTOR_SMOKE_HOST, default 32.197.41.137, ~/smoke_repo), renders
figures/v5scout_bars.png: one panel per objective family, three bars
per seed (dObj vs same-seed opp, lower better).
"""
import glob
import json
import os
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
HOST = os.environ.get('SCULPTOR_SMOKE_HOST', '32.197.41.137')
KEY = os.path.expanduser('~/.ssh/ray-autoscaler_us-east-1.pem')
STORE = os.path.join(REPO, 'cache', 'ablation', 'grid_v5scout')
FIGS_DIR = os.path.join(REPO, 'cache', 'ablation',
                        'grid_v5scout_artifacts', 'figs')
OUT = os.path.join(REPO, 'figures', 'v5scout_bars.png')
FAMS = [('lat', 'latency + gamma*resilience'),
        ('fracb', 'frac_beyond_optimal (hinge)'),
        ('mlu', 'max_util (standalone)'),
        ('prio', 'joint latency+bulk')]
ARMS = [('L3_nomem_sched', 'L3 (guaranteed-flip)', '#1baf7a'),
        ('L5_full_adagrad', 'L5 adagrad', '#e87ba4'),
        ('L5_full_rmsprop', 'L5 rmsprop b=0.9', '#4a3aa7')]
SEEDS = (201, 202, 203, 204, 205)


def pull():
    ssh = 'ssh -i {} -o BatchMode=yes -o ConnectTimeout=15'.format(KEY)
    for src, dst in (
            ('~/smoke_repo/cache/ablation/grid_v5scout/', STORE + '/'),
            ('~/smoke_repo/cache/ablation/grid_v5scout_artifacts/figs/',
             FIGS_DIR + '/')):
        os.makedirs(dst, exist_ok=True)
        try:
            subprocess.run(
                'rsync -az --timeout=45 -e "{}" ubuntu@{}:{} {}'.format(
                    ssh, HOST, src, dst),
                shell=True, timeout=90, check=False)
        except subprocess.TimeoutExpired:
            print('[v5scout] pull timeout; using cached store')


def main():
    pull()
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    n_done = 0
    width = .25
    for ax, (fam, title) in zip(axes, FAMS):
        for ai, (arm, label, color) in enumerate(ARMS):
            vals = []
            for si, seed in enumerate(SEEDS):
                v = np.nan
                for fn in glob.glob(os.path.join(
                        STORE, fam, arm, 'N*',
                        'seed_{}_*.json'.format(seed))):
                    try:
                        d = json.load(open(fn))
                    except (OSError, ValueError):
                        continue
                    if d.get('repo_objective') is not None \
                            and d.get('opp_objective') is not None:
                        v = d['repo_objective'] - d['opp_objective']
                        n_done += 1
                vals.append(v)
            x = np.arange(len(SEEDS)) + (ai - 1) * width
            ax.bar(x, vals, width, color=color,
                   label=label if fam == 'lat' else None)
        ax.set_xticks(np.arange(len(SEEDS)))
        ax.set_xticklabels(['s{}'.format(s) for s in SEEDS], fontsize=8)
        ax.axhline(0, color='k', lw=1)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.25, axis='y')
    axes[0].set_ylabel('objective - same-seed opp (lower better)')
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('v5 scout — L3 vs L5-adagrad vs L5-rmsprop, N=10, '
                 'maxhard seeds 201-205, HiGHS, new stop-v2 '
                 '({} cells landed)'.format(n_done), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT, dpi=150)
    print('wrote v5scout_bars.png ({} cells)'.format(n_done))


if __name__ == '__main__':
    main()
