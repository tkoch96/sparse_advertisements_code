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
LOGS_DIR = os.path.join(REPO, 'cache', 'ablation', 'grid_v5scout_logs')
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
             FIGS_DIR + '/'),
            ('~/v5scout_ws/S*/logs/', LOGS_DIR + '/')):
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
    plt.close(fig)

    # per-family figures (one per dash section, unified-grid style)
    for fam, title in FAMS:
        f2, a2 = plt.subplots(figsize=(7.5, 4.4))
        drawn = False
        for ai, (arm, label, color) in enumerate(ARMS):
            vals = []
            for seed in SEEDS:
                v = np.nan
                for fn in glob.glob(os.path.join(
                        STORE, fam, arm, 'N*',
                        'seed_{}_*.json'.format(seed))):
                    try:
                        d = json.load(open(fn))
                    except (OSError, ValueError):
                        continue
                    if d.get('repo_objective') is not None:
                        v = d['repo_objective'] - d['opp_objective']
                        drawn = True
                vals.append(v)
            x = np.arange(len(SEEDS)) + (ai - 1) * .25
            a2.bar(x, vals, .25, color=color, label=label)
        a2.set_xticks(np.arange(len(SEEDS)))
        a2.set_xticklabels(['s{}'.format(s) for s in SEEDS])
        a2.axhline(0, color='k', lw=1)
        a2.set_ylabel('objective - same-seed opp (lower better)')
        a2.set_title('{} — v5 scout, N=10'.format(title), fontsize=10)
        a2.legend(fontsize=8, frameon=False)
        a2.grid(alpha=.25, axis='y')
        f2.tight_layout()
        if drawn:
            f2.savefig(os.path.join(
                REPO, 'figures', 'v5scout_{}.png'.format(fam)), dpi=150)
        plt.close(f2)

    # status board: per-cell RSS trace, s/iter, VM free RAM (from the
    # [mem] lines run_fork_ladder logs every iteration)
    import re as _re
    MEM = _re.compile(
        r'\[mem\] tag=\S+ rss_mb=(\d+) vms_mb=\d+ peak_mb=\d+ '
        r'sys_avail_mb=(\d+) pid=\d+ t=([0-9.]+) iter=(\d+)')
    cells = {}
    for fn in glob.glob(os.path.join(LOGS_DIR, 'v5s_*.log')):
        base = os.path.basename(fn)
        rows = [tuple(map(float, m.groups()))
                for m in map(MEM.search, open(fn, errors='replace')) if m]
        if rows:
            cells[base] = rows
    f3, (b1, b2, b3) = plt.subplots(1, 3, figsize=(16, 4.2))
    arm_color = {'L3': '#1baf7a', 'L5ada': '#e87ba4', 'L5rms': '#4a3aa7'}
    spi_by_arm = {}
    t_all = []
    for base, rows in cells.items():
        arm = base.split('_')[2]
        c = arm_color.get(arm, '#888')
        iters = [r[3] for r in rows]
        b1.plot(iters, [r[0] / 1024. for r in rows], color=c, alpha=.4,
                lw=1)
        ts = [r[2] for r in rows]
        t_all += [(t, r[1] / 1024.) for t, r in zip(ts, rows)]
        if len(ts) > 5:
            spi = np.median(np.diff(ts))
            spi_by_arm.setdefault(arm, []).append(spi)
    b1.set_xlabel('iteration'); b1.set_ylabel('driver RSS (GB)')
    b1.set_title('per-cell RAM over iterations', fontsize=10)
    b1.grid(alpha=.25)
    for ai, (arm, spis) in enumerate(sorted(spi_by_arm.items())):
        b2.bar([ai], [np.median(spis)], color=arm_color.get(arm, '#888'))
        b2.scatter([ai] * len(spis), spis, color='k', s=8, zorder=3)
    b2.set_xticks(range(len(spi_by_arm)))
    b2.set_xticklabels(sorted(spi_by_arm), fontsize=9)
    b2.set_ylabel('s per iteration (median; dots=cells)')
    b2.set_title('iteration time by arm', fontsize=10)
    b2.grid(alpha=.25, axis='y')
    if t_all:
        t_all.sort()
        t0 = t_all[0][0]
        b3.plot([(t - t0) / 3600. for t, _ in t_all],
                [v for _, v in t_all], lw=.8, color='#2a78d6')
        b3.set_xlabel('hours since sweep start')
        b3.set_ylabel('VM avail RAM (GB)')
        b3.set_title('sweep VM free memory', fontsize=10)
        b3.grid(alpha=.25)
    f3.suptitle('v5 scout status — {} cells with telemetry'.format(
        len(cells)), fontsize=11)
    f3.tight_layout(rect=[0, 0, 1, 0.92])
    if cells:
        f3.savefig(os.path.join(REPO, 'figures', 'v5scout_status.png'),
                   dpi=150)
        print('wrote v5scout_status.png ({} cells)'.format(len(cells)))
    plt.close(f3)


if __name__ == '__main__':
    main()
