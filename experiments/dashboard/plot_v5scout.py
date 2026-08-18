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
            ('~/v5scout_ws/S*/logs/', LOGS_DIR + '/'),
            ('~/v5scout2_ws/S*/logs/', LOGS_DIR + '/')):
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
