"""Adagrad transient smoke dash figures (Tom 2026-08-18).

Pulls logs + results from the profiler VM (env SCULPTOR_SMOKE_HOST,
default 32.197.41.137, ~/smoke_repo + ~/adagrad_smoke_ws), then renders:

  figures/dashboards/adagrad_smoke/adagrad_smoke_grads.png   per-arm |g| and alpha_t vs gradient
                                    call (log y) — the transient and
                                    whether warmup-skip un-freezes alpha
  figures/dashboards/adagrad_smoke/adagrad_smoke_obj.png     final objective - same-seed opp per
                                    arm x seed + iters-at-exit

Arms: smk_L3_stock / smk_L5_stock / smk_L5_wskip (WARMUP_SKIP=5).
"""
import glob
import json
import os
import re
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from helpers.figpaths import fig_path  # -> figures/dashboards/<dashboard>/

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOST = os.environ.get('SCULPTOR_SMOKE_HOST', '32.197.41.137')
KEY = os.path.expanduser('~/.ssh/ray-autoscaler_us-east-1.pem')
LOGS = os.path.join(REPO, 'cache', 'ablation', 'adagrad_smoke_logs')
STORE = os.path.join(REPO, 'cache', 'ablation', 'adagrad_smoke')
FIGS = os.path.join(REPO, 'figures')
ARMS = [('smk_L3_stock', 'L3 control (guaranteed-flip; no adagrad)', '#1baf7a'),
        ('smk_L5_stock', 'L5 stock adagrad', '#e87ba4'),
        ('smk_L5_wskip', 'L5 warmup-skip K=5', '#2a78d6'),
        ('smk_L5_wskip20', 'L5 warmup-skip K=20', '#eda100'),
        ('smk_L5_rmsprop', 'L5 rmsprop b=0.99', '#4a3aa7')]
_LINE = re.compile(
    r'\[adagrad\] call=(\d+) \|g\|=([0-9.e+-]+) G=([0-9.e+-]+) '
    r'alpha_t=([0-9.e+-]+)')


def pull():
    ssh = 'ssh -i {} -o BatchMode=yes -o ConnectTimeout=15'.format(KEY)
    for src, dst in (
            ('~/adagrad_smoke_ws/S*/logs/', LOGS + '/'),
            ('~/adagrad_smoke2_ws/S*/logs/', LOGS + '/'),
            ('~/smoke_repo/cache/ablation/adagrad_smoke/', STORE + '/')):
        os.makedirs(dst, exist_ok=True)
        try:
            subprocess.run(
                'rsync -az --timeout=45 -e "{}" ubuntu@{}:{} {}'.format(
                    ssh, HOST, src, dst),
                shell=True, timeout=90, check=False)
        except subprocess.TimeoutExpired:
            print('[adagrad-smoke] pull timeout (VM gone?); using cache')


def series_by_cell():
    """{(arm, seed): [(call, |g|, alpha)]} from per-cell queue logs."""
    out = {}
    for fn in glob.glob(os.path.join(LOGS, '*.log')):
        base = os.path.basename(fn)
        m = re.match(r'(smk_[A-Za-z0-9_]+)_N(\d+)_s(\d+)_', base)
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(3))
        rows = []
        for line in open(fn, errors='replace'):
            lm = _LINE.search(line)
            if lm:
                rows.append((int(lm.group(1)), float(lm.group(2)),
                             float(lm.group(4))))
        if rows:
            out[arm, seed] = rows
    return out


def main():
    pull()
    os.makedirs(FIGS, exist_ok=True)
    ser = series_by_cell()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    drawn = False
    for arm, label, color in ARMS:
        for seed in (201, 202, 203):
            rows = ser.get((arm, seed))
            if not rows:
                continue
            calls = [r[0] for r in rows]
            axes[0].plot(calls, [r[1] for r in rows], color=color,
                         alpha=.55, lw=1.1,
                         label=label if seed == 201 else None)
            axes[1].plot(calls, [r[2] for r in rows], color=color,
                         alpha=.55, lw=1.1,
                         label=label if seed == 201 else None)
            drawn = True
    for ax, ttl in zip(axes, ('|g| per gradient call',
                              'alpha_t per gradient call')):
        ax.set_yscale('log')
        ax.axvspan(0, 5.5, color='#f5d76e', alpha=.25)
        ax.text(5.6, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-6,
                ' warmup window', fontsize=7, color='#8a6d00', va='bottom')
        ax.set_xlabel('gradient call')
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('AdaGrad transient smoke — does the iter-1-5 spike '
                 'freeze the learning rate? (mlu, maxhard, seeds '
                 '201-203)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    if drawn:
        fig.savefig(fig_path('adagrad_smoke_grads.png'), dpi=150)
        print('wrote adagrad_smoke_grads.png')
    plt.close(fig)

    # final objective bars
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    width = .16
    seeds = (201, 202, 203)
    got = False
    for ai, (arm, label, color) in enumerate(ARMS):
        dvals, ivals = [], []
        for seed in seeds:
            fs = glob.glob(os.path.join(
                STORE, arm, 'N*', 'seed_{}_*.json'.format(seed)))
            d = None
            for f in fs:
                try:
                    d = json.load(open(f))
                except (OSError, ValueError):
                    pass
            if d and d.get('repo_objective') is not None:
                dvals.append(d['repo_objective'] - d['opp_objective'])
                ivals.append(d.get('n_iters') or 0)
                got = True
            else:
                dvals.append(np.nan)
                ivals.append(0)
        x = np.arange(len(seeds)) + (ai - 2) * width
        a1.bar(x, dvals, width, color=color, label=label)
        a2.bar(x, ivals, width, color=color, label=label)
    a1.set_xticks(np.arange(len(seeds)))
    a1.set_xticklabels(['s{}'.format(s) for s in seeds])
    a1.set_ylabel('objective - same-seed opp (ms-equiv; lower better)')
    a1.axhline(0, color='k', lw=1)
    a1.set_title('final quality', fontsize=10)
    a1.legend(fontsize=8, frameon=False)
    a1.grid(alpha=.25, axis='y')
    a2.set_xticks(np.arange(len(seeds)))
    a2.set_xticklabels(['s{}'.format(s) for s in seeds])
    a2.set_ylabel('iterations at exit')
    a2.set_title('exit iteration (500 = cap)', fontsize=10)
    a2.grid(alpha=.25, axis='y')
    fig.tight_layout()
    if got:
        fig.savefig(fig_path('adagrad_smoke_obj.png'), dpi=150)
        print('wrote adagrad_smoke_obj.png')
    plt.close(fig)
    if not (drawn or got):
        print('[adagrad-smoke] no data yet')


if __name__ == '__main__':
    main()
