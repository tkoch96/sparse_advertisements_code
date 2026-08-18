"""Profiler scaling figures (Tom 2026-08-17): from the smoke picks' logs
(mirrored locally by profiles_pull.sh) render
  1. s/iter box-whisker by deployment size (first iteration excluded —
     cache-building noise)
  2. stacked mean phase time by size (driver timers: grads / measure /
     info / stop)
  3. worker LP internals (last cumulative timing summary per pick)
NOTE: lb-grad vs resilience-grad split needs a dedicated timer inside
the grads phase — not yet instrumented; grads is shown whole.
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS = os.path.expanduser('~/sculptor_dashboard/prof_logs')
MEM_RE = re.compile(r'\[mem\] tag=iter_start .* t=([0-9.]+) iter=(\d+)')
TIMER_RE = re.compile(r'Timer: ([a-z_]+) -- ([0-9.]+) s')
WORKER_RE = re.compile(r'^\s*.*?([a-z_]+)\s+([0-9.]+)%\s+\(([0-9.]+) ms\)')


def label_size(label):
    m = re.search(r'a(\d+)', label)
    return int(m.group(1)) if m else None


def parse_log(fn):
    iters, timers = [], {}
    worker_last = {}
    block = None
    for ln in open(fn, errors='replace'):
        m = MEM_RE.search(ln)
        if m:
            iters.append((float(m.group(1)), int(m.group(2))))
        m = TIMER_RE.search(ln)
        if m:
            timers.setdefault(m.group(1), []).append(float(m.group(2)))
        if 'timing summary' in ln:
            block = {}
            continue
        if block is not None:
            m = WORKER_RE.match(ln.replace('\x1b', ' '))
            if m and '%' in ln:
                block[m.group(1)] = float(m.group(3))
            elif ln.strip().startswith('=='):
                worker_last = block or worker_last
                block = None
    if block:
        worker_last = block
    iters.sort()
    # per-iter deltas EXCLUDING the first iteration's delta
    deltas = [iters[i + 1][0] - iters[i][0] for i in range(1, len(iters) - 1)
              if 0 < iters[i + 1][0] - iters[i][0] < 3600]
    return deltas, timers, worker_last


def main():
    picks = {}
    for fn in sorted(glob.glob(os.path.join(LOGS, 'profile_*.log'))):
        label = os.path.basename(fn)[len('profile_'):].rsplit('_s', 1)[0]
        sz = label_size(label)
        if sz:
            picks[(sz, label)] = parse_log(fn)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    if picks:
        sizes = sorted({sz for sz, _ in picks})
        # 1. box-whisker s/iter by size
        data = [sum((picks[k][0] for k in picks if k[0] == s), [])
                for s in sizes]
        keep = [(s, d) for s, d in zip(sizes, data) if d]
        if keep:
            axes[0].boxplot([d for _, d in keep],
                            labels=[str(s) for s, _ in keep],
                            showfliers=True)
        axes[0].set_xlabel('deployment size (pops)')
        axes[0].set_ylabel('s per iteration (iter>1)')
        axes[0].set_title('iteration time vs size', fontsize=10)
        axes[0].grid(alpha=.25)
        # 2. stacked mean phase time by size
        phases = ['grads', 'measure', 'info', 'stop']
        colors = ['#2a78d6', '#2f9e6e', '#eda100', '#999999']
        bottoms = np.zeros(len(sizes))
        for ph, c in zip(phases, colors):
            vals = []
            for s in sizes:
                v = sum((picks[k][1].get(ph, []) for k in picks
                         if k[0] == s), [])
                vals.append(float(np.mean(v)) if v else 0.0)
            axes[1].bar([str(s) for s in sizes], vals, bottom=bottoms,
                        color=c, label=ph)
            bottoms += np.asarray(vals)
        axes[1].set_xlabel('deployment size (pops)')
        axes[1].set_ylabel('mean s per phase')
        axes[1].set_title('driver phase breakdown', fontsize=10)
        axes[1].legend(fontsize=8, frameon=False)
        # 3. worker LP internals for the LARGEST completed pick
        big = max((k for k in picks if picks[k][2]), default=None)
        if big:
            wl = picks[big][2]
            items = sorted(wl.items(), key=lambda kv: -kv[1])[:7]
            axes[2].barh([k for k, _ in items][::-1],
                         [v for _, v in items][::-1], color='#4a3aa7')
            axes[2].set_xlabel('cumulative ms (worker 0)')
            axes[2].set_title('worker LP internals @ size {}'.format(
                big[0]), fontsize=10)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, 'no pick logs mirrored yet', ha='center')
            ax.axis('off')
    fig.suptitle('Scoping smokes — scaling profile (first iteration '
                 'excluded)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(_REPO_ROOT, 'figures', 'profiler_scaling.png')
    fig.savefig(out, dpi=140)
    print('wrote', out)


if __name__ == '__main__':
    main()
