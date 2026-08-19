"""EODS-25 run-inspection dash (Tom 2026-08-19: 'ram, cpus, ram per
worker, ram on head, every type of logged time spent in statistical
formats, iteration, convergence — a dash to inspect the run').

Inputs: cache/eods/v1_dash/{sys_samples.csv, timing_lines.txt,
mem_iter.txt} (head-side dash_harvest, pulled by refresh.py).
Output: figures/eods25_run.png — one multi-panel board.
"""
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DASH = os.path.join(REPO, 'cache', 'eods', 'v1_dash')
OUT = os.path.join(REPO, 'figures', 'eods25_run.png')

SOLVE_RE = re.compile(r'([0-9.]+)ms per iter')
GRAD_RE = re.compile(r'(latency|resilience) benefit grad took ([0-9.]+)s')
# both formats: new '[wt] ... name=12%' and old block ' name  12.34%  (ms)'
TIMERCAT_RE = re.compile(r'(\w+)=([0-9.]+)%')
TIMERCAT_OLD_RE = re.compile(r'(\w+)\s+([0-9.]+)%\s+\(')
ITER_RE = re.compile(r'rss_mb=(\d+) .*sys_avail_mb=(\d+) .*'
                     r't=([0-9.]+) iter=(\d+)')
OBJ_RE = re.compile(r'objective[:= ]+(-?[0-9]+\.[0-9]+)', re.I)


def main():
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    (ax_ram, ax_solve, ax_cat), (ax_grad, ax_iter, ax_obj) = axs

    # -- system panel: RAM + workers + load
    fn = os.path.join(DASH, 'sys_samples.csv')
    if os.path.exists(fn):
        rows = [l.split(',') for l in open(fn) if l.count(',') >= 7]
        if rows:
            t = np.array([float(r[0]) for r in rows]); t = (t - t[0]) / 3600
            ax_ram.plot(t, [float(r[2]) / 1024 for r in rows], 'k-',
                        label='sys avail GB')
            ax_ram.plot(t, [float(r[7]) / 1024 for r in rows], 'r-',
                        label='driver RSS GB')
            ax_ram.plot(t, [float(r[4]) / 1024 for r in rows], 'b-',
                        label='worker RSS p50 GB')
            ax_ram.plot(t, [float(r[6]) / 1024 for r in rows], 'b:',
                        label='worker RSS max GB')
            ax2 = ax_ram.twinx()
            ax2.plot(t, [float(r[1]) for r in rows], 'g--', alpha=.6,
                     label='loadavg')
            ax2.set_ylabel('load (green)')
            la = rows[-1]
            ax_ram.set_title(
                'system: {} workers, load {}, avail {}G'.format(
                    la[3], la[1], int(la[2]) // 1024))
            ax_ram.legend(fontsize=7); ax_ram.set_xlabel('hours')
            ax_ram.grid(alpha=.3)

    tl = os.path.join(DASH, 'timing_lines.txt')
    txt = open(tl, errors='replace').read() if os.path.exists(tl) else ''

    # -- per-solve time distribution
    solves = [float(m) / 1000 for m in SOLVE_RE.findall(txt)]
    if solves:
        s = np.array(solves)
        ax_solve.hist(s, bins=40, color='#4a3aa7')
        ax_solve.set_title('per-solve s (n={}, p50={:.1f}, p90={:.1f})'.format(
            len(s), np.percentile(s, 50), np.percentile(s, 90)))
        ax_solve.set_xlabel('seconds'); ax_solve.grid(alpha=.3)

    # -- worker timing categories (from summarize_timing blocks)
    cats = {}
    for name, pct in (TIMERCAT_RE.findall(txt)
                      + TIMERCAT_OLD_RE.findall(txt)):
        if float(pct) > 0:
            cats.setdefault(name, []).append(float(pct))
    if cats:
        names = sorted(cats, key=lambda k: -np.median(cats[k]))[:8]
        ax_cat.boxplot([cats[n] for n in names], vert=False,
                       tick_labels=[n[:26] for n in names])
        ax_cat.set_title('worker time share by category (%)')
        ax_cat.tick_params(labelsize=7); ax_cat.grid(alpha=.3)

    # -- grad phase durations
    lat = [float(v) for k, v in GRAD_RE.findall(txt) if k == 'latency']
    res = [float(v) for k, v in GRAD_RE.findall(txt) if k == 'resilience']
    if lat or res:
        ax_grad.plot(lat, 'o-', label='latency grad s', ms=3)
        ax_grad.plot(res, 's-', label='resilience grad s', ms=3)
        ax_grad.set_title('grad phase durations per training iter')
        ax_grad.legend(fontsize=8); ax_grad.grid(alpha=.3)
        ax_grad.set_xlabel('grad call #')

    # -- iteration progress + driver RSS per iter
    mi = os.path.join(DASH, 'mem_iter.txt')
    if os.path.exists(mi):
        pts = ITER_RE.findall(open(mi, errors='replace').read())
        if pts:
            tt = np.array([float(p[2]) for p in pts])
            order = np.argsort(tt); tt = (tt[order] - tt[order][0]) / 3600
            it = np.array([int(p[3]) for p in pts])[order]
            rss = np.array([int(p[0]) for p in pts])[order] / 1024
            ax_iter.plot(tt, it, 'k.-')
            ax_iter.set_ylabel('iter'); ax_iter.set_xlabel('hours')
            a2 = ax_iter.twinx(); a2.plot(tt, rss, 'r--', alpha=.6)
            a2.set_ylabel('driver RSS GB (red)')
            spi = np.diff(tt * 3600) / np.maximum(np.diff(it), 1)
            ttl = ('iterations (s/iter p50={:.0f})'.format(
                np.percentile(spi, 50)) if len(spi) else 'iterations')
            ax_iter.set_title(ttl); ax_iter.grid(alpha=.3)

    # -- objective trace (best-effort grep)
    objs = [float(v) for v in OBJ_RE.findall(txt)][-500:]
    if objs:
        ax_obj.plot(objs, '.-', ms=2)
        ax_obj.set_title('objective mentions over time (convergence)')
        ax_obj.grid(alpha=.3)
    else:
        ax_obj.set_title('objective trace: no lines yet')

    fig.suptitle('EODS-25 run inspection — 96w, incremental LP, MC=1')
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print('[plot_eods25_run] wrote', OUT)


if __name__ == '__main__':
    main()
