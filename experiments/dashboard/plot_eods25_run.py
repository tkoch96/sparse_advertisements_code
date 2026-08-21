"""EODS-25 run-inspection dash (Tom 2026-08-19: 'ram, cpus, ram per
worker, ram on head, every type of logged time spent in statistical
formats, iteration, convergence — a dash to inspect the run').

Inputs: cache/eods/v1_dash/{sys_samples.csv, timing_lines.txt,
mem_iter.txt} (head-side dash_harvest, pulled by refresh.py).
Output: figures/dashboards/eods25/eods25_run.png — one multi-panel board.
"""
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from helpers.figpaths import fig_path  # -> figures/dashboards/<dashboard>/

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
# env-parameterized so the same module renders any EODS campaign
# (Tom 2026-08-20: EODS-32 dash 'just like EODS-25')
DASH = os.path.join(REPO, os.environ.get(
    'EODS_DASH_DIR', 'cache/eods/v1_dash'))
PREFIX = os.environ.get('EODS_FIG_PREFIX', 'eods25')
TITLE = os.environ.get(
    'EODS_RUN_TITLE', 'EODS-25 run inspection — 96w, incremental LP, MC=1')
OUT = fig_path(PREFIX + '_run.png')

SOLVE_RE = re.compile(r'([0-9.]+)ms per iter')
GRAD_RE = re.compile(r'(latency|resilience) benefit grad took ([0-9.]+)s')
# both formats: new '[wt] ... name=12%' and old block ' name  12.34%  (ms)'
TIMERCAT_RE = re.compile(r'(\w+)=([0-9.]+)%')
TIMERCAT_OLD_RE = re.compile(r'(\w+)\s+([0-9.]+)%\s+\(')
ITER_RE = re.compile(r'rss_mb=(\d+) .*sys_avail_mb=(\d+) .*'
                     r't=([0-9.]+) iter=(\d+)')
OBJ_RE = re.compile(r'objective[:= ]+(-?[0-9]+\.[0-9]+)', re.I)
IT_RE = re.compile(r'\[it\] t=\S+ iter=(\d+) obj=(\S+) pseudo=(\S+) '
                   r'rd=(\S+) rde=\S+ rad=(\S+) n_on=(\d+)')
ADAGRAD_RE = re.compile(r'\[adagrad\] call=(\d+) \|g\|=(\S+) G=(\S+) '
                        r'alpha_t=(\S+)')


def _d0(vals):
    # Tom 2026-08-20 v2: per-series ROBUST NORMALIZATION. Plain
    # delta-from-start still let one series' huge early transient set
    # the axis scale (believed drops thousands in the first iters, GT
    # moves tenths). Center each series on its median and scale by its
    # own p10-p90 span; the axis is clipped to [-3, 3] so outliers
    # can't hide the ongoing movement of every line.
    v = np.asarray([float(x) for x in vals], dtype=float)
    if v.size == 0:
        return v
    med = float(np.median(v))
    span = float(np.percentile(v, 90) - np.percentile(v, 10))
    if span < 1e-9:
        span = abs(med) if abs(med) > 1e-9 else 1.0
    return (v - med) / span


def main():
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    (ax_ram, ax_solve, ax_cat), (ax_grad, ax_iter, ax_obj) = axs
    # default titles so pre-training panels explain themselves instead
    # of rendering blank (Tom 2026-08-19)
    ax_solve.set_title('per-solve times — awaiting training data')
    ax_cat.set_title('worker time share — awaiting [wt] batches')
    ax_grad.set_title('grad durations — awaiting training iters')
    ax_iter.set_title('iteration pace — awaiting first iter')

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

    # -- convergence: [it] per-iter metrics (falls back to grep)
    # (objective_history backfill overlay REMOVED, Tom 2026-08-20: a
    # garbage near-zero point in one archived series produced a -19k
    # outlier that wrecked the axis, and the live run now carries its
    # own full history. Data still collected in objective_history.json
    # if ever wanted back.)
    # EODS-32 (other agent's cluster, read-only pull) overlay —
    # eods25 view only; the eods32 tab carries its own [it] series
    e32 = os.path.join(REPO, 'cache', 'eods', 'eods32_live', 'it.txt')
    if PREFIX == 'eods25' and os.path.exists(e32):
        its32 = IT_RE.findall(open(e32, errors='replace').read())
        if its32:
            n32 = [int(a[0]) for a in its32]
            ax_obj.plot(n32, _d0([a[1] for a in its32]), 'x-',
                        color='#c98f1e', ms=3, label='32: GT obj')
            ax_obj.plot(n32, _d0([a[2] for a in its32]), '+-',
                        color='#8f5f00', ms=3, alpha=.5,
                        label='32: believed')
    its = IT_RE.findall(txt)
    if its:
        it_n = [int(a[0]) for a in its]
        gt = [float(a[1]) for a in its]
        pse = [float(a[2]) for a in its]
        rd = [float(a[3]) for a in its]
        non = [int(a[5]) for a in its]
        ax_obj.plot(it_n, _d0(pse), 'k.-', ms=3,
                    label='believed obj (normalized)')
        ax_obj.plot(it_n, _d0(gt), 'g.-', ms=3, alpha=.8,
                    label='GT obj (normalized)')
        ax_obj.set_ylabel('per-series robust-normalized')
        ax_obj.set_ylim(-3, 3)
        a2 = ax_obj.twinx()
        a2.semilogy(it_n, np.maximum(rd, 1e-12), 'r--', alpha=.5,
                    label='rolling delta')
        ads = ADAGRAD_RE.findall(txt)
        if ads:
            a2.semilogy([int(x[0]) for x in ads],
                        [max(float(x[3]), 1e-12) for x in ads],
                        'b-', alpha=.6, lw=1, label='alpha_t (rmsprop)')
        a2.set_ylabel('rolling delta (red) / alpha_t (blue), log')
        a2.legend(fontsize=6, loc='lower left')
        ax_obj.set_xlabel('iter (n_on last={})'.format(non[-1]))
        ax_obj.set_title('convergence (per-series normalized) + stop + step size')
        ax_obj.legend(fontsize=7, loc='upper right')
        ax_obj.grid(alpha=.3)
    else:
        objs = [float(v) for v in OBJ_RE.findall(txt)][-500:]
        if objs:
            ax_obj.plot(objs, '.-', ms=2)
            ax_obj.set_title('objective mentions (no [it] lines yet)')
            ax_obj.grid(alpha=.3)
        else:
            ax_obj.set_title('convergence: historical runs (dashed); live [it] pending')

    fig.suptitle(TITLE)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print('[plot_eods25_run] wrote', OUT)


if __name__ == '__main__':
    main()
