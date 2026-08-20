"""EODS-25 dash figures (Tom 2026-08-19: 'let L6 rip at a large
problem, scored through the classical eval_latency_failure metrics').

Reads ONLY the distilled cache/eods/v1_dash/ dir (populated head-side
by experiments.eods.dash_harvest via remote_harvest + pulled by
refresh.py — the raw per-cell pickles never leave the head).

Renders:
  figures/eods25_status.png   cell board (done/running/pending, wall,
                              iter progress, driver RSS + sys-avail —
                              RAM is THE failure mode at dpsize 25)
  figures/eods25_results.png  per-strategy classical metrics across
                              completed sims (only once >=1 cell done)
"""
import glob
import json
import os
import pickle
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
# env-parameterized so the same module renders any EODS campaign
# (Tom 2026-08-20: EODS-32 dash 'just like EODS-25')
DASH = os.path.join(REPO, os.environ.get(
    'EODS_DASH_DIR', 'cache/eods/v1_dash'))
PREFIX = os.environ.get('EODS_FIG_PREFIX', 'eods25')
DPSIZE = int(os.environ.get('EODS_DPSIZE', '25'))
LABEL = os.environ.get('EODS_LABEL', 'actual-25')
FIG_DIR = os.path.join(REPO, 'figures')
SEEDS = [1]  # single-deployment scope (Tom 2026-08-19)

MEM_RE = re.compile(
    r'rss_mb=(\d+) vms_mb=\d+ peak_mb=(\d+) sys_avail_mb=(\d+) '
    r'pid=\d+ t=([0-9.]+) iter=(\d+)')
SEED_RE = re.compile(r"_s(\d+)[_.]")

SOLN_ORDER = ['sparse', 'painter', 'one_per_peering', 'anyopt',
              'one_per_pop', 'anycast']
SOLN_COLOR = {'sparse': '#4a3aa7', 'painter': '#e87ba4',
              'one_per_peering': '#1baf7a', 'anyopt': '#c98f1e',
              'one_per_pop': '#5a92c9', 'anycast': '#999999'}


def load_mem():
    """{seed: [(t, iter, rss_gb, sys_avail_gb), ...]} from mem_iter.txt."""
    by_seed = {}
    fn = os.path.join(DASH, 'mem_iter.txt')
    if not os.path.exists(fn):
        return by_seed
    for ln in open(fn, errors='replace'):
        try:
            logname, rest = ln.split('\t', 1)
        except ValueError:
            continue
        sm = SEED_RE.search(logname)
        m = MEM_RE.search(rest)
        if not (sm and m):
            continue
        by_seed.setdefault(int(sm.group(1)), []).append(
            (float(m.group(4)), int(m.group(5)),
             int(m.group(1)) / 1024.0, int(m.group(3)) / 1024.0))
    for s in by_seed:
        by_seed[s].sort()
    return by_seed


def cell_states():
    done, running = {}, set()
    for fn in glob.glob(os.path.join(DASH, 'seed_*_eods.json')):
        try:
            rec = json.load(open(fn))
            done[int(rec['seed'])] = rec
        except (ValueError, KeyError):
            pass
    for fn in glob.glob(os.path.join(DASH, '*.inprog')):
        m = re.search(r'seed_(\d+)_', os.path.basename(fn))
        if m:
            running.add(int(m.group(1)))
    return done, running


def status_fig():
    done, running = cell_states()
    mem = load_mem()
    f, (ax, axm) = plt.subplots(
        2, 1, figsize=(11, 7),
        gridspec_kw={'height_ratios': [1.0, 1.6]})

    for i, s in enumerate(SEEDS):
        if s in done:
            c, txt = '#1baf7a', 'done {:.1f}h'.format(
                done[s].get('wall_s', 0) / 3600.0)
        elif s in running:
            rows = mem.get(s, [])
            c = '#c98f1e'
            txt = ('iter {} rss {:.0f}G'.format(rows[-1][1], rows[-1][2])
                   if rows else 'startup')
        else:
            c, txt = '#cccccc', 'pending'
        ax.barh(0, 1, left=i, height=0.7, color=c, edgecolor='white')
        ax.text(i + 0.5, 0.55, 's{}'.format(s), ha='center', fontsize=9)
        ax.text(i + 0.5, -0.62, txt, ha='center', fontsize=7, rotation=25)
    ax.set_xlim(0, len(SEEDS)); ax.set_ylim(-1.1, 1.1)
    ax.axis('off')
    ax.set_title('EODS {} — single cell (seed 1): {}'.format(
        LABEL,
        'DONE' if done else ('running' if running or mem else 'pending')))

    for s, rows in sorted(mem.items()):
        t0 = rows[0][0]
        axm.plot([(r[0] - t0) / 3600.0 for r in rows],
                 [r[2] for r in rows], label='s{} rss'.format(s))
    if mem:
        allrows = sorted(r for rows in mem.values() for r in rows)
        t0 = allrows[0][0]
        axm.plot([(r[0] - t0) / 3600.0 for r in allrows],
                 [r[3] for r in allrows], 'k--', lw=1,
                 label='sys avail')
    axm.set_xlabel('hours since first iter'); axm.set_ylabel('GB')
    axm.set_title('driver RSS per cell + system available '
                  '(box: 185G; historical OOM was a 64G head)')
    axm.grid(alpha=0.3)
    if mem:
        axm.legend(fontsize=7, ncol=4)
    f.tight_layout()
    out = os.path.join(FIG_DIR, PREFIX + '_status.png')
    f.savefig(out, dpi=110); plt.close(f)
    print('[plot_eods25] wrote', out)


def _num(v):
    """Per-sim entry -> scalar, tolerating dict entries."""
    if isinstance(v, dict):
        v = v.get('avg_latency_difference')
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def results_fig():
    pkl = os.path.join(DASH, 'metrics_by_dpsize.pkl')
    mbd = None
    if os.path.exists(pkl):
        try:
            mbd = pickle.load(open(pkl, 'rb')).get(DPSIZE)
        except Exception as e:
            print('[plot_eods25] merged pkl unreadable:', e)
    if mbd is None:
        # placeholder keeps the figure (and the dash slot) alive until
        # this campaign's dpsize lands in the merge
        f, ax = plt.subplots(figsize=(8, 2))
        ax.text(.5, .5, 'classical eval results appear here when the '
                'cell completes its eval battery', ha='center', va='center')
        ax.axis('off')
        f.savefig(os.path.join(FIG_DIR, PREFIX + '_results.png'), dpi=110)
        plt.close(f)
        return
    panels = [
        ('stats_best_latencies', -1.0,
         'Avg suboptimality, normal (ms)'),
        ('stats_popp_failures_latency_optimal_specific', -1.0,
         'Avg suboptimality, link failure (ms)'),
        ('stats_pop_failures_latency_optimal_specific', -1.0,
         'Avg suboptimality, site failure (ms)'),
    ]
    f, axs = plt.subplots(1, len(panels), figsize=(13, 4))
    for ax, (key, sign, title) in zip(axs, panels):
        d = mbd.get(key, {})
        solns = [s for s in SOLN_ORDER if s in d]
        for i, sol in enumerate(solns):
            vals = [sign * _num(v) for v in np.atleast_1d(d[sol])]
            vals = [v for v in vals if np.isfinite(v)]
            if not vals:
                continue
            ax.bar(i, np.mean(vals), color=SOLN_COLOR.get(sol, '#777'),
                   label='{} (n={})'.format(sol, len(vals)))
            ax.scatter([i] * len(vals), vals, color='k', s=8, zorder=3)
        ax.set_xticks(range(len(solns)))
        ax.set_xticklabels(solns, rotation=30, ha='right', fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, axis='y')
    f.suptitle('EODS {} — classical eval, sims completed so far '
               '(dots = per-sim)'.format(LABEL))
    f.tight_layout()
    out = os.path.join(FIG_DIR, PREFIX + '_results.png')
    f.savefig(out, dpi=110); plt.close(f)
    print('[plot_eods25] wrote', out)


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    status_fig()
    results_fig()
