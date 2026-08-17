"""Hard-objectives v4 panels: per-objective OWN-objective ladder over N.

Reads cache/ablation/hardobj_v3/<obj>/<Ldir>/N*/seed_*_*.json directly
(the cells carry repo_objective + same-seed opp_objective; rescore_fork
keeps them authoritative) and renders one panel per objective in the
house format: mean over landed seeds of (objective - same-seed opp),
0 = one-per-peering, lower = better. Data-driven: empty panels are
skipped, partial seeds fine.

Outputs figures/hardobj_v4_<obj>.png + a combined 3-panel.
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
# HARDOBJ_ROOT / HARDOBJ_OUT_PREFIX let the solver-fork (HiGHS) tabs
# reuse this renderer against their own store (Tom 2026-08-17).
ROOT = os.path.join(REPO, os.environ.get(
    'HARDOBJ_ROOT', 'cache/ablation/hardobj_v3'))
OUT_PREFIX = os.environ.get('HARDOBJ_OUT_PREFIX', 'hardobj_v4')
FIGS = os.path.join(REPO, 'figures')
NS = [1, 2, 5, 10, 20, 50]
ARMS = [('L1_nomc_fixed', 'L1 no_mc+fixed', '#2a78d6'),
        ('L2_nomc_sched', 'L2 no_mc+sched', '#eb6834'),
        ('L3_nomem_sched', 'L3 no_mem+sched', '#1baf7a'),
        ('L4_nodir_sched', 'L4 no_dir+sched', '#eda100'),
        ('L5_full_sched', 'L5 full+sched', '#e87ba4'),
        ('L6_full_slotted', 'L6 slotted WHEN', '#4a3aa7')]
OBJS = [('fracb', 'frac_beyond_optimal (hinge)'),
        ('mlu', 'max_util v2'),
        ('prio', 'joint latency+bulk')]


def load(obj):
    """{(arm, N): [(seed, obj - same-seed opp)]}"""
    out = {}
    for fn in glob.glob(os.path.join(ROOT, obj, '*', 'N*', 'seed_*_*.json')):
        try:
            d = json.load(open(fn))
        except (OSError, ValueError):
            continue
        if d.get('solve_error') or d.get('repo_objective') is None \
                or d.get('opp_objective') is None:
            continue
        parts = fn.split(os.sep)
        arm, ndir = parts[-3], parts[-2]
        out.setdefault((arm, int(ndir[1:])), []).append(
            (d['seed'], float(d['repo_objective']) - float(d['opp_objective'])))
    return out


def painter_ref(obj):
    """Mean (painter - opp) in COST orientation from the archived painter
    evals (old stores are benefit-oriented, so cost-diff = opp - obj).
    mlu's scalar changed between eras (lat_plus -> max_util v2), so its
    archived painter is NOT comparable and we return None."""
    if obj == 'mlu':
        return None
    diffs = []
    for store in ('cache/model_error/hardB3v2_scores.json',
                  'cache/model_error/hardB3_scores.json'):
        p = os.path.join(REPO, store)
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            continue
        for k, v in d.items():
            if str(k).startswith('painter:%s:' % obj):
                diffs.append(v['opp_val'] - v['obj_val'])
        if diffs:
            break
    return float(np.mean(diffs)) if diffs else None


def panel(ax, obj, title):
    data = load(obj)
    drawn = False
    for arm, label, color in ARMS:
        xs, ys = [], []
        for n in NS:
            vals = [v for _, v in data.get((arm, n), [])]
            if vals:
                xs.append(n)
                ys.append(float(np.mean(vals)))
        if xs:
            ax.plot(xs, ys, 'o-', color=color, label=label, lw=1.5, ms=4)
            drawn = True
    ax.axhline(0, color='k', lw=1, linestyle='--')
    # reference window (Tom 2026-08-17): y spans opp -10% .. painter +10%
    # of the painter-opp gap, so both anchors frame the plot
    pref = painter_ref(obj)
    if pref is not None and pref > 0:
        ax.axhline(pref, color='#888', lw=1.2, linestyle=':')
        ax.text(0.99, pref, ' painter', va='bottom', ha='right',
                transform=ax.get_yaxis_transform(), fontsize=8, color='#888')
        ax.set_ylim(-0.1 * pref, 1.1 * pref)
    ax.set_xscale('log')
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlabel('measurement budget N')
    ax.set_ylabel('objective - same-seed opp')
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=.25)
    return drawn


def main():
    os.makedirs(FIGS, exist_ok=True)
    any_drawn = False
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (obj, title) in zip(axes, OBJS):
        d = panel(ax, obj, title)
        any_drawn = any_drawn or d
        f2, a2 = plt.subplots(figsize=(7.5, 4.6))
        if panel(a2, obj, title):
            a2.legend(fontsize=8, frameon=False)
            f2.tight_layout()
            f2.savefig(os.path.join(FIGS, '{}_{}.png'.format(OUT_PREFIX, obj)),
                       dpi=150)
        plt.close(f2)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Hard objectives v4 — L1-L6 ladder, 10 deployments '
                 '(0 = one-per-peering; lower = better)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if any_drawn:
        fig.savefig(os.path.join(FIGS, '{}_3panel.png'.format(OUT_PREFIX)), dpi=150)
        print('wrote hardobj_v4 figures')
    else:
        print('no hardobj_v3 data yet')


if __name__ == '__main__':
    main()
