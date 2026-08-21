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
from helpers.figpaths import fig_path  # dashboard figures -> figures/dashboards/<dashboard>/

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
# HARDOBJ_ARMS="dir:label:color,..." overrides the rung set (Tom
# 2026-08-18: the v5 scout compares step-size policies, not the L1-L6
# ladder, but must render through THIS engine so the dash view is
# identical to the unified grid / future full panel).
_arms_env = os.environ.get('HARDOBJ_ARMS')
if _arms_env:
    ARMS = [tuple(x.split(':', 2)) for x in _arms_env.split(',')]
# HARDOBJ_PAINTER_ROOT: painter-reference store override — scout
# stores carry no painter rung; the maxhard v2 painters share world,
# seeds and objective scalars, so their band transfers exactly.
PROOT = os.path.join(REPO, os.environ.get(
    'HARDOBJ_PAINTER_ROOT', os.environ.get(
        'HARDOBJ_ROOT', 'cache/ablation/hardobj_v3')))
# HARDOBJ_OBJS: comma list of key:title pairs — makes the objective set
# env-driven so the unified grid view renders latency+resilience as an
# equal first-class panel next to the hard objectives (Tom 2026-08-17).
_objs_env = os.environ.get('HARDOBJ_OBJS')
if _objs_env:
    OBJS = [tuple(x.split(':', 1)) for x in _objs_env.split(',')]
else:
    OBJS = [('fracb', 'frac_beyond_optimal (hinge)'),
            ('mlu', 'max_util v2'),
            ('prio', 'joint latency+bulk')]


N_ERRS = {}

# Per-family y-axis units (Tom 2026-08-18: "clarify what the units are
# showing" — every scalar is ms-EQUIVALENT, not the raw metric its name
# suggests; mlu in particular is alpha*MLU folded into a latency sum,
# NOT a utilization ratio). Formulas live in the dash captions.
UNITS = {
    'lat': 'ms (avg routed lat + 50*bad_frac + gamma*resilience)',
    'fracb': 'ms/unit vol (hinge excess past optimal+10ms)',
    'mlu': 'ms-equiv (A*minMLU + routed lat + 3A*bad; A=10x floor)',
    'prio': 'ms-equiv (avg lat + 100*bulk-congested frac)',
}


def load(obj, field='objective'):
    """{(arm, N): [(seed, arm - same-seed opp)]} for the objective
    scalar (default) or field='avg_lat' (the stored steady latency,
    used to decompose the mlu composite: delta-objective minus
    delta-avg_lat ~ A*(delta-minMLU + 3*delta-bad))."""
    out = {}
    N_ERRS[obj] = 0
    keys = (('repo_objective', 'opp_objective') if field == 'objective'
            else ('avg_lat', 'opp_avg_lat'))
    for fn in glob.glob(os.path.join(ROOT, obj, '*', 'N*', 'seed_*_*.json')):
        try:
            d = json.load(open(fn))
        except (OSError, ValueError):
            continue
        if d.get('solve_error'):
            # dash is the source of truth (Tom 2026-08-18): count and
            # DISPLAY crashed cells instead of silently skipping them
            N_ERRS[obj] += 1
            continue
        if d.get(keys[0]) is None or d.get(keys[1]) is None:
            continue
        parts = fn.split(os.sep)
        arm, ndir = parts[-3], parts[-2]
        out.setdefault((arm, int(ndir[1:])), []).append(
            (d['seed'], float(d[keys[0]]) - float(d[keys[1]])))
    return out


def painter_ref(obj):
    """Mean (painter - opp). PRIMARY source: the grid's OWN painter rung
    cells (ROOT/<obj>/painter/N*/seed_*_painter.json) — same world, same
    era, same objective scalar as the arms, so every panel can band to
    opp..painter +/-10% (Tom 2026-08-17). Fallback: archived painter
    evals (cost-diff = opp - obj; mlu excluded there — its scalar
    changed between eras)."""
    diffs = []
    for fn in glob.glob(os.path.join(
            PROOT, obj, 'painter', 'N*', 'seed_*_painter.json')):
        try:
            d = json.load(open(fn))
        except (OSError, ValueError):
            continue
        if d.get('repo_objective') is not None \
                and d.get('opp_objective') is not None:
            diffs.append(float(d['repo_objective'])
                         - float(d['opp_objective']))
    if diffs:
        return float(np.mean(diffs))
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


def load_prio_lex():
    """{(arm, N): [(seed, Lstar_gap, bulk_frac, opp_bulk_frac)]} from
    the lexicographic pair persisted by run_fork_ladder (Tom 2026-08-18:
    priority strictly first, bulk best-effort — a PAIR, never summed).
    Lstar_gap = opp_Lstar - Lstar in BENEFIT convention, i.e. positive
    = arm worse, 0 = opp: same panel semantics as every other family,
    but provably opp-floored (capability metric)."""
    out = {}
    for fn in glob.glob(os.path.join(ROOT, 'prio', '*', 'N*',
                                     'seed_*_*.json')):
        try:
            d = json.load(open(fn))
        except (OSError, ValueError):
            continue
        pl = d.get('prio_lex') or {}
        if pl.get('Lstar') is None or pl.get('opp_Lstar') is None:
            continue
        parts = fn.split(os.sep)
        arm, ndir = parts[-3], parts[-2]
        out.setdefault((arm, int(ndir[1:])), []).append(
            (d.get('seed'), float(pl['opp_Lstar']) - float(pl['Lstar']),
             pl.get('bulk_frac'), pl.get('opp_bulk_frac')))
    return out


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
    if pref is not None and np.isfinite(pref) and abs(pref) > 1e-9:
        ax.axhline(pref, color='#888', lw=1.2, linestyle=':')
        ax.text(0.99, pref, ' painter', va='bottom', ha='right',
                transform=ax.get_yaxis_transform(), fontsize=8, color='#888')
        # opp (=0) at the BOTTOM edge, painter at the TOP, +/-10% of the
        # gap (Tom 2026-08-17); axis inverts if painter < opp
        pad = 0.1 * abs(pref)
        sgn = 1.0 if pref >= 0 else -1.0
        ax.set_ylim(-sgn * pad, pref + sgn * pad)
    ax.set_xscale('log')
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlabel('measurement budget N')
    ax.set_ylabel('objective - same-seed opp\n[{}]'.format(
        UNITS.get(obj, 'ms-equivalent')), fontsize=8)
    _ne = N_ERRS.get(obj, 0)
    if _ne:
        ax.text(0.02, 0.98, '{} CRASHED CELLS'.format(_ne),
                transform=ax.transAxes, va='top', fontsize=9,
                color='#c02f4e', fontweight='bold')
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=.25)
    return drawn


def main():
    os.makedirs(FIGS, exist_ok=True)
    any_drawn = False
    fig, axes = plt.subplots(1, len(OBJS),
                             figsize=(5.4 * len(OBJS), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (obj, title) in zip(axes, OBJS):
        d = panel(ax, obj, title)
        any_drawn = any_drawn or d
        if obj == 'prio':
            lex = load_prio_lex()
            if lex:
                # lexicographic 2-panel (Tom 2026-08-18): left = the
                # priority capability gap (opp exact floor); right =
                # bulk deliverable at ZERO priority cost. Never summed.
                f2, (a2, a3) = plt.subplots(1, 2, figsize=(13.5, 4.6))
                for arm, label, color in ARMS:
                    xs, ys, bs = [], [], []
                    for n in NS:
                        rows = lex.get((arm, n), [])
                        if rows:
                            xs.append(n)
                            ys.append(float(np.mean(
                                [r[1] for r in rows])))
                            bb = [r[2] for r in rows
                                  if r[2] is not None]
                            bs.append(float(np.mean(bb))
                                      if bb else np.nan)
                    if xs:
                        a2.plot(xs, ys, 'o-', color=color, label=label,
                                lw=1.5, ms=4)
                        a3.plot(xs, bs, 'o-', color=color, lw=1.5, ms=4)
                ob = [r[3] for rows in lex.values() for r in rows
                      if r[3] is not None]
                if ob:
                    a3.axhline(float(np.mean(ob)), color='k', lw=1,
                               linestyle='--')
                    a3.text(0.99, float(np.mean(ob)), ' opp',
                            va='bottom', ha='right', fontsize=8,
                            transform=a3.get_yaxis_transform())
                a2.axhline(0, color='k', lw=1, linestyle='--')
                for ax_ in (a2, a3):
                    ax_.set_xscale('log')
                    ax_.set_xticks(NS)
                    ax_.set_xticklabels([str(n) for n in NS])
                    ax_.set_xlabel('measurement budget N')
                    ax_.grid(alpha=.25)
                a2.set_ylabel('priority L* - opp (ms; capability, '
                              'opp exact floor)', fontsize=8)
                a2.set_title(title + ' — PRIORITY (lexicographic '
                             'primary)', fontsize=10)
                a2.legend(fontsize=8, frameon=False)
                a3.set_ylabel('bulk delivered fraction at zero '
                              'priority cost', fontsize=8)
                a3.set_title('BULK (lexicographic secondary; higher '
                             'better)', fontsize=10)
                f2.tight_layout()
                f2.savefig(fig_path('{}_{}.png'.format(OUT_PREFIX, obj)), dpi=150)
                plt.close(f2)
                continue
        if obj == 'mlu':
            # mlu decomposition (Tom 2026-08-18: "what does 5 mean?"):
            # the composite hides whether a delta is utilization or the
            # latency tie-break. Left: full objective delta. Right: the
            # latency share (avg_lat - opp_avg_lat) from the same cells;
            # objective-minus-latency-share ~ A*(dMLU + 3*dbad), i.e.
            # the pure utilization+stranding component.
            f2, (a2, a3) = plt.subplots(1, 2, figsize=(13.5, 4.6))
            if panel(a2, obj, title + ' — full composite'):
                lat_data = load(obj, field='avg_lat')
                for arm, label, color in ARMS:
                    xs, ys = [], []
                    for n in NS:
                        vals = [v for _, v in lat_data.get((arm, n), [])]
                        if vals:
                            xs.append(n)
                            ys.append(float(np.mean(vals)))
                    if xs:
                        a3.plot(xs, ys, 'o-', color=color, label=label,
                                lw=1.5, ms=4)
                a3.axhline(0, color='k', lw=1, linestyle='--')
                a3.set_xscale('log')
                a3.set_xticks(NS)
                a3.set_xticklabels([str(n) for n in NS])
                a3.set_xlabel('measurement budget N')
                a3.set_ylabel('avg_lat - same-seed opp [ms]', fontsize=8)
                a3.set_title('latency share of the composite '
                             '(remainder = A*(dMLU + 3*dbad))',
                             fontsize=9)
                a3.grid(alpha=.25)
                a2.legend(fontsize=8, frameon=False)
                f2.tight_layout()
                f2.savefig(fig_path('{}_{}.png'.format(OUT_PREFIX, obj)), dpi=150)
            plt.close(f2)
            continue
        f2, a2 = plt.subplots(figsize=(7.5, 4.6))
        if panel(a2, obj, title):
            a2.legend(fontsize=8, frameon=False)
            f2.tight_layout()
            f2.savefig(fig_path('{}_{}.png'.format(OUT_PREFIX, obj)),
                       dpi=150)
        plt.close(f2)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(os.environ.get(
        'HARDOBJ_TITLE',
        'Hard objectives v4 — L1-L6 ladder, 10 deployments '
        '(0 = one-per-peering; lower = better)'), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if any_drawn:
        fig.savefig(fig_path('{}_{}panel.png'.format(OUT_PREFIX, len(OBJS))), dpi=150)
        print('wrote hardobj_v4 figures')
    else:
        print('no hardobj_v3 data yet')


if __name__ == '__main__':
    main()
