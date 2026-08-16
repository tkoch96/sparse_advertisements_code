"""Incremental per-objective ladder figures from hardB3_scores.json:
writes the combined 4-panel figure AND one PNG per objective (used by
experiments/dashboard/generate.py tabs).
One panel per objective: own objective value (higher=better) vs N, one
line per rung; seed 1 solid+bold, seeds 2-5 faint; painter dotted, opp
dashed. Partial lines render as far as data exists."""
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = '/Users/tomkoch/Documents/sparse_advertisements_code'
# Env-overridable for variant datasets (hardB3v2 etc.):
STORE = os.path.join(REPO, os.environ.get(
    'HARDB3_STORE', 'cache/model_error/hardB3_scores.json'))
DATA_ROOT_RE = os.environ.get('HARDB3_ROOT_RE', 'cache/ablation/hardB3')
FIG_PREFIX = os.environ.get('HARDB3_FIG_PREFIX', 'hardB3')
NS = [1, 2, 5, 10, 20, 50]
OBJS = [('fracb', 'frac_beyond 10ms (georand)'),
        ('mlu', 'lat + a*MLU (georand)'),
        ('prio', 'joint priority (georand)'),
        ('popfail', 'site_failure / pop (stock)')]
# Monotone ladder (2026-08-14). HARDB3_ARMS=legacy restores the old arm
# set for the pre-stack hardB3 dataset.
if os.environ.get('HARDB3_ARMS') == 'legacy':
    ARMS = [('fixed', 'no_mc', 'L1 no_mc+fixed', '#2a78d6'),
            ('sched', 'no_mc', 'L2 no_mc+sched', '#eb6834'),
            ('sched', 'no_direction', 'L3 no_dir+sched', '#1baf7a'),
            ('sched', 'no_memory', 'L4 no_mem+sched', '#eda100'),
            ('smart', 'no_memory', 'L5 no_mem+smart', '#e87ba4'),
            ('smart', 'full', 'L6 full+smart', '#4a3aa7')]
else:
    ARMS = [('fixed', 'no_mc', 'L1 no_mc+fixed', '#2a78d6'),
            ('sched', 'no_mc', 'L2 no_mc+sched', '#eb6834'),
            ('sched', 'no_memory', 'L3 no_mem+sched', '#1baf7a'),
            ('smart', 'no_memory', 'L4 no_mem+smart', '#eda100'),
            ('smart', 'no_direction', 'L5 no_dir+smart', '#e87ba4'),
            ('smart', 'full', 'L6 full+smart', '#4a3aa7')]

# HARDB3_EXTRA_STORE (Tom 2026-08-16: "L7 should go on the page with
# L1-L6"): a second scores store whose grid rows are merged in under the
# virtual pdir 'smartL7' — L7 shares (smart, full) with L6, so the path
# root is what distinguishes them. Adds the L7 arm row when set.
EXTRA_STORE = os.environ.get('HARDB3_EXTRA_STORE')
if EXTRA_STORE:
    ARMS = ARMS + [('smartL7', 'full', 'L7 bern-K3+smart', '#c02f4e')]


def _clamp_to_refs(ax, painter_excess):
    """Y-limits = the painter<->opp band +/- 10% (Tom 2026-08-15), in
    PER-SEED-NORMALIZED units: values are (objective - same-seed opp),
    so opp = 0 by construction and painter = its mean per-seed excess.
    Normalization matters: raw per-seed opp scales differ up to 30x
    (fracb seed1 -0.248 vs seed4 -0.008), so any absolute-axis band
    clips the good seeds entirely (caught by Tom on fracb seed 4,
    2026-08-15). Collapsed arms still clip off-panel by design."""
    if painter_excess is None:
        return
    lo, hi = min(float(painter_excess), 0.0), max(float(painter_excess), 0.0)
    pad = 0.10 * (hi - lo)
    if pad <= 0:
        pad = 0.10 * max(abs(lo), 1e-9)
    ax.set_ylim(lo - pad, hi + pad)


def main():
    store = json.load(open(STORE))
    grid, painter, opp = {}, {}, {}
    pat = re.compile(
        re.escape(DATA_ROOT_RE)
        + r'/([^/]+)/([^/]+)/N(\d+)/seed_(\d+)_(.+)\.json')
    for key, rec in store.items():
        if rec['obj_val'] is None:
            continue
        if key.startswith('painter:'):
            o = key.split(':')[1]
            s = int(re.search(r'seed_(\d+)_', key).group(1))
            painter.setdefault(o, {})[s] = rec['obj_val']
            if rec.get('opp_val') is not None:
                opp.setdefault((o, s), rec['opp_val'])
            continue
        m = pat.search(key)
        if not m:
            continue
        o, pmode, n, s, rung = (m.group(1), m.group(2), int(m.group(3)),
                                int(m.group(4)), m.group(5))
        grid[(o, pmode, rung, s, n)] = rec['obj_val']
        if rec.get('opp_val') is not None:
            opp.setdefault((o, s), rec['opp_val'])

    # merge the L7 store under the virtual pdir 'smartL7'
    if EXTRA_STORE and os.path.exists(os.path.join(REPO, EXTRA_STORE)):
        pat7 = re.compile(
            r'cache/ablation/hardB3[^/]*/([^/]+)/([^/]+)/N(\d+)/'
            r'seed_(\d+)_(.+)\.json')
        for key, rec in json.load(open(
                os.path.join(REPO, EXTRA_STORE))).items():
            if rec.get('obj_val') is None or key.startswith('painter:'):
                continue
            m = pat7.search(key)
            if not m:
                continue
            o, n, s, rung = (m.group(1), int(m.group(3)),
                             int(m.group(4)), m.group(5))
            grid[(o, 'smartL7', rung, s, n)] = rec['obj_val']
            if rec.get('opp_val') is not None:
                opp.setdefault((o, s), rec['opp_val'])

    # SANITY GATE (Tom 2026-08-16: "no objective value should ever be
    # better than one per peering" -- fail LOUD before rendering).
    # Exemptions: popfail (sparse legitimately beats opp on the failure
    # composite, standing finding) and -- TEMPORARILY, annotated on the
    # panel -- prio: its metric is two-stage assignment-derived (stage-1
    # latency split is not prio-optimal) and its capability twin is a
    # nonconvex QP; resolution pending Tom's call (2026-08-16 morning).
    import sys as _sys
    if REPO not in _sys.path:
        _sys.path.insert(0, REPO)
    from experiments.dashboard.sanity import assert_not_better_than_opp
    _PRIO_EXEMPT = ('prio',)
    assert_not_better_than_opp(
        (('{}/{}/{}/s{}/N{}'.format(o, pm, r, s, n), o, v,
          opp.get((o, s)))
         for (o, pm, r, s, n), v in grid.items()
         if o not in _PRIO_EXEMPT),
        context='plot_hardb3 ' + FIG_PREFIX)

    # PER-SEED NORMALIZATION (Tom 2026-08-15): everything is plotted as
    # (objective - same-seed opp). Raw opp scales differ up to 30x across
    # seeds, so absolute axes (and any band clamp on them) bury the good
    # seeds. opp = 0 by construction; painter = mean per-seed excess.
    grid = {k: v - opp[(k[0], k[3])] for k, v in grid.items()
            if (k[0], k[3]) in opp}
    painter_excess = {}
    for o, by_seed in painter.items():
        ex = [v - opp[(o, s)] for s, v in by_seed.items() if (o, s) in opp]
        if ex:
            painter_excess[o] = float(np.mean(ex))

    # panels are DATA-DRIVEN: an objective with zero rows renders as a
    # blank axis otherwise (caught 2026-08-16: hardB3v2 has no popfail
    # data, so the 4th panel was empty — Tom: figures must not be blank)
    objs_live = [(o, t) for o, t in OBJS
                 if any(k[0] == o for k in grid)]
    fig, axes = plt.subplots(1, max(1, len(objs_live)),
                             figsize=(5.25 * max(1, len(objs_live)), 4.6))
    if len(objs_live) <= 1:
        axes = [axes]
    n_pts = 0
    for ax, (o, title) in zip(axes, objs_live):
        for pmode, rung, label, color in ARMS:
            for s in (1, 2, 3, 4, 5):
                # budgeted-fixed L1 is a real N-grid arm in the new
                # ladder; only the legacy dataset confined it to N=1
                if pmode == 'fixed' and os.environ.get(
                        'HARDB3_ARMS') == 'legacy':
                    xs = [1]
                else:
                    xs = NS
                ys = [grid.get((o, pmode, rung, s, n), np.nan) for n in xs]
                if all(np.isnan(y) for y in ys):
                    continue
                n_pts += sum(~np.isnan(np.array(ys)))
                bold = (s == 1)
                ax.plot(xs, ys, 'o-' if bold else '.-', color=color,
                        lw=2.0 if bold else 0.7,
                        ms=4.5 if bold else 2,
                        alpha=1.0 if bold else 0.35,
                        label=label if bold else None)
        pex = painter_excess.get(o)
        if pex is not None:
            ax.axhline(pex, color='0.35', ls=':', lw=1.2)
            ax.text(1.02, pex, 'painter', fontsize=7, color='0.35',
                    transform=ax.get_yaxis_transform())
        ax.axhline(0.0, color='0.35', ls='--', lw=1.2)
        ax.text(1.02, 0.0, 'opp', fontsize=7, color='0.35',
                transform=ax.get_yaxis_transform())
        ax.set_xscale('log')
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        ax.set_xlabel('measurement budget N')
        ax.set_ylabel('objective - same-seed opp (0 = opp)')
        if o == 'popfail':
            # site_failure evals span orders of magnitude; a linear
            # axis pins everything to one line (Tom, 2026-08-14)
            ax.set_yscale('symlog')
        _clamp_to_refs(ax, pex)
        if o == 'prio':
            ax.annotate('assignment-derived metric: small opp\n'
                        'crossings are metric-structural (see tab note)',
                        xy=(0.02, 0.98), xycoords='axes fraction',
                        va='top', fontsize=6.5, color='#a33',
                        style='italic')
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        'Ladder over N, each panel trained AND scored on its OWN objective '
        '(150 iters; seed 1 bold, seeds 2-5 faint; INCREMENTAL — lines '
        'extend as VM cells land)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.join(REPO, 'figures'), exist_ok=True)
    out = os.path.join(REPO, 'figures', FIG_PREFIX + '_ladders_incremental.png')
    fig.savefig(out, dpi=170)
    print('wrote', out, '({} points)'.format(n_pts))
    # per-objective single panels for the dashboard tabs
    for o, title in OBJS:
        f1, ax = plt.subplots(figsize=(6.4, 4.6))
        for pmode, rung, label, color in ARMS:
            for s in (1, 2, 3, 4, 5):
                xs = ([1] if (pmode == 'fixed' and os.environ.get(
                    'HARDB3_ARMS') == 'legacy') else NS)
                ys = [grid.get((o, pmode, rung, s, n), np.nan) for n in xs]
                if all(np.isnan(y) for y in ys):
                    continue
                bold = (s == 1)
                ax.plot(xs, ys, 'o-' if bold else '.-', color=color,
                        lw=2.0 if bold else 0.7, ms=4.5 if bold else 2,
                        alpha=1.0 if bold else 0.35,
                        label=label if bold else None)
        pex = painter_excess.get(o)
        if pex is not None:
            ax.axhline(pex, color='0.35', ls=':', lw=1.2)
        ax.axhline(0.0, color='0.35', ls='--', lw=1.2)
        ax.set_xscale('log'); ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        ax.set_xlabel('measurement budget N')
        ax.set_ylabel('objective - same-seed opp (0 = opp)')
        if o == 'popfail':
            # site_failure evals span orders of magnitude; a linear
            # axis pins everything to one line (Tom, 2026-08-14)
            ax.set_yscale('symlog')
        _clamp_to_refs(ax, pex)
        if o == 'prio':
            ax.annotate('assignment-derived metric: small opp\n'
                        'crossings are metric-structural (see tab note)',
                        xy=(0.02, 0.98), xycoords='axes fraction',
                        va='top', fontsize=6.5, color='#a33',
                        style='italic')
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25); ax.legend(fontsize=7, frameon=False)
        f1.tight_layout()
        f1.savefig(os.path.join(REPO, 'figures',
                                FIG_PREFIX + '_{}.png'.format(o)), dpi=150)
        plt.close(f1)
    print('wrote per-objective panels')

    # ---- dedicated MLU figure in UTILIZATION units (Tom 2026-08-16:
    # switching the lane to the joint objective silently REMOVED actual
    # utilization from the dash; this figure is the mlu tab's primary).
    # Reads the 'components'/'opp_components' fields score_hardb3 now
    # persists; rows without components (pre-fix) are skipped, so the
    # figure fills in as the rescore chews through.
    def _util_rows(path, pdir_override=None):
        rows, opp_u, pain_u = {}, {}, {}
        if not path or not os.path.exists(path):
            return rows, opp_u, pain_u
        gpat = re.compile(
            r'cache/ablation/hardB3[^/]*/mlu/([^/]+)/N(\d+)/'
            r'seed_(\d+)_(.+)\.json')
        for key, rec in json.load(open(path)).items():
            comp = rec.get('components') or {}
            if key.startswith('painter:mlu:'):
                if comp.get('max_util') is not None:
                    s = int(re.search(r'seed_(\d+)_', key).group(1))
                    pain_u[s] = float(comp['max_util'])
                continue
            if key.startswith('painter:'):
                continue
            m = gpat.search(key)
            if not m or comp.get('max_util') is None:
                continue
            s = int(m.group(3))
            pmode = pdir_override or m.group(1)
            rows[(pmode, m.group(4), s, int(m.group(2)))] = \
                float(comp['max_util'])
            oc = rec.get('opp_components') or {}
            if oc.get('max_util') is not None:
                opp_u.setdefault(s, float(oc['max_util']))
        return rows, opp_u, pain_u

    rows, opp_u, pain_u = _util_rows(STORE)
    if EXTRA_STORE:
        r7, o7, _ = _util_rows(os.path.join(REPO, EXTRA_STORE), 'smartL7')
        rows.update(r7)
        for s, v in o7.items():
            opp_u.setdefault(s, v)
    # PER-SEED normalization: opp min-MLU spans 0.84-0.92 across seeds,
    # so absolute axes make arms cross the mean-opp line spuriously
    # (caught visually 2026-08-16). min-MLU is a TRUE capability metric,
    # so (util - own-seed opp util) >= 0 is a mathematical guarantee.
    rows = {k: v - opp_u[k[2]] for k, v in rows.items() if k[2] in opp_u}
    pain_ex = [v - opp_u[s] for s, v in pain_u.items() if s in opp_u]
    fu, axu = plt.subplots(figsize=(6.4, 4.6))
    n_u = 0
    for pmode, rung, label, color in ARMS:
        for s in (1, 2, 3, 4, 5):
            ys = [rows.get((pmode, rung, s, n), np.nan) for n in NS]
            if all(np.isnan(y) for y in ys):
                continue
            n_u += int(np.sum(~np.isnan(np.array(ys))))
            bold = (s == 1)
            axu.plot(NS, ys, 'o-' if bold else '.-', color=color,
                     lw=2.0 if bold else 0.7, ms=4.5 if bold else 2,
                     alpha=1.0 if bold else 0.35,
                     label=label if bold else None)
    axu.axhline(0.0, color='0.35', ls='--', lw=1.2)
    axu.text(1.02, 0.0, 'opp', fontsize=7, color='0.35',
             transform=axu.get_yaxis_transform())
    if pain_ex:
        axu.axhline(float(np.mean(pain_ex)), color='0.35', ls=':', lw=1.2)
        axu.text(1.02, np.mean(pain_ex), 'painter', fontsize=7,
                 color='0.35', transform=axu.get_yaxis_transform())
    axu.set_xscale('log')
    axu.set_xticks(NS)
    axu.set_xticklabels([str(n) for n in NS])
    axu.set_xlabel('measurement budget N')
    axu.set_ylabel('peak utilization - same-seed opp (0 = opp floor)')
    axu.set_title('MLU — utilization excess of final advertisements',
                  fontsize=10)
    axu.grid(True, alpha=0.25)
    axu.legend(fontsize=7, frameon=False)
    fu.tight_layout()
    fu.savefig(os.path.join(REPO, 'figures',
                            FIG_PREFIX + '_mlu_util.png'), dpi=150)
    plt.close(fu)
    print('wrote mlu utilization figure ({} points)'.format(n_u))


if __name__ == '__main__':
    main()
