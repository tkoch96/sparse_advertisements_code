"""Per-deployment x per-ablation-arm convergence figures (Tom 2026-09-07:
"for each deployment, for each L, a labeled convergence figure so I can
compare what the different L's did on the SAME deployment").

Per-cell convergence PDFs were never harvested by the ablation queue, but
every fork arm stored its GT-objective series in the cell JSON, and full/
L6's curves were recovered from the paper convergence figures. This
rebuilds the figures from those series.

Layout:
  figures/actual-10-ablation/deployment_<NN>/
      L2_no_mc/convergence_over_iterations.pdf
      L3_no_memory/...   L4_no_memory_dir/...   L5_expl_none/...
      L6_full_SCULPTOR/...
      _overlay_all_arms.pdf        <- all arms on one axis, THE comparison
  figures/actual-10-ablation/README.txt

Each single-arm figure carries dotted painter (this deployment's one-shot
painter objective) and OPP reference lines so an arm reads in context.
Capability order (ascending), matching the ladder / README (L6 = SCULPTOR):
  L1 painter (baseline, no training — reference line only)
  L2 no_mc | L3 no_memory | L4 no_memory_dir | L5 expl_none | L6 full
"""
import argparse
import glob
import json
import os
import re
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# capability-ascending; (rung, L-rank, human label, color)
LADDER = [
    ('no_mc',         2, 'monte-carlo off (painter + one-flip search)', 'tab:brown'),
    ('no_memory',     3, 'monte-carlo on, no memory',                   'tab:orange'),
    ('no_memory_dir', 4, '+ direction (flip-threshold step)',           'tab:purple'),
    ('expl_none',     5, '+ memory (continuous advertisement)',         'tab:blue'),
    ('full',          6, '+ entropic exploration = SCULPTOR',           'tab:green'),
]


L_OF = {rung: L for rung, L, _lab, _c in LADDER}
L_OF['painter'] = 1


def armslug(rung):
    return 'L{}_{}'.format(L_OF[rung],
                           'full_SCULPTOR' if rung == 'full' else rung)


# harvested names (evaluations.ablation_cell.harvest_figs: '<label>_<run-dir
# suffix>'; the fork's run dir is '<rung>-dep<seed>-N<n>-<mode>', or
# '<rung>-dep<seed>-fixed' for the scrubbed full rung), e.g.
#   cdf_testing_feature-actual-10_no_mc-dep1-N10-smart.pdf
#   ME_cdf_testing_feature-actual-10_full-dep2-fixed.pdf
#   cdf_testing_feature-actual-10_expl_none-dep1-N10-smart_state-12.pkl
_HARVEST = re.compile(r'^(?P<me>ME_)?(?P<label>.+?)_(?P<rung>[a-z_]+)-dep(?P<seed>\d+)-'
                      r'(?P<rest>[^_]+?)(?:_state-(?P<n>\d+))?\.(?P<ext>pdf|pkl)$')


def organize_harvest(figs_dir, out_dir):
    """The REAL runtime figures (Tom 2026-09-07 rerun): file every
    harvested per-cell PDF / state pickle from run_ablation_cdf's
    artifacts_figs dir into the same deployment_NN/L#_<rung>/ tree the
    series-rebuilt figures use, so the two are drop-in comparable."""
    n = 0
    skipped = []
    os.makedirs(out_dir, exist_ok=True)
    for fn in sorted(glob.glob(os.path.join(figs_dir, '*'))):
        m = _HARVEST.match(os.path.basename(fn))
        if not m or m.group('rung') not in L_OF:
            skipped.append(os.path.basename(fn))
            continue
        seed = int(m.group('seed'))
        adir = os.path.join(out_dir, 'deployment_{:02d}'.format(seed),
                            armslug(m.group('rung')))
        os.makedirs(adir, exist_ok=True)
        if m.group('ext') == 'pkl':
            dst = 'final_state-{}.pkl'.format(m.group('n'))
        elif m.group('me'):
            dst = 'model_error_over_iterations.pdf'
        else:
            dst = 'convergence_over_iterations.pdf'
        shutil.copy(fn, os.path.join(adir, dst))
        n += 1
    if skipped:
        print('[tree] skipped {} unrecognized: {}'.format(
            len(skipped), skipped[:8]))
    with open(os.path.join(out_dir, 'README.txt'), 'w') as fh:
        fh.write(
            'Per-deployment ablation convergence figures -- the GENUINE '
            'runtime make_plots output of each cell, harvested by\n'
            'run_ablation_cdf.py --artifacts-figs (not rebuilt from series).\n\n'
            'deployment_NN/L<rank>_<rung>/convergence_over_iterations.pdf\n'
            '                            model_error_over_iterations.pdf '
            '(when the cell produced one)\n'
            '                            final_state-<N>.pkl (hot-start / '
            'post-mortem artifact)\n\n'
            'Capability order (ascending; L6 = SCULPTOR):\n'
            '  L1 painter        one-shot baseline\n'
            '  L2 no_mc          monte-carlo off (painter + one-flip search)\n'
            '  L3 no_memory      monte-carlo on, no memory\n'
            '  L4 no_memory_dir  + direction (flip-threshold step)\n'
            '  L5 expl_none      + memory (continuous advertisement)\n'
            '  L6 full           + entropic exploration = SCULPTOR '
            '(depstore cache-hit cells have no runtime figure)\n')
    print('[tree] filed {} artifacts under {}'.format(n, out_dir))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--series', default=None,
                    help='ablation_series_export.json (arms/painter/opp)')
    ap.add_argument('--full-curves', default=None,
                    help='full_curves_extracted_a10.json')
    ap.add_argument('--out-dir', default='figures/actual-10-ablation')
    ap.add_argument('--from-harvest', default=None,
                    help='instead of rebuilding from series: file the '
                         'harvested per-cell PDFs/pickles in this '
                         'artifacts_figs dir into the same tree')
    a = ap.parse_args()

    if a.from_harvest:
        return organize_harvest(a.from_harvest, a.out_dir)
    assert a.series and a.full_curves, '--series and --full-curves required'

    S = json.load(open(a.series))
    full_curves = json.load(open(a.full_curves))
    arms, painter, opp = S['arms'], S['painter'], S['opp']
    seeds = sorted(arms, key=int)
    os.makedirs(a.out_dir, exist_ok=True)

    def series_for(seed, rung):
        if rung == 'full':
            fc = full_curves.get(seed)
            return list(zip(fc['iters'], fc['objs'])) if fc else None
        pts = arms.get(seed, {}).get(rung)
        return [(p[0], p[1]) for p in pts] if pts else None

    n_fig = 0
    for seed in seeds:
        dep_dir = os.path.join(a.out_dir, 'deployment_{:02d}'.format(int(seed)))
        os.makedirs(dep_dir, exist_ok=True)
        pv = painter.get(seed)
        ov = opp.get(seed)
        overlay, oleg = plt.subplots(figsize=(7.2, 4.6))

        for rung, L, label, color in LADDER:
            pts = series_for(seed, rung)
            if not pts:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            # overlay
            oleg.plot(xs, ys, color=color, lw=1.6,
                      label='L{} {} ({})'.format(L, rung, label))
            # single-arm figure
            armslug = 'L{}_{}'.format(
                L, 'full_SCULPTOR' if rung == 'full' else rung)
            adir = os.path.join(dep_dir, armslug)
            os.makedirs(adir, exist_ok=True)
            f, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.plot(xs, ys, color=color, lw=1.8,
                    label='L{} {}'.format(L, rung))
            if pv is not None:
                ax.axhline(pv, color='tab:red', lw=.9, ls=':',
                           label='painter (L1, one-shot)')
            if ov is not None:
                ax.axhline(ov, color='k', lw=.9, ls=':',
                           label='one-per-peering (optimal)')
            ax.set_xlabel('training iteration')
            ax.set_ylabel('ground-truth objective (lower better)')
            ax.set_title('deployment {} — L{} {}\n{}'.format(
                seed, L, rung, label), fontsize=9)
            ax.legend(fontsize=7); ax.grid(alpha=.3)
            f.tight_layout()
            f.savefig(os.path.join(adir, 'convergence_over_iterations.pdf'),
                      bbox_inches='tight')
            plt.close(f)
            n_fig += 1

        if pv is not None:
            oleg.axhline(pv, color='tab:red', lw=1.0, ls=':',
                         label='L1 painter (one-shot baseline)')
        if ov is not None:
            oleg.axhline(ov, color='k', lw=1.0, ls=':',
                         label='one-per-peering (optimal)')
        oleg.set_xlabel('training iteration')
        oleg.set_ylabel('ground-truth objective (lower better)')
        oleg.set_title('deployment {} — all ablation arms'.format(seed))
        oleg.legend(fontsize=7); oleg.grid(alpha=.3)
        overlay.tight_layout()
        overlay.savefig(os.path.join(dep_dir, '_overlay_all_arms.pdf'),
                        bbox_inches='tight')
        plt.close(overlay)

    with open(os.path.join(a.out_dir, 'README.txt'), 'w') as fh:
        fh.write(
            'Per-deployment ablation convergence figures.\n\n'
            'deployment_NN/  -- one folder per actual-10 paper deployment '
            '(NN = seed 1..20).\n'
            '  _overlay_all_arms.pdf  -- ALL arms on one axis; the '
            'same-deployment comparison.\n'
            '  L<rank>_<rung>/convergence_over_iterations.pdf  -- one arm, '
            'with painter + OPP reference lines.\n\n'
            'Capability order (ascending; L6 = SCULPTOR):\n'
            '  L1 painter        one-shot baseline (no training; ref line only)\n'
            '  L2 no_mc          monte-carlo off (painter + one-flip search)\n'
            '  L3 no_memory      monte-carlo on, no memory\n'
            '  L4 no_memory_dir  + direction (flip-threshold step)\n'
            '  L5 expl_none      + memory (continuous advertisement)\n'
            '  L6 full           + entropic exploration = SCULPTOR\n\n'
            'y = ground-truth objective (lower better); the trained-objective\n'
            'series (steady + gamma*resilience) each arm optimized. full/L6\n'
            'curves recovered from the paper convergence figures; the rest\n'
            'from the ablation cell JSONs.\n')
    print('wrote {} single-arm figures + {} overlays under {}'.format(
        n_fig, len(seeds), a.out_dir))


if __name__ == '__main__':
    main()
