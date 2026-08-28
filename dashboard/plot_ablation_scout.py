"""Ablation scout dash figure (Tom 2026-08-26: 'run a full ablation test
on the mac, 1 deployment, size small, 100 iters and set up a dashboard
like v5 scout').

Reads the LOCAL ladder workspace (SCULPTOR_ABL_SCOUT_WS) written by
run_fork_ladder.py -- seed_<s>_<rung>.json per arm -- and renders
figures/dashboards/ablation_scout/ablation_scout_bars.png:
  left  panel: avg latency minus same-deployment OPP per rung (lower
               better; painter drawn as a reference line)
  right panel: wall minutes + iterations per rung.
Also converts each arm's convergence PDF (artifacts/figs) to PNG panels.
"""
import glob
import json
import os
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from helpers.figpaths import fig_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.environ.get(
    'SCULPTOR_ABL_SCOUT_WS',
    '/private/tmp/claude-501/-Users-tomkoch-Documents-sparse-advertisements'
    '-code/62afc954-644c-46d9-a6c9-b0f9ce589095/scratchpad/abl_ws')
OUT = fig_path('ablation_scout_bars.png')
RUNG_ORDER = ['full', 'expl_none', 'no_direction', 'no_memory_dir',
              'no_memory', 'no_mc', 'painter']
LABELS = {'full': 'L6 (SCULPTOR)', 'expl_none': 'no exploration',
          'no_direction': 'no direction', 'no_memory_dir': 'no mem+dir',
          'no_memory': 'no memory', 'no_mc': 'no MC', 'painter': 'PAINTER'}


def load():
    rows = {}
    for fn in glob.glob(os.path.join(WS, 'artifacts', '**', 'seed_*_*.json'),
                        recursive=True):
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        rows[d.get('rung') or os.path.basename(fn).split('_', 2)[2]
             .rsplit('.', 1)[0]] = d
    return rows


def render():
    rows = load()
    if not rows:
        return None
    rungs = [r for r in RUNG_ORDER if r in rows] + \
            [r for r in rows if r not in RUNG_ORDER]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    xs = np.arange(len(rungs))
    diffs = [rows[r].get('diff_vs_opp') for r in rungs]
    colors = ['#c026a8' if r == 'full' else
              ('#1c2733' if r == 'painter' else '#4a6fa5') for r in rungs]
    a1.bar(xs, [d if d is not None else 0 for d in diffs], color=colors)
    for x, d in zip(xs, diffs):
        if d is not None:
            a1.annotate('{:.2f}'.format(d), (x, d), ha='center',
                        va='bottom', fontsize=8)
    a1.set_xticks(xs)
    a1.set_xticklabels([LABELS.get(r, r) for r in rungs], rotation=25,
                       ha='right', fontsize=8)
    a1.set_ylabel('avg latency - OPP (ms)\nlower is better')
    a1.set_title('ablation ladder: seed 1, small, 100 iters', fontsize=10)
    a1.grid(alpha=.25, axis='y')
    walls = [(rows[r].get('wall_s') or 0) / 60 for r in rungs]
    iters = [rows[r].get('n_iters') or 0 for r in rungs]
    a2.bar(xs - .2, walls, width=.4, label='wall (min)', color='#c9862b')
    a2b = a2.twinx()
    a2b.bar(xs + .2, iters, width=.4, label='iters', color='#2f9e6e')
    a2.set_xticks(xs)
    a2.set_xticklabels([LABELS.get(r, r) for r in rungs], rotation=25,
                       ha='right', fontsize=8)
    a2.set_ylabel('wall (min)', color='#c9862b')
    a2b.set_ylabel('iterations', color='#2f9e6e')
    a2.set_title('cost per arm', fontsize=10)
    a2.grid(alpha=.25, axis='y')
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    # convergence PDFs -> PNGs beside the bars
    outdir = os.path.dirname(OUT)
    for pdf in glob.glob(os.path.join(WS, 'artifacts', 'figs', '*.pdf')):
        png = os.path.join(outdir, 'conv_' +
                           os.path.basename(pdf)[:-4] + '.png')
        if os.path.exists(png) and \
                os.path.getmtime(png) >= os.path.getmtime(pdf):
            continue
        subprocess.run(['sips', '-s', 'format', 'png', '--resampleWidth',
                        '760', pdf, '--out', png], capture_output=True)
    return OUT


def render_grid_bars():
    """v5scout-style panels from the objective-dimension grid store
    (Tom 2026-08-27): one panel per objective, bar groups per seed,
    one bar per rung -- final gap vs same-seed OPP (lower better),
    mean over N with min-max whiskers. Painter included as reference."""
    import collections
    store = os.path.join(REPO, 'cache', 'ablation', 'grid_objdim_maxhard')
    vals = collections.defaultdict(list)   # (obj, seed, rung) -> [gap per N]
    objs, seeds = set(), set()
    for fn in glob.glob(os.path.join(store, '*', 'N*', 'seed_*_*.json')):
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        rung = d.get('rung') or 'painter'
        obj = fn.split(os.sep)[-3]
        seed = int(d.get('seed', -1))
        gap = d.get('diff_vs_opp')
        if gap is None and d.get('repo_objective') is not None \
                and d.get('opp_objective') is not None:
            gap = d['repo_objective'] - d['opp_objective']
        if gap is None or abs(gap) > 1e5:
            continue
        vals[(obj, seed, rung)].append(float(gap))
        objs.add(obj); seeds.add(seed)
    if not vals:
        return None
    # sentinel guard (same rule as the difficulty analyses): a NO_ROUTE-
    # priced final is thousands of x the objective's real scale and
    # squashes every honest bar. Clip per objective at 50x the median
    # |gap| and annotate the clipped bars.
    import numpy as _np
    clip = {}
    for obj in objs:
        allv = [abs(v) for (o, _s, _r), vs in vals.items() if o == obj
                for v in vs]
        clip[obj] = 50 * max(float(_np.median(allv)), 1e-9)
    objs = sorted(objs); seeds = sorted(seeds)
    order = [r for r in RUNG_ORDER if any((o, s, r) in vals
             for o in objs for s in seeds)]
    colors = {'full': '#c026a8', 'expl_none': '#e87ba4',
              'no_direction': '#4a3aa7', 'no_memory_dir': '#4a6fa5',
              'no_memory': '#1baf7a', 'no_mc': '#c9862b',
              'painter': '#1c2733'}
    fig, axes = plt.subplots(1, len(objs), figsize=(3.1 * len(objs), 3.4),
                             squeeze=False)
    w = 0.8 / max(len(order), 1)
    for ax, obj in zip(axes[0], objs):
        for k, rung in enumerate(order):
            xs, ys, lo, hi = [], [], [], []
            clipped = []
            for si, seed in enumerate(seeds):
                v = vals.get((obj, seed, rung))
                if not v:
                    continue
                cv = [min(x, clip[obj]) for x in v]
                if max(v) > clip[obj]:
                    clipped.append(si)
                xs.append(si - 0.4 + w * (k + 0.5))
                ys.append(np.mean(cv))
                lo.append(np.mean(cv) - min(cv))
                hi.append(max(cv) - np.mean(cv))
            if xs:
                ax.bar(xs, ys, width=w, color=colors.get(rung, '#999'),
                       yerr=[lo, hi], error_kw={'lw': .6},
                       label=LABELS.get(rung, rung))
                for si in clipped:
                    ax.annotate('^', (si - 0.4 + w * (k + 0.5),
                                      clip[obj] * 0.98),
                                ha='center', fontsize=7, color='#c0392b')
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels(['dep {}'.format(s) for s in seeds], fontsize=8)
        ax.set_title(obj, fontsize=9)
        ax.grid(alpha=.25, axis='y')
        ax.axhline(0, color='#888', lw=.6)
    axes[0][0].set_ylabel('final objective - same-seed OPP\n(lower better)',
                          fontsize=8)
    axes[0][-1].legend(fontsize=5.5, loc='upper right')
    fig.suptitle('objective-dimension ablation grid: 250 iters, N mean '
                 '(whiskers = N range)', fontsize=10)
    fig.tight_layout()
    out = fig_path('ablation_scout_grid_bars.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def refresh_hardobj_view():
    """Recreate the v5scout house plots (plot_hardobj_v4 engine) over the
    hardness grid (Tom 2026-08-27: 'same plots as scout v5'). Builds the
    arm-directory symlink view the engine expects, then invokes it with
    the grid's rungs/objectives."""
    import subprocess
    SRC = os.path.join(REPO, 'cache/ablation/grid_objdim_maxhard')
    DST = os.path.join(REPO, 'cache/ablation/grid_objdim_maxhard_hardobjview')
    for fn in glob.glob(os.path.join(SRC, '*', 'N*', 'seed_*_*.json')):
        parts = fn.split(os.sep)
        obj, ndir, base = parts[-3], parts[-2], parts[-1]
        rung = base.split('_', 2)[2].rsplit('.', 1)[0]
        d = os.path.join(DST, obj, rung, ndir)
        os.makedirs(d, exist_ok=True)
        lnk = os.path.join(d, base)
        if not os.path.islink(lnk):
            os.symlink(fn, lnk)
    env = dict(os.environ)
    env.update({
        'HARDOBJ_ROOT': 'cache/ablation/grid_objdim_maxhard_hardobjview',
        'HARDOBJ_OUT_PREFIX': 'grid_objdim',
        'HARDOBJ_ARMS': ('full:L6 SCULPTOR:#c026a8,'
                         'expl_none:no exploration:#e87ba4,'
                         'no_direction:no direction:#4a3aa7,'
                         'no_memory_dir:no mem+dir:#4a6fa5,'
                         'no_memory:no memory:#1baf7a,'
                         'no_mc:no MC:#c9862b'),
        'HARDOBJ_TITLE': ('Hardness grid (MAXHARD world) -- full ladder x 5 objectives, '
                          '3 deployments, 250 iters '
                          '(0 = one-per-peering; lower = better)'),
        'HARDOBJ_OBJS': ('avg_latency:latency + g*resilience,'
                         'per_site_cost:site cost,max_util:MLU,'
                         'frac_beyond_optimal:frac beyond optimal,'
                         'joint_priority:joint priority'),
    })
    subprocess.run([os.environ.get('PYTHON',
                    '/Users/tomkoch/Documents/venv312/bin/python'),
                    '-m', 'dashboard.plot_hardobj_v4'],
                   cwd=REPO, env=env, capture_output=True, timeout=600)


if __name__ == '__main__':
    print(render())
    print(render_grid_bars())
    refresh_hardobj_view()
