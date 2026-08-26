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


if __name__ == '__main__':
    print(render())
