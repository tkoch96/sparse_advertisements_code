"""Two-panel mesh plot over N: median steady latency and median
fraction-of-traffic-congested-under-popp-failures, one line per ladder
rung, with painter and one-per-peering reference bands.

Inputs: cache/ablation/mesh_georand (rescored solver JSONs),
cache/model_error/rerank/mesh_N*/seed_*.json (LP-scored pfail),
cache/model_error/opp_ref_georand.json.
Output: figures/mesh_georand_overN.png (+ .pdf)
"""
import glob
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NS = [1, 2, 5, 10, 20, 50]
RUNGS = ['no_mc', 'no_memory', 'no_direction', 'full']
# categorical palette (validated reference order: blue, orange, aqua, yellow)
COLORS = {'no_mc': '#2a78d6', 'no_memory': '#eb6834',
          'no_direction': '#1baf7a', 'full': '#eda100'}


def load():
    pfail = {}
    for fn in glob.glob('cache/model_error/rerank/mesh_N*/seed_*.json'):
        n = int(fn.split(os.sep)[3].split('_N')[1])
        r = json.load(open(fn))
        for rung, e in r['rungs'].items():
            if e.get('popp_fail'):
                pfail.setdefault((rung, n), []).append(
                    e['popp_fail']['mean'])
    scong, clean = {}, {}
    for e in json.load(open('cache/model_error/steady/mesh_steady.json')):
        if not e.get('solved'):
            continue
        n = int(e['dir'].split(os.sep)[-1][1:])
        scong.setdefault((e['rung'], n), []).append(
            e['steady_congested_frac'])
        if e['clean_avg_lat'] is not None:
            clean.setdefault((e['rung'], n), []).append(e['clean_avg_lat'])
    opp = json.load(open('cache/model_error/opp_ref_georand.json'))
    return scong, clean, pfail, opp


def main():
    scong, clean, pfail, opp = load()
    opp_lat = st.median(v['avg_lat'] for v in opp.values() if v)
    opp_pf = st.median(v['pfail_mean'] for v in opp.values() if v)
    refs = {
        'scong': (0.0, st.median(scong.get(('painter', 1), [np.nan]))),
        'clean': (opp_lat, st.median(clean.get(('painter', 1), [np.nan]))),
        'pfail': (opp_pf, st.median(pfail.get(('painter', 1), [np.nan]))),
    }
    panels = (
        (scong, 'median frac. traffic congested\nor stranded, STEADY state',
         refs['scong'], False),
        (clean, 'median steady latency of\nROUTED traffic (ms)',
         refs['clean'], False),
        (pfail, 'median frac. traffic congested\nunder popp FAILURES',
         refs['pfail'], False),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for ax, (data, ylab, (ref_o, ref_p), logy) in zip(axes, panels):
        for rung in RUNGS:
            ys = [st.median(data[(rung, n)]) if (rung, n) in data else np.nan
                  for n in NS]
            ax.plot(NS, ys, 'o-', color=COLORS[rung], label=rung, lw=2,
                    ms=5)
        ax.axhline(ref_o, color='0.35', ls='--', lw=1.2)
        ax.axhline(ref_p, color='0.35', ls=':', lw=1.2)
        ax.text(NS[-1] * 1.1, ref_o, 'opp', va='center', fontsize=8,
                color='0.35')
        ax.text(NS[-1] * 1.1, ref_p, 'painter', va='center', fontsize=8,
                color='0.35')
        ax.set_xscale('log')
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel('measurement budget N')
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle('georand mesh over measurement budget (100 iters, cache '
                 'off, MC=1; medians of 5 deployments; dashed=one-per-'
                 'peering, dotted=painter)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs('figures', exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig('figures/mesh_georand_overN.{}'.format(ext), dpi=170)
    print('wrote figures/mesh_georand_overN.png')


if __name__ == '__main__':
    main()
