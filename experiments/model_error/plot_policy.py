"""Final policy-ladder figure: three panels over N (steady congested
fraction, clean routed latency, failure congestion), one line per ladder
arm, painter + one-per-peering references.

Data: cache/model_error/rerank/policy_<arm>_N*/seed_*.json and
cache/model_error/steady/policy_steady*.json (built by rerank_ladder /
steady_metrics); painter refs from the mesh datasets; opp refs from
cache/model_error/opp_ref_georand.json.
Output: figures/policy_ladder_overN.{png,pdf}
"""
import glob
import json
import os
import os as _os2
import statistics as st
_STAT = st.mean if _os2.environ.get('POLICY_PLOT_STAT', 'median') == 'mean' else st.median
_OUT = _os2.environ.get('POLICY_PLOT_OUT', 'policy_ladder_overN')
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NS = [1, 2, 5, 10, 20, 50]
ARMS = [('L1B_nomc_fixedN', 'L1 no_mc+fixed', '#2a78d6'),
        ('L2_nomc_sched', 'L2 no_mc+sched', '#eb6834'),
        ('L3_nodir_sched', 'L3 no_dir+sched', '#1baf7a'),
        ('L4_nomem_sched', 'L4 no_mem+sched', '#eda100'),
        ('L5_nomem_smart', 'L5 no_mem+smart', '#e87ba4'),
        ('L6_full_smart', 'L6 full+smart', '#4a3aa7')]


def load():
    pfail, scong, clean = {}, {}, {}
    for fn in glob.glob('cache/model_error/rerank/policy_*_N*/seed_*.json'):
        tag = fn.split(os.sep)[3]           # policy_<arm>_N<n>
        arm = tag[len('policy_'):tag.rindex('_N')]
        n = int(tag[tag.rindex('_N') + 2:])
        r = json.load(open(fn))
        for e in r['rungs'].values():
            if e.get('popp_fail'):
                pfail.setdefault((arm, n), []).append(e['popp_fail']['mean'])
    for sf in glob.glob('cache/model_error/steady/policy_steady*.json'):
        for e in json.load(open(sf)):
            if not e.get('solved'):
                continue
            parts = e['dir'].split(os.sep)
            arm, nd = parts[-2], parts[-1]
            scong.setdefault((arm, int(nd[1:])), []).append(
                e['steady_congested_frac'])
            if e['clean_avg_lat'] is not None:
                clean.setdefault((arm, int(nd[1:])), []).append(
                    e['clean_avg_lat'])
    return scong, clean, pfail


def main():
    scong, clean, pfail = load()
    opp = json.load(open('cache/model_error/opp_ref_georand.json'))
    opp_lat = _STAT([v['avg_lat'] for v in opp.values() if v])
    opp_pf = _STAT([v['pfail_mean'] for v in opp.values() if v])
    # painter refs from the mesh rerank/steady sidecars
    p_pf = [e['popp_fail']['mean']
            for fn in glob.glob('cache/model_error/rerank/mesh*_N1/seed_*.json')
            for rung, e in json.load(open(fn))['rungs'].items()
            if rung == 'painter' and e.get('popp_fail')]
    p_cl, p_sc = [], []
    for sf in ('cache/model_error/steady/mesh_steady.json',
               'cache/model_error/steady/mesh_v2_steady.json'):
        if not os.path.exists(sf):
            continue
        for e in json.load(open(sf)):
            if e.get('solved') and e.get('rung') == 'painter':
                p_sc.append(e['steady_congested_frac'])
                if e['clean_avg_lat'] is not None:
                    p_cl.append(e['clean_avg_lat'])
    refs = {
        'scong': (0.0, _STAT(p_sc) if p_sc else np.nan),
        'clean': (opp_lat, _STAT(p_cl) if p_cl else np.nan),
        'pfail': (opp_pf, _STAT(p_pf) if p_pf else np.nan),
    }
    panels = ((scong, 'median frac. congested/stranded, STEADY', 'scong'),
              (clean, 'median steady latency of ROUTED traffic (ms)', 'clean'),
              (pfail, 'median frac. congested under popp FAILURES', 'pfail'))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (data, ylab, rk) in zip(axes, panels):
        for arm, label, color in ARMS:
            ys = [_STAT(data[(arm, n)]) if (arm, n) in data else np.nan
                  for n in NS]
            ax.plot(NS, ys, 'o-', color=color, label=label, lw=1.8, ms=4.5)
        ro, rp = refs[rk]
        ax.axhline(ro, color='0.35', ls='--', lw=1.1)
        ax.axhline(rp, color='0.35', ls=':', lw=1.1)
        ax.text(55, ro, 'opp', va='center', fontsize=8, color='0.35')
        ax.text(55, rp, 'painter', va='center', fontsize=8, color='0.35')
        ax.set_xscale('log')
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        ax.set_xlabel('measurement budget N')
        ax.set_ylabel(ylab.replace('median', 'mean') if _STAT is st.mean else ylab)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Policy ladder over budget N (georand, 100 iters max, '
                 'exit-on-budget, cache off, MC=1; ' + ('MEANS' if _STAT is st.mean else 'medians') + ' of 5 deployments; '
                 'dashed=one-per-peering, dotted=painter)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs('figures', exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig('figures/{}.{}'.format(_OUT, ext), dpi=170)
    print('wrote figures/{}.png'.format(_OUT))


if __name__ == '__main__':
    main()
