"""Five-panel policy-ladder figure over N (Tom, 2026-08-13): the three
plot_policy panels (steady congested frac, clean routed latency,
popp-failure congestion) plus popp-failure AFFECTED-user routed latency
and pop-failure congestion, one line per ladder arm.

Data: cache/model_error/steady/policy_steady*.json (steady_metrics),
cache/model_error/failure/policy_failure*.json (failure_metrics);
opp refs from cache/model_error/opp_ref_georand.json; painter refs from
the mesh sidecars (panels 1-3 only; no painter failure_metrics run yet).
Env: POLICY_PLOT_STAT=mean|median (default median), POLICY_PLOT_OUT.
Output: figures/<out>.{png,pdf}
"""
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_STAT = st.mean if os.environ.get('POLICY_PLOT_STAT', 'median') == 'mean' \
    else st.median
_OUT = os.environ.get('POLICY_PLOT_OUT', 'policy_ladder_overN_5panel')

NS = [1, 2, 5, 10, 20, 50]
ARMS = [('L1_nomc_fixed', 'L1 no_mc+fixed', '#2a78d6'),
        ('L2_nomc_sched', 'L2 no_mc+sched', '#eb6834'),
        ('L3_nodir_sched', 'L3 no_dir+sched', '#1baf7a'),
        ('L4_nomem_sched', 'L4 no_mem+sched', '#eda100'),
        ('L5_nomem_smart', 'L5 no_mem+smart', '#e87ba4'),
        ('L6_full_smart', 'L6 full+smart', '#4a3aa7')]


def _key(dirpath):
    parts = dirpath.split(os.sep)
    return parts[-2], int(parts[-1][1:])


def load():
    data = {k: {} for k in
            ('scong', 'clean', 'popp_cong', 'popp_afflat', 'pop_cong')}
    for sf in glob.glob('cache/model_error/steady/policy_steady*.json'):
        for e in json.load(open(sf)):
            if not e.get('solved'):
                continue
            k = _key(e['dir'])
            data['scong'].setdefault(k, []).append(
                e['steady_congested_frac'])
            if e['clean_avg_lat'] is not None:
                data['clean'].setdefault(k, []).append(e['clean_avg_lat'])
    for ff in glob.glob('cache/model_error/failure/policy_failure*.json'):
        for e in json.load(open(ff)):
            if not e.get('solved'):
                continue
            k = _key(e['dir'])
            data['popp_cong'].setdefault(k, []).append(
                e['popp']['cong_mean'])
            if e['popp']['affected_routed_lat_mean'] is not None:
                data['popp_afflat'].setdefault(k, []).append(
                    e['popp']['affected_routed_lat_mean'])
            data['pop_cong'].setdefault(k, []).append(e['pop']['cong_mean'])
    return data


def main():
    data = load()
    opp = json.load(open('cache/model_error/opp_ref_georand.json'))
    opp_lat = _STAT([v['avg_lat'] for v in opp.values() if v])
    opp_pf = _STAT([v['pfail_mean'] for v in opp.values() if v])
    p_pf = [e['popp_fail']['mean']
            for fn in glob.glob(
                'cache/model_error/rerank/mesh*_N1/seed_*.json')
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

    sname = 'mean' if _STAT is st.mean else 'median'
    panels = (
        ('scong', '{} frac congested/stranded, STEADY'.format(sname),
         0.0, _STAT(p_sc) if p_sc else np.nan),
        ('clean', '{} steady latency of ROUTED traffic (ms)'.format(sname),
         opp_lat, _STAT(p_cl) if p_cl else np.nan),
        ('popp_cong', '{} frac congested, popp FAILURES'.format(sname),
         opp_pf, _STAT(p_pf) if p_pf else np.nan),
        ('popp_afflat',
         '{} routed latency of AFFECTED users, popp FAILURES (ms)'.format(
             sname), None, None),
        ('pop_cong', '{} frac congested, pop FAILURES'.format(sname),
         None, None),
    )
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.6))
    for ax, (dk, ylab, ref_opp, ref_painter) in zip(axes, panels):
        d = data[dk]
        for arm, label, color in ARMS:
            ys = [_STAT(d[(arm, n)]) if (arm, n) in d else np.nan
                  for n in NS]
            ax.plot(NS, ys, 'o-', color=color, label=label, lw=1.8, ms=4.5)
        if ref_opp is not None and np.isfinite(ref_opp):
            ax.axhline(ref_opp, color='0.35', ls='--', lw=1.1)
            ax.text(55, ref_opp, 'opp', va='center', fontsize=8,
                    color='0.35')
        if ref_painter is not None and np.isfinite(ref_painter):
            ax.axhline(ref_painter, color='0.35', ls=':', lw=1.1)
            ax.text(55, ref_painter, 'painter', va='center', fontsize=8,
                    color='0.35')
        ax.set_xscale('log')
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        ax.set_xlabel('measurement budget N')
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        'Policy ladder over budget N — 5 metrics (georand, 100 iters max, '
        'exit-on-budget, cache off, MC=1, congestion-aware objective; '
        '{}s of 5 deployments; dashed=one-per-peering, dotted=painter)'
        .format(sname.upper()), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs('figures', exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig('figures/{}.{}'.format(_OUT, ext), dpi=170)
    print('wrote figures/{}.png'.format(_OUT))


if __name__ == '__main__':
    main()
