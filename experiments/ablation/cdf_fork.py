"""CDF views of the fork-ladder benefit between painter and OPP.

Two figures + a quantile table from trusted rescore JSONs:

  1. Across-seed CDFs of the combined objective per rung:
       absolute panel : x = combined diff vs OPP (ms; OPP = 0, lower better)
       percentage panel: x = % of the painter->OPP gap closed on the same
                         seed (painter = 0%, OPP = 100%; plot_normalized
                         convention: pct = 100*(1 - d_rung/d_painter))
  2. If the JSONs carry per-scenario data (rescore_fork run with
     SCULPTOR_RESCORE_STORE_SCENARIOS=1), across-scenario CDFs of latency
     under single-popp failure: absolute (rung - OPP, ms per scenario) and
     percentage of the painter-OPP per-scenario gap closed.

    python -m experiments.ablation.cdf_fork --in-dir cache/ablation/fork_5x200 \
        --gamma 0.1 [--out figures/dashboards/misc/fork_5x200_cdf.pdf]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from experiments.ablation.plot_normalized import combined_diffs  # noqa: E402

LADDER = [
    ('painter',      'painter',                              'tab:red'),
    ('no_mc',        'painter + one-flip search',            'tab:brown'),
    ('no_memory',    'monte-carlo',                          'tab:orange'),
    ('no_direction', '+ memory (continuous adv)',            'tab:olive'),
    ('expl_none',    '+ direction (full-vector step)',       'tab:blue'),
    ('expl_random',  '+ random exploration',                 'tab:cyan'),
    ('full',         '+ entropic exploration (= SCULPTOR)',  'tab:green'),
]


def _cdf_xy(vals):
    v = np.sort(np.asarray(vals, dtype=float))
    y = np.arange(1, len(v) + 1) / len(v)
    return v, y


def seed_cdfs(diffs):
    """(abs, pct) per rung: abs = [combined diff per seed],
    pct = [% painter->OPP gap closed per seed] (seeds with painter>0)."""
    pain = diffs.get('painter', {})
    out_abs, out_pct = {}, {}
    for rung, per_seed in diffs.items():
        out_abs[rung] = [per_seed[s] for s in sorted(per_seed)]
        pcts = []
        for s in sorted(per_seed):
            p = pain.get(s)
            if rung == 'painter' or p is None or p <= 0:
                continue
            pcts.append(100.0 * (1.0 - per_seed[s] / p))
        if pcts:
            out_pct[rung] = pcts
    return out_abs, out_pct


def scenario_arrays(in_dir):
    """{rung: {seed: {'lats': [per-scenario abs lat under failure],
                      'opp_lats': [...]}}} when stored; {} otherwise."""
    out = {}
    for fn in glob.glob(os.path.join(in_dir, 'seed_*_*.json')):
        with open(fn) as f:
            r = json.load(f)
        if not r.get('rescored'):
            continue
        per_s = (r.get('fail_popp') or {}).get('per_scenario_lats')
        opp_per_s = (r.get('fail_popp') or {}).get('opp_per_scenario_lats')
        if per_s and opp_per_s:
            out.setdefault(r['rung'], {})[r['seed']] = {
                'lats': per_s, 'opp_lats': opp_per_s}
    return out


def quantile_table(vals_by_rung, unit, fmt='{:>10.2f}'):
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    hdr = '{:<14}'.format('rung') + ''.join('{:>10}'.format('p{:g}'.format(q * 100)) for q in qs) + '{:>6}'.format('n')
    lines = ['CDF quantiles ({}):'.format(unit), hdr, '-' * len(hdr)]
    for rung, _, _ in LADDER:
        if rung not in vals_by_rung:
            continue
        v = np.asarray(vals_by_rung[rung], dtype=float)
        lines.append('{:<14}'.format(rung) +
                     ''.join(fmt.format(np.quantile(v, q)) for q in qs) +
                     '{:>6d}'.format(len(v)))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--gamma', type=float, default=0.1)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    diffs = combined_diffs(args.in_dir, args.gamma)
    if not diffs:
        print('no trusted rescores in {}'.format(args.in_dir)); return
    abs_by_rung, pct_by_rung = seed_cdfs(diffs)
    scen = scenario_arrays(args.in_dir)

    n_rows = 2 if scen else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 4.2 * n_rows), squeeze=False)

    ax = axes[0][0]
    for rung, label, color in LADDER:
        if rung not in abs_by_rung:
            continue
        x, y = _cdf_xy(abs_by_rung[rung])
        ax.step(x, y, where='post', label=label, color=color, marker='o', ms=3)
    ax.set_xscale('symlog', linthresh=1.0)
    ax.axvline(0, color='k', lw=.6, ls=':')
    ax.set_xlabel('combined objective vs OPP (ms, gamma={:g}; OPP=0)'.format(args.gamma))
    ax.set_ylabel('CDF over seeds')
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)

    ax = axes[0][1]
    for rung, label, color in LADDER:
        if rung not in pct_by_rung:
            continue
        x, y = _cdf_xy(pct_by_rung[rung])
        ax.step(x, y, where='post', label=label, color=color, marker='o', ms=3)
    for v, nm in ((0, 'painter'), (100, 'OPP')):
        ax.axvline(v, color='k', lw=.6, ls=':')
        ax.text(v, 1.02, nm, ha='center', fontsize=7)
    ax.set_xscale('symlog', linthresh=100.0)
    ax.set_xlabel('% of painter->OPP gap closed (combined)')
    ax.set_ylabel('CDF over seeds')
    ax.grid(alpha=.3)

    print(quantile_table(abs_by_rung, 'absolute: combined ms vs OPP'))
    print()
    if pct_by_rung:
        print(quantile_table(pct_by_rung, '% painter->OPP gap closed', fmt='{:>10.1f}'))

    if scen:
        # pool scenarios across seeds; per-scenario benefit vs OPP + % of the
        # painter-OPP per-scenario gap closed (scenario-matched within seed)
        ax = axes[1][0]
        pooled_abs = {}
        for rung, label, color in LADDER:
            if rung not in scen:
                continue
            allv = []
            for s, d in scen[rung].items():
                allv.extend(np.asarray(d['lats']) - np.asarray(d['opp_lats']))
            pooled_abs[rung] = allv
            x, y = _cdf_xy(allv)
            ax.step(x, y, where='post', label=label, color=color)
        ax.set_xscale('symlog', linthresh=1.0)
        ax.axvline(0, color='k', lw=.6, ls=':')
        ax.set_xlabel('per-failure-scenario latency vs OPP (ms)')
        ax.set_ylabel('CDF over scenarios (all seeds)')
        ax.legend(fontsize=7)
        ax.grid(alpha=.3)

        ax = axes[1][1]
        pain = scen.get('painter', {})
        for rung, label, color in LADDER:
            if rung == 'painter' or rung not in scen:
                continue
            pooled = []
            for s, d in scen[rung].items():
                if s not in pain:
                    continue
                r = np.asarray(d['lats']) - np.asarray(d['opp_lats'])
                p = np.asarray(pain[s]['lats']) - np.asarray(pain[s]['opp_lats'])
                ok = p > 0
                pooled.extend(100.0 * (1.0 - r[ok] / p[ok]))
            if pooled:
                x, y = _cdf_xy(pooled)
                ax.step(x, y, where='post', label=label, color=color)
        ax.set_xscale('symlog', linthresh=100.0)
        for v in (0, 100):
            ax.axvline(v, color='k', lw=.6, ls=':')
        ax.set_xlabel('% of painter->OPP per-scenario gap closed')
        ax.set_ylabel('CDF over scenarios (all seeds)')
        ax.grid(alpha=.3)
        print('\n(scenario-level CDFs included: per-scenario data present)')
    else:
        print('\n(no per-scenario arrays in JSONs -> seed-level CDFs only; '
              'rerun rescore with SCULPTOR_RESCORE_STORE_SCENARIOS=1 to add them)')

    out = args.out or os.path.join(args.in_dir, 'cdf_fork_gamma{:g}.pdf'.format(args.gamma))
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    print('\nwrote {}'.format(out))


if __name__ == '__main__':
    main()
