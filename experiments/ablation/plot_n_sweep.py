"""Effect of the measurement budget N on the ablation ladder.

Reads the N-sweep dirs (OUT_ROOT/N<k>/seed_*_*.json, trusted rescores only)
plus the fixed-mode study as the N=infinity anchor, and produces:
  * table: median (and per-seed spread) combined objective per (rung, N)
  * table: mean probes actually spent per (rung, N)
  * figure: median combined vs N per rung (log-x), plus the headline
    gap curve (no_memory - full) vs N  [hypothesis: big gap at small N,
    collapsing as N grows]

    python -m experiments.ablation.plot_n_sweep \
        --n-root cache/ablation/nsweep --anchor cache/ablation/fork_small_20x200_v3 \
        --gamma 0.1 --out figures/nsweep_effect.pdf
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

RUNGS = ['painter', 'no_mc', 'no_memory', 'no_direction', 'expl_none', 'expl_random', 'full']
COLORS = {'painter': 'tab:red', 'no_mc': 'tab:brown', 'no_memory': 'tab:orange',
          'no_direction': 'tab:olive', 'expl_none': 'tab:blue',
          'expl_random': 'tab:cyan', 'full': 'tab:green'}


def load_dir(d, gamma):
    """{rung: {seed: (combined, probes_spent)}} from trusted rescores."""
    out = {}
    for fn in glob.glob(os.path.join(d, 'seed_*_*.json')):
        with open(fn) as f:
            r = json.load(f)
        if not r.get('rescored'):
            continue
        fp = r['fail_popp']
        comb = r['diff_vs_opp'] + gamma * (
            fp['avg_lat_under_failure_abs'] - fp['opp_avg_lat_under_failure_abs'])
        out.setdefault(r['rung'], {})[r['seed']] = (comb, r.get('probes_spent'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-root', required=True)
    ap.add_argument('--anchor', default=None, help='fixed-mode dir = N=inf anchor')
    ap.add_argument('--gamma', type=float, default=0.1)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    n_dirs = {}
    for d in sorted(glob.glob(os.path.join(args.n_root, 'N*'))):
        m = re.match(r'N(\d+)$', os.path.basename(d))
        if m:
            n_dirs[int(m.group(1))] = load_dir(d, args.gamma)
    assert n_dirs, 'no N dirs found under {}'.format(args.n_root)
    ns = sorted(n_dirs)
    anchor = load_dir(args.anchor, args.gamma) if args.anchor else None

    # ---- tables ----------------------------------------------------------
    print('\nmedian combined (gamma={}) per rung x N   [anchor = fixed mode]'.format(args.gamma))
    hdr = '{:<14}'.format('rung') + ''.join('{:>10}'.format('N=' + str(n)) for n in ns) + \
          ('{:>10}'.format('fixed') if anchor else '')
    print(hdr); print('-' * len(hdr))
    med = {}
    for rung in RUNGS:
        cells = ''
        for n in ns:
            vals = [v[0] for v in n_dirs[n].get(rung, {}).values()]
            med[rung, n] = float(np.median(vals)) if vals else None
            cells += '{:>10}'.format('{:.2f}'.format(med[rung, n]) if vals else '—')
        if anchor:
            av = [v[0] for v in anchor.get(rung, {}).values()]
            cells += '{:>10}'.format('{:.2f}'.format(np.median(av)) if av else '—')
        print('{:<14}'.format(rung) + cells)

    print('\nmean probes actually spent (of budget N):')
    print(hdr); print('-' * len(hdr))
    for rung in RUNGS:
        cells = ''
        for n in ns:
            vals = [v[1] for v in n_dirs[n].get(rung, {}).values() if v[1] is not None]
            cells += '{:>10}'.format('{:.1f}'.format(np.mean(vals)) if vals else '—')
        print('{:<14}'.format(rung) + cells + ('{:>10}'.format('n/a') if anchor else ''))

    gap = [(n, med.get(('no_memory', n)), med.get(('full', n))) for n in ns]
    print('\nheadline gap (no_memory - full) vs N:')
    for n, a, b in gap:
        if a is not None and b is not None:
            print('  N={:<4} no_memory={:>8.2f}  full={:>8.2f}  gap={:>8.2f}'.format(n, a, b, a - b))

    # ---- figure ----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for rung in RUNGS:
        ys = [med.get((rung, n)) for n in ns]
        if all(y is None for y in ys):
            continue
        ax1.plot(ns, ys, marker='o', label=rung, color=COLORS.get(rung))
        if anchor:
            av = [v[0] for v in anchor.get(rung, {}).values()]
            if av:
                ax1.axhline(np.median(av), color=COLORS.get(rung), ls=':', lw=.8, alpha=.6)
    ax1.set_xscale('log'); ax1.set_yscale('symlog', linthresh=1.0)
    ax1.set_xticks(ns); ax1.set_xticklabels([str(n) for n in ns])
    ax1.set_xlabel('measurement budget N'); ax1.set_ylabel('median combined vs OPP (ms)')
    ax1.legend(fontsize=7); ax1.grid(alpha=.3)
    ax1.set_title('ladder vs N (dotted = fixed-mode anchor)')

    gx = [n for n, a, b in gap if a is not None and b is not None]
    gy = [a - b for n, a, b in gap if a is not None and b is not None]
    ax2.plot(gx, gy, marker='s', color='k')
    ax2.axhline(0, color='k', lw=.6, ls=':')
    ax2.set_xscale('log'); ax2.set_xticks(gx); ax2.set_xticklabels([str(n) for n in gx])
    ax2.set_xlabel('measurement budget N'); ax2.set_ylabel('no_memory - full (ms, median)')
    ax2.set_title('hypothesis curve: gap vs N')
    ax2.grid(alpha=.3)

    out = args.out or os.path.join(args.n_root, 'nsweep_effect.pdf')
    fig.tight_layout(); fig.savefig(out, bbox_inches='tight')
    print('\nwrote', out)


if __name__ == '__main__':
    main()
