"""Normalized ablation ladder: per seed, PAINTER = 0%, one_per_peering
(optimal) = 100%; each rung shows the fraction of the painter->optimal
gap it closes on the combined objective:

    pct(rung) = 100 * (1 - combined_diff(rung) / combined_diff(painter))

where combined_diff = (steady - opp_steady) + gamma * (fail_abs - opp_fail_abs)
(opp's own combined_diff is 0 by construction). Rungs worse than painter go
negative; nothing exceeds 100 unless it beats optimal.

    python -m experiments.ablation.plot_normalized \
        --in-dir cache/ablation/fork_full_res --gamma 4 \
        [--in-dir2 cache/ablation/fork_full --gamma2 0]
"""
import argparse
import glob
import json
import os

import numpy as np

LADDER = [
    ('no_memory',    'monte-carlo',                          'tab:orange'),
    ('no_direction', '+ memory (continuous adv)',            'tab:olive'),
    ('expl_none',    '+ direction (full-vector step)',       'tab:blue'),
    ('expl_random',  '+ random exploration',                 'tab:cyan'),
    ('full',         '+ entropic exploration (= SCULPTOR)',  'tab:green'),
]


def combined_diffs(in_dir, gamma):
    """{rung: {seed: combined_diff_vs_opp}} using only trusted rescores."""
    out = {}
    for fn in glob.glob(os.path.join(in_dir, 'seed_*_*.json')):
        with open(fn) as f:
            r = json.load(f)
        if not r.get('rescored'):
            continue
        d = r['diff_vs_opp']
        if gamma and r.get('fail_eval') == 'lp_driver_v2':
            fp = r['fail_popp']
            d = d + gamma * (fp['avg_lat_under_failure_abs'] - fp['opp_avg_lat_under_failure_abs'])
        elif gamma:
            continue  # gamma requested but failure terms not scored yet
        out.setdefault(r['rung'], {})[r['seed']] = d
    return out


def normalized(diffs):
    """{rung: [pct per seed]} for seeds where painter is available."""
    pain = diffs.get('painter', {})
    out = {}
    for rung, per_seed in diffs.items():
        if rung == 'painter':
            continue
        vals = []
        for seed, d in per_seed.items():
            p = pain.get(seed)
            if p is None or p <= 0:
                continue
            vals.append(100.0 * (1.0 - d / p))
        if vals:
            out[rung] = vals
    return out


def draw(ax, norm, title):
    for key, label, color in LADDER:
        if key not in norm:
            continue
        xs = np.sort(np.asarray(norm[key]))
        ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where='post',
                label='{} (n={})'.format(label, len(xs)), color=color)
    ax.axvline(0, color='tab:red', lw=1.2, ls=':')
    ax.axvline(100, color='black', lw=1.2, ls=':')
    ax.text(0, 1.02, 'PAINTER', color='tab:red', fontsize=8, ha='center')
    ax.text(100, 1.02, 'optimal', color='black', fontsize=8, ha='center')
    ax.set_xlabel('% of painter$\\to$optimal gap closed')
    ax.set_ylabel('CDF over deployments')
    ax.set_title(title)
    ax.set_xlim(0, 110)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc='upper left')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', required=True)
    p.add_argument('--gamma', type=float, default=4.0)
    p.add_argument('--in-dir2', default=None)
    p.add_argument('--gamma2', type=float, default=0.0)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_panels = 2 if args.in_dir2 else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.2))
    axes = np.atleast_1d(axes)

    norm1 = normalized(combined_diffs(args.in_dir, args.gamma))
    draw(axes[0], norm1, 'Resilience-trained, combined obj (gamma={})'.format(args.gamma))
    print('\n== {} (gamma={}) =='.format(args.in_dir, args.gamma))
    for key, label, _ in LADDER:
        if key in norm1:
            v = np.asarray(norm1[key])
            print('{:38s} n={:2d} median {:6.1f}%  mean {:6.1f}%'.format(
                label, len(v), float(np.median(v)), float(v.mean())))

    if args.in_dir2:
        norm2 = normalized(combined_diffs(args.in_dir2, args.gamma2))
        draw(axes[1], norm2, 'Latency-only trained, steady obj (gamma={})'.format(args.gamma2))
        print('\n== {} (gamma={}) =='.format(args.in_dir2, args.gamma2))
        for key, label, _ in LADDER:
            if key in norm2:
                v = np.asarray(norm2[key])
                print('{:38s} n={:2d} median {:6.1f}%  mean {:6.1f}%'.format(
                    label, len(v), float(np.median(v)), float(v.mean())))

    out = args.out or os.path.join(args.in_dir, 'fork_ladder_normalized.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace('.png', '.pdf'))
    print('\nwrote {} (+.pdf)'.format(out))


if __name__ == '__main__':
    main()
