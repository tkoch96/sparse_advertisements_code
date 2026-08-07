"""Plot the fork-based ablation ladder, framed BOTTOM-UP: start from
PAINTER, build SCULPTOR by adding features one at a time.

Headline metric (panels A/B) is THE pipeline objective:
    overall = avg_lat_steady + gamma * avg_lat_under_single_popp_failures
computed for each arm and for one_per_peering (diff shown). Components
shown in panels C (steady) and D (failure term). gamma is configurable
(--gamma, default 4 = wrapper_eval's) so resilience-on and resilience-off
ladders can be ranked under the SAME objective.

    python -m experiments.ablation.plot_fork_ladder --in-dir cache/ablation/fork_full_res
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

LADDER = [
    ('painter',      'PAINTER (repo)',                       'tab:red'),
    ('no_memory',    'monte-carlo',                          'tab:orange'),
    ('no_direction', '+ memory (continuous adv)',            'tab:olive'),
    ('expl_none',    '+ direction (full-vector step)',       'tab:blue'),
    ('expl_random',  '+ random exploration',                 'tab:cyan'),
    ('full',         '+ entropic exploration (= SCULPTOR)',  'tab:green'),
]


def load(in_dir):
    out = {}
    for fn in sorted(glob.glob(os.path.join(in_dir, 'seed_*_*.json'))):
        with open(fn) as f:
            r = json.load(f)
        out.setdefault(r['rung'], {})[r['seed']] = r
    return out


def _cdf(ax, vals, **kw):
    xs = np.sort(np.asarray(vals))
    ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where='post', **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--gamma', type=float, default=4.0,
                   help='Weight on the failure term in the combined objective.')
    p.add_argument('--clip-ms', type=float, default=100.0)
    args = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = load(args.in_dir)
    if not data:
        raise SystemExit('no results in {}'.format(args.in_dir))
    have_fail = any(str(r.get('fail_eval', '')).startswith('lp_')
                    for d in data.values() for r in d.values())

    if have_fail:
        fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(13.5, 9.5))
    else:
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
        axC = axD = None

    hdr = 'combined objective (gamma={})'.format(args.gamma) if have_fail else 'steady avg latency'
    print('\nDiff vs one_per_peering on {} (ms):'.format(hdr))
    print('{:38s} {:>3s} {:>9s} {:>9s} {:>9s}'.format('rung', 'n', 'mean', 'median', 'p90'))
    for pos, (key, label, color) in enumerate(LADDER):
        if key not in data:
            continue
        rows = list(data[key].values())
        steady = np.array([r['diff_vs_opp'] for r in rows])
        if have_fail:
            fail_ours = np.array([r['fail_popp']['avg_lat_under_failure_abs'] for r in rows])
            fail_opp = np.array([r['fail_popp']['opp_avg_lat_under_failure_abs'] for r in rows])
            overall = steady + args.gamma * (fail_ours - fail_opp)
        else:
            overall = steady
        shown = np.clip(overall, None, args.clip_ms)
        nblow = int((overall > args.clip_ms).sum())
        print('{:38s} {:3d} {:9.2f} {:9.2f} {:9.2f}{}'.format(
            label, len(overall), overall.mean(), float(np.median(overall)),
            float(np.percentile(overall, 90)),
            '   [{} clipped]'.format(nblow) if nblow else ''))
        y = np.full(len(shown), pos, dtype=float)
        axA.scatter(shown, y, color=color, s=45, zorder=3)
        axA.scatter([np.median(shown)], [pos], color=color, s=220, marker='|',
                    linewidths=3, zorder=4)
        _cdf(axB, shown, label=label, color=color)
        if axC is not None:
            _cdf(axC, np.clip(steady, None, args.clip_ms), label=label, color=color)
            _cdf(axD, np.clip(fail_ours - fail_opp, None, args.clip_ms),
                 label=label, color=color)

    for ax in [a for a in (axA, axB, axC, axD) if a is not None]:
        ax.axvline(0, color='black', lw=1.2, ls=':', zorder=2)
    axB.plot([], [], color='black', lw=1.2, ls=':', label='one_per_peering (optimal) = 0')
    axA.set_yticks(range(len(LADDER)))
    axA.set_yticklabels([l for _, l, _ in LADDER], fontsize=8)
    axA.invert_yaxis()
    axA.set_xlabel('Overall objective $-$ one_per_peering (ms)')
    axA.set_title('A. Building SCULPTOR from PAINTER ({})'.format(hdr))
    axA.grid(alpha=.3, axis='x')
    axB.set_xlabel('Overall objective $-$ one_per_peering (ms)')
    axB.set_ylabel('CDF over deployments')
    axB.set_title('B. CDF, combined objective')
    axB.grid(alpha=.3)
    axB.legend(fontsize=7, loc='lower right')
    if axC is not None:
        axC.set_xlabel('Steady avg latency $-$ opp (ms)')
        axC.set_ylabel('CDF over deployments')
        axC.set_title('C. Component: steady-state')
        axC.grid(alpha=.3)
        axC.legend(fontsize=7, loc='lower right')
        axD.set_xlabel('Avg latency under popp failures $-$ opp same (ms)')
        axD.set_ylabel('CDF over deployments')
        axD.set_title('D. Component: failure term (LP re-assignment per failure)')
        axD.grid(alpha=.3)
        axD.legend(fontsize=7, loc='lower right')

    out = args.out or os.path.join(args.in_dir, 'fork_ladder.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace('.png', '.pdf'))
    print('\nwrote {} (+.pdf)'.format(out))


if __name__ == '__main__':
    main()
