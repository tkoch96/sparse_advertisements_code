"""Aggregate + plot the ablation results.

Three panels:
  A. CDF (over seeds) of final objective minus one_per_peering, per arm.
  B. Paired per-ladder-step improvement CDFs (same-seed differences --
     the RNG-robust signal, since all arms share each seed's deployment).
  C. Median +/- IQR convergence trace (objective - opp) vs iteration.

    python -m experiments.ablation.plot_ablation --in-dir cache/ablation/full
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

ARM_LABELS = {
    'painter': 'PAINTER (greedy, avg-of-options)',
    'greedy_mc': '+ (d) monte-carlo model',
    'coord_mc': '+ (b) memory (continuous adv)',
    'fullgrad': '+ (c) direction (full grad + momentum)',
    'fullgrad_entropy': '+ (a) entropic exploration',
    'sparse_ref': 'SCULPTOR (repo sparse, reference)',
    'sparse_ref_earlystop': 'SCULPTOR (repo defaults, early stop)',
}
ARM_COLORS = {
    'painter': 'tab:red',
    'greedy_mc': 'tab:orange',
    'coord_mc': 'tab:olive',
    'fullgrad': 'tab:blue',
    'fullgrad_entropy': 'tab:green',
    'sparse_ref': 'black',
    'sparse_ref_earlystop': 'gray',
}
LADDER = ['painter', 'greedy_mc', 'coord_mc', 'fullgrad', 'fullgrad_entropy']
STEP_LABELS = {
    ('painter', 'greedy_mc'): '(d) monte-carlo',
    ('greedy_mc', 'coord_mc'): '(b) memory',
    ('coord_mc', 'fullgrad'): '(c) direction',
    ('fullgrad', 'fullgrad_entropy'): '(a) entropy',
    ('fullgrad_entropy', 'sparse_ref'): 'ladder top vs repo SCULPTOR',
}


def load(in_dir):
    """Returns {seed: {arm: result_dict}}. Accepts both whole-seed files
    (seed_N.json, run_ablation.py) and per-arm files
    (seed_N_<arm>.json, run_ablation2.py), merging the latter."""
    out = {}
    for fn in sorted(glob.glob(os.path.join(in_dir, 'seed_*.json'))):
        with open(fn) as f:
            r = json.load(f)
        s = r['seed']
        if s not in out:
            out[s] = r
        else:
            out[s]['arms'].update(r.get('arms', {}))
    return out


def _cdf(ax, vals, **kw):
    xs = np.sort(np.asarray(vals))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.step(xs, ys, where='post', **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', required=True)
    p.add_argument('--out', default=None, help='Figure path (default <in-dir>/ablation_cdf.png)')
    args = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results = load(args.in_dir)
    if not results:
        raise SystemExit('no seed_*.json in {}'.format(args.in_dir))

    arms_present = [a for a in ARM_LABELS if any(a in r['arms'] for r in results.values())]
    diffs = {a: {s: r['arms'][a]['diff_vs_opp'] for s, r in results.items() if a in r['arms']}
             for a in arms_present}

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 4.8))

    # ---------------- Panel A: final-objective CDFs ----------------
    print('\n=== Final objective - one_per_peering (ms), over {} seeds ==='.format(len(results)))
    print('{:40s} {:>4s} {:>8s} {:>8s} {:>8s}'.format('arm', 'n', 'mean', 'median', 'p90'))
    for a in arms_present:
        vals = np.array(list(diffs[a].values()))
        print('{:40s} {:4d} {:8.2f} {:8.2f} {:8.2f}'.format(
            ARM_LABELS[a], len(vals), vals.mean(), float(np.median(vals)),
            float(np.percentile(vals, 90))))
        _cdf(axA, vals, label='{} (n={})'.format(ARM_LABELS[a], len(vals)),
             color=ARM_COLORS[a])
    axA.set_xlabel('Final avg latency $-$ one_per_peering (ms)')
    axA.set_ylabel('CDF over deployments')
    axA.set_title('A. Final quality per arm')
    axA.grid(alpha=.3)
    axA.legend(fontsize=7, loc='lower right')

    # ---------------- Panel B: paired per-step improvements ----------------
    print('\n=== Paired per-step improvement (ms, positive = feature helped) ===')
    print('{:34s} {:>4s} {:>8s} {:>8s} {:>9s}'.format('step', 'n', 'mean', 'median', 'win rate'))
    steps = list(zip(LADDER[:-1], LADDER[1:]))
    if 'sparse_ref' in arms_present:
        steps.append(('fullgrad_entropy', 'sparse_ref'))
    for prev, nxt in steps:
        if prev not in diffs or nxt not in diffs:
            continue
        shared = sorted(set(diffs[prev]) & set(diffs[nxt]))
        if not shared:
            continue
        imp = np.array([diffs[prev][s] - diffs[nxt][s] for s in shared])
        lbl = STEP_LABELS[(prev, nxt)]
        win = float((imp > 0).mean())
        print('{:34s} {:4d} {:8.2f} {:8.2f} {:8.0%}'.format(lbl, len(imp), imp.mean(),
                                                            float(np.median(imp)), win))
        _cdf(axB, imp, label='{} (win {:.0%})'.format(lbl, win), color=ARM_COLORS[nxt])
    axB.axvline(0, color='gray', lw=1, ls='--')
    axB.set_xlabel('Same-seed improvement from adding feature (ms)')
    axB.set_ylabel('CDF over deployments')
    axB.set_title('B. Marginal value of each feature (paired)')
    axB.grid(alpha=.3)
    axB.legend(fontsize=7, loc='lower right')

    # ---------------- Panel C: convergence traces ----------------
    for a in LADDER:
        if a not in diffs:
            continue
        traces = []
        for s, r in results.items():
            if a not in r['arms'] or 'trace' not in r['arms'][a]:
                continue
            tr = np.asarray(r['arms'][a]['trace'], dtype=float) - r['opp_obj']
            traces.append(tr)
        if not traces:
            continue
        L = max(len(t) for t in traces)
        # pad early-stopped runs with their final value
        M = np.vstack([np.concatenate([t, np.full(L - len(t), t[-1])]) for t in traces])
        med = np.median(M, axis=0)
        q1, q3 = np.percentile(M, [25, 75], axis=0)
        x = np.arange(L)
        axC.plot(x, med, color=ARM_COLORS[a], label=ARM_LABELS[a], lw=1.5)
        axC.fill_between(x, q1, q3, color=ARM_COLORS[a], alpha=.15, lw=0)
    axC.set_xlabel('Iteration')
    axC.set_ylabel('Objective $-$ one_per_peering (ms)')
    axC.set_title('C. Convergence (median, IQR band)')
    axC.grid(alpha=.3)
    axC.legend(fontsize=7, loc='upper right')

    out = args.out or os.path.join(args.in_dir, 'ablation_cdf.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace('.png', '.pdf'))
    print('\nwrote {} (+.pdf)'.format(out))


if __name__ == '__main__':
    main()
