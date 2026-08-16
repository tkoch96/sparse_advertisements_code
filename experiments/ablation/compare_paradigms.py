"""Overlay comparison for the gradient-step-paradigm A/B: GT objective per
iteration + realized flips per iteration + final stats, one curve per step
paradigm, one column per seed. Every arm shares the deployment and the
canonical init for its seed, so within a column the ONLY difference is the
step policy (SCULPTOR_ABLATION_GRAD_SCALE / _ALPHA).

Parametric over arms and seeds -- the arm set is a CLI/JSON input, not a
constant, so new alpha grids reuse this plotter instead of forking it:

    python -m experiments.ablation.compare_paradigms \
        --ws-root /path/to/stepalpha --json-root cache/ablation/step_alpha_v3 \
        --arms auto:auto-scale\\ (stock) a005:fixed\\ alpha=0.05 \
        --seeds 1,2,3 --out figures/step_alpha_v3.png

Per-arm workspace layout: <ws-root>/<arm>/runs/<run-dir>/state-*.pkl (the
solver's own per-iteration metrics); scores: <json-root>/<arm>/seed_<s>_full.json.
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SP))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from helpers import threshold_a  # noqa: E402

# legacy default: the 2026-08-14 A/B (single seed, ws dirs next to this file)
DEFAULT_ARMS = ['auto:auto-scale (stock)', 'fixed001:fixed alpha=0.01',
                'fixed005:fixed alpha=0.05', 'fixed020:fixed alpha=0.20']
COLORS = ['#4a3aa7', '#2a78d6', '#eda100', '#eb6834', '#2f9e6e', '#b03060']


def load_run(ws, seed):
    """(gt_series, flips_per_iter) for one arm/seed from the solver's state
    pickle; the run dir is matched on dep<seed> so arms sharing a workspace
    across seeds never cross-read."""
    pkls = [p for p in glob.glob(os.path.join(ws, 'runs', '*', 'state-*.pkl'))
            if 'dep{}-'.format(seed) in p or 'dep{}/'.format(seed) in p]
    if not pkls:
        return None, None
    latest = max(pkls, key=lambda p: int(re.search(r'state-(\d+)', p).group(1)))
    with open(latest, 'rb') as f:
        st = pickle.load(f)
    m = st['metrics'] if isinstance(st, dict) and 'metrics' in st else st
    gt = [float(v) for v in m['actual_nonconvex_objective']]
    advs = m['advertisements']
    flips = [int((threshold_a(np.asarray(advs[i - 1]))
                  != threshold_a(np.asarray(advs[i]))).sum())
             for i in range(1, len(advs))]
    return gt, flips


def dash_train_obj(steady_tag, failure_tag, arms, seeds, gamma=0.1):
    """The DASHBOARD's training objective, per (arm, seed), read from the
    same two stores plot_policy5 uses: steady soft cost + gamma * Sum_popps
    (soft cost under that popp's failure). Lower = better; one-per-peering
    is the floor on the steady part. Returns (totals, opp_totals) with
    totals[arm][seed] and opp_totals[seed]."""
    def _load(kind, tag):
        fn = os.path.join(REPO, 'cache', 'model_error', kind,
                          '{}.json'.format(tag))
        with open(fn) as f:
            return json.load(f)

    steady, failure = _load('steady', steady_tag), _load('failure', failure_tag)
    st, fa, opp_st, opp_fa = {}, {}, {}, {}
    for e in steady:
        if e['dir'] == 'REFS':
            if e['rung'] == 'opp' and e.get('train_obj') is not None:
                opp_st[e['seed']] = e['train_obj']
        elif e.get('train_obj') is not None:
            st[(_arm_of(e['dir'], arms), e['seed'])] = e['train_obj']
    for e in failure:
        v = e.get('popp', {}).get('obj_cost_sum')
        if v is None:
            continue
        if e['dir'] == 'REFS':
            if e['rung'] == 'opp':
                opp_fa[e['seed']] = v
        else:
            fa[(_arm_of(e['dir'], arms), e['seed'])] = v
    totals = {}
    for (arm, s), v in st.items():
        if arm is not None and (arm, s) in fa:
            totals.setdefault(arm, {})[s] = v + gamma * fa[(arm, s)]
    opp_totals = {s: opp_st[s] + gamma * opp_fa[s]
                  for s in set(opp_st) & set(opp_fa)}
    return totals, opp_totals


def _arm_of(dirpath, arms):
    parts = dirpath.rstrip('/').split(os.sep)
    for name, _ in arms:
        if name in parts:
            return name
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ws-root', default=SP,
                   help="root holding <arm> workspaces (legacy: this dir, "
                        "with arms named 'paradigm_<arm>')")
    p.add_argument('--ws-prefix', default='',
                   help="prefix on the workspace dir name (legacy: 'paradigm_')")
    p.add_argument('--json-root', default=None,
                   help='root holding <arm>/seed_<s>_full.json score files '
                        '(default: <ws-root>/<arm>/out)')
    p.add_argument('--arms', nargs='+', default=DEFAULT_ARMS,
                   help='arm specs "dir:label"')
    p.add_argument('--seeds', default='1')
    p.add_argument('--out', default=os.path.join(SP, 'paradigm_comparison.png'))
    p.add_argument('--title', default='Gradient-step paradigms')
    p.add_argument('--steady-tag', default=None,
                   help="cache/model_error/steady/<tag>.json -- with "
                        "--failure-tag, also report the DASH training "
                        "objective (steady + gamma*sum popp-fail)")
    p.add_argument('--failure-tag', default=None)
    p.add_argument('--gamma', type=float, default=0.1)
    args = p.parse_args()

    arms = [(a.split(':', 1)[0], a.split(':', 1)[1] if ':' in a else a.split(':', 1)[0])
            for a in args.arms]
    seeds = [int(s) for s in args.seeds.split(',')]

    fig, ax = plt.subplots(2, len(seeds), figsize=(5.4 * len(seeds), 8),
                           squeeze=False)
    rows = ['%-22s %s' % ('paradigm', '  '.join(
        'seed%-1d final/best  flips noop' % s for s in seeds))]
    for ai, (name, label) in enumerate(arms):
        color = COLORS[ai % len(COLORS)]
        ws = os.path.join(args.ws_root, args.ws_prefix + name)
        jroot = (os.path.join(args.json_root, name) if args.json_root
                 else os.path.join(ws, 'out'))
        cells = []
        for si, seed in enumerate(seeds):
            gt, flips = load_run(ws, seed)
            if gt is None:
                cells.append('%s' % 'NO DATA'.center(26))
                continue
            ax[0][si].plot(range(len(gt)), gt, color=color, label=label, lw=1.6)
            ax[1][si].plot(range(1, len(flips) + 1), np.cumsum(flips),
                           color=color, label=label, lw=1.6)
            jf = os.path.join(jroot, 'seed_{}_full.json'.format(seed))
            r = json.load(open(jf)) if os.path.exists(jf) else {}
            cells.append('%7.3f/%7.3f %4d %3d' % (
                gt[-1], min(gt), sum(flips), sum(1 for f in flips if f == 0)))
            rows.append('  seed %d %-18s final_gt=%7.3f best_gt=%7.3f '
                        'avg_lat=%7.3f diff_opp=%+6.3f flips=%3d noop=%d/%d'
                        % (seed, label, gt[-1], min(gt),
                           r.get('avg_lat', float('nan')),
                           r.get('diff_vs_opp', float('nan')), sum(flips),
                           sum(1 for f in flips if f == 0), len(flips)))
        rows.insert(len(rows) - len(seeds) if len(seeds) else 0, '')
    for si, seed in enumerate(seeds):
        ax[0][si].set_title('seed %d' % seed, fontsize=11)
        ax[0][si].set_ylabel('GT objective (cost)')
        ax[0][si].set_xlabel('iteration')
        ax[0][si].grid(alpha=.25)
        ax[1][si].set_ylabel('cumulative flips')
        ax[1][si].set_xlabel('iteration')
        ax[1][si].grid(alpha=.25)
    ax[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle(args.title + '  (lower objective = better)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print('wrote', args.out)
    if args.steady_tag and args.failure_tag:
        totals, opp = dash_train_obj(args.steady_tag, args.failure_tag,
                                     arms, seeds, args.gamma)
        rows.append('')
        rows.append('DASH METRIC -- training objective (steady + %.2f*sum '
                    'popp-fail soft cost; LOWER=better; opp = floor)'
                    % args.gamma)
        rows.append('%-22s %s   mean     mean-vs-opp' % (
            'paradigm', ' '.join('seed%d ' % s for s in seeds)))
        figB, axB = plt.subplots(figsize=(7.5, 4.4))
        for ai, (name, label) in enumerate(arms):
            by_seed = totals.get(name, {})
            common = [s for s in seeds if s in by_seed and s in opp]
            if not common:
                rows.append('%-22s NO DATA' % label)
                continue
            vals = [by_seed[s] for s in common]
            deltas = [by_seed[s] - opp[s] for s in common]
            rows.append('%-22s %s  %7.3f  %+7.3f' % (
                label, ' '.join('%6.2f' % by_seed[s] for s in common),
                float(np.mean(vals)), float(np.mean(deltas))))
            axB.bar(ai, np.mean(deltas), color=COLORS[ai % len(COLORS)],
                    width=.62)
            axB.errorbar(ai, np.mean(deltas),
                         yerr=(np.std(deltas) / max(1, len(deltas) ** .5)),
                         color='k', capsize=4, lw=1)
        axB.set_xticks(range(len(arms)))
        axB.set_xticklabels([l for _, l in arms], fontsize=8, rotation=12)
        axB.axhline(0, color='k', lw=1)
        axB.set_ylabel('mean training objective - same-seed opp')
        axB.set_title(args.title + ' -- dash training objective\n'
                      '(lower = better; 0 = one-per-peering)', fontsize=10)
        axB.grid(alpha=.25, axis='y')
        figB.tight_layout()
        outB = os.path.splitext(args.out)[0] + '_trainobj.png'
        figB.savefig(outB, dpi=160)
        print('wrote', outB)
    txt = '\n'.join(r for r in rows if r.strip())
    print(txt)
    with open(os.path.splitext(args.out)[0] + '_table.txt', 'w') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
