"""Shared scaffolding for the non-latency objective evaluations.

Each objective suite answers the same question -- *how do the six solution
types compare under THIS objective?* -- so the mechanics of scoring every
strategy's advertisement and drawing a bar comparison are factored here. What
differs is the scoring function, which each suite supplies.

This deliberately does NOT reuse the latency+resilience phases. Those read
`latency_delta_*` and weight by UG volume; applied to an MLU or priority
optimisation they produce plausible-looking numbers that mean nothing, which
is exactly the confusion the 2026-08-21 split exists to end.
"""
import os
import traceback

import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

from helpers.helpers import save_fig
from helpers.figpaths import fig_path  # noqa: F401  (kept for callers)


def score_all_strategies(ctx, score_one, metric_key):
    """Score every solved advertisement with `score_one(sas, adv) -> float`.

    Writes metrics[metric_key][sim][strategy] and returns that dict. A
    strategy that raises is recorded as None and reported, rather than
    aborting the comparison -- the point of this evaluation is the spread
    across strategies, so one bad arm should not cost you the other five.
    """
    metrics = ctx.metrics
    metrics.setdefault(metric_key, {})
    for sim in range(ctx.N_TO_SIM):
        rets = (metrics.get('compare_rets') or {}).get(sim) or {}
        advs = rets.get('adv_solns') or {}
        metrics[metric_key].setdefault(sim, {})
        for strategy in ctx.soln_types:
            try:
                adv = advs[strategy][0]
            except (KeyError, IndexError):
                print("[{}] no solution for {}".format(metric_key, strategy))
                metrics[metric_key][sim][strategy] = None
                continue
            try:
                metrics[metric_key][sim][strategy] = float(score_one(ctx.sas, adv))
            except Exception:
                print("[{}] scoring failed for {}".format(metric_key, strategy))
                traceback.print_exc()
                metrics[metric_key][sim][strategy] = None
    return metrics[metric_key]


def bar_comparison(ctx, metric_key, ylabel, title, out_name, lower_is_better=True):
    """One bar per solution type, averaged over sims. Saved for later perusal."""
    scores = (ctx.metrics.get(metric_key) or {})
    strategies, values = [], []
    for strategy in ctx.soln_types:
        vals = [scores.get(sim, {}).get(strategy) for sim in scores]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        strategies.append(strategy)
        values.append(float(np.mean(vals)))
    if not strategies:
        print("[{}] nothing to plot -- every strategy scored None".format(metric_key))
        return None

    order = np.argsort(values)
    if not lower_is_better:
        order = order[::-1]
    strategies = [strategies[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(range(len(values)), values, color='#4a7fb5')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title('{}  ({}, {} better)'.format(
        title, ctx.dpsize, 'lower' if lower_is_better else 'higher'), fontsize=9)
    ax.grid(alpha=.25, axis='y')
    for i, v in enumerate(values):
        ax.text(i, v, '{:.4g}'.format(v), ha='center', va='bottom', fontsize=7)
    fig.tight_layout()
    save_fig(out_name)
    plt.close(fig)
    print('[{}] wrote figures/{}'.format(metric_key, out_name))
    return dict(zip(strategies, values))


def announce(ctx, module_name, what):
    print('\n' + '=' * 66)
    print('{} -- objective={} dpsize={}'.format(module_name, ctx.objective, ctx.dpsize))
    print(what)
    print('=' * 66)
