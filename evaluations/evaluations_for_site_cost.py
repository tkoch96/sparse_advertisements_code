"""Evaluation for the per_site_cost objective.

per_site_cost is avg_latency + alpha * sum(active site cost), so the
comparison that matters is how many sites each solution type keeps lit --
that is the term the objective trades latency against.

Scores each advertisement by its active-site count: the number of PoPs with
at least one popp advertised on any prefix. That is a property of the
advertisement itself, so it needs no LP solve and cannot fail the way a
scoring solve can.
"""
import numpy as np

from evaluations._objective_eval_base import (
    score_all_strategies, bar_comparison, announce)

OBJECTIVES = ('per_site_cost',)


def _active_sites(sas, adv):
    """PoPs with at least one advertised popp, on any prefix."""
    on = np.asarray(adv) > .5
    if on.ndim == 1:
        on = on[:, None]
    lit_popps = np.where(on.any(axis=1))[0]
    pops = set()
    for poppi in lit_popps:
        try:
            pops.add(sas.popps[poppi][0])
        except (IndexError, TypeError):
            continue
    return len(pops)


def run(ctx):
    announce(ctx, 'evaluations_for_site_cost',
             'counting active sites per solution type')
    score_all_strategies(ctx, _active_sites, 'active_sites_by_strategy')
    bar_comparison(ctx, 'active_sites_by_strategy',
                   ylabel='active sites (PoPs lit)',
                   title='Site cost by solution type',
                   out_name='site_cost_comparison_{}.pdf'.format(ctx.dpsize),
                   lower_is_better=True)
    return ctx.metrics
