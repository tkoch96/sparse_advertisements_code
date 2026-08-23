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

from evaluations.objectives._objective_eval_base import (
    score_all_strategies, bar_comparison, announce, objective_value_scorer)

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


def _site_cost_load(sas, adv):
    """Per-site cost-load contributions under the objective's own LP:
    {pop: site_cost[pop] * volume_landing_on_pop / total_volume}. The
    adv-based active-site max/avg was degenerate whenever every strategy
    lights every site (any small deployment) -- load-aware versions
    discriminate (Tom 2026-08-23 'site costs are still all the same')."""
    from helpers.helpers import threshold_a
    from core.solve_lp_assignment import solve_generic_lp_with_failure_catch
    a = threshold_a(adv)
    rti, _ = sas.calculate_ground_truth_ingress(a)
    ret = solve_generic_lp_with_failure_catch(sas, rti, 'per_site_cost', adv=a)
    if not ret.get('solved'):
        raise ValueError('site-cost LP unsolved')
    vols_by_poppi = ret.get('vols_by_poppi')
    if not vols_by_poppi:
        raise ValueError('LP returned no per-popp volumes')
    per_pop = {}
    tv = 0.0
    for poppi, vol in vols_by_poppi.items():
        try:
            pop = sas.popps[int(poppi)][0]
        except (IndexError, ValueError, TypeError):
            continue
        v = float(vol)
        per_pop[pop] = per_pop.get(pop, 0.0) + v
        tv += v
    if tv <= 0:
        raise ValueError('no routed volume')
    costs = sas.site_costs
    return {p: float(costs.get(p, 0.0)) * v / tv for p, v in per_pop.items()}


def _max_site_cost_load(sas, adv):
    return max(_site_cost_load(sas, adv).values())


def _avg_site_cost_load(sas, adv):
    d = _site_cost_load(sas, adv)
    return float(np.mean(list(d.values())))


def _weighted_site_cost(sas, adv):
    """Traffic-weighted site cost: solve the objective's own LP for the
    advertisement, then sum(site_cost[pop] * volume landing on pop) /
    total volume. Unlike the active-site count (identical across
    solutions when everyone lights all sites, e.g. actual-3), this
    reflects HOW MUCH traffic each solution parks on expensive sites."""
    from helpers.helpers import threshold_a
    from core.solve_lp_assignment import solve_generic_lp_with_failure_catch
    a = threshold_a(adv)
    rti, _ = sas.calculate_ground_truth_ingress(a)
    ret = solve_generic_lp_with_failure_catch(sas, rti, 'per_site_cost', adv=a)
    if not ret.get('solved'):
        raise ValueError('site-cost LP unsolved')
    vols_by_poppi = ret.get('vols_by_poppi')
    if not vols_by_poppi:
        raise ValueError('LP returned no per-popp volumes')
    costs = sas.site_costs
    num = den = 0.0
    for poppi, vol in vols_by_poppi.items():
        try:
            pop = sas.popps[int(poppi)][0]
        except (IndexError, ValueError, TypeError):
            continue
        v = float(vol)
        num += float(costs.get(pop, 0.0)) * v
        den += v
    return num / den if den > 0 else None


def run(ctx):
    announce(ctx, 'evaluations_for_site_cost',
             'counting active sites per solution type')
    score_all_strategies(ctx, _active_sites, 'active_sites_by_strategy')
    score_all_strategies(ctx, _weighted_site_cost,
                         'weighted_site_cost_by_strategy')
    score_all_strategies(ctx, _max_site_cost_load,
                         'max_site_cost_load_by_strategy')
    score_all_strategies(ctx, _avg_site_cost_load,
                         'avg_site_cost_load_by_strategy')
    score_all_strategies(ctx, objective_value_scorer('per_site_cost'),
                         'objective_value_by_strategy')
    bar_comparison(ctx, 'active_sites_by_strategy',
                   ylabel='active sites (PoPs lit)',
                   title='Site cost by solution type',
                   out_name='site_cost_comparison_{}.pdf'.format(ctx.dpsize),
                   lower_is_better=True)
    return ctx.metrics
