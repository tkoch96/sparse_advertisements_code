"""Evaluation for frac_beyond_optimal -- traffic within X ms of optimal.

The objective maximises the fraction of traffic served within a latency
threshold of the best achievable, so the comparison is that fraction per
solution type, not mean latency: two advertisements can share a mean and
differ sharply in how much traffic sits in the tail.

Threshold is 10ms by default, overridable with SCULPTOR_FRAC_BEYOND_MS.

Metric definition (fixed 2026-09-02): capacity-aware LP assignment via
solve_lp_with_failure_catch; fraction of VOLUME whose assigned latency is
within the threshold of that user's own optimal. (The pre-fix version
reused calc_pct_volume_within_latency, a capacity-blind route-reachability
curve that multi-counts users across prefixes -- see _frac_within's
comment; papertable32b values were recomputed and the pickle corrected.)
"""
import os

import numpy as np

from evaluations.objectives._objective_eval_base import (
    score_all_strategies, bar_comparison, announce, objective_value_scorer)

OBJECTIVES = ('frac_beyond_optimal',)

THRESHOLD_MS = float(os.environ.get('SCULPTOR_FRAC_BEYOND_MS', '10'))


def _frac_within(sas, adv):
    # Definition (Tom 2026-09-02, two fixes same day):
    #   * traffic ASSIGNED by the trained objective's own soft LP
    #     (solve_generic_lp_with_failure_catch, 'frac_beyond_optimal') --
    #     "evaluate with what we trained with"; a min-avg-latency
    #     assignment operates the network for a different objective and
    #     under-credits threshold-shaped advertisements. This matches how
    #     every other objective suite evaluates.
    #   * the reported number is the HARD count on that assignment:
    #     volume (counted once) within THRESHOLD_MS of each user's own
    #     optimal, capacity-aware.
    # (History: the original form reused calc_pct_volume_within_latency,
    # a capacity-blind route-reachability curve that multi-counts users
    # across prefixes -- it inflated painter +26pts / deflated sparse
    # -50pts at sim 0 and inverted the ordering.)
    from core.solve_lp_assignment import solve_generic_lp_with_failure_catch
    a = np.asarray(adv, dtype=float)
    rti, _ = sas.calculate_ground_truth_ingress(a)
    ret = solve_generic_lp_with_failure_catch(
        sas, rti, 'frac_beyond_optimal', adv=a)
    lats = np.asarray(ret['lats_by_ug'], dtype=float)
    vols = np.asarray(sas.ug_vols, dtype=float)
    best = np.asarray([min(sas.ug_perfs[ug].values()) for ug in sas.ugs],
                      dtype=float)
    return float(vols[(lats - best) <= THRESHOLD_MS].sum() / vols.sum())


def run(ctx):
    announce(ctx, 'evaluations_for_frac_beyond_optimal',
             'fraction of traffic within {:.0f}ms of optimal'.format(THRESHOLD_MS))
    score_all_strategies(ctx, _frac_within, 'frac_within_threshold_by_strategy')
    score_all_strategies(ctx, objective_value_scorer('frac_beyond_optimal'),
                         'objective_value_by_strategy')
    bar_comparison(ctx, 'frac_within_threshold_by_strategy',
                   ylabel='fraction of volume within {:.0f}ms'.format(THRESHOLD_MS),
                   title='Traffic within {:.0f}ms of optimal'.format(THRESHOLD_MS),
                   out_name='frac_within_{:.0f}ms_{}.pdf'.format(
                       THRESHOLD_MS, ctx.dpsize),
                   lower_is_better=False)
    return ctx.metrics
