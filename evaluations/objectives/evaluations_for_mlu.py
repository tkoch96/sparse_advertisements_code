"""Evaluation for the MLU objectives (max_util, lat_plus_max_util).

Scores every solution type by the metric the objective actually optimised:
the best achievable peak link utilisation for that advertisement, via
`core.solve_lp_assignment.solve_min_mlu` -- which its own docstring calls THE
canonical MLU of an advertisement, so there is one implementation to fix.

Deliberately does NOT run the latency+resilience phases. Those weight
millisecond deltas by UG volume; an MLU run scored that way yields numbers
that look valid and answer a different question.

Still to build out (2026-08-21): MLU under single-link and single-PoP failure,
which is the interesting comparison and needs a failure-scenario loop like the
one in evaluations_for_latency_plus_resilience.run.
"""
from evaluations.objectives._objective_eval_base import (
    score_all_strategies, bar_comparison, announce, objective_value_scorer)
from core.solve_lp_assignment import solve_min_mlu

OBJECTIVES = ('max_util', 'lat_plus_max_util')


def _mlu(sas, adv):
    # solve_min_mlu wants routed_through_ingress ({prefix: {ug: popp}}),
    # NOT the advertisement matrix -- passing adv raw died with
    # "'numpy.ndarray' object has no attribute 'items'" on this module's
    # first real run (2026-08-22). Compute the ground-truth routing for
    # the thresholded adv first; returns (mlu, routable_vol_frac).
    from helpers.helpers import threshold_a
    rti, _ = sas.calculate_ground_truth_ingress(threshold_a(adv))
    import os
    os.environ.setdefault('SCULPTOR_MIN_MLU_TIMELIMIT', '120')
    try:
        ret = solve_min_mlu(sas, rti)
    except RuntimeError as e:
        if 'status 9' not in str(e):
            raise
        # TIME_LIMIT on huge advs (one_per_peering at actual-32 was the
        # 2026-08-26 casualty: MLU column '-'). Fall back to the
        # assignment-peak MLU from the steady latency LP -- defined for
        # every advertisement, slightly conservative (upper bound).
        from core.solve_lp_assignment import solve_generic_lp_with_failure_catch
        from core.hard_objectives import _max_util_from_ret
        st = solve_generic_lp_with_failure_catch(sas, rti, 'avg_latency',
                                                 no_persistent=True)
        peak = _max_util_from_ret(st, sas.link_capacities_arr,
                                  sas.n_popps, sas=sas)
        if peak is None:
            raise
        print('[mlu] min-MLU timed out; assignment-peak fallback {:.3f}'
              .format(peak))
        return float(peak)
    if isinstance(ret, tuple):
        mlu, _routable = ret
        if mlu is None:
            raise ValueError('nothing routes under this advertisement')
        return mlu
    if isinstance(ret, dict):
        for k in ('mlu', 'objective', 'max_util', 'obj'):
            if k in ret:
                return ret[k]
        raise KeyError('solve_min_mlu returned {} with no MLU-like key'
                       .format(sorted(ret)))
    return ret


def run(ctx):
    announce(ctx, 'evaluations_for_mlu',
             'scoring peak link utilisation per solution type')
    score_all_strategies(ctx, _mlu, 'mlu_by_strategy')
    score_all_strategies(ctx, objective_value_scorer(ctx.kwargs.get('generic_objective') or 'max_util'),
                         'objective_value_by_strategy')
    bar_comparison(ctx, 'mlu_by_strategy',
                   ylabel='peak link utilisation',
                   title='MLU by solution type',
                   out_name='mlu_comparison_{}.pdf'.format(ctx.dpsize),
                   lower_is_better=True)
    return ctx.metrics
