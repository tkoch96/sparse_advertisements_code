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
from evaluations._objective_eval_base import (
    score_all_strategies, bar_comparison, announce)
from core.solve_lp_assignment import solve_min_mlu

OBJECTIVES = ('max_util', 'lat_plus_max_util')


def _mlu(sas, adv):
    ret = solve_min_mlu(sas, adv)
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
    bar_comparison(ctx, 'mlu_by_strategy',
                   ylabel='peak link utilisation',
                   title='MLU by solution type',
                   out_name='mlu_comparison_{}.pdf'.format(ctx.dpsize),
                   lower_is_better=True)
    return ctx.metrics
