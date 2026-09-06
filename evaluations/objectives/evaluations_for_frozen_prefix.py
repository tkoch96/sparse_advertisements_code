"""Evaluation for the frozen_prefix objective (Tom 2026-09-06).

The question this suite answers: with the user->prefix allocation FROZEN
(no reactive DNS re-steering after a failure -- BGP fallback within the
pinned prefix is the only adaptation), how do the solution types compare
on the three failure metrics, kept distinct:

  frozen_fail_latency_by_strategy    ms, routed & uncongested volume
  frozen_fail_cong_by_strategy       fraction on over-capacity popps
  frozen_fail_no_route_by_strategy   fraction stranded (pinned prefix has
                                     no surviving popp)

Every strategy's advertisement is pinned by the SAME rule (the frozen_prefix
LP's own allocation) and swept over single-popp failures. The
One-per-peering row is additionally scored as the REACTIVE-OPTIMAL anchor
(`reactive_*` keys: assignment re-optimized after each failure on the
full-availability advertisement), because frozen one-per-peering has no
backup on any prefix (a single popp each) and strands users -- its frozen
numbers are still recorded under the frozen_* keys for the text. The paper
table's 'frozen_anchor' extractor shows the reactive numbers on that row.

All LPs are driver-side on the sim's own deployment; no worker pool.
"""
from evaluations.objectives._objective_eval_base import (
    score_all_strategies, score_all_strategies_multi, bar_comparison,
    announce, objective_value_scorer)

OBJECTIVES = ('frozen_prefix',)

FROZEN_KEYS = (
    'frozen_steady_latency_by_strategy',
    'frozen_fail_latency_by_strategy',
    'frozen_fail_cong_by_strategy',
    'frozen_fail_no_route_by_strategy',
    'frozen_fail_worst_cong_by_strategy',
    'frozen_fail_worst_no_route_by_strategy',
)
REACTIVE_KEYS = (
    'reactive_steady_latency_by_strategy',
    'reactive_fail_latency_by_strategy',
    'reactive_fail_cong_by_strategy',
    'reactive_fail_no_route_by_strategy',
)
ANCHOR_STRATEGY = 'one_per_peering'


def _score(sas, adv, strategy):
    from core.frozen_prefix_eval import (frozen_failure_metrics,
                                         reactive_optimal_metrics)
    fm = frozen_failure_metrics(sas, adv, which='popps')
    out = {
        'frozen_steady_latency_by_strategy': fm['steady_latency_ms'],
        'frozen_fail_latency_by_strategy': fm['fail_latency_ms'],
        'frozen_fail_cong_by_strategy': fm['fail_frac_cong'],
        'frozen_fail_no_route_by_strategy': fm['fail_frac_no_route'],
        'frozen_fail_worst_cong_by_strategy': fm['worst_frac_cong'],
        'frozen_fail_worst_no_route_by_strategy': fm['worst_frac_no_route'],
    }
    if strategy == ANCHOR_STRATEGY:
        rm = reactive_optimal_metrics(sas, adv, which='popps')
        out.update({
            'reactive_steady_latency_by_strategy': rm['steady_latency_ms'],
            'reactive_fail_latency_by_strategy': rm['fail_latency_ms'],
            'reactive_fail_cong_by_strategy': rm['fail_frac_cong'],
            'reactive_fail_no_route_by_strategy': rm['fail_frac_no_route'],
        })
    return out


def run(ctx):
    announce(ctx, 'evaluations_for_frozen_prefix',
             'frozen user->prefix failover: latency / % congested / % no-route '
             'per solution type (One-per-peering row = reactive-optimal anchor)')
    score_all_strategies_multi(ctx, _score, FROZEN_KEYS + REACTIVE_KEYS)
    score_all_strategies(ctx, objective_value_scorer('frozen_prefix'),
                         'objective_value_by_strategy')
    bar_comparison(ctx, 'frozen_fail_latency_by_strategy',
                   ylabel='latency under failure, frozen (ms)',
                   title='Frozen failover latency by solution type',
                   out_name='frozen_prefix_failure_latency_{}.pdf'.format(ctx.dpsize),
                   lower_is_better=True)
    bar_comparison(ctx, 'frozen_fail_no_route_by_strategy',
                   ylabel='fraction no-route under failure, frozen',
                   title='Frozen failover no-route by solution type',
                   out_name='frozen_prefix_failure_no_route_{}.pdf'.format(ctx.dpsize),
                   lower_is_better=True)
    return ctx.metrics
