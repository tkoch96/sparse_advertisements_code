"""Evaluation for the joint_priority objective.

joint_priority solves a strict-priority-like LP (the B4/SWAN semantics):
the high-priority traffic class is served first at minimum latency, then
bulk traffic fills the remaining capacity. The LP is
`core.solve_lp_assignment.solve_joint_latency_bulk_download`, registered
under the name 'joint_priority' in generic_lp_functions (2026-08-22 --
it was documented but unregistered since birth, so this module used to
print a not-dispatchable notice instead of scoring).

Two scores per solution type:
  priority_by_strategy       fraction of ALL traffic (HPrio + bulk)
                             served uncongested once bulk is placed
                             (1 - fraction_congested_volume_with_bulk;
                             higher is better -- 'priority placement')
  hprio_latency_by_strategy  volume-weighted mean latency of the
                             high-priority class alone (lower better)

evaluations/testing_priorities.py is the legacy standalone driver with
the paper-plot output shape.
"""
import numpy as np

from evaluations.objectives._objective_eval_base import (
    score_all_strategies, bar_comparison, announce, objective_value_scorer)

OBJECTIVES = ('joint_priority',)


def _joint_ret(sas, adv):
    from helpers.helpers import threshold_a
    from core.solve_lp_assignment import solve_joint_latency_bulk_download
    rti, _ = sas.calculate_ground_truth_ingress(threshold_a(adv))
    ret = solve_joint_latency_bulk_download(sas, rti, 'joint_priority')
    if not ret.get('solved'):
        raise ValueError('joint-priority LP unsolved for this advertisement')
    return ret


def _priority_placement(sas, adv):
    ret = _joint_ret(sas, adv)
    return 1.0 - float(ret['fraction_congested_volume_with_bulk'])


def _hprio_latency(sas, adv):
    ret = _joint_ret(sas, adv)
    lats = np.asarray(ret['lats_by_ug'], dtype=float).flatten()
    vols = np.asarray(sas.whole_deployment_ug_vols, dtype=float).flatten()
    n = min(len(lats), len(vols))
    if n == 0:
        raise ValueError('no HPrio latencies returned')
    return float(np.sum(lats[:n] * vols[:n]) / (np.sum(vols[:n]) + 1e-9))


def _hprio_frac_routed(sas, adv):
    """Fraction of high-priority volume actually routed (not NO_ROUTE)."""
    ret = _joint_ret(sas, adv)
    lats = np.asarray(ret['lats_by_ug'], dtype=float).flatten()
    vols = np.asarray(sas.whole_deployment_ug_vols, dtype=float).flatten()
    n = min(len(lats), len(vols))
    from helpers.constants import NO_ROUTE_LATENCY
    routed = lats[:n] < NO_ROUTE_LATENCY * 0.99
    tv = float(np.sum(vols[:n]))
    return float(np.sum(vols[:n][routed]) / tv) if tv > 0 else None


def _bulk_routable(sas, adv):
    """Fraction of bulk demand placed on links that stay uncongested
    (util <= 1) once bulk lands -- 'bulk routable before congestion'."""
    ret = _joint_ret(sas, adv)
    bulk_utils = ret['bulk_vols_by_poppi']
    ok_popps = {pi for pi, u in bulk_utils.items() if u <= 1.0}
    placed_ok = 0.0
    bulk_vols = np.asarray(sas.whole_deployment_ug_bulk_vols,
                           dtype=float).flatten()
    for ugi, pathvols in (ret.get('bulk_paths_by_ug') or {}).items():
        for poppi, vol_pct in pathvols:
            if poppi in ok_popps and ugi < len(bulk_vols):
                placed_ok += vol_pct * bulk_vols[ugi]
    total_bulk = float(np.sum(bulk_vols))
    return placed_ok / total_bulk if total_bulk > 0 else None


def _joint_at_ratio(sas, adv, bv, rti=None):
    """Direct-driver joint LP at LPrio/HPrio ratio bv. Returns the
    fraction of HPRIO traffic congested once bulk lands
    (fraction_congested_volume_with_bulk) -- the exact metric the
    B4/SWAN paper figure plots. Bulk itself is ELASTIC (links may pack
    to the 100x bulk cap by design); the harm signal is HPrio traffic
    sharing links that bulk pushed past capacity. Empirically verified
    2026-08-23: SCULPTOR sim0 at SWAN ratio 4.0 -> 0.183."""
    import numpy as np
    from helpers.helpers import threshold_a
    from core.solve_lp_assignment import solve_joint_latency_bulk_download
    a = threshold_a(adv)
    base_vol = dict(sas.whole_deployment_ug_to_vol)
    sas.ug_to_bulk_vol = {ug: v * bv for ug, v in sas.ug_to_vol.items()}
    sas.whole_deployment_ug_to_bulk_vol = {
        ug: v * bv for ug, v in base_vol.items()}
    sas.whole_deployment_ug_bulk_vols = np.array(
        [sas.whole_deployment_ug_to_bulk_vol[u]
         for u in sas.whole_deployment_ugs])
    if rti is None:
        # rti is bulk-volume-independent -- callers bisecting over bv
        # should compute it once and pass it in (~10x fewer rti calcs)
        rti, _ = sas.calculate_ground_truth_ingress(a)
    ret = solve_joint_latency_bulk_download(sas, rti, 'joint_priority')
    if not ret.get('solved'):
        # infeasible at this bulk ratio = the bulk cannot be placed at
        # all -- for the bisection that IS 'fully congested', not an
        # error. Raising here recorded None ('-') for exactly the GOOD
        # strategies, whose critical ratios push into infeasible
        # territory (Tom 2026-08-26: crit bulk ratio missing for most).
        return 1.0, ret
    return float(ret['fraction_congested_volume_with_bulk']), ret


def _critical_bulk_ratio(sas, adv):
    """Max LPrio/HPrio ratio keeping HPrio congestion <= 5% of HPrio
    volume, bisected. An any-congestion threshold is degenerate: the
    steady-state LP saturates some link to exactly util 1.0, so any
    bulk at all congests >0 HPrio (every strategy floored at 0.25).
    SWAN reference 4.0, B4 6.0."""
    lo, hi = 0.25, 16.0
    from helpers.helpers import threshold_a
    rti, _ = sas.calculate_ground_truth_ingress(threshold_a(adv))
    # measure harm RELATIVE to the bv->0 baseline: the steady-state LP
    # leaves ties at exactly util 1.0, so a fixed slice of HPrio counts
    # congested the moment any bulk exists -- an absolute threshold
    # floors every strategy at lo. Critical = max ratio where bulk has
    # added <= 5 points of HPrio congestion beyond that baseline.
    f_lo, _ = _joint_at_ratio(sas, adv, lo, rti=rti)
    if f_lo >= 0.95:
        # already (essentially) fully congested before bulk matters --
        # the relative +5pt threshold is unreachable and would return
        # the ceiling for the WORST strategy (anycast). Floor it.
        return lo
    eps = f_lo + 0.05
    f_hi, _ = _joint_at_ratio(sas, adv, hi, rti=rti)
    if f_hi <= eps:
        return hi
    for _ in range(10):
        if hi - lo <= 0.05 * max(lo, 1e-9):
            break
        mid = 0.5 * (lo + hi)
        f, _ = _joint_at_ratio(sas, adv, mid, rti=rti)
        if f > eps:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _hprio_congestion_at_swan(sas, adv):
    """Fraction of HPrio traffic congested at the SWAN ratio (4.0)."""
    f, _ = _joint_at_ratio(sas, adv, 4.0)
    return f


def run(ctx):
    announce(ctx, 'evaluations_for_priorities',
             'strict-priority (B4/SWAN) placement per solution type')
    score_all_strategies(ctx, _priority_placement, 'priority_by_strategy')
    score_all_strategies(ctx, objective_value_scorer('joint_priority'),
                         'objective_value_by_strategy')
    score_all_strategies(ctx, _hprio_latency, 'hprio_latency_by_strategy')
    score_all_strategies(ctx, _hprio_frac_routed, 'hprio_frac_routed_by_strategy')
    score_all_strategies(ctx, _bulk_routable, 'bulk_routable_by_strategy')
    score_all_strategies(ctx, _critical_bulk_ratio,
                         'critical_bulk_ratio_by_strategy')
    score_all_strategies(ctx, _hprio_congestion_at_swan,
                         'hprio_cong_swan_by_strategy')
    bar_comparison(ctx, 'priority_by_strategy',
                   ylabel='fraction served uncongested (with bulk)',
                   title='Priority placement by solution type',
                   out_name='priority_placement_{}.pdf'.format(ctx.dpsize),
                   lower_is_better=False)
    bar_comparison(ctx, 'hprio_latency_by_strategy',
                   ylabel='HPrio volume-weighted latency (ms)',
                   title='High-priority latency by solution type',
                   out_name='priority_hprio_latency_{}.pdf'.format(ctx.dpsize),
                   lower_is_better=True)
    return ctx.metrics
