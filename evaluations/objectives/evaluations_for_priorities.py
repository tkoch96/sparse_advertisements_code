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
    _bulk_sweep_all(ctx)
    _gen_priority_paper_figures(ctx)
    return ctx.metrics


# ---- bulk-volume sweep + the B4/SWAN paper figures --------------------
# Ported from old_scripts/testing_priorities.py:gen_paper_plots (Tom
# 2026-08-30): congestion-vs-ratio curves and the HPrio-latency CDF at
# the ratio of interest, emitted as priority_bulk_traffic_congestion.pdf
# and priority_bulk_traffic_latency.pdf.
import numpy as _np

BULK_SWEEP_VALS = [round(float(x), 3) for x in _np.linspace(0.1, 9.0, 20)]
_BV_OF_INTEREST = 5.0     # latency CDF slice
_B4_RATIO, _SWAN_RATIO = 6.0, 4.0


def _bulk_sweep_all(ctx):
    """metrics['bulk_sweep_by_strategy'][sim][strategy] =
    {'congestion_over_bulk_vals': {bv: frac},
     'bulk_latencies_over_bulk_vals': {bv: lats_array}}.
    EXPENSIVE (~20 joint LPs per strategy): cached in the metrics pickle
    and skipped when complete unless SCULPTOR_RECALC contains
    'bulk_sweep'. Sim count capped by SCULPTOR_BULK_SWEEP_SIMS (default
    1 -- the paper figure plots sim 0, matching the legacy script)."""
    import os
    from helpers.helpers import threshold_a
    metrics = ctx.metrics
    key = 'bulk_sweep_by_strategy'
    metrics.setdefault(key, {})
    recalc = 'bulk_sweep' in os.environ.get('SCULPTOR_RECALC', '')
    n_sims = min(ctx.N_TO_SIM,
                 int(os.environ.get('SCULPTOR_BULK_SWEEP_SIMS', '1')))
    cache = getattr(ctx, '_per_sim_sas', None) or {}
    for sim in range(n_sims):
        rets = (metrics.get('compare_rets') or {}).get(sim) or {}
        advs = rets.get('adv_solns') or {}
        metrics[key].setdefault(sim, {})
        sas = cache.get(sim) or ctx.sas
        for strategy in ctx.soln_types:
            have = metrics[key][sim].get(strategy)
            if (not recalc and have
                    and set(have.get('congestion_over_bulk_vals') or {})
                    >= set(BULK_SWEEP_VALS)):
                continue
            try:
                adv = advs[strategy][0]
            except (KeyError, IndexError):
                metrics[key][sim][strategy] = None
                continue
            print('[bulk_sweep] sim {} strategy {} ({} ratios)'.format(
                sim, strategy, len(BULK_SWEEP_VALS)), flush=True)
            try:
                rti, _ = sas.calculate_ground_truth_ingress(
                    threshold_a(adv))
                cong, lats = {}, {}
                for bv in BULK_SWEEP_VALS:
                    f, ret = _joint_at_ratio(sas, adv, bv, rti=rti)
                    cong[bv] = f
                    lats[bv] = _np.asarray(
                        ret.get('bulk_lats_by_ug')
                        if isinstance(ret, dict) else None)
                metrics[key][sim][strategy] = {
                    'congestion_over_bulk_vals': cong,
                    'bulk_latencies_over_bulk_vals': lats}
            except Exception:
                import traceback
                traceback.print_exc()
                metrics[key][sim][strategy] = None
    return metrics[key]


def _gen_priority_paper_figures(ctx):
    """The two paper figures from the sweep (legacy names/format)."""
    from helpers.helpers import get_cdf_xy
    from helpers.paper_plotting_functions import (
        get_figure, save_figure, solution_to_marker,
        solution_to_line_color, solution_to_plot_label)
    metrics = ctx.metrics
    sweep = metrics.get('bulk_sweep_by_strategy') or {}
    sims = [s for s, d in sweep.items()
            if d and any(v for v in d.values())]
    if not sims:
        print('[priority figs] no sweep data -- skipping', flush=True)
        return
    solutions = ['anycast', 'anyopt', 'one_per_pop', 'painter',
                 'sparse', 'one_per_peering']

    # -- congestion vs LPrio/HPrio ratio
    f, ax = get_figure(l=3.5)
    for solution in solutions:
        ys = []
        for bv in BULK_SWEEP_VALS:
            vals = [sweep[s][solution]['congestion_over_bulk_vals'][bv]
                    for s in sims if sweep[s].get(solution)]
            if vals:
                ys.append(float(_np.mean(vals)))
        if len(ys) != len(BULK_SWEEP_VALS):
            continue
        ax.plot(BULK_SWEEP_VALS, ys,
                marker=solution_to_marker[solution],
                color=solution_to_line_color[solution],
                label=solution_to_plot_label[solution])
    ax.set_xlabel('LPrio/HPrio Ratio', fontsize=18)
    ax.set_ylabel('Fraction HPrio\nTraffic Congested', fontsize=18)
    ax.set_yticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    ax.text(_B4_RATIO - .75, .7, 'B4\nRatio', fontsize=18)
    ax.axvline(_B4_RATIO, 0, 1, linestyle='--', color='black')
    ax.text(_SWAN_RATIO - 2.7, .6, 'SWAN\nRatio', fontsize=18)
    ax.axvline(_SWAN_RATIO, 0, 1, linestyle='--', color='black')
    ax.grid(True)
    save_figure('priority_bulk_traffic_congestion.pdf')

    # -- HPrio latency CDF at the ratio of interest
    bv = min(BULK_SWEEP_VALS, key=lambda x: abs(x - _BV_OF_INTEREST))
    f, ax = get_figure(l=3.5)
    for solution in solutions[::-1]:
        diffs, wts = [], []
        for s in sims:
            d = sweep[s].get(solution)
            if not d:
                continue
            lats = d['bulk_latencies_over_bulk_vals'].get(bv)
            vols = (metrics.get('ug_to_vol') or {}).get(s)
            if lats is None or vols is None:
                continue
            for lat, vol in zip(_np.asarray(lats).flatten(),
                                _np.asarray(vols).flatten()):
                diffs.append(float(lat))
                wts.append(float(vol))
        if solution == 'anycast' or not diffs:
            x = _np.linspace(0, 250, num=100)
            cdf_x = _np.zeros(100)
        else:
            x, cdf_x = get_cdf_xy(list(zip(diffs, wts)), weighted=True)
        ax.plot(x[::10], cdf_x[::10],
                marker=solution_to_marker[solution],
                color=solution_to_line_color[solution],
                label=solution_to_plot_label[solution])
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlabel('HPrio Latency (ms)', fontsize=18)
    ax.set_ylabel('CDF of Traffic', fontsize=18)
    ax.set_yticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    ax.set_xlim([0, 250])
    ax.grid(True)
    save_figure('priority_bulk_traffic_latency.pdf')
    print('[priority figs] wrote priority_bulk_traffic_{congestion,'
          'latency}.pdf', flush=True)
