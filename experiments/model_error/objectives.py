"""Candidate HARD objective functions for the measurement-budget study.

Three objectives, each exercising a difficulty class from DIMENSIONS.md
axis 4 that the current avg-latency objective does not:

  (a) frac_users_beyond      threshold/indicator (class c): fraction of
                             volume NOT within X ms of its optimal
                             latency, X in {10,50,100}. Minimize.
  (b) latency_plus_max_util  nonlinear max functional (class b): steady
                             avg latency + alpha * max link utilization,
                             alpha scaled so the terms are comparable.
                             Minimize.
  (c) frac_congested_under_popp_failures
                             joint + rare-event (classes c+d): mean over
                             single-popp failures of the LP's congested
                             volume fraction. Minimize.

Two layers:

1. GENERIC-LP-CONTRACT implementations (solve_lp_frac_beyond_optimal,
   solve_lp_lat_plus_max_util, solve_lp_popp_failure_congestion): same
   signature and return contract as solve_lp_assignment's registered
   objectives (site_failure is the template), so they rope through
   Generic_Objective -> solve_generic_lp_with_failure_catch -> workers
   (path_distribution_computer) exactly like 'avg_latency'. Call
   `register()` to insert them into
   solve_lp_assignment.generic_lp_functions -- one line at driver AND
   worker startup wires them into training; no production files are
   modified. Objective names: 'frac_beyond_optimal',
   'lat_plus_max_util', 'popp_failure_congestion'. Tunables ride
   lp_kwargs (frac_beyond_x, mlu_alpha, popp_failure_sample_k).

2. Driver-side TRUSTED-SCORING wrappers (frac_users_beyond,
   latency_plus_max_util, frac_congested_under_popp_failures,
   evaluate_all): rescore_fork recipe (fresh process, RAY_ADDRESS=local,
   no Worker_Manager), for characterizing difficulty and re-ranking
   existing advertisements.

Pure math lives in _frac_beyond / _max_util_from_ret / vols-helpers so
unit tests (test_objectives_unit.py, no Gurobi) cover both layers.

CLI smoke (world knobs via env, same as everything else here):
    python -m experiments.model_error.objectives --seed 1 --dpsize small
"""
import argparse
import json
import os
import sys

import numpy as np

from solve_lp_assignment import obj_round

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Pure math helpers (unit-tested without Gurobi)
# ---------------------------------------------------------------------------

def _ug_vols_for(sas, n):
    """Volume vector matching a lats_by_ug of length n (workers hold
    subdeployments, the driver holds the whole deployment)."""
    for attr in ('whole_deployment_ug_vols', 'ug_vols'):
        v = getattr(sas, attr, None)
        if v is not None and len(np.asarray(v).flatten()) == n:
            return np.asarray(v, dtype=float).flatten()
    return np.ones(n)


def _perf_floor_best_lats(sas, n):
    """Capacity-blind per-UG optimal: min latency over each UG's routes.
    Available identically on driver and workers (ug_perfs); the
    capacity-aware alternative (LP under one-per-peering) is the
    driver-side best_lats_by_ug() below -- pass it via lp_kwargs
    best_lats when scoring."""
    perfs = getattr(sas, 'whole_deployment_ug_perfs', None) or sas.ug_perfs
    ugs = getattr(sas, 'whole_deployment_ugs', None) or sas.ugs
    if len(ugs) != n:
        ugs = sas.ugs
    return np.asarray([min(perfs[ug].values()) for ug in ugs], dtype=float)


def _frac_beyond(lats, best_lats, x, vols):
    gaps = np.asarray(lats, dtype=float) - np.asarray(best_lats, dtype=float)
    return float(np.average(gaps > x, weights=np.asarray(vols, dtype=float)))


def _max_util_from_ret(ret, caps, n_popps, sas=None):
    """Max over real links of assigned volume / capacity.

    Loads are rebuilt from paths_by_ug (vol_pct x ug volume) when
    possible: the ret['vols_by_poppi'] convention is inconsistent across
    LP layers -- driver LPs return it already divided by capacity
    (utilization, solve_lp_assignment.py ~:278/:1157) while the
    persistent worker LP returns raw volumes, so consuming it directly
    double-divides on the driver (verified 2026-08-14). paths_by_ug's
    vol_pct is a fraction of the UG's own volume in every layer, so it
    is convention-proof. Falls back to vols_by_poppi (assumed raw
    volumes) when paths_by_ug or sas is unavailable."""
    caps = np.asarray(caps, dtype=float).flatten()[:n_popps]
    pbu = ret.get('paths_by_ug')
    if pbu and sas is not None:
        vols = _ug_vols_for(sas, len(np.asarray(
            ret['lats_by_ug']).flatten()))
        loads = np.zeros(n_popps)
        for ug, pathvols in pbu.items():
            ui = int(ug) if isinstance(ug, (int, np.integer)) else None
            if ui is None or ui >= len(vols):
                continue
            for pv in pathvols:
                poppi, vol_pct = (pv if isinstance(pv, (list, tuple))
                                  else (pv, 1.0))
                if int(poppi) < n_popps:
                    loads[int(poppi)] += float(vol_pct) * float(vols[ui])
    else:
        vols_by_poppi = ret['vols_by_poppi']
        if isinstance(vols_by_poppi, dict):
            loads = np.zeros(n_popps)
            for poppi, v in vols_by_poppi.items():
                if int(poppi) < n_popps:
                    loads[int(poppi)] = float(np.sum(v))
        else:
            loads = np.asarray(
                vols_by_poppi, dtype=float).flatten()[:n_popps]
    with np.errstate(divide='ignore', invalid='ignore'):
        utils = np.where(caps > 0, loads / caps, 0.0)
    return float(np.max(utils)) if len(utils) else 0.0


# ---------------------------------------------------------------------------
# Generic-LP-contract objectives (registerable; worker-compatible).
# Template: solve_lp_assignment.solve_lp_assignment_site_failure.
# ---------------------------------------------------------------------------

_OWN_KWARGS = ('adv', 'best_lats', 'frac_beyond_x', 'mlu_alpha',
               'popp_failure_sample_k')


def _inner_kwargs(kwargs):
    return {k: v for k, v in kwargs.items() if k not in _OWN_KWARGS}


def solve_lp_frac_beyond_optimal(sas, routed_through_ingress, obj, **kwargs):
    """(a) objective = volume-weighted fraction of UGs more than X ms above
    their optimal latency (lower better). lp_kwargs: frac_beyond_x
    (default 10), best_lats (default: capacity-blind per-UG floor)."""
    from solve_lp_assignment import solve_generic_lp_with_failure_catch
    x = float(kwargs.get('frac_beyond_x', 10.0))
    steady = solve_generic_lp_with_failure_catch(
        sas, routed_through_ingress, 'avg_latency', **_inner_kwargs(kwargs))
    if not steady.get('solved'):
        return steady
    lats = np.asarray(steady['lats_by_ug'], dtype=float).flatten()
    best = kwargs.get('best_lats')
    if best is None:
        best = _perf_floor_best_lats(sas, len(lats))
    # SCULPTOR_FRAC_BEYOND_REL=0.10: threshold is 10% OF each UG's
    # optimal (Tom's "% of users within 10% of optimal", 2026-08-14)
    # instead of absolute-ms frac_beyond_x. Env so workers agree with
    # the driver without lp_kwargs plumbing.
    rel = os.environ.get('SCULPTOR_FRAC_BEYOND_REL')
    if rel is not None:
        x = float(rel) * np.asarray(best, dtype=float)
    out = dict(steady)
    # generic-LP convention: 'objective' is a BENEFIT (higher better;
    # avg_latency returns ~-avg_lat). Negate our minimize-quantity.
    out['steady_avg_lat'] = -float(steady['objective'])
    out['frac_beyond_x'] = float(rel) if rel is not None else x
    out['frac_beyond'] = _frac_beyond(lats, best, x, _ug_vols_for(sas, len(lats)))
    # SCALAR = CAPABILITY metric (2026-08-16, Tom's invariant: nothing
    # may score better than one-per-peering): min achievable volume-
    # weighted excess-ms beyond (optimal + x) over the adv's ingress
    # options -- the hinge LP (solve_lp_assignment.solve_min_hinge_excess,
    # canonical Gurobi home) is monotone in the option set, so opp is an
    # exact floor. The old assignment-derived frac stays available as the
    # 'frac_beyond' component (a constrained arm CAN beat opp on that one
    # -- 25 store rows did, caught on the dash). REL mode keeps per-UG
    # thresholds. SCULPTOR_FRACB_SCALAR=assign restores the old scalar.
    if os.environ.get('SCULPTOR_FRACB_SCALAR', 'hinge') == 'hinge':
        try:
            from solve_lp_assignment import solve_min_hinge_excess
            base = np.asarray(best, dtype=float)
            if np.ndim(x):                      # REL mode: per-UG threshold
                thr_base, thr_ms = base + np.asarray(x, dtype=float), 0.0
            else:
                thr_base, thr_ms = base, float(x)
            excess, _, fb_cap = solve_min_hinge_excess(
                sas, routed_through_ingress, thr_base, x_ms=thr_ms)
            if excess is not None:
                out['hinge_excess_ms'] = float(excess)
                out['frac_beyond_capability'] = (float(fb_cap)
                                                if fb_cap is not None else None)
                out['objective'] = obj_round(-float(excess))
                return out
        except Exception as e:
            print('[fracb] hinge LP failed ({}); assignment scalar'.format(e))
    out['objective'] = obj_round(-out['frac_beyond'])
    return out


def _min_mlu_from_rti(sas, rti):
    """Thin delegator to THE canonical MLU implementation
    (solve_lp_assignment.solve_min_mlu, Gurobi -- Tom 2026-08-15: one
    place to fix bugs). See its docstring for semantics: best-achievable
    peak utilization over the adv's per-prefix ingress options, monotone,
    opp floor <= 1/scale by anycast provisioning."""
    from solve_lp_assignment import solve_min_mlu
    mlu, _routable_frac = solve_min_mlu(sas, rti)
    return mlu


def solve_lp_lat_plus_max_util(sas, routed_through_ingress, obj, **kwargs):
    """(b) objective (lower better, returned negated per benefit
    convention) = routed_avg_lat + P*bad_frac + alpha*(MLU + bad_frac).

    REWORKED 2026-08-15 (Tom: stranded volume must hurt MORE than
    non-stranded, WITHOUT the 30k sentinel entering the scalar):
    - routed_avg_lat: vol-weighted avg latency over ROUTED volume only
      (sentinel-marked volume excluded from the average -- same
      decomposition as solve_lp_assignment._soft_bounded_objective).
    - P*bad_frac: the existing bounded congestion/no-route price
      (SCULPTOR_SOFT_CONG_PENALTY, default 50ms per unit fraction).
    - alpha*(MLU + STRAND_MULT*bad_frac): the utilization term charges
      stranded volume AS IF fully congested, so shedding volume can
      NEVER lower this term -- the pure-MLU stranding exploit (sparse
      beat opp by parking 72% of volume on the no-route pseudo-ingress)
      is monotonically unprofitable. alpha defaults to the vol-weighted
      per-UG optimal floor (terms comparable); SCULPTOR_LATMLU_STRAND_MULT
      (default 1.0) tunes the in-term stranding charge.
    Net: stranding a unit of volume costs ~(P + alpha) ms-equivalent vs
    its routed latency (~15-40) -- dominant but same order of magnitude,
    so gradients stay stable."""
    from solve_lp_assignment import solve_generic_lp_with_failure_catch
    from constants import NO_ROUTE_LATENCY
    steady = solve_generic_lp_with_failure_catch(
        sas, routed_through_ingress, 'avg_latency', **_inner_kwargs(kwargs))
    if not steady.get('solved'):
        return steady
    lats = np.asarray(steady['lats_by_ug'], dtype=float).flatten()
    vols = _ug_vols_for(sas, len(lats))
    alpha = kwargs.get('mlu_alpha')
    if alpha is None:
        alpha = float(np.average(_perf_floor_best_lats(sas, len(lats)),
                                 weights=vols))
    # MLU term = BEST-ACHIEVABLE peak utilization for this advertisement
    # (min-Y LP over the adv's per-prefix ingress options), NOT the peak
    # of the latency-optimal split (degenerate ~1.0 for every adv).
    # Provisioning is anycast*scale, so opp is mathematically <= 1/scale
    # (~0.909 at the default 1.1) — a hard sanity floor.
    # SCULPTOR_LATMLU_TERM=assign restores the old assignment-peak term.
    if os.environ.get('SCULPTOR_LATMLU_TERM', 'minlp') == 'minlp':
        try:
            mlu = _min_mlu_from_rti(sas, routed_through_ingress)
        except Exception as e:
            print('[lat_plus_max_util] min-MLU LP failed ({}); falling '
                  'back to assignment-peak'.format(e))
            mlu = None
        if mlu is None:
            mlu = _max_util_from_ret(steady, sas.link_capacities_arr,
                                     sas.n_popps, sas=sas)
    else:
        mlu = _max_util_from_ret(steady, sas.link_capacities_arr,
                                 sas.n_popps, sas=sas)
    # sentinel-scale decomposition (mirrors _soft_bounded_objective):
    # bad_w ~ 1 for pure-sentinel UGs, ~0 for real latencies.
    P = float(os.environ.get('SCULPTOR_SOFT_CONG_PENALTY', '50'))
    smult = float(os.environ.get('SCULPTOR_LATMLU_STRAND_MULT', '1.0'))
    tv = float(np.sum(vols))
    S = float(np.sum(lats * vols))
    bad_w = np.clip((lats - 200.0) / max(NO_ROUTE_LATENCY - 200.0, 1.0),
                    0.0, 1.0)
    B = float(np.sum(bad_w * vols))
    routed_lat = max(0.0, S - NO_ROUTE_LATENCY * B) / max(tv - B, 1e-9)
    bad_frac = B / max(tv, 1e-9)
    out = dict(steady)
    out['steady_avg_lat'] = routed_lat
    out['bad_frac'] = bad_frac
    out['max_util'] = mlu
    out['mlu_alpha'] = float(alpha)
    out['objective'] = obj_round(-1.0 * (routed_lat + P * bad_frac
                               + float(alpha) * (mlu + smult * bad_frac)))
    return out


def solve_lp_max_util(sas, routed_through_ingress, obj, **kwargs):
    """STANDALONE MLU objective, v2 (Tom 2026-08-16: 'implement MLU as a
    standalone objective ... ensure all users are routed ... much more
    weight on MLU than latency'). A first-class generic objective (use
    exactly like testing_priorities uses joint_latency_bulk_download):

        objective = -( A*minMLU + routed_lat + G*bad_frac )

    - minMLU: BEST-ACHIEVABLE peak utilization for the advertisement
      (solve_lp_assignment.solve_min_mlu, THE canonical Gurobi LP).
      Routing ALL reachable users is a hard conservation constraint
      inside every LP here -- shedding is never a solver freedom.
    - A = SCULPTOR_MLU_WEIGHT_MULT (default 10) x the vol-weighted
      per-UG optimal-latency floor: MLU carries ~10x the latency
      term's weight (~85-90% of the objective's dynamic range), so
      latency is a tie-break, not a goal.
    - G = 3*A per unit stranded fraction: an advertisement that leaves
      users with NO ingress at all (the only shedding channel) pays
      strictly more than any achievable MLU gain -- learning to
      not-route is provably unprofitable -- while staying at the
      tens-of-ms scale (NEVER the 30s sentinel; Tom's gradient-
      stability ruling).
    v1 (force_mlu fallback Y) was a latency-greedy concentration
    artifact, gameable and quarantined -- do not resurrect."""
    from solve_lp_assignment import solve_generic_lp_with_failure_catch
    from solve_lp_assignment import solve_min_mlu
    from constants import NO_ROUTE_LATENCY
    steady = solve_generic_lp_with_failure_catch(
        sas, routed_through_ingress, 'avg_latency', **_inner_kwargs(kwargs))
    if not steady.get('solved'):
        return steady
    lats = np.asarray(steady['lats_by_ug'], dtype=float).flatten()
    vols = _ug_vols_for(sas, len(lats))
    floor_mean = float(np.average(_perf_floor_best_lats(sas, len(lats)),
                                  weights=vols))
    A = float(os.environ.get('SCULPTOR_MLU_WEIGHT_MULT', '10')) * floor_mean
    G = 3.0 * A
    try:
        mlu, _ = solve_min_mlu(sas, routed_through_ingress)
    except Exception as e:
        print('[max_util] min-MLU LP failed ({}); assignment peak'.format(e))
        mlu = None
    if mlu is None:
        mlu = _max_util_from_ret(steady, sas.link_capacities_arr,
                                 sas.n_popps, sas=sas)
    # sentinel-scale decomposition (as lat_plus): routed-only latency +
    # stranded/congested volume fraction
    tv = float(np.sum(vols))
    S = float(np.sum(lats * vols))
    bad_w = np.clip((lats - 200.0) / max(NO_ROUTE_LATENCY - 200.0, 1.0),
                    0.0, 1.0)
    B = float(np.sum(bad_w * vols))
    routed_lat = max(0.0, S - NO_ROUTE_LATENCY * B) / max(tv - B, 1e-9)
    bad_frac = B / max(tv, 1e-9)
    out = dict(steady)
    out['max_util'] = float(mlu)
    out['steady_avg_lat'] = routed_lat
    out['bad_frac'] = bad_frac
    out['mlu_alpha'] = A
    out['objective'] = obj_round(-(A * float(mlu) + routed_lat + G * bad_frac))
    return out


def solve_lp_popp_failure_congestion(sas, routed_through_ingress, obj,
                                     **kwargs):
    """(c) objective = mean over single-popp failures of the LP's congested
    volume fraction (lower better; in [0,1]). Requires kwargs adv (as
    site_failure does); without it returns the steady solve unchanged.
    lp_kwargs: popp_failure_sample_k -- deterministic stride sample of
    scenarios for tractable training (default: exhaustive)."""
    from solve_lp_assignment import solve_generic_lp_with_failure_catch
    adv = kwargs.get('adv')
    steady = solve_generic_lp_with_failure_catch(
        sas, routed_through_ingress, 'avg_latency', **_inner_kwargs(kwargs))
    if not steady.get('solved') or adv is None:
        return steady
    sample_k = kwargs.get('popp_failure_sample_k')
    poppis = list(range(sas.n_popps))
    if sample_k and int(sample_k) < len(poppis):
        # deterministic stride, not RNG: same scenarios every call so
        # gradients see a stable objective (site_failure's rationale)
        step = max(1, len(poppis) // int(sample_k))
        poppis = poppis[::step][:int(sample_k)]
    fracs = []
    a = np.asarray(adv, dtype=float)
    for poppi in poppis:
        a_fail = a.copy()
        a_fail[poppi, :] = 0
        if a_fail.sum() == 0:
            fracs.append(1.0)
            continue
        fail_routed, _ = sas.calculate_ground_truth_ingress(a_fail)
        fail_ret = solve_generic_lp_with_failure_catch(
            sas, fail_routed, 'avg_latency', **_inner_kwargs(kwargs))
        fracs.append(float(fail_ret['fraction_congested_volume'])
                     if fail_ret.get('solved') else 1.0)
    out = dict(steady)
    # benefit convention: higher better -> negative mean congested fraction
    out['steady_avg_lat'] = -float(steady['objective'])
    out['popp_failure_n_scenarios'] = len(fracs)
    out['popp_failure_max_frac'] = float(np.max(fracs)) if fracs else 0.0
    out['popp_failure_mean_frac'] = float(np.mean(fracs)) if fracs else 0.0
    out['objective'] = obj_round(-out['popp_failure_mean_frac'])
    return out


def solve_lp_frozen_failure(sas, routed_through_ingress, obj, **kwargs):
    """(d) objective = steady avg latency + gamma_f * mean-over-failure
    BGP-fallback latency with FROZEN prefix assignments (Tom 2026-08-14):
    on failure users stay on their pinned prefix and fail over per BGP
    ingress priority -- no LP re-assignment. Reuses the existing eval
    mechanism (static_failure_eval.assess_static_failure_resilience;
    prices no-route and over-capacity popps at NO_ROUTE_LATENCY).
    Standard-contract fields come from the steady avg_latency LP; only
    'objective' is overridden. Tunables via env (worker-safe):
    SCULPTOR_FROZEN_GAMMA (default 1.0), SCULPTOR_FROZEN_WHICH
    (popps|pops, default popps)."""
    from solve_lp_assignment import solve_generic_lp_with_failure_catch
    from experiments.static_failure_eval import (
        assess_static_failure_resilience)
    steady = solve_generic_lp_with_failure_catch(
        sas, routed_through_ingress, 'avg_latency', **_inner_kwargs(kwargs))
    if not steady.get('solved'):
        return steady
    adv = kwargs.get('adv')
    if adv is None:
        return steady  # no advertisement context: fall back to steady
    gamma_f = float(os.environ.get('SCULPTOR_FROZEN_GAMMA', '1.0'))
    which = os.environ.get('SCULPTOR_FROZEN_WHICH', 'popps')
    res = assess_static_failure_resilience(
        sas, np.asarray(adv, dtype=float), which=which)
    out = dict(steady)
    out['steady_avg_lat'] = -float(steady['objective'])
    out['frozen_avg_lat_failure'] = float(res['avg_lat_failure'])
    out['frozen_frac_no_route_failure'] = float(
        res['frac_no_route_failure'])
    out['frozen_gamma'] = gamma_f
    # benefit convention (higher better): steady LP benefit minus the
    # frozen-failure latency penalty, scaled like the RB term.
    out['objective'] = obj_round(float(steady['objective'])
                        - gamma_f * float(res['avg_lat_failure']))
    return out


REGISTERED_OBJECTIVES = {
    'frac_beyond_optimal': solve_lp_frac_beyond_optimal,
    'lat_plus_max_util': solve_lp_lat_plus_max_util,
    'max_util': solve_lp_max_util,
    'popp_failure_congestion': solve_lp_popp_failure_congestion,
    'frozen_failure_latency': solve_lp_frozen_failure,
}


def register():
    """Insert these objectives into solve_lp_assignment's registry. Call at
    driver AND worker startup (workers import their own module copies) to
    make them selectable via generic_objective='<name>'."""
    from solve_lp_assignment import generic_lp_functions
    generic_lp_functions.update(REGISTERED_OBJECTIVES)
    return sorted(REGISTERED_OBJECTIVES)


# ---------------------------------------------------------------------------
# Driver-side trusted-scoring wrappers (rescore_fork recipe)
# ---------------------------------------------------------------------------

def _steady_ret(sas, adv):
    ret = sas.solve_lp_with_failure_catch(np.asarray(adv, dtype=float))
    if not ret.get('solved', True):
        return None
    return ret


def best_lats_by_ug(sas):
    """Capacity-aware optimal per-UG latency: LP under the one-per-peering
    advertisement (everything reachable, LP assigns best), matching the
    repo's optimal baseline convention."""
    ret = _steady_ret(sas, np.eye(sas.n_popps))
    assert ret is not None, 'one-per-peering LP unsolved'
    return np.asarray(ret['lats_by_ug'], dtype=float)


def frac_users_beyond(sas, adv, xs=(10, 50, 100), best_lats=None, ret=None):
    """(a) For each X: volume-weighted fraction of UGs whose latency is
    MORE than X ms above their optimal. Minimize. Indicator/threshold
    objective: linear in per-UG indicators but discontinuous in latency,
    so marginal beliefs must get individual users right near the
    threshold -- averaging no longer forgives per-user errors."""
    vols = np.asarray(sas.ug_vols, dtype=float)
    if best_lats is None:
        best_lats = best_lats_by_ug(sas)
    if ret is None:
        ret = _steady_ret(sas, adv)
    if ret is None:
        return {int(x): 1.0 for x in xs}
    lats = np.asarray(ret['lats_by_ug'], dtype=float)
    return {int(x): _frac_beyond(lats, best_lats, x, vols) for x in xs}


def max_link_utilization(sas, ret):
    return _max_util_from_ret(ret, sas.link_capacities_arr, sas.n_popps, sas=sas)


def latency_plus_max_util(sas, adv, alpha=None, best_lats=None):
    """(b) steady avg latency + alpha * max link utilization. Minimize.
    alpha calibration: default = volume-weighted mean OPTIMAL latency, so
    driving the hottest link from empty to full costs as much as
    doubling everyone's optimal latency -- comparable contributions by
    construction. Override with SCULPTOR_OBJ_MAXUTIL_ALPHA or the
    alpha argument."""
    vols = np.asarray(sas.ug_vols, dtype=float)
    if best_lats is None:
        best_lats = best_lats_by_ug(sas)
    if alpha is None:
        alpha = float(os.environ.get(
            'SCULPTOR_OBJ_MAXUTIL_ALPHA',
            float(np.average(best_lats, weights=vols))))
    ret = _steady_ret(sas, adv)
    if ret is None:
        return {'objective': float('inf'), 'avg_lat': float('inf'),
                'max_util': 1.0, 'alpha': float(alpha)}
    avg_lat = float(np.average(np.asarray(ret['lats_by_ug'], dtype=float),
                               weights=vols))
    mlu = max_link_utilization(sas, ret)
    return {'objective': avg_lat + alpha * mlu, 'avg_lat': avg_lat,
            'max_util': mlu, 'alpha': float(alpha)}


def frac_congested_under_popp_failures(sas, adv):
    """(c) Mean over single-popp failure scenarios of the LP's congested
    volume fraction (plus the max scenario). Minimize. Needs the JOINT
    re-landing of a failed popp's users -- the canonical
    marginals-insufficient objective."""
    a = np.asarray(adv, dtype=float)
    per_scen = []
    for popp in sas.popps:
        a2 = np.copy(a)
        a2[sas.popp_to_ind[popp], :] = 0
        if a2.sum() == 0:
            per_scen.append(1.0)
            continue
        ret = _steady_ret(sas, a2)
        per_scen.append(1.0 if ret is None
                        else float(ret['fraction_congested_volume']))
    return {'mean': float(np.mean(per_scen)),
            'max': float(np.max(per_scen)),
            'n_scenarios': len(per_scen),
            'per_scenario': per_scen}


def evaluate_all(sas, adv, xs=(10, 50, 100)):
    best = best_lats_by_ug(sas)
    ret = _steady_ret(sas, adv)
    return {
        'frac_users_beyond': frac_users_beyond(
            sas, adv, xs=xs, best_lats=best, ret=ret),
        'latency_plus_max_util': latency_plus_max_util(
            sas, adv, best_lats=best),
        'frac_congested_under_popp_failures':
            frac_congested_under_popp_failures(sas, adv),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--dpsize', default='small')
    args = ap.parse_args()

    # rescore_fork's exact isolation recipe
    os.environ['RAY_ADDRESS'] = 'local'
    os.environ.setdefault(
        'RAY_TMPDIR', '/tmp/ray_obj_{}'.format(os.getpid()))
    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(args.seed)
    os.environ.setdefault('MPLBACKEND', 'Agg')

    from constants import DEFAULT_EXPLORE
    from wrapper_eval import capacity
    from deployment_setup import get_random_deployment
    from sparse_advertisements_v3 import Sparse_Advertisement_Eval
    from helpers import deployment_to_prefixes

    dep = get_random_deployment(args.dpsize)
    dep['generic_objective'] = 'avg_latency'
    sas = Sparse_Advertisement_Eval(
        dep, verbose=False, lambduh=0, with_capacity=capacity,
        explore=DEFAULT_EXPLORE, using_resilience_benefit=False, gamma=0,
        n_prefixes=deployment_to_prefixes(dep),
        generic_objective='avg_latency')

    opp = np.eye(sas.n_popps)
    anycast = np.zeros((sas.n_popps, sas.n_popps))
    anycast[:, 0] = 1
    for name, adv in (('one_per_peering', opp), ('anycast', anycast)):
        out = evaluate_all(sas, adv)
        out['frac_congested_under_popp_failures'].pop('per_scenario')
        print('== {} =='.format(name))
        print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# Lexicographic priority/bulk capability pair (Tom-ratified 2026-08-18:
# "we NEEEED to optimize high-priority traffic. bulk is completely
# best-effort" — strict priority is a hard requirement, so the metric is
# a PAIR, never a weighted sum. Stage A: capability latency for the
# priority class (monotone; opp exact floor). Stage B: max bulk volume
# deliverable WITHOUT costing priority anything — optimized over the
# ENTIRE latency-optimal face, not one solver vertex (the vertex
# conditioning was why the legacy two-stage scalar let arms beat opp).
# ---------------------------------------------------------------------------

def prio_lex_pair(sas, routed_through_ingress, face_tol=1e-6):
    """Returns (L_star_benefit, bulk_frac) or (None, None).
    L_star_benefit: the avg_latency capability scalar (benefit
    convention, obj_round'd — IDENTICAL semantics to the lat family).
    bulk_frac: fraction of total bulk volume deliverable within TRUE
    link capacities by SOME latency-optimal priority routing."""
    from solve_lp_assignment import (solve_generic_lp_with_failure_catch,
                                     get_paths_by_ug, NO_PATH_INGRESS,
                                     obj_round)
    import gpshim as gp
    from scipy.sparse import lil_matrix
    from constants import NO_ROUTE_LATENCY
    steady = solve_generic_lp_with_failure_catch(
        sas, routed_through_ingress, 'avg_latency', no_persistent=True)
    if not steady.get('solved'):
        return None, None
    L_star = steady['objective']
    x_raw = np.asarray(steady['raw_solution'], dtype=float).flatten()

    available_paths, _ = get_paths_by_ug(sas, routed_through_ingress)
    n_paths = len(available_paths)
    if len(x_raw) > n_paths:
        x_raw = x_raw[1:]
    lats = np.empty(n_paths)
    for i, (ug, poppi) in enumerate(available_paths):
        if poppi == NO_PATH_INGRESS(sas):
            lats[i] = NO_ROUTE_LATENCY
        else:
            lats[i] = sas.whole_deployment_ug_perfs[ug][sas.popps[poppi]]
    face_val = float(np.dot(lats, x_raw))

    vols = np.asarray(sas.whole_deployment_ug_vols, dtype=float).flatten()
    bulk = np.asarray(sas.whole_deployment_ug_bulk_vols,
                      dtype=float).flatten()
    total_bulk = float(bulk.sum())
    if total_bulk <= 0:
        return L_star, None
    n_ug = sas.whole_deployment_n_ug
    n_popps = sas.n_popps + 1
    from solve_lp_assignment import _apply_capacity_headroom
    caps = np.concatenate([_apply_capacity_headroom(
        sas.link_capacities_arr.flatten(), sas), np.array([0.0])])

    # joint (x, b, short) LP over the latency-optimal face:
    #   x: priority path vols (conservation ==, lat.x <= face value;
    #      sentinel no-route path allowed, huge cap — steady semantics)
    #   b: bulk path vols on REAL ingresses only (sentinel barred:
    #      undelivered bulk must show up as shortfall, never fake-route)
    #   short: per-ug undelivered bulk;  min sum(short)
    nv = 2 * n_paths + n_ug
    A_eq = lil_matrix((2 * n_ug, nv))
    b_eq = np.concatenate([vols[:n_ug], bulk[:n_ug]])
    A_ub = lil_matrix((n_popps + 1, nv))
    caps_row = caps.copy()
    caps_row[NO_PATH_INGRESS(sas)] = 1e9
    b_ub = np.concatenate([caps_row, [face_val * (1 + 1e-9) + face_tol]])
    for pli, (ug, poppi) in enumerate(available_paths):
        ugi = sas.whole_deployment_ug_to_ind[ug]
        A_eq[ugi, pli] = 1.0
        A_ub[poppi, pli] = 1.0
        A_ub[n_popps, pli] = lats[pli]
        if poppi != NO_PATH_INGRESS(sas):
            A_eq[n_ug + ugi, n_paths + pli] = 1.0
            A_ub[poppi, n_paths + pli] = 1.0
    for ugi in range(n_ug):
        A_eq[n_ug + ugi, 2 * n_paths + ugi] = 1.0   # b + short == bulk
    model = gp.Model()
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = 15.0
    z = model.addMVar(nv, lb=0)
    model.addConstr(A_eq.tocsr() @ z == b_eq)
    model.addConstr(A_ub.tocsr() @ z <= b_ub)
    cvec = np.concatenate([np.zeros(2 * n_paths), np.ones(n_ug)])
    model.setObjective(cvec @ z, gp.GRB.MINIMIZE)
    model.optimize()
    if model.status != gp.GRB.OPTIMAL:
        print('[prio-lex] stage-B status {}; bulk_frac=None'.format(
            model.status))
        return L_star, None
    short = float(model.objVal)
    return L_star, obj_round(max(0.0, 1.0 - short / total_bulk))
