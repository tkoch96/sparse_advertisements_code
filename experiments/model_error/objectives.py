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
    out['objective'] = -out['frac_beyond']
    return out


def solve_lp_lat_plus_max_util(sas, routed_through_ingress, obj, **kwargs):
    """(b) objective = steady avg latency + alpha * max link utilization
    (lower better). lp_kwargs: mlu_alpha (default: vol-weighted mean of
    the per-UG optimal floor, making the two terms comparable)."""
    from solve_lp_assignment import solve_generic_lp_with_failure_catch
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
    mlu = _max_util_from_ret(steady, sas.link_capacities_arr, sas.n_popps, sas=sas)
    out = dict(steady)
    # benefit convention: steady['objective'] ~= -avg_lat (higher better),
    # so subtract the utilization penalty from it.
    out['steady_avg_lat'] = -float(steady['objective'])
    out['max_util'] = mlu
    out['mlu_alpha'] = float(alpha)
    out['objective'] = float(steady['objective']) - float(alpha) * mlu
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
    out['objective'] = -out['popp_failure_mean_frac']
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
    out['objective'] = (float(steady['objective'])
                        - gamma_f * float(res['avg_lat_failure']))
    return out


REGISTERED_OBJECTIVES = {
    'frac_beyond_optimal': solve_lp_frac_beyond_optimal,
    'lat_plus_max_util': solve_lp_lat_plus_max_util,
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
        dep, verbose=False, lambduh=0.00001, with_capacity=capacity,
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
