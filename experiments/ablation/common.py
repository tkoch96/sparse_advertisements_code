"""Shared problem representation + ground-truth evaluator for the
PAINTER -> SCULPTOR feature-ladder ablation.

Standalone: reuses only `deployment_setup.get_random_deployment` (seeded)
and constants from the repo. Ground-truth routing and the capacity LP are
reimplemented here with scipy (HiGHS) so the ablation does not consume
Gurobi WLS sessions and parallelizes freely.

Semantics mirror the repo:
  * Per prefix, a UG lands on the advertised+reachable popp with the
    highest preference value (pref = n_popp + 1 - ingress_priority, i.e.
    lowest priority rank wins), as in
    `optimal_adv_wrapper.calculate_ground_truth_ingress`.
  * The user->prefix volume assignment minimizes volume-weighted average
    latency subject to link capacities; volume that cannot fit anywhere
    is charged NO_ROUTE_LATENCY (the repo's failure-catch LP semantics).
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from constants import NO_ROUTE_LATENCY  # noqa: E402

ADVERTISEMENT_THRESHOLD = 0.5


def threshold_a(a):
    return (a > ADVERTISEMENT_THRESHOLD).astype(np.float64)


class Problem:
    """Immutable ground-truth view of one random 'small' deployment."""

    def __init__(self, seed, dpsize='small'):
        # SCULPTOR_DEPLOYMENT_SEED pins both np.random and random inside
        # get_random_deployment, exactly as the repo drivers do.
        os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(seed)
        from deployment_setup import get_random_deployment
        from helpers import deployment_to_prefixes
        dep = get_random_deployment(dpsize)

        self.seed = seed
        self.dpsize = dpsize
        self.deployment = dep  # kept for the real-sparse reference arm

        self.ugs = list(dep['ugs'])
        self.n_ug = len(self.ugs)
        self.vols = np.array([dep['ug_to_vol'][ug] for ug in self.ugs], dtype=np.float64)
        self.total_vol = float(self.vols.sum())

        self.popps = list(dep['popps'])
        self.n_popp = len(self.popps)
        popp_to_ind = {popp: i for i, popp in enumerate(self.popps)}
        self.popp_to_ind = popp_to_ind

        self.n_prefixes = int(deployment_to_prefixes(dep))

        # lat[u, p]: latency of ug u through popp p; +inf if unreachable.
        # pref[u, p]: BGP preference, higher wins; -inf if unreachable.
        self.lat = np.full((self.n_ug, self.n_popp), np.inf)
        self.pref = np.full((self.n_ug, self.n_popp), -np.inf)
        prios = dep['ingress_priorities']
        for ui, ug in enumerate(self.ugs):
            for popp, l in dep['ug_perfs'][ug].items():
                pi = popp_to_ind[popp]
                self.lat[ui, pi] = l
                self.pref[ui, pi] = self.n_popp + 1 - prios[ug][popp]
        self.reachable = [np.where(np.isfinite(self.lat[u]))[0] for u in range(self.n_ug)]

        self.caps = np.array([dep['link_capacities'][popp] for popp in self.popps],
                             dtype=np.float64)

        self._eval_cache = {}

    # ------------------------------------------------------------------
    # Ground-truth routing
    # ------------------------------------------------------------------
    def gt_ingress_col(self, active_bool):
        """True ingress per UG for one prefix advertising `active_bool`
        (bool array over popps). Returns int array, -1 = no route."""
        if not active_bool.any():
            return np.full(self.n_ug, -1, dtype=np.int64)
        masked = np.where(active_bool[None, :], self.pref, -np.inf)
        choice = np.argmax(masked, axis=1)
        ok = masked[np.arange(self.n_ug), choice] > -np.inf
        return np.where(ok, choice, -1)

    def gt_avail(self, adv):
        """Availability matrix (n_ug x n_popp bool): the popps actually
        offered by the thresholded advertisement (one true ingress per
        prefix, per the ground-truth preference order)."""
        adv_bool = np.asarray(adv) > ADVERTISEMENT_THRESHOLD
        avail = np.zeros((self.n_ug, self.n_popp), dtype=bool)
        for j in range(adv_bool.shape[1]):
            col = self.gt_ingress_col(adv_bool[:, j])
            ok = col >= 0
            avail[np.where(ok)[0], col[ok]] = True
        return avail

    def score_avail(self, avail, with_capacity=True):
        """Objective (avg latency, ms) when each UG may use exactly the
        popps flagged in `avail`. Fast path: everyone takes their best
        option; falls back to the capacity LP only if a link overflows."""
        lat_masked = np.where(avail, self.lat, np.inf)
        best = lat_masked.min(axis=1)
        has = np.isfinite(best)
        lats = np.where(has, best, NO_ROUTE_LATENCY)
        if not with_capacity:
            return float(np.dot(lats, self.vols) / self.total_vol)
        choice = np.argmin(lat_masked, axis=1)
        loads = np.zeros(self.n_popp)
        np.add.at(loads, choice[has], self.vols[has])
        if (loads <= self.caps + 1e-9).all():
            return float(np.dot(lats, self.vols) / self.total_vol)
        return self._solve_lp([list(np.where(avail[u])[0]) for u in range(self.n_ug)])

    # ------------------------------------------------------------------
    # Objective: capacity-constrained volume-weighted average latency
    # ------------------------------------------------------------------
    def assign_and_score(self, options_per_u):
        """min avg latency (ms) routing each UG's volume over its options,
        subject to link capacities; unroutable volume gets NO_ROUTE_LATENCY."""
        # Fast path: everyone takes their best-latency option.
        best_lat = np.full(self.n_ug, NO_ROUTE_LATENCY, dtype=np.float64)
        best_popp = np.full(self.n_ug, -1, dtype=np.int64)
        for u, opts in enumerate(options_per_u):
            if opts:
                lats = self.lat[u, opts]
                k = int(np.argmin(lats))
                best_lat[u] = lats[k]
                best_popp[u] = opts[k]
        loads = np.zeros(self.n_popp)
        routed = best_popp >= 0
        np.add.at(loads, best_popp[routed], self.vols[routed])
        if (loads <= self.caps + 1e-9).all():
            return float(np.dot(best_lat, self.vols) / self.total_vol)
        return self._solve_lp(options_per_u)

    def _solve_lp(self, options_per_u):
        # Variables: x[u, o] >= 0 per option, + slack s_u (NO_ROUTE volume).
        c, rows_eq, cols_eq, x_popp, x_of = [], [], [], [], []
        var_i = 0
        for u, opts in enumerate(options_per_u):
            for p in opts:
                c.append(self.lat[u, p])
                rows_eq.append(u); cols_eq.append(var_i)
                x_popp.append(p); x_of.append(u)
                var_i += 1
            # slack
            c.append(NO_ROUTE_LATENCY)
            rows_eq.append(u); cols_eq.append(var_i)
            x_popp.append(-1); x_of.append(u)
            var_i += 1
        n_var = var_i
        A_eq = sp.csr_matrix((np.ones(n_var), (rows_eq, cols_eq)),
                             shape=(self.n_ug, n_var))
        b_eq = self.vols
        ub_rows = [i for i in range(n_var) if x_popp[i] >= 0]
        A_ub = sp.csr_matrix((np.ones(len(ub_rows)),
                              ([x_popp[i] for i in ub_rows], ub_rows)),
                             shape=(self.n_popp, n_var))
        res = linprog(np.array(c), A_ub=A_ub, b_ub=self.caps,
                      A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
        if not res.success:
            raise RuntimeError('ablation LP failed: {}'.format(res.message))
        return float(res.fun / self.total_vol)

    def evaluate(self, adv):
        """Ground-truth avg latency (ms) of a (thresholded) advertisement."""
        adv_bool = np.asarray(adv) > ADVERTISEMENT_THRESHOLD
        key = adv_bool.tobytes()
        try:
            return self._eval_cache[key]
        except KeyError:
            pass
        val = self.score_avail(self.gt_avail(adv_bool))
        self._eval_cache[key] = val
        return val

    def one_per_peering_objective(self):
        """Idealized reference: every popp on its own prefix -> every UG can
        use any reachable popp (capacity still enforced)."""
        return self.assign_and_score([list(r) for r in self.reachable])
