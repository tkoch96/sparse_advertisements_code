"""Speed-optimized continuous/greedy arms for large (actual-N) problems.

Same algorithms and belief semantics as arms.py; the only change is HOW
the MC objective is estimated during probing:

  * Per-column choices are drawn with a per-(column-pattern, sample) seeded
    RNG, so a column's realization is deterministic given the pattern --
    common random numbers across probes fall out automatically.
  * A ProbeContext holds, per sample, the count matrix
        C[u, p] = number of advertised columns whose realization sends u to p
    Flipping one coordinate only swaps one column's contribution:
        C' = C - contrib(old_col) + contrib(new_col)
    so each probe costs two sparse updates + a score instead of a full
    resample of every column.

Validated to produce EXACTLY the same estimates as a direct full
re-evaluation under the same sampling scheme (see validation in session
log); the scheme itself is statistically identical to estimators.mc_estimate
(different CRN stream, same model).
"""
import zlib

import numpy as np

from .common import ADVERTISEMENT_THRESHOLD, NO_ROUTE_LATENCY, threshold_a
from .arms import (ArmConfig, DESIRED_MAX_VAL, GRAD_CLIP_VAL, _mk_result,
                   _rescale, sculptor_init)
from .belief_fast import FastPreferenceBelief
from .estimators import entropy_of_distribution


def _waterfill_score(pb, avail, penalty_lat, with_capacity=True, rounds=3):
    """Volume-weighted avg latency for an availability matrix.

    Feasible case is exact (everyone on their best popp). When links
    overload, approximate the LP by water-filling: each overloaded link
    keeps a proportional share of the demand hitting it; spilled volume
    retries its next-best remaining option. Volume still unplaced after
    `rounds` is charged `penalty_lat` (bounded, not NO_ROUTE -- see
    penalty_lat docstring)."""
    lat_masked = np.where(avail, pb.lat, np.inf)
    best = lat_masked.min(axis=1)
    has = np.isfinite(best)
    # ESTIMATOR-only: no-route volume charged the BOUNDED penalty, not
    # NO_ROUTE_LATENCY -- 30000ms swamps the latency-scale gradients
    # (see feedback_gradient_stability). Ground-truth eval keeps the LP/NRL.
    lats = np.where(has, best, penalty_lat)
    if not with_capacity:
        return float(np.dot(lats, pb.vols)) / pb.total_vol
    choice = lat_masked.argmin(axis=1)
    loads = np.zeros(pb.n_popp)
    np.add.at(loads, choice[has], pb.vols[has])
    if (loads <= pb.caps + 1e-9).all():
        return float(np.dot(lats, pb.vols)) / pb.total_vol
    # --- overload: iterative proportional spill ---
    lat_w = lat_masked.copy()
    rv = pb.vols.copy()                       # remaining volume per UG
    rv[~has] = 0.0                            # no-route users held aside
    cost = float(penalty_lat * pb.vols[~has].sum()) if (~has).any() else 0.0
    cap_left = pb.caps.copy()
    for _ in range(rounds):
        active = rv > 1e-12
        if not active.any():
            break
        b = lat_w[active].min(axis=1)
        ok = np.isfinite(b)
        au = np.where(active)[0]
        # users with no remaining option: charge penalty
        dead = au[~ok]
        if len(dead):
            cost += float(rv[dead].sum() * penalty_lat)
            rv[dead] = 0.0
        au = au[ok]
        if not len(au):
            break
        ch = lat_w[au].argmin(axis=1)
        demand = np.zeros(pb.n_popp)
        np.add.at(demand, ch, rv[au])
        with np.errstate(divide='ignore', invalid='ignore'):
            keep = np.where(demand > 0, np.minimum(1.0, cap_left / np.maximum(demand, 1e-12)), 1.0)
        placed = rv[au] * keep[ch]
        cost += float(np.dot(placed, pb.lat[au, ch]))
        cap_left -= np.minimum(demand, cap_left)
        rv[au] -= placed
        # spilled users lose this option for the next round
        spill = keep[ch] < 1.0
        if spill.any():
            lat_w[au[spill], ch[spill]] = np.inf
    left = rv.sum()
    if left > 1e-9:
        cost += float(left * penalty_lat)
    return cost / pb.total_vol


def exact_score_avail(pb, avail):
    """Exact capacity LP score with vectorized construction (the
    common.Problem version builds its matrices with Python list appends,
    which dominates runtime at actual-N scale)."""
    import scipy.sparse as sp
    from scipy.optimize import linprog
    lat_masked = np.where(avail, pb.lat, np.inf)
    best = lat_masked.min(axis=1)
    has = np.isfinite(best)
    lats = np.where(has, best, float(NO_ROUTE_LATENCY))
    choice = lat_masked.argmin(axis=1)
    loads = np.zeros(pb.n_popp)
    np.add.at(loads, choice[has], pb.vols[has])
    if (loads <= pb.caps + 1e-9).all():
        return float(np.dot(lats, pb.vols)) / pb.total_vol
    ru, rp = np.nonzero(avail)
    n_opt = len(ru)
    n_var = n_opt + pb.n_ug                      # options + per-UG NRL slack
    c = np.concatenate([pb.lat[ru, rp], np.full(pb.n_ug, float(NO_ROUTE_LATENCY))])
    rows_eq = np.concatenate([ru, np.arange(pb.n_ug)])
    cols_eq = np.arange(n_var)
    A_eq = sp.csr_matrix((np.ones(n_var), (rows_eq, cols_eq)),
                         shape=(pb.n_ug, n_var))
    A_ub = sp.csr_matrix((np.ones(n_opt), (rp, np.arange(n_opt))),
                         shape=(pb.n_popp, n_var))
    res = linprog(c, A_ub=A_ub, b_ub=pb.caps, A_eq=A_eq, b_eq=pb.vols,
                  bounds=(0, None), method='highs')
    if not res.success:
        raise RuntimeError('exact_score_avail LP failed: ' + str(res.message))
    return float(res.fun / pb.total_vol)


def _col_realization(belief, col_bool, sample_key):
    """Deterministic (u_idx, p_idx) realization for one column pattern.
    sample_key: (mc_seed, sample_i)."""
    det_u, det_p, multi_u, flat, off = belief._col_struct(col_bool)
    n = len(multi_u)
    if n:
        rng = np.random.default_rng(
            (sample_key[0], sample_key[1], zlib.crc32(col_bool.tobytes())))
        counts = off[1:] - off[:-1]
        idx = off[:-1] + (rng.random(n) * counts).astype(np.int64)
        u = np.concatenate([det_u, multi_u])
        p = np.concatenate([det_p, flat[idx]])
        return u, p
    return det_u, det_p


class ProbeContext:
    """Per-iteration estimator: base advertisement + n_mc sampled count
    matrices; probes are evaluated as single-column swaps."""

    def __init__(self, problem, belief, base_ab, mc_seed, n_mc=5):
        self.pb = problem
        self.belief = belief
        self.base_ab = base_ab
        self.mc_seed = mc_seed
        self.n_mc = n_mc
        self.counts = []           # per sample: int16 (n_ug, n_popp)
        self.base_contrib = {}     # (j, sample) -> (u, p) of base col j
        for s in range(n_mc):
            C = np.zeros((problem.n_ug, problem.n_popp), dtype=np.int16)
            for j in range(base_ab.shape[1]):
                col = base_ab[:, j]
                if not col.any():
                    self.base_contrib[(j, s)] = (np.zeros(0, np.int64), np.zeros(0, np.int64))
                    continue
                u, p = _col_realization(belief, col, (mc_seed, s))
                self.base_contrib[(j, s)] = (u, p)
                np.add.at(C, (u, p), 1)
            self.counts.append(C)

    @property
    def penalty_lat(self):
        """Bounded congestion penalty for the ESTIMATOR only (the exact LP
        stays in ground-truth eval). Charging overflow at NO_ROUTE_LATENCY
        (30000ms) would swamp the latency gradients (scale mismatch); 3x
        the worst finite latency is strong but same-scale."""
        try:
            return self._penalty_lat
        except AttributeError:
            finite = self.pb.lat[np.isfinite(self.pb.lat)]
            self._penalty_lat = float(min(NO_ROUTE_LATENCY, 3 * finite.max()))
            return self._penalty_lat

    def _score(self, C, with_capacity=True):
        """Estimator score from a count matrix: exact when capacity is
        satisfied, water-filling approximation of the LP when overloaded."""
        return _waterfill_score(self.pb, C > 0, self.penalty_lat,
                                with_capacity=with_capacity)

    def estimate_base(self, with_capacity=True):
        return float(np.mean([self._score(C, with_capacity) for C in self.counts]))

    def _excl(self, j, s):
        """Best latency / choice / loads with column j removed from the
        base advertisement (cached per (j, sample))."""
        key = (j, s)
        try:
            return self._excl_cache[key]
        except AttributeError:
            self._excl_cache = {}
        except KeyError:
            pass
        pb = self.pb
        C = self.counts[s]
        u0, p0 = self.base_contrib[(j, s)]
        if len(u0):
            np.subtract.at(C, (u0, p0), 1)
        lat_masked = np.where(C > 0, pb.lat, np.inf)
        best = lat_masked.min(axis=1)
        choice = lat_masked.argmin(axis=1)
        has = np.isfinite(best)
        loads = np.zeros(pb.n_popp)
        np.add.at(loads, choice[has], pb.vols[has])
        if len(u0):
            np.add.at(C, (u0, p0), 1)
        out = (best, choice, has, loads)
        self._excl_cache[key] = out
        return out

    def estimate_col_swap(self, j, new_col, with_capacity=True):
        return float(np.mean(self.col_swap_vals(j, new_col, with_capacity)))

    def col_swap_vals(self, j, new_col, with_capacity=True):
        """Per-sample objective values with column j replaced by new_col.
        Incremental: O(rows-that-changed) per sample, full-LP fallback
        only when the fast capacity check fails."""
        pb = self.pb
        vals = []
        has_new = new_col.any()
        for s in range(self.n_mc):
            best, choice, has, loads = self._excl(j, s)
            if has_new:
                u1, p1 = _col_realization(self.belief, new_col, (self.mc_seed, s))
            else:
                u1 = p1 = np.zeros(0, np.int64)
            if len(u1):
                l1 = pb.lat[u1, p1]
                better = l1 < best[u1]
                rows, rp, rl = u1[better], p1[better], l1[better]
            else:
                rows = np.zeros(0, np.int64); rp = rows; rl = np.zeros(0)
            # volume-weighted score with the swapped rows (bounded no-route
            # charge, consistent with _waterfill_score's estimator semantics)
            lats = np.where(has, best, self.penalty_lat)
            base_dot = float(np.dot(lats, pb.vols))
            delta = float(np.dot(rl - lats[rows], pb.vols[rows]))
            val = (base_dot + delta) / pb.total_vol
            if with_capacity:
                new_loads = loads.copy()
                moved = rows[has[rows]]
                if len(moved):
                    np.add.at(new_loads, choice[moved], -pb.vols[moved])
                if len(rows):
                    np.add.at(new_loads, rp, pb.vols[rows])
                if not (new_loads <= pb.caps + 1e-9).all():
                    # overloaded: full water-fill on the swapped counts (~10ms)
                    C = self.counts[s]
                    u0, p0 = self.base_contrib[(j, s)]
                    if len(u0):
                        np.subtract.at(C, (u0, p0), 1)
                    if len(u1):
                        np.add.at(C, (u1, p1), 1)
                    val = self._score(C, with_capacity=True)
                    if len(u1):
                        np.subtract.at(C, (u1, p1), 1)
                    if len(u0):
                        np.add.at(C, (u0, p0), 1)
            vals.append(val)
        return vals

    def estimate_flip(self, p, j, value, with_capacity=True):
        new_col = self.base_ab[:, j].copy()
        new_col[p] = value
        return self.estimate_col_swap(j, new_col, with_capacity)

    def estimate_col_swap_exact(self, j, new_col):
        """Exact-LP estimate of a column swap (slow; used to re-rank a
        shortlist in the greedy arm)."""
        vals = []
        has_new = new_col.any()
        for s in range(self.n_mc):
            C = self.counts[s]
            u0, p0 = self.base_contrib[(j, s)]
            u1, p1 = (_col_realization(self.belief, new_col, (self.mc_seed, s))
                      if has_new else (np.zeros(0, np.int64), np.zeros(0, np.int64)))
            if len(u0):
                np.subtract.at(C, (u0, p0), 1)
            if len(u1):
                np.add.at(C, (u1, p1), 1)
            vals.append(exact_score_avail(self.pb, C > 0))
            if len(u1):
                np.subtract.at(C, (u1, p1), 1)
            if len(u0):
                np.add.at(C, (u0, p0), 1)
        return float(np.mean(vals))

    def estimate_base_exact(self):
        return float(np.mean([exact_score_avail(self.pb, C > 0) for C in self.counts]))


def _rescale_clip(a, g, cfg):
    """arms._rescale variant for spiky (capacity/no-route) gradients:
    saturate each coordinate at DESIRED_MAX_VAL, then always apply the
    amplify-to-one-flip logic. The repo's damp branch (x0.1/max) freezes
    ALL coordinates when ONE spikes, which left the continuous arms inert
    at actual-N scale."""
    g = np.clip(g, -DESIRED_MAX_VAL, DESIRED_MAX_VAL)
    return _rescale(a, g, cfg)


def _probe_gradient_fast(pb, belief, a, cfg, rng, last_results=None):
    """arms._probe_gradient with ProbeContext instead of independent
    mc_estimate calls, plus the repo's REMEASURE mechanism
    (gradients_latency_benefit's best_from_last_time): 40% of the probe
    budget re-probes last iteration's largest-|gradient| coordinates, so
    partial coordinate movement can accumulate across iterations. Without
    this, at actual-N scale (10k+ coords vs 60 probes) a coordinate is
    almost never probed twice and the continuous arms go inert.

    Returns (g, results) where results = {(p, j): grad} for persistence."""
    n = pb.n_popp * pb.n_prefixes
    probes = []
    if last_results:
        n_re = int(cfg.probe_budget * 0.4)   # repo: pct_explore=60
        for (p, j), val in sorted(last_results.items(), key=lambda kv: -abs(kv[1])):
            if len(probes) >= n_re:
                break
            if abs(val) < .01:
                continue  # not worth the cost (repo threshold)
            if abs(ADVERTISEMENT_THRESHOLD - a[p, j]) > ADVERTISEMENT_THRESHOLD * 7 / 10:
                continue  # saturated on/off (repo skip)
            probes.append((p, j))
    chosen = set(probes)
    flat = a.flatten()
    w = ADVERTISEMENT_THRESHOLD - np.abs(flat - ADVERTISEMENT_THRESHOLD) + .01
    w = w / w.sum()
    k = min(cfg.probe_budget - len(probes), n - len(probes))
    if k > 0:
        for ii in rng.choice(n, size=min(cfg.probe_budget, n), replace=False, p=w):
            p, j = divmod(int(ii), pb.n_prefixes)
            if (p, j) in chosen:
                continue
            probes.append((p, j))
            if len(probes) >= cfg.probe_budget:
                break
    mc_seed = int(rng.integers(2 ** 31))
    ab = a > ADVERTISEMENT_THRESHOLD
    ctx = ProbeContext(pb, belief, ab, mc_seed, n_mc=cfg.n_mc)
    g = np.zeros((pb.n_popp, pb.n_prefixes))
    results = {}
    for p, j in probes:
        est_on = ctx.estimate_flip(p, j, True)
        est_off = ctx.estimate_flip(p, j, False)
        x = a[p, j] - ADVERTISEMENT_THRESHOLD
        sig = cfg.sigmoid_k * np.exp(-cfg.sigmoid_k * x) / (1 + np.exp(-cfg.sigmoid_k * x)) ** 2
        gv = float(np.clip((est_on - est_off) * sig, -GRAD_CLIP_VAL, GRAD_CLIP_VAL))
        g[p, j] = gv
        results[(p, j)] = gv
    return g, results


def run_greedy_mc_fast(problem, cfg):
    pb = problem
    rng = np.random.default_rng(cfg.seed + 1000)
    adv = np.zeros((pb.n_popp, pb.n_prefixes))
    adv[:, 0] = 1.0
    belief = FastPreferenceBelief(pb)
    belief.measure(adv)
    trace = [pb.evaluate(adv)]
    strikes = 0
    it = 0
    for it in range(1, cfg.max_iter + 1):
        mc_seed = int(rng.integers(2 ** 31))
        ab = adv > .5
        ctx = ProbeContext(pb, belief, ab, mc_seed, n_mc=cfg.n_mc)
        offs = np.argwhere(~ab[:, 1:])
        if len(offs) == 0:
            break
        k = min(cfg.probe_budget, len(offs))
        sel = offs[rng.choice(len(offs), size=k, replace=False)]
        # greedy argmaxes a single gain, so every candidate gets the
        # exact-LP estimate (fast: vectorized construction, ~20ms/solve
        # at actual-10 scale; the feasible fast path is often free)
        base_exact = ctx.estimate_base_exact()
        best_gain, best_flip = cfg.min_gain_ms, None
        for p, jm in sel:
            j = int(jm) + 1
            new_col = ab[:, j].copy()
            new_col[int(p)] = True
            est = ctx.estimate_col_swap_exact(j, new_col)
            if base_exact - est > best_gain:
                best_gain, best_flip = base_exact - est, (int(p), j)
        if best_flip is None:
            strikes += 1
            if strikes >= cfg.greedy_patience:
                break
            trace.append(trace[-1])
            continue
        strikes = 0
        adv[best_flip] = 1.0
        belief.measure(adv)
        trace.append(pb.evaluate(adv))
    return _mk_result('greedy_mc', pb, adv, trace, belief, it)


def run_coord_mc_fast(problem, cfg):
    pb = problem
    rng = np.random.default_rng(cfg.seed + 2000)
    a = sculptor_init(pb, rng)
    belief = FastPreferenceBelief(pb)
    belief.measure(a)
    trace = [pb.evaluate(a)]
    last_thresh = threshold_a(a)
    it = 0
    last_results = None
    for it in range(1, cfg.max_iter + 1):
        g, last_results = _probe_gradient_fast(pb, belief, a, cfg, rng, last_results)
        if not np.any(g):
            trace.append(trace[-1])
            continue
        p, j = np.unravel_index(int(np.argmax(np.abs(g))), g.shape)
        g_single = np.zeros_like(g)
        g_single[p, j] = g[p, j]
        g_single = _rescale_clip(a, g_single, cfg)
        a = np.clip(a - cfg.alpha * g_single, 0, 1)
        th = threshold_a(a)
        if not np.array_equal(th, last_thresh):
            belief.measure(a)
            last_thresh = th
        trace.append(pb.evaluate(a))
    return _mk_result('coord_mc', pb, a, trace, belief, it)


def _max_info_measure_fast(pb, belief, a, cfg, rng):
    ab = a > ADVERTISEMENT_THRESHOLD
    n = pb.n_popp * pb.n_prefixes
    flat = a.flatten()
    w = ADVERTISEMENT_THRESHOLD - np.abs(flat - ADVERTISEMENT_THRESHOLD) + .01
    w = w / w.sum()
    k = min(cfg.n_ent_candidates, n)
    idx = rng.choice(n, size=k, replace=False, p=w)
    seed = int(rng.integers(2 ** 31))
    ctx = ProbeContext(pb, belief, ab, seed, n_mc=cfg.n_ent_samples)
    best_ent, best_flip = cfg.min_entropy, None
    for ii in idx:
        p, j = divmod(int(ii), pb.n_prefixes)
        new_col = ab[:, j].copy()
        new_col[p] = ~new_col[p]
        vals = ctx.col_swap_vals(j, new_col, with_capacity=False)
        ent = entropy_of_distribution(vals)
        if ent > best_ent:
            best_ent, best_flip = ent, (p, j)
    if best_flip is not None:
        cand = ab.copy()
        cand[best_flip] = ~cand[best_flip]
        belief.measure(cand)


def run_fullgrad_fast(problem, cfg, entropy=False, name='fullgrad'):
    pb = problem
    rng = np.random.default_rng(cfg.seed + (4000 if entropy else 3000))
    a = sculptor_init(pb, rng)
    last_a = a.copy()
    belief = FastPreferenceBelief(pb)
    belief.measure(a)
    trace = [pb.evaluate(a)]
    last_thresh = threshold_a(a)
    it = 0
    last_results = None
    for it in range(1, cfg.max_iter + 1):
        g, last_results = _probe_gradient_fast(pb, belief, a, cfg, rng, last_results)
        g = _rescale_clip(a, g, cfg)
        w = a - cfg.alpha * g + cfg.beta * (a - last_a)
        last_a = a
        a = np.clip(w, 0, 1)
        th = threshold_a(a)
        if not np.array_equal(th, last_thresh):
            belief.measure(a)
            last_thresh = th
        if entropy:
            _max_info_measure_fast(pb, belief, a, cfg, rng)
        trace.append(pb.evaluate(a))
    return _mk_result(name, pb, a, trace, belief, it)


def run_fullgrad_entropy_fast(problem, cfg):
    return run_fullgrad_fast(problem, cfg, entropy=True, name='fullgrad_entropy')


def get_fast_arm_funcs():
    from .painter_fast import run_painter_fast
    return {
        'painter': run_painter_fast,
        'greedy_mc': run_greedy_mc_fast,
        'coord_mc': run_coord_mc_fast,
        'fullgrad': run_fullgrad_fast,
        'fullgrad_entropy': run_fullgrad_entropy_fast,
    }
