"""The five ablation arms, PAINTER -> SCULPTOR, each adding one feature:

  1. painter           greedy single flip-ON, deterministic avg-of-candidates
                       estimate, capacity-blind (learns preferences from
                       measurements like painter_v5's outer loop)
  2. greedy_mc         + (d) monte-carlo objective model w/ capacity LP
  3. coord_mc          + (b) memory: continuous advertisement, single-
                       coordinate step per iter, threshold at the end
  4. fullgrad          + (c) direction: step on ALL probed coordinates at
                       once, with momentum (repo: w = a - alpha*g + beta*(a - a_last))
  5. fullgrad_entropy  + (a) entropic exploration: one extra max-entropy
                       measurement per iter (solve_max_information analogue)

All arms share the problem instance, the preference-belief model, the
probe budget, and the measurement machinery. Constants mirror
sparse_advertisements_v3.py (alpha=.01 for lambduh<=.01, beta=.3,
sigmoid_k=5, GRAD_CLIP_VAL=10, DESIRED_MAX_VAL=5).
"""
from dataclasses import dataclass

import numpy as np

from .common import ADVERTISEMENT_THRESHOLD, threshold_a
from .belief import PreferenceBelief
from .estimators import mc_estimate, outcome_distribution, entropy_of_distribution

GRAD_CLIP_VAL = 10.0
DESIRED_MAX_VAL = 5.0


@dataclass
class ArmConfig:
    max_iter: int = 200
    probe_budget: int = 60      # coordinate probes per iteration
    n_mc: int = 5               # MC_NUM in the repo
    alpha: float = 0.01         # set_alpha() for lambduh <= .01
    beta: float = 0.3           # gradient momentum
    sigmoid_k: float = 5.0      # heaviside gradient parameter
    n_ent_candidates: int = 40  # flips scored per max-info search
    n_ent_samples: int = 15     # samples per entropy estimate
    min_entropy: float = 0.05   # min_explore_value analogue
    min_gain_ms: float = 1e-3   # greedy arms: minimum accepted gain
    greedy_patience: int = 3    # stop after N gain-less iters (sampled cands)
    seed: int = 0               # arm-level RNG (probes, MC seeds)


def _mk_result(name, problem, adv, trace, belief, iters):
    adv_t = threshold_a(np.asarray(adv, dtype=float))
    return {
        'arm': name,
        'final_obj': problem.evaluate(adv_t),
        'trace': [float(x) for x in trace],
        'n_measurements': belief.n_measurements,
        'iters_run': int(iters),
        'n_on': int(adv_t.sum()),
    }


# ----------------------------------------------------------------------
# 1. PAINTER: greedy flip-ON by deterministic avg-of-candidates estimate
# ----------------------------------------------------------------------
def run_painter(problem, cfg):
    pb = problem
    adv = np.zeros((pb.n_popp, pb.n_prefixes))
    adv[:, 0] = 1.0  # anycast on the first prefix, as the repo painter
    belief = PreferenceBelief(pb)
    belief.measure(adv)
    trace = [pb.evaluate(adv)]
    ug_reach = [np.where(np.isfinite(pb.lat[:, p]))[0] for p in range(pb.n_popp)]
    from .common import NO_ROUTE_LATENCY

    it = 0
    for it in range(1, cfg.max_iter + 1):
        # Expected latency per (UG, prefix) under the belief
        exp_lat = np.full((pb.n_ug, pb.n_prefixes), float(NO_ROUTE_LATENCY))
        col_cands, col_active = {}, {}
        for j in range(pb.n_prefixes):
            colact = np.where(adv[:, j] > .5)[0]
            if len(colact) == 0:
                continue
            col_active[j] = set(int(x) for x in colact)
            cands_j = []
            for u in range(pb.n_ug):
                cands = belief.candidates(u, colact)
                cands_j.append(cands)
                if cands:
                    exp_lat[u, j] = pb.lat[u, cands].mean()
            col_cands[j] = cands_j
        cur_lat = exp_lat.min(axis=1)

        best_gain, best_flip = cfg.min_gain_ms, None
        for j in range(1, pb.n_prefixes):  # col 0 stays pure anycast
            others = [jj for jj in range(pb.n_prefixes) if jj != j]
            other_min = exp_lat[:, others].min(axis=1)
            actset = col_active.get(j, set())
            cands_j = col_cands.get(j)
            for p in range(pb.n_popp):
                if adv[p, j] > .5:
                    continue
                gain = 0.0
                for u in ug_reach[p]:
                    lt = belief.loses_to[u]
                    old = cands_j[u] if cands_j is not None else []
                    new_c = [q for q in old if p not in lt.get(q, ())]
                    pl = lt.get(p)
                    if pl is None or actset.isdisjoint(pl):
                        new_c = new_c + [p]
                    if not new_c:
                        # partial-order corner case: no direct winner known
                        new_c = [q for q in (actset | {p}) if np.isfinite(pb.lat[u, q])]
                    new_u = min(other_min[u], pb.lat[u, new_c].mean())
                    gain += (cur_lat[u] - new_u) * pb.vols[u]
                gain /= pb.total_vol
                if gain > best_gain:
                    best_gain, best_flip = gain, (p, j)
        if best_flip is None:
            break
        adv[best_flip] = 1.0
        belief.measure(adv)
        trace.append(pb.evaluate(adv))
    return _mk_result('painter', pb, adv, trace, belief, it)


# ----------------------------------------------------------------------
# 2. + (d): monte-carlo objective model (capacity-aware)
# ----------------------------------------------------------------------
def run_greedy_mc(problem, cfg):
    pb = problem
    rng = np.random.default_rng(cfg.seed + 1000)
    adv = np.zeros((pb.n_popp, pb.n_prefixes))
    adv[:, 0] = 1.0
    belief = PreferenceBelief(pb)
    belief.measure(adv)
    trace = [pb.evaluate(adv)]

    strikes = 0
    it = 0
    for it in range(1, cfg.max_iter + 1):
        mc_seed = int(rng.integers(2 ** 31))  # common random numbers
        ab = adv > .5
        base = mc_estimate(pb, belief, ab, mc_seed, n_mc=cfg.n_mc)
        offs = np.argwhere(~ab[:, 1:])  # (p, j-1); col 0 stays anycast
        if len(offs) == 0:
            break
        k = min(cfg.probe_budget, len(offs))
        sel = offs[rng.choice(len(offs), size=k, replace=False)]
        best_gain, best_flip = cfg.min_gain_ms, None
        for p, jm in sel:
            cand = ab.copy()
            cand[p, jm + 1] = True
            est = mc_estimate(pb, belief, cand, mc_seed, n_mc=cfg.n_mc)
            if base - est > best_gain:
                best_gain, best_flip = base - est, (int(p), int(jm) + 1)
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


# ----------------------------------------------------------------------
# Shared machinery for the continuous arms
# ----------------------------------------------------------------------
def sculptor_init(problem, rng):
    """init_advertisement('using_objective') analogue: anycast col 0,
    one-pop-per-prefix on the next n_pop cols, sparse random tail."""
    n_popp, n_pref = problem.n_popp, problem.n_prefixes
    a = .35 * np.ones((n_popp, n_pref)) + .2 * (rng.uniform(size=(n_popp, n_pref)) - .5)
    a[:, 0] = .55
    pops = sorted(set(popp[0] for popp in problem.popps))
    for i, pop in enumerate(pops[:n_pref - 1]):
        inds = [problem.popp_to_ind[popp] for popp in problem.popps if popp[0] == pop]
        a[inds, i + 1] = .55
    start = min(len(pops) + 1, n_pref)
    if start < n_pref:
        prob_ons = np.linspace(.05, .005, num=n_pref - start)
        for i in range(n_pref - start):
            on = rng.uniform(size=n_popp) < prob_ons[i]
            a[on, start + i] = .55
    a += .02 * (rng.uniform(size=a.shape) - .5)
    return np.clip(a, 0, 1)


def _probe_gradient(pb, belief, a, cfg, rng):
    """Sample probe coordinates (weighted toward near-threshold entries,
    like the repo's explore step) and finite-difference the MC objective
    through the heaviside sigmoid. Returns d(latency)/da."""
    n = pb.n_popp * pb.n_prefixes
    flat = a.flatten()
    w = ADVERTISEMENT_THRESHOLD - np.abs(flat - ADVERTISEMENT_THRESHOLD) + .01
    w = w / w.sum()
    k = min(cfg.probe_budget, n)
    idx = rng.choice(n, size=k, replace=False, p=w)
    mc_seed = int(rng.integers(2 ** 31))
    g = np.zeros((pb.n_popp, pb.n_prefixes))
    ab = a > ADVERTISEMENT_THRESHOLD
    for ii in idx:
        p, j = divmod(int(ii), pb.n_prefixes)
        on = ab.copy(); on[p, j] = True
        off = ab.copy(); off[p, j] = False
        est_on = mc_estimate(pb, belief, on, mc_seed, n_mc=cfg.n_mc)
        est_off = mc_estimate(pb, belief, off, mc_seed, n_mc=cfg.n_mc)
        x = a[p, j] - ADVERTISEMENT_THRESHOLD
        sig = cfg.sigmoid_k * np.exp(-cfg.sigmoid_k * x) / (1 + np.exp(-cfg.sigmoid_k * x)) ** 2
        g[p, j] = (est_on - est_off) * sig
    return np.clip(g, -GRAD_CLIP_VAL, GRAD_CLIP_VAL)


def _rescale(a, g, cfg):
    """Repo's 'ensure approximately one flip' rescale: amplify a small
    gradient until the nearest coordinate would just cross the threshold,
    damp a huge one."""
    max_val = float(np.max(np.abs(g)))
    if max_val == 0:
        return g
    if max_val < DESIRED_MAX_VAL:
        mask = np.abs(g) > 1e-3
        # movement per unit gradient is -alpha * g
        with np.errstate(divide='ignore', invalid='ignore'):
            alphas = (ADVERTISEMENT_THRESHOLD - a[mask]) / (cfg.alpha * -g[mask])
        alphas = alphas[alphas > 0]
        if len(alphas):
            mult = min(float(alphas.min()), DESIRED_MAX_VAL / max_val) * 1.0001
        else:
            mult = 1.0
        return g * mult
    return g * .1 / max_val


def run_coord_mc(problem, cfg):
    """+ (b) memory: continuous advertisement values, but only the single
    largest-|gradient| coordinate moves each iteration."""
    pb = problem
    rng = np.random.default_rng(cfg.seed + 2000)
    a = sculptor_init(pb, rng)
    belief = PreferenceBelief(pb)
    belief.measure(a)
    trace = [pb.evaluate(a)]
    last_thresh = threshold_a(a)

    it = 0
    for it in range(1, cfg.max_iter + 1):
        g = _probe_gradient(pb, belief, a, cfg, rng)
        if not np.any(g):
            trace.append(trace[-1])
            continue
        p, j = np.unravel_index(int(np.argmax(np.abs(g))), g.shape)
        g_single = np.zeros_like(g)
        g_single[p, j] = g[p, j]
        g_single = _rescale(a, g_single, cfg)
        a = np.clip(a - cfg.alpha * g_single, 0, 1)
        th = threshold_a(a)
        if not np.array_equal(th, last_thresh):
            belief.measure(a)
            last_thresh = th
        trace.append(pb.evaluate(a))
    return _mk_result('coord_mc', pb, a, trace, belief, it)


def _max_info_measure(pb, belief, a, cfg, rng):
    """solve_max_information analogue: among candidate single-flip
    advertisements, measure the one whose predicted objective
    distribution has maximum entropy (if above the floor)."""
    ab = a > ADVERTISEMENT_THRESHOLD
    n = pb.n_popp * pb.n_prefixes
    flat = a.flatten()
    w = ADVERTISEMENT_THRESHOLD - np.abs(flat - ADVERTISEMENT_THRESHOLD) + .01
    w = w / w.sum()
    k = min(cfg.n_ent_candidates, n)
    idx = rng.choice(n, size=k, replace=False, p=w)
    seed = int(rng.integers(2 ** 31))
    best_ent, best_adv = cfg.min_entropy, None
    for ii in idx:
        p, j = divmod(int(ii), pb.n_prefixes)
        cand = ab.copy()
        cand[p, j] = ~cand[p, j]
        vals = outcome_distribution(pb, belief, cand, seed, n_samples=cfg.n_ent_samples)
        ent = entropy_of_distribution(vals)
        if ent > best_ent:
            best_ent, best_adv = ent, cand
    if best_adv is not None:
        belief.measure(best_adv)


def run_fullgrad(problem, cfg, entropy=False, name='fullgrad'):
    """+ (c) direction: momentum step on all probed coordinates at once.
    With entropy=True, also + (a): one max-entropy measurement per iter."""
    pb = problem
    rng = np.random.default_rng(cfg.seed + (4000 if entropy else 3000))
    a = sculptor_init(pb, rng)
    last_a = a.copy()
    belief = PreferenceBelief(pb)
    belief.measure(a)
    trace = [pb.evaluate(a)]
    last_thresh = threshold_a(a)

    it = 0
    for it in range(1, cfg.max_iter + 1):
        g = _probe_gradient(pb, belief, a, cfg, rng)
        g = _rescale(a, g, cfg)
        w = a - cfg.alpha * g + cfg.beta * (a - last_a)
        last_a = a
        a = np.clip(w, 0, 1)
        th = threshold_a(a)
        if not np.array_equal(th, last_thresh):
            belief.measure(a)
            last_thresh = th
        if entropy:
            _max_info_measure(pb, belief, a, cfg, rng)
        trace.append(pb.evaluate(a))
    return _mk_result(name, pb, a, trace, belief, it)


def run_fullgrad_entropy(problem, cfg):
    return run_fullgrad(problem, cfg, entropy=True, name='fullgrad_entropy')


ARM_FUNCS = {
    'painter': run_painter,
    'greedy_mc': run_greedy_mc,
    'coord_mc': run_coord_mc,
    'fullgrad': run_fullgrad,
    'fullgrad_entropy': run_fullgrad_entropy,
}
ARM_ORDER = ['painter', 'greedy_mc', 'coord_mc', 'fullgrad', 'fullgrad_entropy']
