"""Vectorized PAINTER arm -- same greedy algorithm + belief semantics as
arms.run_painter, but the per-candidate (popp, prefix) gain evaluation is
numpy-vectorized so it scales to actual-10-size problems.

For each prefix column j we maintain, from the belief's candidate matrix:
    S[u] = sum of candidate latencies,  N[u] = candidate count
Adding popp p to column j changes, exactly:
  * kills: recorded (u, l, w=p) pairs with l currently a candidate in j
  * entry: p enters for u unless a recorded winner over p is active in j
so the new expected latency is (S - killed_lat + lat[:,p]*enter) /
(N - killed_cnt + enter), NO_ROUTE when the denominator hits zero.
Gain aggregation matches arms.run_painter's signed volume-weighted sum.
"""
import numpy as np

from .common import NO_ROUTE_LATENCY, threshold_a


def _col_cand_matrix(belief, col_bool):
    """(n_ug, n_popp) bool candidate matrix for one column (belief_fast's
    _col_struct semantics, materialized densely)."""
    cand = col_bool[None, :] & belief.reach
    if len(belief._cu):
        m = col_bool[belief._cw] & col_bool[belief._cl]
        if m.any():
            cand[belief._cu[m], belief._cl[m]] = False
    return cand


def run_painter_fast(problem, cfg, belief_cls=None):
    from .belief_fast import FastPreferenceBelief
    belief_cls = belief_cls or FastPreferenceBelief
    pb = problem
    lat0 = np.where(np.isfinite(pb.lat), pb.lat, 0.0)  # masked-safe latencies
    adv = np.zeros((pb.n_popp, pb.n_prefixes))
    adv[:, 0] = 1.0
    belief = belief_cls(pb)
    belief.measure(adv)
    trace = [pb.evaluate(adv)]

    it = 0
    for it in range(1, cfg.max_iter + 1):
        cols = []  # per prefix: (act_bool, cand, S, N, E)
        for j in range(pb.n_prefixes):
            act = adv[:, j] > .5
            if not act.any():
                cols.append((act, None, None, None,
                             np.full(pb.n_ug, float(NO_ROUTE_LATENCY))))
                continue
            cand = _col_cand_matrix(belief, act)
            N = cand.sum(axis=1)
            S = (cand * lat0).sum(axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                E = np.where(N > 0, S / np.maximum(N, 1), float(NO_ROUTE_LATENCY))
            cols.append((act, cand, S, N, E))
        E_all = np.stack([c[4] for c in cols], axis=1)      # (n_ug, n_pref)
        cur_lat = E_all.min(axis=1)

        best_gain, best_flip = cfg.min_gain_ms, None
        for j in range(1, pb.n_prefixes):                    # col 0 stays anycast
            act, cand, S, N, E = cols[j]
            others = [jj for jj in range(pb.n_prefixes) if jj != j]
            other_min = E_all[:, others].min(axis=1)
            for p in range(pb.n_popp):
                if adv[p, j] > .5:
                    continue
                if cand is None:
                    # empty column: p alone becomes the candidate set
                    new_e = np.where(belief.reach[:, p], lat0[:, p], float(NO_ROUTE_LATENCY))
                    gain = float(np.dot(pb.vols,
                                        cur_lat - np.minimum(other_min, new_e)))
                else:
                    S2, N2 = S, N
                    if len(belief._cu):
                        km = (belief._cw == p) & act[belief._cl]
                        if km.any():
                            ku, kl = belief._cu[km], belief._cl[km]
                            live = cand[ku, kl]
                            if live.any():
                                ku, kl = ku[live], kl[live]
                                dS = np.zeros(pb.n_ug); dN = np.zeros(pb.n_ug)
                                np.add.at(dS, ku, lat0[ku, kl])
                                np.add.at(dN, ku, 1.0)
                                S2 = S - dS
                                N2 = N - dN
                    enter = belief.reach[:, p].copy()
                    if len(belief._cu):
                        bm = (belief._cl == p) & act[belief._cw]
                        if bm.any():
                            enter[belief._cu[bm]] = False
                    S3 = S2 + lat0[:, p] * enter
                    N3 = N2 + enter
                    with np.errstate(invalid='ignore', divide='ignore'):
                        new_e = np.where(N3 > 0, S3 / np.maximum(N3, 1),
                                         float(NO_ROUTE_LATENCY))
                    gain = float(np.dot(pb.vols,
                                        cur_lat - np.minimum(other_min, new_e)))
                gain /= pb.total_vol
                if gain > best_gain:
                    best_gain, best_flip = gain, (p, j)
        if best_flip is None:
            break
        adv[best_flip] = 1.0
        belief.measure(adv)
        trace.append(pb.evaluate(adv))

    adv_t = threshold_a(adv)
    return {
        'arm': 'painter',
        'final_obj': pb.evaluate(adv_t),
        'trace': [float(x) for x in trace],
        'n_measurements': belief.n_measurements,
        'iters_run': int(it),
        'n_on': int(adv_t.sum()),
    }
