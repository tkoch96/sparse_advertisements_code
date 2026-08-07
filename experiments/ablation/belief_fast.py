"""Vectorized PreferenceBelief -- same model + API as belief.py, but the
per-column candidate computation and MC sampling are numpy-vectorized so
actual-10-scale problems (thousands of UGs, hundreds of popps, tens of
prefixes) stay tractable.

Constraint store: flat arrays (cu, cl, cw) meaning "for UG cu, popp cl
loses to popp cw". A popp l is dominated in advertised set A iff some
recorded winner w for (u, l) is also in A. Identical semantics to
belief.py's loses_to dicts.

Swapped in for belief.PreferenceBelief once validated (import in
run_ablation flows through belief.py, which re-exports this class when
SCULPTOR_ABLATION_FAST_BELIEF != '0').
"""
import numpy as np


class FastPreferenceBelief:
    def __init__(self, problem):
        self.problem = problem
        self.reach = np.isfinite(problem.lat)          # (n_ug, n_popp)
        # constraint arrays; grown in measure()
        self._cu = np.zeros(0, dtype=np.int64)
        self._cl = np.zeros(0, dtype=np.int64)
        self._cw = np.zeros(0, dtype=np.int64)
        self._pair_seen = set()                        # dedupe (u, l, w)
        self.n_measurements = 0
        self._measured_cols = set()
        self._col_cache = {}

    # -- compat with belief.PreferenceBelief -------------------------------
    @property
    def loses_to(self):
        """Per-UG {loser: set(winners)} view (built on demand; used only by
        the painter arm's incremental logic on small problems)."""
        out = [dict() for _ in range(self.problem.n_ug)]
        for u, l, w in zip(self._cu, self._cl, self._cw):
            out[u].setdefault(int(l), set()).add(int(w))
        return out

    def candidates(self, u, active_poppis):
        act = np.zeros(self.problem.n_popp, dtype=bool)
        act[list(active_poppis)] = True
        cand = act & self.reach[u]
        m = (self._cu == u)
        if m.any():
            l, w = self._cl[m], self._cw[m]
            dom = act[w]
            cand[l[dom & cand[l]]] = False
        return [int(p) for p in np.where(cand)[0]]

    # ----------------------------------------------------------------------
    def measure(self, adv):
        pb = self.problem
        adv_bool = np.asarray(adv) > 0.5
        self.n_measurements += 1
        new_u, new_l, new_w = [], [], []
        changed = False
        for j in range(adv_bool.shape[1]):
            col = adv_bool[:, j]
            if not col.any():
                continue
            key = col.tobytes()
            if key in self._measured_cols:
                continue
            self._measured_cols.add(key)
            changed = True
            active = np.where(col)[0]
            chosen = pb.gt_ingress_col(col)
            routed = np.where(chosen >= 0)[0]
            for u in routed:
                c = int(chosen[u])
                ru = self.reach[u]
                for p in active:
                    p = int(p)
                    if p == c or not ru[p]:
                        continue
                    k = (int(u), p, c)
                    if k not in self._pair_seen:
                        self._pair_seen.add(k)
                        new_u.append(u); new_l.append(p); new_w.append(c)
        if new_u:
            self._cu = np.concatenate([self._cu, np.asarray(new_u, dtype=np.int64)])
            self._cl = np.concatenate([self._cl, np.asarray(new_l, dtype=np.int64)])
            self._cw = np.concatenate([self._cw, np.asarray(new_w, dtype=np.int64)])
        if changed:
            self._col_cache.clear()

    # ----------------------------------------------------------------------
    def _col_struct(self, col_bool):
        """Cached vectorized candidate structure for one prefix column.

        Returns (det_u, det_p, multi_u, flat_cands, offsets):
          det_u/det_p     : UGs with exactly one candidate
          multi_u         : UGs with >= 2 candidates
          flat_cands      : concatenated candidate poppis for multi_u rows
          offsets         : CSR-style offsets into flat_cands (len = len(multi_u)+1)
        """
        key = col_bool.tobytes()
        try:
            return self._col_cache[key]
        except KeyError:
            pass
        cand = col_bool[None, :] & self.reach          # (n_ug, n_popp)
        if len(self._cu):
            # kill dominated losers: winner and loser both advertised
            act_w = col_bool[self._cw]
            act_l = col_bool[self._cl]
            m = act_w & act_l
            if m.any():
                cand[self._cu[m], self._cl[m]] = False
        counts = cand.sum(axis=1)
        det_rows = np.where(counts == 1)[0]
        det_p = np.argmax(cand[det_rows], axis=1) if len(det_rows) else np.zeros(0, dtype=np.int64)
        multi_u = np.where(counts >= 2)[0]
        if len(multi_u):
            rows, cols = np.nonzero(cand[multi_u])
            offsets = np.concatenate([[0], np.cumsum(counts[multi_u])]).astype(np.int64)
            flat_cands = cols.astype(np.int64)
        else:
            offsets = np.zeros(1, dtype=np.int64)
            flat_cands = np.zeros(0, dtype=np.int64)
        out = (det_rows.astype(np.int64), det_p.astype(np.int64),
               multi_u.astype(np.int64), flat_cands, offsets)
        self._col_cache[key] = out
        return out

    def col_options(self, col_bool):
        """belief.py-compatible view: (det_u, det_p, multi_list)."""
        det_u, det_p, multi_u, flat, off = self._col_struct(col_bool)
        multi = [(int(u), flat[off[i]:off[i + 1]]) for i, u in enumerate(multi_u)]
        return det_u, det_p, multi

    def sample_avail(self, adv_bool, rng):
        pb = self.problem
        avail = np.zeros((pb.n_ug, pb.n_popp), dtype=bool)
        for j in range(adv_bool.shape[1]):
            col = adv_bool[:, j]
            if not col.any():
                continue
            det_u, det_p, multi_u, flat, off = self._col_struct(col)
            if len(det_u):
                avail[det_u, det_p] = True
            n = len(multi_u)
            if n:
                counts = (off[1:] - off[:-1])
                idx = off[:-1] + (rng.random(n) * counts).astype(np.int64)
                avail[multi_u, flat[idx]] = True
        return avail

    def expected_prefix_latency(self, u, active_poppis):
        cands = self.candidates(u, active_poppis)
        if not cands:
            return None
        return float(np.mean(self.problem.lat[u, cands]))
