"""Preference belief shared by all learning arms.

Mirrors the repo's model (path_distribution_computer parent_tracker +
get_ingress_probabilities): latencies (ug_perfs) are known exactly, the
BGP preference ORDER is not. Measuring an advertisement reveals, per
prefix, which ingress each UG chose; that ingress is then known to beat
every other popp co-advertised on that prefix that the UG can reach
("parent" constraints). For an unmeasured configuration, a UG lands
uniformly at random on any advertised+reachable popp not dominated by a
known parent in the advertised set.

Per-column-pattern candidate sets are cached (invalidated when new
constraints arrive) because the arms evaluate many single-flip
perturbations of the same base advertisement.
"""
import numpy as np


class PreferenceBelief:
    def __init__(self, problem):
        self.problem = problem
        # loses_to[u][p] = set of popps known to beat popp p for UG u
        self.loses_to = [dict() for _ in range(problem.n_ug)]
        self.n_measurements = 0
        self._measured_cols = set()  # dedupe identical prefix columns
        self._version = 0
        self._col_cache = {}

    # ------------------------------------------------------------------
    def candidates(self, u, active_poppis):
        """Popps in `active_poppis` that UG u could land on under current
        knowledge: reachable and not dominated by a co-advertised parent."""
        lt = self.loses_to[u]
        lat_u = self.problem.lat[u]
        act = set(int(p) for p in active_poppis)
        out = []
        for p in act:
            if not np.isfinite(lat_u[p]):
                continue
            dom = lt.get(p)
            if dom is not None and not act.isdisjoint(dom):
                continue
            out.append(p)
        return sorted(out)

    # ------------------------------------------------------------------
    def measure(self, adv):
        """Reveal ground truth for the thresholded advertisement, update
        constraints. Counts one measurement (all prefixes at once, like
        the repo's measure_ingresses)."""
        pb = self.problem
        adv_bool = np.asarray(adv) > 0.5
        self.n_measurements += 1
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
            active = [int(p) for p in np.where(col)[0]]
            chosen = pb.gt_ingress_col(col)
            for u in np.where(chosen >= 0)[0]:
                c = int(chosen[u])
                lt = self.loses_to[u]
                for p in active:
                    if p == c or not np.isfinite(pb.lat[u, p]):
                        continue
                    lt.setdefault(p, set()).add(c)
        if changed:
            self._version += 1
            self._col_cache.clear()

    # ------------------------------------------------------------------
    def col_options(self, col_bool):
        """Cached per-UG candidate structure for one prefix column.

        Returns (det_u, det_p, multi) where det_u/det_p are int arrays for
        UGs whose ingress is fully determined, and multi is a list of
        (u, np.array(candidate_poppis)) for UGs with residual uncertainty.
        """
        key = col_bool.tobytes()
        try:
            return self._col_cache[key]
        except KeyError:
            pass
        active = [int(p) for p in np.where(col_bool)[0]]
        det_u, det_p, multi = [], [], []
        for u in range(self.problem.n_ug):
            cands = self.candidates(u, active)
            if not cands:
                continue
            if len(cands) == 1:
                det_u.append(u)
                det_p.append(cands[0])
            else:
                multi.append((u, np.array(cands, dtype=np.int64)))
        out = (np.array(det_u, dtype=np.int64), np.array(det_p, dtype=np.int64), multi)
        self._col_cache[key] = out
        return out

    def sample_avail(self, adv_bool, rng):
        """One monte-carlo realization as an availability matrix
        (n_ug x n_popp bool): the popps each UG would obtain."""
        pb = self.problem
        avail = np.zeros((pb.n_ug, pb.n_popp), dtype=bool)
        for j in range(adv_bool.shape[1]):
            col = adv_bool[:, j]
            if not col.any():
                continue
            det_u, det_p, multi = self.col_options(col)
            if len(det_u):
                avail[det_u, det_p] = True
            for u, cands in multi:
                avail[u, cands[rng.integers(len(cands))]] = True
        return avail

    def expected_prefix_latency(self, u, active_poppis):
        """PAINTER-style estimate: mean latency over still-possible
        candidates for one prefix (None if no route)."""
        cands = self.candidates(u, active_poppis)
        if not cands:
            return None
        return float(np.mean(self.problem.lat[u, cands]))
