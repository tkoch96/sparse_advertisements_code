"""Byte-equivalence fence for the scenario-options loop elimination
(Tom 2026-08-25: 'anything that eliminates a loop'). The OLD
implementations of _compute_scenario_options and
_sample_scenario_realizations are frozen below verbatim (pre-e8a8f5c+);
both old and new run on the same seeded worker over init-like, flipped,
and popp-row-zero advertisements with identical RNG streams, and the
full routed_through_ingress nested dicts must compare EQUAL.

    python unit_tests/test_scenario_options_equiv.py [--pickle ...]
"""
import copy, os, sys, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core.path_distribution_computer import LP_SOLVE_DEBUG, LBX_DENSITY

# ---------------------------------------------------------------------
# frozen OLD implementations (verbatim snapshot)
# ---------------------------------------------------------------------
class _Old:
    def _compute_scenario_options(self, a, verb=False, **kwargs):
        """Populate self.rti_data (per-(ug,prefix) ingress options + probabilities)
        for advertisement `a`. Deterministic; pattern-cached."""
        ts_total = time.time()
        _dbg_pc0 = len(getattr(self, 'pattern_cache', {}))

        # --- 1. Initialize Containers ---
        # Instead of nested dicts, we build the flat lists required for vectorization directly.
        self.rti_data = {
            "meta_data": [],  # List of tuples: (ui, pref_i, ug_name)
            "all_probs": [],  # legacy per-scenario lists (sim_rti_better)
            "all_poppis": [],
            "blocks": [],     # [(lengths:int16[], choices_pad:int16[nxm])]
        }

        # Ensure persistent cache exists (persist this across function calls)
        if not hasattr(self, 'pattern_cache'):
            self.pattern_cache = {}

        # Local variable speedups to avoid self lookups in loop
        ugs = self.whole_deployment_ugs
        # Assumed to be {ui: [poppi, poppi...]} or {ui: {poppi: data}}
        ui_to_poppi = self.whole_deployment_ui_to_poppi 

        # --- 2. Process Availability Matrix (a) ---
        # Assuming threshold_a logic is effectively: > 1e-6 means active
        a_log = (a > 1e-6) 

        # Iterate over prefixes (columns of a)
        for pref_i in range(a.shape[1]):
            col = a_log[:, pref_i]
            
            # Optimization: If no POPs are active for this prefix, skip entirely
            if not np.any(col):
                continue

            # Create a hashable signature for this availability state
            tloga = tuple(col)

            # --- CACHE CHECK ---
            key = np.packbits(col).tobytes()
            if key in self.pattern_cache:
                # HIT: compact padded entry (uis:int32, lengths:int16,
                # choices_pad:int16[n_scen x max_n], -1 padded). Appended
                # as a BLOCK; _sample_scenario_realizations assembles the
                # padded sampling matrices with vectorized copies instead
                # of per-scenario python (Tom 2026-08-19 pattern_cache
                # compaction: ~7x RAM, hit path faster than list appends).
                uis_e, lens_e, pad_e = self.pattern_cache[key]
                md = self.rti_data["meta_data"]
                for ui in uis_e:
                    md.append((ui, pref_i, ugs[ui]))
                self.rti_data["blocks"].append((lens_e, pad_e))
                continue

            # --- CACHE MISS: Calculate Logic (VECTORIZED, Tom 2026-08-19
            # startup_optimizations: this per-UG python loop was 23% of
            # every cold solve at actual-25 per the line-level flamegraph)
            # One-time CSR of ui -> potential popps; per miss the valid
            # set is a boolean fancy-index + reduceat, no per-UG python.
            active_poppis = np.where(col)[0]
            active_poppis_set = set(active_poppis)

            if not hasattr(self, '_uipop_csr'):
                _flat, _offs = [], [0]
                for _ui in range(self.whole_deployment_n_ug):
                    _flat.extend(ui_to_poppi[_ui])
                    _offs.append(len(_flat))
                self._uipop_csr = (
                    np.asarray(_flat, dtype=np.int32),
                    np.asarray(_offs, dtype=np.int64))
            _flat, _offs = self._uipop_csr

            # Blocked (ui, child) pairs from active parents. Compact path
            # (SCULPTOR_COMPACT_PT, Tom 2026-08-20): _pt_csr groups rows by
            # parent, so we touch only ACTIVE parents' rows — the legacy
            # python scan walked EVERY (ug,child,parent) entry per miss,
            # which scales ~n_ug*n_popp^2 with measurements (354MB and
            # millions of entries at actual-25 late-run).
            blocked_user_child = set()
            _csr = getattr(self, '_pt_csr', None)
            if _csr is not None:
                _pt_parents, _pt_offs, _pt_rows = _csr
                if _pt_parents.shape[0]:
                    for _j in np.nonzero(col[_pt_parents])[0]:
                        for _bui, _bchild in _pt_rows[
                                _pt_offs[_j]:_pt_offs[_j + 1]].tolist():
                            blocked_user_child.add((_bui, _bchild))
            else:
                # legacy dict path (SCULPTOR_COMPACT_PT=0 or no update yet)
                for ug, child, parent in self.parent_tracker:
                    parenti = self.popp_to_ind[parent]
                    if parenti in active_poppis_set:
                        ui = self.whole_deployment_ug_to_ind[ug]
                        childi = self.popp_to_ind[child]
                        blocked_user_child.add((ui, childi))

            # 3. Build routing for this state — fully vectorized.
            mask = col[_flat]                     # popp physically up
            if blocked_user_child:
                # sparse corrections: locate each blocked (ui, child)
                # inside ui's CSR segment and clear it
                for (_bui, _bchild) in blocked_user_child:
                    seg = slice(_offs[_bui], _offs[_bui + 1])
                    mask[seg] &= (_flat[seg] != _bchild)
            lens_all = np.add.reduceat(
                mask.astype(np.int16),
                _offs[:-1].clip(max=max(len(_flat) - 1, 0)))
            # reduceat quirk: empty segments (offs[i]==offs[i+1]) copy the
            # next element — zero them explicitly
            _empty = (_offs[1:] - _offs[:-1]) == 0
            if _empty.any():
                lens_all[_empty] = 0
            keep = lens_all > 0
            uis_e = np.where(keep)[0].astype(np.int32)
            lens_e = lens_all[keep].astype(np.int16)
            n_scen = int(uis_e.shape[0])
            valid_flat = _flat[mask]
            max_n = int(lens_e.max()) if n_scen else 0
            pad_e = np.full((n_scen, max_n), -1, dtype=np.int16)
            if n_scen:
                _colidx = np.arange(max_n)[None, :] < lens_e[:, None]
                pad_e[_colidx] = valid_flat
            md = self.rti_data["meta_data"]
            for _ui in uis_e:
                md.append((int(_ui), pref_i, ugs[_ui]))
            self.pattern_cache[np.packbits(col).tobytes()] = (
                uis_e, lens_e, pad_e)
            self.rti_data["blocks"].append((lens_e, pad_e))

        self.timing['pmat_organize'] += time.time() - ts_total
        if LP_SOLVE_DEBUG and getattr(self, 'worker_i', -1) in (0, 'drv'):
            print('[lpdbg-pc] w={} pattern_misses={} pattern_total={} '
                  't_scen={:.3f}s'.format(
                      getattr(self, 'worker_i', 'drv'),
                      len(self.pattern_cache) - _dbg_pc0,
                      len(self.pattern_cache), time.time() - ts_total),
                  flush=True)


    def _sample_scenario_realizations(self):
        """Monte-carlo draw of self.MC_NUM joint route realizations from the
        scenario options in self.rti_data (populated by
        _compute_scenario_options). Returns routed_through_ingress:
        {mc_index: {prefix: {ug: popp}}}."""
        # --- 3. Vectorized Simulation (Previously sim_rti_better) ---
        # Now self.rti_data is fully populated. We proceed with the vectorized selection.

        blocks = self.rti_data.get("blocks") or []
        if blocks:
            # Compact-block path (pattern-cache producer, Tom 2026-08-19):
            # assemble padded matrices with vectorized block copies.
            # Arithmetic matches the legacy path exactly — P rows are the
            # same uniform [1/n]*n float64 values, cumsum'd identically.
            lens_all = np.concatenate([b[0] for b in blocks]).astype(np.int64)
            self.rti_data["num_scenarios"] = int(lens_all.shape[0])
            if self.rti_data["num_scenarios"] == 0:
                return {}
            self.rti_data["max_choices"] = int(lens_all.max())
            P_matrix = np.zeros((self.rti_data["num_scenarios"], self.rti_data["max_choices"]))
            self.rti_data["choices_matrix"] = np.full(
                (self.rti_data["num_scenarios"], self.rti_data["max_choices"]), -1, dtype=int)
            row = 0
            for lens_e, pad_e in blocks:
                nrow, ncol = pad_e.shape
                self.rti_data["choices_matrix"][row:row + nrow, :ncol] = pad_e
                row += nrow
            mask = (np.arange(self.rti_data["max_choices"])[None, :]
                    < lens_all[:, None])
            P_matrix[mask] = np.repeat(1.0 / lens_all, lens_all)
        else:
            self.rti_data["num_scenarios"] = len(self.rti_data["all_probs"])
            if self.rti_data["num_scenarios"] == 0:
                return {}

            self.rti_data["max_choices"] = max(len(p) for p in self.rti_data["all_probs"])

            # Create Padded Matrix
            P_matrix = np.zeros((self.rti_data["num_scenarios"], self.rti_data["max_choices"]))
            self.rti_data["choices_matrix"] = np.full((self.rti_data["num_scenarios"], self.rti_data["max_choices"]), -1, dtype=int)

            for i, (probs, pops) in enumerate(zip(self.rti_data["all_probs"], self.rti_data["all_poppis"])):
                n = len(probs)
                P_matrix[i, :n] = probs
                self.rti_data["choices_matrix"][i, :n] = pops

        # CDF Construction
        cdf = np.cumsum(P_matrix, axis=1)
        cdf[:, -1] = 1.0 # Force sum to 1.0 to avoid float precision issues

        # Offset Trick for Vectorized Search
        # Shifts the values of every row so we can search a single flattened array
        offsets = np.arange(self.rti_data["num_scenarios"])
        cdf_offset = cdf + offsets[:, None]

        # Generate Random Numbers
        rand_vals = np.random.rand(self.rti_data["num_scenarios"], self.MC_NUM)
        rand_offset = rand_vals + offsets[:, None]

        # Flatten for searchsorted
        cdf_flat = cdf_offset.ravel()
        rand_flat = rand_offset.ravel()

        # Binary Search (Finds insertion point in flattened CDF)
        insert_indices = np.searchsorted(cdf_flat, rand_flat)

        # Map back to 2D indices
        idx_selections_flat = insert_indices % self.rti_data["max_choices"]
        idx_selections = idx_selections_flat.reshape(self.rti_data["num_scenarios"], self.MC_NUM)

        # Retrieve selected POP indices
        row_indices = np.arange(self.rti_data["num_scenarios"])[:, None]
        selected_poppis = self.rti_data["choices_matrix"][row_indices, idx_selections]

        # --- 4. Construct Final Output Dictionary ---
        routed_through_ingress = {}

        for i, (ui, pref_i, ug_name) in enumerate(self.rti_data["meta_data"]):
            simulated_routes = selected_poppis[i] # Array of size MC_NUM
            
            for mci, poppi in enumerate(simulated_routes):
                if mci not in routed_through_ingress:
                    routed_through_ingress[mci] = {}
                
                # Ensure structure exists
                if pref_i not in routed_through_ingress[mci]:
                    routed_through_ingress[mci][pref_i] = {}
                
                # Assuming self.popps is a list/dict of actual POP objects
                routed_through_ingress[mci][pref_i][ug_name] = self.popps[poppi]

        return routed_through_ingress



def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickle', default='cache/popp_failure_latency_'
                    'comparison_testing_feature-actual-20_dep_sweep_20.pkl')
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()
    from unit_tests.bench_path_distribution import (
        build_worker, init_like_advertisement, seed_parent_tracker_from_init)
    w = build_worker(a.pickle)
    rng = np.random.RandomState(a.seed)
    base = init_like_advertisement(w, rng)
    seed_parent_tracker_from_init(w, base)

    advs = [base.copy()]
    for _ in range(4):                       # single flips
        x = base.copy()
        x[rng.randint(w.n_popp), rng.randint(w.n_prefixes)] ^= True
        advs.append(x)
    for _ in range(3):                       # RB row-zeros
        x = base.copy(); x[rng.randint(w.n_popp), :] = False
        advs.append(x)

    old_cso = types.MethodType(_Old._compute_scenario_options, w)
    old_ssr = types.MethodType(_Old._sample_scenario_realizations, w)
    fails = 0
    for k, adv in enumerate(advs):
        results = {}
        for label, cso, ssr in (('old', old_cso, old_ssr),
                                ('new', w._compute_scenario_options,
                                 w._sample_scenario_realizations)):
            # identical pattern-cache state for both arms
            w.pattern_cache = {}
            for attr in ('_uipop_poskey',):
                if hasattr(w, attr):
                    delattr(w, attr)
            cso(adv.astype(np.float64))
            np.random.seed(1000 + k)
            results[label] = ssr()
        if results['old'] != results['new']:
            fails += 1
            print('MISMATCH adv {}'.format(k))
            for mci in results['old']:
                for pref in results['old'][mci]:
                    o = results['old'][mci][pref]
                    n = results['new'].get(mci, {}).get(pref)
                    if o != n:
                        print('  mci={} pref={} old_n={} new_n={}'.format(
                            mci, pref, len(o), 0 if n is None else len(n)))
                        break
        else:
            print('adv {:>2d}: EQUAL ({} scenarios)'.format(
                k, w.rti_data['num_scenarios']))
    print('FAIL' if fails else 'ALL EQUAL -- fence holds')
    sys.exit(1 if fails else 0)

if __name__ == '__main__':
    main()
