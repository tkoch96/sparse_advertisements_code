"""Monte-carlo OFF worker for the ablation ladder ('no_mc' rung).

Subclasses the production worker (_LocalPathDistributionComputer) and turns
off the monte-carlo route simulation, replacing it with a single
DETERMINISTIC "avg-of-options pseudo-path" realization, PAINTER-style:

  * For each (ug, prefix) scenario the worker computes the expected latency
    over the scenario's ingress options (probability-weighted average) and
    represents the scenario by ONE pseudo-path priced at that expectation
    (structural representative = the option with lowest true latency; on a
    (ug, popp) collision across prefixes the lower expectation wins, which
    is what the LP would have picked anyway).
  * Link capacities are effectively removed ("huge caps"): a fictitious
    averaged path cannot meaningfully congest a specific link, and the
    pre-monte-carlo estimator this rung models was capacity-blind.
  * MC_NUM = 1, so generic_objective_pdf solves exactly one LP per
    latency_benefit call and the returned benefit distribution collapses to
    a point mass (the trivial-distribution branch).

The class is injected via the worker_comms_ray.ACTOR_CLS seam by
run_fork_ladder when SCULPTOR_ABLATION_MC='0'. It never raises inside the
actor (handle_msg would swallow the traceback into 'ERROR'); instead it
counts violations, and the driver-side fork asserts the counters every
iteration via the 'abl_mc_stats' RPC:

  mc_num == 1, stock_sample_calls == 0, point_mass_violations == 0,
  pseudo_calls > 0  -- proving the flag binds on every worker, every
  iteration. In-process numbers are still untrusted; rescore_fork (fresh
  process, stock code, real capacities) produces all reported metrics.
"""
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from core.path_distribution_computer import _LocalPathDistributionComputer  # noqa: E402


class Abl_MC_Off_Worker(_LocalPathDistributionComputer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MC_NUM = 1  # one deterministic pseudo-realization per LB call
        self._abl_pseudo_price = {}   # (ug, poppi) -> expected latency; single-use
        # Deterministic-realization caches (Tom 2026-08-31: the per-
        # scenario python loop below ran on EVERY latency-benefit call --
        # ~75% of no_mc's worker wall at actual-10, vs ~10% rti share for
        # the MC arms whose sampling is vectorized). Everything here is a
        # pure function of the block content, so caching cannot change
        # results -- only skip recomputation.
        self._abl_blk_cache = {}   # block bytes -> (ug->rep, [(ug,rep,exp)], n_scen)
        self._abl_adv_cache = {}   # per-adv tuple of (pref_i, block key) -> (out, prices, n_scen)
        self._abl_mc = {
            'pseudo_calls': 0,          # _sample_scenario_realizations invocations
            'stock_sample_calls': 0,    # any stock MC sampler reached (must stay 0)
            'point_mass_violations': 0, # pdf came back non-degenerate (must stay 0)
            'pdf_calls': 0,
        }
        print('[ablation-fork] worker {}: mc-off (deterministic pseudo-path, '
              'huge caps, MC_NUM=1)'.format(self.worker_i), flush=True)

    # ---- sampling override: deterministic avg-of-options pseudo-path ---- #
    def _sample_scenario_realizations(self):
        self._abl_mc['pseudo_calls'] += 1
        self._abl_pseudo_price = {}
        rd = self.rti_data
        blocks = rd.get('blocks') or []
        meta = rd.get('block_meta') or []
        out = {0: {}}
        if blocks and meta:
            # compact-block structures (post-2026-08-25 mainline; the
            # legacy all_probs/meta_data lists are no longer populated).
            # Mainline samples UNIFORMLY over each scenario's options, so
            # the deterministic expectation is the plain mean.
            #
            # Two-level deterministic cache (Tom 2026-08-31): the whole
            # composed realization is a pure function of the block
            # contents, so key blocks by their bytes and the full adv by
            # the tuple of (pref_i, block key).
            blk_keys = []
            for (lens_e, pad_e), (pref_i, _names_e, _uis_e) in zip(
                    blocks, meta):
                blk_keys.append((int(pref_i),
                                 lens_e.tobytes() + pad_e.tobytes()))
            adv_key = tuple(blk_keys)
            hit = self._abl_adv_cache.get(adv_key)
            if hit is not None and os.environ.get(
                    'SCULPTOR_ABL_RTI_CACHE_CHECK', '0') == '1':
                # smoke-mode self-check: recompute uncached and compare;
                # count mismatches (never raise -- handle_msg would
                # swallow it into 'ERROR')
                c_out, c_prices, c_n = hit
                f_out, f_prices, f_n = self._abl_uncached_realization(
                    blocks, meta)
                if (f_out != c_out or f_prices != c_prices
                        or f_n != c_n):
                    self._abl_mc['rti_cache_mismatch'] = \
                        self._abl_mc.get('rti_cache_mismatch', 0) + 1
                    print('[ablation-fork] RTI CACHE MISMATCH (adv key '
                          'len {})'.format(len(adv_key)), flush=True)
                else:
                    ok = self._abl_mc.get('rti_cache_check_ok', 0) + 1
                    self._abl_mc['rti_cache_check_ok'] = ok
                    if ok == 1 or ok % 200 == 0:
                        print('[ablation-fork] rti cache self-check: {} '
                              'hit(s) verified equal'.format(ok),
                              flush=True)
            if hit is not None:
                cached_out, cached_prices, n_scen = hit
                # inner dicts copied: downstream owns the realization;
                # prices are read-only then REBOUND (never mutated), so
                # the cached dict itself is safe to hand over
                self._abl_pseudo_price = cached_prices
                rd['num_scenarios'] = n_scen
                return ({0: {pi: dict(d)
                             for pi, d in cached_out.items()}}
                        if n_scen else {})
            n_scen = 0
            for bk, ((lens_e, pad_e), (pref_i, names_e, uis_e)) in zip(
                    blk_keys, zip(blocks, meta)):
                ent = self._abl_blk_cache.get(bk[1])
                if ent is None:
                    ug_to_rep, scen_list = {}, []
                    for j, ug_name in enumerate(names_e):
                        n = int(lens_e[j])
                        if n <= 0:
                            continue
                        poppis = pad_e[j, :n].astype(int)
                        lats = self.lat_matrix[poppis, int(uis_e[j])]
                        exp_lat = float(lats.mean())
                        rep = int(poppis[int(np.argmin(lats))])
                        ug_to_rep[ug_name] = self.popps[rep]
                        scen_list.append((ug_name, rep, exp_lat))
                    ent = (ug_to_rep, scen_list)
                    self._abl_blk_cache[bk[1]] = ent
                ug_to_rep, scen_list = ent
                for ug_name, rep, exp_lat in scen_list:
                    prev = self._abl_pseudo_price.get((ug_name, rep))
                    if prev is None or exp_lat < prev:
                        self._abl_pseudo_price[ug_name, rep] = exp_lat
                out[0].setdefault(pref_i, {}).update(ug_to_rep)
                n_scen += len(scen_list)
            rd['num_scenarios'] = n_scen
            # bounded: candidate advs churn; insertion-order eviction
            # keeps the recent working set (candidates recur within and
            # across adjacent rounds) without unbounded growth
            if len(self._abl_adv_cache) >= 512:
                del self._abl_adv_cache[next(iter(self._abl_adv_cache))]
            if len(self._abl_blk_cache) >= 4096:
                del self._abl_blk_cache[next(iter(self._abl_blk_cache))]
            self._abl_adv_cache[adv_key] = (
                {pi: dict(d) for pi, d in out[0].items()},
                dict(self._abl_pseudo_price), n_scen)
            return out if n_scen else {}
        rd['num_scenarios'] = len(rd['all_probs'])
        if rd['num_scenarios'] == 0:
            return {}
        perfs_all = self.whole_deployment_ug_perfs
        for (ui, pref_i, ug_name), probs, poppis in zip(
                rd['meta_data'], rd['all_probs'], rd['all_poppis']):
            perfs = perfs_all[ug_name]
            lats = [perfs[self.popps[pi]] for pi in poppis]
            exp_lat = float(np.dot(probs, lats))
            rep = poppis[int(np.argmin(lats))]
            prev = self._abl_pseudo_price.get((ug_name, rep))
            if prev is None or exp_lat < prev:
                self._abl_pseudo_price[ug_name, rep] = exp_lat
            try:
                out[0][pref_i][ug_name] = self.popps[rep]
            except KeyError:
                out[0][pref_i] = {ug_name: self.popps[rep]}
        return out

    def _abl_uncached_realization(self, blocks, meta):
        """The ORIGINAL per-scenario loop, kept verbatim as the reference
        implementation for the cache self-check (and nothing else)."""
        prices, out0 = {}, {}
        n_scen = 0
        for (lens_e, pad_e), (pref_i, names_e, uis_e) in zip(blocks, meta):
            for j, ug_name in enumerate(names_e):
                n = int(lens_e[j])
                if n <= 0:
                    continue
                poppis = pad_e[j, :n].astype(int)
                lats = self.lat_matrix[poppis, int(uis_e[j])]
                exp_lat = float(lats.mean())
                rep = int(poppis[int(np.argmin(lats))])
                prev = prices.get((ug_name, rep))
                if prev is None or exp_lat < prev:
                    prices[ug_name, rep] = exp_lat
                out0.setdefault(pref_i, {})[ug_name] = self.popps[rep]
                n_scen += 1
        return out0, prices, n_scen

    # Stock samplers must be unreachable; count instead of raising (handle_msg
    # would swallow an exception into 'ERROR') and let the driver assert.
    def sim_rti(self):
        self._abl_mc['stock_sample_calls'] += 1
        return super().sim_rti()

    def sim_rti_better(self):
        # (same guard as sim_rti: stock MC samplers must be unreachable)
        self._abl_mc['stock_sample_calls'] += 1
        return super().sim_rti_better()

    # ---- pricing override: pseudo-paths priced at expected latency ------ #
    def _path_obj_coeffs(self, available_paths, obj, site_cost_alpha):
        coeffs = super()._path_obj_coeffs(available_paths, obj, site_cost_alpha)
        if self._abl_pseudo_price:
            # prices only ever exist for obj='avg_latency' training LBs
            for i, key in enumerate(available_paths):
                try:
                    coeffs[i] = self._abl_pseudo_price[key]
                except KeyError:
                    pass  # e.g. the NO_PATH sentinel: keep NO_ROUTE_LATENCY
        return coeffs

    def solve_generic_lp_persistent(self, routed_through_ingress, obj, **kwargs):
        try:
            return super().solve_generic_lp_persistent(routed_through_ingress, obj, **kwargs)
        finally:
            # Prices are single-use: they belong to the pseudo-realization
            # built by the immediately preceding _sample_scenario_realizations
            # call. Any OTHER LP solve (e.g. driver-requested ground-truth
            # solves via _cmd_solve_lp) must see true per-path latencies.
            self._abl_pseudo_price = {}

    # ---- capacity override: "huge caps" (capacity-blind estimator) ------ #
    def _compute_static_caps(self):
        caps = super()._compute_static_caps()
        # 10x total deployment volume: no link can ever bind, well-scaled for
        # the LP. Applies in training AND in-process eval (which is untrusted
        # anyway; rescore_fork re-evaluates with real capacities).
        huge = 10.0 * float(np.sum(self.whole_deployment_ug_vols))
        return np.full_like(caps, huge)

    # ---- self-checks + driver RPC --------------------------------------- #
    def generic_objective_pdf(self, obj, a, **kwargs):
        x, pdfx = super().generic_objective_pdf(obj, a, **kwargs)
        self._abl_mc['pdf_calls'] += 1
        if self.MC_NUM != 1 or np.count_nonzero(pdfx) != 1:
            self._abl_mc['point_mass_violations'] += 1
        return x, pdfx

    def _cmd_abl_mc_stats(self, data):
        # RPC for the driver's per-iteration binding assertion: a STOCK
        # worker answers 'ERROR' to this command, which is the injection-
        # failure signal _abl_assert_mc catches.
        return dict(self._abl_mc, mc_num=self.MC_NUM)
