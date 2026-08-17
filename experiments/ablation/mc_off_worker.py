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

# Solver-fork seam (Tom 2026-08-17): this module is imported FRESH inside
# ray worker processes (the actor class module), so the bare-name import
# below must be re-routed here — the driver-side aliases don't exist in
# this process. No-op when SCULPTOR_LP_BACKEND is unset.
if os.environ.get('SCULPTOR_LP_BACKEND'):
    from experiments.solver_fork.run_equivalence import install_aliases
    install_aliases()

from path_distribution_computer_ray import _LocalPathDistributionComputer  # noqa: E402


class Abl_MC_Off_Worker(_LocalPathDistributionComputer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MC_NUM = 1  # one deterministic pseudo-realization per LB call
        self._abl_pseudo_price = {}   # (ug, poppi) -> expected latency; single-use
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
        rd['num_scenarios'] = len(rd['all_probs'])
        if rd['num_scenarios'] == 0:
            return {}
        out = {0: {}}
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
