"""Ablation fork of the REAL SCULPTOR solver.

Subclasses Sparse_Advertisement_Solver. sparse_advertisements_v3.solve()
is now an orchestrator over named sub-steps (_solve_setup,
_solve_iter_begin, _solve_compute_gradients, _solve_apply_step,
_solve_post_step_measure, _solve_max_info_phase, _solve_check_stop,
_solve_iter_end, _solve_finalize); this fork REPLICATES the orchestrator
verbatim and inserts per-iteration ASSERTIONS at the seams so every run
proves its ablation flags are actually binding. Feature mechanisms are
overrides of existing repo methods; NO repo behavior is modified.

Flags (read at construction):
  SCULPTOR_ABLATION_MEMORY    '1' (default) | '0'
      '0': advertisement rounded to {0,1} after every step AND at init
      (a memory-less algorithm holds no continuous state).
      ASSERT: advertisement is binary at iteration start and after step.
  SCULPTOR_ABLATION_DIRECTION '1' (default) | '0'
      '0': single-coordinate descent -- gradient masked to its largest
      |component| and the realized step (momentum included) projected
      onto that coordinate. Additionally the samplers' cross-iteration
      remeasure state (last_lb_calls_results,
      last_rb_calls_results_{popp,pop}) is cleared after every gradient
      call: re-probing previously-large gradient coordinates is
      directional information and must not inform direction-off rungs
      (the wide explore pool is untouched -- painter-fair).
      ASSERT: gradient has <=1 nonzero component; realized step changes
      <=1 coordinate; remeasure state empty at every sampling pass.
  SCULPTOR_ABLATION_EXPLORE   'default' | 'random' | 'none'
      'none': solve_max_information returns None (no extra measurement).
      'random': one random adjacent flip is measured instead.
      ASSERT: path_measures does not grow during the max-info phase when
      'none'; grows by <= n_max_info_iter otherwise.
  SCULPTOR_ABLATION_MC        '1' (default) | '0'
      '0': monte-carlo route simulation replaced by a single deterministic
      avg-of-options pseudo-path with huge link capacities
      (experiments/ablation/mc_off_worker.Abl_MC_Off_Worker, injected via
      the worker_comms_ray.ACTOR_CLS seam by run_fork_ladder).
      ASSERT (every iteration, via the 'abl_mc_stats' worker RPC): every
      worker is the mc-off class, MC_NUM==1, zero stock-sampler calls,
      zero non-point-mass benefit pdfs, pseudo-path builder ran.
  SCULPTOR_ABLATION_PROBE_MODE 'fixed' (default) | 'gated'
      'gated': uncertainty-gated measure-XOR-step (Tom's design,
      2026-08-10). Each iteration: compute gradients as usual and also
      their per-coordinate uncertainty (sigma of the flip-delta from the
      LB pdfs the workers already return). If U > PROBE_C and the probe
      budget is not exhausted, spend the iteration on a MEASUREMENT (the
      rung's max-info mechanism; falls back to measuring the current
      advertisement) and do NOT step. Otherwise STEP and measure NOTHING
      (stock SCULPTOR measured the deployed advertisement after every
      step; under gating N is the TOTAL measurement budget for solve()). U = g^2-weighted mean sign-error probability
      Phi(-|g_i|/sigma_i) over ALL probed gradient coordinates (full
      support size; LB term only -- RB coords without sigma excluded). 'fixed'
      reproduces stock semantics exactly.
      ASSERT (every iteration): TOTAL measurements during solve() -- as
      path_measures growth, so every measurement path counts -- never
      exceed PROBE_N; probe iterations never exceed PROBE_N.
  SCULPTOR_ABLATION_PROBE_C   float, default 1.0: INITIAL threshold (auto-c
      anneals down from it; with AUTO_C=0 it is the static threshold)
  SCULPTOR_ABLATION_PROBE_N   int, default 5 (gated-mode probe budget)
  SCULPTOR_ABLATION_PROBE_AUTO_C '1' (default) | '0'
      Auto-learn c (Tom's scheme, 2026-08-10): budget N should spread over
      ~the first PROBE_FRAC of an assumed PROBE_TCONV-iteration
      convergence horizon, i.e. probe on a fraction N/(FRAC*TCONV) of
      iterations -> c targets the (1 - N/(FRAC*TCONV)) quantile
      (~95-98th pct) of ALL U values observed so far. c starts at
      PROBE_C (high) and anneals toward the quantile estimate with a
      ~5-iteration time constant; every time a probe FIRES, c doubles
      (multiplier persists). c components logged each iteration.
  SCULPTOR_ABLATION_PROBE_TCONV int, default 300 (assumed convergence horizon)
  SCULPTOR_ABLATION_PROBE_FRAC  float, default 0.75 (fraction of horizon)
  SCULPTOR_ABLATION_ASSERTS   '1' (default) | '0'  -- disable checks

A summary line '[ablation-assert] SUMMARY ...' prints at the end of every
run with the number of checks performed (violations raise immediately).
"""
import os
import pickle
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sparse_advertisements_v3 import Sparse_Advertisement_Solver, _log_mem  # noqa: E402
from constants import ADVERTISEMENT_THRESHOLD  # noqa: E402
from helpers import threshold_a  # noqa: E402


class Ablation_Sparse_Advertisement_Solver(Sparse_Advertisement_Solver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.abl_memory = os.environ.get('SCULPTOR_ABLATION_MEMORY', '1') == '1'
        self.abl_direction = os.environ.get('SCULPTOR_ABLATION_DIRECTION', '1') == '1'
        self.abl_explore = os.environ.get('SCULPTOR_ABLATION_EXPLORE', 'default')
        assert self.abl_explore in ('default', 'random', 'none')
        self.abl_mc = os.environ.get('SCULPTOR_ABLATION_MC', '1') == '1'
        self.abl_probe_mode = os.environ.get('SCULPTOR_ABLATION_PROBE_MODE', 'fixed')
        assert self.abl_probe_mode in ('fixed', 'gated')
        self.abl_probe_c = float(os.environ.get('SCULPTOR_ABLATION_PROBE_C', '1.0'))
        self.abl_probe_n = int(os.environ.get('SCULPTOR_ABLATION_PROBE_N', '5'))
        self.abl_probe_auto_c = os.environ.get('SCULPTOR_ABLATION_PROBE_AUTO_C', '1') == '1'
        self.abl_probe_tconv = int(os.environ.get('SCULPTOR_ABLATION_PROBE_TCONV', '300'))
        self.abl_probe_frac = float(os.environ.get('SCULPTOR_ABLATION_PROBE_FRAC', '0.75'))
        self.abl_probes_spent = 0
        self._abl_grad_sigma = {}
        self._abl_probe_U = None
        self._abl_U_history = []
        self._abl_c_mult = 1.0
        self.abl_assert = os.environ.get('SCULPTOR_ABLATION_ASSERTS', '1') == '1'
        self._abl_checks = {'iter_start_binary': 0, 'grad_single': 0, 'grad_finite': 0,
                            'step_single': 0, 'step_binary': 0, 'max_info_budget': 0,
                            'mc_off_workers': 0}
        self.abl_nan_grad_iters = 0
        print('[ablation-fork] memory={} direction={} explore={} mc={} probe_mode={} '
              '(c={}, N={}) asserts={}'.format(
            self.abl_memory, self.abl_direction, self.abl_explore, self.abl_mc,
            self.abl_probe_mode, self.abl_probe_c, self.abl_probe_n,
            self.abl_assert), flush=True)

    # ================= feature mechanisms (method overrides) ============
    def init_advertisement(self):
        a = super().init_advertisement()
        # All rungs must start from the SAME initialization (the repo's
        # init(), seeded by SCULPTOR_DEPLOYMENT_SEED). The first rung of a
        # seed persists the canonical init; every other rung asserts
        # byte-equality against it BEFORE any rung-specific transform.
        fn = os.environ.get('SCULPTOR_ABLATION_INIT_FILE')
        if fn and self.abl_assert:
            if os.path.exists(fn):
                ref = np.load(fn)
                assert a.shape == ref.shape and np.array_equal(a, ref), \
                    '[ablation-assert] initialization differs from canonical ({})'.format(fn)
                print('[ablation-assert] init matches canonical ({})'.format(fn), flush=True)
            else:
                np.save(fn, a)
                print('[ablation-assert] canonical init saved ({})'.format(fn), flush=True)
            self._abl_checks['init_canonical'] = self._abl_checks.get('init_canonical', 0) + 1
        if not self.abl_memory:
            a = threshold_a(a)  # memory-less: same init, thresholded (no continuous state)
        return a

    def impose_advertisement_constraint(self, a):
        a = super().impose_advertisement_constraint(a)
        if not self.abl_direction:
            # project the WHOLE step (incl. momentum) onto the gradient's
            # chosen coordinate; solve() sets last_advertisement = a_k (the
            # pre-step advertisement) just before calling us.
            coord = getattr(self, '_abl_step_coord', None)
            la = getattr(self, 'last_advertisement', None)
            if coord is not None and la is not None and np.shape(la) == np.shape(a):
                proj = np.clip(np.array(la, dtype=float), 0, 1)
                proj.flat[coord] = np.asarray(a).flat[coord]
                a = proj
            self._abl_step_coord = None
        if not self.abl_memory:
            a = threshold_a(a)
        return a

    # ============ gated probing (PROBE_MODE=gated) ======================
    def _assemble_lb_gradients(self, calls, all_lb_rets, a, L_grad):
        # capture per-coordinate flip-delta sigma from the LB pdfs the
        # workers already return (and stock code throws away): the two
        # arms of a flip probe are independent MC draws, so
        # var(delta) = var(before) + var(after).
        if self.abl_probe_mode == 'gated':
            # For each probed coordinate capture (|raw delta|, sigma of the
            # raw delta). heaviside_gradient scales the delta by a sigmoid
            # slope, so sign-error must be computed on RAW delta vs RAW
            # sigma (same units); the gradient g is used only as the
            # relevance weight. The delta is a difference of two MEANS over
            # MC_NUM draws, so its estimation noise is the standard error:
            # (var_b + var_a) / MC_NUM.
            from constants import ADVERTISEMENT_THRESHOLD as _T  # noqa: F401
            MC_NUM = 5  # path_distribution_computer default (no_mc: var=0 anyway)
            stats = {}
            for i, (ind, before_then_after) in enumerate(calls):
                var = 0.0
                means = []
                for j in (2 * i, 2 * i + 1):
                    mean_j, (x, p) = all_lb_rets[j]
                    means.append(float(mean_j))
                    var += self._abl_pdf_var(x, p)
                if len(means) == 2:
                    # SIGNED raw delta = after - before (matches this_grad's sign)
                    delta = (means[1] - means[0]) if before_then_after == 'ba' \
                        else (means[0] - means[1])
                else:
                    delta = 0.0
                stats[ind] = (delta, (var / MC_NUM) ** 0.5)
            self._abl_grad_sigma = stats
        return super()._assemble_lb_gradients(calls, all_lb_rets, a, L_grad)

    @staticmethod
    def _abl_pdf_var(x, p):
        x = np.asarray(x, dtype=float).flatten()
        p = np.asarray(p, dtype=float).flatten()
        psum = p.sum()
        if psum <= 0:
            return 0.0
        m = float((x * p).sum() / psum)
        return max(0.0, float((x * x * p).sum() / psum) - m * m)

    def _abl_capture_rb(self, store_attr, calls_advs, all_lb_rets):
        """Accumulate (signed raw delta, variance) per coordinate for RB
        probes. Each call is a (failed_off, failed_on) ret pair; the
        assembled gradient uses heaviside(before=failed_on, after=failed_off)
        so signed raw = failed_off - failed_on."""
        MC_NUM = 5
        store = {}
        ind = 0
        for coord in calls_advs:
            off_mean, (ox, op) = all_lb_rets[ind]
            on_mean, (nx, np_) = all_lb_rets[ind + 1]
            raw = float(off_mean) - float(on_mean)
            var = (self._abl_pdf_var(ox, op) + self._abl_pdf_var(nx, np_)) / MC_NUM
            s = store.setdefault(coord, [0.0, 0.0])
            s[0] += raw
            s[1] += var
            ind += 2
        setattr(self, store_attr, store)

    def _assemble_rb_popp_gradients(self, calls, all_lb_rets, advertisement, grad_rb):
        if self.abl_probe_mode == 'gated':
            coords = [(self.popp_to_ind[c[0]], c[2]) for c in calls]
            self._abl_capture_rb('_abl_rb_stats_popp', coords, all_lb_rets)
        return super()._assemble_rb_popp_gradients(calls, all_lb_rets, advertisement, grad_rb)

    def _assemble_rb_pop_gradients(self, calls, all_lb_rets, advertisement, grad_rb):
        if self.abl_probe_mode == 'gated':
            coords = [(self.popp_to_ind[c[0]], c[2]) for c in calls]
            self._abl_capture_rb('_abl_rb_stats_pop', coords, all_lb_rets)
        return super()._assemble_rb_pop_gradients(calls, all_lb_rets, advertisement, grad_rb)

    def _abl_probe_uncertainty(self, g):
        """U = g^2-weighted mean of P(sign error) = Phi(-|g_i|/sigma_i) over
        ALL probed coordinates with nonzero gradient (the full support size —
        cost is trivial). g^2 weighting = expected fraction of step ENERGY
        pushed the wrong way: sign-error damage on a coordinate scales with
        displacement x gradient, so the near-zero-gradient explore tail
        (whose sign is genuinely 50/50 but harmless) self-discounts
        quadratically instead of needing an arbitrary top-k cutoff.
        Also returns the aggregate noise-to-signal ratio for logging."""
        from math import erfc, sqrt
        # LB + RB composition (heuristic per Tom): probes are independent, so
        # raw deltas add with the objective's term weights and variances add
        # with the squared weights; sign-error is one Gaussian tail on the sum.
        gamma = float(self.get_gamma())
        w_L, w_R = (1.0, gamma) if gamma <= 1 else (1.0 / gamma, 1.0)
        alpha = float(os.environ.get('SCULPTOR_ALPHA_POP', '0'))
        rb_popp = getattr(self, '_abl_rb_stats_popp', {})
        rb_pop = getattr(self, '_abl_rb_stats_pop', {})
        combined = {}
        for ind, (delta, s) in self._abl_grad_sigma.items():
            combined[ind] = [w_L * delta, (w_L * s) ** 2]
        for src, w in ((rb_popp, w_R), (rb_pop, w_R * alpha)):
            if w == 0.0:
                continue
            for ind, (raw, var) in src.items():
                c = combined.setdefault(ind, [0.0, 0.0])
                c[0] += w * raw
                c[1] += (w ** 2) * var
        entries = []
        for ind, (raw, var) in combined.items():
            gv = float(np.asarray(g)[ind])
            if gv != 0.0:
                entries.append((abs(gv), abs(raw), var ** 0.5))
        if not entries:
            return 0.0, 0.0, 0
        wsum, num, d2, s2 = 0.0, 0.0, 0.0, 0.0
        ratios = []
        for gmag, delta, s in entries:
            # sign-error on the composed RAW delta (same units as sigma); the
            # scaled gradient g is only the relevance weight
            p_err = 0.5 * erfc(delta / (s * sqrt(2.0))) if s > 0 else 0.0
            w = gmag * gmag
            num += w * p_err
            wsum += w
            d2 += delta * delta
            s2 += s * s
            ratios.append(delta / s if s > 0 else float('inf'))
        med_snr = float(np.median(ratios)) if ratios else float('inf')
        self._abl_probe_med_snr = med_snr
        return num / wsum, (s2 / d2 if d2 > 0 else float('inf')), len(entries)

    def _abl_probe_current_c(self):
        """Auto-learned threshold (Tom's scheme): anneal from the initial
        (high) c toward the target quantile of the U history, x2 for every
        probe already fired. Static c when AUTO_C=0."""
        if not self.abl_probe_auto_c:
            return self.abl_probe_c, float('nan'), float('nan')
        # probe on ~N/(FRAC*TCONV) of iterations -> target quantile of U
        rate = self.abl_probe_n / max(1.0, self.abl_probe_frac * self.abl_probe_tconv)
        q_target = min(0.999, max(0.5, 1.0 - rate))
        if len(self._abl_U_history) >= 3:
            q_hat = float(np.quantile(np.asarray(self._abl_U_history), q_target))
        else:
            q_hat = self.abl_probe_c  # warmup: no estimate yet
        anneal = float(np.exp(-self.iter / 5.0))  # fast anneal, ~5-iter tau
        c = (q_hat + (self.abl_probe_c - q_hat) * anneal) * self._abl_c_mult
        return c, q_hat, anneal

    def _abl_probe_decision(self, grads):
        """gated mode: True -> spend this iteration on a measurement."""
        U, nsr, k = self._abl_probe_uncertainty(grads)
        self._abl_probe_U = U
        self._abl_U_history.append(U)
        c, q_hat, anneal = self._abl_probe_current_c()
        want = U > c
        can = self.abl_probes_spent < self.abl_probe_n
        decision = want and can
        print('[probe-gate] iter={} U={:.4f} nsr={:.3f} med_snr={:.2f} k={} '
              'c={:.4f} (q_hat={:.4f} anneal={:.2f} mult={:g}) '
              'spent={}/{} -> {}'.format(
            self.iter, U, nsr, getattr(self, '_abl_probe_med_snr', float('nan')),
            k, c, q_hat, anneal, self._abl_c_mult,
            self.abl_probes_spent, self.abl_probe_n, 'PROBE' if decision else
            ('step (budget exhausted, U high)' if want else 'step')), flush=True)
        if decision:
            self._abl_c_mult *= 2.0  # back-off: every fired probe doubles c
        return decision

    def _abl_do_probe_iteration(self):
        """Measurement instead of a step: use the rung's max-info mechanism;
        if it yields nothing (e.g. explore=none), measure the current
        advertisement so the iteration still buys real information."""
        pm_before = int(getattr(self, 'path_measures', 0))
        self._solve_max_info_phase()
        if int(getattr(self, 'path_measures', 0)) == pm_before:
            self._solve_post_step_measure()  # fallback: measure current adv
        self.abl_probes_spent += 1

    def _abl_assert_measure_budget(self):
        """TOTAL measurements during solve() never exceed the budget N --
        asserted on path_measures growth so any measurement path counts."""
        total = int(getattr(self, 'path_measures', 0)) - \
            int(getattr(self, '_abl_pm_solve_start', 0))
        if self.abl_assert:
            assert total <= self.abl_probe_n, \
                ('[ablation-assert] measurement budget exceeded: {} measured > N={} '
                 '(probe iterations: {})'.format(total, self.abl_probe_n,
                                                 self.abl_probes_spent))
            assert self.abl_probes_spent <= self.abl_probe_n, \
                ('[ablation-assert] probe iterations exceed budget: {} > {}'.format(
                    self.abl_probes_spent, self.abl_probe_n))
            self._abl_checks['measure_budget'] = self._abl_checks.get('measure_budget', 0) + 1

    # Cross-iteration remeasure state: the gradient samplers re-probe
    # coordinates whose PREVIOUS-iteration gradients were large
    # (last_lb_calls_results / last_rb_calls_results_{popp,pop}). That is
    # directional information carried across iterations, so direction-off
    # rungs must not see it: we assert the containers are empty before each
    # sampling pass and clear them after (Tom, 2026-08-09). The wide
    # explore pool is untouched -- painter-style multi-index exploration
    # with a best-coordinate pick stays a fair one-flip baseline.
    _ABL_REMEASURE_STATE = ('last_lb_calls_results',
                            'last_rb_calls_results_popp',
                            'last_rb_calls_results_pop')

    def _abl_clear_remeasure_state(self):
        for attr in self._ABL_REMEASURE_STATE:
            if hasattr(self, attr):
                delattr(self, attr)

    def gradients(self, a, add_metrics=True):
        if not self.abl_direction:
            if self.abl_assert:
                for attr in self._ABL_REMEASURE_STATE:
                    assert not getattr(self, attr, None), \
                        ('[ablation-assert] direction-off: {} nonempty at gradient '
                         'time (cross-iteration remeasure state leaked)'.format(attr))
                self._abl_checks['remeasure_cleared'] = \
                    self._abl_checks.get('remeasure_cleared', 0) + 1
        g = super().gradients(a, add_metrics=add_metrics)
        if not self.abl_direction:
            self._abl_clear_remeasure_state()
        bad = ~np.isfinite(g)
        if bad.any():
            # NaN guard (repo bug: failure-scenario LPs with zero routable
            # volume propagate NaN and collapse the advertisement)
            self.abl_nan_grad_iters += 1
            print('[ablation-fork] WARNING: {} non-finite gradient entries zeroed '
                  '(occurrence {})'.format(int(bad.sum()), self.abl_nan_grad_iters), flush=True)
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.abl_direction:
            # (memory-less selection+amplification already happened inside
            # _rescale_gradient; g is single-nonzero there and argmax keeps
            # the same coordinate)
            flat = int(np.abs(g).argmax())
            mask = np.zeros_like(g)
            mask.flat[flat] = g.flat[flat]
            g = mask
            self._abl_step_coord = flat
        return g

    def _rescale_gradient(self, net_grad, a):
        if not self.abl_direction:
            # single-coordinate rungs flip exactly one coordinate per
            # iteration (SCULPTOR's own design intent): SELECT the best viable coordinate and
            # amplify it to a guaranteed threshold crossing. (The stock cap,
            # DESIRED_MAX_VAL/max, limits steps to ~0.05 which the
            # every-iteration rounding erases -- the memory-less rung could
            # never flip anything; coordinate rotation freezes the memory
            # rung the same way.) The memory flag decides only whether the
            # crossed value is then rounded (impose) or stays continuous
            # just past threshold -- reversible, probe-hot: that IS memory.
            # Selection must happen HERE so the amplified coordinate is the
            # one that survives masking: pre-negation semantics -- the step
            # moves a by +alpha*net_grad, so net>0 pushes on, net<0 off.
            aa = np.asarray(a)
            viable = ((aa <= ADVERTISEMENT_THRESHOLD) & (net_grad > 1e-12)) | \
                     ((aa > ADVERTISEMENT_THRESHOLD) & (net_grad < -1e-12))
            if viable.any():
                gsel = np.where(viable, net_grad, 0.0)
                flat = int(np.abs(gsel).argmax())
                out = np.zeros_like(net_grad)
                out.flat[flat] = net_grad.flat[flat]
                alpha_needed = (ADVERTISEMENT_THRESHOLD - aa.flat[flat]) / \
                               (self.alpha * net_grad.flat[flat])
                mult = float(alpha_needed) * 1.0001
                print('Modified gradient by a factor of {} to ensure exactly one flip '
                      '(single-coordinate policy, coord {})'.format(mult, flat))
                return out * mult
            # no viable coordinate: emit zero step (no flip this iter)
            return np.zeros_like(net_grad)
        return super()._rescale_gradient(net_grad, a)

    def solve_max_information(self, current_advertisement):
        if self.abl_explore == 'none':
            return None
        if self.abl_explore == 'random':
            a = threshold_a(np.copy(current_advertisement))
            poppi = np.random.randint(self.n_popps)
            prefi = np.random.randint(self.n_prefixes)
            a[poppi, prefi] = 1.0 - a[poppi, prefi]
            if a.sum() == 0:
                return None
            return a
        return super().solve_max_information(current_advertisement)

    # ================= assertions =======================================
    def _abl_assert_iter_start(self):
        if not self.abl_assert:
            return
        if not self.abl_memory:
            aa = np.asarray(self.optimization_advertisement)
            assert np.all(np.isclose(aa, 0) | np.isclose(aa, 1)), \
                '[ablation-assert] memory-off: advertisement not binary at iter start'
            self._abl_checks['iter_start_binary'] += 1

    def _abl_assert_gradient(self, g):
        if not self.abl_assert:
            return
        assert np.all(np.isfinite(g)), '[ablation-assert] non-finite gradient escaped the guard'
        self._abl_checks['grad_finite'] += 1
        if not self.abl_direction:
            assert np.count_nonzero(g) <= 1, \
                '[ablation-assert] direction-off: gradient has >1 nonzero component'
            self._abl_checks['grad_single'] += 1

    def _abl_assert_step(self):
        if not self.abl_assert:
            return
        cur = np.asarray(self.optimization_advertisement)
        if not self.abl_direction:
            la = np.clip(np.array(self.last_advertisement, dtype=float), 0, 1)
            nch = int((~np.isclose(cur, la, atol=1e-12)).sum())
            assert nch <= 1, ('[ablation-assert] direction-off: realized step changed '
                              '{} coordinates (must be <=1)'.format(nch))
            self._abl_checks['step_single'] += 1
        if not self.abl_memory:
            assert np.all(np.isclose(cur, 0) | np.isclose(cur, 1)), \
                '[ablation-assert] memory-off: advertisement not binary after step'
            self._abl_checks['step_binary'] += 1

    def _abl_assert_mc(self):
        """MC-off binding proof, every iteration: fan the 'abl_mc_stats' RPC
        to all workers. A stock worker answers 'ERROR' (unknown cmd), which
        is exactly the injection-failure signal we want to catch."""
        if not self.abl_assert or self.abl_mc:
            return
        stats = self.worker_manager.send_receive_workers(
            pickle.dumps(('abl_mc_stats', None)))
        for wi, st in stats.items():
            assert isinstance(st, dict), \
                ('[ablation-assert] mc-off: worker {} did not report mc stats '
                 '(mc-off worker class not injected?): {!r}'.format(wi, st))
            assert st['mc_num'] == 1, \
                '[ablation-assert] mc-off: worker {} MC_NUM={}'.format(wi, st['mc_num'])
            assert st['stock_sample_calls'] == 0, \
                ('[ablation-assert] mc-off: worker {} reached a stock MC sampler '
                 '{} time(s)'.format(wi, st['stock_sample_calls']))
            assert st['point_mass_violations'] == 0, \
                ('[ablation-assert] mc-off: worker {} returned {} non-point-mass '
                 'benefit pdf(s)'.format(wi, st['point_mass_violations']))
            assert st['pseudo_calls'] > 0, \
                '[ablation-assert] mc-off: worker {} never built a pseudo-path'.format(wi)
        self._abl_checks['mc_off_workers'] += 1

    def _abl_assert_max_info(self, pm_before):
        if not self.abl_assert:
            return
        grew = int(getattr(self, 'path_measures', 0)) - pm_before
        if self.abl_explore == 'none':
            assert grew == 0, ('[ablation-assert] explore=none: max-info phase issued '
                               '{} measurement(s)'.format(grew))
        else:
            budget = int(getattr(self, 'n_max_info_iter', 1))
            assert grew <= budget, ('[ablation-assert] explore={}: max-info phase issued '
                                    '{} measurement(s), budget {}'.format(self.abl_explore, grew, budget))
        self._abl_checks['max_info_budget'] += 1

    def _abl_assert_summary(self):
        print('[ablation-assert] SUMMARY (violations raise immediately, so all listed '
              'checks HELD): {}'.format(self._abl_checks), flush=True)
        if self.abl_probe_mode == 'gated':
            total = int(getattr(self, 'path_measures', 0)) - \
                int(getattr(self, '_abl_pm_solve_start', 0))
            print('[probe-gate] FINAL: {} total measurements (budget N={}), '
                  '{} probe iterations, c={}'.format(
                      total, self.abl_probe_n, self.abl_probes_spent,
                      self.abl_probe_c), flush=True)

    # ================= solve(): replica of the repo orchestrator ========
    # Verbatim copy of Sparse_Advertisement_Solver.solve() with
    # _abl_assert_* calls inserted at the seams.
    def solve(self, **kwargs):
        if not self._solve_setup(**kwargs):
            return
        self._abl_pm_solve_start = int(getattr(self, 'path_measures', 0))
        self._solve_t_start = time.time()
        self.t_per_iter = 0

        if not self.simulated:
            self.last_measured_advertisement = self.optimization_advertisement

        self._broadcast_training_mode(True)

        try:
            while not self.stop:

                timers = []
                t_last = time.time()

                self._solve_iter_begin()
                self._abl_assert_iter_start()

                grads = self._solve_compute_gradients()
                self._abl_assert_gradient(grads)
                self._abl_assert_mc()

                ## grads
                timers.append(time.time() - t_last)
                t_last = time.time()

                if self.abl_probe_mode == 'gated':
                    # measure-XOR-step under a TOTAL measurement budget: N
                    # bounds every measurement in the run. Stock SCULPTOR
                    # measured the deployed advertisement after every step;
                    # gated mode does NOT -- step iterations measure nothing,
                    # and only probe iterations may measure. The budget
                    # assertion is on total path_measures growth, so ANY
                    # measurement path that slips through gets caught.
                    if self._abl_probe_decision(grads):
                        self._abl_do_probe_iteration()
                    else:
                        self._solve_apply_step(grads)
                        self._abl_assert_step()
                    self._abl_assert_measure_budget()
                    timers.append(time.time() - t_last)
                    t_last = time.time()
                    _log_mem('iter_post_measure', iter=self.iter)
                else:
                    self._solve_apply_step(grads)
                    self._abl_assert_step()

                    self._solve_post_step_measure()

                    ## measure
                    timers.append(time.time() - t_last)
                    t_last = time.time()
                    _log_mem('iter_post_measure', iter=self.iter)

                    _abl_pm_before = int(getattr(self, 'path_measures', 0))
                    self._solve_max_info_phase()
                    self._abl_assert_max_info(_abl_pm_before)

                ## info
                timers.append(time.time() - t_last)
                t_last = time.time()

                self._solve_check_stop()

                _log_mem('iter_post_stop_tracker', iter=self.iter)
                self.iter += 1

                ## stop
                timers.append(time.time() - t_last)
                t_last = time.time()

                self._solve_iter_end(timers)

        finally:
            self._broadcast_training_mode(False)

        self._solve_finalize()
        self._abl_assert_summary()


# Ladder rungs, cumulative from full SCULPTOR downward. Each entry is the
# env-flag dict the driver applies before constructing the solver.
RUNGS = {
    'full':        {'SCULPTOR_ABLATION_MEMORY': '1', 'SCULPTOR_ABLATION_DIRECTION': '1', 'SCULPTOR_ABLATION_EXPLORE': 'default', 'SCULPTOR_ABLATION_MC': '1'},
    'expl_random': {'SCULPTOR_ABLATION_MEMORY': '1', 'SCULPTOR_ABLATION_DIRECTION': '1', 'SCULPTOR_ABLATION_EXPLORE': 'random', 'SCULPTOR_ABLATION_MC': '1'},
    'expl_none':   {'SCULPTOR_ABLATION_MEMORY': '1', 'SCULPTOR_ABLATION_DIRECTION': '1', 'SCULPTOR_ABLATION_EXPLORE': 'none', 'SCULPTOR_ABLATION_MC': '1'},
    'no_direction': {'SCULPTOR_ABLATION_MEMORY': '1', 'SCULPTOR_ABLATION_DIRECTION': '0', 'SCULPTOR_ABLATION_EXPLORE': 'none', 'SCULPTOR_ABLATION_MC': '1'},
    'no_memory':   {'SCULPTOR_ABLATION_MEMORY': '0', 'SCULPTOR_ABLATION_DIRECTION': '0', 'SCULPTOR_ABLATION_EXPLORE': 'none', 'SCULPTOR_ABLATION_MC': '1'},
    # bottom link of the ladder (connects painter <-> no_memory): monte-carlo
    # OFF on top of the no_memory semantics.
    'no_mc':       {'SCULPTOR_ABLATION_MEMORY': '0', 'SCULPTOR_ABLATION_DIRECTION': '0', 'SCULPTOR_ABLATION_EXPLORE': 'none', 'SCULPTOR_ABLATION_MC': '0'},
}
RUNG_ORDER = ['full', 'expl_random', 'expl_none', 'no_direction', 'no_memory', 'no_mc']
