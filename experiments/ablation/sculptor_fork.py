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
      step; under gating N is the TOTAL measurement budget for solve()).
      U = g^2-weighted mean sign-error probability Phi(-|raw_i|/sigma_i)
      over ALL probed coordinates, LB and RB terms COMPOSED per the
      objective's own weights (independent probes: weighted raw-delta
      sums, squared-weight variance sums; raw deltas vs standard-error
      sigmas -- never the heaviside-scaled gradient vs raw sigma).
      'fixed' reproduces stock semantics exactly. With
      SCULPTOR_ABLATION_FIXED_BUDGET=1 (budgeted-fixed, Tom 2026-08-14):
      measure EVERY iteration until the PROBE_N budget is spent (may
      overshoot by 1: post-step + max-info can both measure in one
      iteration), then KEEP TRAINING on beliefs -- step-only iterations
      like a gated 'step', with the same uncertainty-factor decay --
      until the normal convergence criterion / horizon ends the run.
      (The 2026-08-12 exit-on-budget form of budgeted-fixed stopped at
      ~iteration N, i.e. ~at init -- degenerate L1B, retired.)
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
      ~5-iteration time constant; every time a probe FIRES, c doubles,
      and the multiplier DECAYS back toward 1 with a ~PROBE_MULT_TAU-
      iteration time constant (refractory burst-suppression, NOT a
      permanent ratchet). c components logged each iteration.
  SCULPTOR_ABLATION_PROBE_MULT_TAU float, default 10 (refractory decay)
  SCULPTOR_ABLATION_PROBE_TCONV int, default 300 (assumed convergence horizon)
  SCULPTOR_ABLATION_PROBE_FRAC  float, default 0.75 (fraction of horizon)
  SCULPTOR_ABLATION_GRAD_BASE 'threshold' (default) | 'bernoulli'
      L7 (Tom, 2026-08-15): the base configuration gradient flip pairs
      condition on. 'threshold' = stock: probes condition on
      threshold_a(a) (mode point mass) and assembly scales the raw flip
      delta by the sigmoid slope -- a biased estimator of the partial of
      the multilinear extension F(a)=E[f(A)], A~Bern(a). 'bernoulli' =
      the unbiased estimator: ONE base A drawn per gradient call (K=1,
      same LP cost), every LB and RB flip pair conditions on the SAME A
      (implemented by mapping queued probe advs through A=(u<a) in the
      latency_benefit override -- production's threshold_a at the
      compression seam is the identity on binary advs, so production
      code is untouched), and assembly uses the RAW after-before delta
      (multilinearity: F is linear in a_ij; the sigmoid slope re-weights
      by distance-to-threshold, double-counting what the sampled base
      already encodes).
      ASSERTS: queued bases binary; drawn base differs from
      threshold_a(a) on fractional coords at plausible frequency (vs
      expected sum min(a,1-a)); raw-delta assembly exercised (sigmoid
      path structurally unreachable). Logs per-iteration coin-flip mass
      sum a(1-a) -- the variance self-anneal Tom asked to track.
      Incompatible with SCULPTOR_ALPHA_POP>0 (the pop-RB sampler
      thresholds its base inside production code; loud refusal at init).
  SCULPTOR_ABLATION_GRAD_BASE_K int, default 1 (Tom 2026-08-15): number
      of drawn bases per gradient call under GRAD_BASE=bernoulli. Every
      flip pair is evaluated under all K bases (flush-time K-expansion
      of the probe queue) and averaged back to one (mean, mixture-pdf)
      per entry -- K x the LP cost; the mixture pdf hands the gate's
      sigma capture the cross-base variance as well.
  SCULPTOR_ABLATION_ASSERTS   '1' (default) | '0'  -- disable checks

A summary line '[ablation-assert] SUMMARY ...' prints at the end of every
run with the number of checks performed (violations raise immediately).
"""
import collections
import copy
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
        # Read every ablation flag once at construction (all runtime
        # behavior branches on these); see module docstring for semantics.
        super().__init__(*args, **kwargs)
        # stop-v2 now lives in the HEAD (sparse_advertisements_v3,
        # merged 2026-08-16); SCULPTOR_STOP_RULE env applies there.
        self.abl_memory = os.environ.get('SCULPTOR_ABLATION_MEMORY', '1') == '1'
        self.abl_direction = os.environ.get('SCULPTOR_ABLATION_DIRECTION', '1') == '1'
        self.abl_explore = os.environ.get('SCULPTOR_ABLATION_EXPLORE', 'default')
        assert self.abl_explore in ('default', 'random', 'none')
        self.abl_mc = os.environ.get('SCULPTOR_ABLATION_MC', '1') == '1'
        self.abl_probe_mode = os.environ.get('SCULPTOR_ABLATION_PROBE_MODE', 'fixed')
        # fixed: stock semantics (measure every new advertisement).
        # gated: original U>c gate (2026-08-10).
        # scheduled: unconditional probe every ~TCONV/N iterations.
        # smart: gated + (b) stale+plateau + (c) prediction-mismatch +
        #        (d) surprise-adaptive threshold (Tom, 2026-08-12).
        assert self.abl_probe_mode in ('fixed', 'gated', 'scheduled', 'smart', 'adaptive', 'slotted')
        # exit-on-budget applies to FIXED-BUDGET (L1) ONLY as of
        # 2026-08-14 late (Tom: "never exit because we ran out of
        # measurements — that's only for L1"). Gated/scheduled/smart
        # NEVER set exit_reason='budget_exhausted': budget exhaustion
        # stops MEASURING, training runs to the normal criterion/horizon.
        # This flag only selects L1's exit-vs-coast variant.
        self.abl_exit_on_budget = os.environ.get(
            'SCULPTOR_ABLATION_EXIT_ON_BUDGET', '1') == '1'
        self.abl_exit_reason = None
        # smart-gate tunables
        self.abl_smart_stale_frac = float(os.environ.get(
            'SCULPTOR_ABLATION_SMART_STALE_FRAC', '0.5'))   # gap = frac*TCONV/N
        self.abl_smart_plateau_w = int(os.environ.get(
            'SCULPTOR_ABLATION_SMART_PLATEAU_W', '5'))
        self.abl_smart_plateau_eps = float(os.environ.get(
            'SCULPTOR_ABLATION_SMART_PLATEAU_EPS', '0.01'))  # rel. to belief range
        self.abl_smart_sign_w = int(os.environ.get(
            'SCULPTOR_ABLATION_SMART_SIGN_W', '5'))
        self.abl_smart_sign_rate = float(os.environ.get(
            'SCULPTOR_ABLATION_SMART_SIGN_RATE', '0.5'))
        self.abl_smart_surprise_rel = float(os.environ.get(
            'SCULPTOR_ABLATION_SMART_SURPRISE_REL', '0.2'))
        self.abl_smart_surprise_factor = float(os.environ.get(
            'SCULPTOR_ABLATION_SMART_SURPRISE_FACTOR', '0.5'))
        self._abl_belief_hist = collections.deque(maxlen=64)
        self._abl_predreal = collections.deque(maxlen=self.abl_smart_sign_w)
        self._abl_pending_pred = None
        self._abl_last_probe_iter = 0    # last actual MEASUREMENT
        self._abl_last_attempt_iter = 0  # last FIRED decision (throttle)
        self._abl_pending_probe_ctx = None
        self._abl_preprobe_belief = None
        self._abl_probe_reasons = collections.Counter()
        self.abl_probe_c = float(os.environ.get('SCULPTOR_ABLATION_PROBE_C', '1.0'))
        self.abl_probe_n = int(os.environ.get('SCULPTOR_ABLATION_PROBE_N', '5'))
        self.abl_probe_auto_c = os.environ.get('SCULPTOR_ABLATION_PROBE_AUTO_C', '1') == '1'
        self.abl_probe_tconv = int(os.environ.get('SCULPTOR_ABLATION_PROBE_TCONV', '300'))
        self.abl_probe_frac = float(os.environ.get('SCULPTOR_ABLATION_PROBE_FRAC', '0.75'))
        self.abl_probes_spent = 0
        # decision-aware WHAT (L6', Tom 2026-08-16): PROBE_TARGET=decision
        # probes the one-flip neighbor with the largest expected regret of
        # deciding unprobed. Needs (delta, sigma) capture + the periodic
        # MC-refresh even under scheduled probing (both are elsewhere gated
        # on gated/smart -- without this, MC_NUM=1 pins sigma=0 and every
        # score collapses to the grounding fallback).
        self._abl_decision_target = os.environ.get(
            'SCULPTOR_ABLATION_PROBE_TARGET', 'maxinfo') == 'decision'
        self._abl_grad_sigma = {}
        self._abl_probe_U = None
        self._abl_U_history = []
        self._abl_c_mult = 1.0
        # --- sigma refresh + entropy-in-U (Tom's design, 2026-08-14 late) ---
        # At MC_NUM=1, gradient-flip pdfs go point-mass once measurements pin
        # beliefs -> sigma=0 -> U dies right after the first probe. Fix:
        # every SIGMA_REFRESH iterations evaluate gradients under
        # MC_NUM_EXPLORE draws and EWMA the per-coordinate variances;
        # between refreshes sigma is floored by the EWMA.
        self.abl_sigma_refresh = int(os.environ.get(
            'SCULPTOR_ABLATION_SIGMA_REFRESH',
            str(max(2, int(round(0.5 * self.abl_probe_tconv
                                 / max(1, self.abl_probe_n)))))))
        self._abl_var_ewma = {}
        self._abl_sigma_refresh_iter = False
        # U = U_sigma + U_ENT_W * (adjacency entropy / running anchor):
        # (a) WHEN to measure combines gradient sign-error risk with how
        # much information remains adjacent. (b) WHAT to measure stays
        # entropy-argmax over adjacencies (fallback: current adv) -- NOTE
        # (Tom): the measure-here-vs-measure-adjacent policy is a real
        # tradeoff we may revisit; for now target selection ignores sigma.
        self.abl_u_ent_w = float(os.environ.get(
            'SCULPTOR_ABLATION_U_ENT_W', '0.01'))
        self._abl_ent_anchor = None
        self._abl_ent_ratio = 0.0
        self._abl_U_sig = None
        self._abl_U_ent = None
        self._abl_med_sigma = None
        # --- L7: gradient base configuration (Tom, 2026-08-15) ---
        self.abl_grad_base = os.environ.get('SCULPTOR_ABLATION_GRAD_BASE', 'threshold')
        assert self.abl_grad_base in ('threshold', 'bernoulli')
        if self.abl_grad_base == 'bernoulli':
            assert float(os.environ.get('SCULPTOR_ALPHA_POP', '0')) == 0, \
                ('[ablation-fork] GRAD_BASE=bernoulli is incompatible with '
                 'SCULPTOR_ALPHA_POP>0: the pop-RB sampler thresholds its base '
                 'inside production code and would silently condition on the '
                 'mode, not the drawn A')
        # K bases per gradient entry (Tom 2026-08-15: 'let's just see what
        # happens' at K=3): every flip pair is evaluated under all K drawn
        # bases and averaged -- K x the LP cost per gradient.
        self.abl_grad_base_k = int(os.environ.get(
            'SCULPTOR_ABLATION_GRAD_BASE_K', '1'))
        assert self.abl_grad_base_k >= 1
        self._l7_bern_u = None        # per-gradient-call uniform draws (list of K); A_k = (u_k < a)
        self._l7_bern_active = False  # True only while gradient flip probes flush
        self._l7_diff_total = 0       # run total: coords where A != threshold_a(a)
        self._l7_expdiff_total = 0.0  # run total: expected disagreements sum min(a,1-a)
        self.abl_assert = os.environ.get('SCULPTOR_ABLATION_ASSERTS', '1') == '1'
        self._abl_checks = {'iter_start_binary': 0, 'grad_single': 0, 'grad_finite': 0,
                            'step_single': 0, 'step_binary': 0, 'max_info_budget': 0,
                            'mc_off_workers': 0}
        self.abl_nan_grad_iters = 0
        print('[ablation-fork] memory={} direction={} explore={} mc={} probe_mode={} '
              '(c={}, N={}) grad_base={} (K={}) asserts={}'.format(
            self.abl_memory, self.abl_direction, self.abl_explore, self.abl_mc,
            self.abl_probe_mode, self.abl_probe_c, self.abl_probe_n,
            self.abl_grad_base, self.abl_grad_base_k, self.abl_assert), flush=True)

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

    # ============ step-paradigm seam (Tom, 2026-08-14 night) ============
    # NOTE: the GRAD_SCALE branch lives in the single _rescale_gradient
    # override further down. A SECOND definition here silently lost to it
    # (same class, later def wins) and made the seam dead code for every
    # rung -- 2026-08-16: fixed-alpha arms were really auto-scale runs.
    # ============ gated probing (PROBE_MODE=gated) ======================
    # sigma capture stack (_abl_var_smooth/_abl_pdf_var/_abl_capture_rb/
    # _assemble_*_gradients) merged into the HEAD 2026-08-16.

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
        sigmas = []
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
            sigmas.append(s)
        med_snr = float(np.median(ratios)) if ratios else float('inf')
        self._abl_probe_med_snr = med_snr
        self._abl_med_sigma = float(np.median(sigmas)) if sigmas else 0.0
        # U = U_sigma + W * entropy-of-adjacencies ratio (Tom, 2026-08-14):
        # WHEN to measure blends (i) risk of stepping the wrong way with
        # (ii) how much information remains adjacent (last explore's best
        # value over its running EWMA anchor). WHAT to measure remains
        # entropy-argmax with the measure-current fallback -- revisit the
        # here-vs-adjacent targeting tradeoff later.
        u_sig = num / wsum
        u_ent = self.abl_u_ent_w * min(2.0, self._abl_ent_ratio)
        self._abl_U_sig, self._abl_U_ent = float(u_sig), float(u_ent)
        return (u_sig + u_ent,
                (s2 / d2 if d2 > 0 else float('inf')), len(entries))

    def _abl_probe_current_c(self):
        """Auto-learned threshold (Tom's scheme): anneal from the initial
        (high) c toward the target quantile of the U history, x2 for every
        probe already fired. Static c when AUTO_C=0."""
        if not self.abl_probe_auto_c:
            return self.abl_probe_c, float('nan'), float('nan')
        # REFRACTORY, not ratchet: the post-probe doubling relaxes back
        # toward 1 with a ~MULT_TAU-iteration time constant, so c stays
        # auto-learning after probes fire (a permanent multiplier locked
        # spending at ~2 probes regardless of budget -- 2026-08-10 N-sweep).
        # refractory scales with the INTENDED inter-probe spacing so large
        # budgets can physically be spent (fixed tau=10 capped spending at
        # ~iters/10 regardless of N -- scaling smoke 2026-08-11)
        tau_default = max(2.0, 0.5 * self.abl_probe_frac * self.abl_probe_tconv
                          / max(1, self.abl_probe_n))
        tau = float(os.environ.get('SCULPTOR_ABLATION_PROBE_MULT_TAU', str(tau_default)))
        self._abl_c_mult = 1.0 + (self._abl_c_mult - 1.0) * float(np.exp(-1.0 / tau))
        # probe on ~N/(FRAC*TCONV) of iterations -> target quantile of U;
        # Q_AGGR (Tom, 2026-08-14 late: spend was still low at ~18/50)
        # shifts the quantile down (e.g. 95th -> 85th) for more probes.
        rate = self.abl_probe_n / max(1.0, self.abl_probe_frac * self.abl_probe_tconv)
        # Q_AGGR: 0.10 briefly (2026-08-14 eve) made the gate burst-spend
        # the whole budget by ~iter 40; pared back to 0 (Tom) — the
        # sched backstop guarantees grounding instead.
        aggr = float(os.environ.get('SCULPTOR_ABLATION_PROBE_Q_AGGR', '0'))
        q_target = max(0.4, min(0.999, 1.0 - rate) - aggr)
        if len(self._abl_U_history) >= 1:
            # learn from the FIRST U sample; anneal from the initial c with
            # a short time constant (Tom 2026-08-14: tau=1.5 validated --
            # first probe ~iter 8, clean budget spend; tau=0.5 front-loaded
            # probes and exhausted explore early)
            # U_WINDOW (Tom, 2026-08-15, DEFAULT ON at TCONV/2): U is
            # nonstationary (drifts down as measurements resolve beliefs,
            # hardest on memory arms) -- a FULL-history quantile stays
            # anchored to the early high-U era, so c sits above the
            # current regime and criterion (a) under-fires (L6: ~30% of
            # probes came from the sched backstop). A trailing window
            # tracks the current regime. =0 restores full-history.
            _uw = int(os.environ.get(
                'SCULPTOR_ABLATION_U_WINDOW',
                str(max(10, self.abl_probe_tconv // 2))))
            _hist = self._abl_U_history[-_uw:] if _uw > 0 \
                else self._abl_U_history
            q_hat = float(np.quantile(np.asarray(_hist), q_target))
        else:
            q_hat = self.abl_probe_c  # warmup: no estimate yet
        _atau = float(os.environ.get(
            'SCULPTOR_ABLATION_PROBE_ANNEAL_TAU', '1.5'))
        anneal = float(np.exp(-self.iter / _atau))
        c = (q_hat + (self.abl_probe_c - q_hat) * anneal) * self._abl_c_mult
        return c, q_hat, anneal

    def _abl_probe_decision(self, grads):
        """gated mode: True -> spend this iteration on a measurement."""
        U, nsr, k = self._abl_probe_uncertainty(grads)
        self._abl_probe_U = U
        self._abl_U_history.append(U)
        c, q_hat, anneal = self._abl_probe_current_c()
        self._abl_probe_c_now = c
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
            # refractory doubling moved to _abl_commit_probe_bookkeeping —
            # it applies only if the probe actually MEASURES (Tom review
            # 2026-08-15: skipped probes were paying the penalty)
            self._abl_last_attempt_iter = self.iter
            self._abl_pending_probe_ctx = {}
        return decision

    def _abl_decision_probe_target(self):
        # implementation merged into the HEAD as _decision_probe_target
        # (default SCULPTOR_MAXINFO_TARGET=decision); kept as an alias
        # for the fork's probe-XOR-step harness.
        return self._decision_probe_target()

    def _abl_do_probe_iteration(self):
        """Measurement instead of a step: use the rung's max-info mechanism;
        if it yields nothing (e.g. explore=none), measure the current
        advertisement so the iteration still buys real information.

        Returns True iff a measurement happened. Tom's rule (2026-08-14):
        if explore found nothing worth measuring AND the current adv has
        already been measured, IGNORE the probe request -- spend nothing,
        stop nothing; the caller turns the iteration into a normal step.
        (Pre-fix this fallback re-measured the same adv until the explore
        re-pick killed the run -- the MC_NUM=1 degenerate-pdf pathology.)"""
        pm_before = int(getattr(self, 'path_measures', 0))
        # SCULPTOR_ABLATION_PROBE_TARGET (Tom 2026-08-16, WHAT/WHEN
        # decomposition): 'current' = probe measures the CURRENT
        # advertisement only (pure grounding -- the L1-L5 arms);
        # 'maxinfo' (legacy default) = the rung's max-info targeting
        # (the smart-WHAT delta, isolated at L6/L7);
        # 'decision' (L6') = expected-regret targeting over the one-flip
        # adjacency, grounding at current when nothing scores.
        _tgt = os.environ.get('SCULPTOR_ABLATION_PROBE_TARGET', 'maxinfo')
        if _tgt == 'decision':
            adv = self._abl_decision_probe_target()
            if adv is not None:
                # probe diagnostic (Tom 2026-08-16 "what is it picking, how
                # much does that reduce uncertainty"): BEFORE state here;
                # AFTER state resolved at the next gradients() call, once
                # the measurement has been folded into beliefs.
                _d = dict(getattr(self, '_abl_decision_choice', {}) or {})
                _b = getattr(self, 'current_pseudo_objective', None)
                _d.update({
                    'iter': int(self.iter),
                    'belief_before': (float(_b) if _b is not None
                                      and np.isfinite(_b) else None)})
                try:
                    _d['U_before'] = float(self._abl_probe_uncertainty(
                        self._abl_last_grads)[0])
                except Exception:
                    _d['U_before'] = None
                self.measure_ingresses(adv)
                if int(getattr(self, 'path_measures', 0)) > pm_before:
                    self._abl_pending_probe_diag = _d
                    self._abl_probe_reasons['decision_adjacent'] += 1
                    self.abl_probes_spent += 1
                    self._abl_last_probe_iter = self.iter
                    return True
            # no scored candidate (or measurement no-op): ground at the
            # current adv -- the bias-reset backbone; skip if already known
            cur = tuple(threshold_a(np.asarray(
                self.optimization_advertisement, dtype=float)).flatten())
            if cur in getattr(self, 'measured', {}):
                self._abl_probe_skips = 1 + getattr(
                    self, '_abl_probe_skips', 0)
                print('[probe-gate] iter={} probe SKIPPED (decision: no '
                      'scored candidate; current already measured; {} '
                      'skips)'.format(self.iter, self._abl_probe_skips),
                      flush=True)
                return False
            self._solve_post_step_measure()
            if int(getattr(self, 'path_measures', 0)) == pm_before:
                return False
            self._abl_probe_reasons['decision_ground'] += 1
            self.abl_probes_spent += 1
            self._abl_last_probe_iter = self.iter
            return True
        if _tgt == 'current':
            cur = tuple(threshold_a(np.asarray(
                self.optimization_advertisement, dtype=float)).flatten())
            if cur in getattr(self, 'measured', {}):
                self._abl_probe_skips = 1 + getattr(
                    self, '_abl_probe_skips', 0)
                print('[probe-gate] iter={} probe SKIPPED (target=current '
                      'already measured; {} skips)'.format(
                          self.iter, self._abl_probe_skips), flush=True)
                return False
            self._solve_post_step_measure()
            if int(getattr(self, 'path_measures', 0)) == pm_before:
                return False
            self.abl_probes_spent += 1
            self._abl_last_probe_iter = self.iter
            return True
        self._solve_max_info_phase()
        # Track adjacency-entropy for the U combination: best candidate
        # value over a running EWMA anchor of itself. Updated whenever
        # explore actually evaluated candidates (probe iterations only;
        # between probes the last ratio holds).
        _ev = getattr(self, '_last_explore_value', None)
        if _ev is not None and np.isfinite(_ev) and _ev > 0:
            self._abl_ent_anchor = float(_ev) if self._abl_ent_anchor is None \
                else 0.75 * self._abl_ent_anchor + 0.25 * float(_ev)
            self._abl_ent_ratio = float(_ev) / max(self._abl_ent_anchor, 1e-9)
        if getattr(self, '_explore_remeasure_stop', None) is not None:
            # Legacy SCULPTOR_REMEASURE_STOP=1 path: no probe happened,
            # none is counted; the main loop ends training gracefully.
            return True
        if int(getattr(self, 'path_measures', 0)) == pm_before:
            cur = tuple(threshold_a(np.asarray(
                self.optimization_advertisement, dtype=float)).flatten())
            if cur in getattr(self, 'measured', {}):
                self._abl_probe_skips = 1 + getattr(
                    self, '_abl_probe_skips', 0)
                print('[probe-gate] iter={} probe SKIPPED: no informative '
                      'candidate and current adv already measured '
                      '({} skips so far)'.format(
                          self.iter, self._abl_probe_skips), flush=True)
                # NOTE: only the ATTEMPT clock was advanced (at decision
                # time) — (b)/(s) staleness still measures from the last
                # actual measurement
                return False
            self._solve_post_step_measure()  # fallback: measure current adv
        self.abl_probes_spent += 1
        self._abl_last_probe_iter = self.iter
        return True

    # ============ scheduled + smart probing (2026-08-12) =================

    def _abl_slotted_decision(self):
        """Slotted WHEN (Tom 2026-08-16: "mean measurement rate stays
        evenly spaced; bias measurements to where they're needed WITHIN
        their expected interval"). Probe k owns slot k*period +- w
        (w = period/2, slots tile TCONV exactly), so the budget is always
        fully spent and the long-run rate IS the schedule. Within a slot:
        fire from the slot START when the last grounding surprise was hot
        (the model demonstrably drifting), from the CENTER when quiet;
        the slot END force-fires (schedule = backstop). Skipped probes
        retry every iteration until the slot closes -- no budget leak."""
        period = max(1, int(round(float(self.abl_probe_tconv)
                                  / max(1, self.abl_probe_n))))
        w = max(1, period // 2)
        k = self.abl_probes_spent + 1          # next probe, 1-indexed
        can = k <= self.abl_probe_n
        # resolve last grounding surprise (same pending mechanism as
        # 'adaptive'; _abl_last_surprise_val persists for the hot test,
        # _abl_last_surprise is one-shot for the gate record)
        if getattr(self, '_abl_surprise_pending', None) is not None:
            pre, probe_iter = self._abl_surprise_pending
            b = getattr(self, 'current_pseudo_objective', None)
            if b is not None and np.isfinite(b) and self.iter > probe_iter:
                span = max(abs(getattr(self, '_stopv2_b0', float(b))
                               - getattr(self, '_stopv2_best', float(b))), 1e-9)
                surprise = abs(float(b) - pre) / span
                self._abl_last_surprise = float(surprise)
                self._abl_last_surprise_val = float(surprise)
                self._abl_surprise_pending = None
                print('[probe-gate] slotted surprise={:.4f}'.format(surprise),
                      flush=True)
        theta = float(os.environ.get('SCULPTOR_ABLATION_SURPRISE_THETA', '0.02'))
        hot = (getattr(self, '_abl_last_surprise_val', None) or 0.0) > theta
        center = k * period
        earliest, latest = center - w, center + w
        due = self.iter >= (earliest if hot else center)
        force = self.iter >= latest
        decision = can and (due or force)
        if decision:
            self._abl_last_attempt_iter = self.iter
            self._abl_pending_probe_ctx = {}
            b = getattr(self, 'current_pseudo_objective', None)
            self._abl_surprise_pending = (
                (float(b) if b is not None and np.isfinite(b) else 0.0),
                int(self.iter))
        print('[probe-gate] iter={} mode=slotted k={}/{} slot=[{},{}] '
              'hot={} spent={} -> {}'.format(
                  self.iter, k, self.abl_probe_n, earliest, latest, hot,
                  self.abl_probes_spent,
                  'PROBE' if decision else 'step'), flush=True)
        return decision

    def _abl_adaptive_decision(self):
        """new-L6 WHEN (Tom 2026-08-16): surprise-adapted grounding.
        Iteration-clocked AIMD on the probe interval K -- start at
        K0 = TCONV/N; after each grounding, the REALIZED belief surprise
        (the one bias-immune error signal: how much the measurement moved
        the belief, relative to the achieved belief span) shrinks K
        multiplicatively on big surprise / grows it on small; clamped to
        [1, 3*K0]. No model-self-assessed uncertainty anywhere (the L7
        autopsy: a biased model never volunteers that it needs checking;
        ~290 of ~400 smart-gate firings were the dumb backstop). The
        K_max clamp IS the staleness backstop. Probe target is always
        'current' (pure grounding)."""
        K0 = max(1.0, float(self.abl_probe_tconv) / max(1, self.abl_probe_n))
        if not hasattr(self, '_abl_K'):
            self._abl_K = float(K0)
            self._abl_surprise_pending = None
        if self._abl_surprise_pending is not None:
            pre, probe_iter = self._abl_surprise_pending
            b = getattr(self, 'current_pseudo_objective', None)
            if b is not None and np.isfinite(b) and self.iter > probe_iter:
                span = max(abs(getattr(self, '_stopv2_b0', float(b))
                               - getattr(self, '_stopv2_best', float(b))), 1e-9)
                surprise = abs(float(b) - pre) / span
                theta = float(os.environ.get(
                    'SCULPTOR_ABLATION_SURPRISE_THETA', '0.02'))
                self._abl_last_surprise = float(surprise)
                oldK = self._abl_K
                if surprise > theta:
                    self._abl_K = max(1.0, self._abl_K * 0.5)
                else:
                    self._abl_K = min(3.0 * K0, self._abl_K * 1.3)
                print('[probe-gate] adaptive surprise={:.4f} theta={} '
                      'K {:.1f}->{:.1f}'.format(surprise, theta, oldK,
                                                self._abl_K), flush=True)
                self._abl_surprise_pending = None
        due = (self.iter - self._abl_last_probe_iter) >= int(round(self._abl_K))
        can = self.abl_probes_spent < self.abl_probe_n
        decision = due and can
        if decision:
            self._abl_last_attempt_iter = self.iter
            self._abl_pending_probe_ctx = {}
            b = getattr(self, 'current_pseudo_objective', None)
            self._abl_surprise_pending = (
                (float(b) if b is not None and np.isfinite(b) else 0.0),
                int(self.iter))
        print('[probe-gate] iter={} mode=adaptive K={:.1f} since_last={} '
              'spent={}/{} -> {}'.format(
                  self.iter, self._abl_K,
                  self.iter - self._abl_last_probe_iter,
                  self.abl_probes_spent, self.abl_probe_n,
                  'PROBE' if decision else 'step'), flush=True)
        return decision

    def _abl_scheduled_decision(self):
        """scheduled mode: unconditional probe every ~TCONV/N iterations --
        no self-assessment, spends exactly N over the horizon."""
        period = max(1, int(round(self.abl_probe_tconv
                                  / max(1, self.abl_probe_n))))
        due = (self.iter - self._abl_last_probe_iter) >= period
        can = self.abl_probes_spent < self.abl_probe_n
        retry_ok = (self.iter - self._abl_last_attempt_iter
                    >= min(3, period))
        decision = due and can and retry_ok
        if decision:
            self._abl_last_attempt_iter = self.iter
            self._abl_pending_probe_ctx = {}
        print('[probe-gate] iter={} mode=scheduled period={} since_last={} '
              'spent={}/{} -> {}'.format(
                  self.iter, period, self.iter - self._abl_last_probe_iter,
                  self.abl_probes_spent, self.abl_probe_n,
                  'PROBE' if decision else 'step'), flush=True)
        return decision

    def _abl_track_belief(self):
        """Per-iteration belief bookkeeping shared by the smart gate:
        appends the solver's believed (pseudo) objective, resolves the
        pending predicted-vs-realized pair from the last step, and applies
        the (d) surprise adjustment after a probe."""
        b = getattr(self, 'current_pseudo_objective', None)
        if b is None or not np.isfinite(b):
            return
        prev = self._abl_belief_hist[-1] if self._abl_belief_hist else None
        self._abl_belief_hist.append(float(b))
        if prev is None:
            return
        realized = float(b) - prev
        # (c) bookkeeping: pair the last STEP's predicted delta with what
        # the belief actually did (pairs spanning probe iterations are
        # dropped -- measurement corrections are not model mispredictions)
        if self._abl_pending_pred is not None:
            self._abl_predreal.append((self._abl_pending_pred, realized))
            self._abl_pending_pred = None
        # (d) surprise: how much did the last probe's measurement move the
        # belief, relative to the belief's own range? big surprise -> the
        # model was wrong -> LOWER the bar for the next probe.
        if self._abl_preprobe_belief is not None:
            scale = max(max(self._abl_belief_hist) - min(self._abl_belief_hist),
                        1e-9)
            surprise = abs(float(b) - self._abl_preprobe_belief) / scale
            if surprise > self.abl_smart_surprise_rel:
                self._abl_c_mult *= self.abl_smart_surprise_factor
                print('[probe-gate] surprise={:.3f} > {} -> c_mult *= {} '
                      '(now {:g})'.format(
                          surprise, self.abl_smart_surprise_rel,
                          self.abl_smart_surprise_factor, self._abl_c_mult),
                      flush=True)
            else:
                self._abl_c_mult *= 2.0  # unsurprising probe: back off
            self._abl_preprobe_belief = None

    def _abl_smart_decision(self, grads):
        """smart mode: probe when ANY of
          (a) U > c                       (model admits uncertainty)
          (b) stale >= TCONV/(2N) AND believed objective plateaued
              (quiet + ungrounded -> verify before trusting convergence)
          (c) sign-disagreement rate of predicted-vs-realized believed
              deltas above threshold  (local model can't predict itself)
        Budget-capped; every probe logs which criteria fired."""
        U, nsr, k = self._abl_probe_uncertainty(grads)
        self._abl_probe_U = U
        self._abl_U_history.append(U)
        c, q_hat, anneal = self._abl_probe_current_c()
        self._abl_probe_c_now = c
        fire_a = U > c

        stale_gap = max(2, int(round(self.abl_smart_stale_frac
                                     * self.abl_probe_tconv
                                     / max(1, self.abl_probe_n))))
        stale = (self.iter - self._abl_last_probe_iter) >= stale_gap
        hist = list(self._abl_belief_hist)
        plateau = False
        if len(hist) >= self.abl_smart_plateau_w + 1:
            deltas = np.abs(np.diff(hist[-(self.abl_smart_plateau_w + 1):]))
            scale = max(max(hist) - min(hist), 1e-9)
            plateau = float(np.mean(deltas)) < self.abl_smart_plateau_eps * scale
        fire_b = stale and plateau

        fire_c = False
        if len(self._abl_predreal) >= self.abl_smart_sign_w:
            dis = [1.0 if (abs(r) > 1e-12 and np.sign(p) != np.sign(r)) else 0.0
                   for p, r in self._abl_predreal]
            fire_c = float(np.mean(dis)) >= self.abl_smart_sign_rate

        # (s) scheduled backstop (Tom, 2026-08-14 late): never let the
        # self-assessed criteria starve spending -- if no probe has fired
        # in SCHED_FALLBACK_MULT x the intended spacing, probe anyway.
        sched_gap = max(2, int(round(
            float(os.environ.get('SCULPTOR_ABLATION_SCHED_FALLBACK_MULT',
                                 '1.25'))
            * self.abl_probe_tconv / max(1, self.abl_probe_n))))
        fire_s = (self.iter - self._abl_last_probe_iter) >= sched_gap

        reasons = ''.join(tag for tag, f in
                          (('a', fire_a), ('b', fire_b), ('c', fire_c),
                           ('s', fire_s)) if f)
        # MIN-GAP conservatism (Tom 2026-08-16c: "firing a bit too
        # early -- bias toward the fixed schedule"): the self-assessed
        # criteria (a/b/c) cannot fire before MINGAP_FRAC x the
        # scheduled spacing since the last MEASUREMENT; the (s)
        # backstop is exempt (it IS the schedule anchor).
        mingap = int(round(
            float(os.environ.get('SCULPTOR_ABLATION_SMART_MINGAP_FRAC',
                                 '0.7'))
            * self.abl_probe_tconv / max(1, self.abl_probe_n)))
        if reasons and not fire_s and \
                (self.iter - self._abl_last_probe_iter) < mingap:
            print('[probe-gate] iter={} min-gap guard: reasons={} held '
                  '(since_last={} < mingap={})'.format(
                      self.iter, reasons,
                      self.iter - self._abl_last_probe_iter, mingap),
                  flush=True)
            reasons = ''
        can = self.abl_probes_spent < self.abl_probe_n
        # staleness (b) and backstop (s) ride the MEASUREMENT clock; the
        # ATTEMPT clock only throttles expensive max-info retries after a
        # skip (Tom review 2026-08-15: skips must not reset (b)/(s) — a
        # run whose explore keeps re-picking measured advs would defer
        # the backstop's grounding guarantee indefinitely)
        retry_ok = (self.iter - self._abl_last_attempt_iter
                    >= min(3, sched_gap))
        decision = bool(reasons) and can and retry_ok
        print('[probe-gate] iter={} mode=smart U={:.4f} c={:.4f} '
              'stale={}/{} plateau={} signrate_n={} reasons={} '
              'spent={}/{} -> {}'.format(
                  self.iter, U, c, self.iter - self._abl_last_probe_iter,
                  stale_gap, plateau, len(self._abl_predreal),
                  reasons or '-', self.abl_probes_spent, self.abl_probe_n,
                  'PROBE' if decision else 'step'), flush=True)
        if decision:
            self._abl_last_attempt_iter = self.iter
            # bookkeeping committed only if the probe MEASURES (see
            # _abl_commit_probe_bookkeeping)
            self._abl_pending_probe_ctx = {
                'reasons': reasons, 'fire_c': fire_c,
                'belief': (self._abl_belief_hist[-1]
                           if self._abl_belief_hist else None)}
        return decision

    def _abl_commit_probe_bookkeeping(self):
        """Post-MEASUREMENT bookkeeping: refractory doubling (gated),
        reasons census + pre-probe belief for the (d) surprise pairing
        (smart). Runs ONLY when a probe actually measured — fired-but-
        skipped probes leave no statistical trace, so probe_reasons and
        c-trajectories stay quantitatively interpretable (Tom review
        2026-08-15)."""
        ctx = getattr(self, '_abl_pending_probe_ctx', None) or {}
        if self.abl_probe_mode == 'gated':
            self._abl_c_mult *= 2.0  # back-off: measured probe doubles c
        if self.abl_probe_mode == 'smart':
            if ctx.get('reasons'):
                self._abl_probe_reasons[ctx['reasons']] += 1
            self._abl_preprobe_belief = ctx.get('belief')
            if ctx.get('fire_c'):
                self._abl_predreal.clear()  # fresh window post-grounding
        self._abl_pending_probe_ctx = None

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
        """Forget which coordinates had large gradients last iteration --
        the direction-off rungs' guarantee that no cross-iteration
        directional information reaches the samplers."""
        for attr in self._ABL_REMEASURE_STATE:
            if hasattr(self, attr):
                delattr(self, attr)

    # ============ L7: Bernoulli gradient base (GRAD_BASE=bernoulli) =====
    def flush_latency_benefit_queue(self, **kwargs):
        # Sampling seam (L7 override (a)), at FLUSH time so one place
        # handles K>=1 drawn bases. While gradient flip probes are being
        # flushed, each queued continuous adv is replaced by its K drawn
        # bases A_k = (u_k < a): fractional coords ~ Bern(a_ij), coords the
        # sampler forced to exactly 0.0/1.0 stay deterministic (u in
        # [0,1)), and ALL probes of this gradient call share the SAME K
        # u-matrices -> the same K drawn bases (LB and RB alike).
        # Production's compression then applies threshold_a, which is the
        # identity on a binary matrix -- the override lives entirely in
        # the fork. The K replicas of each entry are flushed k-adjacent
        # and averaged back to one (mean, (x, pdf)) per original entry:
        # mean of means, and mean of pdfs (the K-base MIXTURE, so the
        # gate's sigma capture sees cross-base variance too).
        if not getattr(self, '_l7_bern_active', False) or not self.lb_args_queue:
            return super().flush_latency_benefit_queue(**kwargs)
        K = self.abl_grad_base_k
        orig = self.lb_args_queue
        expanded = []
        for entry_args, entry_kwa in orig:
            a = np.asarray(entry_args[0], dtype=np.float64)
            for u in self._l7_bern_u:
                A = (u < a).astype(np.float64)
                if self.abl_assert:
                    assert np.all((A == 0.0) | (A == 1.0)), \
                        '[ablation-assert] l7: queued probe base not binary'
                    self._abl_checks['l7_base_binary'] = \
                        self._abl_checks.get('l7_base_binary', 0) + 1
                expanded.append(((A,) + tuple(entry_args[1:]),
                                 copy.deepcopy(entry_kwa)))
        self.lb_args_queue = expanded
        rets = super().flush_latency_benefit_queue(**kwargs)
        assert len(rets) == K * len(orig), \
            '[ablation-assert] l7: flush returned {} rets for {} expanded ' \
            'entries'.format(len(rets), K * len(orig))
        averaged = []
        for i in range(len(orig)):
            group = rets[K * i:K * i + K]
            mean = float(np.mean([g[0] for g in group]))
            x0, p0 = group[0][1]
            ps = [np.asarray(g[1][1], dtype=float).flatten() for g in group]
            if all(p.shape == ps[0].shape for p in ps) and all(
                    np.asarray(g[1][0]).flatten().shape ==
                    np.asarray(x0).flatten().shape for g in group):
                p_mix = np.mean(ps, axis=0)
                averaged.append((mean, (np.asarray(x0).flatten(), p_mix)))
            else:
                averaged.append((mean, (np.asarray(x0).flatten(),
                                        np.asarray(p0).flatten())))
        return averaged

    def heaviside_gradient(self, before, after, a_ij):
        # Assembly seam (L7 override (b)): under Bernoulli semantics the
        # multilinear extension is LINEAR in a_ij, so dF/da_ij is the raw
        # flip delta -- the sigmoid slope would re-weight by distance-to-
        # threshold, double-counting what the sampled base encodes. This
        # one seam covers the LB assembler and both RB assemblers; the
        # sigmoid path is structurally unreachable while the flag is set.
        if self.abl_grad_base == 'bernoulli':
            self._abl_checks['l7_raw_delta'] = \
                self._abl_checks.get('l7_raw_delta', 0) + 1
            return after - before
        return super().heaviside_gradient(before, after, a_ij)

    def gradients(self, a, add_metrics=True):
        # Wraps SCULPTOR's combined LB+RB gradient: NaN-guards it, and for
        # direction-off rungs masks it to one coordinate and proves the
        # remeasure state carried nothing in (see module docstring).
        if not self.abl_direction:
            if self.abl_assert:
                for attr in self._ABL_REMEASURE_STATE:
                    assert not getattr(self, attr, None), \
                        ('[ablation-assert] direction-off: {} nonempty at gradient '
                         'time (cross-iteration remeasure state leaked)'.format(attr))
                self._abl_checks['remeasure_cleared'] = \
                    self._abl_checks.get('remeasure_cleared', 0) + 1
        if self.abl_grad_base == 'bernoulli':
            # K drawn bases per gradient call (K=1: one base per
            # iteration); binding bookkeeping compares the ACTUAL drawn
            # bases against the thresholded base stock would have used.
            K = self.abl_grad_base_k
            aa = np.clip(np.asarray(a, dtype=np.float64), 0.0, 1.0)
            self._l7_bern_u = [np.random.uniform(size=aa.shape)
                               for _ in range(K)]
            thr = aa > ADVERTISEMENT_THRESHOLD
            d = int(sum(((u < aa) != thr).sum() for u in self._l7_bern_u))
            e = float(np.minimum(aa, 1.0 - aa).sum()) * K
            cf = float((aa * (1.0 - aa)).sum())
            self._l7_diff_total += d
            self._l7_expdiff_total += e
            print('[l7] iter={} K={} coinflips={:.3f} base_diff={} expected~{:.2f}'.format(
                getattr(self, 'iter', -1), K, cf, d, e), flush=True)
            if self.abl_assert and e >= 14.0:
                # P(no disagreement) <= exp(-14) if sampling binds -- a trip
                # here means the drawn bases are not actually being used.
                assert d >= 1, ('[ablation-assert] l7: drawn bases identical to '
                                'thresholded base despite {:.1f} expected '
                                'disagreements'.format(e))
            self._l7_bern_active = True
        try:
            g = super().gradients(a, add_metrics=add_metrics)
        finally:
            self._l7_bern_active = False
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
        # (NaN guard, belief/stop-v2 tracking, and probe-diag resolution
        # now run in the HEAD's gradients(); merged 2026-08-16.)
        # re-stash the MASKED gradient for single-coordinate rungs so the
        # decision-WHAT relevance weight sees what the step will see
        self._abl_last_grads = g
        return g

    def _rescale_gradient(self, net_grad, a):
        # Step-size POLICIES (auto/fixed/adagrad/dog) live in the HEAD
        # (SCULPTOR_GRAD_SCALE, default adagrad; merged 2026-08-16).
        # This override keeps ONLY the single-coordinate guaranteed-flip
        # policy that DEFINES the direction-off rungs (alpha cancels
        # algebraically there, so head policies cannot apply).
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

    # make_plots/_plot_model_error merged into the HEAD 2026-08-16.

    def solve_max_information(self, current_advertisement):
        """The EXPLORATION rung knob: SCULPTOR's entropic max-information
        measurement proposal (default), a random adjacent flip ('random'),
        or nothing ('none')."""
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
        # direction-off binding proof: <=1 nonzero component, always finite.
        if not self.abl_assert:
            return
        assert np.all(np.isfinite(g)), '[ablation-assert] non-finite gradient escaped the guard'
        self._abl_checks['grad_finite'] += 1
        if not self.abl_direction:
            assert np.count_nonzero(g) <= 1, \
                '[ablation-assert] direction-off: gradient has >1 nonzero component'
            self._abl_checks['grad_single'] += 1

    def _abl_assert_step(self):
        # memory/direction binding proof on the REALIZED step (momentum
        # included): binary adv for memory-off, <=1 changed coordinate for
        # direction-off.
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
        # exploration binding proof: measurement growth during the max-info
        # phase matches the rung's budget (0 for explore=none).
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
        # End-of-run receipt: counts of every per-iteration check that held
        # (violations raise at the offending iteration, so a printed count
        # IS the proof the flag bound throughout).
        print('[ablation-assert] SUMMARY (violations raise immediately, so all listed '
              'checks HELD): {}'.format(self._abl_checks), flush=True)
        if self.abl_grad_base == 'bernoulli':
            print('[l7] FINAL: raw_delta_calls={} base_binary_checks={} '
                  'base_diffs={} expected~{:.1f}'.format(
                      self._abl_checks.get('l7_raw_delta', 0),
                      self._abl_checks.get('l7_base_binary', 0),
                      self._l7_diff_total, self._l7_expdiff_total), flush=True)
            if self.abl_assert:
                assert self._abl_checks.get('l7_raw_delta', 0) > 0, \
                    '[ablation-assert] l7: raw-delta assembly never exercised'
                assert self._abl_checks.get('l7_base_binary', 0) > 0, \
                    ('[ablation-assert] l7: no queued probe ever passed through '
                     'the Bernoulli sampling seam (latency_benefit override '
                     'not reached with the mode active)')
                if self._l7_expdiff_total >= 10.0:
                    assert self._l7_diff_total > 0, \
                        ('[ablation-assert] l7: drawn base never differed from '
                         'the thresholded base over the whole run')
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

                if self.abl_probe_mode in ('gated', 'scheduled', 'smart', 'adaptive', 'slotted'):
                    # measure-XOR-step under a TOTAL measurement budget: N
                    # bounds every measurement in the run. Step iterations
                    # measure nothing; only probe iterations may measure.
                    # The budget assertion is on total path_measures growth,
                    # so ANY measurement path that slips through gets caught.
                    if self.abl_probe_mode == 'smart':
                        self._abl_track_belief()
                        probe = self._abl_smart_decision(grads)
                    elif self.abl_probe_mode == 'slotted':
                        probe = self._abl_slotted_decision()
                    elif self.abl_probe_mode == 'adaptive':
                        probe = self._abl_adaptive_decision()
                    elif self.abl_probe_mode == 'scheduled':
                        probe = self._abl_scheduled_decision()
                    else:
                        probe = self._abl_probe_decision(grads)
                    _gate_rec = {'iter': int(self.iter),
                                 'K': (float(self._abl_K)
                                       if hasattr(self, '_abl_K') else None),
                                 'surprise': getattr(
                                     self, '_abl_last_surprise', None),
                                 'U': getattr(self, '_abl_probe_U', None),
                                 'c': getattr(self, '_abl_probe_c_now', None),
                                 'U_sig': getattr(self, '_abl_U_sig', None),
                                 'U_ent': getattr(self, '_abl_U_ent', None),
                                 'med_sigma': getattr(
                                     self, '_abl_med_sigma', None),
                                 'ent_ratio': float(self._abl_ent_ratio),
                                 'ent_anchor': (
                                     float(self._abl_ent_anchor)
                                     if self._abl_ent_anchor is not None
                                     else None),
                                 'refresh': bool(
                                     self._abl_sigma_refresh_iter),
                                 'probe': bool(probe),
                                 'spent': int(self.abl_probes_spent),
                                 'uf': float(self.uncertainty_factor),
                                 'explore_val': None}
                    if not hasattr(self, '_abl_gate_hist'):
                        self._abl_gate_hist = []
                    self._abl_gate_hist.append(_gate_rec)
                    self._abl_last_surprise = None  # one record per resolution
                    self._last_explore_value = None
                    if probe:
                        probed = self._abl_do_probe_iteration()
                        _gate_rec['explore_val'] = getattr(
                            self, '_last_explore_value', None)
                        _gate_rec['spent'] = int(self.abl_probes_spent)
                        if probed:
                            self._abl_commit_probe_bookkeeping()
                        else:
                            # probe request ignored (nothing informative,
                            # current adv already measured) -> normal step
                            self._abl_pending_probe_ctx = None
                            _gate_rec['skipped'] = True
                            probe = False
                    if not probe:
                        # Preserve stock's uncertainty_factor decay invariant
                        # (2026-08-14): stock decays the factor every iteration
                        # inside solve_max_information; under gating that code
                        # only runs on probe iterations, so a factor spike
                        # suppressed U, which suppressed probes, which made the
                        # decay unreachable -- a deadlock that froze runs at
                        # uncertainty_factor ~16k (dep3). Decay on step
                        # iterations with stock's alpha and floor.
                        self.uncertainty_factor = max(
                            1.0, self.uncertainty_factor * (1 - .25))
                        a_before = np.array(self.optimization_advertisement,
                                            dtype=float)
                        self._solve_apply_step(grads)
                        self._abl_assert_step()
                        if self.abl_probe_mode == 'smart':
                            # (c) raw material: first-order predicted change
                            # in the believed objective from this step
                            da = (np.asarray(self.optimization_advertisement,
                                             dtype=float) - a_before)
                            self._abl_pending_pred = float(
                                np.dot(np.asarray(grads).flatten(),
                                       da.flatten()))
                    self._abl_assert_measure_budget()
                    # Budget exhaustion stops MEASURING, never TRAINING
                    # (Tom, 2026-08-14 late: "never exit because we ran
                    # out of measurements — that's only for L1"). The
                    # gate's spend cap already blocks further probes;
                    # remaining iterations run as belief-driven steps to
                    # the normal convergence criterion / horizon.
                    # (Pre-change runs recorded exit_reason=
                    # 'budget_exhausted' here — quarantined.)
                    timers.append(time.time() - t_last)
                    t_last = time.time()
                    _log_mem('iter_post_measure', iter=self.iter)
                elif (os.environ.get('SCULPTOR_ABLATION_FIXED_BUDGET',
                                     '0') == '1'
                      and not self.abl_exit_on_budget
                      and (int(getattr(self, 'path_measures', 0))
                           - int(getattr(self, '_abl_pm_solve_start', 0)))
                          >= self.abl_probe_n):
                    # Budgeted-fixed COAST variant (EXIT_ON_BUDGET=0):
                    # budget spent -> keep training on beliefs to the
                    # horizon. NOT the L1 arm -- Tom's final definition
                    # (2026-08-14 late): L1 = measure every one of the
                    # first N iterations, then IMMEDIATELY EXIT (the
                    # exit branch below, default). L1 varies with N via
                    # how many trained iterations it gets.
                    self.abl_probes_spent = (
                        int(getattr(self, 'path_measures', 0))
                        - int(getattr(self, '_abl_pm_solve_start', 0)))
                    self.uncertainty_factor = max(
                        1.0, self.uncertainty_factor * (1 - .25))
                    self._solve_apply_step(grads)
                    self._abl_assert_step()

                    ## measure (skipped: fixed budget spent — coasting)
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

                    if os.environ.get('SCULPTOR_ABLATION_FIXED_BUDGET',
                                      '0') == '1':
                        # track spend (may overshoot by 1: post-step +
                        # max-info can both measure in one iteration).
                        # L1 semantics (Tom 2026-08-14 late): budget
                        # spent -> IMMEDIATELY EXIT (default
                        # exit-on-budget); EXIT_ON_BUDGET=0 -> coast
                        # branch above instead.
                        self.abl_probes_spent = (
                            int(getattr(self, 'path_measures', 0))
                            - int(getattr(self, '_abl_pm_solve_start', 0)))
                        if (self.abl_exit_on_budget
                                and self.abl_probes_spent
                                >= self.abl_probe_n):
                            self.abl_exit_reason = 'budget_exhausted'

                ## info
                timers.append(time.time() - t_last)
                t_last = time.time()

                if (getattr(self, '_explore_remeasure_stop', None) is not None
                        and self.abl_exit_reason is None):
                    # covers both the gated probe path and stock fixed-mode
                    # max-info: remeasure means beliefs are resolved.
                    self.abl_exit_reason = 'remeasure_triggered'

                self._solve_check_stop()

                _log_mem('iter_post_stop_tracker', iter=self.iter)
                self.iter += 1

                ## stop
                timers.append(time.time() - t_last)
                t_last = time.time()

                self._solve_iter_end(timers)

                if self.abl_exit_reason == 'budget_exhausted':
                    # Tom's exit-training criterion (2026-08-12): with no
                    # measurements left there is no way to update beliefs;
                    # further steps are pure model-drift (the georand
                    # collapse mechanism). Stop at the last grounded point.
                    print('[ablation-fork] EXIT: measurement budget '
                          'exhausted at iter {} ({}/{} probes; reasons: {})'
                          .format(self.iter, self.abl_probes_spent,
                                  self.abl_probe_n,
                                  dict(self._abl_probe_reasons) or 'n/a'),
                          flush=True)
                    break
                elif self.abl_exit_reason == 'remeasure_triggered':
                    info = getattr(self, '_explore_remeasure_stop', {}) or {}
                    print('[ablation-fork] EXIT: REMEASURE TRIGGERED at iter '
                          '{} -- explore re-selected an already-measured '
                          'advertisement (flips={}, methodology={}); beliefs '
                          'are resolved, further probes are circular. '
                          'Stopping training gracefully ({}/{} probes spent).'
                          .format(info.get('iter', self.iter),
                                  info.get('flips'), info.get('methodology'),
                                  self.abl_probes_spent, self.abl_probe_n),
                          flush=True)
                    break

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
