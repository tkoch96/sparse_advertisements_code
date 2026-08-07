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
      onto that coordinate.
      ASSERT: gradient has <=1 nonzero component; realized step changes
      <=1 coordinate.
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
        self.abl_assert = os.environ.get('SCULPTOR_ABLATION_ASSERTS', '1') == '1'
        self._abl_checks = {'iter_start_binary': 0, 'grad_single': 0, 'grad_finite': 0,
                            'step_single': 0, 'step_binary': 0, 'max_info_budget': 0,
                            'mc_off_workers': 0}
        self.abl_nan_grad_iters = 0
        print('[ablation-fork] memory={} direction={} explore={} mc={} asserts={}'.format(
            self.abl_memory, self.abl_direction, self.abl_explore, self.abl_mc,
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

    def gradients(self, a, add_metrics=True):
        g = super().gradients(a, add_metrics=add_metrics)
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

    # ================= solve(): replica of the repo orchestrator ========
    # Verbatim copy of Sparse_Advertisement_Solver.solve() with
    # _abl_assert_* calls inserted at the seams.
    def solve(self, **kwargs):
        if not self._solve_setup(**kwargs):
            return
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
