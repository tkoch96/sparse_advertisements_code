"""Driver for the fork-based ablation ladder.

For one (seed, rung): builds the seeded 'small' deployment, spins up the
REAL repo worker stack, runs Ablation_Sparse_Advertisement_Solver with
that rung's feature flags, and scores the final advertisement with the
REPO's own evaluation (measured_objective + ground-truth LP user
latencies). one_per_peering (and optionally the repo painter) are
computed through the same pipeline for reference.

    python -m experiments.ablation.run_fork_ladder --seed 1 --rung full \
        --port 31800 --max-iter 200 --out-dir cache/ablation/fork_ladder

Rungs: full, expl_random, expl_none, no_direction, no_memory, no_mc (see
sculptor_fork.RUNGS) plus 'painter' (repo painter baseline, no fork).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)



def avg_lat(sas, adv):
    """Repo-eval volume-weighted average user latency for an advertisement."""
    lats = np.asarray(sas.get_ground_truth_user_latencies(adv))
    vols = np.asarray(sas.ug_vols)
    return float(np.average(lats, weights=vols))


def run_one(seed, rung, port, max_iter, out_dir, dpsize='small'):
    # One (seed, rung) cell: build seeded deployment -> real worker stack ->
    # fork solver under the rung's flags (+ optional gated probing via
    # SCULPTOR_ABLATION_PROBE_*) -> in-run scoring on a PRISTINE eval stack
    # (still untrusted; rescore_fork is authoritative) -> semantic run-dir.
    out_fn = os.path.join(out_dir, 'seed_{}_{}.json'.format(seed, rung))
    if os.path.exists(out_fn):
        print('[seed {} {}] exists, skipping'.format(seed, rung), flush=True)
        return out_fn

    from experiments.ablation.sculptor_fork import (
        Ablation_Sparse_Advertisement_Solver, RUNGS)
    if rung != 'painter':
        for k, v in RUNGS[rung].items():
            os.environ[k] = v
    os.environ['SCULPTOR_MAX_ITER'] = str(max_iter)
    # fair budget: no early stop -- UNLESS the cell opts into one via
    # SCULPTOR_ABLATION_MIN_ITER (the stop-v2 flow, Tom 2026-08-16)
    os.environ['SCULPTOR_MIN_ITER'] = os.environ.get(
        'SCULPTOR_ABLATION_MIN_ITER', str(max_iter))
    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(seed)
    # canonical per-seed init: first rung writes it, all others assert equality
    os.makedirs(out_dir, exist_ok=True)
    os.environ['SCULPTOR_ABLATION_INIT_FILE'] = os.path.abspath(
        os.path.join(out_dir, 'init_dep{}.npy'.format(seed)))
    os.environ['SCULPTOR_DISABLE_PARALLEL_STRATEGIES'] = '1'
    os.environ.setdefault('MPLBACKEND', 'Agg')

    from helpers.constants import DEFAULT_EXPLORE
    from evaluations.wrapper_eval import capacity
    from core.deployment_setup import get_random_deployment
    from core.sparse_advertisements_v3 import Sparse_Advertisement_Eval
    from core.worker_comms import Worker_Manager
    from helpers.helpers import deployment_to_prefixes, threshold_a

    # resilience config follows the pipeline default (SCULPTOR_USE_RESILIENCE,
    # now default-on with wrapper_eval's gamma=4); set =0 for pure latency
    use_res = os.environ.get('SCULPTOR_USE_RESILIENCE', '1') == '1'
    # resilience requires gamma>0 (constructor asserts); gamma=0 runs
    # (hard objectives, hardB3v2) train resilience-free by definition
    _g = float(os.environ.get('SCULPTOR_ABLATION_GAMMA', '0.1') or 0)
    if _g <= 0:
        use_res = False
    # SCULPTOR_ABLATION_GAMMA overrides the default 0.1 (Tom 2026-08-15:
    # standalone default is ALWAYS 0.1, never wrapper_eval's 4 -- at small,
    # gamma=4 puts every iteration in the 'gradient very large' damp branch
    # (RB grads saturate the clip) and training freezes under threshold
    # semantics; ~0.1 keeps the resilience term active without destroying
    # the one-flip step dynamics).
    gamma_val = float(os.environ.get('SCULPTOR_ABLATION_GAMMA', '0.1')) if use_res else 0

    t0 = time.time()
    # GC scope = CWD-relative runs/ (matching where the solver actually
    # checkpoints, constants.RUN_DIR='runs'). Using _REPO_ROOT here made
    # cleanup reach across workspaces and delete OTHER processes' live
    # checkpoint dirs (the 105-run massacre + a smoke casualty).
    _runs_root = os.path.abspath('runs')
    _runs_before = set(os.listdir(_runs_root)) if os.path.isdir(_runs_root) else set()
    deployment = get_random_deployment(dpsize)
    deployment['port'] = port
    n_prefixes = deployment_to_prefixes(deployment)

    sas = Sparse_Advertisement_Eval(
        deployment, verbose=True, lambduh=0, with_capacity=capacity,
        explore=DEFAULT_EXPLORE, using_resilience_benefit=use_res, gamma=gamma_val,
        n_prefixes=n_prefixes, generic_objective=os.environ.get('SCULPTOR_ABLATION_OBJECTIVE', 'avg_latency'),
    )
    # 'no_mc' rung: swap the worker actor class for the deterministic
    # pseudo-path worker BEFORE the solve-phase workers start. The seam is
    # reverted right after solve so the pristine scoring stack below gets
    # stock workers. sculptor_fork._abl_assert_mc verifies the injection
    # actually took (a stock worker answers 'ERROR' to the stats RPC).
    import ray
    import core.worker_comms as worker_comms
    _stock_actor_cls = worker_comms_ray.ACTOR_CLS
    if os.environ.get('SCULPTOR_ABLATION_MC', '1') == '0':
        from experiments.ablation.mc_off_worker import Abl_MC_Off_Worker
        worker_comms_ray.ACTOR_CLS = ray.remote(Abl_MC_Off_Worker)
        print('[ablation-fork] mc-off worker class injected', flush=True)
    wm = Worker_Manager(sas.get_init_kwa(), deployment)
    wm.start_workers()
    result = {'seed': seed, 'rung': rung, 'max_iter': max_iter,
              'using_resilience': use_res, 'gamma': float(gamma_val)}
    try:
        sas.set_worker_manager(wm)
        sas.update_deployment(deployment)
        sas.solutions = {}

        # one_per_peering reference through the repo pipeline
        sas.solve_one_per_peering()
        opp_adv = sas.solutions['one_per_peering']['advertisement']
        result['opp_avg_lat'] = avg_lat(sas, opp_adv)
        result['opp_objective'] = float(sas.solutions['one_per_peering']['objective'])

        if rung == 'painter':
            # Painter is measurement-UNBOUNDED by construction: painter_v5
            # measures every iteration (measure_ingresses + stop_tracker's
            # measured_objective) and stops on convergence/max_n_iter, with
            # nothing counting those measurements. For budget-fair
            # comparison against the N-capped rungs, SCULPTOR_ABLATION_
            # PAINTER_BUDGET=N caps painter's loop at N iterations (Tom,
            # 2026-08-12). path_measures is recorded either way.
            _pb = os.environ.get('SCULPTOR_ABLATION_PAINTER_BUDGET')
            if _pb:
                from core.painter import Painter_Adv_Solver as _PAS
                _orig_init = _PAS.__init__

                def _budget_init(self, *a, **kw):
                    _orig_init(self, *a, **kw)
                    self.max_n_iter = int(_pb)
                _PAS.__init__ = _budget_init
                try:
                    sas.solve_painter()
                finally:
                    _PAS.__init__ = _orig_init
            else:
                sas.solve_painter()
            adv = sas.solutions['painter']['advertisement']
            result['repo_objective'] = float(sas.solutions['painter']['objective'])
            result['n_iters'] = None
            result['painter_budget'] = int(_pb) if _pb else None
            result['n_advs_measured'] = int(
                sas.solutions['painter'].get('n_advs') or -1)
            result['painter_iters'] = int(getattr(sas.painter, 'iter', -1))
        else:
            solver = Ablation_Sparse_Advertisement_Solver(
                sas.output_deployment(), **sas.get_init_kwa())
            solver.set_worker_manager(sas.get_worker_manager())
            solver.compute_one_per_peering_solution()
            try:
                solver.solve()
            except Exception as e:
                import traceback; traceback.print_exc()
                result['solve_error'] = str(e)
            try:
                # writes convergence_over_iterations.pdf etc. into the run
                # dir (same call + guard as the repo's solve_sparse; it can
                # IndexError on some runs -- old diagnostic code)
                solver.make_plots()
            except Exception as e:
                print('make_plots failed (non-fatal): {}'.format(e))
            adv = threshold_a(np.asarray(solver.optimization_advertisement, dtype=float))
            try:
                result['repo_objective'] = float(solver.measured_objective(adv))
            except Exception as e:
                # degenerate advertisements (e.g. all-off after a NaN step)
                # crash the repo objective; record and keep the run's data
                result['repo_objective'] = None
                result['repo_objective_error'] = str(e)
            # persist objective COMPONENTS (Tom 2026-08-18: the mlu
            # composite hides whether a panel delta is utilization or
            # latency tie-break). v2 fix (v5 scout shipped empty {}):
            # _steady_ret solves the PLAIN latency LP — components only
            # exist in the REGISTERED family objective's ret, so call it
            # directly on the adv's ground-truth ingress map.
            try:
                _objname = os.environ.get('SCULPTOR_ABLATION_OBJECTIVE',
                                          'avg_latency')
                from core.solve_lp_assignment import generic_lp_functions
                _fn = generic_lp_functions.get(_objname)
                _rti = None
                if _fn is not None:
                    _rti, _ = sas.calculate_ground_truth_ingress(adv)
                    _ret = _fn(sas, _rti, _objname)
                    if _ret and _ret.get('solved', True):
                        result['obj_components'] = {
                            k: float(_ret[k]) for k in
                            ('max_util', 'steady_avg_lat', 'bad_frac',
                             'mlu_alpha', 'frac_beyond',
                             'frac_beyond_capability', 'hinge_excess_ms')
                            if _ret.get(k) is not None}
                # lexicographic prio capability pair (Tom-ratified
                # 2026-08-18: priority strictly first, bulk best-effort
                # over the latency-optimal FACE; a PAIR, never summed)
                if _objname == 'joint_latency_bulk_download':
                    from core.hard_objectives import \
                        prio_lex_pair
                    if _rti is None:
                        _rti, _ = sas.calculate_ground_truth_ingress(adv)
                    _orti, _ = sas.calculate_ground_truth_ingress(opp_adv)
                    _L, _B = prio_lex_pair(sas, _rti)
                    _Lo, _Bo = prio_lex_pair(sas, _orti)
                    result['prio_lex'] = {
                        'Lstar': _L, 'bulk_frac': _B,
                        'opp_Lstar': _Lo, 'opp_bulk_frac': _Bo}
                    if (_L is not None and _Lo is not None
                            and _L > _Lo + 1e-6):
                        print('[prio-lex] WARNING: arm Lstar beats opp '
                              '({} > {}) — invariant violation'.format(
                                  _L, _Lo))
            except Exception as e:
                import traceback; traceback.print_exc()
                print('component persist failed (non-fatal): {}'.format(e))
            result['n_iters'] = int(getattr(solver, 'iter', -1))
            # anytime-performance (Tom 2026-08-19: quantify convergence
            # SPEED, not just exit iteration — exit includes the stop
            # rule's flat patience tail). First iteration reaching q of
            # the total believed-objective descent.
            try:
                _tr = solver.metrics.get('abl_belief_objective') or []
                if len(_tr) > 3:
                    _its = [t[0] for t in _tr]
                    _bs = [t[1] for t in _tr]
                    _b0 = _bs[0]
                    _bmin = min(_bs)
                    _drop = _b0 - _bmin
                    if _drop > 1e-9:
                        _res = {}
                        for _q in (0.5, 0.9, 0.95):
                            for _it, _b in zip(_its, _bs):
                                if (_b0 - _b) >= _q * _drop:
                                    _res[str(int(_q * 100))] = int(_it)
                                    break
                        result['iters_to'] = _res
            except Exception as _e:
                print('iters_to persist failed (non-fatal): {}'.format(_e))
            result['n_advs_measured'] = int(getattr(solver, 'path_measures', -1))
            result['nan_grad_iters'] = int(getattr(solver, 'abl_nan_grad_iters', 0))
            result['probe_mode'] = getattr(solver, 'abl_probe_mode', 'fixed')
            result['probes_spent'] = int(getattr(solver, 'abl_probes_spent', 0))
            result['exit_reason'] = getattr(solver, 'abl_exit_reason', None)
            result['probe_reasons'] = dict(getattr(
                solver, '_abl_probe_reasons', {}) or {})
            result['probe_skips'] = int(getattr(
                solver, '_abl_probe_skips', 0))
            result['remeasure_skips'] = int(getattr(
                solver, '_explore_remeasure_skips', 0))
            result['gate_hist'] = getattr(solver, '_abl_gate_hist', None)
            # decision-WHAT probe diagnostics (Tom 2026-08-16): per-probe
            # chosen coord/popp, score, p_err, sigma/U/belief before-after
            result['probe_log'] = getattr(solver, '_abl_probe_log', None)

        result['adv'] = np.asarray(adv).tolist()
        result['n_on'] = int(np.asarray(adv).sum())
    finally:
        try:
            wm.stop_workers()
        except Exception as e:
            print('warning: stop_workers raised {}'.format(e))
        # revert the mc-off actor-class injection so the scoring stack
        # (wm2 below) is built from stock workers
        worker_comms_ray.ACTOR_CLS = _stock_actor_cls

    # ---- scoring phase: PRISTINE eval stack ----
    # The solver's modify_ugs (pseudo-UG splitting, seed-dependent) mutates
    # the deployment held by the shared workers, which silently corrupts
    # any scoring done through the same stack. Rebuild everything fresh.
    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(seed)
    deployment2 = get_random_deployment(dpsize)
    deployment2['port'] = port + 400
    sas2 = Sparse_Advertisement_Eval(
        deployment2, verbose=False, lambduh=0, with_capacity=capacity,
        explore=DEFAULT_EXPLORE, using_resilience_benefit=False, gamma=0,
        n_prefixes=n_prefixes, generic_objective=os.environ.get('SCULPTOR_ABLATION_OBJECTIVE', 'avg_latency'),
    )  # scoring stack stays latency-only: the reported metric is unchanged
    wm2 = Worker_Manager(sas2.get_init_kwa(), deployment2)
    wm2.start_workers()
    try:
        sas2.set_worker_manager(wm2)
        sas2.update_deployment(deployment2)
        result['opp_avg_lat'] = avg_lat(sas2, opp_adv)
        result['avg_lat'] = avg_lat(sas2, np.asarray(adv))
        result['diff_vs_opp'] = result['avg_lat'] - result['opp_avg_lat']
    finally:
        try:
            wm2.stop_workers()
        except Exception as e:
            print('warning: wm2.stop_workers raised {}'.format(e))
    result['wall_s'] = round(time.time() - t0, 1)

    # Run-dir retention: the solver checkpoints every run into runs/<ts>/
    # (state pickles, figures, per-iter metrics; ~35MB each). We RETAIN
    # recent run dirs for debugging and garbage-collect old ones so long
    # sweeps can't fill the disk (the 19GB incident).
    #   SCULPTOR_ABLATION_RUNS_KEEP  (default 20): newest N run dirs kept
    #   SCULPTOR_ABLATION_KEEP_RUNS=1: never delete anything
    _srd = getattr(locals().get('solver', None), 'save_run_dir', None)
    # rename the run dir semantically: runs/ablation-<dpsize>-<rung>-dep<seed>
    # (suffixed -N<budget> under gated probing so grids over N never
    # collide on the same dir name within a workspace -- needed for
    # per-run figure/log harvesting)
    if _srd and os.path.isdir(_srd):
        import shutil
        _nsuf = ''
        _pmode = os.environ.get('SCULPTOR_ABLATION_PROBE_MODE', 'fixed')
        if _pmode in ('gated', 'scheduled', 'smart', 'adaptive', 'slotted'):
            _nsuf = '-N{}-{}'.format(
                os.environ.get('SCULPTOR_ABLATION_PROBE_N', '?'), _pmode)
        elif _pmode == 'fixed':
            # budgeted-fixed (L1 v2) is a real N-grid arm: carry N in the
            # dir/fig name or every N's convergence figure collides on
            # <rung>-dep<seed>-fixed.pdf (caught 2026-08-14: dash grid
            # could only link one cell)
            _nsuf = ('-N{}-fixed'.format(
                os.environ.get('SCULPTOR_ABLATION_PROBE_N', '?'))
                if os.environ.get('SCULPTOR_ABLATION_FIXED_BUDGET',
                                  '0') == '1'
                else '-fixed')
        _dst = os.path.join(os.path.dirname(_srd),
                            'ablation-{}-{}-dep{}{}'.format(dpsize, rung, seed, _nsuf))
        try:
            if os.path.isdir(_dst):
                shutil.rmtree(_dst, ignore_errors=True)  # rerun of same cell
            os.rename(_srd, _dst)
            _srd = _dst
        except OSError as e:
            print('warning: run-dir rename failed: {}'.format(e))
    result['save_run_dir'] = _srd
    if os.environ.get('SCULPTOR_ABLATION_KEEP_RUNS', '0') != '1':
        import shutil
        keep = int(os.environ.get('SCULPTOR_ABLATION_RUNS_KEEP', '20'))
        runs_root = os.path.abspath('runs')
        try:
            dirs = sorted((d for d in os.listdir(runs_root)
                           if os.path.isdir(os.path.join(runs_root, d))),
                          key=lambda d: os.path.getmtime(os.path.join(runs_root, d)),
                          reverse=True)
            for d in dirs[keep:]:
                shutil.rmtree(os.path.join(runs_root, d), ignore_errors=True)
        except OSError as e:
            print('warning: runs GC failed: {}'.format(e))

    # de-clutter: remove every run dir spawned during this invocation
    # (incidental solver/eval instances) except the semantically renamed one
    try:
        import shutil as _sh
        keep_name = os.path.basename(result.get('save_run_dir') or '')
        for d in set(os.listdir(_runs_root)) - _runs_before:
            if d != keep_name:
                _sh.rmtree(os.path.join(_runs_root, d), ignore_errors=True)
    except OSError as e:
        print('warning: stray run-dir cleanup failed: {}'.format(e))

    os.makedirs(out_dir, exist_ok=True)
    with open(out_fn, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print('[seed {} {}] avg_lat={:.3f} diff={:+.3f} iters={} ({:.0f}s)'.format(
        seed, rung, result['avg_lat'], result['diff_vs_opp'],
        result.get('n_iters'), result['wall_s']), flush=True)
    return out_fn


def main():
    # CLI wrapper; every knob beyond (seed, rung, iters, size) arrives via
    # SCULPTOR_* env so sweep scripts stay thin.
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--rung', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--max-iter', type=int, default=200)
    p.add_argument('--dpsize', default='small')
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()
    run_one(args.seed, args.rung, args.port, args.max_iter, args.out_dir,
            dpsize=args.dpsize)


if __name__ == '__main__':
    main()
