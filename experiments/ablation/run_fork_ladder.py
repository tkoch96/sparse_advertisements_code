"""Driver for the fork-based ablation ladder.

For one (seed, rung): builds the seeded 'small' deployment, spins up the
REAL repo worker stack, runs Ablation_Sparse_Advertisement_Solver with
that rung's feature flags, and scores the final advertisement with the
REPO's own evaluation (measured_objective + ground-truth LP user
latencies). one_per_peering (and optionally the repo painter) are
computed through the same pipeline for reference.

    python -m experiments.ablation.run_fork_ladder --seed 1 --rung full \
        --port 31800 --max-iter 200 --out-dir cache/ablation/fork_ladder

Rungs: full, expl_random, expl_none, no_direction, no_memory (see
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
    os.environ['SCULPTOR_MIN_ITER'] = str(max_iter)   # fair budget: no early stop
    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(seed)
    # canonical per-seed init: first rung writes it, all others assert equality
    os.makedirs(out_dir, exist_ok=True)
    os.environ['SCULPTOR_ABLATION_INIT_FILE'] = os.path.abspath(
        os.path.join(out_dir, 'init_dep{}.npy'.format(seed)))
    os.environ['SCULPTOR_DISABLE_PARALLEL_STRATEGIES'] = '1'
    os.environ.setdefault('MPLBACKEND', 'Agg')

    from constants import DEFAULT_EXPLORE
    from wrapper_eval import capacity, gamma as EVAL_GAMMA
    from deployment_setup import get_random_deployment
    from sparse_advertisements_v3 import Sparse_Advertisement_Eval
    from worker_comms import Worker_Manager
    from helpers import deployment_to_prefixes, threshold_a

    # resilience config follows the pipeline default (SCULPTOR_USE_RESILIENCE,
    # now default-on with wrapper_eval's gamma=4); set =0 for pure latency
    use_res = os.environ.get('SCULPTOR_USE_RESILIENCE', '1') == '1'
    # SCULPTOR_ABLATION_GAMMA overrides wrapper_eval's gamma (=4). At small,
    # gamma=4 puts every iteration in the 'gradient very large' damp branch
    # (RB grads saturate the clip) and training freezes; ~0.01 keeps the
    # resilience term active without destroying the one-flip step dynamics.
    gamma_val = float(os.environ.get('SCULPTOR_ABLATION_GAMMA', EVAL_GAMMA)) if use_res else 0

    t0 = time.time()
    _runs_root = os.path.join(_REPO_ROOT, 'runs')
    _runs_before = set(os.listdir(_runs_root)) if os.path.isdir(_runs_root) else set()
    deployment = get_random_deployment(dpsize)
    deployment['port'] = port
    n_prefixes = deployment_to_prefixes(deployment)

    sas = Sparse_Advertisement_Eval(
        deployment, verbose=True, lambduh=0.00001, with_capacity=capacity,
        explore=DEFAULT_EXPLORE, using_resilience_benefit=use_res, gamma=gamma_val,
        n_prefixes=n_prefixes, generic_objective='avg_latency',
    )
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
            sas.solve_painter()
            adv = sas.solutions['painter']['advertisement']
            result['repo_objective'] = float(sas.solutions['painter']['objective'])
            result['n_iters'] = None
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
            result['n_iters'] = int(getattr(solver, 'iter', -1))
            result['n_advs_measured'] = int(getattr(solver, 'path_measures', -1))
            result['nan_grad_iters'] = int(getattr(solver, 'abl_nan_grad_iters', 0))

        result['adv'] = np.asarray(adv).tolist()
        result['n_on'] = int(np.asarray(adv).sum())
    finally:
        try:
            wm.stop_workers()
        except Exception as e:
            print('warning: stop_workers raised {}'.format(e))

    # ---- scoring phase: PRISTINE eval stack ----
    # The solver's modify_ugs (pseudo-UG splitting, seed-dependent) mutates
    # the deployment held by the shared workers, which silently corrupts
    # any scoring done through the same stack. Rebuild everything fresh.
    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(seed)
    deployment2 = get_random_deployment(dpsize)
    deployment2['port'] = port + 400
    sas2 = Sparse_Advertisement_Eval(
        deployment2, verbose=False, lambduh=0.00001, with_capacity=capacity,
        explore=DEFAULT_EXPLORE, using_resilience_benefit=False, gamma=0,
        n_prefixes=n_prefixes, generic_objective='avg_latency',
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
    if _srd and os.path.isdir(_srd):
        import shutil
        _dst = os.path.join(os.path.dirname(_srd),
                            'ablation-{}-{}-dep{}'.format(dpsize, rung, seed))
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
        runs_root = os.path.join(_REPO_ROOT, 'runs')
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
