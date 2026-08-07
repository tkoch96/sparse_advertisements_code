"""Reference arm: run the REAL repo SCULPTOR (Sparse_Advertisement_Solver,
Ray workers + Gurobi) on the same seeded 'small' deployments as the
standalone ablation arms, and score its final advertisement with the SAME
standalone evaluator so the CDF is apples-to-apples.

Results are merged into the existing seed_<N>.json files under
arms['sparse_ref'].

Run AFTER run_ablation.py, sequentially (respects the Gurobi WLS
concurrent-session limit):

    python -m experiments.ablation.run_sparse_ref --n-seeds 10 \
        --max-iter 200 --out-dir cache/ablation/full
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


def run_one(seed, max_iter, port, out_dir):
    out_fn = os.path.join(out_dir, 'seed_{}.json'.format(seed))
    if not os.path.exists(out_fn):
        print('[seed {}] no ablation JSON at {}; run run_ablation.py first'.format(seed, out_fn))
        return None
    with open(out_fn) as f:
        result = json.load(f)
    if 'sparse_ref' in result['arms']:
        print('[seed {}] sparse_ref already present, skipping'.format(seed))
        return out_fn

    os.environ['SCULPTOR_MAX_ITER'] = str(max_iter)
    # Fair-budget reference: the ablation arms always run the full
    # max_iter, so hold the repo solver's convergence early-stop off
    # until the same budget is spent.
    os.environ['SCULPTOR_MIN_ITER'] = str(max_iter)
    os.environ['SCULPTOR_DISABLE_PARALLEL_STRATEGIES'] = '1'
    os.environ.setdefault('MPLBACKEND', 'Agg')

    # Build the standalone problem FIRST (it pins SCULPTOR_DEPLOYMENT_SEED and
    # the numpy RNG exactly as run_ablation.run_seed does, so the repo solver
    # below sees the byte-identical deployment).
    from experiments.ablation.common import Problem, threshold_a
    problem = Problem(seed)

    from constants import DEFAULT_EXPLORE
    from wrapper_eval import capacity
    from sparse_advertisements_v3 import Sparse_Advertisement_Eval
    from worker_comms import Worker_Manager
    from helpers import deployment_to_prefixes

    deployment = problem.deployment
    deployment['port'] = port
    n_prefixes = deployment_to_prefixes(deployment)

    t0 = time.time()
    sas = Sparse_Advertisement_Eval(
        deployment,
        verbose=True,
        lambduh=0.00001,          # negligible L1: pure-latency comparison
        with_capacity=capacity,
        explore=DEFAULT_EXPLORE,
        using_resilience_benefit=False,
        gamma=0,
        n_prefixes=n_prefixes,
        generic_objective='avg_latency',
    )
    wm = Worker_Manager(sas.get_init_kwa(), deployment)
    wm.start_workers()
    try:
        sas.set_worker_manager(wm)
        sas.update_deployment(deployment)
        sas.solutions = {}  # normally created inside compare_different_solutions
        sas.solve_sparse()
        adv = threshold_a(np.asarray(sas.solutions['sparse']['advertisement'], dtype=float))
    finally:
        try:
            wm.stop_workers()
        except Exception as e:
            print('warning: stop_workers raised {}'.format(e))

    final_obj = problem.evaluate(adv)
    r = {
        'arm': 'sparse_ref',
        'final_obj': final_obj,
        'diff_vs_opp': final_obj - result['opp_obj'],
        'n_measurements': int(sas.solutions['sparse'].get('n_advs', -1)),
        'iters_run': max_iter,
        'n_on': int(adv.sum()),
        'wall_s': round(time.time() - t0, 1),
        'repo_objective': float(sas.solutions['sparse']['objective']),
    }
    result['arms']['sparse_ref'] = r
    with open(out_fn, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print('[seed {}] sparse_ref: obj={:.2f} diff={:+.2f} ({:.0f}s)'.format(
        seed, r['final_obj'], r['diff_vs_opp'], r['wall_s']), flush=True)
    return out_fn


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--n-seeds', type=int, default=10)
    p.add_argument('--seed-base', type=int, default=1)
    p.add_argument('--max-iter', type=int, default=200)
    p.add_argument('--port-base', type=int, default=31700)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()

    for i, seed in enumerate(range(args.seed_base, args.seed_base + args.n_seeds)):
        run_one(seed, args.max_iter, args.port_base + i, args.out_dir)


if __name__ == '__main__':
    main()
