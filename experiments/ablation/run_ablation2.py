"""(seed, arm)-parallel ablation driver for large dpsizes (actual-N).

Differences from run_ablation.py:
  * tasks are (seed, arm) pairs -> per-task files seed_<N>_<arm>.json, so
    a 16-vCPU box is saturated even though one seed's arm chain is long
  * uses the fast arm implementations (arms_fast / painter_fast)
  * Problem/opp built once per task (deployment loads from the prebuilt
    disk cache in a few seconds)

plot_ablation.load() accepts both file shapes.

    python -m experiments.ablation.run_ablation2 --dpsize actual-10 \
        --n-seeds 12 --max-iter 200 --jobs 8 --out-dir cache/ablation/actual-10_full
"""
import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

ARM_ORDER = ['painter', 'greedy_mc', 'coord_mc', 'fullgrad', 'fullgrad_entropy']


def run_task(task):
    seed, arm, cfg_kwargs, out_dir = task
    # keep BLAS single-threaded; parallelism is at the process level
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    from experiments.ablation.common import Problem
    from experiments.ablation.arms import ArmConfig
    from experiments.ablation.arms_fast import get_fast_arm_funcs

    out_fn = os.path.join(out_dir, 'seed_{}_{}.json'.format(seed, arm))
    if os.path.exists(out_fn):
        print('[seed {} {}] already done, skipping'.format(seed, arm), flush=True)
        return out_fn

    cfg_kwargs = dict(cfg_kwargs)
    dpsize = cfg_kwargs.pop('dpsize')
    t0 = time.time()
    problem = Problem(seed, dpsize=dpsize)
    cfg = ArmConfig(seed=seed, **cfg_kwargs)
    opp = problem.one_per_peering_objective()
    r = get_fast_arm_funcs()[arm](problem, cfg)
    r['diff_vs_opp'] = r['final_obj'] - opp
    r['wall_s'] = round(time.time() - t0, 1)
    result = {
        'seed': seed,
        'dpsize': dpsize,
        'opp_obj': opp,
        'n_ug': problem.n_ug,
        'n_popp': problem.n_popp,
        'n_prefixes': problem.n_prefixes,
        'cfg': cfg_kwargs,
        'arms': {arm: r},
    }
    with open(out_fn, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print('[seed {} {}] obj={:.2f} diff={:+.2f} meas={} iters={} ({:.0f}s)'.format(
        seed, arm, r['final_obj'], r['diff_vs_opp'], r['n_measurements'],
        r['iters_run'], r['wall_s']), flush=True)
    return out_fn


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--n-seeds', type=int, default=12)
    p.add_argument('--seed-base', type=int, default=1)
    p.add_argument('--dpsize', default='actual-10')
    p.add_argument('--max-iter', type=int, default=200)
    p.add_argument('--probe-budget', type=int, default=60)
    p.add_argument('--n-mc', type=int, default=5)
    p.add_argument('--arms', default=','.join(ARM_ORDER))
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    arms = args.arms.split(',')
    cfg_kwargs = {
        'max_iter': args.max_iter,
        'probe_budget': args.probe_budget,
        'n_mc': args.n_mc,
        'dpsize': args.dpsize,
    }
    seeds = list(range(args.seed_base, args.seed_base + args.n_seeds))
    # interleave arms so long tasks spread across the pool early: for each
    # arm index, all seeds (slowest arms first)
    order = ['fullgrad_entropy', 'fullgrad', 'coord_mc', 'greedy_mc', 'painter']
    tasks = [(s, a, cfg_kwargs, args.out_dir)
             for a in order if a in arms for s in seeds]

    t0 = time.time()
    if args.jobs > 1:
        from multiprocessing import get_context
        with get_context('spawn').Pool(args.jobs) as pool:
            for fn in pool.imap_unordered(run_task, tasks):
                print('[written] {}'.format(fn), flush=True)
    else:
        for task in tasks:
            run_task(task)
    print('ALL TASKS DONE in {:.0f}s -> {}'.format(time.time() - t0, args.out_dir))


if __name__ == '__main__':
    main()
