"""Driver for the PAINTER -> SCULPTOR feature-ladder ablation.

For each seed: build one 'small' deployment, compute the one_per_peering
reference objective, run every arm on the SAME instance, and write a JSON
with per-arm final objectives + traces. Seeds run in parallel processes
(scipy/HiGHS only -- no Gurobi sessions consumed).

Smoke:
    python -m experiments.ablation.run_ablation --n-seeds 3 --max-iter 20 \
        --out-dir cache/ablation/smoke
Full:
    python -m experiments.ablation.run_ablation --n-seeds 30 --max-iter 200 \
        --jobs 4 --out-dir cache/ablation/full

Afterwards:
    python -m experiments.ablation.plot_ablation --in-dir <out-dir>
"""
import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def run_seed(task):
    seed, cfg_kwargs, arms, out_dir = task
    from experiments.ablation.common import Problem
    from experiments.ablation.arms import ArmConfig, ARM_FUNCS

    out_fn = os.path.join(out_dir, 'seed_{}.json'.format(seed))
    if os.path.exists(out_fn):
        print('[seed {}] already done, skipping'.format(seed), flush=True)
        return out_fn

    t0 = time.time()
    # dpsize rides inside cfg_kwargs so the task-tuple shape stays stable
    # (spawn workers re-import this module mid-run).
    cfg_kwargs = dict(cfg_kwargs)
    dpsize = cfg_kwargs.pop('dpsize', 'small')
    problem = Problem(seed, dpsize=dpsize)
    cfg = ArmConfig(seed=seed, **cfg_kwargs)
    opp = problem.one_per_peering_objective()
    result = {
        'seed': seed,
        'dpsize': dpsize,
        'opp_obj': opp,
        'n_ug': problem.n_ug,
        'n_popp': problem.n_popp,
        'n_prefixes': problem.n_prefixes,
        'cfg': cfg_kwargs,
        'arms': {},
    }
    for arm in arms:
        ta = time.time()
        r = ARM_FUNCS[arm](problem, cfg)
        r['diff_vs_opp'] = r['final_obj'] - opp
        r['wall_s'] = round(time.time() - ta, 1)
        result['arms'][arm] = r
        print('[seed {}] {}: obj={:.2f} diff={:+.2f} meas={} iters={} ({:.0f}s)'.format(
            seed, arm, r['final_obj'], r['diff_vs_opp'], r['n_measurements'],
            r['iters_run'], r['wall_s']), flush=True)
    result['wall_s'] = round(time.time() - t0, 1)
    with open(out_fn, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    return out_fn


def main():
    from experiments.ablation.arms import ARM_ORDER
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--n-seeds', type=int, default=3)
    p.add_argument('--seed-base', type=int, default=1, help='Seeds are seed_base..seed_base+n-1.')
    p.add_argument('--dpsize', default='small')
    p.add_argument('--max-iter', type=int, default=20)
    p.add_argument('--probe-budget', type=int, default=60)
    p.add_argument('--n-mc', type=int, default=5)
    p.add_argument('--arms', default=','.join(ARM_ORDER))
    p.add_argument('--jobs', type=int, default=1)
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
    tasks = [(s, cfg_kwargs, arms, args.out_dir) for s in seeds]

    t0 = time.time()
    if args.jobs > 1:
        from multiprocessing import get_context
        with get_context('spawn').Pool(args.jobs) as pool:
            for fn in pool.imap_unordered(run_seed, tasks):
                print('[done] {}'.format(fn), flush=True)
    else:
        for task in tasks:
            run_seed(task)
    print('ALL SEEDS DONE in {:.0f}s -> {}'.format(time.time() - t0, args.out_dir))


if __name__ == '__main__':
    main()
