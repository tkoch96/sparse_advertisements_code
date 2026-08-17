"""One EODS cell = one (dpsize, sim) unit of evaluate_over_deployment_sizes,
speaking the queue's run_fork_ladder CLI + result-JSON convention so the
existing queue / governor / dash / fleet machinery runs it unchanged.

Mapping (Tom 2026-08-17, EODS modernization):
  --seed S     -> sim index (SCULPTOR_DEPLOYMENT_SEED=S; one sim per cell)
  --rung       -> 'eods' (fixed)
  --dpsize     -> actual-N token, e.g. 'actual-10' (cell translates to the
                  classic 'testing_feature-actual-N' string)
  --out-dir    -> store dir; writes seed_<S>_eods.json (queue done-marker,
                  n_iters=1) + seed_<S>_metrics.pkl (the per-cell
                  checkpoint pickle, evaluate_all_metrics' own format via
                  use_performance_metrics_fn)
  --max-iter   -> ignored (kept for CLI compat)

Strategy set via SCULPTOR_SOLN_TYPES (comma list). Merge with
experiments.eods.merge_eods into the classic metrics_by_dpsize cache so
evaluate_over_deployment_sizes' paper plots run unchanged.
"""
import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--rung', default='eods')
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--max-iter', type=int, default=0)
    ap.add_argument('--dpsize', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    out_fn = os.path.join(args.out_dir,
                          'seed_{}_eods.json'.format(args.seed))
    if os.path.exists(out_fn):
        print('[eods] exists, skipping', out_fn)
        return
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(args.seed)
    os.environ.setdefault('MPLBACKEND', 'Agg')

    n_sites = args.dpsize.replace('actual-', '')
    dpsize_str = 'testing_feature-actual-{}'.format(n_sites)
    pkl_fn = os.path.join(args.out_dir,
                          'seed_{}_metrics.pkl'.format(args.seed))

    t0 = time.time()
    from actual_deployment_eval_latency_failure import evaluate_all_metrics
    metrics = evaluate_all_metrics(
        dpsize_str, args.port, nsim=1,
        use_performance_metrics_fn=pkl_fn)

    rec = {'seed': args.seed, 'rung': 'eods', 'dpsize': args.dpsize,
           'n_iters': 1,   # progress convention: 1 sim per cell
           'soln_types': sorted(
               os.environ.get('SCULPTOR_SOLN_TYPES', 'painter').split(',')),
           'metrics_pkl': os.path.basename(pkl_fn),
           'stats_keys': sorted(k for k in (metrics or {})
                                if str(k).startswith('stats_')),
           'wall_s': round(time.time() - t0, 1),
           'lp_backend': os.environ.get('SCULPTOR_LP_BACKEND', 'gurobi')}
    with open(out_fn, 'w') as f:
        json.dump(rec, f)
    print('[eods] done {} sim {} in {:.0f}s -> {}'.format(
        args.dpsize, args.seed, rec['wall_s'], out_fn), flush=True)


if __name__ == '__main__':
    main()
