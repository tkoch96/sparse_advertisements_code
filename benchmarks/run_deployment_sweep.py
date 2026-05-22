"""Run evaluate_over_deployment_sizes with low nsim for fast turnaround.

Replicates pull_results_new from evaluate_over_deployment_sizes.py but with
n_sim_by_dpsize overridable from the env so we can start with nsim=1 across
all sizes (results cache into cache/testing_feature_cache_fn.pkl and per-size
eval pickles, so a follow-up run with higher nsim just adds trials).

The dpsizes are [3, 5, 10, 15, 20, 25, len(POP_TO_LOC['vultr'])] (= 32 for
the current Vultr topology), matching the paper's deployment-size sweep.

Usage (on the cluster head):
  cd /home/ubuntu/sparse_advertisements_code
  SCULPTOR_MAX_ITER=200 \\
  SCULPTOR_N_WORKERS=32 \\
  SCULPTOR_CAPACITY_HEADROOM=0.2 \\
  SCULPTOR_DEPLOYMENT_SWEEP_NSIM=1 \\
  PYTHONUNBUFFERED=1 \\
  /home/ubuntu/venv312/bin/python -u benchmarks/run_deployment_sweep.py --port 31520

Env vars:
  SCULPTOR_DEPLOYMENT_SWEEP_NSIM   -- nsim per dpsize (default 1). Override
                                       to scale up later runs.
  SCULPTOR_DEPLOYMENT_SWEEP_SIZES  -- comma-separated explicit dpsize list
                                       (default: 3,5,10,15,20,25,<n_vultr>).
                                       Use to re-run only specific sizes.
  SCULPTOR_DEPLOYMENT_SWEEP_TAG    -- run tag suffix (default: dep_sweep).
                                       Namespaces the per-dpsize eval pickle.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

# Project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
os.chdir(_REPO)

# Enable the early-stop + adaptive-budget fixes by default for the sweep.
# Both validated locally on small × 200 to deliver ~1.5x training speedup
# with no measurable quality regression on the failure-eval (paper-metric
# %≤10ms actually improved 84.9% → 90.9% at small, within single-seed
# noise envelope). Override with SCULPTOR_STOP_DROP_ADV_DELTA=0 /
# SCULPTOR_ADAPTIVE_PROBE_BUDGET=0 if needed.
os.environ.setdefault('SCULPTOR_STOP_DROP_ADV_DELTA', '1')
os.environ.setdefault('SCULPTOR_ADAPTIVE_PROBE_BUDGET', '1')

# Alias worker_comms -> worker_comms_ray (mirrors run_ray.py)
import worker_comms_ray as _ray_mod  # noqa: E402
sys.modules['worker_comms'] = _ray_mod

from constants import POP_TO_LOC, CACHE_DIR, RUN_DIR  # noqa: E402
from eval_latency_failure import evaluate_all_metrics  # noqa: E402
from wrapper_eval import global_performance_metrics_fn  # noqa: E402


def _find_save_run_dir_for_dpsize(dpsize):
    """Find the latest <ts>-testing_feature-actual-{dpsize}-sparse directory
    in RUN_DIR. Used for hot-starting sparse training from a previous run's
    saved state (e.g. after killing a sweep mid-training and restarting)."""
    suffix = f"-testing_feature-actual-{dpsize}-sparse"
    candidates = []
    if not os.path.isdir(RUN_DIR):
        return None
    for name in os.listdir(RUN_DIR):
        if not name.endswith(suffix):
            continue
        # Confirm at least one state-*.pkl is present
        path = os.path.join(RUN_DIR, name)
        try:
            has_state = any(f.startswith('state-') and f.endswith('.pkl')
                            for f in os.listdir(path))
        except OSError:
            has_state = False
        if not has_state:
            continue
        # Filenames are <unix_ts>-<dpsize>-sparse; sort by timestamp prefix
        try:
            ts = int(name.split('-')[0])
        except ValueError:
            continue
        candidates.append((ts, name))
    if not candidates:
        return None
    candidates.sort(reverse=True)  # latest first
    return candidates[0][1]


def _dpsize_already_completed(dpsize, tag):
    """Check whether this dpsize's eval pickle is fully populated -- i.e.,
    compare_rets[0]['n_advs'] exists and has data for every strategy. If
    so, evaluate_all_metrics will skip strategy_compare via its own
    metrics['compare_rets'][random_iter]['n_advs'] guard."""
    os.environ['SCULPTOR_RUN_TAG'] = f"{tag}_{dpsize}"
    pkl_path = global_performance_metrics_fn(f"testing_feature-actual-{dpsize}")
    if not os.path.exists(pkl_path):
        return False
    try:
        m = pickle.load(open(pkl_path, 'rb'))
    except Exception:
        return False
    cr = m.get('compare_rets', {}).get(0)
    if not isinstance(cr, dict): return False
    n_advs = cr.get('n_advs')
    if not isinstance(n_advs, dict): return False
    # Need every soln_type to have at least one entry
    return all(isinstance(v, list) and len(v) > 0 for v in n_advs.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--cache-name", default="testing_feature_cache_fn.pkl",
                        help="Top-level metrics-by-dpsize pickle (default matches "
                             "evaluate_over_deployment_sizes.pull_results_new).")
    args = parser.parse_args()

    # Default dpsizes match evaluate_over_deployment_sizes.pull_results_new
    default_sizes = [3, 5, 10, 15, 20, 25, len(POP_TO_LOC['vultr'])]
    sizes_env = os.environ.get('SCULPTOR_DEPLOYMENT_SWEEP_SIZES', '')
    if sizes_env:
        dpsizes = [int(x.strip()) for x in sizes_env.split(',') if x.strip()]
    else:
        dpsizes = default_sizes

    nsim = int(os.environ.get('SCULPTOR_DEPLOYMENT_SWEEP_NSIM', '1'))
    tag = os.environ.get('SCULPTOR_DEPLOYMENT_SWEEP_TAG', 'dep_sweep')

    print("="*72)
    print(f"=== deployment sweep ===")
    print(f"  dpsizes:           {dpsizes}")
    print(f"  nsim per dpsize:   {nsim}")
    print(f"  MAX_ITER:          {os.environ.get('SCULPTOR_MAX_ITER', '(default)')}")
    print(f"  N_WORKERS:         {os.environ.get('SCULPTOR_N_WORKERS', '(default)')}")
    print(f"  CAPACITY_HEADROOM: {os.environ.get('SCULPTOR_CAPACITY_HEADROOM', '(default)')}")
    print(f"  STOP_DROP_ADV_DELTA:    {os.environ.get('SCULPTOR_STOP_DROP_ADV_DELTA')}")
    print(f"  ADAPTIVE_PROBE_BUDGET:  {os.environ.get('SCULPTOR_ADAPTIVE_PROBE_BUDGET')}")
    print(f"  tag:               {tag}")
    print(f"  port:              {args.port}")
    print("="*72, flush=True)

    cache_fn = os.path.join(CACHE_DIR, args.cache_name)
    if os.path.exists(cache_fn):
        metrics_by_dpsize = pickle.load(open(cache_fn, 'rb'))
        print(f"[sweep] loaded existing cache {cache_fn} with {len(metrics_by_dpsize)} dpsizes")
    else:
        metrics_by_dpsize = {}
        print(f"[sweep] no existing cache, starting fresh")

    overall_start = time.time()
    for dpsize in dpsizes:
        dp_start = time.time()
        # Use the same naming convention as pull_results_new for cache reuse
        dpsize_str = f"testing_feature-actual-{dpsize}"
        # Tag the per-dpsize eval pickle so this run is separable from any
        # leftover state (per evaluate_all_metrics' SCULPTOR_RUN_TAG handling).
        os.environ['SCULPTOR_RUN_TAG'] = f"{tag}_{dpsize}"
        print(f"\n{'='*72}", flush=True)
        print(f"[sweep] === dpsize={dpsize}  dpsize_str={dpsize_str}  nsim={nsim} ===", flush=True)
        print(f"{'='*72}", flush=True)

        # Hot-start logic:
        # - If the per-dpsize eval pickle has compare_rets[0]['n_advs']
        #   already populated, every soln_type already trained -- skip
        #   save_run_dir so evaluate_all_metrics enters its
        #   "already calculated, continue" branch and only re-runs missing
        #   eval phases.
        # - Otherwise, look in RUN_DIR for the latest
        #   <ts>-testing_feature-actual-{dp}-sparse directory. If one
        #   exists, pass it as save_run_dir so sparse hot-starts from its
        #   last saved state (state-N.pkl, saved every 5 iters).
        save_run_dir_for_dp = None
        if _dpsize_already_completed(dpsize, tag):
            print(f"[sweep] dpsize={dpsize} compare_rets already populated; "
                  f"evaluate_all_metrics will skip training and only fill missing eval phases.",
                  flush=True)
        else:
            cand = _find_save_run_dir_for_dpsize(dpsize)
            if cand:
                save_run_dir_for_dp = cand
                # Count saved state files for visibility
                try:
                    n_states = len([f for f in os.listdir(os.path.join(RUN_DIR, cand))
                                    if f.startswith('state-') and f.endswith('.pkl')])
                except OSError:
                    n_states = '?'
                print(f"[sweep] dpsize={dpsize} HOT-START from {cand} ({n_states} saved state files)",
                      flush=True)
            else:
                print(f"[sweep] dpsize={dpsize} no prior save_run_dir; starting fresh",
                      flush=True)

        try:
            metrics = evaluate_all_metrics(
                dpsize_str, args.port,
                save_run_dir=save_run_dir_for_dp, nsim=nsim)
        except KeyboardInterrupt:
            print(f"[sweep] interrupted during dpsize={dpsize}", flush=True)
            raise
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"[sweep] dpsize={dpsize} failed; continuing to next size", flush=True)
            continue

        # Extract just the stats_* keys (matches pull_results_new)
        metrics_by_dpsize[dpsize] = {k: v for k, v in metrics.items() if 'stats' in k}
        pickle.dump(metrics_by_dpsize, open(cache_fn, 'wb'))
        dp_wall = time.time() - dp_start
        print(f"[sweep] dpsize={dpsize} done in {dp_wall:.1f}s "
              f"(cumulative {time.time()-overall_start:.1f}s)", flush=True)

    overall = time.time() - overall_start
    print(f"\n[sweep] ALL DONE in {overall:.1f}s ({overall/60:.1f} min). "
          f"Wrote {len(metrics_by_dpsize)} dpsizes to {cache_fn}", flush=True)


if __name__ == '__main__':
    main()
