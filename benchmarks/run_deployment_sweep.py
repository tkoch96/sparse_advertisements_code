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
  SCULPTOR_DEPLOYMENT_SWEEP_NSIM   -- nsim per dpsize. Single int (applied
                                       to all sizes, default 1) OR a
                                       comma-separated list parallel to
                                       SCULPTOR_DEPLOYMENT_SWEEP_SIZES
                                       (mirrors evaluate_over_deployment_sizes
                                       n_sim_by_dpsize). Example: SIZES=25,32
                                       NSIM=3,2 -> 3 sims at 25, 2 at 32.
  SCULPTOR_DEPLOYMENT_SWEEP_SIZES  -- comma-separated explicit dpsize list
                                       (default: 3,5,10,15,20,25,<n_vultr>).
                                       Use to re-run only specific sizes.
  SCULPTOR_DEPLOYMENT_SWEEP_TAG    -- run tag suffix (default: dep_sweep).
                                       Namespaces the per-dpsize eval pickle.
"""
from __future__ import annotations

import argparse
import gc
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


def _log_mem(tag, **extra):
    """Minimal /proc-based driver memory snapshot. Mirrors the helper in
    sparse_advertisements_v3.py so the sweep launcher can drop a marker at
    dpsize boundaries without importing v3 just for this."""
    rss_kb = sys_avail_kb = -1
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    rss_kb = int(line.split()[1]); break
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    sys_avail_kb = int(line.split()[1]); break
    except (FileNotFoundError, PermissionError):
        return
    extras = ' '.join(f'{k}={v}' for k, v in extra.items())
    print(f'[mem] tag={tag} rss_mb={rss_kb//1024} sys_avail_mb={sys_avail_kb//1024} '
          f'pid={os.getpid()} t={time.time():.2f} {extras}', flush=True)


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
    metrics['compare_rets'][random_iter]['n_advs'] guard.

    Note: this only checks random_iter=0. Callers that care about full
    nsim coverage should use _count_random_iters_done().
    """
    return _count_random_iters_done(dpsize, tag, 1) >= 1


def _count_random_iters_done(dpsize, tag, max_check):
    """Return how many random_iters of this dpsize have populated
    compare_rets in the eval pickle. Counts contiguously from 0 (i.e.
    if compare_rets[0] is populated but compare_rets[2] isn't and we
    haven't reached compare_rets[2] yet, returns 1).

    Used by the hot-start path so we can hot-start random_iter=N when
    random_iters 0..N-1 are already done but N hasn't started/finished.
    """
    os.environ['SCULPTOR_RUN_TAG'] = f"{tag}_{dpsize}"
    pkl_path = global_performance_metrics_fn(f"testing_feature-actual-{dpsize}")
    if not os.path.exists(pkl_path):
        return 0
    try:
        m = pickle.load(open(pkl_path, 'rb'))
    except Exception:
        return 0
    compare_rets = m.get('compare_rets', {}) or {}
    done = 0
    for ri in range(max_check):
        cr = compare_rets.get(ri)
        if not isinstance(cr, dict):
            break
        n_advs = cr.get('n_advs')
        if not isinstance(n_advs, dict):
            break
        if not all(isinstance(v, list) and len(v) > 0 for v in n_advs.values()):
            break
        done += 1
    return done


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

    nsim_env = os.environ.get('SCULPTOR_DEPLOYMENT_SWEEP_NSIM', '1')
    nsim_parts = [int(x.strip()) for x in nsim_env.split(',') if x.strip()]
    if len(nsim_parts) == 1:
        nsim_by_dpsize = {dp: nsim_parts[0] for dp in dpsizes}
    elif len(nsim_parts) == len(dpsizes):
        nsim_by_dpsize = dict(zip(dpsizes, nsim_parts))
    else:
        raise SystemExit(
            f"SCULPTOR_DEPLOYMENT_SWEEP_NSIM={nsim_env!r} has {len(nsim_parts)} "
            f"values but SIZES has {len(dpsizes)} ({dpsizes}). Pass a single "
            f"int or a list of the same length.")
    tag = os.environ.get('SCULPTOR_DEPLOYMENT_SWEEP_TAG', 'dep_sweep')

    print("="*72)
    print(f"=== deployment sweep ===")
    print(f"  dpsizes:           {dpsizes}")
    print(f"  nsim per dpsize:   {[nsim_by_dpsize[dp] for dp in dpsizes]}")
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
    _log_mem('sweep_start')
    for dpsize in dpsizes:
        dp_start = time.time()
        _log_mem('dpsize_start', dpsize=dpsize)
        # Use the same naming convention as pull_results_new for cache reuse
        dpsize_str = f"testing_feature-actual-{dpsize}"
        # Tag the per-dpsize eval pickle so this run is separable from any
        # leftover state (per evaluate_all_metrics' SCULPTOR_RUN_TAG handling).
        os.environ['SCULPTOR_RUN_TAG'] = f"{tag}_{dpsize}"
        nsim_dp = nsim_by_dpsize[dpsize]
        print(f"\n{'='*72}", flush=True)
        print(f"[sweep] === dpsize={dpsize}  dpsize_str={dpsize_str}  nsim={nsim_dp} ===", flush=True)
        print(f"{'='*72}", flush=True)

        # Hot-start logic (NSIM-aware):
        # - Count how many random_iters are already done in the eval pickle.
        # - If all `nsim_dp` random_iters are done, training is skipped entirely;
        #   pass save_run_dir=None and evaluate_all_metrics will only fill
        #   missing eval phases.
        # - If some are done and some aren't, build a per-random_iter list:
        #   None for done ones (skipped by inner loop), latest-run-dir for the
        #   FIRST un-done one (hot-starts from state-N.pkl), None for any
        #   further un-done ones (start fresh in a new run dir; we never
        #   point multiple un-done iters at the same hot-start dir because
        #   they'd corrupt each other's state-N.pkl writes).
        # - If none are done and we have a prior run dir, hot-start
        #   random_iter=0 from it.
        n_done = _count_random_iters_done(dpsize, tag, nsim_dp)
        save_run_dir_for_dp = None
        if n_done >= nsim_dp:
            print(f"[sweep] dpsize={dpsize} all {n_done}/{nsim_dp} random_iters complete; "
                  f"evaluate_all_metrics will skip training and only fill missing eval phases.",
                  flush=True)
        else:
            cand = _find_save_run_dir_for_dpsize(dpsize)
            if nsim_dp > 1:
                # Always pass a list of length nsim_dp when nsim>1
                # (evaluate_all_metrics asserts on shape).
                save_run_dir_for_dp = [None] * nsim_dp
                if cand:
                    save_run_dir_for_dp[n_done] = cand   # hot-start the first un-done
                    try:
                        n_states = len([f for f in os.listdir(os.path.join(RUN_DIR, cand))
                                        if f.startswith('state-') and f.endswith('.pkl')])
                    except OSError:
                        n_states = '?'
                    print(f"[sweep] dpsize={dpsize} {n_done}/{nsim_dp} done; "
                          f"HOT-START random_iter={n_done} from {cand} ({n_states} saves); "
                          f"random_iters {n_done+1}..{nsim_dp-1} start fresh.",
                          flush=True)
                else:
                    print(f"[sweep] dpsize={dpsize} {n_done}/{nsim_dp} done; "
                          f"no prior run dir; remaining random_iters start fresh.",
                          flush=True)
            else:
                # nsim_dp == 1 -- single value, original semantics.
                if cand:
                    save_run_dir_for_dp = cand
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
                save_run_dir=save_run_dir_for_dp, nsim=nsim_dp)
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
        # Force GC at dpsize boundary so any unreferenced state from this
        # dpsize's SAS / wm / metrics dict doesn't carry into the next
        # dpsize's iov_enter (session 9 forensics showed ~6 GB sticking
        # across the transition). Cheap, no risk -- if there's nothing
        # to collect, gc.collect() just returns quickly.
        n_collected = gc.collect()
        dp_wall = time.time() - dp_start
        print(f"[sweep] dpsize={dpsize} done in {dp_wall:.1f}s "
              f"(cumulative {time.time()-overall_start:.1f}s); "
              f"gc.collect freed {n_collected} objects", flush=True)
        _log_mem('dpsize_done', dpsize=dpsize, wall_s=int(dp_wall), gc_freed=n_collected)

    overall = time.time() - overall_start
    print(f"\n[sweep] ALL DONE in {overall:.1f}s ({overall/60:.1f} min). "
          f"Wrote {len(metrics_by_dpsize)} dpsizes to {cache_fn}", flush=True)


if __name__ == '__main__':
    main()
