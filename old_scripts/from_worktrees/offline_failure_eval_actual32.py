"""Offline failure-eval against the recovered actual-32 advertisements.

Loads:
  - state-202.pkl: deployment + SCULPTOR advertisement at iter 202
  - popp_failure_latency_comparison_actual-32.pkl: all 6 strategies' advs

For each strategy:
  - assess_failure_resilience(adv, which='popps')  → popp-failure latency Δ
  - assess_failure_resilience(adv, which='pops')   → pop-failure latency Δ

Prints aggregated metrics: mean Δ, % within {10, 50, 100} ms of optimal.

Spins up a local Ray cluster (no AWS needed). Worker count tunable via
SCULPTOR_N_WORKERS env var (default 4).

Run:
  cd ~/Documents/sparse_advertisements_code
  ~/Documents/venv312/bin/python benchmarks/offline_failure_eval_actual32.py
"""
import os, sys, time, copy, pickle
import numpy as np

# Make the project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray

# Alias worker_comms to the Ray version (mirrors run_ray.py)
import worker_comms_ray as _ray_mod
sys.modules['worker_comms'] = _ray_mod

from constants import NO_ROUTE_LATENCY
from sparse_advertisements_v3 import Sparse_Advertisement_Eval
from worker_comms_ray import Worker_Manager
from wrapper_eval import assess_failure_resilience


NO_ROUTE_LATENCY_LOCAL = NO_ROUTE_LATENCY  # 30000
STATE_PKL = os.path.expanduser(
    '~/Documents/sparse_advertisements_code/recovered_actual32/state-202.pkl')
COMPARE_PKL = os.path.expanduser(
    '~/Documents/sparse_advertisements_code/recovered_actual32/popp_failure_latency_comparison_actual-32.pkl')

STRATS = ['sparse', 'painter', 'anyopt', 'one_per_pop', 'anycast', 'one_per_peering']
THRESHOLDS = [-10, -50, -100]


def summarize_rows(rows, total_vol, n_failure_iters):
    """Compute global-mean Δ + % within thresholds, treating unaffected
    (user, failure) pairs as Δ=0. Matches the 'Option 1' aggregation
    used in the actual-10 N=3 cross-seed writeup."""
    denom = total_vol * n_failure_iters
    sum_vol_diff = 0.0
    sum_vol_aff = 0.0
    sum_within = {th: 0.0 for th in THRESHOLDS}
    for fields in rows:
        if len(fields) == 6:
            _, vol, _, _, perf1, perf2 = fields
        else:
            _, vol, _, _, perf1, perf2, _ = fields
        sum_vol_aff += vol
        if perf1 == NO_ROUTE_LATENCY_LOCAL or perf2 == NO_ROUTE_LATENCY_LOCAL:
            continue
        d_v = perf1 - perf2
        sum_vol_diff += vol * d_v
        for th in THRESHOLDS:
            if d_v >= th:
                sum_within[th] += vol
    unaffected = max(0.0, denom - sum_vol_aff)
    out = {
        'mean_diff': sum_vol_diff / denom if denom > 0 else float('nan'),
        'pct_aff': 100.0 * sum_vol_aff / denom if denom > 0 else 0.0,
    }
    for th in THRESHOLDS:
        out[f'pct_within_{abs(th)}ms'] = 100.0 * (sum_within[th] + unaffected) / denom if denom > 0 else 0.0
    return out


def main():
    print("=== offline failure-eval (actual-32 recovered) ===")

    print(f"\nLoading {STATE_PKL}")
    state = pickle.load(open(STATE_PKL, 'rb'))
    deployment = state['deployment']
    print(f"  n_ugs={len(deployment['ug_perfs'])}, n_popps={len(deployment['popps'])}")

    print(f"\nLoading {COMPARE_PKL}")
    cmp_data = pickle.load(open(COMPARE_PKL, 'rb'))
    advs = cmp_data['compare_rets'][0]['adv_solns']
    print(f"  strategies with adv: {sorted(advs.keys())}")
    ug_vols_raw = cmp_data['ug_to_vol'][0]
    if isinstance(ug_vols_raw, dict):
        total_vol = float(sum(ug_vols_raw.values()))
    else:
        total_vol = float(np.sum(np.asarray(ug_vols_raw, dtype=float)))
    print(f"  total deployment volume: {total_vol:.1f}")

    # Set the run-tag so our partial-eval pickle is namespaced
    os.environ['SCULPTOR_RUN_TAG'] = 'offline_failure_eval'

    print("\n=== ray.init() ===")
    ray.init(num_cpus=int(os.environ.get('SCULPTOR_N_WORKERS', 4)) + 2,
             ignore_reinit_error=True, log_to_driver=False)

    # Pre-cleanup: pop any lingering env vars that would taint the eval
    os.environ.pop('SCULPTOR_CAPACITY_HEADROOM', None)  # eval should see full caps

    print("\n=== constructing SAS ===")
    sas = Sparse_Advertisement_Eval(
        deployment, verbose=True, lambduh=0.000007, with_capacity=True,
        explore='entropy', using_resilience_benefit=True, gamma=1.0,
        n_prefixes=advs['sparse'][0].shape[1],
        generic_objective='avg_latency')

    print("\n=== starting Ray workers ===")
    wm = Worker_Manager(sas.get_init_kwa(), deployment)
    wm.start_workers()
    sas.set_worker_manager(wm)
    sas.update_deployment(deployment)
    print(f"  workers up: {sas.get_n_workers()}")

    results = {}
    for which in ['popps', 'pops']:
        results[which] = {}
        for strat in STRATS:
            try:
                adv = advs[strat][0]
            except (KeyError, IndexError):
                print(f"\n[skip] {strat}: no adv in compare_rets")
                continue
            print(f"\n=== {strat} | which={which} ===")
            t0 = time.time()
            ret = assess_failure_resilience(sas, adv, which=which)
            elapsed = time.time() - t0
            rows = ret['mutable']['latency_delta_specific']
            iters = set()
            for f in rows:
                iters.add(f[3] if len(f) >= 6 else None)
            n_iters = max(1, len(iters))
            summary = summarize_rows(rows, total_vol, n_iters)
            results[which][strat] = summary
            print(f"  elapsed: {elapsed:.1f}s | n_failure_scenarios={n_iters} | rows={len(rows)}")
            print(f"  mean Δ:      {summary['mean_diff']:+.3f} ms")
            print(f"  % within 10ms:  {summary['pct_within_10ms']:.2f}%")
            print(f"  % within 50ms:  {summary['pct_within_50ms']:.2f}%")
            print(f"  % within 100ms: {summary['pct_within_100ms']:.2f}%")
            print(f"  % affected:     {summary['pct_aff']:.2f}%")

    print("\n" + "=" * 70)
    print("=== FINAL SUMMARY (actual-32, single seed=1) ===")
    print("=" * 70)
    for which in results:
        print(f"\n{'POPP' if which == 'popps' else 'POP'}-FAILURE")
        print(f"  {'strat':<20s}{'mean Δ':>12s}{'%≤10ms':>10s}{'%≤50ms':>10s}{'%≤100ms':>11s}")
        for strat in STRATS:
            if strat not in results[which]:
                continue
            s = results[which][strat]
            print(f"  {strat:<20s}{s['mean_diff']:>12.3f}{s['pct_within_10ms']:>9.2f}%{s['pct_within_50ms']:>9.2f}%{s['pct_within_100ms']:>10.2f}%")

    # Save aggregated results
    out_path = os.path.expanduser(
        '~/Documents/sparse_advertisements_code/recovered_actual32/offline_failure_eval_summary.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nsaved summary to {out_path}")

    ray.shutdown()


if __name__ == '__main__':
    main()
