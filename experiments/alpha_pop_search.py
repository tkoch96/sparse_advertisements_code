"""Single-config driver for tuning the pop-failure resilience gradient
(`gradients_resilience_benefit_pop`) via SCULPTOR_ALPHA_POP.

Trains sparse + painter + one_per_peering on one deployment (seed-pinned for
A/B comparability) and reports the *exact* paper-plot pop-failure metrics
used by `evaluate_over_deployment_sizes.py::make_paper_plots`:

  * stats_pop_failures_latency_optimal_specific.avg_latency_difference
       -- "Avg Suboptimality During Site Failure (ms)" paper-plot Y
  * stats_pop_failures_latency_optimal_specific.frac_vol_congested
       -- "Pct Volume Congested During Site Failure"
  * stats_latency_thresholds_fail_pop[ri][-10|-50|-100]
       -- "Pct Traffic Within X ms of Optimal (Site Failure)" paper-plot Y
          (each sim_idx contributes one fraction-of-CDF lookup)

It is intentionally narrow: only the pop-failure phase of the eval is
relevant; the heavier flash-crowd / volume-multiplier phases still run
because they share infrastructure with the failure eval, but we ignore
their outputs.

Usage
-----
    python -m experiments.alpha_pop_search \
        --alpha 0.1 --anneal-end 0 \
        --dpsize small --port 31970 --nsim 1 --max-iter 100 \
        --seed 1 --tag a010 --out /tmp/alphagrid/a010.json

Outputs a JSON with the per-strategy metrics + Obj-trace monotonicity from
the sparse training log (parsed post-hoc from stdout/stderr is the caller's
job; this driver writes the metric JSON only).
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Make sibling modules importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


LAT_THRESHOLDS = [-10, -50, -100]   # paper plot thresholds (ms below optimal)


def _aggregate_paper_metrics(metrics, soln_types):
    """Mirror of `get_failure_metric_arr` in eval_latency_failure.py, scoped
    to the pop-failure key. Returns the same paper-plot numbers
    make_paper_plots() consumes, but per-strategy.

    For each strategy we produce:
      avg_lat_diff      -- avg(perf1-perf2) weighted by volume
      frac_vol_congested-- frac of volume routed to NO_ROUTE under failure
      pct_within_<X>_ms -- mean over sims of (1 - frac_uncongested * cdf(X))
                           expressed as a percentage. Mirrors the
                           `100 - 100*np.mean(...)` formula make_paper_plots
                           applies to stats_latency_thresholds_fail_pop.
    """
    from helpers import get_cdf_xy
    from constants import NO_ROUTE_LATENCY

    key = 'pop_failures_latency_optimal_specific'
    sims = metrics.get(key, {})
    nsim = len(sims)

    out = {}
    for soln in soln_types:
        # Per-sim threshold stats (volume-weighted CDF lookup at each threshold)
        per_sim_threshold = {lt: [] for lt in LAT_THRESHOLDS}
        all_diffs = []
        all_vols = []
        total_vol_congested = 0.0
        total_vol_seen = 0.0
        total_actual_vol = 0.0

        for ri in range(nsim):
            inner = sims.get(ri, {})
            entries = inner.get(soln) or []
            if not entries:
                # missing sim for this soln (e.g. strategy failed)
                continue
            this_diffs, this_vols = [], []
            this_vol_total = 0.0
            this_vol_congested = 0.0
            for fields in entries:
                # fields = (diff, vol, ug, element, perf1, perf2) or
                #          (diff, vol, ug, element, perf1, perf2, _)
                diff, vol = fields[0], fields[1]
                perf1, perf2 = fields[4], fields[5]
                total_actual_vol += vol
                if perf1 == NO_ROUTE_LATENCY:
                    # best-case scenario already congested -- skip
                    continue
                this_vol_total += vol
                total_vol_seen += vol
                if perf2 != NO_ROUTE_LATENCY:
                    all_diffs.append(perf1 - perf2)
                    all_vols.append(vol)
                    this_diffs.append(perf1 - perf2)
                    this_vols.append(vol)
                else:
                    this_vol_congested += vol
                    total_vol_congested += vol

            frac_congested = this_vol_congested / (this_vol_total + 1e-9)
            try:
                xs, cdf_x = get_cdf_xy(list(zip(this_diffs, this_vols)), weighted=True)
                for lt in LAT_THRESHOLDS:
                    xi = int(np.argmin(np.abs(xs - lt)))
                    per_sim_threshold[lt].append(
                        (1 - frac_congested) * cdf_x[xi] + frac_congested)
            except (IndexError, ValueError):
                for lt in LAT_THRESHOLDS:
                    per_sim_threshold[lt].append(frac_congested)

        if all_diffs:
            avg_lat_diff = float(np.average(all_diffs, weights=all_vols))
        else:
            avg_lat_diff = float('nan')
        frac_vol_congested = float(total_vol_congested / (total_vol_seen + 1e-9))

        # Paper-plot formula: 100 * (1 - mean(per_sim_threshold))
        threshold_pct = {}
        for lt in LAT_THRESHOLDS:
            vals = per_sim_threshold[lt]
            if vals:
                threshold_pct['pct_within_{}_ms'.format(abs(lt))] = float(100 * (1 - np.mean(vals)))
            else:
                threshold_pct['pct_within_{}_ms'.format(abs(lt))] = float('nan')

        out[soln] = {
            'avg_lat_diff_ms_pop_fail': avg_lat_diff,
            'frac_vol_congested_pop_fail': frac_vol_congested,
            'nsim_with_data': sum(1 for vs in per_sim_threshold[LAT_THRESHOLDS[0]] if vs is not None),
            **threshold_pct,
        }
    return out


def _parse_obj_trace_from_log(log_path):
    """Optional: if a sparse-training stdout log was captured, extract the
    per-iter Obj line and report monotonicity for sim 0.

    Looks for `Actual: NP:..., LB:..., RB:..., Total:..., Obj: X` *immediately
    after* each `LEARNING ITERATION : N` line. Only counts the first such line
    per iter (the "modeled" objective, not the post-step one).
    """
    import re
    if not log_path or not Path(log_path).exists():
        return None
    iter_re = re.compile(r'LEARNING ITERATION\s*:\s*(\d+)')
    act_re = re.compile(r'Actual:\s*NP:[^,]+,\s*LB:[^,]+,\s*RB:[^,]+,\s*Total:[^,]+,\s*Obj:\s*([\-0-9.e+]+)')
    objs = []
    saw_first = False
    with open(log_path) as f:
        for line in f:
            if iter_re.search(line):
                saw_first = False
                continue
            m = act_re.search(line)
            if m and not saw_first:
                try:
                    objs.append(float(m.group(1)))
                except ValueError:
                    pass
                saw_first = True
    if len(objs) < 2:
        return None
    objs = np.asarray(objs)
    diffs = np.diff(objs)
    n_up = int((diffs > 0).sum())
    n_dn = int((diffs < 0).sum())
    return {
        'n_iters_seen': len(objs),
        'obj_start': float(objs[0]),
        'obj_end': float(objs[-1]),
        'obj_min': float(objs.min()),
        'mono_down_steps': n_dn,
        'bump_up_steps': n_up,
        'bump_pct': float(n_up / (n_up + n_dn + 1e-9) * 100),
        'largest_bump': float(diffs[diffs > 0].max() if n_up else 0.0),
    }


def run_one_config(args):
    """Run a single (alpha, anneal_end) config. Returns the result dict."""
    # Env knobs flow into the rest of the codebase via os.environ
    os.environ['SCULPTOR_USE_RESILIENCE'] = '1'
    os.environ['SCULPTOR_ALPHA_POP'] = str(args.alpha)
    os.environ['SCULPTOR_ALPHA_POP_ANNEAL_END_ITER'] = str(args.anneal_end)
    os.environ['SCULPTOR_MAX_ITER'] = str(args.max_iter)
    if args.seed is not None:
        os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(args.seed)
    # Skip the parallel-strategy fanout to keep per-config memory bounded
    os.environ['SCULPTOR_DISABLE_PARALLEL_STRATEGIES'] = '1'
    # Non-interactive matplotlib (eval still produces some figures)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    # Tag the run dir + cache pickle uniquely
    os.environ['SCULPTOR_RUN_TAG'] = args.tag

    # Late import so env vars are honoured by initializers
    from eval_latency_failure import evaluate_all_metrics

    soln_types = args.solns.split(',')

    t0 = time.time()
    evaluate_all_metrics(
        args.dpsize,
        args.port,
        nsim=args.nsim,
        soln_types=soln_types,
    )
    train_eval_seconds = time.time() - t0

    # The pickle filename is built deterministically from dpsize + tag
    pkl = Path('cache') / f'popp_failure_latency_comparison_{args.dpsize}_{args.tag}.pkl'
    if not pkl.is_absolute():
        pkl = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / pkl
    if not pkl.exists():
        raise RuntimeError(f'Expected pickle not found: {pkl}')

    with open(pkl, 'rb') as f:
        metrics = pickle.load(f)

    per_strategy = _aggregate_paper_metrics(metrics, soln_types)

    # Sparse-minus-painter is the RNG-robust signal (both share deployment seed)
    gap = {}
    if 'sparse' in per_strategy and 'painter' in per_strategy:
        for k in ('avg_lat_diff_ms_pop_fail', 'pct_within_10_ms',
                  'pct_within_50_ms', 'pct_within_100_ms',
                  'frac_vol_congested_pop_fail'):
            try:
                gap[k] = per_strategy['sparse'][k] - per_strategy['painter'][k]
            except (KeyError, TypeError):
                pass

    return {
        'config': {
            'alpha': args.alpha,
            'anneal_end': args.anneal_end,
            'dpsize': args.dpsize,
            'seed': args.seed,
            'nsim': args.nsim,
            'max_iter': args.max_iter,
            'tag': args.tag,
            'solns': soln_types,
        },
        'wallclock_seconds': train_eval_seconds,
        'per_strategy_pop_fail': per_strategy,
        'sparse_minus_painter': gap,
        'obj_trace_summary': _parse_obj_trace_from_log(args.train_log),
        'pickle_path': str(pkl),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--alpha', type=float, required=True,
                   help='SCULPTOR_ALPHA_POP weight on the pop-failure resilience gradient.')
    p.add_argument('--anneal-end', type=int, default=0,
                   help='SCULPTOR_ALPHA_POP_ANNEAL_END_ITER (0 = no anneal).')
    p.add_argument('--dpsize', default='small')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--nsim', type=int, default=1)
    p.add_argument('--max-iter', type=int, default=100)
    p.add_argument('--seed', type=int, default=None,
                   help='SCULPTOR_DEPLOYMENT_SEED (recommended for A/B).')
    p.add_argument('--tag', required=True,
                   help='Unique tag for this config. Used in cache filename + run dir.')
    p.add_argument('--solns', default='sparse,painter,one_per_peering',
                   help='Comma-separated solution types to train + eval.')
    p.add_argument('--train-log', default=None,
                   help='Optional path to the captured stdout log for Obj-trace monotonicity parsing.')
    p.add_argument('--out', required=True, help='Output JSON path.')
    args = p.parse_args()

    Path(os.path.dirname(args.out) or '.').mkdir(parents=True, exist_ok=True)
    result = run_one_config(args)

    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f'\n[alpha_pop_search] wrote {args.out}')
    print(f'  sparse vs painter (site-fail Δlat, ms): '
          f'{result["sparse_minus_painter"].get("avg_lat_diff_ms_pop_fail", float("nan")):+.3f}')


if __name__ == '__main__':
    main()
