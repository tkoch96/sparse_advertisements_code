"""Per-phase timing + crash diagnostics for eval_all_solution_types.

Drives evaluate_all_metrics on a chosen dpsize and produces a per-phase
report:
  - wall time
  - populated soln_types after the phase
  - whether any traceback was logged
  - which LP code path the phase exercises

Designed to be re-run before-and-after fixes so we can compare. Output
written to benchmarks/eval_phase_<RUN_TAG>.json next to a full log.

Usage:
  cd ~/Documents/sparse_advertisements_code  # or worktree
  ~/Documents/venv312/bin/python benchmarks/eval_phase_baseline.py \
    --dpsize small --max-iter 10 --tag baseline
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import time
import traceback
import contextlib
from collections import OrderedDict

# ---- Project setup ---------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
# repo root is now TWO levels up (moved under experiments/ 2026-08-21);
# this script chdir's to _REPO, so getting the depth wrong breaks every
# relative cache/ path and the top-level imports below.
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

# Run_ray.py aliases worker_comms → worker_comms; mirror that here so the
# rest of the codebase finds the right module.
import core.worker_comms as _ray_mod  # noqa: E402
sys.modules['worker_comms'] = _ray_mod

import ray  # noqa: E402

# ---- Phase identifiers (matched against the "----X for deployment number" lines) ----
PHASE_KEYS = [
    'strategy_compare',          # compare_different_solutions over 6 strategies
    'pct_volume_within_latency', # pure-Python, no LP
    'failure_resilience',        # popp+pop failures via solve_lp_with_failure_catch_mp
    'diurnal',                   # 24h × N intensities
    'flash_crowd',               # per-metro flash crowd
]
# Map "log header" → phase key, so we can detect the phase by parsing
# the print() lines evaluate_all_metrics emits at each block's top.
PHASE_HEADER_PREFIXES = {
    '-----Deployment number': 'strategy_compare',
    '-----Volume calc for deployment number': 'pct_volume_within_latency',
    '-----Failure calc for deployment number': 'failure_resilience',
    '-----Diurnal calc for deployment number': 'diurnal',
    '-----Flash crowd calc for deployment number': 'flash_crowd',
}


class _TeeStream(io.TextIOBase):
    """Splits writes to (original_stream, log_file) AND tracks phase boundaries."""

    def __init__(self, real, logfile_handle, on_phase_start):
        self._real = real
        self._log = logfile_handle
        self._on_phase_start = on_phase_start
        self._buf = ''

    def write(self, s):
        # forward verbatim
        self._real.write(s)
        try:
            self._log.write(s)
            self._log.flush()
        except Exception:
            pass
        # accumulate then split on newline so we can match headers exactly
        self._buf += s
        if '\n' in self._buf:
            lines = self._buf.split('\n')
            self._buf = lines[-1]
            for line in lines[:-1]:
                self._maybe_mark_phase(line)
        return len(s)

    def _maybe_mark_phase(self, line):
        for prefix, key in PHASE_HEADER_PREFIXES.items():
            if prefix in line:
                self._on_phase_start(key, line)
                return

    def flush(self):
        self._real.flush()
        try:
            self._log.flush()
        except Exception:
            pass


def _populated_strats(metrics: dict, field: str, soln_types: list) -> list:
    """For a given metrics[field][0][solution], which solutions are non-empty?"""
    out = []
    sub = metrics.get(field, {}).get(0, {})
    for s in soln_types:
        v = sub.get(s)
        if v is None:
            continue
        if isinstance(v, (list, tuple)) and len(v) == 0:
            continue
        if isinstance(v, dict) and len(v) == 0:
            continue
        out.append(s)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dpsize', default='small')
    p.add_argument('--max-iter', type=int, default=10,
                   help='SCULPTOR_MAX_ITER override (keep tiny for baseline)')
    p.add_argument('--n-workers', type=int, default=4)
    p.add_argument('--tag', required=True,
                   help='SCULPTOR_RUN_TAG; output written to benchmarks/eval_phase_<tag>.json + .log')
    p.add_argument('--port', type=int, default=31415)
    p.add_argument('--seed', type=int, default=1,
                   help='SCULPTOR_DEPLOYMENT_SEED for reproducible deployments')
    p.add_argument('--no-clear', action='store_true',
                   help='Skip clearing the run-tagged cache file (useful for resume diagnostics)')
    args = p.parse_args()

    os.environ['SCULPTOR_MAX_ITER'] = str(args.max_iter)
    os.environ['SCULPTOR_N_WORKERS'] = str(args.n_workers)
    os.environ['SCULPTOR_RUN_TAG'] = args.tag
    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(args.seed)
    # Match the headroom default we ship at scale
    os.environ.setdefault('SCULPTOR_CAPACITY_HEADROOM', '0.2')

    # Now safe to import codepaths that read env at import time
    from evaluations.wrapper_eval import global_performance_metrics_fn, global_soln_types
    from evaluations.eval_all_solution_types import evaluate_all_metrics

    pkl_path = global_performance_metrics_fn(args.dpsize)
    if not args.no_clear and os.path.exists(pkl_path):
        print(f"[setup] removing stale {pkl_path}")
        os.remove(pkl_path)

    log_path = os.path.join('benchmarks', f'eval_phase_{args.tag}.log')
    json_path = os.path.join('benchmarks', f'eval_phase_{args.tag}.json')
    os.makedirs('benchmarks', exist_ok=True)

    # ---- Phase tracking ----
    timings = OrderedDict()  # phase_key -> total wall seconds
    counts = OrderedDict()   # phase_key -> how many times we entered the phase header
    state = {'current': None, 'start': None, 'overall_start': time.time()}

    def on_phase_start(key, line):
        # finalize previous phase
        if state['current'] is not None and state['start'] is not None:
            dt = time.time() - state['start']
            timings[state['current']] = timings.get(state['current'], 0.0) + dt
        state['current'] = key
        state['start'] = time.time()
        counts[key] = counts.get(key, 0) + 1

    logf = open(log_path, 'w')
    real_stdout, real_stderr = sys.stdout, sys.stderr
    tee_out = _TeeStream(real_stdout, logf, on_phase_start)
    tee_err = _TeeStream(real_stderr, logf, on_phase_start)
    sys.stdout = tee_out
    sys.stderr = tee_err

    crash_info = None
    try:
        evaluate_all_metrics(args.dpsize, args.port, soln_types=list(global_soln_types))
    except SystemExit as e:
        crash_info = f"SystemExit: code={e.code}"
        traceback.print_exc()
    except BaseException:
        crash_info = traceback.format_exc()
        traceback.print_exc()
    finally:
        # close last phase
        if state['current'] is not None and state['start'] is not None:
            dt = time.time() - state['start']
            timings[state['current']] = timings.get(state['current'], 0.0) + dt
        sys.stdout = real_stdout
        sys.stderr = real_stderr
        try:
            logf.close()
        except Exception:
            pass

    overall = time.time() - state['overall_start']

    # ---- Analyse populated fields in the metrics pickle ----
    metrics_summary = {}
    if os.path.exists(pkl_path):
        try:
            metrics = pickle.load(open(pkl_path, 'rb'))
        except Exception as e:
            metrics_summary['error'] = f"failed to load pickle: {e}"
            metrics = {}
    else:
        metrics = {}
        metrics_summary['error'] = f"pickle never written: {pkl_path}"

    # Which strategies actually have a converged advertisement in compare_rets?
    cr = metrics.get('compare_rets', {}).get(0)
    if isinstance(cr, dict) and 'adv_solns' in cr:
        adv_solns = cr['adv_solns']
        metrics_summary['strategies_with_adv'] = [
            s for s in global_soln_types
            if isinstance(adv_solns.get(s), list) and len(adv_solns[s]) > 0
        ]
    else:
        metrics_summary['strategies_with_adv'] = []

    # Failure / flash / diurnal field populations
    failure_fields = [
        'popp_failures_latency_optimal_specific',
        'pop_failures_latency_optimal_specific',
    ]
    flash_fields = ['resilience_to_congestion']
    diurnal_fields = ['diurnal']
    pct_lat_fields = ['pct_volume_within_latency']

    def _summarise_fields(fields):
        return {f: _populated_strats(metrics, f, list(global_soln_types)) for f in fields}

    metrics_summary['failure'] = _summarise_fields(failure_fields)
    metrics_summary['flash_crowd'] = _summarise_fields(flash_fields)
    metrics_summary['diurnal'] = _summarise_fields(diurnal_fields)
    metrics_summary['pct_volume_within_latency'] = _summarise_fields(pct_lat_fields)

    # ---- Pull tracebacks out of the log so we can attribute crashes per-phase ----
    tracebacks = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as fh:
            log_lines = fh.read().splitlines()
        i = 0
        while i < len(log_lines):
            if 'Traceback (most recent call last):' in log_lines[i]:
                # collect until blank line or another "Traceback" marker
                tb = [log_lines[i]]
                j = i + 1
                while j < len(log_lines):
                    if log_lines[j].strip() == '' and j + 1 < len(log_lines) and not (log_lines[j + 1].startswith(' ') or log_lines[j + 1].startswith('\t')):
                        break
                    tb.append(log_lines[j])
                    j += 1
                tracebacks.append('\n'.join(tb))
                i = j
            else:
                i += 1
    metrics_summary['n_tracebacks_in_log'] = len(tracebacks)
    metrics_summary['first_tracebacks'] = tracebacks[:6]

    # ---- Write summary ----
    summary = {
        'tag': args.tag,
        'dpsize': args.dpsize,
        'max_iter': args.max_iter,
        'n_workers': args.n_workers,
        'overall_wall_s': round(overall, 2),
        'phase_wall_s': {k: round(v, 2) for k, v in timings.items()},
        'phase_enter_count': dict(counts),
        'crash_info': crash_info,
        'metrics': metrics_summary,
        'pkl_path': pkl_path,
        'log_path': log_path,
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- Print the report ----
    print()
    print("=" * 72)
    print(f"=== eval_phase_baseline: tag={args.tag} dpsize={args.dpsize} ===")
    print("=" * 72)
    print(f"overall wall: {overall:.1f}s")
    print()
    print(f"{'phase':<28s} {'wall(s)':>10s}  {'entries':>8s}")
    for k in PHASE_KEYS:
        w = timings.get(k, 0.0)
        c = counts.get(k, 0)
        print(f"  {k:<26s} {w:>10.1f}  {c:>8d}")
    if crash_info:
        print()
        print(f"crash: {crash_info[:300]}")
    print()
    print(f"strategies with adv: {metrics_summary['strategies_with_adv']}")
    print(f"failure populated: {metrics_summary['failure']}")
    print(f"flash_crowd populated: {metrics_summary['flash_crowd']}")
    print(f"diurnal populated: {metrics_summary['diurnal']}")
    print(f"pct_volume_within_latency populated: {metrics_summary['pct_volume_within_latency']}")
    print(f"tracebacks observed: {metrics_summary['n_tracebacks_in_log']}")
    print()
    print(f"log:  {log_path}")
    print(f"json: {json_path}")
    print(f"pkl:  {pkl_path}")


if __name__ == '__main__':
    main()
