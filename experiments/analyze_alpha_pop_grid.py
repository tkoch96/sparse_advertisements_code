"""Aggregate + display the JSON outputs of alpha_pop_search.py.

Reads every *.json in --in, builds a table keyed by (alpha, anneal_end),
and prints:
  1. Per-strategy paper-plot pop-failure metrics for sparse, painter, OPP.
  2. Sparse-minus-painter gaps (the RNG-robust signal).
  3. Obj-trace monotonicity (bump_pct, final Obj).
  4. Ranked recommendations: which alpha config best closes the
     sparse-vs-painter gap on pct_within_10ms_site_failure (the headline
     paper metric).

Usage:
    python experiments/analyze_alpha_pop_grid.py --in /tmp/alpha_pop_grid
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _fmt(x, w=8, prec=3):
    if x is None:
        return ' ' * w
    if isinstance(x, float):
        if not np.isfinite(x):
            return f'{"nan":>{w}}'
        return f'{x:>+{w}.{prec}f}'
    return f'{x:>{w}}'


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--in', dest='in_dir', required=True,
                   help='Directory containing alpha_pop_search.py JSON outputs.')
    p.add_argument('--rank-by', default='pct_within_10_ms',
                   choices=['avg_lat_diff_ms_pop_fail', 'pct_within_10_ms',
                            'pct_within_50_ms', 'pct_within_100_ms',
                            'frac_vol_congested_pop_fail'],
                   help='Metric used for the ranked recommendation table.')
    args = p.parse_args()

    rows = []
    for jp in sorted(Path(args.in_dir).glob('*.json')):
        with open(jp) as f:
            d = json.load(f)
        rows.append(d)
    if not rows:
        print(f'No JSON results found in {args.in_dir}', file=sys.stderr)
        sys.exit(1)

    # ---- Per-strategy headline table ----
    print('\n=== Per-strategy site-failure metrics (paper plot quantities) ===')
    print('alpha   anneal   strategy          avg_Δlat   frac_cong   %within_10ms   %within_50ms   %within_100ms')
    print('-' * 105)
    for r in rows:
        c = r['config']
        for soln in r['per_strategy_pop_fail']:
            m = r['per_strategy_pop_fail'][soln]
            print(f"{c['alpha']:>5.2f}   {c['anneal_end']:>5d}   {soln:<16s}"
                  f"  {_fmt(m.get('avg_lat_diff_ms_pop_fail'), 8, 3)}   "
                  f"{_fmt(m.get('frac_vol_congested_pop_fail'), 8, 4)}   "
                  f"{_fmt(m.get('pct_within_10_ms'), 12, 2)}   "
                  f"{_fmt(m.get('pct_within_50_ms'), 12, 2)}   "
                  f"{_fmt(m.get('pct_within_100_ms'), 13, 2)}")

    # ---- Sparse-minus-painter gap (RNG-robust) ----
    print('\n=== sparse − painter gap (negative = sparse better) ===')
    print('alpha   anneal   Δavg_Δlat   Δ%within_10ms   Δ%within_50ms   Δ%within_100ms')
    print('-' * 84)
    for r in rows:
        c = r['config']
        g = r['sparse_minus_painter']
        print(f"{c['alpha']:>5.2f}   {c['anneal_end']:>5d}   "
              f"{_fmt(g.get('avg_lat_diff_ms_pop_fail'), 9, 3)}   "
              f"{_fmt(g.get('pct_within_10_ms'), 13, 2)}   "
              f"{_fmt(g.get('pct_within_50_ms'), 13, 2)}   "
              f"{_fmt(g.get('pct_within_100_ms'), 14, 2)}")

    # ---- Training monotonicity ----
    print('\n=== Sparse training Obj-trace monotonicity ===')
    print('alpha   anneal   n_iters   Obj_start   Obj_end   Obj_min   mono↓   bump↑   bump%    max_bump')
    print('-' * 98)
    for r in rows:
        c = r['config']
        o = r.get('obj_trace_summary') or {}
        if not o:
            print(f"{c['alpha']:>5.2f}   {c['anneal_end']:>5d}   (no log)")
            continue
        print(f"{c['alpha']:>5.2f}   {c['anneal_end']:>5d}   "
              f"{o['n_iters_seen']:>7d}   {o['obj_start']:>+9.3f}   {o['obj_end']:>+7.3f}   "
              f"{o['obj_min']:>+7.3f}   {o['mono_down_steps']:>5d}   {o['bump_up_steps']:>5d}   "
              f"{o['bump_pct']:>5.1f}%   {o['largest_bump']:>+8.3f}")

    # ---- Ranked recommendation ----
    rank_key = args.rank_by
    print(f'\n=== Ranked by sparse − painter on {rank_key} (best first) ===')

    # For latency-style metrics, lower (more negative) is better.
    # For percentage-within-threshold, HIGHER is better → so we want
    # sparse − painter as POSITIVE.
    is_pct = rank_key.startswith('pct_within_')
    sortable = []
    for r in rows:
        v = r['sparse_minus_painter'].get(rank_key)
        if v is None or not np.isfinite(v):
            continue
        # ranking direction: lower=better for latency/congestion, higher=better for pct
        rank_v = -v if is_pct else v
        sortable.append((rank_v, r))
    sortable.sort()
    for rank_v, r in sortable[:8]:
        c = r['config']
        v = r['sparse_minus_painter'].get(rank_key)
        marker = '🥇' if r is sortable[0][1] else '  '
        print(f"  {marker} alpha={c['alpha']:.2f} anneal={c['anneal_end']:>3d}  "
              f"sparse-painter[{rank_key}]={v:+.3f}  (lower-is-better={not is_pct})")


if __name__ == '__main__':
    main()
