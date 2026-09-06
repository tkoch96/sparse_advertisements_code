"""deployment_scaling_summary -- the gist of the sweeps, one row per
metric (Tom 2026-08-30).

    python evaluations/deployment_scaling_summary.py \
        --cache-fn <metrics_by_dpsize.pkl> [--prefix-cache-fn <pkl>] \
        --out-dir figures/scaling_summary

For every metric: SCULPTOR vs the best baseline per size, the GAP at the
smallest and largest size, and a verdict on how the gap moves with scale
-- grow / hold / shrink -- plus the % change and a Spearman rho of
gap-vs-size (directional signal only: n=6 sizes, so no p-value theater).
Emits deployment_scaling_summary.csv (and prefix_scaling_summary.csv
when the prefix pickle is given/present).

The gap change %% is (gap_max - gap_min) / max(|gap_min|, |gap_max|)
-- bounded and safe for zero/sign-flipping gaps.  sculptor_leads counts
sizes where SCULPTOR beats the best baseline, so a gap that "shrinks"
into a deficit is visible at a glance.  Size 3 is dropped on the
deployment_size axis (marker-contaminated; paper figures start at 5).
"""
import argparse
import csv
import os
import pickle
import re
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

BASELINES = ['painter', 'one_per_pop', 'anyopt', 'anycast']
ORACLE = 'one_per_peering'
SPARSE = 'sparse'

# (objective, metric, direction, extractor)
# direction: '<' lower-better, '>' higher-better, '0' closest-to-zero
# ('0' is for failure suboptimality: ingress priorities are not the
# latency rank, so failures can DECREASE latency and a huge negative
# subopt is drift from optimal, not a win -- rank by |value|)
# extractor(metrics_for_size, solution) -> float or None


def _stats_scalar(key, field=None, scale=1.0):
    def _x(m, sol):
        row = (m.get(key) or {}).get(sol)
        if row is None:
            return None
        if field is not None:
            if not isinstance(row, dict):
                return None
            row = row.get(field)
        if isinstance(row, dict):          # per-sim dict -> mean
            vals = []
            for v in row.values():
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass               # empty/partial sim (prefix shim)
            row = float(np.mean(vals)) if vals else None
        if row is None:
            return None
        try:
            v = float(row)
        except (TypeError, ValueError):
            return None
        return None if not np.isfinite(v) else scale * v
    return _x


METRICS = [
    ('Latency', 'Subopt normal (ms)', '<',
     _stats_scalar('stats_best_latencies', scale=-1.0)),
    ('Failure robustness', 'Subopt ingress-fail (ms)', '0',
     _stats_scalar('stats_popp_failures_latency_optimal_specific',
                   'avg_latency_difference', scale=-1.0)),
    ('Failure robustness', '% cong ingress-fail', '<',
     _stats_scalar('stats_popp_failures_latency_optimal_specific',
                   'frac_vol_congested', scale=100.0)),
    ('Failure robustness', 'Subopt site-fail (ms)', '0',
     _stats_scalar('stats_pop_failures_latency_optimal_specific',
                   'avg_latency_difference', scale=-1.0)),
    ('Failure robustness', '% cong site-fail', '<',
     _stats_scalar('stats_pop_failures_latency_optimal_specific',
                   'frac_vol_congested', scale=100.0)),
    ('Congestion resilience', 'Flash crowd intensity', '>',
     _stats_scalar('stats_resilience_to_congestion')),
    ('Congestion resilience', 'Diurnal intensity', '>',
     _stats_scalar('stats_diurnal')),
]


def _spearman(xs, ys):
    if len(xs) < 3:
        return None
    rx = np.argsort(np.argsort(xs)).astype(float)
    ry = np.argsort(np.argsort(ys)).astype(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(cache_fn, axis_name):
    m_by = pickle.load(open(cache_fn, 'rb'))
    sizes = sorted(m_by, key=lambda k: int(re.search(r'\d+', str(k)).group()))
    if axis_name == 'deployment_size':
        # size 3 is marker-contaminated; the paper's figures start at 5
        sizes = [s for s in sizes
                 if int(re.search(r'\d+', str(s)).group()) != 3]
    rows = []
    for objective, metric, direction, fx in METRICS:
        per_size = []
        for sz in sizes:
            sv = fx(m_by[sz], SPARSE)
            bl = {b: fx(m_by[sz], b) for b in BASELINES}
            bl = {b: v for b, v in bl.items() if v is not None}
            if sv is None or not bl:
                continue
            if direction == '0':
                bname, bval = min(bl.items(), key=lambda kv: abs(kv[1]))
                gap = abs(bval) - abs(sv)
            elif direction == '<':
                bname, bval = min(bl.items(), key=lambda kv: kv[1])
                gap = bval - sv
            else:
                bname, bval = max(bl.items(), key=lambda kv: kv[1])
                gap = sv - bval
            per_size.append((sz, sv, bval, bname, gap))
        if len(per_size) < 3:
            continue
        gaps = [g for _s, _v, _b, _n, g in per_size]
        g_min, g_max = gaps[0], gaps[-1]
        denom = max(abs(g_min), abs(g_max))
        if denom < 1e-6:               # materiality floor: 1e-9-scale
            denom = 0.0                # gaps are noise, not a trend
        pct = (100.0 * (g_max - g_min) / denom) if denom else 0.0
        rho = _spearman(list(range(len(gaps))), gaps)
        if denom == 0:
            verdict = 'hold'
        elif pct >= 25 and (rho or 0) > 0:
            verdict = 'grow'
        elif pct <= -25 and (rho or 0) < 0:
            verdict = 'shrink'
        else:
            verdict = 'hold'
        n_lead = sum(1 for g in gaps if g > 0)
        last = per_size[-1]
        rows.append({
            'axis': axis_name,
            'objective': objective,
            'metric': metric,
            'direction': direction,
            'sculptor_at_max': round(last[1], 3),
            'best_baseline_at_max': round(last[2], 3),
            'best_baseline_name': last[3],
            'gap_at_min': round(g_min, 3),
            'gap_at_max': round(g_max, 3),
            'gap_change_pct': round(pct, 1),
            'spearman_rho': round(rho, 2) if rho is not None else '',
            'sculptor_leads': '{}/{}'.format(n_lead, len(gaps)),
            'verdict': verdict,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-fn', required=True,
                    help='metrics_by_dpsize.pkl for the size axis')
    ap.add_argument('--prefix-cache-fn', default=None,
                    help='metrics-by-prefix pickle (optional)')
    ap.add_argument('--out-dir', default='figures/scaling_summary')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    def _write(rows, name):
        if not rows:
            print('no rows for {} -- skipped'.format(name))
            return
        pth = os.path.join(a.out_dir, name)
        with open(pth, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print('wrote {} ({} rows)'.format(pth, len(rows)))
        for r in rows:
            print('  {:<24s} {:<26s} {:>6s}  gap {:>9} -> {:>9}  ({:>6}%%, leads {})'
                  .format(r['objective'], r['metric'], r['verdict'],
                          r['gap_at_min'], r['gap_at_max'],
                          r['gap_change_pct'], r['sculptor_leads']).replace('%%','%'))

    _write(summarize(a.cache_fn, 'deployment_size'),
           'deployment_scaling_summary.csv')
    if a.prefix_cache_fn and os.path.exists(a.prefix_cache_fn):
        _write(summarize(a.prefix_cache_fn, 'prefix_budget'),
               'prefix_scaling_summary.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
