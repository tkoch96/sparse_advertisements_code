"""Emit cache/dashboard/cost_calibration.json for the Cost-estimator tab.

Re-anchored 2026-08-23 on the LIVE prefixbudget3 ladder (64 workers,
c8g.16xl-class): parses the newest cluster_runs harvest for per-size
sec/iter and whole-size wall, so the estimate tracks the latest training
data automatically. Per Tom: actual-32 is assumed to run at actual-25's
per-iter speed. Rerun: python -m dashboard.cost_calibration
"""
import glob
import json
import os
import re
import statistics
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'cache', 'dashboard', 'cost_calibration.json')


def parse_latest_harvest():
    """Per-size sec/iter + whole-size wall from the newest run.log."""
    logs = sorted(glob.glob(os.path.join(
        REPO, 'cache', 'cluster_runs', '*', 'logs', 'run.log')),
        key=os.path.getmtime)
    if not logs:
        return None
    fn = logs[-1]
    text = open(fn, errors='replace').read()
    marks = [(float(t), int(d)) for t, d in re.findall(
        r"\[mem\] tag=dpsize_start .*? t=([\d.]+) dpsize=(\d+)", text)]
    pts = [(float(t), int(i)) for t, i in re.findall(
        r"\[mem\] tag=iter_start rss_mb=\d+ .*? t=([\d.]+) iter=(\d+)", text)]

    def size_at(t):
        s = None
        for mt, md in marks:
            if mt <= t:
                s = md
        return s

    per = {}
    for (a_t, a_i), (b_t, b_i) in zip(pts, pts[1:]):
        if b_i <= a_i:
            continue
        d = b_t - a_t
        if 0 < d < 3600:
            per.setdefault(size_at(a_t), []).append(d)
    sec_iter = {str(s): {'n': len(v),
                         'mean': round(sum(v) / len(v), 1),
                         'median': round(statistics.median(v), 1)}
                for s, v in per.items() if s}
    walls = {m[0]: float(m[1]) for m in re.findall(
        r"\[sweep\] dpsize=(\d+) done in ([\d.]+)s", text)}
    wid = [int(w) for w in re.findall(r"Worker (\d+) --", text)]
    return {
        'source': os.path.relpath(fn, REPO),
        'harvested_utc': time.strftime('%Y-%m-%d %H:%M UTC',
                                       time.gmtime(os.path.getmtime(fn))),
        'workers': (max(wid) + 1) if wid else 64,
        'sec_per_iter': sec_iter,
        'wall_total_sec': walls,
    }


def main():
    meas = parse_latest_harvest()
    # anchor: latest measured size-25 mean; Tom 2026-08-23: "extrapolate
    # the current size 25 run and assume 32 is basically the same speed"
    t25 = None
    if meas and '25' in meas.get('sec_per_iter', {}):
        t25 = meas['sec_per_iter']['25']['mean']
    anchor = ({'size': 32, 'workers': meas['workers'], 't_iter_sec': t25,
               'source': 'actual-25 live ({}, mean of {} iters); '
                         'actual-32 assumed equal per Tom'.format(
                             meas['source'], meas['sec_per_iter']['25']['n'])}
              if t25 else
              {'size': 32, 'workers': 64, 't_iter_sec': 470,
               'source': 'pf32_run.log (fallback -- no harvest found)'})
    calib = {
        'generated_utc': time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()),
        'status': ('MEASURED anchor from the live prefixbudget3 ladder'
                   if t25 else
                   'GUESSTIMATE -- no harvest parsed, pf32 fallback anchor'),
        'anchor': anchor,
        'measured': meas,
        # alpha sanity: prefixbudget3 10->25 gives ln(177.3/20.3)/ln(2.5)=2.37
        'defaults': {'alpha': 2.4, 'iters_per_eval': 200,
                     'overhead_min_per_eval': 15,
                     'beta': 2.0, 'headroom_frac': 0.20},
        'rss': {'base_mb': 520, 'state32_gb': 83.1,
                'source': 'eods32 production worker p50 1849 MB @ 64w'},
        'families': [
            {'name': 'c8g', 'ram_gb_per_core': 2.0, 'usd_core_hr': 0.0399},
            {'name': 'm8g', 'ram_gb_per_core': 4.0, 'usd_core_hr': 0.0450},
            {'name': 'r8g', 'ram_gb_per_core': 8.0, 'usd_core_hr': 0.0530},
        ],
        'sizes': [3, 5, 10, 15, 20, 25, 32],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(calib, f, indent=1)
    print('wrote', OUT, '(anchor: {} s/iter, {})'.format(
        anchor['t_iter_sec'], 'measured' if t25 else 'fallback'))


if __name__ == '__main__':
    main()
