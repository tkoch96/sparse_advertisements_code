"""Emit cache/dashboard/cost_calibration.json for the Cost-estimator tab.

Numbers are GUESSTIMATES anchored on measured points; the 24xl smoke
(2026-08-21) will tighten per-iter timing + steady-state worker RSS.
Rerun: python -m dashboard.cost_calibration
"""
import json
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'cache', 'dashboard', 'cost_calibration.json')

CALIB = {
    'generated_utc': time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()),
    'status': 'GUESSTIMATE — anchor measured (pf32); scaling exponents '
              'are guesses pending the 24xl smoke + a size sweep on EC2',
    # Anchor: preflight c8g.24xl, actual-32, 64 workers: ~470 s per
    # lb-iteration => 470*64 core-seconds per iteration at N=32.
    'anchor': {'size': 32, 'workers': 64, 't_iter_sec': 470,
               'source': 'pf32_run.log (measured, c8g.24xl, 2026-08-19)'},
    # Mac sweep (3-iter combined-flush shakeout, ~3 workers, M-series —
    # different silicon, startup included; used only to sanity the
    # size-scaling exponent, alpha~2.3):
    'mac_sweep_wall_sec': {'3': 100.8, '5': 1359.8, '10': 2843.2,
                           '15': 5488.2},
    'defaults': {'alpha': 2.3, 'iters_per_eval': 30,
                 'startup_min_per_eval': 10, 'beta': 2.0,
                 'headroom_frac': 0.20},
    # Worker RSS model: mb(N, W) = base + state32_gb*1024*(N/32)^beta/W.
    # base+shard measured on eods32 production (p50 1849 MB @ N=32 W=64).
    'rss': {'base_mb': 520, 'state32_gb': 83.1,
            'source': 'eods32 production worker p50 1849 MB @ 64w; '
                      '16xl smoke iter-1 p50 2.25 GB @ 48w consistent'},
    # On-demand us-east-1 Graviton4, $ per vCPU-hour (approx 2026-08).
    'families': [
        {'name': 'c8g', 'ram_gb_per_core': 2.0, 'usd_core_hr': 0.0399},
        {'name': 'm8g', 'ram_gb_per_core': 4.0, 'usd_core_hr': 0.0450},
        {'name': 'r8g', 'ram_gb_per_core': 8.0, 'usd_core_hr': 0.0530},
    ],
    'sizes': [3, 5, 10, 15, 20, 25, 32],
}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(CALIB, f, indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
