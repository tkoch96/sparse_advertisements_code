#!/usr/bin/env python
"""Fast refresh loop for the Cluster runs tab.

    python -m cluster.dashloop                 # every 30s, forever
    python -m cluster.dashloop --interval 60
    python -m cluster.dashloop --once

Why this exists separately from `dashboard/refresh.py`
-----------------------------------------------------
That loop walks every registered experiment's plotting steps; one full
cycle takes well over its own 180 s interval, so the cluster tab could
only ever be as fresh as the slowest unrelated experiment. On 2026-08-21
that showed a finished run as still in flight for eleven minutes.

This loop does the three cheap things the live tab actually needs --
harvest, redraw, regenerate -- and nothing else. It is safe to run
alongside the slow loop: `dashboard.generate` writes `index.html` via
`os.replace`, so concurrent writers can never produce a torn page, and
last-writer-wins is the correct outcome for a page that is entirely
derived from files on disk.

Cost when idle is one EC2 describe per cycle, and not even that when no
run is registered as live.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cluster import vmlib as V  # noqa: E402

PY = sys.executable


def _run(argv, tag, timeout=300):
    try:
        p = subprocess.run(argv, cwd=V.REPO, capture_output=True, text=True,
                           timeout=timeout,
                           env=dict(os.environ, PYTHONPATH=V.REPO,
                                    MPLBACKEND='Agg'))
    except subprocess.TimeoutExpired:
        print('[dashloop] {} TIMED OUT after {}s'.format(tag, timeout),
              flush=True)
        return False
    if p.returncode != 0:
        # Loud, unlike the slow loop which sends step output to DEVNULL. A
        # refresh loop that fails silently is how a dashboard goes stale
        # while still looking alive.
        print('[dashloop] {} rc={}: {}'.format(
            tag, p.returncode, (p.stderr or '').strip()[-400:]), flush=True)
        return False
    return True


def _log_fingerprint():
    """(path, size) for every harvested log -- cheap staleness signal."""
    out = []
    if not os.path.isdir(V.RUNS_DIR):
        return tuple(out)
    for name in sorted(os.listdir(V.RUNS_DIR)):
        p = os.path.join(V.RUNS_DIR, name, 'logs', 'run.log')
        try:
            out.append((name, os.path.getsize(p)))
        except OSError:
            continue
    return tuple(out)


def cycle(state):
    live = V.live_runs()
    if live:
        _run([PY, '-m', 'cluster.harvest_all', '--quiet'], 'harvest')

    # Replot only when a log actually grew. Redrawing five figures per run
    # every 30 s against an ever-growing log is the one part of this that
    # would not stay cheap.
    fp = _log_fingerprint()
    if fp != state.get('fp'):
        _run([PY, '-m', 'dashboard.plot_cluster_timing', '--all'], 'plot')
        state['fp'] = fp
        state['dirty'] = True

    # Regenerate every cycle regardless: the status card carries elapsed
    # time and cost-so-far, which move even when the log does not.
    _run([PY, '-m', 'dashboard.generate'], 'generate')
    state['dirty'] = False
    return len(live)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--interval', type=int, default=30)
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args(argv)

    state = {}
    n = 0
    while True:
        n += 1
        t0 = time.time()
        try:
            live = cycle(state)
        except Exception as e:                    # noqa: BLE001
            print('[dashloop] cycle {} raised: {}'.format(n, e), flush=True)
            live = -1
        if n == 1 or n % 20 == 0:
            print('[dashloop] cycle {} ok ({} live run(s), {:.1f}s)'.format(
                n, live, time.time() - t0), flush=True)
        if a.once:
            return 0
        time.sleep(max(5, a.interval - (time.time() - t0)))


if __name__ == '__main__':
    raise SystemExit(main())
