#!/usr/bin/env python
"""Pull every live run's logs and results. The unattended safety net.

    python -m cluster.harvest_all            # pull whatever is live
    python -m cluster.harvest_all --quiet    # for cron / the refresh loop

This is the belt to `expctl`'s braces. `expctl status/watch/kill` pull on
demand and `vmctl stop` refuses to stop over unharvested bytes, but all of
those need someone to be running them. This one is wired into the
dashboard refresh loop (every 180 s) and is safe to run from cron: with no
VM running it makes one EC2 describe call and exits 0.

The record it leaves is `cache/cluster_runs/<run_id>/harvest.json`, whose
`last_harvest_utc` the dashboard card shows -- so a harvest that has
quietly stopped working is visible as a stale timestamp rather than as a
missing log discovered after the box is gone.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cluster import vmlib as V      # noqa: E402
from cluster import expctl          # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args(argv)

    runs = V.live_runs()
    if not runs:
        if not a.quiet:
            print('no live runs')
        return 0

    # One describe for all of them; a per-run lookup would make this
    # O(runs) API calls on every refresh cycle.
    try:
        states = {d['id']: d for d in V.describe()}
    except Exception as e:                        # noqa: BLE001
        print('harvest_all: cannot reach EC2: {}'.format(e), file=sys.stderr)
        return 1

    n_ok = n_skip = n_err = 0
    for m in runs:
        inst = states.get(m.get('instance_id'))
        if not inst or inst['state'] != 'running':
            n_skip += 1
            continue
        try:
            got = expctl.harvest(m, inst['ip'], verbose=False)
        except Exception as e:                    # noqa: BLE001
            n_err += 1
            print('harvest_all: {} FAILED: {}'.format(m['run_id'], e),
                  file=sys.stderr)
            continue
        n_ok += 1
        if not a.quiet:
            lp = V.local_log(m['run_id'])
            size = os.path.getsize(lp) if os.path.exists(lp) else 0
            print('{}: log {}{}'.format(
                m['run_id'], V.human_bytes(size),
                '  WARN ' + '; '.join(got['errors']) if got['errors'] else ''))

    if not a.quiet:
        print('harvested {} run(s); {} on stopped VMs; {} failed'.format(
            n_ok, n_skip, n_err))
    return 1 if n_err else 0


if __name__ == '__main__':
    raise SystemExit(main())
