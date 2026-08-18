"""Phase-2 gate: build_ug_perfs_min (vectorized, optional multiprocess)
must equal build_ug_perfs + the mainline line-912 reduction, exactly.

    python -m experiments.depcache.test_phase2_gate [--pops p1,p2,...]
"""
import argparse
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    from helpers import parse_lat
    from experiments.depcache.shard_loader import (
        build_ug_perfs, build_ug_perfs_min)
    ap = argparse.ArgumentParser()
    ap.add_argument('--pops', default='vtrtokyo,vtrwarsaw,vtrstockholm')
    ap.add_argument('--shards', default='cache/lat_shards')
    args = ap.parse_args()
    pops = set(args.pops.split(','))
    shards = (args.shards if os.path.isabs(args.shards)
              else os.path.join(_REPO_ROOT, args.shards))

    t0 = time.time()
    ref = build_ug_perfs(shards, pops, {}, {}, parse_lat)
    for ug in ref:
        for popp in ref[ug]:
            ref[ug][popp] = float(np.min(ref[ug][popp]))
    t_ref = time.time() - t0

    timings = {'phase1+reduce': t_ref}
    for name, procs in (('phase2 serial', 1), ('phase2 x8', 8)):
        t0 = time.time()
        got = build_ug_perfs_min(shards, pops, {}, {}, procs=procs)
        timings[name] = time.time() - t0
        assert set(ref) == set(got), name
        bad = sum(1 for ug in ref if ref[ug] != got[ug])
        assert bad == 0, (name, bad)
    print('EXACT over {} ugs | '.format(len(ref)) + ' | '.join(
        '{} {:.1f}s'.format(k, v) for k, v in timings.items()))


if __name__ == '__main__':
    main()
