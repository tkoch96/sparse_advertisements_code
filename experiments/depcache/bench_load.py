"""Load-time benchmark: CSV loop vs shard loader across pop counts
(Tom 2026-08-17). Both arms produce the identical end product (per-
(ug,popp) minimum latencies, no ignore/violate filters) so timings are
apples-to-apples. Appends rows to cache/depcache_bench.json after each
measurement so partial results plot immediately.

    nice -n 10 python -m experiments.depcache.bench_load \
        --npops 2,4,8,12,16,20,26
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def csv_arm(csv_fn, pops):
    from helpers import parse_lat
    keep = set(pops)
    up = {}
    for row in open(csv_fn):
        fields = row.strip().split(',')
        if len(fields) != 6 or fields[2] not in keep:
            continue
        _, ip, pop, peer, _x, lat = fields
        lat = parse_lat(lat)
        ug = ('tmp', ip)
        key = (pop, peer)
        d = up.get(ug)
        if d is None:
            up[ug] = {key: lat}
        else:
            v = d.get(key)
            if v is None or lat < v:
                d[key] = lat
    return up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npops', default='2,4,8,12,16,20,26')
    ap.add_argument('--shards', default='cache/lat_shards')
    ap.add_argument('--csv', default='cache/vultr_ingress_latencies_by_dst.csv')
    ap.add_argument('--out', default='cache/depcache_bench.json')
    args = ap.parse_args()
    from experiments.depcache.shard_loader import build_ug_perfs_min
    shards = os.path.join(_REPO_ROOT, args.shards)
    csv_fn = os.path.join(_REPO_ROOT, args.csv)
    out_fn = os.path.join(_REPO_ROOT, args.out)
    all_pops = sorted(f.split('.npz')[0] for f in os.listdir(shards)
                      if f.endswith('.npz'))
    rows = []
    if os.path.exists(out_fn):
        rows = json.load(open(out_fn))
    done = {r['n_pops'] for r in rows}
    for n in [int(x) for x in args.npops.split(',')]:
        if n in done:
            continue
        pops = all_pops[:n]
        t0 = time.time()
        got = build_ug_perfs_min(shards, pops, {}, {}, procs=1)
        t_shard = time.time() - t0
        n_ugs = len(got)
        del got
        t0 = time.time()
        ref = csv_arm(csv_fn, pops)
        t_csv = time.time() - t0
        del ref
        rows.append({'n_pops': n, 'shard_s': round(t_shard, 1),
                     'csv_s': round(t_csv, 1), 'ugs': n_ugs})
        json.dump(rows, open(out_fn, 'w'), indent=1)
        print('n_pops={} shard {:.1f}s csv {:.1f}s ({:.1f}x)'.format(
            n, t_shard, t_csv, t_csv / max(t_shard, 1e-9)), flush=True)


if __name__ == '__main__':
    main()
