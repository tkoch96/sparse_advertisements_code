"""Shard-backed replacement for the 4.3GB latency-CSV parse in
deployment_setup (Tom 2026-08-17). Loads only the requested pops'
binary shards and reproduces the CSV loop's ug_perfs EXACTLY:
per-(ug, popp) latency lists in global row order (shards preserve it),
same ignore_popps / violate_sol filters, same parse_lat application.
Key-insertion order differs from the CSV interleave, which is safe —
the caller sorts ugs/popps immediately after.
"""
import json
import os

import numpy as np


def available(shard_dir):
    return bool(shard_dir) and os.path.exists(
        os.path.join(shard_dir, 'manifest.json'))


def build_ug_perfs(shard_dir, considering_pops, ignore_popps,
                   violate_sol, parse_lat, ug_perfs=None):
    """Returns the same ug_perfs dict the CSV loop builds (optionally
    extending a caller-provided dict, matching the loop's semantics)."""
    ug_perfs = {} if ug_perfs is None else ug_perfs
    for pop in sorted(set(considering_pops)):
        npz_fn = os.path.join(shard_dir, pop + '.npz')
        if not os.path.exists(npz_fn):
            continue
        z = np.load(npz_fn)
        with open(os.path.join(shard_dir, pop + '.strings.json')) as f:
            pools = json.load(f)
        ips, peers = pools['ips'], pools['peers']
        ip_id, peer_id, lat_s = z['ip_id'], z['peer_id'], z['lat']
        # vector prefilters over the small string pools
        peer_ok = np.asarray(
            [(pop, p) not in ignore_popps for p in peers], dtype=bool)
        ip_ok = np.asarray(
            [not violate_sol.get(('tmp', ip), False) for ip in ips],
            dtype=bool)
        keep = peer_ok[peer_id] & ip_ok[ip_id]
        # parse_lat memoized over unique latency strings (heavy repeats)
        uniq, inv = np.unique(lat_s[keep], return_inverse=True)
        parsed = [parse_lat(u) for u in uniq]
        kip = ip_id[keep]
        kpeer = peer_id[keep]
        for j in range(len(kip)):
            ug = ('tmp', ips[kip[j]])
            key = (pop, peers[kpeer[j]])
            lat = parsed[inv[j]]
            try:
                ug_perfs[ug][key].append(lat)
            except KeyError:
                try:
                    ug_perfs[ug][key] = [lat]
                except KeyError:
                    ug_perfs[ug] = {key: [lat]}
    return ug_perfs
