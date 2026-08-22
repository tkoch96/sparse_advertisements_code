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
        # shards store raw float seconds; parse_lat = *1000 + clamp
        lat_f = lat_s[keep] * 1000.0
        uniq, inv = np.unique(lat_f, return_inverse=True)
        from helpers.helpers import MIN_LATENCY, MAX_LATENCY
        parsed = [min(max(u, MIN_LATENCY), MAX_LATENCY) for u in uniq]
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


def _pop_min_worker(job):
    """Per-pop vectorized reduction: (pop, shard_dir, ignore_popps_pop,
    violate_keys) -> (pop, [(ip, peer, min_lat)])."""
    import json as _json
    pop, shard_dir, ignored_peers, violate_ips, min_lat, max_lat = job
    npz_fn = os.path.join(shard_dir, pop + '.npz')
    if not os.path.exists(npz_fn):
        return pop, []
    z = np.load(npz_fn)
    with open(os.path.join(shard_dir, pop + '.strings.json')) as f:
        pools = _json.load(f)
    ips, peers = pools['ips'], pools['peers']
    ip_id, peer_id, lat_s = z['ip_id'], z['peer_id'], z['lat']
    peer_ok = np.asarray([p not in ignored_peers for p in peers], bool)
    ip_ok = np.asarray([ip not in violate_ips for ip in ips], bool)
    keep = peer_ok[peer_id] & ip_ok[ip_id]
    if not keep.any():
        return pop, []
    kip = ip_id[keep].astype(np.uint64)
    kpeer = peer_id[keep].astype(np.uint64)
    # vectorized parse_lat: *1000 then clamp (helpers.parse_lat verbatim)
    lat = lat_s[keep] * 1000.0
    np.clip(lat, min_lat, max_lat, out=lat)
    key = kip * np.uint64(len(peers)) + kpeer
    # sort + reduceat groupby: np.minimum.at is a notoriously slow
    # ufunc.at path (~100x) at tens of millions of rows (26-pop load
    # hung for 30+ min before this swap; math identical)
    order = np.argsort(key, kind='stable')
    ks = key[order]
    ls = lat[order]
    starts = np.flatnonzero(np.concatenate(([True], ks[1:] != ks[:-1])))
    uniq = ks[starts]
    mins = np.minimum.reduceat(ls, starts)
    u_ip = (uniq // np.uint64(len(peers))).astype(np.int64)
    u_peer = (uniq % np.uint64(len(peers))).astype(np.int64)
    return pop, [(ips[u_ip[i]], peers[u_peer[i]], float(mins[i]))
                 for i in range(len(uniq))]


def build_ug_perfs_min(shard_dir, considering_pops, ignore_popps,
                       violate_sol, procs=1, ug_perfs=None):
    """Vectorized phase-2 loader: returns ug_perfs with per-(ug, popp)
    MINIMUM latencies directly (the downstream reduction block is
    idempotent: np.min(scalar) == scalar). Equivalent to
    build_ug_perfs + the line-912 reduction; gated in
    unit_tests/test_phase2_gate.py."""
    from helpers.helpers import MIN_LATENCY, MAX_LATENCY
    ug_perfs = {} if ug_perfs is None else ug_perfs
    violate_ips = {ug[1] for ug, v in violate_sol.items() if v}
    jobs = []
    for pop in sorted(set(considering_pops)):
        ignored = {peer for (p, peer) in ignore_popps if p == pop}
        jobs.append((pop, shard_dir, ignored, violate_ips,
                     MIN_LATENCY, MAX_LATENCY))
    if procs > 1:
        import multiprocessing as mp
        with mp.Pool(procs) as pool:
            results = pool.map(_pop_min_worker, jobs)
    else:
        results = [_pop_min_worker(j) for j in jobs]
    for pop, triples in results:
        for ip, peer, m in triples:
            ug = ('tmp', ip)
            try:
                ug_perfs[ug][pop, peer] = m
            except KeyError:
                ug_perfs[ug] = {(pop, peer): m}
    return ug_perfs
