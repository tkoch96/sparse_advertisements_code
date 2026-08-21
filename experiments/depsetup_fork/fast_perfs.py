"""Array-native deployment-creation fork, phase B1 (Tom 2026-08-18:
"keep everything about parsing separate by pop/ug/ingress until the
LAST possible second").

Insight that makes byte-exactness cheap: for actual-N sizes
DO_UG_CLUSTERING is False, so the pipeline is parse -> per-(ug,popp)
min -> key-level filters -> do_filter quota selection. The ONLY
expensive stage is materializing the parse product as a Python
dict-of-LISTS holding every raw measurement (tens of millions of
floats + list objects). Everything downstream only ever consumes
np.min(lats) per (ug, popp).

So B1 keeps the raw measurements as numpy arrays per pop shard,
reduces to per-(ug, popp) minima VECTORIZED (sort + minimum.reduceat,
the np.minimum.at path is ~100x slower), and materializes a
dict-of-SCALARS -- 10-30x fewer Python objects -- in EXACTLY the
insertion order the original shard loop would have used (pops sorted,
first-appearance row order within a pop). The untouched mainline
load_actual_perfs then runs its own tail verbatim (its min loop
becomes np.min(scalar) no-ops), which makes the final output
byte-exact by construction, including the RNG stream consumed by
do_filter's shuffles (dict iteration order is preserved, and Python
set/hash behavior sees identical inputs in identical order within a
process).

Install via install() (monkeypatches
experiments.depcache.shard_loader.build_ug_perfs, the seam
deployment_setup already calls when SCULPTOR_LAT_SHARDS is set), or
run the A/B gate: python -m experiments.depsetup_fork.gate_5pop
"""
import json
import os
import time

import numpy as np

from helpers.helpers import MIN_LATENCY, MAX_LATENCY

STAGE_T = {}


def _t(k, t0):
    STAGE_T[k] = STAGE_T.get(k, 0.0) + (time.time() - t0)


def build_ug_perfs_min(shard_dir, considering_pops, ignore_popps,
                       violate_sol, parse_lat, ug_perfs=None):
    """Drop-in replacement for shard_loader.build_ug_perfs returning
    per-(ug, popp) MIN latencies as np.float64 scalars instead of raw
    measurement lists. Downstream np.min(lats) is a value-preserving
    no-op on scalars, and min(clamped measurements) == clamp(min) since
    the parse_lat clamp is monotone. Insertion order mirrors the
    original loop exactly: pops sorted, (ip, peer) pairs by first
    appearance in shard row order."""
    STAGE_T.clear()
    ug_perfs = {} if ug_perfs is None else ug_perfs
    for pop in sorted(set(considering_pops)):
        npz_fn = os.path.join(shard_dir, pop + '.npz')
        if not os.path.exists(npz_fn):
            continue
        t0 = time.time()
        z = np.load(npz_fn)
        with open(os.path.join(shard_dir, pop + '.strings.json')) as f:
            pools = json.load(f)
        ips, peers = pools['ips'], pools['peers']
        ip_id, peer_id, lat_s = z['ip_id'], z['peer_id'], z['lat']
        _t('load', t0)

        t0 = time.time()
        peer_ok = np.asarray(
            [(pop, p) not in ignore_popps for p in peers], dtype=bool)
        ip_ok = np.asarray(
            [not violate_sol.get(('tmp', ip), False) for ip in ips],
            dtype=bool)
        keep = peer_ok[peer_id] & ip_ok[ip_id]
        if not keep.any():
            continue
        kip = ip_id[keep].astype(np.int64)
        kpeer = peer_id[keep].astype(np.int64)
        # parse_lat vectorized: *1000 then clamp; clamp-then-min ==
        # min-then-clamp (monotone), so clamping the minima suffices
        klat = lat_s[keep] * 1000.0
        _t('filter', t0)

        t0 = time.time()
        key = kip * len(peers) + kpeer
        order = np.argsort(key, kind='stable')
        ks = key[order]
        ls = klat[order]
        starts = np.concatenate(
            ([0], np.flatnonzero(np.diff(ks)) + 1))
        mins = np.minimum.reduceat(ls, starts)
        np.clip(mins, MIN_LATENCY, MAX_LATENCY, out=mins)
        upair = ks[starts]
        # first-appearance order of each unique (ip, peer) pair in the
        # ORIGINAL row order (np.unique's return_index is the first
        # occurrence; upair from the sort is the same sorted unique set)
        first_row = np.unique(key, return_index=True)[1]
        appearance = np.argsort(first_row, kind='stable')
        _t('reduce', t0)

        t0 = time.time()
        uip = (upair // len(peers)).tolist()
        upeer = (upair % len(peers)).tolist()
        mins_l = mins.tolist()
        for j in appearance.tolist():
            ug = ('tmp', ips[uip[j]])
            popp_key = (pop, peers[upeer[j]])
            val = np.float64(mins_l[j])
            try:
                ug_perfs[ug][popp_key] = val
            except KeyError:
                ug_perfs[ug] = {popp_key: val}
        _t('materialize', t0)
    return ug_perfs


_ORIG = None


def install():
    """Route deployment_setup's SCULPTOR_LAT_SHARDS seam through the
    array-native minimizing loader."""
    global _ORIG
    from experiments.depcache import shard_loader
    if _ORIG is None:
        _ORIG = shard_loader.build_ug_perfs
    shard_loader.build_ug_perfs = build_ug_perfs_min


def uninstall():
    from experiments.depcache import shard_loader
    if _ORIG is not None:
        shard_loader.build_ug_perfs = _ORIG
