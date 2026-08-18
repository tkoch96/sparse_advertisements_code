"""Phase B2: full array-native fork of load_actual_perfs (Tom
2026-08-18: keep everything sharded/array until the LAST possible
second). All per-measurement and per-(ug,popp) work stays in numpy;
Python dicts materialize only for the final few-thousand-UG survivor
set. Byte-exactness strategy vs mainline (gated by gate_5pop --b2):

- Values: groupby-min via sort + minimum.reduceat is bitwise-equal to
  np.min over the same floats; clamp-then-min == min-then-clamp
  (monotone); np.array(strs, float) uses the same strtod as float(str).
- Order: every order-sensitive step is replicated at the KEY level
  with real Python sets/lists (get_intersection on key lists, set()
  over the same tuple sequence), because mainline's dict/set iteration
  orders feed the RNG-consuming quota selection.
- RNG: np.random.shuffle on a same-length index array consumes the
  identical MT19937 stream as on a Python list and yields the same
  permutation (verified), and len>0 gates are replicated.
- The provider-pruning tail (operating on the small survivor dict) is
  copied verbatim from deployment_setup so its hash-order-dependent
  helpers behave identically.
"""
import os
import time

import numpy as np

from helpers import (MIN_LATENCY, MAX_LATENCY, get_intersection,
                     get_difference, parse_lat)
from constants import POP_TO_LOC

STAGE_T = {}


def _t(k, t0):
    STAGE_T[k] = STAGE_T.get(k, 0.0) + (time.time() - t0)
    return time.time()


def _seg_starts(sorted_keys):
    return np.concatenate(
        ([0], np.flatnonzero(np.diff(sorted_keys)) + 1))


def load_actual_perfs_arrays(considering_pops, **kwargs):
    """Array-native equivalent of deployment_setup.load_actual_perfs
    (shard-backed path). Returns (anycast_latencies, ug_perfs)."""
    import deployment_setup as ds
    STAGE_T.clear()
    t0 = time.time()
    print("Loading performances, only considering pops: {}".format(
        considering_pops))
    lat_fn = os.path.join(ds.CACHE_DIR, 'vultr_ingress_latencies_by_dst.csv')
    assert os.path.exists(lat_fn)
    pop_to_loc = {pop: POP_TO_LOC['vultr'][pop] for pop in considering_pops}
    violate_sol = {}
    for row in open(os.path.join(ds.CACHE_DIR,
                                 'addresses_violating_sol.csv'), 'r'):
        metro, asn, violates = row.strip().split(',')
        violate_sol[metro, asn] = int(violates)
    import geopy.distance
    pop_dists = {}
    for i, popi in enumerate(considering_pops):
        for j, popj in enumerate(considering_pops):
            if j > i:
                continue
            if j == i:
                pop_dists[popi, popj] = 0
                pop_dists[popj, popi] = 0
            pop_dists[popi, popj] = geopy.distance.geodesic(
                pop_to_loc[popi], pop_to_loc[popj]).km
            pop_dists[popj, popi] = pop_dists[popi, popj]

    all_popps = {}
    for row in open(os.path.join(ds.DATA_DIR,
                                 'vultr_peers_inferred.csv'), 'r'):
        pop, peer, _, tp, _ = row.strip().split(',')
        try:
            all_popps[pop, peer].append(tp)
        except KeyError:
            all_popps[pop, peer] = [tp]
    ignore_popps = {popp: None for popp, tps in all_popps.items()
                    if len(set(tps)) == 1 and tps[0] == 'routeserver'}
    t0 = _t('header', t0)

    # ---- shard parse -> per-(ug,popp) minima as flat arrays ----
    import json as _json
    shard_dir = os.environ.get('SCULPTOR_LAT_SHARDS')
    ug_keys = []            # ('tmp', ip) in first-appearance order
    ug_index = {}
    popp_keys = []
    popp_index = {}
    pair_ug, pair_popp, pair_lat = [], [], []
    for pop in sorted(set(considering_pops)):
        npz_fn = os.path.join(shard_dir, pop + '.npz')
        if not os.path.exists(npz_fn):
            continue
        z = np.load(npz_fn)
        with open(os.path.join(shard_dir, pop + '.strings.json')) as f:
            pools = _json.load(f)
        ips, peers = pools['ips'], pools['peers']
        ip_id, peer_id, lat_s = z['ip_id'], z['peer_id'], z['lat']
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
        klat = lat_s[keep] * 1000.0
        key = kip * len(peers) + kpeer
        order = np.argsort(key, kind='stable')
        ks, ls = key[order], klat[order]
        starts = _seg_starts(ks)
        mins = np.minimum.reduceat(ls, starts)
        np.clip(mins, MIN_LATENCY, MAX_LATENCY, out=mins)
        upair = ks[starts]
        first_row = np.unique(key, return_index=True)[1]
        appearance = np.argsort(first_row, kind='stable')
        uip = (upair // len(peers))[appearance]
        upeer = (upair % len(peers))[appearance]
        umin = mins[appearance]
        # map local ip ids -> global ug ranks; register new ugs in
        # first-PAIR-arrival order (mirrors dict insertion order)
        uniq_ip, ip_first = np.unique(uip, return_index=True)
        arrival_ips = uniq_ip[np.argsort(ip_first, kind='stable')]
        local_to_global = np.empty(len(ips), dtype=np.int64)
        for li in arrival_ips.tolist():
            k = ('tmp', ips[li])
            r = ug_index.get(k)
            if r is None:
                r = len(ug_keys)
                ug_index[k] = r
                ug_keys.append(k)
            local_to_global[li] = r
        uniq_peer = np.unique(upeer)
        peer_to_global = np.empty(len(peers), dtype=np.int64)
        for pi in uniq_peer.tolist():
            k = (pop, peers[pi])
            r = popp_index.get(k)
            if r is None:
                r = len(popp_keys)
                popp_index[k] = r
                popp_keys.append(k)
            peer_to_global[pi] = r
        pair_ug.append(local_to_global[uip])
        pair_popp.append(peer_to_global[upeer])
        pair_lat.append(umin)
    pair_ug = np.concatenate(pair_ug)
    pair_popp = np.concatenate(pair_popp)
    pair_lat = np.concatenate(pair_lat)
    n_ug = len(ug_keys)
    print("{} UGs, {} popps read from measurement file".format(
        n_ug, len(popp_keys)))
    t0 = _t('parse_min', t0)

    # ---- single-ingress + <=1ms UG filters (vectorized) ----
    counts = np.bincount(pair_ug, minlength=n_ug)
    o = np.argsort(pair_ug, kind='stable')
    su, sl = pair_ug[o], pair_lat[o]
    st = _seg_starts(su)
    ug_min = np.full(n_ug, np.inf)
    ug_min[su[st]] = np.minimum.reduceat(sl, st)
    keep_ug = (counts >= 2) & (ug_min > 1)
    pmask = keep_ug[pair_ug]
    new_rank = np.cumsum(keep_ug) - 1
    pair_ug = new_rank[pair_ug[pmask]]
    pair_popp = pair_popp[pmask]
    pair_lat = pair_lat[pmask]
    cur_keys = [k for k, m in zip(ug_keys, keep_ug.tolist()) if m]
    print("{} UGs, {} popps after removing 1ms UGs".format(
        len(cur_keys), len(np.unique(pair_popp))))
    t0 = _t('filter_1ms', t0)

    # ---- anycast parse (vectorized) + per-ug min ----
    rows = open(os.path.join(
        ds.CACHE_DIR, 'vultr_anycast_latency_smaller.csv'), 'r').read()
    parts = [l.split(',') for l in rows.split('\n') if l]
    del rows
    skip_ips = set(ip for (m, ip), v in violate_sol.items() if v)
    a_ips = [p[1] for p in parts]
    a_lats = [p[2] for p in parts]
    a_keepm = np.asarray(
        [(l != '-1') and (ip not in skip_ips)
         for ip, l in zip(a_ips, a_lats)], dtype=bool)
    a_ip_arr = np.asarray(a_ips, dtype=object)[a_keepm]
    a_lat_arr = np.asarray(a_lats, dtype=np.float64)[
        a_keepm] * 1000.0
    np.clip(a_lat_arr, MIN_LATENCY, MAX_LATENCY, out=a_lat_arr)
    del parts, a_ips, a_lats
    a_uniq, a_first, a_inv = np.unique(
        a_ip_arr.astype(str), return_index=True, return_inverse=True)
    ao = np.argsort(a_inv, kind='stable')
    a_st = _seg_starts(a_inv[ao])
    a_min = np.minimum.reduceat(a_lat_arr[ao], a_st)
    a_min_by_uniq = np.empty(len(a_uniq))
    a_min_by_uniq[a_inv[ao][a_st]] = a_min
    a_arrival = np.argsort(a_first, kind='stable')
    anycast_keys = [('tmp', ip) for ip in a_uniq[a_arrival].tolist()]
    anycast_vals = {k: np.float64(v) for k, v in zip(
        anycast_keys, a_min_by_uniq[a_arrival].tolist())}
    t0 = _t('anycast', t0)

    # ---- intersect (order via the REAL helper for hash parity) ----
    def _restrict(order_keys):
        """Keep pairs whose ug is in order_keys; remap ranks to the
        new order. cur rank map derives from cur_keys."""
        nonlocal pair_ug, pair_popp, pair_lat, cur_keys
        idx_of = {k: i for i, k in enumerate(cur_keys)}
        old_to_new = np.full(len(cur_keys), -1, dtype=np.int64)
        for ni, k in enumerate(order_keys):
            old_to_new[idx_of[k]] = ni
        m = old_to_new[pair_ug] >= 0
        pair_ug = old_to_new[pair_ug[m]]
        pair_popp = pair_popp[m]
        pair_lat = pair_lat[m]
        cur_keys = list(order_keys)

    # get_intersection args MUST be dicts, as mainline passes: CPython
    # set() PRESIZES its table for a dict argument but grows
    # incrementally for a list, and the different resize history
    # changes the result set's iteration order (found via the 10-pop
    # gate: single adjacent-swap divergences). dict.fromkeys preserves
    # our key order at negligible cost.
    in_both = get_intersection(dict.fromkeys(cur_keys),
                               dict.fromkeys(anycast_keys))
    _restrict(in_both)
    # mainline rebuilds anycast_latencies restricted to in_both here;
    # the NEXT intersection's set-iteration order depends on operand
    # contents, so mirror the restriction at the key level
    anycast_keys = list(in_both)
    print("{} UGs, {} popps after limiting to clients who have an "
          "anycast latency".format(
              len(cur_keys), len(np.unique(pair_popp))))
    t0 = _t('intersect1', t0)

    # ---- SOL physics filter (vectorized over pop pairs) ----
    pops_l = sorted(set(considering_pops))
    pop_of_popp = np.asarray(
        [pops_l.index(p) for p, _ in popp_keys], dtype=np.int64)
    n_pops = len(pops_l)
    ppop = pop_of_popp[pair_popp]
    key2 = pair_ug * n_pops + ppop
    o2 = np.argsort(key2, kind='stable')
    st2 = _seg_starts(key2[o2])
    seg_min = np.minimum.reduceat(pair_lat[o2], st2)
    seg_key = key2[o2][st2]
    lat_by_ug_pop = np.full((len(cur_keys), n_pops), np.nan)
    lat_by_ug_pop[seg_key // n_pops, seg_key % n_pops] = seg_min
    known = np.asarray([k in violate_sol for k in cur_keys], dtype=bool)
    known_bad = np.asarray(
        [bool(violate_sol.get(k, 0)) for k in cur_keys], dtype=bool)
    viol = np.zeros(len(cur_keys), dtype=bool)
    for i in range(n_pops):
        for j in range(n_pops):
            d = pop_dists[pops_l[i], pops_l[j]] * .01
            with np.errstate(invalid='ignore'):
                viol |= (lat_by_ug_pop[:, i] + lat_by_ug_pop[:, j]
                         + 5 <= d)
    # mainline skips the physics check for ANY ug already present in
    # violate_sol (0 or 1) and only appends NEW marks, in iteration
    # order; replicate both
    to_del_mask = (known & known_bad) | (~known & viol)
    for i, k in enumerate(cur_keys):
        if not known[i]:
            violate_sol[k] = 1 if viol[i] else 0
    with open(os.path.join(ds.CACHE_DIR,
                           'addresses_violating_sol.csv'), 'w') as f:
        f.write(''.join('{},{},{}\n'.format(m, a, v)
                        for (m, a), v in violate_sol.items()))
    print("{} UGs violate SOL rules".format(
        sum(violates for violates in violate_sol.values())))
    keep_keys = [k for k, m in zip(cur_keys, to_del_mask.tolist())
                 if not m]
    _restrict(keep_keys)
    in_both = get_intersection(dict.fromkeys(cur_keys),
                               dict.fromkeys(anycast_keys))
    _restrict(in_both)
    print("{} UGs, {} popps after removing SOL and only considering "
          "anycast ones".format(
              len(cur_keys), len(np.unique(pair_popp))))
    t0 = _t('sol', t0)

    # ---- best popp per ug (first-min in arrival order) ----
    o3 = np.argsort(pair_ug, kind='stable')
    su3, sl3 = pair_ug[o3], pair_lat[o3]
    st3 = _seg_starts(su3)
    seg_lens = np.diff(np.concatenate((st3, [len(su3)])))
    seg_min3 = np.minimum.reduceat(sl3, st3)
    exp_min = np.repeat(seg_min3, seg_lens)
    pos = np.arange(len(sl3))
    masked = np.where(sl3 == exp_min, pos, len(sl3))
    first_min = np.minimum.reduceat(masked, st3)
    best_popp_of_ug = np.empty(len(cur_keys), dtype=np.int64)
    best_popp_of_ug[su3[st3]] = pair_popp[o3[first_min]]
    t0 = _t('best_popp', t0)

    if kwargs.get('do_filter', True):
        default_max_n_ug = 15
        max_n_ug = kwargs.get('n_users_per_peer', default_max_n_ug)
        provider_fn = os.path.join(ds.CACHE_DIR,
                                   'vultr_provider_popps.csv')
        provider_popps, provider_popps_d = [], {}
        for row in open(provider_fn, 'r'):
            pop, peer = row.strip().split(',')
            if pop not in considering_pops:
                continue
            provider_popps.append((pop, peer))
            provider_popps_d[pop, peer] = None
        present = np.unique(pair_popp)
        popps_sorted = sorted(popp_keys[i] for i in present.tolist())
        prov_best = np.asarray(
            [popp_keys[b] in provider_popps_d
             for b in best_popp_of_ug.tolist()], dtype=bool)
        # per-popp ug lists in outer-ug order: sort pairs by
        # (popp, ug rank); iterate popps in sorted-key order
        o4 = np.lexsort((pair_ug, pair_popp))
        sp, sug = pair_popp[o4], pair_ug[o4]
        st4 = _seg_starts(sp)
        seg_of_popp = {int(sp[s]): (s, e) for s, e in zip(
            st4, np.concatenate((st4[1:], [len(sp)])))}
        n_total_users, n_peer_was_best, n_provider_was_best = 0, 0, 0
        kept_seq = []           # (popp_key, ug_rank) kept, popp-major
        for popp in popps_sorted:
            s, e = seg_of_popp[popp_index[popp]]
            seg_ugs = sug[s:e]
            pb = prov_best[seg_ugs]
            peer_ugs = seg_ugs[~pb].copy()
            provider_ugs = seg_ugs[pb].copy()
            if len(peer_ugs) > 0:
                np.random.shuffle(peer_ugs)
            if len(provider_ugs) > 0:
                np.random.shuffle(provider_ugs)
            n_keep = np.minimum(len(seg_ugs), max_n_ug)
            _ugs = np.concatenate((peer_ugs, provider_ugs))[0:n_keep]
            kept_seq.append((popp, _ugs))
            n_total_users += n_keep
            n_keeping_peer = np.minimum(len(peer_ugs), max_n_ug)
            n_keeping_provider = np.minimum(
                max_n_ug - n_keeping_peer, len(provider_ugs))
            n_provider_was_best += n_keeping_provider
            n_peer_was_best += n_keeping_peer
        print("Out of {} UGs, {} ({} pct) peer was best, {} ({} pct) "
              "provider was best.".format(
                  n_total_users, n_peer_was_best,
                  round(n_peer_was_best * 100 / n_total_users, 2),
                  n_provider_was_best,
                  round(n_provider_was_best * 100 / n_total_users, 2)))
        if kwargs.get('focus_on_peers', True):
            keep_ugs = list(set(
                cur_keys[u] for popp, ugs in kept_seq
                if popp not in provider_popps for u in ugs.tolist()))
        else:
            keep_ugs = list(set(
                cur_keys[u] for popp, ugs in kept_seq
                for u in ugs.tolist()))
        # ---- LAST POSSIBLE SECOND: materialize survivor dicts ----
        idx_of = {k: i for i, k in enumerate(cur_keys)}
        kept_mask = np.zeros(len(cur_keys), dtype=bool)
        for k in keep_ugs:
            kept_mask[idx_of[k]] = True
        m = kept_mask[pair_ug]
        small_ug = pair_ug[m].tolist()
        small_popp = pair_popp[m].tolist()
        small_lat = pair_lat[m].tolist()
        by_rank = {}
        for u, p, l in zip(small_ug, small_popp, small_lat):
            k = popp_keys[p]
            v = np.float64(l)
            try:
                by_rank[u][k] = v
            except KeyError:
                by_rank[u] = {k: v}
        ug_perfs = {k: by_rank[idx_of[k]] for k in keep_ugs}
        print("{} UGs after limiting to those with a peer "
              "measurement".format(len(ug_perfs)))
        t0 = _t('do_filter', t0)

        # ---- provider-pruning tail: verbatim mainline code on the
        # small survivor dict ----
        n_ugs_by_provider = {provider: 0 for provider in provider_popps}
        n_providers_by_ug = {ug: 0 for ug in ug_perfs}
        for ug in ug_perfs:
            for provider in provider_popps:
                try:
                    ug_perfs[ug][provider]
                    n_ugs_by_provider[provider] += 1
                    n_providers_by_ug[ug] += 1
                except KeyError:
                    continue
        to_del_popps = []
        for popp, n in sorted(n_ugs_by_provider.items(),
                              key=lambda el: el[1]):
            if n < 2:
                to_del_popps.append(popp)
            else:
                break
        print("Removing providers : {} since they don't have enough "
              "measurements.".format(to_del_popps))
        ug_perfs = {ug: {popp: ug_perfs[ug][popp] for popp in
                         get_difference(ug_perfs[ug], to_del_popps)}
                    for ug in ug_perfs}
        provider_popps = get_difference(provider_popps, to_del_popps)
        n_ugs_by_provider = {provider: 0 for provider in provider_popps}
        n_providers_by_ug = {ug: 0 for ug in ug_perfs}
        for ug in ug_perfs:
            for provider in provider_popps:
                try:
                    ug_perfs[ug][provider]
                    n_ugs_by_provider[provider] += 1
                    n_providers_by_ug[ug] += 1
                except KeyError:
                    continue
        cutoff_frac = .35
        to_del = list([ug for ug, n in n_providers_by_ug.items()
                       if n / len(provider_popps) < cutoff_frac])
        print("Removing {} out of {} UGs since they don't have "
              "measurements to enough providers.".format(
                  len(to_del), len(ug_perfs)))
        for ug in to_del:
            del ug_perfs[ug]
        n_ugs_by_provider = {provider: 0 for provider in provider_popps}
        n_providers_by_ug = {ug: 0 for ug in ug_perfs}
        for ug in ug_perfs:
            for provider in provider_popps:
                try:
                    ug_perfs[ug][provider]
                    n_ugs_by_provider[provider] += 1
                    n_providers_by_ug[ug] += 1
                except KeyError:
                    continue
        t0 = _t('prune_tail', t0)
    else:
        # no-filter path: materialize everything (rare; parity only)
        by_rank = {}
        for u, p, l in zip(pair_ug.tolist(), pair_popp.tolist(),
                           pair_lat.tolist()):
            k = popp_keys[p]
            v = np.float64(l)
            try:
                by_rank[u][k] = v
            except KeyError:
                by_rank[u] = {k: v}
        ug_perfs = {k: by_rank[i] for i, k in enumerate(cur_keys)}

    anycast_latencies = {ug: anycast_vals[ug] for ug in ug_perfs}
    ugs = sorted(list(ug_perfs))
    popps = sorted(list(set(
        popp for ug in ugs for popp in ug_perfs[ug])))
    print("{} UGs, {} popps after limiting users".format(
        len(ugs), len(popps)))
    return anycast_latencies, ug_perfs
