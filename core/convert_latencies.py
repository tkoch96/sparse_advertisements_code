"""One-time parallel conversion of the 4.3GB per-ingress latency CSV
into per-pop binary shards (Tom 2026-08-17: deployment creation is
bottlenecked on re-parsing this file for every new pop combination;
at 32 pops the filter passes everything).

Shards: <out>/<pop>.npz with parallel arrays (ip_id, peer_id, lat_ms)
+ <out>/<pop>.strings.json {ips: [...], peers: [...]} + a manifest.
Global row order is preserved (chunk-ordered merge) so the shard loader
reproduces the CSV loop's ug_perfs lists EXACTLY.

    python -m core.convert_latencies \
        --csv cache/vultr_ingress_latencies_by_dst.csv \
        --out cache/lat_shards --procs 8
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def chunk_bounds(fn, n):
    size = os.path.getsize(fn)
    bounds = [0]
    with open(fn, 'rb') as f:
        for i in range(1, n):
            f.seek(size * i // n)
            f.readline()
            bounds.append(f.tell())
    bounds.append(size)
    return [(bounds[i], bounds[i + 1]) for i in range(n)
            if bounds[i] < bounds[i + 1]]


def parse_chunk(job):
    fn, start, end, ci = job
    per_pop = {}
    with open(fn, 'rb') as f:
        f.seek(start)
        while f.tell() < end:
            row = f.readline()
            if not row:
                break
            fields = row.decode('ascii', 'replace').strip().split(',')
            if len(fields) != 6:
                continue
            _, ip, pop, peer, _x, lat = fields
            d = per_pop.setdefault(pop, ([], [], []))
            d[0].append(ip)
            d[1].append(peer)
            d[2].append(lat)
    return ci, per_pop


def build_shards(csv, out, procs=None):
    """Build the per-pop shards from the CSV. Writes into a tmp sibling
    and renames into place so concurrent builders / readers never see a
    half-written shard dir (the manifest is the last thing written, and
    available() keys on the manifest). Returns True on success."""
    procs = procs or max(2, mp.cpu_count() - 2)
    final_out = out
    out = '{}.tmp.{}'.format(final_out, os.getpid())
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    jobs = [(csv, s, e, i) for i, (s, e) in
            enumerate(chunk_bounds(csv, procs * 4))]
    print('parsing {} ({:.1f}GB) in {} chunks x {} procs'.format(
        os.path.basename(csv), os.path.getsize(csv) / 1e9, len(jobs),
        procs), flush=True)
    merged = {}
    _ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp
    with _ctx.Pool(procs) as pool:
        for ci, per_pop in sorted(pool.imap_unordered(parse_chunk, jobs),
                                  key=lambda r: r[0]):
            for pop, (ips, peers, lats) in sorted(per_pop.items()):
                m = merged.setdefault(pop, ([], [], []))
                m[0].append((ci, ips))
                m[1].append((ci, peers))
                m[2].append((ci, lats))
    # chunk-ordered concatenation preserves global row order
    manifest = {}
    for pop, (ips_c, peers_c, lats_c) in sorted(merged.items()):
        ips = [x for _, chunk in sorted(ips_c) for x in chunk]
        peers = [x for _, chunk in sorted(peers_c) for x in chunk]
        lats = [x for _, chunk in sorted(lats_c) for x in chunk]
        ip_pool = sorted(set(ips))
        peer_pool = sorted(set(peers))
        ip_id = {s: i for i, s in enumerate(ip_pool)}
        peer_id = {s: i for i, s in enumerate(peer_pool)}
        np.savez_compressed(
            os.path.join(out, pop + '.npz'),
            ip_id=np.asarray([ip_id[s] for s in ips], dtype=np.uint32),
            peer_id=np.asarray([peer_id[s] for s in peers], dtype=np.uint32),
            lat=np.asarray([float(x) for x in lats],
                           dtype=np.float64))
        with open(os.path.join(out, pop + '.strings.json'), 'w') as f:
            json.dump({'ips': ip_pool, 'peers': peer_pool}, f)
        manifest[pop] = len(ips)
        print('  {}: {} rows'.format(pop, len(ips)), flush=True)
    with open(os.path.join(out, 'manifest.json'), 'w') as f:
        json.dump({'source': os.path.basename(csv),
                   'source_bytes': os.path.getsize(csv),
                   'rows_by_pop': manifest}, f, indent=1)
    # atomic-ish promote: lose the race gracefully if another process
    # finished first (its manifest is just as good)
    try:
        if os.path.isdir(final_out) and not os.listdir(final_out):
            os.rmdir(final_out)
        os.rename(out, final_out)
    except OSError:
        from core.shard_loader import available as _avail
        if not _avail(final_out):
            raise
        import shutil; shutil.rmtree(out, ignore_errors=True)
    print('done in {:.0f}s -> {}'.format(time.time() - t0, final_out))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='cache/vultr_ingress_latencies_by_dst.csv')
    ap.add_argument('--out', default='cache/lat_shards')
    ap.add_argument('--procs', type=int, default=max(2, mp.cpu_count() - 2))
    args = ap.parse_args()
    csv = args.csv if os.path.isabs(args.csv) else os.path.join(_REPO_ROOT, args.csv)
    out = args.out if os.path.isabs(args.out) else os.path.join(_REPO_ROOT, args.out)
    build_shards(csv, out, args.procs)


if __name__ == '__main__':
    main()
