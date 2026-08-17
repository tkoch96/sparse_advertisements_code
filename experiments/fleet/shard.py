"""Shard a queue manifest across N VMs (Tom 2026-08-17, SCALE-500 P1).

Deployment-major: seeds are split across shards per spec, so a VM keeps
every cell of a (spec, seed) pair — deployment build/cache reuse stays
local to one VM. Painter/reference specs (single-seed or tiny) ride with
shard 0.

    python -m experiments.fleet.shard --manifest tools/eods_manifest.json \
        --n 6 --out-dir tools/shards/eods
"""
import argparse
import json
import os


def parse_seeds(spec):
    s = str(spec)
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',') if x]


def shard_manifest(specs, n):
    shards = [[] for _ in range(n)]
    for sp in specs:
        seeds = parse_seeds(sp.get('seeds', '1-5'))
        if len(seeds) == 1:
            shards[0].append(dict(sp))
            continue
        for i in range(n):
            mine = seeds[i::n]
            if not mine:
                continue
            c = dict(sp)
            c['seeds'] = ','.join(str(s) for s in mine)
            shards[i].append(c)
    return shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    specs = json.load(open(args.manifest))
    shards = shard_manifest(specs, args.n)
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.manifest))[0]
    for i, sh in enumerate(shards):
        fn = os.path.join(args.out_dir, '{}.shard{}.json'.format(base, i))
        with open(fn, 'w') as f:
            json.dump(sh, f, indent=1)
        cells = sum(len(parse_seeds(s['seeds']))
                    * len(str(s['n_values']).split(',')) for s in sh)
        print('shard {}: {} specs, {} cells -> {}'.format(
            i, len(sh), cells, fn))


if __name__ == '__main__':
    main()
