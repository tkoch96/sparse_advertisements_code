"""Emit the EODS manifest in the queue's spec format (Tom 2026-08-17).

Cells: one per (dpsize, sim). Canonical grid from
evaluate_over_deployment_sizes.pull_results_new: dpsizes
[3,5,10,15,20,25,32] x nsim [15,20,10,16,15,15,12] = 103 cells.

    python -m experiments.eods.build_manifest \
        --out tools/eods_manifest.json --store cache/eods/v1 \
        --soln-types sparse,painter,anycast,one_per_pop,one_per_peering
"""
import argparse
import json
import os

DPSIZES = [3, 5, 10, 15, 20, 25, 32]
NSIM = [15, 20, 10, 16, 15, 15, 12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='tools/eods_manifest.json')
    ap.add_argument('--store', default='cache/eods/v1')
    ap.add_argument('--soln-types', default='painter')
    ap.add_argument('--backend', default='highs')
    ap.add_argument('--dpsizes', default=None,
                    help='comma subset, e.g. 3,5,10')
    args = ap.parse_args()

    keep = ([int(x) for x in args.dpsizes.split(',')]
            if args.dpsizes else DPSIZES)
    specs = []
    for dp, ns in zip(DPSIZES, NSIM):
        if dp not in keep:
            continue
        specs.append({
            'label': 'eods_a{}'.format(dp),
            'runner': 'experiments.eods.run_eods_cell',
            'out_root': '{}/actual-{}'.format(args.store, dp),
            'rungs': 'eods',
            'probe_mode': 'fixed',        # unused; CLI compat
            'seeds': '1-{}'.format(ns),   # seeds = sim indices
            'n_values': '1',              # single N-dir (progress compat)
            'gamma': '0',
            'max_iter': 1,                # progress denominator: 1/sim
            'dpsize': 'actual-{}'.format(dp),
            'artifacts_figs': '{}_artifacts/figs'.format(args.store),
            'env': {
                'SCULPTOR_SOLN_TYPES': args.soln_types,
                'SCULPTOR_LP_BACKEND': args.backend,
            },
        })
    with open(args.out, 'w') as f:
        json.dump(specs, f, indent=1)
    n = sum(len(range(*[int(x) for x in s['seeds'].split('-')])) + 1
            for s in specs)
    print('wrote {}: {} specs, {} cells'.format(args.out, len(specs), n))


if __name__ == '__main__':
    main()
