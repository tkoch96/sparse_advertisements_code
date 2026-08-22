"""Emit queue manifests for the EODS campaign family (Tom 2026-08-17).

Modes:
  sizes    — evaluate_over_deployment_sizes: dpsizes [3,5,10,15,20,25,32]
             x nsim [15,20,10,16,15,15,12] = 103 cells (seed = sim).
  prefixes — evaluate_over_n_prefixes at one dpsize: N-axis = prefix
             count [30..100 step 5], one cell each (seed fixed at 1).
  obj32    — all objective families at actual-32 through the standard
             grid runner (run_fork_ladder): fam x rung x seeds. RAM
             PREREQUISITE (HANDOFF): parent_tracker uint32 packing or
             r-family VMs — dpsize 25 OOM'd a 64G head pre-packing.

    python -m experiments.eods.build_manifest --mode sizes \
        --out cluster/manifests/eods_manifest.json --soln-types sparse,painter,anycast
    python -m experiments.eods.build_manifest --mode prefixes --dpsizes 20 \
        --out cluster/manifests/eonp_manifest.json
    python -m experiments.eods.build_manifest --mode obj32 --seeds 1-3 \
        --out cluster/manifests/obj32_manifest.json
"""
import argparse
import json

# Tom 2026-08-17 scoping spec: {5:30, 10:20, 15:15, 20:10, 25:10, 32:10}
DPSIZES = [5, 10, 15, 20, 25, 32]
NSIM = [30, 20, 15, 10, 10, 10]
PREFIX_COUNTS = list(range(30, 101, 5))
OBJ32_FAMILIES = {
    'lat': {'gamma': '0.1', 'env': {}},
    'fracb': {'gamma': '0', 'env': {'SCULPTOR_XOBJS': '1',
              'SCULPTOR_ABLATION_OBJECTIVE': 'frac_beyond_optimal'}},
    'mlu': {'gamma': '0', 'env': {'SCULPTOR_XOBJS': '1',
            'SCULPTOR_ABLATION_OBJECTIVE': 'max_util'}},
    'prio': {'gamma': '0', 'env': {'SCULPTOR_XOBJS': '1',
             'SCULPTOR_ABLATION_OBJECTIVE': 'joint_latency_bulk_download'}},
}
OBJ32_COMMON_ENV = {
    'SCULPTOR_ABLATION_PROBE_TARGET': 'current',
    'SCULPTOR_ABLATION_GRAD_SCALE': 'adagrad',
    'SCULPTOR_ABLATION_ALPHA0': '1',
    'SCULPTOR_STOP_RULE': 'v2',
    'SCULPTOR_ABLATION_MIN_ITER': '40',
    'SCULPTOR_ABLATION_PROBE_TCONV': '100',
}
OBJ32_RUNGS = {'L6_full_slotted': ('full', 'slotted'),
               'painter': ('painter', 'fixed')}


def base_env(args):
    env = {'SCULPTOR_SOLN_TYPES': args.soln_types,
           'SCULPTOR_LP_BACKEND': args.backend}
    # --env KEY=VAL (repeatable): extra per-cell env baked into every
    # spec. Needed because the queue exports SCULPTOR_ABLATION_PROBE_*
    # twins derived from the spec's (probe_mode, N, max_iter) queue
    # fields — which for EODS cells are progress-convention placeholders
    # (fixed/1/1), NOT training knobs. The solver checks the unprefixed
    # SCULPTOR_PROBE_* names first, so explicit values here win.
    for kv in (args.env or []):
        k, _, v = kv.partition('=')
        env[k] = v
    return env


def mode_sizes(args):
    keep = ([int(x) for x in args.dpsizes.split(',')]
            if args.dpsizes else DPSIZES)
    return [{
        'label': 'eods_a{}'.format(dp),
        'runner': 'experiments.eods.run_eods_cell',
        'out_root': '{}/actual-{}'.format(args.store, dp),
        'rungs': 'eods', 'probe_mode': 'fixed',
        # --nsim overrides the scoping-spec sim count (Tom 2026-08-19:
        # single-deployment runs at 25/32, not the 10-sim campaign)
        'seeds': '1-{}'.format(args.nsim or ns), 'n_values': '1', 'gamma': '0',
        'max_iter': 1, 'dpsize': 'actual-{}'.format(dp),
        'artifacts_figs': '{}_artifacts/figs'.format(args.store),
        'env': {**base_env(args), 'SCULPTOR_EODS_MODE': 'sizes',
                'SCULPTOR_MAX_ITER': str(args.train_iters)},
    } for dp, ns in zip(DPSIZES, NSIM) if dp in keep]


def mode_prefixes(args):
    keep = [int(x) for x in (args.dpsizes or '20').split(',')]
    return [{
        'label': 'eonp_a{}'.format(dp),
        'runner': 'experiments.eods.run_eods_cell',
        'out_root': '{}/prefixes-actual-{}'.format(args.store, dp),
        'rungs': 'eonp', 'probe_mode': 'fixed',
        'seeds': '1',
        'n_values': ','.join(str(p) for p in PREFIX_COUNTS),
        'gamma': '0', 'max_iter': 1, 'dpsize': 'actual-{}'.format(dp),
        'artifacts_figs': '{}_artifacts/figs'.format(args.store),
        'env': {**base_env(args), 'SCULPTOR_EODS_MODE': 'prefixes'},
    } for dp in keep]


def mode_obj32(args):
    specs = []
    for fam, fcfg in OBJ32_FAMILIES.items():
        for rung_label, (rung, pmode) in OBJ32_RUNGS.items():
            specs.append({
                'label': 'obj32_{}_{}'.format(fam, rung_label),
                'out_root': '{}/obj32/{}/{}'.format(
                    args.store, fam, rung_label),
                'rungs': rung, 'probe_mode': pmode,
                'seeds': '1' if rung == 'painter' else args.seeds,
                'n_values': args.n_values,
                'gamma': fcfg['gamma'], 'max_iter': 500,
                'dpsize': 'actual-32',
                'artifacts_figs': '{}_artifacts/figs'.format(args.store),
                'env': {**OBJ32_COMMON_ENV, **fcfg['env'],
                        'SCULPTOR_LP_BACKEND': args.backend},
            })
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('sizes', 'prefixes', 'obj32'),
                    default='sizes')
    ap.add_argument('--out', required=True)
    ap.add_argument('--store', default='cache/eods/v1')
    ap.add_argument('--soln-types', default='painter')
    ap.add_argument('--backend', default='highs')
    ap.add_argument('--dpsizes', default=None)
    ap.add_argument('--seeds', default='1-3')
    ap.add_argument('--n-values', default='10')
    ap.add_argument('--train-iters', type=int, default=200)
    ap.add_argument('--nsim', type=int, default=None,
                    help='override sims per dpsize (sizes mode)')
    ap.add_argument('--env', action='append', default=[],
                    help='extra per-cell env KEY=VAL (repeatable)')
    args = ap.parse_args()

    specs = {'sizes': mode_sizes, 'prefixes': mode_prefixes,
             'obj32': mode_obj32}[args.mode](args)
    with open(args.out, 'w') as f:
        json.dump(specs, f, indent=1)

    def n_cells(s):
        seeds = s['seeds']
        ns = (len(range(int(seeds.split('-')[0]),
                        int(seeds.split('-')[1]) + 1))
              if '-' in seeds else len(seeds.split(',')))
        return ns * len(str(s['n_values']).split(','))
    print('wrote {} [{}]: {} specs, {} cells'.format(
        args.out, args.mode, len(specs), sum(n_cells(s) for s in specs)))


if __name__ == '__main__':
    main()
