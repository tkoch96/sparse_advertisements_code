#!/usr/bin/env python
"""Ablation grid over objectives x measurement budgets x deployments x
ladder rungs (Tom 2026-08-26: 'add in a third dimension -- the other
objectives').

    python run_ablation_grid.py \
        --number_measurements_allowed [5,10,15,20] \
        --deployments 3 --num_iters 200 --objectives all --dpsize small

Thin manifest generator over experiments/ablation/run_n_sweep_queue.py
(manifest mode): one spec per objective; the queue supplies the global
slot pool, RAM governor, deployment-major ordering, stagger, audit and
rescore. Nothing here duplicates queue logic.
"""
import argparse
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
ALL_OBJECTIVES = ['avg_latency', 'per_site_cost', 'max_util',
                  'frac_beyond_optimal', 'joint_priority']
# active ladder rungs + painter baseline (expl_random retired 2026-08-12)
DEFAULT_RUNGS = ('full,expl_none,no_direction,no_memory_dir,'
                 'no_memory,no_mc,painter')


def _parse_list(v):
    return [x for x in v.strip('[] ').replace(' ', '').split(',') if x]


def _ensure_inits(seeds, dpsize, init_dir, dry=False):
    import numpy as np
    os.makedirs(init_dir, exist_ok=True)
    todo = [s for s in seeds if not os.path.exists(
        os.path.join(init_dir, 'init_dep{}.npy'.format(s)))]
    if not todo or dry:
        if todo:
            print('[grid] (dry) would build inits for seeds', todo)
        return
    os.environ.setdefault('MPLBACKEND', 'Agg')
    sys.path.insert(0, _REPO)
    from core.deployment_setup import get_random_deployment
    from core.sparse_advertisements_v3 import Sparse_Advertisement_Eval
    from helpers.helpers import deployment_to_prefixes
    for s in todo:
        os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(s)
        dep = get_random_deployment(dpsize)
        sas = Sparse_Advertisement_Eval(
            dep, verbose=False, lambduh=0,
            using_resilience_benefit=False, gamma=0,
            n_prefixes=deployment_to_prefixes(dep))
        a0 = sas.init_advertisement()
        np.save(os.path.join(init_dir, 'init_dep{}.npy'.format(s)), a0)
        print('[grid] canonical init seed {} -> init_dep{}.npy '
              '(shape {})'.format(s, s, a0.shape))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--number_measurements_allowed', required=True,
                    help="comma list, brackets ok: '[5,10,15,20]'")
    ap.add_argument('--deployments', type=int, default=3,
                    help='number of deployments (seeds 1..K)')
    ap.add_argument('--num_iters', type=int, default=200)
    ap.add_argument('--objectives', default='all',
                    help="'all' or comma list of {}".format(ALL_OBJECTIVES))
    ap.add_argument('--dpsize', default='small')
    ap.add_argument('--rungs', default=DEFAULT_RUNGS)
    ap.add_argument('--out-root', default=None,
                    help='grid store root (default cache/ablation/'
                         'grid_objdim under the repo)')
    ap.add_argument('--ws-root', default=None)
    ap.add_argument('--slots', type=int, default=4,
                    help='parallel cells (Mac default 4; VM: raise)')
    ap.add_argument('--probe-mode', default='gated')
    ap.add_argument('--gamma', default='0.1',
                    help='gamma for avg_latency (others force 0 via '
                         'objective policy)')
    ap.add_argument('--launch-stagger', type=float, default=None)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    n_values = ','.join(_parse_list(a.number_measurements_allowed))
    objectives = (ALL_OBJECTIVES if a.objectives.strip() in ('all', '')
                  else _parse_list(a.objectives))
    bad = [o for o in objectives if o not in ALL_OBJECTIVES]
    if bad:
        raise SystemExit('unknown objectives: {}'.format(bad))
    out_root = a.out_root or os.path.join(
        _REPO, 'cache', 'ablation', 'grid_objdim')
    ws_root = a.ws_root or os.path.join(out_root + '_ws')
    os.makedirs(ws_root, exist_ok=True)

    # Canonical per-seed inits, shared across EVERY objective and budget
    # (the queue's ablation-fairness contract, extended to the objective
    # dimension): pre-generate deterministically so parallel first cells
    # never race to write them.
    init_dir = os.path.join(ws_root, 'inits')
    _ensure_inits(range(1, a.deployments + 1), a.dpsize, init_dir,
                  dry=a.dry_run)

    manifest = []
    for obj in objectives:
        manifest.append({
            'label': obj,
            'out_root': os.path.join(out_root, obj),
            'probe_mode': a.probe_mode,
            'rungs': a.rungs,
            'seeds': '1-{}'.format(a.deployments),
            'n_values': n_values,
            'gamma': a.gamma if obj == 'avg_latency' else '0',
            'max_iter': a.num_iters,
            'dpsize': a.dpsize,
            'init_src': init_dir,
            'env': {'SCULPTOR_XOBJS': '1',
                    'SCULPTOR_ABLATION_OBJECTIVE': obj},
            'artifacts_figs': os.path.join(
                out_root + '_artifacts', 'figs'),
        })
    mf = os.path.join(ws_root, 'grid_manifest.json')
    with open(mf, 'w') as fh:
        json.dump(manifest, fh, indent=1)
    n_cells = (len(objectives) * len(n_values.split(',')) * a.deployments
               * len(a.rungs.split(',')))
    print('[grid] {} objectives x {} budgets x {} deployments x {} rungs '
          '= {} cells; manifest {}'.format(
              len(objectives), len(n_values.split(',')), a.deployments,
              len(a.rungs.split(',')), n_cells, mf))
    argv = [sys.executable, '-u', '-m',
            'experiments.ablation.run_n_sweep_queue',
            '--manifest', mf, '--ws-root', ws_root,
            '--slots', str(a.slots), '--max-iter', str(a.num_iters),
            '--dpsize', a.dpsize]
    if a.launch_stagger is not None:
        argv += ['--launch-stagger', str(a.launch_stagger)]
    print('[grid] exec:', ' '.join(argv))
    if a.dry_run:
        return 0
    env = dict(os.environ)
    env.setdefault('PYTHONPATH', _REPO)
    env.setdefault('MPLBACKEND', 'Agg')
    return subprocess.call(argv, cwd=_REPO, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
