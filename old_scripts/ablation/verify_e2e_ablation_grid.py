"""(g) Ablation grid: objectives x measurement budgets x deployments x
ladder rungs, through run_ablation_grid.py (Tom 2026-08-26).

    python integration_tests/verify_e2e_ablation_grid.py            # quick lattice
    python integration_tests/verify_e2e_ablation_grid.py --full     # all objectives
    python integration_tests/verify_e2e_ablation_grid.py --iters 10

Runs the real grid driver end to end in a throwaway store and asserts a
result JSON exists for EVERY lattice point with n_iters > 0 (painter is
exempt: baseline, no training loop).
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import integration_tests._common as C  # noqa: E402

_LABEL = 'ablation_grid'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--iters', type=int, default=8)
    ap.add_argument('--full', action='store_true',
                    help='all 5 objectives (default: 2)')
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()
    if not C.preflight():
        return 2
    objectives = ('all' if a.full else 'avg_latency,max_util')
    n_objs = 5 if a.full else 2
    budgets, seeds, rungs = '5,10', 1, 'full,no_mc,painter'
    root = tempfile.mkdtemp(prefix='verify_e2e_ablgrid_')
    out_root = os.path.join(root, 'store')
    log = os.path.join(root, 'grid.log')
    print('=' * 74 + '\ne2e: ablation grid (objectives dimension)\n' + '=' * 74)
    print('objectives : {}\nbudgets    : {}\nseeds      : {}\nrungs      : {}'
          '\niters      : {}\nscratch    : {}'.format(
              objectives, budgets, seeds, rungs, a.iters, root))
    res = C.Result('ablation_grid')
    t0 = time.time()
    rc = C.run([sys.executable, '-u',
                os.path.join(C.REPO, 'run_ablation_grid.py'),
                '--number_measurements_allowed', budgets,
                '--deployments', str(seeds),
                '--num_iters', str(a.iters),
                '--objectives', objectives,
                '--dpsize', 'small',
                '--rungs', rungs,
                '--out-root', out_root,
                '--slots', '3', '--launch-stagger', '2'],
               C.REPO, C.env_for(a.iters, label=_LABEL), log)
    res.wall_s = time.time() - t0
    res.check(rc == 0, 'exit code 0', 'rc={}'.format(rc))
    C.scan_log(log, res)

    objs = (['avg_latency', 'per_site_cost', 'max_util',
             'frac_beyond_optimal', 'joint_priority'] if a.full
            else ['avg_latency', 'max_util'])
    missing, zero_iter = [], []
    for obj in objs:
        for N in budgets.split(','):
            for s in range(1, seeds + 1):
                for rung in rungs.split(','):
                    fn = os.path.join(out_root, obj, 'N{}'.format(N),
                                      'seed_{}_{}.json'.format(s, rung))
                    if not os.path.exists(fn):
                        missing.append(os.path.relpath(fn, out_root))
                        continue
                    d = json.load(open(fn))
                    if rung != 'painter' and not d.get('n_iters'):
                        zero_iter.append(os.path.relpath(fn, out_root))
    n_pts = len(objs) * 2 * seeds * len(rungs.split(','))
    res.check(not missing, 'result JSON at every lattice point',
              'missing {}: {}'.format(len(missing), missing[:4])
              if missing else '{} points'.format(n_pts))
    res.check(not zero_iter, 'every trained cell has n_iters > 0',
              'zero-iter: {}'.format(zero_iter[:4]) if zero_iter
              else 'all trained')
    print('-' * 74)
    res.report()
    ok = res.passed
    if not a.keep and ok:
        shutil.rmtree(root, ignore_errors=True)
    else:
        print('scratch kept: {}'.format(root))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
