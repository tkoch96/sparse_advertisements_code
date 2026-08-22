#!/usr/bin/env python
"""(b) Sweep over deployment sizes -- 3, 4, 5 and 6 PoPs.

    python integration_tests/verify_e2e_dpsizes.py
    python integration_tests/verify_e2e_dpsizes.py --quick
    python integration_tests/verify_e2e_dpsizes.py --dpsizes 3,5,10

Drives the real `evaluations/evaluate_over_deployment_sizes.py` with --plot, so
`make_paper_plots` produces the same figures the paper sweep does.

That driver used to hardcode its sizes, nsim and port, so the only way to run a
small sweep was to edit the file. It now takes --dpsizes / --nsim / --max-iter /
--cache-fn; omitting them reproduces the previous behaviour exactly.

Uses --cache-fn to namespace the results so a test run can never overwrite the
real `testing_feature_cache_fn.pkl`.
"""
import argparse
import os
import pickle
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

_LABEL = 'dpsizes'   # figures/integration_tests/<_LABEL>/

# These verify the pipeline RUNS, not that it converges -- keep them in
# minutes. Override with --iters when you actually want convergence.
DEFAULT_ITERS = 5

DEFAULT_DPSIZES = [3, 4, 5, 6]


def run_case(root, iters, dpsizes):
    res = C.Result('dpsizes   (PoPs={})'.format(dpsizes))
    ws = C.workspace(root, 'dpsizes')
    log = os.path.join(root, 'dpsizes.log')
    # namespaced so a test never clobbers the real sweep cache
    cache_fn = os.path.join(ws, 'cache', 'e2e_dpsize_sweep.pkl')
    t0 = time.time()
    rc = C.run([sys.executable, '-u',
                C.driver('evaluate_over_deployment_sizes.py'),
                '--dpsizes', ','.join(str(s) for s in dpsizes),
                '--nsim', '1', '--max-iter', str(iters),
                '--cache-fn', cache_fn, '--plot'],
               ws, C.env_for(iters, label=_LABEL), log)
    res.wall_s = time.time() - t0
    res.check(rc == 0, 'exit code 0', 'rc={}'.format(rc))
    C.scan_log(log, res)

    if res.check(os.path.exists(cache_fn), 'sweep cache written', cache_fn):
        d = pickle.load(open(cache_fn, 'rb'))
        res.check(set(d) >= set(dpsizes), 'all sizes evaluated',
                  'have {}'.format(sorted(d)))
        empty = [s for s in d if not d[s]]
        res.check(not empty, 'every size has stats',
                  'empty: {}'.format(empty) if empty else
                  '{} sizes'.format(len(d)))
    C.collect(ws, res, 'dpsizes', [cache_fn])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dpsizes', default=None, help='comma list, e.g. 3,4,5,6')
    ap.add_argument('--quick', action='store_true', help='3 iters, 2 sizes')
    ap.add_argument('--iters', type=int, default=None)
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()

    print('=' * 74 + '\ne2e: evaluate over deployment sizes\n' + '=' * 74)
    if not C.preflight():
        return 2
    dpsizes = ([int(x) for x in a.dpsizes.split(',')] if a.dpsizes
               else (DEFAULT_DPSIZES[:2] if a.quick else DEFAULT_DPSIZES))
    iters = 3 if a.quick else (a.iters or DEFAULT_ITERS)
    root = tempfile.mkdtemp(prefix='verify_e2e_dpsizes_')
    print('dpsizes     : {}\niters       : {}\nscratch     : {}'.format(
        dpsizes, iters, root))
    print('=' * 74)
    res = run_case(root, iters, dpsizes)
    print('  -> {} in {:.0f}s'.format('PASS' if res.passed else 'FAIL', res.wall_s))
    return C.finish([res], root, a.keep)


if __name__ == '__main__':
    sys.exit(main())
