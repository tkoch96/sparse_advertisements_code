#!/usr/bin/env python
"""(a) Sweep over prefix budgets at dpsize=small.

    python integration_tests/verify_e2e_prefixes.py
    python integration_tests/verify_e2e_prefixes.py --quick
    python integration_tests/verify_e2e_prefixes.py --budgets 1,2,4,8

Drives the real `evaluations/evaluate_over_n_prefixes.py` (with --plot, so the
paper plots come out of `make_paper_plots` like any other sweep) rather than
reimplementing its loop -- those plots are the point of the evaluation.

The prefix sweep reuses a deployment cached by an ordinary eval run, so the
case seeds one first with `eval_all_solution_types.py`. That mirrors the classic
flow, which assumes the cache is already populated.

Judged on the cache the sweep produced: every requested budget present, and
every one carrying stats. See _common.py for why the exit code isn't enough.
"""
import argparse
import os
import pickle
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

_LABEL = 'prefixes'   # figures/integration_tests/<_LABEL>/

# These verify the pipeline RUNS, not that it converges -- keep them in
# minutes. Override with --iters when you actually want convergence.
DEFAULT_ITERS = 5

# Bracket small's natural budget, which is 6 (helpers.deployment_to_prefixes).
# The floor is not arbitrary: init_advertisement gives prefix 0 to anycast and
# one prefix per PoP, so it needs n_prefixes >= n_pops + 1. `small` has 3 PoPs,
# so budgets below 4 make sparse fail -- as a swallowed IndexError before
# 2026-08-21, and as a clear ValueError since.
DEFAULT_BUDGETS = [4, 6, 8, 10]


def run_case(root, iters, budgets, dpsize='small'):
    res = C.Result('prefixes  ({}, budgets={})'.format(dpsize, budgets))
    ws = C.workspace(root, 'prefixes')
    t0 = time.time()

    seed_log = os.path.join(root, 'prefixes_seed.log')
    rc = C.run([sys.executable, '-u', C.driver('eval_all_solution_types.py'),
                '--dpsize', dpsize],
               ws, C.env_for(iters, label=_LABEL), seed_log)
    if not res.check(rc == 0, 'seed deployment built', 'rc={}'.format(rc)):
        C.scan_log(seed_log, res)
        res.wall_s = time.time() - t0
        return res

    log = os.path.join(root, 'prefixes.log')
    rc = C.run([sys.executable, '-u', C.driver('evaluate_over_n_prefixes.py'),
                '--dpsize', dpsize,
                '--prefixes', ','.join(str(b) for b in budgets),
                '--nsim', '1', '--max-iter', str(iters), '--plot'],
               ws, C.env_for(iters, label=_LABEL), log)
    res.wall_s = time.time() - t0
    res.check(rc == 0, 'exit code 0', 'rc={}'.format(rc))
    C.scan_log(log, res)

    cache_fn = os.path.join(
        ws, 'cache', 'evaluate_over_prefix_numbers_cache_fn_{}.pkl'.format(dpsize))
    if res.check(os.path.exists(cache_fn), 'prefix cache written', cache_fn):
        d = pickle.load(open(cache_fn, 'rb'))
        res.check(set(d) >= set(budgets), 'all budgets evaluated',
                  'have {}'.format(sorted(d)))
        empty = [b for b in d if not d[b]]
        res.check(not empty, 'every budget has stats',
                  'empty: {}'.format(empty) if empty else
                  '{} budgets'.format(len(d)))
    C.collect(ws, res, 'prefixes', [cache_fn])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--budgets', default=None, help='comma list, e.g. 1,2,3,4')
    ap.add_argument('--dpsize', default='small')
    ap.add_argument('--quick', action='store_true', help='3 iters, 2 budgets')
    ap.add_argument('--iters', type=int, default=None)
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()

    print('=' * 74 + '\ne2e: evaluate over prefix budgets\n' + '=' * 74)
    if not C.preflight():
        return 2
    budgets = ([int(x) for x in a.budgets.split(',')] if a.budgets
               else (DEFAULT_BUDGETS[:2] if a.quick else DEFAULT_BUDGETS))
    iters = 3 if a.quick else (a.iters or DEFAULT_ITERS)
    root = tempfile.mkdtemp(prefix='verify_e2e_prefixes_')
    print('budgets     : {}\niters       : {}\nscratch     : {}'.format(
        budgets, iters, root))
    print('=' * 74)
    res = run_case(root, iters, budgets, a.dpsize)
    print('  -> {} in {:.0f}s'.format('PASS' if res.passed else 'FAIL', res.wall_s))
    return C.finish([res], root, a.keep)


if __name__ == '__main__':
    sys.exit(main())
