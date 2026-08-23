#!/usr/bin/env python
"""(c) The paper table, end to end -- solve every objective, emit the table.

    python integration_tests/verify_e2e_objectives.py            # demo1 config
    python integration_tests/verify_e2e_objectives.py --quick    # 5 iters
    python integration_tests/verify_e2e_objectives.py --iters 30

Runs the canonical `generate_paper_table.py --dpsize small
--number_of_deployments 1 --num_training_iter 10 --run_id e2etable` (the
"paper_table_demo1" smoke of 2026-08-23) in a throwaway workspace: one
training+eval cell per objective (avg_latency, per_site_cost, max_util,
frac_beyond_optimal, joint_priority), metrics recomputed, table emitted.
Judged on artifacts, never on rc (the bare-except trap, see _common). The
finished table is printed so a green run ends with the thing you wanted to
see.

This file previously ran one bare evaluation per objective; the table driver
subsumes that (same cells, plus aggregation + emit), so the old test lives on
inside this one.
"""
import argparse
import csv
import os
import re
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
sys.path.insert(0, C.REPO)   # for evaluations.* imports below

_LABEL = 'paper_table'   # figures/integration_tests/<_LABEL>/

DEFAULT_ITERS = 3       # the demo1 config: small, 1 deployment, 10 iters
RUN_ID = 'e2etable'

OBJECTIVES = ['avg_latency', 'per_site_cost', 'max_util',
              'frac_beyond_optimal', 'joint_priority']
METHODS = ['One-per-peering', 'SCULPTOR', 'PAINTER', 'AnyOpt',
           'Anycast', 'Unicast']


def run_case(root, iters, ndeps, dpsize):
    res = C.Result('paper table ({}, {} deployment(s), {} iters)'.format(
        dpsize, ndeps, iters))
    ws = C.workspace(root, 'paper_table')
    log = os.path.join(root, 'paper_table.log')
    t0 = time.time()
    table_out = os.path.join(ws, 'figures', 'paper_table')
    rc = C.run([sys.executable, '-u',
                os.path.join(C.REPO, 'evaluations', 'generate_paper_table.py'),
                '--dpsize', dpsize,
                '--number_of_deployments', str(ndeps),
                '--num_training_iter', str(iters),
                '--run_id', RUN_ID,
                '--out', table_out],
               ws, C.env_for(iters, label=_LABEL), log)
    res.wall_s = time.time() - t0
    res.check(rc == 0, 'exit 0', 'rc={}'.format(rc))
    text = C.scan_log(log, res)

    # a cell that ATTEMPTED (printed "running") must finish; a cell the
    # cache layers skipped is legitimately absent from the log -- that is
    # the ~seconds cached path, the whole point of the pickle layers
    attempted = {obj: bool(re.search(r'\[{}\] running'.format(re.escape(obj)),
                                     text)) for obj in OBJECTIVES}
    for obj in OBJECTIVES:
        if attempted[obj]:
            res.check(re.search(r'\[{}\] DONE'.format(re.escape(obj)), text),
                      '{}: cell DONE'.format(obj))
        else:
            res.check(True, '{}: cached (cell not needed)'.format(obj))

    # per-objective metrics pickles, judged on content (rc=0 is not success)
    from evaluations.generate_paper_table import dpsize_str
    dp_str = dpsize_str(dpsize)
    for obj in OBJECTIVES:
        suffix = '' if obj == 'avg_latency' else '_' + obj
        # the driver is REPO-anchored; RUN_ID namespaces these so they
        # cannot collide with real results
        pkl = os.path.join(C.REPO, 'cache',
                           'popp_failure_latency_comparison_{}_{}{}.pkl'.format(
                               dp_str, RUN_ID, suffix))
        C.check_metrics(pkl, res, prefix='{}: '.format(obj),
                        started_at=t0 if attempted[obj] else None)

    # the table itself
    tex = os.path.join(table_out, 'paper_table.tex')
    csv_fn = os.path.join(table_out, 'paper_table.csv')
    res.check(os.path.exists(tex), 'paper_table.tex written')
    if res.check(os.path.exists(csv_fn), 'paper_table.csv written'):
        rows = [r for r in csv.reader(open(csv_fn))]
        names = [r[0] for r in rows]
        res.check('DIRECTION' in names, 'CSV carries the DIRECTION row')
        missing = [m for m in METHODS if m not in names]
        res.check(not missing, 'all 6 methods in table',
                  'missing {}'.format(missing) if missing else '')
        sr = next((r for r in rows if r and r[0] == 'SCULPTOR'), None)
        filled = sum(1 for v in (sr or [])[1:] if v not in ('', '-'))
        res.check(sr is not None and filled >= 20,
                  'SCULPTOR row substantially populated',
                  '{} filled cells'.format(filled))

    C.collect(ws, res, _LABEL, [tex, csv_fn])

    # a green run should end with the table on screen
    m = re.search(r'\n( +\|.+\n(?:.+\n)+?)\n *wrote ', text)
    if m:
        print('\n' + m.group(1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dpsize', default='small')
    ap.add_argument('--ndeps', type=int, default=1)
    ap.add_argument('--iters', type=int, default=None)
    ap.add_argument('--quick', action='store_true', help='5 iters')
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()

    print('=' * 74 + '\ne2e: the paper table (all objectives -> one table)\n' + '=' * 74)
    if not C.preflight():
        return 2
    iters = 5 if a.quick else (a.iters or DEFAULT_ITERS)
    root = tempfile.mkdtemp(prefix='verify_e2e_paper_table_')
    print('dpsize      : {}\ndeployments : {}\niters       : {}\nscratch     : {}'.format(
        a.dpsize, a.ndeps, iters, root))
    print('=' * 74)
    res = run_case(root, iters, a.ndeps, a.dpsize)
    print('  -> {} in {:.0f}s'.format('PASS' if res.passed else 'FAIL', res.wall_s))
    return C.finish([res], root, a.keep)


if __name__ == '__main__':
    sys.exit(main())
