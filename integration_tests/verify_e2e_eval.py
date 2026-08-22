#!/usr/bin/env python
"""End-to-end verification that `evaluate_all_metrics` actually runs.

    python integration_tests/verify_e2e_eval.py            # small@100, actual-5@3
    python integration_tests/verify_e2e_eval.py --quick    # both at 3 iters
    python integration_tests/verify_e2e_eval.py --only small

The baseline case: run the eval driver at two deployment sizes and check the
metrics pickle it produced -- all six strategies solved, failed_strategies
empty, per-UG latency vectors populated and finite.

Why it does not just check the exit code, and the workspace/Ray hygiene the
harness needs, are documented in _common.py.

actual-5 defaults to 3 iters: it measured 2831s at THREE iterations on a
laptop, so 50 is a cluster setting, not a laptop one.
"""
import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

import numpy as np  # noqa: E402

# These verify the pipeline RUNS, not that it converges. small@100 took 220s
# and actual-5 measured 2831s at THREE iterations, so the defaults are low on
# purpose -- pass --iters when you actually want convergence.
CASES = [
    ('small',    'small',    5),
    ('actual-5', 'actual-5', 3),
]


def run_case(root, label, dpsize, iters):
    _LABEL = label.replace('-', '_')
    res = C.Result('{} @ {} iters'.format(label, iters))
    ws = C.workspace(root, label.replace('-', '_'))
    log = os.path.join(root, 'eval_{}.log'.format(label.replace('-', '_')))
    t0 = time.time()
    rc = C.run([sys.executable, '-u', C.driver('eval_all_solution_types.py'),
                '--dpsize', dpsize], ws, C.env_for(iters, label=_LABEL), log)
    res.wall_s = time.time() - t0
    res.check(rc == 0, 'exit code 0', 'rc={}'.format(rc))
    C.scan_log(log, res)

    pkl = os.path.join(ws, 'cache',
                       'popp_failure_latency_comparison_{}.pkl'.format(dpsize))
    d = C.check_metrics(pkl, res, started_at=t0)
    if d:
        lats = (d.get('latencies') or {}).get(0) or {}
        bad = []
        for s in sorted(C.EXPECTED_STRATEGIES):
            v = lats.get(s)
            if v is None or len(v) == 0:
                bad.append('{}=empty'.format(s)); continue
            if not np.isfinite(np.asarray(v, dtype=float)).all():
                bad.append('{}=non-finite'.format(s))
        res.check(not bad, 'per-UG latencies populated and finite',
                  '; '.join(bad) if bad else
                  'n={} per strategy'.format(len(lats.get('sparse', []))))
    C.collect(ws, res, label.replace('-', '_'), [pkl])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='3 iters per case')
    ap.add_argument('--iters', type=int, default=None)
    ap.add_argument('--only', default=None, help='run just this case label')
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()

    cases = [c for c in CASES if a.only in (None, c[0])]
    if not cases:
        sys.exit('no case matches --only {!r} (have: {})'.format(
            a.only, ', '.join(c[0] for c in CASES)))

    print('=' * 74 + '\ne2e: evaluate_all_metrics\n' + '=' * 74)
    if not C.preflight():
        return 2
    root = tempfile.mkdtemp(prefix='verify_e2e_eval_')
    print('cases       : {}\nscratch     : {}'.format(
        ', '.join(c[0] for c in cases), root))
    print('=' * 74)

    results = []
    for i, (label, dpsize, default_iters) in enumerate(cases):
        if i:
            time.sleep(5)   # let the previous Ray cluster release
        iters = 3 if a.quick else (a.iters or default_iters)
        print('\n[{}/{}] {} @ {} iters'.format(i + 1, len(cases), label, iters),
              flush=True)
        r = run_case(root, label, dpsize, iters)
        results.append(r)
        print('  -> {} in {:.0f}s'.format('PASS' if r.passed else 'FAIL', r.wall_s),
              flush=True)
    return C.finish(results, root, a.keep)


if __name__ == '__main__':
    sys.exit(main())
