"""e2e: the measurement budget binds when driven from the real sweep CLI.

    python integration_tests/verify_e2e_probe_budget.py
    python integration_tests/verify_e2e_probe_budget.py --budgets 2,5,10

Runs `evaluations/evaluate_over_deployment_sizes.py --probe-n N` as a
SUBPROCESS -- the same entry point a cluster run uses -- and judges it on
what the log says, not on the exit code.

What it checks, per budget N:

  1. the sweep announces the policy it is actually running
        [sweep] probing: mode=smart budget=N
  2. every solve() reports its budget on exit
        [probe-budget] EXITING on M path measures
            (= 1 setup grounding + P probes) | budget N=... mode=... skipped=...
  3. P <= N                 -- the budget BINDS (the hard invariant)
  4. M == P + 1             -- the setup grounding is the only measurement
                               outside the budget
  5. any shortfall is EXPLAINED -- every probe that spent nothing logged a
     reason, so `skipped` accounts for the gap. A budget is an upper bound,
     not a target: once the advertisement stops moving, grounding at an
     unchanged point is a no-op and the run legitimately underspends.

(5) is the one that took a bug to learn. `_probe_ground_current` used to
return False silently when `_solve_post_step_measure` found nothing to
measure, so ~2.5 probes per run vanished with no log line and the budget
merely looked leaky. Unexplained shortfall is now a FAILURE here.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

EXIT_RE = re.compile(
    r'\[probe-budget\] EXITING on (\d+) path measures '
    r'\(= (\d+) setup grounding \+ (\d+) probes\) \| budget N=(-?\d+) '
    r'mode=(\S+) skipped=(\d+) iters=(\d+)')
ANNOUNCE_RE = re.compile(r'\[sweep\] probing: mode=(\S+) budget=(\S+)')
SKIP_RE = re.compile(r'\[probe-gate\] iter=\d+ probe SKIPPED')


def run_case(budget, dpsize, iters, port, keep=False):
    ws = tempfile.mkdtemp(prefix='e2e_probe_budget_')
    log_fn = os.path.join(ws, 'run.log')
    env = dict(os.environ)
    env.update({
        'PYTHONUNBUFFERED': '1',
        'SCULPTOR_N_WORKERS': os.environ.get('PROBE_E2E_WORKERS', '4'),
        'MPLBACKEND': 'Agg',
    })
    # the CLI must be what sets the budget -- that is the wiring under test
    for k in ('SCULPTOR_PROBE_N', 'SCULPTOR_PROBE_MODE'):
        env.pop(k, None)
    # NAMESPACE THE RESULT CACHE. wrapper_eval caches per (dpsize, run tag);
    # without a unique tag the second budget loads the first one's pickle,
    # skips training entirely, and the test passes or fails on a run that
    # never solved anything. Cost 2026-08-21: both cases of the first run
    # were cache hits and produced no probe evidence at all.
    env['SCULPTOR_RUN_TAG'] = 'e2eprobe_{}_{}'.format(budget, int(time.time()))
    argv = [sys.executable, '-u',
            os.path.join(_REPO, 'evaluations',
                         'evaluate_over_deployment_sizes.py'),
            '--dpsizes', str(dpsize), '--nsim', '1',
            '--max-iter', str(iters), '--port', str(port),
            '--probe-n', str(budget),
            '--cache-fn', os.path.join(ws, 'metrics.pkl'),
            '--figures-subdir', 'integration_tests/probe_budget']
    t0 = time.time()
    with open(log_fn, 'w') as fh:
        rc = subprocess.call(argv, cwd=_REPO, env=env, stdout=fh,
                             stderr=subprocess.STDOUT)
    wall = time.time() - t0
    text = open(log_fn, errors='replace').read()
    return {'budget': budget, 'rc': rc, 'wall': wall, 'log': log_fn,
            'text': text, 'ws': ws}


def judge(case):
    """-> (ok, lines). Judged on the LOG, never on rc."""
    out, ok = [], True
    b, text = case['budget'], case['text']

    ann = ANNOUNCE_RE.search(text)
    if not ann:
        out.append('    FAIL: sweep never announced its probing policy')
        ok = False
    else:
        out.append('    announce: mode={} budget={}'.format(*ann.groups()))
        if ann.group(2) != str(b):
            out.append('    FAIL: announced budget {} != requested {}'
                       .format(ann.group(2), b))
            ok = False

    # A cached size never solves, so every downstream check is vacuous.
    # Name that explicitly rather than reporting "no evidence line".
    if 'CACHE HIT' in text:
        out.append('    FAIL: the sweep hit a RESULT CACHE -- no solve ran, '
                   'so this case proves nothing about the budget')
        return False, out
    if 'LEARNING ITERATION' not in text:
        out.append('    FAIL: no LEARNING ITERATION in the log -- training '
                   'never started, so the budget was never exercised')
        return False, out

    exits = EXIT_RE.findall(text)
    if not exits:
        out.append('    FAIL: no [probe-budget] EXITING line -- the budget '
                   'left no evidence it was in force')
        return False, out

    for m, setup, probes, n, mode, skipped, iters in exits:
        m, setup, probes = int(m), int(setup), int(probes)
        n, skipped = int(n), int(skipped)
        out.append('    EXITING on {} path measures (= {} setup + {} probes)'
                   ' | N={} mode={} skipped={} iters={}'.format(
                       m, setup, probes, n, mode, skipped, iters))
        if n != b:
            out.append('      FAIL: solver budget {} != requested {}'.format(n, b))
            ok = False
        if probes > b:
            out.append('      FAIL: budget EXCEEDED ({} > {})'.format(probes, b))
            ok = False
        if m != probes + 1:
            out.append('      FAIL: {} measures != {} probes + 1 setup '
                       'grounding'.format(m, probes))
            ok = False
        if probes < b and skipped == 0:
            out.append('      FAIL: underspent ({}/{}) with ZERO logged '
                       'skips -- the shortfall is unexplained'.format(
                           probes, b))
            ok = False
        elif probes < b:
            out.append('      underspend explained: {} probe(s) found '
                       'nothing to measure'.format(skipped))
    return ok, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--budgets', default='3,5')
    ap.add_argument('--dpsize', type=int, default=3)
    ap.add_argument('--iters', type=int, default=25)
    ap.add_argument('--port', type=int, default=32600)
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(',')]

    print('=' * 74)
    print('e2e: measurement budget via evaluate_over_deployment_sizes '
          '--probe-n')
    print('=' * 74)
    print('interpreter : {}'.format(sys.executable))
    print('dpsize      : actual-{}   iters: {}'.format(a.dpsize, a.iters))
    print('budgets     : {}'.format(budgets))
    print('=' * 74)

    all_ok = True
    port = a.port
    for i, b in enumerate(budgets, 1):
        port += 1
        print('\n[{}/{}] budget N={}'.format(i, len(budgets), b), flush=True)
        case = run_case(b, a.dpsize, a.iters, port, a.keep)
        ok, lines = judge(case)
        for l in lines:
            print(l)
        print('    rc={} wall={:.0f}s  log -> {}'.format(
            case['rc'], case['wall'], case['log']))
        print('    -> {}'.format('PASS' if ok else 'FAIL'))
        all_ok = all_ok and ok

    print('\n' + '-' * 74)
    print('ALL {} CASE(S) {}'.format(len(budgets),
                                     'PASSED' if all_ok else 'FAILED'))
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
