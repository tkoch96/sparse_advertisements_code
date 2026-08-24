#!/usr/bin/env python
"""(d) Hot-start: kill a small solve mid-training, resume from state-N.

    python integration_tests/verify_e2e_hotstart.py
    python integration_tests/verify_e2e_hotstart.py --kill-at 30

Phase 1 launches evaluate_over_deployment_sizes on 'small' (1 sim) and
KILLS the driver as soon as runs/<ts>-small-sparse/state-<kill_at>.pkl
appears. Phase 2 relaunches with SCULPTOR_HOTSTART_RUN_DIR pointing at
that dir and must (a) print 'Loading from hotstart dir', (b) resume at an
iteration >= kill_at rather than 0, (c) finish with the ALL DONE banner
and a content-checked metrics pickle.

Exists because hot-start silently broke twice on 2026-08-24 (a list-form
save_run_dir nested to [[dir]]; the join died inside the bare except and
the run 'passed' having solved nothing). This is the regression fence.
"""
import argparse
import glob
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

_LABEL = 'hotstart'
DRIVER = os.path.join(C.REPO, 'evaluations', 'evaluate_over_deployment_sizes.py')


def _spawn(ws, env, log_fn):
    lf = open(log_fn, 'w')
    return subprocess.Popen(
        [sys.executable, '-u', DRIVER, '--dpsizes', 'small', '--nsim', '1',
         '--max-iter', '60', '--cache-fn',
         os.path.join(ws, 'cache', 'hs_e2e.pkl'),
         '--figures-subdir', 'integration_tests/hotstart'],
        cwd=ws, env=env, stdout=lf, stderr=subprocess.STDOUT)


def run_case(root, kill_at):
    res = C.Result('hot-start (small, kill at state-{})'.format(kill_at))
    ws = C.workspace(root, 'hotstart')
    env = C.env_for(60, {'SCULPTOR_RUN_TAG': 'hs_e2e',
                         'SCULPTOR_EVAL_SEED': '777'}, label=_LABEL)
    t0 = time.time()

    # ---- phase 1: train, kill at the target checkpoint -------------------
    log1 = os.path.join(root, 'phase1.log')
    proc = _spawn(ws, env, log1)
    target = None
    deadline = time.time() + 45 * 60
    while time.time() < deadline and proc.poll() is None:
        for d in glob.glob(os.path.join(ws, 'runs', '*-small-sparse')):
            if os.path.exists(os.path.join(d, 'state-%d.pkl' % kill_at)):
                target = os.path.basename(d)
                break
        if target:
            break
        time.sleep(15)
    if not res.check(target is not None,
                     'phase 1 reached state-{}'.format(kill_at),
                     'driver rc={}'.format(proc.poll())):
        proc.kill()
        return res
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(60)
    except subprocess.TimeoutExpired:
        proc.kill()
    res.check(os.path.exists(os.path.join(ws, 'runs', target, 'state-0.pkl')),
              'state-0.pkl present (required for hot-start)')

    # ---- phase 2: resume from the checkpoint -----------------------------
    log2 = os.path.join(root, 'phase2.log')
    env2 = dict(env)
    env2['SCULPTOR_HOTSTART_RUN_DIR'] = target
    env2['RAY_TMPDIR'] = tempfile.mkdtemp(prefix='/tmp/rt_')
    rc = _spawn(ws, env2, log2).wait()
    res.wall_s = time.time() - t0
    txt = open(log2, errors='replace').read()
    res.check('Loading from hotstart dir' in txt, 'hotstart dir loaded')
    iters = [int(x) for x in re.findall(r'TIMING START Iteration (\d+)', txt)]
    res.check(bool(iters) and min(iters) >= kill_at,
              'resumed at iter >= {} (not from 0)'.format(kill_at),
              'first iters: {}'.format(iters[:3]))
    m = re.search(r'ALL DONE in .*?(\d+)/(\d+) sizes ok', txt)
    res.check(bool(m) and m.group(1) == m.group(2) != '0',
              'ALL DONE banner (rc={} is not the judge)'.format(rc))
    C.check_metrics(os.path.join(ws, 'cache',
                    'popp_failure_latency_comparison_small_hs_e2e.pkl'),
                    res, prefix='resumed run: ', started_at=t0)
    C.collect(ws, res, _LABEL, [log1, log2])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kill-at', type=int, default=10,
                    help='checkpoint index to kill at (state-N.pkl)')
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()
    print('=' * 74 + '\ne2e: hot-start kill/resume\n' + '=' * 74)
    if not C.preflight():
        return 2
    root = tempfile.mkdtemp(prefix='verify_e2e_hotstart_')
    print('kill at     : state-{}\nscratch     : {}'.format(a.kill_at, root))
    print('=' * 74)
    res = run_case(root, a.kill_at)
    print('  -> {} in {:.0f}s'.format('PASS' if res.passed else 'FAIL', res.wall_s))
    return C.finish([res], root, a.keep)


if __name__ == '__main__':
    sys.exit(main())
