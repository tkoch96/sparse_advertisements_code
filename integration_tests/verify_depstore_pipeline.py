"""Depstore pipeline integration test (Tom 2026-08-30 choreography).

    python integration_tests/verify_depstore_pipeline.py           # full
    python integration_tests/verify_depstore_pipeline.py --keep

Steps (exactly the agreed sequence):
  (a) SCULPTOR_DEPSTORE_TEST=1 -- a SEMANTIC flag hashed into every
      fingerprint, so this test's artifacts live in a disjoint namespace
  (b) purge any artifacts carrying that flag (clean slate)
  (c) modes: SCULPTOR_DEPSTORE_MODE=train_only / eval_only
  (d) eval_only on the clean cache      -> EXPECT FAILURE (not trained)
  (e) training pass (train_only)        -> trains + PUTs
  (f) train_only again                  -> EXPECT SUCCESS (cache HIT, fast)
  (g) eval_only                         -> EXPECT FAILURE (trained, but
                                           paper evals never ran)
  (h) eval pass = full pipeline          -> runs family evals, stamps marker
  (i) eval_only again                   -> EXPECT SUCCESS (fully cached)
  (j) purge the test namespace

The driver under test is eval_all_solution_types at small@3 iters (the
same harness verify_e2e_eval uses), pinned to one deployment via
SCULPTOR_EVAL_SEED so every step sees the same fingerprint.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
PY = os.path.expanduser('~/Documents/venv312/bin/python')

STEPS = []


def step(name, ok, detail=''):
    STEPS.append((name, ok, detail))
    print('  {} {} {}'.format('PASS' if ok else 'FAIL', name,
                              detail and '-- ' + detail))


def _driver_env(root, mode=None, ray_tmp=None):
    env = dict(os.environ)
    env.update({
        'SCULPTOR_DEPSTORE': '1',
        'SCULPTOR_DEPSTORE_TEST': '1',          # (a)
        'SCULPTOR_DEPSTORE_LOCAL': '1',
        'SCULPTOR_DEPSTORE_ROOT': root,
        'SCULPTOR_EVAL_SEED': '4242',
        'SCULPTOR_MAX_ITER': '3',
        'SCULPTOR_LOG_MEM': '0',
        'MPLBACKEND': 'Agg',
        # workers import core/* by module path; the throwaway ws cwd
        # doesn't contain them (local fork-ws recipe: PYTHONPATH must
        # carry the repo or actors die with ModuleNotFoundError)
        'PYTHONPATH': _REPO,
    })
    if mode:
        env['SCULPTOR_DEPSTORE_MODE'] = mode
    else:
        env.pop('SCULPTOR_DEPSTORE_MODE', None)
    if ray_tmp:
        # per-step Ray namespace (the run_n_sweep_queue pattern): a step
        # can never attach to the previous step's dying cluster. Short
        # path (unix-socket length limit).
        env['RAY_TMPDIR'] = ray_tmp
        env['RAY_ADDRESS'] = 'local'
    return env


_WS_N = [0]


def _run_driver(scratch, root, mode, log_fn, timeout=1800):
    """One driver invocation in a FRESH throwaway workspace per call:
    the legacy metrics-pickle layer must never carry state between
    steps -- the depstore is the only cross-step channel (step (f)
    false-passed via the pickle before this)."""
    from integration_tests import _common as C
    _WS_N[0] += 1
    ws = C.workspace(scratch, 'step{}'.format(_WS_N[0]))
    # Ray hygiene between steps: back-to-back driver runs attach to the
    # previous step's dying local Ray and fake-succeed with dead actors
    # (found via the standalone probe -- the outer except swallows the
    # worker-startup failure and exits rc=0 having done nothing).
    ray_tmp = '/tmp/dspr{}'.format(_WS_N[0])
    argv = [PY, '-u', C.driver('eval_all_solution_types.py'),
            '--dpsize', 'small', '--port', '31777']
    t0 = time.time()
    with open(log_fn, 'w') as lf:
        try:
            rc = subprocess.call(argv, cwd=ws,
                                 env=_driver_env(root, mode, ray_tmp),
                                 stdout=lf, stderr=subprocess.STDOUT,
                                 timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = -99
    return rc, time.time() - t0


def _purge(root):
    env = _driver_env(root)
    code = ('import sys; sys.path.insert(0, {!r});'
            'from core import depstore;'
            'print(depstore.Depstore().purge_test_artifacts())'
            .format(_REPO))
    out = subprocess.run([PY, '-c', code], env=env,
                         capture_output=True, text=True)
    return (out.stdout or '').strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()
    scratch = tempfile.mkdtemp(prefix='depstore_pipe_')
    root = os.path.join(scratch, 'store')
    logd = os.path.join(scratch, 'logs')
    os.makedirs(logd, exist_ok=True)
    print('scratch: {}'.format(scratch))

    try:
        # (b) clean slate
        purged = _purge(root)
        step('(b) purge test namespace', purged.isdigit(),
             '{} removed'.format(purged))

        # (d) eval_only on clean cache -> must FAIL
        rc, dt = _run_driver(scratch, root, 'eval_only',
                             os.path.join(logd, 'd_evalonly.log'))
        step('(d) eval_only on clean cache fails', rc != 0,
             'rc={} ({:.0f}s)'.format(rc, dt))

        # (e) training pass
        rc, dt = _run_driver(scratch, root, 'train_only',
                             os.path.join(logd, 'e_train.log'))
        _put = '[depstore] PUT' in open(os.path.join(logd, 'e_train.log'),
                                        errors='replace').read()
        step('(e) training pass succeeds + PUTs', rc == 0 and _put,
             'rc={} put={} ({:.0f}s)'.format(rc, _put, dt))
        t_train = dt

        # (f) train_only again -> success via cache HIT (much faster)
        rc, dt = _run_driver(scratch, root, 'train_only',
                             os.path.join(logd, 'f_train2.log'))
        hit = 'HIT' in open(os.path.join(logd, 'f_train2.log'),
                            errors='replace').read()
        step('(f) train_only re-run succeeds via HIT',
             rc == 0 and hit and dt < max(60, t_train * 0.7),
             'rc={} hit={} {:.0f}s (train was {:.0f}s)'.format(
                 rc, hit, dt, t_train))

        # (g) eval_only -> must FAIL (no paper_evals_done marker yet)
        rc, dt = _run_driver(scratch, root, 'eval_only',
                             os.path.join(logd, 'g_evalonly.log'))
        step('(g) eval_only before evals fails', rc != 0,
             'rc={} ({:.0f}s)'.format(rc, dt))

        # (h) eval pass = full pipeline (hits training, runs evals,
        #     stamps the marker)
        rc, dt = _run_driver(scratch, root, None,
                             os.path.join(logd, 'h_full.log'))
        _htxt = open(os.path.join(logd, 'h_full.log'),
                     errors='replace').read()
        stamped = ('paper_evals_done stamped for' in _htxt
                   and 'stamped for 0' not in _htxt)
        step('(h) full pipeline succeeds + stamps marker',
             rc == 0 and stamped, 'rc={} stamped={} ({:.0f}s)'.format(
                 rc, stamped, dt))

        # (i) eval_only again -> success
        rc, dt = _run_driver(scratch, root, 'eval_only',
                             os.path.join(logd, 'i_evalonly.log'))
        step('(i) eval_only after evals succeeds', rc == 0,
             'rc={} ({:.0f}s)'.format(rc, dt))

        # (j) purge
        purged = _purge(root)
        step('(j) final purge', purged.isdigit() and int(purged) > 0,
             '{} removed'.format(purged))

        failed = [s for s in STEPS if not s[1]]
        print('\n{}: {}/{} steps passed'.format(
            'FAILED' if failed else 'ALL PASS',
            len(STEPS) - len(failed), len(STEPS)))
        return 1 if failed else 0
    finally:
        if a.keep:
            print('kept: {}'.format(scratch))
        else:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
