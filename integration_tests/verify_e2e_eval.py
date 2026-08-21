#!/usr/bin/env python
"""End-to-end verification that `evaluate_all_metrics` actually runs.

    python integration_tests/verify_e2e_eval.py            # full: small@100, actual-5@50
    python integration_tests/verify_e2e_eval.py --quick    # fast:  small@3,   actual-5@3
    python integration_tests/verify_e2e_eval.py --only small

WHY THIS EXISTS -- and why it does not just check the exit code.

`evaluate_all_metrics` wraps its entire strategy loop in a bare `except:`
(evaluations/eval_latency_failure.py:661) that prints a traceback and continues. A run
whose solver died in the first second still reaches the plotting section,
still returns a metrics dict, and still exits 0. On 2026-08-21 that exact
behaviour cost ~11h of actual-32 training: a broken hot-start exited 0 after
six seconds, the queue read rc=0 as success, and its harvest step deleted the
run directory.

So rc=0 is necessary, not sufficient. Every case below is judged on the
CONTENT of the metrics pickle it produced -- did all six strategies actually
solve, are the per-UG latency vectors populated and finite -- plus a scan of
the log for failure markers the bare except would otherwise have hidden.

Each case runs in a throwaway workspace with `cache/` and `data/` symlinked
back to the repo, so nothing here mutates repo state.
"""
import argparse
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# Cases: (label, dpsize, default iters). The dpsize string is passed straight
# to eval_latency_failure.py --dpsize.
# (label, dpsize, default iters).
# actual-5 measured 2831s (47 min) at THREE iterations on an M-series laptop --
# it rebuilds from the 4.2 GB measurement CSV and carries 2567 UGs. 50 iters is
# a cluster setting, not a laptop one, so the default here is deliberately low;
# pass --iters 50 on a real box.
CASES = [
    ('small',    'small',    100),
    ('actual-5', 'actual-5',   3),
]

# Swallowed tracebacks that already occur on a healthy tree. Each entry is
# (regex, why-it-is-tolerated). These are REPORTED on every run, never
# silently dropped -- a new signature not listed here fails the run, which is
# the regression signal we actually want.
KNOWN_TRACEBACKS = [
    # (regex, why-it-is-tolerated). Empty since 2026-08-21: the only entry was
    # assess_volume_multipliers, whose phase has been removed as dead code.
    # A traceback not listed here FAILS the run -- that is the regression signal.
]

# Hard failure markers -- these mean a strategy or a worker actually died.
FATAL_PATTERNS = [
    (re.compile(r'ERROR cmd '),        'a Ray worker raised inside a command handler'),
    (re.compile(r'\[FATAL\]'),         'driver aborted a required strategy'),
    (re.compile(r'\[INCOMPLETE\]'),    'comparison data is incomplete'),
    (re.compile(r'MemoryError'),       'out of memory'),
    (re.compile(r'raylet.*OOM|killed due to memory pressure'),
                                       'Ray killed a worker under memory pressure'),
]

EXPECTED_STRATEGIES = {'sparse', 'anyopt', 'painter',
                       'anycast', 'one_per_pop', 'one_per_peering'}


class Result:
    def __init__(self, label):
        self.label = label
        self.checks = []          # (ok, name, detail)
        self.warnings = []
        self.wall_s = 0.0

    def check(self, ok, name, detail=''):
        self.checks.append((bool(ok), name, detail))
        return bool(ok)

    @property
    def passed(self):
        return all(ok for ok, _, _ in self.checks)


def preflight():
    """Fail fast on a wrong interpreter instead of two frames deep in a solver.

    gpshim picks its LP backend at import time from SCULPTOR_LP_BACKEND,
    defaulting to 'highs' -- so highspy is a hard requirement on the default
    path. A venv with only gurobipy dies inside `import gpshim`, which the
    bare except then swallows into an unhelpful mid-run traceback.
    """
    backend = os.environ.get('SCULPTOR_LP_BACKEND', 'highs').lower()
    need = {'highs': 'highspy', 'gurobi': 'gurobipy'}.get(backend)
    missing = []
    for mod in [need, 'ray', 'numpy']:
        if not mod:
            continue
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    print('interpreter : {}'.format(sys.executable))
    print('python      : {}'.format(sys.version.split()[0]))
    print('LP backend  : {} (needs {})'.format(backend, need))
    if missing:
        print('\nCANNOT RUN -- missing module(s): {}'.format(', '.join(missing)))
        print('This is almost always the wrong virtualenv. The project venv is')
        print('~/Documents/venv312 (Python 3.12); ~/Documents/venv is 3.14 and')
        print('has no highspy. Re-run with:')
        print('    ~/Documents/venv312/bin/python {} {}'.format(
            os.path.relpath(os.path.abspath(__file__)), ' '.join(sys.argv[1:])))
        return False
    return True


def _make_workspace(root, label):
    """Throwaway workspace whose cache has the INPUTS but none of the RESULTS.

    `evaluate_all_metrics` resume-skips a simulation whose
    compare_rets[...]['n_advs'] is already populated in
    cache/popp_failure_latency_comparison_<dpsize>.pkl. Symlinking the whole
    repo cache therefore makes the run a no-op that exits 0 in a few seconds
    while the harness happily validates last week's numbers. So: symlink every
    cache entry EXCEPT those result pickles, forcing a real computation whose
    output we can then check for freshness.
    """
    ws = os.path.join(root, 'ws_' + label.replace('-', '_'))
    for sub in ('runs', 'logs', 'figures/paper', 'cache'):
        os.makedirs(os.path.join(ws, sub), exist_ok=True)
    dst = os.path.join(ws, 'data')
    if not os.path.lexists(dst):
        os.symlink(os.path.join(_REPO, 'data'), dst)
    for entry in os.listdir(os.path.join(_REPO, 'cache')):
        if entry.startswith('popp_failure_latency_comparison_'):
            continue                      # results, not inputs -- keep them out
        link = os.path.join(ws, 'cache', entry)
        if not os.path.lexists(link):
            os.symlink(os.path.join(_REPO, 'cache', entry), link)
    return ws


def _scan_log(text, res):
    """Classify everything the bare except would have hidden."""
    for pat, why in FATAL_PATTERNS:
        hits = pat.findall(text)
        res.check(not hits, 'no "{}"'.format(pat.pattern[:34]),
                  '{} occurrence(s) -- {}'.format(len(hits), why) if hits else '')

    # Split the log into traceback blocks and classify each one.
    blocks = re.split(r'(?=Traceback \(most recent call last\):)', text)
    blocks = [b for b in blocks if b.startswith('Traceback')]
    known, unknown = 0, []
    for b in blocks:
        head = b[:2000]
        if any(pat.search(head) for pat, _ in KNOWN_TRACEBACKS):
            known += 1
        else:
            # first frame + exception line, for a readable report
            frames = re.findall(r'  File "([^"]+)", line (\d+), in (\S+)', head)
            last = frames[-1] if frames else ('?', '?', '?')
            exc = b.strip().splitlines()[-1][:120] if b.strip() else '?'
            unknown.append('{}:{} in {} -> {}'.format(
                os.path.basename(last[0]), last[1], last[2], exc))
    if known:
        res.warnings.append(
            '{} known-issue traceback(s) swallowed by the bare except:'.format(known))
        for _, why in KNOWN_TRACEBACKS:
            res.warnings.append('    - ' + why)
    res.check(not unknown, 'no unrecognised swallowed tracebacks',
              '; '.join(unknown[:4]) if unknown else '')


def _scan_metrics(pkl_fn, res, started_at=None):
    if not res.check(os.path.exists(pkl_fn), 'metrics pickle written', pkl_fn):
        return
    # cache/ is symlinked back to the repo, so this path may already hold a
    # PREVIOUS run's results. Without this check a run that died instantly
    # would still pass every content assertion below -- the same
    # stale-artifact trap that let a 6-second no-op read as success on
    # 2026-08-21.
    if started_at is not None:
        age = os.path.getmtime(pkl_fn) - started_at
        detail = ('written {:.0f}s into the run'.format(age) if age > 0 else
                  'STALE: mtime predates run start by {:.0f}s'.format(-age))
        if not res.check(age > 0, 'metrics pickle is from THIS run', detail):
            return
    try:
        d = pickle.load(open(pkl_fn, 'rb'))
    except Exception as e:
        res.check(False, 'metrics pickle loads', '{}: {}'.format(type(e).__name__, e))
        return
    res.check(True, 'metrics pickle loads')

    cr = (d.get('compare_rets') or {}).get(0)
    if not res.check(isinstance(cr, dict), 'compare_rets[0] present',
                     'got {}'.format(type(cr).__name__)):
        return

    failed = cr.get('failed_strategies') or []
    res.check(not failed, 'no failed strategies', 'failed: {}'.format(failed))

    solved = set(cr.get('adv_solns') or {})
    missing = EXPECTED_STRATEGIES - solved
    res.check(not missing, 'all 6 strategies solved',
              'missing: {}'.format(sorted(missing)) if missing else '')

    n_advs = cr.get('n_advs') or {}
    empty = [s for s in solved if not n_advs.get(s)]
    res.check(not empty, 'every strategy has an advertisement',
              'no n_advs for: {}'.format(sorted(empty)) if empty else '')

    lats = (d.get('latencies') or {}).get(0) or {}
    bad = []
    for s in sorted(EXPECTED_STRATEGIES):
        v = lats.get(s)
        if v is None or len(v) == 0:
            bad.append('{}=empty'.format(s))
            continue
        try:
            import numpy as np
            arr = np.asarray(v, dtype=float)
            if not np.isfinite(arr).all():
                bad.append('{}=non-finite'.format(s))
        except Exception as e:
            bad.append('{}={}'.format(s, type(e).__name__))
    res.check(not bad, 'per-UG latencies populated and finite',
              '; '.join(bad) if bad else
              'n={} per strategy'.format(len(lats.get('sparse', []))))


def run_case(label, dpsize, iters, port, root, keep_logs):
    res = Result('{} @ {} iters'.format(label, iters))
    ws = _make_workspace(root, label)
    ray_tmp = tempfile.mkdtemp(prefix='/tmp/rt_')
    log_fn = os.path.join(root, 'eval_{}.log'.format(label.replace('-', '_')))

    env = dict(os.environ)
    env.update({
        'PYTHONPATH': _REPO,
        'MPLBACKEND': 'Agg',
        'SCULPTOR_MAX_ITER': str(iters),
        # one traceback per distinct worker error, always (2026-08-21 contract)
        'SCULPTOR_VERBOSE_ERRORS': env.get('SCULPTOR_VERBOSE_ERRORS', '0'),
        # Ray sockets are AF_UNIX and the path cannot exceed 103 bytes; a
        # tempfile.mkdtemp() root under /var/folders/... blows that on macOS.
        'RAY_TMPDIR': ray_tmp,
        # Force a private cluster per case. Without this the second case
        # finds the first case's still-draining cluster ("Connecting to
        # existing Ray cluster at 127.0.0.1:...") and dies when it goes
        # away -- a 2-second failure that looks like a code bug but is
        # pure harness cross-contamination.
        'RAY_ADDRESS': 'local',
    })
    os.makedirs(env['RAY_TMPDIR'], exist_ok=True)

    cmd = [sys.executable, '-u', os.path.join(_REPO, 'evaluations', 'eval_latency_failure.py'),
           '--dpsize', dpsize, '--port', str(port)]
    print('  running: {} (cwd={})'.format(' '.join(cmd[-4:]), os.path.basename(ws)),
          flush=True)

    t0 = time.time()
    with open(log_fn, 'w') as lf:
        rc = subprocess.call(cmd, cwd=ws, env=env, stdout=lf,
                             stderr=subprocess.STDOUT)
    res.wall_s = time.time() - t0

    res.check(rc == 0, 'exit code 0', 'rc={}'.format(rc))
    text = open(log_fn, errors='replace').read()
    _scan_log(text, res)
    # global_performance_metrics_fn(dpsize) -> cache/popp_failure_latency_comparison_<dpsize>.pkl
    _scan_metrics(os.path.join(ws, 'cache',
                  'popp_failure_latency_comparison_{}.pkl'.format(dpsize)), res, t0)
    # The workspace is a tempdir that gets deleted on success, so anything
    # eval_latency_failure wrote there (its comparison PDF, the run log) would
    # vanish with it. Copy the artifacts somewhere durable and report the path
    # -- otherwise "it passed" gives you nothing to actually look at.
    out_dir = os.path.join(_REPO, 'figures', 'evaluations',
                           label.replace('-', '_'))
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    figs = []
    src_figs = os.path.join(ws, 'figures')
    for dp, _, fns in os.walk(src_figs):
        for fn in fns:
            if fn.startswith('.'):
                continue
            src = os.path.join(dp, fn)
            dst = os.path.join(out_dir, fn)
            shutil.copy(src, dst)
            figs.append(dst)
    shutil.copy(log_fn, os.path.join(out_dir, 'eval.log'))
    pkl = os.path.join(ws, 'cache',
                       'popp_failure_latency_comparison_{}.pkl'.format(dpsize))
    if os.path.exists(pkl):
        shutil.copy(pkl, os.path.join(out_dir, os.path.basename(pkl)))
    res.artifacts = out_dir
    res.figs = figs
    res.log_fn = log_fn if keep_logs else None
    if not keep_logs:
        shutil.rmtree(ray_tmp, ignore_errors=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true',
                    help='3 iters per case -- shape check, not a convergence check')
    ap.add_argument('--iters', type=int, default=None,
                    help='override iterations for every case')
    ap.add_argument('--only', default=None, help='run just this case label')
    # --port is vestigial: eval_latency_failure requires it, but under Ray
    # nothing binds it (path_distribution_computer.py (actor layer) sets
    # `self.port = 0  # unused under Ray`). Its only real effect is avoiding
    # the 5-second "NO PORT SPECIFIED" sleep in optimal_adv_wrapper.py:936.
    # Case isolation comes from RAY_ADDRESS=local, not from distinct ports.
    ap.add_argument('--port', type=int, default=31900,
                    help='port passed through to eval_latency_failure '
                         '(required by its CLI; unused under Ray)')
    ap.add_argument('--keep', action='store_true',
                    help='keep the scratch workspaces and logs')
    args = ap.parse_args()

    cases = [c for c in CASES if args.only in (None, c[0])]
    if not cases:
        sys.exit('no case matches --only {!r} (have: {})'.format(
            args.only, ', '.join(c[0] for c in CASES)))

    print('=' * 74)
    print('e2e eval verification')
    print('=' * 74)
    if not preflight():
        return 2
    root = tempfile.mkdtemp(prefix='verify_e2e_eval_')
    print('cases       : {}'.format(', '.join(c[0] for c in cases)))
    print('scratch     : {}'.format(root))
    print('=' * 74)

    results = []
    for i, (label, dpsize, default_iters) in enumerate(cases):
        iters = 3 if args.quick else (args.iters or default_iters)
        print('\n[{}/{}] {} @ {} iters'.format(i + 1, len(cases), label, iters), flush=True)
        if i:
            time.sleep(5)   # let the previous case's Ray cluster fully release
        results.append(run_case(label, dpsize, iters, args.port + i, root, args.keep))
        r = results[-1]
        print('  -> {} in {:.0f}s'.format('PASS' if r.passed else 'FAIL', r.wall_s),
              flush=True)

    print('\n' + '=' * 74)
    print('RESULTS')
    print('=' * 74)
    for r in results:
        print('\n{}  [{}]  {:.0f}s'.format(
            r.label, 'PASS' if r.passed else 'FAIL', r.wall_s))
        for ok, name, detail in r.checks:
            print('   {} {}{}'.format('PASS' if ok else 'FAIL', name,
                                      '  -- ' + detail if detail else ''))
        for w in r.warnings:
            print('   WARN ' + w)
        if getattr(r, 'artifacts', None):
            print('   artifacts -> {}'.format(os.path.relpath(r.artifacts, _REPO)))
            for f in getattr(r, 'figs', []):
                print('      fig: {}'.format(os.path.basename(f)))

    failed = [r for r in results if not r.passed]
    print('\n' + '-' * 74)
    if failed:
        print('FAILED: {} of {} case(s) -- {}'.format(
            len(failed), len(results), ', '.join(r.label for r in failed)))
        print('logs kept in {}'.format(root))
        return 1
    print('ALL {} CASE(S) PASSED'.format(len(results)))
    print('evaluation plots + eval.log + metrics pkl -> figures/evaluations/')
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    else:
        print('scratch kept: {}'.format(root))
    return 0


if __name__ == '__main__':
    sys.exit(main())
