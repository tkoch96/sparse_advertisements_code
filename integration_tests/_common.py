"""Shared machinery for the end-to-end integration tests.

Every check here exists because something once passed when it shouldn't have.
The three that matter:

1. **rc=0 is not success.** `evaluate_all_metrics` wraps its strategy loop in a
   bare `except:` that prints and continues, so a run whose solver died in the
   first second still reaches plotting, returns a metrics dict and exits 0. On
   2026-08-21 the queue read exactly that as success and its harvest step
   deleted 11h of actual-32 training. Cases are judged on artifacts.

2. **Stale artifacts.** The workspace cache symlinks the repo, so a previous
   run's result pickle satisfies every content check — and worse,
   `evaluate_all_metrics` *resume-skips* a sim whose `n_advs` is populated, so
   the run becomes a seconds-long no-op that passes. `workspace()` symlinks
   cache INPUTS but excludes result pickles.

3. **Ray hygiene.** Sockets are AF_UNIX (103-byte path cap), so RAY_TMPDIR goes
   under /tmp; and consecutive cases must not attach to each other's draining
   cluster, so RAY_ADDRESS=local forces a private one.

Note on ports: `--port` is vestigial under Ray — nothing binds it
(`path_distribution_computer` sets `self.port = 0` outright). None of these
tests pass one.
"""
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)

# Markers that mean a strategy or worker actually died.
FATAL_PATTERNS = [
    (re.compile(r'ERROR cmd '),     'a Ray worker raised inside a command handler'),
    (re.compile(r'\[FATAL\]'),      'driver aborted a required strategy'),
    (re.compile(r'\[INCOMPLETE\]'), 'comparison data is incomplete'),
    (re.compile(r'MemoryError'),    'out of memory'),
    (re.compile(r'killed due to memory pressure'), 'Ray OOM-killed a worker'),
]

EXPECTED_STRATEGIES = {'sparse', 'anyopt', 'painter',
                       'anycast', 'one_per_pop', 'one_per_peering'}

# Result-pickle prefixes that must NOT be inherited from the repo cache.
_RESULT_PREFIXES = ('popp_failure_latency_comparison_',
                    'evaluate_over_prefix_numbers_',
                    'testing_feature_cache_fn')


class Result:
    """Accumulates named pass/fail checks for one case."""

    def __init__(self, label):
        self.label = label
        self.checks = []
        self.artifacts = None
        self.figs = []
        self.wall_s = 0.0

    def check(self, ok, name, detail=''):
        self.checks.append((bool(ok), name, detail))
        return bool(ok)

    @property
    def passed(self):
        return all(ok for ok, _, _ in self.checks)

    def report(self):
        print('\n{}  [{}]  {:.0f}s'.format(
            self.label, 'PASS' if self.passed else 'FAIL', self.wall_s))
        for ok, name, detail in self.checks:
            print('   {} {}{}'.format('PASS' if ok else 'FAIL', name,
                                      '  -- ' + detail if detail else ''))
        if self.artifacts:
            print('   artifacts -> {}'.format(os.path.relpath(self.artifacts, REPO)))
            for f in sorted(set(self.figs)):
                print('      fig: {}'.format(f))


def workspace(root, label):
    """Throwaway workspace whose cache holds the INPUTS but none of the RESULTS."""
    ws = os.path.join(root, 'ws_' + label)
    for sub in ('runs', 'logs', 'figures/paper', 'cache'):
        os.makedirs(os.path.join(ws, sub), exist_ok=True)
    link = os.path.join(ws, 'data')
    if not os.path.lexists(link):
        os.symlink(os.path.join(REPO, 'data'), link)
    for entry in os.listdir(os.path.join(REPO, 'cache')):
        if entry.startswith(_RESULT_PREFIXES):
            continue
        dst = os.path.join(ws, 'cache', entry)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(REPO, 'cache', entry), dst)
    return ws


def env_for(iters, extra=None, label=None):
    e = dict(os.environ)
    e.update({
        'PYTHONPATH': REPO,
        'MPLBACKEND': 'Agg',
        'SCULPTOR_MAX_ITER': str(iters),
        'RAY_TMPDIR': tempfile.mkdtemp(prefix='/tmp/rt_'),  # AF_UNIX 103B cap
        'RAY_ADDRESS': 'local',        # private cluster per case
        'SCULPTOR_XOBJS': '1',         # register max_util / frac_beyond_optimal
    })
    # NOTE (2026-08-21): do NOT set SCULPTOR_RUN_TAG here. It was added to
    # stop cases loading each other's cached results, but workspace() above
    # already guarantees that -- it symlinks the INPUT cache entries and
    # skips _RESULT_PREFIXES, and every case runs with cwd=ws, so CACHE_DIR
    # resolves inside the workspace. The tag only appended a suffix to the
    # RESULT filename, which broke check_metrics and
    # evaluate_over_n_prefixes.py (which opens the untagged name directly):
    # 3 of 5 e2e tests went red. Isolation here is by workspace, not by tag.
    if label:
        # keep throwaway test figures out of the real sweep output
        e['SCULPTOR_FIG_SUBDIR'] = os.path.join('integration_tests', label)
    e.update(extra or {})
    return e


def run(argv, ws, env, log_fn):
    with open(log_fn, 'w') as lf:
        return subprocess.call(argv, cwd=ws, env=env, stdout=lf,
                               stderr=subprocess.STDOUT)


def driver(*parts):
    """Absolute path to an evaluations/ driver."""
    return os.path.join(REPO, 'evaluations', *parts)


def scan_log(log_fn, res):
    text = open(log_fn, errors='replace').read()
    for pat, why in FATAL_PATTERNS:
        hits = pat.findall(text)
        res.check(not hits, 'no "{}"'.format(pat.pattern[:30]),
                  '{}x -- {}'.format(len(hits), why) if hits else '')
    blocks = [b for b in re.split(r'(?=Traceback \(most recent call last\):)', text)
              if b.startswith('Traceback')]
    detail = []
    for b in blocks[:4]:
        frames = re.findall(r'  File "([^"]+)", line (\d+), in (\S+)', b[:2000])
        last = frames[-1] if frames else ('?', '?', '?')
        exc = b.strip().splitlines()[-1][:100] if b.strip() else '?'
        detail.append('{}:{} -> {}'.format(os.path.basename(last[0]), last[1], exc))
    res.check(not blocks, 'no swallowed tracebacks', '; '.join(detail))
    return text


def check_metrics(pkl_fn, res, prefix='', started_at=None):
    """Judge a metrics pickle on content, not on the exit code."""
    if not res.check(os.path.exists(pkl_fn), prefix + 'metrics pickle written',
                     pkl_fn):
        return None
    if started_at is not None:
        age = os.path.getmtime(pkl_fn) - started_at
        if not res.check(age > 0, prefix + 'metrics pickle is from THIS run',
                         'written {:.0f}s in'.format(age) if age > 0 else
                         'STALE: predates run start by {:.0f}s'.format(-age)):
            return None
    try:
        d = pickle.load(open(pkl_fn, 'rb'))
    except Exception as e:
        res.check(False, prefix + 'metrics pickle loads',
                  '{}: {}'.format(type(e).__name__, e))
        return None
    cr = (d.get('compare_rets') or {}).get(0)
    if not res.check(isinstance(cr, dict), prefix + 'compare_rets[0] present',
                     'got {}'.format(type(cr).__name__)):
        return d
    failed = cr.get('failed_strategies') or []
    res.check(not failed, prefix + 'no failed strategies', str(failed))
    missing = EXPECTED_STRATEGIES - set(cr.get('adv_solns') or {})
    res.check(not missing, prefix + 'all 6 strategies solved',
              'missing {}'.format(sorted(missing)) if missing else '')
    return d


def collect(ws, res, label, extra_files=(), root_dir='integration_tests'):
    """Copy figures + result pickles out before the scratch dir is deleted.

    Goes to figures/<root_dir>/<label>/ -- `integration_tests` by default, so
    test output never lands beside a real sweep's figures.
    """
    out = os.path.join(REPO, 'figures', root_dir, label)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    for dp, _, fns in os.walk(os.path.join(ws, 'figures')):
        for fn in fns:
            if fn.startswith('.'):
                continue
            shutil.copy(os.path.join(dp, fn), os.path.join(out, fn))
            res.figs.append(fn)
    for f in extra_files:
        if f and os.path.exists(f):
            shutil.copy(f, os.path.join(out, os.path.basename(f)))
    res.artifacts = out
    return out


def preflight():
    """Fail fast on the wrong interpreter instead of deep inside a solver."""
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
    print('LP backend  : {} (needs {})'.format(backend, need))
    if missing:
        print('\nCANNOT RUN -- missing module(s): {}'.format(', '.join(missing)))
        print('Almost always the wrong virtualenv. The project venv is')
        print('~/Documents/venv312 (Python 3.12); ~/Documents/venv is 3.14 and')
        print('has no highspy. Re-run with:')
        print('    ~/Documents/venv312/bin/python {} {}'.format(
            os.path.relpath(os.path.abspath(sys.argv[0])), ' '.join(sys.argv[1:])))
        return False
    return True


def finish(results, root, keep):
    """Common tail: per-case report, summary, scratch cleanup, exit code."""
    print('\n' + '=' * 74 + '\nRESULTS\n' + '=' * 74)
    for r in results:
        r.report()
    failed = [r for r in results if not r.passed]
    print('\n' + '-' * 74)
    if failed:
        print('FAILED: {} of {} case(s)'.format(len(failed), len(results)))
        print('logs kept in {}'.format(root))
        return 1
    print('ALL {} CASE(S) PASSED'.format(len(results)))
    print('figures + result pickles -> figures/integration_tests/')
    if not keep:
        shutil.rmtree(root, ignore_errors=True)
    else:
        print('scratch kept: {}'.format(root))
    return 0

