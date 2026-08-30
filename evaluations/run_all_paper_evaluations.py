"""run_all_paper_evaluations -- THE paper-evaluation entry point.

    python evaluations/run_all_paper_evaluations.py run   <intent> [--only S] [--dry-run] [--no-grab]
    python evaluations/run_all_paper_evaluations.py check <intent> [--only S]
    python evaluations/run_all_paper_evaluations.py grab  <intent> [--only S]
    python evaluations/run_all_paper_evaluations.py <intent>            # = run

One intent file, three verbs (Tom 2026-08-30 consolidation):
  run    execute every enabled stage to COMPLETION ('where': 'vm'
         launches via expctl then blocks on the run log; 'local' is a
         plain subprocess), store each stage's declared outputs as a
         DEPSTORE artifact (family 'paper_files', keyed by the stage's
         config fingerprint -- staleness detection for free via
         eval-era keying), mirror them into the directory canon for
         humans, then grab everything local.
  check  report which stages' artifacts are resolvable, pulling nothing.
  grab   pull resolvable artifacts into their local dst dirs; for any
         miss, print a runbook DERIVED from the stage's own run command.

Stages are data: built-ins cover deployment_sizes / n_prefixes /
paper_table, and any stage may instead declare its own 'cmd' template
(list of argv tokens; '{py}' expands to the right interpreter) -- new
stages need zero code here.

grab_paper_artifacts.py remains as a thin back-compat shim.
"""
import argparse
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

PY_LOCAL = os.path.expanduser('~/Documents/venv312/bin/python')
PY_VM = '/home/ubuntu/venv312/bin/python'
VM_REPO = '/home/ubuntu/sparse_advertisements_code'
DEFAULT_INTENT = os.path.join(
    _REPO, 'evaluations', 'intents', 'paper_intent.example.json')
ARTIFACT_FAMILY = 'paper_files'


# ---------------------------------------------------------------- cmd --
def _stage_cmd(name, spec, where, default_iters=None):
    """argv for one stage. A 'cmd' template in the intent wins (stages
    as data); built-ins cover the three original stages. Iteration count
    is ONE concept: per-stage 'iters' overriding the intent-level
    'iters' (Tom 2026-08-30: it had three spellings across stages)."""
    py = PY_VM if where == 'vm' else PY_LOCAL
    iters = spec.get('iters', default_iters)
    if spec.get('cmd'):
        return [t.replace('{py}', py) for t in spec['cmd']]
    if name == 'deployment_sizes':
        cmd = [py, '-u', 'evaluations/evaluate_over_deployment_sizes.py',
               '--port', str(spec.get('port', 31600)),
               '--cache-fn', spec['cache_fn'],
               '--figures-subdir', spec['figures_subdir'],
               '--dpsizes', ','.join(str(d) for d in spec['dpsizes']),
               '--nsim', ','.join(str(n) for n in spec['nsim'])]
        if iters:
            cmd += ['--max-iter', str(iters)]
        if spec.get('plot'):
            cmd.append('--plot')
        return cmd
    if name == 'n_prefixes':
        cmd = [py, '-u', 'evaluations/evaluate_over_n_prefixes.py',
               '--port', str(spec.get('port', 31602)),
               '--dpsize', str(spec['dpsize']),
               '--prefixes', ','.join(str(p) for p in spec['prefixes']),
               '--nsim', str(spec.get('nsim', 1))]
        if iters:
            cmd += ['--max-iter', str(iters)]
        if spec.get('cache_fn'):
            cmd += ['--cache-fn', spec['cache_fn']]
        if spec.get('figures_subdir'):
            cmd += ['--figures-subdir', spec['figures_subdir']]
        if spec.get('plot'):
            cmd.append('--plot')
        return cmd
    if name == 'paper_table':
        return [py, '-u', 'evaluations/generate_paper_table.py',
                '--dpsize', str(spec['dpsize']),
                '--number_of_deployments', str(spec['nsim']),
                '--num_training_iter', str(iters or spec.get('iters', 150)),
                '--run_id', spec['run_tag'],
                '--objectives', ','.join(spec['objectives']),
                '--out', spec['out']]
    raise SystemExit('stage {!r} has no cmd template and no built-in'
                     .format(name))


def _runbook(name, spec, intent):
    if spec.get('runbook'):
        return spec['runbook']
    where = intent.get('where', 'local')
    cmd = _stage_cmd(name, spec, where,
                     intent.get('iters'))
    if where == 'vm':
        env = dict(intent.get('env') or {}, **(spec.get('env') or {}))
        envs = ' '.join('--env {}={}'.format(k, v) for k, v in env.items())
        return ('python -m cluster.expctl launch {} --label {} {} -- {}'
                .format(intent.get('storage_vm', '<vm>'),
                        '{}-{}'.format(intent.get('run_id', 'paper'), name),
                        envs, ' '.join(cmd)))
    return ' '.join(cmd)


def _stage_fp(name, spec, intent):
    """Depstore fingerprint for a stage's artifact blob: the stage spec
    (minus comments/artifacts) + the merged env, so any config change
    re-keys and stale artifacts can never be served."""
    from core import depstore
    clean = {k: v for k, v in spec.items()
             if not k.startswith('_') and k not in ('artifacts',)}
    cfg = {'paper_stage': name,
           'spec': json.dumps(clean, sort_keys=True),
           'intent_env': json.dumps(
               dict(intent.get('env') or {}, **(spec.get('env') or {})),
               sort_keys=True)}
    fp, _ = depstore.fingerprint(cfg, _warn=False)
    return fp


# ----------------------------------------------------------------- vm --
def _vm(intent):
    from cluster import vmlib as V
    d = V.resolve(intent.get('storage_vm'))
    if isinstance(d, str):
        d = V.describe(d)[0]
    return d


def _remote_py(ip, code, timeout=180):
    from cluster import vmlib as V
    return V.ssh(ip, "{} - <<'RPYEOF'\n{}\nRPYEOF".format(PY_VM, code),
                 timeout=timeout)


# ------------------------------------------------------------- verbs --
def _run_local(name, cmd, env):
    print('[{}] local: {}'.format(name, ' '.join(cmd)))
    return subprocess.call(cmd, cwd=_REPO, env=dict(os.environ, **env))


def _run_vm(name, cmd, env, vm, run_label):
    """Launch via expctl, then BLOCK until the run's real exit code."""
    import re
    import time
    from cluster import vmlib as V
    argv = [PY_LOCAL, '-m', 'cluster.expctl', 'launch', vm,
            '--label', run_label]
    for k, v in env.items():
        argv += ['--env', '{}={}'.format(k, v)]
    argv += ['--'] + cmd
    print('[{}] vm launch: {}'.format(name, ' '.join(argv)))
    out = subprocess.run(argv, cwd=_REPO, capture_output=True, text=True)
    blob = (out.stdout or '') + (out.stderr or '')
    m = re.search(r'expctl watch (\S+)', blob)
    if out.returncode != 0 or not m:
        print(blob[-800:])
        return out.returncode or 1
    run_id = m.group(1)
    d = V.resolve(vm)
    if isinstance(d, str):
        d = V.describe(d)[0]
    log = '/home/ubuntu/cluster_runs/{}/run.log'.format(run_id)
    print('[{}] run {} started -- blocking until completion'.format(
        name, run_id))
    while True:
        time.sleep(60)
        _rc, o, _e = V.ssh(d['ip'],
                           'grep -a "exit_rc=" {} | tail -1'.format(log),
                           timeout=60)
        line = (o or '').strip()
        if 'exit_rc=' in line:
            rrc = int(line.split('exit_rc=')[1].split()[0])
            print('[{}] run {} finished rc={}'.format(name, run_id, rrc))
            return rrc


def _store_artifacts(name, spec, intent):
    """Stage succeeded: bundle its output files into a depstore eval
    artifact (family=paper_files, keyed by the stage fingerprint) and
    mirror them into the human-readable directory canon."""
    outputs = spec.get('outputs') or []
    art = spec.get('artifacts') or {}
    if not outputs:
        return
    fp = _stage_fp(name, spec, intent)
    where = intent.get('where', 'local')
    if where == 'vm':
        code = '''import glob, json, os, sys
sys.path.insert(0, {repo!r})
os.chdir({repo!r})
from core import depstore
files = {{}}
for g in {globs!r}:
    for f in glob.glob(g):
        files[os.path.basename(f)] = open(f, 'rb').read()
st = depstore.Depstore()
rel = st.put_eval({fp!r}, {fam!r}, files,
                  provenance={{'stage': {name!r}}})
canon = {canon!r}
if canon:
    os.makedirs(canon, exist_ok=True)
    for b, blob in files.items():
        open(os.path.join(canon, b), 'wb').write(blob)
print('stored', len(files), 'file(s) at', rel)'''.format(
            repo=VM_REPO, globs=outputs, fp=fp, fam=ARTIFACT_FAMILY,
            name=name, canon=art.get('src_dir') or '')
        d = _vm(intent)
        _rc, o, e = _remote_py(d['ip'], code)
        print('[{}] {}'.format(name, (o or e or '').strip()[-200:]))
    else:
        import glob as _g
        from core import depstore
        files = {}
        for g in outputs:
            for f in _g.glob(os.path.join(_REPO, g)):
                files[os.path.basename(f)] = open(f, 'rb').read()
        st = depstore.Depstore()
        st.put_eval(fp, ARTIFACT_FAMILY, files,
                    provenance={'stage': name})
        dst = os.path.join(_REPO, art.get('dst') or 'figures/paper')
        os.makedirs(dst, exist_ok=True)
        for b, blob in files.items():
            open(os.path.join(dst, b), 'wb').write(blob)
        print('[{}] stored {} file(s) in depstore + {}'.format(
            name, len(files), art.get('dst')))


def _resolve_artifacts(name, spec, intent, pull):
    """check/grab: depstore-first (fingerprint + era aware), directory
    canon as the transition fallback. Returns (ok, detail)."""
    art = spec.get('artifacts') or {}
    fp = _stage_fp(name, spec, intent)
    where = intent.get('where', 'local')
    dst = os.path.join(_REPO, art.get('dst') or 'figures/paper')

    if where == 'local' or spec.get('kind') == 'local_artifact':
        from core import depstore
        st = depstore.Depstore()
        blob = st.get_eval(fp, ARTIFACT_FAMILY)
        if blob:
            if pull:
                os.makedirs(dst, exist_ok=True)
                for b, data in blob.items():
                    open(os.path.join(dst, b), 'wb').write(data)
            return True, 'depstore ({} files)'.format(len(blob))
        ld = art.get('local_dir')
        files = art.get('files') or []
        if ld and files and all(os.path.exists(
                os.path.join(_REPO, ld, f)) for f in files):
            if pull:
                import shutil
                os.makedirs(dst, exist_ok=True)
                for f in files:
                    shutil.copy(os.path.join(_REPO, ld, f), dst)
            return True, 'local dir fallback ({} files)'.format(len(files))
        return False, 'no depstore artifact for fp {} (era-aware miss)' \
            .format(fp)

    # vm mode: resolve on the VM
    d = _vm(intent)
    if d['state'] != 'running':
        return False, 'storage VM is {} -- vmctl start it'.format(
            d['state'])
    code = '''import base64, json, os, sys
sys.path.insert(0, {repo!r}); os.chdir({repo!r})
from core import depstore
st = depstore.Depstore()
blob = st.get_eval({fp!r}, {fam!r})
if blob is None:
    print('MISS')
else:
    out = {{b: base64.b64encode(x).decode() for b, x in blob.items()}}
    print('HIT ' + json.dumps(out))'''.format(
        repo=VM_REPO, fp=fp, fam=ARTIFACT_FAMILY)
    _rc, o, _e = _remote_py(d['ip'], code, timeout=300)
    o = (o or '').strip()
    if o.startswith('HIT '):
        import base64
        files = json.loads(o.split('HIT ', 1)[1])
        if pull:
            os.makedirs(dst, exist_ok=True)
            for b, enc in files.items():
                open(os.path.join(dst, b), 'wb').write(
                    base64.b64decode(enc))
        return True, 'depstore@vm ({} files)'.format(len(files))
    # directory-canon fallback
    files = art.get('files') or []
    src = art.get('src_dir')
    if src and files:
        from cluster import vmlib as V
        script = ';'.join('test -f "{}/{}" && echo OK || echo NO'
                          .format(src, f) for f in files)
        _rc2, o2, _e2 = V.ssh(d['ip'], script, timeout=60)
        oks = (o2 or '').strip().split('\n')
        if oks and len(oks) == len(files) and all(
                x.strip() == 'OK' for x in oks):
            if pull:
                includes = []
                for f in files:
                    includes += ['--include', f]
                os.makedirs(dst, exist_ok=True)
                V.rsync('ubuntu@{}:{}/'.format(d['ip'], src), dst + '/',
                        ip=d['ip'],
                        extra=tuple(includes + ['--exclude', '*']))
            return True, 'dir-canon fallback ({} files)'.format(len(files))
    return False, 'no depstore artifact for fp {} and dir canon ' \
        'incomplete'.format(fp)


def verb_run(intent, a):
    where = intent.get('where', 'local')
    vm = intent.get('storage_vm')
    genv = intent.get('env') or {}
    stages = intent.get('stages') or {}
    want = ([k.strip() for k in a.only.split(',')] if a.only
            else [k for k, v in stages.items() if v.get('enabled')])
    results = {}
    for name in want:
        spec = stages.get(name)
        if spec is None:
            raise SystemExit('stage {!r} not in intent'.format(name))
        if spec.get('kind') == 'local_artifact':
            print('[{}] artifact-only stage -- nothing to run'.format(name))
            results[name] = 0
            continue
        env = dict(genv, **(spec.get('env') or {}))
        cmd = _stage_cmd(name, spec, where, intent.get('iters'))
        if a.dry_run:
            print('[{}] DRY: env={} cmd={}'.format(name, env,
                                                   ' '.join(cmd)))
            results[name] = 0
            continue
        rc = (_run_vm(name, cmd, env, vm,
                      '{}-{}'.format(intent.get('run_id', 'paper'), name))
              if where == 'vm' else _run_local(name, cmd, env))
        results[name] = rc
        if rc != 0:
            print('[{}] FAILED rc={} -- stopping'.format(name, rc))
            break
        _store_artifacts(name, spec, intent)
    ok = not any(results.values())
    if ok and not a.dry_run and not getattr(a, 'no_grab', False):
        print('\n== pulling artifacts local ==')
        verb_grab(intent, a, pull=True)
    print('\n== summary ==')
    for name, rc in results.items():
        print('  {:<20s} {}'.format(name, 'ok' if rc == 0
                                    else 'rc={}'.format(rc)))
    return 0 if ok else 1


def verb_grab(intent, a, pull=True):
    stages = intent.get('stages') or {}
    want = ([k.strip() for k in a.only.split(',')] if a.only
            else [k for k, s in stages.items()
                  if s.get('artifacts') or s.get('outputs')])
    missing = {}
    for name in want:
        spec = stages.get(name)
        if spec is None:
            raise SystemExit('stage {!r} not in intent'.format(name))
        _art = spec.get('artifacts') or {}
        if not (_art.get('files') or spec.get('outputs')):
            print('== {} -- (skeleton stage, no artifacts declared -- '
                  'skipped)'.format(name))
            continue
        ok, detail = _resolve_artifacts(name, spec, intent, pull)
        print('== {} -- {}: {}'.format(
            name, 'ok' if ok else 'MISSING', detail))
        if not ok:
            missing[name] = detail
    if missing:
        print('\nINCOMPLETE -- {} stage(s) need runs:'.format(len(missing)))
        for name, detail in missing.items():
            print('\n  [{}] {}'.format(name, detail))
            print('  produce it with:\n    {}'.format(
                _runbook(name, stages[name], intent)))
        return 1
    print('\nALL PAPER ARTIFACTS {}'.format(
        'PULLED' if pull else 'PRESENT'))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('verb_or_intent')
    ap.add_argument('intent', nargs='?', default=None)
    ap.add_argument('--only', default=None)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-grab', action='store_true')
    a = ap.parse_args()
    if a.verb_or_intent in ('run', 'check', 'grab'):
        verb = a.verb_or_intent
        intent_path = a.intent or DEFAULT_INTENT
    else:
        verb = 'run'
        intent_path = a.verb_or_intent
    intent = json.load(open(intent_path))
    # the runner itself stores/resolves paper_files artifacts, so the
    # intent's depstore env must bind in THIS process too (found
    # 2026-08-30: runner wrote to the repo store while stages wrote to
    # the local one)
    for k, v in (intent.get('env') or {}).items():
        if k.startswith('SCULPTOR_DEPSTORE'):
            os.environ[k] = str(v)
    if verb == 'run':
        return verb_run(intent, a)
    return verb_grab(intent, a, pull=(verb == 'grab'))


if __name__ == '__main__':
    sys.exit(main())
