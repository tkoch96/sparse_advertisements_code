"""grab_paper_artifacts -- pull the paper's artifact set, as declared by
the SAME intent file that runs the evaluations.

    python evaluations/grab_paper_artifacts.py                       # pull all
    python evaluations/grab_paper_artifacts.py --check               # report only
    python evaluations/grab_paper_artifacts.py --only paper_table
    python evaluations/grab_paper_artifacts.py --intent evaluations/intents/other.json

One declaration, two verbs (Tom 2026-08-30): each stage in the intent
file says how to RUN it (run_all_paper_evaluations.py) and, via its
'artifacts' block, what it PRODUCES and where. This script checks the
declared sources (storage VM via vmlib, or local for
kind=local_artifact stages), pulls what exists, and for anything missing
prints a runbook DERIVED FROM the stage's own run command -- never a
hand-maintained duplicate. Exit 0 only when the whole set landed.
"""
import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from cluster import vmlib as V  # noqa: E402
from evaluations.run_all_paper_evaluations import _stage_cmd  # noqa: E402

DEFAULT_INTENT = os.path.join(
    _REPO, 'evaluations', 'intents', 'paper_intent.example.json')


def _runbook(name, spec, intent):
    """The command that produces this stage's artifacts, derived from the
    stage's own runner definition (single source of truth)."""
    if spec.get('runbook'):
        return spec['runbook']
    where = intent.get('where', 'local')
    cmd = _stage_cmd(name, spec, where)
    if where == 'vm':
        env = dict(intent.get('env') or {}, **(spec.get('env') or {}))
        envs = ' '.join('--env {}={}'.format(k, v) for k, v in env.items())
        return ('python -m cluster.expctl launch {} --label {} {} -- {}'
                .format(intent.get('storage_vm', '<vm>'),
                        '{}-{}'.format(intent.get('run_id', 'paper'), name),
                        envs, ' '.join(cmd)))
    return ' '.join(cmd)


def _vm_ip(instance):
    d = V.resolve(instance)
    if isinstance(d, str):
        d = V.describe(d)[0]
    if d['state'] != 'running':
        return None, d['state']
    return d['ip'], 'running'


def _remote_stat(ip, paths):
    """One ssh round-trip: mtime+size per path ('MISSING' when absent)."""
    script = ';'.join(
        'stat -c "%Y %s" "{p}" 2>/dev/null || echo MISSING'.format(p=p)
        for p in paths)
    rc, out, _err = V.ssh(ip, script, timeout=60)
    lines = (out or '').strip().split('\n')
    return {p: (None if line.strip() == 'MISSING' else line.strip())
            for p, line in zip(paths, lines)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--intent', default=DEFAULT_INTENT)
    ap.add_argument('--check', action='store_true',
                    help='report only; pull nothing')
    ap.add_argument('--only', default=None, help='comma list of stages')
    a = ap.parse_args()
    intent = json.load(open(a.intent))
    stages = intent.get('stages') or {}
    want = ([k.strip() for k in a.only.split(',')] if a.only
            else [k for k, s in stages.items() if s.get('artifacts')])
    bad = [k for k in want if k not in stages]
    if bad:
        raise SystemExit('unknown stage(s): {} (have: {})'.format(
            bad, sorted(stages)))

    need_vm = any((stages[k].get('artifacts') or {}).get('src_dir')
                  for k in want)
    ip = None
    if need_vm:
        vm = intent.get('storage_vm')
        ip, state = _vm_ip(vm)
        if ip is None:
            print('storage VM {} is {} -- start it first:\n'
                  '  python -m cluster.vmctl start {}'.format(vm, state, vm))
            return 2

    missing, pulled = {}, 0
    for k in want:
        spec = stages[k]
        art = spec.get('artifacts') or {}
        files = art.get('files') or []
        print('== {}'.format(k))
        if not files:
            print('   (no files declared -- skeleton stage)')
            continue
        if art.get('src_dir'):
            paths = [os.path.join(art['src_dir'], f) for f in files]
            st = _remote_stat(ip, paths)
            absent = [f for f, p in zip(files, paths) if st[p] is None]
            if absent:
                missing[k] = absent
                print('   MISSING on VM: {}'.format(', '.join(absent)))
                continue
            if a.check:
                print('   ok ({} files on VM)'.format(len(paths)))
                continue
            dst = os.path.join(_REPO, art['dst'])
            os.makedirs(dst, exist_ok=True)
            includes = []
            for f in files:
                includes += ['--include', f]
            rc, _o, err = V.rsync(
                'ubuntu@{}:{}/'.format(ip, art['src_dir']), dst + '/',
                ip=ip, extra=tuple(includes + ['--exclude', '*']))
            if rc != 0:
                print('   rsync FAILED: {}'.format((err or '')[:200]))
                missing[k] = ['<rsync failure>']
                continue
            pulled += len(paths)
            print('   pulled {} file(s) -> {}'.format(len(paths),
                                                      art['dst']))
        else:
            ld = os.path.join(_REPO, art['local_dir'])
            absent = [f for f in files
                      if not os.path.exists(os.path.join(ld, f))]
            if absent:
                missing[k] = absent
                print('   MISSING locally: {}'.format(', '.join(absent)))
                continue
            if a.check:
                print('   ok ({} local files)'.format(len(files)))
                continue
            import shutil
            dst = os.path.join(_REPO, art['dst'])
            os.makedirs(dst, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(ld, f), os.path.join(dst, f))
            pulled += len(files)
            print('   copied {} file(s) -> {}'.format(len(files),
                                                      art['dst']))

    if missing:
        print('\nINCOMPLETE -- {} stage(s) need runs:'.format(len(missing)))
        for k, files in missing.items():
            print('\n  [{}] missing: {}'.format(k, ', '.join(files)))
            print('  produce it with:\n    {}'.format(
                _runbook(k, stages[k], intent)))
        return 1
    print('\nALL PAPER ARTIFACTS {} ({} files)'.format(
        'PRESENT' if a.check else 'PULLED', pulled))
    return 0


if __name__ == '__main__':
    sys.exit(main())
