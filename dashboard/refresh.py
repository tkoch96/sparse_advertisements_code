"""Generic dashboard refresher: one registry-driven loop for every
experiment (Tom, 2026-08-14: dashboards are cross-project abstractions;
no per-request custom tooling).

Each EXPERIMENTS entry in generate.py may carry a `refresh` spec:

  'refresh': {
    'remote_harvest': "<shell run on the head via ssh>",   # optional
    'pull': [("<remote repo-rel dir>/", "<local repo-rel dir>/") , ...],
    'steps': [                          # staleness-gated pipeline
      {'in':  ['<repo-rel glob>', ...],   # source data / upstream outs
       'out': ['<repo-rel path>', ...],   # what this step produces
       'argv': [...],                     # command ({repo}/{py} ok)
       'world': 'georand',                # eval-world knobs, resolved
                                          #   from model_error/worlds.py
                                          #   (REQUIRED for eval steps)
       'env': {...},                      # optional extra env
       'every': 4},                       # optional: only attempt on
      ...                                 #   every Nth cycle (cost cap)
    ],
    'evals': [ [argv...], ... ],        # legacy: run every cycle
    'heavy': [ [argv...], ... ],        # legacy: every --heavy-every
  }

A step runs iff an `out` is missing while inputs exist, or the newest
`in` mtime is newer than the oldest `out` mtime. Chain steps by listing
one step's `out` in the next step's `in` (data -> eval store -> figure):
any upstream change propagates, and untouched experiments cost nothing.
Prefer `steps` for all new entries — with `evals`/`heavy` the figure
argv must hard-code its output name, which is how the v2 figure went
stale while overwriting the bad-grads figure (2026-08-14).

argv entries may use {repo} and {py} placeholders. Pulls mirror with
rsync --delete (head is authoritative). After all experiments refresh,
the site regenerates once.

Head address resolution (first hit wins): --host flag, SCULPTOR_HEAD_IP
env, ~/.sculptor_cluster_alert/active_cluster.json (head.public_ip).

Usage:
    python -m dashboard.refresh                # one cycle
    python -m dashboard.refresh --loop 180 --heavy-every 4
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SSH_KEY = os.path.expanduser('~/.ssh/ray-autoscaler_us-east-1.pem')


def head_ip(flag):
    if flag:
        return flag
    if os.environ.get('SCULPTOR_HEAD_IP'):
        return os.environ['SCULPTOR_HEAD_IP']
    try:
        d = json.load(open(os.path.expanduser(
            '~/.sculptor_cluster_alert/active_cluster.json')))
        return d['head']['public_ip']
    except Exception:
        return None


def sh(cmd, **kw):
    return subprocess.run(cmd, **kw).returncode


def log(msg):
    print('{} {}'.format(time.strftime('%H:%M:%SZ', time.gmtime()), msg),
          flush=True)


def _expand_auto(argv):
    """AUTO:<root> -> comma list of that root's N-dirs that already
    contain results (steady/failure --dirs contract). None = no data."""
    for i, a in enumerate(argv):
        if a.startswith('AUTO:'):
            import glob as _g
            root = a[5:]
            ds = sorted(d for d in _g.glob(
                os.path.join(REPO, root, '*', 'N*'))
                if _g.glob(os.path.join(d, 'seed_*_*.json')))
            if not ds:
                return None
            argv[i] = ','.join(os.path.relpath(d, REPO) for d in ds)
            break
    return argv


def _run_cmd(argv, tag, py, env_extra=None):
    argv = _expand_auto([a.format(repo=REPO, py=py) for a in argv])
    if argv is None:
        return
    env = dict(os.environ, PYTHONPATH=REPO, MPLBACKEND='Agg')
    if env_extra:
        env.update(env_extra)
    rc = sh(argv, cwd=REPO, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, env=env)
    if rc != 0:
        log('[refresh:{}] cmd rc={}: {}'.format(
            tag, rc, ' '.join(argv[:4])))


def _newest_mtime(patterns):
    """Newest mtime across repo-relative globs; None if nothing matches."""
    import glob as _g
    newest = None
    for pat in patterns:
        for p in _g.glob(os.path.join(REPO, pat), recursive=True):
            m = os.path.getmtime(p)
            if newest is None or m > newest:
                newest = m
    return newest


STAMP_DIR = os.path.join(REPO, 'cache', 'model_error', '.refresh_stamps')


def _input_fingerprint(patterns):
    """Hash of the SET of matched input paths — catches deletions, which
    mtime comparison cannot (2026-08-14: quarantined arms lingered in
    the eval stores because removing files never makes inputs 'newer')."""
    import glob as _g
    import hashlib
    paths = sorted(p for pat in patterns
                   for p in _g.glob(os.path.join(REPO, pat),
                                    recursive=True))
    return hashlib.md5('\n'.join(paths).encode()).hexdigest(), len(paths)


def _step_staleness(step, stamp_key=None):
    """Return a reason string if the step should run, else None."""
    newest_in = _newest_mtime(step.get('in', []))
    if newest_in is None:
        return None                      # no source data yet
    outs = [os.path.join(REPO, o) for o in step.get('out', [])]
    missing = [o for o in outs if not os.path.exists(o)]
    if missing:
        return 'missing {}'.format(os.path.basename(missing[0]))
    oldest_out = min(os.path.getmtime(o) for o in outs)
    if newest_in > oldest_out:
        return 'inputs newer by {:.0f}s'.format(newest_in - oldest_out)
    if stamp_key:
        fp, n = _input_fingerprint(step.get('in', []))
        sf = os.path.join(STAMP_DIR, stamp_key + '.fp')
        old = open(sf).read().split()[0] if os.path.exists(sf) else None
        if old != fp:
            return 'input set changed ({} files)'.format(n)
    return None


def _write_stamp(step, stamp_key):
    if not stamp_key:
        return
    os.makedirs(STAMP_DIR, exist_ok=True)
    fp, n = _input_fingerprint(step.get('in', []))
    with open(os.path.join(STAMP_DIR, stamp_key + '.fp'), 'w') as f:
        f.write('{} {}\n'.format(fp, n))


def refresh_experiment(exp, ip, cycle, heavy_every, py):
    spec = exp.get('refresh')
    if not spec:
        return
    tag = exp.get('id', '?')
    ssh_base = ['ssh', '-o', 'ConnectTimeout=15',
                '-o', 'StrictHostKeyChecking=accept-new', '-i', SSH_KEY,
                'ubuntu@{}'.format(ip)]
    if ip and spec.get('remote_harvest'):
        sh(ssh_base + [spec['remote_harvest']],
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ip:
        for remote_rel, local_rel in spec.get('pull', []):
            sh(['rsync', '-az', '--delete', '--ignore-errors', '-e',
                'ssh -o ConnectTimeout=15 -o StrictHostKeyChecking='
                'accept-new -i {}'.format(SSH_KEY),
                'ubuntu@{}:sparse_advertisements_code/{}'.format(
                    ip, remote_rel),
                os.path.join(REPO, local_rel)],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i, step in enumerate(spec.get('steps', [])):
        if step.get('every') and cycle % step['every'] != 0:
            continue
        # 'always' steps (Tom, 2026-08-14): CHEAP steps (plots) run
        # every cycle unconditionally — full recomputation of all
        # available stats, so code/label changes propagate without
        # manual renders. Staleness gating is a COST optimization for
        # expensive evals only, never a correctness mechanism.
        stamp_key = '{}_{}'.format(tag, i)
        if step.get('always'):
            reason = 'always'
        else:
            reason = _step_staleness(step, stamp_key)
            if not reason:
                continue
        log('[refresh:{}] step {} ({}) -> {}'.format(
            tag, i, reason, ' '.join(step['argv'][:4])))
        env_extra = dict(step.get('env') or {})
        if step.get('world'):
            from core import worlds
            env_extra = dict(worlds.env(step['world']), **env_extra)
        _run_cmd(list(step['argv']), tag, py, env_extra)
        _write_stamp(step, stamp_key)
    cmds = list(spec.get('evals', []))
    if heavy_every and cycle % heavy_every == 0:
        cmds += list(spec.get('heavy', []))
    for argv in cmds:
        _run_cmd(list(argv), tag, py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host')
    ap.add_argument('--loop', type=int, default=0,
                    help='seconds between cycles; 0 = run once')
    ap.add_argument('--heavy-every', type=int, default=4)
    ap.add_argument('--experiments', help='comma list of experiment ids')
    ap.add_argument('--py', default=sys.executable)
    args = ap.parse_args()

    from dashboard import generate
    only = set(args.experiments.split(',')) if args.experiments else None

    cycle = 0
    while True:
        cycle += 1
        # Re-load the REGISTRY every cycle (2026-08-14: a startup-only
        # import meant registry edits — new tabs, new steps — were
        # silently absent from the running loop, which kept regenerating
        # the site without them).
        try:
            import importlib
            importlib.reload(generate)
        except Exception as e:
            print('[refresh] registry reload failed: {}'.format(e),
                  flush=True)
        # Re-resolve the head IP EVERY cycle (2026-08-14: instance
        # restarts change it; a startup-only lookup silently starved
        # pulls for an hour) — the alert JSON is the source of truth.
        ip = head_ip(args.host)
        if not ip and cycle == 1:
            print('[refresh] no head ip resolved; local-only refresh')
        units = []
        for exp in generate.EXPERIMENTS:
            if only and exp.get('id') not in only:
                continue
            units.append(exp)
            units.extend(exp.get('sections', []))
        for u in units:
            if not u.get('refresh'):
                continue
            try:
                refresh_experiment(u, ip, cycle, args.heavy_every,
                                   args.py)
            except Exception as e:
                print('[refresh:{}] {}'.format(u.get('id'), e),
                      flush=True)
        try:
            generate.main()
        except Exception as e:
            print('[refresh] generate failed: {}'.format(e), flush=True)
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == '__main__':
    main()
