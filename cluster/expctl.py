#!/usr/bin/env python
"""expctl -- push code to a VM, launch an experiment, watch it, pull it back.

    python -m cluster.expctl push head
    python -m cluster.expctl launch head --preset dpsweep --label smoke \
        --dpsizes 3,5,10 --nsim 1 --max-iter 10
    python -m cluster.expctl list
    python -m cluster.expctl status <run_id>
    python -m cluster.expctl watch  <run_id> --interval 300
    python -m cluster.expctl pull   <run_id>
    python -m cluster.expctl kill   <run_id>
    python -m cluster.expctl finish <run_id>

The harvest contract
--------------------
The log is the only thing that explains a failure and it lives on a box
that will be stopped. So:

* `status` pulls before it reports (`--no-pull` opts out).
* `watch` pulls on every tick, and pulls once more when the process exits.
* `kill` pulls after killing.
* `vmctl stop` pulls every live run and **refuses to stop** if any bytes
  are still only on the VM.

`watch` exits 0 only for a clean `done`, and nonzero otherwise, so it can
gate a shell chain. **Do not pipe it** (`watch ... | tail`) -- the pipe
hands you the exit code of `tail`, which reported 0 for a FAILED run on
2026-08-21. That is the same lie as trusting rc == 0, reintroduced one
layer up.

Never trust rc == 0
-------------------
A cell exited 0 in six seconds on 2026-08-20 after a silently-failed
hot-start, and the queue read that as success and deleted eleven hours of
checkpoints. `verdict()` therefore ignores the exit code unless the log
also carries a completion banner, and reports tracebacks, OOM kills and
disk-full markers on their own.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cluster import vmlib as V  # noqa: E402

# Code we push. cache/ and data/ are excluded: they hold the 4.5 GB
# latency CSV and friends, which are already on the VM's EBS volume and
# must not be re-shipped on every code change.
PUSH_EXCLUDES = [
    '.git', '__pycache__', '*.pyc', '.DS_Store',
    'cache', 'data', 'runs', 'logs', 'figures', 'dashboard_site',
    'old_scripts', 'old_handoffs', '.claude', 'dashboard_site',
]

DATA_FILES = [
    'data/vultr_peers_inferred.csv',
    'cache/vultr_ingress_latencies_by_dst.csv',
    'cache/vultr_anycast_latency_smaller.csv',
    'cache/vultr_provider_popps.csv',
]


# ------------------------------------------------------------- presets --

def preset_dpsweep(a, run_id):
    """evaluations/evaluate_over_deployment_sizes.py -- Tom's own sweep.

    Not the run_deployment_sweep.py fork: the fork's three env knobs are
    now CLI flags on the real evaluation, and the `[mem]`/`[sweep]`
    timing instrumentation was ported across on 2026-08-21.
    """
    argv = [V.REMOTE_PY, '-u',
            'evaluations/evaluate_over_deployment_sizes.py',
            '--port', str(a.port),
            '--cache-fn', 'cache/cluster_runs/{}/metrics_by_dpsize.pkl'.format(run_id),
            '--figures-subdir', 'cluster/{}'.format(run_id)]
    if a.dpsizes:
        argv += ['--dpsizes', a.dpsizes]
    if a.nsim:
        argv += ['--nsim', a.nsim]
    if a.max_iter:
        argv += ['--max-iter', str(a.max_iter)]
    # Measurement budget. Forwarded as flags rather than set as env vars so
    # the sweep's own validation runs: it fail-fasts on a typo'd budget,
    # and it implies --probe-mode smart -- a budget left on the default
    # post_step mode is silently no budget at all.
    if a.probe_n:
        argv += ['--probe-n', str(a.probe_n)]
    if a.probe_mode:
        argv += ['--probe-mode', a.probe_mode]
    if a.plot:
        argv += ['--plot']
    env = {
        'PYTHONUNBUFFERED': '1',
        'SCULPTOR_SWEEP_PROGRESS_JSON':
            '{}/{}/progress.json'.format(V.REMOTE_RUNS, run_id),
        # A sweep whose sparse solve dies is a ladder of baselines with the
        # system under evaluation missing -- abort the cell loudly instead
        # of spending the eval phases on it (the 2026-08-22 nsim=20 run
        # burned ~50 min producing 19/20 sparse-less sims before the log
        # was read). Override with --env SCULPTOR_REQUIRE_SOLNS= to allow
        # baseline-only runs.
        'SCULPTOR_REQUIRE_SOLNS': 'sparse',
    }
    if a.nocache:
        # SCULPTOR_RUN_TAG namespaces wrapper_eval's per-size metrics
        # pickle (`popp_failure_latency_comparison_<dpsize><_tag>.pkl`), so
        # every size gets a virgin cache path and actually trains.
        #
        # This matters far more for a TIMING run than a results run: a
        # cached size returns in about a second without logging a single
        # LEARNING ITERATION, and that 1 s lands in the timing tables as
        # if it were a measurement. Seeding the pricing model with it
        # would be worse than having no number at all.
        #
        # Namespacing rather than deleting is deliberate -- the old caches
        # stay on disk untouched, so nothing is destroyed to get a clean
        # measurement.
        env['SCULPTOR_RUN_TAG'] = run_id.replace('-', '_')
    if a.objsize:
        env['SCULPTOR_LOG_OBJSIZE'] = '1'
    # Point the per-worker mem/objsize logs at the run dir instead of /tmp,
    # so the ordinary run-dir harvest collects them. Ray DEDUPLICATES
    # worker stdout, so the driver log carries one worker's census plus a
    # "[repeated Nx across cluster]" marker -- the per-worker files are the
    # only place the actual per-worker distribution survives.
    env['SCULPTOR_WORKER_MEM_LOG_DIR'] = '{}/{}/workers'.format(
        V.REMOTE_RUNS, run_id)
    if a.shards and a.shards != 'off':
        # Without this, deployment setup falls back to a SERIAL Python loop
        # over the 65M-row, 4.5 GB latency CSV -- re-parsed once per
        # deployment size. With it, core/fork_load.py's array fast path
        # loads per-pop binary shards instead (~5x, byte-exact gated at
        # 5/10/16/20/26 pops per core/deployment_setup.py).
        #
        # Not defaulted ON: it changes what the timing run is measuring,
        # and a timing baseline that silently switches code paths mid-study
        # is worthless. `launch` warns when shards exist but are unused, so
        # the choice is visible rather than forgotten.
        env['SCULPTOR_LAT_SHARDS'] = a.shards
    elif a.shards == 'off':
        # Empty string is the documented "deliberately off" value; unset
        # would now mean "use the default", which is the opposite.
        env['SCULPTOR_LAT_SHARDS'] = ''
    pulls = [
        'cache/cluster_runs/{}/'.format(run_id),
        'figures/cluster/{}/'.format(run_id),
        # The per-size checkpoint pickles -- metrics['adv'] (every solved
        # advertisement) and metrics['deployment'] per sim live in these,
        # and they are written after EVERY strategy/eval phase. They land
        # in cache/ (outside the run dir), so without this entry they were
        # never harvested and a stopped box silently ate them
        # (Tom 2026-08-22: losing these = the run's money wasted).
        'cache/popp_failure_latency_comparison_*{}*.pkl'.format(
            '_' + run_id.replace('-', '_') if a.nocache else ''),
    ]
    return argv, env, pulls


def preset_papertable(a, run_id):
    """evaluations/generate_paper_table.py -- every objective at one size.

    --dpsizes carries the (single) size, --nsim the number of deployments,
    --max-iter the training iters; the driver then runs one training+eval
    cell per objective (avg_latency, per_site_cost, max_util,
    frac_beyond_optimal, joint_priority) and emits the two tables.
    """
    dpsize = (a.dpsizes or '32')
    if ',' in dpsize:
        raise SystemExit('papertable takes ONE size (got {})'.format(dpsize))
    tag = run_id.replace('-', '_')
    argv = [V.REMOTE_PY, '-u', 'evaluations/generate_paper_table.py',
            '--dpsize', dpsize,
            '--number_of_deployments', str(a.nsim or 1),
            '--num_training_iter', str(a.max_iter or 150),
            '--run_id', tag,
            '--out', 'figures/cluster/{}/paper_table'.format(run_id)]
    if getattr(a, 'hotstart', ''):
        argv += ['--hotstart', a.hotstart]
    env = {
        'PYTHONUNBUFFERED': '1',
        'SCULPTOR_REQUIRE_SOLNS': 'sparse',
        # per-worker mem logs into the run dir (same rationale as dpsweep)
        'SCULPTOR_WORKER_MEM_LOG_DIR': '{}/{}/workers'.format(
            V.REMOTE_RUNS, run_id),
    }
    pulls = [
        'figures/cluster/{}/'.format(run_id),
        # per-objective cell logs + sweep caches (cache/, outside run dir)
        'cache/table_generate_{}*'.format(tag),
        # the L1 checkpoint pickles: solved advertisements + metrics per
        # objective -- losing these wastes the run (same as dpsweep)
        'cache/popp_failure_latency_comparison_*{}*.pkl'.format(tag),
        'cache/paper_table_condensed_*{}*.pkl'.format(tag),
    ]
    return argv, env, pulls


PRESETS = {'dpsweep': preset_dpsweep,
           'papertable': preset_papertable}


# --------------------------------------------------------------- push --

def cmd_push(a):
    inst = V.resolve(a.ref)
    _require_running(inst)

    # REFUSE TO PUSH OVER A LIVE RUN. Ray actors already alive keep the
    # code they imported, but a sweep that moves to the next deployment
    # size spawns FRESH actors -- which would pick up the new code
    # mid-experiment. That silently makes sizes incomparable: on
    # 2026-08-21 a pending MC_NUM default change (5 -> 1, an estimator
    # change, not just a speed knob) would have applied to actual-32 while
    # actual-25 ran at the old value, inside one experiment.
    live = [m for m in V.live_runs(inst['id'])]
    if live and not a.force:
        print('REFUSING TO PUSH -- {} live run(s) on {}:'.format(
            len(live), inst['id']))
        for m in live:
            print('  {}  ({})'.format(m['run_id'], ' '.join(m.get('cmd', []))[:70]))
        print('\nCode changes would reach any actors spawned from here on '
              '(e.g. the next deployment size),\nmaking sizes within this '
              'run incomparable.')
        print('\n  wait for it to finish, or:')
        print('  python -m cluster.expctl push {} --force   '
              '# only if the change cannot affect results'.format(a.ref))
        return 2
    print('pushing code {} -> {}:{}'.format(V.REPO, inst['id'], V.REMOTE_REPO))
    rc, out, err = V.rsync(V.REPO.rstrip('/') + '/',
                           V.remote(inst['ip'], V.REMOTE_REPO + '/'),
                           excludes=PUSH_EXCLUDES)
    if rc != 0:
        sys.stderr.write(err)
        raise SystemExit('code push failed rc={}'.format(rc))
    print(out.strip())

    if a.data:
        for rel in DATA_FILES:
            src = os.path.join(V.REPO, rel)
            if not os.path.exists(src):
                print('  data: MISSING locally, skipped: {}'.format(rel))
                continue
            print('  data: {} ({})'.format(
                rel, V.human_bytes(os.path.getsize(src))))
            V.ssh(inst['ip'], 'mkdir -p {}'.format(
                shlex.quote(os.path.join(V.REMOTE_REPO,
                                         os.path.dirname(rel)))))
            rc, out, err = V.rsync(src, V.remote(
                inst['ip'], os.path.join(V.REMOTE_REPO, rel)))
            if rc != 0:
                sys.stderr.write(err)
                raise SystemExit('data push failed for {}'.format(rel))

    # The runtime scaffolding the codebase assumes exists. Reproduced from
    # ray-cluster.yaml's setup_commands so a plain start+push works
    # without going through the autoscaler.
    scaffold = '''
cd {repo}
mkdir -p figures logs runs cache cache/deployments cache/cluster_runs {runs}
touch cache/addresses_violating_sol.csv
for d in venv312 cache data figures logs runs; do
  [ -e /home/ubuntu/$d ] || true
done
ln -sfn /home/ubuntu/venv312 /home/ubuntu/venv 2>/dev/null || true
python3 -c "import sys; print('scaffold ok')"
'''.format(repo=V.REMOTE_REPO, runs=V.REMOTE_RUNS)
    rc, out, err = V.ssh(inst['ip'], scaffold)
    print('  ' + (out or err).strip().splitlines()[-1] if (out or err).strip()
          else '  scaffold ok')

    rc, out, _ = V.ssh(inst['ip'],
                       'cd {} && git rev-parse --short HEAD 2>/dev/null; '
                       'df -h / | tail -1'.format(V.REMOTE_REPO))
    print(out.strip())
    return 0


# ------------------------------------------------------------- launch --

_LAUNCHER = '''#!/bin/bash
# generated by cluster/expctl.py -- do not edit on the VM
cd {repo} || exit 90
mkdir -p {rundir} {rundir}/workers {repo}/cache/cluster_runs/{run_id} {repo}/figures/cluster/{run_id}
# The script records its OWN pid. Reading it back from `echo $!` over ssh
# was unreliable: the launch channel does not always close promptly once
# the job is detached, so the launcher could time out on a run that had in
# fact started perfectly (2026-08-21, first real launch).
echo $$ > {rundir}/run.pid
rm -f {rundir}/run.rc
exec {redir} {rundir}/run.log 2>&1
echo "[expctl] run_id={run_id}"
echo "[expctl] host=$(hostname) started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[expctl] git=$(git rev-parse --short HEAD 2>/dev/null || echo none)"
echo "[expctl] cores=$(nproc) mem_gb=$(free -g | awk '/^Mem:/{{print $2}}')"
echo "[expctl] disk=$(df -h / | tail -1)"
echo "[expctl] cmd={cmd_display}"
echo "[expctl] ---------------------------------------------------------"
{exports}
# Disk/RAM sampler: a JSONL sample a minute, and one shout into the log the
# first time free space drops under the floor. Deliberately non-destructive
# -- it tells you the run is about to die rather than deciding for you.
# It inherits the exec'd stdout, so its DISKLOW line shares the log's file
# offset instead of racing the driver with a second `>>` handle.
(
  warned=0
  while true; do
    avail=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
    memav=$(awk '/^MemAvailable:/{{print int($2/1024)}}' /proc/meminfo)
    echo "{{\\"t\\":$(date +%s),\\"disk_avail_gb\\":${{avail:-0}},\\"mem_avail_mb\\":${{memav:-0}}}}" >> {rundir}/sysmon.jsonl
    if [ "${{avail:-999}}" -lt {disk_floor} ] && [ "$warned" -eq 0 ]; then
      echo "[expctl] DISKLOW only ${{avail}}GB free on / (floor {disk_floor}GB)"
      warned=1
    fi
    sleep 60
  done
) &
SAMPLER=$!
{cmd}
rc=$?
kill $SAMPLER 2>/dev/null
echo "[expctl] ---------------------------------------------------------"
echo "[expctl] exit_rc=$rc finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "$rc" > {rundir}/run.rc
'''


def cmd_launch(a):
    inst = V.resolve(a.ref)
    if not a.dry_run:          # --dry-run is for reading the plan on a cold box
        _require_running(inst)

    if getattr(a, 'resume', None):
        # Continue an existing experiment: same run_id, so the same
        # --cache-fn (the metrics pickle ACCUMULATES sizes), the same
        # SCULPTOR_RUN_TAG, the same remote dir and the same dashboard
        # section. Pair it with --dpsizes listing only the sizes still
        # missing: re-entering a completed size would hit its cache,
        # return in ~1s, and overwrite that size's real phase timings
        # with a disk read.
        prev = V.load_manifest(a.resume)
        run_id = prev['run_id']
        label = prev.get('label') or a.label or 'run'
    else:
        prev = None
        label = a.label or a.preset or 'run'
        run_id = '{}-{}'.format(time.strftime('%Y%m%d_%H%M%S'), label)
    rundir = '{}/{}'.format(V.REMOTE_RUNS, run_id)

    if prev and not a.preset:
        # A resume continues the same experiment; making the caller retype
        # --preset invites retyping it WRONG, which would silently change
        # what the run is measuring.
        a.preset = prev.get('preset')
    if a.preset:
        argv, env, pulls = PRESETS[a.preset](a, run_id)
    else:
        if not a.cmd:
            raise SystemExit('give --preset or a raw command after --')
        argv, env, pulls = list(a.cmd), {}, []
    if prev:
        # a resume must not silently DROP env the original segment ran
        # with (2026-08-25: SCULPTOR_HOTSTART_RUN_DIR was supplied via
        # --env on segment 1, the resume rebuilt env from the preset
        # alone, and actual-32 would have retrained from scratch).
        # Preset-computed keys keep their fresh values; everything else
        # inherits. Explicit --env below still overrides both.
        for k, v in (prev.get('env') or {}).items():
            env.setdefault(k, v)
    for kv in a.env or []:
        k, _, v = kv.partition('=')
        env[k] = v
    for extra in a.pull or []:
        pulls.append(extra)

    cmd = ' '.join(shlex.quote(x) for x in argv)
    exports = '\n'.join('export {}={}'.format(k, shlex.quote(str(v)))
                        for k, v in sorted(env.items()))
    script = _LAUNCHER.format(repo=V.REMOTE_REPO, rundir=rundir,
                              redir='>>' if prev else '>',
                              run_id=run_id, cmd=cmd,
                              cmd_display=cmd.replace('"', "'"),
                              exports=exports,
                              disk_floor=a.disk_floor)

    if a.preset == 'dpsweep' and not a.dry_run:
        rc, out, _ = V.ssh(inst['ip'],
                           'test -f {}/{}/manifest.json && echo HAVE || '
                           'echo NONE'.format(
                               V.REMOTE_REPO,
                               (a.shards if a.shards != 'off'
                                else 'cache/lat_shards')))
        if a.shards == 'off':
            print('NOTE: --shards off -- deployment setup will use the '
                  'LEGACY SERIAL loop over the 4.5GB CSV, once per size.\n')
        elif 'HAVE' not in out:
            # Silently falling back would make a run look like a fast-path
            # run in every respect except its wall clock.
            print('WARNING: shard dir {} not found on this VM, so setup '
                  'will FALL BACK to the serial 4.5GB CSV loop.\n'
                  '         Build them with core/convert_latencies.py, or '
                  'pass --shards off to make that explicit.\n'.format(
                      a.shards))

    print('run_id  {}'.format(run_id))
    print('vm      {} ({}) {}'.format(inst['id'], inst['type'], inst['ip']))
    print('cmd     {}'.format(cmd))
    if env:
        print('env     {}'.format(' '.join(
            '{}={}'.format(k, v) for k, v in sorted(env.items()))))

    if a.dry_run:
        print('\n--- launcher (dry run, nothing started) ---')
        print(script)
        return 0

    V.ssh(inst['ip'], 'mkdir -p {}'.format(shlex.quote(rundir)), check=True)
    p = subprocess.run(V.ssh_argv(inst['ip'],
                                  'cat > {}/run.sh && chmod +x {}/run.sh'
                                  .format(rundir, rundir)),
                       input=script, text=True, capture_output=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit('could not write launcher')

    # REGISTER BEFORE STARTING. If the launch ssh misbehaves after the job
    # is already running, an unregistered run is an unharvested run -- the
    # exact failure this whole tool exists to prevent. A manifest for a job
    # that never started is harmless by comparison.
    m = V.save_manifest({
        'run_id': run_id, 'label': label, 'preset': a.preset,
        'instance_id': inst['id'], 'instance_type': inst['type'],
        'launch_ip': inst['ip'],
        'remote_dir': rundir, 'remote_log': rundir + '/run.log',
        'remote_repo': V.REMOTE_REPO,
        'cmd': argv, 'env': env, 'pull': pulls,
        'pid': None, 'started_utc': V.utcnow(), 'started_epoch': time.time(),
        'state': 'launched', 'disk_floor_gb': a.disk_floor,
        'segments': (prev.get('segments') or []) + [{
            'started_utc': prev.get('started_utc'),
            'finished_epoch': prev.get('finished_epoch'),
            'state': prev.get('state'),
            'cmd': prev.get('cmd')}] if prev else [],
        'first_started_utc': (prev.get('first_started_utc')
                              or prev.get('started_utc')) if prev else None,
    })

    # setsid + nohup: survives the ssh session, the laptop sleeping, and
    # wifi dropping. `< /dev/null` so nothing blocks on stdin. The short
    # timeout is deliberate -- the channel sometimes stays open after the
    # job detaches, and waiting on it proves nothing. run.sh writes its own
    # pidfile, so that is what we read back.
    try:
        V.ssh(inst['ip'],
              'setsid nohup {r}/run.sh < /dev/null > /dev/null 2>&1 & '
              'disown 2>/dev/null; exit 0'.format(r=rundir), timeout=25)
    except subprocess.TimeoutExpired:
        pass

    pid = None
    for _ in range(12):
        rc, out, _ = V.ssh(inst['ip'],
                           'cat {}/run.pid 2>/dev/null'.format(rundir),
                           timeout=25)
        if out.strip():
            pid = out.strip().splitlines()[-1]
            break
        time.sleep(5)
    if pid is None:
        print('\nWARNING: no pidfile after 60s. The run is REGISTERED, so '
              'nothing is orphaned -- check it with:\n'
              '  python -m cluster.expctl status {}'.format(run_id))
    m['pid'] = pid
    m['state'] = 'running' if pid else 'launched'
    # a resume supersedes any terminal verdict from the previous segment
    m.pop('finished_epoch', None)
    m.pop('verdict', None)
    V.save_manifest(m)
    V.update_alert(sweep={'run_id': run_id,
                          'pid_file': rundir + '/run.pid',
                          'log_file': m['remote_log'],
                          'outcome': 'in flight since ' + m['started_utc']},
                   note='expctl launch {} on {}'.format(run_id, inst['id']))

    print('pid     {}'.format(pid))
    print('log     {}'.format(m['remote_log']))
    print('local   {}'.format(V.run_dir(run_id)))
    print('\nwatch it:')
    print('  python -m cluster.expctl watch {} --interval 300'.format(run_id))
    print('follow the log live:')
    print('  ssh -i {} {}@{} tail -f {}'.format(
        V.SSH_KEY, V.SSH_USER, inst['ip'], m['remote_log']))
    return 0


# ------------------------------------------------------------ harvest --

def harvest(m, ip, verbose=True):
    """Pull the run dir and every declared result path. Returns a dict.

    Called by status/watch/kill and, critically, by `vmctl stop` before it
    will stop the box.
    """
    d = V.run_dir(m['run_id'])
    os.makedirs(os.path.join(d, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(d, 'results'), exist_ok=True)
    got = {'logs': None, 'results': [], 'errors': []}

    rc, out, err = V.rsync(V.remote(ip, m['remote_dir'].rstrip('/') + '/'),
                           os.path.join(d, 'logs') + '/')
    if rc != 0:
        got['errors'].append('log rsync rc={}: {}'.format(rc, err.strip()[:300]))
    else:
        got['logs'] = out.strip()

    for rel in m.get('pull', []):
        src = rel if rel.startswith('/') else \
            os.path.join(m.get('remote_repo', V.REMOTE_REPO), rel)
        dst = os.path.join(d, 'results', rel.rstrip('/').replace('/', '__'))
        if '*' in rel or '?' in rel:
            # Glob entry: the remote shell expands it, so the local target
            # must be a directory that can hold several matches.
            dst = os.path.join(d, 'results',
                               re.sub(r'[*?]', '', rel.rstrip('/'))
                               .replace('/', '__').rstrip('_'))
            os.makedirs(dst, exist_ok=True)
            dst += '/'
        elif rel.endswith('/'):
            os.makedirs(dst, exist_ok=True)
            dst += '/'
        rc, out, err = V.rsync(V.remote(ip, src), dst)
        if rc == 0:
            got['results'].append(rel)
        elif rc in (23, 24):
            # 23/24 = source did not exist yet. Normal early in a run.
            pass
        else:
            got['errors'].append('{} rsync rc={}: {}'.format(
                rel, rc, err.strip()[:200]))

    with open(os.path.join(d, 'harvest.json'), 'w') as fh:
        json.dump({'last_harvest_utc': V.utcnow(), **got}, fh, indent=1)
    if verbose:
        lp = V.local_log(m['run_id'])
        print('harvested -> {} (log {})'.format(
            d, V.human_bytes(os.path.getsize(lp))
            if os.path.exists(lp) else 'none yet'))
        for e in got['errors']:
            print('  WARN {}'.format(e))
    return got


def cmd_pull(a):
    m = V.load_manifest(a.run_id)
    inst = V.describe(m['instance_id'])[0]
    if inst['state'] != 'running':
        raise SystemExit('{} is {} -- nothing to pull from'.format(
            inst['id'], inst['state']))
    harvest(m, inst['ip'])
    return 0


# ------------------------------------------------------------- status --

_TB = re.compile(r'^Traceback \(most recent call last\)', re.M)
_ERR = re.compile(r'^(\w*(?:Error|Exception|Interrupt)):', re.M)


def _current_segment(log_text):
    """A resumed run APPENDS to run.log, so the file holds every segment.

    Judging the live state on the whole file reports old segments' failures
    as if they were current -- on 2026-08-21 a healthy resumed run showed
    "3 traceback(s)" that all predated the relaunch. The verdict is about
    what is happening NOW, so it reads from the last launch banner.
    """
    i = log_text.rfind('[expctl] run_id=')
    return (log_text[i:], log_text[:i]) if i > 0 else (log_text, '')


def verdict(log_text, rc, killed=False, alive=None):
    """What actually happened, judged from the log rather than the rc.

    Returns (state, headline, details). `rc == 0` alone is never enough:
    the 2026-08-20 incident was a six-second rc-0 "success" that had
    silently failed its hot-start.
    """
    details = []
    log_text, prior = _current_segment(log_text)
    n_prior_tb = len(_TB.findall(prior))
    done = '[sweep] ALL DONE' in log_text
    n_tb = len(_TB.findall(log_text))
    errs = {}
    for mm in _ERR.finditer(log_text):
        errs[mm.group(1)] = errs.get(mm.group(1), 0) + 1
    if n_tb:
        details.append('{} traceback(s): {}'.format(
            n_tb, ', '.join('{}x{}'.format(v, k) for k, v in
                            sorted(errs.items(), key=lambda x: -x[1])[:4])
            or 'unclassified'))
    if 'DISKLOW' in log_text:
        details.append('DISK FLOOR BREACHED -- see the DISKLOW line')
    if 'No space left on device' in log_text:
        details.append('DISK FULL ("No space left on device")')
    if 'Killed' in log_text or 'MemoryError' in log_text:
        details.append('possible OOM (Killed / MemoryError in log)')
    for mm in re.finditer(r'\[sweep\] dpsize=(\d+) FAILED', log_text):
        details.append('dpsize {} failed'.format(mm.group(1)))
    if n_prior_tb:
        details.append('({} traceback(s) in EARLIER segments of this '
                       'resumed run -- not current)'.format(n_prior_tb))

    if rc is None:
        # No exit code was written. `kill` ends run.sh itself, so the
        # launcher's trailing `echo $rc > run.rc` never executes -- and the
        # same is true of an OOM kill or the box disappearing underneath it.
        # A process we KNOW is dead must never be reported as 'in flight':
        # that is a stale-but-plausible state, which is the whole failure
        # class these tools exist to remove.
        if killed:
            return ('killed', 'killed by operator (no exit code written)',
                    details)
        if alive is False:
            return ('died', 'process is gone and wrote no exit code -- '
                            'killed, OOM-killed, or the box went away',
                    details)
        return ('running', 'in flight', details)
    if done and rc == 0 and not details:
        return ('done', 'completed clean', details)
    if done and rc == 0:
        # Finished, but something in the log wants reading before the
        # numbers are trusted -- a traceback, a disk warning, a failed size.
        return ('done-dirty', 'completed, but the log carries warnings',
                details)
    if rc == 0:
        return ('suspect',
                'exited 0 WITHOUT a completion banner -- do not trust this',
                details)
    return ('failed', 'exited rc={}'.format(rc), details)


def _read_local(run_id, name):
    p = os.path.join(V.run_dir(run_id), 'logs', name)
    if not os.path.exists(p):
        return None
    try:
        return open(p, errors='replace').read()
    except IOError:
        return None


def status(m, pull=True, tail=25, quiet=False):
    inst = V.describe(m['instance_id'])[0]
    out = {'run_id': m['run_id'], 'instance': inst}
    if inst['state'] == 'running' and pull:
        try:
            harvest(m, inst['ip'], verbose=False)
        except Exception as e:                    # noqa: BLE001
            out['harvest_error'] = str(e)

    # A VM that is not running cannot be running our process. Leaving this
    # as None let verdict() fall through to 'in flight' for a run whose box
    # had already been stopped -- the same stale-but-plausible state as the
    # killed-run bug, arriving by a different route (2026-08-21, when the
    # instance was stopped externally mid-ladder).
    alive = False if inst['state'] != 'running' else None
    if inst['state'] == 'running':
        # The manifest's pid can be a PREVIOUS segment's (a resume that
        # failed to record its pid probed a dead pid, verdicted 'died',
        # and the watch stamped a live run terminal -- 2026-08-24, twice).
        # run.pid on the box is written by the launcher at every
        # (re)launch: it is the authority. Fall back to the manifest.
        rc, o, _ = V.ssh(inst['ip'],
                         'P=$(cat {}/run.pid 2>/dev/null); P=${{P:-{}}}; '
                         'echo PID=$P; kill -0 $P 2>/dev/null '
                         '&& echo ALIVE || echo DEAD'
                         .format(m['remote_dir'], m.get('pid', '0')))
        alive = 'ALIVE' in o
        mm = re.search(r'PID=(\d+)', o)
        if mm and str(m.get('pid')) != mm.group(1):
            m['pid'] = int(mm.group(1))
            V.save_manifest(m)
        rc, o, _ = V.ssh(inst['ip'], "df -BG --output=avail / | tail -1")
        out['disk_avail'] = o.strip()

    log = _read_local(m['run_id'], 'run.log') or ''
    rcfile = _read_local(m['run_id'], 'run.rc')
    exit_rc = int(rcfile.strip()) if rcfile and rcfile.strip().lstrip('-').isdigit() else None
    if alive:
        exit_rc = None
    state, headline, details = verdict(
        log, exit_rc, killed=(m.get('state') == 'killed' and not alive),
        alive=alive)
    out.update({'alive': alive, 'exit_rc': exit_rc, 'state': state,
                'headline': headline, 'details': details,
                'log_bytes': len(log)})

    prog = _read_local(m['run_id'], 'progress.json')
    if prog:
        try:
            out['progress'] = json.loads(prog)
        except ValueError:
            pass

    elapsed = time.time() - m.get('started_epoch', time.time())
    out['elapsed_s'] = elapsed
    out['cost_usd'] = V.cost_usd(m.get('instance_type', ''), elapsed)

    if quiet:
        return out

    print('=' * 72)
    print('{}   [{}] {}'.format(m['run_id'], state.upper(), headline))
    print('=' * 72)
    print('  vm        {} {} {} ({})'.format(
        inst['id'], inst['type'], inst['state'], inst['ip'] or '-'))
    print('  process   {}'.format(
        'alive (pid {})'.format(m.get('pid')) if alive else
        ('exit_rc={}'.format(exit_rc) if exit_rc is not None else 'unknown')))
    print('  elapsed   {}   ~${:.2f} of compute so far'.format(
        V.human_dt(elapsed), out['cost_usd'] or 0))
    if out.get('disk_avail'):
        print('  disk free {}'.format(out['disk_avail']))
    print('  log       {} local'.format(V.human_bytes(len(log))))
    for d in details:
        print('  !! {}'.format(d))
    p = out.get('progress')
    if p:
        sizes = p.get('sizes', [])
        done = p.get('done', {})
        print('  progress  {}/{} sizes  (phase={})'.format(
            len(done), len(sizes), p.get('phase')))
        for s in sizes:
            e = done.get(str(s))
            if e is None:
                mark = 'running' if p.get('current') == s else '-'
                print('     actual-{:<3} {}'.format(s, mark))
            else:
                print('     actual-{:<3} {:<8} {}'.format(
                    s, 'ok' if e.get('ok') else 'FAILED',
                    V.human_dt(e.get('wall_s'))))
    if tail and log:
        print('  --- last {} lines '.format(tail) + '-' * 40)
        # tqdm redraws with \r; splitting on it is the difference between
        # seeing "0%" forever and seeing the actual position.
        lines = log.replace('\r', '\n').rstrip().splitlines()
        for line in lines[-tail:]:
            print('  | ' + line[:200])
    print('  local     {}'.format(V.run_dir(m['run_id'])))
    return out


def cmd_status(a):
    if a.run_id:
        status(V.load_manifest(a.run_id), pull=not a.no_pull, tail=a.tail)
        return 0
    runs = V.live_runs()
    if not runs:
        print('no live runs. `expctl list` shows finished ones.')
        return 0
    for m in runs:
        status(m, pull=not a.no_pull, tail=a.tail)
    return 0


def cmd_list(a):
    runs = V.all_runs()
    if not runs:
        print('no runs registered under {}'.format(V.RUNS_DIR))
        return 0
    print('{:<30} {:<12} {:<21} {:<10} {}'.format(
        'RUN ID', 'STATE', 'INSTANCE', 'ELAPSED', 'CMD'))
    for m in runs:
        el = m.get('finished_epoch', time.time()) - m.get('started_epoch', 0)
        print('{:<30} {:<12} {:<21} {:<10} {}'.format(
            m['run_id'], m.get('state', '?'), m.get('instance_id', '-'),
            V.human_dt(el), ' '.join(m.get('cmd', []))[:60]))
    return 0


# -------------------------------------------------------------- watch --

def cmd_watch(a):
    m = V.load_manifest(a.run_id)
    t0 = time.time()
    while True:
        st = status(m, pull=True, tail=a.tail)
        if st.get('alive') is False or st['instance']['state'] != 'running':
            print('\nprocess is gone -- final harvest')
            inst = st['instance']
            if inst['state'] == 'running':
                harvest(m, inst['ip'])
                rb, lb, ok = V.harvest_gap(m, inst['ip'])
                if ok and rb > lb:
                    print('WARNING: {} bytes still only on the VM'.format(rb - lb))
                else:
                    print('log fully harvested ({})'.format(V.human_bytes(lb)))
            # Segment guard (2026-08-24): a watch bound to a KILLED
            # segment can reach this line after a resume has already
            # relaunched the run -- stamping 'died' over a live process
            # (both dashboards showed dead runs while 23 processes
            # trained). Re-read the manifest; if the pid changed, this
            # watch's verdict belongs to a previous segment: drop it.
            _fresh = V.load_manifest(m['run_id'])
            if _fresh and _fresh.get('pid') != m.get('pid'):
                print('\n[watch] run was relaunched (pid {} -> {}) -- '
                      'verdict belongs to the old segment, not stamping.'
                      .format(m.get('pid'), _fresh.get('pid')))
                return 0
            m['state'] = st['state']
            m['finished_epoch'] = time.time()
            m['verdict'] = st['headline']
            V.save_manifest(m)
            print('\nVERDICT: {} -- {}'.format(st['state'].upper(),
                                               st['headline']))
            print('\nWhen you are done looking, stop the box:')
            print('  python -m cluster.vmctl stop {}'.format(m['instance_id']))
            return 0 if st['state'] in ('done',) else 1
        if a.max_wait and time.time() - t0 > a.max_wait:
            print('\n--max-wait reached; run still going. Nothing stopped.')
            return 0
        time.sleep(a.interval)


def cmd_kill(a):
    m = V.load_manifest(a.run_id)
    inst = V.describe(m['instance_id'])[0]
    if inst['state'] == 'running':
        # Kill the whole PROCESS GROUP, not just pid + direct children:
        # generate_paper_table's cell subprocesses (and their Ray fleets)
        # are grandchildren -- pkill -P orphaned them, and up to three
        # zombie cells trained for hours on stale code, interleaving into
        # the live cell logs (2026-08-25).
        V.ssh(inst['ip'], 'PG=$(ps -o pgid= -p {pid} | tr -d " "); '
                          'pkill -TERM -P {pid} 2>/dev/null; kill -TERM {pid} '
                          '2>/dev/null; sleep 5; '
                          '[ -n "$PG" ] && kill -KILL -- -$PG 2>/dev/null; '
                          'kill -KILL {pid} 2>/dev/null; '
                          'true'.format(pid=m.get('pid', 0)))
        print('sent TERM/KILL to pid {}'.format(m.get('pid')))
        harvest(m, inst['ip'])
    m['state'] = 'killed'
    m['finished_epoch'] = time.time()
    V.save_manifest(m)
    return 0


def cmd_ack(a):
    """Deliberately leave a VM up for a bounded window.

    Satisfies the Stop hook (`cluster/hooks/remind_live_vm.py`) so it stops
    firing every turn -- but only for this run, only until the deadline,
    and only while the run is genuinely still running. See that module's
    docstring for why all three limits matter.
    """
    m = V.load_manifest(a.run_id)
    until = time.time() + a.minutes * 60
    path = os.path.expanduser('~/.sculptor_cluster_alert/vm_ack.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump({'run_id': m['run_id'], 'instance_id': m.get('instance_id'),
                   'until': until, 'until_utc': time.strftime(
                       '%Y-%m-%dT%H:%M:%SZ', time.gmtime(until)),
                   'reason': a.reason, 'acked_utc': V.utcnow()}, fh, indent=1)
    print('{} acknowledged for {} min (until {} UTC)'.format(
        m['run_id'], a.minutes,
        time.strftime('%H:%M', time.gmtime(until))))
    print('reason: {}'.format(a.reason))
    print('The reminder returns after that, and immediately if the run '
          'stops before then.')
    return 0


def cmd_finish(a):
    m = V.load_manifest(a.run_id)
    inst = V.describe(m['instance_id'])[0]
    if inst['state'] == 'running':
        harvest(m, inst['ip'])
        rb, lb, ok = V.harvest_gap(m, inst['ip'])
        if ok and rb > lb:
            raise SystemExit('{} bytes still unharvested -- not marking '
                             'finished'.format(rb - lb))
    st = status(m, pull=False, tail=0, quiet=True)
    # Never overwrite a terminal state with a non-terminal one. `finish`
    # used to assign st['state'] unconditionally, which turned a run that
    # `kill` had correctly marked 'killed' back into 'running' -- the
    # manifest then claimed a dead process was in flight (2026-08-21).
    if not (st['state'] == 'running' and m.get('state') in
            ('killed', 'failed', 'done', 'done-dirty', 'suspect', 'died')):
        m['state'] = st['state']
    m['verdict'] = st['headline']
    m['finished_epoch'] = time.time()
    V.save_manifest(m)
    print('{}: {} -- {}'.format(m['run_id'], st['state'], st['headline']))
    return 0


# --------------------------------------------------------------- misc --

def _require_running(inst):
    if inst['state'] != 'running':
        raise SystemExit(
            '{} is {}. Start it first:\n  python -m cluster.vmctl start {}'
            .format(inst['id'], inst['state'], inst['id']))


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='expctl', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('push', help='rsync the repo (code only by default)')
    p.add_argument('ref')
    p.add_argument('--force', action='store_true',
                   help='push even though a run is live on the VM (see the '
                        'warning it prints -- this can make deployment '
                        'sizes within one run incomparable)')
    p.add_argument('--data', action='store_true',
                   help='also ship the big data/cache CSVs (slow; only '
                        'needed on a box that has never had them)')
    p.set_defaults(fn=cmd_push)

    p = sub.add_parser('launch', help='start a detached experiment')
    p.add_argument('ref')
    p.add_argument('--preset', choices=sorted(PRESETS))
    p.add_argument('--label', default=None)
    p.add_argument('--hotstart', default='',
                   help="papertable: 'obj:remote_runs_dir,...' passed "
                        "through to generate_paper_table --hotstart")
    p.add_argument('--resume', default=None, metavar='RUN_ID',
                   help='continue an existing run: same run_id, cache-fn, '
                        'run tag, remote dir and dashboard section; the log '
                        'is APPENDED. Pass --dpsizes with only the sizes '
                        'still missing.')
    p.add_argument('--dpsizes', default=None, help='e.g. 3,5,10')
    p.add_argument('--nsim', default=None, help='int or list parallel to --dpsizes')
    p.add_argument('--max-iter', type=int, default=None)
    p.add_argument('--probe-n', default=None,
                   help='measurement budget per solve(): an int, or the '
                        'literal "prefixes" for each deployment\'s own '
                        'prefix count. Implies --probe-mode smart.')
    p.add_argument('--probe-mode', default=None,
                   choices=['post_step', 'scheduled', 'slotted', 'gated',
                            'smart'],
                   help='WHEN-probing policy; post_step (stock) has no '
                        'budget.')
    p.add_argument('--port', type=int, default=31415)
    p.add_argument('--plot', action='store_true')
    p.add_argument('--env', action='append', help='K=V, repeatable')
    p.add_argument('--pull', action='append',
                   help='extra repo-relative path to harvest, repeatable')
    p.add_argument('--nocache', action='store_true',
                   help='force every size to actually train, by namespacing '
                        'the per-size metrics cache to this run id. REQUIRED '
                        'for honest timing: a cached size returns in ~1s and '
                        'reports that as its wall time. Deletes nothing.')
    p.add_argument('--shards', nargs='?', const='cache/lat_shards',
                   default='cache/lat_shards', metavar='DIR',
                   help='per-pop latency shard dir for deployment setup '
                        '(SCULPTOR_LAT_SHARDS). ON BY DEFAULT since '
                        '2026-08-21; pass --shards off for the legacy '
                        'SERIAL 4.5GB CSV loop.')
    p.add_argument('--objsize', action='store_true',
                   help='per-worker object-size census into the worker mem '
                        'logs (SCULPTOR_LOG_OBJSIZE=1)')
    p.add_argument('--disk-floor', type=int, default=25,
                   help='GB free below which the log shouts DISKLOW')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('cmd', nargs='*', help='raw command after -- (no preset)')
    p.set_defaults(fn=cmd_launch)

    p = sub.add_parser('status'); p.add_argument('run_id', nargs='?')
    p.add_argument('--no-pull', action='store_true')
    p.add_argument('--tail', type=int, default=25)
    p.set_defaults(fn=cmd_status)

    sub.add_parser('list').set_defaults(fn=cmd_list)

    p = sub.add_parser('pull'); p.add_argument('run_id')
    p.set_defaults(fn=cmd_pull)

    p = sub.add_parser('watch')
    p.add_argument('run_id')
    p.add_argument('--interval', type=int, default=300)
    p.add_argument('--tail', type=int, default=15)
    p.add_argument('--max-wait', type=int, default=0)
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser('kill'); p.add_argument('run_id')
    p.set_defaults(fn=cmd_kill)

    p = sub.add_parser('finish', help='final harvest + record the verdict')
    p.add_argument('run_id')
    p.set_defaults(fn=cmd_finish)

    p = sub.add_parser('ack', help='deliberately leave the VM up for a while')
    p.add_argument('run_id')
    p.add_argument('--minutes', type=int, default=30)
    p.add_argument('--reason', required=True,
                   help='why this run is being left to run')
    p.set_defaults(fn=cmd_ack)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    raise SystemExit(main())
