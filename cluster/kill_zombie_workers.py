#!/usr/bin/env python3
"""Kill zombie SCULPTOR Ray actor processes on the cluster's worker node(s).

When a driver crashes or is force-killed, the Ray actor processes it spawned
can be left running on the worker node — eating CPU/memory until something
explicitly reaps them. `ray.kill()` requires an ActorHandle, which we don't
have once the driver is gone, so we drop one level down and pkill the
actor processes by name.

The Ray actor processes show up in `ps` as
`ray::_LocalPathDistributionComputer*` or `ray::Path_Distribution_Computer*`.

Usage (from local Mac):
  python cluster/kill_zombie_workers.py            # kill all matching actor procs
  python cluster/kill_zombie_workers.py --dry-run  # report but don't kill

Discovers worker IPs via the AWS API, SSHs to each, pkills the matching
processes. Idempotent: if there are no orphan actors it's a quick no-op.
"""
import argparse
import os
import subprocess
import sys

AWS_BIN = os.path.expanduser('~/Documents/venv312/bin/aws')
SSH_KEY = os.path.expanduser('~/.ssh/ray-autoscaler_us-east-1.pem')

# What we match against. The Ray actor processes' argv contains the actor
# class name; pkill -f matches the whole argv.
ACTOR_PATTERN = '[r]ay::_LocalPathDistributionComputer|[r]ay::Path_Distribution_Computer'


def _run(cmd, timeout=60):
    """Returns (rc, stdout, stderr)."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, '', 'TIMEOUT'


def find_worker_ips():
    """Discover SCULPTOR worker-node IPs via AWS. The head is the
    m7g.4xlarge; workers are c7g.16xlarge (configured in cluster/ray-cluster.yaml)."""
    rc, out, err = _run(
        '{aws} ec2 describe-instances '
        '--filters "Name=tag:project,Values=sculptor" '
        '"Name=instance-state-name,Values=running" '
        '--query "Reservations[].Instances[].[InstanceType,PublicIpAddress]" '
        '--output text'.format(aws=AWS_BIN))
    if rc != 0:
        print('AWS describe-instances failed: {}'.format(err), file=sys.stderr)
        return []
    workers = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        itype, ip = parts[0], parts[1]
        # Worker pool is c7g.* per cluster/ray-cluster.yaml; head is m7g.*
        if itype.startswith('c7g.') or 'worker' in itype:
            workers.append((itype, ip))
    return workers


def kill_on_node(ip, dry_run=False):
    """SSH to a worker, count + (optionally) kill matching actor processes."""
    # First, count what's there.
    rc, out, err = _run(
        'ssh -i {key} -o StrictHostKeyChecking=no -o ConnectTimeout=15 '
        '-o BatchMode=yes ubuntu@{ip} '
        '"pgrep -afc \\"{pat}\\" 2>/dev/null || echo 0"'.format(
            key=SSH_KEY, ip=ip, pat=ACTOR_PATTERN),
        timeout=30)
    before = (out or '0').strip().splitlines()[-1] if out else '0'
    print('  {} (before): {} matching actor procs'.format(ip, before))
    if dry_run:
        rc, out, err = _run(
            'ssh -i {key} -o StrictHostKeyChecking=no -o ConnectTimeout=15 '
            '-o BatchMode=yes ubuntu@{ip} '
            '"pgrep -af \\"{pat}\\" 2>/dev/null | head -10"'.format(
                key=SSH_KEY, ip=ip, pat=ACTOR_PATTERN), timeout=30)
        if out.strip():
            for line in out.strip().splitlines():
                print('    | ' + line)
        return
    # Kill: SIGTERM first, then SIGKILL after grace period.
    _run(
        'ssh -i {key} -o StrictHostKeyChecking=no -o ConnectTimeout=15 '
        '-o BatchMode=yes ubuntu@{ip} '
        '"pkill -TERM -f \\"{pat}\\" 2>/dev/null; sleep 3; '
        ' pkill -KILL -f \\"{pat}\\" 2>/dev/null; sleep 1"'.format(
            key=SSH_KEY, ip=ip, pat=ACTOR_PATTERN), timeout=30)
    rc, out, err = _run(
        'ssh -i {key} -o StrictHostKeyChecking=no -o ConnectTimeout=15 '
        '-o BatchMode=yes ubuntu@{ip} '
        '"pgrep -afc \\"{pat}\\" 2>/dev/null || echo 0"'.format(
            key=SSH_KEY, ip=ip, pat=ACTOR_PATTERN), timeout=30)
    after = (out or '0').strip().splitlines()[-1] if out else '0'
    print('  {} (after):  {} matching actor procs'.format(ip, after))


def find_head_ip():
    """Discover the SCULPTOR head IP via AWS (m7g.* instance type)."""
    rc, out, err = _run(
        '{aws} ec2 describe-instances '
        '--filters "Name=tag:project,Values=sculptor" '
        '"Name=instance-state-name,Values=running" '
        '--query "Reservations[].Instances[].[InstanceType,PublicIpAddress]" '
        '--output text'.format(aws=AWS_BIN))
    if rc != 0:
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith('m7g.'):
            return parts[1]
    return None


def kill_head_orphan_drivers(head_ip, dry_run=False):
    """Find any orphan run_deployment_sweep.py drivers on the head.
    The "current" sweep is whichever PID is in /tmp/cluster_runs/latest.pid;
    anything else with that argv is an orphan from a previous kill that
    didn't fully tear down. These are head-side (driver) zombies,
    distinct from worker-node (actor) zombies.
    """
    # Get all matching PIDs and the current one.
    # Wrap remote command in SINGLE quotes so the local shell doesn't
    # expand $(cat ...) and $current -- they must run remotely.
    rc, out, err = _run(
        "ssh -i {key} -o StrictHostKeyChecking=no -o ConnectTimeout=15 "
        "-o BatchMode=yes ubuntu@{ip} "
        "'current=$(cat /tmp/cluster_runs/latest.pid 2>/dev/null || echo NONE); "
        " echo current_pid=$current; "
        " pgrep -af \"[r]un_deployment_sweep.py\"'".format(
            key=SSH_KEY, ip=head_ip), timeout=30)
    if rc != 0:
        print('  could not list head processes (rc={})'.format(rc))
        return
    current_pid = None
    orphans = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('current_pid='):
            current_pid = line.split('=', 1)[1]
            continue
        parts = line.split(None, 1)
        if len(parts) >= 1 and parts[0].isdigit():
            pid = parts[0]
            if pid != current_pid:
                orphans.append((pid, parts[1] if len(parts) > 1 else ''))
    print('  head {} (current pid={}, orphans={})'.format(head_ip, current_pid, len(orphans)))
    for pid, argv in orphans:
        print('    - pid={} {}'.format(pid, argv[:80]))
    if dry_run or not orphans:
        return
    pids = ' '.join(p for p, _ in orphans)
    _run(
        "ssh -i {key} -o StrictHostKeyChecking=no -o ConnectTimeout=15 "
        "-o BatchMode=yes ubuntu@{ip} "
        "'kill -9 {pids} 2>/dev/null; sleep 1; "
        " pgrep -af \"[r]un_deployment_sweep.py\" | wc -l'".format(
            key=SSH_KEY, ip=head_ip, pids=pids), timeout=30)
    print('  killed {} orphan driver(s) on head'.format(len(orphans)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='List what would be killed, but do not kill.')
    args = ap.parse_args()

    print('== head-side orphan drivers ==')
    head_ip = find_head_ip()
    if head_ip:
        kill_head_orphan_drivers(head_ip, dry_run=args.dry_run)
    else:
        print('  no head instance found via AWS')

    print()
    print('== worker-side orphan actors ==')
    workers = find_worker_ips()
    if not workers:
        print('  no worker nodes found via AWS')
    for itype, ip in workers:
        kill_on_node(ip, dry_run=args.dry_run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
