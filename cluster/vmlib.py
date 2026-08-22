"""Shared plumbing for the cluster tools (`vmctl.py`, `expctl.py`).

Everything that both CLIs need lives here: EC2 lookup, SSH/rsync wrappers,
the local run registry, and the cluster-alert JSON contract.

Design notes for whoever picks this up next
-------------------------------------------
* **boto3 direct, not Ray.** The autoscaler is a fine way to bring a fleet
  up from nothing, but for the one-VM-at-a-time workflow it is a slow,
  stateful layer over `start-instances`. These tools start an *existing*
  stopped instance and talk to it over plain SSH. Public IPs are assigned
  fresh on every start, so nothing may cache an IP across a stop.
* **The harvest contract is the point.** A run's log is the only artifact
  that explains a failure, and it lives on an ephemeral box. So: every
  status poll pulls, every kill pulls, and `vmctl stop` refuses to stop a
  VM whose runs have unharvested bytes. See `harvest_gap()`.
* **The alert JSON is load-bearing.** `~/.sculptor_cluster_alert/
  liveness_check.py` runs from cron every 10 min and pages Tom when the
  file disagrees with AWS. Any lifecycle transition must call
  `update_alert()` or it raises a false CRIT.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time

REGION = os.environ.get('SCULPTOR_AWS_REGION', 'us-east-1')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO, 'cache', 'cluster_runs')
ALERT_JSON = os.path.expanduser('~/.sculptor_cluster_alert/active_cluster.json')
SSH_KEY = os.path.expanduser('~/.ssh/ray-autoscaler_us-east-1.pem')
SSH_USER = 'ubuntu'
REMOTE_REPO = '/home/ubuntu/sparse_advertisements_code'
REMOTE_PY = '/home/ubuntu/venv312/bin/python'
REMOTE_RUNS = '/home/ubuntu/cluster_runs'

# On-demand $/hr, us-east-1, for the cost-so-far readout. Approximate by
# design -- this is for "am I burning money faster than I thought", not
# for billing. Spot is charged differently; the tools do not use spot.
HOURLY_USD = {
    'c8g.24xlarge': 3.83, 'c8g.16xlarge': 2.55, 'c8g.12xlarge': 1.91,
    'c8g.8xlarge': 1.28, 'c8g.4xlarge': 0.64,
    'c7g.16xlarge': 2.32, 'c7g.12xlarge': 1.74, 'c7g.8xlarge': 1.16,
    'c7g.4xlarge': 0.58,
    'm8g.16xlarge': 2.81, 'm8g.8xlarge': 1.41,
    'm7g.16xlarge': 2.61, 'm7g.12xlarge': 1.96, 'm7g.8xlarge': 1.31,
    'm7g.4xlarge': 0.65, 'm7g.2xlarge': 0.33, 'm7g.large': 0.08,
    'r8g.8xlarge': 1.67,
}

# EBS gp3 is $0.08/GB-month. Growing 50 -> 300 GB costs ~$0.03/hr, which
# is under 1% of a c8g.24xlarge and buys us out of the failure mode that
# has cost more compute than every other bug combined (2026-08-20: a full
# disk killed an 11h actual-32 run at iteration 125).
GP3_USD_PER_GB_MONTH = 0.08


# --------------------------------------------------------------- EC2 ---

def ec2():
    import boto3
    return boto3.client('ec2', region_name=REGION)


def describe(instance_id=None, include_terminated=False):
    """Return a list of instance dicts, newest-tagged first.

    Deliberately does NOT filter on tag:project -- the preflight box has
    no project tag and we still need to see it, since an untagged running
    instance is exactly the kind of thing that quietly bills for a week.
    """
    kw = {'InstanceIds': [instance_id]} if instance_id else {}
    out = []
    for res in ec2().describe_instances(**kw)['Reservations']:
        for i in res['Instances']:
            if not include_terminated and i['State']['Name'] == 'terminated':
                continue
            tags = {t['Key']: t['Value'] for t in i.get('Tags', [])}
            out.append({
                'id': i['InstanceId'],
                'type': i['InstanceType'],
                'state': i['State']['Name'],
                'ip': i.get('PublicIpAddress'),
                'private_ip': i.get('PrivateIpAddress'),
                'name': tags.get('Name', ''),
                'project': tags.get('project', ''),
                'launched': i['LaunchTime'].isoformat(),
                'volumes': [b['Ebs']['VolumeId']
                            for b in i.get('BlockDeviceMappings', [])],
                'root_device': i.get('RootDeviceName', '/dev/sda1'),
            })
    return out


def aliases():
    """Short names -> instance ids, from cluster/vms.json.

    Exists because two of the boxes share the Name tag 'ray-sculptor-head',
    so tag lookup alone is ambiguous.
    """
    p = os.path.join(REPO, 'cluster', 'vms.json')
    try:
        return json.load(open(p)).get('aliases', {})
    except (IOError, ValueError):
        return {}


def resolve(ref):
    """Accept an alias from vms.json, an instance id, a Name tag, or a
    unique substring of either."""
    ref = aliases().get(ref, ref)
    if ref and ref.startswith('i-'):
        got = describe(ref)
        if got:
            return got[0]
        raise SystemExit('no such instance: {}'.format(ref))
    cands = [d for d in describe()
             if ref and (ref in d['name'] or ref in d['id'])]
    if len(cands) == 1:
        return cands[0]
    known = '\n'.join('  {} {:<10} {}'.format(d['id'], d['state'], d['name'])
                      for d in describe())
    alias_help = '\n'.join('  {:<10} -> {}'.format(k, v)
                           for k, v in sorted(aliases().items()))
    if not cands:
        raise SystemExit('no instance matches {!r}.\nInstances:\n{}\n'
                         'Aliases (cluster/vms.json):\n{}'.format(
                             ref, known, alias_help))
    raise SystemExit('{!r} is ambiguous: {}\nUse an alias instead:\n{}'.format(
        ref, ', '.join(d['id'] for d in cands), alias_help))


def volume_size(volume_id):
    return ec2().describe_volumes(VolumeIds=[volume_id])['Volumes'][0]['Size']


# ------------------------------------------------------- SSH / rsync ---

SSH_OPTS = [
    '-i', SSH_KEY,
    '-o', 'StrictHostKeyChecking=accept-new',
    # The public IP is recycled across instances, so a stale known_hosts
    # entry otherwise fails verification *silently* when stderr is
    # discarded (dashboard/README.md documents the same bite).
    '-o', 'UserKnownHostsFile=/dev/null',
    '-o', 'LogLevel=ERROR',
    '-o', 'ConnectTimeout=15',
    '-o', 'ServerAliveInterval=20',
    '-o', 'ServerAliveCountMax=3',
]


def ssh_argv(ip, remote_cmd=None):
    argv = ['ssh'] + SSH_OPTS + ['{}@{}'.format(SSH_USER, ip)]
    if remote_cmd is not None:
        argv.append(remote_cmd)
    return argv


def ssh(ip, remote_cmd, check=False, timeout=180, quiet=True):
    """Run one command on the VM. Returns (rc, stdout, stderr)."""
    p = subprocess.run(ssh_argv(ip, remote_cmd), capture_output=True,
                       text=True, timeout=timeout)
    if check and p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit('ssh failed (rc={}): {}'.format(
            p.returncode, remote_cmd[:200]))
    if not quiet and p.stderr.strip():
        sys.stderr.write(p.stderr)
    return p.returncode, p.stdout, p.stderr


def ssh_ready(ip, tries=1, delay=10):
    for k in range(tries):
        try:
            rc, _, _ = ssh(ip, 'true', timeout=20)
            if rc == 0:
                return True
        except subprocess.TimeoutExpired:
            pass
        if k + 1 < tries:
            time.sleep(delay)
    return False


def rsync(src, dst, ip=None, excludes=(), delete=False, extra=(),
          timeout=600):
    """rsync with the cluster's ssh options. Remote side is whichever of
    src/dst is written as `user@host:path`.

    Flags are kept to the portable set: macOS ships **openrsync**
    (protocol 29, "rsync 2.6.9 compatible"), which rejects `--info=`,
    `--out-format=` and friends outright rather than ignoring them.
    """
    argv = ['rsync', '-az', '--partial', '--stats',
            '-e', ' '.join(shlex.quote(a) for a in ['ssh'] + SSH_OPTS)]
    for e in excludes:
        argv += ['--exclude', e]
    if delete:
        argv.append('--delete')
    argv += list(extra)
    argv += [src, dst]
    # A timeout is not optional here. harvest_all() runs inside the
    # dashboard refresh loop; an rsync that hangs on a half-open TCP
    # connection would stall every experiment's refresh indefinitely, and
    # the loop has no watchdog of its own. Better a failed pull that the
    # next cycle retries than a dashboard frozen at a stale hour.
    try:
        p = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return (124, '', 'rsync timed out after {}s: {} -> {}'.format(
            timeout, src, dst))
    return p.returncode, p.stdout, p.stderr


def remote(ip, path):
    return '{}@{}:{}'.format(SSH_USER, ip, path)


# ----------------------------------------------------- run registry ---

def run_dir(run_id):
    return os.path.join(RUNS_DIR, run_id)


def load_manifest(run_id):
    p = os.path.join(run_dir(run_id), 'manifest.json')
    if not os.path.exists(p):
        raise SystemExit('unknown run {!r} (no {})'.format(run_id, p))
    return json.load(open(p))


def save_manifest(m):
    d = run_dir(m['run_id'])
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, 'manifest.json.tmp')
    with open(tmp, 'w') as fh:
        json.dump(m, fh, indent=1, sort_keys=True)
    os.replace(tmp, os.path.join(d, 'manifest.json'))
    return m


def all_runs():
    if not os.path.isdir(RUNS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(RUNS_DIR)):
        p = os.path.join(RUNS_DIR, name, 'manifest.json')
        if os.path.exists(p):
            try:
                out.append(json.load(open(p)))
            except ValueError:
                continue
    return out


def live_runs(instance_id=None):
    """Runs not yet marked finished/harvested-final."""
    return [m for m in all_runs()
            if m.get('state') in (None, 'launched', 'running', 'killed')
            and (instance_id is None or m.get('instance_id') == instance_id)]


def local_log(run_id):
    return os.path.join(run_dir(run_id), 'logs', 'run.log')


def harvest_gap(m, ip):
    """Bytes on the VM that are not yet on the Mac, for this run's log.

    Returns (remote_bytes, local_bytes, reachable). This is the single
    check that `vmctl stop` gates on -- it is deliberately about *bytes*
    rather than mtimes, because a truncated pull and a complete one have
    the same mtime and only the byte count tells them apart.
    """
    rc, out, _ = ssh(ip, 'stat -c %s {} 2>/dev/null || echo -1'.format(
        shlex.quote(m['remote_log'])))
    if rc != 0:
        return (None, None, False)
    try:
        rbytes = int(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return (None, None, False)
    lp = local_log(m['run_id'])
    lbytes = os.path.getsize(lp) if os.path.exists(lp) else 0
    return (rbytes, lbytes, True)


# ------------------------------------------------------ alert JSON ---

def read_alert():
    try:
        return json.load(open(ALERT_JSON))
    except (IOError, ValueError):
        return {}


def update_alert(active=None, instance=None, note=None, sweep=None):
    """Patch the liveness-cron contract file. MUST be called on every VM
    start/stop -- the cron pages Tom on any disagreement with AWS."""
    cfg = read_alert()
    if active is not None:
        cfg['active'] = bool(active)
    cfg['last_updated'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    if note:
        cfg['agent_session'] = note
    if instance:
        head = cfg.setdefault('head', {})
        head['instance_id'] = instance.get('id', head.get('instance_id'))
        head['public_ip'] = instance.get('ip')
        head['ssh_key'] = '~/.ssh/ray-autoscaler_us-east-1.pem'
        head['ssh_user'] = SSH_USER
        if instance.get('state_note'):
            head['state'] = instance['state_note']
    if sweep is not None:
        cfg['sweep'] = sweep
    os.makedirs(os.path.dirname(ALERT_JSON), exist_ok=True)
    tmp = ALERT_JSON + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(cfg, fh, indent=1)
    os.replace(tmp, ALERT_JSON)
    return cfg


# ----------------------------------------------------------- misc ---

def human_bytes(n):
    if n is None:
        return '?'
    for unit in ('B', 'K', 'M', 'G', 'T'):
        if abs(n) < 1024 or unit == 'T':
            return '{:.0f}{}'.format(n, unit) if unit == 'B' else \
                   '{:.1f}{}'.format(n, unit)
        n /= 1024.0


def human_dt(seconds):
    if seconds is None:
        return '?'
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return '{}h{:02d}m'.format(h, m)
    if m:
        return '{}m{:02d}s'.format(m, s)
    return '{}s'.format(s)


def cost_usd(instance_type, seconds, disk_gb=0):
    rate = HOURLY_USD.get(instance_type)
    if rate is None:
        return None
    disk_rate = disk_gb * GP3_USD_PER_GB_MONTH / 730.0
    return (rate + disk_rate) * seconds / 3600.0


def utcnow():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def parse_utc(s):
    """'2026-08-21T17:28:22Z' -> epoch seconds.

    calendar.timegm, not mktime-minus-time.timezone: the latter ignores
    DST and put a four-minute-old run an hour into its past.
    """
    if not s:
        return None
    import calendar
    try:
        return calendar.timegm(time.strptime(s, '%Y-%m-%dT%H:%M:%SZ'))
    except ValueError:
        return None
