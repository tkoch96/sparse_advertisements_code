#!/usr/bin/env python
"""vmctl -- VM lifecycle for the SCULPTOR boxes.

    python -m cluster.vmctl list
    python -m cluster.vmctl start head --disk 300
    python -m cluster.vmctl status head
    python -m cluster.vmctl df head
    python -m cluster.vmctl grow-disk head --gb 500
    python -m cluster.vmctl ssh head -- 'tail -5 /tmp/foo.log'
    python -m cluster.vmctl stop head
    python -m cluster.vmctl terminate head --yes

`<ref>` is an instance id, a Name tag, or any unambiguous substring of
either -- `head`, `preflight`, `i-0428` all work.

Two things this tool does that hand-written boto3 kept forgetting:

1. **Grows the disk on start.** `--disk N` (default: whatever
   SCULPTOR_VM_MIN_DISK_GB says, 300) modifies the EBS volume *and*
   grows the partition and filesystem over SSH, so the space is actually
   usable. gp3 is $0.08/GB-month -- going 50 -> 300 GB costs about $0.03
   an hour against a $3.83/hr box. A full disk killed an 11-hour
   actual-32 run at iteration 125 on 2026-08-20 and destroyed its
   checkpoints on the way down; that is the whole reason this is a
   default rather than a flag you remember to pass.

2. **Refuses to stop a VM with unharvested logs.** `stop` walks every
   live run registered against the instance, pulls it, and compares
   remote to local byte counts. If bytes are missing it exits nonzero
   and tells you what would be lost. `--force` overrides, and says so.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cluster import vmlib as V  # noqa: E402

MIN_DISK_GB = int(os.environ.get('SCULPTOR_VM_MIN_DISK_GB', '300'))


# ------------------------------------------------------------ helpers ---

def _wait_state(instance_id, want, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = V.describe(instance_id)[0]
        if d['state'] == want:
            return d
        time.sleep(5)
    raise SystemExit('timed out waiting for {} to be {}'.format(
        instance_id, want))


_GROW_FS = r'''
set -e
src=$(findmnt -no SOURCE /)
fstype=$(findmnt -no FSTYPE /)
part=${src#/dev/}
disk=$(lsblk -no PKNAME "$src")
num=$(cat /sys/class/block/$part/partition)
echo "root=$src fs=$fstype disk=/dev/$disk part=$num"
sudo growpart /dev/$disk $num || echo "growpart: nothing to do"
case "$fstype" in
  xfs) sudo xfs_growfs / ;;
  *)   sudo resize2fs "$src" ;;
esac
df -h /
'''


def _grow_disk(inst, target_gb, verbose=True):
    """Grow the root EBS volume to target_gb and extend the filesystem.

    Returns the new size in GB. No-op (and says so) if the volume is
    already at least that big. EBS allows exactly one modification per
    volume per 6 hours -- the tool reports that rather than retrying.
    """
    vol = inst['volumes'][0]
    cur = V.volume_size(vol)
    if cur >= target_gb:
        if verbose:
            print('  disk: {} already {} GB (>= {}), no change'.format(
                vol, cur, target_gb))
        return cur
    if verbose:
        extra_hr = (target_gb - cur) * V.GP3_USD_PER_GB_MONTH / 730.0
        print('  disk: growing {} {} -> {} GB (+${:.4f}/hr)'.format(
            vol, cur, target_gb, extra_hr))
    try:
        V.ec2().modify_volume(VolumeId=vol, Size=target_gb)
    except Exception as e:                       # noqa: BLE001
        print('  disk: modify_volume FAILED: {}'.format(e))
        print('  disk: (EBS allows one resize per volume per 6h -- if that '
              'is the cause, the box still runs, just with the old size)')
        return cur
    # Wait for the volume to leave 'modifying'; the filesystem can be
    # grown as soon as state is 'optimizing', we do not wait for
    # 'completed' (that can take hours on a large volume).
    t0 = time.time()
    while time.time() - t0 < 300:
        st = V.ec2().describe_volumes_modifications(VolumeIds=[vol])
        mods = st.get('VolumesModifications', [])
        state = mods[0]['ModificationState'] if mods else 'completed'
        if state in ('optimizing', 'completed'):
            break
        time.sleep(5)
    if inst.get('ip'):
        rc, out, err = V.ssh(inst['ip'], _GROW_FS, timeout=180)
        if verbose:
            for line in (out or '').strip().splitlines():
                print('  disk: ' + line)
            if rc != 0:
                print('  disk: filesystem grow rc={} -- {}'.format(
                    rc, (err or '').strip()[:300]))
    return target_gb


# Descending, within a family. A stopped instance is PINNED to its AZ, so
# when that AZ is out of (say) c8g.24xlarge there is no waiting it out --
# the only moves are to retype or to snapshot into another AZ. Retyping
# keeps the volume, the instance id and the Elastic IP, so it is cheap and
# reversible: `vmctl start head --type c8g.24xlarge` later puts it back.
# Ordered by (cores desc, then family preference), NOT confined to one
# family: on 2026-08-21 every c8g size in us-east-1f was exhausted, so a
# same-family ladder walked 24xl -> 4xl and still failed. A stopped
# instance is pinned to its AZ, so cross-family is the only remaining
# move short of snapshotting elsewhere.
#
# ARM64 ONLY -- the AMI is Graviton (ami-06683ebc6ba468d04, Ubuntu 22.04
# arm64). An x86 type here would fail to boot, which is a far worse
# failure than "no capacity".
_ARM_LADDER = [
    'c8g.16xlarge', 'c7g.16xlarge', 'm8g.16xlarge', 'm7g.16xlarge',
    'c8g.12xlarge', 'c7g.12xlarge', 'm7g.12xlarge',
    'r8g.8xlarge', 'c8g.8xlarge', 'c7g.8xlarge', 'm8g.8xlarge',
    'm7g.8xlarge', 'c8g.4xlarge', 'c7g.4xlarge', 'm7g.4xlarge',
]

_FALLBACK_LADDER = {
    'c8g': _ARM_LADDER, 'c7g': _ARM_LADDER, 'm8g': _ARM_LADDER,
    'm7g': _ARM_LADDER, 'r8g': _ARM_LADDER,
}


def _ladder_from(instance_type):
    """Try the requested type first, then progressively smaller ARM types.

    Starting from the requested type's position keeps us from silently
    UPgrading to something more expensive than asked for.
    """
    ladder = _FALLBACK_LADDER.get(instance_type.split('.')[0])
    if not ladder:
        return [instance_type]
    if instance_type in ladder:
        return ladder[ladder.index(instance_type):]
    return [instance_type] + ladder


def _retype(instance_id, new_type):
    V.ec2().modify_instance_attribute(InstanceId=instance_id,
                                      InstanceType={'Value': new_type})


def _start_with_capacity_fallback(inst, a):
    """Start the box, walking down the size ladder on InsufficientCapacity.

    `--type X` pins the type (retyping to X first if needed) and disables
    the walk, so an explicit request fails loudly rather than quietly
    running the experiment on half the cores.
    """
    from botocore.exceptions import ClientError

    # --type sets where the ladder STARTS, not a pin. A failed downgrade
    # walk leaves the instance at the bottom rung, so without this the next
    # start would begin from the smallest type it last tried -- silently
    # running the next experiment on a quarter of the cores.
    # --exact-type is the pin, for when a specific size is the point.
    if a.type and a.type != inst['type']:
        print('retyping {} {} -> {} (ladder start)'.format(
            inst['id'], inst['type'], a.type))
        _retype(inst['id'], a.type)
        inst['type'] = a.type
    if a.exact_type:
        candidates = [a.exact_type]
        if a.exact_type != inst['type']:
            _retype(inst['id'], a.exact_type)
            inst['type'] = a.exact_type
    else:
        candidates = _ladder_from(inst['type'])

    last = None
    for i, itype in enumerate(candidates):
        if itype != inst['type']:
            print('  no capacity for {}; retyping to {}'.format(
                inst['type'], itype))
            try:
                _retype(inst['id'], itype)
            except ClientError as e:
                print('  retype to {} refused: {}'.format(
                    itype, str(e)[:120]))
                continue
            inst['type'] = itype
        rate = V.HOURLY_USD.get(itype, 0)
        print('starting {} ({}, ~${:.2f}/hr)'.format(inst['id'], itype, rate))
        try:
            V.ec2().start_instances(InstanceIds=[inst['id']])
            if i:
                print('  NOTE: running on {} rather than {} -- us-east-1{} '
                      'had no capacity for the larger size. Timing numbers '
                      'must be normalised by core count.'.format(
                          itype, candidates[0], 'f'))
            return inst
        except ClientError as e:
            if 'InsufficientInstanceCapacity' not in str(e):
                raise
            last = e
            if a.exact_type:
                break
    raise SystemExit(
        'no capacity in this AZ for any of {}.\nLast error: {}\n\n'
        'Options: wait and retry, pass --type with a family that has room, '
        'or snapshot the volume into another AZ.'.format(
            ', '.join(candidates), str(last)[:200]))


def _set_protection(instance_id, on, terminate_too=True, verbose=True):
    """EC2 stop/terminate protection -- an account-level lock, not a norm.

    On 2026-08-21 another agent working in the same AWS account issued
    `stop-instances` on the box mid-ladder, killing a run at 4/7 sizes.
    The PreToolUse hook in this repo only guards the session it is loaded
    in; it cannot bind a different agent, a different checkout, or the
    console. `DisableApiStop` can: the API itself refuses, whoever asks.

    `vmctl stop` clears it automatically just before stopping, so the
    friction lands only on paths that bypass these tools -- which is
    exactly where it belongs.
    """
    ec2 = V.ec2()
    try:
        ec2.modify_instance_attribute(InstanceId=instance_id,
                                      DisableApiStop={'Value': bool(on)})
        if terminate_too:
            ec2.modify_instance_attribute(
                InstanceId=instance_id,
                DisableApiTermination={'Value': bool(on)})
    except Exception as e:                        # noqa: BLE001
        if verbose:
            print('  protection: could not {} ({})'.format(
                'enable' if on else 'disable', str(e)[:160]))
        return False
    if verbose:
        print('  protection: stop/terminate protection {}'.format(
            'ENABLED -- other agents cannot stop this box'
            if on else 'cleared'))
    return True


def _protection_state(instance_id):
    """-> {'stop': bool|None, 'terminate': bool|None}."""
    ec2 = V.ec2()
    out = {}
    for attr, key, field in (
            ('disableApiStop', 'stop', 'DisableApiStop'),
            ('disableApiTermination', 'terminate', 'DisableApiTermination')):
        try:
            r = ec2.describe_instance_attribute(InstanceId=instance_id,
                                                Attribute=attr)
            out[key] = bool((r.get(field) or {}).get('Value', False))
        except Exception:                         # noqa: BLE001
            out[key] = None
    return out


def _tag_in_use(instance_id, note):
    """Breadcrumb in the console so another operator sees why it is locked."""
    try:
        V.ec2().create_tags(Resources=[instance_id], Tags=[
            {'Key': 'sculptor:in-use', 'Value': note[:255]}])
    except Exception:                             # noqa: BLE001
        pass


def _df(ip):
    rc, out, _ = V.ssh(ip, "df -h / /home 2>/dev/null | sed 1d; "
                           "echo ---; df -h --output=avail / | tail -1")
    return out


# ------------------------------------------------------------ commands --

def cmd_list(a):
    rows = V.describe()
    if not rows:
        print('no non-terminated instances in {}'.format(V.REGION))
        return 0
    runs = V.all_runs()
    print('{:<21} {:<15} {:<10} {:<16} {:>6} {:>9}  {}'.format(
        'INSTANCE', 'TYPE', 'STATE', 'PUBLIC IP', 'DISK', '$/HR', 'NAME'))
    for d in rows:
        gb = V.volume_size(d['volumes'][0]) if d['volumes'] else 0
        rate = V.HOURLY_USD.get(d['type'])
        mine = [m for m in runs if m.get('instance_id') == d['id']
                and m.get('state') in ('launched', 'running')]
        tail = d['name'] + ('  [{} live run(s)]'.format(len(mine))
                            if mine else '')
        print('{:<21} {:<15} {:<10} {:<16} {:>5}G {:>9}  {}'.format(
            d['id'], d['type'], d['state'], d['ip'] or '-', gb,
            '${:.2f}'.format(rate) if rate else '?', tail))
    burning = [d for d in rows if d['state'] in ('running', 'pending')]
    if burning:
        tot = sum(V.HOURLY_USD.get(d['type'], 0) for d in burning)
        print('\nBURNING: {} instance(s), ~${:.2f}/hr (~${:.0f}/day)'.format(
            len(burning), tot, tot * 24))
    else:
        print('\nNothing running. $0/hr compute.')
    return 0


def cmd_start(a):
    inst = V.resolve(a.ref)
    if inst['state'] == 'running':
        print('{} already running at {}'.format(inst['id'], inst['ip']))
    elif inst['state'] != 'stopped':
        raise SystemExit('{} is {}, cannot start'.format(
            inst['id'], inst['state']))
    else:
        _start_with_capacity_fallback(inst, a)
        inst = _wait_state(inst['id'], 'running', timeout=a.timeout)
        print('  running, public ip {}'.format(inst['ip']))

    # Alert JSON goes active BEFORE the slow SSH wait: if the tool dies
    # in the wait, the cron must still know a VM is up.
    V.update_alert(active=True, instance=dict(
        inst, state_note='started {} by cluster.vmctl'.format(V.utcnow())),
        note='vmctl start {} ({})'.format(inst['id'], inst['type']))

    print('  waiting for sshd', end='', flush=True)
    for _ in range(a.timeout // 10):
        if V.ssh_ready(inst['ip']):
            print(' ok')
            break
        print('.', end='', flush=True)
        time.sleep(10)
    else:
        print(' TIMEOUT')
        raise SystemExit('instance is running but ssh never came up; '
                         'ip={} -- it is still billing, stop it or retry'
                         .format(inst['ip']))

    if a.protect:
        _set_protection(inst['id'], True)
        _tag_in_use(inst['id'], 'started {} by cluster.vmctl; stop via '
                                '`python -m cluster.vmctl stop <ref>` '
                                '(clears protection + harvests logs first)'
                                .format(V.utcnow()))

    target = a.disk if a.disk is not None else MIN_DISK_GB
    if target:
        _grow_disk(inst, target)
    print('\n' + _df(inst['ip']))
    print('ssh: ssh -i {} {}@{}'.format(V.SSH_KEY, V.SSH_USER, inst['ip']))
    return 0


def _gate_run(m, ip, kill=False):
    """Can this run's box be stopped without losing anything?

    Returns (ok, explanation).

    The byte comparison is only meaningful once the writer has stopped. A
    LIVE run's log grows between the harvest and the stat, so comparing
    bytes against a running process always shows a gap -- the first
    version of this gate did exactly that and would have blocked every
    stop, training everyone to reach for --force. So:

      * process alive  -> refuse on those grounds (stopping the VM kills
                          a running experiment), unless --kill was asked
                          for, in which case kill it and then gate;
      * process dead   -> harvest and compare bytes. Retry once, because
                          a process that exited during the harvest leaves
                          a real but transient gap.
    """
    from cluster import expctl

    rc, out, _ = V.ssh(ip, 'kill -0 {} 2>/dev/null && echo ALIVE || echo DEAD'
                           .format(m.get('pid') or 0))
    alive = 'ALIVE' in out
    if alive and not kill:
        return (False, 'STILL RUNNING (pid {}) -- stopping the VM would '
                       'kill it. Use --kill to end it deliberately, or '
                       'expctl watch to see it out'.format(m.get('pid')))
    if alive and kill:
        print(' killing pid {} ...'.format(m.get('pid')), end='', flush=True)
        try:
            expctl.cmd_kill(type('A', (), {'run_id': m['run_id']})())
        except SystemExit:
            pass
        time.sleep(5)

    for attempt in (1, 2):
        try:
            expctl.harvest(m, ip, verbose=False)
        except Exception as e:                    # noqa: BLE001
            return (False, 'harvest raised: {}'.format(e))
        rb, lb, reachable = V.harvest_gap(m, ip)
        if not reachable:
            return (False, 'could not stat the remote log')
        if rb <= lb:
            return (True, V.human_bytes(lb))
        if attempt == 1:
            time.sleep(3)
    return (False, '{} of {} bytes pulled -- {} SHORT'.format(
        lb, rb, V.human_bytes(rb - lb)))


def cmd_stop(a):
    inst = V.resolve(a.ref)
    if inst['state'] == 'stopped':
        print('{} already stopped'.format(inst['id']))
        V.update_alert(active=False, instance=inst,
                       note='vmctl stop (already stopped) {}'.format(
                           V.utcnow()))
        return 0
    if inst['state'] != 'running':
        raise SystemExit('{} is {}'.format(inst['id'], inst['state']))

    # --- the harvest gate -------------------------------------------------
    runs = V.live_runs(inst['id'])
    unharvested = []
    if runs and not a.skip_harvest:
        print('final harvest of {} live run(s) before stop'.format(len(runs)))
        for m in runs:
            print('  {} ...'.format(m['run_id']), end='', flush=True)
            ok, why = _gate_run(m, inst['ip'], kill=a.kill)
            if ok:
                print(' ok ({})'.format(why))
            else:
                print(' {}'.format(why))
                unharvested.append((m['run_id'], why))

    if unharvested and not a.force:
        still_running = any('STILL RUNNING' in why for _rid, why in unharvested)
        print('\nREFUSING TO STOP:')
        for rid, why in unharvested:
            print('  {}: {}'.format(rid, why))
        if still_running:
            print('\n  python -m cluster.vmctl stop {} --kill    '
                  '# end it, harvest, then stop'.format(a.ref))
            print('  python -m cluster.expctl watch <run_id>   '
                  '# let it finish first')
        else:
            print('\nFix the pull, or re-run with --force to stop anyway.')
        print('\nThe VM is still running and still billing (~${:.2f}/hr).'
              .format(V.HOURLY_USD.get(inst['type'], 0)))
        return 2
    if unharvested and a.force:
        print('\n--force: stopping with {} unharvested run(s). '
              'These logs are being abandoned:'.format(len(unharvested)))
        for rid, why in unharvested:
            print('  {}: {}'.format(rid, why))

    # Clear protection only now -- after the gate has passed. Dropping it
    # earlier would open the window this is meant to close.
    _set_protection(inst['id'], False)
    _tag_in_use(inst['id'], 'stopped {} by cluster.vmctl'.format(V.utcnow()))
    print('stopping {}'.format(inst['id']))
    V.ec2().stop_instances(InstanceIds=[inst['id']])
    if a.wait:
        _wait_state(inst['id'], 'stopped', timeout=a.timeout)
        print('  stopped')
    V.update_alert(active=False, instance=dict(
        inst, ip=None, state_note='stopped {} by cluster.vmctl (EBS data '
                                  'intact)'.format(V.utcnow())),
        note=a.note or 'vmctl stop {} at {}'.format(inst['id'], V.utcnow()))
    print('alert JSON set active=false')
    return 0


def cmd_terminate(a):
    inst = V.resolve(a.ref)
    if not a.yes:
        raise SystemExit(
            'terminate DESTROYS the EBS volume and everything on it '
            '({} {} {}). Pass --yes if that is really what you want.'
            .format(inst['id'], inst['type'], inst['name']))
    _set_protection(inst['id'], False)
    V.ec2().terminate_instances(InstanceIds=[inst['id']])
    V.update_alert(active=False, instance=dict(inst, ip=None,
                   state_note='TERMINATED {}'.format(V.utcnow())),
                   note='vmctl terminate {}'.format(inst['id']))
    print('terminated {}'.format(inst['id']))
    return 0


def cmd_ip(a):
    print(V.resolve(a.ref)['ip'] or '')
    return 0


def cmd_ssh(a):
    inst = V.resolve(a.ref)
    if inst['state'] != 'running':
        raise SystemExit('{} is {}'.format(inst['id'], inst['state']))
    if a.cmd:
        return subprocess.call(V.ssh_argv(inst['ip'], ' '.join(a.cmd)))
    return subprocess.call(V.ssh_argv(inst['ip']))


def cmd_df(a):
    inst = V.resolve(a.ref)
    print(_df(inst['ip']))
    return 0


def cmd_grow(a):
    inst = V.resolve(a.ref)
    _grow_disk(inst, a.gb)
    print(_df(inst['ip']))
    return 0


def cmd_protect(a):
    inst = V.resolve(a.ref)
    _set_protection(inst['id'], bool(a.on))
    print(_protection_state(inst['id']))
    return 0


def cmd_status(a):
    inst = V.resolve(a.ref)
    gb = V.volume_size(inst['volumes'][0]) if inst['volumes'] else 0
    print('{}  {}  {}'.format(inst['id'], inst['type'], inst['state']))
    print('  name      {}'.format(inst['name'] or '-'))
    print('  public ip {}'.format(inst['ip'] or '-'))
    print('  disk      {} GB'.format(gb))
    prot = _protection_state(inst['id'])
    print('  protected stop={} terminate={}'.format(
        prot.get('stop'), prot.get('terminate')))
    rate = V.HOURLY_USD.get(inst['type'])
    if rate and inst['state'] == 'running':
        print('  cost      ~${:.2f}/hr (~${:.0f}/day if left up)'.format(
            rate, rate * 24))
    if inst['state'] == 'running':
        print()
        print(_df(inst['ip']))
        rc, out, _ = V.ssh(inst['ip'],
                           "uptime; echo; free -g | sed -n '1,2p'")
        print(out)
    runs = [m for m in V.all_runs() if m.get('instance_id') == inst['id']]
    if runs:
        print('  registered runs:')
        for m in runs:
            print('    {:<28} {:<10} {}'.format(
                m['run_id'], m.get('state', '?'), m.get('label', '')))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='vmctl', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('list', help='every non-terminated instance + burn rate') \
        .set_defaults(fn=cmd_list)

    p = sub.add_parser('start', help='start a stopped instance, grow its disk')
    p.add_argument('ref')
    p.add_argument('--disk', type=int, default=None,
                   help='grow root volume to this many GB (default {}; '
                        '0 disables)'.format(MIN_DISK_GB))
    p.add_argument('--type', default=None,
                   help='where the capacity ladder STARTS (retypes the '
                        'stopped box; volume, instance id and Elastic IP '
                        'are kept). Capacity failures then walk DOWN the '
                        'cross-family ARM ladder from there.')
    p.add_argument('--exact-type', default=None,
                   help='pin exactly this type; fail loudly rather than '
                        'quietly running on fewer cores.')
    p.add_argument('--no-protect', dest='protect', action='store_false',
                   default=True,
                   help='do NOT set EC2 stop/terminate protection. By '
                        'default the box is locked so another agent or '
                        'session cannot stop it mid-run (this happened '
                        '2026-08-21); vmctl stop clears it automatically.')
    p.add_argument('--timeout', type=int, default=600)
    p.set_defaults(fn=cmd_start)

    p = sub.add_parser('stop', help='harvest-gated stop')
    p.add_argument('ref')
    p.add_argument('--force', action='store_true',
                   help='stop even if logs are unharvested (says what is lost)')
    p.add_argument('--kill', action='store_true',
                   help='end any still-running experiment first, then '
                        'harvest and stop (without this, a live run blocks '
                        'the stop rather than being killed by surprise)')
    p.add_argument('--skip-harvest', action='store_true',
                   help='do not even try to pull (implies you accept the loss)')
    p.add_argument('--wait', action='store_true')
    p.add_argument('--note', default=None, help='agent_session note for the alert JSON')
    p.add_argument('--timeout', type=int, default=600)
    p.set_defaults(fn=cmd_stop)

    p = sub.add_parser('terminate', help='DESTRUCTIVE: kills the EBS volume too')
    p.add_argument('ref')
    p.add_argument('--yes', action='store_true')
    p.set_defaults(fn=cmd_terminate)

    p = sub.add_parser('ip'); p.add_argument('ref'); p.set_defaults(fn=cmd_ip)
    p = sub.add_parser('df'); p.add_argument('ref'); p.set_defaults(fn=cmd_df)
    p = sub.add_parser('status'); p.add_argument('ref')
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser('protect', help='set/clear EC2 stop protection')
    p.add_argument('ref')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--on', action='store_true')
    g.add_argument('--off', action='store_true')
    p.set_defaults(fn=cmd_protect)

    p = sub.add_parser('grow-disk')
    p.add_argument('ref'); p.add_argument('--gb', type=int, required=True)
    p.set_defaults(fn=cmd_grow)

    p = sub.add_parser('ssh', help='interactive shell, or -- <cmd>')
    p.add_argument('ref')
    p.add_argument('cmd', nargs='*')
    p.set_defaults(fn=cmd_ssh)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    raise SystemExit(main())
