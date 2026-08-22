#!/usr/bin/env python3
"""Stop hook: do not end a turn with a VM quietly burning money.

Wired up in `.claude/settings.json`. Fires when the agent is about to
finish its turn. If the cluster-alert JSON says a VM is active, it blocks
once (exit 2) with a reminder of what is running, what it costs per hour,
and whether any run still has bytes only on the VM.

Deliberately cheap and offline: it reads `~/.sculptor_cluster_alert/
active_cluster.json` and the local run manifests, and makes no EC2 or SSH
call. A hook that took ten seconds would get disabled within a day. The
consequence is that it trusts the alert JSON -- which every lifecycle
path in `vmctl` updates, and which the liveness cron independently
cross-checks against AWS every 10 minutes.

`stop_hook_active` is honoured, so it interrupts at most once per turn and
cannot loop.

Acknowledgement
---------------
A reminder that fires on every single turn is a reminder that gets turned
off. So it can be *satisfied*, not merely repeated:

    python -m cluster.expctl ack <run_id> --minutes 30 --reason "..."

That writes `~/.sculptor_cluster_alert/vm_ack.json` and buys quiet for the
stated window. Three things keep the snooze honest:

* it **expires** -- the reminder comes back, which is the heartbeat Tom
  wants on long runs rather than silence;
* it is **tied to one run id** -- acknowledging run A says nothing about
  run B;
* it is **void the moment the run stops running** -- a finished, failed or
  killed run alarms immediately regardless of any outstanding ack. The
  whole point is to notice the end of a run, and that is exactly when a
  stale snooze would do the most damage.
"""

import json
import os
import sys
import time

ALERT = os.path.expanduser('~/.sculptor_cluster_alert/active_cluster.json')
ACK = os.path.expanduser('~/.sculptor_cluster_alert/vm_ack.json')
REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
RUNS = os.path.join(REPO, 'cache', 'cluster_runs')
def _rates():
    """Single source: cluster/vmlib. A local copy went stale the moment
    the run moved to c7g.16xlarge and the burn line silently vanished."""
    try:
        sys.path.insert(0, REPO)
        from cluster.vmlib import HOURLY_USD
        return HOURLY_USD
    except Exception:                             # noqa: BLE001
        return {}


def _read_ack():
    try:
        ack = json.load(open(ACK))
    except (IOError, ValueError):
        return None
    return ack if ack.get('until', 0) > time.time() else None


def _still_running(m):
    """True only while the run's own log says it is still going.

    Read from the harvested log rather than the manifest's `state`, which
    is only rewritten when a watcher sees the run out -- an interrupted
    watcher would otherwise leave a dead run looking alive forever, and
    the ack would go on suppressing the one alarm that matters.
    """
    d = os.path.join(RUNS, m['run_id'], 'logs')
    rc_txt = ''
    try:
        rc_txt = open(os.path.join(d, 'run.rc')).read().strip()
    except (IOError, OSError):
        pass
    if rc_txt:
        return False                    # the launcher wrote an exit code
    try:
        log = open(os.path.join(d, 'run.log'), errors='replace').read()
    except (IOError, OSError):
        return True                     # nothing harvested yet; assume live
    return '[sweep] ALL DONE' not in log and '[expctl] exit_rc=' not in log


def main():
    try:
        payload = json.load(sys.stdin)
    except (ValueError, IOError):
        payload = {}
    if payload.get('stop_hook_active'):
        return 0                       # already interrupted once this turn

    try:
        cfg = json.load(open(ALERT))
    except (IOError, ValueError):
        return 0
    if not cfg.get('active'):
        return 0

    head = cfg.get('head', {})
    lines = ['A VM IS STILL RUNNING -- do not end the turn without dealing '
             'with it.', '']
    lines.append('  instance   {}'.format(head.get('instance_id', '?')))
    lines.append('  public ip  {}'.format(head.get('public_ip', '?')))
    lines.append('  active since {}'.format(cfg.get('last_updated', '?')))

    live, stale = [], []
    if os.path.isdir(RUNS):
        for name in sorted(os.listdir(RUNS)):
            mp = os.path.join(RUNS, name, 'manifest.json')
            hp = os.path.join(RUNS, name, 'harvest.json')
            try:
                m = json.load(open(mp))
            except (IOError, ValueError):
                continue
            if m.get('state') not in ('launched', 'running', 'killed'):
                continue
            live.append(m)
            age = None
            if os.path.exists(hp):
                age = time.time() - os.path.getmtime(hp)
            if age is None or age > 1800:
                stale.append((name, age))

    # An outstanding, unexpired ack for a run that is STILL RUNNING buys
    # quiet. A run that has stopped voids it -- see the module docstring.
    ack = _read_ack()
    if ack and live:
        acked = [m for m in live if m['run_id'] == ack.get('run_id')]
        if acked and len(acked) == len(live) and _still_running(acked[0]):
            return 0

    if live:
        lines.append('  live runs  {}'.format(
            ', '.join(m['run_id'] for m in live)))
        rate = _rates().get(live[0].get('instance_type'), 0)
        if rate:
            lines.append('  burn       ~${:.2f}/hr (~${:.0f}/day)'.format(
                rate, rate * 24))
    if stale:
        lines.append('')
        lines.append('  !! LOGS NOT RECENTLY HARVESTED:')
        for name, age in stale:
            lines.append('     {} -- last pull {}'.format(
                name, 'never' if age is None
                else '{:.0f} min ago'.format(age / 60.0)))

    lines += ['', 'Do one of these before finishing:',
              '  python -m cluster.expctl status <run_id>   # pulls + verdict',
              '  python -m cluster.expctl watch  <run_id>   # stay with it',
              '  python -m cluster.vmctl  stop   <ref>      # harvest + stop',
              '',
              'If the run is meant to keep going while Tom watches, say so '
              'explicitly in your reply -- then this reminder is satisfied.']
    sys.stderr.write('\n'.join(lines) + '\n')
    return 2


if __name__ == '__main__':
    sys.exit(main())
