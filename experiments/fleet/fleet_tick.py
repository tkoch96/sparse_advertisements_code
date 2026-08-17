"""Fleet-aggregated progress ticker (SCALE-500 P0).

Every --tick seconds: for each fleet VM (registry `fleet` list, falling
back to the single `head`), run the same remote probe progress_tick uses
(landed + in-flight iterations from a manifest, RAM/CPU/cells), sum the
fleet, attach per-VM rows, reuse progress_tick's regression/EMA ETA, and
write dashboard_site/progress.json. Drop-in replacement for
progress_tick --loop when a fleet is up.

    nohup python -m experiments.fleet.fleet_tick --loop \
        --manifest tools/eods_manifest.json &
"""
import argparse
import json
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.dashboard import progress_tick as pt
from experiments.fleet import registry


def probe_vm(ip, manifest):
    remote = pt.REMOTE_PY.replace('grid_georand_r4_manifest.json',
                                  os.path.basename(manifest))
    try:
        out = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=10', '-i', pt.SSH_KEY,
             'ubuntu@{}'.format(ip),
             '/home/ubuntu/venv/bin/python -c '
             '"import sys; exec(sys.stdin.read())"'],
            input=remote, capture_output=True, text=True, timeout=25)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def tick(manifest):
    vms = registry.fleet() or [
        {'instance_id': 'head', 'public_ip':
         registry.load()['head']['public_ip'], 'state': 'running'}]
    rows, tot = [], None
    for vm in vms:
        if vm.get('state') in ('terminated', 'launch_failed'):
            continue
        d = probe_vm(vm['public_ip'], manifest)
        rows.append({'id': vm['instance_id'], 'ip': vm['public_ip'],
                     'ok': d is not None,
                     **({k: d[k] for k in ('done_cells', 'cells_running',
                                           'ram_pct', 'cpu_pct',
                                           'inflight_it')} if d else {})})
        if d:
            if tot is None:
                tot = dict(d)
            else:
                for k in ('done_it', 'est_total', 'done_cells',
                          'inflight_it', 'cells_running'):
                    tot[k] += d[k]
                tot['ram_pct'] = max(tot['ram_pct'], d['ram_pct'])
                tot['cpu_pct'] = (tot['cpu_pct'] + d['cpu_pct']) / 2.0
    if tot is None:
        return 'no reachable VMs'
    # total_cells is per-manifest global; VM probes each count the FULL
    # manifest denominator — normalize to one copy
    tot['total_cells'] = rows and max(
        r.get('done_cells', 0) for r in rows) or tot['total_cells']
    d0 = probe_vm(vms[0]['public_ip'], manifest)
    if d0:
        tot['total_cells'] = d0['total_cells']
    tot['fleet'] = rows
    pt._add_rate_and_eta(tot)
    with open(os.path.join(_REPO_ROOT, 'dashboard_site',
                           'progress.json'), 'w') as f:
        json.dump(tot, f)
    return 'ok fleet={} done={}/{}'.format(
        len(rows), tot['done_cells'], tot['total_cells'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--loop', action='store_true')
    ap.add_argument('--tick', type=int, default=30)
    ap.add_argument('--manifest', required=True)
    args = ap.parse_args()
    while True:
        print(time.strftime('%H:%M:%S'), tick(args.manifest), flush=True)
        if not args.loop:
            break
        time.sleep(args.tick)


if __name__ == '__main__':
    main()
