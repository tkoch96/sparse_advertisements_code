"""Assign shards and start queues across the fleet (SCALE-500 P1).

    python -m cluster.fleet.shard --manifest cluster/manifests/eods_manifest.json \
        --n <fleet size> --out-dir tools/shards/eods
    python -m cluster.fleet.fleet_launch --shard-dir tools/shards/eods \
        --manifest-base eods_manifest [--env SCULPTOR_LP_BACKEND=highs ...]

Shard JSONs are DATA (not code): pushed via scp to each VM's repo tools/
dir. Queue launched detached per VM; ram_watchdog armed.
"""
import argparse
import glob
import os
import subprocess

from cluster.fleet import registry

KEY = os.path.expanduser('~/.ssh/ray-autoscaler_us-east-1.pem')
SSH = ['ssh', '-i', KEY, '-o', 'StrictHostKeyChecking=no',
       '-o', 'ConnectTimeout=20', '-o', 'BatchMode=yes']
SCP = ['scp', '-i', KEY, '-o', 'StrictHostKeyChecking=no',
       '-o', 'BatchMode=yes']

# NOTE 2026-08-22: run_n_sweep_queue was REMOVED with the ablation fork;
# this campaign-era launcher cannot run until pointed at a new queue.
LAUNCH = ('cd ~/sparse_advertisements_code && nohup setsid env {env} '
          '~/venv/bin/python -u -m experiments.ablation.run_n_sweep_queue '
          '--manifest tools/{mf} --ws-root ~/fleet_ws --slots {slots} '
          '--workers-per-run {wpr} --port0 51000 --launch-stagger 8 '
          '--no-rescore < /dev/null >> ~/fleet_queue.log 2>&1 & '
          'sleep 2; nohup setsid bash ~/ram_watchdog.sh < /dev/null '
          '>> ~/ram_watchdog.log 2>&1 & echo LAUNCHED')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-dir', required=True)
    ap.add_argument('--manifest-base', required=True)
    ap.add_argument('--slots', type=int, default=36)
    ap.add_argument('--workers-per-run', type=int, default=4)
    ap.add_argument('--env', nargs='*', default=[
        'SCULPTOR_LP_BACKEND=highs', 'SCULPTOR_RAY_NUM_CPUS=6',
        'SCULPTOR_QUEUE_PASSES=3', 'SCULPTOR_CELL_TIMEOUT=5400'])
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(
        args.shard_dir, args.manifest_base + '.shard*.json')))
    vms = [e for e in registry.fleet()
           if e.get('state') in ('bootstrapped', 'running', 'queue_running')]
    assert len(shards) <= len(vms), (len(shards), len(vms))
    for shard_fn, vm in zip(shards, vms):
        ip = vm['public_ip']
        mf = os.path.basename(shard_fn)
        subprocess.run(SCP + [shard_fn, 'ubuntu@{}:sparse_advertisements_code/tools/{}'.format(ip, mf)],
                       check=True)
        cmd = LAUNCH.format(env=' '.join(args.env), mf=mf,
                            slots=args.slots, wpr=args.workers_per_run)
        r = subprocess.run(SSH + ['ubuntu@{}'.format(ip), cmd],
                          capture_output=True, text=True, timeout=60)
        ok = 'LAUNCHED' in r.stdout
        print(vm['instance_id'], ip, mf, 'OK' if ok else 'FAILED')
        registry.upsert({**vm, 'shard': mf,
                         'state': 'queue_running' if ok else 'launch_failed'})


if __name__ == '__main__':
    main()
