"""Drain + collect + terminate the fleet (SCALE-500 P1).

    python -m cluster.fleet.fleet_down --store cache/eods/v1 [--keep-head]

Per VM: final rsync of the store to the Mac, terminate, registry update.
The head (i-0428c395787bc3ca0) is never terminated — only fleet VMs.
"""
import argparse
import os
import subprocess

import boto3

from cluster.fleet import registry

HEAD_INSTANCE = 'i-0428c395787bc3ca0'
KEY = os.path.expanduser('~/.ssh/ray-autoscaler_us-east-1.pem')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--region', default='us-east-1')
    args = ap.parse_args()
    ec2 = boto3.client('ec2', region_name=args.region)
    for vm in list(registry.fleet()):
        iid, ip = vm['instance_id'], vm.get('public_ip')
        if iid == HEAD_INSTANCE:
            continue
        if ip:
            subprocess.run(
                ['rsync', '-az', '--timeout=300', '-e',
                 'ssh -i {} -o StrictHostKeyChecking=no -o BatchMode=yes'.format(KEY),
                 'ubuntu@{}:sparse_advertisements_code/{}/'.format(ip, args.store),
                 os.path.join(args.store, '')])
        try:
            ec2.terminate_instances(InstanceIds=[iid])
            print('terminated', iid)
        except Exception as e:
            print('terminate failed', iid, e)
        registry.upsert({**vm, 'state': 'terminated'})
    print('fleet drained; head untouched')


if __name__ == '__main__':
    main()
