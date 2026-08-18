"""Provision a fleet of spot VMs from the head's AMI (SCALE-500 P1).

    # one-time: bake an AMI from the prepared head (env + caches on disk)
    python -m experiments.fleet.fleet_up --bake-ami

    # launch: N spot instances, register in the fleet JSON, bootstrap
    python -m experiments.fleet.fleet_up --image-id ami-... --count 6 \
        --type c7i.24xlarge --sha <git sha>

Bootstrap over ssh (never scp code — git doctrine): fetch+checkout the
pinned SHA, apply the known new-host fixes (worker_comms venv symlink,
ray on PATH, RAY_TMPDIR dirs). Cells stay idempotent, so spot reclaims
cost at most the in-flight cells.
"""
import argparse
import subprocess
import sys
import time

import boto3

from experiments.fleet import registry

REGION = 'us-east-1'
HEAD_INSTANCE = 'i-0428c395787bc3ca0'
KEY = '~/.ssh/ray-autoscaler_us-east-1.pem'
SSH = ['ssh', '-i', KEY.replace('~', __import__('os').path.expanduser('~')),
       '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=20',
       '-o', 'BatchMode=yes']

BOOTSTRAP = r'''set -e
cd ~/sparse_advertisements_code
# the AMI may be baked mid-git-operation on the head: clear stale locks
rm -f .git/index.lock .git/shallow.lock
git fetch -q origin && git checkout -qf {sha}
sudo ln -sf ~/venv/bin/ray /usr/local/bin/ray || true
mkdir -p /tmp/ray_tmp
echo BOOTSTRAP_OK $(git rev-parse --short HEAD)
'''


def bake_ami(ec2):
    name = 'sculptor-fleet-{}'.format(int(time.time()))
    r = ec2.create_image(InstanceId=HEAD_INSTANCE, Name=name,
                         NoReboot=True)
    print('AMI requested:', r['ImageId'],
          '(available in ~10-20 min; pass as --image-id)')


def launch(ec2, args):
    mkt = ({'MarketType': 'spot',
            'SpotOptions': {'SpotInstanceType': 'one-time',
                            'InstanceInterruptionBehavior': 'terminate'}}
           if args.spot else None)
    kw = dict(ImageId=args.image_id, InstanceType=args.type,
              MinCount=args.count, MaxCount=args.count,
              KeyName=args.key_name,
              SecurityGroupIds=[args.sg])
    if mkt:
        kw['InstanceMarketOptions'] = mkt
    insts = ec2.run_instances(**kw)['Instances']
    ids = [i['InstanceId'] for i in insts]
    print('launched:', ids)
    ec2.get_waiter('instance_running').wait(InstanceIds=ids)
    desc = ec2.describe_instances(InstanceIds=ids)
    out = []
    for res in desc['Reservations']:
        for i in res['Instances']:
            e = {'instance_id': i['InstanceId'],
                 'public_ip': i.get('PublicIpAddress'),
                 'instance_type': args.type, 'spot': bool(args.spot),
                 'state': 'running', 'shard': None,
                 'launched_at': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                              time.gmtime())}
            registry.upsert(e)
            out.append(e)
    # bootstrap (retry: sshd needs a minute)
    for e in out:
        boot = BOOTSTRAP.format(sha=args.sha)
        for attempt in range(10):
            r = subprocess.run(
                SSH + ['ubuntu@{}'.format(e['public_ip']), boot],
                capture_output=True, text=True)
            if 'BOOTSTRAP_OK' in r.stdout:
                print(e['instance_id'], r.stdout.strip().splitlines()[-1])
                registry.upsert({**e, 'state': 'bootstrapped'})
                break
            time.sleep(20)
        else:
            print('BOOTSTRAP FAILED', e['instance_id'], file=sys.stderr)
            registry.upsert({**e, 'state': 'bootstrap_failed'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bake-ami', action='store_true')
    ap.add_argument('--image-id')
    ap.add_argument('--count', type=int, default=1)
    ap.add_argument('--type', default='c7i.24xlarge')
    ap.add_argument('--spot', action='store_true', default=True)
    ap.add_argument('--on-demand', dest='spot', action='store_false')
    ap.add_argument('--sha', default='origin/main')
    ap.add_argument('--sg', default='sg-083047c25a833bc2f')
    ap.add_argument('--key-name', default='ray-autoscaler_us-east-1')
    args = ap.parse_args()
    ec2 = boto3.client('ec2', region_name=REGION)
    if args.bake_ami:
        bake_ami(ec2)
        return
    assert args.image_id, '--image-id required (or --bake-ami first)'
    launch(ec2, args)


if __name__ == '__main__':
    main()
