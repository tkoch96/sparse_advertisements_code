"""depstorectl -- ops for the global deployment/eval store.

  python -m cluster.depstorectl ls [--evals]
  python -m cluster.depstorectl fsck
  python -m cluster.depstorectl why-miss [--min-iters N] [--config k=v ...]
  python -m cluster.depstorectl mirror [--bucket s3://...]   # push to S3
  python -m cluster.depstorectl ingest <pickle> --dpsize X [--n-iters N]
                                       [--legacy] [--config k=v ...]

The store root resolves exactly as core.depstore does (env
SCULPTOR_DEPSTORE_ROOT / SCULPTOR_DEPSTORE_LOCAL). `mirror` pushes every
object missing-or-changed to S3 (bucket auto-created on first use);
restore is `aws s3 sync` or a future `mirror --pull`.
"""
import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from core import depstore  # noqa: E402

DEFAULT_BUCKET = os.environ.get('SCULPTOR_DEPSTORE_S3',
                                's3://sculptor-depstore-tomkoch')


def _kv(pairs):
    out = {}
    for p in pairs or []:
        k, _, v = p.partition('=')
        out[k] = v
    return out


def cmd_ls(a):
    st = depstore.Depstore()
    seen = {}
    for e in st.index():
        if e.get('kind') in ('training', 'eval'):
            seen[e['path']] = e
    if not seen:
        print('store empty ({})'.format(st.root))
        return 0
    print('store: {} ({:.1f} MB)'.format(st.root, st._dir_size_mb()))
    for path, e in sorted(seen.items()):
        if e['kind'] == 'training':
            mfn = os.path.join(st.root, path, 'manifest.json')
            extra = ''
            if os.path.exists(mfn):
                m = json.load(open(mfn))
                k = m.get('key', {})
                extra = ' obj={} seed={} era={}'.format(
                    k.get('SCULPTOR_GENERIC_OBJECTIVE') or
                    k.get('SCULPTOR_ABLATION_OBJECTIVE') or '?',
                    k.get('SCULPTOR_DEPLOYMENT_SEED') or '?',
                    (m.get('provenance') or {}).get('era_tag', ''))
            print('  T {:<44s} it={:<5}{}'.format(
                e['fp'], e.get('n_iters'), extra))
        elif a.evals:
            print('  E {} {}@{}'.format(e['fp'], e.get('family'),
                                        e.get('eval_era')))
    return 0


def cmd_fsck(a):
    st = depstore.Depstore()
    bad = ok = 0
    for e in st.index():
        if e.get('kind') not in ('training', 'eval'):
            continue
        d = os.path.join(st.root, e['path'])
        mfn = os.path.join(d, 'manifest.json')
        if not os.path.exists(mfn):
            if os.path.isdir(d):
                print('  BAD (no manifest): {}'.format(e['path']))
                bad += 1
            continue
        m = json.load(open(mfn))
        for fn, want in m.get('checksums', {}).items():
            p = os.path.join(d, fn)
            if not os.path.exists(p) or depstore._sha256_file(p) != want:
                print('  BAD (checksum): {}/{}'.format(e['path'], fn))
                bad += 1
                break
        else:
            ok += 1
    print('fsck: {} ok, {} bad'.format(ok, bad))
    return 1 if bad else 0


def cmd_why_miss(a):
    st = depstore.Depstore()
    print(st.why_miss(min_iters=a.min_iters, config=_kv(a.config)))
    return 0


def cmd_mirror(a):
    import boto3
    from botocore.exceptions import ClientError
    st = depstore.Depstore()
    bucket = (a.bucket or DEFAULT_BUCKET).replace('s3://', '').rstrip('/')
    s3 = boto3.client('s3', region_name='us-east-1')
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        try:
            print('creating bucket {}'.format(bucket))
            s3.create_bucket(Bucket=bucket)
        except ClientError as e:
            print('S3 access denied ({}).\n'
                  'The ray-cluster IAM user has no S3 permissions. One-time '
                  'admin fix (run as your admin identity):\n'
                  '  aws iam put-user-policy --user-name ray-cluster '
                  '--policy-name sculptor-depstore-s3 --policy-document '
                  "'{{\"Version\":\"2012-10-17\",\"Statement\":[{{"
                  "\"Effect\":\"Allow\",\"Action\":[\"s3:CreateBucket\","
                  "\"s3:ListBucket\",\"s3:GetObject\",\"s3:PutObject\"],"
                  "\"Resource\":[\"arn:aws:s3:::{b}\","
                  "\"arn:aws:s3:::{b}/*\"]}}]}}'"
                  .format(str(e)[:80], b=bucket))
            return 2
    remote = {}
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket):
        for o in page.get('Contents', []):
            remote[o['Key']] = o['Size']
    pushed = skipped = 0
    for dirpath, _d, files in os.walk(st.root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            key = os.path.relpath(p, st.root)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            if remote.get(key) == sz:
                skipped += 1
                continue
            s3.upload_file(p, bucket, key)
            pushed += 1
    print('mirror -> s3://{}: pushed {}, unchanged {}'.format(
        bucket, pushed, skipped))
    return 0


def cmd_ingest(a):
    """Best-effort ingest of an EXISTING metrics pickle (compare_rets/
    adv style). Legacy artifacts are tagged so modern-era lookups can
    never silently hit them (their CORE_ERA key is 'legacy')."""
    import pickle
    import numpy as np
    st = depstore.Depstore()
    try:
        m = pickle.load(open(a.pickle, 'rb'))
    except Exception as e:
        print('cannot ingest {}: not a metrics pickle ({})'.format(
            a.pickle, e))
        return 1
    cfg = _kv(a.config)
    cfg['dpsize'] = a.dpsize
    if a.legacy:
        cfg['CORE_ERA'] = 'legacy:' + os.path.basename(a.pickle)
    n = 0
    crs = m.get('compare_rets') or {}
    deps = m.get('deployment') or {}
    for sim, ret in crs.items():
        if not ret:
            continue
        try:
            adv = np.asarray(ret['adv_solns']['sparse'][0])
        except (KeyError, TypeError, IndexError):
            continue
        c = dict(cfg, sim=str(sim))
        fp = st.put_training(
            adv, a.n_iters, deployment=deps.get(sim), config=c,
            provenance={'ingested_from': os.path.basename(a.pickle),
                        'era_tag': 'legacy' if a.legacy else 'modern'})
        if fp:
            n += 1
    print('ingested {} training artifact(s) from {}'.format(n, a.pickle))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('ls'); p.add_argument('--evals', action='store_true')
    p.set_defaults(fn=cmd_ls)
    p = sub.add_parser('fsck'); p.set_defaults(fn=cmd_fsck)
    p = sub.add_parser('why-miss')
    p.add_argument('--min-iters', type=int, default=0)
    p.add_argument('--config', nargs='*', default=[])
    p.set_defaults(fn=cmd_why_miss)
    p = sub.add_parser('mirror'); p.add_argument('--bucket', default=None)
    p.set_defaults(fn=cmd_mirror)
    p = sub.add_parser('ingest')
    p.add_argument('pickle')
    p.add_argument('--dpsize', required=True)
    p.add_argument('--n-iters', type=int, default=150)
    p.add_argument('--legacy', action='store_true')
    p.add_argument('--config', nargs='*', default=[])
    p.set_defaults(fn=cmd_ingest)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == '__main__':
    main()
