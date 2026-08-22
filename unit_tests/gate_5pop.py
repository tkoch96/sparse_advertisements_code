"""5-pop dummy A/B gate for the array-native deployment-creation fork
(Tom 2026-08-18). Runs mainline load_actual_perfs twice in a sandboxed
CACHE_DIR -- once with the stock shard loader, once with the fork's
vectorized scalar-min loader -- same np.random seed, same process (so
hash ordering matches), and demands BYTE-EXACT equality of the outputs:
key ORDER included, float values bitwise, because downstream RNG
(do_filter shuffles, sub-ms tie-break noise) consumes state in dict
iteration order.

The sandbox copies addresses_violating_sol.csv (the SOL stage rewrites
it) and symlinks every read-only input, so nothing touches the real
cache. Usage:
  python -m unit_tests.gate_5pop [--pops p1,p2,...]
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

POPS5 = ['vtramsterdam', 'vtratlanta', 'vtrchicago', 'vtrmiami',
         'vtrnewyork']
SEED = 42


def _sandbox(real_cache):
    sb = os.path.join(REPO, 'cache', 'depsetup_fork_sandbox')
    os.makedirs(os.path.join(sb, 'deployments'), exist_ok=True)
    for fn in ('vultr_ingress_latencies_by_dst.csv',
               'vultr_anycast_latency_smaller.csv',
               'vultr_provider_popps.csv', 'lat_shards'):
        dst = os.path.join(sb, fn)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(os.path.join(real_cache, fn)), dst)
    return sb


def _reset_violate(real_cache, sb):
    shutil.copyfile(os.path.join(real_cache, 'addresses_violating_sol.csv'),
                    os.path.join(sb, 'addresses_violating_sol.csv'))


def _walk_equal(a, b, path='root'):
    """Exact recursive equality with ORDER for dicts; returns list of
    mismatch descriptions (empty = equal)."""
    errs = []
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = list(a), list(b)
        if ka != kb:
            errs.append('{}: key order/set differs ({} vs {} keys)'.format(
                path, len(ka), len(kb)))
            return errs
        for k in ka:
            errs += _walk_equal(a[k], b[k], '{}[{}]'.format(path, k))
            if len(errs) > 20:
                return errs
    else:
        eq = (a == b)
        if hasattr(eq, 'all'):
            eq = bool(np.asarray(eq).all())
        if not eq:
            errs.append('{}: {} != {}'.format(path, a, b))
    return errs


def run_arm(ds, pops, arm, do_filter=True):
    from core import fast_perfs
    np.random.seed(SEED)
    t0 = time.time()
    if arm == 'b2':
        # exercise the MERGED mainline seam end-to-end (2026-08-18)
        os.environ['SCULPTOR_DEPSETUP_ARRAYS'] = '1'
        anycast, ug_perfs = ds.load_actual_perfs(
            considering_pops=pops, do_filter=do_filter)
    else:
        # baseline arms must pin the legacy loop now that the seam
        # defaults ON
        os.environ['SCULPTOR_DEPSETUP_ARRAYS'] = '0'
        if arm == 'b1':
            fast_perfs.install()
        else:
            fast_perfs.uninstall()
        anycast, ug_perfs = ds.load_actual_perfs(
            considering_pops=pops, do_filter=do_filter)
        fast_perfs.uninstall()
        os.environ.pop('SCULPTOR_DEPSETUP_ARRAYS', None)
    dt = time.time() - t0
    return anycast, ug_perfs, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pops', default=','.join(POPS5))
    ap.add_argument('--arms', default='b2',
                    help='comma list of arms to gate vs orig: b1,b2')
    ap.add_argument('--no-filter', action='store_true',
                    help='pass do_filter=False (RNG never consumed; '
                         'isolates the deterministic pipeline)')
    args = ap.parse_args()
    pops = args.pops.split(',')

    import core.deployment_setup as ds
    real_cache = ds.CACHE_DIR
    sb = _sandbox(real_cache)
    ds.CACHE_DIR = sb
    os.environ['SCULPTOR_LAT_SHARDS'] = os.path.join(sb, 'lat_shards')

    from core import fast_perfs

    print('=== gate: pops={} seed={} sandbox={}'.format(pops, SEED, sb))
    _reset_violate(real_cache, sb)
    any_o, ugp_o, dt_o = run_arm(ds, pops, 'orig', do_filter=not args.no_filter)
    print('[orig] {:.1f}s  ugs={} '.format(dt_o, len(ugp_o)))

    results = {}
    for arm in args.arms.split(','):
        _reset_violate(real_cache, sb)
        any_f, ugp_f, dt_f = run_arm(ds, pops, arm, do_filter=not args.no_filter)
        stages = {}
        if arm == 'b1':
            stages = dict(fast_perfs.STAGE_T)
        elif arm == 'b2':
            from core import fork_load
            stages = dict(fork_load.STAGE_T)
        print('[{}] {:.1f}s  ugs={}  stages: {}'.format(
            arm, dt_f, len(ugp_f),
            {k: round(v, 2) for k, v in stages.items()}))
        errs = _walk_equal(ugp_o, ugp_f, 'ug_perfs')
        errs += _walk_equal(any_o, any_f, 'anycast')
        n_entries = sum(len(v) for v in ugp_o.values())
        if errs:
            print('GATE FAIL [{}] ({} mismatches, first 20):'.format(
                arm, len(errs)))
            for e in errs[:20]:
                print('  ', e)
            sys.exit(1)
        print('GATE PASS [{}]: byte-exact over {} ugs / {} entries + {} '
              'anycast; order preserved. speedup {:.2f}x '
              '({:.1f}s -> {:.1f}s)'.format(
                  arm, len(ugp_o), n_entries, len(any_o),
                  dt_o / max(dt_f, 1e-9), dt_o, dt_f))
        results[arm] = dt_f


if __name__ == '__main__':
    main()
