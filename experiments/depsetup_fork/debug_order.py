"""Locate the first key-order divergence between mainline and B2 in
the no-filter (deterministic) pipeline at a given pop set."""
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from experiments.depsetup_fork.gate_5pop import (_sandbox, _reset_violate,
                                                 run_arm, POPS5, SEED)


def main():
    pops = sys.argv[1].split(',') if len(sys.argv) > 1 else POPS5
    import deployment_setup as ds
    real_cache = ds.CACHE_DIR
    sb = _sandbox(real_cache)
    ds.CACHE_DIR = sb
    os.environ['SCULPTOR_LAT_SHARDS'] = os.path.join(sb, 'lat_shards')

    _reset_violate(real_cache, sb)
    any_o, ugp_o, dt_o = run_arm(ds, pops, 'orig', do_filter=False)
    _reset_violate(real_cache, sb)
    any_f, ugp_f, dt_f = run_arm(ds, pops, 'b2', do_filter=False)

    ko, kf = list(ugp_o), list(ugp_f)
    so, sf = set(ko), set(kf)
    print('orig {} keys, b2 {} keys; set-equal: {}'.format(
        len(ko), len(kf), so == sf))
    if so != sf:
        only_o = list(so - sf)[:5]
        only_f = list(sf - so)[:5]
        print('only in orig:', only_o)
        print('only in b2:', only_f)
        for k in only_o[:2]:
            print('orig[{}] = {}'.format(k, ugp_o[k]))
        for k in only_f[:2]:
            print('b2[{}] = {}'.format(k, ugp_f[k]))
    n = min(len(ko), len(kf))
    div = -1
    for i in range(n):
        if ko[i] != kf[i]:
            div = i
            break
    if div < 0:
        print('no positional divergence in first {} keys'.format(n))
    else:
        print('first order divergence at index {} / {}'.format(div, n))
        print(' orig[{}..]: {}'.format(div, ko[div:div + 3]))
        print(' b2  [{}..]: {}'.format(div, kf[div:div + 3]))
        for k in (ko[div], kf[div]):
            print(' {} -> orig popps {} | b2 popps {}'.format(
                k, list(ugp_o.get(k, {})), list(ugp_f.get(k, {}))))
    # inner-dict order check on common prefix
    inner_div = 0
    for i in range(min(div if div > 0 else n, n)):
        k = ko[i]
        if list(ugp_o[k]) != list(ugp_f[k]):
            print('inner order diverges at ug {} ({}):'.format(i, k))
            print('  orig:', list(ugp_o[k]))
            print('  b2  :', list(ugp_f[k]))
            inner_div += 1
            if inner_div > 3:
                break
    # value check
    nv = 0
    for k in so & sf:
        for p in ugp_o[k]:
            if p in ugp_f[k] and ugp_o[k][p] != ugp_f[k][p]:
                nv += 1
                if nv < 4:
                    print('value diff {} {}: {} vs {}'.format(
                        k, p, ugp_o[k][p], ugp_f[k][p]))
    print('value mismatches: {}'.format(nv))
    a_ko, a_kf = list(any_o), list(any_f)
    print('anycast set-equal: {}, order-equal: {}'.format(
        set(a_ko) == set(a_kf), a_ko == a_kf))


if __name__ == '__main__':
    main()
