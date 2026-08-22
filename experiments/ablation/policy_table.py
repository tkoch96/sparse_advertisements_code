"""Policy-ladder analysis: the one-stop table for the current study design
(solver rungs x probing policies, georand world, exit-on-budget).

Auto-discovers arms under --root (default cache/ablation/policy_ladder,
layout <root>/<arm>/N<budget>/seed_<s>_<rung>.json, rescored by the queue)
and prints, per arm x N: median/per-seed combined score, healthy counts,
median exit iteration, pooled probe-reason attribution. When the
LP-scored sidecars exist (cache/model_error/rerank/policy_* and
cache/model_error/steady/policy_steady.json, produced by
core.rerank_ladder / steady_metrics), also prints
steady congestion, clean routed latency, and failure congestion medians.

    python -m experiments.ablation.policy_table [--root ...]
"""
import argparse
import collections
import glob
import json
import os
import statistics as st
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def comb(r):
    return r['diff_vs_opp'] + 4 * (
        r['fail_popp']['avg_lat_under_failure_abs']
        - r['fail_popp']['opp_avg_lat_under_failure_abs'])


def discover(root):
    arms = {}
    for fn in glob.glob(os.path.join(root, '*', 'N*', 'seed_*.json')):
        parts = fn.split(os.sep)
        arm, ndir = parts[-3], parts[-2]
        r = json.load(open(fn))
        if not r.get('rescored'):
            continue
        arms.setdefault((arm, int(ndir[1:])), []).append(r)
    return arms


def lp_sidecars(arm, n):
    """(steady_cong, clean_lat, pfail) medians if the sidecar files exist."""
    out = [None, None, None]
    rer = glob.glob('cache/model_error/rerank/policy_{}_N{}/seed_*.json'
                    .format(arm, n))
    if rer:
        pf = []
        for fn in rer:
            r = json.load(open(fn))
            for e in r['rungs'].values():
                if e.get('popp_fail'):
                    pf.append(e['popp_fail']['mean'])
        if pf:
            out[2] = st.median(pf)
    sfn = 'cache/model_error/steady/policy_steady.json'
    if os.path.exists(sfn):
        sc, cl = [], []
        for e in json.load(open(sfn)):
            if not e.get('solved'):
                continue
            p = e['dir'].split(os.sep)
            if len(p) >= 2 and p[-2].endswith(arm) and p[-1] == 'N{}'.format(n):
                sc.append(e['steady_congested_frac'])
                if e['clean_avg_lat'] is not None:
                    cl.append(e['clean_avg_lat'])
        if sc:
            out[0] = st.median(sc)
        if cl:
            out[1] = st.median(cl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='cache/ablation/policy_ladder')
    args = ap.parse_args()
    arms = discover(args.root)
    order = sorted(arms, key=lambda k: (k[0], k[1]))
    print('{:>22} | {:>8} {:>7} {:>8} | {:>7} {:>9} {:>7} | {:>18}'.format(
        'arm', 'median', 'healthy', 'exit-it', 'st.cong', 'clean-ms',
        'pfail', 'probe reasons'))
    for arm, n in order:
        rs = arms[(arm, n)]
        cs = [comb(r) for r in rs]
        reasons = collections.Counter()
        for r in rs:
            for k, v in (r.get('probe_reasons') or {}).items():
                reasons[k] += v
        sc, cl, pf = lp_sidecars(arm, n)
        fmt = lambda v, f: ('{:' + f + '}').format(v) if v is not None else '--'
        print('{:>22} | {:>+8.0f} {:>5}/{} {:>8.0f} | {:>7} {:>9} {:>7} | {}'.format(
            '{} N{}'.format(arm, n), st.median(cs),
            sum(c < 100 for c in cs), len(cs),
            st.median(r['n_iters'] for r in rs),
            fmt(sc, '.1%'), fmt(cl, '.1f'), fmt(pf, '.1%'),
            dict(reasons) or '-'))


if __name__ == '__main__':
    main()
