"""Merge per-cell EODS pickles into the classic metrics_by_dpsize cache
(the exact shape evaluate_over_deployment_sizes.pull_results_new writes),
so the existing paper-plot path runs unchanged.

Merge semantics follow eval_ladder_metrics' per-seed convention: purely
mechanical — list-valued stats entries per soln_type are concatenated
across sims in seed order; scalar/dict entries must agree (asserted)
or be per-sim lists already. NO metric logic here.

    python -m experiments.eods.merge_eods --store cache/eods/v1 \
        --out cache/eods/v1/metrics_by_dpsize.pkl
"""
import argparse
import glob
import os
import pickle
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def merge_stats(per_sim):
    """Merge a list of per-sim metrics dicts (each: stats_* keyed by
    soln_type holding per-sim lists of length 1) by concatenation."""
    merged = {}
    for m in per_sim:
        for k, v in m.items():
            if not str(k).startswith('stats_'):
                continue
            if k not in merged:
                merged[k] = {}
            if isinstance(v, dict):
                for soln, vals in v.items():
                    if isinstance(vals, list):
                        merged[k].setdefault(soln, []).extend(vals)
                    else:
                        merged[k].setdefault(soln, []).append(vals)
            else:
                merged[k].setdefault('_scalar', []).append(v)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', default='cache/eods/v1')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    store = (args.store if os.path.isabs(args.store)
             else os.path.join(_REPO_ROOT, args.store))
    out = args.out or os.path.join(store, 'metrics_by_dpsize.pkl')

    metrics_by_dpsize = {}
    for dp_dir in sorted(glob.glob(os.path.join(store, 'actual-*'))):
        dp = int(dp_dir.split('actual-')[-1])
        per_sim = []
        for fn in sorted(
                glob.glob(os.path.join(dp_dir, 'N1', 'seed_*_metrics.pkl')),
                key=lambda f: int(re.search(r'seed_(\d+)_', f).group(1))):
            with open(fn, 'rb') as f:
                per_sim.append(pickle.load(f))
        if per_sim:
            metrics_by_dpsize[dp] = merge_stats(per_sim)
            print('dpsize {}: merged {} sims, {} stats keys'.format(
                dp, len(per_sim), len(metrics_by_dpsize[dp])))
    with open(out, 'wb') as f:
        pickle.dump(metrics_by_dpsize, f)
    print('wrote', out)


if __name__ == '__main__':
    main()
