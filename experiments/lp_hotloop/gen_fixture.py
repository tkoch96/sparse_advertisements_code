"""Capture (deployment, adv) -> solution fixture for the LP hot-loop
sandbox (Tom 2026-08-19: unit test inputs->outputs across semi-random
requests; workshop optimizations against it)."""
import os, random, sys, pickle
import numpy as np
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _R)
os.environ.setdefault('SCULPTOR_LP_BACKEND','highs'); os.environ.setdefault('MPLBACKEND','Agg')

def build_worker(dpsize='decent'):
    random.seed(31415); np.random.seed(31415)
    from core.deployment_setup import get_random_deployment
    from core.path_distribution_computer import _LocalPathDistributionComputer
    dep = get_random_deployment(dpsize, port=31601)
    return _LocalPathDistributionComputer(0, dep, {'lambduh':1.0,'gamma':0,
        'verbose':False,'n_prefixes':None,'with_capacity':False,
        'save_run_dir':None,'generic_objective':'avg_latency'})

def requests(w, n=25):
    rng = np.random.default_rng(11)
    base = (rng.random((w.n_popps, w.n_prefixes)) > .8).astype(np.float64)
    yield base.copy()
    for i in range(n - 1):
        a = base.copy()  # semi-random: base plus 1-3 flips (grad-like)
        for f in range(1 + int(rng.integers(0, 3))):
            idx = (int(rng.integers(0, w.n_popps)), int(rng.integers(0, w.n_prefixes))); a[idx] = 1.0 - a[idx]
        yield a

if __name__ == '__main__':
    w = build_worker()
    recs = []
    for i, a in enumerate(requests(w)):
        np.random.seed(5000 + i)
        b, (x, px) = w.latency_benefit(a.copy(), retnow=True, generic_obj='avg_latency')
        recs.append({'adv': a, 'benefit': float(b),
                     'pdf_sum': float(np.sum(px)), 'exp': float(np.sum(x*px))})
        print(i, round(float(b), 6))
    pickle.dump(recs, open(os.path.join(_R,'experiments/lp_hotloop/fixture_decent.pkl'),'wb'))
    print('wrote', len(recs), 'records')
