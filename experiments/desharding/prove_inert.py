"""Empirical proof for the desharding claim (Tom 2026-08-19: 'All
objects involving splitting UGs are just not used by the solver.
Convince yourself of that.')

Builds the tiny test deployment, splits it into TWO UG chunks, and
constructs three worker views:
    w_full   : n_chunks=1 split (the whole thing — what tests use)
    w_slice0 : static + slice 0
    w_slice1 : static + slice 1
Then evaluates latency_benefit for identical advertisement patterns on
all three. If the sliced ug_* keys were consumed by the solver, w_slice0
and w_slice1 would disagree with w_full (different UG subsets). If they
are inert (solver reads whole_deployment_* only), all three agree
exactly.

Also reports the shapes that expose which view each object is built on:
len(ugs) vs len(whole_deployment_ugs), n_ug, big_lbx width, and the lbx
grid bounds (init_all_vars derives lbx from self.ug_vols — if that read
ever hit sliced stats, per-worker lbx grids would diverge, violating the
'important that every worker has the same lbx' comment).

    SCULPTOR_LP_BACKEND=highs python -m experiments.desharding.prove_inert
"""
import os
import random
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault('SCULPTOR_LP_BACKEND', 'highs')
os.environ.setdefault('MPLBACKEND', 'Agg')

INIT_KWA = {
    'lambduh': 1.0, 'gamma': 0, 'verbose': False, 'n_prefixes': None,
    'with_capacity': False, 'save_run_dir': None,
    'generic_objective': 'avg_latency',
}


def build_worker(subdep, static=None):
    from path_distribution_computer_ray import _LocalPathDistributionComputer
    return _LocalPathDistributionComputer(
        worker_i=0, subdeployment=subdep, init_kwargs=dict(INIT_KWA),
        static_dep=static)


def bench(w, advs):
    """Evaluate each adv pattern immediately; return list of (benefit,
    pdf-checksum) so equality means bitwise-identical answers."""
    out = []
    for i, a in enumerate(advs):
        # sim_rti draws MC routing samples from the GLOBAL numpy RNG —
        # pin it per pattern so all three workers see identical draws,
        # else MC noise masquerades as a slice effect
        np.random.seed(1000 + i)
        benefit, (x, px) = w.latency_benefit(
            a.copy(), retnow=True, generic_obj='avg_latency')
        out.append((float(benefit), float(np.sum(x * px)),
                    float(np.sum(px))))
    return out


def main():
    random.seed(31415)
    np.random.seed(31415)
    from deployment_setup import get_random_deployment
    from helpers import (split_deployment_by_ug,
                         split_deployment_by_ug_separated)

    dep = get_random_deployment('really_friggin_small', port=31600)
    full = split_deployment_by_ug(dep, n_chunks=1)[0]
    static, slices = split_deployment_by_ug_separated(dep, n_chunks=2)
    print('slice sizes:', [len(s['ugs']) for s in slices],
          'whole:', len(dep['ugs']))

    w_full = build_worker(full)
    w_s0 = build_worker(slices[0], static)
    w_s1 = build_worker(slices[1], static)

    for name, w in (('full', w_full), ('slice0', w_s0), ('slice1', w_s1)):
        print('{:>7s}: len(ugs)={} n_ug={} whole={} big_lbx={} '
              'lbx=[{:.6f},{:.6f}]'.format(
                  name, len(w.ugs), w.n_ug,
                  len(w.whole_deployment_ugs),
                  getattr(w, 'big_lbx', np.zeros((0, 0))).shape,
                  float(w.lbx[0]), float(w.lbx[-1])))

    n_popps = len(w_full.popps)
    n_pref = w_full.n_prefixes
    rng = np.random.default_rng(7)
    advs = [(rng.random((n_popps, n_pref)) > .5).astype(np.float64)
            for _ in range(5)]
    advs.append(np.ones((n_popps, n_pref)))

    r_full, r_s0, r_s1 = bench(w_full, advs), bench(w_s0, advs), bench(w_s1, advs)
    agree01 = all(np.allclose(a, b, atol=0, rtol=0)
                  for a, b in zip(r_s0, r_s1))
    agree0f = all(np.allclose(a, b, atol=0, rtol=0)
                  for a, b in zip(r_s0, r_full))
    print('\nper-pattern (benefit, E[x*px], sum px):')
    for i, (a, b, c) in enumerate(zip(r_full, r_s0, r_s1)):
        print(' adv{}: full={} s0={} s1={}'.format(i, a, b, c))
    print('\nslice0 == slice1 : {}'.format(agree01))
    print('slice0 == full   : {}'.format(agree0f))
    print('\nVERDICT: slices {} the solver output'.format(
        'DO NOT affect' if (agree01 and agree0f) else 'AFFECT'))


if __name__ == '__main__':
    main()
