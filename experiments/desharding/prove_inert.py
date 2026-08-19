"""Desharding parity gate (Tom 2026-08-19).

Pre-removal, this script proved the UG slices bitwise-inert (three
worker views, identical benefits, MC pinned — see NOTES.md). The
EXPECTED values below were captured under PYTHONHASHSEED=0 from the
PRE-removal code on the raw deployment, and the post-removal code
reproduces them bitwise (verified 2026-08-19).

HASH-SEED CAVEAT (hard-won): benefits are only reproducible ACROSS
processes with PYTHONHASHSEED pinned — set/dict iteration order of
string keys shifts RNG alignment otherwise. Same-process comparisons
don't need it. This gate re-execs itself with PYTHONHASHSEED=0.

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

# Captured 2026-08-19 pre-removal (see NOTES.md).
EXPECTED = [
    -9.914223903535007,
    -41.92469654196053,
    -10.183844102827221,
    -15.447982984283588,
    -38.979463851859634,
    -14.900979397521688,
]

if os.environ.get('PYTHONHASHSEED') != '0':
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    random.seed(31415)
    np.random.seed(31415)
    from deployment_setup import get_random_deployment
    from path_distribution_computer_ray import _LocalPathDistributionComputer

    dep = get_random_deployment('really_friggin_small', port=31600)
    w = _LocalPathDistributionComputer(
        worker_i=0, deployment=dep, init_kwargs=dict(INIT_KWA))
    print('worker: len(ugs)={} n_ug={} whole={}'.format(
        len(w.ugs), w.n_ug, len(w.whole_deployment_ugs)))
    assert len(w.ugs) == len(w.whole_deployment_ugs), \
        'worker no longer holds the full deployment?'

    n_popps = len(w.popps)
    n_pref = w.n_prefixes
    rng = np.random.default_rng(7)
    advs = [(rng.random((n_popps, n_pref)) > .5).astype(np.float64)
            for _ in range(5)]
    advs.append(np.ones((n_popps, n_pref)))

    got = []
    for i, a in enumerate(advs):
        np.random.seed(1000 + i)
        benefit, _ = w.latency_benefit(
            a.copy(), retnow=True, generic_obj='avg_latency')
        got.append(float(benefit))

    ok = True
    for i, (g, e) in enumerate(zip(got, EXPECTED)):
        match = (g == e)
        ok = ok and match
        print(' adv{}: got={!r} expected={!r} {}'.format(
            i, g, e, 'OK' if match else 'MISMATCH'))
    print('\nGATE: {}'.format('PASS (bitwise)' if ok else 'FAIL'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
