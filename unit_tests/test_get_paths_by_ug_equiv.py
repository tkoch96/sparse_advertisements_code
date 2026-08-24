"""get_paths_by_ug optimization equivalence (2026-08-24 hot-loop pass).

Rebuilds the pre-optimization implementation inline and asserts the
optimized core/solve_lp_assignment.get_paths_by_ug returns IDENTICAL
(available_paths, paths_by_ug-normalized) on real deployments and
randomized rti draws -- the LP input must be byte-equal, not just
equivalent-ish."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SCULPTOR_XOBJS', '1')


def _reference(sas, rti):
    from core.solve_lp_assignment import NO_PATH_INGRESS, NO_ROUTE_LATENCY
    from helpers.helpers import get_difference
    paths_by_ug = {}
    for prefixi in sorted(rti):
        for ug in sas.whole_deployment_ugs:
            if rti[prefixi].get(ug) is None:
                continue
            poppi = sas.popp_to_ind[rti[prefixi][ug]]
            paths_by_ug.setdefault(ug, []).append(poppi)
    ugs_no = get_difference(list(sas.whole_deployment_ugs), list(paths_by_ug))
    out = {}
    _lm = getattr(sas, 'lat_matrix', None)
    for ug in sorted(paths_by_ug):
        if _lm is not None:
            _u = sas.whole_deployment_ug_to_ind[ug]
            so = sorted(set(paths_by_ug[ug]), key=lambda el: _lm[el, _u])
        else:
            _perf = sas.whole_deployment_ug_perfs[ug]
            so = sorted(set(paths_by_ug[ug]),
                        key=lambda el: _perf.get(sas.popps[el],
                                                 NO_ROUTE_LATENCY))
        for poppi in so:
            out[ug, poppi] = None
    ap = sorted(list(out), key=lambda el: el[0])
    for ug in ugs_no:
        ap.append((ug, NO_PATH_INGRESS(sas)))
    return ap, paths_by_ug


def main():
    import pickle
    from unit_tests.bench_path_distribution import build_worker
    from core.solve_lp_assignment import get_paths_by_ug
    w = build_worker('cache/popp_failure_latency_comparison_'
                     'testing_feature-actual-3_papertable3.pkl')
    rng = np.random.RandomState(7)
    for trial in range(5):
        # random rti draw: subset of ugs routed per prefix
        rti = {}
        for pref in range(min(6, w.n_prefixes)):
            m = {}
            for ui in rng.choice(w.n_ug, size=w.n_ug // 2, replace=False):
                ug = w.whole_deployment_ugs[ui]
                m[ug] = w.popps[rng.randint(w.n_popp)]
            rti[pref] = m
        ref_ap, ref_pbu = _reference(w, rti)
        new_ap, new_pbu = get_paths_by_ug(w, rti)
        assert new_ap == ref_ap, 'available_paths diverged (trial {})'.format(trial)
        assert set(new_pbu) == set(ref_pbu) and all(
            sorted(new_pbu[k]) == sorted(ref_pbu[k]) for k in ref_pbu), \
            'paths_by_ug diverged (trial {})'.format(trial)
    print('EQUIVALENT: 5/5 randomized rti trials byte-identical')


if __name__ == '__main__':
    main()
