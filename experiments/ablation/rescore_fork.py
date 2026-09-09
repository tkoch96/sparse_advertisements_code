"""Authoritative re-scoring of fork-ladder results. Deliberately boring:

  * one fresh subprocess per seed (never shares a process with a solver)
  * RAY_ADDRESS=local + private RAY_TMPDIR: never attaches to a running
    cluster (attaching to the sweep's busy cluster crashed evals)
  * NO Worker_Manager / no _mp fan-out: every LP is a driver-side
    sas.solve_lp_with_failure_catch call in a plain loop

Metrics per advertisement (all repo LP code, NO_ROUTE volume charged
NO_ROUTE_LATENCY):
  avg_lat                      : steady volume-weighted avg latency
  fail_popp/fail_pop:
    avg_lat_under_failure_abs  : mean over single popp/pop failures of the
                                 vol-weighted avg latency (LP re-assignment)
    opp_avg_lat_under_failure_abs : same for one_per_peering
The pipeline objective is avg_lat + gamma * avg_lat_under_failure_abs
(combined at plot time so gamma is a display choice).

    python -m experiments.ablation.rescore_fork --in-dir cache/ablation/fork_full_res --all
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


MARKER = 'lp_driver_v2'


class FullObjectiveScorer:
    """THE trusted scorer as a reusable object (2026-09-09, for the full-
    objective-over-iterations curve): the same evaluator, steady LP and
    single-peering failure sweep rescore_seed uses. full(adv, gamma) =
    steady avg latency + gamma * SUM over popp failures of the avg latency
    under that failure (30 s no-route sentinel charged), i.e. the ladder's
    full objective; opp_full(gamma) is the one-per-peering anchor."""

    def __init__(self, seed, dpsize, dep_file=None):
        os.environ['RAY_ADDRESS'] = 'local'
        os.environ['RAY_TMPDIR'] = '/tmp/ray_rescore_{}'.format(os.getpid())
        os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(seed)
        os.environ.setdefault('MPLBACKEND', 'Agg')
        from helpers.constants import DEFAULT_EXPLORE
        from evaluations.wrapper_eval import capacity
        from core.deployment_setup import get_random_deployment
        from core.sparse_advertisements_v3 import Sparse_Advertisement_Eval
        from helpers.helpers import deployment_to_prefixes
        dep_file = dep_file or os.environ.get('SCULPTOR_ABLATION_DEP_FILE', '')
        if dep_file:
            import pickle as _pickle
            dep = _pickle.load(open(dep_file.format(seed=seed), 'rb'))
        else:
            dep = get_random_deployment(dpsize)
        dep['generic_objective'] = 'avg_latency'
        self.sas = Sparse_Advertisement_Eval(
            dep, verbose=False, lambduh=0, with_capacity=capacity,
            explore=DEFAULT_EXPLORE, using_resilience_benefit=False, gamma=0,
            n_prefixes=deployment_to_prefixes(dep), generic_objective='avg_latency')
        self.vols = np.asarray(self.sas.ug_vols)
        self.n_popps = int(self.sas.n_popps)
        self._opp = None

    def steady(self, adv):
        ret = self.sas.solve_lp_with_failure_catch(np.asarray(adv, dtype=float))
        return float(np.average(np.asarray(ret['lats_by_ug']), weights=self.vols))

    def fail_mean(self, adv):
        a = np.asarray(adv, dtype=float)
        per = []
        for popp in self.sas.popps:
            a2 = np.copy(a)
            a2[[self.sas.popp_to_ind[popp]], :] = 0
            if a2.sum() == 0:
                per.append(30000.0)
                continue
            per.append(self.steady(a2))
        return float(np.mean(per))

    def _drop_lp_cache(self):
        # The evaluator memoizes every LP it solves (keyed by advertisement,
        # with full per-UG path maps). One full() = 1 + n_popps LPs; scoring a
        # cell's whole iteration history cached ~3000 of them per process and
        # 12 processes filled a 123 GB box in 10 min (2026-09-09). Nothing
        # here re-solves an advertisement, so the cache buys nothing: drop it.
        try:
            self.sas.linear_prog_soln_cache = {k: {} for k in self.sas.linear_prog_soln_cache}
        except Exception:
            pass

    def full(self, adv, gamma):
        v = self.steady(adv) + float(gamma) * self.n_popps * self.fail_mean(adv)
        self._drop_lp_cache()
        return v

    def opp_full(self, gamma):
        if self._opp is None:
            self._opp = (self.steady(np.eye(self.n_popps)), self.fail_mean(np.eye(self.n_popps)))
            self._drop_lp_cache()
        return self._opp[0] + float(gamma) * self.n_popps * self._opp[1]


def rescore_seed(seed, in_dir, dpsize):
    fns = sorted(glob.glob(os.path.join(in_dir, 'seed_{}_*.json'.format(seed))))
    todo = []
    for fn in fns:
        with open(fn) as f:
            r = json.load(f)
        # SCULPTOR_RESCORE_REDO_CONG=1: additive backfill pass -- re-score
        # cells that were rescored BEFORE congestion harvesting existed
        # (Tom 2026-08-31: congested-volume columns for the ladder table)
        _redo_cong = os.environ.get('SCULPTOR_RESCORE_REDO_CONG', '0') == '1'
        if 'adv' in r and (r.get('fail_eval') != MARKER
                           or (_redo_cong
                               and 'steady_frac_congested' not in r)):
            todo.append((fn, r))
    if not todo:
        print('[rescore seed {}] nothing to do'.format(seed), flush=True)
        return

    # isolation: fresh local Ray, never attach to a running cluster
    os.environ['RAY_ADDRESS'] = 'local'
    os.environ['RAY_TMPDIR'] = '/tmp/ray_rescore_{}'.format(os.getpid())
    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(seed)
    os.environ.setdefault('MPLBACKEND', 'Agg')

    from helpers.constants import DEFAULT_EXPLORE
    from evaluations.wrapper_eval import capacity
    from core.deployment_setup import get_random_deployment
    from core.sparse_advertisements_v3 import Sparse_Advertisement_Eval
    from helpers.helpers import deployment_to_prefixes

    # dep-file mode (Tom 2026-08-31, ablation CDF on paper deployments):
    # score on the SAME pinned deployment the cells ran on, not a fresh
    # seeded draw ('{seed}' in the path expands)
    _dep_file = os.environ.get('SCULPTOR_ABLATION_DEP_FILE', '')
    if _dep_file:
        import pickle as _pickle
        _dep_file = _dep_file.format(seed=seed)
        dep = _pickle.load(open(_dep_file, 'rb'))
        print('[rescore seed {}] deployment from {} (dep-file mode)'
              .format(seed, _dep_file), flush=True)
    else:
        dep = get_random_deployment(dpsize)
    dep['generic_objective'] = 'avg_latency'
    sas = Sparse_Advertisement_Eval(
        dep, verbose=False, lambduh=0, with_capacity=capacity,
        explore=DEFAULT_EXPLORE, using_resilience_benefit=False, gamma=0,
        n_prefixes=deployment_to_prefixes(dep), generic_objective='avg_latency')
    vols = np.asarray(sas.ug_vols)

    NO_ROUTE_MARK = 29999.0   # lats >= this are NO_ROUTE/congested charges

    def score(adv):
        ret = sas.solve_lp_with_failure_catch(np.asarray(adv, dtype=float))
        lats = np.asarray(ret['lats_by_ug'])
        mean = float(np.average(lats, weights=vols))
        frac_cong = float(vols[lats >= NO_ROUTE_MARK].sum() / vols.sum())
        return mean, frac_cong

    def steady(adv):
        return score(adv)[0]

    def scenarios(which):
        if which == 'popps':
            for popp in sas.popps:
                yield [sas.popp_to_ind[popp]]
        else:
            for pop in sas.pops:
                yield [sas.popp_to_ind[p] for p in sas.popps if p[0] == pop]

    # SCULPTOR_RESCORE_STORE_SCENARIOS=1: additionally persist the
    # per-failure-scenario latencies (for scenario-level CDFs, cdf_fork.py).
    # Default (unset) keeps the original aggregate-only JSON schema.
    store_scen = os.environ.get('SCULPTOR_RESCORE_STORE_SCENARIOS', '0') == '1'

    def fail_abs(adv, which):
        a = np.asarray(adv, dtype=float)
        per_s, per_s_cong = [], []
        for failed in scenarios(which):
            a2 = np.copy(a)
            a2[failed, :] = 0
            if a2.sum() == 0:
                per_s.append(float(np.average(
                    np.full(len(vols), 30000.0), weights=vols)))
                per_s_cong.append(1.0)
                continue
            m, fc = score(a2)
            per_s.append(m)
            per_s_cong.append(fc)
        return float(np.mean(per_s)), per_s, float(np.mean(per_s_cong))

    opp_adv = np.eye(sas.n_popps)
    opp_steady, opp_steady_cong = score(opp_adv)
    opp_fail, opp_fail_scen, opp_fail_cong = {}, {}, {}
    for w in ('popps', 'pops'):
        opp_fail[w], opp_fail_scen[w], opp_fail_cong[w] = fail_abs(opp_adv, w)

    for fn, r in todo:
        old = r.get('diff_vs_opp')
        r['avg_lat'], r['steady_frac_congested'] = score(r['adv'])
        r['opp_avg_lat'] = opp_steady
        r['opp_steady_frac_congested'] = opp_steady_cong
        r['diff_vs_opp'] = r['avg_lat'] - opp_steady
        for which, key in (('popps', 'fail_popp'), ('pops', 'fail_pop')):
            mean_abs, per_scen, mean_cong = fail_abs(r['adv'], which)
            r[key] = {
                'avg_lat_under_failure_abs': mean_abs,
                'opp_avg_lat_under_failure_abs': opp_fail[which],
                'avg_frac_congested': mean_cong,
                'opp_avg_frac_congested': opp_fail_cong[which],
            }
            if store_scen:
                r[key]['per_scenario_lats'] = per_scen
                r[key]['opp_per_scenario_lats'] = opp_fail_scen[which]
        r.pop('opp_fail', None)
        r['rescored'] = True
        r['fail_eval'] = MARKER
        r['driver_diff_vs_opp'] = old
        with open(fn, 'w') as f:
            json.dump(r, f, indent=2, default=float)
        comb = r['diff_vs_opp'] + 4 * (r['fail_popp']['avg_lat_under_failure_abs']
                                       - r['fail_popp']['opp_avg_lat_under_failure_abs'])
        print('[rescore seed {} {}] steady={:+.3f} combined(g4)={:+.3f}'.format(
            seed, r['rung'], r['diff_vs_opp'], comb), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', required=True)
    p.add_argument('--dpsize', default='small')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--all', action='store_true')
    p.add_argument('--port-base', type=int, default=0, help='unused; kept for CLI compat')
    args = p.parse_args()

    if args.seed is not None:
        rescore_seed(args.seed, args.in_dir, args.dpsize)
        return
    assert args.all
    seeds = sorted({int(os.path.basename(fn).split('_')[1])
                    for fn in glob.glob(os.path.join(args.in_dir, 'seed_*_*.json'))})
    for seed in seeds:
        subprocess.run([sys.executable, '-m', 'experiments.ablation.rescore_fork',
                        '--in-dir', args.in_dir, '--dpsize', args.dpsize,
                        '--seed', str(seed)], cwd=_REPO_ROOT)


if __name__ == '__main__':
    main()
