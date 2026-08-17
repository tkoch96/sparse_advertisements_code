"""Fresh-process recompute of a fracb cell's hinge scalar for BOTH the
trained adv and one-per-peering, under the backend given by
SCULPTOR_LP_BACKEND. Prints hinge excess + objective for each. Usage:

    env <world+xobjs env> SCULPTOR_LP_BACKEND=highs \
        python -m experiments.solver_fork.debug_fracb_cell /tmp/fracb_worst.json
"""
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    spec = json.load(open(sys.argv[1]))
    for k, v in spec['env'].items():
        os.environ[k] = v
    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(spec['seed'])
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('RAY_ADDRESS', 'local')
    os.environ.setdefault('RAY_TMPDIR', '/tmp/ray_dbg_fracb')

    from constants import DEFAULT_EXPLORE
    from wrapper_eval import capacity
    from deployment_setup import get_random_deployment
    from sparse_advertisements_v3 import Sparse_Advertisement_Eval
    from helpers import deployment_to_prefixes
    from experiments.model_error.objectives import (
        register, solve_lp_frac_beyond_optimal)
    import gpshim

    register()
    dep = get_random_deployment('small')
    dep['generic_objective'] = 'avg_latency'
    sas = Sparse_Advertisement_Eval(
        dep, verbose=False, lambduh=0, with_capacity=capacity,
        explore=DEFAULT_EXPLORE, using_resilience_benefit=False, gamma=0,
        n_prefixes=deployment_to_prefixes(dep),
        generic_objective='avg_latency')

    def score(name, adv):
        rti, _ = sas.calculate_ground_truth_ingress(
            np.asarray(adv, dtype=float))
        out = solve_lp_frac_beyond_optimal(sas, rti, 'frac_beyond_optimal')
        print('[{}] backend={} objective={} hinge_excess_ms={} '
              'frac_beyond={} solved={}'.format(
                  name, gpshim.BACKEND,
                  round(out.get('objective', float('nan')), 6),
                  round(out['hinge_excess_ms'], 6)
                  if out.get('hinge_excess_ms') is not None else None,
                  round(out.get('frac_beyond', float('nan')), 6),
                  out.get('solved')), flush=True)
        return out

    order = os.environ.get('DEBUG_ORDER', 'opp_first')
    if order == 'arm_first':
        score('arm', np.asarray(spec['adv'], dtype=float))
        score('opp', np.eye(sas.n_popps))
    else:
        score('opp', np.eye(sas.n_popps))
        score('arm', np.asarray(spec['adv'], dtype=float))


if __name__ == '__main__':
    main()
