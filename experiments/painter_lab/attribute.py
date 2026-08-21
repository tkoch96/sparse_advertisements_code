"""Painter RAM attribution (Tom 2026-08-20): which retained structure
grows during the painter measure/recompute loop?

Wraps stop_tracker to deep-size candidate attributes at every painter
iteration. Run small enough for the Mac: actual-10.

Usage: python -m experiments.painter_lab.attribute testing_feature-actual-10
"""
import os, sys, time

os.environ.setdefault('SCULPTOR_DEPLOYMENT_SEED', '1')
os.environ.setdefault('SCULPTOR_LP_BACKEND', 'highs')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ['SCULPTOR_DISABLE_PARALLEL_STRATEGIES'] = '1'
os.environ.setdefault('SCULPTOR_PAINTER_MEASURE_CAP', '6')

import numpy as np


def deep_sz(o, seen=None, depth=0):
    if seen is None:
        seen = set()
    if id(o) in seen or depth > 8:
        return 0
    seen.add(id(o))
    if isinstance(o, np.ndarray):
        return o.nbytes
    s = sys.getsizeof(o, 0)
    if isinstance(o, dict):
        s += sum(deep_sz(k, seen, depth+1) + deep_sz(v, seen, depth+1)
                 for k, v in o.items())
    elif isinstance(o, (list, tuple, set, frozenset)):
        s += sum(deep_sz(x, seen, depth+1) for x in o)
    elif hasattr(o, '__dict__'):
        s += deep_sz(o.__dict__, seen, depth+1)
    return s


ATTRS = ['measured_prefs', 'parent_tracker', 'calc_cache',
         'ug_perfs', 'whole_deployment_ug_perfs', 'popp_to_ug',
         'measured', 'all_rb_calls_results_popps', 'linear_prog_soln_cache',
         'this_time_ip_cache', 'ingress_priorities', 'advs']


def _rss_mb():
    import resource
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v // (1024 * 1024) if sys.platform == 'darwin' else v // 1024


def main():
    dpsize = sys.argv[1] if len(sys.argv) > 1 else 'testing_feature-actual-10'
    from core.deployment_setup import get_random_deployment
    from helpers.constants import DEFAULT_EXPLORE
    from helpers.helpers import deployment_to_prefixes
    dep = get_random_deployment(dpsize)
    from core.sparse_advertisements_v3 import Sparse_Advertisement_Eval
    sas = Sparse_Advertisement_Eval(
        dep, verbose=False, lambduh=0, with_capacity=False,
        explore=DEFAULT_EXPLORE, using_resilience_benefit=False, gamma=0,
        n_prefixes=deployment_to_prefixes(dep),
        generic_objective='avg_latency')
    sas.update_deployment(dep)
    sas.solutions = {}

    import core.painter as painter_mod
    solver_holder = []

    # find the painter solver object: solve_painter constructs it; easiest
    # hook is patching Painter_Adv_Solver.stop_tracker on the class.
    orig_stop = painter_mod.Painter_Adv_Solver.stop_tracker

    def instrumented_stop(self, **kwargs):
        orig_stop(self, **kwargs)
        rows = []
        seen = set()               # SHARED across attrs: partition, no
        seen.add(id(self))         # double counting via back-references
        # size generic_objective LAST so its .sas backref doesn't swallow
        # the whole solver graph into its number
        items = sorted(vars(self).items(),
                       key=lambda kv: kv[0] == 'generic_objective')
        for a, v in items:
            if a == 'calc_cache':
                for ck, cv in v.all_caches.items():
                    b = deep_sz(cv, seen)
                    if b > 1e6:
                        rows.append(('calc_cache.' + ck, b))
                continue
            try:
                b = deep_sz(v, seen)
            except Exception:
                continue
            if b > 1e6:
                rows.append((a, b))
        rows.sort(key=lambda kv: -kv[1])
        print('[attr] iter={} peak_rss={}MB  '.format(self.iter, _rss_mb()) +
              '  '.join('{}={:.0f}MB'.format(k, b/1e6) for k, b in rows),
              flush=True)

    painter_mod.Painter_Adv_Solver.stop_tracker = instrumented_stop
    t0 = time.time()
    sas.solve_painter()
    print('[attr] DONE wall={:.0f}s peak_rss={}MB'.format(
        time.time() - t0, _rss_mb()), flush=True)


if __name__ == '__main__':
    main()
