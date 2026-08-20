"""Standalone painter cell — RAM/time profiling lab (Tom 2026-08-20).

Mirrors sparse_advertisements_v3._solve_one_strategy_in_subprocess:
build the seeded deployment, construct Sparse_Advertisement_Eval with
NO worker pool, run solve_painter end-to-end. A sampler thread logs
[pm] rss lines so we can see the footprint over phases.

Usage (on the lab VM):
  SCULPTOR_PAINTER_MEASURE_CAP=10 python -m experiments.painter_lab.run_standalone \
      testing_feature-actual-25
"""
import os, sys, time, threading

os.environ.setdefault('SCULPTOR_DEPLOYMENT_SEED', '1')
os.environ.setdefault('SCULPTOR_LP_BACKEND', 'highs')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ['SCULPTOR_DISABLE_PARALLEL_STRATEGIES'] = '1'


def _rss_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) // 1024
    except Exception:
        return -1
    return -1


def sampler(stop_evt, peak):
    while not stop_evt.is_set():
        r = _rss_mb()
        peak[0] = max(peak[0], r)
        print('[pm] t={:.0f} rss_mb={}'.format(time.time(), r), flush=True)
        stop_evt.wait(5)


def main():
    dpsize = sys.argv[1] if len(sys.argv) > 1 else 'testing_feature-actual-25'
    stop_evt, peak = threading.Event(), [0]
    threading.Thread(target=sampler, args=(stop_evt, peak), daemon=True).start()

    t0 = time.time()
    print('[pm] phase=import rss={}MB'.format(_rss_mb()), flush=True)
    from deployment_setup import get_random_deployment
    from constants import DEFAULT_EXPLORE
    from helpers import deployment_to_prefixes

    dep = get_random_deployment(dpsize)
    t_dep = time.time()
    print('[pm] phase=deployment_built t={:.0f}s rss={}MB'.format(
        t_dep - t0, _rss_mb()), flush=True)

    from sparse_advertisements_v3 import Sparse_Advertisement_Eval
    sas = Sparse_Advertisement_Eval(
        dep, verbose=True, lambduh=0, with_capacity=False,
        explore=DEFAULT_EXPLORE, using_resilience_benefit=False, gamma=0,
        n_prefixes=deployment_to_prefixes(dep),
        generic_objective='avg_latency')
    sas.update_deployment(dep)
    sas.solutions = {}
    t_sas = time.time()
    print('[pm] phase=sas_built t={:.0f}s rss={}MB'.format(
        t_sas - t_dep, _rss_mb()), flush=True)

    sas.solve_painter()
    t_p = time.time()
    print('[pm] phase=painter_done t={:.0f}s rss={}MB peak={}MB'.format(
        t_p - t_sas, _rss_mb(), peak[0]), flush=True)
    soln = sas.solutions.get('painter', {})
    obj = soln.get('objective')
    print('[pm] RESULT dpsize={} painter_obj={} wall_total={:.0f}s '
          'peak_rss={}MB'.format(dpsize, obj, t_p - t0, peak[0]), flush=True)
    stop_evt.set()


if __name__ == '__main__':
    main()
