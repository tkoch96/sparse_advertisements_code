#!/usr/bin/env python
"""Latency-breakdown bench for Path_Distribution_Computer (Tom 2026-08-24).

    python unit_tests/bench_path_distribution.py                # all scenarios
    python unit_tests/bench_path_distribution.py --candidates 60
    python unit_tests/bench_path_distribution.py --scenarios warm,cold

Instantiates ONE _LocalPathDistributionComputer directly (no Ray) on a
real harvested deployment and drives the actual hot path
(latency_benefit -> generic_objective_pdf -> solve_generic_lp_persistent)
with a stream of single-popp-flip candidates -- the same shape as the
compressed-LB mega-batches. Reports the worker's own phase timing table,
per-solve wall, HiGHS model size, and RSS per scenario:

  warm       defaults (incremental UB diffs, basis preserved)
  cold       SCULPTOR_LP_INCREMENTAL=0 (full sweep + cold simplex)
  mlu_off    alternate avg_latency/max_util, SCULPTOR_LP_INCR_MLU=0
  mlu_on     same alternation with SCULPTOR_LP_INCR_MLU=1
  rebuild    warm + init_persistent_lp() rebuild every --rebuild-every

The questions this answers (2026-08-24): (a) is optimize warm-started;
(b) is a periodic model rebuild worth it for RAM/latency; (c) where the
per-call milliseconds actually go.
"""
import argparse
import copy
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SCULPTOR_XOBJS', '1')
os.makedirs('logs', exist_ok=True)

import numpy as np


def rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024 if sys.platform == 'darwin' else 1024)


def build_worker(pickle_fn):
    import pickle
    m = pickle.load(open(pickle_fn, 'rb'))
    deployment = copy.deepcopy(m['deployment'][0])
    deployment['port'] = 0
    kwa = dict(m['settings'][0] or {})
    kwa.pop('save_run_dir', None)
    kwa['verbose'] = False
    from core.path_distribution_computer import _LocalPathDistributionComputer
    t0 = time.time()
    w = _LocalPathDistributionComputer(0, deployment, kwa)
    print('worker init: {:.1f}s | popps={} ugs={} prefixes={}'.format(
        time.time() - t0, w.n_popp, w.n_ug, w.n_prefixes))
    return w


def gradient_step_load(w, n_steps=4, flips_per_step=12, seed=1234,
                       congested=False):
    """The realistic hot-loop shape (Tom 2026-08-24): a handful of
    gradient-step ITERATIONS, each = one base advertisement (the step's
    optimization point) plus its compressed-LB flip batch; the base
    moves a few popps between steps like an applied gradient update.
    Yields (step_idx, adv); per-step accounting exposes variance."""
    rng = np.random.RandomState(seed)
    base = np.zeros((w.n_popp, w.n_prefixes))
    base[:, 0] = 1                                # anycast prefix
    # congested=True models the belief-phase regime: few popps on, caps
    # exceeded, so the standard LP goes infeasible and the persistent
    # solver falls back to MLU mode -- the real source of the
    # standard<->MLU alternation that disables incrementality.
    density = (w.n_popp // 20) if congested else (w.n_popp // 6)
    for j in range(1, w.n_prefixes):
        base[rng.choice(w.n_popp, size=max(2, density),
                        replace=False), j] = 1
    if congested:
        base[rng.choice(w.n_popp, size=int(w.n_popp * .8),
                        replace=False), 0] = 0
    for step in range(n_steps):
        yield step, base.copy()
        for _ in range(flips_per_step):
            a = base.copy()
            i, j = rng.randint(w.n_popp), rng.randint(w.n_prefixes)
            a[i, j] = 1 - a[i, j]
            yield step, a
        # apply a "gradient step": toggle a few entries of the base
        for _ in range(3):
            i, j = rng.randint(w.n_popp), rng.randint(w.n_prefixes)
            base[i, j] = 1 - base[i, j]


def run_scenario(w, name, n, env, objs, rebuild_every=0):
    for k, v in env.items():
        os.environ[k] = v
    for k in w.timing:
        w.timing[k] = 0
    # distinct candidates defeat the lb memo, but clear it anyway so no
    # scenario inherits another's cache
    try:
        w.calc_cache.all_caches['lb'] = {}
    except Exception:
        pass
    rss0, t0 = rss_mb(), time.time()
    n_done = 0
    step_walls = {}
    for i, (step, a) in enumerate(gradient_step_load(
            w, n_steps=n[0], flips_per_step=n[1],
            congested=(len(objs) > 1 or objs[0] == 'congested'))):
        if rebuild_every and i and i % rebuild_every == 0:
            trb = time.time()
            for attr in ('_active_keys', '_last_active_vars', '_last_mlu'):
                if hasattr(w, attr):
                    delattr(w, attr)
            w.var_pool = {}
            w.init_persistent_lp()
            print('    [rebuild] at candidate {} took {:.2f}s'.format(
                i, time.time() - trb))
        _tc = time.time()
        w.latency_benefit(a, generic_obj='avg_latency')
        step_walls.setdefault(step, []).append(time.time() - _tc)
        n_done += 1
    wall = time.time() - t0
    tt = {k: v for k, v in sorted(w.timing.items(), key=lambda kv: -kv[1])
          if v > 0.01}
    print('\n== {} ==  {} solves in {:.1f}s -> {:.0f} ms/solve | '
          'var_pool={} | RSS {:.0f}->{:.0f} MB'.format(
              name, n_done, wall, 1000 * wall / max(n_done, 1),
              len(getattr(w, 'var_pool', {})), rss0, rss_mb()))
    for k, v in tt.items():
        print('    {:38s} {:7.2f}s  ({:4.0f}%)'.format(k, v, 100 * v / wall))
    for step in sorted(step_walls):
        d = step_walls[step]
        print('    step {:>2d}: {:5.0f} ms/solve mean, {:5.0f} min, '
              '{:6.0f} max over {} solves'.format(
                  step, 1000 * sum(d) / len(d), 1000 * min(d),
                  1000 * max(d), len(d)))
    return wall / max(n_done, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pickle', default='cache/popp_failure_latency_'
                    'comparison_testing_feature-actual-20_dep_sweep_20.pkl')
    ap.add_argument('--steps', type=int, default=4,
                    help='gradient-step iterations (3-5 typical)')
    ap.add_argument('--flips-per-step', type=int, default=12)
    ap.add_argument('--rebuild-every', type=int, default=15)
    ap.add_argument('--scenarios', default='warm,cold,mlu_off,mlu_on,rebuild')
    a = ap.parse_args()
    w = build_worker(a.pickle)
    lat, mlu = 'avg_latency', 'max_util'
    S = {
        'warm':    (dict(SCULPTOR_LP_INCREMENTAL='1'), [lat], 0),
        'cold':    (dict(SCULPTOR_LP_INCREMENTAL='0'), [lat], 0),
        'mlu_off': (dict(SCULPTOR_LP_INCREMENTAL='1',
                         SCULPTOR_LP_INCR_MLU='0'), ['congested'], 0),
        'mlu_on':  (dict(SCULPTOR_LP_INCREMENTAL='1',
                         SCULPTOR_LP_INCR_MLU='1'), ['congested'], 0),
        'rebuild': (dict(SCULPTOR_LP_INCREMENTAL='1'), [lat],
                    a.rebuild_every),
    }
    results = {}
    names = a.scenarios.split(',')
    # pass 0 warms the var_pool/model so scenario ORDER does not decide
    # the winner (the pool is built once and shared); report pass 1.
    for rep in range(2):
        if rep:
            print('\n======== steady-state pass ========')
        for name in names:
            env, objs, reb = S[name.strip()]
            r = run_scenario(w, name, (a.steps, a.flips_per_step),
                             env, objs, reb)
            if rep:
                results[name] = r
    print('\n== summary (ms/solve) ==')
    base = results.get('warm')
    for k, v in results.items():
        print('  {:10s} {:7.0f}{}'.format(k, v * 1000,
              '   ({:+.0f}% vs warm)'.format(100 * (v - base) / base)
              if base and k != 'warm' else ''))


if __name__ == '__main__':
    main()
