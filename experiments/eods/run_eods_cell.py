"""One EODS-family cell, speaking the queue's run_fork_ladder CLI +
result-JSON convention so queue/governor/dash/fleet run it unchanged.

Modes (SCULPTOR_EODS_MODE):
  sizes (default) — evaluate_over_deployment_sizes unit: one (dpsize,
      sim). dpsize token 'actual-N' -> 'testing_feature-actual-N'.
      --seed = sim index; the cell's N-dir is always N1.
  prefixes — evaluate_over_n_prefixes unit: one (dpsize, prefix_num).
      The queue's swept N VALUE (delivered per-cell as
      SCULPTOR_ABLATION_PROBE_N) IS the prefix count, so the dash
      N-axis natively shows prefixes. Deployment comes from the classic
      popp_failure_latency_comparison_<dpsize>.pkl (same as the
      standalone script).

Both modes call eval_all_solution_types.evaluate_all_metrics (the module
BOTH standalone sweeps use) with nsim=1 and a per-cell resumable pickle
(use_performance_metrics_fn). Strategies via SCULPTOR_SOLN_TYPES.
Merge with experiments.eods.merge_eods.
"""
import argparse
import json
import os
import pickle
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--rung', default='eods')
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--max-iter', type=int, default=0)
    ap.add_argument('--dpsize', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    mode = os.environ.get('SCULPTOR_EODS_MODE', 'sizes')
    out_fn = os.path.join(args.out_dir,
                          'seed_{}_{}.json'.format(args.seed, args.rung))
    if os.path.exists(out_fn):
        print('[eods] exists, skipping', out_fn)
        return
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(args.seed)
    os.environ.setdefault('MPLBACKEND', 'Agg')

    soln_types = [s for s in os.environ.get(
        'SCULPTOR_SOLN_TYPES', 'painter').split(',') if s]
    n_sites = args.dpsize.replace('actual-', '')
    pkl_fn = os.path.join(args.out_dir,
                          'seed_{}_metrics.pkl'.format(args.seed))
    kwargs = {'nsim': 1, 'use_performance_metrics_fn': pkl_fn,
              'soln_types': soln_types}
    # SCULPTOR_EODS_HOTSTART_DIR (Tom 2026-08-20: training state IS
    # resumable — the solver saves state-N.pkl in its run dir and
    # load_optimization_state hot-starts from the newest one; the
    # popp_to_users blocker was fixed 2026-08-19). Value = run-dir name
    # relative to RUN_DIR (the workspace runs/ dir). Passing it also
    # bypasses the compare_rets resume-skip, forcing sparse to continue
    # training from the saved state.
    _hs = os.environ.get('SCULPTOR_EODS_HOTSTART_DIR')
    if _hs:
        print('[eods] HOTSTART from run dir: {}'.format(_hs), flush=True)
        # Fail loud BEFORE evaluate_all_metrics: its strategy loop wraps
        # everything in a bare except that prints and continues, so a
        # broken hot-start silently degrades to a 6-second no-op that
        # exits 0 -- which makes the queue treat it as success and rmtree
        # the run dir, destroying the very checkpoints we're resuming
        # from (lost 11h of actual-32 training this way, 2026-08-21).
        from helpers.constants import RUN_DIR
        _hs_dir = os.path.join(RUN_DIR, _hs)
        _base = os.path.join(_hs_dir, 'state-0.pkl')
        try:
            if 'deployment' not in pickle.load(open(_base, 'rb')):
                raise KeyError("no 'deployment' key in {}".format(_base))
        except Exception as e:
            sys.exit('[eods] FATAL hotstart: {} unusable ({}: {}). '
                     'state-0.pkl holds the deployment and is required; '
                     'never prune it.'.format(_base, type(e).__name__, e))
        import re as _re
        _states = {}
        for _f in os.listdir(_hs_dir):
            _m = _re.match(r'state-(\d+)\.pkl$', _f)
            if _m and _f != 'state-0.pkl':
                _states[int(_m.group(1))] = _f
        if not _states:
            sys.exit('[eods] FATAL hotstart: {} has no state-N.pkl beyond '
                     'state-0 -- nothing to resume from.'.format(_hs_dir))
        # load_optimization_state() resumes from the HIGHEST-numbered
        # state, so that is the one that must actually be readable --
        # counting filenames is not enough. A disk that fills mid-
        # pickle.dump leaves it truncated, and UnpicklingError is not a
        # ValueError, so _solve_setup's cold-start handler does not catch
        # it: it lands in the same bare except -> rc=0 -> queue rmtree.
        _newest = _states[max(_states)]
        _newest_fn = os.path.join(_hs_dir, _newest)
        try:
            pickle.load(open(_newest_fn, 'rb'))
        except Exception as e:
            sys.exit('[eods] FATAL hotstart: newest checkpoint {} is '
                     'unreadable ({}: {}) -- it is the one the solver '
                     'resumes from. Move/delete it and retry; the run '
                     'then resumes from the next-newest.'.format(
                         _newest_fn, type(e).__name__, e))
        print('[eods] hotstart base OK, {} resumable checkpoint(s), '
              'newest={}'.format(len(_states), _newest), flush=True)
        kwargs['save_run_dir'] = _hs

    if mode == 'prefixes':
        # the queue's swept N value = prefix count for this cell
        prefix_num = int(os.environ['SCULPTOR_ABLATION_PROBE_N'])
        dpsize_str = 'actual-{}'.format(n_sites)
        from helpers.constants import CACHE_DIR
        base = pickle.load(open(os.path.join(
            CACHE_DIR,
            'popp_failure_latency_comparison_actual-{}.pkl'.format(
                n_sites)), 'rb'))
        kwargs.update({'n_prefixes': prefix_num,
                       'prefix_deployment': base['deployment'][3]})
        unit = 'prefixes={}'.format(prefix_num)
    else:
        dpsize_str = 'testing_feature-actual-{}'.format(n_sites)
        unit = 'sim={}'.format(args.seed)

    t0 = time.time()
    from evaluations.eval_all_solution_types import evaluate_all_metrics
    # activate AFTER all module imports: activating earlier reorders the
    # star-import circularity and eval_all_solution_types loses
    # get_random_deployment (NameError, found on the instrumented
    # pre-flight relaunch 2026-08-20)
    if os.environ.get('SCULPTOR_STARTUP_TIMELOG') == '1':
        import helpers.timelog as timelog
        timelog.activate()
    metrics = evaluate_all_metrics(dpsize_str, args.port, **kwargs)

    rec = {'seed': args.seed, 'rung': args.rung, 'dpsize': args.dpsize,
           'mode': mode, 'unit': unit,
           'n_iters': 1,   # progress convention: 1 unit per cell
           'soln_types': sorted(soln_types),
           'metrics_pkl': os.path.basename(pkl_fn),
           'stats_keys': sorted(k for k in (metrics or {})
                                if str(k).startswith('stats_')),
           'wall_s': round(time.time() - t0, 1),
           'lp_backend': os.environ.get('SCULPTOR_LP_BACKEND', 'gurobi')}
    with open(out_fn, 'w') as f:
        json.dump(rec, f)
    print('[eods] done {} {} in {:.0f}s -> {}'.format(
        args.dpsize, unit, rec['wall_s'], out_fn), flush=True)


if __name__ == '__main__':
    main()
