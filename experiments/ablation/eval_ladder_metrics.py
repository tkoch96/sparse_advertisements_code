"""Score the fork-ladder advertisements with the REPO's full evaluation
pipeline (eval_all_solution_types.evaluate_all_metrics) -- the same functions
that produce every metric in evaluate_over_deployment_sizes:

  stats_best_latencies, stats_latency_thresholds_{normal,fail_popp,fail_pop},
  stats_{popp,pop}_failures_latency_optimal_specific  (latency deltas for the
      users OF the failed element -- the "relevant users" metrics),
  stats_resilience_to_congestion (flash crowds),
  stats_diurnal.

NO metric logic is reimplemented. This driver only:
  1. loads the ladder advertisements from the fork-ladder JSONs
     (adv saved at train end; one JSON per (seed, rung)),
  2. rebuilds each seed's deployment exactly as the ladder/rescore do
     (SCULPTOR_DEPLOYMENT_SEED + get_random_deployment on the SAME host
     that trained -- deployment construction is data-cache dependent),
  3. pre-populates evaluate_all_metrics' checkpoint pickle so its training
     phase sees each rung as an already-solved strategy (the same
     'n_advs'-present convention the pipeline itself uses to resume), with
     the per-solution latencies computed by the same calls its phase 1 makes,
  4. registers the rung names in wrapper_eval.default_metrics templates
     (runtime augmentation; no repo edit),
  5. calls evaluate_all_metrics and dumps every 'stats_*' key.

Run on the machine that trained the ladder (deployment consistency!):
  python -m experiments.ablation.eval_ladder_metrics \
      --in-dir cache/ablation/fork_a10_v2 --dpsize actual-10 \
      --port 46600 --metrics-fn cache/ablation/ladder_a10_eval_metrics.pkl \
      --out cache/ablation/ladder_a10_stats.pkl

Fresh process only (rule: never score in a process where a solver ran).
Worker count via SCULPTOR_N_WORKERS (mind RAM if a sweep is
running on the same license).
"""
import argparse
import copy
import glob
import json
import os
import pickle
import re
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


RUNG_ORDER = ['painter', 'no_mc', 'no_memory', 'no_direction',
              'expl_none', 'expl_random', 'full']


def load_ladder_advs(in_dir):
    """{seed: {rung: adv ndarray}} from fork-ladder JSONs (adv = trained
    advertisement; present whether or not the JSON has been rescored)."""
    out = {}
    for fn in sorted(glob.glob(os.path.join(in_dir, 'seed_*_*.json'))):
        m = re.match(r'seed_(\d+)_(.+)\.json$', os.path.basename(fn))
        if not m:
            continue
        with open(fn) as f:
            r = json.load(f)
        if 'adv' not in r or not r['adv']:
            print('[ladder-eval] {}: no adv, skipping'.format(fn))
            continue
        out.setdefault(int(m.group(1)), {})[m.group(2)] = np.asarray(r['adv'], dtype=float)
    return out


def _canary_check(in_dir, merged, kept_seeds):
    """Cross-validate merged per-seed latencies against the independent
    rescore numbers stored in the fork JSONs (when present). Any large
    disagreement means in-process contamination slipped through -- refuse
    to produce stats from corrupted data."""
    n_checked, worst = 0, 0.0
    for ri, seed in enumerate(kept_seeds):
        for rung, lats in merged.get('latencies', {}).get(ri, {}).items():
            fn = os.path.join(in_dir, 'seed_{}_{}.json'.format(seed, rung))
            if not os.path.exists(fn) or lats is None or not len(lats):
                continue
            with open(fn) as f:
                r = json.load(f)
            if not r.get('rescored') or r.get('avg_lat') is None:
                continue
            vols = np.asarray(merged['ug_to_vol'][ri])
            mine = float(np.average(np.asarray(lats), weights=vols))
            ref = float(r['avg_lat'])
            # generous tolerance: LP degeneracy noise + regime differences
            # are ~ms scale; contamination errors are orders of magnitude
            tol = max(5.0, 0.2 * abs(ref))
            diff = abs(mine - ref)
            worst = max(worst, diff / (abs(ref) + 1e-9))
            n_checked += 1
            assert diff <= tol, (
                '[ladder-eval] CANARY FAILED seed {} {}: merged avg_lat {:.2f} vs '
                'rescore {:.2f} -- in-process contamination suspected'.format(
                    seed, rung, mine, ref))
    print('[ladder-eval] canary vs rescore: {} cells checked, worst rel diff {:.1%} OK'.format(
        n_checked, worst))


def _dump_stats(m, seeds, soln_types, out_fn):
    stats = {k: m[k] for k in m if k.startswith('stats_')}
    stats['_seeds'] = seeds
    stats['_soln_types'] = soln_types
    pickle.dump(stats, open(out_fn, 'wb'))
    print('[ladder-eval] stats written: {}'.format(out_fn))
    for k in sorted(stats):
        if k.startswith('_'):
            continue
        print('== {} =='.format(k))
        try:
            for sol in soln_types:
                print('  {:<16} {}'.format(sol, stats[k].get(sol)))
        except Exception as e:
            print('  (unprintable: {})'.format(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--dpsize', default='actual-10')
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--metrics-fn', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default=None, help='comma list; default all found')
    ap.add_argument('--single-seed-worker', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args()

    ladder = load_ladder_advs(args.in_dir)
    if args.seeds:
        keep = {int(s) for s in args.seeds.split(',')}
        ladder = {s: v for s, v in ladder.items() if s in keep}
    seeds = sorted(ladder)
    assert seeds, 'no ladder JSONs found in {}'.format(args.in_dir)
    rungs = [r for r in RUNG_ORDER if any(r in ladder[s] for s in seeds)]
    soln_types = rungs + ['one_per_peering']
    print('[ladder-eval] seeds={} solutions={}'.format(seeds, soln_types))

    # ---------------------------------------------------------------------
    # CONTAMINATION GUARD. Scoring multiple SEEDS through one process /
    # shared worker stack silently corrupts results (verified 2026-08-09:
    # the small seed-3 no_direction blowup adv, +21k ms in three
    # independent scorers, scored +0.99 in a multi-seed process; the same
    # driver run seed-alone reproduced +20,977). Isolation unit = one
    # fresh subprocess per seed (same as rescore_fork). The parent process
    # never scores anything: it merges the per-seed pickles mechanically
    # and lets evaluate_all_metrics' own stats section aggregate (every
    # phase sees complete data and skips; no LP is re-solved).
    # ---------------------------------------------------------------------
    merged_mode = False
    if not args.single_seed_worker and len(seeds) > 1:
        merged_mode = True
        import subprocess
        per_seed_fns = []
        for s in seeds:
            fn = args.metrics_fn + '.seed{}'.format(s)
            per_seed_fns.append((s, fn))
            if os.path.exists(fn):
                print('[ladder-eval] per-seed pickle exists, reusing: {}'.format(fn))
                continue
            cmd = [sys.executable, '-m', 'experiments.ablation.eval_ladder_metrics',
                   '--in-dir', args.in_dir, '--dpsize', args.dpsize,
                   '--port', str(args.port + 10 * s), '--metrics-fn', fn,
                   '--out', fn + '.stats_unused', '--seeds', str(s),
                   '--single-seed-worker']
            env = dict(os.environ)
            env.pop('RAY_TMPDIR', None)  # each child gets its own pid-based dir
            print('[ladder-eval] spawning isolated seed {} worker'.format(s), flush=True)
            rc = subprocess.call(cmd, env=env)
            if rc != 0:
                print('[ladder-eval] WARNING: seed {} worker rc={} (skipping)'.format(s, rc))
        # merge per-seed pickles: combined[k][ri] <- seed_i[k][0]
        merged = None
        kept_seeds = []
        for s, fn in per_seed_fns:
            if not os.path.exists(fn):
                continue
            with open(fn, 'rb') as f:
                one = pickle.load(f)
            ri = len(kept_seeds)
            kept_seeds.append(s)
            if merged is None:
                merged = {k: {} for k in one}
            for k in merged:
                if isinstance(one.get(k), dict) and 0 in one[k]:
                    merged[k][ri] = one[k][0]
        assert merged and kept_seeds, 'no per-seed results to merge'
        pickle.dump(merged, open(args.metrics_fn, 'wb'))
        print('[ladder-eval] merged {} seeds -> {}'.format(kept_seeds, args.metrics_fn))
        _canary_check(args.in_dir, merged, kept_seeds)
        seeds = kept_seeds  # ri order for the stats pass

    os.environ.setdefault('MPLBACKEND', 'Agg')

    os.environ.setdefault('MPLBACKEND', 'Agg')
    # Isolation (same as rescore_fork): NEVER attach to a running Ray cluster.
    # worker_comms tries address='auto' first, which on a busy sweep host
    # attaches to the sweep's Ray -- and dies with it when that rung ends.
    # Must be set before the first repo import below (_ensure_ray runs at
    # worker_comms import time).
    os.environ['RAY_ADDRESS'] = 'local'
    os.environ.setdefault('RAY_TMPDIR', '/tmp/ray_ladder_eval_{}'.format(os.getpid()))

    # IMPORTANT: import eval_all_solution_types BEFORE any module that imports
    # sparse_advertisements_v3. sparse_advertisements_v3:93 imports from
    # eval_all_solution_types while Sparse_Advertisement_Eval is defined much
    # later (line ~640); importing sparse first hands eval_all_solution_types a
    # partial module during its star-import and its namespace permanently
    # misses the class (NameError in every later eval phase). Production
    # drivers (evaluate_over_deployment_sizes) import this module first too.
    from evaluations.eval_all_solution_types import evaluate_all_metrics

    # ---- register rung names in the repo's default metric templates -------
    import evaluations.wrapper_eval as wrapper_eval
    for k, per_iter in wrapper_eval.default_metrics.items():
        for i, v in per_iter.items():
            if isinstance(v, dict):
                for sol in soln_types:
                    v.setdefault(sol, [])

    from helpers.constants import DEFAULT_EXPLORE
    from evaluations.wrapper_eval import capacity, gamma, lambduh
    from core.deployment_setup import get_random_deployment
    from core.sparse_advertisements_v3 import Sparse_Advertisement_Eval
    from core.worker_comms import Worker_Manager
    from helpers.helpers import deployment_to_prefixes

    # ---- build the pre-populated checkpoint pickle ------------------------
    # (skipped in merged mode: per-seed workers already computed everything;
    # the merged pickle is on disk and the call below only runs the stats
    # aggregation, since every phase sees complete data and skips itself)
    if merged_mode:
        m = evaluate_all_metrics(
            args.dpsize, args.port, nsim=len(seeds), soln_types=soln_types,
            use_performance_metrics_fn=args.metrics_fn)
        _dump_stats(m, seeds, soln_types, args.out)
        return

    metrics = copy.deepcopy(wrapper_eval.default_metrics)
    for k in metrics:
        for ri in range(len(seeds)):
            if ri not in metrics[k]:
                metrics[k][ri] = copy.deepcopy(metrics[k][0])

    # One Worker_Manager for this seed, exactly like evaluate_all_metrics'
    # own phase loop (update_deployment needs workers to populate
    # best_lats_by_ug via compute_one_per_peering_solution).
    wm = None
    try:
        for ri, seed in enumerate(seeds):
            print('[ladder-eval] preparing seed {} (sim index {})'.format(seed, ri), flush=True)
            os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(seed)
            deployment = get_random_deployment(args.dpsize)
            deployment['port'] = args.port
            n_prefixes = deployment_to_prefixes(deployment)
            sas = Sparse_Advertisement_Eval(
                deployment, verbose=False, lambduh=lambduh, with_capacity=capacity,
                explore=DEFAULT_EXPLORE,
                using_resilience_benefit=(os.environ.get('SCULPTOR_USE_RESILIENCE', '1') == '1'),
                gamma=gamma, n_prefixes=n_prefixes, generic_objective='avg_latency')
            if wm is None:
                wm = Worker_Manager(sas.get_init_kwa(), deployment)
                wm.start_workers()
            sas.set_worker_manager(wm)
            sas.update_deployment(deployment)

            advs = dict(ladder[seed])
            n_prefs_trained = next(iter(advs.values())).shape
            advs['one_per_peering'] = np.eye(sas.n_popps)
            for rung, adv in advs.items():
                if adv.shape[0] != sas.n_popps:
                    raise ValueError(
                        'seed {} {}: adv shape {} vs deployment n_popps {} -- '
                        'deployment mismatch, run on the training host'.format(
                            seed, rung, adv.shape, sas.n_popps))

            # same calls evaluate_all_metrics' phase 1 makes per solution
            metrics['deployment'][ri] = deployment
            metrics['settings'][ri] = sas.get_init_kwa()
            metrics['ug_to_vol'][ri] = sas.ug_vols
            metrics['compare_rets'][ri] = {
                'n_advs': 1,
                'adv_solns': {sol: [adv] for sol, adv in advs.items()},
            }
            for sol, adv in advs.items():
                lats = sas.solve_lp_with_failure_catch(adv)['lats_by_ug']
                metrics['adv'][ri][sol] = adv
                metrics['latencies'][ri][sol] = lats
            metrics['best_latencies'][ri] = copy.copy(sas.best_lats_by_ug)
            print('[ladder-eval] seed {}: trained shape {}, {} solutions injected'.format(
                seed, n_prefs_trained, len(advs)), flush=True)
            del sas
    finally:
        if wm is not None:
            try:
                wm.stop_workers()  # eval phases start their own fleet
            except Exception as e:
                print('[ladder-eval] warning: stop_workers raised {}'.format(e))

    os.makedirs(os.path.dirname(os.path.abspath(args.metrics_fn)), exist_ok=True)
    pickle.dump(metrics, open(args.metrics_fn, 'wb'))
    print('[ladder-eval] checkpoint pickle written: {}'.format(args.metrics_fn))

    # ---- hand off to the repo pipeline ------------------------------------
    m = evaluate_all_metrics(
        args.dpsize, args.port, nsim=len(seeds), soln_types=soln_types,
        use_performance_metrics_fn=args.metrics_fn)
    _dump_stats(m, seeds, soln_types, args.out)


if __name__ == '__main__':
    main()
