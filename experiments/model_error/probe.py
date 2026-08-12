"""Iteration-0 model-error probe: how far off is SCULPTOR's objective
estimate before it has learned anything?

For each seed: build a deployment (all generator knobs via env), stand up
the real solver stack, run _solve_setup() only (init advertisement +
baseline measurement = the exact iteration-0 belief state of every run),
then compare
    believed = solver.modeled_objective(init_adv)      (model belief)
    true     = solver.measured_objective(init_adv)     (ground-truth sim)
plus the belief's own spread (std of the latency-benefit distribution the
MC machinery returns). Errors are aggregated across seeds per knob
setting; the point is to find knob settings where |believed - true| and
the believed std become non-trivial WITHOUT changing problem size.

Knobs (all env, all default to stock 'small'):
    SCULPTOR_VOL_SPREAD      log-uniform UG volumes, s in {unset,2,4,6}
    SCULPTOR_LAT_SPREAD      within-tier latency noise multiplier (1=stock)
    SCULPTOR_ROUTE_VIOLATION priority-swap probability (.05=stock)
    SCULPTOR_SCALE_FACTOR    capacity headroom over anycast (1.1=stock)

Per-seed subprocess isolation (rule 3: never trust shared-stack numbers);
RAY_ADDRESS=local + unique RAY_TMPDIR per child (gotcha: stray attach).

Usage:
    python -m experiments.model_error.probe --seeds 1-10 \
        --tag baseline [--dpsize small] [--port0 61000]
    SCULPTOR_LAT_SPREAD=4 python -m experiments.model_error.probe \
        --seeds 1-10 --tag lat4
Child mode (internal): --child-seed N --port P
Results: one JSON per tag under cache/model_error/.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time

KNOB_ENVS = ('SCULPTOR_VOL_SPREAD', 'SCULPTOR_LAT_SPREAD',
             'SCULPTOR_ROUTE_VIOLATION', 'SCULPTOR_SCALE_FACTOR',
             'SCULPTOR_LAT_MODEL', 'SCULPTOR_PREF_MODEL')
MARKER = 'MODEL_ERROR_RESULT '


def run_child(seed, port, dpsize, iters=0):
    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(seed)
    os.environ['SCULPTOR_DISABLE_PARALLEL_STRATEGIES'] = '1'
    os.environ.setdefault('MPLBACKEND', 'Agg')

    if iters > 0:
        # Trajectory mode: run the REAL full-rung solver for `iters`
        # gradient steps with ZERO measurements after the baseline (the
        # fork's gated probe mode with a static threshold above U's max
        # of 1.0 -- its per-iteration assertions prove no probe fires),
        # then evaluate model error AT the trajectory endpoint. This is
        # the drift question: how wrong do beliefs get as the optimizer
        # walks away from its last measurement?
        os.environ['SCULPTOR_ABLATION_PROBE_MODE'] = 'gated'
        os.environ['SCULPTOR_ABLATION_PROBE_AUTO_C'] = '0'
        os.environ['SCULPTOR_ABLATION_PROBE_C'] = '2.0'
        os.environ['SCULPTOR_ABLATION_PROBE_N'] = '1'
        os.environ['SCULPTOR_ABLATION_PROBE_TCONV'] = str(iters)
        os.environ['SCULPTOR_MAX_ITER'] = str(iters)
        os.environ['SCULPTOR_MIN_ITER'] = str(iters)

    import numpy as np
    from constants import DEFAULT_EXPLORE
    from wrapper_eval import capacity, gamma as EVAL_GAMMA
    from deployment_setup import get_random_deployment
    from sparse_advertisements_v3 import (
        Sparse_Advertisement_Eval, Sparse_Advertisement_Solver)
    from worker_comms import Worker_Manager
    from helpers import deployment_to_prefixes

    use_res = os.environ.get('SCULPTOR_USE_RESILIENCE', '1') == '1'
    # default matches the ladder/N-sweep studies (gamma=0.1), not
    # wrapper_eval's gamma=4
    gamma_val = (float(os.environ.get('SCULPTOR_ABLATION_GAMMA', '0.1'))
                 if use_res else 0)

    deployment = get_random_deployment(dpsize)
    deployment['port'] = port
    n_prefixes = deployment_to_prefixes(deployment)
    sas = Sparse_Advertisement_Eval(
        deployment, verbose=True, lambduh=0.00001, with_capacity=capacity,
        explore=DEFAULT_EXPLORE, using_resilience_benefit=use_res,
        gamma=gamma_val, n_prefixes=n_prefixes, generic_objective='avg_latency',
    )
    wm = Worker_Manager(sas.get_init_kwa(), deployment)
    wm.start_workers()
    try:
        sas.set_worker_manager(wm)
        sas.update_deployment(deployment)
        if iters > 0:
            from experiments.ablation.sculptor_fork import (
                Ablation_Sparse_Advertisement_Solver)
            solver = Ablation_Sparse_Advertisement_Solver(
                sas.output_deployment(), **sas.get_init_kwa())
        else:
            solver = Sparse_Advertisement_Solver(
                sas.output_deployment(), **sas.get_init_kwa())
        solver.set_worker_manager(sas.get_worker_manager())
        solver.compute_one_per_peering_solution()
        if iters > 0:
            # skip finalize: reset_ugs() would desync the belief state
            # from the deployment before our evaluation
            solver._solve_finalize = lambda: None
            solver.solve()
            assert getattr(solver, 'abl_probes_spent', 0) == 0, \
                'gate fired despite static c=2.0'
        else:
            assert solver._solve_setup(), 'unexpected hot-start in probe'

        a0 = copy.deepcopy(solver.optimization_advertisement)

        # Probe set: the measured init adv (belief should be ~exact: it was
        # just measured) plus UNMEASURED perturbations at increasing flip
        # distance -- the belief about those is what gradient steps rely
        # on, so that is where model error actually matters.
        rng = np.random.RandomState(1000 + seed)
        n_popp, n_pref = a0.shape
        probe_advs = [('init', 0, a0)]
        for k in (1, 8, 32):
            for rep in range(3):
                ap = copy.deepcopy(a0)
                idx = rng.choice(n_popp * n_pref, size=k, replace=False)
                for ii in idx:
                    i, j = ii // n_pref, ii % n_pref
                    ap[i, j] = 0.55 if ap[i, j] < 0.5 else 0.01
                probe_advs.append(('flip{}'.format(k), rep, ap))

        # ALL beliefs first, then all ground truths: ground-truth calls can
        # update stored ingress state and must not feed the belief.
        beliefs = []
        for _, _, a in probe_advs:
            lb, (benefits, probs) = solver.latency_benefit_fn(
                copy.deepcopy(a), retnow=True)
            pr = np.asarray(probs, dtype=float) + 1e-12
            pr = pr / pr.sum()
            b = np.asarray(benefits, dtype=float)
            ex = float((b * pr).sum())
            std = float(np.sqrt(max(0.0, (b * b * pr).sum() - ex ** 2)))
            rb = (float(solver.resilience_benefit_fn(
                copy.deepcopy(a), retnow=True)) if use_res else 0.0)
            beliefs.append((float(lb), std, rb))

        probes = []
        for (kind, rep, a), (lb_b, lb_std, rb_b) in zip(probe_advs, beliefs):
            lb_t = float(solver.get_ground_truth_latency_benefit(
                copy.deepcopy(a)))
            rb_t = (float(solver.get_ground_truth_resilience_benefit(
                copy.deepcopy(a))) if use_res else 0.0)
            probes.append({
                'kind': kind, 'rep': rep,
                'lb_believed': lb_b, 'lb_true': lb_t,
                'err_lat': lb_b - lb_t,
                'rb_believed': rb_b, 'rb_true': rb_t,
                'err_res_weighted': float(gamma_val) * (rb_b - rb_t),
                'lb_std_believed': lb_std,
            })

        from helpers import threshold_a
        drift = None
        if iters > 0 and solver.metrics.get('advertisements'):
            first = threshold_a(np.asarray(solver.metrics['advertisements'][0]))
            drift = int(np.sum(first != threshold_a(np.asarray(a0))))
        out = {
            'seed': seed, 'dpsize': dpsize, 'gamma': float(gamma_val),
            'iters': iters,
            'n_advs_measured': int(getattr(solver, 'path_measures', -1)),
            'drift_bits_since_step1': drift,
            'probes': probes,
            'knobs': {k: os.environ.get(k) for k in KNOB_ENVS},
        }
        print(MARKER + json.dumps(out), flush=True)
    finally:
        try:
            wm.stop_workers()
        except Exception as e:
            print('warning: stop_workers raised {}'.format(e))


def parse_seeds(spec):
    if '-' in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(',')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='1-10')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--dpsize', default='small')
    ap.add_argument('--port0', type=int, default=61000)
    ap.add_argument('--iters', type=int, default=0,
                    help='trajectory mode: run the full-rung solver this '
                         'many measurement-free iterations, then probe '
                         'model error at the endpoint (0 = iteration-0 '
                         'probe at the measured init)')
    ap.add_argument('--child-seed', type=int, default=None)
    ap.add_argument('--port', type=int, default=None)
    args = ap.parse_args()

    if args.child_seed is not None:
        run_child(args.child_seed, args.port, args.dpsize, iters=args.iters)
        return

    assert args.tag, '--tag required (names the results file)'
    seeds = parse_seeds(args.seeds)
    results, failed = [], []
    for i, s in enumerate(seeds):
        env = dict(os.environ)
        env.update({
            'RAY_ADDRESS': 'local',
            'RAY_TMPDIR': '/tmp/ray_moderr_{}_{}'.format(os.getpid(), s),
        })
        cmd = [sys.executable, '-m', 'experiments.model_error.probe',
               '--child-seed', str(s), '--port', str(args.port0 + 40 * i),
               '--dpsize', args.dpsize, '--iters', str(args.iters)]
        t0 = time.time()
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith(MARKER)]
        if p.returncode != 0 or not line:
            failed.append(s)
            print('[probe] seed {} FAILED rc={} (tail: {})'.format(
                s, p.returncode, (p.stdout + p.stderr)[-400:]), flush=True)
            continue
        r = json.loads(line[-1][len(MARKER):])
        results.append(r)
        by_kind = {}
        for p in r['probes']:
            by_kind.setdefault(p['kind'], []).append(p)
        parts = []
        for kind in ('init', 'flip1', 'flip8', 'flip32'):
            ps = by_kind.get(kind, [])
            if ps:
                mae = sum(abs(p['err_lat']) for p in ps) / len(ps)
                parts.append('{} lat_mae={:.4f}'.format(kind, mae))
        print('[probe] seed {}: {} ({:.0f}s)'.format(
            s, '  '.join(parts), time.time() - t0), flush=True)

    if results:
        import statistics as st
        by_kind = {}
        for r in results:
            for p in r['probes']:
                by_kind.setdefault(p['kind'], []).append(p)
        kinds_summary = {}
        for kind, ps in by_kind.items():
            lat = [p['err_lat'] for p in ps]
            res = [p['err_res_weighted'] for p in ps]
            stds = [p['lb_std_believed'] for p in ps]
            kinds_summary[kind] = {
                'n': len(ps),
                'err_lat_mean': st.mean(lat),
                'err_lat_mae': st.mean(abs(x) for x in lat),
                'err_lat_std': st.stdev(lat) if len(lat) > 1 else 0.0,
                'err_res_weighted_mean': st.mean(res),
                'err_res_weighted_mae': st.mean(abs(x) for x in res),
                'lb_std_believed_mean': st.mean(stds),
            }
        summary = {
            'tag': args.tag, 'dpsize': args.dpsize, 'n_seeds': len(results),
            'failed_seeds': failed,
            'knobs': {k: os.environ.get(k) for k in KNOB_ENVS},
            'by_kind': kinds_summary,
            'results': results,
        }
        out_dir = os.path.join('cache', 'model_error')
        os.makedirs(out_dir, exist_ok=True)
        fn = os.path.join(out_dir, '{}.json'.format(args.tag))
        with open(fn, 'w') as f:
            json.dump(summary, f, indent=2)
        print('[probe] {} (n={} seeds) -> {}'.format(
            args.tag, len(results), fn), flush=True)
        for kind in ('init', 'flip1', 'flip8', 'flip32'):
            ks = kinds_summary.get(kind)
            if ks:
                print('  {:>6}: lat MAE={:.4f} (bias {:+.4f}, sd {:.4f}) '
                      'res-wtd MAE={:.4f} believed-std={:.5f}'.format(
                          kind, ks['err_lat_mae'], ks['err_lat_mean'],
                          ks['err_lat_std'], ks['err_res_weighted_mae'],
                          ks['lb_std_believed_mean']), flush=True)


if __name__ == '__main__':
    main()
