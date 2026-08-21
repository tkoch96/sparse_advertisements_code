"""Backend-equivalence driver (Tom 2026-08-17): N small runs, EXACT same
deployment + init per seed, SCULPTOR stack routed through the solver-fork
(gpshim) with backend gurobi vs highs. Acceptance = essential solution
equivalence at the OBJECTIVE level (degenerate LPs make solution-level
agreement impossible; single-trial same-seed comparisons are additionally
gradient-RNG noisy — judge the paired table, not any one seed).

Orchestrate (spawns per-cell subprocesses, queue-style workspaces):
    python -m experiments.solver_fork.run_equivalence \
        --backends highs,gurobi --seeds 1-10 --parallel 8 \
        --out-dir cache/solver_fork/equiv_v1

Report over finished cells:
    python -m experiments.solver_fork.run_equivalence --report \
        --out-dir cache/solver_fork/equiv_v1

Internal per-cell entrypoint: --one-cell --backend B --seed S ...
"""
import argparse
import json
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

def install_aliases():
    """No-op since the 2026-08-17 mainline merge: core modules import
    gpshim directly, so backend selection is purely the
    SCULPTOR_LP_BACKEND env var. Kept for callers' backward compat."""
    import core.gpshim as gpshim
    want = os.environ.get('SCULPTOR_LP_BACKEND', 'gurobi')
    assert gpshim.BACKEND == want, (gpshim.BACKEND, want)


def gurobi_license_ok(py):
    r = subprocess.run(
        [py, '-c', 'import gurobipy as gp\n'
         'e = gp.Env(params={"OutputFlag": 0}); m = gp.Model(env=e)\n'
         'x = m.addVar(); m.setObjective(x); m.addConstr(x >= 1)\n'
         'm.optimize(); print("OK")'],
        capture_output=True, text=True, timeout=120)
    return r.returncode == 0 and 'OK' in r.stdout


def one_cell(args):
    # backend env is already exported by the orchestrator; enforce+verify
    os.environ['SCULPTOR_LP_BACKEND'] = args.backend
    install_aliases()
    from experiments.ablation.run_fork_ladder import run_one
    import core.gpshim as gpshim
    print('[solver-fork] cell start backend={} seed={} (gpshim={})'.format(
        args.backend, args.seed, gpshim.BACKEND), flush=True)
    out_fn = run_one(args.seed, 'full', args.port, args.max_iter,
                     args.out_dir, dpsize=args.dpsize)
    # stamp the backend into the result JSON for report-time provenance
    with open(out_fn) as f:
        rec = json.load(f)
    rec['lp_backend'] = args.backend
    with open(out_fn, 'w') as f:
        json.dump(rec, f)
    print('[solver-fork] cell done backend={} seed={} -> {}'.format(
        args.backend, args.seed, out_fn), flush=True)


def orchestrate(args):
    import queue as queue_mod
    import threading

    py = sys.executable
    backends = [b.strip() for b in args.backends.split(',') if b.strip()]
    seeds = parse_seeds(args.seeds)
    for b in backends:
        assert b in ('gurobi', 'highs'), b

    if 'gurobi' in backends and not gurobi_license_ok(py):
        print('[equiv] GUROBI LICENSE DEAD -> skipping gurobi cells '
              '(re-run with --backends gurobi once renewed)', flush=True)
        backends = [b for b in backends if b != 'gurobi']
        if not backends:
            sys.exit(2)

    cells = [(b, s) for b in backends for s in seeds]
    # skip finished cells (same convention as the queue: output JSON exists)
    todo = []
    for b, s in cells:
        out_fn = os.path.join(args.out_dir, b, 'seed_{}_full.json'.format(s))
        if os.path.exists(out_fn):
            print('[equiv] exists, skipping {} seed {}'.format(b, s))
        else:
            todo.append((b, s))
    print('[equiv] {} cells to run ({} skipped as done)'.format(
        len(todo), len(cells) - len(todo)), flush=True)
    if not todo:
        report(args)
        return

    q = queue_mod.Queue()
    for c in todo:
        q.put(c)
    launch_lock = threading.Lock()
    last_launch = [0.0]
    failures = []

    def slot_worker(slot):
        # queue-style per-slot workspace: cwd-relative runs/ isolated per
        # slot so semantic run-dir renames + retention GC can't collide
        # across concurrent cells (the 105-run massacre lesson)
        ws = os.path.join(args.ws_root, 'S{}'.format(slot))
        for sub in ('runs', 'logs', 'figures/paper'):
            os.makedirs(os.path.join(ws, sub), exist_ok=True)
        for link in ('cache', 'data'):
            lp = os.path.join(ws, link)
            if not os.path.islink(lp):
                try:
                    os.symlink(os.path.join(_REPO_ROOT, link), lp)
                except FileExistsError:
                    pass
        while True:
            try:
                backend, seed = q.get_nowait()
            except queue_mod.Empty:
                return
            out_dir = os.path.join(args.out_dir, backend)
            os.makedirs(out_dir, exist_ok=True)
            env = dict(os.environ)
            env.update({
                'PYTHONPATH': _REPO_ROOT,
                'SCULPTOR_LP_BACKEND': backend,
                'SCULPTOR_N_WORKERS': str(args.workers_per_run),
                'MPLBACKEND': 'Agg',
                'RAY_ADDRESS': 'local',
                'RAY_TMPDIR': '/tmp/ray_eq_S{}'.format(slot),
            })
            cmd = [py, '-u', '-m', 'experiments.solver_fork.run_equivalence',
                   '--one-cell', '--backend', backend, '--seed', str(seed),
                   '--port', str(args.port0 + 20 * slot),
                   '--max-iter', str(args.max_iter),
                   '--dpsize', args.dpsize, '--out-dir', out_dir]
            # serialize launches: simultaneous deployment builds are the
            # proven memory peak (queue lesson)
            with launch_lock:
                wait = last_launch[0] + args.launch_stagger - time.time()
                if wait > 0:
                    time.sleep(wait)
                last_launch[0] = time.time()
            log = os.path.join(ws, 'logs',
                               'equiv_{}_s{}.log'.format(backend, seed))
            print('[equiv] launch {} seed {} (slot {}, log {})'.format(
                backend, seed, slot, log), flush=True)
            t0 = time.time()
            with open(log, 'w') as lf:
                try:
                    rc = subprocess.call(cmd, cwd=ws, env=env, stdout=lf,
                                         stderr=subprocess.STDOUT,
                                         timeout=args.cell_timeout)
                except subprocess.TimeoutExpired:
                    rc = -99
            status = 'ok' if rc == 0 else 'rc={}'.format(rc)
            print('[equiv] finished {} seed {} in {:.0f}s ({})'.format(
                backend, seed, time.time() - t0, status), flush=True)
            if rc != 0:
                failures.append((backend, seed, rc, log))

    threads = [threading.Thread(target=slot_worker, args=(i,), daemon=True)
               for i in range(args.parallel)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        print('[equiv] FAILURES:', failures, flush=True)
    report(args)


def parse_seeds(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def report(args):
    import numpy as np
    print('\n===== solver_fork equivalence report =====')
    dirs = {b: os.path.join(args.out_dir, b) for b in ('gurobi', 'highs')}
    recs = {b: {} for b in dirs}
    for b, d in dirs.items():
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.startswith('seed_') and fn.endswith('_full.json'):
                seed = int(fn.split('_')[1])
                with open(os.path.join(d, fn)) as f:
                    recs[b][seed] = json.load(f)
    seeds = sorted(set(recs['gurobi']) | set(recs['highs']))
    if not seeds:
        print('no cells found under', args.out_dir)
        return

    # init/deployment identity: per-seed init files must be byte-identical
    for s in seeds:
        fa = os.path.join(dirs['gurobi'], 'init_dep{}.npy'.format(s))
        fb = os.path.join(dirs['highs'], 'init_dep{}.npy'.format(s))
        if os.path.exists(fa) and os.path.exists(fb):
            same = open(fa, 'rb').read() == open(fb, 'rb').read()
            if not same:
                print('  !! INIT MISMATCH seed {} — arms are NOT comparable'
                      .format(s))

    hdr = ('seed', 'g.obj', 'h.obj', 'g.lat', 'h.lat', 'g-h.lat',
           'g.iters', 'h.iters', 'g.wall', 'h.wall', 'adv.ham')
    print(('{:>5} {:>9} {:>9} {:>8} {:>8} {:>8} {:>7} {:>7} {:>7} {:>7}'
           ' {:>7}').format(*hdr))
    dlats, dobjs = [], []
    for s in seeds:
        g = recs['gurobi'].get(s)
        h = recs['highs'].get(s)

        def fmt(r, k, w='{:.2f}'):
            if r is None or r.get(k) is None:
                return '-'
            return w.format(r[k])
        ham = '-'
        if g and h and g.get('adv') and h.get('adv'):
            ga = np.asarray(g['adv'])
            ha = np.asarray(h['adv'])
            ham = str(int(np.sum(ga != ha))) if ga.shape == ha.shape else 'shape!'
        if g and h and g.get('avg_lat') is not None and h.get('avg_lat') is not None:
            dlats.append(g['avg_lat'] - h['avg_lat'])
        if (g and h and g.get('repo_objective') is not None
                and h.get('repo_objective') is not None):
            dobjs.append(g['repo_objective'] - h['repo_objective'])
        print(('{:>5} {:>9} {:>9} {:>8} {:>8} {:>8} {:>7} {:>7} {:>7} {:>7}'
               ' {:>7}').format(
            s, fmt(g, 'repo_objective'), fmt(h, 'repo_objective'),
            fmt(g, 'avg_lat'), fmt(h, 'avg_lat'),
            '{:.2f}'.format(g['avg_lat'] - h['avg_lat'])
            if g and h and g.get('avg_lat') is not None
            and h.get('avg_lat') is not None else '-',
            fmt(g, 'n_iters', '{:d}') if g else '-',
            fmt(h, 'n_iters', '{:d}') if h else '-',
            fmt(g, 'wall_s', '{:.0f}'), fmt(h, 'wall_s', '{:.0f}'), ham))
    if dlats:
        dlats = np.asarray(dlats)
        print('\npaired avg_lat delta (gurobi - highs, ms): '
              'mean {:+.3f}  median {:+.3f}  max|d| {:.3f}  n={}'.format(
                  dlats.mean(), float(np.median(dlats)),
                  float(np.abs(dlats).max()), len(dlats)))
    if dobjs:
        dobjs = np.asarray(dobjs)
        print('paired repo_objective delta (gurobi - highs): '
              'mean {:+.3f}  median {:+.3f}  max|d| {:.3f}  n={}'.format(
                  dobjs.mean(), float(np.median(dobjs)),
                  float(np.abs(dobjs).max()), len(dobjs)))
    print('(reminder: same-seed single-trial deltas are gradient-RNG '
          'noise-dominated; judge the distribution)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--one-cell', action='store_true')
    p.add_argument('--report', action='store_true')
    p.add_argument('--backend', default=None)
    p.add_argument('--backends', default='highs,gurobi')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--seeds', default='1-10')
    p.add_argument('--port', type=int, default=None)
    p.add_argument('--port0', type=int, default=32200)
    p.add_argument('--max-iter', type=int, default=200)
    p.add_argument('--dpsize', default='small')
    p.add_argument('--out-dir', default='cache/solver_fork/equiv_v1')
    p.add_argument('--ws-root', default=os.path.join(_REPO_ROOT, 'eq_ws'))
    p.add_argument('--parallel', type=int, default=4)
    p.add_argument('--workers-per-run', type=int, default=4)
    p.add_argument('--launch-stagger', type=float, default=20.0)
    p.add_argument('--cell-timeout', type=float, default=float(
        os.environ.get('SCULPTOR_CELL_TIMEOUT', '7200')))
    args = p.parse_args()
    args.out_dir = os.path.join(_REPO_ROOT, args.out_dir) \
        if not os.path.isabs(args.out_dir) else args.out_dir
    if args.one_cell:
        assert args.backend in ('gurobi', 'highs') and args.seed is not None
        one_cell(args)
    elif args.report:
        report(args)
    else:
        orchestrate(args)


if __name__ == '__main__':
    main()
