"""Dual-solve parity experiment (Tom 2026-08-17): for every objective
family, 5 small-deployment trainings in which EVERY LP the HiGHS backend
solves — driver and workers, one-shot and persistent — is re-solved by
gurobi on the exact same model (SCULPTOR_GPSHIM_DUAL=1) and compared:
|d objective| and ||d x||_2 per solve.

    python -m experiments.solver_fork.run_dual_parity \
        --out-dir cache/solver_fork/dual_parity --parallel 2
    python -m experiments.solver_fork.run_dual_parity --report \
        --out-dir cache/solver_fork/dual_parity
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

FAMILIES = ('lat', 'fracb', 'mlu', 'prio')


def family_env(fam):
    m = json.load(open(os.path.join(
        _REPO_ROOT, 'tools', 'grid_georand_manifest.json')))
    sp = [s for s in m
          if s['out_root'].endswith('{}/L5_full_sched'.format(fam))][0]
    env = dict(sp['env'])
    env['SCULPTOR_ABLATION_GAMMA'] = sp['gamma']
    env['SCULPTOR_ABLATION_PROBE_MODE'] = sp['probe_mode']
    env['SCULPTOR_ABLATION_PROBE_N'] = '10'
    return env


def one_cell(args):
    os.environ['SCULPTOR_LP_BACKEND'] = 'highs'
    from experiments.ablation.run_fork_ladder import run_one
    run_one(args.seed, 'full', args.port, args.max_iter,
            args.out_dir, dpsize='small')


def orchestrate(args):
    import queue as qm
    import threading
    py = sys.executable
    cells = [(f, s) for f in FAMILIES for s in range(1, 6)]
    q = qm.Queue()
    for c in cells:
        dual_dir = os.path.join(args.out_dir, 'dual', '{}_s{}'.format(*c))
        done = os.path.join(args.out_dir, c[0],
                            'seed_{}_full.json'.format(c[1]))
        if os.path.exists(done) and os.path.isdir(dual_dir):
            print('[dual] exists, skipping', c)
        else:
            q.put(c)
    lock = threading.Lock()
    last = [0.0]

    def slot(si):
        ws = os.path.join(args.ws_root, 'S{}'.format(si))
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
                fam, seed = q.get_nowait()
            except qm.Empty:
                return
            out_dir = os.path.join(args.out_dir, fam)
            os.makedirs(out_dir, exist_ok=True)
            env = dict(os.environ)
            env.update(family_env(fam))
            env.update({
                'PYTHONPATH': _REPO_ROOT,
                'MPLBACKEND': 'Agg', 'RAY_ADDRESS': 'local',
                'RAY_TMPDIR': '/tmp/ray_dp_S{}'.format(si),
                'SCULPTOR_LP_BACKEND': 'highs',
                'SCULPTOR_N_WORKERS': str(args.workers_per_run),
                'SCULPTOR_GPSHIM_DUAL': '1',
                'SCULPTOR_GPSHIM_DUAL_OUT': os.path.join(
                    args.out_dir, 'dual', '{}_s{}'.format(fam, seed)),
            })
            cmd = [py, '-u', '-m',
                   'experiments.solver_fork.run_dual_parity',
                   '--one-cell', '--seed', str(seed),
                   '--port', str(args.port0 + 20 * si),
                   '--max-iter', str(args.max_iter),
                   '--out-dir', out_dir]
            with lock:
                w = last[0] + args.launch_stagger - time.time()
                if w > 0:
                    time.sleep(w)
                last[0] = time.time()
            log = os.path.join(ws, 'logs', 'dual_{}_s{}.log'.format(fam, seed))
            print('[dual] launch {} seed {}'.format(fam, seed), flush=True)
            t0 = time.time()
            with open(log, 'w') as lf:
                rc = subprocess.call(cmd, cwd=ws, env=env, stdout=lf,
                                     stderr=subprocess.STDOUT,
                                     timeout=args.cell_timeout)
            print('[dual] finished {} seed {} in {:.0f}s rc={}'.format(
                fam, seed, time.time() - t0, rc), flush=True)

    ths = [__import__('threading').Thread(target=slot, args=(i,),
                                          daemon=True)
           for i in range(args.parallel)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    report(args)


def report(args):
    import glob
    import numpy as np
    print('\n===== dual-solve parity report (gurobi vs HiGHS, same LP) =====')
    grand = []
    for fam in FAMILIES:
        recs = []
        for fn in glob.glob(os.path.join(
                args.out_dir, 'dual', '{}_s*'.format(fam), '*.jsonl')):
            with open(fn) as f:
                recs += [json.loads(l) for l in f if l.strip()]
        solved = [r for r in recs if 'dobj' in r]
        if not solved:
            print('{:>6}: no records yet'.format(fam))
            continue
        dobj = np.array([r['dobj_rel'] for r in solved])
        l2 = np.array([r['l2'] for r in solved])
        l2r = np.array([r['l2_rel'] for r in solved])
        zero_l2 = float((l2 <= 1e-6).mean())
        print('{:>6}: {:6d} solves | dobj_rel max {:.2e} p99 {:.2e} | '
              '||dx|| max {:.3g} mean {:.3g} | ||dx||/||x|| max {:.2e} | '
              'exact-x {:.1%}'.format(
                  fam, len(solved), dobj.max(), np.percentile(dobj, 99),
                  l2.max(), l2.mean(), l2r.max(), zero_l2))
        grand += solved
    if grand:
        dobj = np.array([r['dobj_rel'] for r in grand])
        print('\nALL: {} solves, max relative objective delta {:.2e} '
              '(machine-precision agreement = solvers interchangeable at '
              'the objective level; L2>0 at equal objective = degenerate '
              'alternate optima, not error)'.format(len(grand), dobj.max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--one-cell', action='store_true')
    p.add_argument('--report', action='store_true')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--port', type=int, default=None)
    p.add_argument('--port0', type=int, default=34200)
    p.add_argument('--max-iter', type=int, default=40)
    p.add_argument('--out-dir', default='cache/solver_fork/dual_parity')
    p.add_argument('--ws-root', default=os.path.join(_REPO_ROOT, 'dp_ws'))
    p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--workers-per-run', type=int, default=4)
    p.add_argument('--launch-stagger', type=float, default=15.0)
    p.add_argument('--cell-timeout', type=float, default=5400)
    args = p.parse_args()
    args.out_dir = (args.out_dir if os.path.isabs(args.out_dir)
                    else os.path.join(_REPO_ROOT, args.out_dir))
    if args.one_cell:
        one_cell(args)
    elif args.report:
        report(args)
    else:
        orchestrate(args)


if __name__ == '__main__':
    main()
