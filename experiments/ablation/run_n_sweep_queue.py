"""Work-queue N-sweep driver: near-maximum core efficiency.

Replaces run_n_sweep.sh's per-seed lane affinity with a global cell queue:
every (N, seed, rung) cell is independent work, SLOTS workers pull cells
greedily, so there is no straggler tail (tail = one longest cell). The
per-seed affinity existed only to serialize the canonical-init FIRST
write; this driver REQUIRES all init_dep<seed>.npy pre-present (copied
from --init-src and verified) so no such race exists — the init-equality
assertion still runs inside every cell.

Sizing: each cell runs 1 driver + WORKERS_PER_RUN Gurobi workers, so
Gurobi sessions = SLOTS * WORKERS_PER_RUN. Default 28 slots x 1 worker
= 28 sessions (the proven-safe level) ~= 56 busy cores on a 64-vCPU head.

    python -m experiments.ablation.run_n_sweep_queue \
        --out-root cache/ablation/nsweep_v2 \
        --init-src cache/ablation/fork_small_20x200_v3 \
        --ws-root /home/ubuntu/nsweep_ws_q [--n-values 1,2,5,10,20]
        [--seeds 1-20] [--max-iter 200] [--slots 28] [--workers-per-run 1]

Audit gate (same rules as run_n_sweep.sh, incl. the code-version guard)
and the trusted rescore run as pipeline stages after the queue drains.
"""
import argparse
import glob
import json
import os
import queue
import shutil
import subprocess
import sys
import threading

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# heavy rungs first so the queue's tail is a cheap cell, not a 'full' run
RUNGS_HEAVY_FIRST = ['full', 'expl_random', 'expl_none', 'no_direction',
                     'no_memory', 'no_mc', 'painter']


def parse_seeds(spec):
    if '-' in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(',')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--init-src', required=True)
    ap.add_argument('--ws-root', required=True)
    ap.add_argument('--n-values', default='1,2,5,10,20')
    ap.add_argument('--rungs', default=','.join(RUNGS_HEAVY_FIRST))
    ap.add_argument('--seeds', default='1-20')
    ap.add_argument('--max-iter', type=int, default=200)
    ap.add_argument('--slots', type=int, default=28)
    ap.add_argument('--workers-per-run', type=int, default=1)
    ap.add_argument('--port0', type=int, default=56000)
    ap.add_argument('--dpsize', default='small')
    ap.add_argument('--probe-mode', default='gated')
    ap.add_argument('--gamma', default='0.1')
    ap.add_argument('--no-rescore', action='store_true')
    ap.add_argument('--py', default=sys.executable)
    args = ap.parse_args()

    n_values = [int(n) for n in args.n_values.split(',')]
    seeds = parse_seeds(args.seeds)

    # ---- init preseeding: MANDATORY (this is what makes cells independent)
    src_inits = {s: os.path.join(args.init_src, 'init_dep{}.npy'.format(s)) for s in seeds}
    missing = [s for s, p in src_inits.items() if not os.path.exists(p)]
    assert not missing, 'missing canonical inits for seeds {} in {}'.format(missing, args.init_src)
    for N in n_values:
        d = os.path.join(args.out_root, 'N{}'.format(N))
        os.makedirs(d, exist_ok=True)
        for s, p in src_inits.items():
            dst = os.path.join(d, os.path.basename(p))
            if not os.path.exists(dst):
                shutil.copy(p, dst)

    # ---- build the queue (skip completed cells)
    rungs = [r for r in RUNGS_HEAVY_FIRST if r in args.rungs.split(',')]
    cells = []
    for rung in rungs:                      # heavy-first across the whole queue
        for N in n_values:
            for s in seeds:
                out_fn = os.path.join(args.out_root, 'N{}'.format(N),
                                      'seed_{}_{}.json'.format(s, rung))
                if not os.path.exists(out_fn):
                    cells.append((N, s, rung))
    q = queue.Queue()
    for c in cells:
        q.put(c)
    print('[queue] {} cells to run, {} slots x {} workers/run'.format(
        len(cells), args.slots, args.workers_per_run), flush=True)

    failures = []
    flock = threading.Lock()

    def slot_worker(slot):
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
        env = dict(os.environ)
        env.update({
            'PYTHONPATH': _REPO_ROOT,
            'SCULPTOR_ABLATION_GAMMA': args.gamma,
            'SCULPTOR_N_WORKERS': str(args.workers_per_run),
            'MPLBACKEND': 'Agg',
            'RAY_ADDRESS': 'local',
            'RAY_TMPDIR': '/tmp/ray_q_S{}'.format(slot),
            'SCULPTOR_ABLATION_PROBE_MODE': args.probe_mode,
            'SCULPTOR_ABLATION_PROBE_TCONV': str(args.max_iter),
        })
        while True:
            try:
                N, s, rung = q.get_nowait()
            except queue.Empty:
                return
            env['SCULPTOR_ABLATION_PROBE_N'] = str(N)
            log = os.path.join(ws, 'logs', 'N{}_s{}_{}.log'.format(N, s, rung))
            cmd = [args.py, '-u', '-m', 'experiments.ablation.run_fork_ladder',
                   '--seed', str(s), '--rung', rung,
                   '--port', str(args.port0 + 20 * slot),
                   '--max-iter', str(args.max_iter), '--dpsize', args.dpsize,
                   '--out-dir', os.path.join(args.out_root, 'N{}'.format(N))]
            with open(log, 'w') as lf:
                rc = subprocess.call(cmd, cwd=ws, env=env, stdout=lf, stderr=subprocess.STDOUT)
            if rc != 0:
                with flock:
                    failures.append((N, s, rung, rc))
                    print('[queue] FAIL N={} seed={} rung={} rc={}'.format(N, s, rung, rc), flush=True)
            q.task_done()

    threads = [threading.Thread(target=slot_worker, args=(i,), daemon=True)
               for i in range(args.slots)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print('[queue] sweep done; failures: {}'.format(len(failures)), flush=True)

    # ---- audit gate (same rules as run_n_sweep.sh)
    bad = 0
    for fn in glob.glob(os.path.join(args.out_root, 'N*', 'seed_*_*.json')):
        r = json.load(open(fn))
        if r['rung'] == 'painter':
            continue
        # budget-exhausted exits (Tom's exit-training criterion) legally end
        # before max_iter -- accept them iff the budget really was spent
        early_ok = (r.get('exit_reason') == 'budget_exhausted'
                    and r.get('probes_spent', 0) >= 1)
        if r.get('solve_error') or (
                (r.get('n_iters') or 0) < args.max_iter + 1 and not early_ok):
            print('[audit] BAD:', fn, r.get('n_iters'), str(r.get('solve_error'))[:40])
            bad += 1
        if args.probe_mode != 'fixed' and r.get('probe_mode') != args.probe_mode:
            print('[audit] BAD (stale code, probe_mode={}):'.format(r.get('probe_mode')), fn)
            bad += 1
    print('[audit] {} bad runs'.format(bad), flush=True)
    if bad or failures:
        print('[queue] AUDIT FAILED')
        sys.exit(1)

    if not args.no_rescore:
        sem = threading.Semaphore(8)

        def rescore(N, s):
            with sem:
                env = dict(os.environ, RAY_ADDRESS='local', MPLBACKEND='Agg',
                           RAY_TMPDIR='/tmp/ray_qrs_{}_{}'.format(N, s))
                subprocess.call([args.py, '-m', 'experiments.ablation.rescore_fork',
                                 '--in-dir', os.path.join(args.out_root, 'N{}'.format(N)),
                                 '--dpsize', args.dpsize, '--seed', str(s)],
                                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rs = [threading.Thread(target=rescore, args=(N, s), daemon=True)
              for N in n_values for s in seeds]
        for t in rs:
            t.start()
        for t in rs:
            t.join()
        print('[queue] rescore done', flush=True)
    print('[queue] ALL DONE', flush=True)


if __name__ == '__main__':
    main()
