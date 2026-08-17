"""Work-queue N-sweep driver: near-maximum core efficiency.

Replaces run_n_sweep.sh's per-seed lane affinity with a global cell queue:
every (N, seed, rung) cell is independent work, SLOTS workers pull cells
greedily, so there is no straggler tail (tail = one longest cell). The
per-seed affinity existed only to serialize the canonical-init FIRST
write; this driver REQUIRES all init_dep<seed>.npy pre-present (copied
from --init-src and verified) so no such race exists — the init-equality
assertion still runs inside every cell.

Sizing: each cell runs 1 driver + WORKERS_PER_RUN Gurobi workers, so
Gurobi sessions = SLOTS * WORKERS_PER_RUN -- NOT a sizing constraint
(see WLS policy note in experiments/ablation/README.md): size slots to RAM
(90% target) and cores. Default 28 slots x 1 worker
~= 56 busy cores on a 64-vCPU head; scale slots up with the box.

    python -m experiments.ablation.run_n_sweep_queue \
        --out-root cache/ablation/nsweep_v2 \
        --init-src cache/ablation/fork_small_20x200_v3 \
        --ws-root /home/ubuntu/nsweep_ws_q [--n-values 1,2,5,10,20]
        [--seeds 1-20] [--max-iter 200] [--slots 28] [--workers-per-run 1]

Audit gate (same rules as run_n_sweep.sh, incl. the code-version guard)
and the trusted rescore run as pipeline stages after the queue drains.

--manifest MODE (2026-08-15, Tom: "a balancing solution that doesn't
have these problems"): the multi-queue chains partitioned slots
STATICALLY per (objective, policy) — when one queue drained, its slots
died idle while another still had an hours-long tail (the hb3v3
rebalance incident). A manifest runs EVERY cell group in ONE process
with ONE global slot pool: no partitions, no idle slots, no
reverse-order supplemental hacks. The manifest is a JSON list of
specs; each spec carries what used to be a whole queue invocation:

    [{"label": "prio_smart",                # artifact/figure prefix
      "out_root": "cache/ablation/hardB3v2/prio/smart",
      "init_src": "cache/ablation/nsweep_v2_inits_georand",  # optional,
                                            # falls back to --init-src
      "probe_mode": "smart", "rungs": "no_memory,no_direction,full",
      "seeds": "1-5", "n_values": "1,2,5,10,20,50",
      "gamma": "0", "max_iter": 200,        # both optional (CLI default)
      "env": {"SCULPTOR_XOBJS": "1",        # per-spec env overrides
              "SCULPTOR_ABLATION_OBJECTIVE": "joint_latency_bulk_download"},
      "artifacts_figs": "cache/ablation/hardB3v2_artifacts/figs"},  # optional:
                                            # inline per-cell convergence-
                                            # figure harvest (+ run-dir rm)
     ...]

Cell order is deployment-major ACROSS specs (Tom's standing rule: one
deployment x all N x ALL objectives first). Launches are globally
staggered (--launch-stagger, default 30s) so simultaneous deployment
builds can't thrash the box — no hand-tuned per-queue sleep choreography.
Audit + rescore run per spec. The classic single-spec CLI is unchanged
(it builds a one-spec manifest internally).
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
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# heavy rungs first so the queue's tail is a cheap cell, not a 'full' run
RUNGS_HEAVY_FIRST = ['full', 'expl_random', 'expl_none', 'no_direction',
                     'no_memory', 'no_mc', 'painter']


class MemGovernor:
    """RAM-targeted slot admission (Tom 2026-08-15: 'deploy cores to
    achieve ram utilization of 90%' -- RAM, not cores, is what crashes
    the box). Reuses the repo's /proc/meminfo MemAvailable pattern
    (sparse_advertisements_v3._log_mem / path_distribution_computer.
    get_node_mem_avail_mb). A worker may START a cell only when the
    PROJECTED utilization after adding one estimated cell stays under
    mem_target and a spike reserve remains (deployment builds peak well
    above steady RSS -- the proven thrash mechanism). Per-cell RSS is
    EWMA-estimated live from (used - baseline)/active. Config file
    (SCULPTOR_QUEUE_GOVERNOR, default ~/queue_governor.json) is re-read
    on every decision so max_active / mem_target are LIVE-tunable
    mid-run; max_active is bounded by RAM/cores, NOT Gurobi sessions
    (WLS policy note in experiments/ablation/README.md). On hosts
    without /proc/meminfo (the Mac) the governor is a no-op beyond the
    max_active cap."""

    def __init__(self, max_active, mem_target=0.90, est_cell_gb=6.0,
                 reserve_gb=8.0, sample_s=15.0):
        self.cfg_path = os.environ.get(
            'SCULPTOR_QUEUE_GOVERNOR',
            os.path.expanduser('~/queue_governor.json'))
        self.max_active = max_active
        self.mem_target = mem_target
        self.est_cell_gb = est_cell_gb
        self.reserve_gb = reserve_gb
        self.sample_s = sample_s
        self.active = 0
        self.lock = threading.Lock()
        self._last_log = 0.0
        m = self._meminfo()
        self.enabled = m is not None
        self.baseline_used_gb = (m[0] - m[1]) if m else 0.0
        print('[governor] enabled={} max_active={} target={:.0%} '
              'baseline_used={:.1f}G cfg={}'.format(
                  self.enabled, self.max_active, self.mem_target,
                  self.baseline_used_gb, self.cfg_path), flush=True)

    @staticmethod
    def _meminfo():
        """(total_gb, available_gb) or None (no /proc)."""
        path = os.environ.get('SCULPTOR_MEMINFO_PATH', '/proc/meminfo')
        try:
            d = {}
            with open(path) as f:
                for line in f:
                    parts = line.split()
                    if parts and parts[0].rstrip(':') in ('MemTotal', 'MemAvailable'):
                        d[parts[0].rstrip(':')] = int(parts[1]) / 1048576.0
            if 'MemTotal' in d and 'MemAvailable' in d:
                return d['MemTotal'], d['MemAvailable']
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        return None

    def _refresh_cfg(self):
        try:
            with open(self.cfg_path) as f:
                cfg = json.load(f)
            self.max_active = int(cfg.get('max_active', self.max_active))
            self.mem_target = float(cfg.get('mem_target', self.mem_target))
            self.reserve_gb = float(cfg.get('reserve_gb', self.reserve_gb))
            if cfg.get('est_cell_gb'):
                self.est_cell_gb = float(cfg['est_cell_gb'])
        except (FileNotFoundError, ValueError, KeyError):
            pass

    def try_acquire(self):
        with self.lock:
            self._refresh_cfg()
            if self.active >= self.max_active:
                return False
            if not self.enabled:
                self.active += 1
                return True
            m = self._meminfo()
            if m is None:
                self.active += 1
                return True
            total, avail = m
            used = total - avail
            # live per-cell estimate once enough cells are running
            if self.active >= 3:
                per = (used - self.baseline_used_gb) / self.active
                if per > 0.5:
                    self.est_cell_gb = 0.7 * self.est_cell_gb + 0.3 * per
            projected = (used + self.est_cell_gb) / total
            ok = (projected <= self.mem_target
                  and avail >= self.reserve_gb + self.est_cell_gb)
            now = time.time()
            if now - self._last_log > 120:
                self._last_log = now
                print('[governor] active={} max={} used={:.0f}G/{:.0f}G '
                      '({:.0%}) est_cell={:.1f}G projected={:.0%} '
                      'admit={}'.format(
                          self.active, self.max_active, used, total,
                          used / total, self.est_cell_gb, projected, ok),
                      flush=True)
            if ok:
                self.active += 1
            return ok

    def release(self):
        with self.lock:
            self.active -= 1


def parse_seeds(spec):
    if '-' in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(',')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', default=None)   # required unless --manifest
    ap.add_argument('--init-src', default=None)   # required unless --manifest
                                                  # (with --manifest: the
                                                  # per-spec init_src default)
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
    ap.add_argument('--manifest', default=None,
                    help='JSON list of cell-group specs; one global slot '
                         'pool serves them all (see module docstring)')
    ap.add_argument('--launch-stagger', type=float, default=30.0,
                    help='global min seconds between cell launches '
                         '(deployment-build thrash guard)')
    args = ap.parse_args()

    # ---- resolve specs: manifest, or a single spec from the classic CLI
    if args.manifest:
        with open(args.manifest) as f:
            specs = json.load(f)
        assert isinstance(specs, list) and specs, 'manifest must be a non-empty JSON list'
    else:
        assert args.out_root and args.init_src, '--out-root/--init-src required without --manifest'
        specs = [{'label': os.path.basename(args.out_root.rstrip('/')) or 'spec0',
                  'out_root': args.out_root}]
    for i, sp in enumerate(specs):
        sp.setdefault('label', 'spec{}'.format(i))
        assert sp.get('out_root'), 'spec {} missing out_root'.format(sp['label'])
        sp.setdefault('init_src', args.init_src)
        assert sp['init_src'], 'spec {} missing init_src'.format(sp['label'])
        sp['max_iter'] = int(sp.get('max_iter', args.max_iter))
        sp['gamma'] = str(sp.get('gamma', args.gamma))
        sp['probe_mode'] = sp.get('probe_mode', args.probe_mode)
        sp['dpsize'] = sp.get('dpsize', args.dpsize)
        sp['seeds_list'] = parse_seeds(str(sp.get('seeds', args.seeds)))
        sp['n_list'] = [int(n) for n in str(sp.get('n_values', args.n_values)).split(',')]
        sp['rungs_list'] = [r for r in RUNGS_HEAVY_FIRST
                            if r in str(sp.get('rungs', args.rungs)).split(',')]
        assert sp['rungs_list'], 'spec {} selected no rungs'.format(sp['label'])
        sp['env'] = {k: str(v) for k, v in sp.get('env', {}).items()}

    # ---- init preseeding: MANDATORY (this is what makes cells independent)
    for sp in specs:
        src_inits = {s: os.path.join(sp['init_src'], 'init_dep{}.npy'.format(s))
                     for s in sp['seeds_list']}
        missing = [s for s, p in src_inits.items() if not os.path.exists(p)]
        assert not missing, 'missing canonical inits for seeds {} in {}'.format(
            missing, sp['init_src'])
        for N in sp['n_list']:
            d = os.path.join(sp['out_root'], 'N{}'.format(N))
            os.makedirs(d, exist_ok=True)
            for s, p in src_inits.items():
                dst = os.path.join(d, os.path.basename(p))
                if not os.path.exists(dst):
                    shutil.copy(p, dst)

    def build_missing_cells():
        """Re-scannable work list (2026-08-16, Tom: 'there should be a
        central queue of jobs... CPUs report back when finished'). Called
        once per PASS: a cell is work iff its JSON is absent AND no FRESH
        .inprog marker exists (another slot/queue is computing it; stale
        markers > SCULPTOR_CELL_TIMEOUT are ignored). This makes purges,
        failures, and killed cells re-enter the queue on the next pass
        instead of waiting for a whole follow-up sweep."""
        timeout_s = float(os.environ.get('SCULPTOR_CELL_TIMEOUT', '7200'))
        found = []
        for s_ in all_seeds:
            for sp_ in specs:
                if s_ not in sp_['seeds_list']:
                    continue
                for rung_ in sp_['rungs_list']:
                    for N_ in sp_['n_list']:
                        out_fn_ = os.path.join(
                            sp_['out_root'], 'N{}'.format(N_),
                            'seed_{}_{}.json'.format(s_, rung_))
                        if os.path.exists(out_fn_):
                            continue
                        marker = out_fn_ + '.inprog'
                        try:
                            if (os.path.exists(marker) and
                                    time.time() - os.path.getmtime(marker)
                                    < timeout_s):
                                continue
                        except OSError:
                            pass
                        found.append((sp_, N_, s_, rung_))
        return found

    # ---- build the queue (skip completed cells)
    # Deployment-major ordering (Tom 2026-08-14): complete every (rung, N)
    # of seed k before starting seed k+1 -- and ACROSS specs (Tom
    # 2026-08-15: one deployment x all N x ALL objectives first) -- so
    # full per-deployment lines land first while the queue stays
    # saturated (no per-seed barriers).
    all_seeds = []
    for sp in specs:
        for s in sp['seeds_list']:
            if s not in all_seeds:
                all_seeds.append(s)
    cells = build_missing_cells()
    q = queue.Queue()
    for c in cells:
        q.put(c)
    # RAM-targeted governor: worker THREADS are provisioned to the
    # config's max_active when larger than --slots (so a live config can
    # raise concurrency without relaunching); the governor then admits
    # cells up to whichever is smaller at decision time.
    gov = MemGovernor(max_active=args.slots)
    n_threads = max(args.slots, gov.max_active)
    print('[queue] {} cells to run ({} specs), {} threads (governor '
          'max_active={}) x {} workers/run'.format(
              len(cells), len(specs), n_threads, gov.max_active,
              args.workers_per_run), flush=True)

    failures = []
    flock = threading.Lock()
    launch_lock = threading.Lock()
    last_launch = [0.0]

    def harvest_cell(sp, ws, N, s, rung):
        """Inline artifact harvest (spec['artifacts_figs']): copy THIS
        cell's convergence figure to the artifacts dir under the canonical
        '<label>_<rundir-suffix>.pdf' name, then remove the run dir (keeps
        head disk flat without an external harvest loop). Matches only the
        cell's own run dir so shared slots can never cross-label specs."""
        figs = sp.get('artifacts_figs')
        if not figs:
            return
        os.makedirs(figs, exist_ok=True)
        pat = os.path.join(ws, 'runs',
                           'ablation-{}-{}-dep{}-*'.format(sp['dpsize'], rung, s))
        for d in glob.glob(pat):
            suffix = os.path.basename(d).replace(
                'ablation-{}-'.format(sp['dpsize']), '', 1)
            src = os.path.join(d, 'convergence_over_iterations.pdf')
            if os.path.exists(src):
                shutil.copy(src, os.path.join(
                    figs, '{}_{}.pdf'.format(sp['label'], suffix)))
            me = os.path.join(d, 'model_error_over_iterations.pdf')
            if os.path.exists(me):
                shutil.copy(me, os.path.join(
                    figs, 'ME_{}_{}.pdf'.format(sp['label'], suffix)))
            shutil.rmtree(d, ignore_errors=True)

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
        while True:
            # RAM admission gate BEFORE pulling a cell: blocks (not
            # exits) while the box is at target utilization, so slots
            # re-open as cells finish or memory frees.
            while not gov.try_acquire():
                if q.empty():
                    return
                time.sleep(gov.sample_s)
            try:
                sp, N, s, rung = q.get_nowait()
            except queue.Empty:
                gov.release()
                return
            env = dict(os.environ)
            env.update({
                'PYTHONPATH': _REPO_ROOT,
                'SCULPTOR_ABLATION_GAMMA': sp['gamma'],
                'SCULPTOR_N_WORKERS': str(args.workers_per_run),
                'MPLBACKEND': 'Agg',
                'RAY_ADDRESS': 'local',
                'RAY_TMPDIR': '/tmp/ray_q_S{}'.format(slot),
                'SCULPTOR_ABLATION_PROBE_MODE': sp['probe_mode'],
                'SCULPTOR_ABLATION_PROBE_TCONV': str(sp['max_iter']),
                'SCULPTOR_ABLATION_PROBE_N': str(N),
            })
            env.update(sp['env'])
            log = os.path.join(ws, 'logs', '{}_N{}_s{}_{}.log'.format(
                sp['label'], N, s, rung))
            # 'runner' spec field (Tom 2026-08-17, EODS/fleet): any module
            # speaking run_fork_ladder's CLI + result-JSON convention can
            # be queued — EODS cells, future fleet workloads. Default
            # unchanged.
            cmd = [args.py, '-u', '-m',
                   sp.get('runner', 'experiments.ablation.run_fork_ladder'),
                   '--seed', str(s), '--rung', rung,
                   '--port', str(args.port0 + 20 * slot),
                   '--max-iter', str(sp['max_iter']), '--dpsize', sp['dpsize'],
                   '--out-dir', os.path.join(sp['out_root'], 'N{}'.format(N))]
            # global launch stagger: simultaneous deployment BUILDS are the
            # proven memory peak; serialize cell starts across all slots.
            with launch_lock:
                wait = last_launch[0] + args.launch_stagger - time.time()
                if wait > 0:
                    time.sleep(wait)
                last_launch[0] = time.time()
            out_fn = os.path.join(sp['out_root'], 'N{}'.format(N),
                                  'seed_{}_{}.json'.format(s, rung))
            marker = out_fn + '.inprog'
            try:
                with open(marker, 'w') as mf:
                    json.dump({'pid': os.getpid(), 'slot': slot,
                               'ts': time.time()}, mf)
            except OSError:
                pass
            timeout_s = float(os.environ.get('SCULPTOR_CELL_TIMEOUT', '7200'))
            with open(log, 'w') as lf:
                try:
                    rc = subprocess.call(cmd, cwd=ws, env=env, stdout=lf,
                                         stderr=subprocess.STDOUT,
                                         timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    rc = -99
                    print('[queue] TIMEOUT {} N={} seed={} rung={} after '
                          '{}s -> killed, re-queued next pass'.format(
                              sp['label'], N, s, rung, int(timeout_s)),
                          flush=True)
            try:
                os.remove(marker)
            except OSError:
                pass
            if rc != 0:
                with flock:
                    failures.append((sp['label'], N, s, rung, rc))
                    print('[queue] FAIL {} N={} seed={} rung={} rc={}'.format(
                        sp['label'], N, s, rung, rc), flush=True)
            else:
                harvest_cell(sp, ws, N, s, rung)
            gov.release()
            q.task_done()

    max_passes = int(os.environ.get('SCULPTOR_QUEUE_PASSES', '3'))
    for _pass in range(max_passes):
        if _pass > 0:
            cells = build_missing_cells()
            if not cells:
                break
            print('[queue] pass {}: re-queuing {} missing cells (failures/'
                  'timeouts/late-added work)'.format(_pass + 1, len(cells)),
                  flush=True)
            for c in cells:
                q.put(c)
        threads = [threading.Thread(target=slot_worker, args=(i,),
                                    daemon=True)
                   for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    print('[queue] sweep done; failures: {}'.format(len(failures)), flush=True)

    # ---- audit gate (same rules as run_n_sweep.sh), per spec
    bad = 0
    for sp in specs:
        for fn in glob.glob(os.path.join(sp['out_root'], 'N*', 'seed_*_*.json')):
            r = json.load(open(fn))
            if r['rung'] == 'painter':
                continue
            # Legitimate early exits end before max_iter. As of 2026-08-14
            # late only L1 (budgeted-fixed) produces budget_exhausted;
            # remeasure_triggered is legacy (skip-not-stop replaced it) --
            # both stay ACCEPTED for old datasets, neither can occur from
            # gated/scheduled/smart under current semantics.
            early_ok = ((r.get('exit_reason') == 'budget_exhausted'
                         and r.get('probes_spent', 0) >= 1)
                        or r.get('exit_reason') == 'remeasure_triggered'
                        # stop-v2 convergence exit (2026-08-16): legitimate
                        # for any cell running SCULPTOR_STOP_RULE=v2
                        or r.get('exit_reason') == 'stop_v2')
            if r.get('solve_error') or (
                    (r.get('n_iters') or 0) < sp['max_iter'] + 1 and not early_ok):
                print('[audit] BAD:', fn, r.get('n_iters'), str(r.get('solve_error'))[:40])
                bad += 1
            if sp['probe_mode'] != 'fixed' and r.get('probe_mode') != sp['probe_mode']:
                print('[audit] BAD (stale code, probe_mode={}):'.format(r.get('probe_mode')), fn)
                bad += 1
    print('[audit] {} bad runs'.format(bad), flush=True)
    if bad or failures:
        print('[queue] AUDIT FAILED')
        sys.exit(1)

    if not args.no_rescore:
        sem = threading.Semaphore(8)

        def rescore(sp, N, s):
            with sem:
                env = dict(os.environ, RAY_ADDRESS='local', MPLBACKEND='Agg',
                           RAY_TMPDIR='/tmp/ray_qrs_{}_{}_{}'.format(sp['label'], N, s))
                subprocess.call([args.py, '-m', 'experiments.ablation.rescore_fork',
                                 '--in-dir', os.path.join(sp['out_root'], 'N{}'.format(N)),
                                 '--dpsize', sp['dpsize'], '--seed', str(s)],
                                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rs = [threading.Thread(target=rescore, args=(sp, N, s), daemon=True)
              for sp in specs for N in sp['n_list'] for s in sp['seeds_list']]
        for t in rs:
            t.start()
        for t in rs:
            t.join()
        print('[queue] rescore done', flush=True)
    print('[queue] ALL DONE', flush=True)


if __name__ == '__main__':
    main()
