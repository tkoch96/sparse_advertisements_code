> 📜 **HISTORICAL** — session-6 snapshot. For current state see
> [README.md](README.md) and the highest-numbered `HANDOFF_SESSION_*.md`.
> File:line references in this document may be stale against current code.

# Handoff to session 6

Picking up a SCULPTOR research codebase mid-experiment. Read these
six files in order before doing anything:

  1. CLUSTER_RUNBOOK.md       — AWS Ray ops; cluster usage
  2. OVERNIGHT_SUMMARY.md     — session 2 results (per-iter timing baselines)
  3. SESSION_4_SUMMARY.md     — session 4 results; headroom finding
  4. SESSION_5_SUMMARY.md     — what just happened; current state of play
  5. RESEARCH_ROADMAP.md      — original next-steps plan + overlays
  6. HANDOFF.md               — pre-cluster context (skim only)

## Headline

- Headroom (`SCULPTOR_CAPACITY_HEADROOM=0.2`, RB-grad/value auto-skipped)
  is the established frontrunner; ships as the default in
  `benchmarks/headroom_n_trials.sh`.
- **actual-10 N=3 cross-seed** result is clean: sparse ≈ one_per_peering
  on normal-LP and popp-failure.
- **actual-32 single-trial** result is recovered: sparse 28.96 ms vs
  one_per_peering 28.35 ms (within 0.61 ms / 97.4% within 10ms). Saved
  at `recovered_actual32/` (gitignored binaries, README committed).
- Step 1 (batched `getAttr("X", active_vars)`) shipped and validated
  on a parametrized sweep: 15-22% wall savings at decent scale, scales
  better with bigger var_pool.
- Cluster torn down. $0 idle.

## 🚨 FIRST TASK: recover the missing actual-32 failure-mode metrics offline

The cluster eval at actual-32 wrote a pickle (`recovered_actual32/popp_
failure_latency_comparison_actual-32.pkl`) but the failure-eval fields
(`popp_failures_latency_optimal_specific`, `pop_failures_*`, all the
sub-eval fields) are **empty** for every strategy. The strategy-compare
phase completed (all 6 strategies' advertisements + normal-LP latencies
are there) but `assess_failure_resilience` either crashed silently
(try/except at eval_all_solution_types.py:325-328 swallows per-strategy
exceptions) or never ran. The cluster log was lost when sshd wedged.

**eval_latency_failure has a check_calced_everything resume mechanism.**
It'll skip the populated strategy-compare phase and run only the empty
failure-eval phases. Steps:

```bash
cd ~/Documents/sparse_advertisements_code
# Make sure the recovered pickle is at the expected cache path:
mkdir -p cache/
cp recovered_actual32/popp_failure_latency_comparison_actual-32.pkl \
   cache/popp_failure_latency_comparison_actual-32_actual32_n32_evalonly.pkl

# Run eval_latency_failure locally, hot-resuming from the saved pickle.
# It'll skip strategy-compare (compare_rets full), skip pct_volume_within_latency
# (populated), and run the failure-eval phase which is what's empty.
SCULPTOR_RUN_TAG=actual32_n32_evalonly \
  SCULPTOR_N_WORKERS=4 \
  ~/Documents/venv312/bin/python run_ray.py eval_latency_failure \
  --port 31510 --dpsize actual-32 \
  > /tmp/local_failure_eval.log 2>&1
```

Expected wall: ~30-60 min locally with 4 workers (779 popps × 6
strategies × ~2 LP solves each = ~9k LP solves, plus pop-failure +
sub-evals).

If it crashes:
- Grep the log for traceback. The most likely culprits are:
  - `solve_generic_lp_persistent`'s `exit(0)` on infeasibility at
    path_distribution_computer.py:228 (kills the worker actor →
    RayActorError → caught by try/except line 325-328 → empty data)
  - Ray serialization size limits on the per-popp call_args list in
    solve_lp_with_failure_catch_mp
  - Some assertion failure in the eval path at actual-32 scale
- The bug needs fixing for production-quality eval. Likely a small
  patch — e.g., replace `exit(0)` with `return {'solved': False}` so
  the worker survives infeasible failure scenarios.

Once it runs, the resulting pickle will have populated failure-mode
fields. Then compute the cross-strategy comparison matching the
paper's figure 6 (panel a — CDF of latency Δ under single link
failure; panel b — % within 50ms under link failure; panel c — same
under site failure). The Option-1 global-mean aggregation method is
documented in `benchmarks/worker_perf_investigation.md` and in
`recovered_actual32/README.md`.

Then update `recovered_actual32/README.md` with the actual-32 failure
numbers so the comparison to the paper is complete.

## Then: priority queue

(1) Cross-seed actual-32 (N=3 trials, seeds 1/2/3) — match the
    actual-10 phase B protocol. With Step 1 in place and per-strategy
    checkpoint (5b349be), per-trial cost ~7-10h. 3-trial parallel ~$20.
    Bring up cluster (`ray up cluster/ray-cluster.yaml`), launch via
    `benchmarks/headroom_n_trials.sh actual-32 200 3 1 2 3`. Be wary
    of painter — slow at actual-32. Consider skipping painter from
    soln_types if you don't need that specific baseline. Tear down
    every session.

(2) Step 2 (Gurobi MVar refactor) — bigger LP-orchestration win
    beyond Step 1. Sketch in `benchmarks/worker_perf_investigation.md`.
    Replace the per-var `addVar` + Column pattern in
    `path_distribution_computer.py:solve_unified_lp` with `addMVar` +
    sparse coefficient matrix. Validate with
    `tests/test_worker_perf_sweep.py` pattern: env-var toggle, ON/OFF
    on warmed-up var_pool, fingerprint match required. Expected
    payoff at actual-32: another ~20-30% wall reduction.

(3) Reply to Gurobi support thread about session-limit ask. Latest
    message draft (from session 5) was for ~256 sessions; reconfirm
    based on now-measured per-trial timing.

## Operational notes (mostly unchanged from session 4)

- ALWAYS run experiments on the cloud cluster, not local Mac.
  Exceptions in this handoff: the failure-eval recovery (intentionally
  local) and any new optimization scaffolding via `tests/test_worker_perf*.py`.
- Cron heartbeats don't reliably fire when session is dormant — use
  ScheduleWakeup or rely on user pinging for status.
- Per-trial PDFs in figures/ get RACED by concurrent trials. Use the
  per-seed metrics pickles in cache/, not the PDFs.
- Cluster yaml file_mount source is `~/Documents/sparse_advertisements_code`
  (main branch), not the worktree. FF-merge any branch work to main
  before `ray rsync-up`.
- `ray rsync-up` syncs to head only; existing worker nodes have stale
  code. Either terminate workers (re-spawn from head's file_mount on
  next demand) or restart the cluster on code changes.
- Tear down EVERY session that brings up the cluster.

## Critical environment

- venv: `~/Documents/venv312/bin/python`
- AWS: configured locally; instances tagged `project=sculptor` for
  teardown verification
- Gurobi: WLS license at `~/gurobi.lic`. Per Nicholas at Gurobi support
  (session 5 thread): current session limit is 2 (not 3 as memory
  earlier said). Asking for more is in flight.
- ntfy topic from session 3: sculptor-tk-95c9decb99ed7220

## Critical commits past handoff (newest first)

```
[Step 1 sweep results commit] — sweep landed: 15-22% wall savings
56754dc   Update worker-perf writeup: Step 1 shipped
aff7142   Recovered actual-32 single-trial result + writeup
90110ea   Step 1: batched getAttr(X, active_vars) for raw_x extraction
5b349be   Checkpoint metrics pickle after each strategy
7dc2706   Worker-perf investigation writeup
8e04b95   Remove intermediate i%50 summarize_timing trigger
26a04e2   Accumulate self.timing per-batch
0b6608f   Worker perf + correctness scaffold
ddd51f7   ray.put static deployment context once at start_workers
24ededb   Emit summarize_timing at end of every calc_compressed_lb batch
e17814a   Cleanup: remove stochastic_lp + consolidate flags
6fcb689   Skip resilience_benefit value under headroom mode
```

## Where the recovered actual-32 data lives

```
recovered_actual32/
├── README.md                                              (committed)
├── state-202.pkl                                          (771MB, gitignored)
│   └─ SCULPTOR converged advertisement at iter 202, deployment,
│      metrics, parent_tracker, measured_prefs. Sufficient to re-run
│      anything offline.
├── popp_failure_latency_comparison_actual-32.pkl          (37MB, gitignored)
│   └─ All 6 strategies' converged advertisements + normal-LP per-UG
│      latencies. Failure-mode fields are empty (the 🚨 first task above).
└── offline_failure_eval_summary.pkl                       (if first task succeeds)
```
