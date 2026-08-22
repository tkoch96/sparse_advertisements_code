# Handoff to session 10

Picking up a SCULPTOR research codebase mid-stride. **Read this doc
first**; `HANDOFF_SESSION_8.md` for the deeper architectural context
(objective registry, worker-adv plumbing fix, site_failure objective)
and `SESSION_4_SUMMARY.md` for the headroom/stop_tracker history.
Older handoffs are reference only.

## 🎯 Long-term north star

**Scale SCULPTOR emulations to evaluate very quickly on generic, cheap
compute.** Two pillars:

1. **Spot-instance compute.** Default deployment target is interruptible
   spot capacity (c7g.16xlarge or similar). This requires the sweep to
   be resilient to preemption: hot-start from disk checkpoints, idempotent
   per-`random_iter` writes, no all-or-nothing multi-hour runs that lose
   everything on a single eviction.
2. **Per-iter / per-LP computational efficiency inside SCULPTOR.** Drive
   down memory + CPU per iter so a given dpsize fits on smaller boxes
   (preferring `m7g.4xlarge` over `m7g.8xlarge` over `m7g.16xlarge`).
   Each level down is roughly half the spot $/hr.

Every change should be evaluated against these: does it let us run on
smaller / cheaper / more-interruptible hardware?

## 🚨 Active state (cluster is OFF)

Cluster stopped 2026-05-26 after dpsize=25 OOM-killed the m7g.4xlarge
head ~14.5h into a combined `SIZES=25,32 NSIM=3,2` boost. Specifically:
sparse driver process hit ~34GB (27GB RSS + 7GB shmem), painter parallel-
subprocess added ~22GB, Ray overhead pushed total above the 63GB head
RAM. Linux OOM-killer fired; head went into post-OOM degraded state
(SSH unresponsive, AWS reachability check failed).

| instance | type | state | notes |
|---|---|---|---|
| `i-09a6ff2823b0bb304` (head) | m7g.4xlarge | **stopped** | EBS preserved: all sweep state, runs/state-N.pkl checkpoints, per-dpsize pickles |
| `i-08c7b448c78a6f50f` (worker) | c7g.16xlarge spot | **terminated** | Was one-time spot (no stop option). Re-provision via `ray up` |

**Cost while parked:** ~$5-10/mo head EBS. Zero compute.

**Sweep state preserved on head EBS:**
- `cache/popp_failure_latency_comparison_testing_feature-actual-{3,5,10,15,20}_dep_sweep_*.pkl` — random_iter 0..4 complete (from prior boost)
- `cache/popp_failure_latency_comparison_testing_feature-actual-25_dep_sweep_25.pkl` — random_iter 0 only; iter 1 was in progress, never written
- `runs/<ts>-testing_feature-actual-25-sparse/state-N.pkl` — partial training checkpoint, last save was ≤iter 45 (state-N saved every 5 iters; from `[adaptive-budget]` log tail before SSH died)
- dpsize=32 pickle does not exist (never trained)

Local `cache/` already has the random_iter=0..4 pickles SCP'd this session — those are the post-boost state of dpsizes 3..20.

## 🚨 Critical path: diagnose large-instance crashes

The OOM root cause is known (sparse driver footprint × parallel painter
subprocess). What's NOT known: **where inside SCULPTOR the memory
actually accumulates**, and whether the growth is bounded (a single
expensive op) or unbounded (a leak across iters).

The session-9 mem instrumentation is enough to bracket whole-iter phases
(`grad`, `measure`, `stop_tracker`) but doesn't pinpoint individual
allocators. Goal for the next session: **enable detailed logging that,
post-hoc, lets us answer "what operation grew RSS by N MB at iter K?"
for any spike.**

### Pathways toward detailed logging (open to ideas)

1. **Structured event stream.** Emit one JSON line per significant event
   (LP solve, worker fan-out, cache insert, pickle write) with
   `{tag, iter, dpsize, rss_mb, sys_avail_mb, dt_ms, payload_size_mb,
   stacktrace_short}`. Cheap to grep, easy to analyze in pandas.
   Live alongside the existing `[mem]` text format. Suggested gate:
   `SCULPTOR_DEEP_TRACE=1`. File output to `runs/<ts>-actual-N-sparse/trace.jsonl`
   so it's easy to pull off the cluster.
2. **memray / tracemalloc snapshots at iter boundaries.** Take a heap
   snapshot at iter_start + iter_post_stop_tracker for the first 10
   iters and every 25 thereafter. Diff successive snapshots to find
   which allocators grow. memray has lower overhead than tracemalloc
   and writes a `.bin` we can pull and analyze locally.
3. **Per-worker mem deltas.** `_log_mem_worker` already exists in
   `path_distribution_computer.py` at 4 tags (worker_proc_start,
   worker_init_msg_received, update_deployment_enter/done). Extend to
   tag every gradient call entry/exit + cache state size. The driver
   currently can't see per-worker RSS climb (workers are remote Ray
   actors); without this, a worker leak is invisible.
4. **Gurobi model-size + solve-time logging.** Each LP call already has
   internal timing (`[InfoTiming]`, `[Timing]` prints). Wrap every
   `model.optimize()` to emit `{model_vars, model_constrs, nz_count,
   solve_ms, peak_solve_mem_mb}`. Gurobi has env-level APIs for this.
   Especially important because the WLS license throttles after 3
   concurrent sessions — knowing which LPs are big vs small helps
   prioritize what to optimize.
5. **Cache-size telemetry.** `MAX_CACHE_SIZE=8000` in
   `path_distribution_computer.py:57` is the only explicit bound. Most
   other caches (e.g., `self.measured`, `clear_new_measurement_caches`'s
   purged set, the latency-benefit dict) have no instrumentation.
   Per-iter cache cardinality + estimated bytes would catch unbounded
   growth.

A reasonable first session-10 pass: implement (1) + a stripped-down
version of (4), launch on `dpsize=10` (cheap, well-understood) for
~30 iters, sanity-check the trace, then on `dpsize=25` to see where
the actual growth happens.

### Mitigation options to pair with the relaunch

Don't relaunch the combined SIZES=25,32 sweep without picking one of
these (per `project_sculptor_dpsize25_oom_post_session8.md` memory):

| option | head $/hr delta | wall delta | risk |
|---|---|---|---|
| Upgrade head to m7g.8xlarge (128GB) | +$0.65/hr | same | Lowest risk. New IP after start. Doesn't fix the underlying inefficiency. |
| Disable parallel soln_types (run sparse → painter sequentially) | $0 | +50-100% | Halves peak head memory. dpsize=32 still uncertain (sparse alone was 34GB). Code change to `compare_different_solutions`. |
| Reduce per-driver memory (aggressive cache eviction, smaller caches) | $0 | small | Real risk of convergence regression. Needs validation at small first. |

For the long-term north star (cheap generic compute), **the
cache/memory work is the right investment** — but the deep-logging pass
should come first to know what to cut.

## 🎯 Longer-term performance opportunities

Once the OOM is understood, these are the structural opportunities to
make SCULPTOR run faster on smaller boxes. Roughly ordered by leverage.

### A. `path_distribution_computer.py`

The 1000+ line module that does most worker-side LP and cache work.
Suspected hot spots based on session-9 reading:

- **Cache-bypass paths.** Session 4 found one (`verbose_workers=True`
  bypassing the LB cache → 20s/iter waste); commit 219f078 fixed it.
  There may be others. Audit every kwarg that conditionally skips a
  cache lookup or pickle.
- **LP cache memory model.** `MAX_CACHE_SIZE = 8000` is a fixed bound;
  no LRU, no per-entry size tracking. Could be replaced with a memory-
  budget cache (e.g., `cachetools.LRUCache(maxsize=N_bytes)`) tuned to
  fit in available RAM. Critical for spot instances where you don't
  control the box size as tightly.
- **Per-worker RSS at update_deployment.** Already partly instrumented;
  the `[mem-worker]` tags at `update_deployment_enter/done` will reveal
  how much each worker grows when a new dpsize is loaded. If the
  growth scales with dpsize^2 (LP matrix size), that's the spot-
  compute bottleneck.

### B. High-level architecture

- **Sparse training is currently a single driver process that fans out
  to remote Ray workers.** The driver process itself holds enormous
  state (~27GB for dpsize=25). Could this state be partitioned across
  multiple driver processes, each handling a subset of UGs / PoPPs?
  Map-reduce style. Probably non-trivial to refactor but breaks the
  single-driver memory ceiling that forces head upgrades.
- **The compare_different_solutions parallel subprocess fan-out
  (commit df0ad32, "Run non-sparse strategies in parallel subprocesses")
  is what doubles head memory.** Worth reconsidering whether the
  subprocess isolation is necessary, or whether thread-pool execution
  (sharing the deployment in memory) would work. Need to check what
  state the strategies mutate.
- **Spot-instance preemption resilience.** The sweep currently
  hot-starts from `runs/state-N.pkl` saved every 5 iters. That's
  reasonable RPO (~5 iters of work lost) but worker-state recovery
  is untested. Need a deliberate preemption test: kill the worker
  mid-sweep, verify ray up brings up a new one and training continues.

### C. Gurobi ↔ framework interaction

- **WLS license is the binding throughput constraint.** Academic WLS
  allows ~3 sustained concurrent sessions; we run 32 workers each with
  their own Gurobi env. Every sweep accumulates `Overage for too long`
  errors. The license isn't hard-killing us yet but is slowing things
  down silently (sessions get throttled). Options:
  - Pool Gurobi sessions across workers (one shared `Env`, multiple
    `Model` per session) — needs an architectural change to Gurobi
    usage in `solve_lp_assignment.py`.
  - Switch to a different LP solver for the inner gradient probes
    (HiGHS is open-source, no concurrency limit, slower per-solve but
    might net positive). Keep Gurobi for the final solve.
  - Pay for a commercial WLS license. Probably not in scope for a
    student project.
- **LP model rebuilding vs warm-starting.** Most LPs in
  `solve_lp_assignment.py` get built fresh each call. Gurobi supports
  warm starts from previous solutions. For the gradient-probe LPs
  (which differ only in 1-bit advertisement flips), warm starts could
  be 10x faster. Needs validation.

## What session 9 did (short)

1. **Extended `benchmarks/run_deployment_sweep.py`** to accept per-dpsize
   NSIM as a comma-list parallel to SIZES (mirrors
   `evaluate_over_deployment_sizes.py`'s `n_sim_by_dpsize` pattern).
   Single-int form preserved for backward compat. See memory
   `project_sculptor_sweep_per_dpsize_nsim.md`.
2. **Added `log_mem(tag, worker_i=None, **extra)` shared helper** in
   `helpers.py`. Same `[mem]` format as existing `_log_mem` in
   `sparse_advertisements_v3.py` (kept duplicated). Importable
   from any module without circular-import risk.
3. **Instrumented `init_optimization_vars` and `measure_ingresses`**
   with `iov_*` and `mi_*` mem tags (sparse_advertisements_v3.py +
   optimal_adv_wrapper.py). These were the gap regions before the
   OOM diagnosis.
4. **Launched combined `SIZES=25,32 NSIM=3,2` sweep**, OOM'd after 14.5h.
   Per-iter timing breakdown captured: grad ~80s, measure ~10s,
   stop_tracker phase ~100s. The 100s "stop_tracker phase" is almost
   entirely `solve_max_information` (which always returned `None`
   under this config), not the stop check itself.
5. **Confirmed `SCULPTOR_ADAPTIVE_PROBE_BUDGET=1` is firing** and
   reducing grad-phase budget over iters (440 → 110 by iter 46).
   **No equivalent adaptive scaling exists for `info_support_size`**
   (max_info budget) — opportunity flagged in section A.
6. **Stopped head, terminated worker (spot).** Cluster fully off.

## 🚨 Uncommitted state

**Nothing in session 8 OR session 9 is committed yet.** Suggested commit
order if you want to land before resuming:

1. `path_distribution_computer*.py` worker-adv plumbing fix (session 8 bug)
2. `worker_comms*.py` `SCULPTOR_WORKER_INIT_STAGGER_SEC` + mem logging
3. `solve_lp_assignment.py` site_failure LP + retire dead LPs (session 8)
4. `experiments/` registry + driver + site_failure spec (session 8)
5. `tests/test_lp_correctness.py` site_failure tests (session 8)
6. `helpers.py` shared `log_mem` helper (session 9)
7. `sparse_advertisements_v3.py` `iov_*` mem tags in init_optimization_vars (session 9)
8. `optimal_adv_wrapper.py` `mi_*` mem tags in measure_ingresses (session 9)
9. `benchmarks/run_deployment_sweep.py` per-dpsize NSIM (session 9)
10. `HANDOFF_SESSION_9.md` (this doc)
11. `RESEARCH_ROADMAP.md` north-star pointer at top (session 9)

Run `git status` to see the exact delta. The instrumented files were
pushed to head via scp during the session; they'll be redeployed via
`file_mounts` when `ray up` runs next.

## Cluster ops cheatsheet

| action | command |
|---|---|
| Resume cluster (provisions fresh worker, rsyncs code, restores from EBS on head) | `~/Documents/venv312/bin/ray up -y cluster/ray-cluster.yaml` |
| Status check after resume | See "status one-liner" template in HANDOFF_SESSION_8.md (replace IP) |
| Tear down (destroys EBS — would lose head state) | `~/Documents/venv312/bin/ray down -y cluster/ray-cluster.yaml` |
| AWS state | `~/Documents/venv312/bin/aws ec2 describe-instances --filters "Name=tag:project,Values=sculptor" --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,PublicIpAddress]' --output table` |
| Console output (post-OOM forensics) | `~/Documents/venv312/bin/aws ec2 get-console-output --instance-id <head-id> --latest --output text \| tail -40` |
| CloudWatch CPU history | See `aws cloudwatch get-metric-statistics` call in session 9 transcript |

## Hazards & gotchas (additions from session 9)

(Carry forward from HANDOFF_SESSION_8.md; new items below.)

- **dpsize≥25 OOMs on m7g.4xlarge head with current parallel-soln_types
  config + post-session-8 LPs.** See `project_sculptor_dpsize25_oom_post_session8.md`
  memory. Must mitigate before relaunching.
- **AWS instance-status `impaired` is the canonical signal for OOM
  wedge.** `running` alone is misleading — the OS can be dead while the
  hypervisor still reports "running". Always check
  `describe-instance-status` if SSH times out.
- **Worker is a one-time Spot Instance.** It can only be terminated
  (not stopped). When the user wants the cluster off, head gets
  `stop-instances`, worker gets `terminate-instances`. `ray up`
  re-provisions a fresh worker; EBS lost, but no critical data lives
  there.
- **SSH ConnectTimeout doesn't bound TCP retry timeouts.** Setting
  `ConnectTimeout=2` can still take ~10s to actually fail. Wrap in
  Bash tool `timeout` parameter for hard caps, or live with the
  imprecision.
- **Per-dpsize NSIM env var.** `SCULPTOR_DEPLOYMENT_SWEEP_NSIM` accepts
  either a single int or a comma-list parallel to
  `SCULPTOR_DEPLOYMENT_SWEEP_SIZES`. Length mismatch raises
  `SystemExit`. Use this instead of writing per-dpsize launcher scripts.

## Key files modified this session (uncommitted)

```
helpers.py                              # +log_mem(tag, worker_i=None, **extra) shared helper
sparse_advertisements_v3.py             # +iov_* mem tags in init_optimization_vars
optimal_adv_wrapper.py                  # +mi_* mem tags in measure_ingresses
benchmarks/run_deployment_sweep.py      # per-dpsize NSIM (single int OR comma-list)
HANDOFF_SESSION_9.md                    # this doc
RESEARCH_ROADMAP.md                     # north-star pointer at top
```

Plus all session-8 files still uncommitted; see HANDOFF_SESSION_8.md
"Key files modified this session" for that list.

## Critical environment

- Local venv: `~/Documents/venv312/bin/python`
- Local repo: `~/Documents/sparse_advertisements_code`
- AWS: instances tagged `project=sculptor`. Head
  `i-09a6ff2823b0bb304` (stopped), worker terminated.
- Gurobi: WLS license at `~/gurobi.lic`. 3 concurrent-session
  baseline. See `project_gurobi_wls_concurrency.md` memory.
- ntfy topic: `sculptor-tk-95c9decb99ed7220`
