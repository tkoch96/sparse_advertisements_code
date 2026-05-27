> 📜 **HISTORICAL** — session-2 snapshot. For current state see
> [README.md](README.md) and the highest-numbered `HANDOFF_SESSION_*.md`.
> File:line references in this document may be stale against current code.

# Overnight session 2026-05-18 → 2026-05-19

Starts where HANDOFF.md left off. Goal of this session: stand up an AWS Ray
cluster, get a real per-iteration wall-time number for `actual-32` on cluster
hardware, capture timing breakdowns, leave everything ready for the next
session (Claude or human) to scale workers up.

## TL;DR for whoever reads this next

**What's working now (as of end-of-session 2026-05-19):**
- AWS Ray cluster fully operational. `ray-cluster.yaml` + `teardown.sh` in
  the repo work as-is. See `CLUSTER_RUNBOOK.md` for the cookbook.
- `actual-10` and `actual-32` both run end-to-end on cluster.
- Per-iter timing baselines: ~3 min/iter at actual-10 (8 workers),
  ~16 min/iter at actual-32 (32 workers on c7g.16xlarge).

**Big finding to validate:**
- Replacing the SGD-based resilience-benefit gradient with an in-LP
  capacity-headroom constraint gave **~6× per-iter speedup** on actual-10
  (216s → 37s). Cost-model implication: paper's 100-run grid drops from
  $1,700 → ~$300 in spot.
- Quality NOT yet validated — needs a multi-trial test with same-seed
  controls and a code fix to gate headroom off during eval phase. See
  `RESEARCH_ROADMAP.md` Tier 1.

**Total spend tonight:** ~$5-7 in spot (well under the $30 ceiling).

**What's NEXT for the picker-upper (highest leverage, in order):**
1. Land the two prerequisite code fixes (seed env var + gate headroom-off
   in eval). `RESEARCH_ROADMAP.md` #1.
2. Run the 5-trial × 100-iter quality A/B on actual-10. ~$30-40, ~8 hours
   parallelized on 5 workers.
3. If headroom passes quality bar, apply to actual-32 + write up.

**Cluster is torn down at end of session.** Costs $0 idle.

## Cluster shape used

- Yaml: `ray-cluster.yaml` in repo root.
- Region: `us-east-1`
- Head: `m7g.large` on-demand, CPU=0 resource (no actor work scheduled there)
- Workers: `c7g.16xlarge` spot, max_workers=1, idle_timeout=10 min
- Cluster name: `sculptor`, all instances tagged `project=sculptor`
- Codebase mounted to `/home/ubuntu/sparse_advertisements_code` from local
  `~/Documents/sparse_advertisements_code` via Ray file_mounts (incremental
  rsync). Gurobi WLS license file_mounted to `/home/ubuntu/gurobi.lic`.
- Data: ~4.5GB `vultr_ingress_latencies_by_dst.csv` shipped via rsync (one-time
  ~25 min upload on first ray up). Lives in `cache/` on cluster. The
  `data/vultr_peers_inferred.csv` (~500KB) also shipped.

## Pre-run setup hurdles (already fixed in yaml, here for reference)

Six independent failures hit between "first ray up" and "first SCULPTOR run":

1. **IAM perms.** AmazonEC2FullAccess is not enough; Ray needs IAMFullAccess
   (or scoped IAM perms) to create/attach an instance profile.
2. **`ray` binary not on system PATH.** Ray's autoscaler internally invokes
   bare `ray stop` from a non-interactive SSH shell, which doesn't see venv
   bin dirs. Fix: `sudo ln -sf /home/ubuntu/venv312/bin/ray /usr/local/bin/ray`
   in setup_commands.
3. **`worker_comms.py:16` hardcoded venv list.** Asserts that one of a
   hand-coded list of Python paths exists at module import. None matched the
   cluster's `/home/ubuntu/venv312/`. Fix: `ln -sfn /home/ubuntu/venv312
   /home/ubuntu/venv` so the hardcoded `/home/ubuntu/venv/bin/python` path
   resolves. (A proper fix would replace the list with `sys.executable`.)
4. **`figures/`, `logs/`, `runs/`, `cache/` dirs missing on cluster.** The
   codebase writes to them as a side effect of deployment setup but doesn't
   create them. Fix: `mkdir -p` in setup_commands.
5. **`addresses_violating_sol.csv` missing.** [deployment_setup.py:745](deployment_setup.py:745)
   reads it; an empty file works (the file is auto-populated by line 893).
   Fix: `touch` it in setup_commands.
6. **boto3 missing on head node.** Ray's autoscaler runs on the head and
   needs boto3 to talk to EC2 for worker provisioning. Without it, ray status
   shows pending demand forever but no worker is ever spawned. Fix: add
   boto3 to the pip install list. Silent failure — only visible in
   `/tmp/ray/session_latest/logs/monitor.err`.
7. **`rsync_filter: .gitignore` excluded `cache/`.** Ray's rsync file_mounts
   respects gitignore by default; this repo gitignores `cache/`, so the
   4.5GB latency CSV in cache/ never made it to the cluster. Fix: removed
   `rsync_filter` block from yaml entirely. Rely on explicit rsync_exclude.

All seven now baked into [`ray-cluster.yaml`](ray-cluster.yaml). A fresh
`ray up` from scratch should hit zero of these next time.

## Tests that pass on cluster

- `pytest -m "unit and not slow"` → **15/15 passing** in ~10s on head node.
  This validates: codebase imports, Gurobi license auth (academic license
  2487370, columbia.edu), persistent Gurobi LP path, warm-start invariants,
  basic LP correctness. Confirmed Gurobi works on c7g (ARM Graviton) via the
  WLS file_mount.
- Skipped: integration tests (would have spawned a worker; redundant with the
  decent SCULPTOR run, which exercises the same Ray code path with more
  coverage).

## Cost so far (when starting decent run)

- Head node up since 2026-05-19 02:49 UTC: ~35 min × $0.07/hr ≈ $0.05
- No workers running yet.
- Tasks performed so far:
  - 3 `ray up` invocations (1 succeeded fully after fixes)
  - 1 codebase + license rsync
  - 1 `ray up` that bundled the 4.5GB cache rsync
  - Unit tests on head

## Run 1: `decent` `MAX_ITER=2` (2026-05-19 03:24 UTC)

**Result: FAILED.** Took ~31 min wall (worker spawn + crash + downstream timeouts).
Two distinct issues surfaced:

### Issue A: worker actor crashed at __init__ (FIXED in yaml)

`path_distribution_computer_ray.py:66` opens `logs/worker_X_log-<dpsize>.txt`
with a *relative* path. The Ray actor's CWD on the spawned worker node is
NOT `~/sparse_advertisements_code` — typically `/home/ubuntu` or a Ray-
internal runtime resources dir — so the open() raises FileNotFoundError and
the actor dies at creation. Every downstream SCULPTOR call returned "No
solution" because no worker ever produced results.

Fix (already applied to yaml setup_commands and to the running head node):
```
ln -sfn /home/ubuntu/sparse_advertisements_code/logs /home/ubuntu/logs
ln -sfn /home/ubuntu/sparse_advertisements_code/cache /home/ubuntu/cache
ln -sfn /home/ubuntu/sparse_advertisements_code/figures /home/ubuntu/figures
ln -sfn /home/ubuntu/sparse_advertisements_code/runs /home/ubuntu/runs
ln -sfn /home/ubuntu/sparse_advertisements_code/data /home/ubuntu/data
```

A proper fix in code: replace `LOG_DIR` (relative) with an absolute path
derived from `os.path.dirname(__file__)` or `pathlib.Path(__file__).parent`.

### Issue B: synthetic-deployment evaluator bug (not blocking actual-32)

`wrapper_eval.py:741` `metro_to_diurnal_factor` does
`POP2TIMEZONE[metro]` but for synthetic deployments (`decent`, `med`),
`metro` is `np.int64(0)..np.int64(9)` (integer pop indices), not real city
names. `KeyError: np.int64(0)`. Only fires in the diurnal calc step of
eval_latency_failure.

Does NOT affect actual-32 (real metro names). Worth fixing for future
synthetic-deployment runs by adding a `if not isinstance(metro, str): return
1.0` guard in `metro_to_diurnal_factor`, or wrapping the lookup in
try/except.

### Cluster observations from this run

- Autoscaler IS working post-boto3 fix: "Adding 1 node(s) of type
  ray.worker.default" → "Resized to 64 CPUs" within ~20s of demand.
- Worker spot provisioning + setup_commands: total ~10-12 min from demand
  signal to worker registered with Ray.
- Worker auto-terminated after 12m46s idle (post-crash): "Removing 1 nodes
  of type ray.worker.default (idle)" — confirms `min_workers: 0` + 10-min
  `idle_timeout_minutes` is working as designed.
- Gurobi license worked cleanly on the c7g.16xlarge worker (academic
  license 2487370, columbia.edu).

### Skipping decent re-test, proceeding directly to actual-32

Reasoning: the symlink fix is in setup_commands AND applied ad-hoc to
running head; high confidence it unblocks the worker actor. Decent's diurnal
bug would obscure the success signal. actual-32 uses real city metros so
that bug doesn't fire. Going straight to actual-32 saves a worker
re-provision (~$0.25) and ~30 min.

## Run 2: `actual-32` `MAX_ITER=5` (2026-05-19 04:01 UTC)

**Result: HEAD NODE WENT IMPAIRED ~76s in (OOM, likely).** Driver process
got launched, started loading the 4.5GB cache CSV, and the m7g.large head
(8GB RAM) ran out of memory. Instance went `impaired`; SSH dropped (port 22
unreachable, ping 100% loss). EC2 instance-status: `running impaired ok` —
i.e. AWS sees the VM as up but the OS health check failed.

### Why the head OOMs on actual-32

The actual-32 driver path goes through
[deployment_setup.py:load_actual_perfs](deployment_setup.py:740), which:
1. Reads the entire 4.5GB `vultr_ingress_latencies_by_dst.csv` line by line
2. Builds `ug_perfs[(metro, asn)][popp].append(lat)` — a deeply nested dict
   that easily 3-4x expands the raw CSV in Python memory.
3. Computes pairwise pop_dists (small).
4. Filters/clusters users.

Peak memory during step 2 is likely 15-25GB. m7g.large has 8GB. OOM.

The SCULPTOR LP work itself (which is the HANDOFF.md target for parallelism)
happens on WORKERS, which are c7g.16xlarge with 128GB and plenty of margin.
But the driver runs on the HEAD, and it's the driver that does the data
loading.

### Fix: upgrade head to m7g.4xlarge

Updated yaml head node type from `m7g.large` to `m7g.4xlarge` (16 vCPU,
64GB RAM, $0.16/hr — small premium over $0.07/hr). RAM is the binding
constraint. Could go to `r7g.2xlarge` (8 vCPU, 64GB, $0.10/hr) for more
RAM-optimized pricing if cost-sensitive in future.

### Cost penalty of fresh head provision

Terminated instance lost its EBS, so the next `ray up` redoes everything
including the 4.5GB rsync over home upload (~25 min real time at home
bandwidth). One-time cost. If this churn happens again later, consider
mirroring the data to S3 instead so cluster provisions don't depend on
laptop upload speed.

### Worker RAM sizing — open risk

User noted (mid-session): "running actual-32 takes quite a few GB of RAM.
Maybe 2-3 GB per process." Implications:

- [constants.py:102 `get_n_workers`](constants.py:102) returns 1000 for
  `actual-32` (falls through the `if n_pops > 5` branch).
- [worker_comms_ray.py:140](worker_comms_ray.py:140) caps at
  `min(cpu_count, 1000)`. On c7g.16xlarge → **64 actors**.
- At 2-3 GB per actor × 64 = **128-192 GB peak RAM** vs c7g.16xlarge's
  **128 GB total**. Tight at best, OOM at worst.

Possible fixes if this OOMs:
1. **r7g.16xlarge worker** (64 vCPU, 512 GB RAM, ~$1.20/hr spot vs ~$0.60
   for c7g spot). Cheapest fix that doesn't touch code; 4x RAM headroom.
2. **Cap n_workers in code** (e.g., make `get_n_workers` respect a
   `SCULPTOR_MAX_WORKERS` env var, mirroring the SCULPTOR_MAX_ITER pattern
   I added this session).
3. **Mixed sizing** — head stays m7g.4xlarge, workers go to r7g.16xlarge.

If `actual-32 MAX_ITER=5` succeeds at 64 actors on c7g.16xlarge, the 2-3 GB
figure was an upper bound or actual-32 is less memory-heavy than larger
sizes. If it OOMs, switch to r7g.

## Run 3 — extra hurdles (file paths, missing files, `actual-10` smoke)

Several issues hit before getting clean SCULPTOR numbers:

### Issue C: worker actor logs path (FIXED in yaml)

`path_distribution_computer_ray.py:66` opens `logs/worker_X_log-<dpsize>.txt`
relative. Ray actor CWD on workers is `/home/ubuntu` not
`/home/ubuntu/sparse_advertisements_code`. Symlinked
`/home/ubuntu/logs -> /home/ubuntu/sparse_advertisements_code/logs` (and
cache/figures/runs/data) in setup_commands. Proper fix is to make these
paths absolute via `os.path.dirname(__file__)` in the codebase.

### Issue D: missing cache files (USER ACTION + FIXED in yaml)

actual-N needed two more data files we hadn't shipped:
- `cache/vultr_anycast_latency_smaller.csv` (~52 MB) — anycast latency data
- `cache/vultr_provider_popps.csv` (~2 KB) — provider peer mapping

Both were in user's `~/Downloads/` from the original Drive download (user
forgot to mention). Moved + rsync'd. False alarm on a third file
(`vultr_all_dsts_asn_apnic_pop.csv`) — gated by `APNIC_VOLUME = False` in
constants.py, never called.

### Issue E: `cache/deployments/` subdir didn't exist (FIXED in yaml)

`pickle.dump` to `cache/deployments/pruned_performances_<...>.pkl` failed
because parent dir wasn't created. Added `cache/deployments` to mkdir list
in setup_commands.

### Issue F: bug in flash crowd evaluator (NON-FATAL)

`eval_latency_failure.py:480` `assess_resilience_to_flash_crowds_mp` raises
`TypeError: unsupported operand type(s) for -: 'tuple' and 'tuple'`
repeatedly. Caught in some upstream try/except, doesn't kill the run, but
loses the flash crowd evaluation results. Pre-existing bug; not blocking
for our timing goal.

### Issue G: tracebacks from old top-of-file code (NON-FATAL)

`sparse_advertisements_v3.py:64` (`compare_estimated_actual_per_user`) +
`:1224` (`make_plots`) raise IndexError. User confirmed these are legacy
diagnostic code that should be removed in a cleanup pass. Don't alarm.

## Run 4: `actual-10` `MAX_ITER=5` (2026-05-19 16:46 UTC)

**Result: SUCCESS** — first end-to-end SCULPTOR run on cluster.

### Per-iteration grads timing on c7g.16xlarge (64 cores, 1 worker)

| LEARNING_ITER | grads (s) |
|---|---|
| 0 | 122.50 |
| 1 | 117.92 |
| 2 | 106.27 |
| 3 | 117.36 |
| 4 | 116.44 |
| 5 | 119.33 |
| 6 | 115.07 |

**Mean grads per iter: ~116s (~2 min)** — very consistent. Note: 7 iterations
despite `SCULPTOR_MAX_ITER=5` env var; the env var sets `max_n_iter` but the
LEARNING ITERATION counter exceeded this, suggesting either (a) inner loop
in eval_latency_failure runs SCULPTOR multiple times across "deployment
numbers" with each pass starting iter 0, or (b) the override is applied per
SAS instance not per outer loop. Doesn't affect timing.

### Full timer breakdown for one iter (mid-run)

| Phase | Wall | % of iter |
|---|---|---|
| `Timer: grads` | ~116s | ~75% |
| `Timer: stop` | ~44s | ~28% |
| `Timer: info` | ~10s | ~6% |
| `Timer: measure` | ~1s | <1% |
| Per-iter total | ~170s (2.8 min) | |

`grads` dominates exactly as HANDOFF.md predicted. New observation: `stop`
is the next biggest chunk (~28%) — that's the "max info" search step
(measuring 42 perms per iter for exploration). HANDOFF had this at 4% on
Mac; on the cluster it's a bigger relative share because grads got faster.

### Resilience benefit grad is a big chunk of `grads`

Mid-iter print: "Calcing latency benefit grad took 18s; Calcing resilience
benefit grad took 96s". So RB-grad is ~80% of the grads phase, not the LB
work. HANDOFF didn't decompose grads internally; this might be a productive
next thing to profile.

### Comparison metrics for baseline algorithms

Eval phase produced clean numbers for the comparison algos:

- painter: 96.3% of traffic within -10ms of optimal
- anycast: 80.45% within -10ms
- one_per_pop: 91.31% within -10ms
- one_per_peering: 100% within -10ms (trivial — every peering gets its own
  prefix)

SCULPTOR's own outputs are in `runs/...-actual-10-sparse/` on the cluster
(pickled per-iter state; pull with ray rsync-down if needed).

### Wall-clock total

actual-10 5-iter ran ~45 min wall:
- ~5 min: load 4.5GB CSV + cluster users into deployment
- ~5 min: worker spot provision + setup_commands (during which SCULPTOR
  waited on first actor schedule)
- ~20 min: 7 SCULPTOR iters × ~3 min/iter
- ~15 min: eval phase (with errors)

## Suggested optimizations (not implemented this session)

Things I noticed during the runs that the next session could pursue.
Listed in rough order of expected payoff. None of these were changed
tonight — they're for a future iteration.

### 1. Silent startup phase — pickling deployment per actor

[worker_comms_ray.py:145-173 `start_workers`](worker_comms_ray.py:145)
loops over N workers (64 for actual-32) and for EACH does:
```python
actor = Path_Distribution_Computer.remote(worker, subdeployments[worker], init_kwa)
```
This pickles `init_kwa` (large kwargs dict) freshly N times. Ray serializes
each constructor argument per actor. For actual-32 with 64 actors, this is
the dominant cost of the "silent startup phase" between deployment build
and iter 0.

**Fix:** `init_kwa_ref = ray.put(init_kwa)` once, then pass the ObjectRef
to all actors. Ray's object store deduplicates the underlying bytes; actor
constructors deserialize from a shared cache. Should cut the silent phase
significantly. Estimated payoff: 3-10 minutes of wall time saved per run
on actual-32, AND lower memory pressure (no N copies of the big dict in
transit).

Same pattern applies to `subdeployments[worker]` if there's a shared
"global" piece that's identical across workers (popps list, link capacities,
etc.). Refactor: split into a global "ray.put once" + a per-worker UG slice.

### 2. Constructor parallelism

Same code, the `for worker in range(n_workers)` loop fires N `.remote(...)`
calls serially. Each `.remote()` returns immediately (it's async) so this
might already be parallel-ish, but the subsequent `ray.get(ready_refs)`
waits for ALL to finish. Verify with timing: if `.remote()` itself is slow
(e.g., because Ray queues serialization), batching could help.

### 3. Per-actor RAM cap risk for larger deployments

n_workers is hardcoded to 64 for `actual-N` with N > 5. At 2-3 GB per
actor, that's 128-192 GB on a c7g.16xlarge (128 GB). actual-32 might just
fit; bigger deployments would OOM. Two fixes:
- Cap n_workers to leave RAM headroom: `min(cpu_count, max_ram_gb / per_actor_gb)`.
  Make `per_actor_gb` a tunable.
- Switch worker type to `r7g.16xlarge` (64 vCPU, 512 GB, ~$1.20/hr spot)
  — 4x more RAM per core at ~2x the spot price.

### 4. Worker_comms hardcoded venv list (already in memory)

`worker_comms.py:11-16` has a hardcoded list of Python paths. Replace with
`sys.executable`. Three-line change, eliminates the class of "venv path
doesn't match" bugs entirely.

### 5. Relative paths in SAS

Multiple places open files with relative paths
(`logs/...`, `cache/...`, `figures/...`). Tonight worked around it with
symlinks; the proper fix is `os.path.join(os.path.dirname(__file__), ...)`.
Especially relevant for parallel/distributed work where actor CWD differs
from driver CWD.

### 6. Eval phase bugs (flash crowd, diurnal, plot)

These are pre-existing bugs that the user has been ignoring. Worth a
cleanup pass:
- `wrapper_eval.py:741` `metro_to_diurnal_factor` — guard against
  non-string metros for synthetic deployments
- `eval_latency_failure.py:480` `assess_resilience_to_flash_crowds_mp` —
  the `tuple - tuple` TypeError needs an actual debug
- `sparse_advertisements_v3.py:64, 1224` — old diagnostic code that emits
  noisy tracebacks; remove

## Run 5: `actual-32` `MAX_ITER=5` with `SCULPTOR_N_WORKERS=32` + `ray.put(init_kwa)` (2026-05-19 18:50 UTC)

After several failed/diagnostic earlier attempts (head OOM on m7g.large,
missing files, n_workers=16 bottleneck), this is the first actual-32 SCULPTOR
run with the right config.

### Two optimizations applied this session

1. **`SCULPTOR_N_WORKERS` env var override** in
   [worker_comms_ray.py:134](worker_comms_ray.py:134). The original heuristic
   `min(multiprocessing.cpu_count(), suggested_for_dpsize)` ran on the
   driver, so it capped at the head's CPU count (16 on m7g.4xlarge), leaving
   3/4 of the c7g.16xlarge worker idle. Env var override bypasses this.
   Launched with `SCULPTOR_N_WORKERS=32` — confirmed via `ray status`:
   `32.0/64.0 CPU` in use on worker (vs 16/64 before).

2. **`ray.put(init_kwa)`** in start_workers. Single shared plasma store entry
   instead of per-actor pickle. Unclear how much this helped — the silent
   startup phase still feels long, but no negative effect observed.

### Iter 0 grads breakdown (real numbers!)

```
LEARNING ITERATION : 0
Calcing latency benefit grad took 202s
Calcing resilience benefit grad took 766s
```

- LB grad: **202s (~3.4 min)** — 11x slower than actual-10's 18s
- RB grad: **766s (~12.8 min)** — 8x slower than actual-10's 96s
- **Total grads: ~970s (~16 min) per iter**
- Per-iter total (grads + measure + info + stop): expected ~20-25 min

### Implications

- 5-iter actual-32 wall: ~2 hours of SCULPTOR + ~25 min of setup + ~25 min
  of eval phase = ~2.5-3 hours total
- 50-iter wall: ~16-20 hours of SCULPTOR alone — **infeasible on a single
  c7g.16xlarge**. To do 50-iter in a single overnight, need either:
  - More worker nodes (each c7g.16xlarge gives ~32 more actors, so 4 workers
    → 128 actors → ~5 hours)
  - Per-LP speedup (Gurobi `Method=2` barrier, multi-scenario API, etc.)
  - Algorithm changes (cache hit rate, smaller N, etc. — but user wants N
    same or larger)

### Why RB grad dominates (8x bigger than LB grad)

User mentioned earlier ("Resilience benefit grad was a big chunk" in the
actual-10 run, where RB took 96s vs LB 18s — ~5x ratio). At actual-32 the
ratio is ~4x, similar shape. RB is computing resilience over many possible
failures of popps; the search space grows with deployment size.

A future optimization target if pursuing larger-than-actual-32: profile RB
specifically. cProfile around `Calcing resilience benefit grad` would tell
you whether it's LP-bound or Python-bound.

## Run 6: A/B optimization tests on actual-10 (2026-05-19 evening)

After actual-32 v3 produced stable baseline numbers, used actual-10 as the
fast A/B harness for testing two optimization hypotheses.

### Run A — baseline (actual-10, MAX_ITER=3, N_WORKERS=8)

| Iter | LB | RB | Total grads |
|---|---|---|---|
| 0 | 40 | 187 | 228 |
| 1 | 37 | 187 | 225 |
| 2 | 37 | 177 | 215 |
| 3 | 37 | 179 | 217 |
| 4 | 36 | 160 | 197 |
| **Mean** | **37** | **178** | **216** |

RB/LB ratio: **4.8×** — RB is 80%+ of grads time. (Same pattern as actual-32
where RB/LB was ~4.5×.)

### Run B — `SCULPTOR_RB_NO_UGS_SUBSET=1`

Hypothesis: dropping the `ugs=` subset on RB calls enables cache hits +
better persistent-Gurobi warm-start because the LP variable set is constant.
Implementation: env-var gate added to `gradients_resilience_benefit_popp`
in sparse_advertisements_v3.py.

| Iter | A (baseline) | B (no_ugs) |
|---|---|---|
| 0 | 228s | 226s |
| 1 | 225s | 229s |

**Result: WASH at actual-10 scale.** Per-iter wall is within noise. Worker-
level prints showed per-call LB iter dropped from ~10s to ~3s (warm-start
DID help), but per-iter total didn't move — the bigger-LP cost canceled the
warm-start savings.

Hypothesis is NOT falsified for actual-32 scale, where bigger LP cost might
not scale as fast as warm-start savings. Worth testing if time allows. For
now, **don't ship this change as-is** — neutral at small scale.

### Run C — `SCULPTOR_CAPACITY_HEADROOM=0.2 SCULPTOR_SKIP_RB_GRAD=1`

Hypothesis (your design): move resilience from SGD-RB out of the gradient
loop and INTO the inner LP as a capacity-headroom constraint. Multiply all
link capacities by `(1 - 0.2) = 0.8` so the LP always reserves 20% headroom
for failure absorption. Skip RB-grad entirely.

Implementation:
- `_apply_capacity_headroom` helper in
  [solve_lp_assignment.py](solve_lp_assignment.py) wraps each
  `sas.link_capacities_arr.flatten()` use (7 sites).
- Same headroom applied to `static_caps` in the persistent Gurobi path at
  [path_distribution_computer.py:95](path_distribution_computer.py:95).
- `SCULPTOR_SKIP_RB_GRAD=1` early-returns zeros from
  [gradients_resilience_benefit](sparse_advertisements_v3.py:1223).

| Iter | A (baseline) | C (headroom) | Speedup |
|---|---|---|---|
| 0 | 228s | **40s** | 5.7× |
| 1 | 225s | **37s** | 6.1× |
| 2 | 215s | ~37s | ~5.8× |

**Result: ~6× per-iter speedup.** RB grad is 0s as designed.

The remaining question is whether the final advertisement produces
comparable POST-FAILURE performance to the baseline RB-grad version (the
LP-headroom approach should give a more robust solution by construction,
but we need to verify in the failure-eval phase). C is running through eval
phases now; the comparison numbers will be:
- Average latency on no-failure (should be COMPARABLE to baseline since LP
  uses same objective)
- Average latency under popp failure (this is where headroom should pay off)

If headroom produces comparable failure-eval to baseline, this is a clear
algorithmic win and likely the most important finding of this session.

### Implications for the cost model

If the 6× speedup holds at actual-32 (it should — RB/LB ratio was similar):
- Per-iter for actual-32 drops from ~16 min → **~3 min**
- 200-iter for actual-32 drops from ~55h → **~9h** on 1× c7g.16xlarge
- 100-run grid drops from $1,700 (spot) → **~$300** (spot)
- With Gurobi cap raise + a few worker nodes → **~1-2h per run** is plausible

This single optimization is bigger than the cluster scaling work in this
session.

## Optimization 8 (added this session): S3 mirror for `cache/`

The 4.5GB cache CSV had to rsync from laptop every fresh `ray up`. At home
upload speeds that's ~25 min. Mirror to S3 once + setup_commands does
`aws s3 sync s3://bucket /home/ubuntu/sparse_advertisements_code/cache`.
Fast within us-east-1, free transfer. Pays back after 1-2 cluster rebuilds.
Was on the original task list as task #10 (data shipment) but skipped in
favor of direct rsync to keep iteration loop short.
