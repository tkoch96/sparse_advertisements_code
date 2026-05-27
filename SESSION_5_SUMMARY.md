> 📜 **HISTORICAL** — session-5 snapshot. For current state see
> [README.md](README.md) and the highest-numbered `HANDOFF_SESSION_*.md`.

# Session 5 summary (2026-05-21)

Starts where SESSION_4_SUMMARY.md left off. Goal this session: verify
the session-4 cache-bypass fix didn't break things, get a clean
cross-seed actual-10 read, take the headroom approach to actual-32,
and iterate on per-iter speedups. Read in order: HANDOFF.md →
CLUSTER_RUNBOOK.md → OVERNIGHT_SUMMARY.md → SESSION_4_SUMMARY.md →
this file → RESEARCH_ROADMAP.md.

## TL;DR

- **stop_tracker cache-bypass fix verified safe** (within same-seed noise envelope)
- **Phase B finished: actual-10 N=3 cross-seed headroom data is clean**. sparse ≈ one_per_peering on normal-LP and popp-failure
- **actual-32 single-trial result recovered** from a wedged cluster via stop-start. SCULPTOR's converged advertisement (iter 202) preserved + normal-LP comparison for all 6 strategies
- **Step 1 timing optimization shipped** (batched `getAttr(X, active_vars)` for raw_x extraction): 15-22% wall savings at decent/small scales, fingerprint-match correctness, scales with var_pool size
- **Big code cleanup**: deleted ~2200 lines of stochastic_lp + session-3 dispatcher code; consolidated `SCULPTOR_SKIP_RB_GRAD` into `SCULPTOR_CAPACITY_HEADROOM>0`
- **Eval-checkpoint hook landed**: `compare_different_solutions` now pickles after every strategy via an `on_strategy_complete` callback — protects against painter / anyopt crash losing the SCULPTOR advertisement
- **Cluster torn down**, $0 idle. ~$15 total cluster spend this session

## What's working

- Code state on `main` is at HEAD with all session-5 commits (worktree branch FF-merges cleanly)
- All 15 unit tests still pass
- `tests/test_worker_perf.py` + `tests/test_worker_perf_sweep.py` give a correctness-gated bench for future LP-optim PRs
- Recovered actual-32 data at `recovered_actual32/` (gitignored binaries, README.md committed)

## What this session changed (commits, newest first)

```
0b6608f → 7dc2706 → 90110ea → aff7142 → 56754dc

[hash]    [summary]
56754dc   Update worker-perf writeup: Step 1 (batched getAttr X) shipped
aff7142   Recovered actual-32 single-trial result + writeup
90110ea   Step 1: batched getAttr(X, active_vars) for raw_x extraction
5b349be   Checkpoint metrics pickle after each strategy in compare_different_solutions
7dc2706   Worker-perf investigation writeup
8e04b95   Remove intermediate i%50 summarize_timing trigger
26a04e2   Accumulate self.timing per-batch (was overwrite per LP call)
0b6608f   Worker perf + correctness scaffold (parametrized over real deployments)
ddd51f7   ray.put static deployment context once at start_workers (10× at N=64)
24ededb   Emit summarize_timing at end of every calc_compressed_lb batch
e17814a   Cleanup: remove stochastic_lp + consolidate flags + add timing debug
6fcb689   Skip resilience_benefit value under headroom mode
```

Plus: latest Step-1-sweep commit landed after `aff7142`.

## What this session found (results)

### Phase B at actual-10 — N=3 cross-seed (headroom + skip-RB-grad)

Trials at seeds 1, 3, 4. Pickles preserved in cache/ on cluster (not pulled locally since they were already analyzed mid-session and the cluster is now down):

| metric | sparse | one_per_peering | painter |
|---|---:|---:|---:|
| Normal-LP mean Δ (ms) | **−0.10** | 0 (ref) | −1.11 |
| Normal-LP % ≤ −10ms | **99.8%** | 100% | 97.7% |
| Popp-failure mean Δ (global Option-1) | **−0.009** | 0 (ref) | −0.034 |
| Pop-failure mean Δ (global Option-1) | −0.259 | 0 (ref) | −0.125 |

Sparse essentially ties the idealized `one_per_peering` upper bound
on every axis at actual-10.

### actual-32 single-trial recovered result (seed=1, MAX_ITER=200)

Converged at iter 202 (stop=True). Normal-LP comparison across all 6
strategies (5173 UGs, 779 popps, 32 pops):

| strat | mean lat (ms) | mean Δ vs OPP | % ≤ −10ms | % ≤ −50ms | % ≤ −100ms |
|---|---:|---:|---:|---:|---:|
| one_per_peering | 28.35 | 0 (ref) | 100% | 100% | 100% |
| **sparse** | **28.96** | **−0.61** | **97.4%** | 99.9% | 100% |
| painter | 29.36 | −1.01 | 94.6% | 99.8% | 100% |
| one_per_pop | 33.50 | −5.15 | 83.5% | 97.7% | 99.7% |
| anyopt | 44.55 | −16.20 | 73.1% | 88.8% | 95.0% |
| anycast | 54.59 | −26.24 | 70.3% | 84.0% | 90.0% |

Sparse stays within 0.61 ms / 97.4% of the idealized upper bound at
production scale. Slight degradation vs actual-10 (was 0.10 ms / 99.8%
there) is expected — more popps means more places to leave performance
on the table — but still solidly the top non-idealized strategy.

### Step 1 timing optimization — sweep at small + decent

Batched `getAttr("X", active_vars)` replaces a per-var loop over the
full var_pool (which grows to 228k at decent, projected 0.5-2M at
actual-32):

| dpsize | n_perms | var_pool | wall reduction | per-LP savings |
|---|---:|---:|---:|---:|
| small | 32 | 2,231 | -15.5% | 1.5 ms/solve |
| small | 132 | 2,231 | -15.4% | 1.9 ms/solve |
| decent | 32 | 228,453 | -21.7% | ~91 ms/solve |
| decent | 132 | 228,453 | -18.2% | 86 ms/solve |

Per-LP savings scale linearly with var_pool size (legacy code iterates
var_pool; Step 1 iterates only active_vars). Projected actual-32 win:
~260-600 ms/solve, ~3+ hours saved per full trial.

Correctness fingerprint matches in every config.

### What did NOT work (proven dead ends with explanations)

- **Multi-Scenario Gurobi API**: `optimize()` is 0.5% of wall at
  actual-32. Already retired session-4. Confirmed by per-LP-component
  breakdown in `benchmarks/step1_sweep_results.md`.
- **Skip `setAttr("UB", all_vars, [0]*N)` (Optimization 1)**: saved
  51s setup at decent but cost 34s in Gurobi's `optimize()`. Apparent
  Gurobi-basis-bookkeeping interaction. Reverted.

## Open questions / hypotheses

1. **Why did the cluster eval phase silently produce empty failure-mode
   fields?** Strategy-compare + pct_volume_within_latency completed
   (data present), but assess_failure_resilience for both popp and pop
   failures wrote no rows for any of the 6 strategies. The
   try/except at eval_latency_failure.py:325-328 catches per-strategy
   exceptions and continues silently. The cluster log was lost when
   sshd wedged before we could grep for tracebacks. Could be: Ray
   serialization size limits on the per-popp call_args list, OOM
   during the LP fan-out, or `exit(0)` from `solve_generic_lp_persistent`
   on an infeasible failure scenario (which exits the worker actor and
   propagates as RayActorError).

2. **Step 2 (MVar refactor) potential**: Step 1 saves ~20% at decent.
   A larger refactor that uses Gurobi's `addMVar` + sparse coefficient
   matrices (matching the OLD non-persistent path's bulk API but with
   persistence) would speed up `solve_unified_lp_not_optimize` and
   raw_x simultaneously. Not yet attempted. Expected payoff at
   actual-32: another ~20-30%.

3. **Does var_pool ever need clearing?** Answer (this session): no
   — once the hot paths stop iterating var_pool, its size becomes
   irrelevant for perf. Memory cost (~few hundred bytes/var × 1M vars
   = ~500MB/actor) is also fine on c7g.16xlarge.

4. **N=3-5 cross-seed actual-32**: only have N=1 for now. Paper-grade
   needs more seeds.

## Suggested next steps (priority order)

### 0. Immediate (within next session)

- **Recover actual-32 failure-mode metrics offline** — run
  `eval_latency_failure.evaluate_all_metrics` locally against the
  recovered pickle. The check_calced_everything machinery will skip
  the already-done strategy-compare/pct_vol_within_latency phases and
  run only the failure-eval. ~30-60 min locally. Steps:
  1. Set `SCULPTOR_RUN_TAG=actual32_n32_evalonly` 
  2. Copy `recovered_actual32/popp_failure_latency_comparison_actual-32.pkl`
     to `cache/popp_failure_latency_comparison_actual-32_actual32_n32_evalonly.pkl`
  3. `python run_ray.py eval_latency_failure --port 31510 --dpsize actual-32`
     (local Ray, 4-8 workers)
  4. Watch for the failure-eval traceback — that diagnosing answers
     open question #1
  5. Diff the resulting metrics into the recovered_actual32 writeup

### 1. Tier 1 — finish the actual-32 single-trial result

- Compute the recovered failure-eval (above)
- Cross-reference against the paper's figure 6 panels (b) and (c)
- Update `recovered_actual32/README.md` with the comparable numbers

### 2. Tier 1 — cross-seed actual-32 (N=3 trials)

- With the new commits in place (Step 1 + per-strategy checkpoint),
  per-trial cost should be ~7-10h instead of 10-12h
- 3 trials in parallel × 7h × $0.85/hr ≈ $20
- Need to bring up cluster again (`ray up ray-cluster.yaml`). Be wary
  of painter at actual-32 — it adds hours. Maybe skip painter from
  soln_types for now and add it back when investigating fig 6.

### 3. Tier 2 — Step 2 MVar refactor

- Bigger LP-orchestration win. Sandbox-test with the existing
  scaffold (`tests/test_worker_perf_sweep.py` pattern). Sketch in
  `benchmarks/worker_perf_investigation.md`.
- Goal: another 20-30% wall reduction at actual-32 scale.
- Risk: requires care to preserve warm-start basis semantics.

### 4. Tier 2 — diagnose the failure-eval crash

- Run #0 (offline failure-eval) will produce a traceback if the crash
  reproduces locally. Patch the underlying bug.
- Likely culprits: `solve_generic_lp_persistent`'s `exit(0)` on
  infeasibility (path_distribution_computer.py:228), large
  call_args serialization in solve_lp_with_failure_catch_mp.

### 5. Tier 3 — paper grid (`evaluate_over_deployment_sizes`)

- The {3, 5, 10, 15, 20, 25, 32} × N seeds grid. With Step 1 + skip
  painter, projected cost ~$200-400 for the whole grid.

## Files / artifacts of interest

| path | what it is |
|---|---|
| `recovered_actual32/state-202.pkl` (771MB) | SCULPTOR converged advertisement + deployment + metrics. Gitignored. |
| `recovered_actual32/popp_failure_latency_comparison_actual-32.pkl` (37MB) | All 6 strategies' advs + normal-LP latencies. Missing failure-mode fields. Gitignored. |
| `recovered_actual32/README.md` | Headline + provenance + how to use |
| `benchmarks/worker_perf_investigation.md` | Methodology + measurement + what works/doesn't |
| `benchmarks/step1_sweep_results.md` | Quantified Step 1 win across 5 configurations |
| `tests/test_worker_perf.py` | Parametrized scaffold with correctness gate |
| `tests/test_worker_perf_sweep.py` | ON/OFF sweep for any toggleable optimization |
| `path_distribution_computer.py` | The worker; has the env-var `SCULPTOR_DISABLE_RAW_X_BATCH` toggle |
| `eval_latency_failure.py` | The eval driver; checkpoint hook + per-strategy save on crash |

## Operational notes that bit prior agents (mostly unchanged from session 4)

- Cluster yaml file_mount source is `~/Documents/sparse_advertisements_code` (main branch), not the worktree. FF-merge before `ray rsync-up`
- `ray rsync-up` syncs to head only, NOT to running worker nodes — to push code changes to workers, either terminate the worker (it'll re-spawn with fresh code from head's file_mount) or use `ray rsync-up --all-nodes` if Ray supports it on your version
- Tear down EVERY session that brings up the cluster. `./teardown.sh`. Verify with `aws ec2 describe-instances --filters Name=tag:project,Values=sculptor`
- `painter` is slow at actual-32 (5-10 min/iter × 30 iters = 2-5 hours alone). Consider dropping it from soln_types if you don't need that specific baseline for your figure
- The deployment dict has SOME fields per-UG (`ug_perfs`, `ug_to_vol`, etc.) and SOME static (`popps`, `link_capacities`, `whole_deployment_*`). The session-5 `ray.put`-static refactor exploits this; helpers.py:split_deployment_by_ug_separated returns them separately for actor init.
- The new `SCULPTOR_GRB_DUMP=<dir>` env var (uncommitted) instruments worker 0 to write its first 3 LP solves as .mps files + a Gurobi log file. Useful for sending to Gurobi support.

venv: `~/Documents/venv312/bin/python`
AWS: configured locally
Gurobi: WLS license at `~/gurobi.lic` (session limit 2 per Nicholas's email; we asked for more)
