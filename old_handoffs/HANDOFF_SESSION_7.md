> 📜 **HISTORICAL** — session-7 snapshot. For current state see
> [README.md](README.md) and the highest-numbered `HANDOFF_SESSION_*.md`.
> File:line references in this document may be stale against current code.

# Handoff to session 7

Picking up a SCULPTOR research codebase with an actively-running cluster
sweep. Read these files in order before doing anything:

  1. CLUSTER_RUNBOOK.md       — AWS Ray ops; cluster usage
  2. OVERNIGHT_SUMMARY.md     — session 2 results (per-iter timing baselines)
  3. SESSION_4_SUMMARY.md     — session 4 results; headroom finding
  4. SESSION_5_SUMMARY.md     — session 5 results
  5. HANDOFF_SESSION_6.md     — session 6 entry; what session 6's agent saw on arrival
  6. recovered_actual32/README.md — final actual-32 single-trial result + paper-comparable failure/diurnal/flash stats
  7. benchmarks/worker_perf_investigation.md — Step 1 timing wins from session 5
  8. benchmarks/step1_sweep_results.md       — Step 1 cross-dpsize sweep results
  9. RESEARCH_ROADMAP.md      — original next-steps plan + session overlays
  10. HANDOFF.md              — pre-cluster context (skim only)

## 🚨 ACTIVE: deployment-size sweep running on cluster

A `benchmarks/run_deployment_sweep.py` job is in flight on the cluster
(head node IP: `3.239.5.131`). Don't blow it away. Status check command:

```bash
ssh -i ~/.ssh/ray-autoscaler_us-east-1.pem -o StrictHostKeyChecking=no \
    ubuntu@3.239.5.131 \
    'pgrep -fa "python.*run_deployment_sweep" | grep -v "bash -c" | head -1 && \
     grep -E "^\[sweep\]|Stopped train loop|adaptive-budget" /tmp/sweep.log | tail -15 && \
     echo --- && free -m | head -2'
```

State at handoff:
- dpsize 3, 5, 10, 15: **COMPLETE** (skipped on resume via `compare_rets`-populated check)
- dpsize 20: hot-started from `runs/1779444055-testing_feature-actual-20-sparse/`,
  sparse stopped at iter 140 (new stop-fix), currently in failure-eval phase
- dpsize 25, 32: pending; will run fresh sparse training under the new stop+budget fixes

Per-dpsize eval pickles are at `cache/popp_failure_latency_comparison_testing_feature-actual-{N}_dep_sweep_{N}.pkl`.
Top-level aggregate at `cache/testing_feature_cache_fn.pkl`.

ETA from handoff: ~3-5 hr (dpsize=20 ~15-30 min remaining, then dpsize=25 ~30-90 min, then dpsize=32 ~60-90 min — all with the new fixes engaged).

## What session 6 did (this session)

Three big themes: **bug fixes**, **memory/throughput optimizations**, and a **deployment-size sweep launch**.

### Bug fixes (eval-phase correctness)

1. `5980124` Fix diurnal + flash_crowd eval crashes on synthetic deployments
   - diurnal: `metro_to_diurnal_factor` fell over on synthetic `np.int64`
     metros because `POP2TIMEZONE` only has real-name keys. Added a
     deterministic-hash fallback so synthetic metros still get varied
     timezones.
   - flash_crowd: `assess_resilience_to_flash_crowds_mp` was zipping
     `raw_solution` (a dict) as if it were a list of values. Fix
     handles both forms.
   - Net: diurnal + flash_crowd phases populate fully at all dpsizes
     they didn't before.

2. `7d2f945` Replace exit(0) on LP infeasibility with no-route sentinel
   - `path_distribution_computer.py:236` was calling `exit(0)` when an
     LP came back infeasible. That kills the Ray actor → silently
     caught as RayActorError → eval data for that strategy lost.
     Replaced with a `lats_by_ug = full(NRL)` sentinel dict that
     downstream consumers treat correctly.

3. `da63fb8` Gate SCULPTOR_CAPACITY_HEADROOM strictly on `_in_training`
   - Three sites used to read the env var without checking
     `_in_training` → headroom could leak into eval. Plus the
     training-mode toggle had no try/finally → an SGD-loop exception
     would leave `_in_training=True` permanently. Both fixed.

### Memory / throughput optimizations

4. `4ff85c3` Cap driver memory in eval phases
   - `assess_failure_resilience` ran with `cache_res=True` →
     `linear_prog_soln_cache` accumulated all per-LP results across
     strategies. At actual-32 that's ~25 GB of dead-weight cache.
     Flipped to `cache_res=False` + added
     `clear_lp_caches()` between phases. **This alone would have
     prevented the OOM that killed the first actual-32 eval attempt.**

5. `7017a31` Strip heavy LP-result fields + per-strategy checkpoint + heap trim
   - `light_result=True` kwarg on `solve_lp_with_failure_catch_mp`:
     strips `raw_solution`, `available_paths`, `vols_by_poppi`,
     `solved`, `objective` on driver receive — saves ~80% of each
     LP-result's memory.
   - Per-strategy `gc.collect()` + glibc `malloc_trim(0)` after each
     strategy in the failure-eval loop. Without this, RSS grew
     monotonically across strategies even with light_result.
   - Per-strategy `pickle.dump` checkpoint so OOM mid-strategy
     preserves prior strategies' data.

6. `4469ab8` Three failure-eval throughput wins
   - **Skip update_dep** in failure-eval call_args (was hardcoded
     `i%20==0` even though dep never changes; each reset wiped the
     gti cache + cost ~3s).
   - **Enable `do_cache=True`** for the worker's `solve_lp` handler
     (was hardcoded False; with #1, the cache now stays warm and
     hits across the 1622 failure scenarios).
   - **Precompute one-per-peering reference LPs once, share across
     strategies** — these are the same per (which, failed_popp_or_pop),
     not strategy-specific. Cut LP count in half for failure-eval.
   - End-to-end on actual-32 failure-eval went from "OOM after 3
     strategies" → 5/6 strategies completing in ~35 min (limited
     ultimately by Gurobi WLS 32-min cap firing during painter's
     local LP).

7. `df0ad32` Run non-sparse strategies in parallel subprocesses
   - `compare_different_solutions` now forks 5 cheap-strategy
     subprocesses concurrently with sparse. Each subprocess gets
     its own `Sparse_Advertisement_Eval` without a worker_manager;
     `solve_lp_with_failure_catch` routes to the local
     non-persistent LP. anyopt's MC phase has a serial-LP fallback
     when `worker_manager` is None.
   - Saves ~30-60 min per dpsize at scale on the strategy_compare
     phase wall.

8. `a1d6a26` SGD: env-gated stop-fix + adaptive probe budget; sweep auto-hotstart
   - `SCULPTOR_STOP_DROP_ADV_DELTA=1`: drops `el[3]`
     (rolling_adv_delta) clause from `stopping_condition`. That
     clause requires the MAX |Δ| across all adv elements to drop
     below 0.01, but proximal L1 makes a few bits oscillate forever
     near the threshold, so the clause never fired → every dpsize≥5
     hit MAX_ITER=200. With the clause dropped + epsilon tightened
     ×0.1, training stops on the objective clauses when they
     genuinely indicate convergence.
   - `SCULPTOR_ADAPTIVE_PROBE_BUDGET=1`: scales `lb_support_size`
     down as rolling_delta shrinks (sqrt schedule, min floor =
     `max(10, 5*n_pop)`). Reduces per-iter LP probe work in the
     converged-tail regime.
   - Both validated locally on small × 200 with N_WORKERS=1:
     baseline 202 iters / 329s, fixes 138 iters / 247s (~1.3x faster
     at small, much larger relative win at bigger dpsizes).
   - Both default-ON in `benchmarks/run_deployment_sweep.py`.

### Sweep infrastructure

9. `52e0cab` Add deployment-sweep launcher with configurable nsim
   - `benchmarks/run_deployment_sweep.py` wraps the existing
     `evaluate_over_deployment_sizes.pull_results_new` logic with:
     - `SCULPTOR_DEPLOYMENT_SWEEP_NSIM` (default 1)
     - `SCULPTOR_DEPLOYMENT_SWEEP_SIZES` (default
       `3,5,10,15,20,25,n_vultr`)
     - `SCULPTOR_DEPLOYMENT_SWEEP_TAG` (default `dep_sweep`)
     - Per-dpsize pickle write after each completes so a mid-sweep
       crash preserves smaller-size results.

10. `a1d6a26` Auto-hot-start logic in the sweep launcher
    - Before each dpsize: check if its eval pickle has
      `compare_rets[0]['n_advs']` populated. If yes → pass
      `save_run_dir=None` so evaluate_all_metrics' resume logic skips
      training. If no → search RUN_DIR for the latest
      `<ts>-testing_feature-actual-{dp}-sparse/` directory; if found,
      pass it as `save_run_dir` so sparse hot-starts from its last
      saved state. Lets us kill + restart the sweep without losing
      prior iters.

### Documentation

11. `4789ac1` recovered_actual32/README.md update with FINAL eval-resume
    stats (paper-comparable Fig 12 metric for failure, plus volume
    multipliers / diurnal / flash crowd tables). **This is the
    canonical writeup of the actual-32 single-trial result.**

12. `fb226bc` `tests/test_worker_perf_warmup_curve.py` — pytest scaffold
    that sweeps `N_WARMUP ∈ {0, 1, 3, 5, 10}` to quantify the var_pool
    warmup effect on Step 1. Carry-over from session 5 work.

13. This handoff doc.

## Critical commits past session 6 handoff (newest first)

```
4bdbe61   Merge main (README + recovered_actual32 stats) into branch
a1d6a26   SGD: env-gated stop-fix + adaptive probe budget; sweep auto-hotstart
4789ac1   Append full eval-resume stats to recovered_actual32/README.md
52e0cab   Add deployment-sweep launcher with configurable nsim
4469ab8   Three failure-eval throughput wins: drop reset churn + gti cache + OPP-ref dedup
7017a31   Strip heavy LP-result fields + per-strategy checkpoint + heap trim in failure-eval
7d2f945   Replace exit(0) on LP infeasibility with no-route sentinel return
4ff85c3   Cap driver memory in eval phases: kill failure-eval LP cache + clear between phases
fb226bc   Add var_pool-size warmup-curve sweep for Step 1
da63fb8   Gate SCULPTOR_CAPACITY_HEADROOM strictly on _in_training; make toggle exception-safe
df0ad32   Run non-sparse strategies in parallel subprocesses (simulated mode)
5980124   Fix diurnal + flash_crowd eval crashes on synthetic deployments
```

## 🚨 FIRST TASK: monitor the sweep through to completion

The sweep was started at ~12:25 UTC (mid-session restart after the
stop-fix + adaptive-budget landings). Monitor it via the command at
the top of this doc. Look for:

- **Phase progression per dpsize**: strategy_compare → pct_volume →
  failure → vol_mult → diurnal → flash_crowd → next dpsize
- **`[sweep] dpsize=X done in Ys (cumulative Zs)`** lines mark completions
- **`[adaptive-budget]` lines** — these appear during sparse training
  as the per-iter probe budget shrinks. Should fire on dpsize 25 and
  32 (sparse training from scratch). Did NOT fire on dpsize 20
  because sparse hot-started past where the budget shrinks.
- **Errors** — `Traceback|GurobiError|Overage|Killed`. Zero acceptable.

When the sweep finishes:

1. **Pull both pickles locally** for archival:
   ```bash
   scp -i ~/.ssh/ray-autoscaler_us-east-1.pem ubuntu@3.239.5.131:/home/ubuntu/sparse_advertisements_code/cache/testing_feature_cache_fn.pkl .
   scp -i ~/.ssh/ray-autoscaler_us-east-1.pem 'ubuntu@3.239.5.131:/home/ubuntu/sparse_advertisements_code/cache/popp_failure_latency_comparison_testing_feature-actual-*_dep_sweep_*.pkl' .
   ```

2. **Compute paper-comparable Fig 12 + Fig 7 metrics across all 7 dpsizes**.
   The aggregate pickle has `stats_*` keys per dpsize already; use
   `evaluate_over_deployment_sizes.make_paper_plots` to generate
   the actual paper figures.

3. **Compare to actual-32 single-trial result** in
   `recovered_actual32/README.md` to see where dpsize=32 lands (should
   be similar but not identical — different seed because the sweep
   uses a freshly-generated `testing_feature-actual-32` deployment
   via `get_random_deployment`).

4. **Tear down the cluster** — `ray down -y cluster/ray-cluster.yaml`. Verify
   no leftover EC2 instances tagged `project=sculptor`.

## 🚨 KEY NEXT GOAL: run the sweep at nsim > 1 for paper-quality figures

`nsim=1` gives a single-seed point per dpsize, which is noise-dominated
per the documented `SCULPTOR_DEPLOYMENT_SEED noise` finding. To produce
publication-quality Figure 12 / Figure 7 plots you need cross-seed
averages. The sweep launcher supports this via
`SCULPTOR_DEPLOYMENT_SWEEP_NSIM`. Original paper used nsim 6-20 per
dpsize.

The recommended way to scale up:
1. After this nsim=1 sweep finishes, decide on a target nsim (probably
   start with nsim=3 for a quick second pass)
2. Re-launch the sweep with that nsim. **Key insight: the per-dpsize
   eval pickle is shared across nsim runs because `random_iter`
   indexes into `metrics[k][random_iter]`. So the new run continues
   filling in `random_iter=1, 2, ...` while the
   first run's `random_iter=0` is preserved.** This requires
   `evaluate_all_metrics` to support resuming with a higher nsim;
   double-check this works as expected — it might require deleting
   the per-dpsize pickle and re-running everything from scratch with
   the new nsim.
3. If full re-run is needed for higher nsim: budget accordingly.
   Estimated wall for nsim=3 sweep with new fixes:
   ~12-18 hr (3 × current ~4-6 hr remaining). Cost ~$10-15.

## Hazards & operational notes

- **WLS license limit is 2 sessions baseline, hard-cut at 32 min of
  overage**. We've been running 32+ sessions sustained for 10+ hours
  without the cap firing — unclear if Nicholas/Gurobi has quietly
  raised it or if we're getting lucky. Don't launch local Gurobi
  while the cluster sweep is running (combined session count is what
  the WLS tracks). MPS-dump request from Nicholas is still pending —
  the small.mps file we made is at `/tmp/gurobi_dumps/` on Tom's
  laptop; for decent/actual32 dumps, wait until cluster sweep is
  done OR add an env-var gated MPS dump path inside
  solve_lp_assignment.py and run with the env var set on the next
  sweep.
- **Cluster file_mount only syncs to head**. Worker stale code is OK
  for changes that only affect driver-side code (the case for all
  this session's changes); worker-side changes (e.g.
  path_distribution_computer.py) require terminating + respawning
  the worker. We did one such cycle this session for the
  `do_cache=True` flip; took ~5 min for worker setup_commands to
  complete.
- **Cluster cost** ~$0.70/hr combined (head m7g.4xlarge + 1 worker
  c7g.16xlarge). Total session cluster spend: probably $15-20.
- **Cron heartbeats unreliable**; rely on `ScheduleWakeup` or user
  pings.
- **Per-trial PDFs in `figures/` get RACED by concurrent trials**;
  use the per-seed metrics pickles in `cache/`, not the PDFs.
- **`cluster/ray-cluster.yaml` has `min_workers: 1`** (temporarily set this
  session). Revert to `0` before tear-down so future `ray up`s start
  with the autoscaler-on-demand default.

## Critical environment

- venv: `~/Documents/venv312/bin/python`
- AWS: configured locally; instances tagged `project=sculptor`
- Gurobi: WLS license at `~/gurobi.lic`. Session limit 2 per Nicholas
  (session 6 thread). Asking for more is in flight; representative
  MPS + log files owed to Nicholas (see hazards).
- ntfy topic from session 3: sculptor-tk-95c9decb99ed7220

## Where the deployment-sweep data lives

```
cache/
├── testing_feature_cache_fn.pkl                          (top-level: {dpsize: {stats_*: ...}})
├── popp_failure_latency_comparison_testing_feature-actual-3_dep_sweep_3.pkl    (per-dpsize)
├── popp_failure_latency_comparison_testing_feature-actual-5_dep_sweep_5.pkl
├── popp_failure_latency_comparison_testing_feature-actual-10_dep_sweep_10.pkl
├── popp_failure_latency_comparison_testing_feature-actual-15_dep_sweep_15.pkl
├── popp_failure_latency_comparison_testing_feature-actual-20_dep_sweep_20.pkl
├── popp_failure_latency_comparison_testing_feature-actual-25_dep_sweep_25.pkl   (pending)
└── popp_failure_latency_comparison_testing_feature-actual-32_dep_sweep_32.pkl   (pending)

runs/
├── 1779427572-testing_feature-actual-3-sparse/    (sparse state-N.pkl files for hot-start)
├── 1779428040-testing_feature-actual-5-sparse/
├── 1779432478-testing_feature-actual-10-sparse/
├── 1779438235-testing_feature-actual-15-sparse/
└── 1779444055-testing_feature-actual-20-sparse/
                                                   (25, 32 will be created as the sweep runs)

recovered_actual32/                                (separate from sweep — single-trial actual-32 result)
├── README.md                                       (committed; full paper-comparable stats)
├── popp_failure_latency_comparison_actual-32_FULL.pkl  (47 MB, gitignored — local archive)
└── state-202.pkl                                   (771 MB, gitignored — original sparse training state)
```
