# Research roadmap

## 🎯 Current north star (session 9, 2026-05-26)

**Scale SCULPTOR emulations to evaluate quickly on generic, cheap
compute** — spot instances + computational efficiency inside SCULPTOR.
The work below should be evaluated against: does it let us run on
smaller, cheaper, more-interruptible hardware? **See
`HANDOFF_SESSION_9.md` for the current state, the critical-path OOM
diagnosis work, and the longer-term perf opportunities in
`path_distribution_computer.py`, the parallel-soln_types architecture,
and Gurobi usage.**

## ⚠ Session-4 addendum (2026-05-21)

Most of the items below have been touched or completed. **Read
`SESSION_4_SUMMARY.md` first** — it has the up-to-date state.

Short version of what session 4 did:

- Built Option A (stochastic LP via Gurobi Multi-Scenario API) end-to-end:
  solver in `stochastic_lp.py`, unit tests, SCULPTOR-loop integration
  behind `SCULPTOR_USE_STOCHASTIC_LP_GRAD=1`.
- Empirically: at small × 100 iter (single seed) **headroom beats both
  RB-grad and stochastic-LP on normal-LP and popp-failure quality**.
  Pop-failure is the noisy column (range 4 ms across seeds at N=3).
- IS variant diverged. Math is unbiased on average but per-iter variance
  too high for SGD. See SESSION_4_SUMMARY.md "IS variance issue".
- Fixed a perf bug in stop_tracker: `verbose_workers=True` was bypassing
  the worker LP cache, costing ~20s/iter at actual-10. One-line fix at
  `sparse_advertisements_v3.py:1877`. **All session-1/2/3 timing numbers
  are overstated by ~2× as a result.**
- Phase B (headroom × N=5 at actual-10 × 150 iter) running at end of
  session.

Items in this file that remain relevant:

- Tier 1 #2 (headroom sweep) — never done; next session should run it.
- Tier 2 #4 (re-enable summarize_timing) — partially relevant; the
  [Timing] prints in stop_tracker already exist and gave us the
  cache-bypass clue.
- Tier 2 #5 (Gurobi multi-scenario API) — **done**, but its SCULPTOR-loop
  integration loses to headroom in single-seed testing.
- Tier 3 #11 (stochastic-LP follow-up paper) — `stochastic_lp.py` is now
  a working starting point.

---

(Original roadmap, written after session 2, follows. Items partially
done are not edited inline; see SESSION_4_SUMMARY.md for the truth.)

Next-steps plan for SCULPTOR after session 2 (overnight cluster work
2026-05-19). Captures what we tried, what worked, what didn't, and what to
do next — in priority order.

Read `OVERNIGHT_SUMMARY.md` first for empirical results and
`CLUSTER_RUNBOOK.md` for the operational setup. This file is the *what to
do* list, not the *what we did* list.

---

## The big finding to validate

**Replacing the SGD-based resilience-benefit gradient with an in-LP
capacity-headroom constraint gives ~6× per-iter speedup on actual-10.**

Per-iter wall: 216s (baseline RB-grad) → 37s (headroom + skip-RB-grad)
on `actual-10` with `SCULPTOR_N_WORKERS=8`. The SCULPTOR algorithm code
path is untouched; we just (a) reserve 20% of link capacity in the inner
LP and (b) skip the SGD-RB gradient entirely.

**Status: speed validated, quality NOT yet validated** (single-trial
comparison was muddied by random-seed differences across A/B runs AND by
the headroom also applying to eval-phase LPs, shifting the "optimal"
reference for everyone).

Cost-model implication: if this scales to actual-32 as the actual-10/-32
LB:RB ratios suggest it will, the 200-iter run for the paper drops from
~55 h → ~10 h, **and the 100-run grid drops from $1,700 → $300 in spot**.

This is the most important loose end.

---

## Tier 1 — validate the headline (1-3 are prerequisites for the paper)

### 1. Multi-trial quality A/B at convergence

The question: when SCULPTOR converges (not 5-10 iters), how does the
headroom version compare to baseline on:
- avg latency vs `one_per_peering` (idealized upper bound)
- avg latency under simulated popp failure
- gap to `painter` (the closest comparable baseline)

**Plan:**
- 5 trial pairs at actual-10 with `SCULPTOR_MAX_ITER=100` (or 150 — see
  user's "you really only see how sparse does after many iterations")
- Each pair: same deployment seed, A (RB-grad) and C (headroom)
- Compare the eval-phase metrics, not just per-iter timing

**Prerequisites** (cannot run cleanly without these):

1. **`SCULPTOR_DEPLOYMENT_SEED` env var**. Currently each run regenerates a
   random deployment, so A and C see different problem instances. To
   fix: thread an env-var-overrideable seed into:
   - `deployment_setup.get_random_deployment` (`np.random.seed` near top)
   - SCULPTOR's `init_advertisement` (also seeded — see line ~224 of
     `sparse_advertisements_v3.py`)
   ~10 lines of code. Test that the same seed produces byte-identical
   first-iter `tmp_a` between runs.

2. **Gate headroom OFF during eval phase**. Currently `_apply_capacity_headroom`
   in `solve_lp_assignment.py` reads `SCULPTOR_CAPACITY_HEADROOM` on every
   LP solve, including the eval-phase `evaluate_all_metrics`. This shifts
   the "optimal" reference for ALL solutions (sparse, painter, anyopt, etc.)
   not just SCULPTOR. Fix options:
   - Have eval phase explicitly `os.environ.pop('SCULPTOR_CAPACITY_HEADROOM')`
     before calling `evaluate_all_metrics` and restore after. ~3 lines.
   - Cleaner: add a `sas._in_training` boolean, set/unset around the
     gradient loop, check it in `_apply_capacity_headroom`. ~6 lines.

**Resources:**
- Cluster: bump `max_workers: 5` in yaml so 5 trial pairs run in parallel
- Wall time: ~6-8 hours for A runs (5 × 100 iters × ~3 min/iter / 5 parallel)
  + ~1 hour for C runs (5 × 100 iters × ~37s/iter / 5 parallel) + ~2 hours
  eval. So ~10 hours total.
- Cost: 5 spot c7g.16xlarge × ~$0.80/hr × ~8 h = **~$30-40**

**Output:** a `quality_ab_results.csv` with columns: seed, condition (A/C),
solution, avg_lat_vs_optimal, popp_fail_diff, pop_fail_diff. Plus a 1-page
analysis: does C preserve SCULPTOR's gap-to-painter? Does it preserve the
gap to `one_per_peering`?

### 2. Headroom sweep

Same setup as #1 but ONE seed, sweep `SCULPTOR_CAPACITY_HEADROOM` ∈
{0.05, 0.10, 0.15, 0.20, 0.25, 0.30}. Find the smallest value that still
gives competitive failure-resilience.

Paper-quality figure (this is a real figure for the writeup):
- x-axis: headroom fraction
- y-axis: SCULPTOR's gap to `one_per_peering` under popp failure
- Annotate: baseline (no headroom, RB-grad on) as horizontal reference line
- Sweet spot is the smallest headroom that's within noise of baseline

**Resources:** 6 runs × ~1 hour at actual-10/100 iters in parallel = ~1-2 h
on 6 workers. ~$5-10.

### 3. Apply best-found-headroom to actual-32

Once #1 and #2 land, run the chosen headroom value on actual-32 at full
iteration count (200) to confirm the 6× scales and quality holds at the
target deployment size.

Single run. With 32 workers on c7g.16xlarge and ~3 min/iter expected:
~10 hours wall, ~$8 in spot.

---

## Tier 2 — structural optimizations (independent of #1-3)

### 4. Re-enable `summarize_timing` to profile RB grad's 80%

`path_distribution_computer.py:65` has the existing timing infrastructure
with a `return` at line 66 disabling it. Remove the return and add a call
to `summarize_timing()` at the end of `gradients_resilience_benefit_popp`.

Existing timing keys: `optimize`, `solve_unified_lp_not_optimize`,
`get_paths_by_ug`, `organizing_results`, `pmat_organize`,
`solve_generic_lp_persistent`, `total_rti_calc`, etc.

This will tell you whether the 80% of grads time is:
- Gurobi-bound (then: multi-scenario API, Method=2, HiGHS)
- Python-orchestration (then: vectorize, batch, cache)
- Ray fan-out (then: larger batches per remote call, ray.put for shared
  args)

Critical: do this BEFORE attacking RB-grad further. We've been guessing
based on aggregate timing; this gives the precise breakdown.

### 5. Gurobi multi-scenario API for the inner LP

Even with headroom replacing RB-grad, LB-grad still calls the inner LP
many times per iter with structurally-related variations (different
toggled bits in the advertisement). Gurobi's multi-scenario API
(`setObjectiveN`, scenario parameter sets) batches N parametric LPs into
one `optimize()` call, exploiting shared basis.

Estimated payoff: 1.5-3× per-LP solve speedup. Independent of headroom
optimization — they stack.

Implementation lives in `path_distribution_computer.py:solve_generic_lp_persistent`
and `init_persistent_lp` (~lines 74-145 area). Budget: half-day to wire
up, half-day to test correctness against current.

### 6. ~~Refactor `worker_comms.py:11-16` hardcoded venv paths~~ **OBSOLETE**

Resolved by the Ray-only refactor (session 10, 2026-05-27). The hardcoded
venv `PYTHON` paths were used by the old ZMQ Worker_Manager to spawn
`path_distribution_computer.py` as subprocesses. The ZMQ path is gone;
workers are Ray actors and no subprocess-spawn step exists anymore.

### 7. Replace relative paths with `os.path.dirname(__file__)`-based absolutes

Multiple places open files with relative paths (`logs/...`, `cache/...`,
`figures/...`). Symlinks in setup_commands work around it, but the proper
fix is `os.path.join(os.path.dirname(__file__), 'logs', ...)`. Then the
Ray actor's CWD doesn't matter.

Files affected: `path_distribution_computer.py:66`,
`optimal_adv_wrapper.py:558`, `sparse_advertisements_v3.py:1138`, plus
`deployment_setup.py` opens. Maybe 30-40 occurrences total. Probably
~1-2 hours of focused work.

---

## Tier 3 — paper hygiene + scaling out

### 8. Fix eval-phase bugs

Three known bugs that produce log noise and partial result loss:

- **`eval_latency_failure.py:480`** `assess_resilience_to_flash_crowds_mp`
  raises `TypeError: unsupported operand type(s) for -: 'tuple' and 'tuple'`
  repeatedly. Flash-crowd eval results partially lost.

- **`wrapper_eval.py:741`** `metro_to_diurnal_factor` does
  `POP2TIMEZONE[metro]` but for synthetic dpsizes (`decent`, `med`) the
  metros are np.int64 not strings. Doesn't affect actual-N runs (real city
  names). Easy fix: `if not isinstance(metro, str): return 1.0`.

- **`sparse_advertisements_v3.py:64, 1224`** (`compare_estimated_actual_per_user`
  and `make_plots`) raise IndexError on most runs. Old diagnostic code.
  Either delete or guard with try/except.

All three: ~2 hours of focused work to clean up. Worth doing before the
multi-trial validation runs (#1) so the result logs are clean.

### 9. S3 mirror for cache/

The 4.5GB latency CSV currently lives only on Tom's Mac. Every fresh
`ray up` rsyncs it from the laptop over home upload (~25 min for the
initial transfer). Mirror it to a private S3 bucket:

```bash
aws s3 mb s3://sculptor-tom-data
aws s3 cp cache/vultr_ingress_latencies_by_dst.csv s3://sculptor-tom-data/
aws s3 cp cache/vultr_anycast_latency_smaller.csv s3://sculptor-tom-data/
aws s3 cp cache/vultr_provider_popps.csv s3://sculptor-tom-data/
aws s3 cp data/vultr_peers_inferred.csv s3://sculptor-tom-data/data/
```

Add to `ray-cluster.yaml` setup_commands:
```yaml
- aws s3 sync s3://sculptor-tom-data /home/ubuntu/sparse_advertisements_code/cache
- aws s3 sync s3://sculptor-tom-data/data /home/ubuntu/sparse_advertisements_code/data
```

Workers need IAM permission for S3 read. Add to the worker node_config:
```yaml
IamInstanceProfile: { Name: ray-cluster-s3-read }
```
(after creating the role in IAM console.)

Within us-east-1, S3 transfer is free + fast (gigabit+). Initial setup:
~15 min upload from Mac. Then every fresh `ray up` saves ~25 min.

Storage cost: ~$0.10/month for 5GB. Trivial.

### 10. Cluster scale-out yaml variants

Once headroom is validated, the 100-run grid for the paper becomes
tractable. Create a `ray-cluster-grid.yaml` with:
- `max_workers: 8` (8 spot c7g.16xlarge = 512 vCPU, fits in current 640
  Gurobi quota)
- Same yaml otherwise

For multi-tenancy (running multiple grid jobs concurrently on the same
cluster), use Ray placement groups + per-job actor pools.

Alternatively: keep `max_workers: 1` and spin up N independent clusters
in parallel (cluster_name: sculptor-{seed}), each its own SCULPTOR run.
Simpler scheduling, no shared-resource contention. Cost is the same.

### 11. Stochastic-LP follow-up paper

If the headroom approach lands cleanly, the natural follow-up is a
proper scenario-based stochastic LP formulation: per failure scenario,
solve the routing LP, optimize expected objective across scenarios. Uses
Gurobi's multi-scenario API to do this efficiently.

Heavier — see OVERNIGHT_SUMMARY.md "stochastic LP" discussion. Belongs
in a separate paper or as a future-work section.

---

## Code TODO inventory

Stuff in the codebase that bit us this session and should be cleaned up
opportunistically:

- `gradients_resilience_benefit_pop` (line 1108 of v3.py) is unused
  (alpha=0). Remove or document why kept.
- ~~`worker_comms.py:11-16` hardcoded paths (#6 above)~~ — gone with the Ray-only refactor.
- `path_distribution_computer.py:66` `summarize_timing` `return` (#4 above).
- All `os.path.join` calls in driver/worker code paths should use
  `__file__`-based absolute paths (#7 above).
- `eval_latency_failure.py` and `wrapper_eval.py` eval bugs (#8 above).
- `sparse_advertisements_v3.py:64, 1224` old diagnostic code (#8 above).
- `SCULPTOR_MAX_ITER` is off-by-2 — actual-10 with MAX_ITER=3 runs 5 iters,
  with MAX_ITER=5 runs 7 iters. Inner loop probably counts differently. Worth
  hunting down for clean comparison runs.
- `get_n_workers(deployment_size)` in constants.py returns 1000 for
  `actual-N` with N>5, then capped by cpu_count. The 1000 is arbitrary and
  has no per-actor RAM consideration. Should be `min(cpu_count, max_ram_gb
  / per_actor_gb)` for robustness.

---

## How to use Claude (the agent) most effectively on this

Based on the session-2 experience:

- **Start the session with**: "Read CLUSTER_RUNBOOK.md, OVERNIGHT_SUMMARY.md,
  and RESEARCH_ROADMAP.md before doing anything." Three files give 80% of
  context in ~10 min of agent time.
- **Don't ask the agent to run a long experiment without `nohup`-detached
  pattern + Monitor for streaming events.** SSH dies, runs die, agent
  loses track.
- **For A/B experiments**: agent writes the dispatcher, you decide
  cost/parallelism. Agent should NOT spin clusters > $30 spot/day without
  explicit OK.
- **End every cluster session with `./teardown.sh`.** Hard-code this into
  the agent's workflow.
- **Use actual-10 as the iteration harness** for ALL algorithmic
  experiments. ~5 min/run with cached deployment. Don't iterate on
  actual-32 (1+ hour/run) for fast-feedback work.
- **Memory files in `~/.claude/projects/.../memory/`** captured the
  hard-won lessons (Ray gotchas, sculptor-specific quirks). Future agents
  should keep extending these.
