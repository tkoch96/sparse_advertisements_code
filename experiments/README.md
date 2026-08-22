# experiments/

Campaign drivers and investigations. **Nothing in mainline imports anything
here** — that is the rule this directory is kept to. If code under `core/`,
`evaluations/`, `helpers/` or `dashboard/` needs something, it gets promoted
out (see "Promotion rule" below).

Reorganised 2026-08-21: 18 directories → 6. Twelve modules were promoted into
`core/`, `dashboard/` and `unit_tests/`; concluded programs were folded into
`old_handoffs/`. See the "Where things went" table at the bottom if you are
looking for something that used to be here.

---

## Live campaigns

### `ablation/` — the feature-ablation ladder machinery
The drivers your ablation campaigns actually run. Imported by 5 modules and
named in `CLUSTER_RUNBOOK.md`.

| file | role |
|---|---|
| `run_n_sweep_queue.py` | **Work-queue N-sweep driver.** Near-maximum core efficiency; the queue that shepherds cells across slots. This is the one that harvests + `rmtree`s run dirs on `rc == 0`. |
| `run_fork_ladder.py` | Per-`(seed, rung)` driver: builds the fork, runs it, harvests. |
| `sculptor_fork.py` | Ablation fork of the real solver — subclasses `Sparse_Advertisement_Solver`. |
| `mc_off_worker.py` | Monte-carlo OFF worker for the `no_mc` rung. |
| `rescore_fork.py` | Authoritative re-scoring of fork results. Deliberately boring. |
| `eval_ladder_metrics.py` | Scores ladder advertisements through the repo's full eval pipeline. |
| `policy_table.py`, `table_fork.py`, `plot_normalized.py`, `cdf_fork.py`, `compare_paradigms.py` | Analysis + figures. |
| `test_mc_off_unit.py` | No-Gurobi checks for the `no_mc` plumbing. |

### `eods/` — the EODS campaign family (25 **and** 32)
Not just one run: `run_eods_cell.py` is the shared cell driver for the whole
family. It speaks the queue's `run_fork_ladder` CLI and result-JSON convention
so the queue/governor/dash/fleet run it unchanged.

| file | role |
|---|---|
| `run_eods_cell.py` | One EODS cell. **Holds the hot-start gate** (2026-08-21) that refuses to run against an unusable checkpoint dir rather than exiting 0 and letting the queue delete it. |
| `merge_eods.py` | Merges per-cell pickles into the classic `metrics_by_dpsize` cache. |
| `build_manifest.py` | Emits queue manifests for the family. |
| `dash_harvest.py` | Head-side harvest, driven by `dashboard/refresh.py`'s `remote_harvest`. |
| `profile_startup.py`, `smoke_profile.py`, `profiler_plots.py`, `profiler_status.py` | Startup profiling — "which operations in startup cost what". |
| `HANDOFF_EODS32.md` | Canonical EODS-32 state. |

### `ablation_study/` — the concluded ladder program's **results**
Data, not code — this is why there are two ablation directories. 180 result
JSONs, the `small_no_resilience` figure set, and `HANDOFF.md`, which opens
**"ABLATION PROGRAM CONCLUDED (2026-08-19)"**. Its verdict: L6c (conservative
WHEN) wins the trained lat+resilience objective, L6-slotted is the least stable
arm, rmsprop > adagrad, probing frequency is a stability knob rather than an
accuracy one. Also holds `SCALE_500_PLAN.md`.

---

## Investigations

Named for what they investigated, so the name survives you forgetting the
context. All are finished except `worker_ram`.

### `worker_ram/` — **open**: worker RAM is still a live problem
`bench.py` measures, outside the iteration loop, what the worker actually
retains. `FINDINGS.md` has the attribution from a real actual-25 production
run at 80 workers:

- `parent_tracker` grows 93 → 354 MB, with string-tuple dict keys, **replicated
  identically in all 80 workers** (~28 GB/box)
- `calc_cache.lb` ~190 MB, of which **99% is keys**, not values

That is the starting point for anything further on worker memory.

### `deployment_sizes_full_timing_investigation/` — per-phase eval timing
Renamed from `benchmarks/`. Two drivers the README points you at, plus ten
`eval_phase_*.json` result captures (which are most of the 4.1 MB) and two
write-ups: `step1_sweep_results.md` (batched `getAttr`) and
`worker_perf_investigation.md` (inner-loop worker timing).

- `run_deployment_sweep.py` — sweeps dpsizes with per-size nsim, hot-start and
  env config. **Caveat: this forks `evaluations/evaluate_over_deployment_sizes.py`
  rather than calling it.** It reimplements `pull_results_new` against
  `evaluate_all_metrics`, adding three env knobs
  (`SCULPTOR_DEPLOYMENT_SWEEP_{NSIM,SIZES,TAG}`). Those knobs are worth porting
  onto the mainline function so this fork can go.
- `eval_phase_baseline.py` — per-phase timing + crash diagnostics.

### `freezing_prefix_assignments_investigation/` — `SCULPTOR_EVAL_VOLSCEN`
Renamed from `eval_opt/`. `ab_eval.py` freezes advertisement solutions and
recomputes the evals twice, legacy vs the volscen fast path, requiring LP
results to be **exactly identical**. The feature is in mainline but
**defaults to off**; this is the harness that would re-prove it.

---

## Promotion rule

Develop here; move it out when it becomes production. Concretely: **the moment
`core/`, `evaluations/`, `helpers/` or `dashboard/` needs to import it, it
belongs in that package, not here.** Before 2026-08-21 five modules had crossed
that line without moving, so `core/` was importing upward into an experiments
folder.

The same applies to tests. Two of the directories dissolved on 2026-08-21 were
carrying passing test batteries that sat outside `unit_tests/` and had no
`pytest` marker, so `pytest -m unit` never ran them: `solver_fork`'s gpshim
battery (6 tests, cross-checking the shim against an independently built
`scipy.linprog`) and `model_error`'s objectives battery (17). Salvaging and
marking them took the suite from **20 passing to 43**.

A third pair — `test_phase2_gate.py` and `gate_5pop.py`, the byte-exactness
gates for `core/fork_load.py` and `core/shard_loader.py` — also moved to
`unit_tests/`, but they are `__main__` scripts rather than pytest modules, so
`pytest` collects nothing from them. Run them by hand, or convert them.

**If an experiment produces a test worth keeping, move it to `unit_tests/` AND
give it a marker** — otherwise it is invisible and its loss would be silent.

## Where things went

| was | now |
|---|---|
| `depcache/shard_loader.py` | `core/shard_loader.py` |
| `depcache/convert_latencies.py` | `core/convert_latencies.py` |
| `depsetup_fork/fork_load.py` | `core/fork_load.py` |
| `model_error/objectives.py` | `core/hard_objectives.py` |
| `model_error/worlds.py` | `core/worlds.py` |
| `static_failure_eval.py` | `core/static_failure_eval.py` |
| `dashboard/` | `dashboard/` (top level) |
| `model_error/plot_policy5.py`, `failure_metrics.py`, `steady_metrics.py` | `dashboard/` |
| `depcache/plot_bench.py` | `dashboard/` |
| `fleet/` | `cluster/fleet/` |
| `solver_fork/test_gpshim_unit.py` | `unit_tests/` |
| `model_error/test_objectives_unit.py` | `unit_tests/` |
| `depcache/test_phase2_gate.py`, `depsetup_fork/gate_5pop.py` | `unit_tests/` |
| `depcache/` + `depsetup_fork/` READMEs | `old_handoffs/DEPLOYMENT_SPEEDUP_PROGRAM.md` |
| `solver_fork/README.md` | `old_handoffs/SOLVER_FORK_MIGRATION.md` |
| `model_error/DIMENSIONS.md`, `FORMATS.md` | `old_handoffs/MODEL_UNCERTAINTY_DIMENSIONS.md` |
| `desharding/NOTES.md` | `old_handoffs/UG_DESHARDING_SURVEY.md` |

Deleted outright: `startup_optimizations/` (a design doc for work never built),
`lp_hotloop/` (its replay script loaded a fixture filename that does not exist —
it could not run), `painter_lab/`, `pattern_cache/`,
`reducing_iteration_timing/`.

Two name collisions worth knowing, because both look like something they are
not:

- **`pattern_cache`** — the *directory* is gone, but `self.pattern_cache` in
  `core/path_distribution_computer.py` is a live runtime cache and is untouched.
- **desharding vs sharding** — `desharding/` was about **UG slices shipped to
  workers**, and its removal landed (`split_deployment_by_ug` is gone from
  `helpers/helpers.py`). **Latency/deployment sharding is a different, live
  thing**: `core/shard_loader.py` + `core/convert_latencies.py`, opt-in via
  `SCULPTOR_LAT_SHARDS`, backing `cache/lat_shards`.
