# Ablation study handoff (2026-08-08) — IN PROGRESS

Continuing agent: read this + `experiments/ablation/sculptor_fork.py`'s
docstring first. The study is NOT done. All code is committed + pushed
(commits `solver: split solve()...` and `ablation: flag-gated fork...`).

## Goal

Quantify each of SCULPTOR's methodological leaps over PAINTER by building
SCULPTOR up one feature at a time (Tom's framing: painter → monte-carlo →
+memory → +direction → +exploration = SCULPTOR), on the REAL solver, with
the pipeline objective **avg_lat_steady + gamma · avg_lat_under_single_popp_failures**
(LP re-assignment per failure; NO_ROUTE volume charged 30000ms), reported
vs one_per_peering (= "optimal", the 0 reference).

## Non-negotiable design rules (Tom-specified, hard-won)

1. **No side-implementations.** Rungs are env-flag overrides on a fork
   (`Ablation_Sparse_Advertisement_Solver`) of the production solver.
   `sparse_advertisements_v3.py` may only be modified by *visual*
   refactors (code motion into named sub-functions to create override
   seams). `solve()` is now an orchestrator over 9 named sub-steps;
   `_rescale_gradient` is extracted for the same reason.
2. **Assertions prove the flags bind, every iteration**: binary state for
   memory-off, ≤1-nonzero gradient AND ≤1-coordinate realized step
   (momentum included) for direction-off, measurement budget for
   exploration, byte-equal canonical init across rungs per seed
   (`init_dep<seed>.npy` in the out-dir). `[ablation-assert] SUMMARY`
   prints per run; violations raise.
3. **Never trust in-process scoring.** Scoring after ANY solver ran in
   the same process is contaminated (pattern-specific, sometimes 1000x;
   see chip "Investigate in-process eval contamination"). ALL reported
   numbers come from `rescore_fork.py`: fresh process per seed,
   RAY_ADDRESS=local (never attach to a running cluster), NO workers,
   driver-side `solve_lp_with_failure_catch` in a plain loop.
4. **One flip per iteration** is SCULPTOR's step-design intent. For
   single-coordinate rungs the stock rescale cap froze training entirely;
   the fork guarantees the selected viable coordinate crosses threshold
   each iteration. Memory-off also rounds init (memory-less = no
   continuous state, ever). Memory-on keeps continuous values around the
   same flip cadence (reversibility/probe-heat = what memory means).
5. **gamma is size-dependent and small**: gamma=4 (wrapper_eval default)
   saturates GRAD_CLIP and freezes training via the damp branch. Current
   choice: **gamma=0.1** (SCULPTOR_ABLATION_GAMMA) for small; Tom
   suggested ~0.01-0.1 scale. Resilience benefit ON for all SCULPTOR
   rungs (uniform objective); painter/opp are objective-blind baselines.
6. Run dirs are semantic + self-contained:
   `runs/ablation-<dpsize>-<rung>-dep<seed>/` with solve.log, convergence
   figure (make_plots is wired in), state pickles. Stray/numbered dirs
   auto-deleted; newest-20 GC. Logs also under `logs/fork_*`.

## Rungs (cumulative; RUNGS dict in sculptor_fork.py)

painter (repo, no fork) → no_memory ("monte-carlo": binary, single-coord,
no explore) → no_direction (+memory: continuous, single-coord) →
expl_none (+direction: full vector) → expl_random (+random explore) →
full (+entropic explore = SCULPTOR). The **monte-carlo OFF flag is NOT
BUILT YET** (would connect painter↔no_memory): intercept
path_distribution_computer sampling, replace MC path with avg-of-options
pseudo-path, huge caps — do via visual refactor + fork override.

## Results so far (all trusted rescores)

- **small, latency-only, 30 seeds x 200 iters** (archived in
  `ablation_study/small_no_resilience/`, PRE-dates the one-flip fixes for
  single-coordinate rungs — its no_memory/no_direction rungs used old
  frozen semantics): painter 5.10 → MC 4.35 → +mem 3.04 → +dir 1.02 →
  explore ≈1.0 (medians, steady vs opp).
- **small, gamma=0.1, 5 seeds x 200 iters, CURRENT semantics**
  (`cache/ablation/fork_5x200/`): medians (combined) painter +338,
  all SCULPTOR rungs +0.9–1.3. THE STORY IS THE TAILS: painter
  catastrophic 3/5; single-coordinate rungs (no_memory, no_direction)
  each blew up on exactly 1 seed (+131, +23208 — genuine training
  outcomes, not bugs); direction-bearing rungs never catastrophically
  failed (worst +2.7; full had one mild +39). Direction = robustness.
- Known repo bugs found (chips filed): NaN gradients under long-budget
  training (fork guards + counts as nan_grad_iters; roots in 0/0 in
  failure-scenario LPs); in-process eval contamination (may affect
  compare_different_solutions historicals); advertisement collapse crash.

## Running right now (2026-08-08)

- LOCAL: `decent` generalization, **1 seed** x 200 iters x 6 rungs,
  gamma=0.1 (`logs/fork_decent_local_chain.sh` → sweep+rescore; results
  to `cache/ablation/fork_decent/`). Tom wants to see if the
  direction-robustness pattern holds at ~13x coordinate count.
- VM (head i-0428c395787bc3ca0, c7g.16xlarge @ 100.54.8.15, 32 workers):
  actual-10 RESTARTED with corrected code (old actual-10 data deleted —
  implementation errors), 5 seeds x 200 x 6 rungs, gamma=0.1
  (`run_a10_v2_chain.sh` on head → `cache/ablation/fork_a10_v2/`).
  `logs/a10v2_finalize_local.sh` on the Mac auto-pulls, then STOPS the
  head and downsizes it to m7g.4xlarge. ALWAYS tear down when done.
  Update ~/.sculptor_cluster_alert/active_cluster.json on lifecycle
  events.

## How to run things

- Sweep: `SCULPTOR_ABLATION_GAMMA=0.1 SEEDS="..." MAX_ITER=... DPSIZE=...
  OUT_DIR=... PORT_BASE=... RUNGS="..." ./experiments/ablation/run_fork_ladder_sweep.sh`
  (sequential on purpose: Gurobi WLS sessions; unique ports).
- Rescore (ALWAYS before reading numbers):
  `python -m experiments.ablation.rescore_fork --in-dir <dir> [--dpsize d] --all`
- Tables: combined = diff_vs_opp + gamma*(fail_popp.avg_lat_under_failure_abs
  − fail_popp.opp_avg_lat_under_failure_abs) from the JSONs.
- Figures: plot_fork_ladder.py (4 panels, --gamma display choice),
  plot_normalized.py (% of painter→optimal gap closed; Tom likes this).
- Smoke before anything big: 1 seed x 10-30 iters; verify
  `[ablation-assert] SUMMARY` and measurement counts per rung.

## Open items / next steps

1. Collect + table the local `decent` run and the VM actual-10 v2 run
   (medians + per-seed, combined at gamma=0.1 — Tom asks "table" often;
   keep them coming unprompted at milestones).
2. Build the **monte-carlo OFF** rung (see above) to complete the ladder
   bottom link.
3. Tails story needs more seeds at small (5 → 30) if Tom wants
   significance on blowup rates; ~12-16h local.
4. gamma sensitivity (0.01 vs 0.1 vs 1) is uncharacterized beyond smokes.
5. Directory reorganization prompt exists (Tom has it) — repo-wide
  experiments/ restructure + DEFAULTS_AUDIT; don't start it while runs
  are live.
6. Failure-term presentation: optionally split into %-volume-congested +
   routed-latency (paper style) alongside the combined scalar.

## Gotchas that will bite you

- Don't edit imported-by-live-runs files mid-sweep; don't delete
  `runs/17*` while a sweep is mid-run (active dirs are numbered until
  renamed at run end). Don't run local Gurobi-heavy things concurrently
  with the VM's 32 sessions.
- `iters` in results is max_iter+2 (known off-by-2). opp_fail ≈ opp
  steady always (structurally failure-proof). Fresh-process rescoring has
  a ~0.2ms LP-degeneracy noise floor. Same-seed reruns differ (probe RNG
  isn't pinned by SCULPTOR_DEPLOYMENT_SEED) — never compare single runs.
- The Mac disk and the head disk both filled once (runs/ checkpoints).
  GC exists now, but watch `df` on long sweeps.
