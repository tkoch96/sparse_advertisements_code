# Ablation ladder: PAINTER → SCULPTOR (agent notes)

Feature-by-feature ablation of SCULPTOR's methodological leaps over PAINTER,
run as env-flag overrides on a fork of the REAL production solver. Companion
doc: `ablation_study/HANDOFF.md` (older; this file supersedes it on
mechanics). Everything here obeys Tom's non-negotiables:

1. **No side-implementations.** Rungs = flags on
   `sculptor_fork.Ablation_Sparse_Advertisement_Solver`. Production files
   may only be changed by *visual* refactors (verbatim code motion creating
   override seams). Current seams: `solve()` orchestrator sub-steps,
   `_rescale_gradient`, `_assemble_lb_gradients`,
   `_assemble_rb_{popp,pop}_gradients`,
   `get_ingress_probabilities_and_sim` split
   (`_compute_scenario_options` / `_sample_scenario_realizations`),
   `_path_obj_coeffs`, `worker_comms_ray.ACTOR_CLS`.
2. **Assertions prove every flag binds, every iteration** —
   `[ablation-assert] SUMMARY` prints per run; violations raise.
3. **Never trust in-process scoring.** All reported numbers come from
   `rescore_fork.py` (fresh process per seed, RAY_ADDRESS=local, driver-side
   LPs). In-process eval contamination is CONFIRMED (2026-08-09: a +21k
   blowup adv scored +0.99 in a multi-seed shared-worker process); the
   repo's own `evaluate_all_metrics` sequential scoring shares this hazard.
4. **Canonical per-seed inits**: first run writes `init_dep<seed>.npy` to
   the out-dir; every other run asserts byte-equality. Copy init files
   between out-dirs to enforce cross-study deployment identity (the
   N-sweep does this against the fixed-mode study).

## Rungs (cumulative; RUNGS in sculptor_fork.py)

painter → **no_mc** (deterministic avg-of-options pseudo-path, huge caps;
`Abl_MC_Off_Worker` injected via ACTOR_CLS; SCULPTOR_ABLATION_MC=0)
→ no_memory (binary adv, one-flip) → no_direction (+continuous memory)
→ expl_none (+full-vector steps) → expl_random → full (= SCULPTOR).
Direction-off rungs also clear the samplers' cross-iteration remeasure
state every gradient call (no directional carryover; wide explore pool
untouched — painter-fair).

## Gated probing (measurement-budget experiments)

`SCULPTOR_ABLATION_PROBE_MODE=gated`: measure-XOR-step under a TOTAL
measurement budget `PROBE_N` (assert: path_measures growth ≤ N, every
iteration). U = g²-weighted sign-error probability over ALL probed
coordinates, LB+RB composed (independent probes: weighted raw sums,
squared-weight variance sums), raw deltas vs standard-error sigmas (never
compare heaviside-scaled g to raw sigma). Auto-c: anneal from PROBE_C
toward the (1 − N/(FRAC·TCONV)) quantile of U history; every fired probe
doubles c. Debug: `[probe-gate]` line per iteration.

## Gamma / penalty facts (small, seed-1 bisection 2026-08-10)

gamma 0.1/0.2 stable; 0.3/1.0 collapse (RB:LB gradient-scale pathology —
NOT fixable by penalty alone); penalty 1000 at gamma 1 "stabilizes" but is
failure-blind under canonical scoring. `SCULPTOR_NO_ROUTE_LATENCY`
env-gates the sentinel for TRAINING only — rescoring keeps canonical 30000.
Studies run gamma=0.1.

## Scripts

- `run_fork_ladder.py` — one (seed, rung) run + in-run (untrusted) scoring.
- `run_fork_ladder_sweep.sh` — sequential sweep (single machine).
- `run_n_sweep.sh` — parallel-lane N-budget sweep. Lanes MUST use isolated
  workspaces (cwd-relative runs/): concurrent run_fork_ladder invocations
  in one cwd delete each other's checkpoint dirs (105/140 runs lost once).
  Includes audit gate: accept n_iters ≥ MAX_ITER+1 (off-by-2 is sometimes
  off-by-1 on clean runs) and require solve_error absent — never trust
  lane exit codes ("0 failures" can hide swallowed solve errors).
- `rescore_fork.py` — THE trusted scorer. `--all` or `--seed`;
  `SCULPTOR_RESCORE_STORE_SCENARIOS=1` persists per-failure-scenario
  latencies for scenario CDFs.
- `table_fork.py` — tables: per-seed + median combined (gamma display
  choice), steady-only, painter-anchored per-seed-paired benefit with
  quantiles, %-of-painter→OPP-gap.
- `cdf_fork.py` — across-seed (and per-scenario) CDFs, abs + % views.
- `plot_n_sweep.py` — ladder-vs-N tables/figure + no_memory−full gap curve
  with fixed-mode anchor.
- `eval_ladder_metrics.py` — score ladder advs through the repo's FULL
  `evaluate_all_metrics` (all stats_* metrics). Per-seed subprocess
  isolation + canary vs rescore (do not weaken; see rule 3).
- `mc_off_worker.py`, `test_mc_off_unit.py` — no_mc worker + no-Gurobi units.

## Datasets (cache/ablation/)

- `fork_5x200/` — small 5×200, pre-remeasure-clearing semantics.
- `fork_small_20x200_v3/` — small 20×200 FIXED mode, clean-v3 semantics
  (remeasure clearing + no_mc). Canonical init source. = N=∞ anchor.
- `nsweep/N{1,2,5,10,20}/` — gated N-budget sweep, same 20 deployments.
- `fork_a10_v2/` — actual-10, seeds 1–4 complete (sweep killed at 25/30),
  OLD semantics (predates remeasure clearing).
- `ladder_{small,a10}_eval_stats.pkl` — repo-metrics (evaluate_all_metrics)
  stats via eval_ladder_metrics.

## Operational gotchas

- Remote pkill: patterns match the ssh session's own command line — always
  bracket (`run_fork_ladde[r]`) and never mention the unbracketed name
  elsewhere in the same command.
- Deployment builds are host-dependent for actual-N (measurement caches):
  score actual-N ONLY on the training host. small/decent are synthetic and
  cross-host identical (hash-verify when in doubt).
- `worker_comms_ray._ensure_ray` attaches to any running cluster
  ('auto') — always set RAY_ADDRESS=local + unique RAY_TMPDIR for
  side-jobs or they die with the neighbor's Ray.
