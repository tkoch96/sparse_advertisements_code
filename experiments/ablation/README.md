# Ablation study: PAINTER → SCULPTOR, and measurement policies (agent notes)

Feature-by-feature ablation of SCULPTOR's methodological leaps, run as
env-flag overrides on a fork of the REAL production solver, now extended
with a **measurement-policy axis** (how/when the solver spends real
measurements). Companion docs: `ablation_study/HANDOFF.md` (live status)
and `experiments/model_error/DIMENSIONS.md` (the variable-space card +
dated findings log — read it first for what has been measured).

Tom's non-negotiables:

1. **No side-implementations.** Everything = flags on
   `sculptor_fork.Ablation_Sparse_Advertisement_Solver`. Production files
   change only by *visual* refactors (verbatim code motion creating
   override seams). Seams: `solve()` sub-steps, `_rescale_gradient`,
   `_assemble_lb_gradients`, `_assemble_rb_{popp,pop}_gradients`,
   `get_ingress_probabilities_and_sim` split, `_path_obj_coeffs`,
   `worker_comms_ray.ACTOR_CLS`.
2. **Assertions prove every flag binds, every iteration** —
   `[ablation-assert] SUMMARY` prints per run; violations raise.
3. **Never trust in-process scoring.** All reported numbers come from
   `rescore_fork.py` (fresh process per seed, RAY_ADDRESS=local,
   driver-side LPs). In-process eval contamination is CONFIRMED.
4. **Canonical per-seed inits**: first run writes `init_dep<seed>.npy`;
   every other run asserts byte-equality (SCULPTOR_ABLATION_INIT_FILE).
5. **Max cores, all the time**: multi-arm studies run every arm
   concurrently with a global slot allocation (~24 one-worker runners on
   the 64-core/123GB head; 28 OOM'd once). Never sequential small waves.
6. **Smoke first, every time**: 2-iter full-shape dry run of any chain
   into a throwaway out-root before committing compute.

## Solver rungs (env flags, cumulative; RUNGS in sculptor_fork.py)

painter → **no_mc** (deterministic avg-of-options pseudo-path, huge caps;
`Abl_MC_Off_Worker` via ACTOR_CLS; SCULPTOR_ABLATION_MC=0) → no_memory
(binary adv, one-flip) → no_direction (+continuous memory; clears
cross-iteration remeasure state) → full (= SCULPTOR). The old expl_none /
expl_random rungs are RETIRED (2026-08-12): traditional exploration is
off the table under budgeted probing; probe-target selection is part of
'full'.

## Measurement policies (SCULPTOR_ABLATION_PROBE_MODE)

- `fixed` — stock semantics: measure every new advertisement (N=∞).
- `gated` — original U>c gate (kept for reproducing 2026-08-10..11 data).
- `scheduled` — unconditional probe every ~TCONV/N iterations.
- `smart` — probe when ANY of (Tom's EM-with-measurements design,
  2026-08-12; all criteria believed-side only, ~free):
  (a) U > c (annealed as in gated);
  (b) no measurement in ≥ STALE_FRAC·TCONV/N iters AND the believed
      objective plateaued (rolling |Δ| below PLATEAU_EPS × belief range);
  (c) sign-disagreement rate of predicted-vs-realized believed deltas
      (first-order g·Δa vs realized) ≥ SIGN_RATE over SIGN_W steps;
  plus (d) surprise-adaptive threshold: a probe that moves the belief by
  > SURPRISE_REL × range halves the c-multiplier, otherwise it doubles.
  Every probe logs `[probe-gate] ... reasons=a|b|c`; per-run counts land
  in the result JSON (`probe_reasons`).
- **Exit-on-budget** (SCULPTOR_ABLATION_EXIT_ON_BUDGET, default ON for
  budgeted modes): training STOPS when the Nth measurement is spent — an
  agent that can no longer update beliefs walks into nonsense land (the
  georand collapse mechanism). Result JSON records
  `exit_reason='budget_exhausted'`; the queue audit accepts these.

The current ladder (Tom's progression, 2026-08-12): L1 no_mc+fixed →
L2 no_mc+scheduled → L3 no_direction+sched → L4 no_memory+sched →
L5 no_memory+smart → L6 full+smart. Verdict at N∈{5,20} (georand,
100 iters, LB_CACHE=0, MC_NUM=1, γ=0.1): **L6 full+smart N=20 is 5/5
healthy (median +16, exits ~iter 62)** — the only fully reliable MC
configuration; L3 is dead under any policy; partial measurement
destabilizes no_mc (L2) relative to both none and every-iteration.

## Estimator-config knobs (interact with everything; see DIMENSIONS.md)

`SCULPTOR_MC_NUM` (worker MC draws, default 5; the fork's gate sigmas
follow it — fixed 2026-08-12, earlier gated runs under-probed at MC=1) ·
`SCULPTOR_LB_CACHE` (0 = fresh MC per belief call; the memoized default
freezes beliefs between measurements — calibration poison, but also a
gradient stabilizer; cache-off did NOT improve outcomes) · world knobs
(`SCULPTOR_LAT_MODEL=geo`, `SCULPTOR_PREF_MODEL=random`,
`SCULPTOR_GEO_NOISE`, `SCULPTOR_VOL_SPREAD`, `SCULPTOR_SCALE_FACTOR`)
live in deployment_setup.py.

## Scripts (current, post-cleanup 2026-08-12)

Run:
- `run_fork_ladder.py` — one (seed, rung) run; semantic run dirs encode
  rung/seed/N/policy for artifact harvesting.
- `run_n_sweep_queue.py` — THE harness: global cell queue, mandatory
  pre-seeded inits, audit gate (accepts budget-exhausted exits;
  probe_mode stale-code guard), built-in per-seed rescore. Multi-arm
  chains live on the head as chain_*.sh (see HANDOFF); pattern: one
  queue invocation per arm, all arms backgrounded concurrently, per-arm
  out-roots (JSON names don't encode policy!), harvest
  convergence_over_iterations.pdf + gzipped logs per arm, delete run
  dirs to keep disk flat.

Score/analyze (trusted, driver-side LP):
- `rescore_fork.py` — THE trusted scorer (combined = steady + 4×failure
  excess vs one-per-peering; 30 000 ms no-route sentinel).
- `policy_table.py` — the policy-ladder table (combined, healthy counts,
  exit iters, probe reasons, + LP sidecars when present).
- `experiments/model_error/{rerank_ladder,steady_metrics,plot_mesh}.py` —
  interpretable metrics (failure congestion, steady congestion, clean
  routed latency) and the three-panel over-N figures.
- `table_fork.py`, `cdf_fork.py` (+`plot_normalized.py`) — fixed-mode-era
  tables/CDFs; still valid for the 2026-08-08..11 datasets.
- `eval_ladder_metrics.py` — repo-metrics path (per-seed subprocess
  isolation + rescore canary; do not weaken).
- `mc_off_worker.py`, `test_mc_off_unit.py` — no_mc worker + units.

Deleted 2026-08-12 (superseded; in git history): the standalone
scipy-arms stack (arms*, belief*, common, estimators, painter_fast,
run_ablation*, run_sparse_ref, plot_ablation, plot_fork_ladder),
run_n_sweep.sh, run_fork_ladder_sweep.sh, plot_n_sweep.py, vm_runbook.sh.

## Datasets (cache/ablation/)

- `policy_ladder/<arm>/N*/` — CURRENT: the policy-ladder study
  (+ `policy_ladder_artifacts/` = every run's convergence figure + log).
- `mesh_georand{,_v2}/` — the gated dense mesh + full replicate
  (+ `mesh_georand_v2_artifacts/`).
- `nsweep_v2_{georand,maxhard,georand_nocache}/` — the harder-world
  extremes waves; `nsweep_v2_inits_georand/` = canonical georand inits.
- `nsweep_mini/` — stock-world extremes (the original inversion result).
- `fork_small_20x200_v3/`, `fixedmode_replica1/`, `fork_5x200/`,
  `ladder_*_eval_stats.pkl` — fixed-mode era (still trusted).
- Retired/quarantined: `nsweep/` (ratchet bug), head-side
  `nsweep_STALE_fixedmode_replicas`, `nsweep_v2_UNVALIDATED_GATE`,
  `nsweep_full_PARTIAL_UNRESCORED` (sweep killed at 437/700).

## Operational gotchas (each cost real time)

1. Deploy = scp + md5-verify both sides + banner check.
2. **pkill self-match**: bracket patterns (`run_fork_ladde[r]`), never
   write the unbracketed name anywhere in the same remote command — and
   NEVER combine pkill with a heredoc containing the target names (kill
   and stage in separate ssh sessions; bitten twice on 2026-08-12).
3. Workspaces: never run run_fork_ladder with cwd=repo; per-arm ws-roots
   + distinct port ranges for concurrent queues.
4. RAY_ADDRESS=local + unique RAY_TMPDIR for every side-job.
5. Audit JSONs, never exit codes; accept n_iters < max only with
   exit_reason='budget_exhausted'.
6. Same-seed single trials are noise; MC-arm outcomes are ~Bernoulli
   survival draws — compare rates/paired counts across replicates, never
   single runs (the v1-mesh 'dose-response' that vanished in v2).
7. Detached remote jobs: setsid nohup + redirect; local launcher ssh
   channels linger and later die with 255 — harmless, ignore.
8. Head disk is 49GB: harvest+delete run dirs per arm,
   SCULPTOR_ABLATION_RUNS_KEEP small when not harvesting, watch df in
   every heartbeat (disk-full = silent rc=1 + truncated logs).
