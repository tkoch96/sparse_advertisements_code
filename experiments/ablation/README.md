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
  With `SCULPTOR_ABLATION_FIXED_BUDGET=1` (budgeted-fixed, Tom
  2026-08-14): measure every iteration until the N budget is spent,
  then keep training on beliefs (no measurements) to the normal
  convergence criterion / horizon. This is the L1 arm's semantics —
  the old exit-on-budget form (L1B, stopped ~at init) is retired.
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
  georand collapse mechanism). AS OF 2026-08-14 LATE this exit exists
  ONLY in budgeted-fixed (L1): gated/scheduled/smart NEVER budget-exit —
  exhaustion stops MEASURING and training runs to the normal criterion/
  horizon. The queue audit still ACCEPTS `budget_exhausted` for
  backward compatibility, but only L1 can produce it.

THE CURRENT LADDER is v3 (Tom 2026-08-16, WHAT/WHEN decomposition;
see HANDOFF.md for the full table): probing is PURE GROUNDING
(SCULPTOR_ABLATION_PROBE_TARGET=current — probe measures the CURRENT
adv) on the fixed schedule through L5 (L1 nomc+budgeted-fixed → L2
nomc+sched → L3 nomem+sched → L4 nodir+sched → L5 full+sched); the
smart deltas are isolated on top: L6 = +smart WHAT (max-info probe
targeting, PROBE_TARGET=maxinfo, same fixed schedule) and L7 = +smart
WHEN (the gate, min-gap-guarded via
SCULPTOR_ABLATION_SMART_MINGAP_FRAC=0.7 toward the schedule). Dirs are
literal: policy_ladder_v3/L<k>_<desc>. Old L7 (Bernoulli gradient
base, SCULPTOR_ABLATION_GRAD_BASE) is PAUSED. v2 headline (pre-fix
semantics, old ladder): L6 beat L4 by ~8 units at N=1-2 — superseded
by the v3 rerun. POST-FIX SEMANTICS everywhere as of 2026-08-16:
lambduh==0 (all files) and SCULPTOR_SIG_CUTOFF=p5 (percentile-scaled
remeasure significance; the absolute .01 starved persistence on
quantized objectives — the fracb L6 freeze).

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
- `run_n_sweep_queue.py` — THE harness. `--manifest specs.json` mode
  (2026-08-15): many cell-group specs in ONE global slot pool (per-spec
  env/gamma/probe-mode/dpsize/out_root), global --launch-stagger,
  inline artifact harvest, RAM governor (MemGovernor, config
  ~/queue_governor.json live-tunable, target 90% RAM, hard cap = the
  RAM/cores, NOT Gurobi sessions -- see the WLS policy note below). Single-spec CLI unchanged: global
  cell queue, mandatory pre-seeded inits, audit gate, built-in
  per-seed rescore. Multi-arm
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

## Result & artifact formats (2026-08-14 era; for future agents)

**Run result JSON** (`seed_<s>_<rung>.json`), key fields: `adv` (final
continuous advertisement matrix), `n_iters`, `max_iter`, `probe_mode`,
`probes_spent`, `exit_reason` (null = full horizon | `budget_exhausted`
| `remeasure_triggered` -- both early exits are LEGITIMATE and audit-
accepted), `solve_error` (null = clean), `rescored` (true after
trusted rescoring), `avg_lat`/`opp_avg_lat`/`diff_vs_opp` (rescore),
`fail_popp`/`fail_pop` (rescore aggregates incl. 30s sentinel).

**Out-root layouts**: classic ladder `.../<arm>/N<k>/seed_<s>_<rung>.json`
(canonical init `init_dep<s>.npy` alongside); hardB3 objective grids
`.../<obj>/{fixed,sched,smart}/N<k>/seed_<s>_<rung>.json`.

**Artifacts harvest convention** (chain-end AND safe mid-run: cp fig,
gzip log, rm run dir -- COMPLETED/renamed `ablation-*` dirs only,
never timestamp-named in-flight dirs):
- figures -> `<artifacts>/figs/[<wslabel>_]<rung>-dep<seed>-N<n>-<pmode>.pdf`
  (fixed mode: `..._<rung>-dep<seed>-fixed.pdf`; painter runs emit NO
  convergence figure)
- logs -> `<artifacts>/logs/[<wslabel>_]N<n>_s<seed>_<rung>.log.gz`
The dashboard (experiments/dashboard/) consumes exactly these names --
keep them stable or update its `conv_grid` patterns.


## Gurobi WLS policy note (2026-08-16, Tom-ratified)

The long-standing "~28 concurrent Gurobi session budget" was a myth
propagated across agent handoffs. The facts:

- Official academic-WLS baseline is **2 concurrent sessions**
  (https://support.gurobi.com/hc/en-us/articles/34672988479633-What-are-the-restrictions-on-using-an-academic-WLS-license),
  with a "Too many sessions" error only after EXTENDED overage and
  ~5-minute token aging
  (https://support.gurobi.com/hc/en-us/articles/34567582787345-How-do-I-resolve-the-error-Too-many-sessions).
- Empirically, this project's license sustains 20-48+ concurrent
  sessions for multi-hour campaigns with zero license errors;
  oversubscription manifests as optimize-request WAITING, not account
  cuts or hard time-outs.
- Therefore session count is NOT a pool-sizing constraint. Size worker
  pools to RAM (target 90% utilization; ~/queue_governor.json is
  live-tunable) and cores.

**Update 2026-08-20: HiGHS is now the DEFAULT backend** (gpshim falls
back to highs, not gurobi, when SCULPTOR_LP_BACKEND is unset). The
single-box facts above still hold, but Gurobi showed a real scaling
limit on multi-node fleets: sessions from several machines sustained
above the WLS baseline were license-killed after ~30 min ("Overage for
too long", 2026-08-20 eods32 fleet — sparse strategy died mid-training;
see experiments/eods/HANDOFF_EODS25.md). Standard campaigns run highs
and touch Gurobi zero times; gurobi is opt-in for the quadratic
objectives only.


## LB-cache A/B (2026-08-16, Tom-ratified: CACHE ON is the standard)

Full paired grid (7 arms x 6 N x 5 seeds, identical deployments/inits,
adagrad a0=1, stop-v2) run twice: `policy_ladder_v3` (SCULPTOR_LB_CACHE=0)
vs `policy_ladder_v3_LBCACHE` (=1). Result over 167 matched cells:

- quality (diff_vs_opp), paired ON-OFF: median +0.000, mean +0.114 ms;
  per-arm medians: L1/L2 bit-identical, L4 -0.09 (ON better), L5 +0.18,
  L6' +0.27, L7 +0.32 -- the probe-sensitive rungs pay a tiny,
  consistent-signed price, far below the +-0.7 single-trial noise floor.
- wall-clock: 3.1x faster per cell (480s vs 1501s mean), and cache-off
  additionally suffers unbounded late-run slowdowns (belief-support
  growth; observed minutes/iter stalls).

DECISION: default stays SCULPTOR_LB_CACHE=1 everywhere; =0 is reserved
for measurement-validity studies (fresh-MC-draw semantics), not for
production or ladder runs.
