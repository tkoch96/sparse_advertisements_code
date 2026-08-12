# Ablation study handoff (2026-08-12) — POLICY-LADDER ERA

## ⚠ Current design & status (2026-08-12 evening; supersedes everything below)

The study pivoted to Tom's **EM-with-measurements** framing: solver rungs
× measurement-spending policies under a FIXED budget N with
**exit-on-budget** (training stops when the Nth measurement is spent).
Full semantics in `experiments/ablation/README.md`; findings log in
`experiments/model_error/DIMENSIONS.md`. Old expl_* rungs RETIRED.

Ladder (georand, 100 iters, LB_CACHE=0, MC_NUM=1, γ=0.1):
L1 no_mc+fixed → L2 no_mc+scheduled → L3 no_direction+sched →
L4 no_memory+sched → L5 no_memory+smart → L6 full+smart.

Sparse verdict (N∈{5,20}, all audits clean, dataset
`cache/ablation/policy_ladder/`): **L6 full+smart N=20 = 5/5 healthy,
median +16, exits ~iter 62** — the only fully reliable MC arm; L3 dead
under any policy; partial measurement destabilizes no_mc (L2). Smart-gate
criteria attribution works ((b) stale+plateau dominates at N=20, (c)
prediction-mismatch at N=5). Dense N∈{1,2,10,50} fill + budgeted-fixed
L1B arm (auto-queued) were IN FLIGHT at handoff-write time; painter
references come from `mesh_georand{,_v2}` (no new runs needed).
Artifacts (every run's convergence figure + gzipped log):
`cache/ablation/policy_ladder_artifacts/`.

Everything below this section is the 2026-08-11 record (mesh era) —
still-valid data inventory + gotchas, but pre-pivot design.

# ——— 2026-08-11 record — IN PROGRESS (historical) ———

Read this + `experiments/ablation/README.md` (mechanics/agent notes) first.
Supersedes the 2026-08-08 handoff. Worktree branch
`claude/elated-blackburn-13c10e` holds the authoritative code (main repo +
VM are synced copies — ALWAYS md5-verify after deploying anywhere).

## ⚠ Live state right now (updated 2026-08-11 ~18:45Z)

- **VM head i-0428c395787bc3ca0 (c7g.16xlarge, 100.54.8.15) is RUNNING,
  IDLE, and has NO auto-teardown** — all queued work finished and was
  pulled to the Mac; STOP DECISION PENDING WITH TOM. Update
  `~/.sculptor_cluster_alert/active_cluster.json` on lifecycle events.
- Nothing in flight. The 2026-08-11 program (below) is COMPLETE:
  mini extremes (stock), model-uncertainty probes, full N-sweep (killed
  by Tom at 437/700 after the pivot — partial UNRESCORED data in
  `cache/ablation/nsweep_full_PARTIAL_UNRESCORED`), and the harder-world
  extremes waves (`nsweep_v2_{georand,maxhard,georand_nocache}`).
- The model-uncertainty workstream lives in `experiments/model_error/`
  (READ ITS `DIMENSIONS.md` — the variable-space card + all measured
  results). New env knobs in deployment_setup
  (`SCULPTOR_LAT_MODEL/GEO_NOISE/PREF_MODEL/LAT_SPREAD/ROUTE_VIOLATION`)
  and path_distribution_computer (`SCULPTOR_MC_NUM`, `SCULPTOR_LB_CACHE`).
- Tom's standing rules: **test code before using it** (smoke first, every
  time); only validated experiments may occupy the machine; hourly status
  reports during long runs; no multi-seed sweeps without his go.

## 2026-08-11 headline results (details in model_error/DIMENSIONS.md)

1. **The stock 'small' world made information worthless** — 3 latency
   tiers + providers-everywhere = model ≈ measurement (mini extremes:
   no_memory ≥ full everywhere). Structural, knob-resistant.
2. **In realistic worlds the result inverts categorically** (georand =
   geodesic×1.3±30-50ms + random prefs; maxhard adds noise×2, vol
   spread, zero cap slack): no_memory's final advs STRAND ~2/3 OF
   TRAFFIC IN STEADY STATE in 20/20 runs; full survives ~half its runs
   with real solutions (georand N1 median: 30.4ms steady, 5% congested
   under median popp failure) and converts probes into wins (paired
   3/5→5/5 as N 1→50 georand; 18× median at maxhard N50). Machinery
   gates information value; hardness gates machinery value.
3. **lb-cache discovery**: benefit(A)+pdf memoized per thresholded adv,
   cleared ONLY on measurement → under gated probing beliefs freeze;
   this was ~all of the model's overconfidence (calibration 0.40→0.99
   stock with SCULPTOR_LB_CACHE=0). BUT cache-off does NOT improve
   outcomes (georand wave 3: full blowups 2→4/5) — frozen beliefs act
   as a gradient stabilizer. Honesty ≠ performance; supports the
   objective-worsening trigger over belief-side fixes.
4. **Hard objectives reveal what avg_lat hides** (20-seed re-rank):
   MLU separates memory/direction rungs (0.238 vs 0.305/0.327);
   popp-failure congestion: painter 18%, ALL ladder rungs 0.000 (and
   degenerate at stock caps). New registerable objectives
   ('frac_beyond_optimal', 'lat_plus_max_util',
   'popp_failure_congestion') in experiments/model_error/objectives.py
   (+17 unit tests), generic-LP contract, register() wires them in.

## The experiment program (Tom's design)

Two threads:
1. **Ladder ablation** (painter → no_mc → no_memory → no_direction →
   expl_none → expl_random → full): quantify each SCULPTOR feature on the
   REAL solver. Fixed-mode 20×200 study DONE (see Results).
2. **Measurement-budget (N) study**: gated probing = measure-XOR-step
   under a TOTAL budget N (stock SCULPTOR measured every step). U =
   composed LB+RB sign-error probability; auto-learned threshold c
   (quantile-targeted, annealed, refractory-doubling with tau scaled to
   intended probe spacing). Hypothesis: small N separates the rungs
   (information efficiency matters), large N collapses the gap.

## Results so far (ALL numbers from rescore_fork — never in-process)

- **Fixed-mode 20×200 clean-v3** (`cache/ablation/fork_small_20x200_v3`,
  rescored, = canonical init source + N=∞ anchor): medians — all rungs
  close ~most of painter's gap (painter median combined +784; rungs
  ~1.1–1.6; no_mc ~5 = bottom link verified). THE TAILS carry the story:
  catastrophic blowups out of 20 — no_direction 5, no_memory 3, no_mc 1,
  full 1, expl_none 0, expl_random 0. Plus one rescored fixed-mode
  replica (`cache/ablation/fixedmode_replica1`, n=14): pooled rates ≈
  no_memory 21%, no_direction 18%, direction-bearing 0–3%. Blowups are
  mostly STOCHASTIC (seed sets differ across replicas; seeds 1 and 12
  look high-risk). Interpretation Tom liked: **monte-carlo/LP modeling
  buys the mean; memory and direction buy the tail; memory WITHOUT
  direction is a noise integrator** (removing the remeasure carryover
  exposed this — see below).
- **Remeasure clearing** (direction-off rungs forget cross-iteration
  gradient info): raised no_direction's blowup rate vs the old 5×200
  study — the carryover had been an accidental stabilizer. Old-semantics
  datasets are labeled in README.
- **In-process eval contamination CONFIRMED** (multi-seed shared-worker
  scoring turned a +21k blowup into +0.99). All scoring must be fresh
  per-seed processes; eval_ladder_metrics has per-seed isolation + a
  canary vs rescore. The repo's own evaluate_all_metrics shares the
  hazard for paper historicals — open investigation for Tom.
- **Gamma bisection** (seed-1, 50-iter smokes): γ=0.1/0.2 stable,
  γ≥0.3 collapses (RB:LB gradient-scale pathology; NOT fixable by the
  NO_ROUTE penalty knob alone — γ=1+penalty1000 goes failure-blind
  instead). `SCULPTOR_NO_ROUTE_LATENCY` env-gates the sentinel for
  training only. Studies run γ=0.1.
- **Gated probing so far**: ratchet bug (permanent c-doubling) capped all
  arms at ~1–2 probes → first N-sweep was invalid (retired). Refractory
  decay fixed it (smoke: N=5 → 4 probes spent, +0.52). Fixed tau=10 then
  capped spending at ~iters/10 (N=50 spent 10, N=100 spent 8) → tau now
  scales with intended spacing (deployed, validated only in the mini test
  in flight). Sleeper finding: ~1–4 measurements ≈ fixed-mode median
  quality at small (fixed mode spends ~125/run) — a big
  measurement-efficiency claim if it survives the validated rerun.
  KNOWN GAP (Tom aware, not yet approved to build): U cannot detect
  confident divergence — a run that diverges keeps low U (precise, wrong
  beliefs). Proposed: objective-worsening trigger (probe/flag when
  believed objective degrades k iters running). Multiple blowups would
  have been caught by it.

## nsweep_mini verdict (2026-08-11, run clean: 20/20, audit 0 bad, rescored)

Hypothesis NOT supported at the extremes. Combined(g4) medians (lower
better), 5 seeds/cell: N=1 — full +8.93 (1 blowup/5), no_memory +5.05
(0 blowups); N=50 — full +5.68 (1 blowup/5), no_memory +3.77 (0
blowups); fixed-mode anchor (seeds 1–5 of 20x200_v3) — full +5.40,
no_memory +5.17. full does NOT beat no_memory at N=1, and no_memory is
nominally better at both N (n=5, medians within seed-noise range —
treat direction, not magnitude, as the signal). full blew up 2/10 cells
under gating (vs 1/20 fixed-mode) — consistent with U-gate blindness to
confident divergence (the known gap). Sleeper finding VALIDATED on the
fixed gate: 1 measurement ≈ fixed-mode median quality (fixed spends
~125/run). Gate spend differs by rung: no_memory spent all 50/50, full
spent 14–27/50. Per the decision rule this is the "redesign with Tom"
branch — full N grid NOT launched. Data: `cache/ablation/nsweep_mini`
(head + Mac).

## Dataset inventory (cache/ablation/)

TRUSTED: `fork_small_20x200_v3` (fixed clean-v3, rescored) ·
`fixedmode_replica1` (rescored) · `fork_5x200` (old semantics, rescored) ·
`ladder_{small,a10}_eval_stats.pkl` (repo-metrics via eval_ladder_metrics).
RETIRED/QUARANTINED: `nsweep` (ratchet gate + killed cells, AUDIT FAILED)
· `nsweep_STALE_fixedmode_replicas` on head (stale-code deploy; N2..N20
subdirs are UNRESCORED fixed-mode replicas — useful for more blowup-rate
n) · `nsweep_v2_UNVALIDATED_GATE` on head (pre-tau-fix gate).
IN FLIGHT: `nsweep_mini` (the extremes test).
`fork_a10_v2` (actual-10 seeds 1–4, OLD semantics, killed at 25/30):
usable but predates remeasure clearing; actual-N rescoring must run ON
THE TRAINING HOST (deployment builds are measurement-cache-dependent).

## Code map (experiments/ablation/)

`sculptor_fork.py` — all flags (see its docstring; MEMORY / DIRECTION /
EXPLORE / MC / PROBE_MODE gated|fixed / PROBE_N / AUTO_C / TCONV / FRAC /
MULT_TAU auto-scaled), per-iteration binding assertions, probe gate.
`mc_off_worker.py` + ACTOR_CLS seam — no_mc rung. `run_fork_ladder.py` —
one cell; GC now CWD-scoped (was repo-absolute: caused cross-workspace
checkpoint deletion twice). `run_n_sweep_queue.py` — THE harness for
grids: global cell queue, no straggler tail, mandatory pre-seeded inits,
audit gate incl. stale-code guard (probe_mode must match), built-in
rescore. SIZING: slots×workers = Gurobi sessions ≤ ~28; memory ≈
4GB/slot — 28 slots OOM'd a 123GB box alongside other work; 20 is safe.
`run_n_sweep.sh` — legacy lane harness (straggler-prone; superseded).
`rescore_fork.py` (+SCULPTOR_RESCORE_STORE_SCENARIOS) · `table_fork.py`
(incl. painter-anchored quantile table) · `cdf_fork.py` · `plot_n_sweep.py`
· `eval_ladder_metrics.py` · `test_mc_off_unit.py`.

## Operational gotchas (each cost us real time)

1. **Deploy = scp + md5-verify + banner-check.** A stale deploy burned 7h
   producing fixed-mode replicas labeled as an N-sweep. The audit's
   probe_mode guard now catches it — keep that pattern for new flags.
2. **Remote pkill self-match**: bracket patterns (`fork_ladde[r]`) and
   never write the unbracketed name ANYWHERE in the same remote command
   (including echo/env text). Multiple ssh sessions killed themselves.
3. **Never run run_fork_ladder with cwd=repo while anything else runs**
   — use a workspace dir (runs/logs/figures + cache/data symlinks).
4. **RAY_ADDRESS=local + unique RAY_TMPDIR for every side-job** or it
   attaches to a neighbor's Ray and dies with it.
5. Lane/queue exit codes lie ("0 failures" while runs died) —
   ALWAYS audit JSONs: solve_error absent, n_iters ≥ MAX_ITER+1 (the
   off-by-2 is sometimes off-by-1 — don't flag 201), probe_mode correct.
6. Same-seed single trials are noise (probe RNG unpinned); blowups are
   probabilistic — compare distributions/rates, never single runs.
7. Orchestrator scripts: verdict-line-gated waits (process checks
   self-match); detach with setsid/start_new_session (harness-child
   nohup gets reaped).

## Next steps (in order)

1. Collect `nsweep_mini` verdict — NOTE: the previous session's watchers
   are ALL STOPPED (handed off clean); arm your own. Check
   `logs/nsweep_mini_chain.log` on the head for ALL DONE / AUDIT FAILED,
   then table no_memory vs full at N=1 vs N=50. If the
   extremes trend is interesting → full N grid via run_n_sweep_queue
   (20 slots, all 7 rungs, N grid per Tom, ~4–6h); else redesign with
   Tom (he floated 'scheduled' probing as an alternative arm).
2. Whatever runs next: smoke first (Tom's rule), md5-verified deploy.
3. Pending Tom decisions: divergence trigger (recommended, small);
   rescore remaining fixed-mode replicas for tighter blowup rates
   (cheap); a10 clean-semantics rerun; repo-pipeline contamination
   investigation; gamma/soft-penalty workstream (needs objective
   redesign, not penalty knob).
4. When work pauses: STOP THE INSTANCE (manual!), update alert JSON,
   pull any unpulled results first (rsync dirs listed above).
