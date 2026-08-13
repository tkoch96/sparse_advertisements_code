# Ablation study handoff (2026-08-13) — OBJECTIVE-BUG-FIX ERA

## For the incoming agent: where we are and why

**The high-level goal is unchanged**: Tom's EM-with-measurements program —
quantify, on the real solver, how much each SCULPTOR feature (solver
rungs) and each measurement-spending policy (fixed / scheduled / smart,
under a fixed budget N with exit-on-budget) buys, across worlds of
varying hardness (stock synthetic → georand → actual-10), producing
paper-grade ablation figures. We are still on that path. The current
phase is **methodology refinement and bug-weeding**: the headline
experiments exist and run cleanly end-to-end; what changed is our
understanding of what some of their numbers meant.

### The big discovery (2026-08-13): the training objective was
### congestion-blind, at TWO layers

When capacity is exceeded, the LP falls back to an MLU formulation whose
returned scalar priced over-capacity volume at its REAL (small) latency —
while evaluation charges it NO_ROUTE_LATENCY. Shedding users IMPROVED
the number the solver optimizes. Verified concretely: georand full+smart
N=50 seed 1 stranded 74% of volume while its objective read −19.61
("better" than the 20.98 ms one-per-peering reference) and evaluation
read 22,387 ms. **Every georand "collapse" in the ladder/mesh datasets
is this mis-specified objective being optimized successfully, not solver
divergence.** Stock-small and actual-10 results are unaffected (nothing
congests there).

Two fixes, both env-gated `SCULPTOR_CONGESTION_AWARE_OBJ` (default ON;
=0 reproduces pre-fix behavior), pricing congested volume exactly like
no-route volume (Tom's design: reuse the NO_ROUTE_LATENCY sentinel,
which training already scales down via `SCULPTOR_NO_ROUTE_LATENCY` —
documented choice 1000 ms for training, canonical 30000 for eval):
1. `solve_lp_assignment.py` MLU fallbacks (driver scalar; commit
   9b2a3d8, pushed). **A/B showed this alone does NOT change behavior**
   — it fixes metrics/stop-tracking, but gradients come from the worker
   belief path. All three A/B-v1 arms still collapsed; the fixed arms'
   GT curves now show the collapse honestly (a cliff at iter ~70) while
   the believed objective declines smoothly through it.
2. `path_distribution_computer.solve_generic_lp_persistent` post-solve
   block (worker beliefs/gradients + corrected link-level congestion
   accounting; deployed to head+worktree, md5 9941f6ad...; COMMITTED as
   9635e01 on claude/elated-blackburn-13c10e, pushed).

### ✅ RESOLVED (2026-08-13 afternoon): A/B v2 (fixab2) — FIX VALIDATED

`chain_fixab2.sh` ran clean 12:03–12:15Z (georand seed 1, full+smart,
N=50, 100 iters, same init; runs are fast — ~11 min — because gated
probing only measures ~30×). Both surviving arms audit clean
(n_iters=102, solve_error absent, probe_mode=smart). Rescored verdict
(`cache/ablation/fixab2/`, pulled to Mac):
- **legacy** (fix off): steady **+22,179.6**, combined(g4) +107,631.7 —
  collapsed; believed repo_objective 19.59 (claims better than the
  20.98 ms reference) vs true avg_lat 22,200 ms. Beliefs off by 3
  orders of magnitude.
- **fixed30k** (fix on, sentinel 30000 = defaults): steady **+1.717**,
  combined −235.8 — healthy; believed 22.7022 vs true 22.7014 —
  **beliefs match ground truth to 4 significant figures**. Probe
  reasons shifted to (a)-dominated (20/27).
- **fixed1k** (fix on, sentinel 1000): **dies deterministically** — NOT
  a bug in fix #2. The 1000 ms sentinel collapses believed-LB
  distributions to point masses (prob 1.0 at one value), every explore
  candidate's value falls to the −1e6 sentinel, the max-information
  step re-picks an already-measured adv, and STOCK code's re-measure
  guard at `sparse_advertisements_v3.py:2081-2089` prints 'woops' and
  calls **`exit(0)`** — hence rc=0, no traceback, run dir never
  renamed. Conclusion: the documented "1000 for training" sentinel
  choice is dead with the congestion-aware objective; **use the 30000
  default** (empirically stable here despite the old gradient-scale
  concern; n=1, the ladder rerun is the real test).

Caveat: fixab2 is n=1 per arm (same-seed single trials are noisy), but
the belief-calibration evidence (19.6-believed/22,200-true → 22.70/22.70)
is mechanism-level, not a noisy comparison.

### ✅ COMPLETE (2026-08-13 16:25Z): full ladder rerun — collapses GONE

155/155 runs, all audited + rescored. Steady stranding = 0.000 for
every arm/N except L2 no_mc N5/N10 (0.13/0.17 — the real no_mc
fragility). Clean routed lat 21.4–25.9 ms (opp 20.98); simpler arms sit
closer to opp. pfail ~0.026–0.06 all arms (painter 0.093) except L2's
N5/N10 spikes. Exit census: 132 budget_exhausted, 12
remeasure_triggered (ALL L6; at N50 stopped after 2–7 probes — the
measurement-efficiency sleeper now has a mechanism), 11 full-horizon.
NEW SEMANTICS shipped mid-rerun (Tom's design): REMEASURE-STOP —
explore re-selecting a measured adv = graceful training stop
([REMEASURE-STOP] banner, exit_reason='remeasure_triggered'), replacing
stock exit(0) which silently killed 14/30 first-pass L6 cells (audit
now accepts the new exit_reason; commits 816a0aa + audit patch, all
pushed). Evals: steady_metrics + rerank_ladder pumped on the Mac in a
clean plotws (scratchpad), figure=plot_policy (mean + median) emailed
to Tom (report #1) 16:45Z. Data: cache/ablation/policy_ladder_fixed
(+_artifacts) on head AND Mac.

### ⚠ IN FLIGHT: minimal actual-10 (Tom's spec, 2026-08-13)

`~/chain_a10min.sh` on head (md5 ba87c49a...): L2+L6 at N∈{1,20} +
painter ref, seed 1, γ=2, 150 iters, smoke-first, rescore ON HEAD.
Driver log logs/a10min_driver.log → "A10MIN COMPLETE"; out
cache/ablation/a10_policy/{L2_nomc_sched,L6_full_smart}/N{1,20} +
painter/N5. Progress reports EMAILED to tomkoch123@gmail.com via
scratchpad send_report.py (reuses budgeter emailer creds; Tom's
standing request 2026-08-13: email reports with figures/numbers/tables).

### (superseded) original rerun plan

Tom (2026-08-13 ~12:35Z): "we basically need to rerun all the
experiments" — the policy-ladder-over-N figure is the one to fix. Note
`mc_off_worker` conditionally delegates to the real
`solve_generic_lp_persistent`, so even no_mc arms touch the fixed path.
Two chains run CONCURRENT on the head (26 slots total ≤ ~28 Gurobi
budget), both smoked (2-iter) first, both with pure defaults (fix ON,
sentinel 30000), same inits (`nsweep_v2_inits_georand`), georand,
100 iters, γ=0.1, out `cache/ablation/policy_ladder_fixed/` +
`_artifacts/`:
- `~/chain_fixladder.sh` (md5 8d690277...): L3–L6 ×
  N∈{1,2,5,10,20,50} × seeds 1–5 = 120 runs, 19 slots; logs
  `logs/fixladder_driver.log` / `fixladder_chain.log`; prints
  "LADDER COMPLETE".
- `~/chain_fixladder2.sh` (md5 f61674fb...): L1 no_mc+fixed at N=1
  (5 runs, as chain_policy3 did) + L2 no_mc+sched over the dense grid
  (30 runs), 7 slots; logs `logs/fixladder2_driver.log` /
  `fixladder2_chain.log`; prints "NOMC COMPLETE".
Target: 155 result JSONs. Compare against pre-fix
`cache/ablation/policy_ladder/` — this re-bases the paper's core
figure (regenerate via experiments/ablation/policy_table.py +
experiments/model_error/plot_policy.py). actual-10: NOTHING was
running on the VM when Tom asked to kill it (phases 1+2 completed
earlier; 5 JSONs safe on head AND pulled to Mac
`cache/ablation/a10_policy/`).

### Everything else in flight / recently landed

- Policy ladder + dense N (georand, 310 runs + artifacts incl. every
  convergence figure): `cache/ablation/policy_ladder/`,
  `policy_ladder_artifacts/`. Findings (pre-fix lens!): L6 full+smart
  peaked 5/5 healthy at N=20; N=50 failures are now explained by the
  objective bug + exit-on-budget never firing (gate spends ≤~40 of 50).
  L1B (budgeted-fixed no_mc) is DEGENERATE (N iterations ≈ init) — drop
  from figures. L3 no_direction dead everywhere. Painter: verified it
  uses only 4 measurements (budget-fair already); budget cap available
  (`SCULPTOR_ABLATION_PAINTER_BUDGET`), measurement count now recorded.
- actual-10 (seed 1, N=5, γ=2, phases 1+2 done, rescored on head):
  ALL five arms within 0.3 ms of one-per-peering (45.8 ms world) — real
  deployment is forgiving; no collapses. `cache/ablation/a10_policy/`.
  Next: more seeds / N per Tom's priority order (budget-conscious).
- Analysis/plots: `experiments/ablation/policy_table.py`,
  `experiments/model_error/{plot_policy,plot_mesh,rerank_ladder,
  steady_metrics}.py` (mean/median switchable via env). Figures in
  `figures/`. `experiments/model_error/DIMENSIONS.md` = findings log.
- Notifier: `~/.sculptor_cluster_alert/heartbeat.py` rewritten to text
  Tom EXPERIMENT PROGRESS (via budgeter emailer SMS) every 3h; liveness
  cron unchanged.
- VM: c7g.16xlarge head (100.54.8.15) up; disk ~10G free (tight — clean
  before big runs); manual stop policy unchanged; ~$100-115 spent over
  the program so far.

### Watchers: all session-bound monitors RELINQUISHED

This session's watchers/waiters (Monitor tasks + until-loop ssh waiters
for the mesh, policy ladder, dense pass, a10 phases, and fixab2) were
session-bound and have been explicitly stopped — **nothing is watching
the head from the old session**. What DOES survive: (1) the head-side
chains themselves run under setsid and finish unattended (fixab2 will
print `FIXAB COMPLETE` to `logs/fixab2_driver.log` and rescore on its
own); (2) the SYSTEM-level cron notifiers on the Mac — liveness_check
(10 min) and the progress-SMS heartbeat (3 h) — keep texting Tom
regardless of any agent session. Incoming agent: re-arm your own
watcher for whatever you run (pattern: scratchpad watch script + a
persistent Monitor, heartbeats 15-20 min with disk in every beat), and
check `logs/fixab2_driver.log` first thing — the A/B likely finished
unattended.

Tom's standing rules (all still in force): smoke-first (2-iter dry runs
of full chain shape); max cores always (concurrent arms, ~24 runners);
trusted rescoring only; md5-verified deploys; NEVER edit imported
modules while runs are in flight (violated once — got away with it);
pkill+heredoc never in the same ssh; audit JSONs not exit codes.

## ——— 2026-08-12 record (policy-ladder era) ———

## ⚠ Current design & status (2026-08-12 evening; superseded above)

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
