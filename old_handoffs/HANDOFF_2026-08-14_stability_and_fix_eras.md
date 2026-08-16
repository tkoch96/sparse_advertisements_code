# HISTORY archived from ablation_study/HANDOFF.md on 2026-08-15
# (chronological era notes: stability stack, fix nights, objective bugs)

# Ablation study handoff (2026-08-14 evening) — STABILITY-STACK ERA

## Where we are (historical; superseded by the section above)

Tom's program continues: hard objectives + trusted policy ladder on the
real solver, debugged interactively today down to the gradient/gate
mechanics. Current single source of truth for live state:
`~/.sculptor_cluster_alert/active_cluster.json`.

### IN FLIGHT right now

- **Policy ladder v2 (trusted)** on the head — **c8g.24xlarge, 96 vCPU,
  IP 32.197.14.235** (upgraded today; ~$3.9/hr; restart changes the IP —
  update the alert JSON + `SCULPTOR_HEAD_IP`/`--host` for the dashboard
  refresher). Chain `~/chain_ladder_v2c.sh` under setsid (survives any
  session): smoke-gated, 6 arm queues, seeds 1-5, N{1,2,5,10,20,50},
  100 iters, gamma 0.1, georand, 27 slots, DEPLOYMENT-MAJOR cell order
  (complete per-deployment N-lines land first — Tom's explicit
  priority). Launched 17:07Z 2026-08-14; ~2-2.5h; prints
  `LADDERV2 COMPLETE` to `logs/ladder_v2_driver.log` and self-harvests
  figures/logs to `cache/ablation/policy_ladder_v2_artifacts`. Queue
  audit + trusted rescore built in. Out: `cache/ablation/policy_ladder_v2`.
- **Dashboard refresh loop** on the MAC (survives session end): nohup
  pid recorded in `/tmp/dashboard_refresh.log` era — `pgrep -f
  "experiments.dashboard.refresh"`; runs
  `python -m experiments.dashboard.refresh --loop 180 --heavy-every 4`.
  Kill/restart with the same command. Serves localhost:8643 via the
  `hardb3-dash` launch.json entry (python http.server on
  `dashboard_site/`).

### The stability stack (all committed+pushed on claude/elated-blackburn-13c10e)

Today's debugging (interactive with Tom, single-cell local smokes on
the Mac — dep3/dep1 georand N10/N50 are the canonical repro cells,
~1-6 min each) produced, in causal order:
1. `477824f` running EWMA anchor for typical_high_uncertainty (set-once
   anchor let sentinel-scale explore ranges blow uncertainty_factor
   7.5→16386, blinding the gate) + per-iteration factor decay in the
   fork (gating had made stock's decay unreachable — deadlock).
2. `2179fe3` SOFT-BOUNDED believed objective: -(avg routed lat) -
   SCULPTOR_SOFT_CONG_PENALTY(default 50)×frac(congested+noroute),
   replacing sentinel-inside-the-average at all 3 congestion-aware
   sites (Tom's design: penalties, not HUGE penalties); + damp-branch
   repair in _rescale_gradient (large grads now damp to
   DESIRED_MAX_VAL=5 not 0.1 — the 0.1 froze deployments with
   legitimately-large flip-gradients, e.g. dep3 frozen at init in BOTH
   prior ladder eras). lats_by_ug keeps sentinel marks; eval/rescore
   unchanged (canonical 30000).
3. `<head of branch>` gate c-learning: warmup=1 U-sample, anneal tau
   1.5 (SCULPTOR_ABLATION_PROBE_ANNEAL_TAU). First U>c probe ~iter 8,
   criterion (a) dominant. tau=0.5 tested and REJECTED (front-loads
   probes, exhausts explore, early remeasure-stop). A snap-to-quantile
   variant (c = quantile(U-history) from the first sample, no anneal)
   smoked clean (first probe iter 7) but is NOT committed — candidate
   if Tom revisits.
Validation: dep3 N10 frozen +9.60 → +3.13 (full budget, all criteria);
dep1 control +1.87 (baseline preserved, zero damp hits).
Also shipped: probe-gate instrumentation panels in every gated run's
convergence figure (U vs c + probe markers, budget staircase,
uncertainty factor, chosen explore values) — bottom rows of the grid.

### Dataset trust table

- `policy_ladder_v2` (in flight): TRUSTED — full stack.
- `policy_ladder_fixed`: BAD-GRADS ERA (pre-stack; dep-dependent
  damp-freezes + gate blindness). Kept for the before/after story;
  eval outputs quarantined in `cache/model_error/BADGRADS_ERA/` and
  `PREFIX_ERA_congestion_bug/` (glob-contamination: NEVER leave
  superseded outputs under names the plot globs match — see
  experiments/model_error/FORMATS.md).
- `hardA_poprb`, `hardB3` (415/630, PAUSED), `hardC_maxhard`,
  `a10_policy`: all PRE-STACK. Same trust caveats apply in principle
  (esp. smart arms). Tom has NOT yet ruled on invalidating/rerunning
  these under the stack. hardB3 resume manifest: `~/resume_hardB3.sh`
  on the head (md5 ada5dde6...; do not resume without Tom's call on
  the mixed-era question).

### HB3V3 — the FINAL hard-objectives run (2026-08-15 ~02:35Z, Tom's calls)

Supersedes the 100-iter hardB3v2 chain (killed mid-prio; ALL 100-iter
data quarantined to PRESKIP_ERA_v2/iters100_era; Mac score store reset).
chain_hb3v3.sh (setsid): 200 ITERS (100 left visible room), 'mlu' =
PURE max_util, windowed-c gate DEPLOYED (fork md5 475b8463...,
U_WINDOW default TCONV/2=100), ALL THREE OBJECTIVES CONCURRENT (Tom:
"prioritize one deployment x all N x all objectives over one objective
at a time") — 9 queues (3 obj x fixed 1/sched 3/smart 4 slots = 24),
each deployment-major, staggered. Same data root hardB3v2/<obj>/<pdir>
so dash/scoring stay wired. Verdict HB3V3 COMPLETE in
logs/hb3v3_driver.log; ~10-13h ETA (~13:00-16:00Z). Monitor armed
(30 min). STANDING RULE REINFORCED TWICE TONIGHT: pkill patterns must
not share a remote command with mv/paths containing the unbracketed
name — killed the ssh shell twice.

### PURE MLU + windowed-c ENABLED (2026-08-15 ~02:20Z, Tom's calls)

1. U_WINDOW now DEFAULT ON (TCONV/2) in sculptor_fork — trailing-window
   quantile for c. NOT yet deployed to the head; hardB3v2 runs stay
   full-history-c for era consistency (mlu-pure chain sets U_WINDOW=0
   explicitly). Deploy fork at the next clean boundary.
2. 'mlu' now means PURE max link utilization: new objective 'max_util'
   in experiments/model_error/objectives.py (registered; 17 unit tests
   pass; lat_plus_max_util kept for legacy). score_hardb3 OBJ_OF['mlu']
   -> 'max_util'; plot/dash labels updated. chain_mlu_pure.sh on the
   head (setsid) SELF-WAITS for HARDB3V2 COMPLETE, quarantines the
   lat+mlu-era mlu block to PRESKIP_ERA_v2/mlu_latplus_era, reruns 180
   cells under max_util (dirs fixed/sched/smart preserved), prints
   MLUPURE COMPLETE to logs/mlupure_driver.log.
AGENT TODO at HARDB3V2 COMPLETE: purge stale 'hardB3v2/mlu/' keys from
cache/model_error/hardB3v2_scores.json on the Mac (path-keyed store
lingers over quarantined files), send interim email (policy ladder +
fracb/prio), arm mlupure monitor; FINAL email at MLUPURE COMPLETE, then
teardown decision (STOP instance, alert JSON).

### GATE V3 CANDIDATES (2026-08-15 ~02:15Z, designed with Tom, NOT enabled)

Census over final-gate smart runs: L6's probes are ~30% pure sched-
backstop ('s'=135 vs (a)=90) while L4 self-fires fine ((a)=290) — the
full-history U-quantile lags nonstationary U (drifts down as
measurements resolve beliefs, hardest on memory arms) so c stays high.
Candidates, all flag-gated DEFAULT OFF, ready to A/B:
1. SCULPTOR_ABLATION_U_WINDOW=<n> — trailing-window quantile for c
   (implemented in _abl_probe_current_c; suggest n=TCONV/2=50).
2. Decisive-flip / expected-improvement step (designed only, NOT
   implemented): enforce flip guarantee on the REALIZED step (post
   momentum/prox), land coordinates decisively past threshold; flip
   only when filtered delta confidently positive, else measure/stop.
   Motivation: paradigm A/B (seeds 1-3, TCONV-matched, scratchpad
   paradigm_* workspaces): auto-scale vs fixed alpha endpoint
   differences are within seed noise; big fixed alpha (0.2) doubles
   flips + halves no-op iters (~30% waste even at best) at no quality
   cost. Step-paradigm seam exists: SCULPTOR_ABLATION_GRAD_SCALE=
   auto|fixed + SCULPTOR_ABLATION_ALPHA (fork _rescale_gradient/
   set_alpha overrides).
NOTE smart probes: backstop only changes the TRIGGER — target is
always the rung's max-info (max-entropy adjacency for explore rungs,
current adv for explore=none rungs).

### QUEUED: hardB3 v2 — monotone ladder x hard objectives (2026-08-14 ~22:40Z)

Tom (stepping away): "queue the multi-objective variant with our new
fixes". chain_hardb3v2.sh on the head (setsid) SELF-WAITS for SMART3
COMPLETE then runs objectives SEQUENTIALLY (fracb -> mlu -> prio), each
= full monotone ladder (fixed:no_mc FIXED_BUDGET / sched:no_mc,no_memory
/ smart:no_memory,no_direction,full), N{1,2,5,10,20,50} x seeds 1-5,
gamma=0, 100 iters, 26 staggered slots, deployment-major. 540 cells
(~6-9h overnight). Out: cache/ablation/hardB3v2/<obj>/<pmode>/;
painter refs COPIED from old hardB3 (painter unaffected by solver
fixes; rescored under current code). Verdict: HARDB3V2 COMPLETE in
logs/hardb3v2_driver.log. Mac side ready: score_hardb3 --root/--store,
plot_hardb3 HARDB3_{STORE,ROOT_RE,FIG_PREFIX,ARMS} envs (legacy arm set
env-gated for the old dataset), dash tab 'Ablation: hard objectives v2'
(store hardB3v2_scores.json, figs figs_hb3v2 symlink) with always-render
plot step. ALSO: smart3 rerun semantics = budget stops MEASURING never
TRAINING (exit-on-budget is L1-only now); Q_AGGR back to 0; prior smart
data quarantined in PRESKIP_ERA_v2/exit_on_budget_era. OPEN (b) work
with Tom: flip-per-iteration guarantee is approximate (momentum/prox
break it; ~43% realized) — theory discussion delivered, min-progress
variant designed, NOT implemented. Also pending: results EMAIL to Tom
at SMART3 COMPLETE + final at HARDB3V2 COMPLETE (send_report.py
pattern, budgeter emailer creds); teardown decision after hardB3v2.

### LADDER RELABELED — monotone, one capability per arm (2026-08-14 ~21:45Z)

Tom's final ladder (display labels; DATA DIRS keep historical names):
  L1 no_mc+fixed   (budgeted-fixed: measure first N iters, EXIT)  dir L1_nomc_fixed
  L2 no_mc+sched   (+evenly-spaced measuring)                     dir L2_nomc_sched
  L3 no_mem+sched  (+congestion-aware belief LP — the "MC" rung;  dir L4_nomem_sched
                    at MC_NUM=1 the delta is belief realism)
  L4 no_mem+smart  (+combined-U smart gate)                       dir L5_nomem_smart
  L5 no_dir+smart  (+memory: continuous advs, single-coord steps) dir L5_nodir_smart (NEW ARM)
  L6 full+smart    (+direction AND explore targeting, bundled)    dir L6_full_smart
  off-ladder ref: no_dir+sched                                    dir L3_nodir_sched
The old ordering had no_direction (memory-bearing) at position 3 —
non-monotone. NEVER rename the data dirs mid-program; the label→dir map
lives in plot_policy5.ARMS + the dash intro. L5 arm launched ~21:37Z
(chain_l5p.sh, 30 cells, L5P COMPLETE marker in logs/l5p_driver.log).

### Objective scalar: FOURTH blind site fixed (2026-08-14 ~20:15Z)

Tom's dash sanity read ("L1 below one-per-peering on the training-
objective panel — impossible") exposed that the 08-13 congestion-aware
fix patched only the MLU FALLBACK paths: the PRIMARY min-latency LP
(and solve_generic_lp + the failure-catch's inner site) still returned
-model.objVal/obj_norm — congestion-blind. A 74%-congested collapsed
adv scored ~3.9ms vs the 14.4ms opp floor. Now ONE helper
(`_soft_bounded_objective` in solve_lp_assignment.py) is used by all
three sites (site-cost variant deliberately untouched — hardB3's own
objective definition). Validated: collapsed adv 3.92 -> 49.74, opp
unchanged 14.42. plot_policy5 now has a hard SANITY GATE: any arm cell
below the per-seed opp train-objective raises AssertionError
(SCULPTOR_SANITY_ASSERT=0 bypasses). Eval stores quarantined to
WRONGWORLD_STOCK_EVAL/blind_primary_lp/ and recomputed.
ALSO: L1 FINAL definition (Tom): measure every one of the first N
iterations then IMMEDIATELY EXIT (exit-on-budget, varies with N);
coast variant is EXIT_ON_BUDGET=0 only. L2 = evenly spaced over the
horizon (scheduled — already correct). Coast-era L1 runs quarantined
(PRESKIP_ERA_v2/L1_coast_era; NOTE: their N5/N20-seed2 cells genuinely
COLLAPSED on-host — belief drift after budget exhaustion, the exact
mechanism exit-on-budget was designed to prevent).

### Dashboard auto-update (2026-08-14 18:00Z): staleness-gated `steps`

refresh.py now supports declarative `steps` in a registry entry's
`refresh` spec: each step lists `in` globs + `out` paths (+argv,
optional env/every) and runs ONLY when an out is missing or inputs are
newer — make-style, chained data → eval store → figure. Tom's rule:
every figure a tab shows must be the pinned `out` of a step (env
`POLICY_PLOT_OUT` etc.), never a plot script's default filename — the
bare plot_policy5 call in the old spec kept rendering the BAD-GRADS
filename, so the v2 figure sat stale while the bad-grads figure got
silently overwritten with v2 data (caught + fixed; bad-grads figure
re-rendered from the BADGRADS_ERA quarantine, mean stat). v2 + hardB3
entries converted; `evals`/`heavy` are legacy. Loop relaunched with
logging: `nohup python -m experiments.dashboard.refresh --loop 180 >>
/private/tmp/dashboard_refresh.log 2>&1` (log lines only when steps
actually run). Contract documented in experiments/dashboard/README.md.

### Wrong-world eval incident (2026-08-14 18:15Z, caught by Tom)

The first policy_ladder_v2 eval stores were computed WITHOUT the world
env knobs (steady/failure_metrics build the eval world from env — no
knobs = STOCK world), so every arm showed ~12 ms clean latency, BELOW
the georand opp refs — impossible numbers. Training data unaffected
(head runs carry their own env). Wrong stores quarantined in
`cache/model_error/WRONGWORLD_STOCK_EVAL/`; recomputed under
`SCULPTOR_LAT_MODEL=geo SCULPTOR_PREF_MODEL=random` (the world pair,
same as score_hardb3 sets internally; LB_CACHE/MC_NUM are training-only
knobs). The env now lives IN the registry steps, so refresh always
passes it. RULE: an eval step's env must pin the world knobs; sanity
gate = per-seed arm means must sit ABOVE the per-seed opp ref.
Follow-up worth doing (NOT done — no source edits while eval children
run): have steady/failure_metrics record their world env in the store
and make plots refuse mismatched worlds.

### Dashboard system (cross-project abstraction — Tom's design rule)

`experiments/dashboard/`: generate.py (registry EXPERIMENTS: left pane
= experiments, middle tabs = sub-experiments; renderers
objective_ladder/static/ladder_links; conv-figure link grids on every
tab), refresh.py (THE ONLY refresh mechanism: registry-driven
harvest/pull-with---delete/evals/plots; add a `refresh` spec to a
registry entry, never write ad-hoc scripts — Tom was explicit),
score_hardb3.py, plot_hardb3.py, plot_policy5 in model_error. Page
self-reloads every 180s. README has format contracts.

### Relinquishment (this session's watchers)

- The session Monitor watching ladder-v2 heartbeats is STOPPED at
  handoff. NOTHING session-bound watches the head. Re-arm your own:
  grep `LADDERV2 COMPLETE` / `AUDIT FAILED` in
  `logs/ladder_v2_driver.log` + JSON count vs 155 + disk, every
  ~15-20 min.
- SURVIVES without any agent: (1) the head chain (setsid, self-
  completing, self-harvesting); (2) the Mac dashboard refresh loop
  (nohup); (3) the Mac cron notifiers (liveness 10-min + progress-SMS
  3h) — they text Tom regardless.
- Progress-report EMAILS to tomkoch123@gmail.com are a standing Tom
  request: scratchpad send_report.py pattern (budgeter emailer creds;
  scratchpad gets wiped — recreate from
  ~/Documents/budgeter/emailer.py import, EmailMessage + attachments).

### Next steps, in order

1. When `LADDERV2 COMPLETE`: verify 155/155 by file count (never trust
   rc/'failures: 0' alone), evals auto-flow via the refresh loop
   (steady/failure tags *_v2), render plot_policy5 (mean), EMAIL Tom
   figure+tables, compare vs BADGRADS era.
2. Ask Tom: hardB3 resume (and whether pre-stack smart cells get
   invalidated); same question for hardA/hardC datasets.
3. Teardown per standing policy when work pauses: pull everything,
   STOP the instance (i-0428c395787bc3ca0), update alert JSON. Note
   the resize: it restarts as c8g.24xlarge (~$3.9/hr) — downsize to
   c7g.16xlarge via modify-instance-attribute if the next phase
   doesn't need 96 cores.
4. Standing rules unchanged: smoke-first; md5-verified deploys; audit
   JSONs not exit codes; one-deployment-first result ordering; general
   bug fixes go to the general codebase, experiment-specific code
   stays in experiments/; bracket pkill patterns; never edit imported
   modules on a host with runs in flight (atomic mv staging).

---

# HISTORY: prior eras below (2026-08-13 and earlier) — do not act on

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

### ✅ COMPLETE (2026-08-13 20:42Z): minimal actual-10 — VM STOPPED

L2+L6 at N∈{1,20} + painter (seed 1, γ=2, 150 iters, smoke passed,
all rescored on head, data pulled to Mac cache/ablation/a10_policy).
Steady diff vs opp (ms): L2 +0.103/+0.037/+0.106 at N1/5/20;
L6 +0.713/+0.304/+0.295; painter +0.889 (WORST — every ladder arm
beats painter on the real deployment); prior N5: L3 +0.099 L4 +0.069
L5 +0.270. No remeasure-stops on actual-10 (all budget-exhausted).
Reports #1 (ladder) + #2 (a10) + #3 (teardown) EMAILED to
tomkoch123@gmail.com via scratchpad send_report.py (budgeter emailer
creds; Tom's standing request: email progress reports w/ figures+
tables). **HEAD i-0428c395787bc3ca0 STOPPED 20:58Z** per standing
teardown policy; EBS retained (a10 run dirs + measurement caches
preserved for host-dependent rescoring). Restart:
`aws ec2 start-instances --instance-ids i-0428c395787bc3ca0`.

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
