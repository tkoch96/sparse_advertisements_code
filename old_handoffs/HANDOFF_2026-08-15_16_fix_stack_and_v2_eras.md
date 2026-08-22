# Ablation study handoff (2026-08-15) — FINAL-SEMANTICS ERA

## READ THIS SECTION ONLY. Everything below the HISTORY divider is
## chronological context — consult when you need the why, not the what.

Single source of live state: `~/.sculptor_cluster_alert/active_cluster.json`
(head IP lives there; EVERYTHING — watchers, dashboard refresher — must
resolve the IP from it per-use, never cache it: the instance restarts
under load and changes IP).

### IN FLIGHT (2026-08-15 ~02:40Z)

- **NOTE (03:00Z): the mlu block runs under chain_mlu3.sh** (verdict
  MLU3 COMPLETE in logs/mlu3_driver.log) — the first pure-MLU
  implementation was degenerate (peak-util of a latency-optimal split
  ≡ 1.0 for every adv; caught on the dash). Fixed: max_util now uses
  the MLU-MINIMIZING LP (force_mlu kwarg on
  solve_generic_lp_with_failure_catch — skips primaries, runs the
  min-Y fallback; validated opp 2.81 < painter 2.87 < sparse 4.22).
  hb3v3's own end-of-chain audit will print AUDIT FAILED for the
  missing mlu cells — EXPECTED, ignore; completion = 540 by count
  (fracb/prio from hb3v3 + mlu from mlu3).
- **hb3v3** on the head (c8g.24xlarge, i-0428c395787bc3ca0, ~$3.9/hr):
  the FINAL hard-objectives grid. Monotone ladder x {fracb=
  frac_beyond_optimal, mlu=PURE max_util, prio=joint_latency_bulk_
  download}, 200 iters, gamma=0, windowed-c gate, THREE OBJECTIVES
  CONCURRENT (9 queues, 24 slots, staggered), deployment-major
  everywhere (Tom's standing rule: one deployment x all N x ALL
  OBJECTIVES first). Out: `cache/ablation/hardB3v2/<obj>/<pdir>/`
  (fixed/sched/smart). Verdict `HB3V3 COMPLETE` in
  `logs/hb3v3_driver.log`; target 540 JSONs; ETA ~13:00-16:00Z.
  Chain script archived at `tools/chain_hb3v3.sh`.
- **Mac dashboard refresh loop** (nohup, `pgrep -f
  dashboard.refresh`, log /private/tmp/dashboard_refresh.log)
  — THE one updater: pulls, evals, figures, site, all self-healing.
- **Session monitor** (30-min beats) watches the chain; the Mac cron
  notifiers (below) text Tom regardless of any agent.

### ON COMPLETION (agent TODO, in order)

1. Audit by COUNT (540) + per-queue rc lines; never trust exit codes.
2. Evals/scoring flow automatically; verify the hard-objectives tab.
3. EMAIL Tom results: `python cluster/send_report.py "<subject>"
   <body.txt> [figures...]` (Gmail app-password inside; attachments
   supported). Include per-objective tables + figures.
4. Teardown per standing policy: pull everything, STOP the instance
   (`aws ec2 stop-instances --instance-ids i-0428c395787bc3ca0`, aws
   CLI at ~/Documents/venv312/bin/aws), update the alert JSON.
   Instance restarts as c8g.24xlarge — downsize if the next phase
   doesn't need 96 cores.

### THE LADDER (final, Tom-ratified 2026-08-14 late)

Monotone — each arm adds ONE capability. DISPLAY label vs ON-DISK dir
(NEVER rename data dirs; the map lives in plot_policy5.ARMS +
generate.py registry 'arms'):

| label | adds | dir |
|---|---|---|
| L1 no_mc+fixed | budgeted-fixed: measure first N iters then EXIT (only arm allowed to exit on budget; varies with N via trained iters) | L1_nomc_fixed |
| L2 no_mc+sched | evenly-spaced measuring (period TCONV/N), normal end | L2_nomc_sched |
| L3 no_mem+sched | congestion-aware belief LP (the "MC" rung; at MC_NUM=1 the real delta is belief realism not ensembles) | L4_nomem_sched |
| L4 no_mem+smart | combined-U smart gate | L5_nomem_smart |
| L5 no_dir+smart | memory: continuous advs + thresholding, single-coordinate steps | L5_nodir_smart |
| L6 full+smart | direction (full-gradient steps) AND explore targeting (bundled — noted on dash) | L6_full_smart |

Off-ladder: L3_nodir_sched (kept in data, not displayed). Ladder
worlds/env via `experiments/model_error/worlds.py` (SINGLE source of
world knobs — georand = LAT_MODEL=geo PREF_MODEL=random).

GATE (smart) final semantics: U = U_sigma + 0.01*min(2, entropy_ratio);
sigmas from per-coordinate flip-delta variance, EWMA'd + refreshed under
MC_NUM_EXPLORE=5 every SIGMA_REFRESH iters; c = annealed quantile of a
TRAILING U window (U_WINDOW default TCONV/2 — full-history lagged
nonstationary U and starved criterion (a): L6 was ~30% backstop-fired);
criteria (a) U>c, (b) stale+plateau, (c) sign-mismatch, (s) sched
backstop at 1.25x spacing. Probe TARGET = rung's max-info (max-entropy
adjacency for explore rungs, current adv otherwise), skip-not-stop on
already-measured. Budget exhaustion stops MEASURING never TRAINING
(exit-on-budget is L1-only). Explore evaluates candidates under
MC_NUM_EXPLORE=5 (worker RPC set_mc_num, restored after).

### L7 STATUS (2026-08-15 ~03:15Z): IMPLEMENTED + smoke-passed.
### Flag `SCULPTOR_ABLATION_GRAD_BASE=threshold(default)|bernoulli` in
### sculptor_fork.py, FORK-ONLY (Tom: no production edits). Override (a)
### = fork latency_benefit override mapping queued probe advs through
### A=(u<a), one u per gradient call, shared by LB+RB-popp probes
### (production threshold_a at compression is identity on binary advs —
### that's the trick). Override (b) = fork heaviside_gradient override,
### raw after-before (covers all 3 assemblers). bernoulli+ALPHA_POP>0
### refuses at init (pop-RB sampler pre-thresholds in production).
### 2-iter smoke: 1920 probes through the seam, 960 raw-delta pairs,
### base_diffs 465 vs 466.2 expected — binding proven. Local 200-iter
### classic-objective run (seed 1) COMPLETE, rc=0, full horizon, no
### solve_error: 46687 raw pairs, base_diffs 21249 vs 21166 expected
### (+0.4%); coinflips annealed 70.8 -> 60.5. In-process avg_lat 11.03
### vs opp 9.74 (UNTRUSTED — rescore before comparing). NOTE it ran at
### gamma=4.0 (runner falls back to wrapper_eval's EVAL_GAMMA when
### SCULPTOR_ABLATION_GAMMA unset) and TRAINED FINE — the gamma=4
### freeze-at-small warning in run_fork_ladder.py:76 was derived under
### threshold semantics; possibly interesting L7 signal, possibly one
### seed. Set SCULPTOR_ABLATION_GAMMA explicitly for ladder cells.
### Local-Mac ws recipe for standalone run_fork_ladder = queue's
### (PYTHONPATH, short RAY_TMPDIR, figures/logs dirs, cache/data
### symlinks); see memory + convergence PDF sent to Tom 03:20Z.
### run_fork_ladder standalone gamma default now 0.1 (Tom-ratified).
### COMPARISON SMOKE (Tom-requested, ~04:00Z; seed 1, same deployment+
### init, N=10, gamma=0.1, TCONV=200, RESCORED lp_driver_v2; SINGLE
### SEED — rates not verdicts): L6 steady 10.79 (opp+0.67) / L3 10.95
### (+0.91) / L7 12.34 (+2.55). L7 fail_popp avg 776ms (!!) vs ~11 for
### L3/L6 — L7 ended SPARSEST (106 advs vs 121/123) and strands volume
### under a popp failure (no-route sentinel). GATE BEHAVIOR SPLIT: L7
### fired 9/10 on criterion (c) sign-mismatch with U 0.10-0.19, budget
### gone by iter 74; L6 fired 8/10 on (b) staleness with U 0.03-0.08.
### Honest-sigma prediction visible (U up ~3x under bernoulli); but
### suspected paradigm/hyperparam mismatch: raw deltas re-weight
### coordinates (no sigmoid distance-to-threshold suppression) and may
### need their own lambduh/step tuning + the pipage rounding endgame
### (final threshold@0.5 of a hedged fractional adv may be the
### fragility mechanism). Data: scratchpad l7cmp_ws/{L3,L6,L7}/out.
### NEXT: bias falsifier (planned first anyway), multi-seed before any
### verdict, and consider the rounding endgame before ladder cells.

### OVERNIGHT (Tom 2026-08-15 ~04:00Z "L1-L7 all objectives all
### deployments by morning"): chain_l7k3.sh ARMED on head (setsid,
### ~/chain_l7k3.sh, verdicts in logs/l7k3_driver.log, archived
### tools/chain_l7k3.sh): waits for hb3v3+mlu3 (540 + queue drain,
### Gurobi WLS session budget forbids concurrent grids / second box),
### then 2-iter smoke, then 120 cells: L7 (GRAD_BASE=bernoulli,
### GRAD_BASE_K=3 — flush-time K-expansion, every flip pair averaged
### over 3 drawn bases, ~3x LP cost; K=3+K=1 smokes passed with exact
### count arithmetic) x rung full x smart x seeds 1-5 x N{1,2,5,10,20,
### 50} x {classic g=.1 -> policy_ladder_v2_L7K3/L7K3_full_smart;
### fracb/mlu/prio g=0 XOBJS -> hardB3v2_L7K3/<obj>/smart}. Same BASE
### env + nsweep_v2_inits_georand inits as L1-L6 data. Deploys
### md5-verified (fork + run_fork_ladder + chain). Dashboard registry:
### L7K3 pull/harvest/scoring/eval steps wired (stores
### hardB3v2_L7K3_scores.json, policy_{steady,failure}_v2_l7k3);
### DISPLAY tabs not yet added — do when cells exist. NO teardown at
### 540 (audit+email only); teardown after L7K3 COMPLETE + pull.
### Timeline honest: hb3v2 (L1-L6 hard) done ~13-17Z; L7K3 starts
### then, ~5-9h => L7 fills through 2026-08-15 afternoon/evening Z.

### MLU STRANDING BUG + UNIFIED CHAIN (2026-08-15 ~12:30Z, Tom-directed)
### Pure max_util is GAMEABLE BY STRANDING: the min-Y fallback LP parks
### no-route volume on a 100000-cap pseudo-ingress (solve_lp_assignment
### caps concat), so sparse advs that strand users show peak util BELOW
### one-per-peering — impossible under honest MLU (opp's feasible set is
### a superset). 113/160 mlu cells beat opp in the scores store (seed 3
### opp 3.54 vs arms ~1.0). FIX (Tom: "reuse latency + alpha*MLU — a
### working solution"): mlu lane -> lat_plus_max_util EVERYWHERE
### (scorer OBJ_OF, dash intro, training specs). Pure-MLU dataset
### quarantined by the chain to cache/ablation/PUREMLU_STRANDING_ERA_mlu.
### BALANCER (Tom: "solution that doesn't have these problems"):
### run_n_sweep_queue --manifest = MANY cell-group specs, ONE global
### slot pool, per-spec env/gamma/probe-mode/out-root, built-in
### --launch-stagger, per-spec audit+rescore, inline per-cell
### convergence-fig harvest (artifacts_figs). CLI mode unchanged (one
### internal spec). Both modes smoke-passed locally.
### chain_unified.sh ARMED on head 12:16Z (~/chain_unified.sh, verdicts
### logs/unified_driver.log): waits fracb+prio=360 + queue drain ->
### quarantines mlu -> manifest smoke -> 300-cell dense manifest
### (~/uni_manifest.json): mlu-redo L1-L6 under lat_plus_max_util (180)
### + L7K3 grid (120, mlu spec lat_plus). Old l7k3 waiter left running
### ON PURPOSE — its gates make it a harmless post-hoc re-verifier.
### Supplemental reverse-order queues (hb3v2R2_ws_*) are eating the
### prio/fracb tails since ~12:05Z. mlu3's late AUDIT FAILED = expected
### (its queues killed / quarantined mid-flight).
### A10 CHAIN ARMED (Tom ~16:30Z, explicitly BEHIND unified):
### chain_a10.sh on head (setsid, verdicts logs/a10_driver.log,
### archived tools/): waits UNIFIED COMPLETE + drain -> init-prep (5x
### 2-iter actual-10 cells write canonical inits -> cache/ablation/
### a10_inits; doubles as data-pipeline canary) -> 140-cell manifest
### (~/a10_manifest.json): {classic g.1, fracb, mlu lat_plus, prio} x
### {L1 fixed / L2-3 sched / L4-6 smart / L7 bernK3 smartL7 dir} x
### seeds 1-5 x N=10, 200 iters, dpsize actual-10, STOCK world, out
### cache/ablation/a10v2/<obj>/<pdir>. RAM governor owns concurrency.
### Dash tabs for a10v2 + L7K3 + mlu-redo: wire when cells land.

### ZIPF KNOB (Tom's next-experiment idea; NEEDS REWORK — 2026-08-15):
### SCULPTOR_ZIPF implemented in deployment_setup (vol power-law +
### global-popularity Gumbel-PL prefs, dedicated RandomStates, z=0
### bit-identical; binding logs [zipf]). THREE local smokes (L3 + L6
### twins, z=0/1/2, seed 1): current formulation makes the problem
### EASIER at high z (L3 rel gap 8.1%->1.0%; L6-L3 gap within noise,
### z=1 even negative-for-L6) and opp latency climbs via MIX (top-10%
### UGs carry 97.6% vol at z=2) — NOT congestion (survived
### SCALE_FACTOR 1.3). Mechanisms: global favorites = MORE predictable
### ingress; concentration shrinks effective problem; anycast-derived
### caps absorb pressure. Candidate fixes (theory-backed, NOT yet
### implemented): per-UG-PRIVATE zipf prefs (concentrated but
### independent -> unpredictable) x tight caps (SCALE_FACTOR->1.0,
### the coupling axis). Tom: "maybe that idea needs more work" —
### PARKED. Data: scratchpad zipf_ws/.

### DASH SELF-CONSISTENCY PASS (overnight 2026-08-16 03:00-04:30Z,
### Tom's acceptance bar: no blanks, painter below nearly all,
### ABSOLUTELY nothing above opp, verified by LOOKING at the PNGs):
### - fracb scalar -> CAPABILITY metric: min hinge-excess-ms beyond
###   (opt+10ms) via solve_lp_assignment.solve_min_hinge_excess
###   (canonical Gurobi home; monotone => opp floor EXACT; validated on
###   the 25 formerly-crossing cells; 17/17 unit tests). Old
###   assignment-derived fraction kept as 'frac_beyond' component.
###   EVAL-side only (Mac) — NOT deployed to head: in-flight L7 fracb
###   cells keep training on the same objective as the L1-L6 data
###   (uniform train/eval gap = fair ladder). Env revert:
###   SCULPTOR_FRACB_SCALAR=assign.
### - prio: two-stage assignment-derived (stage-1 latency split not
###   prio-optimal); capability twin is a NONCONVEX QP (oversubscribe x
###   significance bilinear) -> cannot fix honestly overnight. Panel
###   annotated + tab note + temporary exemption in the sanity gate.
###   MORNING DECISION for Tom: (a) redefine prio to a jointly-linear
###   co-optimizable objective (exact floor), or (b) keep + documented
###   exemption.
### - sanity gate: dashboard/sanity.py
###   assert_not_better_than_opp wired into plot_hardb3 (popfail
###   globally exempt — legitimately beats opp; prio temporarily).
###   Violations now CRASH the figure step loudly.
### - score_hardb3: prunes store rows whose files vanished (era-mixing
###   fix) + persists ret components (max_util etc.) incl. opp
###   components. MLU tab primary figure = UTILIZATION units
###   (per-seed opp-normalized, 0 = exact floor); joint scalar panel
###   secondary. All mlu-lane figure sanity restored (stale
###   pure-MLU-era painter rows were poisoning the panel).
### - per-seed opp normalization EVERYWHERE (absolute axes bury good
###   seeds, opp spans 30x across seeds); blank panels now data-driven
###   (popfail dropped when dataless); L7 merged onto BOTH the
###   hard-objectives tables/figures (virtual pdir smartL7 +
###   HARDB3_EXTRA_STORE) AND the classic 6-panel (plot_policy5 ARMS +
###   L7K3_full_smart). Refresh loop reproduces all of it cleanly
###   (registry-driven; verified 0 errors).

### STANDALONE MLU LANE 'mlupure' (Tom 2026-08-16 ~11:00Z: "the
### entire point was to optimize solely MLU"): max_util REIMPLEMENTED
### in objectives.py as a first-class standalone objective (use like
### testing_priorities uses joint_latency_bulk_download):
### objective = -(A*minMLU + routed_lat + 3A*bad_frac), A =
### SCULPTOR_MLU_WEIGHT_MULT(10) x optimal floor => ~90% of weight on
### MLU, latency = tie-break only; routing-all is a hard LP constraint,
### advertisement-level stranding priced 3A (provably unprofitable,
### bounded, no 30k). Validated: opp floor holds, mlu_share 89.6%,
### 2-iter training smoke clean, 17/17 unit tests. v1 force_mlu
### quarantined -- never resurrect. Lane QUEUED: 4 specs appended to
### ~/a10_manifest.json (now 350 cells: a10 140 actual-10 + mlupure 210
### small-georand L1-L7), per-spec dpsize support added to
### run_n_sweep_queue (manifest mixes dpsizes in one governed pool).
### Out: cache/ablation/hardB3v2_PURE/mlupure/<pdir>. Scorer OBJ_OF
### 'mlupure'->max_util wired; dash tab pending data. FRACB TRAINING
### PDFs NOTE (Tom asked): fracb cells trained on the assignment-
### derived metric, whose opp line is legitimately crossable -- the
### trainer's signal was correct/self-consistent; only the dash's
### capability metric forbids crossing (expected training-vs-dash
### discrepancy on fracb convergence PDFs).

### FRACB L6 FREEZE ROOT-CAUSED + TWO TOM-ORDERED FIXES (2026-08-16
### ~12-14Z): fracb_smart_full-dep2-N50 froze (realized step ~0 from
### iter ~60, LB spikes continuing). NOT the prox (lambduh*alpha=1e-7),
### NOT alpha decay (constant .01), NOT NaN guard (0 fires), NOT probe
### starvation (52 probes, max-info scanning: best entropy ~0.045 —
### belief pinned everywhere). PROVEN CAUSE (A/B, one env var): the
### ABSOLUTE remeasure-significance cutoff |delta|<.01 — calibrated for
### ~20ms latency objectives — wipes persistence on the QUANTIZED
### fraction-scale fracb (range ~0.25): 50/100 iters with zero
### significant remeasures under 'abs' vs 0/100 under the fix; late-run
### flips 9->12, median late step 2x. prio (continuous scalar) shows NO
### freeze — fracb-specific, as Tom suspected. FIXES (both DEFAULT ON):
### (1) lambduh == 0 in EVERY file (call sites + constructor defaults;
### prox now identity; set_alpha branch unchanged -> alpha .01);
### (2) SCULPTOR_SIG_CUTOFF=p5 (default): cutoff = 5th percentile of
### prior-iteration |gradient| distribution, LB + RB-popp sites in
### sparse_advertisements_v3 ('abs' reverts). DEPLOY GATED on UNIFIED
### COMPLETE (waiter armed) so the L7K3 lane isn't split mid-lane;
### a10+mlupure train uniformly under the fixes. L7K3 fracb cells:
### 0/27 landed post-hinge-deploy (uncontaminated; re-check final 3 at
### audit). Quality effect of the fixes = multi-seed question; the
### fracb-lane comparison under new semantics is a candidate rerun.
### NOTE fracb training PDFs: trained on assignment metric whose opp
### line is crossable — trainer signal was correct (Tom's Q).

### MLU SEMANTICS FINAL (Tom-ratified ~13:15Z, deployed to head
### md5=4d6ed37): objective = -(routed_lat + P*bad_frac + alpha*(MLU +
### bad_frac)); MLU = _min_mlu_from_rti best-achievable min-Y LP over
### the adv's per-prefix ingress options (NOT the latency-assignment
### peak, which pins at 1.0). opp hard floor <= 1/1.1 = 0.909 by the
### anycast provisioning argument (Tom); measured 0.84-0.89 across
### seeds at default SCALE_FACTOR — IN the 85-95 target band, so NO
### world change (all lanes share georand deployments). Stranding
### priced ~(P+alpha)=65ms/unit, bounded, no 30k sentinel. Forensic:
### 17/17 objective units pass; seed-3 guilty arm now scores mlu
### 0.8904 > opp 0.8893 with opp winning overall. ALSO FOUND: min-Y
### FALLBACK LP drops most paths (60% vol spuriously no-route under
### opp!) — it is the failure-catch fallback, failure metrics suspect
### wherever primaries went infeasible; spawn-task filed, fix +
### fallback-vs-primary parity unit test needed. See memory
### project_mlu_objective_semantics.
### CORRECTION (~14:30Z, after empirical diff): get_paths_by_ug drops
### NOTHING (2043 paths, 0 pseudo, all 225 UGs under opp) — the
### "path-dropping fallback bug" is FALSIFIED, spawn-task dismissed.
### Actual mechanism of the pure-MLU era: force_mlu returned Y of the
### fallback LP whose objective is (1/ALPHA)*Y + sum(lat*v), ALPHA=.1
### -> latency-greedy; Y = CONCENTRATION of that split (opp
### concentrates hardest = 3.54, sparse forces spread = ~1.0). The
### sentinel marks were DESIGNED congestion pricing (2026-08-13 fix),
### so failure-catch semantics are as-intended and policy_ladder_v2
### classic L1-L6 was NEVER contaminated — no classic rerun needed
### beyond adding L7 (Tom's rerun order was based on the wrong
### narrative; corrected + re-scoped 14:35Z). solve_min_mlu (canonical
### Gurobi, deployed) is the correct MLU regardless.

### L7 PROPOSAL — Bernoulli-base gradient (Tom-directed 2026-08-15;
### HIGHEST-PRIORITY next experiment)

Theory session with Tom (2026-08-15): the canonical continuous form of
the problem is the MULTILINEAR EXTENSION F(a) = E[f(A)], A_ij ~
Bernoulli(a_ij) independent. Its exact partial is a flip-delta with the
OTHER coordinates SAMPLED from Bern(a): ∂F/∂a_ij = E[f(A,ij=1) −
f(A,ij=0)]. Stock SCULPTOR evaluates the same pair but conditions on
threshold_a(a) (mode point mass) + sigmoid weighting — a BIASED
estimator of the same quantity. L7 = L6 with the unbiased estimator:

L7 overrides the gradient computation in TWO distinct respects (both
under one flag `SCULPTOR_ABLATION_GRAD_BASE=threshold(default)|
bernoulli`; each gets its own binding assert):

- OVERRIDE (a) — MONTE-CARLO SIMULATE THE ADVERTISEMENT (the sampling
  seam): the base configuration the flip pairs condition on. Stock:
  `gradients_latency_benefit`'s a_effective = threshold_a(a), and
  compress_lb_args_queue re-thresholds base_adv before shipping. L7:
  draw ONE base A ~ Bern(a) per iteration and probe every flip pair
  against A (K=1 — SAME LP cost as stock; base+delta compression
  already shares one base per batch). The RB samplers
  (gradients_resilience_benefit_{popp,pop}) condition on the same
  advertisement — they must use the SAME drawn A (matters for
  gamma>0 runs).
- OVERRIDE (b) — NO SIGMOID (the assembly seam): in
  `_assemble_lb_gradients` (+ the RB assemblers), stock computes
  this_grad = heaviside_gradient(before, after, a[ind]) — the
  derivative of the surrogate before + (after−before)·σ(k(a−T)).
  Under Bernoulli semantics the interpolation in a_ij is LINEAR
  (multilinearity), so ∂F/∂a_ij = after − before, RAW. The sigmoid
  slope is not just unnecessary, it's WRONG for F: it re-weights by
  distance-to-threshold, which double-counts what the sampling
  already encodes (a_ij is the probability, not a position on a
  sigmoid).
- NOT overridden: `_rescale_gradient`'s one-flip amplification is a
  separate step-policy axis (existing GRAD_SCALE=auto|fixed seam) —
  keep stock/auto for L7 unless separately ablated.
- Endgame: pipage-style rounding replaces threshold@0.5 — round
  fractional coords one at a time in the sign of the estimated
  partial (F coordinatewise-linear ⇒ lossless per step); costs n
  probe pairs, once, at train end.
- Nice-to-have: common-random-numbers pairing (same route-realization
  draw for both arms of a flip pair) — the variance cut LB_CACHE
  provided by accident, without the calibration poison.
- Why (predictions): (i) unbiasedness makes the gate's sigmas honest —
  U's sign-error model Phi(-|d|/sigma) assumes zero bias, and a
  mode-conditioned consistent error reads as sigma≈0/U≈0 blindness;
  (ii) undecided coords hedge over each other → carries the
  swap/coupling info a thresholded base hides; (iii) ||a||_1 becomes
  EXACTLY E[#advs on] (prox-L1 turns honest). Expect ≈L6 in
  easy/separable worlds; pull-ahead under tight caps / heavy-tail
  volumes. Notable: variance self-anneals — Σ a(1−a) (effective
  coin-flips per draw) decays as prox-L1 pushes corners; log it.
- Cheap falsifier BEFORE any ladder spend (local, one afternoon):
  freeze a mid-run, estimate the gradient K times under both bases vs
  a high-K reference — variance decays 1/K, the surviving plateau is
  the measured BIAS in ms. If ~0 in georand, L7 only matters in
  harder worlds — run it there.
- Binding asserts (program rules): under bernoulli the base is binary,
  differs from threshold_a(a) on fractional coords at plausible
  frequency, and the sigmoid path is unreachable.
- Related: the open (b) decisive-flip variant is the DISCRETE twin
  (explicit sequential test, no relaxation). Running both cleanly
  separates "sequential testing under noise" from "hedged coupling" —
  the two theorized values of continuity.

### WORLD-KNOB PROPOSAL — zipf-ian user volumes (Tom, 2026-08-15)

Current `SCULPTOR_VOL_SPREAD` is log-uniform exp(s·U), a BOUNDED tail
(deployment_setup.py:1604). Add a true heavy-tail model:
`SCULPTOR_VOL_MODEL=zipf` + `SCULPTOR_VOL_ZIPF_A` (exponent ~1.1–2.0;
lower = heavier), same RNG-consumption-per-UG convention as the
existing knob, surfaced ONLY via experiments/model_error/worlds.py
(single source of world knobs — add e.g. 'georand_zipf'). Rationale
(theory session): a few dominant UGs make flip-deltas spiky and raise
coordinate coupling near capacity — predicted to separate the
memory/direction rungs and to be a regime where L7's hedging matters;
also more realistic than bounded spread (real traffic is power-law).
Pairs with cap tightness (SCULPTOR_SCALE_FACTOR) for the
noise-axis-vs-coupling-axis 2x2 the ladder decomposition predicts.

### RESULTS SO FAR (trusted, final semantics)

- **Policy ladder v2 COMPLETE** (210 cells; dataset
  cache/ablation/policy_ladder_v2; emailed to Tom 23:15Z).
  HEADLINE: info-efficiency RESTORED — paired per-seed, L6 beats L4 by
  ~8 objective units at N=1-2 (5/5 seeds at N=2), parity at N>=20.
  The earlier "memory hurts" was the remeasure-stop/exit-on-budget
  self-truncation artifact. Also: L1 worst everywhere + non-monotone
  (exits mid-descent at large N; N50 mean 180 vs L6 140); L3
  (no_mem+sched) surprisingly strong at N>=5 — evenly-spaced grounding
  + congestion-aware beliefs ≈ most of the value; smart gate's edge
  concentrates at tiny N. Sparse arms legitimately BEAT one-per-peering
  on the failure-resilience composite (opp advertises every popp so
  every failure congests someone) — steady latency is where below-opp
  is impossible (enforced by assert).
- **Step-paradigm A/B** (local, seeds 1-3, TCONV-matched; figure+table
  in figures/paradigm_comparison_s123.png, runner
  experiments/ablation/compare_paradigms.py, seam SCULPTOR_ABLATION_
  GRAD_SCALE=auto|fixed + SCULPTOR_ABLATION_ALPHA): auto-scale vs fixed
  alpha endpoint differences within seed noise; large alpha (0.2)
  doubles flips + halves no-op iters (~30% waste at best) at no quality
  cost. OPEN (b) agenda with Tom: decisive-flip / expected-improvement
  step (designed in HISTORY, not implemented).

### INFRASTRUCTURE (dashboards, email, SMS — the support system)

- **Dashboard** (`dashboard/`, served localhost:8643 via
  launch.json 'hardb3-dash', site dashboard_site/): TWO tabs — policy
  ladder + hard objectives. ONE updater:
  `python -m dashboard.refresh --loop 180`. Registry-driven
  `steps` per experiment: in-globs -> out-paths; expensive evals
  staleness-gated (mtime + input-set FINGERPRINT so deletions/
  quarantines propagate), cheap plots `'always': True` every cycle
  (staleness is a cost optimization, never correctness). Self-healing:
  per-cycle IP re-resolution from the alert JSON, per-cycle registry
  reload (importlib), ssh/rsync StrictHostKeyChecking=accept-new.
  Eval steps declare `'world':` resolved from worlds.py. Figures shown
  on tabs MUST be pinned step outputs (env POLICY_PLOT_OUT etc).
  Full contract: dashboard/README.md.
- **Sanity machinery** (do not remove): plot_policy5 hard-asserts any
  arm's STEADY objective below per-seed one-per-peering (always an
  eval bug — caught the congestion-blind primary LP); opp/painter refs
  flow through the SAME eval pipeline as arms (REFS entries, evaluated
  FIRST + opp_canary re-eval LAST -> [CANARY-WARNING] on drift =
  in-process eval contamination). All LP objective scalars go through
  `_soft_bounded_objective` in solve_lp_assignment.py (gated to
  avg_latency at generic/bulk primaries — hard objectives keep their
  own model.objVal).
- **Email** (standing Tom request — progress reports w/ figures):
  `cluster/send_report.py <subject> <body_file> [attachments...]`
  (Gmail SMTP, app password embedded, from/to tomkoch123@gmail.com).
- **SMS/texting** (system-level, survives agents): Mac cron —
  `~/.sculptor_cluster_alert/liveness_check.py` every 10 min (texts on
  cluster staleness/crash via budgeter emailer -> 6313393842@vtext.com)
  + `heartbeat.py` every 3h (progress SMS). Contract: update
  `active_cluster.json` on EVERY lifecycle event or the liveness cron
  false-alarms.
- **Watcher pattern** (session-bound, re-arm your own): script in
  tools/watch_policy_ladder.sh (adapt: count JSONs vs target + verdict
  greps + disk + memfree; IP from alert JSON) + persistent Monitor,
  15-30 min beats. Count-based completion; per-arm rc lines; NEVER
  exit codes alone.
- **VM ops**: proven envelope ~26-28 concurrent cells (memory: each
  ~4-6GB, deployment-BUILD phase is the peak — 40+ simultaneous builds
  thrashed the 185GB box TWICE; stagger queue launches 45-150s).
  Gurobi WLS ~28 sessions. pkill: bracket patterns AND never put the
  unbracketed name in the same remote command as mv/paths (killed the
  ssh shell twice tonight). Standalone run_fork_ladder does NOT set
  PROBE_TCONV (queue sets it = max_iter) — set it manually for
  cluster-comparable local cells.

### DATASET TRUST TABLE

- `policy_ladder_v2` (7 arm dirs): TRUSTED — final semantics.
- `hardB3v2/<obj>` dirs: being written by hb3v3 (TRUSTED when
  HB3V3 COMPLETE); painter_georand/painter_stock refs copied from old
  hardB3 (painter unaffected by solver fixes; rescored under current
  code).
- `PRESKIP_ERA_v2/*`: quarantined eras (pre-skip, interim-gate,
  exit-on-budget, coast-L1, lat+mlu, iters100). NEVER let these back
  under glob-matched names.
- `WRONGWORLD_STOCK_EVAL/*`: quarantined eval stores (stock-world,
  blind-scalar eras).
- Legacy (bad-grads policy_ladder_fixed, old hardB3, hardA/hardC,
  a10_policy, mesh_*): pre-stack; not displayed; rerun before using.

### NEXT STEPS

1. **L7 Bernoulli-base gradient — HIGHEST PRIORITY (Tom, 2026-08-15):**
   see L7 PROPOSAL section above. Order of operations: (a) the cheap
   bias-plateau falsifier locally, (b) 2-iter smoke of the flag +
   binding asserts, (c) ladder cells vs L6 (georand first, then the
   hard worlds where the theory says it should separate).
2. World knob: zipf-ian user volumes (see WORLD-KNOB PROPOSAL above);
   pair with cap tightness for the noise-vs-coupling 2x2; natural
   companion world for the L7 vs L6 A/B.
3. Babysit hb3v3 -> completion TODO list above.
4. Tom's open (b) agenda: decisive-flip step variant (+ possibly
   windowed-c A/B analysis from hb3v3 gate histories). NOTE: run
   alongside L7 when possible — decisive-flip is L7's discrete twin
   (see L7 PROPOSAL 'Related').
5. Pending Tom: a10/actual-N reruns under final semantics; hardA/C
   invalidation question; paper-figure regeneration.

