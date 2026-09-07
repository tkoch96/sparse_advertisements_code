# SCRAPBOOK: frozen_prefix objective (started 2026-09-05)

Motivation: SIGCOMM #950 Reviewer A + meta-review — "proactive framing
relies on reactive DNS updates post-failure." Answer: an objective where
the user->prefix allocation is a SINGLE static vector that must work in
normal operation AND under failures (BGP fallback only, no re-steering).

## Design (Tom, 2026-09-05, confirmed in-session)

Brand-new objective, name `frozen_prefix`. NOT a retrofit of
site_failure / frozen_failure_latency (those stay as-is).

* **Decision variables: w(ug, prefix)** — strictly prefix per user, not
  route. Prefix identity never lost (kills the old pin-inversion problem
  in solve_lp_assignment_site_failure / static_failure_eval).
* Routes are coefficient data: for the normal case and each sampled popp
  failure, the routing model gives prefix->popp per user (MC realization
  for steady; `calculate_ground_truth_ingress(a_fail)` per failure,
  same precedent as solve_lp_popp_failure_congestion + site_failure).
* One shared w across all scenarios IS the freeze. w is JOINTLY
  optimized (hedged static allocation), not steady-optimal-then-frozen.
* Objective = soft(normal) + gamma * mean_f soft(failure_f), all through
  one joint LP; per-scenario soft term:
      -avg_lat_routed - P_nr*frac_no_route - P_c*frac_congested
  no_route > congestion, both moderate (defaults P_nr=50, P_c=25
  ms-equivalents, env-tunable) — per the gradient-stability rule, NO
  NO_ROUTE_LATENCY=30000 hard prices anywhere in the objective.
* Linear throughout: latency = sum w*lat_f; no-route = coefficient on
  dead (ug,prefix) pairs per scenario; congestion via per-(scenario,
  popp) overflow vars o >= load - cap.

## Failure subsampling (per-iteration, explore/exploit)

* Sample K popps to fail per ITERATION; hold the set fixed within the
  iteration (all probe pairs + MC realizations score the same scenario
  set -> clean finite differences), resample across iterations.
* Explore/exploit like the existing RB machinery (80/20-ish): exploit =
  popps with high recent failure-term cost / volume share; explore =
  uniform. Driver tracks the history and ships the EXPLICIT kill list to
  workers per flush (env vars don't reach live Ray actors; kwargs on the
  'calc_compressed_lb' jobs do).
* Eval time: exhaustive popp failures (or large fixed sample at
  actual-32) — reported numbers independent of the training schedule.

## Integration checklist (from the 2026-09-05 plumbing map)

- [ ] LP function + registration in core/hard_objectives.py
      REGISTERED_OBJECTIVES['frozen_prefix'] (needs SCULPTOR_XOBJS=1,
      read at import time in hard_objectives.py:695 +
      solve_lp_assignment.py:2014)
- [ ] kill-list plumbing: flush_latency_benefit_queue_generic stamps
      per-flush kwargs -> worker latency_benefit -> generic_objective_pdf
      -> LP call
- [ ] ObjectiveSpec in experiments/ (smoke driver run_objective.py)
- [ ] unit tests in unit_tests/test_lp_correctness.py
- [ ] eval suite evaluations/objectives/evaluations_for_frozen_prefix.py
- [ ] objective_hooks._ROUTES entry (missing = CRASH after full solve)
- [ ] generate_paper_table: OBJECTIVE_REQUIRED_KEY (missing = silently
      reads avg_latency pickle), GROUPS entry, KEY_COLUMNS (Tom curates)
- [ ] wrapper_eval.default_metrics keys (missing = pruned on resume)
- [ ] intent paper_table.objectives + env
- [ ] depstore SEMANTIC_KNOBS: SCULPTOR_FROZEN_PREFIX_* knobs

## Traps / notes

* Steady scenario uses the MC realization's routed_through_ingress;
  failure scenarios use ground-truth ingress on the zeroed adv
  (precedented inconsistency — same as popp_failure_congestion).
* NOT in _PERSISTENT_GUROBI_OBJECTIVES -> non-persistent branch, full
  model build per MC realization. Acceptable for v1; revisit if slow.
* Trains as base Generic_Objective (LP-scalar finite differences);
  using_resilience_benefit=False, gamma baked into the LP scalar ->
  no double counting via measured_objective.
* calculate_ground_truth_ingress returns popp TUPLES not indices
  (static_failure_eval.py:16 note); gti cache is per prefix-column so
  failure sweeps mostly hit cache.
* Smoke rule: actual-10 / small first, nsim>=2 (worker rebirth bug).

## Integration bugs found during small-deployment smoke (2026-09-05)

**Bug B — HiGHS global-scheduler contamination (FIXED).** My joint LP is
much larger than the other LPs (n_pairs + n_scen*n_popps vars). Engaging
HiGHS's GLOBAL parallel scheduler left process-wide solver state that made
a LATER, genuinely-feasible objective-independent LP (anyopt's avg-lat/MLU
fallback) falsely report INFEASIBLE -> the pre-existing `exit(0)` landmine
at solve_lp_assignment.py:1787 (also 759, 1460) killed the process with
rc=0. Proof: identical adv+deployment solves clean (status 2) in a fresh
process, infeasible after a frozen strategy solve in the same one; 0 no-
route users. Fix: `model.Params.Threads = 1` + `_clear_highs_model()`
(highspy `h.clear()`) in core/frozen_prefix.py. NOTE the `exit(0)` landmine
still exists in shared code -- worth a separate fix (dead `return
{'solved': False}` right after proves intent) but I did NOT touch shared LP
code.

**Bug A — "Objective frozen_prefix not implemented" on worker (RESOLVED:
not a code bug).** The generic-LP dispatcher does `generic_lp_functions[obj]`
under `except KeyError: pass`; when frozen_prefix isn't in the worker's dict
the lookup KeyErrors -> misroutes to solve_generic_lp -> that message. Root
cause = the [Ray multi-node env gap]: a STALE raylet started WITHOUT
SCULPTOR_XOBJS -> its actors skip the import-time registration tail hook.
Diagnostic proof (both workers, after a clean `ray stop`): XOBJS='1',
frozen_in_dict=True, single module core.solve_lp_assignment, not-impl=0,
run completed. Fix for local runs: `ray stop --force` before launch so
ray.init starts a fresh raylet inheriting XOBJS=1. On the cluster XOBJS is
set at node ray-start, so max_util/frac_beyond already rely on this and
frozen_prefix is no different. (Left in place: solve_lp_frozen_prefix wraps
its body to re-raise any *internal* KeyError as a loud RuntimeError so a
future realization edge case surfaces clearly instead of being swallowed.)

## Status

- 2026-09-05: design confirmed by Tom; LP + sampler + plumbing + unit tests
  (9/9, +13 LP-correctness) done. Bug B fixed (Threads=1 + highspy clear).
  Bug A = stale-raylet env gap, not a code defect (clean ray restart).
  **Full small-deployment training smoke PASSES under HiGHS**: 7 strategies,
  0 infeasible, 0 tracebacks; frozen_prefix lp_objective sparse -16.50 vs
  OPP -15.24 vs painter -24.08 (sparse near-OPP, beats painter -- the story).
  ALL local, HiGHS backend (Tom: never gurobi).

## Paper-table integration + plugin registry (2026-09-06, DONE)

Tom's ask: pipe THREE failure metrics into the table -- latency, %
congested, % no-route (kept distinct; static_failure_eval conflated the
last two) -- and stop maintaining ~8 scattered objective lists.

**Metrics** (core/frozen_prefix_eval.py, vectorized; suite
evaluations/objectives/evaluations_for_frozen_prefix.py):
  frozen_fail_{latency,cong,no_route}_by_strategy  strict-frozen, every
      method; pin = the frozen_prefix LP's OWN per-(ug,prefix) allocation
      (LP now returns `frozen_prefix_pairs`; no popp->prefix inversion),
      hedged on the training stride sample (SCULPTOR_FROZEN_PREFIX_N_FAIL),
      measured exhaustively (cap: SCULPTOR_FROZEN_PREFIX_EVAL_N_FAIL).
  reactive_fail_*_by_strategy  the UPPER ANCHOR, computed for
      one_per_peering only: assignment RE-OPTIMIZED per failure on the
      full-availability adv (LP per failure -> sampled,
      SCULPTOR_FROZEN_PREFIX_ANCHOR_N_FAIL default 50; 0 = all).
  Table (group 'Frozen failover' / tex 'Frozen Failover'): 'frozen_anchor'
  extractor shows every method frozen and the One-per-peering row as the
  reactive ceiling; caption sentence added when the group is present.
  Small-smoke numbers (45 single-popp failures): reactive-opt 7.49ms/0%/0%;
  sparse 7.97/6.24%/0.79%; painter 11.57/10.41%/0.17%; frozen-OPP
  7.14/0%/2.22% (no backup -> strands). The 2-path frozen oracle
  (core/frozen_prefix_oracle.py) congests 17% -- NOT a valid ceiling
  (strict primary+backup is more restrictive than sparse's spreading);
  kept as evidence, not used in the table.

**Registry** (core/objective_registry.py = THE one place). Consumers now
DERIVE: hard_objectives.REGISTERED_OBJECTIVES, generic_objective.
OBJECTIVE_CLASSES, objective_hooks._ROUTES, generate_paper_table
{GROUPS via _EXTRACTORS, KEY_COLUMNS, OBJECTIVE_REQUIRED_KEY,
DEFAULT_OBJECTIVES, OBJECTIVE_ALIASES, TEX_*_DISPLAY} + --objectives name
validation, wrapper_eval.default_metrics (suite keys now resume-stable),
depstore.SEMANTIC_KNOBS (17 objective knobs moved), experiments/objectives
ObjectiveSpecs (experiments/site_failure.py + frozen_prefix.py deleted),
integration_tests/verify_e2e_objectives.OBJECTIVES. Intent
paper_intent.por.json paper_table.objectives += frozen_prefix (NOTE: this
re-keys the stage fingerprint -> `run --only paper_table` will run the new
cell on the VM; `grab` of already-banked artifacts may need the dir-canon
fallback until then). unit_tests/test_objective_registry.py pins the
legacy literals (snapshot) + complete wiring per plugin. README recipe
rewritten. Per-objective KNOWN GAP: eval_all_solution_types --objective
help text still lists names by hand (cosmetic).

## actual-10 cloud smoke (2026-09-06, i-09a6 m8g.16xlarge, run 20260906_105413-frozen_a10_smoke)

nsim=2, 100 iters, HiGHS, papertable preset. Deployment 1 (308 popps,
3127 ugs): sparse to it 100 (~1.2 min/iter, obj 20.74 -> 20.67, converged
by ~it 30) + all 6 baselines, banked. Deployment 2 (286 popps) died at
it 10: [ray-recover] rebuilt the pool from the pool's BIRTH deployment
(deployment 1) -> worker IndexError (popp 303 into a 286-row adv).
Latent nsim>1 infra bug, made deterministic by frozen_prefix's memory
footprint. FIXED: Worker_Manager.sync_respawn_state (commit c8d9e73).
Relaunched --resume with SCULPTOR_N_WORKERS=40.

**Sizing finding for the actual-32 cell**: frozen_prefix workers are
~3.2 GB RSS each at actual-10 (64 workers -> 107-205 GB sawtooth on a
247 GB box; Ray's memory monitor reaps at 95% ~ every 3-4 iters -> pool
rebuild treadmill). The joint LP carries n_fail scenario blocks of
per-popp overflow vars. At actual-32: cap workers (~40-48 on this box)
and/or lower SCULPTOR_FROZEN_PREFIX_N_FAIL from 20; measure RSS first.

### RESULT (run 20260906_105413-frozen_a10_smoke, complete 19:50Z; VM stopped)
actual-10, nsim=2, 100 iters, HiGHS, 40 workers (0 rebuilds). Harvested to
cache/cluster_runs/20260906_105413-frozen_a10_smoke/ (pickle + table + logs).
Frozen failover columns (nsim=2 means; One-per-peering row = reactive anchor):
  method            steady  fail_lat  %cong   %no-route  objective
  OPP (reactive)    37.00   38.97     0.00    0.000      -74.87
  SCULPTOR          37.21   37.14     9.90    0.066      -75.55
  PAINTER           36.35   36.43    16.17    0.059      -77.01
  Unicast           37.32   37.27    15.64    0.012      -80.03
  AnyOpt            40.88   40.80    12.38    0.005      -94.26
  Anycast           56.84   55.99     1.25    0.000     -114.05
Frozen OPP (pickle, not table): no-route 0.32-0.35% (stranding pathology,
small at actual-10 because ~300 popps give backups). Per-sim spread is wide
(SCULPTOR cong 5.3% vs 14.5%; painter 12.1% vs 20.2%) -> nsim=2 is thin.
Reads: SCULPTOR best of the practical methods on the objective (closest to
OPP) and on % congested (9.9 vs painter 16.2); painter ~0.7ms better on
frozen latency here. CAVEAT for the table: frozen rows' latency is over
UNCONGESTED volume only while the reactive anchor has 0% congestion and pays
latency to stay under hard caps -- so "SCULPTOR 37.1 < anchor 39.0" is not
a win over the ceiling; state this in the caption or report the anchor's
latency alongside its %cong.
Ops: ~4.9h VM (~$14 incl. two wasted relaunches). expctl verdict says
"exited 0 WITHOUT a completion banner" for papertable runs -- banner-string
mismatch with generate_paper_table, not a failure; worth teaching expctl
the papertable driver's final line ('total Xs').

### User-level trace of the actual-10 result (2026-09-06 evening)
Failing the heaviest popp under SCULPTOR's frozen allocation (Miami/AS7195,
62.7 vol, 122 shares): 100% of displaced traffic keeps a route on its pinned
prefix (mechanics OK), but only 2/10 traced users get the good backup
(another Miami popp, +5-9ms); 8/10 are sent to Madrid/Amsterdam at +80-120ms
onto popps that are 1-4x OVER capacity. Why: (1) BGP preference is not
latency-aligned -- user 1 has Madrid/1299 (pref 300, 237ms) ranked above
Miami/1299 (pref 299, 139ms) on the same prefix; (2) SCULPTOR's 26 prefixes
are ALL broad near-anycast variants (44-179 popps, 6-8 sites each), so the
fallback is a BGP lottery -- for this failure SCULPTOR ~= painter ~= unicast
(~+90ms, 82-98% onto over-cap popps); (3) low-anycast popps have caps of
0.5-13 units, so absorbing a 62-unit displacement without congestion needs
spreading that single-winner fallback cannot do. Best-available surviving
latency was within ~1-3ms of the original for every user.
Tom's read: objectives show it works; the knob question is how to bite
harder on no-route/congestion without numerical trouble -> divide latency
by 10, keep penalties (same argmin as penalties x10; smaller scalars).

### Objective-native levers (Tom: "not some global thing")
frozen_prefix's tunables are declared ONCE on its plugin
(core/objective_registry.py lp_defaults): frozen_gamma, frozen_n_fail,
frozen_top_load (heaviest-loaded popps ALWAYS in the kill set),
frozen_explore_frac, frozen_no_route_penalty, frozen_congestion_penalty,
frozen_lat_scale. Resolution: lp_kwargs_for(obj) = defaults + env
overrides (lp_env_overrides, SCULPTOR_FROZEN_PREFIX_*) -> Generic_Objective.
lp_kwargs (explicit driver kwargs still win) -> every driver LP call, and
FrozenPrefixObjective.per_call_lp_kwargs ships the frozen_* levers to workers
with the kill list, so actors price exactly as the driver (no env on the
actor side). Eval pin + objective_value_scorer use lp_kwargs_for too.
TRAP fixed same day: experiment_specs must NOT bake lp_defaults into the
ObjectiveSpec's explicit lp_kwargs (explicit wins -> env overrides ignored;
A/B arm B ran with lat_scale=1.0 and had to be redone as B2).
Small A/B (same deployment/seed, 30 iters): A baseline, B2 lat_scale=0.1,
C lat_scale=0.1 + top_load=5 -- results appended below when scored.

### Small A/B results (same deployment/seed, 30 iters, exhaustive 45 failures)
  arm  lat_scale top_load | steady fail_lat  %cong  %noroute   (effective lat_scale
  A        1.0       0    |  7.71    7.81    6.21%   0.72%      read back from each
  B2       0.1       0    |  8.25    8.41    6.93%   0.63%      arm's trained LP ->
  C        0.1       5    |  8.60    8.69    8.69%   0.37%      levers flow end-to-end)
  painter                 | 11.39   11.46   11.59%   0.18%
  reactive anchor         |  7.00    7.36    0.00%   0.00%
lat/10 helps no-route (-12%; -49% with top_load) at +0.5-0.9ms steady, but
CONGESTION RISES: with latency de-weighted the LP leans on penalties, and
P_nr:P_c = 50:25 makes congesting cheaper than stranding; prefixes got
BROADER (A 20/19 -> B2 29/26 popps) = more surviving options = less
stranding but fallback onto small-cap popps. Both arms still fill the
heaviest popp to exactly cap: the failure block is a MEAN over 20 scenarios
at gamma=1, so the heaviest failure carries ~1/20 the weight of normal ->
next lever = frozen_gamma (arms D: gamma 4; E: gamma 4 + lat 0.1 + top 5).
  D   gamma=4 lat 1.0 top 0      |  7.72    7.80    7.74%   0.73%   (gamma alone: nothing)
  E   gamma=4 lat 0.1 top 5      |  8.86    9.00    7.58%   0.23%   (best no-route, 3x vs A)
No arm beats A's 6.2% congestion. Structural: the LP prices congestion on
EXCESS volume (linear) while the metric flags ALL volume on a popp even 1%
over cap -> at-cap loading is free to the LP and maximally fragile (every
arm puts 70.0 on the 70.0-cap popp). Next lever: frozen_cap_headroom (LP
solves against caps*h; try 0.9) -- arms F (E + headroom .9), F0 (A + .9).
  F0  A + cap_headroom .9        |  8.05    8.14    7.36%   0.66%   (headroom: no help)
  F   E + cap_headroom .9        |  9.52    9.60    7.99%   0.09%   (no-route ~solved)
DIAGNOSIS (diag_cong.py): re-pinning against ALL 45 failures vs the 20-popp
training sample barely moves congestion (A 6.21->6.43, F 7.99->7.29) -> NOT
a train/eval sampling gap. Steady over-cap popps = 0 -> headroom had nothing
to fix. Caps are not tiny at small (min 6.7 / median 32 / max 70); the bind
is AGGREGATE: vol 1361 vs ~1450 total cap (the 1.1x provisioning). A 40-70
unit popp failure dumps its users onto a few BGP-preferred same-site
neighbours (('0','1') load 55.6 / cap 28.1); ~10% global slack can absorb
that only if fallback is DIVERSIFIED, and single-winner BGP fallback within
BROAD prefixes cannot diversify (same popp set -> same fallback for every
prefix). All arms learned broader prefixes. => residual frozen congestion is
structural at 1.1x unless the optimizer learns NARROWER, different-fallback
prefixes. Arms G (P_c=100 > P_nr=50) / H (G + n_fail=45) test whether pricing
congestion above stranding pushes it there.
  G   E + P_c=100 (> P_nr=50)    |  8.89    8.98    6.16%   0.41%   (congestion back to
  H   G + n_fail=45 (all)        |  8.87    9.05    6.55%   0.26%    baseline; frontier)
VERDICT (2026-09-06, small, 30 iters, nsim=1 -- directional): the levers
move the solution along a no-route <-> congestion FRONTIER. No-route is
tunable from 0.72% down to 0.09% (F); congestion has a floor ~6.2% that no
pricing breaks (G matches baseline congestion while cutting no-route 43%;
pricing congestion above stranding trades back). Prefixes stay/get BROADER
in every arm (A 20 -> H 33 popps on the widest). Structural cause per the
diag: 1.1x aggregate provisioning + single-winner BGP fallback inside broad
prefixes cannot diversify a 40-70 unit displacement. Paper framing: frozen
residual congestion is the price of NO re-steering at 1.1x; the reactive
anchor's 0% exists only because it re-optimizes.
Candidate defaults (Tom to pick): G-like (gamma 4, lat 0.1, top 5, P_c 100):
baseline congestion, -43% no-route, +1.2ms; or F-like (headroom .9) if
no-route matters most (0.09%, +1.8ms, +1.8pt congestion). Validate on
actual-10 nsim>=2 before the paper cell -- small is tiny (45 popps, 3 sites).

## Size-32 paper cell plan (Tom 2026-09-07: "pop it onto paper table")
Target: frozen_prefix at dpsize 32 with nsim=3 (like the other objectives),
produced under the CAMPAIGN tag 20260823_130342_papertable32b so the paper
table aggregates it. Trap: the intent's paper_table stage has nsim=1 (the
avg_latency cell only ever ran at 1) and the driver used ONE global nsim --
raising it to 3 would re-run the multi-day avg_latency cell. Fix (done):
per-objective counts + env in generate_paper_table (--nsim-by-objective,
--env-by-objective), passed through by run_all_paper_evaluations from the
intent stage keys `nsim_by_objective` / `env_by_objective`. Intent now has
frozen_prefix: 3 deployments, SCULPTOR_N_WORKERS=24 (memory: ~3.2 GB/worker
at actual-10; heavier at 32). So `run_all_paper_evaluations run <intent>
--only paper_table` on the VM launches exactly the frozen cell at nsim=3
and re-emits the table; `grab` pulls artifacts. Lever defaults for the
cell: set on the plugin after the actual-10 arm-G run reads out (pending).
Cost: unknown at 32 -- quote from the first ~20 iterations' pace; frozen
LP is heavier than avg_latency (non-persistent, 21 scenario blocks).

### Size-32 execution plan, refined (2026-09-07)
The pipeline's `run` executes AND stores on intent['storage_vm'] (i-0428,
16 vCPU/61 GB -- the dash box) with no per-stage override; the campaign's
five size-32 pickles exist on BOTH i-0428 and i-09a6. The storage VM had
OLD code and reported `frozen_prefix ok 1 sim(s)` -- the avg_latency-pickle
fallback (no required key) -- pushed new code there: now MISSING/want 3.
joint_priority shows "2 sim(s) [FAILED strategies in sims: [0]]" on both
boxes: the 09-03 backfill fixed the metrics but left failed_strategies in
compare_rets[0]; coverage now excludes that sim (n=2>=1 still ok) -- table
aggregation is unaffected but worth cleaning.
Split: COMPUTE the frozen cell on i-09a6 via
  expctl launch i-09a6 --preset papertable --run-tag 20260823_130342_papertable32b
    --dpsizes 32 --nsim 3 --max-iter 150 --objectives frozen_prefix
    --env SCULPTOR_XOBJS=1 --env SCULPTOR_LP_BACKEND=highs --env SCULPTOR_N_WORKERS=24
(preset now has --run-tag/--nsim-by-objective; harvest scoped to the
objective), then move the ONE frozen pickle to the storage VM and run the
intent `run --only paper_table` THERE (frozen covered at 3 -> no training;
emit table -> depstore store -> grab -> paper repo). Lever defaults for the
cell = whatever the actual-10 arm-G run supports; set on the plugin first
so the cell needs no lever env.

### actual-10 arm-G cell (run 20260906_225909-frozen_a10_G1, nsim=1, 100 it)
CAVEAT: drew a DIFFERENT deployment than the baseline run's sim 0 (OPP steady
41.8 vs 37.0 ms; painter frozen cong 1.05% vs 12.1%) -- per-cell deployment
draws are how the pipeline works (strategies compared WITHIN a cell), so the
lever A/B rests on the small sweep, not on this. Within its deployment:
  OPP(reactive) 41.65ms 0.00% 0.000% | SCULPTOR 41.91 1.20% 0.028% |
  PAINTER 43.56 1.05% 0.040% | Unicast 44.92 1.22% 0.004% | Anycast 53.70 1.36% 0
SCULPTOR best-of-practical on all three (congestion a wash: this deployment
has slack). Heaviest popp still at cap (63/63); its failure -> 92% onto
over-cap popps, +85ms; widest prefix 206 popps. Memory: 40 workers grew
131->197 GB over 80 iters (~4.9 GB/worker, cache growth; the 2/iter refresh
does not reclaim) -> size-32 cell starts at 20 workers. Plugin lp_defaults
set to arm G. Run 2.7h ~$7.6 (+$0.6 abandoned nsim=2 start).

### Size-32 campaign cell LAUNCHED 2026-09-07 05:48Z: run 20260907_014815-frozen32
on i-09a6 (expctl papertable --run-tag 20260823_130342_papertable32b, nsim 3,
150 it, --objectives frozen_prefix, highs, SCULPTOR_N_WORKERS=20, plugin
defaults = arm G). Cost quote from the first ~20 iterations. Aftercare:
expctl finish -> xfer_pickle.py (scratchpad) i-09a6 -> i-0428 for
cache/popp_failure_latency_comparison_testing_feature-actual-32_20260823_130342_papertable32b_frozen_prefix.pkl
-> on storage VM: run_all_paper_evaluations run <intent> --only paper_table
(frozen covered at 3 -> aggregation only) -> grab -> paper repo figures.

### Size-32 cell KILLED at iteration 1 for a cost quote (2026-09-07 06:20Z)
777 popps / 5169 ugs: 18.7 s per LB probe (worker log), ~40-45 min/iteration
with 20 workers -> ~105 h/deployment -> nsim=3 x 150 it ~ 13 days ~ $900.
Startup (belief calc + pool) ~14 min; ~$1.5 spent; VM stopped, EBS intact.
Profile (actual-10, 308 popps/3127 ugs, 71,810 pairs): whole call 0.99 s of
which HiGHS solve 0.96 s -> Python assembly negligible; the cost IS the LP.
Levers measured (actual-10): n_fail 20/10/5 -> 0.99/0.52/0.33 s (linear;
obj -21.09/-21.03/-20.95); HiGHS ipm 6x SLOWER (5.9 s) -> keep simplex.
Options at size 32 (per-iteration ~42 min @ n_fail 20, ~21 min @ n_fail 10,
20 workers; ~x1.3 faster with 24-28 workers if memory holds):
  nsim3 x150it: n_fail20 ~13d/$900 | n_fail10 ~6.6d/$450
  nsim3 x 80it:              ~7d/$480 |            ~3.5d/$240
  nsim1 x150it:            ~4.4d/$300 |            ~2.2d/$150
(actual-10 objective plateaued by iter ~30-50 -> 80 iterations likely enough.)
Awaiting Tom's pick. Restart cost is trivial (belief memo).

## LIFTED formulation (2026-09-07, commit aa26b47) -- exact, LP size independent of K
Tom: "there's a ton of redundant information across the 20 problems". Yes:
a single-popp failure k only moves the pairs whose normal winner is k, each
to ONE fixed BGP fallback (2nd-best pref in the prefix). So scenario k's
load on popp j = L_j + (volume falling from k onto j): normal load + sparse
delta. Lifted LP: aux var L_j (normal load, written once), normal overflow
o0_j, and a row + overflow var only for (k, j) with a non-empty delta;
untouched popps share o0_j with cost weight 1 + gamma/K*(K - [j killed] -
m_j). nnz ~ 2*pairs instead of (K+1)*pairs. Vectorized fallbacks via
popp_by_ug_indicator top-2 per prefix column (== per-scenario gti, 0
mismatches everywhere). Stacked kept as reference: frozen_formulation=
'stacked' / SCULPTOR_FROZEN_PREFIX_FORMULATION. Tests:
unit_tests/test_frozen_prefix_lifted.py (4) + test_frozen_prefix.py (9).

Ladder (worst relative objective gap stacked vs lifted: 7.3e-9 = solver tol):
  rung       adv                pairs   K=20 stacked->lifted      exhaustive stacked->lifted
  small      allon/rand8/opp    .2-2k   .023->.002 / 1.7x / 4.3x  33x / 2.5x / 8.7x
  actual-5   allon/rand8/opp    3-46k   82x / 2.1x / 5.4x          1020x / 12x / 113x
  actual-10  allon/rand8/opp    3-89k   116x / 70x / 7.1x          1620x / 220x / 133x
  actual-10  SCULPTOR a10 soln  72k     0.98s -> 0.35s (2.7x)      28.7s -> 3.6s (7.9x)
Training regime = last row: broad SCULPTOR prefixes -> base pair LP is the
remaining cost (nnz 1.5M -> 147k, time only 2.7x: near cold-solve floor;
persistent avg_latency LP warm-solves in 0.036s on the same adv -- warm
starts across probes are the next order-of-magnitude lever, but they
re-open the HiGHS shared-state landmine). Exhaustive failures (K=n_popps)
now 3.6s at actual-10: affordable, not free (coupling rows cost simplex
iterations). Solver modes on lifted (actual-10 SCULPTOR adv): dual simplex
0.35s (best); primal 1.5s; ipm 0.85s (2.8s exhaustive -- only wins there);
pdlp 12s. Python assembly ~0.05s/call.
actual-32 rung: running on i-09a6 (VM restarted 16:16Z, Tom: "really scale
this up ... fine to work on there").
actual-32 rung (i-09a6, 777 popps / 5190 ugs, campaign avg_latency pickle's
SCULPTOR adv = 204,195 pairs; gti cross-check on a 20-winner sample, 0
mismatches; log ~/ladder_a32.log on the VM):
  allon_1prefix  K=20   stacked 12.7s -> lifted 0.075s (169x)   obj equal 1e-16
  allon_1prefix  K=777  stacked 424s  -> lifted 0.080s (5273x)  obj equal 8e-16
  SCULPTOR adv   K=20   stacked 7.7s  -> lifted 3.1s  (2.5x)    obj equal 3e-12
                        lifted rows 7.2k, vars 206k, nnz 418k
  SCULPTOR adv   K=777  lifted > 30s TimeLimit (unsolved) -> exhaustive is NOT
                        a training option at 32; stacked skipped (158M nnz).
Read: at the training regime the remaining cost is the 200k-column pair LP
itself (2.5x, same ratio as actual-10); n_fail barely matters for the lifted
model (rows only), so keep n_fail=20 and forget the n_fail=10 lever. Next
order-of-magnitude levers, not done: (a) column reduction -- most of a
user's ~39 prefixes are dominated (bad winner AND bad fallback); column
generation keeps it exact; (b) warm starts across probe pairs (persistent
model) -- re-opens the HiGHS shared-state landmine.
Training smokes with the lifted default (same box, concurrent): actual-10
20260907_123220-frozen_a10_lifted 32 workers ~35 s/iter vs stacked G1 run 40
workers ~97 s/iter (~3.5x worker-normalized); actual-5 20260907_123206-
frozen_a5_lifted 16 workers ~45 s/iter. Both clean through iter 25/37.
actual-10 lifted smoke DONE (20260907_123220-frozen_a10_lifted, 32 workers,
100 it, total 3862 s vs stacked G1 9403 s at 40 workers -> ~3x worker-
normalized; 0 tracebacks; harvested to cache/cluster_runs/...). Table
(within-cell; deployment draw differs slightly from G1):
  OPP(reactive) 41.80ms 0.00% 0.000% | SCULPTOR 42.01 1.05% 0.029% |
  PAINTER 44.00 0.88% 0.031% | Unicast 44.99 1.02% 0.004% | AnyOpt 50.64 1.09% |
  Anycast 53.44 1.29% 0
Same read as G1: SCULPTOR best-of-practical on latency + no-route,
congestion a wash. Size-32 cell launch (nsim 3 x 150 it, 32 workers,
~2.5-3.5 d / $170-240) blocked by the permission classifier -> needs Tom.
actual-5 lifted smoke DONE (20260907_123206-frozen_a5_lifted, 16 workers,
100 it, 6927 s total incl. eval, clean). Table: OPP(reactive) 54.37ms 0/0 |
SCULPTOR 54.66 1.77% 0.039% | PAINTER 55.53 1.46% 0.064% | Unicast 57.69
1.32% 0.003% | AnyOpt 60.96 2.00% | Anycast 62.77 1.82% 0. Same read.
TRAP (2026-09-07 18:27Z): size-32 cell 20260907_141609-frozen32_lifted,
launched while the a5 smoke was still running, ATTACHED to the a5 run's Ray
cluster ("Connecting to existing Ray cluster") instead of starting its own;
a5's end-of-run teardown stopped Ray and the size-32 driver died 2 s later,
rc=1, NO traceback anywhere (driver, workers, dmesg). The a10+a5 pair had
survived only because the earlier-finishing one did not own the cluster.
Rule: one SCULPTOR driver per box, or give the second its own Ray. Relaunched
alone as 20260907_143058-frozen32_lifted2 (nsim 3 x 150 it, 32 workers).

## Latency-weight lever proof on small + the PIN finding (2026-09-07 evening)
Tom: "divide latencies by 100 ... prove that changing this lever changes the
outputs; in theory near-0 no-route/congestion at the expense of latency".
TWO HARNESS BUGS FOUND FIRST: (1) the A-H arms (2026-09-06) shared popps +
volumes but had DIFFERENT link capacities and ingress priorities per arm
(hash check) -- and score_arms pinned every arm on arm A's deployment, so
the "6.2% congestion floor / no arm beats A" verdict is contaminated.
(2) Even with SCULPTOR_DEPLOYMENT_SEED fixed, arms differed: Python string-
hash randomization changes set/dict iteration order inside the deployment
generator -> different RNG consumption -> different priorities/caps.
PYTHONHASHSEED=0 + SCULPTOR_DEPLOYMENT_SEED=31415 gives identical
deployments (hash 0e53d06081 on all four arms). Any cross-process A/B needs
both; the paper table is unaffected (all strategies share one process).
Arms (small, 30 it, 4 workers, lifted LP, gamma 4 top 5 P_c 100 P_nr 50;
exhaustive 45-failure sweep; pin = stride-20 as the code then defaulted):
  arm    lat_scale headroom | steady fail_lat  %cong  %noroute | LP%excess LP%noroute
  S10      0.1      1.0     |  9.90   10.00    2.50%   0.252%  |  0.000%    0.070%
  S100     0.01     1.0     | 10.97   11.06    2.46%   0.189%  |  0.000%    0.000%
  S1000    0.001    1.0     | 12.18   12.20    2.82%   0.054%  |  0.000%    0.000%
  S100H    0.01     0.9     | 11.22   11.28    2.60%   0.493%  |  0.863%    1.881%  (headroom HURTS)
  painter                   | 13.28   13.36    4.68%   0.042%  | reactive anchor 7.29/7.65/0/0
Lever works: latency up 2.3ms, no-route 0.25% -> 0.05% (LP-side no-route 0
from 0.01 down). Congestion flat ~2.5-2.8% even though the LP sees 0.000%
excess -> the LP is blind to what the table counts. DIAGNOSIS (exhaustive
pin instead of stride-20 pin, same trained advs):
  S10   stride pin 2.50% / 0.252%  ->  exhaustive pin 0.08% / 0.204%  (+0.3ms)
  S1000 stride pin 2.82% / 0.054%  ->  exhaustive pin 0.00% / 0.038%  (+1.3ms)
The residual congestion was FAILURES OUTSIDE THE PIN'S 20-POPP SAMPLE
landing on popps the pin loaded to exactly cap (the one over-cap event left
in S10: load/cap 1.017 flagging 3.6% of volume). Not structural; a sampling
artifact of the eval pin. Now cheap to fix with the lifted LP -> pin_pairs
defaults to ALL popps (SCULPTOR_FROZEN_PREFIX_PIN_N_FAIL=k to sample;
SCULPTOR_FROZEN_PREFIX_PIN_TIME_LIMIT default 1800 s via the new
frozen_time_limit LP lever). Every strategy row gets the same exhaustive
pin, so the table stays fair. Consequences: (a) the 2026-09-06 "structural
6% floor" story is retracted; (b) the running size-32 cell
(20260907_143058-frozen32_lifted2) trains fine but its END-OF-RUN EVAL will
use whatever pin code is loaded then -> re-score its stored advs with the
exhaustive pin afterwards if needed; (c) with the exhaustive pin, Tom's
extreme is reachable: lat_scale 0.001 -> 0.00% cong / 0.04% no-route at
+3.5 ms vs lat_scale 0.1. Next: rerun a5/a10 at lat_scale 0.01 (storage
VM) and re-score the existing a5/a10 lat-0.1 advs under the exhaustive pin.
Exhaustive-pin RE-SCORE of the existing a5/a10 lifted smokes (lat_scale 0.1
advs, same pickles; scratchpad rescore_pin.py; pin secs a10: 0.3 stride /
2.8 exhaustive):
  actual-5   strategy   stride20: lat %cong %nr    -> exhaustive: lat %cong %nr
             SCULPTOR   54.66 1.77% 0.039%         -> 54.88 0.09% 0.036%
             painter    55.53 1.46% 0.064%         -> 55.67 0.37% 0.087%
             anyopt     60.96 2.00% 0.001%         -> 61.36 1.39% 0.001%
             anycast    62.77 1.82% 0              -> unchanged (no options)
             reactive anchor 54.37 / 0 / 0
  actual-10  SCULPTOR   42.01 1.05% 0.029%         -> 42.28 0.10% 0.043%
             painter    44.00 0.88% 0.031%         -> 44.03 0.44% 0.059%
             anyopt     50.64 1.09% 0.001%         -> 50.50 1.17% 0.001%
             reactive anchor 41.80 / 0 / 0
With the pin hedged against every failure SCULPTOR DOMINATES painter on all
three metrics at both sizes (congestion 0.09-0.10% vs 0.37-0.44%; painter's
advs have fewer backups so the pin can only do so much for them). The
"congestion a wash" read of the earlier tables was the stride-pin artifact.
size-32 cell 20260907_143058 will need this re-score on its stored advs
(its eval code may still stride-pin).
