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
