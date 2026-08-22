# HANDOFF: EODS-32 — state as of 2026-08-20 ~20:30Z

Written by the 2026-08-20 session ("optimization day 2"). Everything
below is committed AND pushed (origin/main) AND deployed to the head
(107.22.173.189) with the bitwise parity gate green (run again after
any further code changes: old_handoffs/UG_DESHARDING_SURVEY.md/prove_inert.py).
Supersedes the EODS-25 handoff (archived at old_handoffs/HANDOFF_2026-08-20_EODS25_era.md).

## HIGH-LEVEL GOAL (Tom)
Demonstrate as many evals over as many random deployments as possible.
To do that we must run lots of evals very quickly => lots of cores.
Distilled: "running lots of large evals" -> "running these things on
lots of cores." Target ~1000 cores; path: valid 25 (DONE) -> valid 32
on the head -> multi-node fleet.

## STATE: EODS-25 seed-1 COMPLETE + VALID (2026-08-20 19:29Z)
Headline (avg latency ms / objective): sparse 29.58/7.198 vs
one_per_peering 28.94/7.071 vs painter 31.80/7.768; one_per_pop
34.20/8.364, anyopt 45.05/10.845, anycast 52.95/12.277.
Suboptimality vs opp: sparse +0.52ms normal / +3.94ms ingress-fail
(0% congested); painter +2.80 / +10.37. Traffic within 10ms of
optimal: sparse 96.9%, painter 90.5%. Full stats in
cache/eods/v1/actual-25/N1/seed_1_metrics.pkl on the head; ALSO
installed at the canonical path evaluate_over_deployment_sizes reads:
cache/popp_failure_latency_comparison_testing_feature-actual-25.pkl
(prior file backed up alongside with .pre_eods_backup suffix).
CAVEAT: that script runs nsim=15 for dpsize 25 — sim 0 will be reused
(skip via n_advs check) but it will try to SOLVE sims 1-14 fresh;
adjust nsim if you only want the existing column. NOTE stats_* keys
are NOT persisted in the pkl (assembly recomputes them from raw
sections each call — works, but persisting them at assembly would be
a nice fix). Known gap: stats_volume_multipliers absent — congestion
assert fires at +10-14% inflation even for trained advs (and for OPP
at other sizes) — capacity-model tightness, not undertraining;
metric redesign is a Tom-level paper decision.
25 training state backed up: head ~/eods25_backup_1738/ (state-145/150
+ training log).

## THE NEXT SOLUTION (Tom, verbatim priority — do this first)
(a) DON'T solve one-per-peering across every worker. Solve it ONCE
    (driver), distribute the result to workers, done.
    Context: compute_one_per_peering_solution runs in
    Optimal_Adv_Wrapper.__init__ for EVERY worker actor. At 32 that is
    a ~268k-column LP (the opp adv exposes the full (ug,popp) matrix)
    solved identically 64 times, AND it permanently seeds each
    worker's persistent-LP var_pool with 268k columns, which every
    later fallback-path solve then sweeps. Ship the opp solution (or
    just skip the worker-side compute — check what workers actually
    consume from it: likely only the driver's copy is ever used).
(b) Verify the fix on a local Mac smoke (the queue-harness recipe or
    old_scripts/attribute.py-style direct construction;
    bench pattern in experiments/freezing_prefix_assignments_investigation/ab_eval.py).
(c) Verify on the pre-flight VM at actual-32 with IMMEDIATE EXIT and
    incredibly verbose logging ON:
    SCULPTOR_STARTUP_TIMELOG=1 SCULPTOR_LP_SOLVE_DEBUG=1 (+
    SCULPTOR_TIMELOG_MIN to taste). The VM is STOPPED (not
    terminated): i-04d7439fa93efaf2a, c8g.24xlarge, fully
    bootstrapped (venv+repo+caches). Resume via boto3 from the head:
    ec2.start_instances; private IP may persist (was 172.31.77.22) —
    verify. ~$3.4/hr while running — stop/terminate when done.
(d) Remove / turn OFF the verbose logging for production runs (both
    env flags default off already — just don't set them; consider
    deleting the [lpdbg] hooks after the investigation concludes).
(e) Status update to Tom; then PREPARE (do not launch without his go)
    the actual-32 on the HEAD: 64 workers (NOT 80 — 80w OOM'd a
    192GB box during belief; worker RSS ~2.2GB at 32), via
    ~/eods32_launch_full.sh pattern BUT direct-invoke or queue with
    cluster/manifests/eods32_manifest.json env (already carries VOLSCEN,
    COMPACT_RB, PAINTER_MEASURE_CAP, REQUIRE_SOLNS). Add
    SCULPTOR_LP_INCR_MLU=1 SCULPTOR_LP_ADAPTIVE_MLU=1 (validated
    2.05x, bit-identical on Mac; NOT yet validated at 32 — the (c)
    run should confirm). NO belief memo exists yet for seed-1-at-32
    (no run has completed the belief phase) — first completed belief
    persists it for all future 32 runs.

## STARTUP COST INVESTIGATION (today's findings, evidence-grade)
- 97% of worker time in belief phase = solve_generic_lp_persistent ->
  model.optimize() ([wt]: lp_persistent 50% + optimize 47%; py-spy
  12/12 samples; [lpdbg] per-solve proof).
- Every solve takes the FALLBACK full-reset path: incremental LP is
  permanently disabled because standard solves are infeasible in the
  belief regime -> MLU fallback -> _last_mlu guard kills
  incrementality for the next solve. Fixes SHIPPED (env-gated,
  default off): SCULPTOR_LP_INCR_MLU, SCULPTOR_LP_ADAPTIVE_MLU.
- [lpdbg] at 32: paths ~268k EVERY belief solve; simplex_iters 7-12k;
  t_opt grows 8.5s -> 64s as var_pool grows 268k -> 473k in ~6 calls
  (~15k new keys/call). OPEN MYSTERY: pool exceeds the full (ug,popp)
  matrix (268k) — something mints NEW keys per call; prime suspect:
  per-call pseudo-ug identity splitting. INVESTIGATE before/with (a).
- get_ground_truth_resilience_benefit computes NOTHING (0.00s,
  short-circuits even with gamma=4) — Tom suspected this; confirmed.
- RB backups (rb_backups) 1.8GB at decent = 34% of painter peak;
  compact-RB (SCULPTOR_COMPACT_RB=1) validated. Painter measurement
  cap validated (SCULPTOR_PAINTER_MEASURE_CAP). Eval volscen fast
  path (SCULPTOR_EVAL_VOLSCEN=1) validated: 7.7x eval recompute,
  legacy-matching results; diurnal was 70% of the size-20 eval tail.
- Legacy diurnal/flash aggregation EXCLUDES congested ugs
  (exact-sentinel filter) — historical curves were flat under volume
  surges for the wrong reason. ug-sentinel pricing (auto with
  volscen) makes it vertex-stable. Metric redesign = Tom decision.

## SHIPPED + VALIDATED TODAY (all committed/pushed/deployed/gated)
gpshim default gurobi->highs; SCULPTOR_REQUIRE_SOLNS fail-fast;
compact parent_tracker (COMPACT_PT default-on, 12-seed A/B clean, 5%
faster); compact RB memo (COMPACT_RB, bitwise-equal fold); painter
per-iter LP-cache clear + measurement cap; eval volscen + ug-sentinel
pricing + reader guard; eval phase [mem] markers; startup timelog +
lpdbg instrumentation (env-gated); hot-start VALIDATED (used in
production for the 25 eval pivot); belief-LP fixes INCR_MLU +
ADAPTIVE_MLU (Mac-validated only). Dash: normalized convergence panel
+ alpha_t trace. Docs: HiGHS-default sweep. Skill:
.claude/skills/large-sim (Mac-local, gitignored).

## INFRA STATE
- Head 107.22.173.189: IDLE, code current (checksum-audited), parity
  gate green. eods25 queue/cell processes killed (clean exit after
  [eods] done). Dash pipeline + cron autochecker still live.
- Preflight VM i-04d7439fa93efaf2a: STOPPED (EBS kept). Resume for
  (c), then stop/terminate. Logs on it: ~/pf32_run_uninstrumented.log
  (baseline belief ~85% at kill), ~/pf32_run_oom80w.log (80w OOM),
  ~/pf32_diag.log + /tmp/lpdbg_sample.txt (per-solve anatomy).
- Cluster alert file (~/.sculptor_cluster_alert/active_cluster.json)
  is current — keep updating on every lifecycle event.
- Costs today: painter lab ~$1.50 (terminated), preflight ~$12
  (stopped). One-VM law stands: no new instances without Tom.

## WATCHERS RELINQUISHED — RE-ARM ON PICKUP
All session monitors/waiters are STOPPED (this session's harness dies
with it). When you start any long run, re-arm:
1. 30-min content-based head monitor (pattern in this session: SSH ->
   [it] tail + probe-gate + crash-greps + free -g; ALERT on
   PROC-DEAD/Traceback/MemoryError/Killed/[eods] done; note [it]
   silence is NORMAL during strategy/eval phases — check log mtime).
2. Completion waiter on the result json (until -f seed_1_eods.json).
3. Mac cron dash autochecker (3-min) is still installed and running.
GOTCHAS for your shell plumbing (cost this session real time):
- pkill -f 'run_eods_cell' MATCHES YOUR OWN SSH SESSION (the pattern
  appears in its argv) -> exit 255 + half-dead kills. Use
  pgrep -f "ru[n]_eods_cell" (char-class breaks self-match).
- zsh does NOT word-split unquoted $VARS (env lists silently collapse
  into one bogus assignment — verify flags landed via
  /proc/PID/environ, not by assumption).
- rsync file lists: destination paths are FLAT unless you use -R or
  per-file destinations (a run_eods_cell.py once landed at repo root).
- eval_all_solution_types import order is circular-import-fragile: only
  LAZY imports of sparse_advertisements_v3 symbols (see _log_mem shim).

## VALIDITY CRITERIA + PLAYBOOK (unchanged from EODS-25 handoff)
Stats populated, volume-multiplier status noted (assert expected to
fire — document per-solution), REQUIRE_SOLNS aborts cell (exit 43) if
sparse dies. OOM -> fewer workers. Resume traps: delete stale
seed_*_metrics.pkl/json before relaunch UNLESS hot-starting
(SCULPTOR_EODS_HOTSTART_DIR bypasses skip on purpose; state saves
every 5 iters). Harnesses: prove_inert (bitwise), ab_eval (frozen-adv
eval exactness), belief_bench pattern (/tmp on Mac), fork-ladder
paired-seed A/B (~3min/cell at small, 100 iters).
