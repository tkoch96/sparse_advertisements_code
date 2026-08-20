# HANDOFF: EODS-25/32 — state as of 2026-08-19 ~21:15Z

Written by the 2026-08-19 session (the "optimization day"). Everything
below is committed AND pushed (origin/main @ d2a2040) AND deployed to
the head (107.22.173.189, c8g.24xlarge 96c/185G) with the bitwise
parity gate green (experiments/desharding/prove_inert.py).

## HIGH-LEVEL GOAL (Tom, restated 2026-08-20)
Demonstrate as many evals over as many random deployments as possible.
To do that we must run lots of evals very quickly, which means lots of
cores. The goal distills from "running lots of large evals" to
"running these things on lots of cores." Target: ~1000 cores. Path: valid single-deployment results at 25 then 32 on
the 96-core head, then multi-node fleet (bandwidth is per-node:
~96w/node for training, ~32 effective for cold phases -> ~10-12 nodes;
serial driver floor <10-20s/iter becomes the binding constraint at
that scale — async/pipelined grads is the known counter). Measured
scaling model + core-seconds numbers in KNOWN NUMBERS below.

## OVERNIGHT 08-19/20 DELTA (read before acting)
- KeyError popp-map bug turned BLOCKING (killed sparse at iter 67) —
  FIXED (solve_lp_assignment sort key via lat_matrix, NO_ROUTE for
  missing pairs). 96-worker OOM-reap treadmill -> run at 80 workers.
  Resume-trap reminder: clear seed_1_metrics.pkl+json before any
  relaunch or the cell skips training on the stale checkpoint.
- Belief memo WORKS in production (3 restarts at ~7min startup).
- HOT-START now wired: SCULPTOR_EODS_HOTSTART_DIR=<run-dir name under
  ws/runs> on run_eods_cell resumes training from newest state-N.pkl
  (popp_to_users guard makes it safe). VALIDATED 2026-08-20 ~11:20Z:
  actual-5 kill-resume smoke (kill -9 iter 23 -> resume state-25 ->
  evals) clean end-to-end; guard fired and lazily rebuilt. Trusted.
- The parallel "32 agent" fleet (3x c8g.24xlarge, ~$87): TERMINATED
  2026-08-20 ~09:57Z per Tom. FULL FORENSICS (2026-08-20, from the
  32-agent's session transcript c4da311e + cache/eods/eods32_live/):
  three attempts, no valid result.
  TIMELINE: 00:58Z start; ray up -> IAM PassRole DENIED -> manual
  boto3 launch, 3 nodes up 01:07Z (turnup ~9min). Then ~5h lost:
  us-east-1 arm64 apt mirror 503s + a root-owned orphan apt holding
  worker1's lock ~2h (pkill missed the perm domain), 6.9G rsync + a
  broken wait-loop/SSH timeout ~2.2h (session dormant til 06:06Z).
  v1 06:10Z (250w): OOM in belief — worker RSS at 32 is ~2.2G (vs
  1.4G at 25); 83/node x 2.2G ~ 182G on 185G nodes; raylet died,
  worker0 wedged -> force-stop/start. v2 06:46Z (120w): INVALID —
  v1's partial metrics pkl tripped the already-computed skip,
  training silently skipped ("done in 612s", zero [it]). v3 07:16Z
  (120w, pkl cleared): belief ~9min (288-core bandwidth), iters 0-10
  at ~110s/iter (healthy), then SPARSE DIED ~07:52Z:
  GurobiError WLS "Overage for too long, 3 active sessions, over
  baseline for 32 minutes". MECHANISM (high confidence): launcher
  env had SCULPTOR_LP_BACKEND=highs but multi-node Ray actors get
  the env of each node's `ray start`, NOT the driver's — remote
  workers silently defaulted to gurobi (gpshim selects backend at
  import from process env). Single-node runs are immune (env
  inherited); the current 25 log has ZERO gurobi mentions. After
  sparse died the harness continued baselines+evals ~2h (invalid).
  COUNTERMEASURES: (1) 32 re-runs ON THE HEAD via the queue
  (single-node => env inherited => highs); (2) SCULPTOR_REQUIRE_SOLNS
  env (in eods32 manifest, =sparse): required-strategy failure aborts
  the cell (exit 43) instead of hours of baseline evals; (3) monitors
  alert on [it]-silence >20min and on "GurobiError"; (4) FLEET
  PREREQ for the 1000-core plan: propagate SCULPTOR_* env to remote
  actors (ray runtime_env env_vars, or bake exports into every
  node's ray start script) + clear stale metrics pkls before
  relaunch + worker count from measured per-size RSS (32: ~2.2G/w).
- CURRENT: clean 25 run (fixed code) training on the head — iter ~52
  at ~10:15Z, GT obj 7.35, ~95-120s/iter, 3+/10 probes spent, no stop
  attempts yet. Expect stop window ~iter 100-150, evals after, valid
  result mid-afternoon. Dash: 25 curves + (dormant) 32 overlay + all
  buttons. Believed-vs-GT gap (pseudo ~19k vs obj ~7.4) never closes —
  open scientific question, discuss with Tom before treating as bug.

## IMMEDIATE NEXT STEP
The running 25 cell completes -> check evals VALID (stats populated,
volume-multiplier assert silent) -> launch the single 32 cell
(~/eods32_launch_full.sh; consider PROBE_TCONV ~ realistic horizon).
Then: fleet plan for 1000 cores (nodes x width, Ray multi-node — the
autoscaler-era code supports it; per-node plasma dedupe already in).

## SCOPE (Tom, revised mid-day)
ONE deployment at actual-25 (seed 1), then ONE at actual-32 — through
run_eods_cell -> eval_latency_failure.evaluate_all_metrics (classical
paper metrics). NOT the 10-sim campaign; NOT ablation-ladder semantics.
Manifests: tools/eods25_manifest.json, tools/eods32_manifest.json
(1 cell each). Launchers on head: ~/eods25_launch_full.sh (96w),
~/eods32_launch_full.sh (96w) — both carry the full env.

## RUNNING NOW
The definitive 25 cell (relaunched ~20:23Z): 96 workers, belief phase
-> training ~21:15Z. It has EVERYTHING: full optimization stack,
staleness-floor probing, budget-hold stop AT THE LIVE SITE, [it]/[wt]
telemetry, memo-persist (this run writes the seed-1 belief memo — the
LAST 50-min startup for this seed). Watch: dash tab EODS-25
(convergence panel = believed vs GT + stop signal + historical dashed
context from state-pickle backfill), live tail (20s), live convergence
PDF, flamegraph. Monitors: 30-min heartbeat in-session; cron
autochecker (3-min) self-heals the dash pipeline.

## THE DAY'S SHIPPED CHANGES (all bitwise-gated; quality-proven
## n=30 paired A/B: mean delta -0.6% of scale, p=0.70, identical iters)
1. Desharding: UG-slice machinery removed everywhere; workers get the
   FULL deployment via one ray.put (plasma). Grads are what's
   distributed. Gate: experiments/desharding (bitwise, PYTHONHASHSEED
   pinned — cross-process repro REQUIRES pinned hash seed).
2. pattern_cache compact repr (uis/lens/pad int16, packed keys, no
   probs) + vectorized block assembly. ~16x RAM. experiments/
   pattern_cache/bench.py = the repr bench (run at scale too: 850MB ->
   82-130MB @217 patterns).
3. measured_latency_benefits: ndarray (was dict, ~17x); lat_matrix
   built once driver-side, shipped in deployment dict (zero-copy per
   node).
4. Incremental LP activation (set-diff vs prev active; SCULPTOR_LP_
   INCREMENTAL=1 default) + vectorized _to_highs_bound in gpshim.
5. MC_NUM=1 standard for EODS (Tom); env in launchers.
6. Miss-path vectorization (CSR ui->popps + mask/reduceat; was 23% of
   cold solves per line-level flamegraph).
7. Belief memoization (SCULPTOR_BELIEF_MEMO=1 default): persists the
   iov bootstrap keyed on (deployment/init/MC/gamma/obj/world); loads
   in seconds. Validated 10-seed load-vs-compute (p=0.49). Fixing it
   also fixed the HISTORICAL hot-start popp_to_users bug
   (_ensure_popp_to_users lazy rebuild).
8. Stop rule (Tom's design): staleness floor (PROBE_MAX_STALENESS,
   default=slot period — measure if no grounding in X iters and budget
   remains) + budget-hold (no exit while probes banked unless current
   adv is ground-truth measured). BOTH stop-check sites (2852+2984)
   via _post_stop_check — the first attempt patched only the dead
   site (caught via empty dash panel).
9. Logging: [it] per-iter (obj/pseudo/rd/rde/rad/n_on), [wt] compact
   worker timing, UTC timestamps, compact worker errors
   (SCULPTOR_VERBOSE_ERRORS=1 for stacks), cold-start one-liner.

## KNOWN NUMBERS (measured today, actual-25 96w)
startup ~50min (belief calc; memoized hereafter; phase is memory-
bandwidth-bound ~24-32 effective cores/box); training ~90-150s/iter
(load/N holds: ~10-14k core-s/iter; 500-core abstraction: ~35-50s/iter
+ serial floor <10-20s); worker ~1.4G RSS; eval phases measure 20+
times (eval-measurement cost unprofiled — driver py-spy it if slow).

## OPEN / WATCH
- Stop behavior at 25: prior runs exited iter ~30 prematurely (REL vs
  inflated honest-init, MC=1 noise, grounded-gate timing — full
  analysis in session log). The hold should carry it further; belief
  scale was still ~19.5k (NO_ROUTE penalty territory) at iter 33.
- Eval assert: assess_volume_multipliers ValueError = undertrained adv
  stranding traffic under inflation. If it fires again with a properly
  trained adv — investigate as a real bug.
- Task chip pending: KeyError (vtrtokyo,9824) in solve_lp_assignment
  sort lambda (one-off worker failure; latent popp map mismatch).
- v5 scout CLOSED: findings synthesis in session log (L6c wins lat
  2.13/no blowups; L6 slotted least stable 9 blowups incl seed-202
  cluster; L4 high-variance; rmsprop>adagrad; probing frequency is a
  STABILITY knob). Consider experiments/ablation/V5_SCOUT_FINDINGS.md.
- Future perf: delta evaluation of near-identical candidates (order-of-
  magnitude on belief phase), eval-phase measurement profiling,
  generic_objective/ug_perfs arrayify for >96w, serial-floor async at
  500 cores.
- 32 launch: after a VALID 25 result. Launcher ready; memo makes its
  restarts cheap; PROBE_TCONV consider ~real horizon.

## HARNESSES (use these; they caught 2 real bugs today)
- experiments/desharding/prove_inert.py — bitwise gate (self-pins hash)
- scratchpad ab*.sh pattern — paired-seed statistical A/B
- experiments/lp_hotloop — fixture replay (correctness+timing)
- experiments/pattern_cache/bench.py — repr/timing bench
- experiments/startup_optimizations/NOTES.md — startup program record
