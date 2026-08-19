# HANDOFF: evaluate_over_deployment_sizes @ size 25 ONLY

Written 2026-08-19 ~03:30 UTC by the outgoing session. Goal (Tom):
"let L6 rip at a large problem and see how it does with our classical
evaluations (eval_latency_failure.py) which we do nicely through
evaluate_over_deployment_sizes.py" — dpsize 25 ONLY, all current fixes.

## Code state (all in main, HEAD ~ef4b413)

Merged & validated:
- HiGHS backend via gpshim (SCULPTOR_LP_BACKEND=highs is THE backend;
  OBJ_ROUND=4 standard).
- rmsprop (bias-corrected, beta=0.9) is the DEFAULT step policy for
  direction-on rungs (L5/L6); SCULPTOR_ABLATION_GRAD_SCALE=adagrad
  restores legacy.
- stop-v2 bundle: honest init, REL=0.001, IMP=0, trend clause
  (TREND_EPS=1e-3). Runs ride while believed objective declines.
- depsetup array loader: load_actual_perfs auto-routes when
  SCULPTOR_LAT_SHARDS points at shards (~5x on fresh pop-set builds;
  byte-exact gated 5/10/16/20/26 pops). Shards exist on BOTH VMs at
  <repo>/cache/lat_shards.
- Slotted WHEN (L6) in mainline (SCULPTOR_PROBE_MODE=slotted etc.).
- Lexicographic prio pair + obj_components + iters_to_{50,90,95}
  persisted per ablation cell (run_fork_ladder only — the EODS path
  does NOT persist these).
- Queue harvest fallback for timestamped run dirs.

DEPLOYED: head (107.22.173.189) repo AND sweep VM (32.197.41.137)
~/smoke_repo both carry this exact state as of this handoff.

NOT merged (open threads, do not block on them):
- experiments/reducing_iteration_timing: persistent-inner-LP candidate
  — actual-15 A/B RUNNING on head in ~/rit_repo + ~/rit_out_a15
  (COPY repo; do not confuse with the main repo). Verdict = compare
  "Timer: grads" means base vs persist. If persist wins big, consider
  it for the EODS run only AFTER a byte-exactness/quality gate.
- prio_lex is EVAL-side only; training still joint_latency_bulk.

## The task

evaluate_over_deployment_sizes.py at dpsize 25 ONLY, standard NSIM
(Tom's earlier spec used 10 sims at size 25 — CONFIRM), all soln
types, HiGHS, full new-code defaults. The interesting arm is sparse
(SCULPTOR full = L6 semantics in mainline; probe mode via env if the
driver exposes it — VERIFY what evaluate_over_deployment_sizes runs by
default before launching).

## Hardware & known hazards (READ ALL)

- dpsize=25 OOM history: OOM-killed the sparse driver on a 64GB head.
  Current head has 185G; sweep VM (32.197.41.137, r8g.8xlarge) has
  247G and is the SAFER choice once the v5 grid drains (~morning).
  Watch driver RSS; disable parallel soln types if RAM climbs
  (SCULPTOR_DISABLE_PARALLEL_STRATEGIES=1 pattern).
- Startup at size 25 is HOURS single-core (profiler: >3h at a20+).
  The depsetup array loader (needs SCULPTOR_LAT_SHARDS=<repo>/cache/
  lat_shards in env) cuts the load_actual_perfs share ~5x. Startup
  silence 5-15+ min while pickling deployment to workers is NORMAL.
- Deployment caches: pruned_performances_<pops>.pkl and
  actual_deployment_cache_<pops>_seed<k>.pkl under cache/deployments
  make repeat runs fast BUT keys lack a world fingerprint — for stock
  world this is fine; do NOT reuse across world-knob changes.
- SCULPTOR_DEPLOYMENT_SEED determines the 25-pop subset (random
  choice); pin it and record it.
- Tracebacks from sparse_advertisements_v3 top-of-file legacy code are
  NON-FATAL noise; don't alarm on them.
- pkill on these boxes: ALWAYS bracket patterns (run_fork[_]ladder)
  AND never put the target string un-bracketed anywhere in your own
  command line (paths included) — four self-kill incidents on
  2026-08-18 alone. Use separate kill/verify SSH sessions.
- Cluster alerting contract: update
  ~/.sculptor_cluster_alert/active_cluster.json on lifecycle events.
- Cost: both VMs are on-demand; tear down / stop what you finish with
  (Tom's standing rule: always tear down at the end).

## Concurrent state to respect

- Sweep VM runs the v5 grid tail + L4nd (top-5 flip redo) + L6c
  (conservative WHEN: TCONV=650, SURPRISE_THETA=0.08) queues until
  ~morning. Do NOT start EODS-25 there until
  `pgrep -c -f run_n_sweep` returns 0 (or coordinate slots).
- Head runs the rit a15 A/B (~/rit_out_a15). Leave it; harvest its
  verdict into experiments/reducing_iteration_timing/README.md.
- Mac loops: dash refresh/sync/ticker/profiles_pull + 30-min monitor
  cycles (scratchpad night_monitor.sh pattern). Dash:
  http://107.22.173.189/ (basic auth). "The dash is the source of
  truth" — monitor RESULTS CONTENT, not just processes.

## Suggested run shape (validate, then commit to it)

1. Smoke: dpsize 25, NSIM=1, single soln type (sparse) on the sweep
   VM post-drain; confirm startup completes, RSS profile, one full
   training + eval_latency_failure round trip.
2. Full: NSIM per Tom's answer (default 10), all soln types,
   sequential-strategies if RAM demands. Use a queue/manifest pattern
   (run_eods_cell speaks run_fork_ladder CLI; evaluate_all_metrics is
   resumable via its pickle).
3. Dash tab for it (house pattern: pull + steps + figures; see
   grid_v5scout wiring in experiments/dashboard/generate.py).
4. 30-min monitors; RAM is the failure mode to watch at 25.
