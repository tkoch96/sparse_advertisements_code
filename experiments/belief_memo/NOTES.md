# Initial-belief memoization (Tom 2026-08-19; designed, building next)

GOAL: persist the startup belief computation (~50 min at actual-25,
recomputed identically on every same-seed restart — 3x paid today) so
restarts load it in seconds.

PINNED (2026-08-19 evening):
- The expensive block is inside init_optimization_vars: the
  get_ground_truth_resilience_benefit call + the lazily-triggered
  worker "calcing latency benefits" batch (the 704 per-popp lb evals).
  Timeline evidence (s1v2 log): iov_post_gt_resilience_benefit stamps
  BEFORE the 71-min gap; solve_post_init_optim_vars stamps after —
  so part of the cost is lazy work triggered between the gt tag and
  iov exit (the modeled_objective calls' first-touch worker batches).
- OPEN QUESTION (correctness-critical): does the 71-min product live
  driver-side only (current_*_benefit + metrics + belief arrays), or
  does required state also land worker-side during the lazy batch
  (beyond rebuildable caches)? If worker-side state matters, restore
  must rebroadcast it or accept first-iter recompute per worker.
  Answer by diffing driver attrs + a worker census before/after the
  block on a small run.

DESIGN:
- Persist at solve_post_init_optim_vars time (after ALL lazy work):
  {current_objective, current_latency_benefit,
   current_resilience_benefit, current_pseudo/effective_objective,
   metrics tail, + whatever the open question adds}.
- Key: (deployment seed + pops fingerprint, init adv hash, MC_NUM,
  objective name, gamma, probe config, world knobs). Same no-world-
  fingerprint discipline as deployment caches — key them in.
- Load-or-compute guard in iov; SCULPTOR_BELIEF_MEMO=0 opt-out.
- GATES: bitwise parity (prove_inert-style: loaded-vs-computed run
  must produce identical first-iter grads on small), plus the n-seed
  A/B harness if any doubt remains.
