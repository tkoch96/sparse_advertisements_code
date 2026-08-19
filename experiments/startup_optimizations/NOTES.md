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

## 2026-08-19 late: memo zero-iter ROOT CAUSE + fix
Load-arm crash = AttributeError popp_to_users in first gradient call —
IDENTICAL root cause to the historical hot-start bug (memory:
popp_to_users). It's derived state from the bootstrap's LP solve
(save_ug_ingress_decisions), skipped by both hot-start and memo-load.
FIX (in, gated): payload includes popp_to_users + _ensure_popp_to_users
lazy-rebuild guard at both kill-sampling readers (one LP solve when
absent) — the guard also repairs classic hot-starting. 10-seed
load-vs-compute A/B rerunning.

## Miss-path vectorization spec (pdc ~879-902, flamegraph 23% of cold solves)
Precompute once per worker: CSR of ui->potential popps
(flat_pops uint16, offs int32 over whole_deployment uis).
Per miss column: active = bool[n_popp] from packed key;
mask = active[flat_pops]; parent-blocked pairs (small at bootstrap;
from parent_tracker) applied as sparse corrections to mask;
lens = np.add.reduceat(mask, offs[:-1]); entries = the masked flat
array segmented -> exactly the compact (uis, lens, pad) entry the
cache now stores, built with the same vectorized padding as the
sampler. Zero per-UG python. Gate: bitwise (prove_inert) + the
pattern_cache bench for entry-content equality + timing.
