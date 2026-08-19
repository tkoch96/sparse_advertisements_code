# Desharding survey (Tom 2026-08-19: "there's no UG sharding across workers
# — remove the references; is it comments? unused objects?")

## Verdict: it is LIVE machinery shipping (probably) UNUSED data, plus
## misleading comments. Not just cosmetic.

What the code actually does (verified 2026-08-19):

1. GRADS are what's distributed. flush_latency_benefit_queue_generic
   (sparse_advertisements_v3.py:531) splits the LB-call queue across
   workers (split_seq(lb_args_queue[1:], n_workers)) and CONCATENATES
   results — no cross-worker reduce. Each worker answers its jobs over
   the FULL deployment (whole_deployment_* attrs). Tom's mental model
   ("4k grads -> 500 workers x 8 each") matches the code.

2. Nevertheless start_workers ships per-worker UG SLICES:
   split_deployment_by_ug (helpers.py:144) slices ug_perfs / ug_to_vol /
   ug_to_bulk_vol / ug_to_ip / ug_anycast_perfs / ingress_priorities per
   worker; whole_deployment_* keys ride along as replicated statics.
   The pdc's heavy arrays are built from whole_deployment_* (e.g.
   generic_benefit intersects against self.whole_deployment_ugs, pdc
   matrices from whole_deployment_ug_perfs) — the sliced keys look
   functionally inert for gradient compute.

3. The reshard-on-resize machinery (worker_comms_ray.py request_add/
   remove_workers, ~150 lines) exists mainly to re-slice this data.

## Inventory of references (worker path only; fleet/shard.py and
## depcache lat_shards are DIFFERENT, legitimate concepts — keep):
- helpers.py: split_deployment_by_ug, split_deployment_by_ug_separated
- optimal_adv_wrapper.py:1019-1050 + optimal_adv_wrapper_ray.py:907-938
  (send subdeployments)
- worker_comms_ray.py: 27 refs (split, ship, reshard on pool resize)
- path_distribution_computer_ray.py:48-74 (actor takes subdeployment)
- path_distribution_computer.py: subset_ugs/which_ugs branches in
  generic_benefit (EVAL-time flows pass explicit ug lists — verify
  before touching; flash-crowd/diurnal evals use which_ugs)

## Why it matters beyond cosmetics (RAM):
- The replicated whole_deployment_* statics are the un-shardable
  per-worker data floor. split_deployment_by_ug_separated already
  ray.put()s statics ONCE (plasma object store). Plasma gives
  ZERO-COPY sharing for numpy arrays on the same node — but
  whole_deployment_ug_perfs is dict-of-dicts, which plasma must
  deserialize per worker (full copy each). Converting the static
  structures to numpy arrays (the depsetup_fork array discipline,
  extended to runtime) would let N workers share ONE physical copy →
  the <0.5G/worker goal becomes achievable without changing compute.

## Proposed sequence (do NOT hot-swap under the live cell):
1. memprof run confirms slice keys' share + whole_deployment share
   (rides on next cell, SCULPTOR_WORKER_MEMPROF=1).
2. Patch A (safe, mechanical): stop slicing — ship {'ugs': all_ugs}
   + static ref to every worker; delete reshard machinery; fix
   comments. Gate: byte-identical grads on a small deployment.
3. Patch B (the RAM fix): arrayify whole_deployment statics ->
   plasma zero-copy. Gate: same + RSS-per-worker before/after.
