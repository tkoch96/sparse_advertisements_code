# depsetup_fork — array-native deployment creation

Phase 3 of the deployment-creation speedup program (see
`experiments/depcache/README.md` for phases 1-2). Tom's directive
(2026-08-18): fork deployment_setup's expensive path and keep parsing
sharded (by pop / ug / ingress) until the LAST possible second.

## Key facts discovered scoping this

- For `actual-N` sizes `DO_UG_CLUSTERING = False` — **Birch clustering
  is not in the path at all**. The pipeline is: shard/CSV parse →
  per-(ug,popp) min → single-ingress + ≤1ms filters → anycast
  intersect → SOL physics filter → best-popp → `do_filter` (~15 UGs
  per ingress, RNG quota selection) → small final dict.
- The final object IS small (Tom's observation): do_filter caps to a
  few thousand UGs. The expensive artifact is the INTERMEDIATE
  dict-of-lists holding every raw measurement (~50M floats + list
  objects at 26-32 pops) — everything downstream consumes only
  `np.min(lats)` per (ug, popp).
- Ordering matters for exactness: `get_intersection` returns
  hash-ordered lists and `do_filter` shuffles lists built in dict
  iteration order, so the RNG outcome depends on dict insertion
  order. The fork therefore mirrors the original's insertion order
  exactly (pops sorted, first-appearance row order), which makes the
  whole downstream — including RNG draws — byte-identical within a
  process.

## B1 (this directory)

`fast_perfs.build_ug_perfs_min`: vectorized per-pop groupby-min
(sort + `np.minimum.reduceat`; `minimum.at` is ~100x slower) over the
depcache npz shards, materializing per-(ug,popp) MIN latencies as
np.float64 SCALARS in original insertion order. Installed via the
existing `SCULPTOR_LAT_SHARDS` seam (monkeypatches
`shard_loader.build_ug_perfs`); mainline `load_actual_perfs` runs
unmodified — its min loop degenerates to `np.min(scalar)` no-ops.
Clamp-then-min == min-then-clamp (monotone), so clamping the reduced
minima preserves `parse_lat` semantics bit-for-bit.

Gate: `python -m experiments.depsetup_fork.gate_5pop` — sandboxed
CACHE_DIR (copies addresses_violating_sol.csv, which the SOL stage
rewrites; symlinks all read-only inputs), same seed + same process for
both arms, demands byte-exact nested equality INCLUDING key order.

## B2 (`fork_load.py`) — RESULTS 2026-08-18

Full array-native load_actual_perfs: arrays end-to-end, dict
materialization only at the final survivor set. Byte-exact gate PASS
at 5 pops AND 10 pops (keys, key order, values bitwise, RNG stream).

| pops | orig (Mac) | B2 | speedup |
|------|-----------|-----|---------|
| 5    | 70-144 s (cache-warmth variance) | ~40 s | 1.7-3.6x |
| 10   | 192-382 s | ~90 s | 2.2-4.2x |

Gap widens with pop count (orig scales superlinearly, B2 ~linearly).
B1 is superseded (can even lose to a warm-cache orig).

B2 10-pop stage profile (s): parse_min 13, filter_1ms 4, anycast 2,
sol 22, best_popp 18, do_filter 30 — the remaining fat is key-level
python comprehensions (~854k-element), worth another ~2x if needed.

### Exactness war stories (why the gate is strict)

- CPython `set()` PRESIZES its hash table for a dict argument but
  grows incrementally for a list/iterator; identical key sequences
  then yield DIFFERENT set iteration orders (adjacent-swap
  divergences at 10 pops, invisible at 5). get_intersection args must
  therefore be dicts exactly as mainline passes them
  (`dict.fromkeys(keys)`).
- `np.random.shuffle` on a same-length ndarray consumes the identical
  MT19937 stream as on a list (verified) — quota selection reproduces
  exactly.
- mainline's SOL stage SKIPS the physics check for any UG already in
  addresses_violating_sol.csv (0 or 1) and only physics-checks new
  UGs; the fork replicates this including the file rewrite.

### Clustering note (Tom asked about removing it)

Birch clustering is NOT in the actual-N path (`DO_UG_CLUSTERING =
False`) — there is nothing to remove; it only runs for the RIPE
prototype sizes. The wins here came from killing the raw-measurement
dict-of-lists and the python tail loops.

### Next

- Bench at 26 pops on the head (orig ~650 s known) for the scaling
  point that matters to EODS/fleet.
- If adopted: wire behind an env seam (e.g. SCULPTOR_DEPSETUP_FORK=1)
  in deployment_setup so cache-MISS builds use it; cache-HIT paths
  are unaffected. Pre-baked AMI caches remain the zero-risk
  alternative for repeated pop-sets; this fork is the win for FRESH
  pop-set builds (exactly the EODS fleet pattern).

## Hazards

- The per-(pops) caches (`pruned_performances_*.pkl`,
  `actual_deployment_cache_*`) make all of this a once-per-host cost;
  AMI pre-baking those caches remains the zero-risk alternative for
  fleet scale (depcache README, phase 3 option A).
- Cache keys lack a world fingerprint (existing hazard, still open).
