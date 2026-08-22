# Deployment-creation speedup program (depcache + depsetup_fork), 2026-08-17..18

Dissolved 2026-08-21: the production code is now core/shard_loader.py,
core/fork_load.py and core/convert_latencies.py; the parity gates are in
unit_tests/. These two READMEs are the program's record.

---

# depcache — expedite deployment creation (2026-08-17)

Bottleneck (Tom's catch): `cache/vultr_ingress_latencies_by_dst.csv`
(4.3GB) is re-parsed row-by-row in Python for every NEW pop combination
(pruned_performances cache is keyed per pop set), worst at 32 pops
where the filter passes everything.

## Phase 1 (DONE)
- `convert_latencies.py`: one-time 8-way parallel CSV -> per-pop npz
  shards. 4.3GB -> 689MB / 26 shards in 155s. Global row order
  preserved (chunk-ordered merge).
- `shard_loader.py`: rebuilds the CSV loop's ug_perfs from shards.
  BYTE-EXACT gate: 0 mismatches over 793,706 ugs (2-pop subset).
- Mainline seam (deployment_setup, SCULPTOR_LAT_SHARDS=<dir>):
  empty-iterator guard, CSV fallback when shards absent.
- Measured: 2-pop subset 23.2s -> 5.5s (4x) + 6x disk. Small/mid pop
  sets win big (IO + skip-filtering); at 32 pops the loader's per-row
  python assembly converges toward CSV cost — see phase 2.

## Phase 2 (planned — the 32-pop killer)
- Vectorize the ug_perfs AGGREGATION: downstream consumers reduce the
  per-(ug,popp) lists; push that reduction into numpy (unique-key
  groupby over (ip_id, peer_id)) and gate against the
  pruned_performances output rather than raw lists.
- Parallelize per-pop shard loads across processes (shards are
  independent).
- Bake `cache/lat_shards/` into the fleet AMI (every VM skips the CSV
  forever); pre-bake pruned_performances for the standard pop sets.

## Related hazard (flagged, not yet fixed)
`actual_deployment_cache_<pops>_seed<k>.pkl` and
`pruned_performances_<pops>.pkl` keys ignore WORLD env knobs — same
hazard class as the 2026-08-17 shared-RNG catch. Any world-varying
campaign at actual-N should verify these caches are world-invariant
inputs (raw measurement prep: yes) vs world-dependent products
(deployment cache with randomized volumes: NO — needs world in key).

---

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

Gate: `python -m unit_tests.gate_5pop` — sandboxed
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

### MERGED INTO MAINLINE 2026-08-18 (Tom-ratified)

deployment_setup.load_actual_perfs routes to
fork_load.load_actual_perfs_arrays whenever SCULPTOR_LAT_SHARDS points
at available shards (exception -> legacy fallback).
SCULPTOR_DEPSETUP_ARRAYS=0 pins the legacy loop (the gate's baseline
arm does this). Bench: byte-exact 4.99x/5.09x/4.83x at 16/20/26 pops;
post-merge gate re-verified through the seam (3.03x at 5 pops).
Cache-HIT paths (pruned_performances / actual_deployment_cache) are
untouched; cache-MISS builds get the win — exactly the fresh-pop-set
EODS/fleet pattern. AMI cache pre-baking remains complementary.

## Hazards

- The per-(pops) caches (`pruned_performances_*.pkl`,
  `actual_deployment_cache_*`) make all of this a once-per-host cost;
  AMI pre-baking those caches remains the zero-risk alternative for
  fleet scale (depcache README, phase 3 option A).
- Cache keys lack a world fingerprint (existing hazard, still open).
