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
