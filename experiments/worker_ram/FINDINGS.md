# Worker RAM/caching dense pass — findings (2026-08-20)

Evidence: head memprof at actual-25 (production, 80w) + isolated bench
(experiments/worker_ram/bench.py) at small/decent + cache-stats smoke
(SCULPTOR_CACHE_STATS) at actual-5.

## Measured attribution (per worker, actual-25 production)
- parent_tracker: 93 -> 354MB (grows with measurements; string-tuple
  dict keys; replicated identically in ALL 80 workers ~= 28GB/box)
- calc_cache.lb: ~190MB — 99% of it is KEYS (get_a_cache_rep tuples,
  105.7KB/entry at decent; values are compact)
- pattern_cache: ~148MB (post-compaction)
- process baseline ~520MB; total ~2GB plateau

## Ranked opportunities
1. [IMPLEMENTED, validating] parent_tracker: CSR-by-parent int32 +
   plasma ship (SCULPTOR_COMPACT_PT). ~12B/entry vs ~100B, one copy
   per node vs per worker, miss-path touches only active parents.
2. lb-cache keys -> np.packbits(threshold_a(a)).tobytes(): 490x
   smaller (221B vs 105.7KB at decent), 32x faster to build. Makes
   MAX_CACHE_SIZE=8000 cost ~2MB instead of ~800MB worst-case, so the
   cap can rise and the full-flush in check_clear_cache can become
   LRU. (Cache-stats smoke: cross-measurement-epoch key reoccurrence
   is only ~13%, so retention ACROSS clears is not worth building —
   this is about within-epoch capacity + key bytes.)
3. rti_data transient churn: ~20MB fresh alloc per latency_benefit
   call at decent (choices_matrix int64 8.6MB -> int16 2.2MB; P_matrix
   f64 8.6MB -> f32 4.3MB; meta_data 24k tuples 2.8MB rebuilt/call).
   Reuse buffers + narrow dtypes => ~4x less churn; churn drives the
   glibc-arena RSS growth (workers 520MB -> 2GB plateau).
4. var_pool / persistent LP model: 223k columns after 150 calls with
   only 23k active (90% cold). Periodic model compaction (rebuild
   dropping cold columns) bounds a structure that reaches 100k-2M at
   scale. Care: rebuild loses HiGHS basis warmth for one solve.
5. MALLOC_ARENA_MAX=2 on workers (Linux): free ~10-15% RSS for
   many-threaded long-lived processes.

## Explicitly rejected
- LRU/retention for Calc_Cache dicts across measurement clears:
  reoccurrence ~13% (actual-5 smoke) — the optimizer moves on; the
  clear-everything policy is approximately right there.

## Eval-tail program (2026-08-20 afternoon) — SHIPPED
- SCULPTOR_EVAL_VOLSCEN=1: diurnal/flash/vol-mult scenarios ship as
  vol/cap vectors; rti computed once per adv; persistent LP with
  per-scenario RHS updates. 7.7x on eval recompute at small (77s->10s);
  diurnal was 70% of the size-20 eval tail (2206s of 3064s).
- ug-sentinel pricing (auto with volscen): congestion-touched ugs
  priced exactly NO_ROUTE_LATENCY, killing vertex-dependent mixture
  leakage into the ==sentinel aggregation filter. Validated: diurnal
  curves match legacy exactly; 25/4176 scalars differ >1e-6 (degenerate
  taint-set edges).
- LATENT LEGACY BUG documented: the flash/diurnal aggregation silently
  EXCLUDES congested ugs (exact-sentinel filter), so legacy curves are
  flat under volume surges. Metric redesign (volume-weighted congested
  fraction) is a paper-level decision — not changed here.
- Eval phase mem/time markers ([mem] tag=eval_*) permanent.
