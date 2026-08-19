# pattern_cache representation bench (2026-08-19)

pattern_cache memoizes per advertisement-column pattern (tuple-of-bool
over popps) the full routing state: for every UG, valid ingress popps
(after parent_tracker blocking) + uniform probs. Vital (miss path =
parent_tracker scan + per-UG filtering, the pmat_organize cost) but the
repr is Python-object soup and unbounded between probe-driven clears —
the dominant worker-RAM growth at dpsize 25 (~3.3G/iter aggregate).

Bench (real cache content via the production miss path, 200 distinct
patterns, 14318 (ug,pops) pairs, tiny deployment — per-PAIR constants
are scale-free and extrapolate):

| scheme                          | per pair | vs current |
|---------------------------------|----------|------------|
| A current list[(ui,pops,probs)] | 264.5 B  | 100%       |
| B drop probs (always 1/n)       | 157.1 B  | 59%        |
| C CSR per entry (int32)         | 21.6 B   | 8.2%       |
| D C + np.packbits key           | 19.8 B   | 7.5%       |
| D2 D + uint16 pops              | 14.9 B   | **5.6%**   |
| E global arena + packed key     | 20.4 B   | 7.7%       |

Hit-path reconstruction (rebuild the exact current tuples): 8.8x slower
relative, ~20us/entry absolute — noise vs the ~100ms+ LP call it feeds.
Better: consumers read the arrays directly (rti_data assembly), skipping
reconstruction entirely.

RECOMMENDATION: D2. At the observed ~2.1G/worker pattern-cache-dominated
growth, D2 => ~120MB/worker: decouples worker RAM from probe spacing,
removes the need for aggressive probe-driven clears (better hit rates),
and is a prerequisite-remover for the many-light-workers plan.
Implementation: encode on miss directly (no list intermediate), iterate
arrays on hit; gate with the desharding-style bitwise parity check
(PYTHONHASHSEED pinned) before adoption.

Keys: even packed, keys are per-distinct-column; key count == entry
count. The arena (E) wins only when entries shrink further (delta
encoding vs base pattern — future work; gradient candidates differ from
base in 1-2 popps, so entries are near-duplicates and delta encoding
could take another order of magnitude. Not attempted here.)

## 2026-08-19 AT-SCALE VALIDATION (head, testing_feature-actual-25, seed 1)
217 distinct patterns = 946,120 (ug,pops) pairs (avg ~4360 ugs/entry,
~40 pops/ug):

| scheme | total | per pair | vs current |
|--------|-------|----------|------------|
| A current | 850.48M | 942.6B | 100% |
| B noprobs | 471.06M | 522.1B | 55% |
| C csr | 158.34M | 175.5B | 18.6% |
| D2 csr+key+uint16 | **82.27M** | 91.2B | **9.7%** |

- Storage theory CONFIRMED: 217 patterns = 850MB explains the multi-GB
  worker growth; D2 = 10.3x at true scale.
- Current repr degrades SUPERLINEARLY vs tiny-scale extrapolation
  (942 vs 265 B/pair): large ints not interned, unlike small-int-cached
  tiny runs. Tiny-scale benches UNDERSTATE the win.
- CAVEAT vs tiny-scale conclusion: naive hit-path reconstruction is
  12.7ms/entry at scale (~380ms per 30-column lb call) — NOT viable.
  Adoption = D2 storage + ARRAY-NATIVE consumer (rti_data assembly in
  get_ingress_probabilities/sim_rti reads CSR directly). Contained
  refactor, parity-gated; likely faster than today's list appends.
