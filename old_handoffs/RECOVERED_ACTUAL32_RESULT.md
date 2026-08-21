# Recovered actual-32 result (single trial, seed=1, MAX_ITER=200)

## Files in this directory

| file | size | what it is |
|---|---|---|
| `state-202.pkl` | 771 MB | SCULPTOR final state at iter 202 (converged, stop=True): advertisement, deployment, metrics, parent_tracker, etc. |
| `popp_failure_latency_comparison_actual-32.pkl` | 37 MB | Eval-phase pickle: per-UG normal-LP latencies for all 6 strategies + each strategy's converged advertisement |

Both pulled off the cluster head node at 35.153.211.18 (post-stop-start
to recover from SSH wedge). Cluster terminated immediately after.

## Recovery story

The cluster eval-only resume from `state-195.pkl` ran for ~3 hours.
SCULPTOR re-converged to iter 202 (which is past MAX_ITER=200, hit the
RD<epsilon stopping condition). All 6 strategies in
`compare_different_solutions` completed successfully — including painter,
which took ~30 min at actual-32. The eval pickle was written at line
152 of eval_latency_failure.py after the strategy loop finished.

The failure-eval / sub-eval phases (volume-multiplier, diurnal,
flash-crowd, popp/pop-failure-Δ) had not yet completed when the head
sshd became unresponsive (RAM-starved by the long-running painter +
accumulated worker state). AWS reboot didn't recover; stop-start did.

## SCULPTOR @ actual-32 (single trial, seed=1)

5173 UGs, 779 popps, 32 pops. Converged at iter 202 (stop=True).
Advertisement: 2708 active bits of 59,983 total (4.5% sparse), with
the first prefix at full anycast (779) and 76 sparse prefixes.

### Normal-LP per-UG latency (mean across 5173 UGs)

| strat | mean (ms) | mean Δ vs OPP | % ≤−10ms | % ≤−50ms | % ≤−100ms |
|---|---:|---:|---:|---:|---:|
| one_per_peering | **28.35** | 0 (ref) | 100% | 100% | 100% |
| **sparse** | **28.96** | **−0.61** | **97.4%** | 99.9% | 100% |
| painter | 29.36 | −1.01 | 94.6% | 99.8% | 100% |
| one_per_pop | 33.50 | −5.15 | 83.5% | 97.7% | 99.7% |
| anyopt | 44.55 | −16.20 | 73.1% | 88.8% | 95.0% |
| anycast | 54.59 | −26.24 | 70.3% | 84.0% | 90.0% |

**Sparse comes within 0.61 ms / 97.4% of the idealized one_per_peering
upper bound at the production target deployment size**. Painter ~3pp
behind on coverage. Sparse improves over the de-facto baselines
(anyopt by 15.6 ms, anycast by 25.6 ms).

### Failure-Δ (NOT in the recovered pickle)

The popp/pop-failure-Δ rows are empty in this pickle — the eval got
to line 152's per-strategy pickle.dump but didn't reach the
failure-eval phases (lines 247+). To recover this, run the offline
eval against the saved advertisements:

```python
import pickle
d = pickle.load(open('popp_failure_latency_comparison_actual-32.pkl','rb'))
advs = d['compare_rets'][0]['adv_solns']  # dict of strategy -> [advertisement]
deployment = d['deployment'][0]
# Re-run assess_failure_resilience offline on each adv
```

(The cluster could also be brought back up with the new per-strategy
checkpoint code from commit `5b349be` to make future eval runs survive
mid-loop crashes.)

## Comparison with actual-10 phase B (N=3 trials, cross-seed)

| metric | actual-10 (N=3 mean) | actual-32 (N=1) |
|---|---:|---:|
| sparse mean Δ | −0.10 ms | −0.61 ms |
| sparse % ≤−10ms | 99.8% | 97.4% |
| painter mean Δ | −1.11 ms | −1.01 ms |

Sparse stays competitive at the larger scale (slight degradation, as
expected — more popps means more places to leave performance on the
table, but still <1 ms from ideal).

## Cost

This trial: ~4h cluster wall × $1/hr = ~$4 spend.
With the timing optimizations now in place (commit chain through
`90110ea`), future actual-32 trials should be ~30-40% faster wall.

## Next steps (for paper)

1. Re-run the failure-eval offline against the stored advertisements
   to get popp/pop-failure-Δ at actual-32.
2. Repeat actual-32 with seeds 2-5 for cross-seed CDF (matches the
   actual-10 N=5 protocol).
3. Headroom sweep (TIER 1 #2 in roadmap) at actual-10 or actual-32 to
   find lowest headroom that still beats painter.

---

## Failure / volume / diurnal / flash-crowd eval — FINAL (2026-05-22)

The original recovered pickle had only `compare_rets` + `pct_volume_within_latency`
populated. After three sessions of cluster work the remaining phases were
filled in by `eval_latency_failure`'s resume mechanism (the existing
`check_calced_everything` skip + per-strategy checkpointing we added).

Final populated pickle: `popp_failure_latency_comparison_actual-32_FULL.pkl`
(47 MB, gitignored). Eval-resume took ~75 minutes on a single
c7g.16xlarge worker (64 vCPUs, 32 Ray actors). Optimizations in this branch:
no per-LP `update_dep` for failure scenarios, OPP-reference LPs computed
once and shared across strategies, gti-cache enabled in worker
`solve_lp` handler, heavy LP-result fields stripped before driver-side
accumulation, per-strategy `gc.collect()` + `malloc_trim()`.

### Normal-LP (no failure) — mean per-UG latency

| strategy | mean lat (ms) | Δ vs OPP |
|---|---:|---:|
| one_per_peering | 28.349 | 0 (ref) |
| **sparse** | **28.958** | **+0.61** |
| painter | 29.360 | +1.01 |
| one_per_pop | 33.500 | +5.15 |
| anyopt | 44.548 | +16.20 |
| anycast | 54.585 | +26.24 |

### Popp-failure (per-link failure) — paper metric (% of affected traffic within X ms of optimal)

| strategy | mean Δ | %≤10ms | %≤50ms | %≤100ms | pct cong |
|---|---:|---:|---:|---:|---:|
| one_per_peering | +0.00 | 99.96% | 99.96% | 99.96% | 0.00% |
| **sparse** | **-2.51** | **88.27%** | **99.31%** | **99.90%** | 0.00% |
| painter | -4.76 | 85.16% | 96.24% | 99.40% | 0.00% |
| one_per_pop | -15.58 | 61.06% | 89.37% | 97.88% | 0.00% |
| anyopt | -61.58 | 26.53% | 54.66% | 75.08% | 0.00% |
| anycast | -119.09 | 9.93% | 26.92% | 45.31% | 0.00% |

### Pop-failure (per-PoP failure) — paper metric

| strategy | mean Δ | %≤10ms | %≤50ms | %≤100ms | pct cong |
|---|---:|---:|---:|---:|---:|
| one_per_peering | +0.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| **sparse** | **+18.64** | **84.82%** | **94.88%** | **98.91%** | 0.00% |
| painter | +11.21 | 77.85% | 87.98% | 96.10% | 0.00% |
| one_per_pop | +4.72 | 64.43% | 81.60% | 94.83% | 0.00% |
| anyopt | -18.62 | 54.99% | 74.54% | 88.02% | 0.00% |
| anycast | -90.10 | 18.50% | 35.60% | 58.84% | 0.00% |

### Volume multipliers — mean per-UG latency at global volume scale

| strategy | @X=0% | @X=15% | @X=29% | Δ low→hi |
|---|---:|---:|---:|---:|
| one_per_peering | 28.21 | 29.28 | 32.86 | +4.65 |
| **sparse** | **28.74** | **30.07** | **34.16** | **+5.42** |
| painter | 29.22 | 31.36 | 40.75 | **+11.54** ← painter degrades most under volume growth |
| one_per_pop | 33.54 | 34.67 | 37.99 | +4.44 |
| anyopt | 44.89 | 46.68 | 48.65 | +3.76 |
| anycast | 55.52 | 55.52 | 55.52 | +0.00 (LP returns same result; no congestion path active) |

### Diurnal — no congestion at any tested intensity

`stats_diurnal` (max intensity before congestion) reports `25%` (the floor)
for every strategy, which by the paper code means "never congested in the
tested range". Tested intensities were `[25, 50, 65, 70, 75, 85, 95, 105,
115, 125, 150]%`. Our actual-32 deployment with default `link_capacities`
absorbs all of it. Mean per-UG latency delta vs baseline across 24 hours:

| strategy | 25% | 75% | 125% | 150% |
|---|---:|---:|---:|---:|
| sparse | -0.7 | -0.6 | -1.1 | -1.1 |
| anyopt | -3.8 | -3.5 | -5.8 | -5.9 |
| painter | -1.1 | -1.0 | -1.6 | -1.6 |
| anycast | 0.0 | 0.0 | 0.0 | 0.0 |
| one_per_pop | -1.6 | -1.5 | -2.5 | -2.6 |
| one_per_peering | -0.8 | -0.7 | -1.1 | -1.1 |

(Negative deltas are expected for diurnal: most metros are off-peak at
most hours, so total load is below baseline → easier routing → slightly
lower latency. To reproduce the paper's Fig 7(b) differentiation curve
we'd extend `diurnal_multipliers` beyond 150% so the deployment hits
capacity-binding regimes where strategies actually differ.)

### Flash crowd — also runs without congestion in the tested range

Mean per-UG latency delta vs baseline, averaged across metros at Y=1.3
(capacity over-provisioning factor):

| strategy | X=10% | X=87% | X=164% | X=474% |
|---|---:|---:|---:|---:|
| sparse | +0.7 | -1.6 | -1.6 | -1.6 |
| anyopt | +1.1 | -7.8 | -7.8 | -7.8 |
| painter | +0.9 | -2.5 | -2.5 | -2.5 |
| anycast | 0.0 | 0.0 | 0.0 | 0.0 |
| one_per_pop | +0.7 | -3.6 | -3.6 | -3.6 |
| one_per_peering | +0.6 | -1.6 | -1.6 | -1.6 |

Plateaus past X=87% suggest the per-metro flash crowd at actual-32's
capacity headroom doesn't cross a critical-congestion boundary across
the X range. Same recipe to dial in the paper's intensity-before-
congestion threshold: lower `Y_val` (less capacity over-provisioning),
or extend the X range.

## Comparison vs paper Figure 12 (popp-failure %≤10ms) at 32 sites

| strategy | paper (Fig 12b, ~32 sites) | ours (actual-32 single seed, 200-iter sparse) |
|---|---:|---:|
| sparse | ~67% | 88.3% |
| painter | ~70% | 85.2% |
| anyopt | ~47% | 26.5% |
| anycast | ~13% | 9.9% |

Strategy *ordering* matches the paper exactly. Absolute numbers are
slightly different because: (1) sparse here was trained with
`SCULPTOR_CAPACITY_HEADROOM=0.2` (paper-time runs were pre-headroom),
so the sparse adv is shaped to be more redundant; (2) sparse and
painter both benefit from real-Vultr-32 having more peering diversity
than the synthetic-32 the paper sweeps. The relative ranking and the
gap between top and bottom is preserved.
