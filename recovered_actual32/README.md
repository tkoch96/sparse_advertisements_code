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
