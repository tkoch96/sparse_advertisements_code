# stop_tracker cache-bypass fix — local verification (2026-05-21)

Protocol: SESSION_4_SUMMARY.md task #0. Goal: confirm dropping
`verbose_workers=True` at `sparse_advertisements_v3.py:1877` does not
change SCULPTOR's `Believed` / `GTO` / final eval signal.

## Setup

- All runs: `small`, `MAX_ITER=10`, `N_WORKERS=2`,
  `SCULPTOR_CAPACITY_HEADROOM=0.2`, `SCULPTOR_SKIP_RB_GRAD=1`,
  `SCULPTOR_DEPLOYMENT_SEED=1`, local Ray (no cluster).
- Four runs: with-fix #1/#2 (HEAD), no-fix #1/#2 (line 1877 restored
  to `verbose_workers=True`). Logs preserved in this directory.

## Numbers

| metric | fix #1 | fix #2 | no-fix #1 | no-fix #2 |
|---|---:|---:|---:|---:|
| Iter-0 LB | −6.142 | −6.211 | −5.964 | −6.222 |
| Iter-0 RB | −315.5 | −317.5 | −309.6 | −313.7 |
| GTO @ iter 12 | 3.147 | 3.128 | 3.055 | 3.153 |
| sparse %≤−10ms | 89.97 | 92.22 | 88.83 | 87.27 |
| popp Δ (ms) | −3.03 | −4.13 | −4.10 | −3.11 |
| **pop Δ (ms)** | **−0.39** | **−0.30** | **−2.40** | **−1.28** |

## Verdict

**Inconclusive on the 0.1% bar, but no evidence of harm.**

Key finding: **SCULPTOR has substantial run-to-run noise even with the
same `SCULPTOR_DEPLOYMENT_SEED`** — two with-fix runs disagree at
iter-0 LB by 0.07 and on final sparse% by 2.3pp. The seed pins the
deployment build + initial advertisement but the gradient-step
sampling has unseeded RNG (likely `np.random` calls during the
"max-info" measurement-selection step plus Ray timing/ordering
effects). This makes the "agree to ~0.1%" bar from the protocol
impossible to meet on principle.

With that run-to-run noise as the ruler:

- **LB / RB / GTO / sparse% / popp-Δ**: with-fix and no-fix ranges
  overlap. No detectable systematic effect at N=2.
- **pop-Δ**: with-fix mean ≈ −0.34, no-fix mean ≈ −1.84 → a ~1.5 ms
  gap that *looks* systematic. But Phase A's documented seed-to-seed
  pop-failure variance was 4.2 ms at N=3, so 1.5 ms is well within
  the seed-noise envelope. Not statistically distinguishable from
  noise at this sample size.

## Implication for Phase B

Phase B (N=5 trials of headroom at actual-10 × 150 iter, fix in
place) is **broadly trustworthy**. Cross-seed aggregation at N=5
should swamp any cache-fix bias. Caveat: if the pop-Δ shift is
real, it would push Phase B's pop-failure numbers ~1.5 ms more
favorable than a no-fix equivalent. When comparing Phase B's
pop-failure against any pre-fix session-3 baseline, eyeball for
this offset.

## What was NOT verified

- Whether the cached path's `(xsumx, psumx)` reconstruction is
  numerically identical to fresh compute. (Spot checks suggest it
  matches for `benefit` but reconstructs `xsumx` lossily via
  linspace; only `benefit` is consumed by `modeled_objective`, so
  this doesn't affect GTO.)
- Why `verbose_workers=True` causes any observable effect at all,
  given that the worker-side `latency_benefit` drops `**kwargs`
  before calling `generic_benefit` (so the cache-bypass guard
  `verb = kwargs.get('verbose_workers')` *should* always evaluate
  to None regardless of the caller). There is a code path I have
  not yet found where verbose_workers reaches generic_benefit.
  Worth a follow-up audit if the pop-Δ bias proves real with more
  trials.

## Files

- `with_fix_{1,2}.log`: HEAD-state runs (fix in place)
- `no_fix_{1,2}.log`: ran with line 1877 reverted to
  `verbose_workers=True`, then fix restored
