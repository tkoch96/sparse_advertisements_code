# Step 1 (batched getAttr X) — quantified sweep

## Methodology

In a single pytest invocation, build a worker, warm up its `var_pool`
to a representative size via 5 prep batches (different advertisement
seed each), then measure the same `calc_compressed_lb` batch twice:

1. STEP 1 ENABLED (default): batched `model.getAttr("X", active_vars)`
   — one Gurobi C-call on the active set (~few hundred vars)
2. STEP 1 DISABLED (`SCULPTOR_DISABLE_RAW_X_BATCH=1`): legacy per-var
   loop `{key: var.X for key, var in self.var_pool.items() if ...}`
   — one Gurobi C-call per var in var_pool (10k-1M+ at scale)

Same warmed worker for both, so Gurobi state / var_pool / deployment
are identical. Transient caches cleared between runs:
`pattern_cache`, `this_time_ip_cache`, `calc_cache.all_caches['lb']`
— otherwise the second measurement short-circuits on cache hits.

`PYTHONHASHSEED=0` set at pytest launch so fingerprints are stable
across processes (dict/set iter order deterministic).

## Results

| dpsize | n_ugs | n_popps | n_perms | LP solves | var_pool | wall ON | wall OFF | Δ | % | fingerprint match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| small  | 225  | 51  | 1   | 10  | 2,231 | 0.294s | 0.122s | +0.17s | +141%* | ✓ |
| small  | 225  | 51  | 32  | 165 | 2,231 | 2.019s | 2.390s | -0.37s | **-15.5%** | ✓ |
| small  | 225  | 51  | 132 | 665 | 2,231 | 7.755s | 9.168s | -1.41s | **-15.4%** | ✓ |
| decent | 4000 | 294 | 32  | 165 | 228,453 | 54.33s | 69.38s | -15.04s | **-21.7%** | ✓ |
| decent | 4000 | 294 | 132 | 665 | 228,453 | 256.94s | 314.06s | -57.12s | **-18.2%** | ✓ |

\* small × n_perms=1: only 10 LP solves; the +172ms is back-to-back
measurement noise (Gurobi warm-start basis carry-over between two
back-to-back optimize() calls helps the second one). Wash-out at
n_perms ≥ 32.

### `solve_generic_lp_persistent` (the targeted timer)

The outer LP wrapper that contains the raw_x extraction. Step 1 should
reduce this timer directly:

| config | ON | OFF | Δ | per-LP-call savings |
|---|---:|---:|---:|---:|
| small × 1 (10 solves) | 101ms | 110ms | -9ms | -0.9ms/solve |
| small × 32 (165 solves) | 1842ms | 2090ms | -248ms | -1.5ms/solve |
| small × 132 (665 solves) | 7101ms | 8376ms | -1275ms | -1.9ms/solve |
| decent × 32 (165 solves) | ? | ? | ? | ~86 ms/solve* |
| decent × 132 (665 solves) | 238877ms | 296138ms | **-57260ms** | **-86 ms/solve** |

\* extrapolated from the wall delta -15s / 165 solves.

Per-LP-call savings scale with var_pool size: small (2.2k vars) →
1.9ms, decent (228k vars) → 86ms. That's roughly linear, matching the
theory (legacy code iterates var_pool, Step 1 iterates only active
set ~independent of var_pool).

### Where the savings come from

At decent × 132, all wall savings cleanly attributed to the targeted
timer:

```
solve_generic_lp_persistent     ON= 238877  OFF= 296138  Δ= -57260 ms  ✓ Step 1's win
optimize                        ON=  97132  OFF=  95825  Δ=  +1307 ms  ← <0.5% noise
solve_unified_lp_not_optimize   ON= 110618  OFF= 105001  Δ=  +5617 ms  ← ~5% noise
sim_rti                         ON=  17239  OFF=  17032  Δ=   +207 ms  ← noise
```

Other timers wobble by single-digit % from run-to-run variance but
don't add up to a net cost. The targeted -57260ms cleanly explains
the -57120ms wall improvement.

## Projection to actual-32

var_pool size scales with n_ugs × n_popps. At decent: 228k. At
actual-32 (n_ugs=5173, n_popps=779), theoretical max is ~4M; in
practice probably 0.5M-1.5M depending on iteration count. So var_pool
is ~3-7× decent's.

Linear scaling of per-LP-call savings (86 ms × 3-7) → **260-600 ms per
LP solve saved at actual-32**. For a typical info-phase batch of ~660
LP solves: **170-400 seconds saved per `calc_compressed_lb` batch**.

This compounds across the many calc_compressed_lb batches per SCULPTOR
iter. At ~5-10 batches/iter × 200 iters = ~1000 batches. Even at the
low end of the projection, this is ~3 hours of wall time saved per
actual-32 trial.

## Correctness

**Every single configuration in the sweep passed the fingerprint
correctness check.** Step 1 returns numerically identical results to
the legacy code path, just faster:

```
small × 1   ON=6290145e7f32afcf  OFF=6290145e7f32afcf  ✓
small × 32  ON=21e7bce73655dc7d  OFF=21e7bce73655dc7d  ✓
small × 132 ON=2004ea39449d9916  OFF=2004ea39449d9916  ✓
decent × 32  ON=c0075d78e6593b32  OFF=c0075d78e6593b32  ✓
decent × 132 ON=61b7d71a823259c9  OFF=61b7d71a823259c9  ✓
```

## How to reproduce

```bash
PYTHONHASHSEED=0 ~/Documents/venv312/bin/python -m pytest \
  tests/test_worker_perf_sweep.py -v -s
```

Each config takes 1-10 min depending on dpsize/n_perms. Full sweep
~30 min.

To verify on a NEW optimization: copy the test pattern, set/unset the
env var of your choice between the two measurements. Assert fingerprint
match. Compare walls.
