# Worker-side perf investigation (path_distribution_computer)

Goal: identify time-saving opportunities in the inner-loop worker code
(`sim_rti`, `total_rti_calc`, `pmat_organize`, `solve_generic_lp_persistent`,
`solve_unified_lp_not_optimize`). The wider context: at `actual-32` scale
SCULPTOR's per-iter wall is dominated by python orchestration around the
LP, not by Gurobi itself (`optimize` is only ~0.5% of LP-call wall).

## Methodology

1. **Scaffold** (`tests/test_worker_perf.py`): drives `_LocalPathDistribution
   Computer._cmd_calc_compressed_lb` directly with real deployments from
   `deployment_setup.get_random_deployment(dpsize)` and real initial
   advertisements (`init_advertisement`). Parametrized over dpsize ∈
   {small, decent}, seed ∈ {1, 2}, n_perms ∈ {1, 32, 132}.
2. **Correctness gate**: `test_correctness_calc_compressed_lb_reproducible`
   compares two cache-cleared calls' fingerprints (sha256 of sorted
   (job_id, rounded-benefit, xsumx bounds, sparse psumx) tuples). Must
   pass before any optimization is taken seriously.
3. **Timing accumulator**: previously `self.timing[k] = time.time() - ts`
   *overwrote* per LP call; converted to `+=` and reset at start of each
   `calc_compressed_lb` batch. `summarize_timing()` now prints true
   per-batch cumulative per-key breakdown.
4. **PYTHONHASHSEED=0** at pytest launch: makes dict/set iteration order
   stable across runs so the fingerprint is meaningful for pre/post
   comparison (otherwise hash randomization changes LP column order →
   FP-tiny eddy in benefit values → different fingerprint).

## Baseline (decent, n_perms=132, MC_NUM=5 → 660 LP calls/batch)

```
cold wall: 347 s          warm (cache hit): 0.05 s

per-key cumulative (nested timers):
  solve_generic_lp_persistent  332 s   95%   outer LP wrapper
    optimize                   103 s   30%   Gurobi solve
    solve_unified_lp_not_opt   113 s   33%   LP setup (UB churn, obj set, col add)
    get_paths_by_ug             13 s    4%   path enumeration
    organizing_results           4 s    1%   raw_x → vols_by_poppi / lats_by_ug loop
    (outer wrapper python)     ~99 s   29%   obj_coeffs build + raw_x extract + ...
  sim_rti / total_rti_calc      15 s    4%   per-prefix loop + vectorized MC sim
  pmat_organize                  4 s    1%   subset of sim_rti
```

At `actual-32` scale (per the cluster worker log): `optimize` falls to
~0.5%, sim_rti / total_rti_calc grow to ~58%, python wrapper grows to
~32%. Python overhead scales worse than Gurobi.

## Investigation results

### Optimization 1 — track `_last_active_vars`, only deactivate those

**Hypothesis:** `setAttr("UB", all_vars, [0.0] * len(all_vars))` in
`solve_unified_lp` zeroes UB on the *entire* `var_pool` (grows to >10k at
actual-32). Tracking the previously-active set and only zeroing those
should cut the LP-setup time substantially.

**Implementation:** stored `self._last_active_vars` at end of each call;
at start of next call zeroed only those. Activate-or-discover loop and
active-vars-build loop merged into one pass.

**Result:** ❌ **NET LOSS at decent scale** (one-shot bench):
- `solve_unified_lp_not_optimize`: 113s → 62s (−51s, as expected)
- `optimize`: 103s → 137s (**+34s, unexpected**)
- cold wall: 347s → 372s (+25s slower)
- correctness fingerprint matches → LP results identical

**Conclusion:** the all-vars UB-zeroing isn't pure waste. It appears to
interact with Gurobi's warm-start basis carryover or some internal
bookkeeping such that *not doing it* costs more in `optimize()` than it
saves in setup. Without diving into Gurobi internals (closed-source), I
can't pinpoint why. The relationship is robust enough that the
inferred gain disappears reliably.

**Status:** REVERTED. This setup-side optimization is not speedupable in
isolation; reducing UB churn requires a paired Gurobi solver hint.

### Optimization 2 — batch `getAttr("X", active_vars)` for raw_x extraction

**Hypothesis:** `raw_x = {key: var.X for key, var in self.var_pool.items()
if var.X > 1e-7}` iterates *all* `var_pool` entries calling per-var `.X`
(a Gurobi API access). Replacing with `getAttr("X", active_vars)` (batched
read on the small active set) should eliminate the var_pool-size penalty.

**Initial measurement:** ran the bench with a cold/empty `var_pool` →
inconclusive (savings appeared real but were within run-to-run noise at
that scale). Diagnosed the cause: an empty var_pool doesn't expose the
overhead this optimization targets.

**Bench-realism fix:** added a 5-batch warmup before the measured call
so `var_pool` reaches representative production size. **Measured**:
- decent: 0 → 228k entries
- actual-32 (projected): 0 → ~0.5-2M entries

Far bigger than I'd guessed (I'd said 30k). The original
`for var in var_pool.items() ... var.X` loop was doing 228k × Gurobi
C-calls per LP solve at decent, × 660 LP solves = ~150M `.X` calls per
batch — easily the bulk of the outer wrapper time.

**Result with warmed-up bench:** ✅ **NET WIN at decent**:
- cold wall: 141.55s → 131.92s (−9.6s, −6.8%)
- solve_generic_lp_persistent: 133.9s → 123.0s (−10.9s, the targeted savings)
- correctness fingerprint matches (`c0075d78e6593b32` both pre+post)
- run-to-run noise on `optimize` and `solve_unified_lp_not_optimize` is
  ~±20s but the cold wall net is solidly negative

Projected at actual-32: var_pool is 5-10× larger → savings scale
proportionally → ~50-100s saved per `calc_compressed_lb` batch.

**Status:** ✅ **SHIPPED in commit `90110ea`.**

### Run-to-run noise at decent scale

Even with `PYTHONHASHSEED=0`, single-pytest-run timing of identical code
varies by ~5-15% across pytest invocations (system load, JIT effects,
memory allocator state). To get a usable signal on a < 15% optimization,
need N≥3 runs per configuration and ~30+ minutes of total bench wall.

## What's clearly NOT speedupable in isolation

| candidate | status | reason |
|---|---|---|
| Multi-Scenario Gurobi API | NOT WORTH | `optimize` is 0.5% of wall at actual-32; addressing it is rounding error |
| Reduce setAttr("UB", all_vars) churn | NOT (alone) | Cuts python 50%, Gurobi pays back +30%. Net loss. |
| Per-call `var.X` reads in raw_x | LIKELY YES | Replaceable with `getAttr("X", active_vars)`. Inconclusive without N≥3 bench. |
| `sim_rti` per-prefix loop | UNCLEAR | Cache already handles repeated states. Hot for novel states only. |

## What IS speedupable (proven, already shipped)

| change | scale | speedup | commit |
|---|---|---|---|
| `ray.put` static deployment ctx once | start_workers @ N=64 | ~10× (95s → 10s) | `ddd51f7` |
| Skip `resilience_benefit` value under headroom | stop_tracker | 18s → 0.02s | `6fcb689` |
| Skip RB-grad consolidated into headroom env | gradient loop | full RB-grad cost → 0 | `e17814a` |
| Drop `verbose_workers=True` from stop_tracker | (historical) | cache-hit path now active | `219f078` |
| Self.timing accumulator (visibility) | benchmarking | 0% wall, makes optim. work tractable | `26a04e2` |

## Recommended next steps

1. **N≥3-run bench harness:** repeat opt 2 with proper statistical
   sampling. Likely confirms ~80s outer-wrapper savings at decent →
   ~150-400s at actual-32 depending on var_pool size.
2. **Vectorize `obj_coeffs` build** in `solve_generic_lp_persistent`:
   pre-compute a 2D `ug_perfs_arr[ug_idx, popp_idx]` once at deployment
   update; replace the per-path dict-of-dict lookup with numpy gather.
   ~15% of LP wrapper time.
3. **`sim_rti` per-prefix loop's `parent_tracker` iteration** (lines
   638-644): pre-compute `popp_to_blocked_users` mapping when
   parent_tracker updates. Saves O(parent_tracker_size) per cache miss.
4. **Eval-phase `start_workers` repetition:** each eval sub-phase
   (volume-mult, diurnal, flash-crowd) calls `start_workers` afresh.
   With the `ray.put`-static refactor each call is ~10s instead of ~95s,
   so this is no longer painful, but a side-task could pool actors
   across sub-phases.

## Scaffold + accumulator are the lasting contribution from this round.

Future optimization PRs should:
1. Run `pytest tests/test_worker_perf.py -k correctness` first (must
   pass).
2. Run `PYTHONHASHSEED=0 pytest tests/test_worker_perf.py::test_bench_
   calc_compressed_lb[132-1-decent] -v -s` N≥3 times pre+post, average,
   check the per-key breakdown for the targeted savings.
3. Only merge if cold wall drops by >2σ of the noise floor (~30s at
   decent).
