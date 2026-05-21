# Session 4 summary (2026-05-20 → 2026-05-21)

Starts where OVERNIGHT_SUMMARY.md (session 2) and Session-3 left off. Goal
of this session: validate the headroom approach as a paper-ready alternative
to the SGD-RB gradient, plus explore a more rigorous stochastic-LP gradient
as a third option. Also uncovered a major perf bug along the way.

Read in order: CLUSTER_RUNBOOK.md → OVERNIGHT_SUMMARY.md → this file →
RESEARCH_ROADMAP.md.

## TL;DR

- **Headroom (`SCULPTOR_CAPACITY_HEADROOM=0.2 SCULPTOR_SKIP_RB_GRAD=1`) is
  the current frontrunner** for replacing the SGD-based resilience benefit
  gradient. Per N=3 trials at `small` × 50 iter, it consistently beats
  classical RB-grad on normal-LP quality (96-98% vs 90% within −10ms of
  optimal) and ties or beats it on popp-failure resilience. Pop-failure
  results are noisier across seeds (range ~4 ms with N=3).
- **A proper stochastic LP gradient (Option A from the roadmap) is wired
  up** but doesn't beat headroom in single-trial testing. See
  `stochastic_lp.py` + `tests/test_stochastic_lp.py`.
- **Major perf bug found and fixed:** the `verbose_workers=True` kwarg in
  `stop_tracker`'s call to `modeled_objective` was disabling the worker LP
  cache, costing ~20 s/iter at actual-10 (an 800× slowdown on that one
  call). One-line fix landed in commit at HEAD.
- **Phase B running on cluster as of EOS:** N=5 trials of headroom at
  `actual-10` × 150 iter, with the cache fix. ETA ~45 min from launch
  (relaunched 02:44 UTC).
- **No teardown yet** — cluster still has head + (until Phase B finishes)
  one spot worker. Tear down at end with `./teardown.sh`.

## What's working

- Code state on `main` is at HEAD with all session-4 commits (no
  uncommitted changes).
- 19/19 unit tests pass (`pytest tests/test_stochastic_lp.py -m "unit and not slow"`).
- Cluster operational; `ray up ray-cluster.yaml` from scratch works in one
  shot.
- New runner script: `benchmarks/headroom_n_trials.sh <dpsize> <max_iter> <max_concurrent> <seeds...>`.
- Cross-seed plotter: `benchmarks/cross_seed_phase_a_plot.py`.

## What this session changed (code)

1. **`stochastic_lp.py`** (NEW, ~400 lines): solver for the
   scenario-based stochastic LP using Gurobi's Multi-Scenario API.
   Methods: `multi_scenario` (one optimize() call with K scenarios baked
   in via per-scenario Var.ScenNUB; default), `warm` (sequential warm-start),
   `cold` (rebuild model per scenario). Also includes
   `solve_headroom_lp`, scenario factories
   (`single_popp_scenarios`, `single_pop_scenarios`,
   `pick_gradient_step_scenarios`, `pick_gradient_step_scenarios_importance`),
   and `compute_unroutable_volume_fractions`.

2. **`tests/test_stochastic_lp.py`** (NEW, 19 tests): correctness
   (degenerates to nominal, per-scenario matches standalone, feasibility,
   weighted-objective consistency, warm≡cold, multi_scenario agrees with
   warm); 3-approach comparison (headroom 1-LP vs stochastic K-warm vs
   RB-grad-approx K-cold); resampling test; importance-sampling tests
   (cold-start uniform, concentrates on high-impact, unbiased estimator).

3. **`sparse_advertisements_v3.py`** edits:
   - Added `gradients_stochastic_lp` method on `Sparse_Advertisement_Solver`.
   - Added `SCULPTOR_USE_STOCHASTIC_LP_GRAD=1` env var that swaps in the
     new gradient as `gradients_resilience_benefit_fn`.
   - Added `SCULPTOR_STOCHASTIC_LP_K` (default 16) and
     `SCULPTOR_STOCHASTIC_LP_IMPORTANCE=1` env vars.
   - **Bug fix**: dropped `verbose_workers=True` from
     `stop_tracker`'s `modeled_objective(advertisement, ...)` call at
     line 1877. This was disabling the worker's LB cache (cached path
     deliberately skips when `verb` is truthy). ~20 s/iter saved at
     actual-10.

4. **`benchmarks/`**:
   - `sculptor_3way_runner.sh`: 3-condition (A=RB-grad, C=headroom,
     D=stochastic-LP) runner, also supports E (D + importance sampling).
   - `headroom_n_trials.sh`: N-trials × concurrency-capped runner.
     Used for Phase A (N=3 small) and Phase B (N=5 actual-10).
   - `cross_seed_phase_a_plot.py`: aggregates per-seed metrics pickles
     into one CDF PDF.
   - `stochastic_lp_study.py`: standalone speedup + convergence study.
   - `out/phase_a_pdfs/`: Phase A outputs (per-seed pickles +
     `phase_a_cross_seed_comparison.pdf`).

## What this session found (results)

### 100-iter 4-way comparison at `small` (single seed = 1)

| | iters | t/iter | normal Δ | popp-fail Δ | pop-fail Δ |
|---|---:|---:|---:|---:|---:|
| A (RB-grad) | 99 | 4.5 s | −2.37 | −2.90 | **−0.36** |
| **C (headroom)** | 102 | **2.4 s** | **−0.71** | **−1.02** | −1.79 |
| D (stoch LP uniform K=16) | 75 | 5.1 s | −1.48 | −1.32 | −3.37 |
| E (stoch LP + IS K=16) | 79 | 6.6 s | **−20440** | +2.17 | −1.74 |

(Δ = `best_perf − new_perf`, ms; closer to 0 = better. one_per_peering is the 0 reference for failure metrics.)

- **C is best on normal-LP and popp-failure** by a wide margin.
- **A is best on pop-failure** (closest to 0 at −0.36).
- **D doesn't dominate any column** at this seed.
- **E diverged catastrophically** because importance sampling pushed
  per-iter gradient variance way up; SCULPTOR's modeled objective looked
  great (355) but the actual advertisement was junk. See `IS variance
  issue` below.

### N=3 trials of headroom at `small` × 50 iter (Phase A)

| seed | normal-LP %≤−10ms | normal Δ | popp-fail Δ | pop-fail Δ |
|---|---:|---:|---:|---:|
| 1 | 96.78 | −0.54 | −1.46 | −4.06 |
| 2 | 96.66 | −0.75 | −2.35 | −1.57 |
| 3 | 97.76 | −0.80 | −2.06 | +0.14 |
| **mean** | **97.07** | **−0.70** | **−1.96** | **−1.83** |
| range | 1.1 pp | 0.26 ms | 0.89 ms | **4.20 ms** |

- Normal-LP and popp-failure: **very tight cross-seed cluster, much
  better than painter** (painter: 70-79% % within −10ms, −0.5 to −5 ms Δ).
- **Pop-failure has high variance** across seeds (4.2 ms range). With
  N=3 this is not enough to make a confident claim either direction.

### Phase B (in flight at EOS): N=5 trials of headroom at `actual-10` × 150 iter

Relaunched after cache-bypass fix. ETA ~45 min wall (one trial × ~43 min,
others run in parallel). Per-seed deployment sizes vary substantially:
seed=2 drew 2959 UGs (~2.4× the others), so its trial wall is closer to
80-90 min. Results to be aggregated when complete; see `/tmp/cluster_runs/headroom_n/`
on the cluster head.

### IS variance issue (Option A's gradient signal)

When the K=16 scenarios are sampled with probability ∝ failure-impact
(importance sampling), each sample's IS weight is `p_uniform/p_imp`. For
rare scenarios that happen to be sampled, this weight can be substantial
(e.g. ~0.69 for one sample when `p_imp=0.001` with K=16). That single
sample then dominates the per-iter expected_latency estimator and the
finite-difference gradient SCULPTOR is using. Over many iters, the
gradient signal effectively bounces between "minimize THIS rare scenario"
and "minimize THAT rare scenario", producing a model that thinks it's at
a great minimum (modeled objective 355) but whose actual routing is
broken (0.5% of traffic within −10ms of optimal).

The math IS unbiased on average (verified by
`test_importance_sampling_estimator_is_unbiased`). The problem is
per-iter variance.

Possible fixes left as future work:
- Self-normalized importance sampling (SNIS): divide by Σweights. Biased
  but bounded variance.
- Clip the IS weights (cap at e.g. 2× the uniform weight).
- Higher `exploration_weight` (try 0.5 instead of 0.1) — mostly uniform
  with mild importance bias.
- Use the FULL scenario set (no sampling): no IS needed; just expensive.

### Cache-bypass perf bug (the big find)

`path_distribution_computer.py:791`:
```python
if not verb and not subset_ugs:
    # cache lookup — FAST return if hit
    ...
```

`sparse_advertisements_v3.py:1877` (BEFORE fix):
```python
self.current_pseudo_objective = self.modeled_objective(advertisement, verbose_workers=True, verbose=True)
```

The `verbose_workers=True` propagated to the worker's calc_compressed_lb
and forced it to skip the cache lookup. Result: every `stop_tracker`
call did a full LP recompute, ~20 s/iter wasted at actual-10. After fix:
- That call drops from 20.5 s → ~25 ms (verified by `[Timing]` prints
  on a Phase B trial running the OLD code; the matching
  `modeled_objective(threshold_a(advertisement))` call on line 1881
  was always at 25 ms because it doesn't pass `verbose_workers`).
- Per-iter at actual-10 should drop from ~37 s → ~17 s.

This bug had been silently doubling per-iter cost across every SCULPTOR
run in sessions 1-4. Anything pre-fix in `OVERNIGHT_SUMMARY.md`'s timing
table is overstated by ~2×.

## Open questions / hypotheses to test

1. **Does Phase B (actual-10 N=5) reproduce the cross-seed normal-LP
   tightness?** Phase A at small showed 96-98% — a fragile-looking
   narrowness. Could be deployment-class-specific.

2. **Pop-failure variance**: at small, N=3 had range 4.2 ms across seeds.
   Need either more seeds or a different metric (e.g. worst-case across
   pop failures rather than mean).

3. **Can the IS approach be saved?** SNIS + weight clipping + higher
   exploration_weight should all be tried before declaring Option A dead.
   Cheap experiments since the infrastructure exists.

4. **The cache-bypass fix changes ground truth.** Pre-fix runs in
   sessions 1-3 were ALSO affected (the same line wrote junk timings to
   `actual_nonconvex_objective`). Worth re-running Session-3's A vs C at
   actual-10 with the fix to see if conclusions change.

5. **Is `gradients_resilience_benefit_pop`** worth re-enabling for pop
   failure? Currently `pop_failure` weight is alpha=0 (see line 1245);
   the comment says it hurts convergence. With the cache fix, retest.

## Suggested next steps (priority order)

1. **Wait for Phase B to finish, pull results, generate cross-seed CDF
   PDFs** with `benchmarks/cross_seed_phase_a_plot.py` (or its
   actual-10-flavored variant). Report on
   normal+popp+pop-failure aggregates.

2. **Run Phase A (small × N=3) AGAIN** with the cache fix applied, to
   verify the headroom claims hold under correct timing. The
   correctness of the eval metrics shouldn't have changed (cache fix
   only sped up `modeled_objective`), but it's worth verifying.

3. **Try SNIS for the importance-sampling variant of stochastic LP** and
   re-run a 100-iter at small. If SNIS gets E to look like a reasonable
   competitor to D and C, the math is back on track.

4. **Headroom sweep** at small × N=3: try
   `SCULPTOR_CAPACITY_HEADROOM` ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}.
   Find the lowest headroom that still beats painter on normal load.
   Cheap (~30 min wall on cluster).

5. **Tear down cluster** (`./teardown.sh`).

## Known issues / caveats

- **The 4-way comparison (A/C/D/E) was on a single seed (=1)** at small.
  Need more seeds before any algorithmic claim is robust.
- **The original sign convention got flipped in my initial reporting**
  — corrected mid-session. The eval-phase metric is
  `mean(best_perf − new_perf)` per UG, so 0 = matches per-UG-best and
  more negative = worse. Closer to 0 is better.
- **Importance sampling didn't pan out** as a drop-in. The math is right
  but the per-iter variance destroys SCULPTOR's gradient signal. The IS
  unit tests pass; SCULPTOR-loop integration is what fails.
- **The cluster runs in `runs/` and `figures/` on the cluster head are
  raced** by concurrent trials (multiple trials write to the same
  filenames). Per-seed metrics pickles ARE properly tagged. If you want
  per-seed plots, work from the pickles.

## Critical commits this session (newest first)

```
8ede47e  Drop verbose_workers=True from stop_tracker's modeled_objective  <-- the perf fix
...      Headroom N-trial runner with concurrency cap
...      Refactor: parallel arrays, don't capture pid via subshell
ad04180  Gurobi Multi-Scenario API implementation + test agreement
66181cd  Expand comparison test: nominal+popp+pop latency deltas, cap-overflow flag
...      Three-approach comparison test + headroom helper
...      Stochastic LP solver + tests + study (synthetic + actual-N)
ca2ec97  One-seed slim follow-up dispatcher with ntfy.sh push   <-- last session-3 commit
```

## Files that matter

- `stochastic_lp.py` — the new solver
- `tests/test_stochastic_lp.py` — the unit tests
- `sparse_advertisements_v3.py` — modified gradients_resilience_benefit_fn
  swap + the `verbose_workers` fix in `stop_tracker`
- `benchmarks/headroom_n_trials.sh` — the runner used for phase A & B
- `benchmarks/out/phase_a_pdfs/` — Phase A outputs
- `ray-cluster.yaml` — `max_workers: 5` from session 3

## How to read the current Phase B results when it finishes

```
# On cluster (or after `ray rsync-down`):
ls /tmp/cluster_runs/headroom_n/hd_seed*_actual-10_*.log

# Per-seed eval table (replace timestamp):
for s in 1 2 3 4 5; do
  log=$(ls -t /tmp/cluster_runs/headroom_n/hd_seed${s}_actual-10_*.log | head -1)
  tr '\r' '\n' < "$log" | grep "Average latency difference sparse" | head -2
done

# Aggregate plot:
python benchmarks/cross_seed_phase_a_plot.py  # currently small-flavored;
                                              # tweak SEEDS + BASE + path
```
