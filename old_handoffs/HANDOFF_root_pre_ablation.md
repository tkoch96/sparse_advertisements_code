> 📜 **HISTORICAL** — session-1 snapshot, pre-AWS / pre-Ray. For current
> state see [README.md](README.md) and the highest-numbered
> `HANDOFF_SESSION_*.md`. File:line references here are stale against
> current code (the ZMQ Worker_Manager / run_ray.py / hardcoded venv
> paths described below were all removed in the May 2026 Ray-only refactor).

# Handoff: scaling SCULPTOR to actual-32 in hours, not days

## ⚠️ Updated 2026-05-19 — read the session-2 docs first

This file (HANDOFF.md) describes the state **before** the overnight cluster
work of 2026-05-19. Most of the items in the "Short-term plan" section
below have now landed or been superseded by better paths.

**A new agent should read these three files first, in order:**

1. **[CLUSTER_RUNBOOK.md](CLUSTER_RUNBOOK.md)** — How to operate the AWS
   Ray cluster (which is now fully set up). Covers prereqs, the `ray up`
   command, monitoring patterns, teardown. Use as-is — `ray-cluster.yaml`
   in this repo is ready to run.
2. **[OVERNIGHT_SUMMARY.md](OVERNIGHT_SUMMARY.md)** — Empirical results
   from the cluster bring-up + first SCULPTOR runs on real hardware. Has
   per-iter timing data for `actual-10` and `actual-32`, a 6× speedup
   finding from a new in-LP capacity-headroom formulation, cost model for
   the 100-run paper grid.
3. **[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md)** — Prioritized next-steps
   plan, including the multi-trial quality A/B that needs to land to
   validate the headroom finding, plus algorithmic and operational TODOs.

Then HANDOFF.md (this file) for older context.

The rest of this file (below) is preserved verbatim for historical
reference. Tests are still healthy, persistent-Gurobi dispatch still
applies, but the "scaling to actual-32 in hours" goal is **already
achieved at the per-iter level** (16 min/iter on a single c7g.16xlarge
worker, or projected ~3 min/iter post-headroom optimization).

---

This document briefs a new Claude session on the state of the repo and the
work remaining. Goal: run SCULPTOR on the `actual-32` deployment size in a
couple of hours instead of a day or more.

## Repository orientation

Working dir: `/Users/tomkoch/Documents/sparse_advertisements_code/`.

The SCULPTOR algorithm lives in `sparse_advertisements_v3.py` (~2150 lines).
The main entry point users invoke is `eval_latency_failure.py`.

Workers (computing latency benefit and LP solutions) come in two flavors:
- **ZMQ subprocess workers (original):** `worker_comms.py` (`Worker_Manager`) +
  `path_distribution_computer.py` (`Path_Distribution_Computer`). The driver
  spawns N subprocesses and talks to them over `tcp://localhost:<port>`.
- **Ray actor workers (this session's addition):** `worker_comms.py` +
  `path_distribution_computer.py`. Same public API; Ray underneath.

A tiny launcher `run_ray.py` aliases `sys.modules['worker_comms']` to
worker_comms and `runpy`s the requested driver, so existing scripts
work under Ray without edits:

```
python run_ray.py eval_latency_failure --port 31415 --dpsize small
```

LP solving has two paths:
- **Persistent Gurobi** (`path_distribution_computer.py:solve_generic_lp_persistent`).
  Warm-starts across calls; ~10–20× faster than scipy for the MC inner loop.
- **scipy** (`solve_lp_assignment.py:solve_generic_lp` and friends). Used as
  fallback for non-supported objectives and for main-thread (non-worker) callers.

As of this session, `solve_generic_lp_with_failure_catch` dispatches
`avg_latency` and `per_site_cost` to the persistent Gurobi path when the
caller is a worker. Falls back to scipy otherwise.

## Tests (`tests/`)

Run with `pytest -v` from the repo root. Key files:

| File | What it covers |
|---|---|
| `test_smoke.py` | Worker constructs, handle_msg dispatches. |
| `test_lp_correctness.py` | Volume conservation, lats-by-ug consistency, Gurobi-scipy agreement, AND toy LPs with hand-verifiable answers (unlimited-capacity routing, zero-latency popp, tight-capacity overflow). |
| `test_warm_start_realism.py` | Strict, falsifiable warm-start test on the persistent Gurobi path. Asserts `iters(identical) < 5`. Also pins that `_cmd_solve_lp(avg_latency)` routes through Gurobi (not scipy). |
| `test_scaling_timing.py` | Per-LP solve time vs deployment size for `_cmd_solve_lp`. |
| `test_sas_iteration_timing.py` | End-to-end SAS.solve() wall-time per iteration. Projects to 200-iter production loop. |
| `test_convergence.py` | Falsifiable convergence claim: SAS's own modeled_objective is non-increasing over 3 iterations. Doubles as a Ray-stack smoke test. |
| `test_timing_baseline.py` | Throughput probe + dispatch overhead. |

Run subsets:
```
pytest -m unit                     # fast in-process tests
pytest -m "unit and not slow"      # exclude benchmarks
pytest -m "integration"            # Ray-based end-to-end
```

Markers in `pytest.ini`: `unit`, `integration`, `gurobi`, `slow`, `benchmark`.

## Things this session changed and why

1. **`worker_comms.py`, `path_distribution_computer.py`** — Ray-actor
   replacements for the ZMQ workers. Public API matches the originals so
   `sparse_advertisements_v3.py` doesn't need edits. The actor wraps a
   `_LocalPathDistributionComputer` that tests can instantiate directly
   without Ray.
2. **`run_ray.py`** — module-aliasing launcher. Picks the Ray backend without
   touching driver scripts.
3. **`_ActorSocketShim` in `worker_comms.py`** — exposes `.send(msg)` /
   `.recv()` on top of a Ray actor handle, since
   `sparse_advertisements_v3.py` calls those directly in `stop_tracker` and
   `set_iter` loops.
4. **`solve_lp_assignment.solve_generic_lp_with_failure_catch`** — now
   prefers `sas.solve_generic_lp_persistent` (persistent Gurobi) for
   `avg_latency` / `per_site_cost` objectives when called on a worker. This
   was the single biggest perf win in this session: ~5–15× faster per LP.
5. **`constants.py`** — `n_pops_from_dpsize` and `PRINT_FREQUENCY` now
   handle synthetic deployment sizes (`really_friggin_small`, `decent`,
   `med`, `large`). Previously they returned `None` and crashed downstream.

## Numbers we measured

### Per-LP solve time
| size | popps | persistent Gurobi (was scipy) |
|---|---|---|
| small | 45 | ~2.4 ms |
| decent | 272 | ~83 ms (was ~125 ms via scipy) |
| med | 1357 | ~160 ms (was ~297 ms via scipy) |
| large | 15289 | ~6.7 s |

### Full SAS.solve() per-iteration wall time (on user's Mac, ~8 Ray actors)
| size | per-iter | 200-iter projection |
|---|---|---|
| small | ~1.0 s | ~3 min |
| **decent** | **~465 s (~7.75 min)** | **~25.8 h** |
| med | not yet measured | — |

### Breakdown of one decent iteration (very important)
| Phase | Wall time | % of iter |
|---|---|---|
| `Timer: grads` (gradient computation) | **~280 s** | **60%** |
| Unattributed / other | ~140 s | ~30% |
| `Timer: info` | ~24 s | 5% |
| Exploration ("Measured perms", 52 perms) | ~20 s | 4% |
| `modeled_objective` | ~1.8 s | <1% |
| `measure_ingresses` (worker fan-out) | ~2 s | <1% |
| LP solving (inside the above) | small | — |

**The "grads" timer IS the LP work.** The gradient phase orchestrates
thousands of LP solves across MC samples × prefix entries × peer pairs.
Cutting per-LP wall-time directly cuts grads time. So the framing isn't
"grads vs LPs are different things" -- they're the same thing seen from
different layers. The good news: the LP fan-out is embarrassingly parallel.
Worker logs show 8 actors doing latency-benefit work concurrently at ~1.3 s
per LB-iter. With 100 cores, the grads phase should drop near-linearly.

The remaining ~140 s/iter ("unattributed" above) is also algorithm work
that scales -- algorithm-level loops around modeled_objective evaluations,
caching bookkeeping, exploration measure_ingresses. These are NOT the
binding constraint at this scale but they're not free either: 140 s × 200
iters = ~7.8 h on its own. They also benefit from worker parallelism for
the parts that fan out, and from cheaper per-LP solves for the rest.

### Implications for actual-32

User constraints (important): wants **200+ iterations**, does NOT want to
reduce gradient support N (ideally would increase it), and wants the run
to finish in hours.

Given those constraints, the only viable knobs are:
  1. Effective parallelism (more workers).
  2. Per-LP wall time (faster Gurobi, better caching).
  3. Algorithm-level loops outside the LP that can be sped up.

The Mac at 8 cores is the binding constraint. Scaling math:
- At 8 cores: 280 s/iter grads + 185 s/iter other ≈ 26 h for 200 iters on
  decent.
- At 100 cores (Gurobi WSL cap, perfectly parallel): grads drops ~12.5× to
  ~22 s/iter, other phases shrink mostly proportionally too -> per-iter
  ≈ 40-60 s, **200-iter ≈ 2-3 h on decent**.
- At `actual-32` scale: LPs are bigger (more popps) so per-LP wall time
  scales by maybe 2-3×. Even at 100 cores, expect ~3-8 h for 200 iters
  WITHOUT further LP-level speedups.
- To hit the 2-hour target on actual-32: 100 cores AND a 1.5-2× per-LP
  improvement (Gurobi parameter tuning, cache hit-rate, multi-scenario).
  Do not lower N -- the math works without it.

### Warm-start verification
Persistent Gurobi warm-start IS working in the MC inner loop. With the
*identical* rti repeated, simplex iterations drop to 0 (basis already
optimal). That's the speedup we re-confirmed in this session.

### Ray fan-out overhead
`measure_ingresses` is ~2 s at decent, ~10 s at decent with `Calculating RB
backups` (one-time setup). Pickling and message-passing dominate. Could
matter at larger scale; profile if you see it grow disproportionately.

## Outstanding production blocker

The user has not yet run the full eval at `actual-32` post-fixes. The
convergence and iteration-timing tests pass under Ray, so the stack is
healthy in principle — but a real `python run_ray.py eval_latency_failure
--port 31415 --dpsize actual-32` end-to-end run hasn't been verified yet.

That's the first thing the new session should do.

## Short-term plan (one or two sessions)

Given the measured breakdown -- **gradient computation is 60%+ of every
iteration and embarrassingly parallel** -- the highest-leverage moves are
ordered by what each can deliver:

1. **Run on a many-core machine (highest leverage).** The Mac's ~8 cores
   is the binding constraint right now. Same code on a 64- or 100-core box
   (AWS `c7g.16xlarge` or similar) should drop the grads phase by ~10×
   with zero code change. This single move likely brings decent from
   ~26 h → ~3 h and actual-32 from days → single-digit hours.

2. **Run actual-32 end-to-end with the current code and the dispatch fix.**
   Get a *real* per-iter number for actual-32 specifically, so we know how
   much further to push. Command:
   ```
   python run_ray.py eval_latency_failure --port 31415 --dpsize actual-32 2>&1 | tee actual32_run.log
   ```

3. **Profile the "grads" 280s and the ~140s "unattributed".** The grads
   phase is broken down internally in `sparse_advertisements_v3.py`. Add a
   `cProfile` pass around one iteration's grads call, OR dump worker
   `self.timing` mid-run -- each worker accumulates per-phase wall time
   (`optimize`, `get_paths_by_ug`, `organizing_results`,
   `get_ingress_probabilities_by_dict_generic`, `solve_generic_lp_persistent`,
   etc.). Identify whether the 280s is mostly LP, mostly Python, or split.
   The 140s unattributed needs explanation -- likely modeled_objective
   evaluations across many advertisements during gradient calc.

4. **Worker saturation check.** Add a counter in `worker_comms.py:_fanout`
   for time-waiting on Ray refs vs scheduling. If utilization is <60% on
   a big-core machine, the bottleneck is task distribution, not compute.

5. **Tune Gurobi parameters** (only after profiling says LP solve is a
   significant chunk):
   - `model.Params.Method = 2` (barrier instead of dual simplex). Sometimes
     2-3× on large LPs.
   - `model.Params.Presolve = 1` for cheaper presolve on small LPs.

6. **Email Gurobi about WSL seat bump.** Current cap is 100 concurrent
   processes. Academic researchers routinely get bumped to 500-1000. Free,
   email-and-wait, lasts the project. If you ever want to go beyond
   100-core single-node parallelism (which the actual-32 timing suggests
   you might), this becomes binding.

7. **Multi-scenario Gurobi API for batched LPs.** Many of the LPs called
   from `generic_objective_pdf` differ only in their `routed_through_ingress`
   (same adv, different MC sample). Gurobi's `setObjectiveN` /
   multi-scenario API can solve N parametrically-related LPs in one
   `optimize()` call faster than N separate calls. Worth investigating in
   `path_distribution_computer.py:solve_generic_lp_persistent` if profiling
   shows the persistent path is still the bottleneck after the parallelism
   wins.

8. **DO NOT lower gradient support N.** The user explicitly wants 200+
   iterations at current or larger N (it would hurt convergence). The math
   works at current N if items 1-3 above land -- pursue those before
   touching the algorithm.

## Long-term plan (multiple sessions)

1. **Cluster scale-out on AWS spot.** With the Ray backend in place, the
   missing piece is a cluster config:
   - `ray-cluster.yaml` specifying head + worker node types (e.g.,
     `c7g.16xlarge` for cost/efficiency on ARM).
   - `setup_commands` that install your venv + Gurobi + the codebase.
   - Spot instances; head node on-demand.
   Then `ray up`, `ray submit eval_latency_failure.py --dpsize actual-32`.
   Cost: roughly $0.05–0.10 per core-hour on spot. 100 cores × 2 hours ≈
   $10–20 per actual-32 run.
2. **Algorithmic improvements**, in rough order of expected payoff:
   - **Cache hit rate**: instrument `Calc_Cache` access patterns; the user
     estimated 50–80% hit rate, and pushing toward 90% is a direct 2×.
     The hot path is `latency_benefit` lookups during gradient steps.
   - **Smaller gradient support `N`**: configurable; reducing it linearly
     scales total LPs. Sweet spot trades convergence quality for wall-time.
   - **Better exploration**: 52 perms per iteration is substantial. If many
     perms produce near-identical LPs, deduplicate or batch them.
   - **Multi-scenario Gurobi API**: if your perms differ only in objective
     coefficients or RHS, `model.setObjectiveN` can solve many parametric
     variants in one `optimize()` faster than N separate calls.
3. **HiGHS port (only if Gurobi cap is the binding constraint).** Most
   production runs should fit comfortably under 100 Gurobi workers. Switch
   only if you see queueing on Gurobi license tokens AND scaling-up the
   WSL seat count isn't an option. Port lives in
   `path_distribution_computer.py:init_persistent_lp` + `solve_unified_lp`.
   Use `highspy`; the API is more index-based than Gurobi's but supports
   column pool + warm-starts. Budget: ~1 focused day to port and verify.
4. **Production correctness regression suite.** Right now the test suite
   covers LP invariants, warm-start, dispatch, and a 3-iter convergence
   smoke test. Add:
   - A multi-worker integration test (not just 1 worker).
   - Cross-version output comparison: pickle a `compare_different_solutions`
     output at a known seed, then re-run after refactors and diff.

## How to start the next session

Open Claude Code in this directory. The first message should be:

> I'm continuing work on a research codebase at this directory. Please read
> HANDOFF.md for the full briefing. I want to focus on [SHORT-TERM ITEM N]
> first. The goal is to get the actual-32 eval running in hours instead of
> a day. Don't make any code changes until I've reviewed your plan.

Then in subsequent turns: run the actual-32 end-to-end first, look at the
profile output, and decide whether the bottleneck is Gurobi, Python, Ray
serialization, or the algorithm itself. The right next step depends on
that profile.

## File-tree at a glance

```
sparse_advertisements_code/
├── HANDOFF.md                       <- you are here
├── README.md
├── pytest.ini
├── run_ray.py                       <- Ray-backend launcher
├── sparse_advertisements_v3.py      <- SCULPTOR algorithm
├── eval_latency_failure.py          <- primary driver
├── optimal_adv_wrapper.py           <- parent class for workers
├── path_distribution_computer.py    <- ZMQ worker (original)
├── path_distribution_computer.py <- Ray actor wrapper (this session)
├── worker_comms.py                  <- ZMQ Worker_Manager (original)
├── worker_comms.py              <- Ray Worker_Manager (this session)
├── solve_lp_assignment.py           <- LP dispatch + scipy paths
├── deployment_setup.py              <- random deployment builder
├── generic_objective.py             <- objective-fn wrapper
├── constants.py                     <- patched this session
└── tests/                           <- test suite (this session)
```
