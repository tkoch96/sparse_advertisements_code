# Running `evaluate_over_deployment_sizes.py`

The deployment-size sweep: for each dpsize, train SCULPTOR plus every
baseline (painter, anyopt, anycast, one_per_pop, one_per_peering), run the
eval phases, and aggregate into the paper plots.

Written for the next agent. Everything here is verified against a real run
on 2026-08-21 unless it says otherwise.

---

## The short version

```bash
# local smoke -- minutes, proves the pipeline runs
python evaluations/evaluate_over_deployment_sizes.py \
    --dpsizes 3,5 --nsim 1 --max-iter 5 --port 31415 \
    --cache-fn cache/smoke.pkl --figures-subdir smoke

# on a VM, the full ladder (this is what the cluster tools launch)
python -m cluster.vmctl  start head --disk 300
python -m cluster.expctl push   head
python -m cluster.expctl launch head --preset dpsweep --label myrun \
        --dpsizes 3,5,10,15,20,25,32 --nsim 1 --max-iter 3 \
        --nocache --objsize --plot
python -m cluster.expctl watch  <run_id>
python -m cluster.vmctl  stop   head
```

`cluster/README.md` covers the VM side. The rest of this file is the
evaluation itself.

---

## Flags

| flag | meaning |
|---|---|
| `--dpsizes 3,5,10` | comma list of PoP counts. Default `3,5,10,15,20,25,<n_vultr>` |
| `--nsim 1` or `--nsim 3,2` | simulations per size; single int, or a list parallel to `--dpsizes` |
| `--max-iter N` | training iterations (sets `SCULPTOR_MAX_ITER`) |
| `--cache-fn PATH` | where the aggregated per-size stats pickle goes. Namespaces a run |
| `--figures-subdir S` | figures land in `figures/<S>/` instead of colliding |
| `--probe-n N` | **measurement budget** per solve(). Implies `--probe-mode smart`. Also accepts the literal `prefixes` = this deployment's own prefix count, which is the only way to say "one measurement per prefix" across a size sweep. Painter resolves the same value |
| `--probe-mode M` | `post_step` (stock, no budget) / `scheduled` / `slotted` / `gated` / `smart` |
| `--plot` | also run `make_paper_plots` at the end |
| `--port` | vestigial under Ray; nothing binds it |

---

## Reading the log

The sweep is chatty. These are the lines that matter:

```
[sweep] probing: mode=smart budget=5        <- the policy ACTUALLY in force
[sweep] === dpsize=20 dpsize_str=... nsim=1 ===
[mem] tag=dpsize_start ... dpsize=20        <- phase timing anchors
[depsetup] array fast path, shards=cache/lat_shards (20 pops)
[probe-budget] EXITING on 6 path measures (= 1 setup grounding + 5 probes)
                | budget N=5 mode=smart skipped=0 iters=27
[sweep] dpsize=20 done in 3782.0s (63.0 min, 3782.0s/sim)
[sweep] ALL DONE in 12345.0s. 7/7 sizes ok, wrote 7 dpsizes to <cache_fn>
```

**`[sweep] ALL DONE` is the completion contract.** Do not trust `rc == 0`
on its own -- on 2026-08-20 a cell exited 0 in six seconds after a
silently-failed hot-start. `cluster/expctl.py verdict()` judges on this
banner; so should you.

---

## THE TRAP: result caching

`wrapper_eval` caches per-size results at
`cache/popp_failure_latency_comparison_<dpsize_str><_RUN_TAG>.pkl`. A size
whose pickle exists **returns in ~1 second without training**, and the log
says so:

```
[sweep] dpsize=3 done in 2.4s ...  [CACHE HIT -- this wall time is NOT a timing measurement]
```

That is fine for resuming a results run and **poison for a timing run**.
Two ways to force real work:

* `expctl launch --nocache` -- sets `SCULPTOR_RUN_TAG` to the run id, so
  every size gets a virgin cache path. Deletes nothing.
* set `SCULPTOR_RUN_TAG=<something unique>` yourself.

`progress.json` records `cached: true` per size, the dashboard strikes
those rows through, and `plot_cluster_timing` excludes them from the
scaling fit. Do not undo any of that.

---

## Environment knobs that change what you measure

Defaults live in `helpers/constants.py`. **Set them there, not inline** --
five copies of the MC default existed before 2026-08-21 and two were
unreachable.

| var | default | effect |
|---|---|---|
| `SCULPTOR_MC_NUM` | `DEFAULT_MC_NUM = 1` | Monte-Carlo draws per latency-benefit call. NOT just a speed knob: 1 is a single-draw noisy estimator. ~2.6x faster per iteration than 5 |
| `SCULPTOR_MC_NUM_EXPLORE` | `5` | draws during the max-info phase (deliberately higher) |
| `SCULPTOR_PROBE_MODE` | `post_step` | `post_step` has **no budget** |
| `SCULPTOR_PROBE_N` | `DEFAULT_PROBE_N = 10` | measurement budget, only in force when mode != post_step |
| `SCULPTOR_LAT_SHARDS` | auto (`cache/lat_shards`) | array fast path for deployment setup, ~3x. `''` forces the legacy serial 4.5 GB CSV loop |
| `SCULPTOR_MAX_ITER` | dpsize-dependent | training iterations |
| `SCULPTOR_N_WORKERS` | `min(cpu_count, get_n_workers(dpsize))` | Ray actor count |
| `SCULPTOR_LOG_OBJSIZE` | `0` | per-worker object-size census (`--objsize` on expctl) |

**Ray actors do not inherit env set after `ray.init()`.** `SCULPTOR_MC_NUM`
silently had no effect on any real run until 2026-08-21 for exactly this
reason -- it now travels in `init_kwa` (`kwa['mc_num']`), which IS pickled
to every actor. If you add a worker-side knob, use that channel and prove
it with a print, not an assumption.

---

## Budget fairness: painter closed, anyopt open

**Painter: closed.** `core/painter.py` now resolves the *same* budget
SCULPTOR is held to, through the single shared resolver
`helpers.constants.resolve_probe_budget()`, and stops once it has spent
that many iterations (= that many measurements, since painter measures the
live deployment every iteration). `SCULPTOR_PAINTER_MEASURE_CAP` overrides
it; `--probe-mode post_step` (no budget) restores legacy unbounded painter.
Verified locally 2026-08-21 at dpsize=3, `--probe-n prefixes`:

```
[probe-gate] PROBE_N=prefixes -> budget 36
PAINTER measurement budget = 36 (from the probe budget; ...)
```

Both arms resolve 36 from one code path, which is the point of the shared
resolver -- a per-arm copy is how "budget-fair" silently stops being true.

**anyopt: still open.** `core/anyopt.py` has zero references to
`probe_mode`, `probe_n` or `resolve_probe_budget`. It measures once per
non-transit popp -- 100 measurements against a budget of 36 at dpsize=3.
Do not present per-measurement claims *against anyopt* from a budgeted
sweep; SCULPTOR-vs-painter is now fair, SCULPTOR-vs-anyopt is not. anyopt's
count is at least observable (`Measuring anyopt providers: N`), so report
it alongside rather than implying parity.

**The budget only binds if `--max-iter` is high enough.** This is the trap
that replaces the old one. At dpsize=3 / `--max-iter 5` (7 iterations):

```
[probe-budget] EXITING on 4 path measures (= 1 setup grounding + 3 probes)
               | budget N=36 mode=smart skipped=0 iters=7
```

SCULPTOR spent **3 of 36** -- smart mode simply never wanted more probes in
7 iterations. Painter's cap, being a straight iteration cap, *did* bind.
So a short budgeted run caps the baseline and leaves SCULPTOR effectively
unbudgeted: the asymmetry is real but inverted from the old one, and a
"budget-fair" label on it would be wrong in the other direction. Give the
budget enough iterations to actually bind, or report the spend, not the
cap. Note `verify_e2e_probe_budget.py` treats underspend-with-zero-skips as
a FAILURE, so it must be run at an iteration count where the budget binds.

---

## The nsim>1 worker-staleness bug (fixed 2026-08-22)

Until 2026-08-22, any size with `--nsim > 1` silently dropped the sparse
strategy on every sim after the first: `Strategy sparse failed` +
`KeyError: (<pop>, '<peer>')` (or an IndexError in
`_compute_scenario_options`) from every worker, at the very first LB
flush. Root cause: `_cmd_update_deployment` refreshed the worker actor's
deployment dicts but not the DERIVED state -- the persistent Gurobi LP
(constraints/var_pool keyed by the old ug/popp universe), the lbx grids,
and every `hasattr`-guarded lazy cache (`_uipop_csr`, `_pt_csr`,
`rti_data`, `parent_tracker`). Sim 0 was consistent because the actor is
*constructed* with its deployment; sims 1+ dereferenced the new
deployment through the old structures. Every smoke and e2e test ran
nsim=1, which is exactly the blind spot.

The fix makes a full (non-`quick_update`) worker deployment update a
**rebirth**: dispose the Gurobi model, `self.__dict__.clear()`, re-run
`__init__` with the new deployment -- the same code path as
construction, so a new lazy cache added later cannot re-introduce the
bug. `_cmd_solve_lp`'s inline `quick_update=True` swaps are untouched.

Two guards now exist:
* `expctl launch --preset dpsweep` sets `SCULPTOR_REQUIRE_SOLNS=sparse`,
  so a sparse failure aborts the cell instead of burning eval phases on a
  baselines-only comparison. `--env SCULPTOR_REQUIRE_SOLNS=` disables.
* A failed strategy leaves `failed_strategies` in the metrics pickle and
  an empty `(0,)` adv under its key -- detectable, but check for it
  before aggregating old pickles.

The 2026-08-22 morning run (`20260822_072429-prefixbudget`) predates the
fix: its actual-3 pickle has 20 sims of baselines and ONE sparse sim.

---

## How long will this take? (do this arithmetic BEFORE launching)

Measured on `head` (c7g.16xlarge, 64 vCPU, $2.32/hr) 2026-08-21, the full
ladder at `--nsim 1 --max-iter 3` -- the run that produced every number
below -- took 3.8 h wall:

| dpsize | 3 | 5 | 10 | 15 | 20 | 25 | 32 |
|---|---|---|---|---|---|---|---|
| total @3 iter | 221 s | 135 s | 863 s | 1494 s | 3782 s | 4965 s | 2361 s |
| `t per iter` | 12.4 s | 2.2 s | 27.4 s | 58.5 s | 183.1 s | 252.0 s | n/a |

Note the loop ran **5** iterations under `--max-iter 3`, and dpsize=32 came
in *faster* than 25.

Cost model: `per_size = fixed + max_iter * t_per_iter`, where `fixed` is
setup plus the eval phases (all six solution types + failure sims) and does
**not** scale with `--max-iter`. Then multiply by `--nsim`. That gives:

| config | wall | cost |
|---|---|---|
| `--nsim 1  --max-iter 20` | 7.3 h | $17 |
| `--nsim 1  --max-iter 50` | 14.3 h | $33 |
| `--nsim 1  --max-iter 200` | **49 h** | $114 |
| `--nsim 3  --max-iter 200` | **6.1 d** | $342 |
| `--nsim 20 --max-iter 200` | **41 d** | **$2278** |

**`--nsim 20 --max-iter 200` is not an overnight run.** At nsim=20 even
dpsize=3 alone takes 14.6 h, so an 8-hour night completes *zero* sizes.
`--nsim` multiplies the whole ladder; it is the most expensive knob on the
command line and the easiest one to type without noticing.

At `--nsim 1 --max-iter 200`, sizes land at roughly 0.7 / 0.9 / 2.6 / 6.2 /
17.2 / 32.2 / 49.1 h cumulative -- so a night buys the ladder through
dpsize=15, and the pickle is written after every size.

Two things make this an *upper* bound, both unquantified: budgeted probing
skips the expensive live measurement on most iterations (these `t per iter`
numbers come from an unbudgeted `post_step` run that probed every step),
and `stop-v2` can exit before `max_iter` (its docstring records a 500-cap
smoke exiting at 167). Do not assume either rescues you by more than ~5x --
41 days / 5 is still 8 days.

---

## What comes out

* `--cache-fn` pickle: `{dpsize: {stats_*: ...}}`, **written after every
  size** (before 2026-08-21 it was written once at the end, so a crash at
  size 25 discarded 3-20 as well).
* `figures/<--figures-subdir>/`: per-size comparison PDFs, plus the paper
  plots if `--plot`.
* `progress.json` if `SCULPTOR_SWEEP_PROGRESS_JSON` is set (expctl sets
  it): machine-readable per-size wall/sec-per-sim/ok/cached.

One size failing no longer kills the sweep -- it prints the traceback,
records `ok: false`, and continues.

---

## Verifying a change

```bash
python integration_tests/verify_e2e_eval.py --quick          # ~7 min
python integration_tests/verify_e2e_dpsizes.py --quick       # the sweep itself
python integration_tests/verify_e2e_probe_budget.py          # budget binds, ~20 min
python integration_tests/test_convergence_vs_budget.py       # budget sweep + figure
```

`verify_e2e_probe_budget.py` fails on a cache hit, on a missing
`LEARNING ITERATION`, on probes exceeding N, on `measures != probes + 1`,
and on an underspend with zero logged skips. Those last two encode bugs
that were live earlier the same day.
