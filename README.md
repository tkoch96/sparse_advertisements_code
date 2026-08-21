# Sparse Advertisements Code (SCULPTOR)

Research codebase for **SCULPTOR**, an SGD-based optimizer for BGP
advertisement strategies that minimizes average user latency while staying
resilient to PoP/popp failures. Compares SCULPTOR against PAINTER, AnyOpt,
anycast, and several baseline strategies across simulated and
real-deployment scenarios.

---

## 📍 Where to start

| You are... | Read |
|---|---|
| A new contributor | This README, then [unit_tests/README.md](unit_tests/README.md) |
| Picking up mid-stream | [HANDOFF_NEXT.md](HANDOFF_NEXT.md) — the canonical current handoff; earlier ones live in [old_handoffs/](old_handoffs/) |
| Standing up the AWS cluster | [CLUSTER_RUNBOOK.md](CLUSTER_RUNBOOK.md) |
| Looking for the research backlog | [old_handoffs/RESEARCH_ROADMAP.md](old_handoffs/RESEARCH_ROADMAP.md) — stale since 2026-05-27 |

Older handoffs and session snapshots all live in `old_handoffs/`
(`HANDOFF_SESSION_6/7/8/9.md`, `SESSION_4/5_SUMMARY.md`, `OVERNIGHT_SUMMARY.md`,
`RESEARCH_ROADMAP.md`) — useful for archaeology, not authoritative for current
state. Retired code lives in `old_scripts/`; nothing there is imported by the
running pipeline.

---

## Quick start

### Local smoke (laptop, ~2 min)

```bash
# In your venv:
python -m experiments.run_objective \
    --obj avg_latency --dpsize small --port 31510 --max-iter 20
```

This builds a tiny 3-pop deployment, runs all strategies (sparse, painter,
anyopt, anycast, one_per_pop, one_per_peering, random), and writes a
markdown summary to `cache/experiments/run_obj_avg_latency_small.md`.
Sparse uses Ray actors locally — Ray initializes its own local instance on
first import.

### Sweep over deployment sizes (laptop or cluster)

```bash
SCULPTOR_DEPLOYMENT_SWEEP_SIZES=3,5,10 SCULPTOR_DEPLOYMENT_SWEEP_NSIM=1 \
SCULPTOR_MAX_ITER=50 \
python experiments/benchmarks/run_deployment_sweep.py --port 31510
```

Iterates `evaluate_all_metrics` over each dpsize, hot-starts from saved
state where possible. See `experiments/benchmarks/run_deployment_sweep.py` docstring
for all env vars (per-dpsize NSIM lists, headroom, etc.).

### AWS cluster

See [CLUSTER_RUNBOOK.md](CLUSTER_RUNBOOK.md) for full setup.
Short version:

```bash
ray up ray-cluster.yaml -y                    # 5-30 min
ray exec ray-cluster.yaml '... evaluations/eval_latency_failure.py ...'
./teardown.sh                                 # ALWAYS at end of session
```

---

## Architecture overview

```
                    ┌─────────────────────────────────┐
                    │  Driver process (head node)     │
                    │                                 │
   evaluate_all_   ─▶│  Sparse_Advertisement_Eval     │
   metrics()        │  ├─ training loop (SGD)        │
                    │  ├─ compare_different_solutions │
                    │  │  ├─ sparse (this process)    │
                    │  │  └─ painter, anyopt, …       │  ◀── fork subprocesses
                    │  │     (parallel ProcessPool)   │
                    │  └─ eval phases                 │
                    │       (failure resilience,      │
                    │        flash crowd, diurnal)    │
                    └────────┬────────────────────────┘
                             │  (Ray actor RPC)
                             ▼
                    ┌─────────────────────────────────┐
                    │  Ray actors (worker node)        │
                    │  N × Path_Distribution_Computer │
                    │  - latency_benefit calc          │
                    │  - LP solve (persistent LP,    │
                    │    HiGHS default)               │
                    │  - per-worker UG slice           │
                    └─────────────────────────────────┘
```

Key concepts:

- **Sparse training (SGD)** lives in `core/sparse_advertisements_v3.py`
  (`Sparse_Advertisement_Solver`). The main loop does gradient probes →
  step → measure → stop-check, ~80s/iter at dpsize=25 with 32 workers.
- **Workers** are Ray actors (`core/path_distribution_computer.py`)
  managed by `Worker_Manager` (`core/worker_comms.py`). Each holds a
  persistent LP model (HiGHS by default) + cached deployment slice.
- **Strategies (painter, anyopt, etc.)** run concurrently as forked
  subprocesses on the head, in `compare_different_solutions`. They
  don't use Ray workers — they solve via single LPs.
- **Objectives** are registered in `experiments/objectives.py`. New
  objectives are added by registering an `ObjectiveSpec` + an LP function
  in `core/solve_lp_assignment.py`. See "Adding a new objective" below.
- **Eval phases** are the post-training assessments (LP latencies under
  failure scenarios, percent of volume within latency targets, etc.),
  implemented in `evaluations/wrapper_eval.py` + `evaluations/eval_latency_failure.py`.

---

## Code map

The tree is organised into five buckets (restructured 2026-08-21). The repo
root now holds only docs and config — no `.py` files.

| directory | what lives there |
|---|---|
| `core/` | The algorithm and everything it needs to run: SCULPTOR's solver, the LP layer, the Ray worker/actor layer, deployment construction, and the PAINTER / AnyOpt baselines. |
| `evaluations/` | Post-training assessment: `evaluate_all_metrics`, the eval phases, and the per-objective sweep drivers. |
| `helpers/` | Constants and general utilities — logging, timing instrumentation, plot styling. |
| `unit_tests/` | Fast pytest suites. `pytest` runs these by default (`testpaths`). |
| `integration_tests/` | Standalone end-to-end checks that run the real pipeline as a subprocess. See `integration_tests/README.md`. |
| `experiments/` | Campaign drivers — one subdirectory per investigation, plus `benchmarks/` and `ablation_study/`. |
| `tools/` | Operational tooling and campaign manifests. |
| `old_scripts/` | Retired code. Nothing here is imported by the running pipeline. |

Imports are absolute from the repo root (`from helpers.constants import *`).
Modules that are also run as scripts put the repo root on `sys.path`
themselves, so `python evaluations/eval_latency_failure.py ...` works from
anywhere.

### Module inventory

| File | Role |
|---|---|
| `core/sparse_advertisements_v3.py` | SCULPTOR algorithm (SGD), `Sparse_Advertisement_Solver`, `compare_different_solutions` |
| `core/painter.py` | PAINTER + unicast baselines |
| `core/anyopt.py` | AnyOpt baseline |
| `core/optimal_adv_wrapper.py` | Base class `Optimal_Adv_Wrapper`: deployment loading, common LP helpers, `measure_ingresses` |
| `core/path_distribution_computer.py` | Worker-side LP / latency_benefit logic (LP cache, solver shell via `gpshim`). Imported by `_ray.py` for actor body |
| `core/path_distribution_computer.py` | Ray actor wrapper around the above |
| `core/worker_comms.py` | Thin re-export of worker_comms (kept for backward-compat imports) |
| `core/worker_comms.py` | `Worker_Manager`: spawn / fanout / tear down Ray actors |
| `core/solve_lp_assignment.py` | All LP objective implementations (avg_latency, per_site_cost, joint_priority, site_failure) + the persistent-LP solve loop (backend via `gpshim`) |
| `core/generic_objective.py` | `Generic_Objective` — runtime dispatch from objective name → LP function |
| `core/deployment_setup.py` | Build synthetic + actual deployments, link capacities, user volumes |
| `evaluations/wrapper_eval.py` | Eval phase implementations (failure resilience, flash crowd, diurnal) |
| `evaluations/eval_latency_failure.py` | `evaluate_all_metrics()` — primary driver invoked by sweeps |
| `evaluations/actual_deployment_eval_latency_failure.py` | Real-deployment variant (less commonly used) |
| `evaluations/evaluate_over_deployment_sizes.py` | Sweep + plot SCULPTOR vs others as dpsize varies (paper plots) |
| `helpers/paper_plotting_functions.py` | Plot styling primitives |
| `core/realworld_measure_wrapper.py` | Real-deployment glue (RIPE Atlas, advertisement caching) |
| `helpers/helpers.py` | Generic utilities (logging, mem snapshots, deployment splitting, etc.) |
| `helpers/constants.py` | NO_ROUTE_LATENCY, NON_SIMULATED_LINK_CAPACITY, dpsize→n_pop mapping |
| `evaluations/testing_generic_objective.py`, `evaluations/testing_priorities.py`, `evaluations/testing_site_costs.py` | Legacy per-objective drivers, mostly superseded by `experiments/run_objective.py` |
| `helpers/get_smaller_anycast_lats.py` | Sub-sample the anycast latency CSV for faster local testing |
| `evaluations/evals.py` | Legacy eval driver; superseded (not imported by anything) |
| `core/test_polyphase.py` | **Not a test despite the name** — star-imported by `core/path_distribution_computer.py:31`, so it is a hard runtime dependency of the solver |

### `experiments/`

Newer per-objective + per-experiment drivers. Each script has its own docstring.

| File | Role |
|---|---|
| `experiments/objectives.py` | Registry: `ObjectiveSpec` dataclass + map of all objectives |
| `experiments/run_objective.py` | Single CLI driver (`python -m experiments.run_objective --obj <name> --dpsize <size>`) |
| `experiments/site_failure.py` | `site_failure` objective spec (steady + mean-over-PoP-failures with frozen user→prefix) |
| `experiments/static_failure_eval.py` | BGP-fallback failure eval phase shared by site_failure |
| `experiments/painter_hypothesis_sweep.py` | 2D sweep of (scale_factor, vol_spread) testing the painter-degradation hypothesis |

### `experiments/benchmarks/`

Sweep + perf-investigation harnesses with structured output.

| File | Role |
|---|---|
| `experiments/benchmarks/run_deployment_sweep.py` | Cluster-friendly sweep over dpsizes with per-size NSIM, hot-start, env-var config |
| `experiments/benchmarks/eval_phase_baseline.py` | Per-phase timing + crash diagnostics for `evaluate_all_metrics` |

### `unit_tests/`

12 pytest files covering LP correctness, worker behaviour, convergence, perf
sweep. See [unit_tests/README.md](unit_tests/README.md) for fixtures + markers.

### Directories

| Dir | Purpose |
|---|---|
| `data/` | External inputs (AS relationships, latency CSVs, IP lists). Pulled from Drive — see Setup |
| `cache/` | Generated artefacts: deployment pickles, per-experiment metrics, plot inputs |
| `runs/` | Per-training-run state (`state-N.pkl` checkpoints every 5 iters, per-iter stats) |
| `logs/` | Worker stdout/stderr captures; session forensics |
| `figures/` | Generated plots (paper-quality PDFs + diagnostic PNGs) |
| `old_scripts/` | Pre-v3 implementations kept for reference; don't import |

---

## Setup

### Local (Mac/Linux)

```bash
# 1. Python 3.12 venv:
python3.12 -m venv ~/Documents/venv312
source ~/Documents/venv312/bin/activate
pip install --upgrade pip

# 2. Install deps (requirements.txt is incomplete; install transitively):
pip install numpy scipy matplotlib tqdm pandas pickle5 \
            gurobipy scikit-learn scikit-image geopy geoip2 \
            ray[default] boto3 \
            pyzmq pytest

# 3. LP backend: HiGHS is the DEFAULT (as of 2026-08-20) — no license
#    needed. Gurobi is opt-in via SCULPTOR_LP_BACKEND=gurobi (required
#    only for quadratic objectives: squaring/square_rooting). We moved
#    off Gurobi as the default after WLS license scaling issues: on
#    multi-node fleets, sustained sessions above the WLS baseline get
#    license-killed after ~30 min (killed the 2026-08-20 eods32 run).
#    If you do need Gurobi: academic WLS license from
#    https://www.gurobi.com/academia/academic-program-and-licenses/
#    Drop ~/gurobi.lic; confirm: python -c "import gurobipy as g; g.Model().optimize()"

# 4. Data files from Drive (https://drive.google.com/drive/folders/1PvGOPRgkvjTaeq5m2ogyh0zSZ4r6JLcJ):
#    data/vultr_peers_inferred.csv
#    cache/vultr_ingress_latencies_by_dst.csv      (~4.5 GB)
#    cache/vultr_anycast_latency_smaller.csv       (~52 MB)
#    cache/vultr_provider_popps.csv                (~2 KB)

# 5. Sanity check:
python -m experiments.run_objective --obj avg_latency --dpsize small \
    --port 31510 --max-iter 5
```

### AWS cluster

See [CLUSTER_RUNBOOK.md](CLUSTER_RUNBOOK.md) for the IAM perms,
`ray-cluster.yaml` walkthrough, and the standard tear-down ritual. Short
checklist:

- IAM user with `AmazonEC2FullAccess` + `IAMFullAccess`
- `aws configure` locally
- `~/gurobi.lic` (WLS academic; official baseline 2 concurrent sessions -- but see the WLS policy note in experiments/ablation/README.md: empirically 20-48+ sessions sustain fine; size pools to RAM, not sessions)
- `pip install "ray[default]" boto3` in the local venv
- `ray up ray-cluster.yaml -y`

---

## Running experiments

### Environment variables

These knobs control behaviour without code changes. Set via shell env or
in `os.environ` from a launcher.

| Variable | Default | Effect |
|---|---|---|
| `SCULPTOR_MAX_ITER` | unset | Override `Sparse_Advertisement_Solver.solve(max_iter=…)` |
| `SCULPTOR_N_WORKERS` | min(cpu_count, dpsize_suggested) | Max Ray-actor pool size |
| `SCULPTOR_N_WORKERS_DURING_PARALLEL` | unset | If set, sparse training starts with this many workers; ramps up to `SCULPTOR_N_WORKERS` once concurrent parallel strategies finish |
| `SCULPTOR_CAPACITY_HEADROOM` | 0.0 | Multiplier `cap × (1+h)` applied during training only (relaxes the LP cap constraint to give SGD slack); restored to true cap for eval |
| `SCULPTOR_DISABLE_PARALLEL_STRATEGIES` | unset | Run painter / anyopt / etc. serially after sparse instead of concurrently in subprocesses |
| `SCULPTOR_DEPLOYMENT_SEED` | unset | Pin the deployment RNG for reproducible smoke tests |
| `SCULPTOR_DEPLOYMENT_SWEEP_SIZES` | `3,5,10,15,20,25,<n_vultr>` | Comma-separated dpsize list for `experiments/benchmarks/run_deployment_sweep.py` |
| `SCULPTOR_DEPLOYMENT_SWEEP_NSIM` | `1` | Single int OR comma list parallel to SIZES (per-dpsize random_iter count) |
| `SCULPTOR_DEPLOYMENT_SWEEP_TAG` | `dep_sweep` | Suffix on per-dpsize eval pickles |
| `SCULPTOR_RUN_TAG` | unset | Tag for the per-dpsize eval pickle within `evaluate_all_metrics` |
| `SCULPTOR_ADAPTIVE_PROBE_BUDGET` | unset | Shrink gradient probe budget over iters |
| `SCULPTOR_STOP_DROP_ADV_DELTA` | unset | Early-stop threshold on advertisement change |
| `SCULPTOR_LOG_MEM` | `1` | Set to `0` to silence `[mem]` instrumentation |
| `SCULPTOR_WORKER_INIT_STAGGER_SEC` | `0` | Offset between worker spawns (smooths per-actor RAM peaks during init) |
| `SCULPTOR_WORKER_MEM_LOG_DIR` | `/tmp` | Per-worker mem log file directory (Linux only) |

### Port discipline

Each evaluation needs its own port (defaults to 31510 / 31415 / 31618). If
two runs share a port, their workers cross-talk and silently corrupt
results. When running concurrent experiments, pick distinct ports.

### "Scripts need babysitting"

Almost every driver in this repo can run for hours to days at large dpsizes
and routinely hits transient issues (Gurobi WLS throttling, OOM, Ray actor
death, disk space). Hot-start logic exists in most drivers (look for
`state-N.pkl` checkpoints in `runs/`), but always assume any single run
might need to be restarted. For new code changes, smoke at `dpsize=small`
or `dpsize=actual-3` first, then scale up.

---

## Adding new things

### A new objective function

Two pieces: register it + implement the LP.

```python
# 1. In experiments/<name>.py:
from experiments.objectives import ObjectiveSpec, register
register(ObjectiveSpec(
    name='my_new_objective',
    lp_obj_string='my_new_objective',     # the string sas.compare_different_solutions's LP layer expects
    description='What this minimises',
    lp_kwargs={'my_knob': 1.0},
    eval_phases=('static_failure_resilience',),  # plus whatever post-training evals
    gamma=0, using_resilience_benefit=True,
))

# 2. In core/solve_lp_assignment.py: add a function
def solve_lp_assignment_my_new_objective(sas, routed_through_ingress, obj, **kwargs):
    """Return dict with keys:
        objective: float (final LP value)
        solved:    str (Gurobi solution status)
        paths_by_ug: {ug_index: [(poppi, vol_pct), ...]}
        lats_by_ug: numpy array of per-UG latencies
        ... plus any objective-specific fields
    """
    paths_by_ug, available_paths = get_paths_by_ug(sas, routed_through_ingress)
    # ... build Gurobi model, optimize, extract solution ...

# 3. Register the LP function:
generic_lp_functions['my_new_objective'] = solve_lp_assignment_my_new_objective

# 4. Import the spec module from experiments/run_objective.py so it registers at import time.

# 5. (optional) Unit-test in tests/test_lp_correctness.py for a hand-verifiable case.
```

### A new strategy

Add a `solve_<name>` method to `Sparse_Advertisement_Wrapper` (in
`core/sparse_advertisements_v3.py`) that populates `self.solutions[name]` with
the same dict shape as the existing strategies (see `solve_painter` for
the simplest reference). Add the name to `solution_types` and, if it
should run concurrently with sparse, to `_PARALLEL_STRATEGY_NAMES`.

### A new eval phase

Add a function to `evaluations/wrapper_eval.py` that takes `(sas, metrics, …)` and
populates `metrics[<phase_name>][random_iter][solution_type]`. Add the
phase name to the relevant `ObjectiveSpec.eval_phases` tuple. Implement
the same shape as the existing phases (e.g. `assess_failure_resilience`).

---

## Common gotchas

- **Gurobi WLS license is rate-limited.** Academic WLS allows ~3
  sustained concurrent sessions; we routinely run with 32 worker actors.
  Throttling shows up as "Overage for too long" warnings and silently
  slows things down. Avoid running local Gurobi while a cluster sweep
  is active.
- **`evaluations/actual_deployment_eval_latency_failure.py`** is the real-deployment
  path (RIPE Atlas measurements, actual BGP advertisements). It's the
  same shape as `evaluations/eval_latency_failure.py` but with real-world measurement
  glue. Most active development uses the simulated path.
- **dpsize naming.** Synthetic deployments use names like `small` /
  `decent` / `med` (defined in `helpers/constants.py`). Actual deployments use
  `actual-N` where N is the number of PoPs (e.g. `actual-25` = use real
  latencies for 25 randomly-chosen Vultr PoPs).
- **State pickle growth.** `runs/<ts>-*/state-N.pkl` checkpoints grow
  linearly with iteration count (~3 MB/iter at dpsize=25). Old run dirs
  can eat all the disk on the head node — periodically clean up.
- **macOS vs Linux.** `_log_mem_worker` reads `/proc` and is a silent
  no-op on macOS. Mem-instrumentation only fires under Linux (cluster).
