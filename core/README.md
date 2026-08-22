# core/

The algorithm and everything it needs to run. **Nothing here imports
`experiments/`** — that inversion was removed on 2026-08-21 and is the rule
this package is kept to.

## The solver

| file | role |
|---|---|
| `sparse_advertisements_v3.py` | SCULPTOR itself: `Sparse_Advertisement_Solver`, the SGD loop, `compare_different_solutions`, checkpointing to `state-N.pkl`. The largest file in the repo. |
| `optimal_adv_wrapper.py` | Base class `Optimal_Adv_Wrapper` — deployment loading, common LP helpers, `measure_ingresses`. Both the driver and the Ray worker subclass it. |
| `generic_objective.py` | `Generic_Objective` — dispatch from objective name to LP function. |
| `test_polyphase.py` | **Not a test despite the name.** Star-imported by `path_distribution_computer.py`, so it is a hard runtime dependency. Worth renaming. |

## LP layer

| file | role |
|---|---|
| `solve_lp_assignment.py` | All LP objective implementations plus the persistent-LP solve loop. `generic_lp_functions` is the registry. |
| `gpshim.py` | gurobipy-subset facade. `SCULPTOR_LP_BACKEND` selects the backend and **defaults to `highs`** — so `highspy` is required on the default path. |
| `hard_objectives.py` | Extension objectives (`max_util`, `frac_beyond_optimal`, `lat_plus_max_util`, ...), registered when `SCULPTOR_XOBJS=1`. Self-registers as well as being registered from `solve_lp_assignment`'s tail, because the order-dependent version silently left them undispatchable. |
| `static_failure_eval.py` | Static failure resilience, used by `hard_objectives`. |

## Distributed execution (Ray only)

| file | role |
|---|---|
| `path_distribution_computer.py` | Worker-side latency-benefit and LP logic, **and** the Ray actor layer merged in on 2026-08-21. Three names: `Path_Distribution_Computer` (plain class), `_LocalPathDistributionComputer` (adds the `_cmd_*` handlers, what tests use), `Path_Distribution_Computer_Actor` (the `@ray.remote` wrapper). |
| `worker_comms.py` | `Worker_Manager` — spawn / fan out / tear down Ray actors. Ray has been the only backend since the mid-2026 migration; the ZMQ path is gone. |

The `_cmd_*` methods are dispatched by name via `getattr(self, '_cmd_' + cmd)`,
so a grep for callers finds nothing. They are not dead.

## Deployment construction

| file | role |
|---|---|
| `deployment_setup.py` | Builds synthetic and real deployments, link capacities, user volumes. Reads the 4.2 GB `cache/vultr_ingress_latencies_by_dst.csv`. |
| `shard_loader.py` | Loads per-PoP `.npz` latency shards instead of re-parsing that CSV. Opt-in via `SCULPTOR_LAT_SHARDS`. |
| `convert_latencies.py` | **Generates those shards.** Without it they cannot be rebuilt. |
| `fork_load.py` | Array-native deployment load path, default-on. |
| `realworld_measure_wrapper.py` | Real-deployment glue (RIPE Atlas, advertisement caching). Imports `peering_measurements`, which is not in the venv — that import has always been broken here. |
| `worlds.py` | Canonical world definitions. Its own docstring calls it the single source of truth; the dashboard depends on it. |

## Baselines

`painter.py` (PAINTER + unicast) and `anyopt.py` (AnyOpt) — the comparison
strategies SCULPTOR is scored against.

## Gotchas

- `deployment['port']` is vestigial: nothing binds it under Ray. It is kept
  only because `save`/`load_optimization_state` round-trips it.
- `init_advertisement` needs `n_prefixes >= n_pops + 1` — one prefix for
  anycast plus one per PoP. Below that it used to IndexError into a bare
  `except`; it now raises a clear `ValueError`.
