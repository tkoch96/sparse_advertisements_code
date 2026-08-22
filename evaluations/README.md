# evaluations/

Post-training assessment: given advertisements from every solution type, how do
they compare?

## The two halves

`eval_all_solution_types.py` (renamed from `eval_latency_failure.py` on
2026-08-21 — it compares SCULPTOR against PAINTER, AnyOpt, anycast,
one_per_pop and one_per_peering, which the old name did not say) is
**objective-independent**: build the deployment, stand up the workers, solve
every solution type.

Everything after that is **objective-dependent** and lives in a sibling module,
routed by `objective_hooks.py`:

| objective | module |
|---|---|
| `avg_latency` (default) | `evaluations_for_latency_plus_resilience.py` |
| `max_util`, `lat_plus_max_util` | `evaluations_for_mlu.py` |
| `per_site_cost` | `evaluations_for_site_cost.py` |
| `frac_beyond_optimal` | `evaluations_for_frac_beyond_optimal.py` |
| `joint_priority` | `evaluations_for_priorities.py` |

Why: the failure-resilience, flash-crowd and diurnal phases — and all 11 panels
of the comparison figure — are written against latency in milliseconds weighted
by UG volume. Run after an MLU or priority optimisation they produced numbers
that looked valid and meant nothing. `_objective_eval_base.py` holds the
scoring/plotting scaffolding the non-latency suites share.

**Adding an objective:** add a sibling module exposing `OBJECTIVES` and
`run(ctx)`, then register it in `objective_hooks._ROUTES`. Do not add a branch
inside the latency suite. An unregistered objective falls back to the latency
suite *with a printed warning*, so borrowed evaluation is visible.

`joint_priority` is documented in three places but is **not dispatchable** —
it is absent from `generic_lp_functions` and from
`hard_objectives.REGISTERED_OBJECTIVES`. Its module says so rather than
plotting a solve that did not happen.

## Sweeps

| file | role |
|---|---|
| `evaluate_over_deployment_sizes.py` | Sweep + plot vs deployment size. **See [RUNNING_THE_DEPLOYMENT_SWEEP.md](RUNNING_THE_DEPLOYMENT_SWEEP.md) before running it** -- flags, the result-cache trap, which env knobs change what you measure, and why a budgeted sweep is not yet budget-fair. `--dpsizes --nsim --max-iter --cache-fn --plot`; omitting them reproduces the old hardcoded behaviour. `make_paper_plots` lives here and both sweeps use it. |
| `evaluate_over_n_prefixes.py` | Same, swept over prefix budget. `--dpsize --prefixes --nsim --max-iter --plot`. |
| `actual_deployment_eval_latency_failure.py` | Real-deployment variant. |
| `wrapper_eval.py` | The eval phases themselves — failure resilience, flash crowd, diurnal — plus `default_metrics` and `global_performance_metrics_fn`. |

## Two traps worth knowing

**The bare `except:`.** `evaluate_all_metrics` wraps its whole strategy loop in
one. A run whose solver died in the first second still reaches plotting,
returns a metrics dict and exits 0. Never treat `rc == 0` as success — check
the metrics pickle. This cost 11h of actual-32 training on 2026-08-21.

**Resume-skip.** If `compare_rets[...]['n_advs']` is already populated in the
metrics pickle, the sim is skipped outright. A run against a warm cache can be
a seconds-long no-op that looks like a pass. `SCULPTOR_RUN_TAG` namespaces the
pickle filename when you need parallel or A/B runs not to collide.

## Legacy

`evals.py`, `testing_generic_objective.py`, `testing_priorities.py`,
`testing_site_costs.py` are older per-objective drivers, superseded by
`experiments/run_objective.py`. Nothing imports them.
