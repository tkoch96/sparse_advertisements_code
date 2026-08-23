# evaluations/

Post-training assessment: given advertisements from every solution type, how do
they compare?

## Layout

```
evaluations/
  eval_all_solution_types.py       objective-independent half: build deployment,
                                   stand up workers, solve every solution type
  objectives/                      objective-dependent half, one suite per
                                   objective, routed by objectives/objective_hooks.py
  evaluate_over_deployment_sizes.py  sweep + plots vs deployment size
  evaluate_over_n_prefixes.py        sweep vs prefix budget
  wrapper_eval.py                  shared eval phases (failure resilience, flash
                                   crowd, diurnal), bisect_critical_intensities,
                                   default_metrics, global_performance_metrics_fn
  generate_paper_table.py          the paper table: solve -> metrics -> LaTeX/CSV
                                   (invoke via the repo-root launcher)
```

## The two halves

`eval_all_solution_types.py` (renamed from `eval_latency_failure.py` on
2026-08-21 — it compares SCULPTOR against PAINTER, AnyOpt, anycast,
one_per_pop and one_per_peering, which the old name did not say) is
**objective-independent**.

Everything after it is **objective-dependent** and lives in `objectives/`,
routed by `objectives/objective_hooks.py`:

| objective | module |
|---|---|
| `avg_latency` (default) | `evaluations_for_latency_plus_resilience.py` |
| `max_util`, `lat_plus_max_util` | `evaluations_for_mlu.py` |
| `per_site_cost` | `evaluations_for_site_cost.py` |
| `frac_beyond_optimal` | `evaluations_for_frac_beyond_optimal.py` |
| `joint_priority` | `evaluations_for_priorities.py` |

Why: the failure-resilience, flash-crowd and diurnal phases — and all panels of
the comparison figure — are written against latency in milliseconds weighted by
UG volume. Run after an MLU or priority optimisation they produced numbers that
looked valid and meant nothing. `objectives/_objective_eval_base.py` holds the
per-sim scoring scaffolding (`score_all_strategies`, `objective_value_scorer`)
the suites share.

**Adding an objective:** add a module in `objectives/` exposing `OBJECTIVES`
and `run(ctx)`, then register it in `objective_hooks._ROUTES` and (if it needs
its own LP) in `core/solve_lp_assignment.generic_lp_functions`. Do not add a
branch inside the latency suite. An unregistered objective falls back to the
latency suite *with a printed warning*, so borrowed evaluation is visible.

## The paper table

`python generate_paper_table.py --dpsize X --number_of_deployments N
--num_training_iter I --run_id TAG` (repo root) solves anything missing, runs
each objective's evaluation suite, and emits `figures/paper_table/paper_table
.{tex,csv}`. Three pickle layers make repeat calls cheap (fully cached: ~5s);
`FORCE_*` flags at the top of the file (env-overridable) retarget recomputes.
`SCULPTOR_RECALC` targets individual metric families inside the latency suite
(`pct_vol, failure, volume, diurnal, flash, diurnal_bisect, flash_bisect, all`).

## Traps worth knowing

**The bare `except:`.** `evaluate_all_metrics` wraps its whole strategy loop in
one. A run whose solver died in the first second still reaches plotting,
returns a metrics dict and exits 0. Never treat `rc == 0` as success — check
the metrics pickle. This cost 11h of actual-32 training on 2026-08-21.

**Resume-skip.** If `compare_rets[...]['n_advs']` is already populated in the
metrics pickle, the sim is skipped outright. A run against a warm cache can be
a seconds-long no-op that looks like a pass. `SCULPTOR_RUN_TAG` namespaces the
pickle filename when you need parallel or A/B runs not to collide.

**The schema strip.** The metrics loader deletes any pickle key not present in
`wrapper_eval.default_metrics`. A new metric that is computed and dumped but
not registered there is silently erased on the next load (this ate the first
round of bisected critical intensities on 2026-08-23). Register the key.

## Legacy

`evals.py`, `testing_generic_objective.py`, `testing_priorities.py`,
`testing_site_costs.py`, `actual_deployment_eval_latency_failure.py` (the
pre-split monolithic driver) and the `table_generate.py` shim now live in
`old_scripts/`. Nothing imports them.
