"""Site-failure objective: steady-state avg-latency + exhaustive mean over
per-PoP (site) failures with user->prefix frozen to steady-state.

Same shape as static_failure but:
  - Site failures (entire PoP down), not popp failures.
  - Exhaustive over n_pops (typically 3-30) -> deterministic gradient,
    no sampling noise, no annealing required.
  - Combined objective: (1-beta)*steady + beta*mean(soft_failure_obj).

Operationally more meaningful than popp failures: a whole-site outage is a
realistic worst-case the network must absorb.

LP function: solve_lp_assignment_site_failure in solve_lp_assignment.py.
Unit tests: tests/test_lp_correctness.py::test_site_failure_*.
"""
from experiments.objectives import ObjectiveSpec, register


register(ObjectiveSpec(
	name='site_failure',
	lp_obj_string='site_failure',
	lp_kwargs={
		'site_failure_beta': 0.5,
		# Two-component penalty: no_route is HEAVY (true unrecoverable case),
		# congestion is MODERATE (user routed but popp overloaded -- constraint
		# violation). Pick no_route >> congestion so SGD prioritizes avoiding
		# the unrecoverable case.
		'site_failure_no_route_penalty': 50.0,
		'site_failure_congestion_penalty': 2.0,
	},
	eval_phases=('strategy_compare', 'static_failure_resilience'),
	description=(
		'Steady-state avg-latency + exhaustive mean over per-PoP failures, '
		'with each user pinned to the prefix from the steady-state LP. '
		'Per-failure soft objective: -avg_lat_routed - 20 * frac_no_route. '
		'Combined as (1-beta)*steady + beta*mean(soft), default beta=0.5. '
		'Deterministic gradient (no sampling), 1 + n_pops LP solves per call.'
	),
))
