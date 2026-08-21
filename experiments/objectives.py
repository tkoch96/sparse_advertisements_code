"""Registry of SCULPTOR optimization objectives.

Each ObjectiveSpec captures the per-objective knobs needed to:
  - configure Sparse_Advertisement_Eval (gamma, using_resilience_benefit,
    generic_objective string),
  - feed the LP solver (e.g., site_cost_alpha, bulk_cap_limit),
  - shape the deployment generator (e.g., cost_setting for site costs),
  - decide which eval phases to run + how to summarize results.

A spec is a plain dataclass so adding a new objective = adding a literal in
this file; no plugin discovery, no abstract base.

The persistent worker LP in path_distribution_computer.py and the
non-persistent LP in solve_lp_assignment.py both accept the LP kwargs via
**kwargs, so the driver passes spec.lp_kwargs straight through.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ObjectiveSpec:
	# CLI name (e.g., 'avg_latency', 'per_site_cost', 'joint_priority',
	# 'site_failure').
	name: str
	# String passed into Sparse_Advertisement_Eval(..., generic_objective=...).
	# This is the key the LP solver dispatches on (solve_generic_lp_with_failure_catch).
	lp_obj_string: str
	# Extra kwargs forwarded to LP calls (e.g., site_cost_alpha, bulk_cap_limit).
	# Both the persistent worker LP and the scipy fallback pull via kwargs.get().
	lp_kwargs: Dict[str, Any] = field(default_factory=dict)
	# Whether the SGD trainer should compute the per-popp-failure resilience
	# benefit term in its gradient. Most objectives leave this False because
	# the objective itself either ignores failures (avg_latency, per_site_cost,
	# joint_priority) or bakes them in (site_failure).
	using_resilience_benefit: bool = False
	gamma: float = 0.0
	# Per-deployment-builder overrides (passed straight to get_random_deployment).
	# Example: {'cost_type': 'carbon'} for per_site_cost.
	deployment_kwargs: Dict[str, Any] = field(default_factory=dict)
	# Eval phase names to run (consumed by run_objective.py). See evals/ for the
	# implementations registered under these names.
	eval_phases: Tuple[str, ...] = ('strategy_compare', 'pct_volume_within_latency')
	# Env vars set just before training and restored after. Values must be
	# strings (env-var convention). e.g. SCULPTOR_CAPACITY_HEADROOM=0.2.
	train_env: Dict[str, str] = field(default_factory=dict)
	# Free-text description for summary output.
	description: str = ''


_REGISTRY: Dict[str, ObjectiveSpec] = {}


def register(spec: ObjectiveSpec) -> ObjectiveSpec:
	if spec.name in _REGISTRY:
		raise ValueError("Objective {!r} already registered".format(spec.name))
	_REGISTRY[spec.name] = spec
	return spec


def get(name: str) -> ObjectiveSpec:
	try:
		return _REGISTRY[name]
	except KeyError:
		raise KeyError("Unknown objective {!r}. Registered: {}".format(
			name, sorted(_REGISTRY.keys())))


def all_specs() -> List[ObjectiveSpec]:
	return list(_REGISTRY.values())


# --- Built-in objectives ---------------------------------------------------

register(ObjectiveSpec(
	name='avg_latency',
	lp_obj_string='avg_latency',
	description='Minimize traffic-weighted average user latency. Baseline objective.',
	eval_phases=(
		'strategy_compare', 'pct_volume_within_latency',
		'failure_resilience',
		'diurnal', 'flash_crowd',
	),
))

register(ObjectiveSpec(
	name='per_site_cost',
	lp_obj_string='per_site_cost',
	lp_kwargs={'site_cost_alpha': 100.0},  # was constants.DEFAULT_SITE_COST
	deployment_kwargs={'cost_type': 'carbon'},
	description=(
		'Minimize traffic-weighted (latency + site_cost_alpha * site_cost). '
		'Weighted-sum scalarization; alpha is the cost-vs-latency tradeoff knob.'
	),
	eval_phases=('strategy_compare', 'pct_volume_within_latency', 'site_cost_summary'),
))

register(ObjectiveSpec(
	name='joint_priority',
	lp_obj_string='joint_latency_bulk_download',
	# bulk_cap_limit: max total (HPrio + LPrio) volume per link as multiple of
	# capacity. SIGCOMM 2025 paper value is 3.0; existing experiments here ran
	# under 100.0 so that is preserved as the default. Override per-experiment
	# if reproducing the paper exactly.
	lp_kwargs={'bulk_cap_limit': 100.0},
	description=(
		'Joint optimization of HPrio (latency-sensitive) and LPrio (bulk) '
		'traffic on a shared link layer. HPrio LP is solved first, then bulk '
		'fills around it minimizing HPrio-weighted oversubscription -- de '
		'facto strict priority queueing. NOTE: this is NOT an SLO formulation; '
		'plots labeled "SLO" actually report HPrio congestion fraction.'
	),
	eval_phases=('strategy_compare', 'priority_bulk_sweep'),
))

# 'site_failure' is registered in experiments/site_failure.py to keep the
# spec colocated with its LP-function file path and avoid circular imports.
