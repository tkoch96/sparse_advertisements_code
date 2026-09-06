"""THE single place an objective is declared (Tom 2026-09-06).

Before this file, adding an objective meant editing ~8 independent lists:
the extension-LP dict (core/hard_objectives.REGISTERED_OBJECTIVES), the
training-policy map (core/generic_objective.OBJECTIVE_CLASSES), the eval-
suite router (evaluations/objectives/objective_hooks._ROUTES), four tables
in evaluations/generate_paper_table.py (GROUPS, KEY_COLUMNS,
OBJECTIVE_REQUIRED_KEY, DEFAULT_OBJECTIVES/ALIASES + the TeX display maps),
the resume-stability schema (evaluations/wrapper_eval.default_metrics),
the depstore fingerprint knobs (core/depstore.SEMANTIC_KNOBS), the
experiments driver's ObjectiveSpec registry, and the integration test's
objective list. Missing any one of them failed silently (an unrouted
objective crashed after a full solve; a missing required-key made the
table read the avg_latency pickle and report "covered").

Now every one of those consumers DERIVES its list from `PLUGINS` below.
Adding an objective = adding one ObjectivePlugin here (plus its LP function,
eval suite module, and any training-policy subclass, which the plugin
points at by dotted path).

This module is deliberately import-light (dataclasses + importlib only):
LP functions and classes are referenced as 'package.module:attr' strings
and resolved on demand, so the registry can be imported from anywhere in
core/ or evaluations/ without creating a cycle.

Per-consumer views (each consumer calls exactly one of these):
  extension_lp_functions()   core/hard_objectives.REGISTERED_OBJECTIVES
  lp_name_for(name)          the string the LP registry dispatches on
  training_classes()         core/generic_objective.OBJECTIVE_CLASSES
  eval_routes()              evaluations/objectives/objective_hooks._ROUTES
  required_metric_keys()     generate_paper_table.OBJECTIVE_REQUIRED_KEY
  table_groups()             generate_paper_table.GROUPS (column specs)
  key_columns()              generate_paper_table.KEY_COLUMNS
  tex_group_display() / tex_sub_display()
  paper_table_defaults()     generate_paper_table.DEFAULT_OBJECTIVES
  aliases()                  generate_paper_table.OBJECTIVE_ALIASES
  default_metric_keys()      wrapper_eval.default_metrics additions
  semantic_knobs()           core/depstore.SEMANTIC_KNOBS additions
  experiment_specs()         experiments/objectives ObjectiveSpec kwargs

Intent JSON files (evaluations/intents/*.json) stay data; `validate_names`
checks any objective list against the registry.
"""
import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ObjectivePlugin:
	# --- identity ---------------------------------------------------------
	name: str
	description: str = ''
	aliases: Tuple[str, ...] = ()
	# --- solver -----------------------------------------------------------
	# 'pkg.mod:function' for an EXTENSION LP registered into
	# solve_lp_assignment.generic_lp_functions by core/hard_objectives.
	# '' = the LP is built into solve_lp_assignment itself.
	lp: str = ''
	# the string the generic-LP dispatcher is called with (defaults to name;
	# joint_priority's LP is also registered as 'joint_latency_bulk_download').
	lp_name: str = ''
	# 'pkg.mod:Class' Generic_Objective subclass owning the TRAINING policy
	# (gradient components, gamma annealing, per-call LP kwargs). '' = base.
	training_class: str = ''
	# --- experiments/ driver (run_objective.py) ---------------------------
	# extra ObjectiveSpec fields: lp_kwargs, using_resilience_benefit, gamma,
	# deployment_kwargs, eval_phases, train_env. None = not runnable there.
	experiment: Optional[Dict[str, Any]] = None
	# --- evaluation -------------------------------------------------------
	eval_module: str = ''            # evaluations.objectives.<basename>
	required_metric_key: str = ''    # a pickle counts for this objective
	                                 # only if it holds this key
	metric_keys: Tuple[str, ...] = ()  # per-sim {soln: value} keys the suite
	                                   # writes; declared so the eval loader
	                                   # keeps them across resumes
	# --- paper table ------------------------------------------------------
	table_group: str = ''            # stored group label ('' = no columns)
	group_order: int = 999           # full-table section order
	key_order: int = 999             # pruned (paper) table section order
	# (sublabel, direction '<'|'>', extractor, *args) -- extractors are
	# resolved by name inside generate_paper_table (they close over its
	# private helpers); see generate_paper_table._EXTRACTORS.
	table_columns: Tuple[Tuple, ...] = ()
	key_columns: Tuple[str, ...] = ()   # sublabels promoted to the paper table
	tex_group: str = ''
	tex_subs: Dict[str, str] = field(default_factory=dict)
	paper_table_default: bool = False
	default_order: int = 999      # cell run order in DEFAULT_OBJECTIVES
	# --- provenance -------------------------------------------------------
	# env knobs that change RESULTS (enter the depstore fingerprint)
	semantic_knobs: Dict[str, str] = field(default_factory=dict)


PLUGINS: Dict[str, ObjectivePlugin] = {}


def register(plugin: ObjectivePlugin) -> ObjectivePlugin:
	if plugin.name in PLUGINS:
		raise ValueError('objective {!r} already registered'.format(plugin.name))
	PLUGINS[plugin.name] = plugin
	return plugin


def get(name: str) -> ObjectivePlugin:
	try:
		return PLUGINS[name]
	except KeyError:
		raise KeyError('unknown objective {!r}; registered: {}'.format(
			name, sorted(PLUGINS)))


def names():
	return list(PLUGINS)


def resolve(path: str):
	"""'pkg.mod:attr' -> the attribute (module imported on demand)."""
	mod, _, attr = path.partition(':')
	if not mod or not attr:
		raise ValueError('expected "pkg.mod:attr", got {!r}'.format(path))
	return getattr(importlib.import_module(mod), attr)


def canonical(name: str) -> str:
	"""Alias -> registered name (identity for unknown names)."""
	return aliases().get(name, name)


def validate_names(objs, where='objective list'):
	unknown = [o for o in objs if canonical(o) not in PLUGINS]
	if unknown:
		raise KeyError('{}: unknown objective(s) {} (registered: {})'.format(
			where, unknown, sorted(PLUGINS)))
	return [canonical(o) for o in objs]


# ----------------------------------------------------------- consumer views
def extension_lp_functions():
	return {p.name: resolve(p.lp) for p in PLUGINS.values() if p.lp}


def lp_name_for(name):
	p = PLUGINS.get(name)
	return (p.lp_name or p.name) if p else name


def training_classes():
	return {p.name: resolve(p.training_class)
			for p in PLUGINS.values() if p.training_class}


def eval_routes():
	return {p.name: p.eval_module for p in PLUGINS.values() if p.eval_module}


def required_metric_keys():
	return {p.name: p.required_metric_key
			for p in PLUGINS.values() if p.required_metric_key}


def _ordered(attr):
	return sorted((p for p in PLUGINS.values() if p.table_group),
				  key=lambda p: (getattr(p, attr), p.name))


def table_groups():
	"""[(group_label, objective_name, [(sublabel, dir, extractor, *args)])]
	in full-table section order."""
	return [(p.table_group, p.name, list(p.table_columns))
			for p in _ordered('group_order')]


def key_columns():
	"""[(group_label, sublabel)] in paper-table section order."""
	out = []
	for p in _ordered('key_order'):
		out.extend((p.table_group, sub) for sub in p.key_columns)
	return out


def tex_group_display():
	return {p.table_group: p.tex_group for p in PLUGINS.values()
			if p.table_group and p.tex_group}


def tex_sub_display():
	out = {}
	for p in PLUGINS.values():
		out.update(p.tex_subs)
	return out


def paper_table_defaults():
	return [p.name for p in sorted(PLUGINS.values(),
								  key=lambda p: (p.default_order, p.name))
			if p.paper_table_default]


def aliases():
	return {a: p.name for p in PLUGINS.values() for a in p.aliases}


def default_metric_keys():
	out = []
	for p in PLUGINS.values():
		for k in p.metric_keys:
			if k not in out:
				out.append(k)
	return out


def semantic_knobs():
	out = {}
	for p in PLUGINS.values():
		out.update(p.semantic_knobs)
	return out


def experiment_specs():
	"""{name: ObjectiveSpec kwargs} for experiments/objectives.py."""
	out = {}
	for p in PLUGINS.values():
		if p.experiment is None:
			continue
		spec = dict(p.experiment)
		spec.setdefault('lp_obj_string', p.lp_name or p.name)
		spec.setdefault('description', p.description)
		out[p.name] = spec
	return out


# ================================================================ PLUGINS ==
# Column extractor names (resolved in generate_paper_table._EXTRACTORS):
#   'mean' <key>                mean of metrics[key][sim][soln]
#   'pct' <key>                 same x100
#   'stats' <key> [inner] [scale]   cross-sim stats_* aggregate
#   'lat_split' <part>          clean|congested|stranded volume split
#   'mlu_cell_latency'          min-latency-assignment latency (MLU group)
#   'lat_res_objective'         lat + gamma*RB scalar
#   'flash_crowd' / 'diurnal'   bisected critical intensities
#   'frozen_anchor' <frozen_key> <reactive_key> <scale>
#                               frozen metric for every method; the
#                               One-per-peering row shows the re-optimized
#                               (reactive) ceiling instead
_LAT_SPLIT_COLS = (
	('Congested vol', '<', 'lat_split', 'congested'),
	('Stranded vol', '<', 'lat_split', 'stranded'),
)
_OBJ_COL = ('Objective', '>', 'mean', 'objective_value_by_strategy')

register(ObjectivePlugin(
	name='avg_latency',
	default_order=0,
	description='Minimize traffic-weighted average user latency (+ gamma * '
				'resilience benefit during training). Baseline objective.',
	aliases=('latency', 'latency_resilience'),
	training_class='core.generic_objective:LatencyPlusResilienceObjective',
	experiment=dict(eval_phases=(
		'strategy_compare', 'pct_volume_within_latency',
		'failure_resilience', 'diurnal', 'flash_crowd')),
	eval_module='evaluations_for_latency_plus_resilience',
	# the shared/default pickle: no required key (every other objective
	# must prove itself so this pickle can't satisfy their coverage)
	table_group='Latency + g*Resilience', group_order=1, key_order=1,
	table_columns=(
		('Latency (ms)', '<', 'lat_split', 'clean'),
		('Congested vol', '<', 'lat_split', 'congested'),
		('Stranded vol', '<', 'lat_split', 'stranded'),
		('Subopt normal (ms)', '<', 'stats', 'stats_best_latencies', None, -1.0),
		('Subopt PoPP-fail (ms)', '<', 'stats',
		 'stats_popp_failures_latency_optimal_specific', 'avg_latency_difference', -1.0),
		('% cong PoPP-fail', '<', 'stats',
		 'stats_popp_failures_latency_optimal_specific', 'frac_vol_congested', 100.0),
		('Subopt PoP-fail (ms)', '<', 'stats',
		 'stats_pop_failures_latency_optimal_specific', 'avg_latency_difference', -1.0),
		('% cong PoP-fail', '<', 'stats',
		 'stats_pop_failures_latency_optimal_specific', 'frac_vol_congested', 100.0),
		('Flash-crowd resilience', '>', 'stats', 'stats_resilience_to_congestion'),
		('Diurnal resilience', '>', 'stats', 'stats_diurnal'),
		('Objective (lat+g*RB)', '<', 'lat_res_objective'),
	),
	key_columns=('Latency (ms)', '% cong PoPP-fail', '% cong PoP-fail',
				 'Flash-crowd resilience', 'Diurnal resilience'),
	tex_group='Failure Robustness',
	tex_subs={
		'Subopt PoPP-fail (ms)': 'Subopt ingress-fail (ms)',
		'% cong PoPP-fail': '% cong ingress-fail',
		'Subopt PoP-fail (ms)': 'Subopt site-fail (ms)',
		'% cong PoP-fail': '% cong site-fail',
		'Flash-crowd resilience': 'Flash crowd intensity',
		'Diurnal resilience': 'Diurnal intensity',
	},
	paper_table_default=True,
	semantic_knobs={'SCULPTOR_USE_RESILIENCE': '1', 'SCULPTOR_GT_RB': '0'},
))

register(ObjectivePlugin(
	name='max_util',
	default_order=2,
	description='Minimize maximum link utilization (canonical best-achievable '
				'peak util); the group\'s latency column is the min-latency '
				'assignment value.',
	aliases=('mlu',),
	lp='core.hard_objectives:solve_lp_max_util',
	eval_module='evaluations_for_mlu',
	required_metric_key='mlu_by_strategy',
	metric_keys=('mlu_by_strategy', 'objective_value_by_strategy'),
	table_group='MLU', group_order=0, key_order=0,
	table_columns=(
		('Latency (ms)', '<', 'mlu_cell_latency'),
		('MLU', '<', 'mean', 'mlu_by_strategy'),
	) + _LAT_SPLIT_COLS + (_OBJ_COL,),
	key_columns=('Latency (ms)', 'MLU'),
	tex_group='Latency + MLU',
	paper_table_default=True,
	semantic_knobs={'SCULPTOR_OBJ_MAXUTIL_ALPHA': '',
					'SCULPTOR_MLU_WEIGHT_MULT': ''},
))

register(ObjectivePlugin(
	name='lat_plus_max_util',
	description='Steady avg latency + alpha * max link utilization.',
	lp='core.hard_objectives:solve_lp_lat_plus_max_util',
	eval_module='evaluations_for_mlu',
	required_metric_key='mlu_by_strategy',
	semantic_knobs={'SCULPTOR_LATMLU_TERM': '',
					'SCULPTOR_LATMLU_STRAND_MULT': ''},
))

register(ObjectivePlugin(
	name='frac_beyond_optimal',
	default_order=3,
	description='Maximize the volume within 10ms of its one-per-peering '
				'latency (trained-objective soft-LP assignment, hard 10ms '
				'count, volume counted once).',
	lp='core.hard_objectives:solve_lp_frac_beyond_optimal',
	eval_module='evaluations_for_frac_beyond_optimal',
	required_metric_key='frac_within_threshold_by_strategy',
	metric_keys=('frac_within_threshold_by_strategy',
				 'objective_value_by_strategy'),
	table_group='Frac beyond optimal', group_order=2, key_order=3,
	table_columns=(
		('% within 10ms', '>', 'pct', 'frac_within_threshold_by_strategy'),
		_OBJ_COL,
	) + _LAT_SPLIT_COLS,
	key_columns=('% within 10ms',),
	tex_group='Latency Sensitive Services',
	tex_subs={'% within 10ms': '% within 10ms of optimal'},
	paper_table_default=True,
	semantic_knobs={'SCULPTOR_FRACB_SCALAR': '',
					'SCULPTOR_FRAC_BEYOND_REL': '',
					'SCULPTOR_HINGE_NOROUTE_MS': ''},
))

register(ObjectivePlugin(
	name='joint_priority',
	default_order=4,
	description='HPrio (latency-sensitive) LP solved first, LPrio (bulk) '
				'fills around it minimizing HPrio-weighted oversubscription '
				'-- de facto strict priority queueing.',
	aliases=('priorities',),
	lp_name='joint_latency_bulk_download',
	experiment=dict(lp_kwargs={'bulk_cap_limit': 100.0},
					eval_phases=('strategy_compare', 'priority_bulk_sweep')),
	eval_module='evaluations_for_priorities',
	required_metric_key='priority_by_strategy',
	metric_keys=('priority_by_strategy', 'hprio_latency_by_strategy',
				 'hprio_frac_routed_by_strategy', 'bulk_routable_by_strategy',
				 'critical_bulk_ratio_by_strategy',
				 'hprio_cong_swan_by_strategy', 'objective_value_by_strategy'),
	table_group='High + Low Priority Traffic', group_order=3, key_order=2,
	table_columns=(
		('Frac HPrio routed', '>', 'mean', 'hprio_frac_routed_by_strategy'),
		('HPrio latency (ms)', '<', 'mean', 'hprio_latency_by_strategy'),
		('Crit bulk ratio', '>', 'mean', 'critical_bulk_ratio_by_strategy'),
		('HPrio cong @SWAN', '<', 'mean', 'hprio_cong_swan_by_strategy'),
	) + _LAT_SPLIT_COLS + (_OBJ_COL,),
	key_columns=('HPrio latency (ms)', 'Crit bulk ratio'),
	tex_group='Traffic Classes',
	tex_subs={'HPrio cong @SWAN': 'HPrio cong @SWAN'},
	paper_table_default=True,
	semantic_knobs={'SCULPTOR_BULK_SLACK_DOM': '1e3'},
))

register(ObjectivePlugin(
	name='per_site_cost',
	default_order=1,
	description='Minimize traffic-weighted (latency + site_cost_alpha * '
				'site_cost); alpha is the cost-vs-latency knob.',
	aliases=('site_cost',),
	experiment=dict(lp_kwargs={'site_cost_alpha': 100.0},
					deployment_kwargs={'cost_type': 'carbon'},
					eval_phases=('strategy_compare', 'pct_volume_within_latency',
								 'site_cost_summary')),
	eval_module='evaluations_for_site_cost',
	required_metric_key='active_sites_by_strategy',
	metric_keys=('active_sites_by_strategy', 'weighted_site_cost_by_strategy',
				 'max_site_cost_load_by_strategy',
				 'avg_site_cost_load_by_strategy', 'objective_value_by_strategy'),
	table_group='Site cost', group_order=4, key_order=4,
	table_columns=(
		('Wgt max site cost', '<', 'mean', 'max_site_cost_load_by_strategy'),
		('Wgt avg site cost', '<', 'mean', 'weighted_site_cost_by_strategy'),
	) + _LAT_SPLIT_COLS + (_OBJ_COL,),
	key_columns=('Wgt avg site cost',),
	tex_group='Traffic Cost Across Sites',
	paper_table_default=True,
))

register(ObjectivePlugin(
	name='popp_failure_congestion',
	description='Mean over single-popp failures of the LP\'s congested '
				'volume fraction.',
	lp='core.hard_objectives:solve_lp_popp_failure_congestion',
))

register(ObjectivePlugin(
	name='frozen_failure_latency',
	description='(superseded by frozen_prefix) steady latency + gamma * '
				'BGP-fallback failure latency with a single-prefix pin; '
				'prices no-route at NO_ROUTE_LATENCY.',
	lp='core.hard_objectives:solve_lp_frozen_failure',
	semantic_knobs={'SCULPTOR_FROZEN_GAMMA': '', 'SCULPTOR_FROZEN_WHICH': ''},
))

register(ObjectivePlugin(
	name='site_failure',
	description='Steady avg latency + exhaustive mean over per-PoP failures '
				'with each user pinned to its steady-state prefix; soft '
				'no_route/congestion penalties; (1-beta)*steady + beta*mean.',
	experiment=dict(
		lp_kwargs={'site_failure_beta': 0.5,
				   'site_failure_no_route_penalty': 50.0,
				   'site_failure_congestion_penalty': 2.0},
		eval_phases=('strategy_compare', 'static_failure_resilience')),
))

register(ObjectivePlugin(
	name='frozen_prefix',
	default_order=5,
	description='Joint LP over w(ug, prefix) shared across the normal '
				'scenario + per-iteration-sampled single-popp failures: '
				'normal latency + gamma * mean(failure term), soft penalties '
				'(no_route > congestion). The user->prefix freeze is '
				'structural: one static allocation, BGP fallback within the '
				'pinned prefix is the only post-failure adaptation.',
	aliases=('frozen',),
	lp='core.hard_objectives:solve_lp_frozen_prefix',
	training_class='core.generic_objective:FrozenPrefixObjective',
	experiment=dict(eval_phases=('strategy_compare',)),
	eval_module='evaluations_for_frozen_prefix',
	required_metric_key='frozen_fail_latency_by_strategy',
	metric_keys=('frozen_steady_latency_by_strategy',
				 'frozen_fail_latency_by_strategy',
				 'frozen_fail_cong_by_strategy',
				 'frozen_fail_no_route_by_strategy',
				 'frozen_fail_worst_cong_by_strategy',
				 'frozen_fail_worst_no_route_by_strategy',
				 'reactive_steady_latency_by_strategy',
				 'reactive_fail_latency_by_strategy',
				 'reactive_fail_cong_by_strategy',
				 'reactive_fail_no_route_by_strategy',
				 'objective_value_by_strategy'),
	table_group='Frozen failover', group_order=5, key_order=5,
	table_columns=(
		('Steady latency (ms)', '<', 'mean', 'frozen_steady_latency_by_strategy'),
		('Latency (ms)', '<', 'frozen_anchor',
		 'frozen_fail_latency_by_strategy', 'reactive_fail_latency_by_strategy', 1.0),
		('% cong fail', '<', 'frozen_anchor',
		 'frozen_fail_cong_by_strategy', 'reactive_fail_cong_by_strategy', 100.0),
		('% no-route fail', '<', 'frozen_anchor',
		 'frozen_fail_no_route_by_strategy', 'reactive_fail_no_route_by_strategy', 100.0),
	) + _LAT_SPLIT_COLS + (_OBJ_COL,),
	key_columns=('Latency (ms)', '% cong fail', '% no-route fail'),
	tex_group='Frozen Failover',
	tex_subs={'% cong fail': '% cong ingress-fail',
			  '% no-route fail': '% no-route ingress-fail'},
	paper_table_default=True,
	semantic_knobs={
		'SCULPTOR_FROZEN_PREFIX_GAMMA': '',
		'SCULPTOR_FROZEN_PREFIX_N_FAIL': '',
		'SCULPTOR_FROZEN_PREFIX_NO_ROUTE_PENALTY': '',
		'SCULPTOR_FROZEN_PREFIX_CONGESTION_PENALTY': '',
		'SCULPTOR_FROZEN_PREFIX_EXPLORE_FRAC': '',
		'SCULPTOR_FROZEN_PREFIX_EVAL_N_FAIL': '',
		'SCULPTOR_FROZEN_PREFIX_ANCHOR_N_FAIL': '',
	},
))
