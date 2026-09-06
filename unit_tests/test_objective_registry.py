"""The central objective registry (core/objective_registry.py) is THE one
place an objective is declared; these tests pin (a) that every consumer
really derives from it and the wiring is complete for every plugin, and
(b) that the derived views reproduce the legacy literals they replaced
(2026-09-06) so the paper table's shape is unchanged.

Run:
	SCULPTOR_XOBJS=1 pytest -q unit_tests/test_objective_registry.py
"""
import importlib
import os

import numpy as np
import pytest

os.environ.setdefault('SCULPTOR_XOBJS', '1')

from core import objective_registry as R


# --------------------------------------------------------------- snapshots
# the literals the registry replaced (verbatim, 2026-09-06) -- guards the
# paper-table shape through the migration
LEGACY_ROUTES = {
	'avg_latency':         'evaluations_for_latency_plus_resilience',
	'max_util':            'evaluations_for_mlu',
	'lat_plus_max_util':   'evaluations_for_mlu',
	'joint_priority':      'evaluations_for_priorities',
	'per_site_cost':       'evaluations_for_site_cost',
	'frac_beyond_optimal': 'evaluations_for_frac_beyond_optimal',
}
LEGACY_REQUIRED = {
	'per_site_cost': 'active_sites_by_strategy',
	'max_util': 'mlu_by_strategy',
	'lat_plus_max_util': 'mlu_by_strategy',
	'frac_beyond_optimal': 'frac_within_threshold_by_strategy',
	'joint_priority': 'priority_by_strategy',
}
LEGACY_DEFAULTS = ['avg_latency', 'per_site_cost', 'max_util',
				   'frac_beyond_optimal', 'joint_priority']
LEGACY_ALIASES = {'priorities': 'joint_priority', 'latency': 'avg_latency',
				  'latency_resilience': 'avg_latency',
				  'site_cost': 'per_site_cost', 'mlu': 'max_util'}
LEGACY_KEY_COLUMNS = [
	('MLU', 'Latency (ms)'), ('MLU', 'MLU'),
	('Latency + g*Resilience', 'Latency (ms)'),
	('Latency + g*Resilience', '% cong PoPP-fail'),
	('Latency + g*Resilience', '% cong PoP-fail'),
	('Latency + g*Resilience', 'Flash-crowd resilience'),
	('Latency + g*Resilience', 'Diurnal resilience'),
	('High + Low Priority Traffic', 'HPrio latency (ms)'),
	('High + Low Priority Traffic', 'Crit bulk ratio'),
	('Frac beyond optimal', '% within 10ms'),
	('Site cost', 'Wgt avg site cost'),
]
LEGACY_GROUP_ORDER = ['MLU', 'Latency + g*Resilience', 'Frac beyond optimal',
					  'High + Low Priority Traffic', 'Site cost']
LEGACY_GROUP_SUBS = {
	'MLU': ['Latency (ms)', 'MLU', 'Congested vol', 'Stranded vol', 'Objective'],
	'Latency + g*Resilience': [
		'Latency (ms)', 'Congested vol', 'Stranded vol', 'Subopt normal (ms)',
		'Subopt PoPP-fail (ms)', '% cong PoPP-fail', 'Subopt PoP-fail (ms)',
		'% cong PoP-fail', 'Flash-crowd resilience', 'Diurnal resilience',
		'Objective (lat+g*RB)'],
	'Frac beyond optimal': ['% within 10ms', 'Objective', 'Congested vol',
							'Stranded vol'],
	'High + Low Priority Traffic': [
		'Frac HPrio routed', 'HPrio latency (ms)', 'Crit bulk ratio',
		'HPrio cong @SWAN', 'Congested vol', 'Stranded vol', 'Objective'],
	'Site cost': ['Wgt max site cost', 'Wgt avg site cost', 'Congested vol',
				  'Stranded vol', 'Objective'],
}
LEGACY_SEMANTIC_KNOBS = {
	'SCULPTOR_GT_RB': '0', 'SCULPTOR_USE_RESILIENCE': '1',
	'SCULPTOR_OBJ_MAXUTIL_ALPHA': '', 'SCULPTOR_MLU_WEIGHT_MULT': '',
	'SCULPTOR_LATMLU_TERM': '', 'SCULPTOR_LATMLU_STRAND_MULT': '',
	'SCULPTOR_HINGE_NOROUTE_MS': '', 'SCULPTOR_FRACB_SCALAR': '',
	'SCULPTOR_FRAC_BEYOND_REL': '', 'SCULPTOR_FROZEN_GAMMA': '',
	'SCULPTOR_FROZEN_WHICH': '', 'SCULPTOR_BULK_SLACK_DOM': '1e3',
}


@pytest.mark.unit
def test_legacy_views_preserved():
	routes = R.eval_routes()
	for k, v in LEGACY_ROUTES.items():
		assert routes[k] == v
	req = R.required_metric_keys()
	for k, v in LEGACY_REQUIRED.items():
		assert req[k] == v
	assert R.paper_table_defaults()[:len(LEGACY_DEFAULTS)] == LEGACY_DEFAULTS
	al = R.aliases()
	for k, v in LEGACY_ALIASES.items():
		assert al[k] == v
	assert R.key_columns()[:len(LEGACY_KEY_COLUMNS)] == LEGACY_KEY_COLUMNS
	groups = R.table_groups()
	assert [g for g, _, _ in groups][:len(LEGACY_GROUP_ORDER)] == LEGACY_GROUP_ORDER
	for g, _obj, cols in groups:
		if g in LEGACY_GROUP_SUBS:
			assert [c[0] for c in cols] == LEGACY_GROUP_SUBS[g], g
	knobs = R.semantic_knobs()
	for k, v in LEGACY_SEMANTIC_KNOBS.items():
		assert knobs[k] == v, k


@pytest.mark.unit
def test_consumers_derive_from_registry():
	from core.solve_lp_assignment import generic_lp_functions
	import core.hard_objectives as ho
	from core.generic_objective import OBJECTIVE_CLASSES
	from evaluations.objectives import objective_hooks
	from core import depstore
	from evaluations import wrapper_eval
	from experiments import objectives as exp_objectives
	assert ho.REGISTERED_OBJECTIVES == R.extension_lp_functions()
	ho.register()
	for name, fn in R.extension_lp_functions().items():
		assert generic_lp_functions[name] is fn
	assert OBJECTIVE_CLASSES == R.training_classes()
	assert objective_hooks._ROUTES == R.eval_routes()
	for k, v in R.semantic_knobs().items():
		assert depstore.SEMANTIC_KNOBS[k] == v
	for k in R.default_metric_keys():
		assert k in wrapper_eval.default_metrics
	for name in R.experiment_specs():
		assert exp_objectives.get(name).name == name


@pytest.mark.unit
def test_every_plugin_is_completely_wired():
	"""The failure modes the registry exists to prevent: an unrouted
	objective, a table group without a required key, an unresolvable
	LP/class path, an eval module that doesn't import."""
	from core.solve_lp_assignment import generic_lp_functions
	import core.hard_objectives as ho
	ho.register()
	for p in R.PLUGINS.values():
		if p.lp:
			assert callable(R.resolve(p.lp)), p.name
			assert p.name in generic_lp_functions, p.name
		if p.training_class:
			assert isinstance(R.resolve(p.training_class), type), p.name
		if p.eval_module:
			mod = importlib.import_module('evaluations.objectives.' + p.eval_module)
			assert callable(getattr(mod, 'run', None)), p.name
			assert p.name in getattr(mod, 'OBJECTIVES', ()), (
				'{}: eval module {} does not list it in OBJECTIVES'.format(
					p.name, p.eval_module))
		if p.table_group:
			assert p.eval_module, '{}: table columns but no eval suite'.format(p.name)
			assert p.name == 'avg_latency' or p.required_metric_key, (
				'{}: table group needs required_metric_key (else the '
				'avg_latency pickle satisfies its coverage)'.format(p.name))
			for sub in p.key_columns:
				assert sub in [c[0] for c in p.table_columns], (p.name, sub)
		# the LP name the dispatcher is called with must exist
		assert R.lp_name_for(p.name) in generic_lp_functions or not (p.lp or p.lp_name), p.name


@pytest.mark.unit
def test_paper_table_columns_resolve():
	"""generate_paper_table must be able to build every declared column."""
	from evaluations import generate_paper_table as gpt
	assert gpt.GROUPS and gpt.KEY_COLUMNS
	labels = {'{}|{}'.format(g, lab) for g, _o, subs in gpt.GROUPS for lab, _d, _f in subs}
	for g, sub in gpt.KEY_COLUMNS:
		assert '{}|{}'.format(g, sub) in labels, (g, sub)
	assert 'Frozen failover|% no-route fail' in labels
	assert gpt.OBJECTIVE_REQUIRED_KEY['frozen_prefix'] == 'frozen_fail_latency_by_strategy'
	assert 'frozen_prefix' in gpt.DEFAULT_OBJECTIVES
	assert gpt.OBJECTIVE_ALIASES['frozen'] == 'frozen_prefix'


@pytest.mark.unit
def test_validate_names():
	assert R.validate_names(['mlu', 'frozen']) == ['max_util', 'frozen_prefix']
	with pytest.raises(KeyError):
		R.validate_names(['no_such_objective'])
	with pytest.raises(ValueError):
		R.register(R.get('avg_latency'))


@pytest.mark.unit
def test_frozen_suite_metrics_small():
	"""frozen_failure_metrics + reactive_optimal_metrics on the small
	deployment: three DISTINCT metrics, sane ranges, and the frozen
	one-per-peering pathology (strands users) vs the reactive anchor
	(never strands, never congests)."""
	from unit_tests.test_lp_correctness import _setup
	from core.frozen_prefix_eval import (frozen_failure_metrics,
										 reactive_optimal_metrics)
	worker, dep, adv, rti = _setup()
	n = worker.n_popps
	opp = np.eye(n)
	fz = frozen_failure_metrics(worker, opp, which='popps')
	assert fz['n_failures'] == n
	assert 0.0 <= fz['fail_frac_cong'] <= 1.0
	assert 0.0 <= fz['fail_frac_no_route'] <= 1.0
	assert fz['fail_frac_no_route'] > 0, 'frozen OPP must strand (no backup)'
	rc = reactive_optimal_metrics(worker, opp, which='popps', n_fail=0)
	assert rc['n_failures'] == n
	assert rc['fail_frac_no_route'] < 1e-9
	assert rc['fail_frac_cong'] < 1e-9
	assert np.isfinite(rc['fail_latency_ms'])
	# anycast (single prefix, every popp): never strands, may congest
	fa = frozen_failure_metrics(worker, adv, which='popps')
	assert fa['fail_frac_no_route'] < 1e-9
	assert np.isfinite(fa['fail_latency_ms'])
	# eval sampling knob caps the sweep deterministically
	fz5 = frozen_failure_metrics(worker, opp, which='popps', eval_n_fail=5)
	assert fz5['n_failures'] == 5
