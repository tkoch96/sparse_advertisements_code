"""
Smoke tests. These intentionally assert almost nothing -- their purpose is to
prove the fixture scaffolding works and to give you a template for real tests.

When you want to test specific behavior, copy one of these and tighten the
assertions.
"""
import pytest


# ---------------------------------------------------------------------------#
# Deployment fixtures load and look sane.
# ---------------------------------------------------------------------------#
@pytest.mark.unit
def test_tiny_deployment_has_expected_shape(tiny_deployment):
	"""The minimal deployment we build for tests has the fields downstream code
	expects. If this fails after a refactor to deployment_setup, every other
	test will too -- so this gives a clear single point of failure."""
	required = {
		'ugs', 'popps', 'ug_perfs', 'ug_to_vol',
		'whole_deployment_ugs', 'whole_deployment_ug_perfs',
		'whole_deployment_ug_to_vol', 'ingress_priorities',
		'link_capacities', 'dpsize',
	}
	missing = required - set(tiny_deployment.keys())
	assert not missing, "tiny_deployment is missing fields: {}".format(missing)
	assert tiny_deployment['dpsize'] == 'really_friggin_small'


@pytest.mark.unit
def test_worker_deployment_is_full(tiny_deployment, worker_deployment):
	"""Workers receive the full deployment (no UG sharding)."""
	assert set(worker_deployment['ugs']) == set(tiny_deployment['ugs'])


# ---------------------------------------------------------------------------#
# Worker instantiates and exposes the dispatch surface we expect.
# ---------------------------------------------------------------------------#
@pytest.mark.unit
@pytest.mark.gurobi
def test_worker_constructs(worker):
	"""The biggest 'does the setup actually work' check. Building a worker
	exercises:
	  * the deployment-splitting plumbing,
	  * Optimal_Adv_Wrapper.__init__,
	  * init_all_vars (incl. init_persistent_lp -> Gurobi env creation).
	If this passes, downstream tests can assume a usable worker.
	"""
	assert worker.worker_i == 0
	# These attributes are set by init_all_vars / __init__.
	assert hasattr(worker, 'model')           # persistent Gurobi model
	assert hasattr(worker, 'var_pool')
	assert hasattr(worker, 'lbx')


@pytest.mark.unit
@pytest.mark.gurobi
def test_worker_handle_msg_dispatches(worker):
	"""handle_msg is the shim Worker_Manager_ray uses. Confirm it routes a
	pickled (cmd, data) tuple to the right _cmd_* method."""
	import pickle
	msg = pickle.dumps(('increment_iter', None))
	assert worker.handle_msg(msg) == "ACK"


@pytest.mark.unit
@pytest.mark.gurobi
def test_worker_unknown_cmd_does_not_crash(worker):
	"""Unknown commands return 'ERROR' instead of raising. Worker_Manager
	relies on this to keep the cluster alive when a buggy driver sends garbage."""
	import pickle
	msg = pickle.dumps(('this_is_not_a_real_cmd', None))
	assert worker.handle_msg(msg) == "ERROR"
