"""
Pytest fixtures for the sparse_advertisements_code test suite.

Goal: hide the painful setup work behind named fixtures so individual tests
can read like:

    def test_solve_lp_is_fast(worker, tiny_advertisement):
        result = worker._cmd_solve_lp([(0, tiny_advertisement, deployment, False)])
        assert result[0][1]['solved']

The fixtures:

  * tiny_deployment            -- a minimal `really_friggin_small` deployment,
                                  generated once per test session with a fixed
                                  seed for reproducibility.
  * worker_deployment          -- the full deployment, the
                                  shape every worker actually receives.
  * init_kwa                   -- kwargs that Optimal_Adv_Wrapper.__init__ wants.
  * worker (function-scoped)   -- a fresh _LocalPathDistributionComputer
                                  (the non-Ray version) holding its own
                                  Gurobi model. Each test gets a clean one.
  * worker_session (session)   -- the same, but shared across tests. Faster.
                                  Use only when you won't mutate state.
  * gurobi_available           -- session-scoped boolean. Tests can use it
                                  to skip if Gurobi isn't usable on this box.
  * tiny_advertisement         -- a valid advertisement matrix for the worker.
  * lp_timer                   -- helper: times N LP solves, returns stats.

Markers (declared in pytest.ini):
    @pytest.mark.unit         fast, in-process
    @pytest.mark.integration  exercises Ray / Worker_Manager_ray
    @pytest.mark.gurobi       requires a live Gurobi license
    @pytest.mark.slow         > a few seconds

Run subsets:
    pytest                                    # everything
    pytest -m unit                            # fast checks
    pytest -m "not slow"                      # exclude long ones
    pytest tests/test_smoke.py -v             # single file
"""
import os
import sys
import time
import random
import contextlib
from pathlib import Path

import numpy as np
import pytest

# Make the project importable when running `pytest` from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------#
# Session-wide setup: deterministic seeds + a sandboxed log dir so tests don't
# scribble into the real logs/ folder.
# ---------------------------------------------------------------------------#
@pytest.fixture(scope='session', autouse=True)
def _isolate_environment(tmp_path_factory):
	"""Run all tests against a tmp logs/ dir and with fixed RNG seeds."""
	tmp_logs = tmp_path_factory.mktemp("logs")
	# constants.LOG_DIR is imported by name in many modules, so we patch the
	# module attribute. We don't need to undo this because tests run in their
	# own process.
	import constants
	constants.LOG_DIR = str(tmp_logs)

	random.seed(31415)
	np.random.seed(31415)
	yield


# ---------------------------------------------------------------------------#
# Deployment fixtures.
# ---------------------------------------------------------------------------#
@pytest.fixture(scope='session')
def tiny_deployment():
	"""A minimal but real deployment. Built once per session.

	Uses 'really_friggin_small' (2 pops, 20 peers, 75 user groups). Big enough
	that the LP solver does real work, small enough that __init__ + an LP
	solve finish in under a second."""
	random.seed(31415)
	np.random.seed(31415)
	from deployment_setup import get_random_deployment
	dep = get_random_deployment('really_friggin_small', port=31600)
	return dep


@pytest.fixture(scope='session')
def worker_deployment(tiny_deployment):
	"""The dict shape a worker receives from Worker_Manager: the full
	deployment (every worker computes over the entire deployment)."""
	return tiny_deployment


@pytest.fixture(scope='session')
def init_kwa():
	"""Minimum kwargs Optimal_Adv_Wrapper.__init__ accepts. Mirrors
	Worker_Manager.get_init_kwa() defaults."""
	return {
		'lambduh': 1.0,
		'gamma': 0,
		'verbose': False,
		'n_prefixes': None,
		'with_capacity': False,
		'save_run_dir': None,
		'generic_objective': 'avg_latency',
	}


# ---------------------------------------------------------------------------#
# Gurobi availability probe.
# ---------------------------------------------------------------------------#
@pytest.fixture(scope='session')
def gurobi_available():
	"""True iff we can build and solve a trivial Gurobi model. Tests that need
	Gurobi should depend on this and skip when False, e.g.:

	    def test_solve(worker, gurobi_available):
	        if not gurobi_available:
	            pytest.skip("No Gurobi license available on this machine")
	"""
	try:
		import gurobipy as gp
		m = gp.Model("license_probe")
		m.Params.OutputFlag = 0
		x = m.addVar()
		m.setObjective(x)
		m.optimize()
		return True
	except Exception:
		return False


# ---------------------------------------------------------------------------#
# Worker fixtures (the non-Ray local class so tests are fast).
# ---------------------------------------------------------------------------#
def _build_worker(worker_deployment, init_kwa):
	"""Internal: build a _LocalPathDistributionComputer without Ray.

	This is the same class body the Ray actor wraps; instantiating it directly
	gives us a normal Python object whose methods we can call synchronously.
	"""
	from path_distribution_computer_ray import _LocalPathDistributionComputer
	return _LocalPathDistributionComputer(
		worker_i=0,
		deployment=worker_deployment,
		init_kwargs=init_kwa,
	)


@pytest.fixture
def worker(worker_deployment, init_kwa, gurobi_available):
	"""A fresh worker for each test. Skips automatically if Gurobi isn't
	available, since the worker's __init__ builds the persistent LP model.
	Function-scoped so state from one test never leaks into the next."""
	if not gurobi_available:
		pytest.skip("Gurobi license not available on this machine")
	return _build_worker(worker_deployment, init_kwa)


@pytest.fixture(scope='session')
def worker_session(worker_deployment, init_kwa, gurobi_available):
	"""Like `worker`, but shared across the whole test session. Use this for
	read-only tests where you want the (potentially slow) __init__ to happen
	exactly once. DO NOT use for tests that mutate worker state."""
	if not gurobi_available:
		pytest.skip("Gurobi license not available on this machine")
	return _build_worker(worker_deployment, init_kwa)


# ---------------------------------------------------------------------------#
# Small helpers.
# ---------------------------------------------------------------------------#
@pytest.fixture
def tiny_advertisement(worker_deployment):
	"""A valid advertisement matrix for the tiny deployment.

	Shape: (n_popps, n_prefixes). Each column is one prefix; values > 0.5 mean
	the popp is advertised on that prefix. We default to a single prefix that
	includes all popps -- the simplest non-trivial case."""
	n_popps = len(worker_deployment['popps'])
	adv = np.ones((n_popps, 1))
	return adv


class TimingStats:
	"""Tiny stats holder for benchmark fixtures."""
	def __init__(self, samples_s):
		self.samples_s = list(samples_s)

	def summary(self):
		import statistics
		s = self.samples_s
		return {
			'n': len(s),
			'min_ms': 1000 * min(s),
			'mean_ms': 1000 * (sum(s) / len(s)),
			'median_ms': 1000 * statistics.median(s),
			'max_ms': 1000 * max(s),
			'p95_ms': 1000 * sorted(s)[max(0, int(len(s) * 0.95) - 1)],
			'total_s': sum(s),
		}


@pytest.fixture
def lp_timer():
	"""Returns a function that times N invocations of a thunk and returns
	TimingStats. Useful as a building block for benchmark tests.

	Example:
	    def test_solve_lp_throughput(worker, tiny_advertisement, lp_timer, worker_deployment):
	        def one_solve():
	            data = [(0, tiny_advertisement, worker_deployment, False)]
	            worker._cmd_solve_lp(data)
	        stats = lp_timer(one_solve, n=50)
	        print(stats.summary())
	"""
	def _run(thunk, n=10, warmup=2):
		for _ in range(warmup):
			thunk()
		samples = []
		for _ in range(n):
			t0 = time.perf_counter()
			thunk()
			samples.append(time.perf_counter() - t0)
		return TimingStats(samples)
	return _run


@contextlib.contextmanager
def _timer():
	t0 = time.perf_counter()
	out = {}
	yield out
	out['elapsed_s'] = time.perf_counter() - t0


@pytest.fixture
def stopwatch():
	"""Context manager: `with stopwatch() as t: do_work(); print(t['elapsed_s'])`."""
	return _timer
