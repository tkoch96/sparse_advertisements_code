"""
SCULPTOR end-to-end iteration timing.

This is the production-realistic number: "how long does one full gradient
descent iteration take?" It bundles everything that matters -- worker fan-out,
exploration, latency-benefit MC, modeled-objective evaluation, LP solving --
into a single wall-time measurement per iteration.

We run `Sparse_Advertisement_Solver.solve()` for N iterations at each size,
then divide. The output projects production runtime at 200 iterations and at
your worker count.

Run:
	pytest -v -s tests/test_sas_iteration_timing.py
	BENCH_ITER_SIZES="small,decent" pytest -v -s tests/test_sas_iteration_timing.py

Sizes default to ['small'] -- decent and beyond can take minutes per call to
solve() because of how many sub-LPs are evaluated per iteration.
"""
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import pytest


ALL_SIZES = ['really_friggin_small', 'small', 'decent', 'med', 'large']


def _selected_sizes():
	env = os.environ.get('BENCH_ITER_SIZES', '').strip().lower()
	if not env:
		return ['small']
	if env == 'all':
		return list(ALL_SIZES)
	asked = [s.strip() for s in env.split(',') if s.strip()]
	unknown = [s for s in asked if s not in ALL_SIZES]
	if unknown:
		raise ValueError("Unknown BENCH_ITER_SIZES entries: {}".format(unknown))
	return asked


SELECTED_SIZES = _selected_sizes()

# How many SCULPTOR iterations to run per size (excluding setup).
N_ITERS_BY_SIZE = {
	'really_friggin_small': 5,
	'small': 5,
	'decent': 3,
	'med': 2,
	'large': 1,
}

_RESULTS = OrderedDict()


def _build_sas(size):
	"""Build a Sparse_Advertisement_Solver + a running Worker_Manager. Same
	wiring `eval_latency_failure.py` and the convergence test use."""
	import random
	import worker_comms_ray
	sys.modules['worker_comms'] = worker_comms_ray

	from deployment_setup import get_random_deployment
	from sparse_advertisements_v3 import Sparse_Advertisement_Solver
	from worker_comms import Worker_Manager  # alias picks up _ray

	random.seed(31415)
	np.random.seed(31415)
	deployment = get_random_deployment(size, port=31600)
	deployment['generic_objective'] = 'avg_latency'

	n_pops = len(set(p[0] for p in deployment['popps']))
	n_prefixes_safe = max(n_pops + 2, 4)

	sas = Sparse_Advertisement_Solver(
		deployment,
		lambduh=0, gamma=0, verbose=False,
		n_prefixes=n_prefixes_safe, with_capacity=False,
	)
	wm = Worker_Manager(sas.get_init_kwa(), deployment)
	wm.start_workers()
	sas.set_worker_manager(wm)
	sas.update_deployment(deployment)
	return sas, deployment


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.gurobi
@pytest.mark.benchmark
@pytest.mark.parametrize('size', SELECTED_SIZES)
def test_sas_iteration_timing(size):
	"""Run SAS.solve() for a small number of iterations and report wall time
	per iteration. Projects total runtime for the production 200-iteration
	loop."""
	pytest.importorskip("ray")

	print("\n--- size = {} ---".format(size))
	t_setup0 = time.time()
	sas, deployment = _build_sas(size)
	setup_s = time.time() - t_setup0
	print("  setup: {:.2f}s".format(setup_s))

	n_iters = N_ITERS_BY_SIZE.get(size, 3)
	sas.max_n_iter = n_iters

	t0 = time.time()
	try:
		sas.solve()
	finally:
		try:
			sas.worker_manager.stop_workers()
		except Exception:
			pass
	elapsed = time.time() - t0

	# `solve()` runs a setup phase then N optimization iterations. Approximate
	# per-iter cost by dividing; tighter would require instrumenting solve()
	# itself, but for budget projection this is good enough.
	per_iter = elapsed / max(n_iters, 1)
	# Project production: 200 SCULPTOR iterations (per your description).
	proj_200 = per_iter * 200

	print("  ran {} iterations in {:.1f}s  ({:.2f}s/iter)".format(
		n_iters, elapsed, per_iter))
	print("  projection: 200 iterations would take ~{:.1f}min ({:.1f}h)".format(
		proj_200 / 60, proj_200 / 3600))

	_RESULTS[size] = {
		'n_popps': len(deployment['popps']),
		'n_ugs': len(deployment['ugs']),
		'n_iters': n_iters,
		'elapsed_s': elapsed,
		'per_iter_s': per_iter,
		'setup_s': setup_s,
	}


def pytest_terminal_summary(terminalreporter, exitstatus=None, config=None):
	if not _RESULTS:
		return
	tr = terminalreporter
	tr.write_sep("=", "SCULPTOR per-iteration wall time")
	tr.write_line("{:>10} | {:>5} | {:>6} | {:>6} | {:>10} | {:>11} | {:>14}".format(
		'size', 'iters', 'popps', 'ugs', 'per-iter', '200-iter', '200-iter (h)'))
	tr.write_line("-" * 80)
	for size, r in _RESULTS.items():
		tr.write_line("{:>10} | {:>5} | {:>6} | {:>6} | {:>8.2f}s | {:>9.1f}s | {:>13.2f}h".format(
			size, r['n_iters'], r['n_popps'], r['n_ugs'],
			r['per_iter_s'],
			r['per_iter_s'] * 200,
			r['per_iter_s'] * 200 / 3600))

	tr.write_line("")
	tr.write_line("Reading: 'per-iter' is the average wall time for one full SAS iteration")
	tr.write_line("at this size, including exploration + latency-benefit MC + modeled-objective.")
	tr.write_line("'200-iter' is your production loop budget projection.")
	tr.write_line("At 100 Gurobi workers, divide by ~the parallelism fraction your gradient")
	tr.write_line("step actually achieves -- worker saturation in practice is typically 40-80%.")
