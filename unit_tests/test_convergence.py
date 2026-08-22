"""
Convergence tests.

Two kinds of tests live here:

  1. test_gradient_descent_step_reduces_objective (unit, fast):
		The core claim of any gradient-descent algorithm: stepping in the
		negative gradient direction reduces the objective. We compute the
		objective at advertisement A, take a small numerical step, recompute,
		and assert the objective went down (or at least didn't go up by more
		than noise). Doesn't need Worker_Manager / Ray; runs in seconds.

  2. test_sas_solve_converges_end_to_end (integration, slow, skipped by default):
		Builds a real Sparse_Advertisement_Solver with a Ray-backed
		Worker_Manager and runs a handful of iterations of SAS.solve(),
		asserting the final objective is at most the initial objective.
		Slow (seconds to a minute) and depends on Ray + the full stack being
		healthy -- so it doubles as a smoke test for `eval_all_solution_types`.

Markers (from pytest.ini):
  unit         -- in-process, no Ray.
  integration  -- exercises Worker_Manager / Ray actors.
  slow         -- can take >>1s.

Run the cheap one only:
	pytest -v tests/test_convergence.py -m "unit"

Run everything (assumes Ray installed and Gurobi license available):
	pytest -v -s tests/test_convergence.py
"""
import os
import time

import numpy as np
import pytest


def _setup_worker(size='small'):
	"""Same scaffold as test_lp_correctness, but local to this file so it can
	change independently."""
	import random
	random.seed(31415)
	np.random.seed(31415)
	from core.deployment_setup import get_random_deployment
	from core.path_distribution_computer import _LocalPathDistributionComputer

	dep = get_random_deployment(size, port=31600)
	subdep = dep
	init_kwa = {
		'lambduh': 1.0, 'gamma': 0, 'verbose': False,
		'n_prefixes': None, 'with_capacity': False,
		'save_run_dir': None, 'generic_objective': 'avg_latency',
	}
	worker = _LocalPathDistributionComputer(
		worker_i=0, deployment=subdep, init_kwargs=init_kwa)
	return worker, dep


# ---------------------------------------------------------------------------#
# Unit test: gradient direction is consistent with objective.
#
# Strategy: compute the modeled objective at advertisement A. Build a "near
# neighbour" A' that toggles a single popp ON if it had the highest measured
# latency benefit, or OFF if turning it off improves the objective. Compute
# the objective at A'. Across many random A draws, *some* fraction must move
# in the favorable direction -- otherwise the gradient signal in the system
# is broken.
#
# We deliberately keep this test cheap and statistical: it doesn't say "every
# step is improving," it says "the gradient is informative more often than
# random would be." That's the falsifiable claim that fails when something
# real breaks (e.g., a sign error in the latency_benefit code).
# ---------------------------------------------------------------------------#
@pytest.mark.unit
@pytest.mark.gurobi
def test_gradient_step_improves_objective_on_average():
	"""Across N random advertisements, toggling the single most-beneficial
	popp should improve the objective more than half the time. If it doesn't,
	either the objective is noisy beyond usefulness or the latency_benefit
	signal has the wrong sign."""
	worker, deployment = _setup_worker('small')
	n_popps = len(deployment['popps'])
	rng = np.random.default_rng(2026)

	N_TRIALS = 8
	improvements = 0
	deltas = []

	def obj_at(adv):
		"""Modeled objective: -avg_latency (more negative = better)."""
		rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
		ret = worker.solve_generic_lp_persistent(rti, 'avg_latency')
		assert ret['solved'], "LP did not solve during gradient test"
		return float(ret['objective'])

	for trial in range(N_TRIALS):
		# Random adv with at least one popp on.
		adv = (rng.random((n_popps, 1)) > 0.5).astype(float)
		if adv.sum() == 0:
			adv[rng.integers(n_popps), 0] = 1.0
		base_obj = obj_at(adv)

		# Try toggling each popp in turn and keep the best single-flip neighbor.
		best_delta = 0.0
		for j in range(n_popps):
			candidate = adv.copy()
			candidate[j, 0] = 1.0 - candidate[j, 0]
			if candidate.sum() == 0:
				continue  # never go to the empty advertisement
			try:
				cand_obj = obj_at(candidate)
			except Exception:
				continue
			delta = cand_obj - base_obj
			if delta < best_delta:
				best_delta = delta

		deltas.append(best_delta)
		if best_delta < -1e-6:
			improvements += 1

	# Assertion: of the N trials, the best-neighbour step must be a strict
	# improvement in *most* of them. If the LP / objective code is healthy,
	# this is nearly always the case -- at least one popp toggle helps. If
	# it's not, the objective gradient signal is broken.
	frac = improvements / N_TRIALS
	print("\nimprovement fraction across {} trials: {:.0%}".format(N_TRIALS, frac))
	print("min/mean/max delta-objective: {:.4f} / {:.4f} / {:.4f}".format(
		min(deltas), sum(deltas) / len(deltas), max(deltas)))
	assert frac >= 0.5, (
		"Objective gradient signal looks broken: only {:.0%} of random "
		"advertisements had an improving single-popp neighbour. Expected "
		">=50%. Either the LP is wrong or the latency_benefit sign is flipped."
		.format(frac)
	)


# ---------------------------------------------------------------------------#
# Integration: full SAS.solve() convergence smoke test.
#
# Skipped by default (slow + heavy deps). Enable with:
#     pytest -v -s tests/test_convergence.py::test_sas_solve_converges_end_to_end
#
# This is the test that would catch the kind of bug you just hit
# (worker_socket.send broken under Ray) because it runs the actual solve()
# loop through Worker_Manager.
# ---------------------------------------------------------------------------#
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.gurobi
def test_sas_solve_converges_end_to_end():
	"""Build a real Sparse_Advertisement_Solver on a tiny deployment, run a
	few iterations, and assert the final objective is at most the initial
	objective. Failure here means either:
	  * the gradient/optimization machinery is broken, or
	  * the Worker_Manager dispatch is broken (e.g., Ray ActorHandle.send),
		or
	  * Ray itself isn't initialised cleanly.
	Any of these would also break `python eval_all_solution_types.py --dpsize small`.
	"""
	# Lazy imports so this file still parses if Ray isn't installed.
	ray = pytest.importorskip("ray")

	# Use the small (not really_friggin_small) deployment because the SCULPTOR
	# learning loop occasionally returns NaNs on degenerately tiny problems.
	from core.deployment_setup import get_random_deployment
	from core.sparse_advertisements_v3 import Sparse_Advertisement_Solver

	# Worker_Manager needs to be picked up via the same module-aliasing
	# trick run_ray.py uses, so SAS imports the Ray-backed Worker_Manager.
	import sys
	import core.worker_comms as worker_comms
	sys.modules['worker_comms'] = worker_comms

	import random
	random.seed(31415)
	np.random.seed(31415)
	deployment = get_random_deployment('small', port=31600)
	deployment['generic_objective'] = 'avg_latency'

	# n_prefixes constraint: the default 'using_objective' init mode in
	# Sparse_Advertisement_Wrapper.init_advertisement writes to slots
	# [1 .. n_pops] of the prefix axis, so n_prefixes must be >= n_pops + 1.
	# We pick a generous value so the test isn't fragile to per-deployment
	# pop counts.
	n_pops_in_dep = len(set(p[0] for p in deployment['popps']))
	n_prefixes_safe = max(n_pops_in_dep + 2, 4)

	# Use the default `using_objective` initialization (= what production
	# uses). We deliberately do NOT override `init` here so the test exercises
	# the same path real runs do.
	sas = Sparse_Advertisement_Solver(
		deployment,
		lambduh=0, gamma=0, verbose=False,
		n_prefixes=n_prefixes_safe, with_capacity=False,
	)

	# Wire up workers exactly the way eval_all_solution_types.py does (line 129).
	# SAS.solve() depends on `self.worker_manager` being set; the import
	# alias above ensures `Worker_Manager` here is the Ray-backed version.
	from core.worker_comms import Worker_Manager  # resolved via the sys.modules alias
	wm = Worker_Manager(sas.get_init_kwa(), deployment)
	wm.start_workers()
	sas.set_worker_manager(wm)
	sas.update_deployment(deployment)

	# Limit iterations so the test stays bounded.
	sas.max_n_iter = 3

	# IMPORTANT: SCULPTOR optimizes a combined objective (latency benefit +
	# sparsity penalty + resilience). Comparing raw avg_latency at all-ones vs
	# at a sparse output would always look like "regression" because sparsity
	# is the whole point of the algorithm. We measure the algorithm's OWN
	# objective (sas.modeled_objective) at both endpoints -- that's the value
	# the gradient descent is actually trying to drive down. That's also the
	# value the user actually cares about ("ending optimization value lower
	# than the starting one").
	#
	# Reproducibility note: sas.solve() internally calls init_advertisement()
	# under whatever np.random state is current. We seed once for the initial
	# snapshot, then re-seed before solve() so solve()'s own init produces the
	# same starting advertisement we just measured.
	np.random.seed(31415)
	initial_adv = sas.init_advertisement()
	initial_modeled = float(sas.modeled_objective(initial_adv))
	print("\ninitial modeled objective: {:.4f}".format(initial_modeled))

	# Run the actual SAS solver. This drives Worker_Manager fan-out, which is
	# the code path that hit the ActorHandle.send bug.
	#
	# IMPORTANT: we must measure the final objective BEFORE stopping the
	# workers, because modeled_objective itself does worker fan-out (via
	# latency_benefit_fn). If we ran stop_workers first, the next
	# modeled_objective call would see worker_sockets={} and raise KeyError
	# inside flush_latency_benefit_queue_generic when it indexes rets[0].
	np.random.seed(31415)  # re-seed so solve()'s init matches `initial_adv`
	t0 = time.time()
	final_modeled = None
	try:
		sas.solve()
		final_modeled = float(sas.modeled_objective(sas.optimization_advertisement))
		print("final   modeled objective: {:.4f}".format(final_modeled))
	finally:
		try:
			sas.worker_manager.stop_workers()
		except Exception:
			pass
	print("SAS.solve elapsed: {:.1f}s".format(time.time() - t0))

	assert final_modeled is not None, "solve() failed before we could measure final"

	# Convergence claim: SCULPTOR's *own* objective at the final advertisement
	# should be at most its objective at the starting advertisement (modulo
	# a small slack for noise). If this fails, the gradient direction or the
	# worker dispatch is broken; raw avg_latency disagreeing is normal because
	# of the sparsity / resilience terms.
	assert final_modeled <= initial_modeled + 1e-3, (
		"SAS.solve made the modeled objective WORSE: initial={:.4f} -> "
		"final={:.4f}. The algorithm's own objective should monotonically "
		"decrease (modulo noise). Likely causes: wrong gradient sign, worker "
		"dispatch broken (Worker_Manager fan-out failures often manifest this way), "
		"or modeled_objective itself has changed shape."
		.format(initial_modeled, final_modeled)
	)
