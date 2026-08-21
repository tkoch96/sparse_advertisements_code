"""
Verifiable warm-start tests for the worker's LP-solving paths.

Two separate paths in this codebase solve LPs, and they are NOT interchangeable:

  1. solve_generic_lp_persistent
		Persistent Gurobi model in path_distribution_computer.py.
		Called from generic_objective_pdf for the MC inner loop.
		Warm-starts across calls (basis preserved).

  2. solve_generic_lp_with_failure_catch -> solve_generic_lp
		scipy.optimize.linprog in solve_lp_assignment.py.
		Called by _cmd_solve_lp for objective='avg_latency' (the default).
		Fresh problem each call; no warm-start.

Previous iterations of this file silently fell back to -1 when measurements
failed, hiding the fact that we were sometimes measuring scipy and reporting
Gurobi numbers. This rewrite:

  * uses strict attribute access (raises on failure -- a test error, not -1),
  * has one test per LP path,
  * asserts the path identity (Gurobi-ran vs scipy-ran) in each test, so the
	test would fail loudly if the dispatch ever changed.

Run with:
	pytest -v -s tests/test_warm_start_realism.py

Sizes via BENCH_WARM_SIZES env var (default: small, decent):
	BENCH_WARM_SIZES="small,decent,med" pytest -v -s tests/test_warm_start_realism.py
"""
import os
import time
import statistics
from collections import OrderedDict

import numpy as np
import pytest


ALL_SIZES = ['really_friggin_small', 'small', 'decent', 'med', 'large']


def _selected_sizes():
	env = os.environ.get('BENCH_WARM_SIZES', '').strip().lower()
	if not env:
		return ['small', 'decent']
	if env == 'all':
		return list(ALL_SIZES)
	asked = [s.strip() for s in env.split(',') if s.strip()]
	unknown = [s for s in asked if s not in ALL_SIZES]
	if unknown:
		raise ValueError("Unknown BENCH_WARM_SIZES entries: {}".format(unknown))
	return asked


SELECTED_SIZES = _selected_sizes()

# Per-mode sample counts. Tuned per size so each test caps at ~tens of seconds.
N_BY_SIZE = {
	'really_friggin_small': 200,
	'small': 100,
	'decent': 40,
	'med': 15,
	'large': 5,
}
# Number of (prefix, ug) pairs perturbed between consecutive rti's in mode 'close'.
MC_REROUTES_PER_STEP = 5

_RESULTS = OrderedDict()


# ---------------------------------------------------------------------------#
# Fixture construction (duplicated rather than imported so this file is
# self-contained and so changes to conftest can't silently shift its results).
# ---------------------------------------------------------------------------#
def _build_worker(size):
	import random
	random.seed(31415)
	np.random.seed(31415)
	from core.deployment_setup import get_random_deployment
	from core.path_distribution_computer import _LocalPathDistributionComputer

	deployment = get_random_deployment(size, port=31600)
	subdep = deployment
	init_kwa = {
		'lambduh': 1.0, 'gamma': 0, 'verbose': False,
		'n_prefixes': None, 'with_capacity': False,
		'save_run_dir': None, 'generic_objective': 'avg_latency',
	}
	worker = _LocalPathDistributionComputer(
		worker_i=0, deployment=subdep, init_kwargs=init_kwa)
	return worker, subdep, deployment


# ---------------------------------------------------------------------------#
# Strict Gurobi metric helpers.
# ---------------------------------------------------------------------------#
def _gurobi_solve_metrics(model):
	"""Read (iter_count, runtime_s) from the last optimize() call on a Gurobi
	model. Raises RuntimeError if neither the single- nor multi-obj attribute
	name is readable. NO silent fallbacks: a measurement failure surfaces as
	a test error, not a misleading number."""
	iter_count = None
	last_err = None
	for attr in ('IterCount', 'IterCount0', 'IterCount1'):
		try:
			iter_count = float(model.getAttr(attr))
			break
		except Exception as e:
			last_err = e
	if iter_count is None:
		raise RuntimeError(
			"Could not read IterCount from Gurobi model "
			"(tried IterCount, IterCount0, IterCount1). Last error: {}. "
			"This usually means no optimize() ran since the last reset.".format(last_err))
	runtime = None
	for attr in ('Runtime', 'Runtime0', 'Runtime1'):
		try:
			runtime = float(model.getAttr(attr))
			break
		except Exception as e:
			last_err = e
	if runtime is None:
		raise RuntimeError(
			"Could not read Runtime from Gurobi model. Last error: {}".format(last_err))
	return iter_count, runtime


def _stats(samples_s):
	s = sorted(samples_s)
	return {
		'n': len(s),
		'min_ms': 1000 * s[0],
		'p50_ms': 1000 * statistics.median(s),
		'mean_ms': 1000 * (sum(s) / len(s)),
		'p95_ms': 1000 * s[max(0, int(len(s) * 0.95) - 1)],
		'max_ms': 1000 * s[-1],
	}


# ---------------------------------------------------------------------------#
# RTI generators -- vary the input to solve_generic_lp_persistent without
# touching the advertisement matrix (which is what production MC does).
# ---------------------------------------------------------------------------#
def _baseline_rti(worker, adv):
	"""The deterministic routing table for `adv`. Same call the production
	worker would make."""
	rti, _ = worker.calculate_ground_truth_ingress(adv, do_cache=False)
	return rti


def _perturbed_rti(rti_base, worker, rng, k=MC_REROUTES_PER_STEP):
	"""Copy `rti_base` and randomly reroute k (prefix, ug) entries to a
	different popp from the user's available set."""
	new_rti = {pi: dict(ug_to_popp) for pi, ug_to_popp in rti_base.items()}
	all_pairs = [(pi, ug) for pi, ug_map in new_rti.items() for ug in ug_map]
	if not all_pairs:
		return new_rti
	k = min(k, len(all_pairs))
	idx = rng.choice(len(all_pairs), size=k, replace=False)
	for j in idx:
		pi, ug = all_pairs[j]
		current = new_rti[pi][ug]
		available = list(worker.whole_deployment_ug_perfs.get(ug, {}).keys())
		available = [p for p in available if p != current]
		if not available:
			continue
		new_rti[pi][ug] = available[rng.integers(len(available))]
	return new_rti


def _generate_rtis(mode, worker, fixed_adv, n, rng):
	"""Build n rti dicts for the given mode.
	  identical: same baseline rti N times.
	  close:     each rti is the previous one with MC_REROUTES_PER_STEP entries
				 re-routed (gradient-MC mimic).
	  far:       each rti is built from a *fresh random advertisement*. Maximum
				 dissimilarity, worst case for warm-start.
	"""
	if mode == 'identical':
		rti = _baseline_rti(worker, fixed_adv)
		return [rti for _ in range(n)]

	if mode == 'close':
		rtis = [_baseline_rti(worker, fixed_adv)]
		for _ in range(n - 1):
			rtis.append(_perturbed_rti(rtis[-1], worker, rng))
		return rtis

	if mode == 'far':
		n_popps = len(worker.popps)
		rtis = []
		for _ in range(n):
			a = (rng.random((n_popps, 1)) > 0.5).astype(float)
			if a.sum() == 0:
				a[rng.integers(n_popps), 0] = 1.0
			rtis.append(_baseline_rti(worker, a))
		return rtis

	raise ValueError("unknown mode: {}".format(mode))


# ---------------------------------------------------------------------------#
# Test 1: the Gurobi persistent path warm-starts.
#
# We call solve_generic_lp_persistent directly, then assert:
#   (a) Gurobi ACTUALLY ran (Runtime > 0 is readable, no fallback).
#   (b) On 'identical' rti repetition, iter count converges to ~0 (warm-start
#       claim).
#   (c) On 'far' rti, iter count is materially larger.
# If any of these fails, the test FAILS -- no silent numbers.
# ---------------------------------------------------------------------------#
@pytest.mark.gurobi
@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize('size', SELECTED_SIZES)
def test_gurobi_persistent_path_warm_starts(size):
	print("\n--- gurobi-persistent path, size = {} ---".format(size))
	worker, subdep, deployment = _build_worker(size)
	n_popps = len(deployment['popps'])
	n = N_BY_SIZE[size]
	rng = np.random.default_rng(2026)

	# Fixed adv for the 'identical' and 'close' modes (production pattern:
	# adv stays the same, routing varies).
	fixed_adv = (rng.random((n_popps, 1)) > 0.5).astype(float)
	if fixed_adv.sum() == 0:
		fixed_adv[rng.integers(n_popps), 0] = 1.0

	# Warm the persistent model once so __init__ overhead doesn't leak into
	# the first mode.
	warm_rti = _baseline_rti(worker, fixed_adv)
	for _ in range(3):
		ret = worker.solve_generic_lp_persistent(warm_rti, 'avg_latency')
		assert ret.get('solved'), "warmup LP failed to solve"

	# Sanity: Gurobi metrics must be readable after a real solve. If not, the
	# whole test premise is wrong -- fail loudly.
	iters_warm, rt_warm = _gurobi_solve_metrics(worker.model)
	assert rt_warm > 0, (
		"After a warmup solve, Gurobi Runtime is 0 -- this path may not be "
		"using the persistent model. Cannot continue.")

	results = {}
	for mode in ('identical', 'close', 'far'):
		rtis = _generate_rtis(mode, worker, fixed_adv, n, rng)
		# Re-warm with the baseline to neutralize the prior mode's basis state.
		for _ in range(2):
			worker.solve_generic_lp_persistent(warm_rti, 'avg_latency')

		wall_s, iter_counts, gurobi_s = [], [], []
		for rti in rtis:
			t0 = time.perf_counter()
			ret = worker.solve_generic_lp_persistent(rti, 'avg_latency')
			wall_s.append(time.perf_counter() - t0)
			ic, rt = _gurobi_solve_metrics(worker.model)
			iter_counts.append(ic)
			gurobi_s.append(rt)
			assert ret.get('solved'), "{} mode: LP did not solve".format(mode)

		st = _stats(wall_s)
		st['p50_iters'] = statistics.median(iter_counts)
		st['mean_iters'] = sum(iter_counts) / len(iter_counts)
		st['p50_gurobi_ms'] = 1000 * statistics.median(gurobi_s)
		st['p50_python_ms'] = st['p50_ms'] - st['p50_gurobi_ms']
		results[mode] = st
		print("  {:<10} n={n}  wall p50={p50_ms:7.2f}ms  "
			  "(gurobi p50={p50_gurobi_ms:6.2f}ms  python={p50_python_ms:6.2f}ms)  "
			  "iters p50={p50_iters:6.0f}".format(mode, **st))

	_RESULTS[size] = {
		'n_popps': n_popps, 'n_ugs': len(deployment['ugs']),
		'n_solves': n, 'modes': results,
	}

	# The actual warm-start claim, falsifiable. p50 iter count for repeatedly
	# solving the identical LP should be ~0 (basis already optimal). p50 for
	# 'far' should be materially larger because each LP has a different
	# active column set.
	assert results['identical']['p50_iters'] < 5, (
		"Warm-start claim falsified at size {}: identical rti repetition "
		"required p50={} simplex iterations (expected ~0)."
		.format(size, results['identical']['p50_iters'])
	)
	assert results['far']['p50_iters'] > results['identical']['p50_iters'], (
		"Warm-start claim falsified at size {}: 'far' did not require more "
		"iterations than 'identical' (far p50={}, identical p50={})."
		.format(size, results['far']['p50_iters'], results['identical']['p50_iters'])
	)


# ---------------------------------------------------------------------------#
# Test 2: _cmd_solve_lp(avg_latency) DOES route through the persistent Gurobi
# model. solve_lp_assignment.solve_generic_lp_with_failure_catch checks whether
# `sas` is a worker with an initialized persistent model and dispatches there
# for avg_latency / per_site_cost. This test pins that dispatch so a future
# regression (e.g., dropping avg_latency from _PERSISTENT_GUROBI_OBJECTIVES)
# would be detected.
# ---------------------------------------------------------------------------#
@pytest.mark.gurobi
@pytest.mark.benchmark
@pytest.mark.parametrize('size', ['small', 'decent'])
def test_avg_latency_cmd_solve_lp_routes_to_persistent_gurobi(size):
	print("\n--- _cmd_solve_lp(avg_latency) path, size = {} ---".format(size))
	worker, subdep, deployment = _build_worker(size)
	n_popps = len(deployment['popps'])
	subdep_with_obj = dict(subdep, generic_objective='avg_latency')

	# Warm the persistent model so we have a known runtime to compare against.
	adv = np.ones((n_popps, 1))
	rti = _baseline_rti(worker, adv)
	for _ in range(2):
		worker.solve_generic_lp_persistent(rti, 'avg_latency')
	before_iters, before_runtime = _gurobi_solve_metrics(worker.model)
	print("  before: IterCount0={}, Runtime0={:.4f}s".format(
		before_iters, before_runtime))

	# Call _cmd_solve_lp the way Worker_Manager would. If the dispatch goes
	# through persistent Gurobi (correct behaviour), the model's Runtime0
	# attribute updates to reflect the new solve. If it falls back to scipy
	# (broken), the attribute is frozen from the warmup above.
	out = worker._cmd_solve_lp([(0, adv, subdep_with_obj, False)])
	assert out and out[0][1].get('solved'), "_cmd_solve_lp did not solve"

	after_iters, after_runtime = _gurobi_solve_metrics(worker.model)
	print("  after:  IterCount0={}, Runtime0={:.4f}s".format(
		after_iters, after_runtime))

	# Pin: a fresh Gurobi solve must have actually run. Runtime is per-call
	# (not cumulative), so a fresh optimize() overwrites it. If it's exactly
	# unchanged, no Gurobi solve happened and we silently fell back to scipy.
	model_changed = (
		after_iters != before_iters
		or abs(after_runtime - before_runtime) > 1e-9
	)
	assert model_changed, (
		"_cmd_solve_lp for 'avg_latency' did not touch the persistent Gurobi "
		"model (IterCount unchanged at {}, Runtime unchanged at {:.6f}s). "
		"That means the dispatch fell back to scipy. Check "
		"solve_lp_assignment.solve_generic_lp_with_failure_catch -- the "
		"persistent-Gurobi routing for avg_latency is broken."
		.format(after_iters, after_runtime)
	)
	print("  -> confirmed: _cmd_solve_lp for 'avg_latency' routes through "
		  "the persistent Gurobi model.")


# ---------------------------------------------------------------------------#
# Cross-size summary -- only the Gurobi-path test contributes.
# ---------------------------------------------------------------------------#
def pytest_terminal_summary(terminalreporter, exitstatus=None, config=None):
	if not _RESULTS:
		return
	tr = terminalreporter

	tr.write_sep("=", "Gurobi-persistent LP path: per-call timings")
	tr.write_line("{:>10} | {:>6} | {:>9} | {:>9} | {:>9}".format(
		'size', 'popps', 'identical', 'close', 'far'))
	tr.write_line("-" * 56)
	for size, r in _RESULTS.items():
		m = r['modes']
		tr.write_line("{:>10} | {:>6} | {:>7.2f}ms | {:>7.2f}ms | {:>7.2f}ms".format(
			size, r['n_popps'],
			m['identical']['p50_ms'], m['close']['p50_ms'], m['far']['p50_ms']))

	tr.write_sep("-", "Gurobi-only solve time p50 (ms)")
	for size, r in _RESULTS.items():
		m = r['modes']
		tr.write_line("  {:>10}: identical={:.2f}  close={:.2f}  far={:.2f}".format(
			size,
			m['identical']['p50_gurobi_ms'], m['close']['p50_gurobi_ms'],
			m['far']['p50_gurobi_ms']))

	tr.write_sep("-", "Simplex iterations p50 (warm-start signal)")
	for size, r in _RESULTS.items():
		m = r['modes']
		tr.write_line("  {:>10}: identical={:.0f}  close={:.0f}  far={:.0f}".format(
			size,
			m['identical']['p50_iters'], m['close']['p50_iters'],
			m['far']['p50_iters']))

	tr.write_line("")
	tr.write_line("Reading: iters[identical] ~ 0 + iters[far] > iters[identical]")
	tr.write_line("proves warm-start works on the persistent Gurobi path.")
	tr.write_line("These numbers reflect the production MC inner loop pattern.")
	tr.write_line("They do NOT apply to _cmd_solve_lp(avg_latency), which uses scipy.")
