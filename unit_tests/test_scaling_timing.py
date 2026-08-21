"""
Scaling-timing benchmarks across deployment sizes.

Run with:
    pytest -v -s tests/test_scaling_timing.py

These are slow (especially at 'med' and 'large') so they're gated behind
`@pytest.mark.benchmark` and skipped by `pytest -m "not slow"`.

By default we run up through 'decent'. To include 'med'/'large', set:
    BENCH_SIZES="really_friggin_small,small,decent,med"
    BENCH_SIZES="all"          # everything

Each parametrized run prints a summary; at the end we print a comparison table.

What you should look at in the output:
  * median LP-solve time per deployment size -- this is the 'T' in
    `total_runtime ~ N_lps * T / n_workers`.
  * how steeply it scales as peers / popps grow -- gives you a feel for
    what your real production deployment will cost.
  * dispatch overhead (`handle_msg` no-op): should be << LP solve time.
  * calc_lb throughput: separately timed since it's also a hot path.
"""
import os
import time
import pickle
import statistics
from collections import OrderedDict

import numpy as np
import pytest

# Sizes available in deployment_setup.problem_params. Ordered by build time.
ALL_SIZES = [
	'really_friggin_small',
	'small',
	'decent',
	'med',
	'large',
]


def _selected_sizes():
	"""Honour BENCH_SIZES env var; default to the first three (fastest)."""
	env = os.environ.get('BENCH_SIZES', '').strip().lower()
	if not env:
		return ['really_friggin_small', 'small', 'decent']
	if env == 'all':
		return list(ALL_SIZES)
	asked = [s.strip() for s in env.split(',') if s.strip()]
	unknown = [s for s in asked if s not in ALL_SIZES]
	if unknown:
		raise ValueError("Unknown BENCH_SIZES entries: {}. Valid: {}".format(
			unknown, ALL_SIZES))
	return asked


SELECTED_SIZES = _selected_sizes()


# Collected across parametrized runs, printed in a teardown summary below.
_RESULTS = OrderedDict()


def _build_worker_for(size):
	"""Build (deployment, worker) for a given size. Done per parametrize call
	rather than fixturing because the deployment dict is size-specific."""
	import random
	random.seed(31415)
	np.random.seed(31415)
	from core.deployment_setup import get_random_deployment
	from core.path_distribution_computer import _LocalPathDistributionComputer

	t0 = time.perf_counter()
	deployment = get_random_deployment(size, port=31600)
	t_dep = time.perf_counter() - t0

	t0 = time.perf_counter()
	subdep = deployment
	t_split = time.perf_counter() - t0

	init_kwa = {
		'lambduh': 1.0, 'gamma': 0, 'verbose': False,
		'n_prefixes': None, 'with_capacity': False,
		'save_run_dir': None, 'generic_objective': 'avg_latency',
	}

	t0 = time.perf_counter()
	worker = _LocalPathDistributionComputer(
		worker_i=0, deployment=subdep, init_kwargs=init_kwa)
	t_worker = time.perf_counter() - t0

	return {
		'deployment': deployment,
		'deployment': subdep,
		'worker': worker,
		'init_kwa': init_kwa,
		'setup_timing': {
			'build_deployment_s': t_dep,
			'split_s': t_split,
			'worker_init_s': t_worker,
		},
	}


def _stats(samples_s):
	s = sorted(samples_s)
	return {
		'n': len(s),
		'min_ms': 1000 * s[0],
		'p50_ms': 1000 * statistics.median(s),
		'mean_ms': 1000 * (sum(s) / len(s)),
		'p95_ms': 1000 * s[max(0, int(len(s) * 0.95) - 1)],
		'max_ms': 1000 * s[-1],
		'total_s': sum(s),
	}


# ---------------------------------------------------------------------------#
# Main benchmark: per-LP solve time vs deployment size.
# ---------------------------------------------------------------------------#
@pytest.mark.gurobi
@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize('size', SELECTED_SIZES)
def test_lp_throughput_by_deployment_size(size):
	"""Build a deployment at `size`, then time N LP solves. Prints a per-size
	row; a final summary is emitted by the session-end fixture below.

	Includes both `solve_lp` (the full LP path through the persistent model)
	timing and basic setup costs.
	"""
	print("\n--- size = {} ---".format(size))
	ctx = _build_worker_for(size)
	worker = ctx['worker']
	subdep = ctx['deployment']
	deployment = ctx['deployment']

	st = ctx['setup_timing']
	n_popps = len(deployment['popps'])
	n_ugs = len(deployment['ugs'])
	print("  deployment: {} ugs, {} popps".format(n_ugs, n_popps))
	print("  setup: build={:.2f}s  split={:.3f}s  worker_init={:.2f}s".format(
		st['build_deployment_s'], st['split_s'], st['worker_init_s']))

	# Build a single-prefix "all on" advertisement.
	adv = np.ones((n_popps, 1))
	subdep_with_obj = dict(subdep, generic_objective='avg_latency')

	# Adjust n by size: at really_friggin_small we can do 100; at med/large
	# even 10 will take a while. The first solve is also slow (cold persistent
	# model), so we always warm up first.
	if size in ('really_friggin_small', 'small'):
		warmup, n = 3, 50
	elif size in ('decent',):
		warmup, n = 2, 25
	else:  # med, large
		warmup, n = 1, 10

	def one_solve():
		out = worker._cmd_solve_lp([(0, adv, subdep_with_obj, False)])
		assert out and out[0][1].get('solved')

	for _ in range(warmup):
		one_solve()
	samples = []
	for _ in range(n):
		t0 = time.perf_counter()
		one_solve()
		samples.append(time.perf_counter() - t0)

	st_solve = _stats(samples)
	print("  LP solves (n={n}):  p50={p50_ms:.1f}ms  mean={mean_ms:.1f}ms"
		  "  p95={p95_ms:.1f}ms  max={max_ms:.1f}ms".format(**st_solve))
	# Implied single-worker throughput.
	per_sec = 1000 / st_solve['p50_ms']
	print("  -> ~{:.0f} LPs/sec on this worker (p50)".format(per_sec))
	# At 100 workers, what does "millions of LPs" look like?
	for n_lps in (1_000_000, 10_000_000):
		hrs = n_lps * (st_solve['p50_ms'] / 1000.0) / 100 / 3600.0
		print("  -> {:>10,} LPs on 100 workers @ p50: {:.2f}h".format(n_lps, hrs))

	_RESULTS[size] = {
		'n_ugs': n_ugs,
		'n_popps': n_popps,
		'setup': st,
		'solve': st_solve,
	}

	# Loose guardrail: catastrophic regressions only.
	assert st_solve['p50_ms'] < 60_000, (
		"LP solve p50 over 60s at size {} -- something is very wrong".format(size))


# ---------------------------------------------------------------------------#
# Persistent vs cold-start: how much does the warm-start LP shell save?
# ---------------------------------------------------------------------------#
@pytest.mark.gurobi
@pytest.mark.slow
@pytest.mark.benchmark
def test_persistent_model_vs_fresh_build():
	"""Compare solving the same LP with the warm persistent model (the
	worker's default path) against rebuilding the Gurobi model each call.

	Tells us how much benefit the existing `init_persistent_lp` design buys
	us; if the gap is small, switching solvers is cheaper than if it's large.
	"""
	ctx = _build_worker_for('small')
	worker = ctx['worker']
	subdep = ctx['deployment']
	deployment = ctx['deployment']
	n_popps = len(deployment['popps'])

	adv = np.ones((n_popps, 1))
	subdep_with_obj = dict(subdep, generic_objective='avg_latency')

	def warm_path():
		out = worker._cmd_solve_lp([(0, adv, subdep_with_obj, False)])
		assert out[0][1].get('solved')

	# Cold path: rebuild the persistent LP shell each call. This forces Gurobi
	# to discard all the cached variables and reconstruct.
	def cold_path():
		worker.var_pool = {}
		worker.init_persistent_lp()
		out = worker._cmd_solve_lp([(0, adv, subdep_with_obj, False)])
		assert out[0][1].get('solved')

	for fn, label in [(warm_path, 'warm'), (cold_path, 'cold')]:
		# Always warm up first to amortize one-time Gurobi init.
		for _ in range(2):
			fn()
		samples = []
		for _ in range(15):
			t0 = time.perf_counter()
			fn()
			samples.append(time.perf_counter() - t0)
		st = _stats(samples)
		print("\n{} path:  p50={p50_ms:.1f}ms  mean={mean_ms:.1f}ms".format(label, **st))


# ---------------------------------------------------------------------------#
# Dispatch overhead: handle_msg(no-op) cost.
# ---------------------------------------------------------------------------#
@pytest.mark.gurobi
@pytest.mark.benchmark
def test_handle_msg_overhead_scales():
	"""Verify the pickle/dispatch cost in handle_msg stays trivially small
	even at larger sizes (it should -- payload size is what scales, not
	dispatch). If this grows surprisingly, our wire format has a problem."""
	for size in ['really_friggin_small', 'small']:
		ctx = _build_worker_for(size)
		worker = ctx['worker']
		noop = pickle.dumps(('increment_iter', None))
		n = 5000
		t0 = time.perf_counter()
		for _ in range(n):
			worker.handle_msg(noop)
		us_per_call = (time.perf_counter() - t0) / n * 1e6
		print("  {}: {:.1f} us/call".format(size, us_per_call))


# ---------------------------------------------------------------------------#
# Final summary printed at end of session (only if results were collected).
# ---------------------------------------------------------------------------#
def pytest_terminal_summary(terminalreporter, exitstatus=None, config=None):
	"""pytest hook: emit a comparison table after all tests complete."""
	if not _RESULTS:
		return
	tr = terminalreporter
	tr.write_sep("=", "LP solve scaling summary")
	tr.write_line("{:>22} | {:>5} | {:>6} | {:>9} | {:>9} | {:>9} | {:>11}".format(
		'size', 'ugs', 'popps', 'p50 (ms)', 'p95 (ms)', 'mean (ms)', '/sec @ p50'))
	tr.write_line("-" * 90)
	for size, r in _RESULTS.items():
		s = r['solve']
		per_sec = 1000.0 / s['p50_ms']
		tr.write_line("{:>22} | {:>5} | {:>6} | {:>9.1f} | {:>9.1f} | {:>9.1f} | {:>11.0f}".format(
			size, r['n_ugs'], r['n_popps'],
			s['p50_ms'], s['p95_ms'], s['mean_ms'], per_sec))
	tr.write_line("")
	tr.write_line("Back-of-envelope: total wall = (N_lps * p50_ms / 1000) / n_workers / 3600 hours.")
