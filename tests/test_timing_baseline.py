"""
Timing tests / micro-benchmarks.

These are NOT correctness tests -- they're throughput probes. They print
human-readable stats so you can eyeball per-LP solve time, and they set very
generous upper bounds so they fail only if something catastrophically
regresses (e.g., per-LP time goes from 10ms to 10s).

Add `--capture=no` (or `-s`) when running pytest to actually see the print
output:

    pytest tests/test_timing_baseline.py -v -s
"""
import pickle
import pytest


@pytest.mark.gurobi
@pytest.mark.slow
def test_persistent_lp_solve_throughput(worker_session, subdeployment,
                                        tiny_advertisement, lp_timer):
	"""Measure how fast the worker's persistent-LP path solves the tiny
	deployment. Useful as a baseline before/after solver changes."""
	# Build one solve_lp `data` payload. Matches the shape Worker_Manager builds
	# in optimal_adv_wrapper.solve_lp_assignments_workers.
	subdeployment_with_obj = dict(subdeployment)
	subdeployment_with_obj['generic_objective'] = 'avg_latency'

	def one_solve():
		data = [(0, tiny_advertisement, subdeployment_with_obj, False)]
		out = worker_session._cmd_solve_lp(data)
		# Make sure we're not silently no-oping.
		assert out and out[0][1].get('solved'), "LP did not solve"

	stats = lp_timer(one_solve, n=20, warmup=3).summary()

	# Print so you can see it under `pytest -s`.
	print("\nLP solve timing (n={n}):".format(**stats))
	print("  min:    {min_ms:.2f} ms".format(**stats))
	print("  median: {median_ms:.2f} ms".format(**stats))
	print("  mean:   {mean_ms:.2f} ms".format(**stats))
	print("  p95:    {p95_ms:.2f} ms".format(**stats))
	print("  max:    {max_ms:.2f} ms".format(**stats))

	# Very loose upper bound. Tightens only if you've decided what 'normal' is.
	assert stats['median_ms'] < 5000, \
		"LP solve median is suspiciously slow: {:.2f}ms".format(stats['median_ms'])


@pytest.mark.gurobi
@pytest.mark.slow
def test_handle_msg_dispatch_overhead(worker_session, stopwatch):
	"""Measure pickled-tuple dispatch overhead on the worker side. If this is
	a significant fraction of LP solve time, we've over-engineered the API."""
	noop = pickle.dumps(('increment_iter', None))
	n = 1000
	with stopwatch() as t:
		for _ in range(n):
			worker_session.handle_msg(noop)
	per_call_us = (t['elapsed_s'] / n) * 1e6
	print("\nhandle_msg(no-op) dispatch: {:.1f} us/call over {} calls".format(per_call_us, n))
	# Should be well under a millisecond. If it's slower, something is wrong.
	assert per_call_us < 1000, \
		"Dispatch overhead too high: {:.1f} us/call".format(per_call_us)
