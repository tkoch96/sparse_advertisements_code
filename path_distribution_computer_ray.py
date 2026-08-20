"""
Ray-actor version of Path_Distribution_Computer.

Design:
  * The non-Ray version (path_distribution_computer.py) spawns one subprocess
    per worker and communicates via ZMQ REQ/REP sockets. Each subprocess loops
    in `run()` -> `check_for_commands()` decoding pickled (cmd, data) tuples.
  * This module replaces that loop with a Ray actor. Each former `cmd` becomes
    an instance method. A single `handle_msg(msg_bytes)` shim unpacks the
    pickled (cmd, data) tuple and dispatches, which lets worker_comms_ray
    keep the original `send_receive_*` API verbatim -- so the driver code in
    sparse_advertisements_v3.py / optimal_adv_wrapper.py / eval_latency_failure.py
    does not need to change.

We deliberately subclass the original Path_Distribution_Computer instead of
duplicating its ~900 lines of LP / latency-benefit logic. Only __init__ (which
used to do a ZMQ handshake) and the dispatch surface change.

Worker state -- in particular the persistent Gurobi model created by
`init_persistent_lp()` and the per-worker deployment cache -- lives for the
lifetime of the actor, so the LP shell is reused across thousands of solves.
"""
import os
import time
import pickle
import ray

# Inherit ALL the heavy logic (latency_benefit, solve_*_lp, monte-carlo, etc.)
# from the original module by importing it. We override only what changes.
from path_distribution_computer import (
	Path_Distribution_Computer as _BasePathDistComputer,
	_log_mem_worker,
)
from optimal_adv_wrapper import Optimal_Adv_Wrapper
from solve_lp_assignment import solve_generic_lp_with_failure_catch
from constants import LOG_DIR


class _LocalPathDistributionComputer(_BasePathDistComputer):
	"""Same body as the Ray actor below, but without the @ray.remote decorator.

	Exposed so unit tests can instantiate a worker directly (no Ray, no ZMQ)
	and call methods like a normal Python object. The decorated Ray version
	(`Path_Distribution_Computer`) is defined at the bottom of this file by
	wrapping this class.
	"""

	def __init__(self, worker_i, deployment, init_kwargs):
		# Replicate the non-ZMQ portion of the original __init__ (lines 42-62
		# of path_distribution_computer.py). Skip start_connection / run.
		#
		# `deployment` is the FULL deployment (one shared plasma entry,
		# ray.put once by worker_comms_ray.start_workers and
		# auto-dereferenced by Ray at actor-init time). Every worker
		# computes gradient jobs over the entire deployment.
		self.worker_i = worker_i
		self.port = 0  # unused under Ray
		self.logging_iter = 0
		self.timing = {k: 0 for k in [
			'solve_unified_lp_not_optimize', 'optimize', 'get_paths_by_ug',
			'organizing_results', 'get_ingress_probabilities_by_dict_generic',
			'sim_rti', 'total_rti_calc', 'pmat_organize',
			'solve_generic_lp_persistent',
			'solve_generic_lp_not_persistent']}
		self.rti_data = {}
		self.MC_NUM = 5  ## monte carlo simulations to determine distributions

		# Construct the optimization wrapper directly with the deployment
		# and init kwargs supplied by Worker_Manager (no ZMQ handshake).
		Optimal_Adv_Wrapper.__init__(self, deployment, **init_kwargs)

		# Open the per-worker log file the same way the original did.
		log_path = os.path.join(
			LOG_DIR, 'worker_{}_log-{}.txt'.format(self.worker_i, self.dpsize))
		with open(log_path, 'w'):
			pass

		self.init_all_vars()
		# Note: no run() loop and no main_socket. Ray dispatches method calls.
		# Equivalent of the ZMQ version's `worker_proc_start` mem-snapshot,
		# fired at the end of actor __init__ (post init_all_vars so the
		# Gurobi shell + Optimal_Adv_Wrapper state are loaded).
		_log_mem_worker(self.worker_i, 'worker_proc_start',
		                dpsize=getattr(self, 'dpsize', '?'))

	# ------------------------------------------------------------------ #
	# Pickled-tuple dispatcher. Worker_Manager_ray calls this so the
	# existing send_receive_* / send_messages_workers API stays unchanged.
	# ------------------------------------------------------------------ #
	def handle_msg(self, msg_bytes):
		try:
			cmd, data = pickle.loads(msg_bytes)
		except Exception as e:
			print("Worker {} failed to unpickle msg: {}".format(self.worker_i, e))
			return "ERROR"
		method = getattr(self, '_cmd_' + cmd, None)
		if method is None:
			print("Invalid CMD in worker {} : {}".format(self.worker_i, cmd))
			return "ERROR"
		try:
			return method(data)
		except Exception as e:
			# One compact line by default; full stack only when
			# SCULPTOR_VERBOSE_ERRORS=1 (Tom 2026-08-19 log cleanup).
			print("[{}] Worker {} ERROR cmd '{}': {}: {}".format(
				time.strftime('%H:%M:%SZ', time.gmtime()),
				self.worker_i, cmd, type(e).__name__, e))
			if os.environ.get('SCULPTOR_VERBOSE_ERRORS') == '1':
				import traceback
				traceback.print_exc()
			return "ERROR"

	# ------------------------------------------------------------------ #
	# Command handlers. One per branch of the original if/elif dispatch
	# in check_for_commands(). Each takes the `data` payload and returns
	# the value that the original sent back over ZMQ.
	# ------------------------------------------------------------------ #

	def _cmd_calc_lb(self, data):
		ret = []
		self.this_time_ip_cache = {}
		for (args, kwargs) in data:
			ret.append(self.latency_benefit(*args, **kwargs))
		# Original did `del self.this_time_ip_cache`; reset to {} is equivalent
		# and avoids AttributeError if anything still references it.
		self.this_time_ip_cache = {}
		return ret

	def _cmd_solve_lp(self, data):
		ret = []
		ts = time.time()
		self.check_load_rw_measure_wrapper()
		n_iters = 0
		for fields in sorted(data, key=lambda el: el[0]):
			if len(fields) == 4:
				adv_i, adv, deployment, update_dep = fields
			else:
				adv_i, adv, opt_adv, deployment, update_dep = fields
			if update_dep:
				deployment_save = self.output_deployment()
				self.clear_caches()
				self.update_deployment(deployment, quick_update=True,
					verb=False, exit_on_impossible=False)
			self.check_load_rw_measure_wrapper()

			rti, _ = self.calculate_ground_truth_ingress(adv, do_cache=False)
			this_ret = solve_generic_lp_with_failure_catch(
				self, rti, deployment.get('generic_objective'), adv=adv)
			if update_dep:
				self.update_deployment(deployment_save, quick_update=True,
					verb=False, exit_on_impossible=False)
				self.check_load_rw_measure_wrapper()
			ret.append((adv_i, this_ret))
			n_iters += 1
		return ret

	def _cmd_calc_compressed_lb(self, data):
		# Reset self.timing accumulators at the top of each batch so the
		# end-of-batch summarize_timing() shows per-batch cumulative LP-
		# solve breakdown (not lifetime-cumulative or last-call-only). The
		# inner solve_unified_lp etc. now += into self.timing.
		for k in self.timing:
			self.timing[k] = 0.0
		ts = time.time()
		tlp = time.time()
		ret = []
		base_args, base_kwa = data[0]
		base_adv, = base_args
		base_adv = base_adv.astype(bool)
		ret.append({'ans': self.latency_benefit(base_adv, **base_kwa),
			'job_id': base_kwa.get('job_id', -1)})
		i = 0
		last_timing_summary = 0
		for diff, kwa in data[1:]:
			kwa['verbose_workers'] = (base_kwa.get('verbose_workers', False)
				or kwa.get('verbose_workers', False))
			for ind in zip(*diff):
				base_adv[ind] = not base_adv[ind]
			ret.append({'ans': self.latency_benefit(base_adv, **kwa),
				'job_id': kwa.get('job_id', -1)})
			for ind in zip(*diff):
				base_adv[ind] = not base_adv[ind]
			i += 1
			if time.time() - tlp > 100:
				self.print("[{}] {} pct. done calcing latency benefits, {}ms per iter".format(
					time.strftime("%H:%M:%SZ", time.gmtime()),
					round(i * 100.0 / len(data), 1),
					round(1000 * (time.time() - ts) / i)))
				tlp = time.time()
			# (intermediate i%50 summarize_timing trigger removed — the
			# end-of-batch summary below now always fires, giving one clean
			# per-batch breakdown instead of accumulator-growing intermediate
			# snapshots.)
			self.check_clear_cache()
		# Always emit one cumulative summary at end of each batch so that
		# fine-grained parallelism (e.g. 132 perms ÷ 64 actors = 2 per actor)
		# doesn't silently skip the i%50 trigger above. Worker 0 alone (the
		# Ray-dedup'd "[repeated Nx]" message handles the rest).
		if self.worker_i == 0 and len(data) > 1:
			self.summarize_timing()
		return ret

	def _cmd_reset_new_meas_cache(self, _data=None):
		self.clear_new_meas_caches()
		return "ACK"

	def _cmd_update_parent_tracker(self, parents_on):
		for ug in parents_on:
			for beaten_ingress, routed_ingress in parents_on[ug]:
				self.parent_tracker[ug, beaten_ingress, routed_ingress] = True
		if len(parents_on) > 0:
			self.clear_new_meas_caches()
		return "ACK"

	def _cmd_update_parent_tracker_csr(self, payload):
		# SCULPTOR_COMPACT_PT path: payload is either an ObjectRef to (or
		# directly) the (parents, offsets, rows, nonempty) CSR from
		# _encode_parents_on_csr. Zero-copy per node via plasma; replaces
		# the per-worker string-tuple dict (354MB/worker at actual-25).
		parents, offsets, rows, nonempty = payload
		self._pt_csr = (parents, offsets, rows)
		if nonempty:
			self.clear_new_meas_caches()
		return "ACK"

	def _cmd_update_deployment(self, data):
		# The base-class update_deployment short-circuits its worker_manager
		# fan-out branch when self.worker_manager is unset (AttributeError
		# guard), which is the case inside a worker actor.
		deployment, kwargs = data
		_log_mem_worker(self.worker_i, 'update_deployment_enter',
		                dpsize=deployment.get('dpsize', '?'))
		self.update_deployment(deployment, **kwargs)
		_log_mem_worker(self.worker_i, 'update_deployment_done',
		                dpsize=deployment.get('dpsize', '?'))
		self.dump_mem_components('post_update_deployment')
		return "ACK"

	def _cmd_dump_mem_log(self, _data=None):
		# Exposes the inherited dump_mem_log() to the cmd dispatcher so
		# Worker_Manager_ray._collect_and_emit_worker_mem_logs can pull
		# per-worker mem files at end of run via the standard
		# send_receive_workers API.
		return self.dump_mem_log()

	def _cmd_get_node_mem_avail_mb(self, _data=None):
		# Exposes the inherited get_node_mem_avail_mb() so the driver
		# can probe worker-node memory via the standard cmd dispatch.
		# Used by Worker_Manager._maybe_autoscale to monitor worker-node
		# memory pressure (head and worker live on different boxes).
		return self.get_node_mem_avail_mb()

	def _cmd_update_kwa(self, new_kwa):
		if new_kwa.get('n_prefixes') is not None:
			self.n_prefixes = new_kwa.get('n_prefixes')
		if new_kwa.get('gamma') is not None:
			self.gamma = new_kwa.get('gamma')
		if new_kwa.get('with_capacity') is not None:
			self.with_capacity = new_kwa.get('with_capacity')
		return "ACK"

	def _cmd_increment_iter(self, _data=None):
		self.increment_iter()
		return "ACK"

	def _cmd_set_iter(self, data):
		self.iter = data
		return "ACK"

	def _cmd_set_training_mode(self, data):
		# Inherited from Path_Distribution_Computer: toggles self._in_training
		# and pushes new RHS into the persistent Gurobi cap constraints so
		# SCULPTOR_CAPACITY_HEADROOM only applies during the gradient loop.
		self.set_training_mode(data)
		return "ACK"

	def _cmd_set_mc_num(self, data):
		# Explore-time MC override (Tom, 2026-08-14): the driver bumps
		# MC_NUM while evaluating explore candidates (entropy needs a real
		# distribution; training may run MC_NUM=1) and restores it after.
		self.MC_NUM = int(data)
		return "ACK"

	def _cmd_reset_cache(self, _data=None):
		self.clear_caches()
		return "ACK"

	def _cmd_init(self, _data=None):
		# Original called start_connection() to (re)bind a ZMQ socket. No-op for Ray.
		return "ACK"

	def _cmd_end(self, _data=None):
		# Cleanup hook. The Worker_Manager will follow up with ray.kill(actor).
		return "ACK"

	# Convenience: lets the master sanity-check that an actor is alive.
	def ping(self):
		return "ACK"


# Public Ray actor: the same class above wrapped with @ray.remote.
# Worker_Manager_ray imports this name unchanged.
#
# max_restarts=-1: if the actor's node is lost (e.g. AWS reclaims the spot
# worker), Ray may restart the actor on another live node instead of leaving
# it permanently dead. This is defence-in-depth — the Worker_Manager also does
# an app-level rebuild+retry on RayActorError (worker_comms_ray._with_recovery),
# which is the primary recovery path and re-applies deployment state. We do NOT
# set max_task_retries: a transparently-retried task would run against a
# restarted actor that lost its post-__init__ state (update_deployment is not
# replayed by Ray), so we drive the retry at the app level instead.
Path_Distribution_Computer = ray.remote(num_cpus=1, max_restarts=-1)(_LocalPathDistributionComputer)
