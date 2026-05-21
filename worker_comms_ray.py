"""
Ray-based replacement for worker_comms.Worker_Manager.

Public API matches the original (`get_n_workers`, `start_workers`,
`stop_workers`, `update_worker_deployments`, `send_receive_workers`,
`send_receive_messages_workers`, `send_receive_worker`,
`send_messages_workers`) so existing drivers can opt in by importing
Worker_Manager from this module instead of `worker_comms`.

Internally, each "worker" is a long-lived Ray actor
(path_distribution_computer_ray.Path_Distribution_Computer.remote(...)) that
owns its persistent Gurobi model. The pickled (cmd, data) tuples that the
drivers already build are forwarded verbatim to `actor.handle_msg.remote(...)`,
which unpacks and dispatches them on the actor side.
"""
import os
import pickle
import multiprocessing

import ray

from helpers import *
from constants import *


# ---------------------------------------------------------------------- #
# Ray initialisation. We try to attach to a cluster first (auto-discover or
# RAY_ADDRESS), then fall back to local-mode so single-machine runs work
# without any extra setup. Idempotent: safe to import multiple times.
# ---------------------------------------------------------------------- #
def _ensure_ray():
	if ray.is_initialized():
		return
	address = os.environ.get('RAY_ADDRESS')
	# include_dashboard=False: the dashboard pulls in pydantic v1 which is
	# broken on Python 3.14 and we don't need a web UI for batch LP solving.
	# Set RAY_DASHBOARD=1 to re-enable.
	want_dashboard = os.environ.get('RAY_DASHBOARD') == '1'
	init_kwargs = dict(
		ignore_reinit_error=True,
		log_to_driver=True,
		include_dashboard=want_dashboard,
	)
	if address:
		ray.init(address=address, **init_kwargs)
		return
	try:
		# 'auto' attaches to a running cluster on this host if one exists.
		ray.init(address='auto', **init_kwargs)
		return
	except Exception:
		# No cluster -> local-mode. Uses all cores on this machine.
		ray.init(**init_kwargs)


_ensure_ray()

# Import after Ray is initialised so the @ray.remote decorator binds cleanly.
from path_distribution_computer_ray import Path_Distribution_Computer


class _ActorSocketShim:
	"""A tiny adapter so Ray actor handles look enough like a ZMQ REQ socket
	for `sparse_advertisements_v3.py` (and similar callers) that call
	`worker_socket.send(msg)` / `worker_socket.recv()` directly --
	bypassing the Worker_Manager API entirely.

	Public surface:
	  * .send(msg_bytes)       -- fires `actor.handle_msg.remote(msg_bytes)`
								  and records the future for the next recv().
	  * .recv()                -- blocks on the most recent send()'s future
								  and returns the result, pickled (mimics the
								  zmq REQ/REP wire format the original
								  workers used to send back).
	  * .handle_msg            -- pass-through to the actor, so internal
								  Worker_Manager methods can still do
								  `socket.handle_msg.remote(msg)` unchanged.
	  * .actor                 -- the raw Ray ActorHandle, for `ray.kill`.

	The original ZMQ REQ sockets enforced strict send/recv alternation;
	the call sites in sparse_advertisements_v3 follow that pattern, so a
	single-slot queue is sufficient. We don't bother emulating timeouts."""

	def __init__(self, actor):
		self.actor = actor
		self._pending = None  # at most one outstanding remote call

	# --- ZMQ-style wire API (for direct callers) -------------------------- #
	def send(self, msg_bytes):
		# Fire and remember.
		self._pending = self.actor.handle_msg.remote(msg_bytes)

	def recv(self):
		if self._pending is None:
			# Nothing in flight; return a benign empty pickled value rather
			# than raising, since callers sometimes do speculative recv().
			return pickle.dumps(None)
		result = ray.get(self._pending)
		self._pending = None
		return pickle.dumps(result)

	# --- Ray-native API (for Worker_Manager internal use) ----------------- #
	@property
	def handle_msg(self):
		return self.actor.handle_msg

	def __getattr__(self, name):
		# Forward any other attribute access to the underlying actor handle so
		# things like `socket.ping.remote()` keep working.
		return getattr(self.actor, name)


class Worker_Manager:
	"""Drop-in replacement for worker_comms.Worker_Manager that uses Ray actors
	instead of ZMQ subprocesses.

	Compatibility notes:
	  * `self.worker_sockets` is preserved as a dict { worker_i -> actor_handle }.
	    A few call sites in sparse_advertisements_v3.py / optimal_adv_wrapper.py
	    iterate it directly; only the keys are needed there, which still works.
	  * The `send_*` methods accept the same pickled (cmd, data) byte messages.
	"""

	def __init__(self, kwa_settings, deployment):
		self.kwa_settings = kwa_settings
		self.deployment = deployment
		self.dpsize = self.deployment['dpsize']
		self.worker_sockets = {}          # worker_i -> Ray actor handle
		self.worker_to_deployments = {}

	def get_init_kwa(self):
		return self.kwa_settings

	def get_n_workers(self):
		# Same heuristic as the original: min(cpu_count, suggested_for_dpsize).
		# Allow env var override to escape the multiprocessing.cpu_count() trap
		# when the driver runs on a small head node but workers are larger
		# (e.g. driver on m7g.4xlarge has cpu_count==16 but worker c7g.16xlarge
		# has 64; without override we under-utilize the worker by 4x).
		# RAM caveat: at ~2-3 GB per actor for actual-32, cap at 32 on a
		# 128 GB worker to leave headroom.
		env_override = os.environ.get('SCULPTOR_N_WORKERS')
		if env_override is not None:
			try:
				suggested_num_workers = get_n_workers(self.dpsize)
				n = min(int(env_override), suggested_num_workers)
				print("SCULPTOR_N_WORKERS override active: n_workers={}".format(n))
				return n
			except ValueError:
				print("WARNING: SCULPTOR_N_WORKERS={!r} is not an int; falling back".format(env_override))
		cpu_count = multiprocessing.cpu_count()
		suggested_num_workers = get_n_workers(self.dpsize)
		return min(cpu_count, suggested_num_workers)

	# ------------------------------------------------------------------ #
	# Lifecycle
	# ------------------------------------------------------------------ #
	def start_workers(self):
		self.worker_to_deployments = {}
		n_workers = self.get_n_workers()

		# Split the deployment into (shared static dict, per-UG slices) and
		# ray.put the static dict ONCE. The actor constructor then receives
		# the static dict via ObjectRef (auto-dereferenced by Ray's actor
		# protocol from one shared plasma entry) plus its tiny per-UG slice
		# — avoiding 34 MB × n_workers of redundant pickling at actual-32
		# scale (~95 sec → ~10 sec wall for 64 actors).
		static_dep, slices = split_deployment_by_ug_separated(
			self.deployment, n_chunks=n_workers)
		static_dep_ref = ray.put(static_dep)

		# Same trick for init_kwa (already shared across actors).
		init_kwa = self.get_init_kwa()
		init_kwa_ref = ray.put(init_kwa)
		for worker in range(n_workers):
			if len(slices[worker]['ugs']) == 0:
				continue
			assert len(slices[worker]['ugs']) >= 1
			# worker_to_deployments stores the *full* per-worker dict so any
			# driver-side code that reads it (e.g. update_worker_deployments)
			# sees the same shape as before the refactor.
			self.worker_to_deployments[worker] = {**static_dep, **slices[worker]}
			# Construct the actor. The Ray scheduler picks a node/CPU.
			# Each actor's __init__ (in path_distribution_computer_ray) merges
			# the static + slice and sets up its persistent Gurobi model.
			actor = Path_Distribution_Computer.remote(
				worker, slices[worker], init_kwa_ref, static_dep_ref)
			# Wrap in a ZMQ-shaped shim so external code that calls
			# socket.send(msg)/socket.recv() (e.g. sparse_advertisements_v3
			# stop_tracker/set_iter loops) works unchanged. Worker_Manager's
			# own methods still use `.handle_msg.remote(...)` via the shim's
			# pass-through `handle_msg` property.
			self.worker_sockets[worker] = _ActorSocketShim(actor)

		# Block until every actor has finished __init__ -- matches the
		# behaviour of the original which waited for an ACK before returning.
		# Calling any method forces __init__ to complete.
		ready_refs = [self.worker_sockets[w].ping.remote()
					  for w in self.worker_sockets]
		ray.get(ready_refs)

	def update_worker_deployments(self, new_deployment):
		self.deployment = new_deployment
		self.worker_to_deployments = {}
		n_workers = self.get_n_workers()
		print("Splitting deployment into subdeployments...")
		subdeployments = split_deployment_by_ug(self.deployment, n_chunks=n_workers)
		print("Done splitting deployment into subdeployments.")

		print("Sending deployment update messages...")
		refs = []
		for worker in range(n_workers):
			if len(subdeployments[worker]['ugs']) == 0:
				continue
			assert len(subdeployments[worker]['ugs']) >= 1
			self.worker_to_deployments[worker] = subdeployments[worker]
			msg = pickle.dumps(('update_deployment', (subdeployments[worker], {})))
			refs.append(self.worker_sockets[worker].handle_msg.remote(msg))
		print("Waiting for deployment ACK messages...")
		ray.get(refs)

	def stop_workers(self):
		for worker, sock in list(self.worker_sockets.items()):
			# `sock` is an _ActorSocketShim wrapping the actor handle. ray.kill
			# wants the raw actor.
			actor = getattr(sock, 'actor', sock)
			try:
				ray.kill(actor)
			except Exception:
				pass
		self.worker_sockets = {}

	# ------------------------------------------------------------------ #
	# Message-passing primitives
	# ------------------------------------------------------------------ #
	def _fanout(self, items):
		"""items: iterable of (worker_i, msg_bytes). Submit all in flight,
		then ray.get the results. Returns {worker_i: result}."""
		pending = []
		for worker_i, msg in items:
			if worker_i not in self.worker_sockets:
				continue
			pending.append((worker_i, self.worker_sockets[worker_i].handle_msg.remote(msg)))
		rets = {}
		for worker_i, ref in pending:
			rets[worker_i] = ray.get(ref)
		return rets

	def send_receive_workers(self, msg, L_TIMEOUT=100 * 60):
		# Same msg to every worker. Original returns dict {worker_i: result}.
		return self._fanout([(w, msg) for w in self.worker_sockets])

	def send_receive_messages_workers(self, msgs, L_TIMEOUT=100 * 60, **kwargs):
		# msgs is a list indexed by worker_i (matches original semantics).
		n_workers = kwargs.get('n_workers', self.get_n_workers())
		assert len(msgs) == n_workers
		return self._fanout([(i, msgs[i]) for i in range(n_workers)
							 if i in self.worker_sockets])

	def send_receive_worker(self, worker_i, msg):
		return ray.get(self.worker_sockets[worker_i].handle_msg.remote(msg))

	def send_messages_workers(self, msgs):
		"""Send (possibly distinct) messages to workers. The original waits for
		ACKs but does not return the per-worker payloads -- we mirror that.

		`msgs` may be either a dict {worker_i: msg} (as built in
		optimal_adv_wrapper.update_parent_tracker_workers) or a list indexed by
		worker_i. We accept both."""
		if isinstance(msgs, dict):
			items = list(msgs.items())
		else:
			items = list(enumerate(msgs))
		refs = []
		for worker_i, msg in items:
			if worker_i not in self.worker_sockets:
				continue
			refs.append(self.worker_sockets[worker_i].handle_msg.remote(msg))
		# Block on completion. (The original's send_messages_workers also
		# tries to recv ACKs, just without using them.)
		ray.get(refs)
