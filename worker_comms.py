import zmq, pickle, numpy as np, time, copy, multiprocessing, os, threading
from subprocess import call
from helpers import *
from constants import *

## ~ half a second for debugging, .01s for real time
SLEEP_PERIOD = .02

paths = ["/home/tom/venv/bin/python", "/Users/tom/Documents/phd/research/ingress_opt/venv/bin/python",
	'/home/ubuntu/venv/bin/python', '/Users/tomkoch/Documents/venv/bin/python', '/home/ubuntu/env/bin/python']
PYTHON = None
for path in paths:
	if os.path.exists(path):
		PYTHON = path
		break
assert PYTHON is not None

class Worker_Manager:
	def __init__(self, kwa_settings, deployment):

		self.kwa_settings = kwa_settings
		self.deployment = deployment
		self.dpsize = self.deployment['dpsize']
		self.worker_sockets = {}
		# Adaptive-resize state; see worker_comms_ray.Worker_Manager for the
		# semantics (request_add_workers fires from a watcher thread,
		# process_pending_resize runs at sparse iter boundary).
		self.worker_to_deployments = {}
		self._resize_lock = threading.Lock()
		self._pending_add_workers = 0

	def get_init_kwa(self):
		return self.kwa_settings

	def get_n_workers(self):
		# Once workers are running, return the actual count so partitioning
		# code sees the post-ramp size after a request_add_workers expansion.
		if self.worker_sockets:
			return len(self.worker_sockets)
		return self._target_n_workers()

	def _target_n_workers(self):
		# Env var override mirrors the Ray Worker_Manager path so the same
		# SCULPTOR_N_WORKERS=N knob works in local (non-Ray) runs too. When
		# unset, behavior is unchanged: min(cpu_count, dpsize-suggested).
		env_override = os.environ.get('SCULPTOR_N_WORKERS')
		if env_override is not None:
			try:
				n = int(env_override)
				return n
			except ValueError:
				print("WARNING: SCULPTOR_N_WORKERS={!r} is not an int; falling back".format(env_override))
		cpu_count = multiprocessing.cpu_count()
		suggested_num_workers = get_n_workers(self.dpsize)
		return min(cpu_count, suggested_num_workers)

	def update_worker_deployments(self, new_deployment):
		self.deployment = new_deployment
		self.worker_to_deployments = {}
		n_workers = self.get_n_workers()
		print("Splitting deployment into subdeployments...")
		subdeployments = split_deployment_by_ug(self.deployment, n_chunks=n_workers)
		print("Done splitting deployment into subdeployments.")
		
		context = zmq.Context()
		print("Sending deployment update messages...")
		for worker in range(n_workers):
			if len(subdeployments[worker]['ugs']) == 0: continue
			## It would be annoying to make the code work for cases in which a processor focuses on one user
			assert len(subdeployments[worker]['ugs']) >= 1
			self.worker_to_deployments[worker] = subdeployments[worker]
			# send worker startup information
			msg = pickle.dumps(('update_deployment', subdeployments[worker]))
			self.worker_sockets[worker].send(msg)
		print("Waiting for deployment ACK messages...")
		for worker in range(n_workers):
			while True:
				try:
					msg = pickle.loads(self.worker_sockets[worker].recv())
					if msg == 'ACK':
						break
				except:
					time.sleep(.5)

	def start_workers(self, n_workers_override=None):
		"""Spawn the initial pool of ZMQ-worker subprocesses.

		`n_workers_override` lets evaluate_all_metrics start with a smaller
		pool than SCULPTOR_N_WORKERS during the
		concurrent-parallel-strategies window. The remaining slots are
		added later via request_add_workers -> process_pending_resize.
		"""
		# self.worker_to_uis = {}
		self.worker_to_deployments = {}
		if n_workers_override is not None:
			n_workers = n_workers_override
			print("[adaptive-workers] start_workers using override n_workers={}"
				  " (target={})".format(n_workers, self._target_n_workers()))
		else:
			n_workers = self._target_n_workers()
			env_override = os.environ.get('SCULPTOR_N_WORKERS')
			if env_override is not None:
				print("SCULPTOR_N_WORKERS override active: n_workers={}".format(n_workers))
		# print("Splitting deployment into subdeployments.")
		subdeployments = split_deployment_by_ug(self.deployment, n_chunks=n_workers)
		# print("Done splitting deployment into subdeployments.")
		# SCULPTOR_WORKER_INIT_STAGGER_SEC: lightweight offset between worker
		# spawns. Workers still init in PARALLEL, but with staggered start
		# times so their memory peaks (~few seconds into init, when var_pool
		# + RB backups build) don't perfectly overlap. Default 0 preserves
		# original simultaneous-spawn behaviour. Small values (1-3s) give most
		# of the memory-smoothing benefit at small wall-time cost
		# (n_workers * stagger_sec added during init only).
		stagger_sec = float(os.environ.get('SCULPTOR_WORKER_INIT_STAGGER_SEC', '0') or 0)
		if stagger_sec > 0:
			print("Worker init spawn-staggered: {}s offset between spawns".format(stagger_sec))

		context = zmq.Context()
		for worker in range(n_workers):
			if len(subdeployments[worker]['ugs']) == 0: continue
			## It would be annoying to make the code work for cases in which a processor focuses on one user
			# print("Launching working {}".format(worker))
			assert len(subdeployments[worker]['ugs']) >= 1
			base_port = int(self.deployment.get('port', 31415))
			call("{} path_distribution_computer.py {} {} &".format(PYTHON, worker, base_port), shell=True) # VMs
			# send worker startup information
			args = [subdeployments[worker]]
			self.worker_to_deployments[worker] = subdeployments[worker]
			kwargs = self.get_init_kwa()
			self.worker_sockets[worker] = context.socket(zmq.REQ)
			self.worker_sockets[worker].setsockopt(zmq.RCVTIMEO, 100000)
			self.worker_sockets[worker].connect('tcp://localhost:{}'.format(base_port+worker))
			msg = pickle.dumps(('init',(args,kwargs)))
			self.worker_sockets[worker].send(msg)
			# Offset the next spawn without waiting for ACK. Workers init in
			# parallel; ACKs are collected in the second pass below.
			if stagger_sec > 0 and worker + 1 < n_workers:
				time.sleep(stagger_sec)
		# Collect ACKs in parallel (single pass regardless of stagger).
		for worker in range(n_workers):
			while True:
				try:
					msg = pickle.loads(self.worker_sockets[worker].recv())
					if msg == 'ACK':
						break
				except:
					time.sleep(.5)

	# ------------------------------------------------------------------ #
	# Adaptive resize (request from any thread; apply on main thread only)
	# ------------------------------------------------------------------ #
	def request_add_workers(self, n_add):
		"""Thread-safe request to grow the worker pool by `n_add`. See
		worker_comms_ray.Worker_Manager.request_add_workers for the design
		(watcher thread fires this when concurrent parallel-strategy
		subprocesses finish; resize executes on next iter boundary)."""
		with self._resize_lock:
			self._pending_add_workers += int(n_add)
			print("[adaptive-workers] request_add_workers(+{}) (pending={}, current={})".format(
				n_add, self._pending_add_workers, len(self.worker_sockets)), flush=True)

	def process_pending_resize(self):
		"""Main-thread consumer; safe to call from sparse training loop at
		iter boundary (no concurrent fanouts)."""
		with self._resize_lock:
			n_add = self._pending_add_workers
			self._pending_add_workers = 0
		if n_add <= 0:
			return
		try:
			self._do_add_workers(n_add)
		except Exception as e:
			import traceback
			traceback.print_exc()
			print("[adaptive-workers] _do_add_workers({}) FAILED: {}".format(n_add, e), flush=True)

	def _do_add_workers(self, n_add):
		"""Spawn `n_add` ZMQ subprocess workers and re-shard UGs across the
		new total. Existing workers are sent their NEW (smaller) slice via
		the update_deployment cmd; new workers are spawned with their
		slice as part of the init handshake.
		"""
		n_existing = len(self.worker_sockets)
		n_total = n_existing + n_add
		print("[adaptive-workers] growing pool: {} -> {} (+{})".format(
			n_existing, n_total, n_add), flush=True)
		t0 = time.time()

		# Re-shard for the new total.
		subdeployments = split_deployment_by_ug(self.deployment, n_chunks=n_total)

		# 1) Spawn the new subprocesses + connect their sockets.
		context = zmq.Context()
		stagger_sec = float(os.environ.get('SCULPTOR_WORKER_INIT_STAGGER_SEC', '0') or 0)
		base_port = int(self.deployment.get('port', 31415))
		new_workers = []
		for worker in range(n_existing, n_total):
			if len(subdeployments[worker]['ugs']) == 0:
				continue
			call("{} path_distribution_computer.py {} {} &".format(PYTHON, worker, base_port), shell=True)
			self.worker_sockets[worker] = context.socket(zmq.REQ)
			self.worker_sockets[worker].setsockopt(zmq.RCVTIMEO, 100000)
			self.worker_sockets[worker].connect('tcp://localhost:{}'.format(base_port + worker))
			args = [subdeployments[worker]]
			kwargs = self.get_init_kwa()
			msg = pickle.dumps(('init', (args, kwargs)))
			self.worker_sockets[worker].send(msg)
			self.worker_to_deployments[worker] = subdeployments[worker]
			new_workers.append(worker)
			if stagger_sec > 0 and worker + 1 < n_total:
				time.sleep(stagger_sec)

		# 2) Collect init ACKs from the new workers.
		for worker in new_workers:
			while True:
				try:
					msg = pickle.loads(self.worker_sockets[worker].recv())
					if msg == 'ACK':
						break
				except Exception:
					time.sleep(.5)
		t_init = time.time() - t0

		# 3) Re-shard existing workers — send each its NEW (smaller) slice.
		for worker in range(n_existing):
			if worker not in self.worker_sockets:
				continue
			if len(subdeployments[worker]['ugs']) == 0:
				continue
			self.worker_to_deployments[worker] = subdeployments[worker]
			msg = pickle.dumps(('update_deployment', subdeployments[worker]))
			self.worker_sockets[worker].send(msg)
		for worker in range(n_existing):
			if worker not in self.worker_sockets:
				continue
			if len(subdeployments[worker]['ugs']) == 0:
				continue
			while True:
				try:
					m = pickle.loads(self.worker_sockets[worker].recv())
					if m == 'ACK':
						break
				except Exception:
					time.sleep(.5)
		t_reshard = time.time() - t0 - t_init

		print("[adaptive-workers] pool now {} workers; init={:.1f}s reshard={:.1f}s".format(
			len(self.worker_sockets), t_init, t_reshard), flush=True)

	def _collect_and_emit_worker_mem_logs(self):
		"""Pull each worker's per-process mem log file and echo to stdout
		so the lines land in the sweep log alongside driver [mem] events.
		Best-effort: any worker that errors is skipped silently. Must run
		BEFORE workers are sent 'end' / killed, since it relies on the
		ZMQ socket loop to dispatch the dump_mem_log cmd.
		"""
		try:
			msg = pickle.dumps(('dump_mem_log', None))
			results = self.send_receive_workers(msg)
		except Exception as e:
			print('[worker_mem_collect] failed: {}'.format(e), flush=True)
			return
		for worker_i in sorted(results):
			content = results.get(worker_i) or ''
			if content:
				print('--- worker {} mem log start ---'.format(worker_i), flush=True)
				print(content, end='' if content.endswith('\n') else '\n', flush=True)
				print('--- worker {} mem log end ---'.format(worker_i), flush=True)

	def stop_workers(self):
		# Pull per-worker mem logs BEFORE we send 'end' (which causes the
		# worker to close its socket and exit, dropping its log file).
		try:
			self._collect_and_emit_worker_mem_logs()
		except Exception as e:
			print('[worker_mem_collect] outer error: {}'.format(e), flush=True)
		for worker, socket in self.worker_sockets.items():
			try:
				socket.recv()
			except:
				pass
			msg = pickle.dumps(('end','end'))
			try:
				socket.send(msg)
				socket.close()
			except:
				pass
		del self.worker_sockets
		self.worker_sockets = {}

	def send_receive_workers(self, msg, L_TIMEOUT = 100*60):
		n_workers = self.get_n_workers()
		for worker, worker_socket in self.worker_sockets.items():
			worker_socket.send(msg)
		rets = {}
		timeouts = {workeri:time.time() + L_TIMEOUT for workeri in range(n_workers)}
		while True:
			# wait for responses from workers
			for worker in range(n_workers):
				try:
					rets[worker]
				except KeyError:
					try: # check for message from worker
						this_ret = pickle.loads(self.worker_sockets[worker].recv())
						if this_ret != "ERROR":
							rets[worker] = this_ret
						else:
							print("Received error message from worker {}, sending again".format(worker))
							self.worker_sockets[worker].send(msg)
					except zmq.error.Again: # Timeout, must be stll calculating
						if time.time() > timeouts[worker]:
							## resend the message
							print("Potential error in worker {}, no message after {}s. Resending.".format(worker, L_TIMEOUT))
							self.worker_sockets[worker].send(msg)
							timeouts[worker] = time.time() + L_TIMEOUT
				
			if len(rets) == n_workers:
				break
			time.sleep(SLEEP_PERIOD)
		return rets

	def send_receive_messages_workers(self, msgs, L_TIMEOUT = 100*60, **kwargs):
		# send unique message to each worker
		n_workers = kwargs.get('n_workers', self.get_n_workers())
		assert len(msgs) == n_workers
		
		for i,msg in enumerate(msgs):
			self.worker_sockets[i].send(msg)

		rets = {}
		timeouts = {workeri:time.time() + L_TIMEOUT for workeri in range(n_workers)}
		while True:
			# wait for responses from workers
			for msg,worker in zip(msgs, range(n_workers)):
				try:
					rets[worker]
				except KeyError:
					try: # check for message from worker
						this_ret = pickle.loads(self.worker_sockets[worker].recv())
						if this_ret != "ERROR":
							rets[worker] = this_ret
						else:
							print("Received error message from worker {}, sending again".format(worker))
							self.worker_sockets[worker].send(msg)
					except zmq.error.Again: # Timeout, must be stll calculating
						if time.time() > timeouts[worker]:
							## resend the message
							print("Potential error in worker {}, no message after {}s. Resending.".format(worker, L_TIMEOUT))
							self.worker_sockets[worker].send(msg)
							timeouts[worker] = time.time() + L_TIMEOUT
				
			if len(rets) == n_workers:
				break
			time.sleep(SLEEP_PERIOD)
		return rets

	def send_receive_worker(self, worker_i, msg):
		self.worker_sockets[worker_i].send(msg)
		while True:
			try:
				ret = pickle.loads(self.worker_sockets[worker_i].recv())
				if ret != "ERROR":
					break
				else:
					print("received error message from worker {}, sending again".format(worker_i))
					self.worker_sockets[worker_i].send(msg)
			except zmq.error.Again: # Timeout, must be stll calculating
				time.sleep(SLEEP_PERIOD)
				pass
		return ret

	def send_messages_workers(self, msgs):
		# Phase 1: Fire all messages immediately
		# ZMQ handles the buffering, so this loop completes almost instantly.
		for worker, worker_socket in self.worker_sockets.items():
			msg = msgs[worker]
			worker_socket.send(msg)

		# Phase 2: Collect acknowledgments
		# By the time we get here, all workers are processing in parallel.
		for worker, worker_socket in self.worker_sockets.items():
			try:
				# We assume your sockets have RCVTIMEO set as in start_workers
				worker_socket.recv()
			except zmq.error.Again:
				print(f"Worker {worker} timed out receiving ACK.")



