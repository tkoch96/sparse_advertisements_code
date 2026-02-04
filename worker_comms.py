import zmq, pickle, numpy as np, time, copy, multiprocessing, os
from subprocess import call
from helpers import *
from constants import *

## ~ half a second for debugging, .01s for real time
SLEEP_PERIOD = .02

paths = ["/home/tom/venv/bin/python", "/Users/tom/Documents/phd/research/ingress_opt/venv/bin/python",
	'/home/ubuntu/venv/bin/python', '/Users/tomkoch/Documents/venv/bin/python']
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

	def get_init_kwa(self):
		return self.kwa_settings

	def get_n_workers(self):
		cpu_count = multiprocessing.cpu_count()
		suggested_num_workers = get_n_workers(self.dpsize)
		return min(cpu_count, suggested_num_workers)

	def update_worker_deployments(self, new_deployment):
		self.deployment = new_deployment
		self.worker_to_deployments = {}
		n_workers = self.get_n_workers()
		# print("Splitting deployment into subdeployments.")
		subdeployments = split_deployment_by_ug(self.deployment, n_chunks=n_workers)
		# print("Done splitting deployment into subdeployments.")
		
		context = zmq.Context()
		for worker in range(n_workers):
			if len(subdeployments[worker]['ugs']) == 0: continue
			## It would be annoying to make the code work for cases in which a processor focuses on one user
			# print("Updating deployment in worker {}".format(worker))
			assert len(subdeployments[worker]['ugs']) >= 1
			self.worker_to_deployments[worker] = subdeployments[worker]
			# send worker startup information
			# args = [copy.deepcopy(subdeployments[worker])]
			# kwargs = self.get_init_kwa()
			# msg = pickle.dumps(('init',(args,kwargs)))
			msg = pickle.dumps(('update_deployment', subdeployments[worker]))
			self.worker_sockets[worker].send(msg)
			while True:
				try:
					msg = pickle.loads(self.worker_sockets[worker].recv())
					if msg == 'ACK':
						break
				except:
					time.sleep(.5)

	def start_workers(self):
		# self.worker_to_uis = {}
		self.worker_to_deployments = {}
		n_workers = self.get_n_workers()
		# print("Splitting deployment into subdeployments.")
		subdeployments = split_deployment_by_ug(self.deployment, n_chunks=n_workers)
		# print("Done splitting deployment into subdeployments.")
		
		context = zmq.Context()
		for worker in range(n_workers):
			if len(subdeployments[worker]['ugs']) == 0: continue
			## It would be annoying to make the code work for cases in which a processor focuses on one user
			# print("Launching working {}".format(worker))
			assert len(subdeployments[worker]['ugs']) >= 1
			base_port = int(self.deployment.get('port', 31415))
			call("{} path_distribution_computer.py {} {} &".format(PYTHON, worker, base_port), shell=True) # VMs
			# send worker startup information
			args = [copy.deepcopy(subdeployments[worker])]
			self.worker_to_deployments[worker] = subdeployments[worker]
			kwargs = self.get_init_kwa()
			self.worker_sockets[worker] = context.socket(zmq.REQ)
			self.worker_sockets[worker].setsockopt(zmq.RCVTIMEO, 100000)
			self.worker_sockets[worker].connect('tcp://localhost:{}'.format(base_port+worker))
			msg = pickle.dumps(('init',(args,kwargs)))
			self.worker_sockets[worker].send(msg)
		for worker in range(n_workers):
			while True:
				try:
					msg = pickle.loads(self.worker_sockets[worker].recv())
					if msg == 'ACK':
						break
				except:
					time.sleep(.5)

	def stop_workers(self):
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
		for worker, worker_socket in self.worker_sockets.items():
			msg = msgs[worker]
			worker_socket.send(msg)
			## this will fail if it hits a timeout
			worker_socket.recv()



