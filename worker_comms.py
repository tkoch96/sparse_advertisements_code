import zmq, pickle, numpy as np, time, copy, multiprocessing
from subprocess import call
from helpers import *

class Worker_Manager:
	def __init__(self, kwa_settings, deployment):

		self.kwa_settings = kwa_settings
		self.deployment = deployment
		self.worker_sockets = {}

	def get_init_kwa(self):
		return self.kwa_settings

	def get_n_workers(self):
		# return np.minimum(32, multiprocessing.cpu_count() // 2)
		return multiprocessing.cpu_count() // 2

	def start_workers(self):
		# self.worker_to_uis = {}
		self.worker_to_deployments = {}
		n_workers = self.get_n_workers()
		subdeployments = split_deployment_by_ug(self.deployment, n_chunks=n_workers)
		
		context = zmq.Context()
		for worker in range(n_workers):
			if len(subdeployments[worker]['ugs']) == 0: continue
			## It would be annoying to make the code work for cases in which a processor focuses on one user
			assert len(subdeployments[worker]['ugs']) >= 1

			call("../congestion_analysis/venv/bin/python path_distribution_computer.py {} &".format(worker), shell=True)
			# send worker startup information
			args = [subdeployments[worker]]
			self.worker_to_deployments[worker] = subdeployments[worker]
			kwargs = self.get_init_kwa()
			self.worker_sockets[worker] = context.socket(zmq.REQ)
			self.worker_sockets[worker].setsockopt(zmq.RCVTIMEO, 1000)
			while True:
				self.worker_sockets[worker].connect('tcp://localhost:{}'.format(BASE_SOCKET+worker))
				msg = pickle.dumps(('init',(args,kwargs)))
				self.worker_sockets[worker].send(msg)
				time.sleep(1)
				msg = pickle.loads(self.worker_sockets[worker].recv())
				if msg == 'ACK':
					break

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

	def send_receive_workers(self, msg):
		n_workers = self.get_n_workers()
		for worker, worker_socket in self.worker_sockets.items():
			worker_socket.send(msg)
		rets = {}
		while True:
			# wait for responses from workers
			for worker in range(n_workers):
				try:
					rets[worker]
				except KeyError:
					try: # check for message from worker
						rets[worker] = pickle.loads(self.worker_sockets[worker].recv())
					except: # Timeout, must be stll calculating
						if np.random.random() > .99:
							print("received {} rets so far".format(len(rets)))
						time.sleep(.2)
						pass
			if len(rets) == n_workers:
				break
		return rets
