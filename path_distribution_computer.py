import numpy as np, numba as nb, pickle, copy, zmq, time
np.setbufsize(262144*8)

from constants import *
from helpers import *
from optimal_adv_wrapper import Optimal_Adv_Wrapper
from test_polyphase import rescale_pdf


remeasure_a = None
try:
	remeasure_a = pickle.load(open('remeasure_a.pkl','rb'))
except:
	pass


#### When these matrices are really big, helps to use numba, but can't work in multiprocessing scenarios
@nb.njit(fastmath=True,parallel=True)
def large_logical_and(arr1,arr2):
	return np.logical_and(arr1,arr2)

class Path_Distribution_Computer(Optimal_Adv_Wrapper):
	def __init__(self, worker_i):
		self.worker_i = worker_i
		args, kwargs = self.start_connection()
		super().__init__(*args, **kwargs)
		self.calculate_user_latency_by_peer()
		self.with_capacity = kwargs.get('with_capacity', False)

		## Latency benefit for each user is -1 * MAX_LATENCY -> -1 MIN_LATENCY
		## divided by their contribution to the total volume (i.e., multiplied by a weight)
		## important that every worker has the same lbx
		min_vol,max_vol = np.min(self.whole_deployment_ug_vols), np.max(self.whole_deployment_ug_vols)
		total_deployment_volume = np.sum(self.whole_deployment_ug_vols)
		self.lbx = np.linspace(-1*MAX_LATENCY * max_vol / total_deployment_volume, 0,num=LBX_DENSITY)
		self.big_lbx = np.zeros((LBX_DENSITY, len(self.whole_deployment_ugs)))
		for i in range(len(self.whole_deployment_ugs)):
			self.big_lbx[:,i] = copy.copy(self.lbx)


		self.stop = False
		self.calc_cache = Calc_Cache()
		self.this_time_ip_cache = {}
		print('started in worker {}'.format(self.worker_i))
		self.run()

	def start_connection(self):
		context = zmq.Context()
		self.main_socket = context.socket(zmq.REP)
		self.main_socket.setsockopt(zmq.RCVTIMEO, 1000)
		self.main_socket.bind('tcp://*:{}'.format(BASE_SOCKET+self.worker_i))
		while True:
			try:
				init_msg = self.main_socket.recv()
				break
			except zmq.error.Again:
				time.sleep(.01)
		self.main_socket.send(pickle.dumps('ACK'))
		msg_decoded = pickle.loads(init_msg)
		_, data = msg_decoded
		return data

	def clear_caches(self):
		self.calc_cache.clear_all_caches()

	def get_ingress_probabilities_by_a_matmul(self, a, verb=False, **kwargs):
		a = threshold_a(a.astype(np.int32))
		# if np.array_equal(a, remeasure_a): verb = True
		a_log = a.astype(bool)



		self.ingress_probabilities[:,:,:] = 0
		for pref_i in range(self.n_prefixes):
			tloga = tuple(a_log[:,pref_i].flatten())
			##### WARNING -- if number of UGs and number of popps is the same, there could be ambiguity with the broadcasting
			##### but the likelihood of that event is pretty small
			if np.sum(a[:,pref_i]) == 0:
				continue
			try:
				self.ingress_probabilities[:,pref_i,:] = self.calc_cache.all_caches['ing_prob'][tloga]
				continue
			except KeyError:
				try:
					for (i,k), prob in self.this_time_ip_cache[tloga].items():
						# will need a more complicated caching mechanism if ever non-uniform
						self.ingress_probabilities[i,pref_i,k] = 1.0/prob 
					continue
				except KeyError:
					pass
			these_active_popps = np.expand_dims(a_log[:,pref_i],axis=1)
			# a[:,pref_i] is active popps
			### This step takes the longest by far, but I can't figure out how to speed it up
			### perhaps when popp gets large, it makes more sense to store a sparse representation of parent_tracker
			active_parent_indicator = np.logical_and(a_log[:,pref_i], self.parent_tracker)
			# holds ug,popps to delete since they get beaten
			delete_popp_ug_indicator = np.logical_and(these_active_popps, np.any(active_parent_indicator, axis=2).T)
			# UG has route and popp active
			active_popp_ug_indicator = np.logical_and(self.popp_by_ug_indicator_no_rank, these_active_popps)
			# remove popp,ug's that would get beaten #1,0->1 1,1->0,0,1->0,0,0->0
			valid_popp_ug_indicator = np.logical_and(active_popp_ug_indicator,np.logical_not(delete_popp_ug_indicator))
			# now sort based on likelihood
			sortf_arr = {ug:[] for ug in self.ugs}
			for poppi,ugi in zip(*np.where(valid_popp_ug_indicator)):
				ug = self.ugs[ugi]
				popp = self.popps[poppi]
				try:
					d = self.calc_cache.all_caches['distance'][poppi,ug]
				except KeyError:
					ingress = self.popps[poppi]
					d = int(geopy.distance.geodesic(self.pop_to_loc[popp[0]], 
						self.metro_loc[ug[0]]).km)
					self.calc_cache.all_caches['distance'][poppi,ug] = d
				sortf_arr[ug].append((poppi,d))
			### Cache the entries that have non-zero probability
			self.this_time_ip_cache[tloga] = {}
			for ug,ds in sortf_arr.items():
				if len(ds) == 0:
					continue
				ui = self.ug_to_ind[ug]
				most_likely_peers = sorted(ds,key=lambda el : el[1])
				### TODO -- possibly introduce actual likelihoods here
				nmlp = len(most_likely_peers)
				for mlp,_ in most_likely_peers:
					self.ingress_probabilities[mlp,pref_i,ui] = 1 / nmlp
					### Cache the entries that have non-zero probability
					self.this_time_ip_cache[tloga][mlp,ui] = nmlp
			### This simple caching mechanism requires too much memory
			# self.calc_cache.all_caches['ing_prob'][tuple(a_log[:,pref_i].flatten())] = copy.copy(self.ingress_probabilities[:,pref_i,:])

	def latency_benefit(self, a, **kwargs):
		"""Calculates distribution of latency benefit at a given advertisement. Benefit is the sum of 
			benefits across all users."""
		a_effective = threshold_a(a)
		verb = kwargs.get('verb')
		if not kwargs.get('plotit') and not verb:
			try:
				return self.calc_cache.all_caches['lb'][tuple(a_effective.flatten())]
			except KeyError:
				pass

		timers = {
			'capacity': 0,
			'probs': 0,
			'benefit': 0,
			'convolution': 0,
			'start': time.time()
		}

		## Dims are path, prefix, user
		self.get_ingress_probabilities_by_a_matmul(a_effective, **kwargs)
		p_mat = self.ingress_probabilities
		p_mat = p_mat / (np.sum(p_mat,axis=0) + 1e-8)
		benefits = self.measured_latency_benefits
		lbx = self.lbx


		timers['probs'] = time.time() - timers['start']

		subset_ugs = False
		which_ugs = kwargs.get('ugs', None)
		if which_ugs is not None:
			subset_ugs = True
			which_ugs_this_worker = get_intersection(which_ugs, self.ugs)
			if len(which_ugs_this_worker) == 0:
				pdf = np.zeros(lbx.shape)
				pdf[-1] = 1
				return 0, (lbx.flatten(),pdf.flatten())
			which_ugs_i = np.array([self.ug_to_ind[ug] for ug in which_ugs_this_worker])
			all_workers_ugs_i = np.array([self.whole_deployment_ug_to_ind[ug] for ug in which_ugs])
			all_workers_vol = sum(self.whole_deployment_ug_vols[all_workers_ugs_i])

			benefit_renorm = all_workers_vol / np.sum(self.whole_deployment_ug_vols)

			lbx = lbx / benefit_renorm
			benefits = benefits / benefit_renorm
			self.big_lbx = self.big_lbx / benefit_renorm


		p_link_fails = np.zeros(self.n_popp)
		if self.with_capacity:
			## holds P(ingress) for each user
			ingress_px = np.zeros((len(lbx), self.n_ug))
			vol_x = self.vol_x

			all_pref_inds = np.arange(self.n_prefixes)
			for ui in range(self.n_ug):
				all_pv_i = np.where(p_mat[:,:,ui])
				# Holds prefix j, ingress bi, benefit and probability
				all_pv = [(j,benefits[bi,j,ui],p_mat[bi,j,ui],bi) for bi,j in zip(*all_pv_i)]
				if len(all_pv) == 0:
					# this user has no paths
					continue
				if len(all_pv) == 1:
					_, _, p, bi = all_pv[0]
					ingress_px[bi, ui] += p
				else:
					all_pv = sorted(all_pv,key=lambda el : el[1])
					running_probs = np.zeros((self.n_prefixes))
					running_probs[all_pv[0][0]] = all_pv[0][2]
					
					prefs_exist = list(set([el[0] for el in all_pv]))
					for pref_j in get_difference(all_pref_inds, prefs_exist):
						running_probs[pref_j] = 1 

					for i in range(1,len(all_pv)):
						pref_j, _, p, bi = all_pv[i]

						# calculate prob(max latency benefit)
						# we calculate this iteratively, from the smallest to the largest value
						# probability calc is basically probability of this value (p) times probability 
						# other prefixes are one of the smaller values (running prob)
						max_prob = p * np.prod(running_probs[all_pref_inds!=pref_j])
						running_probs[pref_j] += p
						if max_prob == 0 : continue

						ingress_px[bi, ui] += max_prob

			for ingress_i in range(self.n_popp): ### TODO -- expand so that every popp doesn't get their own link
				ug_prob_vols_this_ingress_i = np.where(ingress_px[ingress_i,:])[0]
				ug_prob_vols_this_ingress = np.zeros((len(self.vol_x), self.n_ug))
				for ui in ug_prob_vols_this_ingress_i:
					# Model user as bernoulli on this ingress
					vol_i = np.where(self.ug_to_vol[self.ugs[ui]] - self.vol_x <=0)[0][0]
					ug_prob_vols_this_ingress[vol_i, ui] = ingress_px[ingress_i, ui]
					ug_prob_vols_this_ingress[0, ui] = 1 - ingress_px[ingress_i, ui]
				ug_prob_vols_this_ingress = ug_prob_vols_this_ingress / (np.sum(ug_prob_vols_this_ingress,axis=0) + 1e-8)
				if np.sum(ug_prob_vols_this_ingress.flatten()) == 0:
					continue
				for ui in np.where(np.sum(ug_prob_vols_this_ingress,axis=0)==0):
					ug_prob_vols_this_ingress[0,ui] = 1

				p_vol_this_ingress = self.pdf_sum_function(ug_prob_vols_this_ingress).flatten()
				p_link_fails[ingress_i] = np.sum(p_vol_this_ingress[self.vol_x * self.n_ug > self.link_capacities[ingress_i]])
			
			# if kwargs.get('verb'):
			# 	import matplotlib.pyplot as plt
			# 	f,ax = plt.subplots(2)
			# 	ax[0].set_title("Ingress {}".format(ingress_i))
			# 	ax[0].plot(ug_prob_vols_this_ingress)
			# 	ax[1].plot(p_vol_this_ingress)
			# 	plt.show()
		timers['capacity'] = time.time() - timers['start']

		## holds P(latency benefit) for each user
		px = np.zeros((len(lbx), self.n_ug))

		all_pref_inds = np.arange(self.n_prefixes)
		for ui in range(self.n_ug):
			all_pv_i = np.where(p_mat[:,:,ui])
			## combine benefit with bernoulli link failure
			all_pv = [(j, benefits[bi,j,ui],p_mat[bi,j,ui], p_link_fails[bi]) for bi,j in zip(*all_pv_i)]
			if len(all_pv) == 0:
				# this user has no paths
				continue
			elif len(all_pv) == 1:
				_, lb, p, plf = all_pv[0]
				lbx_i = np.where(lb - lbx <= 0)[0][0]
				px[lbx_i, ui] += p * (1 -  plf)
				px[0, ui] += p * plf
			else:
				## Check to see if it's a trivial calculation
				found_trivial = False
				min_by_pref, max_by_pref = NO_ROUTE_LATENCY*np.ones(self.n_prefixes), -NO_ROUTE_LATENCY*np.ones(self.n_prefixes)
				prefs = []
				for pref_j, lb, p, plf in all_pv:
					min_by_pref[pref_j] = np.minimum(min_by_pref[pref_j],lb)
					max_by_pref[pref_j] = np.maximum(max_by_pref[pref_j],lb)
					prefs.append(pref_j)
				for pref_j in set(prefs):
					if min_by_pref[pref_j] >= np.max(np.delete(max_by_pref, pref_j)):
						### singleton benefit at this benefit
						lb = max_by_pref[pref_j]
						lbx_i = np.where(lb - lbx <= 0)[0][0]
						px[lbx_i, ui] = 1.0
						found_trivial = True
						# print("{} {} {}".format(min_by_pref,max_by_pref,pref_j))
						# exit(0)
						break
				if found_trivial: 
					continue
				# else:
				# 	if np.random.random() > .999:
				# 		print(min_by_pref)
				# 		print(max_by_pref)
				# 		print("\n")
				# 		if np.random.random() > .999:exit(0)


				all_pv = sorted(all_pv,key=lambda el : el[1])
				running_probs = np.zeros((self.n_prefixes))
				running_probs[all_pv[0][0]] = all_pv[0][2]

				prefs_exist = list(set([el[0] for el in all_pv]))
				for pref_j in get_difference(all_pref_inds, prefs_exist):
					running_probs[pref_j] = 1 

				for i in range(1,len(all_pv)):
					pref_j, lb, p, plf = all_pv[i]

					# calculate prob(max latency benefit)
					# we calculate this iteratively, from the smallest to the largest value
					# probability calc is basically probability of this value (p) times probability 
					# other prefixes are one of the smaller values (running prob)
					max_prob = p * np.prod(running_probs[all_pref_inds!=pref_j])
					running_probs[pref_j] += p

					if max_prob == 0 : continue

					lbx_i = np.where(lb - lbx <= 0)[0][0]
					px[lbx_i, ui] += max_prob * (1 - plf)
					px[0, ui] += max_prob * plf
		for ui in reversed(range(self.n_ug)):
			if np.sum(px[:,ui]) == 0:
				# This user experiences no benefit with probability 1
				# no benefit means no path, so it's actually just the most 
				# negative benefit we can give
				px[0,ui] = 1
		px = px / (np.sum(px,axis=0) + 1e-8) # renorm
		if subset_ugs:
			px = px[:,which_ugs_i]
		timers['benefit'] = time.time() - timers['start']

		## Calculate p(sum(benefits)) which is a convolution of the p(benefits)
		xsumx, psumx = self.pdf_sum_function(self.big_lbx[:,:px.shape[1]], px)
		# if subset_ugs: ## reweight to account for the fact that we're not looking at all volume
		# 	print("pre renorm {} {} {} {}".format(xsumx[0],xsumx[-1],np.where(psumx), benefit_renorm))
		# 	xsumx, psumx = rescale_pdf(xsumx.flatten(), psumx.flatten(), benefit_renorm)
		# 	print("post renorm {} {} {}".format(xsumx[0],xsumx[-1],np.where(psumx)))
		benefit = np.sum(xsumx.flatten() * psumx.flatten())


		if np.sum(psumx) < .5:
			print("ERRRRRR : {}".format(np.sum(psumx)))

		if subset_ugs: # reset vars
			lbx = lbx * benefit_renorm
			benefits = benefits * benefit_renorm
			self.big_lbx = self.big_lbx * benefit_renorm
		timers['convolution'] = time.time() - timers['start']
		print(timers)

		self.calc_cache.all_caches['lb'][tuple(a_effective.flatten())] = (benefit, (xsumx.flatten(),psumx.flatten()))
		
		return benefit, (xsumx.flatten(),psumx.flatten())

	def check_for_commands(self):
		# print("checking for commands in worker {}".format(self.worker_i))
		try:
			msg = self.main_socket.recv()
			# print("Received message in worker with length : {}".format(len(msg)))
		except zmq.error.Again:
			return
		try:
			msg = pickle.loads(msg)
		except:
			print("Failed parsing message of length : {}, sending back error message".format(len(msg)))
			pickle.dump(msg, open('error_{}_{}.pkl'.format(int(time.time()), self.worker_i),'wb'))
			self.main_socket.send(pickle.dumps("ERROR")) # should hopefully generate an error in the main thread
			return 
		cmd, data = msg
		# print("received command {} in worker {}".format(cmd, self.worker_i))
		if cmd == 'calc_lb':
			ret = []
			self.this_time_ip_cache = {}
			for (args,kwargs) in data:
				ret.append(self.latency_benefit(*args, **kwargs))
			del self.this_time_ip_cache
		elif cmd == 'calc_compressed_lb':

			ts = time.time()
			ret = []
			# self.this_time_ip_cache = {}
			base_args,base_kwa = data[0]
			base_adv, = base_args
			base_adv = base_adv.astype(bool)
			ret.append(self.latency_benefit(base_adv,**base_kwa))
			i=0
			for diff, kwa in data[1:]:
				for ind in zip(*diff):
					base_adv[ind] = not base_adv[ind]
				ret.append(self.latency_benefit(base_adv, **kwa))
				for ind in zip(*diff):
					base_adv[ind] = not base_adv[ind]

				i += 1
				kwa['verb'] = True
				if i%100 == 0 and kwa.get('verb'):
					print("worker {} : {} pct. done calcing grads, {} s per iter".format(self.worker_i, 
					i * 100.0 /len(data), (time.time() - ts) / i))
			# del self.this_time_ip_cache

		elif cmd == 'reset_new_meas_cache':
			self.calc_cache.clear_new_measurement_caches()
			self.this_time_ip_cache = {}
			ret = "ACK"
		elif cmd == 'update_parent_tracker':
			parents_on = data
			for ui in parents_on:
				for beaten_ingress, routed_ingress in parents_on[ui]:
					self.parent_tracker[ui, beaten_ingress, routed_ingress] = True
			ret = "ACK"
		elif cmd == 'update_deployment':
			deployment = data
			self.update_deployment(deployment)
			self.this_time_ip_cache = {}
			ret = "ACK"
		elif cmd == 'update_kwa':
			new_kwa = data
			if new_kwa.get('n_prefixes') is not None:
				self.n_prefixes = new_kwa.get('n_prefixes')
			if new_kwa.get('gamma') is not None:
				self.gamma = new_kwa.get('gamma')
			if new_kwa.get('with_capacity') is not None:
				self.with_capacity = new_kwa.get('with_capacity')
			ret = 'ACK'
		elif cmd == 'reset_cache':
			self.clear_caches()
			self.this_time_ip_cache = {}
			ret = "ACK"
		elif cmd == 'init':
			self.start_connection()
			return
		elif cmd == 'end':
			self.stop = True
			self.main_socket.close()
			return
		else:
			print("Invalid CMD in worker {} : {}".format(self.worker_i, cmd))
			exit(0)
		self.main_socket.send(pickle.dumps(ret))

	def run(self):
		while not self.stop:
			self.check_for_commands()
			time.sleep(.01)

if __name__ == "__main__":
	worker_i = int(sys.argv[1])
	pdc = Path_Distribution_Computer(worker_i)