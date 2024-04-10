import numpy as np, numba as nb, pickle, copy, zmq, time
np.setbufsize(262144*8)
np.random.seed(31414)
import random
random.seed(31415)
from constants import *
from helpers import *
from test_polyphase import *
from optimal_adv_wrapper import Optimal_Adv_Wrapper

from scipy.sparse import lil_matrix, csr_matrix

remeasure_a = None
try:
	remeasure_a = pickle.load(open('remeasure_a.pkl','rb'))
except:
	pass

USER_OF_INTEREST = None

# dont make this too big or you'll break the VM
# 2000 for lots of workers / smaller VMs. 15000 for fewer workers / large VMs
MAX_CACHE_SIZE = 8000

PROB_TOLERANCE = .05 ## if probabilities differ by more than this much, we have to recalculate things

def get_a_cache_rep(a):
	tups = []
	a = threshold_a(a)
	for x,y in zip(*np.where(a)):
		tups.append((x,y))
	return tuple(sorted(tups))

#### When these matrices are really big, helps to use numba, but can't work in multiprocessing scenarios
@nb.njit(fastmath=True,parallel=True)
def large_logical_and(arr1,arr2):
	return np.logical_and(arr1,arr2)

class Path_Distribution_Computer(Optimal_Adv_Wrapper):
	def __init__(self, worker_i, base_port, **kwargs):
		self.worker_i = worker_i
		self.port = base_port
		self.logging_iter = 0
		self.with_capacity = kwargs.get('with_capacity', False)
		self.pdf_sum_function = sum_pdf_fixed_point


		if kwargs.get('debug', False):
			self.n_prefixes = None
			return
		args, kwargs = self.start_connection()
		super().__init__(*args, **kwargs)

		with open(os.path.join(LOG_DIR, 'worker_{}_log-{}.txt'.format(self.worker_i, self.dpsize)),'w') as f:
			pass
		self.init_all_vars()
		self.run()
		print('started in worker {}'.format(self.worker_i))


	def init_all_vars(self):
		self.calculate_user_latency_by_peer()

		## Latency benefit for each user is -1 * MAX_LATENCY -> -1 MIN_LATENCY
		## divided by their contribution to the total volume (i.e., multiplied by a weight)
		## important that every worker has the same lbx
		min_vol,max_vol = np.min(self.whole_deployment_ug_vols), np.max(self.whole_deployment_ug_vols)
		total_deployment_volume = np.sum(self.whole_deployment_ug_vols)

		min_lbx = np.maximum(-1,-1 * NO_ROUTE_LATENCY * max_vol / total_deployment_volume)
		max_lbx = 0

		self.lbx = np.linspace(min_lbx, max_lbx,num=LBX_DENSITY)
		self.big_lbx = np.zeros((LBX_DENSITY, self.n_ug))
		for i in range(self.n_ug):
			self.big_lbx[:,i] = copy.copy(self.lbx)
		self.lb_range_trackers = {ui: [min_lbx,max_lbx] for ui in range(self.n_ug)}
		self.lb_range_alpha = .005 ## EWMA for update LB range definitions

		self.stop = False
		self.calc_cache = Calc_Cache()
		self.this_time_ip_cache = {}

		self.init_user_px_cache()

		self.iter = 0

	def start_connection(self):
		context = zmq.Context()
		self.main_socket = context.socket(zmq.REP)
		self.main_socket.setsockopt(zmq.RCVTIMEO, 1000)
		self.main_socket.bind('tcp://*:{}'.format(self.worker_i+self.port))
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

	def increment_iter(self):
		self.iter += 1

	def get_limited_cap_latency_multiplier(self):
		LIMITED_CAP_LATENCY_MULTIPLIER = 1.1
		power = 1.02
		# LIMITED_CAP_LATENCY_MULTIPLIER = 5
		# power = 1.05
		# return np.minimum(20, np.power(power,self.iter+1) * LIMITED_CAP_LATENCY_MULTIPLIER)
		return np.minimum(1.5, np.power(power,self.iter+1) * LIMITED_CAP_LATENCY_MULTIPLIER)

	def clear_caches(self):
		self.this_time_ip_cache = {}
		self.calc_cache.clear_all_caches()

	def get_ingress_probabilities_by_dict(self, a, verb=False, **kwargs):
		## Uses dictionaries to do the job
		# if tuple(a.astype(bool).flatten()) == tuple(remeasure_a.astype(bool).flatten()): 
		# 	verb = True
		# 	print("\n\nREMEASURING\n\n")
		a_log = threshold_a(a).astype(bool)


		sum_a = np.sum(a,axis=0)

		timers = {
			'cache_hits': 0,
			'cache_lookups': 0,
			'api': 0,
			'dpi': 0,
			'apugi': 0,
			'vpugi': 0,
			'sort_calc': 0,
			'final_calc': 0,
			'total': 0,
		}

		self.ingress_probabilities = {ui:{} for ui in range(self.n_ug)}
		for pref_i in np.where(sum_a)[0]:
			ts_loop = time.time()
			tloga = tuple(a_log[:,pref_i].flatten())
			##### WARNING -- if number of UGs and number of popps is the same, there could be ambiguity with the broadcasting
			##### but the likelihood of that event is pretty small
			if np.sum(a[:,pref_i]) == 0:
				continue
			try:
				for (poppi,ui), prob in self.this_time_ip_cache[tloga].items():
					# will need a more complicated caching mechanism if ever non-uniform
					self.ingress_probabilities[ui][poppi,pref_i] = 1.0/prob 
				timers['cache_hits'] += 1
				timers['cache_lookups'] += time.time() - ts_loop
				continue
			except KeyError:
				pass

			## i.e, for each user and for each popp. compute whether a parent of that popp is currently active
			active_parent_indicator = {}
			poppis_active = {poppi:None for poppi in np.where(a_log[:,pref_i])[0]}
			for ug,child,parent in self.parent_tracker: ### we should modify parent tracker to map parent to children
				ui = self.ug_to_ind[ug]
				parenti = self.popp_to_ind[parent]
				childi = self.popp_to_ind[child]
				try:
					poppis_active[parenti]
					active_parent_indicator[ui,childi] = 1
				except KeyError:
					continue
			timers['api'] += time.time() - ts_loop; ts_loop=time.time()

			## For active poppi in active_poppis, for user in poppi to users, if not parent active for poppi -> tabulate

			## Group by user
			self.this_time_ip_cache[tloga] = {}
			cacheref = self.this_time_ip_cache[tloga]
			for ui in range(self.n_ug):
				these_poppis = []
				ref = self.ui_to_poppi[ui]
				for poppi in poppis_active:
					try:
						ref[poppi]
					except KeyError:
						continue ### user doesn't have this popp
					try:
						active_parent_indicator[ui,poppi] ### We have an active parent, ignore
						continue
					except KeyError:
						these_poppis.append(poppi)

				if len(these_poppis) == 0:
					continue
				npoppis = len(these_poppis)
				likelihood = 1.0 / npoppis
				for poppi in these_poppis:
					self.ingress_probabilities[ui][poppi,pref_i] = likelihood
					### Cache the entries that have non-zero probability
					cacheref[poppi,ui] = npoppis
			timers['final_calc'] += time.time() - ts_loop; ts_loop=time.time()

		# if np.random.random() > 0 and self.worker_i == 0:
		# 	print('\n')
		# 	for k,v in timers.items():
		# 		print("{} -- {} s".format(k,round(v,5)))

	def init_user_px_cache(self):
		self.user_px = np.zeros((len(self.lbx), self.n_ug))
		self.ingress_px = np.zeros((self.n_popp, self.n_ug))
		self.p_link_fails = np.zeros(self.n_popp)
		self.link_failure_severities = np.zeros(self.n_popp)
		self.big_lbx = np.zeros((LBX_DENSITY, self.n_ug))
		for ui in range(self.n_ug):
			self.big_lbx[:,ui] = np.linspace(self.lb_range_trackers[ui][0], self.lb_range_trackers[ui][1], 
					num=LBX_DENSITY)
		self.user_ip_cache = {
			'init_ip': None,
			'default_px': copy.copy(self.user_px),
			'big_lbx': copy.copy(self.big_lbx),
			'ingress_px': copy.copy(self.ingress_px),
			'p_link_fails': copy.copy(self.p_link_fails),
			'link_failure_severities': copy.copy(self.link_failure_severities),
		}

	def summarize_cache_size(self):
		## 
		for obj,nm in zip([self.user_ip_cache, self.calc_cache, self.this_time_ip_cache], 
			['user ip', 'calc cache', 'this time ip']):
			self.print("{} cache -- {} size".format(nm,round(len(pickle.dumps(obj))/1e6)))

	def lfs_to_penalty(self, lfs):
		"""LFS is the greedy-allocation volume divided by the link capacity. Want to compute a
			multiplicative latency penalty to encourage people to not inundate links. 
			But can't be too rough so that it's unstable."""
		return np.power(lfs, .1)

	def latency_benefit(self, a, **kwargs):
		"""Calculates distribution of latency benefit at a given advertisement. Benefit is the sum of 
			benefits across all users."""

		a_effective = threshold_a(a)
		verb = kwargs.get('verbose_workers')
		# print(tuple(a_effective.astype(bool).flatten()))
		# print(tuple(remeasure_a.astype(bool).flatten()))
		remeasure = False
		if verb:
			try:
				remeasure_a = pickle.load(open('remeasure_a.pkl','rb'))
				if tuple(a_effective.astype(bool).flatten()) == tuple(remeasure_a.astype(bool).flatten()): 
					remeasure = True
					print("\n\n\nREMEASURING\n\n\n")
			except:
				pass

		
		if not verb:
			## don't rely on caching if we want to log / print statistics
			try:
				cache_rep = get_a_cache_rep(a_effective)
				benefit, (xsumx_cache_rep, psumx_cache_rep) = self.calc_cache.all_caches['lb'][cache_rep]
				xsumx = np.linspace(xsumx_cache_rep[0], xsumx_cache_rep[1], num=LBX_DENSITY)
				psumx = np.zeros(LBX_DENSITY)
				for i,d in psumx_cache_rep.items():
					psumx[i] = d
				ret = (benefit, (xsumx,psumx))

				return ret
			except KeyError:
				pass
		# else:
		# 	pickle.dump(self.calc_cache.all_caches['lb'], open("worker_{}_cache.pkl".format(self.worker_i), 'wb'))

		super_verbose=False
		# if os.path.exists('interesting_cache_rep_{}.pkl'.format(self.worker_i)):
		# 	cache_rep = pickle.load(open('interesting_cache_rep_{}.pkl'.format(self.worker_i),'rb'))

		# 	if get_a_cache_rep(a_effective) == cache_rep:
		# 		super_verbose=True


		USER_OF_INTEREST = None #### UI is respect to the whole deployment
		WORKER_OF_INTEREST = None


		timers = {
			'cache_lookup': 0,
			'capacity_1': 0,
			'capacity_2': 0,
			'capacity_3': 0,
			'capacity_4': 0,
			'probs': 0,
			'benefit': 0,
			'convolution': 0,
			'start': time.time()
		}

		## Dims are path, prefix, user
		# self.get_ingress_probabilities_by_a_matmul(a_effective, **kwargs)
		self.get_ingress_probabilities_by_dict(a_effective, **kwargs)
		p_mat = self.ingress_probabilities
		benefits = self.measured_latency_benefits
		lbx = self.lbx

		if super_verbose:
			print("Entering super verbose")
			print(p_mat.get(USER_OF_INTEREST))

		timers['probs'] = time.time()

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

			lbx = copy.copy(self.lbx)
			lbx = lbx / benefit_renorm
			benefits = benefits / benefit_renorm
			self.big_lbx = self.big_lbx / benefit_renorm

		#### SETUP CACHE
		# caching px by ingress probabilities matrix
		# loop over users for which something is different
		ug_inds_to_loop = {}
		if self.user_ip_cache['init_ip'] is not None:
			for ui in range(self.n_ug):
				refa = self.user_ip_cache['init_ip'][ui]
				refb = p_mat[ui]
				
				for k in set(refa).union(set(refb)):
					if np.abs(refa.get(k,0) - refb.get(k,0)) > PROB_TOLERANCE:
						# print(ui)
						# print(get_difference(refa,refb))
						# print(get_difference(refb,refa))
						# print(k)
						# print(refa.get(k,0))
						# print(refb.get(k,0))
						# if np.random.random() > .99:exit(0)

						ug_inds_to_loop[ui] = None
						break
			ug_inds_to_loop = np.array(sorted(list(ug_inds_to_loop)))
			self.user_px = copy.copy(self.user_ip_cache['default_px'])
			self.big_lbx = copy.copy(self.user_ip_cache['big_lbx'])
			if self.with_capacity:
				self.ingress_px = copy.copy(self.user_ip_cache['default_ingress_px'])
				self.p_link_fails = copy.copy(self.user_ip_cache['p_link_fails'])
				self.link_failure_severities = copy.copy(self.user_ip_cache['link_failure_severities'])
		else:
			ug_inds_to_loop = np.arange(self.n_ug)
		changed_popps = {} # notes popps for which I need to recompute p link fails
		if len(ug_inds_to_loop) > 0:
			if self.with_capacity:
				tmp = copy.copy(self.ingress_px)
				self.ingress_px[:,ug_inds_to_loop] = 0
				for poppi,_ in zip(*np.where(tmp != self.ingress_px)):
					changed_popps[poppi] = None

		timers['cache_lookup'] = time.time()

		if super_verbose:
			print("Computing for user super verbose: {}".format(USER_OF_INTEREST in ug_inds_to_loop))


		if self.with_capacity:
			## holds P(ingress) for each user
			vol_x = self.vol_x

			all_pref_inds = np.arange(self.n_prefixes)
			recalc_popps = {} # tracks popps with new p link failure probability
			for ui in ug_inds_to_loop:
				all_pv_i = p_mat[ui]
				# Holds prefix j, ingress poppi, benefit and probability
				all_pv = [(prefj,benefits[poppi,ui],all_pv_i[poppi,prefj],poppi) for poppi,prefj in all_pv_i]
				if len(all_pv) == 0:
					# this user has no paths
					continue
				if len(all_pv) == 1:
					_, _, p, poppi = all_pv[0]
					self.ingress_px[poppi, ui] += p
					changed_popps[poppi] = None 
				else:
					all_pv = sorted(all_pv,key=lambda el : el[1])
					running_probs = np.zeros((self.n_prefixes))
					running_probs[all_pv[0][0]] = all_pv[0][2]
					
					prefs_exist = list(set([el[0] for el in all_pv]))
					for pref_j in get_difference(all_pref_inds, prefs_exist):
						running_probs[pref_j] = 1 

					for i in range(1,len(all_pv)):
						pref_j, _, p, poppi = all_pv[i]

						# calculate prob(max latency benefit)
						# we calculate this iteratively, from the smallest to the largest value
						# probability calc is basically probability of this value (p) times probability 
						# other prefixes are one of the smaller values (running prob)
						max_prob = p * np.prod(running_probs[all_pref_inds!=pref_j])
						running_probs[pref_j] += p
						if max_prob == 0 : continue

						self.ingress_px[poppi, ui] += max_prob
						changed_popps[poppi] = None 
			timers['capacity_1'] = time.time()

			## prob that user j contributes volx i
			for ingress_i in changed_popps:
				ug_prob_vols_this_ingress_i = np.where(self.ingress_px[ingress_i,:] > 0)[0]
				
				include_in_px = []
				definitely_here = 0
				for ui in ug_prob_vols_this_ingress_i:
					# Model user as bernoulli on this ingress
					if self.ingress_px[ingress_i, ui] >= .99:
						definitely_here += self.ug_vols[ui]
						continue
					include_in_px.append(ui)
				n_nontrivial_ugs = len(include_in_px)
				severity = 0
				if n_nontrivial_ugs > 0:
					ug_prob_vols_this_ingress = np.zeros((len(self.vol_x), n_nontrivial_ugs))
					tmpind = 0

					for ui in include_in_px:
						ug_prob_vols_this_ingress[self.ui_to_vol_i[ui], tmpind] = self.ingress_px[ingress_i, ui]
						ug_prob_vols_this_ingress[0, tmpind] = 1 - self.ingress_px[ingress_i, ui]
						tmpind += 1
					x_vol_this_ingress, p_vol_this_ingress = self.pdf_sum_function(self.big_vol_x[:,0:n_nontrivial_ugs],
						ug_prob_vols_this_ingress)
					x_vol_this_ingress = x_vol_this_ingress.flatten()
					p_vol_this_ingress = p_vol_this_ingress.flatten()
					x_vol_this_ingress += definitely_here
					new_p = np.sum(p_vol_this_ingress[x_vol_this_ingress > self.link_capacities[ingress_i]])
					if new_p > 0:
						severity = np.sum(p_vol_this_ingress * x_vol_this_ingress)
				else:
					x_vol_this_ingress = definitely_here
					p_vol_this_ingress = 1
					new_p = x_vol_this_ingress > self.link_capacities[ingress_i]
					if new_p > 0:
						severity = x_vol_this_ingress
				## Doesn't this negate my ability to split across UGs???? ughghghghghg
				# will probably have to do main -> compute prob -> return -> combine prob -> compress -> send to workers
				# ~or~ pretend every link is 1/n_workers capacity and shuffle users
				# ~or~ group all users with similar reachabilities in the same worker
				old_p = self.p_link_fails[ingress_i]
				if np.abs(old_p - new_p) > PROB_TOLERANCE:
					recalc_popps[ingress_i] = None
				self.p_link_fails[ingress_i] = new_p
				self.link_failure_severities[ingress_i] = severity / self.link_capacities[ingress_i]

			if verb:
				for ingress_i in np.where(self.p_link_fails)[0]:
					# self.print("{} Ingress {} ({}) fails with probability {}, users {}".format(self.worker_i,ingress_i, self.popps[ingress_i],
					#  	self.p_link_fails[ingress_i], np.where(self.ingress_px[ingress_i,:] > .1)[0]))
					whole_deployment_uis = [self.whole_deployment_ug_to_ind[self.ugs[ui]] for ui in np.where(self.ingress_px[ingress_i,:])[0]]
					severity = self.link_failure_severities[ingress_i]
					# self.print(np.where(self.ingress_px[ingress_i,:]))
					# self.print("whole deployment uis {} ".format(whole_deployment_uis))
					# self.print("severity is {}".format(severity))


					# ui_global = self.whole_deployment_ug_to_ind[ug]
					uis = np.where(self.ingress_px[ingress_i,:] > .001)[0]
					uis_global = [self.whole_deployment_ug_to_ind[self.ugs[ui]] for ui in uis]
					users_str = "-".join([str(el) for el in uis_global])
					failing_popp = kwargs.get('failing_popp','none')
					if failing_popp != 'none':
						failing_popp = self.popp_to_ind[failing_popp]
					self.log("link_fail_report,{},{},{},{},{},{},{}\n".format(
						self.iter,ingress_i,failing_popp,self.link_capacities[ingress_i],
						severity,users_str,self.p_link_fails[ingress_i]))


					
					ug_prob_vols_this_ingress_i = np.where(self.ingress_px[ingress_i,:] > .1)[0]
					include_in_px = []
					definitely_here = 0
					for ui in ug_prob_vols_this_ingress_i:
						# Model user as bernoulli on this ingress
						if self.ingress_px[ingress_i, ui] >= .99:
							definitely_here += self.ug_vols[ui]
							continue
						include_in_px.append(ui)
					n_nontrivial_ugs = len(include_in_px)
					if n_nontrivial_ugs > 0:
						ug_prob_vols_this_ingress = np.zeros((len(self.vol_x), n_nontrivial_ugs))
						tmpind = 0

						for ui in include_in_px:
							ug_prob_vols_this_ingress[self.ui_to_vol_i[ui], tmpind] = self.ingress_px[ingress_i, ui]
							ug_prob_vols_this_ingress[0, tmpind] = 1 - self.ingress_px[ingress_i, ui]
							tmpind += 1
						x_vol_this_ingress, p_vol_this_ingress = self.pdf_sum_function(self.big_vol_x[:,0:n_nontrivial_ugs],
							ug_prob_vols_this_ingress)
						x_vol_this_ingress = x_vol_this_ingress.flatten()
						p_vol_this_ingress = p_vol_this_ingress.flatten()
						x_vol_this_ingress += definitely_here
					else:
						x_vol_this_ingress = definitely_here
						p_vol_this_ingress = 1
						new_p = x_vol_this_ingress > self.link_capacities[ingress_i]


					# self.print("Ingress capacity is {} while users contribute {}, prob {}".format(self.link_capacities[ingress_i],
					# 	x_vol_this_ingress,p_vol_this_ingress))

			timers['capacity_2'] = time.time()
			timers['capacity_3'] = time.time()
			ug_inds_to_loop = list(ug_inds_to_loop)
			for ingress_i in recalc_popps:
				for ui in self.poppi_to_ui[ingress_i]:
					ug_inds_to_loop.append(ui)
			ug_inds_to_loop = np.array(sorted(list(set(ug_inds_to_loop))))
		

		if len(ug_inds_to_loop) > 0:
			self.user_px[:,ug_inds_to_loop] = 0

		

		timers['capacity_4'] = time.time()

		lb_multiplier_link_failure = self.get_limited_cap_latency_multiplier()

		all_pref_inds = np.arange(self.n_prefixes)


		ug_benefit_updates = []
		for ui in ug_inds_to_loop:

			min_experienced_benefit, max_experienced_benefit = np.inf, -1 * np.inf
			all_pv_i = p_mat[ui]
			## combine benefit with bernoulli link failure
			all_pv = [(prefj, benefits[poppi,ui], all_pv_i[poppi,prefj], self.p_link_fails[poppi], 
				self.link_failure_severities[poppi]) \
				for poppi,prefj in all_pv_i]
			# if verb and self.worker_i == WORKER_OF_INTEREST and self.whole_deployment_ug_to_ind[self.ugs[ui]] == USER_OF_INTEREST:
			# 	self.print("PV : {}".format(all_pv))

			if len(all_pv) == 0:
				# this user has no paths
				continue
			elif len(all_pv) == 1:
				_, lb, p, plf, lfs = all_pv[0]

				if lb > max_experienced_benefit:
					max_experienced_benefit = lb
				if lb < min_experienced_benefit:
					min_experienced_benefit = lb

				if lb <= self.big_lbx[0,ui]:
					## reformat now, this is a big deal
					self.big_lbx[:,ui] = np.linspace(lb*1.5, self.big_lbx[-1,ui])
					lbx_i = np.where(lb - self.big_lbx[:,ui] <= 0)[0][0]
				elif lb >= self.big_lbx[-1,ui]:
					lbx_i = self.big_lbx.shape[0] - 1 
				else:
					lbx_i = np.where(lb - self.big_lbx[:,ui] <= 0)[0][0]
				if (verb or super_verbose) and self.worker_i == WORKER_OF_INTEREST and ui == USER_OF_INTEREST:
					self.print("multi LB {} LBXI {}, low : {} up : {}".format(lb,lbx_i,self.big_lbx[0,ui],
						self.big_lbx[-1,ui]))

				self.user_px[lbx_i, ui] += p * (1 -  plf)
				if plf > 0:
					lb_failure = lb * self.lfs_to_penalty(lfs) * lb_multiplier_link_failure
					# print("{},{},{},{}".format(round(lb,3),round(lfs,3),
					# 	round(lb_multiplier_link_failure,3),round(lb_failure,3)))

					if lb_failure > max_experienced_benefit:
						max_experienced_benefit = lb
					if lb_failure < min_experienced_benefit:
						min_experienced_benefit = lb

					if lb_failure <= self.big_lbx[0,ui]:
						## reformat now, this is a big deal
						self.big_lbx[:,ui] = np.linspace(lb_failure*1.5, self.big_lbx[-1,ui])
						limited_cap_lbxi = np.where(lb_failure - self.big_lbx[:,ui] <= 0)[0][0]
					elif lb_failure >= self.big_lbx[-1,ui]:
						limited_cap_lbxi = self.big_lbx.shape[0] - 1 
					else:
						limited_cap_lbxi = np.where(lb_failure - self.big_lbx[:,ui] <= 0)[0][0]

					self.user_px[limited_cap_lbxi, ui] += p * plf
			else:
				all_pv = sorted(all_pv,key=lambda el : el[1])
				
				running_probs = np.zeros((self.n_prefixes))
				running_probs[all_pv[0][0]] = all_pv[0][2]

				prefs_exist = list(set([el[0] for el in all_pv]))
				for pref_j in get_difference(all_pref_inds, prefs_exist):
					running_probs[pref_j] = 1 

				for i in range(1,len(all_pv)):
					pref_j, lb, p, plf, lfs = all_pv[i]

					# calculate prob(max latency benefit)
					# we calculate this iteratively, from the smallest to the largest value
					# probability calc is basically probability of this value (p) times probability 
					# other prefixes are one of the smaller values (running prob)
					max_prob = p * np.prod(running_probs[all_pref_inds!=pref_j])
					running_probs[pref_j] += p

					if max_prob == 0 : continue

					if lb > max_experienced_benefit:
						max_experienced_benefit = lb
					if lb < min_experienced_benefit:
						min_experienced_benefit = lb

					if lb <= self.big_lbx[0,ui]:
						## reformat now, this is a big deal
						self.big_lbx[:,ui] = np.linspace(lb*1.5, self.big_lbx[-1,ui])
						lbx_i = np.where(lb - self.big_lbx[:,ui] <= 0)[0][0]
					elif lb >= self.big_lbx[-1,ui]:
						lbx_i = self.big_lbx.shape[0] - 1 
					else:
						lbx_i = np.where(lb - self.big_lbx[:,ui] <= 0)[0][0]
					if (verb or super_verbose) and self.worker_i == WORKER_OF_INTEREST and ui == USER_OF_INTEREST:
						self.print("multi LB {} LBXI {}, low : {} up : {}".format(lb,lbx_i,self.big_lbx[0,ui],
							self.big_lbx[-1,ui]))

					self.user_px[lbx_i, ui] += max_prob * (1 - plf)
					if plf > 0:
						lb_failure = lb * self.lfs_to_penalty(lfs)
						# print("{},{},{},{}".format(round(lb,3),round(lfs,3),
						# 	round(lb_multiplier_link_failure,3),round(lb_failure,3)))
						if lb_failure > max_experienced_benefit:
							max_experienced_benefit = lb
						if lb_failure < min_experienced_benefit:
							min_experienced_benefit = lb
						
						if lb_failure <= self.big_lbx[0,ui]:
							## reformat now, this is a big deal
							self.big_lbx[:,ui] = np.linspace(lb_failure*1.5, self.big_lbx[-1,ui])
							limited_cap_lbxi = np.where(lb_failure - self.big_lbx[:,ui] <= 0)[0][0]
						elif lb_failure >= self.big_lbx[-1,ui]:
							limited_cap_lbxi = self.big_lbx.shape[0] - 1 
						else:
							limited_cap_lbxi = np.where(lb_failure - self.big_lbx[:,ui] <= 0)[0][0]
						
						self.user_px[limited_cap_lbxi, ui] += max_prob * plf

			# static assignment to best possible
			max_experienced_benefit = self.best_latency_benefits[ui] 	
			# lower bound it a little more
			min_experienced_benefit = min_experienced_benefit * 1.5
			if min_experienced_benefit == max_experienced_benefit:
				min_experienced_benefit -= .1
			self.lb_range_trackers[ui][0] = (1-self.lb_range_alpha) * self.lb_range_trackers[ui][0] + \
				self.lb_range_alpha * min_experienced_benefit
			self.lb_range_trackers[ui][1] = (1-self.lb_range_alpha) * self.lb_range_trackers[ui][1] + \
				self.lb_range_alpha * max_experienced_benefit

			if np.abs(self.lb_range_trackers[ui][0] - self.big_lbx[0,ui]) > .01 or \
				np.abs(self.lb_range_trackers[ui][1] - self.big_lbx[-1,ui]) > .01:
				ug_benefit_updates.append(ui)

			if self.worker_i == WORKER_OF_INTEREST and ui == USER_OF_INTEREST and (super_verbose or verb):
				print(self.lb_range_trackers[ui])
				print(all_pv)
				print("{} {}".format(min_experienced_benefit/1.5, max_experienced_benefit))

		
		for ui in ug_inds_to_loop:
			if np.sum(self.user_px[:,ui]) == 0:
				# This user experiences no benefit with probability 1
				# no benefit means no path, so it's actually just the most 
				# negative benefit we can give
				self.user_px[0,ui] = 1
				if verb or super_verbose:
					log_str = "no_path_warning,{}\n".format(ui)
					self.log(log_str)
		self.user_px = self.user_px / (np.sum(self.user_px,axis=0) + 1e-8) # renorm

		if verb or super_verbose:
			total_b = 0
			prnts = []
			for bi,ui in zip(*np.where(self.user_px)):
				p = self.user_px[bi,ui]
				if p > .001:
					prnts.append((ui,bi,self.big_lbx[bi,ui],p))
					total_b += p*self.big_lbx[bi,ui]
			if kwargs.get('failing_popp') is None or super_verbose:
				for ui,bi,lb,p in sorted(prnts, key = lambda el : el[0]):
					# get likely popps
					likely_popps = np.where(self.ingress_px[:,ui]>.01)[0]
					likely_popps_str = ""
					likely_popps_str = "-".join(["{} ({} ms)".format(poppi,round(self.ug_perfs[self.ugs[ui]][self.popps[poppi]],2)) for poppi in likely_popps])
					ug = self.ugs[ui]
					ui_global = self.whole_deployment_ug_to_ind[ug]
					log_str = "benefit_estimate,{},{},{},{},{},{},{}\n".format(
						self.iter,ui_global,bi,lb,round(p,2),likely_popps_str,self.logging_iter)
					if (verb or super_verbose) and self.worker_i == WORKER_OF_INTEREST and ui_global == USER_OF_INTEREST:
						self.print("PV : {}".format(p_mat[ui]))
						self.print(log_str)
					self.log(log_str)
				self.logging_iter += 1
			# print("Total : {}".format(total_b))

		## save calc cache for later if its not set
		if self.user_ip_cache['init_ip'] is None:
			# print("Setting cache variables")
			self.user_ip_cache['init_ip'] = p_mat
			self.user_ip_cache['default_px'] = copy.copy(self.user_px)
			self.user_ip_cache['big_lbx'] = copy.copy(self.big_lbx)
			if self.with_capacity:
				self.user_ip_cache['default_ingress_px'] = copy.copy(self.ingress_px)
				self.user_ip_cache['p_link_fails'] = copy.copy(self.p_link_fails)
				self.user_ip_cache['link_failure_severities'] = copy.copy(self.link_failure_severities)

		if subset_ugs:
			px = self.user_px[:,which_ugs_i]
		else:
			px = self.user_px
		timers['benefit'] = time.time()

		## Calculate p(sum(benefits)) which is a convolution of the p(benefits)
		xsumx, psumx = self.pdf_sum_function(self.big_lbx, px)
		xsumx = xsumx.flatten(); psumx = psumx.flatten()
		benefit = np.sum(xsumx * psumx)

		## Update the big LBX object
		for ui in ug_benefit_updates:
			self.big_lbx[:,ui] = np.linspace(self.lb_range_trackers[ui][0], self.lb_range_trackers[ui][1], 
				num=LBX_DENSITY)

		if np.sum(psumx) < .5:
			print("ERRRRRR : {}".format(np.sum(psumx)))
			exit(0)

		if subset_ugs: # reset vars
			lbx = lbx * benefit_renorm
			benefits = benefits * benefit_renorm
			self.big_lbx = self.big_lbx * benefit_renorm

		timers['convolution'] = time.time()
		# if np.random.random() > 0:
		# 	t_order = ['start','probs','cache_lookup','capacity_1','capacity_2',
		# 		'capacity_3','capacity_4','benefit','convolution']
		# 	t_deltas = [timers[t_order[i+1]] - timers[t_order[i]] for i in range(len(t_order)-1)]
		# 	time_str = "  --  ".join("{}--{}ms".format(t_order[i+1],
		# 		int(t_deltas[i]*1000)) for i in range(len(t_order)-1))
		# 	print("Worker {} looping over {} pct of UGs".format(self.worker_i, 
		# 		round(len(ug_inds_to_loop)*100.0/self.n_ug),3))
		# 	print(time_str)
		# 	print("{} seconds total".format(round(time.time() - timers['start'], 3)))
		# 	print('\n')


		### Store compressed versions of these variables
		cache_rep = get_a_cache_rep(a_effective)
		xsumx_cache_rep = (xsumx[0], xsumx[-1])
		psumx_cache_rep = {}
		for i in np.where(psumx)[0]:
			psumx_cache_rep[i] = psumx[i]

		# if kwargs.get('sanitycheck',True):
		# 	for existing_cache_member in list(self.calc_cache.all_caches['lb']):
		# 		if len(get_difference(cache_rep,existing_cache_member)) == 1 or len(get_difference(cache_rep,existing_cache_member)) == 1:
		# 			_benefit, (_xsumx_cache_rep, _psumx_cache_rep) = self.calc_cache.all_caches['lb'][existing_cache_member]
		# 			if np.abs(_benefit - benefit) > 1:
		# 				print("Two very close cache members disagree by a lot: {} vs {}!".format(benefit,_benefit))
		# 				pickle.dump(existing_cache_member, open('interesting_cache_rep_{}.pkl'.format(self.worker_i),'wb'))

		# 				## Literally just reaches into the cache
		# 				real1,_ = self.latency_benefit(a, sanitycheck=False)
		# 				adv_in_cache = np.zeros((self.n_popps,self.n_prefixes))
		# 				for poppi,prefi in existing_cache_member:
		# 					adv_in_cache[poppi,prefi] = 1
		# 				real2,_ = self.latency_benefit(adv_in_cache, sanitycheck=False)
		# 				print("Recalc : {} vs {}".format(real1,real2))


		# 				## No cache on cache_rep
		# 				real1,_ = self.latency_benefit(a, verbose_workers=True, sanitycheck=False)
		# 				adv_in_cache = np.zeros((self.n_popps,self.n_prefixes))
		# 				for poppi,prefi in existing_cache_member:
		# 					adv_in_cache[poppi,prefi] = 1
		# 				real2,_ = self.latency_benefit(adv_in_cache, verbose_workers=True, sanitycheck=False)
		# 				print("Recalc no cache : {} vs {}".format(real1,real2))


		# 				exit(0)

		self.calc_cache.all_caches['lb'][cache_rep] = (benefit, (xsumx_cache_rep, psumx_cache_rep))
		
		if super_verbose:
			print("Exiting super verbose")
			self.print("Storing benefit {}".format(benefit))

		return benefit, (xsumx, psumx)

	def log(self,s):
		self.log_ptr = open(os.path.join(LOG_DIR, 'worker_{}_log-{}.txt'.format(self.worker_i, self.dpsize)),'a')
		self.log_ptr.write(s)
		self.log_ptr.close()

	def print(self, s):
		print("Worker {} -- {}".format(self.worker_i, s))

	def check_clear_cache(self):
		cache_to_clear = self.calc_cache.all_caches['lb']

		if len(cache_to_clear) > MAX_CACHE_SIZE:
			# order of lbx_density + n_popps*n_prefixes per entry
			# self.print("Clearing calc cache, currently size {}".format(
			# 	len(pickle.dumps(self.calc_cache))/1e6))
			# self.print("Clearing calc cache, current len {}".format(len(cache_to_clear)))
			self.this_time_ip_cache = {}
			self.calc_cache.all_caches['lb'] = {}

	def clear_new_meas_caches(self):
		# print("Clearing caches")
		self.this_time_ip_cache = {}
		self.init_user_px_cache()
		self.calc_cache.clear_new_measurement_caches()

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
			# if len(data) > 1:
			# 	if self.worker_i == 0:
			# 		pickle.dump([data,self.output_deployment()], open('cache/tmp.pkl','wb'))
			# 	exit(0)
			ts = time.time()
			tlp = time.time()
			ret = []
			base_args,base_kwa = data[0]
			base_adv, = base_args
			base_adv = base_adv.astype(bool)

			if len(data[1:]) > 1:
				### Prepopulate the cache with the "average" advertisement
				self.init_user_px_cache()
				avg_adv = np.zeros(base_adv.shape)
				for diff, kwa in data[1:]:
					for ind in zip(*diff):
						base_adv[ind] = not base_adv[ind]
					avg_adv += base_adv.astype(np.int32)
					for ind in zip(*diff):
						base_adv[ind] = not base_adv[ind]
				avg_adv = avg_adv / len(data[1:])
				avg_adv = avg_adv > .5
				self.latency_benefit(avg_adv)
			ret.append(self.latency_benefit(base_adv,**base_kwa))
			i=0
			for diff, kwa in data[1:]:
				kwa['verbose_workers'] = base_kwa.get('verbose_workers',False) or kwa.get('verbose_workers',False)
				for ind in zip(*diff):
					base_adv[ind] = not base_adv[ind]
				ret.append(self.latency_benefit(base_adv, **kwa))
				for ind in zip(*diff):
					base_adv[ind] = not base_adv[ind]
				i += 1
				# kwa['verb'] = True
				if time.time() - tlp > 100:
					self.print("{} pct. done calcing latency benefits, {}ms per iter".format( 
						round(i * 100.0 /len(data),1), round(1000*(time.time() - ts) / i)))
					tlp = time.time()
				self.check_clear_cache()
			# if len(data)>10:
			# 	print("Worker {} calcs took {}s".format(self.worker_i, int(time.time() - ts)))

		elif cmd == "solve_lp":
			ret = []
			ts = time.time()
			n_iters,t_per_iter = 0,0
			for adv_i, adv, deployment, update_dep in sorted(data, key = lambda el : el[0]):
				if update_dep:
					deployment_save = self.output_deployment()
					self.clear_caches()
					self.update_deployment(deployment,quick_update=True,verb=False,exit_on_impossible=False)
				this_ret = self.solve_lp_with_failure_catch(adv)
				if update_dep:
					self.update_deployment(deployment_save,quick_update=True,verb=False,exit_on_impossible=False)
				ret.append((adv_i, this_ret))
				n_iters += 1

				t_per_iter = round((time.time() - ts)/n_iters,2)

		elif cmd == 'reset_new_meas_cache':
			self.clear_new_meas_caches()
			ret = "ACK"
		elif cmd == 'update_parent_tracker':
			parents_on = data
			for ug in parents_on:
				for beaten_ingress, routed_ingress in parents_on[ug]:
					self.parent_tracker[ug, beaten_ingress, routed_ingress] = True
			if len(parents_on) > 0:
				self.clear_new_meas_caches()
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
		elif cmd == 'increment_iter':
			self.increment_iter()
			ret = "ACK"
		elif cmd == 'reset_cache':
			self.clear_caches()
			ret = "ACK"
		elif cmd == 'init':
			self.start_connection()
			return
		elif cmd == 'end':
			print("Received end command in worker {}, stopping".format(self.worker_i))
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
		print("Ended run loop in worker {}".format(self.worker_i))

if __name__ == "__main__":
	worker_i = int(sys.argv[1])
	base_port = int(sys.argv[2])
	pdc = Path_Distribution_Computer(worker_i, base_port)