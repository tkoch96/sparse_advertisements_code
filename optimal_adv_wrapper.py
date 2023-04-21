import numpy as np, time, tqdm
np.setbufsize(262144*8)
from constants import *
from helpers import *
from test_polyphase import *
from worker_comms import Worker_Manager


class Ing_Obj:
	def __init__(self, ing):
		self.id = ing
		self.parents = {}
		self.children = {}
		self.alive = True

	def is_leaf(self):
		return len(self.children) == []

	def add_parent(self, parent):
		self.parents[parent.id] = parent

	def add_child(self, child):
		self.children[child.id] = child

	def has_child(self, potential_children):
		return len(get_intersection(self.children, potential_children)) > 0

	def has_parent(self, potential_parents):
		return len(get_intersection(self.parents, potential_parents)) > 0

	def has_info(self):
		return len(self.parents) > 0 or len(self.children) > 0

	def print(self):
		print("Node : {}, parents: {}, children: {}, alive: {}".format(
			self.id, list(self.parents), list(self.children), self.alive))

	def kill(self):
		self.alive = False

class Optimal_Adv_Wrapper:
	### Wrapper class for all solutions to finding optimal advertisements
	def __init__(self, deployment, lambduh=1.0, verbose=True, gamma=0, **kwargs):
		self.verbose = verbose
		self.ts_loop = time.time()
		self.advertisement_cost = self.l1_norm
		self.epsilon = .05 # change in objective less than this -> stop
		self.max_n_iter = 300 # maximum number of learning iterations
		self.lambduh = lambduh # sparsity cost
		self.gamma = gamma # resilience cost
		self.with_capacity = kwargs.get('with_capacity', False)
		self.n_prefixes = kwargs.get('n_prefixes')

		self.pdf_sum_function = sum_pdf_fixed_point

		self.calc_cache = Calc_Cache()
		self.update_deployment(deployment)	

	def set_worker_manager(self, wm):
		self.worker_manager = wm

	def get_worker_manager(self):
		return self.worker_manager

	def get_n_workers(self):
		return self.worker_manager.get_n_workers()

	def send_receive_workers(self, msg):
		return self.worker_manager.send_receive_workers(msg)

	def send_receive_worker(self, worker_i, msg):
		return self.worker_manager.send_receive_worker(worker_i, msg)

	def clear_caches(self):
		self.calc_cache.clear_all_caches()
		self.measured_prefs = {ui: {self.popp_to_ind[popp]: Ing_Obj(self.popp_to_ind[popp]) for popp in \
			self.ug_perfs[self.ugs[ui]]} for ui in range(self.n_ug)}
		try:
			msg = pickle.dumps(['reset_cache', ()])
			self.send_receive_workers(msg)
		except AttributeError:
			# not initialized yet
			pass

	def clear_new_measurement_caches(self):
		self.calc_cache.clear_new_measurement_caches()
		self.this_time_ip_cache = {}
		msg = pickle.dumps(['reset_new_meas_cache',()])
		self.send_receive_workers(msg)

	def get_cache(self):
		return self.calc_cache.get_cache()

	def get_init_kwa(self):
		return {
			'lambduh': self.lambduh, 
			'gamma': self.gamma, 
			'verbose': False,
			'n_prefixes': self.n_prefixes,
			'with_capacity': self.with_capacity,
		}

	def update_cache(self, new_cache):
		self.calc_cache.update_cache(new_cache)
		### Keeps the parent-tracker up to date
		for ui, inds in self.calc_cache.all_caches['parents_on'].items():
			for childi,parenti in inds:
				self.parent_tracker[ui,childi,parenti] = True

	def update_deployment(self, deployment):
		self.deployment = deployment
		self.ugs = deployment['ugs']
		self.n_ug = len(self.ugs)
		self.whole_deployment_ugs = deployment['whole_deployment_ugs']
		self.whole_deployment_n_ug = len(self.whole_deployment_ugs)
		self.ug_to_ind = {ug:i for i,ug in enumerate(self.ugs)}
		self.whole_deployment_ug_to_ind = {ug:i for i,ug in enumerate(self.whole_deployment_ugs)}
		self.ug_perfs = deployment['ug_perfs']
		self.ug_anycast_perfs = deployment['ug_anycast_perfs']
		# Shape of the variables
		self.popps = sorted(list(set(deployment['popps'])))
		self.n_popps = len(self.popps)
		self.pops = list(set([u[0] for u in self.popps]))
		self.popp_to_ind = {k:i for i,k in enumerate(self.popps)}

		self.poppi_to_ui = {}
		for ug in self.ug_perfs:
			for popp in self.ug_perfs[ug]:
				try:
					self.poppi_to_ui[self.popp_to_ind[popp]].append(self.ug_to_ind[ug])
				except KeyError:
					self.poppi_to_ui[self.popp_to_ind[popp]] = [self.ug_to_ind[ug]]
		for poppi in get_difference(list(range(self.n_popps)), self.poppi_to_ui):
			self.poppi_to_ui[poppi] = []

		self.provider_popps = deployment['provider_popps']
		self.n_provider_popps = len(self.provider_popps) # number of provider PoPps
		self.n_popp = len(self.popps)
		if self.n_prefixes is None:
			self.n_prefixes = np.maximum(3,self.n_popp // 3)
		self.n_providers = deployment['n_providers'] # number of provider ASes
		self.provider_popp_inds = [self.popp_to_ind[popp] for popp in self.provider_popps]
		self.ground_truth_ingress_priorities = deployment['ingress_priorities']

		# for popp, prio in self.ground_truth_ingress_priorities[self.ugs[19]].items():
		# 	print("{} -- prio {}".format(self.popp_to_ind[popp],prio))
		# exit(0)

		self.metro_loc = deployment['metro_loc']
		self.pop_to_loc = deployment['pop_to_loc']
		self.link_capacities_by_popp = deployment['link_capacities']

		try:
			self.worker_i
			### Worker thread, need to divide capacities by the number of workers
			### where appropriate
			for popp, cap in self.link_capacities_by_popp.items():
				if popp in self.provider_popps:
					self.link_capacities_by_popp[popp] = cap / N_WORKERS
		except AttributeError:
			pass

		self.link_capacities = {self.popp_to_ind[popp]: self.link_capacities_by_popp[popp] for popp in self.popps}
		self.link_capacities_arr = np.zeros(self.n_popp)
		for poppi, cap in self.link_capacities.items():
			self.link_capacities_arr[poppi] = cap

		self.ingress_priority_inds = {ug:{self.popp_to_ind[popp]: \
			self.ground_truth_ingress_priorities[ug][popp] for popp in self.ug_perfs[ug]} \
			for ug in self.ugs}

		self.ug_to_vol = deployment['ug_to_vol']
		self.whole_deployment_ug_to_vol = deployment['whole_deployment_ug_to_vol']
		self.ug_vols = np.zeros(self.n_ug)
		self.whole_deployment_ug_vols = np.zeros(self.whole_deployment_n_ug)
		for ug, v in self.ug_to_vol.items():
			self.ug_vols[self.ug_to_ind[ug]] = v
		for ug, v in self.whole_deployment_ug_to_vol.items():
			self.whole_deployment_ug_vols[self.whole_deployment_ug_to_ind[ug]] = v
		all_vols = list(self.ug_to_vol.values())
		self.vol_x = np.linspace(0, max(all_vols))
		self.big_vol_x = np.zeros((len(self.vol_x), len(self.whole_deployment_ugs)))
		for i in range(len(self.whole_deployment_ugs)):
			self.big_vol_x[:,i] = copy.copy(self.vol_x)
		self.ui_to_vol_i = {}
		for ui in range(self.n_ug):
			self.ui_to_vol_i[ui] = np.where(self.ug_to_vol[self.ugs[ui]] -\
				self.vol_x <=0)[0][0]



		# ingress,ug -> stores ~ how many times ingress wins for a ug
		self.n_wins = {ug:{i:0 for i in range(self.n_popp)} for ug in self.ugs}

			# if len(self.ug_perfs[self.ugs[ui]]) > 1:
			# 	for popp in self.ug_perfs[self.ugs[ui]]:
			# 		print(self.ingress_priority_inds[self.ugs[ui]][self.popp_to_ind[popp]])
			# 		print("{} {}".format(popp,self.popp_to_ind[popp]))
			# 	print(self.popp_by_ug_indicator[:,ui])
			# 	exit(0)

		try:
			### Worker bee
			print("Allocating objects in worker {}".format(self.worker_i))
			self.parent_tracker = np.zeros((self.n_ug, self.n_popp, self.n_popp), dtype=bool)
			# ijk'th entry of ingress probabilities is probability that user k ingresses over popp i for prefix j
			self.ingress_probabilities = np.zeros((self.n_popp, self.n_prefixes, self.n_ug))\

			self.popp_by_ug_indicator_no_rank = np.zeros((self.n_popp, self.n_ug), dtype=bool)
			for ui in range(self.n_ug):
				for popp in self.ug_perfs[self.ugs[ui]]:
					self.popp_by_ug_indicator_no_rank[self.popp_to_ind[popp],ui] = True

		except AttributeError:
			### Main thread
			print("Allocating objects in main thread")
			with open(os.path.join(CACHE_DIR, 'main_thread_log.txt'),'w') as f:
				pass
			self.parent_tracker = {}
			self.popp_by_ug_indicator = np.zeros((self.n_popp, self.n_ug), np.ushort)
			for ui in range(self.n_ug):
				for popp in sorted(self.ug_perfs[self.ugs[ui]]):
					self.popp_by_ug_indicator[self.popp_to_ind[popp], ui] = self.n_popp + 1 - self.ingress_priority_inds[self.ugs[ui]][self.popp_to_ind[popp]]

		try:
			n_workers = self.get_n_workers()
			print("Splitting and assigning deployments")
			subdeployments = split_deployment_by_ug(self.deployment, n_chunks=n_workers)
			for worker in range(n_workers):
				if len(subdeployments[worker]['ugs']) == 0: continue
				## It would be annoying to make the code work for cases in which a processor focuses on one user
				assert len(subdeployments[worker]['ugs']) >= 1
				# send worker startup information
				self.worker_manager.worker_to_deployments[worker] = subdeployments[worker]
				
				msg = pickle.dumps(('update_kwa', self.get_init_kwa()))
				self.send_receive_worker(worker, msg)

				msg = pickle.dumps(('update_deployment', subdeployments[worker]))
				self.send_receive_worker(worker, msg)
		except AttributeError:
			pass


		# self.rb_backups = {}
		# for ug in self.ug_perfs:
		# 	for popp1 in self.ug_perfs[ug]:
		# 		for popp2 in self.ug_perfs[ug]:
		# 			if popp1 == popp2: continue
		# 			# support popp2 provides to popp1
		# 			# if popp1 fails, popp2 should be really close to the latency
		# 			if self.ug_perfs[ug][popp2] < self.ug_perfs[ug][popp1]:
		# 				backup = 100
		# 			else:
		# 				# should multiply this by probability of user using this ingress anyway
		# 				delta = self.ug_perfs[ug][popp2] - self.ug_perfs[ug][popp1]
		# 				if delta > 50:
		# 					backup = 1
		# 				elif delta > 30:
		# 					backup = 10
		# 				elif delta > 10:
		# 					backup = 30
		# 				elif delta > 5:
		# 					backup = 50
		# 				else:
		# 					backup = 100
		# 			popp1i = self.popp_to_ind[popp1]
		# 			popp2i = self.popp_to_ind[popp2]
		# 			try:
		# 				self.rb_backups[ug,popp1i][popp2i] = backup
		# 			except KeyError:
		# 				self.rb_backups[ug,popp1i] = {popp2i: backup}


		#### TMP 
		self.all_rb_calls_results = {poppi:[] for poppi in range(self.n_popps)}
		self.all_lb_calls_results = []

		self.clear_caches()
		self.calculate_user_latency_by_peer()

	def calculate_user_latency_by_peer(self):
		"""
			Calculate latency from each user to each peer. In practice, you would measure these latencies
			once using Anyopt-type techniques.
			We just simulate latencies randomly (or however).
			This function turns ug -> popp -> lat into 3D array of popp, ug, prefixes for ease of use
		"""
		# try:
		# 	self.worker_i
		# except AttributeError:
		# 	# Only needed for worker bees
		# 	return
		self.mlbs = np.zeros((self.n_popp, self.n_ug))
		total_vol = np.sum(self.whole_deployment_ug_vols)
		for ug in self.ugs:
			ugi = self.ug_to_ind[ug]
			for popp in self.popps:
				popp_i = self.popp_to_ind[popp]
				## benefit per user is -1 * latency * volume fraction
				## we multiply by volume fraction so that later we can just calculate the sum and
				## have that be the correct average benefit
				weight = self.ug_vols[ugi] / total_vol
				self.mlbs[popp_i,ugi] = -1 * self.ug_perfs[ug].get(popp, NO_ROUTE_LATENCY) * weight
					
		# same for each prefix (ease of calculation later)
		self.best_latency_benefits = np.max(self.mlbs,axis=0)
		self.measured_latency_benefits = np.tile(np.expand_dims(self.mlbs, axis=1), 
			(1, self.n_prefixes, 1))

	def l1_norm(self, a):
		return np.sum(np.abs(a).flatten())

	def prefix_cost(self, a):
		c_pref = 1
		has_pref = (np.sum(a,axis=0) > 0).astype(np.int32)
		# cost for different prefs likely not different
		cost_prefs = np.sum(has_pref) * c_pref

		return cost_prefs

	def get_ground_truth_latency_benefit(self, a, **kwargs):
		### Measures actual latency benefit as if we were to advertise 'a'
		a_effective = threshold_a(a)
		
		user_latencies = self.get_ground_truth_user_latencies(a_effective,**kwargs)

		ugs = kwargs.get('ugs',self.ugs)
		benefit = self.benefit_from_user_latencies(user_latencies,ugs)
		return benefit

	def get_max_painter_benefit(self):
		max_improve_over_anycast = 0
		for ug in self.ugs:
			best_user_latency = np.min(list(self.ug_perfs[ug].values()))
			max_improve_over_anycast += (self.ug_anycast_perfs[ug] - best_user_latency) * self.ug_to_vol[ug]
		max_improve_over_anycast = max_improve_over_anycast / np.sum(self.ug_vols)
		return max_improve_over_anycast

	def get_max_overall_benefit(self):
		## benefit is between MIN_LATENCY_FOR_EVERYONE and NO_ROUTE_FOR_EVERYONE
		## scaled to be minimized is -1*NO_ROUTE_FOR_EVERYONE -> -1 * MIN_LATENCY_FOR_EVERYONE
		## made positive is 0 -> -1 * (MIN_LATENCY_FOR_EVERYONE - NO_ROUTE_FOR_EVERYONE)

		normalized_best_overall_benefit = 0
		for ug in self.ugs:
			best_user_latency = np.min(list(self.ug_perfs[ug].values()))
			normalized_best_overall_benefit += (-1 * (best_user_latency - NO_ROUTE_LATENCY) * self.ug_to_vol[ug])
		normalized_best_overall_benefit /= np.sum(self.ug_vols)
		return normalized_best_overall_benefit

	def get_normalized_benefit(self, a, **kwargs):
		a_effective = threshold_a(a)
		user_latencies = self.get_ground_truth_user_latencies(a_effective, **kwargs)
		print([(ug,user_latencies[self.ug_to_ind[ug]]) for ug in self.ugs])
		normalized_benefit = 0
		for ug in self.ugs:
			user_latency = user_latencies[self.ug_to_ind[ug]]
			normalized_benefit += (-1 * (user_latency - NO_ROUTE_LATENCY) * self.ug_to_vol[ug])
		normalized_benefit /= np.sum(self.ug_vols)
		return normalized_benefit

	def summarize_user_latencies(self, a):
		### Summarizes which users are getting good latency and bad latency, peers that would benefit them

		user_latencies = self.get_ground_truth_user_latencies(a)
		cum_by_peer = {}
		help_ug_by_peer = {}
		for ug in self.ug_perfs:
			these_perfs = list(sorted(self.ug_perfs[ug].items(), key = lambda el : el[1]))
			best_peer, best_perf = these_perfs[0]
			current_perf = user_latencies[self.ug_to_ind[ug]]
			if best_perf < current_perf:
				# print("UG {} would see {} ms improvement through {}".format(
				# 	self.ug_to_ind[ug], current_perf - best_perf, self.popp_to_ind[best_peer]))
				try:
					cum_by_peer[self.popp_to_ind[best_peer]] += (current_perf - best_perf)
					help_ug_by_peer[self.popp_to_ind[best_peer]].append(self.ug_to_ind[ug])
				except KeyError:
					cum_by_peer[self.popp_to_ind[best_peer]] = current_perf - best_perf
					help_ug_by_peer[self.popp_to_ind[best_peer]] = [self.ug_to_ind[ug]]

		sorted_cum_by_peer = list(sorted(cum_by_peer.items(), key = lambda el : -1 * el[1]))
		print(sorted_cum_by_peer)
		best_overall_peers = sorted_cum_by_peer[0:3]
		for bop,diff in best_overall_peers:
			print("{} -- {} ms, UGs {} would see corresponding lats {}".format(bop,diff,help_ug_by_peer[bop],
				[round(self.ug_perfs[self.ugs[ui]][self.popps[bop]],2) for ui in help_ug_by_peer[bop]]))
			print([round(el,2) for el in self.recent_grads[bop,:]])

	def get_ground_truth_user_latencies(self, a, **kwargs):
		#### Measures actual user latencies as if we were to advertise 'a'
		user_latencies, ug_ingress_decisions = self.calculate_user_choice(a)
		
		if self.with_capacity:
			link_volumes = np.zeros(self.n_popp)
			ingress_to_users = {}
			for ugi, ingress_i in ug_ingress_decisions.items():
				if ingress_i is None: continue
				link_volumes[ingress_i] += self.ug_vols[ugi]
				try:
					ingress_to_users[ingress_i].append(ugi)
				except KeyError:
					ingress_to_users[ingress_i] = [ugi]
			cap_violations = link_volumes > self.link_capacities_arr

			if kwargs.get('store_metrics'):
				self.metrics['link_utilizations'].append(link_volumes / self.link_capacities_arr)
			# poppi = self.popp_to_ind[('atlanta','3257')]
			# print("PoPP {} ({}) summary".format(self.popps[poppi], poppi))
			# print("Users {} using this ingress".format(ingress_to_users.get(poppi)))

			if (self.verbose or kwargs.get('overloadverb')) and len(np.where(cap_violations)[0]) > 0:
				# print("LV: {}, LC: {}".format(link_volumes, self.link_capacities_arr))
				for poppi in np.where(cap_violations)[0]:
					poppi_failed = kwargs.get('failing', 'none')
					if poppi_failed != 'none':
						poppi_failed = self.popp_to_ind[poppi_failed]
					# print("PoPP {} ({}) inundated when failing {}, severity {}".format(self.popps[poppi], poppi,
					# 	poppi_failed, link_volumes[poppi] - self.link_capacities_arr[poppi]))
					# print("Users {} now using this ingress".format(ingress_to_users[poppi]))
					users_str = "-".join([str(el) for el in ingress_to_users[poppi]])
					self.log("link_fail_report,{},{},{},{},{},{}\n".format(
						self.iter,poppi,poppi_failed,self.link_capacities[poppi],
						link_volumes[poppi],users_str))
			for cap_violation in np.where(cap_violations)[0]:
				for ugi in ingress_to_users[cap_violation]:
					user_latencies[ugi] = NO_ROUTE_LATENCY
		# if self.verbose and len(np.where(cap_violations)[0]) > 0:
		# 	print([round(el,2) for el in user_latencies])
		return user_latencies

	def benefit_from_user_latencies(self, user_latencies, ugs):
		# sum of the benefits, simple model for benefits is 1 / latency
		# allow for calculation over a subset of UGs
		user_benefits = -1 * user_latencies
		# average user benefit -- important that this function is not affected by the number of user groups
		these_inds = np.array([self.ug_to_ind[ug] for ug in ugs])
		these_vols = self.ug_vols[these_inds]
		these_benefits = user_benefits[these_inds]
		return np.sum(these_benefits * these_vols) / np.sum(these_vols)

	def get_ground_truth_resilience_benefit(self, a, **kwargs):
		#### NOTE -- this calculation does not factor in capacity, since that would take a long time
		benefit = 0
		tmp = np.ones(a.shape)
		a = threshold_a(a)
		pre_user_latencies, pre_ug_catchments = self.calculate_user_choice(a, **kwargs)
		total_vol = np.sum(self.ug_vols)
		for popp in self.popps:
			these_uis = self.poppi_to_ui[self.popp_to_ind[popp]]
			these_ugs = [self.ugs[ui] for ui in these_uis]
			if len(these_ugs) == 0: 
				continue
			tmp[self.popp_to_ind[popp],:] = 0

			user_latencies, ug_catchments = self.calculate_user_choice(copy.copy(a * tmp),
				ugs=these_ugs, failing=popp)

			## benefit is user latency under failure - user latency under no failure
			# I might want this to be compared to best possible, but oh well
			these_inds = np.array([self.ug_to_ind[ug] for ug in these_ugs])
			these_vols = self.ug_vols[these_inds]
			# these_users_resilience = -1 * np.sum((user_latencies - pre_user_latencies[these_inds]) *\
			# 	these_vols) / total_vol #/ self.n_popps
			these_users_resilience = -1 * np.sum((user_latencies) *\
				these_vols) / total_vol #/ self.n_popps
			benefit += these_users_resilience

			other_inds = np.array(get_difference(range(self.n_ug), these_inds))
			if len(other_inds) > 0:
				other_vols = self.ug_vols[other_inds]
				other_users_resilience = -1 * np.sum(pre_user_latencies[other_inds] *\
					other_vols) / total_vol
				benefit += other_users_resilience


			tmp[self.popp_to_ind[popp],:] = 1
		return benefit

	def calculate_user_choice(self, a, **kwargs):
		"""Calculates UG -> popp assuming they go to their best performing popp."""
		ugs = kwargs.get('ugs', self.ugs)
		a = threshold_a(a)
		user_latencies = NO_ROUTE_LATENCY * np.ones((len(ugs)))
		routed_through_ingress, _ = self.calculate_ground_truth_ingress(a, ugs=ugs)
		ug_ingress_decisions = {ugi:None for ugi in range(self.n_ug)}
		for prefix_i in range(a.shape[1]):
			for ugi,ug in enumerate(kwargs.get('ugs', ugs)):
				routed_ingress = routed_through_ingress[prefix_i].get(ug)
				if routed_ingress is None:
					latency = NO_ROUTE_LATENCY
				else:
					latency = self.ug_perfs[ug][self.popps[routed_ingress]]
				if latency < user_latencies[ugi]: 
					ug_ingress_decisions[ugi] = routed_ingress
					user_latencies[ugi] = latency
		if kwargs.get('get_ug_catchments', False):
			self.update_ug_ingress_decisions(ug_ingress_decisions)
		if kwargs.get('verb'):
			for ui,poppi in ug_ingress_decisions.items():
				self.log("benefit_estimate,{},{},{},{}\n".format(
					self.iter,ui,poppi,self.mlbs[poppi,ui]))
		return user_latencies, ug_ingress_decisions

	def log(self,s):
		self.log_ptr = open(os.path.join(CACHE_DIR, 'main_thread_log.txt'),'a')
		self.log_ptr.write(s)
		self.log_ptr.close()

	def update_ug_ingress_decisions(self, ug_catchments):
		"""For updating variables that only change with ug catchments."""
		# pairing popps with popps
		# popp -> users -> benefit
		# but also users -> popps
		# support (popp1 -> popp2) is sum_over_users( potential backup popp1 can provide to popp2 for that user)
		# backup is ~ how good that option is compared to user's other options
		# lots of backup for very good relative option
		self.ug_catchments = ug_catchments

		try:
			self.rb_backups # init if needed
		except AttributeError:
			### Resilience benefit backup pre-calcs
			self.rb_backups = np.zeros((self.n_ug,self.n_popps,self.n_popps),dtype=np.ushort)
			for ug in tqdm.tqdm(self.ug_perfs,desc="Calculating RB backups"):
				for popp1 in self.ug_perfs[ug]:
					for popp2 in self.ug_perfs[ug]:
						if popp1 == popp2: continue
						# support popp2 provides to popp1
						# if popp1 fails, popp2 should be really close to the latency
						if self.ug_perfs[ug][popp2] < self.ug_perfs[ug][popp1]:
							backup = 100
						else:
							# should multiply this by probability of user using this ingress anyway
							delta = self.ug_perfs[ug][popp2] - self.ug_perfs[ug][popp1]
							if delta > 50:
								backup = 1
							elif delta > 30:
								backup = 10
							elif delta > 10:
								backup = 30
							elif delta > 5:
								backup = 50
							else:
								backup = 100
						popp1i = self.popp_to_ind[popp1]
						popp2i = self.popp_to_ind[popp2]
						self.rb_backups[self.ug_to_ind[ug],popp1i,popp2i] = backup

		poppi_to_ugi = {}
		for ugi,poppi in self.ug_catchments.items():
			if poppi is None: continue
			try:
				poppi_to_ugi[poppi].append(ugi)
			except KeyError:
				poppi_to_ugi[poppi] = [ugi]

		# self.rb_popp_support = np.zeros((self.n_popps, self.n_popps))
		# for popp1i in poppi_to_ugi:
		# 	for ugi in poppi_to_ugi[popp1i]:
		# 		ug = self.ugs[ugi]
		# 		try:
		# 			self.rb_backups[ug,popp1i]
		# 		except KeyError:
		# 			# ug has no route, or ug has a route to only 1 popp 
		# 			continue
		# 		for popp2i,bck in self.rb_backups[ug,popp1i].items():
		# 			self.rb_popp_support[popp1i,popp2i] += bck

		self.rb_popp_support = np.zeros((self.n_popps, self.n_popps))
		for popp1i in poppi_to_ugi:
			for ugi in poppi_to_ugi[popp1i]:
				for popp2i in range(self.n_popps):
					self.rb_popp_support[popp1i,popp2i] += np.sum(self.rb_backups[ugi,popp1i,popp2i])

		self.rb_popp_support += .0001
		self.popp_sample_probs = (self.rb_popp_support.T / np.sum(self.rb_popp_support, axis=1)).T

	def calculate_ground_truth_ingress(self, a, **kwargs):
		### Returns routed_through ingress -> prefix -> ug -> popp_i
		## and actives prefix -> [active popp indices]

		### Somewhat efficient implementation using matrix logic
		routed_through_ingress = {} # prefix -> UG -> ingress index
		actives = {} # prefix -> indices where we adv
		ugs = kwargs.get('ugs', self.ugs)
		ug_inds = np.array([self.ug_to_ind[ug] for ug in ugs])
		n_ug = len(ugs)

		a = threshold_a(a)

		for prefix_i in range(self.n_prefixes):
			cache_rep = tuple(np.where(a[:,prefix_i])[0].flatten())
			try:
				routed_through_ingress[prefix_i], actives[prefix_i] = self.calc_cache.all_caches['gti'][cache_rep]
				continue
			except KeyError:
				pass
			this_actives = np.where(a[:,prefix_i])[0]
			actives[prefix_i] = this_actives
			this_routed_through_ingress = {}
			if np.sum(a[:,prefix_i]) == 0:
				routed_through_ingress[prefix_i] = this_routed_through_ingress
				self.calc_cache.all_caches['gti'][cache_rep] = (this_routed_through_ingress,this_actives)
				continue
			active_popp_indicator = np.tile(np.expand_dims(a[:,prefix_i],axis=1), (1,n_ug))
			active_popp_ug_indicator = self.popp_by_ug_indicator[:,ug_inds] * active_popp_indicator
			best_available_options = np.argmax(active_popp_ug_indicator,axis=0)
			for ui, bao in zip(range(n_ug), best_available_options):
				if active_popp_ug_indicator[bao,ui] == 0: continue # no route
				this_routed_through_ingress[ugs[ui]] = bao
			routed_through_ingress[prefix_i] = this_routed_through_ingress
			if len(ugs) == self.n_ug:
				self.calc_cache.all_caches['gti'][cache_rep] = (this_routed_through_ingress,this_actives)
		return routed_through_ingress, actives

	def enforce_measured_prefs(self, routed_through_ingress, actives):
		for ui, ug in enumerate(self.ugs):
			for prefix_i in range(self.n_prefixes):
				routed_ingress = routed_through_ingress[prefix_i].get(ug)
				if routed_ingress is None:
					# no path, nothing learned
					continue
				else:
					# observation indidates for this user group what the winning ingress was, 
					# and what the active&available ingresses were
					other_available = get_intersection(actives[prefix_i], self.ingress_priority_inds[ug])
					other_available = get_difference(other_available, [routed_ingress])
					routed_ingress_obj = self.measured_prefs[ui].get(routed_ingress)
					for beaten_ingress in other_available:
						beaten_ingress_obj = self.measured_prefs[ui].get(beaten_ingress)
						beaten_ingress_obj.add_parent(routed_ingress_obj)
						routed_ingress_obj.add_child(beaten_ingress_obj)
						self.parent_tracker[ui,beaten_ingress,routed_ingress] = True
						try:
							self.calc_cache.all_caches['parents_on'][ui][beaten_ingress,routed_ingress] = None
						except KeyError:
							self.calc_cache.all_caches['parents_on'][ui] = {(beaten_ingress,routed_ingress): None}
		
		## Update workers about new parent tracker information
		for worker, worker_socket in self.worker_manager.worker_sockets.items():
			this_deployment_ugs = self.worker_manager.worker_to_deployments[worker]['ugs']
			uis_of_interest = list([self.ug_to_ind[ug] for ug in this_deployment_ugs])
			sub_cache = {}
			for worker_ui, global_ui in enumerate(uis_of_interest):
				try:
					sub_cache[worker_ui] = self.calc_cache.all_caches['parents_on'][global_ui] 
				except KeyError:
					pass
			msg = pickle.dumps(('update_parent_tracker', sub_cache))
			worker_socket.send(msg)
			worker_socket.recv()

	def measure_ingresses(self, a):
		"""Between rounds, measure ingresses from users to deployment given advertisement a."""
		### i.e., this is an actual advertisement measurement, we should aim to limit these :)
		try:
			self.measured
		except AttributeError:
			self.measured = {}
		try:
			self.measured[tuple(a.flatten())]
			return
		except KeyError:
			pass

		## Reset benefit calculations cache since we now have more info
		self.clear_new_measurement_caches()

		self.path_measures += 1
		self.calculate_user_choice(a,get_ug_catchments=True)

		a = threshold_a(a)
		routed_through_ingress, actives = self.calculate_ground_truth_ingress(a,verb=True)

		self.enforce_measured_prefs(routed_through_ingress, actives)
		self.measured[tuple(a.flatten())] = None

	def get_naive_range(self, a, ugs=None):
		overall_best = 0
		overall_worst = 0
		overall_average = 0

		if ugs is None:
			ugs = list(self.ug_to_vol)

		total_ug_vol = sum(list(self.ug_to_vol[ug] for ug in ugs))

		for ug in ugs:
			perfs = self.ug_perfs[ug]
			worst_case_perf = MAX_LATENCY
			best_case_perf = MAX_LATENCY
			has_perf = [self.popp_to_ind[popp] for popp in perfs]
			for prefix_i in range(self.n_prefixes):
				is_on = np.where(a[:,prefix_i])[0]
				is_on_and_route = get_intersection(is_on, has_perf)
				if len(is_on_and_route) > 0:
					these_perfs = [perfs[self.popps[popp]] for popp in is_on_and_route]
					# at the best case, user gets routed to lowest latency "on" probe
					best_case_perf = np.minimum(np.min(these_perfs), best_case_perf)
					# at the worst case, every max hits and the user picks the best of those maxes
					worst_case_perf = np.minimum(np.max(these_perfs), worst_case_perf)

			overall_best += ((-1 * best_case_perf) * self.ug_to_vol[ug])
			overall_worst += ((-1 * worst_case_perf) * self.ug_to_vol[ug])
			overall_average += ((-1 * (best_case_perf + worst_case_perf) / 2) * self.ug_to_vol[ug])

		return {
			'best': overall_best / total_ug_vol,
			'worst': overall_worst / total_ug_vol,
			'avg': overall_average / total_ug_vol,
		}
			

	def actual_nonconvex_objective(self, a):
		# Don't approximate the L0 norm with anything
		# Use actual latencies as if we were to really measure all the paths		

		c_peering = 1
		has_peering = (np.sum(a,axis=1) > 0).astype(np.int32)
		# cost for different peerings may be different
		cost_peerings = np.sum(np.dot(has_peering,c_peering*np.ones(has_peering.shape)))
		
		# cost for different prefs likely not different
		cost_prefs = self.prefix_cost(a)
		norm_penalty = cost_peerings + cost_prefs
		latency_benefit = self.get_ground_truth_latency_benefit(a)
		if self.gamma > 0:
			resilience_benefit = self.get_ground_truth_resilience_benefit(a)
		else:
			resilience_benefit = 0

		return self.lambduh * norm_penalty - (latency_benefit + self.gamma * resilience_benefit)

	def measured_objective(self, a, **kwargs):
		## Approximate L0 norm with whatever approximation we're using
		## Use actual latencies as if we were to really measure all the paths		
		## (Here we actually execute the advertisement)
		norm_penalty = self.advertisement_cost(a)
		latency_benefit = self.get_ground_truth_latency_benefit(a, **kwargs)
		if self.gamma > 0 and kwargs.get('use_resilience', True):
			resilience_benefit = self.get_ground_truth_resilience_benefit(a, **kwargs)
		else:
			resilience_benefit = 0
		if self.gamma <= 1:
			benefit = latency_benefit + self.gamma * resilience_benefit
		else:
			benefit = 1 / self.gamma * latency_benefit + resilience_benefit

		obj = self.lambduh * norm_penalty - benefit

		if self.verbose:
			print("Actual: NP: {}, LB: {}, RB: {}, Total: {}, Obj: {}".format(
				round(norm_penalty,2),round(latency_benefit,2),
				round(resilience_benefit,2),round(benefit,2), round(obj,2)))
		
		return obj


