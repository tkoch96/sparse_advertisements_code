import numpy as np, time, tqdm
np.setbufsize(262144*8)
from constants import *
from helpers import *
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
		self.epsilon = .00005 # change in objective less than this -> stop
		self.max_n_iter = 150 # maximum number of learning iterations
		self.lambduh = lambduh # sparsity cost
		self.gamma = gamma # resilience cost
		self.with_capacity = kwargs.get('with_capacity', False)
		self.n_prefixes = kwargs.get('n_prefixes')

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
			self.ug_perfs[self.ugs[ui]] if popp != 'anycast'} for ui in range(self.n_ug)}
		try:
			msg = pickle.dumps(['reset_cache', ()])
			self.send_receive_workers(msg)
		except AttributeError:
			# not initialized yet
			pass

	def clear_new_measurement_caches(self):
		self.calc_cache.clear_new_measurement_caches()
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
		self.ugs = list(deployment['ugs'])
		self.n_ug = len(self.ugs)
		self.whole_deployment_ugs = deployment['whole_deployment_ugs']
		self.whole_deployment_n_ug = len(self.whole_deployment_ugs)
		self.ug_to_ind = {ug:i for i,ug in enumerate(self.ugs)}
		self.whole_deployment_ug_to_ind = {ug:i for i,ug in enumerate(self.whole_deployment_ugs)}
		self.ug_perfs = deployment['ug_perfs']
		# Shape of the variables
		self.popps = list(set(deployment['popps']))
		self.popp_to_ind = {k:i for i,k in enumerate(self.popps)}
		self.provider_popps = deployment['provider_popps']
		self.n_provider_popps = len(self.provider_popps) # number of provider PoPps
		self.n_popp = len(get_difference(self.popps,['anycast']))
		if self.n_prefixes is None:
			self.n_prefixes = np.maximum(2,self.n_popp // 3)
		self.n_providers = deployment['n_providers'] # number of provider ASes
		self.ground_truth_ingress_priorities = deployment['ingress_priorities']

		self.metro_loc = deployment['metro_loc']
		self.pop_to_loc = deployment['pop_to_loc']
		self.link_capacities_by_popp = deployment['link_capacities']
		self.link_capacities = {self.popp_to_ind[popp]: deployment['link_capacities'][popp] for popp in self.popps}
		self.link_capacities_arr = np.zeros(self.n_popp)
		for poppi, cap in self.link_capacities.items():
			self.link_capacities_arr[poppi] = cap

		self.ingress_priority_inds = {ug:{self.popp_to_ind[popp]: \
			self.ground_truth_ingress_priorities[ug][popp] for popp in self.ug_perfs[ug] if popp != 'anycast'} \
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
		self.vol_x = np.linspace(min(all_vols),max(all_vols))

		# ingress,ug -> stores ~ how many times ingress wins for a ug
		self.n_wins = {ug:{i:0 for i in range(self.n_popp)} for ug in self.ugs}

		self.popp_by_ug_indicator = np.zeros((self.n_popp, self.n_ug))
		for ui in range(self.n_ug):
			for popp in self.ug_perfs[self.ugs[ui]]:
				if popp == 'anycast': continue
				self.popp_by_ug_indicator[self.popp_to_ind[popp], ui] = self.n_popp + 1 - self.ingress_priority_inds[self.ugs[ui]][self.popp_to_ind[popp]]

		## only sub-workers should spawn these arrays if they get too big
		max_entries = 10000e6
		# ijk'th entry of ingress probabilities is probability that user k ingresses over popp i for prefix j
		if self.n_popp * self.n_prefixes * self.n_ug < max_entries:
			self.ingress_probabilities = np.zeros((self.n_popp, self.n_prefixes, self.n_ug))
		# ijk'th entry of parent tracker indicates whether, for ug i, popp j getes beaten by popp k
		if self.n_popp**2 * self.n_ug < max_entries*64: # more lenient since dtype is bool
			self.parent_tracker = np.zeros((self.n_ug, self.n_popp, self.n_popp), dtype=bool)
		self.popp_by_ug_indicator_no_rank = np.zeros((self.n_popp, self.n_ug), dtype=bool)
		for ui in range(self.n_ug):
			for popp in self.ug_perfs[self.ugs[ui]]:
				if popp == 'anycast': continue
				self.popp_by_ug_indicator_no_rank[self.popp_to_ind[popp],ui] = True

		try:
			n_workers = self.get_n_workers()
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
			# not initialized yet, or a worker bee
			pass
		self.clear_caches()
		self.calculate_user_latency_by_peer()

	def calculate_user_latency_by_peer(self):
		"""
			Calculate latency from each user to each peer. In practice, you would measure these latencies
			once using Anyopt-type techniques.
			We just simulate latencies randomly (or however).
			This function turns ug -> popp -> lat into 3D array of popp, ug, prefixes for ease of use
		"""
		self.measured_latencies = np.zeros((self.n_popp, len(self.ugs)))
		self.measured_latency_benefits = np.zeros((self.n_popp, len(self.ugs)))
		total_vol = np.sum(self.whole_deployment_ug_vols)
		for ug in self.ugs:
			ugi = self.ug_to_ind[ug]
			for popp in self.popps:
				popp_i = self.popp_to_ind[popp]
				## benefit per user is -1 * latency * volume fraction
				## we multiply by volume fraction so that later we can just calculate the sum and
				## have that be the correct average benefit
				self.measured_latencies[popp_i,ugi] = self.ug_perfs[ug].get(popp, NO_ROUTE_LATENCY)
				weight = self.ug_vols[ugi] / total_vol
				self.measured_latency_benefits[popp_i,ugi] = -1 * self.ug_perfs[ug].get(popp, NO_ROUTE_LATENCY) * weight
					
		# same for each prefix (ease of calculation later)
		self.measured_latencies = np.tile(np.expand_dims(self.measured_latencies, axis=1), 
			(1, self.n_prefixes, 1))
		self.measured_latency_benefits = np.tile(np.expand_dims(self.measured_latency_benefits, axis=1), 
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
		benefit = self.benefit_from_user_latencies(user_latencies)
		return benefit

	def get_ug_perfs_with_anycast(self):
		ugperf_copy = copy.deepcopy(self.ug_perfs)
		for ug in self.ug_perfs:
			anycast_ingress = [popp for popp,priority in self.ground_truth_ingress_priorities[ug].items() \
				if priority == 0][0]
			ugperf_copy[ug]['anycast'] = ugperf_copy[ug][anycast_ingress]
		return ugperf_copy

	def get_max_painter_benefit(self):
		max_improve_over_anycast = 0
		ug_anycast_perfs = self.get_ug_perfs_with_anycast()
		for ug in self.ugs:
			best_user_latency = np.min(list(self.ug_perfs[ug].values()))
			max_improve_over_anycast += (ug_anycast_perfs[ug]['anycast'] - best_user_latency) * self.ug_to_vol[ug]
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
				[self.ug_perfs[self.ugs[ui]][self.popps[bop]] for ui in help_ug_by_peer[bop]]))
			print(self.recent_grads[bop,:])

	def get_ground_truth_user_latencies(self, a, **kwargs):
		#### Measures actual user latencies as if we were to advertise 'a'
		user_latencies = NO_ROUTE_LATENCY * np.ones((len(self.ugs)))
		routed_through_ingress, _ = self.calculate_ground_truth_ingress(a)
		ug_ingress_decisions = {ugi:None for ugi in range(self.n_ug)}
		for prefix_i in range(a.shape[1]):
			for ugi,ug in enumerate(self.ugs):
				routed_ingress = routed_through_ingress[prefix_i].get(ug)
				if routed_ingress is None:
					latency = NO_ROUTE_LATENCY
				else:
					latency = self.ug_perfs[ug][self.popps[routed_ingress]]
				if latency < user_latencies[ugi]: 
					ug_ingress_decisions[ugi] = routed_ingress
					user_latencies[ugi] = latency
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
			if kwargs.get('verb'):
				print("LV: {}, LC: {}".format(link_volumes, self.link_capacities_arr))
				print(np.where(cap_violations))
			for cap_violation in np.where(cap_violations)[0]:
				for ugi in ingress_to_users[cap_violation]:
					user_latencies[ugi] = NO_ROUTE_LATENCY
		# if kwargs.get('verb'):
		# 	print(user_latencies)
		return user_latencies

	def benefit_from_user_latencies(self, user_latencies):
		# sum of the benefits, simple model for benefits is 1 / latency
		user_benefits = -1 * user_latencies
		# average user benefit -- important that this function is not affected by the number of user groups
		return np.sum(user_benefits * self.ug_vols) / np.sum(self.ug_vols)

	def get_ground_truth_resilience_benefit(self, a):
		benefit = 0
		tmp = np.ones(a.shape)
		for popp in self.popps:
			tmp[self.popp_to_ind[popp],:] = 0
			benefit += self.get_ground_truth_latency_benefit(a * tmp)
			tmp[self.popp_to_ind[popp],:] = 1
		return benefit / len(self.popps)

	def calculate_ground_truth_ingress(self, a):
		### Returns routed_through ingress -> prefix -> ug -> popp_i
		## and actives prefix -> [active popp indices]

		### Somewhat efficient implementation using matrix logic
		routed_through_ingress = {} # prefix -> UG -> ingress index
		actives = {} # prefix -> indices where we adv
		for prefix_i in range(self.n_prefixes):
			try:
				routed_through_ingress[prefix_i], actives[prefix_i] = self.calc_cache.all_caches['gti'][tuple(a[:,prefix_i].flatten())]
				continue
			except KeyError:
				pass
			this_actives = np.where(a[:,prefix_i] == 1)[0]
			actives[prefix_i] = this_actives
			this_routed_through_ingress = {}
			if np.sum(a[:,prefix_i]) == 0:
				routed_through_ingress[prefix_i] = this_routed_through_ingress
				self.calc_cache.all_caches['gti'][tuple(a[:,prefix_i].flatten())] = (this_routed_through_ingress,this_actives)
				continue
			active_popp_indicator = np.tile(np.expand_dims(a[:,prefix_i],axis=1), (1,self.n_ug))
			active_popp_ug_indicator = self.popp_by_ug_indicator * active_popp_indicator
			best_available_options = np.argmax(active_popp_ug_indicator,axis=0)
			for ui, bao in zip(range(self.n_ug), best_available_options):
				if active_popp_ug_indicator[bao,ui] == 0: continue # no route
				this_routed_through_ingress[self.ugs[ui]] = bao
			routed_through_ingress[prefix_i] = this_routed_through_ingress
			self.calc_cache.all_caches['gti'][tuple(a[:,prefix_i].flatten())] = (this_routed_through_ingress,this_actives)
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

		a = threshold_a(a)
		routed_through_ingress, actives = self.calculate_ground_truth_ingress(a)

		self.enforce_measured_prefs(routed_through_ingress, actives)
		# print("Measuring : \n{}".format(a))
		self.measured[tuple(a.flatten())] = None

	def get_naive_range(self, a):
		overall_best = 0
		overall_worst = 0
		overall_average = 0

		total_ug_vol = sum(list(self.ug_to_vol.values()))

		for ug, perfs in self.ug_perfs.items():
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
			resilience_beneift = 0

		return self.lambduh * norm_penalty - (latency_benefit + self.gamma * resilience_benefit)

	def measured_objective(self, a, **kwargs):
		## Approximate L0 norm with whatever approximation we're using
		## Use actual latencies as if we were to really measure all the paths		
		## (Here we actually execute the advertisement)
		norm_penalty = self.advertisement_cost(a)
		latency_benefit = self.get_ground_truth_latency_benefit(a, **kwargs)
		if self.gamma > 0:
			resilience_benefit = self.get_ground_truth_resilience_benefit(a)
		else:
			resilience_benefit = 0
		if self.verbose:
			print("Actual: NP: {}, LB: {}, RB: {}".format(norm_penalty,latency_benefit,resilience_benefit))
		return self.lambduh * norm_penalty - (latency_benefit + self.gamma * resilience_benefit)


