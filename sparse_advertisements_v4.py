
class Optimal_Adv_Wrapper_v2:
	### Wrapper class for all solutions to finding optimal advertisements
	def __init__(self, deployment, lambduh=1.0, verbose=True, gamma=0, **kwargs):
		self.verbose = verbose
		self.epsilon = .05 # change in objective less than this -> stop
		self.max_n_iter = 100 # maximum number of learning iterations
		self.lambduh = lambduh # sparsity cost
		self.gamma = gamma # resilience cost
		self.n_prefixes = kwargs.get('n_prefixes')

		self.update_deployment(deployment)	

	def output_deployment(self):
		deployment = {
			'ugs': self.ugs,
			'ug_perfs': self.ug_perfs,
			'ug_to_vol': self.ug_to_vol,
			'ug_anycast_perfs': self.ug_anycast_perfs,
			'whole_deployment_ugs': self.whole_deployment_ugs,
			'whole_deployment_ug_to_vol': self.whole_deployment_ug_to_vol,
			'popps': self.popps,
			'metro_loc': self.metro_loc,
			'pop_to_loc': self.pop_to_loc,
			'n_providers': self.n_providers,
			'provider_popps': self.provider_popps,
			'ingress_priorities': self.ground_truth_ingress_priorities,
			'link_capacities': self.link_capacities_by_popp,
		}
		return copy.deepcopy(deployment)

	def update_deployment(self, deployment, **kwargs):

		### Get rid of ugs with only one pop
		n_pop_by_ug = {ug:{} for ug in deployment['ug_perfs']}
		for ug in deployment['ug_perfs']:
			for pop,peer in deployment['ug_perfs'][ug]:
				n_pop_by_ug[ug][pop] = None
		ug_to_del = list(ug for ug,v in n_pop_by_ug.items() if len(v) <= 1)
		for ug in ug_to_del:
			for k in ['ugs', 'whole_deployment_ugs']:
				deployment[k] = get_difference(deployment[k], ug_to_del)
			for k in ['ug_perfs', 'ug_to_vol', 'ug_anycast_perfs', 'ingress_priorities', 'whole_deployment_ug_to_vol']:
				deployment[k] = {ug:v for ug,v in deployment[k].items() if ug not in ug_to_del}

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

	def get_ground_truth_user_latencies(self, a, mode='lp', **kwargs):
		#### Measures actual user latencies as if we were to advertise 'a'
		if mode == 'best':
			user_latencies, ug_ingress_decisions = self.calculate_user_choice(a, mode=mode,**kwargs)
		elif mode == 'lp':
			ret = self.solve_lp_with_failure_catch(threshold_a(a))
			if not ret['solved']:
				return self.get_ground_truth_user_latencies(a, mode='best', **kwargs)
			user_latencies = ret['lats_by_ug']
			ug_ingress_decisions = ret['paths_by_ug']
		else:
			raise ValueError("Unknown mode {}".format(mode))
		
		if self.with_capacity:
			link_volumes = np.zeros(self.n_popp)
			ingress_to_users = {}
			for ugi, ingressesvols in ug_ingress_decisions.items():
				if ingressesvols is None: continue
				for ingress_i,vol in ingressesvols:
					link_volumes[ingress_i] += vol
					try:
						ingress_to_users[ingress_i].append(ugi)
					except KeyError:
						ingress_to_users[ingress_i] = [ugi]
			cap_violations = link_volumes > self.link_capacities_arr

			if kwargs.get('store_metrics'):
				self.metrics['link_utilizations'].append(link_volumes / self.link_capacities_arr)

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
					try:
						self.log("link_fail_report,{},{},{},{},{},{}\n".format(
							self.iter,poppi,poppi_failed,self.link_capacities[poppi],
							link_volumes[poppi],users_str))
					except:
						continue
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
				ugs=these_ugs, failing=popp, mode='best')

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

	def calculate_user_choice(self, a, mode = 'lp', **kwargs):
		"""Calculates UG -> popp assuming they act according to decision mode 'mode'."""
		## returns user latencies (np array) and ug_ingress_decisions maps ugi to [(poppi, vol)]
		ugs = kwargs.get('ugs', self.ugs)
		if mode == 'best':
			# every user visits their best popp
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
						ug_ingress_decisions[ugi] = [(routed_ingress,1)]
						user_latencies[ugi] = latency
		elif mode == 'lp':
			ret = self.solve_lp_with_failure_catch(threshold_a(a), ugs=ugs)
			if not ret['solved']:
				## no valid capacity constraining solution, just go with 
				## each user chooses their best
				return self.calculate_user_choice(a, mode='best', **kwargs)
			user_latencies = ret['lats_by_ug']
			ug_ingress_decisions_tmp = ret['paths_by_ug']

			### Need to update this knowing that some ugs may be fixed !!
			## for now, just solving for anything and limiting to the ones I want
			ug_ingress_decisions = {self.ug_to_ind[ug]: ug_ingress_decisions_tmp[self.ug_to_ind[ug]] for ug in ugs}
			if len(ugs) != self.n_ug:
				ug_inds = np.array([self.ug_to_ind[ug] for ug in ugs])
				user_latencies = user_latencies[ug_inds]

		if kwargs.get('get_ug_catchments', False):
			self.update_ug_ingress_decisions(ug_ingress_decisions)
		if kwargs.get('verb'):
			for ui,poppipcts in ug_ingress_decisions.items():
				for poppi,pct in poppipcts:
					self.log("benefit_estimate,{},{},{},{},{}\n".format(
						self.iter,ui,poppi,pct,self.mlbs[poppi,ui]))

		return user_latencies, ug_ingress_decisions

	def log(self,s):
		self.log_ptr = open(os.path.join(CACHE_DIR, 'main_thread_log.txt'),'a')
		self.log_ptr.write(s)
		self.log_ptr.close()

	def calculate_ground_truth_ingress(self, a, **kwargs):
		### Returns routed_through_ingress: prefix -> ug -> popp_i
		## and actives prefix -> [active popp indices]

		### Somewhat efficient implementation using matrix logic
		routed_through_ingress = {} # prefix -> UG -> ingress index
		actives = {} # prefix -> indices where we adv
		ugs = kwargs.get('ugs', self.ugs)
		ug_inds = np.array([self.ug_to_ind[ug] for ug in ugs])
		n_ug = len(ugs)

		a = threshold_a(a)

		for prefix_i in range(a.shape[1]):
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
				## not a subset of user groups, cache result for general utility
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

class Sparse_Advertisement_Solver(Optimal_Adv_Wrapper_v2):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def stop_tracker(self, advs):
		delta_alpha = .7
		self.last_objective = self.obj
		self.obj = self.measured_objective(self.painter_advs_to_sparse_advs(advs),
			use_resilience=False, mode='best')
		self.rolling_delta = (1 - delta_alpha) * self.rolling_delta + delta_alpha * np.abs(self.obj - self.last_objective)
		self.stop = self.stopping_condition([self.iter, self.rolling_delta])

	def solve_sparse_lp(self, **kwargs):



	def solve_sparse(self):
		### Iteratively solves linear programs and learns from mistakes
		print("Initial greedy advertisement comuptation")
		advs = self.solve_sparse_lp(**kwargs)
		self.advs = advs
		
		save_verb = copy.copy(self.verbose)
		self.verbose = False

		self.obj = self.measured_objective(advs) # technically this is a measurement, uncounted
		self.stop = False
		self.rolling_delta = 10
		self.iter = 0
		self.path_measures = 0 
		self.clear_caches()
		ts = time.time()
		print("Starting painter computation")
		while not self.stop:
			# print("Measuring ingresses")
			## conduct measurement with this advertisement strategy
			self.measure_ingresses(advs)
			## recalc sparse decision with new information
			# print("Solving greedy allocation")
			advs = self.solve_sparse_lp(**kwargs)
			self.stop_tracker(advs)
			self.iter += 1

			print("Painter iteration {}, {}s per iter".format(self.iter,
				(time.time() - ts) / (self.iter+1)))

			self.advs = advs
			break

		self.verbose = save_verb





if __name__ == "__main__":
	sas = Sparse_Advertisement_Solver()
	sas.solve_sparse()