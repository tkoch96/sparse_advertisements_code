import numpy as np, time, tqdm, multiprocessing
np.setbufsize(262144*8)
np.set_printoptions(precision=5)
from constants import *
from helpers import *
from subprocess import call, check_output
from generic_objective import Generic_Objective
from worker_comms import Worker_Manager
from solve_lp_assignment import *
from scipy.sparse import csr_matrix, lil_matrix


class Optimal_Adv_Wrapper:
	### Wrapper class for all solutions to finding optimal advertisements
	def __init__(self, deployment, lambduh=1.0, verbose=True, gamma=0, **kwargs):
		self.solution_type = "None"
		self.verbose = verbose
		self.ts_loop = time.time()
		self.advertisement_cost = self.l1_norm
		self.epsilon = .005 # change in objective less than this -> stop
		self.lambduh = lambduh # sparsity cost (unused really)
		self.gamma = gamma # resilience cost
		self.with_capacity = kwargs.get('with_capacity', False)
		self.n_prefixes = kwargs.get('n_prefixes')

		self.generic_objective = Generic_Objective(self, kwargs.get('generic_objective', 'avg_latency'), **kwargs)

		self.calc_cache = Calc_Cache()

		self.update_deployment(deployment)	

		self.set_save_run_dir(**kwargs)

		self.compute_one_per_peering_solution()

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
		self.measured_prefs = {ug: {popp: Ing_Obj(popp) for popp in \
			self.ug_perfs[ug]} for ug in self.ugs}
		self.calc_cache.clear_all_caches()
		# self.linear_prog_soln_cache = {
		# 	'regular': {},
		# 	'failure_catch': {},
		# }
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
		kwa =  {
			'lambduh': self.lambduh, 
			'gamma': self.gamma, 
			'verbose': False,
			'n_prefixes': self.n_prefixes,
			'with_capacity': self.with_capacity,
			'save_run_dir': self.save_run_dir,
		}
		kwa['generic_objective'] = self.generic_objective.obj
		return kwa

	def adv_rep_to_adv(self, adv_rep):
		## Adv rep is a dictionary whose keys are popp,prefix pairs. Signifies which components are "on"

		n_prefixes = max(list([prefix_i for _,prefix_i in adv_rep])) + 1
		adv = np.zeros((self.n_popps, n_prefixes))
		for popp,prefix_i in adv_rep:
			if popp not in self.popps: continue
			adv[self.popp_to_ind[popp], prefix_i] = .55

		return adv

	def solve_lp_assignment(self, adv, **kwargs):
		self.enforce_loaded_rwmw()
		cache_rep = tuple(np.where(threshold_a(adv).flatten())[0])
		computing_best_lats = kwargs.get('computing_best_lats', False)
		try:
			if kwargs.get('verb') or kwargs.get('smallverb'):
				# print("Note, verbose is on so we're ignoring LP cache in woFC")
				raise KeyError
			ret =  self.linear_prog_soln_cache['regular'][cache_rep, computing_best_lats]
			return ret
		except KeyError:
			pass
		ret = solve_lp_assignment(self, adv, **kwargs)
		# ret = solve_lp_assignment(self, adv, **kwargs)
		
		self.linear_prog_soln_cache['regular'][cache_rep, computing_best_lats] = ret
		return ret

	# def load_solution_realworld_measure_wrapper(self, solution_type, deployment):
	# 	solution_type_to_need_load = { ### solution type -> information that needs loading
	# 		'sparse': 'sparse',
	# 		'painter': 'painter',
	# 		'anycast': 'sparse',
	# 		'one_per_pop': 'sparse',
	# 		'one_per_peering': 'None',
	# 	}
	# 	if self.solution_type != solution_type_to_need_load[solution_type]:
	# 		self.update_deployment(deployment)
	# 		self.get_realworld_measure_wrapper()
	# 		self.solution_type = solution_type_to_need_load[solution_type]
	# 		self.get_realworld_measure_wrapper()

	def load_solution_realworld_measure_wrapper(self, solution_type, **kwargs):
		solution_type_to_need_load = { ### solution type -> information that needs loading
			'sparse': 'sparse',
			'painter': 'painter',
			'anycast': 'sparse',
			'one_per_pop': 'sparse',
			'one_per_peering': 'None',
		}
		if self.solution_type != solution_type_to_need_load[solution_type]:
			self.get_realworld_measure_wrapper(**kwargs)
			self.solution_type = solution_type_to_need_load[solution_type]
			self.get_realworld_measure_wrapper(**kwargs)

	def deload_realworld_measure_wrapper(self, deployment):
		## Deloads the real world measure wrapper, and by setting solution_type to None, you don't load any information
		self.solution_type = "None"
		deployment = copy.deepcopy(deployment) ## avoid annoying errors
		self.update_deployment(deployment)

	def solve_lp_with_failure_catch(self, adv, **kwargs):
		self.enforce_loaded_rwmw()
		cache_rep = tuple(np.where(threshold_a(adv).flatten())[0])
		computing_best_lats = kwargs.get('computing_best_lats', False)
		try:
			if kwargs.get('verb') or kwargs.get('smallverb') or kwargs.get('really_bad_fail'):
				# print("Note, verbose is on so we're ignoring LP cache in WFC")
				raise KeyError
			ret = self.linear_prog_soln_cache['failure_catch'][cache_rep, computing_best_lats]
			return copy.deepcopy(ret)
		except KeyError:
			pass
		ret = solve_lp_with_failure_catch(self, adv, **kwargs)
		try:
			if kwargs.get('verb') or kwargs.get('smallverb') or kwargs.get('really_bad_fail'):
				# print("Note, verbose is on so we're ignoring LP cache in WFC")
				raise KeyError
			self.linear_prog_soln_cache['failure_catch'][cache_rep, computing_best_lats] = copy.deepcopy(ret)
		except KeyError:
			pass
		return ret

	def solve_lp_lagrange(self, adv, **kwargs):
		self.enforce_loaded_rwmw()
		cache_rep = tuple(np.where(threshold_a(adv).flatten())[0])
		computing_best_lats = kwargs.get('computing_best_lats', False)
		try:
			if kwargs.get('verb') or kwargs.get('smallverb') or kwargs.get('really_bad_fail'):
				# print("Note, verbose is on so we're ignoring LP cache in WFC")
				raise KeyError
			ret = self.linear_prog_soln_cache['lagrange'][cache_rep, computing_best_lats]
			return copy.deepcopy(ret)
		except KeyError:
			pass
		ret = solve_lp_assignment_lagrange(self, adv, **kwargs)
		self.linear_prog_soln_cache['lagrange'][cache_rep, computing_best_lats] = copy.deepcopy(ret)
		return ret

	def solve_lp_with_failure_catch_weighted_penalty(self, adv, opt_adv, **kwargs):
		self.enforce_loaded_rwmw()
			
		cache_rep = tuple(np.where(threshold_a(np.concatenate([adv,opt_adv],axis=1)).flatten())[0])
		computing_best_lats = kwargs.get('computing_best_lats', False)
		try:
			if kwargs.get('verb') or kwargs.get('smallverb') or kwargs.get('really_bad_fail'):
				# print("Note, verbose is on so we're ignoring LP cache in WFC")
				raise KeyError
			ret = self.linear_prog_soln_cache['penalty'][cache_rep, computing_best_lats]
			return copy.deepcopy(ret)
		except KeyError:
			pass
		optimal_ret = self.solve_lp_with_failure_catch(opt_adv)
		try:
			optimal_latencies = optimal_ret['lats_by_ug']
		except KeyError: # no solution
			return optimal_ret
		ret = solve_lp_assignment_variable_weighted_latency(self, adv, optimal_latencies, **kwargs)
		self.linear_prog_soln_cache['penalty'][cache_rep, computing_best_lats] = copy.deepcopy(ret)

		return ret

	def solve_lp_with_failure_catch_mp(self, advs, **kwargs):
		#### Solves the linear program using worker bees

		adv_i_to_cache_rep = {}

		cache_key = kwargs.get('cache_key','failure_catch')
		worker_cmd = kwargs.get('worker_cmd','solve_lp')

		advs_with_iternum = []
		for adv_i,fields in enumerate(advs):
			if len(fields) == 3:
				adv,dep,flag = fields
				advs_with_iternum.append((adv_i,adv,dep,flag))
			else:
				adv,optimal_adv,dep,flag = fields
				advs_with_iternum.append((adv_i,adv,optimal_adv,dep,flag))

		np.random.shuffle(advs_with_iternum)

		all_rets = [None for i in range(len(advs_with_iternum))]
		if kwargs.get('cache_res', False):
			new_advs = []
			for fields in advs_with_iternum:
				## see if anything has been calculated beore
				if len(fields) == 4: # no penalty
					adv_i,adv,dep,flag = fields
					cache_rep = tuple(np.where(threshold_a(adv).flatten())[0])
				else: # penalty
					adv_i,adv,opt_adv,dep,flag = fields
					cache_rep = tuple(np.where(threshold_a(np.concatenate([adv,opt_adv],axis=1)).flatten())[0])
				adv_i_to_cache_rep[adv_i] = cache_rep
				try:
					all_rets[adv_i] = self.linear_prog_soln_cache[cache_key][cache_rep, False]
				except KeyError:
					new_advs.append(fields)

			advs = new_advs
		else:
			advs = advs_with_iternum

		if len(advs) > 0:
			max_n_in_flight = 350
			n_advs = len(advs)
			global_adv_chunks = split_seq(advs, int(np.ceil(n_advs/max_n_in_flight)))
			for local_adv_chunk in global_adv_chunks:
				adv_chunks = split_seq(local_adv_chunk, self.get_n_workers())
				msgs = []
				for i in range(self.get_n_workers()):
					msg = pickle.dumps([worker_cmd, adv_chunks[i]])
					msgs.append(msg)
				from_workers = self.worker_manager.send_receive_messages_workers(msgs)
				for worker_i, ret in from_workers.items():
					for tr in ret:
						adv_i, answer = tr
						all_rets[adv_i] = answer
						if kwargs.get('cache_res', False):
							cache_rep = adv_i_to_cache_rep[adv_i]
							self.linear_prog_soln_cache[cache_key][cache_rep, False] = answer

		return all_rets

	def update_cache(self, new_cache):
		self.calc_cache.update_cache(new_cache)
		### Keeps the parent-tracker up to date
		for ug, inds in self.calc_cache.all_caches['parents_on'].items():
			for child,parent in inds:
				self.parent_tracker[ug,child,parent] = True

	def add_ug_perf(self, ug, popp, latency):
		### Add performance to UG
		if DEBUG_CLIENT_INFO_ADDING:
			print("Main Thread, adding {}->{}:{}ms".format(ug,popp,round(latency)))
		for pseudo_ug in self.og_ugs_to_new_ugs[ug]:
			self.ug_perfs[pseudo_ug][popp] = latency

	def del_ug_perf(self, ug, popp):
		if DEBUG_CLIENT_INFO_ADDING or ug in UGS_OF_INTEREST:
			print("Main Thread, Deleting {}->{}".format(ug,popp))
		for pseudo_ug in self.og_ugs_to_new_ugs[ug]:
			del self.ug_perfs[pseudo_ug][popp]

			if len(self.ug_perfs[pseudo_ug]) <= 1:
				self.del_ug(pseudo_ug)

	def del_ug(self, ug):
		if DEBUG_CLIENT_INFO_ADDING or ug in UGS_OF_INTEREST:
			print("Main Thread, Deleting {}".format(ug))
		try:
			self.ug_perfs[ug]
			del self.ug_perfs[ug]
		except KeyError:
			for pseudo_ug in self.og_ugs_to_new_ugs[ug]:
				del self.ug_perfs[pseudo_ug]

	def del_client(self, client):
		if DEBUG_CLIENT_INFO_ADDING:
			print("Main Thread, Deleting client {}".format(client))
		relevant_ugs = []
		for ug,clients in self.ug_to_ip.items():
			if client in clients:
				relevant_ugs.append(ug)
		for this_ug in list(set(relevant_ugs)):
			self.ug_to_ip[this_ug] = get_difference(self.ug_to_ip[this_ug], [client])
			if DEBUG_CLIENT_INFO_ADDING:
				print("New set of clients for {}: {}".format(this_ug, self.ug_to_ip[this_ug]))
			if len(self.ug_to_ip[this_ug]) == 0:
				self.del_ug(this_ug)

	def reset_link_capacities(self):
		anycast_catchments = self.rw_measure_wrapper.get_catchments(self.popps)
		if anycast_catchments is not None: ## We possibly don't know them
			anycast_catchments = self.convert_rti_to_pseudo_ugs({0:anycast_catchments})[0]
			from deployment_setup import get_link_capacities_actual_deployment
			self.link_capacities_by_popp = get_link_capacities_actual_deployment(self.output_deployment(), anycast_catchments)

	def compute_one_per_peering_solution(self, **kwargs):
		try:
			self.worker_manager
		except AttributeError: # not ready
			return
		# print("Computing best lats...")c
		revert=False
		try:
			### Temporarily stop using pseudo-ugs
			if self.og_deployment['ugs'] != self.ugs:
				save_dep = self.output_deployment()
				self.update_deployment(self.og_deployment, 
					compute_best_lats=False, clear_caches=False)
				revert=True
		except AttributeError:
			pass

		one_per_ingress_adv = np.identity(self.n_popps)
		ret_overall = self.generic_objective.get_latency_benefit_adv(one_per_ingress_adv)
		objective_lat = ret_overall['objective']
		objective_res = self.generic_objective.get_ground_truth_resilience_benefit(one_per_ingress_adv)
		overall_objective = self.actual_nonconvex_objective(one_per_ingress_adv, 
			latency_benefit_precalc=objective_lat, resilience_benefit_precalc=objective_res)

		# print("{} {}".format(objective_lat, objective_res))
		# exit(0)

		if not ret_overall['solved'] and kwargs.get('exit_on_impossible', True):
			raise ValueError("Cant solve problem in the best case, increase caps.")
		elif not kwargs.get('exit_on_impossible', True):
			pass
		else:
			self.best_lats_by_ug = ret_overall['lats_by_ug']

		try:
			# revert back
			if revert:
				self.update_deployment(save_dep, compute_best_lats=False,
					clear_caches=False)
		except AttributeError:
			pass
		self.optimal_expensive_solution = {
			'obj': ret_overall,
			'latency': objective_lat,
			'resilience': objective_res,
			'overall': overall_objective,
		}


	def check_update_deployment_upon_modify(self):
		### When we modify UG perfs, it could be that many data structures need to change, check that these are changed
		try:
			self.worker_i
			raise ValueError("This method should only be called in the main thread, never in a worker.")
		except AttributeError:
			pass

		self.ugs = sorted(list(self.ug_perfs))
		self.ug_perfs = {ug:self.ug_perfs[ug] for ug in self.ugs}
		self.ug_to_vol = {ug:self.ug_to_vol[ug] for ug in self.ugs}
		self.n_ug = len(self.ugs)
		self.ug_to_ind = {ug:i for i,ug in enumerate(self.ugs)}
		self.ug_to_ip = {ug:self.ug_to_ip[ug] for ug in self.ugs}
		self.ip_to_ug = {ip:ug for ug,ips in self.ug_to_ip.items() for ip in ips}
		
		### Need to update advertisement object when we add popps
		new_popps = sorted(list(set(popp for ug in self.ug_perfs for popp in self.ug_perfs[ug])))
		# print("Updated popp list will have {} popps, old has {}".format(len(new_popps), len(self.popps)))
		# print("Removing {}, adding : {}".format(get_difference(self.popps,new_popps), get_difference(new_popps,self.popps)))
		new_popp_to_ind = {popp:i for i,popp in enumerate(new_popps)}
		removed_popps = get_difference(self.popps, new_popps)
		removed_popps_inds = list([self.popp_to_ind[popp] for popp in removed_popps])
		added_popps = get_difference(new_popps, self.popps)
		added_popps_inds = list([new_popp_to_ind[popp] for popp in added_popps])

		try:
			if len(new_popps) == self.optimization_advertisement.shape[0]: #### TMP!!!!! we should be tracking which popps the optimization advertisement's indices correspond to separately
				raise AttributeError
			# print("Pre opt modify: {}".format(self.optimization_advertisement.shape))
			pre_on = {}
			for prefix_i in range(self.optimization_advertisement.shape[1]):
				pre_on[prefix_i] = list(get_intersection([self.popps[poppi] for poppi in np.where(self.optimization_advertisement[:,prefix_i])[0]], new_popps))
			for remove_poppi in sorted(removed_popps_inds, reverse=True): ## have to go back to front so indices stay consistent
				# print("Removing: {} ({}) ".format(remove_poppi, self.popps[remove_poppi]))
				self.optimization_advertisement = np.concatenate([self.optimization_advertisement[0:remove_poppi,:], self.optimization_advertisement[remove_poppi+1:,:]], axis=0)
				self.last_advertisement = np.concatenate([self.last_advertisement[0:remove_poppi,:], self.last_advertisement[remove_poppi+1:,:]], axis=0)

			for added_poppi in sorted(added_popps_inds): ## have to go front to back so indices stay consistent
				# print("Adding: {} ({})".format(added_poppi, new_popps[added_poppi]))
				self.optimization_advertisement = np.concatenate([self.optimization_advertisement[0:added_poppi,:], .45*np.ones(1,self.n_prefixes), self.optimization_advertisement[added_poppi:,:]], axis=0)
				self.last_advertisement = np.concatenate([self.last_advertisement[0:added_poppi,:], .45*np.ones(1,self.n_prefixes), self.last_advertisement[added_poppi:,:]], axis=0)
			post_on = {}
			for prefix_i in range(self.optimization_advertisement.shape[1]):
				post_on[prefix_i] = list([new_popps[poppi] for poppi in np.where(self.optimization_advertisement[:,prefix_i])[0]])
				this_pre_on = sorted(pre_on[prefix_i])
				this_post_on = sorted(post_on[prefix_i])
				amb = get_difference(this_pre_on, this_post_on)
				bma = get_difference(this_post_on, this_pre_on)
				# print("In pre not post: {}, {}".format(amb, list([self.popp_to_ind[popp] for popp in amb])))
				# print("In post not pre: {}, {}".format(bma, list([new_popp_to_ind[popp] for popp in bma])))
				assert this_pre_on == this_post_on

			# print("Post opt modify: {}".format(self.optimization_advertisement.shape))
		except AttributeError:
			pass

		self.popps = new_popps
		self.popp_to_ind = new_popp_to_ind

		try:
			new_mp = {}
			for ug in self.ug_perfs:
				new_mp[ug] = {}
				for popp,obj in self.measured_prefs.get(ug, {}).items():
					if popp not in self.ug_perfs[ug]:
						continue
					obj.parents = {prnt: obj.parents[prnt] for prnt in get_intersection(self.ug_perfs[ug], obj.parents)}
					obj.children = {child: obj.children[child] for child in get_intersection(self.ug_perfs[ug], obj.children)}
					new_mp[ug][popp] = obj
				for popp in get_difference(self.ug_perfs[ug], self.measured_prefs.get(ug,{})):
					new_mp[ug][popp] = Ing_Obj(popp)
			self.measured_prefs = new_mp
		except AttributeError:
			import traceback
			traceback.print_exc()

		self.whole_deployment_ugs = self.ugs
		self.whole_deployment_n_ug = len(self.whole_deployment_ugs)
		self.whole_deployment_ug_to_ind = {ug:i for i,ug in enumerate(self.whole_deployment_ugs)}
		self.whole_deployment_ug_perfs = self.ug_perfs
		self.ug_anycast_perfs = {ug:self.ug_anycast_perfs[ug] for ug in self.ugs}

		self.n_popps = len(self.popps)
		self.n_popp = len(self.popps)
		self.pops = sorted(list(set([u[0] for u in self.popps])))
		self.n_pops = len(self.pops)
		self.pop_to_ind = {k:i for i,k in enumerate(self.pops)}
		self.pop_to_popps = {}
		for popp in self.popps:
			try:
				self.pop_to_popps[popp[0]].append(popp)
			except KeyError:
				self.pop_to_popps[popp[0]] = [popp]
		self.pop_to_popp_inds = {pop:np.array([self.popp_to_ind[popp] for popp in self.pop_to_popps[pop]]) for pop in self.pops}

		self.make_ui_to_poppi()

		self.provider_popps = get_intersection(self.provider_popps, self.popps)
		self.n_provider_popps = len(self.provider_popps) # number of provider PoPps

		self.n_providers = len(set(popp[1] for popp in self.provider_popps)) # number of provider ASes
		self.provider_popp_inds = [self.popp_to_ind[popp] for popp in self.provider_popps]


		### Start link capacities
		## Remove non-existent popps, add new ones to link capacities
		all_current_link_capacities = list(self.link_capacities_by_popp.values())
		old_popps = list(self.link_capacities_by_popp)
		choices = list(range(len(old_popps)))
		for popp in get_difference(self.popps, list(self.link_capacities_by_popp)): # add new link caps
			self.link_capacities_by_popp[popp] = self.link_capacities_by_popp[old_popps[np.random.choice(choices)]]
		self.link_capacities_by_popp = {popp: self.link_capacities_by_popp[popp] for popp in self.popps}
		## set new link capacities using updated catchment information
		self.reset_link_capacities()

		self.link_capacities = {self.popp_to_ind[popp]: self.link_capacities_by_popp[popp] for popp in self.popps}

		self.link_capacities_arr = np.zeros(self.n_popp)
		for poppi, cap in self.link_capacities.items():
			self.link_capacities_arr[poppi] = cap

		### End link capacities


		try:
			self.og_deployment['link_capacities'] = self.link_capacities_by_popp
			self.og_deployment['popps'] = self.popps
			self.og_deployment['provider_popps'] = self.provider_popps
			self.og_deployment['n_providers'] = self.n_providers
			### DO I need to update the rest of this? It would be annoying, as I'd have to convert between pseudo and normal UGs
			# yup

			still_existing_og_ugs = sorted(list(set([(ug[0],int(ug[1])) for ug in self.ugs])))
			new_ug_perfs = {}
			for ug in still_existing_og_ugs:
				new_ug_perfs[ug] = {}
				for popp in get_intersection(self.og_deployment['ug_perfs'][ug], self.popps):
					new_ug_perfs[ug][popp] = self.og_deployment['ug_perfs'][ug][popp]
			self.og_deployment['ugs'] = still_existing_og_ugs
			self.og_deployment['whole_deployment_ugs'] = still_existing_og_ugs
			self.og_deployment['ug_perfs'] = new_ug_perfs
			self.og_deployment['whole_deployment_ug_perfs'] = new_ug_perfs
			self.og_deployment['ug_to_vol'] = {ug:self.og_deployment['ug_to_vol'][ug] for ug in still_existing_og_ugs}
			self.og_deployment['whole_deployment_ug_to_vol'] = self.og_deployment['ug_to_vol']
			self.og_deployment['ug_to_ip'] = {ug: self.og_deployment['ug_to_ip'][ug] for ug in still_existing_og_ugs}
			self.og_deployment['ug_anycast_perfs'] = {ug: self.og_deployment['ug_anycast_perfs'][ug] for ug in still_existing_og_ugs}

		except AttributeError:
			pass


		self.whole_deployment_ug_to_vol = self.ug_to_vol
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


		try:
			del self.rb_backups
		except AttributeError:
			pass

		## compute best latency by ug
		self.linear_prog_soln_cache = {
			'regular': {},
			'failure_catch': {},
			'penalty': {},
			'lagrange': {},
		}

		### Main thread
		with open(os.path.join(LOG_DIR, 'main_thread_log-{}.txt'.format(self.dpsize)),'w') as f:
			pass
		self.parent_tracker = {}

		self.compute_one_per_peering_solution()
		
		try:					
			del self.rb_backups
		except AttributeError:
			pass
		self.last_lb_calls_results = {}
		self.last_rb_calls_results_popp = {}
		self.all_rb_calls_results_popps = {poppi:[] for poppi in range(self.n_popps)}
		self.all_rb_calls_results_pops = {popi:[] for popi in range(self.n_pops)}
		self.n_resilience_benefit_popp_calls = {}
		self.n_latency_benefit_calls = {}

		self.all_lb_calls_results = []
		self.deployment = self.output_deployment()

	def restart_workers(self):
		try:
			wm = self.get_worker_manager()
		except AttributeError:
			# No worker manager, no problem
			return
		wm.deployment = self.output_deployment()
		wm.stop_workers()
		time.sleep(5)
		wm.start_workers()
		time.sleep(5)

		self.update_parent_tracker_workers()

	def enforce_loaded_rwmw(self):
		if not self.simulated:
			### You need to load realworld measure wrapper before calling this function
			self.rw_measure_wrapper

	def output_specific_deployment(self, solution_type, **kwargs):
		
		solution_type_to_need_load = { ### solution type -> information that needs loading
			'sparse': 'sparse',
			'painter': 'painter',
			'anycast': 'sparse',
			'one_per_pop': 'sparse',
			'one_per_peering': 'None',
		}

		if self.solution_type != solution_type_to_need_load[solution_type]:
			self.solution_type = 'None'
			self.get_realworld_measure_wrapper()
			self.solution_type = solution_type_to_need_load[solution_type]
			self.get_realworld_measure_wrapper()

		ret = self.output_deployment()

		if kwargs.get('reset_on_exit', True):
			self.solution_type = 'None'
			self.get_realworld_measure_wrapper()

		return ret

	def output_deployment(self, copykeys='all'):
		deployment = {
			'ugs': self.ugs,
			'dpsize': self.dpsize,
			'ug_to_ip': self.ug_to_ip,
			'ug_perfs': self.ug_perfs,
			'ug_to_vol': self.ug_to_vol,
			'ug_to_bulk_vol': self.ug_to_bulk_vol,
			'ug_anycast_perfs': self.ug_anycast_perfs,
			'whole_deployment_ugs': self.whole_deployment_ugs,
			'whole_deployment_ug_perfs': self.whole_deployment_ug_perfs,
			'whole_deployment_ug_to_vol': self.whole_deployment_ug_to_vol,
			'whole_deployment_ug_to_bulk_vol': self.whole_deployment_ug_to_bulk_vol,
			'popps': self.popps,
			'site_costs': self.site_costs,
			'metro_loc': self.metro_loc,
			'pop_to_loc': self.pop_to_loc,
			'n_providers': self.n_providers,
			'provider_popps': self.provider_popps,
			'link_capacities': self.link_capacities_by_popp,
			'simulated': self.simulated,
			'port': self.port,
			'generic_objective': self.generic_objective.obj,
		}
		if self.simulated:
			deployment['ingress_priorities'] = self.ground_truth_ingress_priorities
			deployment['whole_deployment_ingress_priorities'] = self.whole_deployment_ground_truth_ingress_priorities

		if copykeys is not None:
			if copykeys == 'all':
				ret = copy.deepcopy(deployment)
			else:
				ret = {}
				for k in deployment:
					if k in copykeys:
						ret[k] = copy.deepcopy(deployment[k])
					else:
						ret[k] = deployment[k]
		else:
			## Be very careful
			ret = deployment

		return ret

	def make_ui_to_poppi(self):
		self.ui_to_poppi = {}
		for ug in self.ug_perfs:
			self.ui_to_poppi[self.ug_to_ind[ug]] = {}
			for popp in self.ug_perfs[ug]:
				self.ui_to_poppi[self.ug_to_ind[ug]][self.popp_to_ind[popp]] = None
		self.poppi_to_ui = {}
		for ug in self.ug_perfs:
			for popp in self.ug_perfs[ug]:
				try:
					self.poppi_to_ui[self.popp_to_ind[popp]].append(self.ug_to_ind[ug])
				except KeyError:
					self.poppi_to_ui[self.popp_to_ind[popp]] = [self.ug_to_ind[ug]]
		for poppi in get_difference(list(range(self.n_popps)), self.poppi_to_ui):
			self.poppi_to_ui[poppi] = []

		self.whole_deployment_ui_to_poppi = {}
		for ug in self.whole_deployment_ug_perfs:
			self.whole_deployment_ui_to_poppi[self.whole_deployment_ug_to_ind[ug]] = {}
			for popp in self.whole_deployment_ug_perfs[ug]:
				self.whole_deployment_ui_to_poppi[self.whole_deployment_ug_to_ind[ug]][self.popp_to_ind[popp]] = None
		self.whole_deployment_poppi_to_ui = {}
		for ug in self.whole_deployment_ug_perfs:
			for popp in self.whole_deployment_ug_perfs[ug]:
				try:
					self.whole_deployment_poppi_to_ui[self.popp_to_ind[popp]].append(self.whole_deployment_ug_to_ind[ug])
				except KeyError:
					self.whole_deployment_poppi_to_ui[self.popp_to_ind[popp]] = [self.whole_deployment_ug_to_ind[ug]]
		for poppi in get_difference(list(range(self.n_popps)), self.whole_deployment_poppi_to_ui):
			self.whole_deployment_poppi_to_ui[poppi] = []

	def update_deployment(self, deployment, **kwargs):
		self.simulated = deployment.get('simulated', True)
		if deployment.get('port') is None:
			print("\n\nWARNING ---- NO PORT SPECIFIED\n\n")
			time.sleep(5)
		self.port = deployment.get('port', 31415)
		if deployment.get('dpsize') == 'small':
			self.max_n_iter = 100
		elif self.simulated:
			self.max_n_iter = 150 # maximum number of learning iterations
		else:
			self.max_n_iter = 150 # maximum number of learning iterations


		self.dpsize = deployment['dpsize']
		self.ugs = sorted(deployment['ugs'])
		self.ug_to_ip = deployment.get('ug_to_ip',None)
		self.n_ug = len(self.ugs)
		self.whole_deployment_ugs = sorted(deployment['whole_deployment_ugs'])
		self.whole_deployment_n_ug = len(self.whole_deployment_ugs)
		self.ug_to_ind = {ug:i for i,ug in enumerate(self.ugs)}
		self.whole_deployment_ug_to_ind = {ug:i for i,ug in enumerate(self.whole_deployment_ugs)}
		self.ug_perfs = deployment['ug_perfs']
		self.whole_deployment_ug_perfs = deployment['whole_deployment_ug_perfs']
		self.ug_anycast_perfs = deployment['ug_anycast_perfs']

		# Shape of the variables
		self.popps = sorted(list(set(deployment['popps'])))
		self.n_popps = len(self.popps)
		self.pops = sorted(list(set([u[0] for u in self.popps])))
		self.site_costs = deployment['site_costs']
		self.n_pops = len(self.pops)
		self.popp_to_ind = {k:i for i,k in enumerate(self.popps)}
		self.pop_to_ind = {k:i for i,k in enumerate(self.pops)}
		self.pop_to_popps = {}
		for popp in self.popps:
			try:
				self.pop_to_popps[popp[0]].append(popp)
			except KeyError:
				self.pop_to_popps[popp[0]] = [popp]
		self.pop_to_popp_inds = {pop:np.array([self.popp_to_ind[popp] for popp in self.pop_to_popps[pop]]) for pop in self.pops}

		try:
			self.clear_caches()
		except:
			pass

		self.make_ui_to_poppi()

		self.provider_popps = deployment['provider_popps']
		self.n_provider_popps = len(self.provider_popps) # number of provider PoPps
		self.n_popp = len(self.popps)
		if self.n_prefixes is None:
			self.n_prefixes = np.maximum(3,self.n_popp // 3)
		self.n_providers = deployment['n_providers'] # number of provider ASes
		self.provider_popp_inds = [self.popp_to_ind[popp] for popp in self.provider_popps]
		if self.simulated:
			self.ground_truth_ingress_priorities = deployment['ingress_priorities']
			self.whole_deployment_ground_truth_ingress_priorities = deployment['whole_deployment_ingress_priorities']

		self.metro_loc = deployment['metro_loc']
		self.pop_to_loc = deployment['pop_to_loc']
		self.link_capacities_by_popp = deployment['link_capacities']
		self.link_capacities = {self.popp_to_ind[popp]: self.link_capacities_by_popp[popp] for popp in self.popps}

		self.link_capacities_arr = np.zeros(self.n_popp)
		for poppi, cap in self.link_capacities.items():
			self.link_capacities_arr[poppi] = cap

		self.ug_to_vol = deployment['ug_to_vol']
		self.ug_to_bulk_vol = deployment.get('ug_to_bulk_vol', None)
		self.whole_deployment_ug_to_vol = deployment['whole_deployment_ug_to_vol']
		self.whole_deployment_ug_to_bulk_vol = deployment.get('whole_deployment_ug_to_bulk_vol',None)
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

		## bulk volumes, if applicable
		if self.ug_to_bulk_vol is not None:
			self.ug_bulk_vols = np.zeros(self.n_ug)
			self.whole_deployment_ug_bulk_vols = np.zeros(self.whole_deployment_n_ug)
			for ug,v in self.ug_to_bulk_vol.items():
				self.ug_bulk_vols[self.ug_to_ind[ug]] = v
			for ug,v in self.whole_deployment_ug_to_bulk_vol.items():
				self.whole_deployment_ug_bulk_vols[self.whole_deployment_ug_to_ind[ug]] = v


		self.deployment = self.output_deployment()
		try:
			self.worker_i
			### Worker thread, need to divide capacities by the number of workers
			### where appropriate

			## want to apportion capacity according to how much of the capacity I think
			## each worker will use
			## users will "likely want to use" low latency popps, so maybe their top 10
			best_n = 10
			global_popp_vol_ctr, this_wrkr_popp_vol_ctr = {popp:0 for popp in self.popps},{popp:0 for popp in self.popps}
			for ug in self.whole_deployment_ugs:
				sorted_popps = sorted(self.whole_deployment_ug_perfs[ug].items(), key = lambda el : el[1])
				for best_popp,perf in sorted_popps[0:best_n]:
					if ug in self.ugs:
						this_wrkr_popp_vol_ctr[best_popp] += self.whole_deployment_ug_to_vol[ug]
					global_popp_vol_ctr[best_popp] += self.whole_deployment_ug_to_vol[ug]

			for popp, cap in self.link_capacities_by_popp.items():
				if global_popp_vol_ctr[popp] > 0 and this_wrkr_popp_vol_ctr[popp] > 0:
					## percent of volume this worker woud likely contribute
					divider = this_wrkr_popp_vol_ctr[popp] / global_popp_vol_ctr[popp]
					divider = 1.05 / divider # be a little more conservative
					# print("Worker {} dividing popp {} by {}".format(self.worker_i, popp, round(divider,2)))
				else:
					divider = 1 # likely won't use any of it, so it doesn't matter
				divider = 1 #POSSIBLy TMP
				self.link_capacities_by_popp[popp] = cap / divider

		except AttributeError:
			pass
		self.link_capacities = {self.popp_to_ind[popp]: self.link_capacities_by_popp[popp] for popp in self.popps}

		try:
			del self.rb_backups
		except AttributeError:
			pass

		if self.simulated:
			self.ingress_priority_inds = {ug:{self.popp_to_ind[popp]: \
				self.whole_deployment_ground_truth_ingress_priorities[ug][popp] for popp in self.whole_deployment_ug_perfs[ug]} \
				for ug in self.whole_deployment_ugs}
			## Used for computing ground truth ingress
			n_paths = sum(len(popps) for ug,popps in self.whole_deployment_ug_perfs.items())
			data = np.zeros((n_paths))
			rows = np.zeros((n_paths))
			cols = np.zeros((n_paths))
			i=0
			self.popp_by_ug_indicator_dict = {}
			for ui in range(self.whole_deployment_n_ug):
				self.popp_by_ug_indicator_dict[ui] = {}
				for popp in sorted(self.whole_deployment_ug_perfs[self.whole_deployment_ugs[ui]]):
					## High pref = routed through there. Ingress priority inds stores prefernece 0 as most-preferred
					pref = self.n_popp + 1 - self.ingress_priority_inds[self.whole_deployment_ugs[ui]][self.popp_to_ind[popp]]
					self.popp_by_ug_indicator_dict[ui][self.popp_to_ind[popp]] = pref
					data[i] = pref
					rows[i] = self.popp_to_ind[popp]
					cols[i] = ui
					i += 1
			self.popp_by_ug_indicator = csr_matrix((data, (rows,cols)), shape=(self.n_popp, self.whole_deployment_n_ug))

		## compute best latency by ug
		self.linear_prog_soln_cache = {
			'regular': {},
			'failure_catch': {},
			'penalty': {},
			'lagrange': {},
		}

		try:
			### Worker bee
			self.worker_i
			if kwargs.get("clear_caches", True):
				self.parent_tracker = {}
				self.init_all_vars()
				self.clear_new_meas_caches()
		except AttributeError:
			### Main thread
			with open(os.path.join(LOG_DIR, 'main_thread_log-{}.txt'.format(self.dpsize)),'w') as f:
				pass
			self.parent_tracker = {}
			if not self.simulated:
				try:
					del self.rw_measure_wrapper
				except AttributeError:
					pass
			self.get_realworld_measure_wrapper()
					
		quick_update = kwargs.get('quick_update', False)
		try:
			if quick_update:
				raise AttributeError
			if kwargs.get('compute_best_lats', True):
				self.compute_one_per_peering_solution()
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

				msg = pickle.dumps(('update_deployment', (subdeployments[worker], kwargs)))
				self.send_receive_worker(worker, msg)
			
		except AttributeError:
			pass

		self.all_rb_calls_results_popps = {poppi:[] for poppi in range(self.n_popps)}
		self.all_rb_calls_results_pops = {popi:[] for popi in range(self.n_pops)}
		self.n_resilience_benefit_popp_calls = {}
		self.n_latency_benefit_calls = {}

		self.all_lb_calls_results = []
		if kwargs.get("clear_caches", True) and not quick_update:
			#### TMP 
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
		self.measured_latency_benefits = {}
		self.best_latency_benefits = {ui: -1 * NO_ROUTE_LATENCY for ui in range(self.n_ug)}
		total_vol = np.sum(self.whole_deployment_ug_vols)
		for ug in self.ugs:
			ugi = self.ug_to_ind[ug]
			for popp in self.popps:
				popp_i = self.popp_to_ind[popp]
				## benefit per user is -1 * latency * volume fraction
				## we multiply by volume fraction so that later we can just calculate the sum and
				## have that be the correct average benefit
				weight = self.ug_vols[ugi] / total_vol
				self.measured_latency_benefits[popp_i,ugi] = -1 * self.ug_perfs[ug].get(popp, NO_ROUTE_LATENCY) * weight
				self.best_latency_benefits[ugi] = np.maximum(self.best_latency_benefits[ugi], self.measured_latency_benefits[popp_i,ugi])
					
		# same for each prefix (ease of calculation later)

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
		return self.generic_objective.get_ground_truth_latency_benefit(a, **kwargs)

	def get_ground_truth_latency_benefit_mp(self, advs, dep, **kwargs):
		### Measures actual latency benefit as if we were to advertise 'a' in batches
		call_args = []
		for i,adv in enumerate(advs):
			call_args.append((threshold_a(adv), dep, i%20==0))
		lp_rets = self.solve_lp_with_failure_catch_mp(call_args, cache_res=False)
		all_rets = []
		ugs = kwargs.get('ugs', self.ugs)
		for lp_ret in lp_rets:
			user_latencies = lp_ret['lats_by_ug']
			benefit = self.benefit_from_user_latencies(user_latencies,ugs)
			all_rets.append(benefit)
		return all_rets

	def get_ground_truth_user_latencies(self, a, mode='lp', **kwargs):
		#### Measures actual user latencies as if we were to advertise 'a'
		self.enforce_loaded_rwmw()
		if mode == 'best':
			user_latencies, ug_ingress_decisions = self.calculate_user_choice(a, mode=mode,**kwargs)
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
				if not kwargs.get('save_metrics', False):
					for cap_violation in np.where(cap_violations)[0]:
						for ugi in ingress_to_users[cap_violation]:
							user_latencies[ugi] = NO_ROUTE_LATENCY
			# if self.verbose and len(np.where(cap_violations)[0]) > 0:
			# 	print([round(el,2) for el in user_latencies])
		elif mode == 'lp':
			ret = self.solve_lp_with_failure_catch(threshold_a(a), **kwargs)
			if not ret['solved']:
				print("In ground truth user latencies, didn't successfully solve LP")
				return self.get_ground_truth_user_latencies(a, mode='best', **kwargs)
			user_latencies = ret['lats_by_ug']
			ug_ingress_decisions = ret['paths_by_ug']
			if kwargs.get('save_ug_ingress_decisions'):
				self.popp_to_users = {}
				for ugi in ug_ingress_decisions:
					for poppi,v in ug_ingress_decisions[ugi]:
						try:
							self.popp_to_users[poppi].append(self.ugs[ugi])
						except KeyError:
							self.popp_to_users[poppi] = [self.ugs[ugi]]
		else:
			raise ValueError("Unknown mode {}".format(mode))
		
		return user_latencies

	def get_top_ugs(self, n=20):
		return sorted(self.ugs, key = lambda el : -1 * self.ug_to_vol[el])[0:n]

	def benefit_from_user_latencies(self, user_latencies, ugs):
		# sum of the benefits, simple model for benefits is 1 / latency
		# allow for calculation over a subset of UGs
		user_benefits = -1 * user_latencies
		# average user benefit -- important that this function is not affected by the number of user groups
		these_inds = np.array([self.ug_to_ind[ug] for ug in ugs])
		these_vols = self.ug_vols[these_inds]
		these_benefits = user_benefits[these_inds]
		return np.sum(these_benefits * these_vols) / np.sum(these_vols)

	def compute_actual_ground_truth_resilience(self):
		return self.dpsize in ['small', 'actual-10']

	def get_ground_truth_resilience_benefit(self, advertisement, **kwargs):
		#### NOTE -- this calculation does not factor in capacity, since that would take a long time

		if self.gamma == 0:
			return 0

		return self.generic_objective.get_ground_truth_resilience_benefit(advertisement, **kwargs)

	def calculate_user_choice(self, a, mode = 'lp', **kwargs):
		"""Calculates UG -> popp assuming they act according to decision mode 'mode'."""
		## returns user latencies (np array) and ug_ingress_decisions maps ugi to [(poppi, vol)]
		self.enforce_loaded_rwmw()
		ugs = kwargs.get('ugs', self.ugs)
		if mode == 'best':
			# every user visits their best popp
			a = threshold_a(a)
			user_latencies = NO_ROUTE_LATENCY * np.ones((len(ugs)))
			kwargs['ugs'] = ugs
			routed_through_ingress, _ = self.calculate_ground_truth_ingress(a, **kwargs)
			ug_ingress_decisions = {ugi:None for ugi in range(self.n_ug)}
			for prefix_i in range(a.shape[1]):
				for ugi,ug in enumerate(kwargs.get('ugs', ugs)):
					routed_ingress = routed_through_ingress[prefix_i].get(ug)
					if routed_ingress is None:
						latency = NO_ROUTE_LATENCY
					else:
						latency = self.ug_perfs[ug][routed_ingress]
					if latency < user_latencies[ugi]: 
						ug_ingress_decisions[ugi] = [(self.popp_to_ind[routed_ingress],1)]
						user_latencies[ugi] = latency
		elif mode == 'lp':
			ret = self.solve_lp_with_failure_catch(threshold_a(a), **kwargs)

			ugs = get_intersection(self.ugs, ugs) ### Might have changed after this call

			user_latencies = ret['lats_by_ug']
			ug_ingress_decisions_tmp = ret['paths_by_ug']


			### Need to update this knowing that some ugs may be fixed !!
			## for now, just solving for anything and limiting to the ones I want
			ug_ingress_decisions = {self.ug_to_ind[ug]: ug_ingress_decisions_tmp.get(self.ug_to_ind[ug],[]) for ug in ugs}
			if len(ugs) != self.n_ug:
				ug_inds = np.array([self.ug_to_ind[ug] for ug in ugs])
				user_latencies = user_latencies[ug_inds]

		if kwargs.get('get_ug_catchments', False):
			try:
				if np.random.random() > .9:
					## occasionally recompute these
					del self.rb_backups
			except AttributeError:
				pass
			self.update_ug_ingress_decisions(ug_ingress_decisions)
		if kwargs.get('verb'):
			for ui,poppipcts in ug_ingress_decisions.items():
				for poppi,pct in poppipcts:
					self.log("benefit_estimate,{},{},{},{},{}\n".format(
						self.iter,ui,poppi,pct,self.measured_latency_benefits[poppi,ui]))

		return user_latencies, ug_ingress_decisions

	def log(self,s):
		self.log_ptr = open(os.path.join(LOG_DIR, 'main_thread_log-{}.txt'.format(self.dpsize)),'a')
		self.log_ptr.write(s)
		self.log_ptr.close()

	def update_ug_ingress_decisions(self, ug_catchments=None):
		"""For updating variables that only change with ug catchments."""
		# pairing popps with popps
		# popp -> users -> benefit
		# but also users -> popps
		# support (popp1 -> popp2) is sum_over_users( potential backup popp1 can provide to popp2 for that user)
		# backup is ~ how good that option is compared to user's other options
		# lots of backup for very good relative option

		poppi_to_ugi = {}
		if ug_catchments is None:
			for ug,popps in self.ug_perfs.items():
				for popp in popps:
					try:
						poppi_to_ugi[self.popp_to_ind[popp]].append(self.ug_to_ind[ug])
					except KeyError:
						poppi_to_ugi[self.popp_to_ind[popp]] = [self.ug_to_ind[ug]]
		else:
			for ugi,poppipcts in ug_catchments.items():
				for poppi,pct in poppipcts:
					try:
						poppi_to_ugi[poppi].append(ugi)
					except KeyError:
						poppi_to_ugi[poppi] = [ugi]
		try:
			self.rb_backups
		except AttributeError:
			### Resilience benefit backup pre-calcs
			self.rb_backups = {}
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
						### proportional to latency difference, UG volume, ability of the backup link to handle the volume
						self.rb_backups[self.ug_to_ind[ug],popp1i,popp2i] = backup * self.ug_to_vol[ug] * self.link_capacities[popp2i]

		self.rb_popp_support = np.zeros((self.n_popps, self.n_popps))
		for popp1i in poppi_to_ugi:
			for ugi in poppi_to_ugi[popp1i]:
				for popp2i in range(self.n_popps):
					self.rb_popp_support[popp1i,popp2i] += self.rb_backups.get((ugi,popp1i,popp2i), 0)

		self.rb_popp_support += .01
		self.popp_backup_sample_probs = (self.rb_popp_support.T / np.sum(self.rb_popp_support, axis=1)).T

	def check_update_calls_to_update_deployment(self, calls_to_update_deployment):
		if len(calls_to_update_deployment) > 0:
			for call in calls_to_update_deployment:
				args = call[1:]
				getattr(self, call[0])(*args)
			self.check_update_deployment_upon_modify()
			self.restart_workers()
		self.check_update_deployment_upon_modify()

	def get_realworld_measure_wrapper(self, **kwargs):
		if self.simulated: return

		## Useful to bootstrap from past runs, but could lead to false understandings of convergence time
		if self.dpsize == 'actual_first_prototype':
			info_prds = list([os.path.join(d, self.solution_type) for d in ['1712340117-actual_first_prototype-sparse']])
		elif self.dpsize == 'actual_second_prototype':
			info_prds = list([os.path.join(d, self.solution_type) for d in ['1712535530-actual_second_prototype-sparse']])
		elif self.dpsize == 'actual_third_prototype':
			info_prds = list([os.path.join(d, self.solution_type) for d in ['1712535530-actual_second_prototype-sparse','1713752460-actual_second_prototype-sparse']])
			
		base_save_run_dirs = list([os.path.join(RUN_DIR, prd) for prd in info_prds])

		try:
			srd = os.path.join(self.save_run_dir, self.solution_type)
			prds = [os.path.join(self.save_run_dir, self.solution_type)]
		except AttributeError:
			srd = None
			prds = [os.path.join(RUN_DIR, '')]
		prds = base_save_run_dirs + prds
		try:
			dep = self.og_deployment
		except AttributeError:
			dep = self.output_deployment()
		from realworld_measure_wrapper import RealWorld_Measure_Wrapper
		self.rw_measure_wrapper = RealWorld_Measure_Wrapper(srd, dep, past_rundirs=prds)
		try:
			self.worker_i
			return
		except AttributeError:
			pass

		### The real world measure wrapper uses real UG information, while this object uses pseudo-UG information
		### So we need to do the conversion correctly
		self.og_ugs_to_new_ugs = {}
		for ug in self.ugs:
			og_ug = (ug[0], int(ug[1]))
			try:
				self.og_ugs_to_new_ugs[og_ug].append(ug)
			except KeyError:
				self.og_ugs_to_new_ugs[og_ug] = [ug]
		routed_through_ingress, actives, calls_to_update_deployment = self.rw_measure_wrapper.reload_info(**kwargs)
		## Convert to using pseudo-UGs
		converted_ret_routed_through_ingress = {}
		for prefix_i,this_routed_through_ingress in routed_through_ingress.items():
			converted_ret_routed_through_ingress[prefix_i] = {}
			for ug,popp in this_routed_through_ingress.items():
				# This might be empty if we're loading state from future iterations where we've already deleted a UG
				# So we do "get"
				for pseudo_ug in self.og_ugs_to_new_ugs.get(ug, []): 
					converted_ret_routed_through_ingress[prefix_i][pseudo_ug] = popp
		self.check_update_calls_to_update_deployment(calls_to_update_deployment)
		for prefix_i in actives:
			actives[prefix_i] = get_intersection(actives[prefix_i], self.popps)

		self.enforce_measured_prefs(converted_ret_routed_through_ingress, actives)

		self.reset_link_capacities()

	def check_load_rw_measure_wrapper(self):
		try:
			self.rw_measure_wrapper
		except AttributeError:
			### Communication with the real-world deployment
			self.get_realworld_measure_wrapper()

	def check_need_measure_actual_deployment(self, a):
		if self.simulated: return
		self.enforce_loaded_rwmw()

		a = threshold_a(a)
	
		cols_to_possibly_measure = []
		for prefix_i in range(a.shape[1]):
			if np.sum(a[:,prefix_i]) == 0:
				continue
			elif np.sum(a[:,prefix_i]) == 1:
				continue
			this_actives = list([self.popps[poppi] for poppi in np.where(a[:,prefix_i])[0]])
			cols_to_possibly_measure.append((prefix_i, this_actives))

		cols_to_measure = []
		if len(cols_to_possibly_measure) > 0:
			### output is routed_through_ingress (prefix_i -> ug -> poppi)
			ret_cols_to_measure = self.rw_measure_wrapper.check_need_measure(cols_to_possibly_measure)
			for prefix_i, popps in ret_cols_to_measure:
				cols_to_measure.append(np.expand_dims(a[:,prefix_i],axis=1))

		return cols_to_measure

	def calculate_ground_truth_ingress(self, a, **kwargs):
		### Returns routed_through_ingress: prefix -> ug -> popp_i
		## and actives prefix -> [active popp indices]

		self.enforce_loaded_rwmw()
		routed_through_ingress = {} # prefix -> UG -> ingress index
		actives = {} # prefix -> popps where we adv
		ugs = kwargs.get('ugs', self.whole_deployment_ugs)
		n_ug = len(ugs)

		a = threshold_a(a)

		if self.simulated:
			### Somewhat efficient implementation using matrix logic
			for prefix_i in range(a.shape[1]):
				cache_rep = tuple(np.where(a[:,prefix_i])[0].flatten())
				try:
					routed_through_ingress[prefix_i], actives[prefix_i] = self.calc_cache.all_caches['gti'][cache_rep]
					continue
				except KeyError:
					pass
				active_inds = np.where(a[:,prefix_i])[0]
				suma = np.sum(a[:,prefix_i])
				this_actives = list([self.popps[poppi] for poppi in active_inds])
				actives[prefix_i] = this_actives
				this_routed_through_ingress = {}
				if suma == 0:
					## Trivial, no route for everyone
					routed_through_ingress[prefix_i] = this_routed_through_ingress
					if n_ug == self.whole_deployment_n_ug and a.shape[1] == self.n_prefixes:
						self.calc_cache.all_caches['gti'][cache_rep] = (this_routed_through_ingress,this_actives)
					continue
				elif suma == 1:
					## Trivial, one route for users who have that option
					popp_active = self.popps[np.where(a[:,prefix_i])[0][0]]
					for ug in ugs:
						try:
							self.whole_deployment_ug_perfs[ug][popp_active]
							this_routed_through_ingress[ug] = popp_active
						except KeyError:
							pass
				else:
					for ug in ugs:
						ui = self.whole_deployment_ug_to_ind[ug]
						best_available_option = None
						for ai in active_inds:
							try:
								pref = self.popp_by_ug_indicator_dict[ui][ai]
								if best_available_option is None:
									best_available_option = ai
									best_available_pref = pref
								elif pref > best_available_pref:
									best_available_option = ai
									best_available_pref = pref
							except KeyError:
								pass
						if best_available_option is None: continue # no route

						## Matrix method
						# active_popp_ug_indicator = self.popp_by_ug_indicator[:,ui].multiply(np.expand_dims(a[:,prefix_i],axis=1)).toarray()
						# best_available_option = np.argmax(active_popp_ug_indicator)
						# if active_popp_ug_indicator[best_available_option] == 0: continue # no route

						this_routed_through_ingress[ug] = self.popps[best_available_option]
				if n_ug == self.whole_deployment_n_ug and a.shape[1] == self.n_prefixes and kwargs.get('do_cache', True):
					## not a subset of user groups, cache result for general utility
					self.calc_cache.all_caches['gti'][cache_rep] = (this_routed_through_ingress,this_actives)
				routed_through_ingress[prefix_i] = this_routed_through_ingress
		else:
			#### ACTUALLY MEASURE THINGS IN THE REAL INTERNET
			### also would make sense for, at least at first, to have some sort of caching of results so that I
			### can recover from errors, restart from iterations and given state, etc..
			## like given algorithm, deployment, current advertisement, iteration, continue optimizing
			
			cols_to_measure = []
			for prefix_i in range(a.shape[1]):
				this_actives = list([self.popps[poppi] for poppi in np.where(a[:,prefix_i])[0]])
				actives[prefix_i] = this_actives
				this_routed_through_ingress = {}
				if np.sum(a[:,prefix_i]) == 0:
					routed_through_ingress[prefix_i] = this_routed_through_ingress
					continue
				elif np.sum(a[:,prefix_i]) == 1:
					## Trivial, one route for users who have that option
					popp_active = self.popps[np.where(a[:,prefix_i])[0][0]]
					for ug in ugs:
						try:
							self.whole_deployment_ug_perfs[ug][popp_active]
							this_routed_through_ingress[ug] = popp_active
						except KeyError:
							pass
					routed_through_ingress[prefix_i] = this_routed_through_ingress
					continue
				cols_to_measure.append((prefix_i, this_actives))

			if len(cols_to_measure) > 0:
				### output is routed_through_ingress (prefix_i -> ug -> poppi)
				ret_routed_through_ingress, calls_to_update_deployment = self.rw_measure_wrapper.measure_advs(cols_to_measure, **kwargs)
				ret_routed_through_ingress = self.convert_rti_to_pseudo_ugs(ret_routed_through_ingress)
				if not kwargs.get('dont_update_deployment', False):
					self.check_update_calls_to_update_deployment(calls_to_update_deployment)

				for prefix_i, this_routed_through_ingress in ret_routed_through_ingress.items():
					## Actives may have changed, recompute
					actives[prefix_i] = get_intersection(actives[prefix_i], self.popps)
					routed_through_ingress[prefix_i] = this_routed_through_ingress

		return routed_through_ingress, actives

	def convert_rti_to_pseudo_ugs(self, ret_routed_through_ingress):
		## Convert to using pseudo-UGs
		converted_ret_routed_through_ingress = {}
		for prefix_i,this_routed_through_ingress in ret_routed_through_ingress.items():
			converted_ret_routed_through_ingress[prefix_i] = {}
			for ug,popp in this_routed_through_ingress.items():
				# This might be empty if we're loading state from future iterations where we've already deleted a UG
				# So we do "get"
				for pseudo_ug in self.og_ugs_to_new_ugs.get(ug, []): 
					converted_ret_routed_through_ingress[prefix_i][pseudo_ug] = popp

		return converted_ret_routed_through_ingress

	def enforce_measured_prefs(self, routed_through_ingress, actives):
		### Saves information about which ingresses beat other ingresses for each user
		### Sends this information to worker bees
		for ui, ug in enumerate(self.ugs):
			for prefix_i in routed_through_ingress:
				routed_ingress = routed_through_ingress[prefix_i].get(ug)
				if routed_ingress is None:
					# no path, nothing learned
					continue
				else:
					# observation indidates for this user group what the winning ingress was, 
					# and what the active&available ingresses were
					other_available = get_intersection(self.ug_perfs[ug], actives[prefix_i])
					other_available = get_difference(other_available, [routed_ingress])

					# print(routed_ingress)
					# print(other_available)
					routed_ingress_obj = self.measured_prefs[ug].get(routed_ingress)

					# routed_ingress_obj.print()
					for beaten_ingress in other_available:
						beaten_ingress_obj = self.measured_prefs[ug].get(beaten_ingress)
						# beaten_ingress_obj.print()	
						try:
							beaten_ingress_obj.add_parent(routed_ingress_obj)
						except AttributeError:
							print("UG {} trying to add parent {} to {} but not in measured prefs, perfs: {}. popps in MP :{}".format(ug, routed_ingress, beaten_ingress, self.ug_perfs[ug], list(self.measured_prefs[ug])))
							exit(0)
						routed_ingress_obj.add_child(beaten_ingress_obj)
						self.parent_tracker[ug,beaten_ingress,routed_ingress] = True
						try:
							self.calc_cache.all_caches['parents_on'][ug][beaten_ingress,routed_ingress] = None
						except KeyError:
							self.calc_cache.all_caches['parents_on'][ug] = {(beaten_ingress,routed_ingress): None}
		self.update_parent_tracker_workers()
	
	def update_parent_tracker_workers(self):	
		## Update workers about new parent tracker information
		try:
			self.worker_manager
		except AttributeError:
			return # no workers to update
		msgs = {}
		for worker in self.worker_manager.worker_sockets:
			this_deployment_ugs = self.ugs
			sub_cache = {}
			for this_deployment_ug in this_deployment_ugs:
				try:
					sub_cache[this_deployment_ug] = self.calc_cache.all_caches['parents_on'][this_deployment_ug] 
				except KeyError:
					pass
			msgs[worker] = pickle.dumps(('update_parent_tracker', sub_cache))
		self.worker_manager.send_messages_workers(msgs)

	def measure_ingresses(self, a, **kwargs):
		"""Between rounds, measure ingresses from users to deployment given advertisement a."""
		### i.e., this is an actual advertisement measurement, we should aim to limit these :)
		self.enforce_loaded_rwmw()

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
		self.calculate_user_choice(a, get_ug_catchments=True, **kwargs)

		a = threshold_a(a)
		routed_through_ingress, actives = self.calculate_ground_truth_ingress(a, verb=True, **kwargs)

		self.enforce_measured_prefs(routed_through_ingress, actives)
		self.measured[tuple(a.flatten())] = None
			
	def actual_nonconvex_objective(self, a, **kwargs):
		# Don't approximate the L0 norm with anything
		# Use actual latencies as if we were to really measure all the paths

		self.enforce_loaded_rwmw()		

		c_peering = 1
		has_peering = (np.sum(a,axis=1) > 0).astype(np.int32)
		# cost for different peerings may be different
		cost_peerings = np.sum(np.dot(has_peering,c_peering*np.ones(has_peering.shape)))
		
		# cost for different prefs likely not different
		cost_prefs = self.prefix_cost(a)
		norm_penalty = cost_peerings + cost_prefs
		latency_benefit = kwargs.get('latency_benefit_precalc', self.get_ground_truth_latency_benefit(a, **kwargs))
		if self.gamma > 0:
			resilience_benefit = kwargs.get('resilience_benefit_precalc', self.get_ground_truth_resilience_benefit(a, **kwargs))
		else:
			resilience_benefit = 0

		if self.gamma <= 1:
			benefit = latency_benefit + self.gamma * resilience_benefit
		else:
			benefit = 1 / self.gamma * latency_benefit + resilience_benefit

		obj = self.lambduh * norm_penalty - benefit

		return obj

	def measured_objective(self, a, **kwargs):
		## Approximate L0 norm with whatever approximation we're using
		## Use actual latencies as if we were to really measure all the paths		
		## (Here we actually execute the advertisement)
		self.enforce_loaded_rwmw()
		norm_penalty = self.advertisement_cost(a)
		latency_benefit = kwargs.get('latency_benefit_precalc', self.get_ground_truth_latency_benefit(a, **kwargs))
		if self.gamma > 0 and kwargs.get('use_resilience', True):
			resilience_benefit = kwargs.get('resilience_benefit_precalc', self.get_ground_truth_resilience_benefit(a, **kwargs))
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

	def set_save_run_dir(self, **kwargs):
		if kwargs.get('save_run_dir', None) is None:
			## initialize save dir
			t_start = int(time.time())
			self.save_run_dir = os.path.join(RUN_DIR, '{}-{}-sparse'.format(t_start, self.dpsize))
			if not os.path.exists(self.save_run_dir):
				call("mkdir {}".format(self.save_run_dir), shell=True)
			if not self.simulated:
				self.get_realworld_measure_wrapper()
		else:
			if RUN_DIR not in kwargs.get('save_run_dir'):
				self.save_run_dir = os.path.join(RUN_DIR, kwargs.get('save_run_dir'))
			else:
				self.save_run_dir = kwargs.get('save_run_dir')
		self.optimal_under_failure_cache_fn = os.path.join(self.save_run_dir, 'optimal_solution_under_failure.pkl')


