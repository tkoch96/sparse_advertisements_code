import geopy.distance, numpy as np, copy, time
from helpers import *
from optimal_adv_wrapper import Optimal_Adv_Wrapper

def vol_to_val(vol):
	# return np.log10(vol + 1)
	return vol

def reduce_to_score(improvements_vols, skip_log=False):
	# score is ~ sum( log(weight) * improvement_in_ms )
	improvements_vols = np.array(improvements_vols)
	if not skip_log: # speedup
		improvements_vols[:,1] = vol_to_val(improvements_vols[:,1])
	return np.sum(np.prod(improvements_vols,axis=1))

def calc_improvements_by_pop_peer(ug_perfs, ug_anycast_perfs, ug_to_vol, being_adved, **kwargs):
	# returns popp -> UG -> improvement
	improvements_by_pop_peer = {}
	ug_decisions = {}
	for ug in ug_perfs:
		if 'anycast' in being_adved:
			ug_perfs[ug]['anycast'] = ug_anycast_perfs[ug]
		vol = ug_to_vol[ug]
		# get best performing ingress
		reachable = get_intersection(being_adved, ug_perfs[ug])
		best_adved_i = np.argmin(np.array([ug_perfs[ug][u] for u in reachable]))
		best_adved = reachable[best_adved_i]
		ug_decisions[ug] = best_adved
		# not being adved
		for u in get_difference(ug_perfs[ug], being_adved):
			diff = ug_perfs[ug][best_adved] - ug_perfs[ug][u]
			pop,peer = u
			try:
				improvements_by_pop_peer[pop,peer]
			except KeyError:
				improvements_by_pop_peer[pop,peer] = {}
			improvements_by_pop_peer[pop,peer][ug] = (np.maximum(diff,0),vol)
	return improvements_by_pop_peer, ug_decisions

def calc_improvements(improvements_by_pop_peer, **kwargs):
	# returns improvements by popp, averaged over UGs
	improvements = {}
	for poppeer in improvements_by_pop_peer:
		these_improvements = {k:v[0] for k,v in improvements_by_pop_peer[poppeer].items()}
		these_vols = {k:v[1] for k,v in improvements_by_pop_peer[poppeer].items()}
		weighted_improvements = [(these_improvements[k], these_vols[k]) for k in these_improvements]
		score = reduce_to_score(weighted_improvements)
		improvements[poppeer] = score
	return improvements


class Painter_Adv_Solver(Optimal_Adv_Wrapper):
	def __init__(self, *args, **kwargs):
		super().__init__(*args,**kwargs)
		# distance between pops
		self.pop_dist_cache = {}
		for popi in self.pop_to_loc:
			for popj in self.pop_to_loc:
				self.pop_dist_cache[popi,popj] = geopy.distance.geodesic(self.pop_to_loc[popi],self.pop_to_loc[popj]).km
		self.calc_popp_to_ug()
		self.stopping_condition = lambda el : el[0] > self.max_n_iter or el[1] < self.epsilon
		self.path_measures = 0 

	def calc_popp_to_ug(self):
		self.popp_to_ug = {}
		for ug in list(self.ug_perfs):
			for popp in self.ug_perfs[ug]:
				try:
					self.popp_to_ug[popp].append(ug)
				except KeyError:
					self.popp_to_ug[popp] = [ug]

	def painter_advs_to_sparse_advs(self, advs, **kwargs):
		painter_budget = self.n_prefixes - 1
		painter_adv = np.zeros((self.n_popp, painter_budget + 1))
		painter_adv[:,0] = 1
		for prefix_i in advs[painter_budget]: # 1 - based
			for popp in advs[painter_budget][prefix_i]:
				painter_adv[self.popp_to_ind[popp],prefix_i] = 1
		return painter_adv

	def stop_tracker(self, **kwargs):
		delta_alpha = .7
		self.last_objective = self.obj
		self.obj = self.measured_objective(self.optimization_advertisement,
			use_resilience=False, mode='best', **kwargs)
		self.rolling_delta = (1 - delta_alpha) * self.rolling_delta + delta_alpha * np.abs(self.obj - self.last_objective)
		print("PAINTER ITER {}, RD {}".format(self.iter, self.rolling_delta))
		self.stop = self.stopping_condition([self.iter, self.rolling_delta])

	def _get_new_improvements(self, being_adved, **kwargs):
		improvements, ug_decisions = calc_improvements_by_pop_peer(self.ug_perfs, self.ug_anycast_perfs,
			self.ug_to_vol, being_adved, **kwargs)
		improvements_average = calc_improvements(improvements, **kwargs)
		return improvements, improvements_average, ug_decisions

	def one_per_pop(self, **kwargs):
		### Doesn't use actual hybridcast performance
		self.solution_type = 'sparse'
		if not self.simulated:
			self.get_realworld_measure_wrapper()
		all_advs = []
		advs = {}
		self.budget = self.n_prefixes - 1
		_, improvements, _ = self._get_new_improvements(['anycast'], **kwargs)
		def to_by_pop(improvs):
			by_pop = {}
			for k,v in improvs.items():
				try:
					by_pop[k[0]].append(v)
				except KeyError:
					by_pop[k[0]] = [v]
			for pop,v in by_pop.items():
				by_pop[pop] = np.mean(np.array(v))
			return by_pop

		pops = self.pops
		advs_by_pfxi = {}
		ret_advs_obj = {}
		for prefix_i in range(self.budget):
			advs_by_pfxi[prefix_i + 1] = {}
			# pick pop that will maximize average improvement
			if pops != []: # no more pops
				_, improvements, _ = self._get_new_improvements(['anycast'] + all_advs, **kwargs)
				improvements_by_pop = to_by_pop(improvements)
				best_pop = pops[np.argmax(np.array([improvements_by_pop[k] for k in pops]))]
				corresponding_pop_peers = list([k for k in self.popps if k[0] == best_pop])
				advs_by_pfxi[prefix_i + 1] = corresponding_pop_peers
				all_advs += corresponding_pop_peers
				pops = get_difference(pops, [best_pop])
							
			ret_advs_obj[prefix_i + 1] = {}
			for pfx, popps in advs_by_pfxi.items():
				ret_advs_obj[prefix_i + 1][pfx] = copy.copy(popps)

		self.optimization_advertisement = self.painter_advs_to_sparse_advs(ret_advs_obj)
		self.optimization_advertisement_representation = {}
		for poppi,prefi in zip(*np.where(threshold_a(self.optimization_advertisement))):
			self.optimization_advertisement_representation[self.popps[poppi], prefi] = None
				
		self.advs = ret_advs_obj

	def painter_v5(self, **kwargs):
		### Wraps painter_v4 with learning preferences
		print("Initial greedy advertisement comuptation")
		self.solution_type = 'painter'
		if not self.simulated:
			self.get_realworld_measure_wrapper()
		advs = self.painter_v4(**kwargs)
		self.advs = advs
		self.optimization_advertisement = self.painter_advs_to_sparse_advs(advs)
		# if os.path.exists('painter_solution.pkl'):
		# 	old_painter_solution = pickle.load(open('painter_solution.pkl','rb'))
		# 	print(self.optimization_advertisement.shape)
		# 	print(old_painter_solution.shape)
		# 	print(np.array_equal(old_painter_solution, self.optimization_advertisement))
		# 	print(np.where(old_painter_solution-self.optimization_advertisement))
		# 	exit(0)
		# else:
		# 	pickle.dump(self.optimization_advertisement, open('painter_solution.pkl','wb'))

		self.last_advertisement = self.painter_advs_to_sparse_advs(advs)

		save_verb = copy.copy(self.verbose)
		self.verbose = False

		self.iter = 0
		if not self.simulated:
			self.calculate_ground_truth_ingress(self.optimization_advertisement, measurement_file_prefix='painter_{}'.format(self.iter))
		self.obj = self.measured_objective(self.optimization_advertisement,
			use_resilience=False, mode='best', measurement_file_prefix='painter_{}'.format(self.iter)) # technically this is a measurement, uncounted
		self.stop = False
		self.rolling_delta = 10
		self.path_measures = 0 
		self.clear_caches()
		ts = time.time()
		print("Starting painter computation")
		while not self.stop:
			# print("Measuring ingresses")
			## conduct measurement with this advertisement strategy
			if not self.simulated:
				self.calculate_ground_truth_ingress(self.optimization_advertisement, measurement_file_prefix='painter_{}'.format(self.iter))
			self.measure_ingresses(self.optimization_advertisement, measurement_file_prefix='painter_{}'.format(self.iter))
			## recalc painter decision with new information
			# print("Solving greedy allocation")
			advs = self.painter_v4(**kwargs)
			self.last_advertisement = copy.deepcopy(self.optimization_advertisement)
			self.optimization_advertisement = self.painter_advs_to_sparse_advs(advs)
			self.optimization_advertisement_representation = {}
			for poppi,prefi in zip(*np.where(threshold_a(self.optimization_advertisement))):
				self.optimization_advertisement_representation[self.popps[poppi], prefi] = None
			self.stop_tracker(measurement_file_prefix='painter_{}'.format(self.iter))
			self.iter += 1

			print("Painter iteration {}, {}s per iter".format(self.iter,
				(time.time() - ts) / (self.iter+1)))

			self.advs = advs
		if not self.simulated:
			self.calculate_ground_truth_ingress(self.optimization_advertisement, measurement_file_prefix='painter_{}'.format(self.iter))

		self.verbose = save_verb

	def painter_v4(self, **kwargs):
		# print("Solving for Painter v4 solution")

		self.budget = self.n_prefixes - 1
		ret_advs_obj = {} # returned object
		advs_by_pfxi = {} # stores pfx->popp mappings during calculation
		# conflicting distance / minimum reuse distance
		cd = kwargs.get('cd')
		assert cd is not None

		pop_to_metro_dist_cache = {}
		to_del = []
		for ug in self.ug_perfs:
			if self.ug_perfs[ug] == {}: to_del.append(ug)
		for ug in to_del: del self.ug_perfs[ug]
		self.calc_popp_to_ug()
		all_popp = list(self.popp_to_ug)
		popp_to_i = {popp:i for i,popp in enumerate(all_popp)}

		def get_available_options_ug(ug, being_adved):
			# An option is 'available' to an ug if its
			# (a) being advertised
			# (b) the ug has a route to it
			# (c) the corresponding pop for the option is not inflated by more than conflicting distance
			# and if we include a peer somewhere, we have to include it everywhere (worst case intra AS assumption)
			all_available_options = get_intersection(being_adved, self.ug_perfs[ug]) # checks (a) and (b)
			if len(all_available_options) == 0: return []

			# Remove impossible options based on our known information
			for popp in get_intersection(self.measured_prefs[ug], all_available_options):
				node = self.measured_prefs[ug][popp]
				if node.has_parent(all_available_options):
					node.kill()
			possible_peers = [popp for popp, node in self.measured_prefs[ug].items() if node.alive]
			available_options_limited = get_intersection(all_available_options, possible_peers)
			## Reset for next time
			for popp in list(self.measured_prefs[ug]):
				self.measured_prefs[ug][popp].alive = True

			## It could be that A > B and B > A if our model is wrong, which will then result in everything dying
			if len(available_options_limited) == 0:
				print("Note -- probably a cycle for {}".format(ug))
				return []

			available_options = []
			dists = []
			for popp in available_options_limited:
				pop,p = popp
				try:
					d = pop_to_metro_dist_cache[self.metro_loc[ug[0]], self.pop_to_loc[pop]]
				except KeyError:
					d = geopy.distance.geodesic(self.metro_loc[ug[0]], self.pop_to_loc[pop]).km
					pop_to_metro_dist_cache[self.metro_loc[ug[0]], self.pop_to_loc[pop]] = d
				dists.append(d)
			min_pop = available_options_limited[np.argmin(np.array(dists))][0]
			available_options = [popp for popp in available_options_limited if \
				self.pop_dist_cache[popp[0], min_pop] < cd] # check (c)
			available_peers = [popp[1] for popp in available_options]
			# Also include popp's too far away for which the same peer is just as close
			ret_available_options = [popp for popp in available_options_limited if \
				popp[1] in available_peers] # final note
			
			return ret_available_options

		# maybe store ug -> pfx perfs (prepoulate "perf" thing)
		ug_to_prefix_perfs = {ug: 1000000*np.ones(self.budget) for ug in self.ug_perfs}
		self.freeze_up_to = -1 # freeze performances up to and including this prefix
		self.impvs_vols = {popp: np.array([[0, vol_to_val(self.ug_to_vol[ug])] for ug in self.popp_to_ug[popp]]) for popp in self.popp_to_ug}
		self.impvs_not_vols = {popp: np.zeros(len(self.popp_to_ug[popp])) for popp in self.popp_to_ug}
		self.popp_ug_inds = {popp: {ug: i for i,ug in enumerate(self.popp_to_ug[popp])} for popp in self.popp_to_ug} ## !
		self.improvement_scores = np.zeros(len(all_popp))
		self.worst_case_improvements = np.zeros(len(all_popp))

		all_ug = list(self.ug_perfs)
		self.ug_i = np.arange(len(all_ug))
		self.ug_to_i = {ug:i for i,ug in enumerate(all_ug)}
		self.popp_to_ug_i = {} ## !
		for popp in all_popp:
			self.popp_to_ug_i[popp] = [self.ug_to_i[ug] for ug in self.popp_to_ug[popp]]

		# Stores improvements each ug would see on a brand new prefix
		self.frozen_impv_by_popp = None

		self.ug_current_perf = {ug:self.ug_anycast_perfs[ug] for ug in self.ug_perfs}
		self.ug_best_perf = {ug: self.ug_anycast_perfs[ug] for ug in self.ug_perfs}
		for ug in self.ug_perfs:
			for u in self.ug_perfs[ug]:
				self.ug_best_perf[ug] = np.minimum(self.ug_best_perf[ug], self.ug_perfs[ug][u])
		ug_i_to_not_consider = []


		def calc_mean_improvements(advs_by_pfx, new_pfxi, last_changed_popp, verb=False):
			npfx = len(advs_by_pfx)

			reuse_prefix = len(advs_by_pfx[new_pfxi]) > 0
			if not reuse_prefix:
				# we've started on a new prefix, update that we're freezing performance calculations
				# for all previous prefixes
				self.freeze_up_to = new_pfxi - 1

			popp_changed = {}
			# We want to calculate updated improvements for <popp,ug> pairs for which their improvement would have changed 
			# since last time
			if last_changed_popp is None: # NEW PREFIX
				# Reset improvements to be improvements each ug would expect for a brand new prefix
				# This bootstraps our improvement calculations so that we save time
				if self.frozen_impv_by_popp is not None:
					for popp in self.popp_to_ug:
						# improvements for popp's we added last time WOULD have changed
						if popp in advs_by_pfx[new_pfxi-1]: continue

						for ug in self.popp_to_ug[popp]:
							impv = self.frozen_impv_by_popp[popp][self.popp_ug_inds[popp][ug]]
							if impv != self.impvs_not_vols[popp][self.popp_ug_inds[popp][ug]]:
								popp_changed[popp] = None
								self.impvs_vols[popp][self.popp_ug_inds[popp][ug]][0] = impv
								self.impvs_not_vols[popp][self.popp_ug_inds[popp][ug]] = impv
						self.improvement_scores[popp_to_i[popp]] = np.sum(np.prod(self.impvs_vols[popp], axis=1))

				if new_pfxi > 0:
					# uges that had some perf at begining of last prefix, but had different perf by the end
					# Are the ones whose improvement for each popp would have changed
					# So we do improvement at start of last prefix + account for deltas over the last prefix
					# ug_to_loop_i = self.ug_i[self.ug_perf_start_pref != self.ug_perf_end_pref]
					# self.ug_perf_start_pref = copy.copy(self.ug_perf_end_pref)

					ug_to_loop_i = set()
					for popp in advs_by_pfx[new_pfxi-1]:
						ug_to_loop_i = ug_to_loop_i.union(self.popp_to_ug_i[popp])
				else:
					ug_to_loop_i = range(len(all_ug))
					self.ug_perf_start_pref = np.zeros((len(all_ug)))
			else:
				# potential improvements change for uges with paths to popps 
				ug_to_loop_i =self.popp_to_ug_i[last_changed_popp]

			if verb:
				print("Looping over {} ug".format(len(ug_to_loop_i)))
				print([all_ug[i] for i in ug_to_loop_i])
			for ug_i in get_difference(ug_to_loop_i, ug_i_to_not_consider): # looping over ug indices
				ug = all_ug[ug_i]
				## TO CALC avg(perf with new peer,pfx) - avg(perf without) for each peer
				# Current performance at current prefix
				pfxi = new_pfxi
				available_options = get_available_options_ug(ug, advs_by_pfx[pfxi])
				if len(available_options) > 0:
					possible_latencies = np.array([self.ug_perfs[ug][u] for u in available_options])
					old_perf = np.mean(possible_latencies)
				else:
					old_perf = self.ug_anycast_perfs[ug]
				old_perf = np.minimum(self.ug_anycast_perfs[ug], old_perf)

				if self.freeze_up_to > -1:
					# performance from already-advertised prefixes
					old_perf = np.minimum(np.min(ug_to_prefix_perfs[ug][0:self.freeze_up_to+1]), old_perf)

				# Cache for later
				ug_to_prefix_perfs[ug][pfxi] = old_perf

				for popp in self.ug_perfs[ug]:
					if reuse_prefix:
						# Calculate new average performance for this prefix
						available_options = get_available_options_ug(ug, advs_by_pfx[pfxi] + [popp])
						if len(available_options) > 0:
							possible_latencies = np.array([self.ug_perfs[ug][u] for u in available_options])
							new_perf = np.mean(possible_latencies)
						else:
							# everything conflicts, performance is same as the old
							new_perf = old_perf
					else:
						# PAINTER will select the new prefix if it's better
						# the new prefixes performance would be ug_perfs -> ug -> popp
						new_perf = self.ug_perfs[ug][popp]
					# Latency is minimum of: anycast latency, perf at this prefix, perfs at all previous prefixes
					new_perf = np.minimum(self.ug_anycast_perfs[ug], new_perf)
					if self.freeze_up_to > -1:
						new_perf = np.minimum(np.min(ug_to_prefix_perfs[ug][0:self.freeze_up_to+1]), new_perf)

					# Sanity check -- Improvement should be positive if old is higher latency
					popp_changed[popp] = None
					impv = old_perf - new_perf
					self.impvs_vols[popp][self.popp_ug_inds[popp][ug]][0] = impv
					self.impvs_not_vols[popp][self.popp_ug_inds[popp][ug]] = impv


			# New prefix --- update improvements each ug would see 
			if last_changed_popp is None:
				self.frozen_impv_by_popp = copy.deepcopy(self.impvs_not_vols)

			# We have latency improvement by popp,ug; convert to average improvement per popp over uges
			for popp in popp_changed:
				self.worst_case_improvements[popp_to_i[popp]] = np.min(self.impvs_not_vols[popp])
				self.improvement_scores[popp_to_i[popp]] = np.sum(np.prod(self.impvs_vols[popp], axis=1))
		best_popp = None
		completely_done = False


		#### Painter v4 solutions have the property that solutions at a higher budget contain 
		### solutions at a lower budget, plus additional advertisements
		for prefix_i in range(self.budget):
			# print("MI : {}, PFX : {}".format(cd, prefix_i))
			last_added_popp = None
			# Reset worst case improvements since we're working on a new prefix
			self.worst_case_improvements = np.zeros(len(all_popp))

			remaining_possible_benefit = None

			advs_by_pfxi[prefix_i] = []
			this_pfx_i = 0
			while True: # keep advertising until stopping conditions are met 
				if completely_done: break
				# print("Prefix: {}, it: {}".format(prefix_i, this_pfx_i))
				calc_mean_improvements(advs_by_pfxi, prefix_i, last_added_popp)
				
				if remaining_possible_benefit is None:
					# First advertisement for this prefix, calculate max benefit
					remaining_possible_benefit = np.sum(self.improvement_scores)

				if remaining_possible_benefit == 0 and this_pfx_i == 0:
					# absolutely no more benefit to get, done allocating
					completely_done = True
					break

				ranked_peers = np.argsort(-1 * self.improvement_scores)
				# a peer ~could~ flip from negative to positive, so this check isn't awesome
				# but its probably pretty close to the truth
				max_mean_improvement_left_to_get = np.sum(self.improvement_scores[self.improvement_scores>0])
				if max_mean_improvement_left_to_get < .01 * remaining_possible_benefit:
					break

				found_peer_to_add = False
				for next_best_peer_i in ranked_peers: # search through peers to find one to add
					# I don't know if this would ever happen, but just a sanity check
					next_best_popp = all_popp[next_best_peer_i]
					if self.improvement_scores[next_best_peer_i] > 0:
						# necessary conditions
						# (a) peer must (on average) add some benefit
						# (b) the worst case average performance for an ug is above an allowable threshold
						found_peer_to_add = True
						best_popp = next_best_popp
							
						break
				this_pfx_i += 1
				if not found_peer_to_add: break

				# Add best popp to advs for this prefix
				advs_by_pfxi[prefix_i].append(best_popp)
				last_added_popp = best_popp

			if self.freeze_up_to > 0:
				# Find uges that get their best performance already
				ug_to_update = set()
				for popp in advs_by_pfxi[prefix_i]:
					ug_to_update = ug_to_update.union(self.popp_to_ug_i[popp])
				
				for ug_i in ug_to_update:
					ug = all_ug[ug_i]
					# If an ug already gets its best performance, dont consider it
					if np.min(ug_to_prefix_perfs[ug][0:self.freeze_up_to]) == self.ug_best_perf[ug]:
						ug_i_to_not_consider.append(ug_i)

			# Change to 1-indexed for the return object
			ret_advs_obj[prefix_i + 1] = {}
			for pfx, popps in advs_by_pfxi.items():
				ret_advs_obj[prefix_i + 1][pfx + 1] = copy.copy(popps)

		return ret_advs_obj


