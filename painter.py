import geopy.distance, numpy as np, copy
from helpers import *
from optimal_adv_wrapper import Optimal_Adv_Wrapper

def vol_to_val(vol):
	return np.log10(vol + 1)

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
				if popp == 'anycast': continue
				if popp[1] in ['regional','hybricast']: continue
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

	def stop_tracker(self, advs):
		delta_alpha = .7
		self.last_objective = self.obj
		self.obj = self.measured_objective(self.painter_advs_to_sparse_advs(advs))
		self.rolling_delta = (1 - delta_alpha) * self.rolling_delta + delta_alpha * np.abs(self.obj - self.last_objective)
		self.stop = self.stopping_condition([self.iter, self.rolling_delta])

	def painter_v5(self, **kwargs):
		### Wraps painter_v4 with learning preferences
		advs = self.painter_v4(**kwargs)
		self.obj = self.measured_objective(self.painter_advs_to_sparse_advs(advs)) # technically this is a measurement, uncounted
		self.stop = False
		self.rolling_delta = 10
		self.iter = 0
		self.path_measures = 0 
		self.clear_caches()
		while not self.stop:
			## conduct measurement with this advertisement strategy
			self.measure_ingresses(self.painter_advs_to_sparse_advs(advs))
			## recalc painter decision with new information
			advs = self.painter_v4(**kwargs)
			self.stop_tracker(advs)
			self.iter += 1

			self.advs = advs

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
			for u in list(self.ug_perfs[ug]):
				if u == 'anycast': continue
				if u[1] in ['hybridcast','regional']: del self.ug_perfs[ug][u]
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

			ui = self.ug_to_ind[ug]
			# Remove impossible options based on our known information
			available_popp_inds = [self.popp_to_ind[popp] for popp in all_available_options]
			# print(available_popp_inds)
			for nodeid in get_intersection(self.measured_prefs[ui], available_popp_inds):
				node = self.measured_prefs[ui][nodeid]
				if node.has_parent(available_popp_inds):
					node.kill()
			possible_peers = [nodeid for nodeid, node in self.measured_prefs[ui].items() if node.alive]
			# print(possible_peers)
			available_options_limited = get_intersection(available_popp_inds, possible_peers)
			# print(available_options_limited)
			# print("\n")
			# Reset for next time
			for nodeid in list(self.measured_prefs[ui]):
				self.measured_prefs[ui][nodeid].alive = True

			available_options = []
			dists = []
			for popp_i in available_options_limited:
				pop,p = self.popps[popp_i]
				try:
					d = pop_to_metro_dist_cache[self.metro_loc[ug[0]], self.pop_to_loc[pop]]
				except KeyError:
					d = geopy.distance.geodesic(self.metro_loc[ug[0]], self.pop_to_loc[pop]).km
					pop_to_metro_dist_cache[self.metro_loc[ug[0]], self.pop_to_loc[pop]] = d
				dists.append(d)
			min_pop = self.popps[available_options_limited[np.argmin(np.array(dists))]][0]
			available_options = [self.popps[poppi] for poppi in available_options_limited if \
				self.pop_dist_cache[self.popps[poppi][0], min_pop] < cd] # check (c)
			available_peers = [popp[1] for popp in available_options]
			# Also include popp's too far away for which the same peer is just as close
			ret_available_options = [self.popps[poppi] for poppi in available_options_limited if \
				self.popps[poppi][1] in available_peers] # final note
			
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

		self.ug_using_pref = {i:-1 for i in range(len(all_ug))}
		self.ug_current_perf = {ug:self.ug_perfs[ug]['anycast'] for ug in self.ug_perfs}
		self.ug_best_perf = {ug: self.ug_perfs[ug]['anycast'] for ug in self.ug_perfs}
		for ug in self.ug_perfs:
			for u in self.ug_perfs[ug]:
				if u == 'anycast': continue
				if u[1] in ['regional','hybridcast']: continue
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
			# for ug_i in ug_to_loop_i: # looping over ug indices
				ug = all_ug[ug_i]
				## TO CALC avg(perf with new peer,pfx) - avg(perf without) for each peer
				# Current performance at current prefix
				pfxi = new_pfxi
				available_options = get_available_options_ug(ug, advs_by_pfx[pfxi])
				if len(available_options) > 0:
					possible_latencies = np.array([self.ug_perfs[ug][u] for u in available_options])
					old_perf = np.mean(possible_latencies)
				else:
					old_perf = self.ug_perfs[ug]['anycast']
				old_perf = np.minimum(self.ug_perfs[ug]['anycast'], old_perf)

				if self.freeze_up_to > -1:
					# performance from already-advertised prefixes
					old_perf = np.minimum(np.min(ug_to_prefix_perfs[ug][0:self.freeze_up_to+1]), old_perf)

				# Cache for later
				ug_to_prefix_perfs[ug][pfxi] = old_perf

				for popp in self.ug_perfs[ug]:
					if popp == 'anycast': continue
					if reuse_prefix:
						# Calculate new average performance for this prefix
						available_options = get_available_options_ug(ug, advs_by_pfx[pfxi] + [popp])
						if len(available_options) > 0:
							possible_latencies = np.array([self.ug_perfs[ug][u] for u in available_options])
							new_perf = np.mean(possible_latencies)
						else:
							# everything conflicts, performance is same as the old
							new_perf = this_pfx_perf
					else:
						# PAINTER will select the new prefix if it's better
						# the new prefixes performance would be ug_perfs -> ug -> popp
						new_perf = self.ug_perfs[ug][popp]
					# Latency is minimum of: anycast latency, perf at this prefix, perfs at all previous prefixes
					new_perf = np.minimum(self.ug_perfs[ug]['anycast'], new_perf)
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
					if next_best_popp == 'anycast': continue 
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