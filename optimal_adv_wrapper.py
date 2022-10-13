import numpy as np
from constants import *
from helpers import *


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
	def __init__(self, deployment, lambduh=1.0, verbose=True,gamma=0,n_prefixes=3):
		self.advertisement_threshold = .5
		self.verbose = verbose
		self.advertisement_cost = self.l1_norm
		self.n_prefixes = n_prefixes
		self.epsilon = .00005 # change in objective less than this -> stop
		self.max_n_iter = 300 # maximum number of learning iterations
		self.lambduh = lambduh # sparsity cost
		self.gamma = gamma # resilience cost


		## VARIOUS CACHES
		self.violates_cache = {}
		self.lb_cache = {}
		self.ground_truth_ingress_cache = {}

		self.update_deployment(deployment)	

	def clear_caches(self):
		## VARIOUS CACHES
		self.violates_cache = {}
		self.lb_cache = {}
		self.ground_truth_ingress_cache = {}
		self.measured_prefs = {ui: {self.popp_to_ind[popp]: Ing_Obj(self.popp_to_ind[popp]) for popp in \
			self.ug_perfs[self.ugs[ui]] if popp != 'anycast'} for ui in range(self.n_ug)}
		self.measured = {}
		self.ingress_probabilities_cache = {}

	def threshold_a(self, a):
		return (a > self.advertisement_threshold).astype(np.float32)
	
	def update_deployment(self, deployment):
		self.ugs = list(deployment['ug_perfs'])
		self.n_ug = len(self.ugs)
		self.ug_to_ind = {ug:i for i,ug in enumerate(self.ugs)}
		self.ug_perfs = deployment['ug_perfs']
		# Shape of the variables
		self.popps = list(set(deployment['popps']))
		self.popp_to_ind = {k:i for i,k in enumerate(self.popps)}
		self.n_popp = len(get_difference(self.popps,['anycast']))
		self.n_providers = deployment['n_providers']
		self.ground_truth_ingress_priorities = deployment['ingress_priorities']

		self.metro_loc = deployment['metro_loc']
		self.pop_to_loc = deployment['pop_to_loc']

		self.ingress_priority_inds = {ug:{self.popp_to_ind[popp]: \
			self.ground_truth_ingress_priorities[ug][popp] for popp in self.ug_perfs[ug] if popp != 'anycast'} \
			for ug in self.ugs}

		# self.link_capacities = np.array([self.graph_md['peering_link_capacities'][str(p)] for p in self.peers])
		self.ug_to_vol = deployment['ug_to_vol']

		# ingress,ug -> stores ~ how many times ingress wins for a ug
		self.n_wins = {ug:{i:0 for i in range(self.n_popp)} for ug in self.ugs}

		self.popp_by_ug_indicator = np.zeros((self.n_popp, self.n_ug))
		self.popp_by_ug_indicator_no_rank = np.zeros((self.n_popp, self.n_ug))
		for ui in range(self.n_ug):
			for popp in self.ug_perfs[self.ugs[ui]]:
				if popp == 'anycast': continue
				self.popp_by_ug_indicator[self.popp_to_ind[popp], ui] = self.n_popp + 1 - self.ingress_priority_inds[self.ugs[ui]][self.popp_to_ind[popp]]
				self.popp_by_ug_indicator_no_rank[self.popp_to_ind[popp],ui] = 1
		self.parent_tracker = np.zeros((self.n_ug,self.n_popp, self.n_popp),dtype=bool)

		self.calculate_user_latency_by_peer()
		self.clear_caches()

	def calculate_user_latency_by_peer(self):
		"""
			Calculate latency from each user to each peer. In practice, you would measure these latencies
			once using Anyopt-type techniques.
			We just simulate latencies randomly (or however).
			This function turns ug -> popp -> lat into 3D array of popp, ug, prefixes for ease of use
		"""
		self.measured_latencies = np.zeros((self.n_popp, len(self.ugs)))
		for ug in self.ugs:
			ugi = self.ug_to_ind[ug]
			for popp in self.popps:
				popp_i = self.popp_to_ind[popp]
				self.measured_latencies[popp_i,ugi] = self.ug_perfs[ug].get(popp, BIG_BAD_VALUE)
		# same for each prefix (ease of calculation later)
		self.measured_latencies = np.tile(np.expand_dims(self.measured_latencies, axis=1), 
			(1, self.n_prefixes, 1))

	def l1_norm(self, a):
		return np.sum(np.abs(a).flatten())

	def get_ground_truth_latency_benefit(self, a):
		### Measures actual latency benefit as if we were to advertise 'a'
		a_effective = self.threshold_a(a)
		
		user_latencies = self.get_ground_truth_user_latencies(a_effective)
		benefit = self.benefit_from_user_latencies(user_latencies)
		return benefit

	def get_ground_truth_user_latencies(self, a):
		#### Measures actual user latencies as if we were to advertise 'a'
		user_latencies = 100 * np.ones((len(self.ugs)))
		routed_through_ingress, _ = self.calculate_ground_truth_ingress(a)
		for prefix_i in range(a.shape[1]):
			for ugi,ug in enumerate(self.ugs):
				routed_ingress = routed_through_ingress[prefix_i].get(ug)
				if routed_ingress is None:
					latency = 100
				else:
					latency = self.ug_perfs[ug][self.popps[routed_ingress]]
				if latency < user_latencies[ugi]: user_latencies[ugi] = latency
		return user_latencies

	def benefit_from_user_latencies(self, user_latencies):
		# sum of the benefits, simple model for benefits is 1 / latency
		user_benefits = -1 * user_latencies
		# average user benefit -- important that this function is not affected by the number of user groups
		return np.mean(user_benefits)

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

		try:
			return self.ground_truth_ingress_cache[tuple(a.flatten())]
		except KeyError:
			pass

		### Somewhat efficient implementation using matrix logic
		routed_through_ingress = {} # prefix -> UG -> ingress index
		actives = {} # prefix -> indices where we adv
		for prefix_i in range(a.shape[1]):
			actives[prefix_i] = np.where(a[:,prefix_i] == 1)[0]
			routed_through_ingress[prefix_i] = {}
			if np.sum(a[:,prefix_i]) == 0:
				continue
			active_popp_indicator = np.tile(np.expand_dims(a[:,prefix_i],axis=1), (1,self.n_ug))
			active_popp_ug_indicator = self.popp_by_ug_indicator * active_popp_indicator
			best_available_options = np.argmax(active_popp_ug_indicator,axis=0)
			for ui, bao in zip(range(self.n_ug), best_available_options):
				if active_popp_ug_indicator[bao,ui] == 0: continue # no route
				routed_through_ingress[prefix_i][self.ugs[ui]] = bao
		self.ground_truth_ingress_cache[tuple(a.flatten())] = (routed_through_ingress,actives)
		return routed_through_ingress, actives

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
		##### (JUST RESET ALL APPROPRIATE CACHES)
		self.lb_cache = {}
		self.ingress_probabilities_cache = {}

		self.path_measures += 1

		a = self.threshold_a(a)
		routed_through_ingress, actives = self.calculate_ground_truth_ingress(a)

		## Need to clear the violates cache since new information from a measurement will influence our belief
		## about whether orderings are impossible
		tups = list(self.violates_cache)
		for k in tups:
			if self.violates_cache[k]: continue
			for prefix_i in range(a.shape[1]):
				if len(get_intersection(k[2],actives[prefix_i])) > 0:
					del self.violates_cache[k]
					break

		# print("Measuring : \n{}".format(a))
		self.measured[tuple(a.flatten())] = None
		for ui, ug in enumerate(self.ugs):
			for prefix_i in range(a.shape[1]):
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

	def actual_nonconvex_objective(self, a):
		# Don't approximate the L0 norm with anything
		# Use actual latencies as if we were to really measure all the paths		

		c_pref = 1
		c_peering = 1

		has_peering = (np.sum(a,axis=1) > 0).astype(np.int32)
		has_pref = (np.sum(a,axis=0) > 0).astype(np.int32)

		# cost for different peerings may be different
		cost_peerings = np.sum(np.dot(has_peering,c_peering*np.ones(has_peering.shape)))
		# cost for different prefs likely not different
		cost_prefs = np.sum(has_pref) * c_peering
		norm_penalty = cost_peerings + cost_prefs
		latency_benefit = self.get_ground_truth_latency_benefit(a)
		resilience_benefit = self.get_ground_truth_resilience_benefit(a)

		return self.lambduh * norm_penalty - (latency_benefit + self.gamma * resilience_benefit)

	def measured_objective(self, a):
		## Approximate L0 norm with whatever approximation we're using
		## Use actual latencies as if we were to really measure all the paths		
		## (Here we actually execute the advertisement)

		norm_penalty = self.advertisement_cost(a)
		latency_benefit = self.get_ground_truth_latency_benefit(a)
		resilience_benefit = self.get_ground_truth_resilience_benefit(a)
		# print("Actual: NP: {}, LB: {}, RB: {}".format(norm_penalty,latency_benefit,resilience_benefit))
		return self.lambduh * norm_penalty - (latency_benefit + self.gamma * resilience_benefit)