import numpy as np
from optimal_adv_wrapper import Optimal_Adv_Wrapper
from helpers import *

class Anyopt_Adv_Solver(Optimal_Adv_Wrapper):
	### Solves for optimal advertisement using ideas from here
	## https://dl.acm.org/doi/pdf/10.1145/3452296.3472935
	def __init__(self, *args, **kwargs):
		super().__init__(*args,**kwargs)

		self.n_montecarlo = 1000

	def solve(self):
		## First, anyopt learns all the pairwise preferences from every client to every transit
		## provider. anyopt then solve for the optimal subset of those popps to enable using monte-carlo?
		## i.e., simulate 1000 scenarios and pick the best one
		## Then add remaining non-transit popps one by one (randomly). If popp reduces latency,
		## keep it. ('one-pass' approach). We randomly choose the prefix against which to
		## measure the peer

		# assume apriori that we predict provider catchments with 100% confidence,
		# since anyopt has a methodology that works pretty well
		choice_arr = list(range(self.n_provider_popps))

		best_adv = None
		for test_i in range(self.n_montecarlo):
			random_adv = np.zeros((self.n_popp, self.n_prefixes))
			for prefi in range(self.n_prefixes):
				randomly_active = np.random.choice(choice_arr, 
					size=np.random.randint(self.n_provider_popps)+1,replace=False)
				randomly_active = [self.provider_popps[ra] for ra in randomly_active]
				randomly_active = np.array([self.popp_to_ind[ra] for ra in randomly_active])
				random_adv[randomly_active, prefi] = 1
			this_adv_obj = self.measured_objective(random_adv)
			if best_adv is None:
				best_adv = random_adv
				best_obj = this_adv_obj
			if this_adv_obj < best_obj:
				best_adv = random_adv
				best_obj = this_adv_obj

		self.stop = False
		self.rolling_delta = 10
		self.iter = 0
		non_transit_popps = get_difference(self.popps, self.provider_popps)
		np.random.shuffle(non_transit_popps)
		popp_i = 0
		keep_popp = []
		## pairwise comparisons among transit, and then measuring each peer separately
		self.path_measures = (int(np.log2(self.n_provider_popps))  + \
			len(non_transit_popps)) // self.n_prefixes
		while popp_i < len(non_transit_popps):
			popp = non_transit_popps[popp_i]
			popp_i += 1
			# In practice we could parallelize this across prefixes
			# but for now it's easier to implement this way (code reuse)
			best_adv[self.popp_to_ind[popp],:] = 1
			obj_after = self.measured_objective(best_adv)
			if obj_after < best_obj:
				keep_popp.append(popp)
			best_adv[self.popp_to_ind[popp],:] = 0
		## "one - pass" method
		for popp in keep_popp:
			best_adv[self.popp_to_ind[popp],0] = 1
		self.obj = self.measured_objective(best_adv) # technically this is a measurement, uncounted

		self.advs = best_adv