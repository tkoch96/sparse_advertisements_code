import numpy as np, numba as nb, pickle, copy
np.setbufsize(262144*8)

from constants import *
from helpers import *
from test_polyphase import sum_pdf_new


remeasure_a = None
try:
	remeasure_a = pickle.load(open('remeasure_a.pkl','rb'))
except:
	pass

#### When these matrices are really big, helps to use numba, but can't work in multiprocessing scenarios
@nb.njit(fastmath=True,parallel=True)
def large_logical_and(arr1,arr2):
	return np.logical_and(arr1,arr2)

class Path_Distribution_Computer:
	def __init__(self, **kwargs):
		self.with_capacity = kwargs.get('with_capacity', False)

		## Latency benefit is -1 * mean latency, so latency benefits must lie in this region
		# what is the right granularity?
		self.lbx = np.linspace(-1*MAX_LATENCY, 0,num=1000)

		self.calc_cache = Calc_Cache()

	def update_deployment(self, deployment):
		self.ugs = list(deployment['ug_perfs'])
		self.n_ug = len(self.ugs)
		self.ug_to_ind = {ug:i for i,ug in enumerate(self.ugs)}
		self.popps = list(set(deployment['popps']))
		self.n_popp = len(get_difference(self.popps,['anycast']))
		self.n_prefixes = np.maximum(2,self.n_popp // 3)
		self.ingress_probabilities = np.zeros((self.n_popp, self.n_prefixes, self.n_ug))
		self.metro_loc = deployment['metro_loc']
		self.pop_to_loc = deployment['pop_to_loc']
		self.popp_to_ind = {k:i for i,k in enumerate(self.popps)}
		self.ug_perfs = deployment['ug_perfs']

		self.ug_to_vol = deployment['ug_to_vol']
		self.ug_vols = np.zeros(self.n_ug)
		for ug, v in self.ug_to_vol.items():
			self.ug_vols[self.ug_to_ind[ug]] = v
		all_vols = list(self.ug_to_vol.values())
		self.vol_x = np.linspace(min(all_vols),max(all_vols))

		self.link_capacities = {self.popp_to_ind[popp]: deployment['link_capacities'][popp] for popp in self.popps}
		self.link_capacities_arr = np.zeros(self.n_popp)
		for poppi, cap in self.link_capacities.items():
			self.link_capacities_arr[poppi] = cap

		self.popp_by_ug_indicator_no_rank = np.zeros((self.n_popp, self.n_ug), dtype=bool)
		for ui in range(self.n_ug):
			for popp in self.ug_perfs[self.ugs[ui]]:
				if popp == 'anycast': continue
				self.popp_by_ug_indicator_no_rank[self.popp_to_ind[popp],ui] = True
		self.parent_tracker = np.zeros((self.n_ug, self.n_popp, self.n_popp), dtype=bool)

	def clear_caches(self):
		self.calc_cache.clear_all_caches()

	def get_ingress_probabilities_by_a_matmul(self, a, verb=False, **kwargs):
		a = threshold_a(a.astype(np.int32))
		if np.array_equal(a, remeasure_a): verb = True
		a_log = a.astype(bool)
		self.ingress_probabilities[:,:,:] = 0
		mprocess = kwargs.get('multiprocess',False)
		for pref_i in range(self.n_prefixes):
			##### WARNING -- if number of UGs and number of popps is the same, there could be ambiguity with the broadcasting
			##### but the likelihood of that event is pretty small
			if np.sum(a[:,pref_i]) == 0:
				continue
			try:
				self.ingress_probabilities[:,pref_i,:] = self.calc_cache.all_caches['ing_prob'][tuple(a_log[:,pref_i].flatten())]
				continue
			except KeyError:
				pass
			these_active_popps = np.expand_dims(a_log[:,pref_i],axis=1)
			# active popps
			tmp_arr = a_log[:,pref_i]
			if mprocess:
				active_parent_indicator = np.logical_and(tmp_arr, self.parent_tracker)
			else: # Numba and multiprocess are not compatible
				active_parent_indicator = large_logical_and(tmp_arr, self.parent_tracker)
			# active_parent_indicator = np.logical_and(tmp_arr, self.parent_tracker)
			# holds ug,popps to delete since they get beaten
			delete_popp_ug_indicator = np.logical_and(these_active_popps, np.any(active_parent_indicator, axis=2).T)
			# UG has route and popp active
			active_popp_ug_indicator = np.logical_and(self.popp_by_ug_indicator_no_rank, these_active_popps)
			# remove popp,ug's that would get beaten #1,0->1 1,1->0,0,1->0,0,0->0
			valid_popp_ug_indicator = np.logical_and(active_popp_ug_indicator,np.logical_not(delete_popp_ug_indicator))
			# now sort based on likelihood
			sortf_arr = {ug:[] for ug in self.ugs}
			active_inds = np.where(valid_popp_ug_indicator > 0)
			for poppi,ugi in zip(active_inds[0],active_inds[1]):
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
			for ug,ds in sortf_arr.items():
				if len(ds) == 0:
					continue
				ui = self.ug_to_ind[ug]
				most_likely_peers = sorted(ds,key=lambda el : el[1])[0:5]

				nmlp = len(most_likely_peers)
				for mlp,_ in most_likely_peers:
					self.ingress_probabilities[mlp,pref_i,ui] = 1 / nmlp
			self.calc_cache.all_caches['ing_prob'][tuple(a_log[:,pref_i].flatten())] = copy.copy(self.ingress_probabilities[:,pref_i,:])

			# for ug in self.ugs:
			# 	# perform a sort on these in particular
			# 	ui = self.ug_to_ind[ug]
			# 	possible_peers = np.where(valid_popp_ug_indicator[:,ui] > 0)[0]
			# 	if len(possible_peers) == 0: continue
			# 	if verb and ui == 0:
			# 		print(possible_peers)					
			# 	most_likely_peers = self.get_n_most_likely_peers_justsort(ug, possible_peers)
			# 	pi = np.zeros((self.n_popp))
			# 	for mlp in most_likely_peers:
			# 		pi[mlp] = 1 / len(most_likely_peers)
			# 	#### TODO -- could incorporate pairwise information here
			# 	# orderings = {}
			# 	# n_combs = np.math.factorial(len(most_likely_peers))
			# 	# for ordering in itertools.permutations(most_likely_peers, len(most_likely_peers)):
			# 	# 	orderings[ordering] = 1 / n_combs # COULD incorporate priors here
			# 	# tot_prob = 1.0
			# 	# ## Calculate the marginal that each ingress wins
			# 	# for ordering in orderings:
			# 	# 	pi[ordering[0]] += orderings[ordering]
			# 	# pi = pi / tot_prob
			# 	self.ingress_probabilities[:,pref_i,ui] = pi

	def latency_benefit(self, a, **kwargs):
		"""Calculates distribution of latency benefit at a given advertisement. Benefit is the sum of 
			benefits across all users."""
		a_effective = threshold_a(a)
		if not kwargs.get('plotit') and not kwargs.get('verb'):
			try:
				return self.calc_cache.all_caches['lb'][tuple(a_effective.flatten())]
			except KeyError:
				pass

		## Dims are path, prefix, user
		self.get_ingress_probabilities_by_a_matmul(a_effective, **kwargs)
		p_mat = self.ingress_probabilities
		p_mat = p_mat / (np.sum(p_mat,axis=0) + 1e-8)
		benefits = self.measured_latency_benefits
		lbx = self.lbx

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

				p_vol_this_ingress = sum_pdf_new(ug_prob_vols_this_ingress).flatten()
				p_link_fails[ingress_i] = np.sum(p_vol_this_ingress[self.vol_x * self.n_ug > self.link_capacities[ingress_i]])
			
			# if kwargs.get('verb'):
			# 	import matplotlib.pyplot as plt
			# 	f,ax = plt.subplots(2)
			# 	ax[0].set_title("Ingress {}".format(ingress_i))
			# 	ax[0].plot(ug_prob_vols_this_ingress)
			# 	ax[1].plot(p_vol_this_ingress)
			# 	plt.show()

		# if kwargs.get('verb'):
		# 	print("PLF: {}".format(p_link_fails))
		
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
				px[0,ui] = 1

		px = px / (np.sum(px,axis=0) + 1e-8)
		## Calculate p(sum(benefits)) which is a convolution of the p(benefits)
		psumx = sum_pdf_new(px)
		# print(px)
		# print(psumx);exit(0)
		### pmf of benefits is now xsumx with probabilities psumx
		xsumx = self.lbx * self.n_ug # possible average benefits across users

		plotit = (kwargs.get('plotit') == True) or np.sum(psumx) < .9 # Checks that this is a probability distribution
		if plotit:
			import matplotlib.pyplot as plt
			# print(a)
			# for _i in range(self.n_prefixes):
			# 	print(benefits[:,_i,:])
			# for _i in range(self.n_prefixes):
			# 	print(p_mat[:,_i,:])
			
			print(np.sum(psumx))
			plt.plot(xsumx * self.n_ug, psumx)
			plt.xlabel("Benefit")
			plt.ylabel("P(Benefit)")
			plt.show()

		benefit = np.sum(xsumx.flatten() * psumx.flatten())
		self.calc_cache.all_caches['lb'][tuple(a_effective.flatten())] = (benefit, (xsumx.flatten(),psumx.flatten()))
		return benefit, (xsumx.flatten(),psumx.flatten())