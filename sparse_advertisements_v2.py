import matplotlib.pyplot as plt, copy, time, numpy as np, itertools, pickle
import networkx as nx
import matplotlib.pyplot as plt, scipy.stats
import sys
np.set_printoptions(threshold=sys.maxsize)
from bgpsim import (
	Announcement,
	ASGraph,
)
from helpers import *
from graph_utils import *

KM_TO_MS = .01
BIG_BAD_VALUE = 1e6 # latency with no path
bp_cache = {}
lb_cache = {}


GRAPH_DIR = "graphs"
TOY_GRAPH_FN = "toy_graph.csv"
TOY_GRAPH_MD_FN = "toy_graph_md.json"

class Sparse_Advertisement_Wrapper:
	def __init__(self, graph_fn=TOY_GRAPH_FN, graph_md_fn=TOY_GRAPH_MD_FN, mu=1.0, verbose=True, cont_grads=False, gaussian_noise_var = .01):
		# (hyper-) parameters
		self.mu = mu
		self.advertisement_threshold = .5
		self.epsilon = .005
		self.max_n_iter = 500
		self.iter_outer = 0
		self.iter = 0
		self.gaussian_noise_var = gaussian_noise_var # initialization noise variance for gaussian 
		self.stopping_condition = lambda el : el[0] > self.max_n_iter or el[1] < self.epsilon or el[2] > 0
		if cont_grads:
			self.latency_benefit_fn = self.latency_benefit_continuous
			self.gradient_fn = self.gradients_continuous
		else:
			self.latency_benefit_fn = self.latency_benefit
			self.gradient_fn = self.gradients

		self.graph_fn = graph_fn
		self.as_relationships = {}
		# at each iteration, create graph object s.t. only active announcments
		# show peering relationships
		self.my_graph = {}
		self.adjacency_graph = {}
		with open(os.path.join(GRAPH_DIR, graph_fn), 'r') as f:
			for row in f:
				if row.startswith("#"): continue
				as1,as2,rel = row.strip().split('|')
				as1 = int(as1); as2 = int(as2); rel = int(rel)
				self.my_graph[as1,as2] = rel
				self.my_graph[as2,as1] = rel * -1
				try:
					self.adjacency_graph[as1].append(as2)
				except KeyError:
					self.adjacency_graph[as1] = [as2]
				try:
					self.adjacency_graph[as2].append(as1)
				except KeyError:
					self.adjacency_graph[as2] = [as1]

		self.edge_lengths = {}
		self.graph_md = json.load(open(os.path.join(GRAPH_DIR, graph_md_fn), 'r'))
		for pair,l in self.graph_md['edge_lengths'].items():
			p1,p2 = [int(el) for el in pair.split('-')]
			self.edge_lengths[p1,p2] = float(l)
			self.edge_lengths[p2,p1] = float(l)
		self.user_networks = [int(un) for un in self.graph_md["user_networks"]]

		# Shape of the variables
		self.content_provider = int(self.graph_md['content_provider'])
		self.peers = []
		for as1,as2 in self.my_graph:
			if as1 == self.content_provider:
				self.peers.append(as2)
			elif as2 == self.content_provider:
				self.peers.append(as1)
		self.peers = list(set(self.peers))
		self.peer_to_ind = {k:i for i,k in enumerate(self.peers)}
		self.n_peers = len(self.peers)
		self.n_prefixes = 2

		# populate transit providers
		rels_by_asn = {}
		for (as1,as2),rel in self.my_graph.items():
			try:
				rels_by_asn[as1].append(rel)
			except KeyError:
				rels_by_asn[as1] = [rel]
		transit_providers = [asn for asn,rels in rels_by_asn.items() if 1 not in rels]
		self.transit_providers = np.array(get_intersection(transit_providers, self.peers))
		self.non_transit_peers = np.array(get_difference(self.peers, self.transit_providers))

		self.n_peers = len(self.peers)

		# measure latency from users to peers
		self.calculate_user_latency_by_peer()
		self.calculate_path_probabilities(init=True)

		print("Creating problem with {} peers, {} prefixes.".format(self.n_peers, self.n_prefixes))

		self.verbose = verbose

	def gradients(self, *args, **kwargs):
		pass

	def gradients_continuous(self, *args, **kwargs):
		pass

	def calculate_best_paths(self, a):
		try:
			return bp_cache[tuple(a)]
		except KeyError:
			pass

		# a is a Nx1 vector where N is the number of peers, 
		# a[i] indicates whether we announce to peer i
		# copy our graph and delete peers who we choose not to advertise to
		graph_cp = copy.copy(self.my_graph)
		peers_to_del = [self.peers[i] for i in range(self.n_peers) if not a[i]]
		for peer in peers_to_del:
			del graph_cp[self.content_provider, peer]
			del graph_cp[peer, self.content_provider]
		# Announce prefix from to all peers
		graph = ASGraph.manual_graph_input(graph_cp)
		sources = [self.content_provider]
		announce = Announcement.make_anycast_announcement(graph, sources)
		g1 = graph.clone()
		# return best paths
		t_s_i = time.time()
		g1.infer_paths(announce)
		bp_cache[tuple(a)] = g1.g.nodes

		return g1.g.nodes

	def calculate_path_probabilities_old(self, init=False):
		"""Update path probabilities based on measured paths."""

		try:
			self.peer_preferences
		except AttributeError:
			self.peer_preferences = {ui: {pi:BIG_BAD_VALUE for pi in range(self.n_peers)} for ui in range(len(self.user_networks))}

		if init:
			self.path_probabilities = BIG_BAD_VALUE * np.ones((self.n_peers, 1, len(self.user_networks)))
			# basic initialization is
			# uniform over paths for each user_netork, same for each prefix
			# (need to check that a path exists)
			self.reachable_peers = {}
			self.uncertainties = {}
			self.measured_prefs = {}
			for ui in range(len(self.user_networks)):
				reachable_peers = np.where(self.measured_latencies[:,0,ui] != BIG_BAD_VALUE)[0]
				self.reachable_peers[ui] = reachable_peers # static, property of the Internet
				reachable_peers = np.array([self.peers[pi] for pi in reachable_peers])
				non_transit_peers = np.array(get_intersection(reachable_peers, self.non_transit_peers))
				non_transit_prefs = np.arange(len(non_transit_peers)) 
				transit_prefs = len(non_transit_prefs) + np.arange(len(self.transit_providers)) # all transit should be reachable
				np.random.shuffle(transit_prefs)
				np.random.shuffle(non_transit_prefs)
				for pi in range(len(reachable_peers)):
					univ_pi = np.where(reachable_peers[pi] == self.peers)[0][0]
					# Simple init -- deprioritize transit
					if reachable_peers[pi] in self.transit_providers:
						self.peer_preferences[ui][univ_pi] = 1
					else:
						self.peer_preferences[ui][univ_pi] = 0

				# For each pair of paths, init uncertainties to prior values
				for pi in range(len(reachable_peers)):
					for pj in range(len(reachable_peers)):
						if pi != pj:
							# could have more complicated prior uncertainty
							self.uncertainties[ui,self.reachable_peers[ui][pi],self.reachable_peers[ui][pj]] = 1
		else:
			# Update probabilities to be consistent with observed pairwise preferences
			for ui in self.measured_prefs:
				for winning_path, active_paths in self.measured_prefs[ui]:
					# indicate we are no longer uncertain about the pairwise preferences between these paths
					# could also propagate lack of uncertainty
					for peer in active_paths:
						self.uncertainties[ui,winning_path,peer] = 0
						self.uncertainties[ui,peer,winning_path] = 0

					arg_max_pref = np.argmin([self.peer_preferences[ui][pi] for pi in active_paths])
					max_pref_path = active_paths[arg_max_pref]
					max_pref = self.peer_preferences[ui][max_pref_path]
					# we only update information if we learned something about preferences
					# the chosen path should be a higher priority than all other active paths
					if self.peer_preferences[ui][winning_path] >= max_pref:
						self.peer_preferences[ui][winning_path] = max_pref - .1
						# resort to give everything an integer preference
						tmp = sorted(self.peer_preferences[ui].items(), key = lambda el : el[1])
						self.peer_preferences[ui] = {}
						i=0
						for k,v in tmp: # TODO -- maybe reconsider this assignment; we arbitrarily break ties here but we may not want to do that
							if v != BIG_BAD_VALUE:
								self.peer_preferences[ui][k] = i
								i += 1
							else:
								self.peer_preferences[ui][k] = BIG_BAD_VALUE
		# Put all preferences into a matrix
		self.path_probabilities = np.expand_dims(self.path_probabilities[:,0,:],axis=1)
		for ui in self.peer_preferences:
			for pi in self.peer_preferences[ui]:
				self.path_probabilities[pi,0,ui] = self.peer_preferences[ui][pi] 

		# Convert preferences to pdf via softmax
		# interpretation is that preference k - 1 is exp(k) times as likely as preference k
		# since we set unreachable paths to preference of BIG_BAD_VALUE, those probabilities are 0
		# softmax_k = .5 + 2 * np.exp(-1 / (self.iter_outer + .001))
		softmax_k = .5 + 4 * np.exp(-1 / (self.iter + .001))
		self.path_probabilities = np.exp(-1 * softmax_k * self.path_probabilities)

		# We tile since the path probabilities don't depend on the prefix being announced, only who
		# you announce the prefix to and the user group
		self.path_probabilities = np.tile(self.path_probabilities, (1, self.n_prefixes, 1))


	def calculate_path_probabilities(self, init=False):
		"""Update path probabilities based on measured paths."""

		
		def violates(ordering, bigger, smallers):
			ordering = np.array(ordering)
			smallers = np.array(smallers)
			wb = np.where(bigger == ordering)[0]
			for s in get_intersection(smallers, ordering):
				if wb > np.where(ordering == s)[0]:
					return True
			return False
		if init:
			# basic initialization is
			# uniform over paths for each user_netork, same for each prefix
			# (need to check that a path exists)
			self.reachable_peers = {}
			self.orderings = {}
			self.measured_prefs = {}
			for ui in range(len(self.user_networks)):
				reachable_peers = np.where(self.measured_latencies[:,0,ui] != BIG_BAD_VALUE)[0]
				self.reachable_peers[ui] = reachable_peers # static, property of the Internet
				reachable_peers = np.array([self.peers[pi] for pi in reachable_peers])
				non_transit_peers = np.array(get_intersection(reachable_peers, self.non_transit_peers))
				# Initialize valid orderings to have uniform probability
				# orderings are w.r.t. reachable peers
				self.orderings[ui] = {}
				n_combs = np.math.factorial(len(self.reachable_peers[ui]))
				for ordering in itertools.permutations(self.reachable_peers[ui],len(self.reachable_peers[ui])):
					self.orderings[ui][ordering] = 1 / n_combs
		else:
			# Update orderings to be consistent with observed pairwise preferences
			# print(self.peers)
			# print(self.user_networks)
			# print(self.measured_prefs)
			for ui in self.measured_prefs:
				# print("UI : {} Before: {}".format(ui, len(self.orderings[ui])))
				for winning_path, active_paths in self.measured_prefs[ui]:
					# delete orderings that violate this measurement, then re-normalize ordering pdf
					all_orderings = list(self.orderings[ui].keys())
					for ordering in all_orderings:
						# print("{} {}".format(ordering, violates(ordering,winning_path,active_paths)))
						if violates(ordering, winning_path, active_paths): 
							del self.orderings[ui][ordering]
				norm_v = sum(self.orderings[ui].values())
				for ordering in self.orderings[ui]:
					self.orderings[ui][ordering] /= norm_v

				# print("UI : {} After: {}".format(ui, len(self.orderings[ui])))

	def calculate_user_latency_by_peer(self):
		"""Calculate latency from each user to each peer. In practice, you would measure these latencies
			once using Anyopt-type techniques.
			For simulation, we use a BGP simulator and a simple AS-AS edge length model.
		"""
		self.measured_latencies_dict = {}
		a = np.zeros((self.n_peers, ))
		for pi, p in enumerate(self.peers):
			# Simulate an advertisement from only this peer
			a[pi] = 1
			best_paths = self.calculate_best_paths(a)
			a[pi] = 0
			for ui, user_network in enumerate(self.user_networks):
				if best_paths[user_network]['best-paths'] == []:
					latency =  BIG_BAD_VALUE
				else:
					best_path = [user_network] + list(best_paths[user_network]['best-paths'][0])
					latency = self.latency_from_path(best_path)
				self.measured_latencies_dict[p, user_network] = latency
		self.measured_latencies = np.zeros((self.n_peers, len(self.user_networks)))
		for pi in range(self.n_peers):
			for ui in range(len(self.user_networks)):
				self.measured_latencies[pi,ui] = self.measured_latencies_dict[self.peers[pi], self.user_networks[ui]]
		self.measured_latencies = np.tile(np.expand_dims(self.measured_latencies,axis=1), (1, self.n_prefixes, 1))

	def latency_benefit_from_bgp(self, a):
		a_effective = self.threshold_a(a)
		user_latencies = self.user_latencies_from_bgp(a_effective)
		benefit = self.benefit_from_user_latencies(user_latencies)
		return benefit

	def user_latencies_from_bgp(self, a):
		user_latencies = BIG_BAD_VALUE * np.ones((len(self.user_networks)))
		for prefix_i in range(a.shape[1]):
			if np.sum(a[:,prefix_i]) == 0:
				continue
			best_paths = self.calculate_best_paths(a[:,prefix_i])
			for i,user_network in enumerate(self.user_networks):
				if best_paths[user_network]['best-paths'] == []:
					latency = BIG_BAD_VALUE
				else:
					best_path = [user_network] + list(best_paths[user_network]['best-paths'][0])
					latency = self.latency_from_path(best_path)
				if latency < user_latencies[i]: user_latencies[i] = latency
		return user_latencies

	def latency_benefit(self, a, calc_uncertainty=False,plotit=False):
		"""Way of estimating expecting latency benefit from an advertisement. Look at expected
			latency for a given advertisement."""
		a_effective = self.threshold_a(a)

		self.path_probabilities = np.zeros((self.n_peers, self.n_prefixes, len(self.user_networks)))
		for ug in range(len(self.user_networks)):
			for pref_i in range(self.n_prefixes):
				possible_peers = get_intersection(self.reachable_peers[ug], np.where(a_effective[:,pref_i])[0])
				pi = np.zeros((self.n_peers))
				for ordering, p in self.orderings[ug].items():
					sub_order = [o for o in ordering if o in possible_peers]
					if sub_order != []:
						pi[sub_order[0]] += p
				self.path_probabilities[:,pref_i,ug] = pi / np.sum(pi + 1e-8)


		# Dims are path, prefix, user

		delta = .1
		# we say latency is very large if there is no path
		# so benefit will be < 0 if there is no path, we clip since that shouldn't contribute negatively to the benefit
		p_mat = self.path_probabilities
		benefits = np.log(1 / self.measured_latencies).clip(0,np.inf) 
		p_mat = p_mat / (np.sum(p_mat,axis=0) + 1e-8)

		min_b, max_b = BIG_BAD_VALUE, -1 * BIG_BAD_VALUE
		for b,p in zip(benefits.flatten(), p_mat.flatten()):
			if p == 0: continue
			min_b = np.minimum(min_b,b)
			max_b = np.maximum(max_b,b)
		n_pts = int(np.ceil((max_b - min_b) / delta))
		if n_pts == 0: # degenerate case -- only one possible benefit, likely the zero advertisement
			benefit = min_b
			xsumx = np.array([min_b])
			psumx = np.array([1.0])
			if calc_uncertainty:
				return benefit, (xsumx,psumx)
			else:
				return benefit
		lbx = np.linspace(min_b,max_b,num=n_pts)
		# holds P(latency benefit) for each user
		px = np.zeros((n_pts, len(self.user_networks)))
		roll_min, roll_max = 0, 0
		users_to_del = []
		for ui in range(len(self.user_networks)):
			minb,maxb = BIG_BAD_VALUE, -1 * BIG_BAD_VALUE
			all_pv = [(j,v,p) for j in range(self.n_prefixes) for v,p in zip(benefits[:,j,ui], p_mat[:,j,ui]) if p > 0]
			ctd_idx_start = False
			if len(all_pv) == 1:
				_, lb, p = all_pv[0]
				lbx_i = np.where(lb - lbx <= 0)[0][0]
				px[lbx_i, ui] += p
				maxb = np.maximum(maxb,lb)
				minb = np.minimum(minb,lb)
			else:
				all_pv = sorted(all_pv,key=lambda el : el[1])
				running_probs = np.ones((self.n_prefixes))
				all_pref_inds = np.arange(self.n_prefixes)
				# print(all_pv)
				for i in range(1,len(all_pv)):
					pref_j, lb, p = all_pv[i]

					# calculate prob(max latency benefit)
					# we calculate this iteratively, from the smallest to the largest value
					# probability calc is basically probability of this value (p) times probability 
					# other prefixes are one of the smaller values (running prob)
					max_prob = p * np.prod(running_probs[all_pref_inds!=pref_j])
					if running_probs[pref_j] == 1:
						running_probs[pref_j] = p
					else:
						running_probs[pref_j] += p

					lbx_i = np.where(lb - lbx <= 0)[0][0]
					px[lbx_i, ui] += max_prob
					maxb = np.maximum(maxb,lb)
					minb = np.minimum(minb,lb)
			if np.sum(px[:,ui]) == 0:
				# This user experiences no benefit with probability 1
				# just remove the user from consideration
				users_to_del.append(ui)
				continue
			if maxb != -1 * BIG_BAD_VALUE:
				roll_max += maxb
				roll_min += minb

		for user in reversed(users_to_del):
			px = np.concatenate([px[:,0:user],px[:,user+1:]],axis=1)

		idx_start = 0
		for ui in range(px.shape[1]):
			idx_start += np.where(px[:,ui] > 0)[0][0]
		px = px / (np.sum(px,axis=0) + 1e-8)
		if roll_min == roll_max:
			# deterministic situation 
			benefit = roll_min / len(self.user_networks)
			xsumx = np.array([benefit])
			psumx = np.array([1.0])
			if calc_uncertainty:
				return benefit, (xsumx,psumx)
			else:
				return benefit

		# post pad to prevent wrap around, then pad it again to make it a power of 2
		l_post_pad = (px.shape[1] + 1) * n_pts
		px = np.concatenate([px,np.zeros(((l_post_pad, px.shape[1])))], axis=0)
		n_fft = int(2**(np.ceil(np.log2(px.shape[0]))))
		px = np.concatenate([px,np.zeros((n_fft - n_pts, px.shape[1]))], axis=0)
		Px = np.fft.fft(px,axis=0)
		Psumx = np.prod(Px,axis=1)
		psumx = np.real(np.fft.ifft(Psumx))
		# maybe not the best but I couldn't figure out how to do the indexing
		n_pts_output = np.where(psumx>1e-4)[0][-1] - idx_start + 1 
		psumx = psumx[idx_start:idx_start+n_pts_output]

		# pmf of benefits is now xsumx with probabilities psumx
		xsumx = np.linspace(roll_min, roll_max, num=n_pts_output) / len(self.user_networks)

		plotit = plotit or np.sum(psumx) < .99 # Checks that this is a probability distribution
		if plotit:
			import matplotlib.pyplot as plt
			print(self.orderings[3])
			print(a)
			print(benefits[:,0,:])
			print(benefits[:,1,:])
			print(p_mat[:,0,:])
			print(p_mat[:,1,:])
			print(idx_start)
			print(n_pts_output)
			print(n_pts)
			print(roll_min)
			print(roll_max)
			print(np.real(np.fft.ifft(Psumx)))
			print(px)
			pickle.dump([px,benefits,p_mat,idx_start,n_pts_output,roll_min,roll_max],open('tmp.pkl','wb'))
			plt.plot(xsumx * len(self.user_networks), psumx)
			plt.show()
			exit(0)


		benefit = np.sum(xsumx * psumx)

		if self.verbose:
			e_user_lat = np.sum((self.measured_latencies * p_mat + 1e-8) / np.sum(p_mat+1e-14, axis=0), axis=0)
			min_e_user_lat = np.min(e_user_lat, axis=0) # minimum over prefixes
			actual_ul = self.user_latencies_from_bgp(a_effective)
			self.metrics['twopart_EL_difference'].append(actual_ul - min_e_user_lat)
			if np.max(np.abs(self.metrics['twopart_EL_difference'][-1])) > 1 and self.iter>30:
				print(self.reachable_peers[0])
				print(self.orderings[0])
				print(self.path_probabilities[:,0,:])
				print(self.path_probabilities[:,1,:])
				print(a_effective)
				print(A_mat[:,0,:])
				print(A_mat[:,1,:])
				print(np.squeeze(p_mat[:,0,:]))
				print(np.squeeze(p_mat[:,1,:]))
				print(np.squeeze(self.measured_latencies[:,0,:]))
				print(e_user_lat)
				print(e_user_lat.shape)
				print(min_e_user_lat)
				print(min_e_user_lat.shape)
				print(actual_ul)
				print("\n")
				exit(0)

		if calc_uncertainty:
			return benefit, (xsumx,psumx)
		else:
			return benefit

	def latency_benefit_continuous(self, a):
		# Wrapper around latency benefit to hopefully make it more continuous
		rolling_sum = 0
		for a_i in range(a.shape[0]):
			for a_j in range(a.shape[1]):
				a_ij = a[a_i,a_j]
				a[a_i,a_j] = 1
				rolling_sum += a_ij * self.latency_benefit(a)
				a[a_i,a_j] = 0
				rolling_sum += (1 - a_ij) * self.latency_benefit(a)
				a[a_i,a_j] = a_ij
		
		return rolling_sum / np.prod(a.shape)

	def latency_from_path(self, path):
		latency = 0
		for edge in range(len(path) - 1):
			latency += self.edge_lengths[path[edge],path[edge+1]]
		return latency * KM_TO_MS * .001

	def benefit_from_user_latencies(self, user_latencies):
		# sum of the benefits, simple model for benefits is 1 / latency
		user_benefits = np.clip(np.log(1 / user_latencies), 0 , np.inf)
		# average user benefit -- important that this function is not affected by the number of user groups
		return np.mean(user_benefits)

	def init_advertisement(self, mode='gaussian'):
		if mode == 'simple':
			return np.random.randint(0,2,size=(self.n_peers, self.n_prefixes)) * 1.0
		elif mode == 'gaussian':
			return self.advertisement_threshold + self.gaussian_noise_var * np.random.normal(size=(self.n_peers, self.n_prefixes))
		else:
			raise ValueError("Adv init {} not recognized.".format(mode))

	def measure_paths(self, a):
		"""Between rounds, measure paths from users to CP given advertisement a."""
		try:
			self.measured
		except AttributeError:
			self.measured = {}

		self.path_measures += 1

		# TODO -- don't remeasure only one peer being active at once, useless

		best_paths = {}
		actives = {}
		a = self.threshold_a(a)

		# print("Measuring : \n{}".format(a))
		self.measured[tuple(a.flatten())] = None
		for prefix_i in range(a.shape[1]):
			actives[prefix_i] = np.where(a[:,prefix_i] == 1)[0]
			if np.sum(a[:,prefix_i]) == 0:
				best_paths[prefix_i] = {un:{'best-paths':[]} for un in self.user_networks}
			else:
				best_paths[prefix_i] = self.calculate_best_paths(a[:,prefix_i])
		for ui, user_network in enumerate(self.user_networks):
			self.measured_prefs[ui] = []
			for prefix_i in range(a.shape[1]):
				bp = best_paths[prefix_i][user_network]['best-paths']
				if bp == []:
					# no path, nothing learned
					continue
				else:
					best_path = list(bp[0])
					peer_network = ([user_network] + best_path)[-2]
					# indidates for this user group what the winning path was, and what the active paths were
					self.measured_prefs[ui].append((self.peer_to_ind[peer_network], actives[prefix_i]))

	def threshold_a(self, a):
		return (a > self.advertisement_threshold).astype(np.float32)

	def objective(self, a, v=False):
		"""Approx actual objective with 1 norm."""
		norm_penalty = np.sum(np.abs(a).flatten())
		latency_benefit = self.latency_benefit_fn(a)
		return self.mu * norm_penalty - latency_benefit

	def actual_objective(self, a):
		"""Actual objective is the number of peering relationships and the number of prefixes."""
		# i.e., don't approximate with the L1 norm
		# and use the actual measured path latencies
		c_pref = .5
		c_peer = .5

		has_peer = (np.sum(a,axis=1) > 0).astype(np.int32)
		has_pref = (np.sum(a,axis=0) > 0).astype(np.int32)

		# cost for different peers may be different
		cost_peers = np.sum(np.dot(has_peer,c_peer*np.ones(has_peer.shape)))
		# cost for different prefs may be different
		cost_prefs = np.sum(has_pref) * c_pref
		norm_penalty = cost_peers + cost_prefs
		latency_benefit = self.latency_benefit_from_bgp(a)
		return self.mu * norm_penalty - latency_benefit

	def outer_objective(self, a):
		"""Objective measured using actual advertisement and subsequent achieved latencies."""
		norm_penalty = np.sum(np.abs(a).flatten())
		latency_benefit = self.latency_benefit_from_bgp(a)
		return self.mu * norm_penalty - latency_benefit

class Sparse_Advertisement_Eval(Sparse_Advertisement_Wrapper):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.sas = Sparse_Advertisement_Solver(**kwargs)

	def compare_peer_value(self,minmu=.01,maxmu=5):
		soln_keys = ['ours','oracle']#,'sparse_greedy']
		soln_peer_alives = {k:[] for k in soln_keys}
		linestyles = ['-','*','^']
		colors = ['r','b','k','orange','fuchsia','sandybrown']
		# mus = np.linspace(maxmu,minmu,num=50)
		mus = np.logspace(np.log10(minmu),np.log10(maxmu),num=20)
		mus = np.flip(mus)
		init_advs = []
		for mu in mus:
			print(mu)
			self.mu = mu; self.sas.mu = mu
			self.compare_different_solutions(which='twopart',n_run=1,verbose=False)
			init_advs.append(self.sas.metrics['twopart_advertisements'][0])

			our_adv = self.sas.get_last_advertisement()
			oracle_adv = self.oracle['l1_advertisement']
			sparse_greedy_adv = self.sparse_greedy['advertisement']

			for k,adv in zip(soln_keys, [our_adv, oracle_adv, sparse_greedy_adv]):
				peers_alive = (np.sum(self.threshold_a(adv),axis=1) > 0).astype(np.int32)
				soln_peer_alives[k].append(peers_alive)

		f,ax = plt.subplots(2,1)
		for si,k in enumerate(soln_keys):
			soln_peer_alives[k] = np.array(soln_peer_alives[k])
			for i in range(self.n_peers):
				ax[0].semilogx(mus, soln_peer_alives[k][:,i] + (-.1  + .1 * si / len(soln_keys)), linestyles[si], c=colors[i], label="{}--{}".format(k,i))

		ax[0].legend()
		ax[0].set_xlabel("Mu")
		ax[0].set_ylabel("Peer On/Off")

		alpha = self.advertisement_threshold / 2
		soln_peer_alives_smoothed = {k: np.zeros((self.n_peers, len(mus) - 1)) for k in soln_keys}
		for si,k in enumerate(soln_keys):
			soln_peer_alives[k] = np.array(soln_peer_alives[k])
			for i in range(self.n_peers):
				avg = soln_peer_alives[k][0,i]
				for mui in range(len(mus) - 1):
					soln_peer_alives_smoothed[k][i,mui] = avg
					avg = alpha * soln_peer_alives[k][mui,i] + (1 - alpha) * avg
				ax[1].semilogx(mus[0:-1], soln_peer_alives_smoothed[k][i,:] + (-.1  + .1 * si / len(soln_keys)), linestyles[si], c=colors[i], label="{}--{}".format(k,i))
		ax[1].legend()
		ax[1].set_xlabel("Mu")
		ax[1].set_ylabel("Peer On/Off")

		rankings = {k:{} for k in soln_keys}
		for peer_i in range(self.n_peers):
			for k in soln_keys:
				crit_mu = np.where(soln_peer_alives_smoothed[k][peer_i,:] > self.advertisement_threshold)[0]
				if len(crit_mu) == 0:
					crit_mu = maxmu + 1
				else:
					crit_mu = mus[crit_mu[0]]
				rankings[k][peer_i] = crit_mu
		for k in rankings:
			peer_rankings = sorted(rankings[k].items(), key = lambda el : el[1])
			peer_rankings = [(self.peers[peer_i], mu) for peer_i, mu in peer_rankings]
			print("{} : sorted rankings : {}".format(k, peer_rankings))


		plt.show()

	def solve_extremes(self, verbose=True):
		# Find minimal and maximal set of advertisements
		# maximal is just everything turned on
		maximal_advertisement = np.ones((self.n_peers, self.n_prefixes))
		maximal_objective = self.objective(maximal_advertisement)

		

		self.cc_cache = {}
		def get_cc(asn):
			cc = [asn]
			try:
				return self.cc_cache[asn]
			except KeyError:
				pass
			for connected_as in self.adjacency_graph[asn]:
				if self.my_graph[asn,connected_as] == -1:
					cc = cc + get_cc(connected_as)
			self.cc_cache[asn] = list(set(cc))
			return cc

		tp_ccs = {}
		for transit_provider in self.transit_providers:
			cc = get_cc(transit_provider)
			tp_ccs[transit_provider] = list(set(cc))
		# find minimal covering set for all user networks
		# in general this is an NP hard problem but the number of TPs is small so even hard problems aren't so hard
		covering_sets = []
		# being lazy for now
		tp_powerset = itertools.chain.from_iterable(itertools.combinations(tp_ccs, r) for r in range(len(tp_ccs)+1))
		valid_coverings = []
		for tp_set in tp_powerset:
			covered_users, found = [], False
			for tp in tp_set:
				covered_users = covered_users + get_intersection(self.user_networks, tp_ccs[tp])
				if get_difference(self.user_networks, covered_users) == []:
					found = True
					break
			if found:
				valid_coverings.append(list(tp_set))
		if valid_coverings == []:
			raise ValueError("Could not find set of transit providers to cover all users!")
		best_covering_len = np.min([len(tp_set) for tp_set in valid_coverings])
		candidate_tp_sets = [tp_set for tp_set in valid_coverings if len(tp_set) == best_covering_len]
		scores = []
		advs = []
		for tp_set in candidate_tp_sets:
			advertisement = np.zeros((self.n_peers, self.n_prefixes))
			for tp in tp_set:
				ind = np.where(np.array(self.peers) == tp)[0][0]
				advertisement[ind] = 1
				advs.append(copy.copy(advertisement))
			scores.append(self.objective(advertisement))

		minimal_objective = np.min(scores)
		minimal_advertisement = advs[np.argmin(scores)]

		self.extreme = {
			'minimal_advertisement': minimal_advertisement,
			'minimal_objective': minimal_objective,
			'maximal_advertisement': maximal_advertisement,
			'maximal_objective': maximal_objective,
		}

	def solve_sparse_greedy(self, init_adv, verbose=True):
		# at each iteration, toggle the entry that yields the largest objective function benefit

		advertisement = np.copy(self.threshold_a(init_adv))
		stop = False
		i = 0
		n_measures = 0
		while not stop:
			pre_obj = self.actual_objective(advertisement)
			deltas = np.zeros((self.n_peers, self.n_prefixes))
			for peer_i in range(self.n_peers):
				for pref_i in range(self.n_prefixes):
					# toggle this entry
					advertisement[peer_i,pref_i] = 1 - advertisement[peer_i,pref_i]
					deltas[peer_i,pref_i] = pre_obj - self.outer_objective(advertisement)
					advertisement[peer_i,pref_i] = 1 - advertisement[peer_i,pref_i]
					n_measures += 1
			# largest additional latency benefit
			best_delta = np.argmax(deltas.flatten())
			best_peer_i = best_delta // self.n_prefixes
			best_pref_i = best_delta % self.n_prefixes
			advertisement[best_peer_i,best_pref_i] = 1 - advertisement[best_peer_i,best_pref_i]

			stop = self.stopping_condition([i,np.abs(deltas[best_peer_i, best_pref_i]), deltas[best_peer_i, best_pref_i]])

			i += 1
		print("Sparse greedy solution measured {} advertisements".format(n_measures))
		self.sparse_greedy = {
			'objective': self.outer_objective(advertisement),
			'advertisement': advertisement,
		}

	def solve_greedy(self, init_adv, verbose=True):
		# at each iteration, toggle the entry that yields the largest delta latency benefit

		advertisement = copy.copy(self.threshold_a(init_adv))
		stop = False
		i = 0
		while not stop:
			pre_lat_ben = self.latency_benefit_fn(advertisement)
			deltas = np.zeros((self.n_peers, self.n_prefixes))
			for peer_i in range(self.n_peers):
				for pref_i in range(self.n_prefixes):
					# toggle this entry
					advertisement[peer_i,pref_i] = 1 - advertisement[peer_i,pref_i]
					# should really measure paths, but whatever
					deltas[peer_i,pref_i] = self.latency_benefit_fn(advertisement) - pre_lat_ben
					advertisement[peer_i,pref_i] = 1 - advertisement[peer_i,pref_i]
			# largest additional latency benefit
			best_delta = np.argmax(deltas.flatten())
			best_peer_i = best_delta // self.n_prefixes
			best_pref_i = best_delta % self.n_prefixes
			advertisement[best_peer_i,best_pref_i] = 1 - advertisement[best_peer_i,best_pref_i]

			stop = self.stopping_condition([i,np.abs(deltas[best_peer_i, best_pref_i]),-1])

			i += 1

		self.greedy = {
			'objective': self.outer_objective(advertisement),
			'advertisement': advertisement,
		}

	def solve_oracle(self,verbose=True):
		# To get the oracle solution, try every possible combination of advertisements
		# Not possible for problems that are too large
		n_arr = self.n_peers * self.n_prefixes
		n_possibilities = 2**n_arr
		if n_possibilities >= 1e6:
			print("Note -- too complex to get oracle solution. Skipping")
			self.oracle = None
			return
		a = np.zeros((n_arr,))
		all_as = []
		objs = np.zeros((n_possibilities,))
		actual_objs = np.zeros((n_possibilities,))
		for i in range(n_possibilities):
			ib = str(bin(i))[2:]
			for j,el in enumerate(ib):
				a[j] = int(el) * 1.0
			objs[i] = self.outer_objective(a.reshape((self.n_peers, self.n_prefixes)))
			actual_objs[i] = self.actual_objective(a.reshape((self.n_peers,self.n_prefixes)))
			all_as.append(copy.deepcopy(a))
		oracle_objective = np.min(objs)

		actual_optimal_oracle_i = np.argmin(actual_objs) # the two objectives are scaled differently, so we choose a common scaling
		actual_oracle_objective = objs[actual_optimal_oracle_i]

		l1_oracle_adv = all_as[np.argmin(objs)].reshape(self.n_peers, self.n_prefixes)
		l0_oracle_adv = all_as[np.argmin(actual_objs)].reshape(self.n_peers, self.n_prefixes)
		if verbose:
			print("Oracle advertisements\n Approx with L1 ({}): {}\nWith L0 ({}): {}".format(
				round(oracle_objective,2), l1_oracle_adv,
				round(actual_oracle_objective,2), l0_oracle_adv))

		self.oracle = {
			'l1_objective': oracle_objective,
			'l0_objective': actual_oracle_objective,
			'l1_advertisement': l1_oracle_adv,
			'l0_advertisement': l0_oracle_adv,
		}

	def compare_different_solutions(self,n_run=10,verbose=True,init_adv=None,which='outerinner'):
		# oracle
		self.solve_oracle(verbose=verbose)
		# Extremes
		self.solve_extremes(verbose=verbose)
		vals = {'ours': [], 'greedy': [], 'sparse greedy': []}
		our_advs = []
		if verbose:
			print(self.peers)
		for i in range(n_run):
			if verbose:
				print(i)
			# Initialize advertisement
			if init_adv is None:
				adv = self.init_advertisement(mode='gaussian')
			else:
				adv = init_adv
			if verbose:
				print("solving ours")
			if which == 'outerinner':
				# Our solution
				self.sas.solve(init_adv=adv)
			elif which == 'twopart':
				self.sas.solve_twopart(init_adv=adv)
			else:
				raise ValueError("Which {} not understood.".format(which))
			our_objective = self.sas.get_last_objective(effective=True) # look at objective for thresholded adv
			our_advs.append(self.sas.get_last_advertisement())
			if verbose:
				print("solving greedy")
			# Greedy solution
			self.solve_greedy(init_adv=adv,verbose=verbose)

			# Sparse greedy solution
			self.solve_sparse_greedy(init_adv=adv,verbose=verbose)

			vals['ours'].append(our_objective)
			vals['greedy'].append(self.greedy['objective'])
			vals['sparse greedy'].append(self.sparse_greedy['objective'])

		if verbose:
			for v, a in zip(vals['ours'], our_advs):
				print("{} ({}) -- {}".format(np.round(a,2).flatten(),self.threshold_a(a.flatten()),v))

			for k in vals:
				if self.oracle is None: 
					v = 1
				else:
					v = self.oracle['l1_objective']
				x,cdf_x = get_cdf_xy(vals[k] + .01 * np.abs(self.extreme['maximal_objective'] - v) * np.random.normal(size=(len(vals[k]),)))
				plt.plot(x,cdf_x,label=k)
			if self.oracle is not None:
				plt.axvline(x=self.oracle['l1_objective'],c='k')
				plt.annotate("L1 Oracle", (self.oracle['l1_objective'],.5))
				plt.axvline(x=self.oracle['l0_objective'],c='k')
				plt.annotate("L0 Oracle", (self.oracle['l0_objective'],.5))
			plt.axvline(x=self.extreme['minimal_objective'],c='turquoise')
			plt.annotate("Minimal", (self.extreme['minimal_objective'],.5))
			plt.axvline(x=self.extreme['maximal_objective'],c='darkorchid')
			plt.annotate("Maximal", (self.extreme['maximal_objective'],.5))
			plt.legend()
			plt.xlabel("Objective Function Value")
			plt.ylabel("CDF of Trials")
			plt.show()

class Sparse_Advertisement_Solver(Sparse_Advertisement_Wrapper):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.sigmoid_k = 5.0
		self.metrics = {}
		self.beta = .3

		self.gradient_support = [(a_i,a_j) for a_i in range(self.n_peers) for a_j in range(self.n_prefixes)]
		max_support = self.n_peers * self.n_prefixes
		self.gradient_support_settings = {
			'calc_every': 20,
			'support_size': np.minimum(self.n_peers * self.n_prefixes,max_support), # setting this value to size(a) turns it off
		}

	def apply_prox_l1(self, w_k):
		"""Applies proximal gradient method to updated variable. Proximal gradient
			for L1 norm is a soft-threshold at the learning rate."""
		return np.sign(w_k) * np.maximum(np.abs(w_k) - self.mu * self.alpha, np.zeros(w_k.shape))

	def heaviside_gradient(self, before, after, a_ij):
		# Gradient of sigmoid function
		# when a_ij goes from zero to one, latency benefit value goes from before to after
		# we approx. that as the continuous function before + (after - before) / (1 + exp(-k * a_ij))
		# return the derivative of this function evaluated at a_ij
		x = a_ij - self.advertisement_threshold
		return (after - before) * self.sigmoid_k * np.exp(-self.sigmoid_k * x) / (1 + np.exp(-self.sigmoid_k * x))**2

	def get_last_advertisement(self):
		return self.metrics['twopart_advertisements'][-1]

	def get_last_objective(self, effective=False):
		if effective:
			return self.outer_objective(self.threshold_a(self.get_last_advertisement()))
		else:
			return self.outer_objective(self.get_last_advertisement())

	def grad_latency_benefit(self, a, inds=None):
		L_grad = np.zeros(a.shape)
		a_effective = self.threshold_a(a)
		if inds is None:
			inds = [(a_i,a_j) for a_i in range(a.shape[0]) for a_j in range(a.shape[1])]
		for a_i,a_j in inds:
			a_ij = a_effective[a_i,a_j] 
			if not a_ij: # off
				before = self.latency_benefit(a_effective)
				a_effective[a_i,a_j] = 1
				after = self.latency_benefit(a_effective)
			else: # on
				after = self.latency_benefit(a_effective)
				a_effective[a_i,a_j] = 0
				before = self.latency_benefit(a_effective)

			L_grad[a_i, a_j] = 	self.heaviside_gradient(before, after, a[a_i,a_j])
			a_effective[a_i,a_j] = a_ij
		return L_grad

	def gradients(self, a):
		# gradient is the proximal gradient of the L1 norm
		# minus mu times gradient of L 
		# gradient of L is calculated via a continuous approximation
		L_grad = self.grad_latency_benefit(a)
		self.metrics['twopart_latency_benefit_grads'].append(L_grad)

		return -1 * L_grad

	def gradients_continuous(self, a):
		inds = self.gradient_support
		if self.iter % self.gradient_support_settings['calc_every'] == 0:
			# periodically calculate all gradients
			inds = [(a_i,a_j) for a_i in range(a.shape[0]) for a_j in range(a.shape[1])]
		L_grad = np.zeros(a.shape)
		for a_i, a_j in inds:
			a_ij = a[a_i, a_j]
			a[a_i,a_j] = 1
			L_grad[a_i,a_j] += self.latency_benefit(a)
			L_grad += a_ij * self.grad_latency_benefit(a, inds)
			a[a_i,a_j] = 0
			L_grad[a_i,a_j] += -1 * self.latency_benefit(a)
			L_grad += (1 - a_ij) * self.grad_latency_benefit(a, inds)
			a[a_i,a_j] = a_ij
		L_grad /= len(inds)
		if self.iter % self.gradient_support_settings['calc_every'] == 0:
			self.update_gradient_support(L_grad)

		self.metrics['twopart_latency_benefit_grads'].append(L_grad)
		return -1 * L_grad

	def impose_advertisement_constraint(self, a):
		"""The convex constraint 0 <= a_ij <= 1 has the simple solution to clip."""
		a = np.clip(a,0,1.0)
		return a

	def make_plots(self, *args, **kwargs):
		f,ax = plt.subplots(4,1)

		# General convergence metrics plot
		which = args[0]

		all_as = np.array(self.metrics['{}_advertisements'.format(which)])
		all_grads = np.array(self.metrics['{}_grads'.format(which)])
		linestyles = ['-','*','^']
		colors = ['orange','brown','aqua','deeppink','peru','grey','k','tan']
		for pref_i in range(self.n_prefixes):
			pref_sty = linestyles[pref_i]
			for peer_i in range(self.n_peers):
				if 'xlimupper' in kwargs:
					ax[0].plot(kwargs['xlimupper'],all_as[:,peer_i,pref_i], pref_sty, c=colors[peer_i%len(colors)], label="Peer {} Prefix {}".format(self.peers[peer_i], pref_i))
				else:
					ax[0].plot(all_as[:,peer_i,pref_i][::5], pref_sty, c=colors[peer_i%len(colors)], label="Peer {} Prefix {}".format(self.peers[peer_i], pref_i))
				ax[1].plot(all_grads[:,peer_i,pref_i], c=colors[peer_i%len(colors)], label="Peer {} Prefix {}".format(self.peers[peer_i], pref_i))

		all_objectives = self.metrics['{}_actual_objectives'.format(which)]
		all_effective_ojectives = self.metrics['{}_effective_objectives'.format(which)]
		ax[0].legend(fontsize=6)
		ax[0].set_ylabel("a")
		ax[1].set_ylabel("Net Grad")
		ax[-2].plot(all_objectives)
		ax[-2].set_ylabel("Numeric Objective")
		ax[-1].plot(all_effective_ojectives)
		ax[-1].set_ylabel("Effective Objective")

		plt.show() 

		if which == 'outer' or which == 'twopart':
			# Make probabilities plot
			n_users = len(self.user_networks)
			n_rows = int(np.ceil(n_users/3))
			f,ax = plt.subplots(n_rows, 3)
			for ui in range(n_users):
				row_i,col_i = ui // 3, ui % 3
				if n_rows == 1:
					access = col_i
				else:
					access = (row_i,col_i)
				user_path_probs = np.array([P[:,0,ui] for P in self.metrics['{}_path_likelihoods'.format(which)]])
				for peer_i in range(self.n_peers):
					ax[access].plot(user_path_probs[:,peer_i],c=colors[peer_i%len(colors)],label="Peer {}".format(self.peers[peer_i]))
			plt.show()
		if which == 'inner' or which == 'twopart':
			# Make latency estimate error plot
			n_users = len(self.user_networks)
			for ui in range(n_users):
				latency_estimate_errors = np.array([EL[ui] for EL in self.metrics['{}_EL_difference'.format(which)]])
				plt.plot(latency_estimate_errors,label="User {}".format(self.user_networks[ui]))
			plt.show()


	def print_adv(self, a):
		for peer_i in range(self.n_peers):
			for pref_i in range(self.n_prefixes):
				print("Peer {} Prefix {}: {}".format(self.peers[peer_i], pref_i, a[peer_i,pref_i]))

	def set_alpha(self):
		self.alpha = .01

	def update_gradient_support(self, gradient):
		gradient = np.abs(gradient)
		inds = [(a_i,a_j) for a_i in range(self.n_peers) for a_j in range(self.n_prefixes)]
		sorted_inds = list(reversed(np.argsort([gradient[a_i,a_j] for a_i,a_j in inds])))
		# Focus on the largest gradients
		self.gradient_support = list([inds[i] for i in sorted_inds[0:self.gradient_support_settings['support_size']]])

	def solve_max_information_old(self, current_advertisement):
		"""Search through neighbors of a, calculate maximum uncertainty."""
		uncertainties = np.zeros(current_advertisement.shape)
		MIN_UNCERTAINTY = .1

		a = np.copy(self.threshold_a(current_advertisement))

		# Calculate uncertainty of all neighbors of a
		for ai in range(a.shape[0]):
			for aj in range(a.shape[1]):
				tmp = a[ai,aj]
				a[ai,aj] = 1 - a[ai,aj]
				_, u = self.latency_benefit(a, calc_uncertainty=True)
				uncertainties[ai,aj] = u
				a[ai,aj] = 1 - a[ai,aj]

		ind = np.unravel_index(np.argmax(uncertainties, axis=None), uncertainties.shape)
		if uncertainties[ind] <= MIN_UNCERTAINTY:
			return None
		else:
			a[ind] = 1 - a[ind]
			return a

	def solve_max_information(self, current_advertisement):
		"""Search through neighbors of a, calculate maximum uncertainty."""
		uncertainties = {}

		a = np.copy(self.threshold_a(current_advertisement))
		current_benefit = self.latency_benefit(a)

		def value_func(u, mode='pursue_entropy'):
			benefits,probs = u
			if np.min(benefits) != np.max(benefits):
				print('meep');exit(0)
			if mode == 'pursue_benefit':
				v = np.sum(benefits[benefits>current_benefit] * probs[benefits>current_benefit])
			elif mode == 'pursue_entropy':
				# TODO -- make the binning common across all comparisons
				v = scipy.stats.entropy(probs+1e-8)
			return v

		MIN_POTENTIAL_VALUE = .01
		n_flips = 1
		max_time = 2 # seconds
		t_start = time.time()
		while True:
			all_inds = [(i,j) for i in range(self.n_peers) for j in range(self.n_prefixes)]
			perms = list(itertools.permutations(all_inds, n_flips))
			np.random.shuffle(perms)
			for flips in perms:
				for flip in flips: # flip bits
					a[flip] = 1 - a[flip]
				if np.sum(a.flatten()) == 0: continue
				_, u = self.latency_benefit(a, calc_uncertainty=True)
				if n_flips > 1: # we can't afford to look through anything but nearest neighbors
					if value_func(u) > MIN_POTENTIAL_VALUE:
						return a
				uncertainties[flips] = u
				for flip in flips: # flip back
					a[flip] = 1 - a[flip]
				if time.time() - t_start > max_time: return None

			potential_value_measure = {}
			# print("CB: {}".format(current_benefit))
			max_benefit = -10
			best_flips = None
			for flips,u in uncertainties.items():
				potential_value_measure[flips] = value_func(u)
				if potential_value_measure[flips] >= max_benefit:
					best_flips = flips
					max_benefit = potential_value_measure[flips]
			# print(best_flips)
			# print("Max potential additional benefit mass : {}".format(max_benefit))
			if best_flips is not None:
				if potential_value_measure[best_flips] > MIN_POTENTIAL_VALUE:
					for flip in best_flips:
						a[flip] = 1 - a[flip]
					if tuple(a.flatten()) in self.measured:
						self.latency_benefit(a, calc_uncertainty=True,plotit=True)
						exit(0)
					return a
			n_flips += 1
			if n_flips == 2:
				return None

	def solve(self, init_adv=None):
		self.set_alpha()
		if init_adv is None:
			outer_advertisement = self.init_advertisement()
		else:
			outer_advertisement = init_adv
		# Initialize model of path probabilities
		self.calculate_path_probabilities(init=True)
		self.measure_paths(outer_advertisement)
		self.calculate_path_probabilities()
		a_km1 = outer_advertisement
		if self.verbose:
			# self.print_adv(advertisement)
			print("Optimizing over {} peers".format(self.n_peers))
			print(self.peers)
			print(self.user_networks)

		stop_outer = False
		self.iter_outer = 0

		t_start = time.time()
		# For analysis
		for oi in ['outer','inner']:
			for k in ['actual_objectives', 'advertisements', 'effective_objectives', 'grads', 
				'latency_benefit_grads', 'prox_l1_grads', 'path_likelihoods', 'EL_difference']:
				self.metrics["{}_{}".format(oi,k)] = []

		rolling_delta_outer = 10
		delta_alpha = .7
		delta_dot_alpha = .9
		current_outer_objective = self.outer_objective(outer_advertisement)


		# Add to metrics
		self.metrics['outer_path_likelihoods'].append(copy.copy(self.path_probabilities))
		self.metrics['outer_actual_objectives'].append(current_outer_objective)
		self.metrics['outer_effective_objectives'].append(self.outer_objective(self.threshold_a(outer_advertisement)))
		self.metrics['outer_advertisements'].append(copy.copy(outer_advertisement))

		while not stop_outer: # outer measurement loop
			stop_inner = False
			self.iter_inner = 0
			rolling_delta_inner = 10
			rolling_delta_dot_inner = -1
			current_inner_objective = self.objective(outer_advertisement)
			inner_advertisement = outer_advertisement
			for k in self.metrics: # reset inner metrics
				if 'inner' in k:
					self.metrics[k] = []
			a_km1 = outer_advertisement
			while not stop_inner:
				# calculate gradients
				grads = self.gradient_fn(inner_advertisement)
				# update advertisement by taking a gradient step with momentum and then applying the proximal gradient for L1
				a_k = inner_advertisement
				w_k = a_k - self.alpha * grads + self.beta * (a_k - a_km1)
				inner_advertisement = self.apply_prox_l1(w_k)
				self.metrics['inner_advertisements'].append(copy.copy(inner_advertisement))
				self.metrics['inner_grads'].append(inner_advertisement - a_k)
				a_km1 = a_k

				# project onto constraints
				# one constraint we may want is Ra <= 1 
				# not sure how to do this

				# another constraint we may want is 0 <= a_ij <= 1
				# the solution is just clipping to be in the set
				# clipping can mess with gradient descent
				inner_advertisement = self.impose_advertisement_constraint(inner_advertisement)

				# re-calculate objective
				last_inner_objective = current_inner_objective
				current_inner_objective = self.objective(inner_advertisement,v=False)
				self.metrics['inner_actual_objectives'].append(current_inner_objective)
				self.metrics['inner_effective_objectives'].append(self.objective(self.threshold_a(inner_advertisement)))

				# Stop when the objective doesn't change, but use an EWMA to track the change so that we don't spuriously exit
				rolling_delta_inner = (1 - delta_alpha) * rolling_delta_inner + delta_alpha * np.abs(current_inner_objective - last_inner_objective)
				rolling_delta_dot_inner = delta_dot_alpha * rolling_delta_dot_inner + (1-delta_dot_alpha) * (current_inner_objective - last_inner_objective)
				stop_inner = self.stopping_condition([self.iter_inner,rolling_delta_inner,rolling_delta_dot_inner])
				self.iter_inner += 1
			
			t_per_iter = (time.time() - t_start) / self.iter_inner
			# if self.verbose:
			# 	print("Stopped inner loop on {}, t per iter: {}".format(self.iter_inner, t_per_iter))
			# 	if self.iter_inner > 2:
			# 		self.make_plots('inner')
			# 		exit(0)
			self.iter_outer += 1
			last_outer_objective = current_outer_objective
			# Now we would actually issue the advertisement based on what our algorithm converged to
			# for now, just accept whatever the algorithm spits out
			# in the future we might want to gradually try out updates our inner algorithm suggests
			# TODO following the new observed catchments, we may want to update our p_i's for each user group
			last_outer_advertisement = outer_advertisement
			outer_advertisement = self.update_outer_advertisement(inner_advertisement,outer_advertisement)
			if self.verbose:
				print("Old outer: {}\n Inner Solution: {}\n New Outer: {}\n".format(last_outer_advertisement,
					inner_advertisement, outer_advertisement))
			current_outer_objective = self.outer_objective(outer_advertisement)

			# After updating advertisement, re-measure the paths to the CP and update path probabilities
			self.measure_paths(outer_advertisement)
			self.calculate_path_probabilities()


			rolling_delta_outer = (1 - delta_alpha) * rolling_delta_outer + delta_alpha * np.abs(current_outer_objective - last_outer_objective)
			stop_outer = self.stopping_condition([self.iter_outer,rolling_delta_outer,-1])

			# Add to metrics
			self.metrics['t_per_iter_inner'] = t_per_iter
			self.metrics['outer_actual_objectives'].append(current_outer_objective)
			self.metrics['outer_effective_objectives'].append(self.outer_objective(self.threshold_a(outer_advertisement)))
			self.metrics['outer_advertisements'].append(copy.copy(outer_advertisement))
			self.metrics['outer_grads'].append(outer_advertisement - last_outer_advertisement)
			self.metrics['outer_path_likelihoods'].append(copy.copy(self.path_probabilities))

		t_per_iter = (time.time() - t_start) / self.iter_outer
		if self.verbose:
			print("Stopped outer loop on {}, t per iter: {}".format(self.iter_outer, t_per_iter))
		self.metrics['t_per_iter_outer'] = t_per_iter

	def solve_twopart(self, init_adv=None):
		self.set_alpha()
		self.measured = {}
		if init_adv is None:
			advertisement = self.init_advertisement()
		else:
			advertisement = init_adv
		self.path_measures = 0 
		# Initialize model of path probabilities
		self.calculate_path_probabilities(init=True)
		self.measure_paths(advertisement)
		self.calculate_path_probabilities()
		a_km1 = advertisement
		if self.verbose:
			# self.print_adv(advertisement)
			print("Optimizing over {} peers".format(self.n_peers))
			print(self.peers)
			print(self.user_networks)

		stop = False
		self.iter = 0

		t_start = time.time()
		# For analysis
		for k in ['actual_objectives', 'advertisements', 'effective_objectives', 'grads', 
			'latency_benefit_grads', 'prox_l1_grads', 'path_likelihoods', 'EL_difference']:
			self.metrics["twopart_" + k] = []

		rolling_delta = 10
		delta_alpha = .7
		delta_dot_alpha = .9
		rolling_delta_dot = -10
		current_objective = self.outer_objective(advertisement)
		last_objective = current_objective

		# Add to metrics
		self.metrics['twopart_actual_objectives'].append(current_objective)
		self.metrics['twopart_effective_objectives'].append(self.objective(self.threshold_a(advertisement)))
		self.metrics['twopart_advertisements'].append(copy.copy(advertisement))

		while not stop:
			# calculate gradients
			grads = self.gradient_fn(advertisement)
			# update advertisement by taking a gradient step with momentum and then applying the proximal gradient for L1
			a_k = advertisement
			w_k = a_k - self.alpha * grads + self.beta * (a_k - a_km1)
			advertisement = self.apply_prox_l1(w_k)
			self.metrics['twopart_advertisements'].append(copy.copy(advertisement))
			self.metrics['twopart_grads'].append(advertisement - a_k)
			a_km1 = a_k

			# another constraint we may want is 0 <= a_ij <= 1
			# the solution is just clipping to be in the set
			# clipping can mess with gradient descent
			advertisement = self.impose_advertisement_constraint(advertisement)

			# Stop when the objective doesn't change, but use an EWMA to track the change so that we don't spuriously exit
			rolling_delta = (1 - delta_alpha) * rolling_delta + delta_alpha * np.abs(current_objective - last_objective)
			stop = self.stopping_condition([self.iter,rolling_delta,rolling_delta_dot])
			self.iter += 1

			# Take a gradient step and update measured paths + probabilities
			if not np.array_equal(self.threshold_a(advertisement), self.threshold_a(a_km1)):
				self.measure_paths(advertisement)
				self.calculate_path_probabilities()

			# re-calculate objective
			last_objective = current_objective
			current_objective = self.objective(advertisement)
			self.metrics['twopart_actual_objectives'].append(current_objective)
			self.metrics['twopart_effective_objectives'].append(self.objective(self.threshold_a(advertisement)))

			# Calculate, advertise & measure information about the prefix that would 
			# give us the most new information
			maximally_informative_advertisement = self.solve_max_information(advertisement)
			if maximally_informative_advertisement is not None:
				self.measure_paths(maximally_informative_advertisement)
				self.calculate_path_probabilities()

			# Add to metrics
			self.latency_benefit(np.ones(advertisement.shape))
			self.metrics['twopart_path_likelihoods'].append(copy.copy(self.path_probabilities))

			t_per_iter = (time.time() - t_start) / self.iter

			if self.iter % 1 == 0 and self.verbose:
				print("Optimizing, iter: {}, t_per_iter : {}".format(self.iter, t_per_iter))
		print("Stopped train loop on {}, t per iter: {}, {} path measures".format(self.iter, t_per_iter, self.path_measures))
		self.metrics['twopart_t_per_iter'] = t_per_iter

def main():
	np.random.seed(31419)
	## Generating graphs
	gen_random_graph('test_graph',n_transit=3,n_user=50)

	# # Sweep mu
	# mu = .001
	# sae = Sparse_Advertisement_Eval(graph_fn="test_graph.csv", graph_md_fn="test_graph_md.json",mu=mu,verbose=False,cont_grads=True)
	# sae.compare_peer_value()


	# # Comparing different solutions
	# mu = .1
	# sae = Sparse_Advertisement_Eval(graph_fn="test_graph.csv", verbose=False,graph_md_fn="test_graph_md.json", mu=mu, cont_grads=False)
	# sae.compare_different_solutions()

	# ## Simple test
	# mu = .1
	# sas = Sparse_Advertisement_Solver(graph_fn="multi_prefix_test.csv", graph_md_fn="multi_prefix_test_md.json", 
	# 	mu=mu,verbose=True,cont_grads=False)
	# sas.solve()
	# sas.make_plots('outer')

	# # Comparing different solutions
	# mu = .1
	# sae = Sparse_Advertisement_Eval(graph_fn="multi_prefix_test.csv", verbose=False,
	# 	graph_md_fn="multi_prefix_test_md.json", mu=mu, cont_grads=False)
	# sae.compare_different_solutions(which='twopart')

	# ## Simple test
	# mu = .1
	# sas = Sparse_Advertisement_Solver(graph_fn="multi_prefix_test.csv", graph_md_fn="multi_prefix_test_md.json", 
	# 	mu=mu,verbose=True,cont_grads=False)
	# sas.solve_twopart()
	# sas.make_plots('twopart')
	# print(sas.threshold_a(sas.get_last_advertisement()))

	# ## Simple test
	# mu = .1
	# sas = Sparse_Advertisement_Solver(graph_fn="test_graph.csv", graph_md_fn="test_graph_md.json", 
	# 	mu=mu,verbose=True,cont_grads=False)
	# sas.solve_twopart()
	# sas.make_plots('twopart')
	# print(sas.threshold_a(sas.get_last_advertisement()))

	# Comparing different solutions
	mu = .1
	sae = Sparse_Advertisement_Eval(graph_fn="test_graph.csv", verbose=False,
		graph_md_fn="test_graph_md.json", mu=mu, cont_grads=False)
	sae.compare_different_solutions(which='twopart')

	# # Sweep mu
	# mu = .001
	# sae = Sparse_Advertisement_Eval(graph_fn="multi_prefix_test.csv", 
	# 	graph_md_fn="multi_prefix_test_md.json",mu=mu,verbose=False,cont_grads=False)
	# sae.compare_peer_value()


if __name__ == "__main__":
	main()