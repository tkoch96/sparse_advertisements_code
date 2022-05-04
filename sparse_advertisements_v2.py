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

CACHE_DIR = "cache"
GRAPH_DIR = "graphs"
TOY_GRAPH_FN = "toy_graph.csv"
TOY_GRAPH_MD_FN = "toy_graph_md.json"

cols = ['firebrick','salmon','orangered','lightsalmon','sienna','lawngreen','darkseagreen','palegoldenrod',
	'darkslategray','deeppink','crimson','mediumpurple','khaki','dodgerblue','lime','black','midnightblue',
	'lightsteelblue']

def _gen_random_graph(*args,**kwargs):
	global bp_cache
	bp_cache = {}
	gen_random_graph(*args,**kwargs)

class Sparse_Advertisement_Wrapper:
	def __init__(self, graph_fn=TOY_GRAPH_FN, graph_md_fn=TOY_GRAPH_MD_FN, mu=1.0, verbose=True, 
			cont_grads=False,  advertisement_cost="l1",
			init={'type':'normal','var':.001}, explore='entropy',
			n_prefixes=2):
		# (hyper-) parameters
		self.mu = mu
		self.advertisement_threshold = .5
		self.epsilon = .0005
		self.max_n_iter = 300
		self.iter_outer = 0
		self.iter = 0
		self.initialization = init
		self.explore = explore
		self.stopping_condition = lambda el : el[0] > self.max_n_iter or el[1] < self.epsilon or el[2] > 0
		# Gradient function
		if cont_grads:
			self.latency_benefit_fn = self.latency_benefit_continuous
			self.gradient_fn = self.gradients_continuous
		else:
			self.latency_benefit_fn = self.latency_benefit
			self.gradient_fn = self.gradients
		# Different types of advertisement cost
		self.advertisement_cost = {
			"l1": self.l1_norm,
			"sigmoid": self.sigmoid_prod_cost,
		}[advertisement_cost]
		if advertisement_cost == "sigmoid":
			self.gradient_fn = self.gradients_sigmoid
			self.proximal = False
			self.sigmoid_cost_k = 5
		else:
			self.proximal = True

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
		self.n_prefixes = n_prefixes

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


		self.verbose = verbose
		if self.verbose:
			print("Creating problem with {} peers, {} prefixes.".format(self.n_peers, self.n_prefixes))


	def gradients(self, *args, **kwargs):
		pass

	def gradients_continuous(self, *args, **kwargs):
		pass

	def gradients_sigmoid(self,*args, **kwargs):
		pass

	def l1_norm(self, a):
		return np.sum(np.abs(a).flatten())

	def sigmoid_sum_cost(self, a):
		# assume peers and prefixes cost the same
		peer_cost = 1 / (1 + np.exp(-1 * self.sigmoid_cost_k * (np.sum(a,axis=1) - self.advertisement_threshold)))
		peer_cost = np.sum(peer_cost)
		prefix_cost = 1 / (1 + np.exp(-1 * self.sigmoid_cost_k * (np.sum(a,axis=0) - self.advertisement_threshold)))
		prefix_cost = np.sum(prefix_cost)

		return peer_cost + prefix_cost

	def sigmoid_prod_cost(self, a):
		# assume peers and prefixes cost the same
		peer_cost = 1 - np.prod(1 - 1 / (1 + np.exp(-1 * self.sigmoid_cost_k * (a - self.advertisement_threshold))),axis=1)
		peer_cost = np.sum(peer_cost)
		prefix_cost = 1 - np.prod(1 - 1 / (1 + np.exp(-1 * self.sigmoid_cost_k * (a - self.advertisement_threshold))),axis=0)
		prefix_cost = np.sum(prefix_cost)

		return peer_cost + prefix_cost

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
					try:
						latency = self.latency_from_path(best_path)
					except KeyError:
						print(a)
						print(best_paths)
						a[pi] = 1
						print(self.calculate_best_paths(a))
						a[pi] = 0
						print("{} {}".format(pi,p))
						exit(0)
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

		# if plotit:
		# 	print(a)
		# 	for _i in range(self.n_prefixes):
		# 		print(p_mat[:,_i,:])
		# 	exit(0)

		min_b, max_b = BIG_BAD_VALUE, -1 * BIG_BAD_VALUE
		for b,p in zip(benefits.flatten(), p_mat.flatten()):
			if p == 0: continue
			min_b = np.minimum(min_b,b)
			max_b = np.maximum(max_b,b)
		n_pts = int(np.ceil((max_b - min_b) / delta))
		if min_b == BIG_BAD_VALUE:
			benefit = 0
			xsumx = np.array([min_b])
			psumx = np.array([1.0])		
			if calc_uncertainty:
				return benefit, (xsumx,psumx)
			else:
				return benefit
		if n_pts <= 1: 
			max_b = min_b + 2 # just inflate it
			n_pts = int(np.ceil((max_b - min_b) / delta))

		lbx = np.linspace(min_b,max_b,num=n_pts)
		# holds P(latency benefit) for each user
		px = np.zeros((n_pts, len(self.user_networks)))
		roll_min, roll_max = 0, 0

		for ui in range(len(self.user_networks)):
			minb,maxb = BIG_BAD_VALUE, -1 * BIG_BAD_VALUE
			all_pv = [(j,v,p) for j in range(self.n_prefixes) for v,p in zip(benefits[:,j,ui], p_mat[:,j,ui]) if p > 0]
			if len(all_pv) == 0:
				# this user has no paths to this prefix
				continue
			if len(all_pv) == 1:
				_, lb, p = all_pv[0]
				lbx_i = np.where(lb - lbx <= 0)[0][0]
				px[lbx_i, ui] += p
				maxb = np.maximum(maxb,lb)
				minb = np.minimum(minb,lb)
			else:
				all_pv = sorted(all_pv,key=lambda el : el[1])
				running_probs = np.zeros((self.n_prefixes))
				try:
					running_probs[all_pv[0][0]] = all_pv[0][2]
				except IndexError:
					print(all_pv)
					print(min_b)
					print(max_b)
					print(p_mat)
					exit(0)
				all_pref_inds = np.arange(self.n_prefixes)
				
				prefs_exist = list(set([el[0] for el in all_pv]))
				prefs_dont_exist = get_difference(list(range(self.n_prefixes)), prefs_exist)
				for pref_j in prefs_dont_exist:
					running_probs[pref_j] = 1 

				for i in range(1,len(all_pv)):
					pref_j, lb, p = all_pv[i]

					# calculate prob(max latency benefit)
					# we calculate this iteratively, from the smallest to the largest value
					# probability calc is basically probability of this value (p) times probability 
					# other prefixes are one of the smaller values (running prob)
					max_prob = p * np.prod(running_probs[all_pref_inds!=pref_j])
					running_probs[pref_j] += p
					if max_prob == 0 : continue

					lbx_i = np.where(lb - lbx <= 0)[0][0]
					px[lbx_i, ui] += max_prob
					maxb = np.maximum(maxb,lb)
					minb = np.minimum(minb,lb)
			if maxb != -1 * BIG_BAD_VALUE:
				roll_max += maxb
				roll_min += minb

		for ui in reversed(range(len(self.user_networks))):
			if np.sum(px[:,ui]) == 0:
				# This user experiences no benefit with probability 1
				# just remove the user from consideration
				px = np.concatenate([px[:,0:ui],px[:,ui+1:]],axis=1)

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
			# print(a)
			# for _i in range(self.n_prefixes):
			# 	print(benefits[:,_i,:])
			# for _i in range(self.n_prefixes):
			# 	print(p_mat[:,_i,:])
			
			print(np.sum(psumx))
			pickle.dump([px,benefits,p_mat,idx_start,n_pts_output,roll_min,roll_max],open('tmp.pkl','wb'))
			plt.plot(xsumx * len(self.user_networks), psumx)
			plt.xlabel("Benefit")
			plt.ylabel("P(Benefit)")
			plt.show()
			exit(0)

		if self.verbose:
			# Add to metrics
			e_user_lat = np.sum((self.measured_latencies * p_mat + 1e-8) / np.sum(p_mat+1e-14, axis=0), axis=0)
			min_e_user_lat = np.min(e_user_lat, axis=0) # minimum over prefixes
			actual_ul = self.user_latencies_from_bgp(a_effective)
			self.metrics['twopart_EL_difference'].append(actual_ul - min_e_user_lat)

		benefit = np.sum(xsumx * psumx)

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

	def init_advertisement(self):
		mode = self.initialization['type']
		if mode == 'random_binary':
			return np.random.randint(0,2,size=(self.n_peers, self.n_prefixes)) * 1.0
		elif mode == 'normal':
			return self.advertisement_threshold + np.sqrt(self.initialization['var']) \
				* np.random.normal(size=(self.n_peers, self.n_prefixes))

		elif mode == 'ones':
			return np.ones((self.n_peers, self.n_prefixes))
		elif mode == 'zeros':
			return np.zeros((self.n_peers, self.n_prefixes))
		elif mode == 'uniform':
			return np.random.random(size=(self.n_peers, self.n_prefixes))
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
		"""Approx actual objective with our belief."""
		norm_penalty = self.advertisement_cost(a)
		latency_benefit = self.latency_benefit_fn(a)
		# print("We believe: NP: {}, LB: {}".format(norm_penalty,latency_benefit))
		return self.mu * norm_penalty - latency_benefit

	def actual_objective(self, a):
		# Don't approximate the L0 norm with anything
		# Use actual latencies as if we were to really measure all the paths		

		c_pref = 1
		c_peer = 1

		has_peer = (np.sum(a,axis=1) > 0).astype(np.int32)
		has_pref = (np.sum(a,axis=0) > 0).astype(np.int32)

		# cost for different peers may be different
		cost_peers = np.sum(np.dot(has_peer,c_peer*np.ones(has_peer.shape)))
		# cost for different prefs likely not different
		cost_prefs = np.sum(has_pref) * c_pref
		norm_penalty = cost_peers + cost_prefs
		latency_benefit = self.latency_benefit_from_bgp(a)
		return self.mu * norm_penalty - latency_benefit

	def outer_objective(self, a):
		# Approximate L0 norm with whatever approximation we're using
		# Use actual latencies as if we were to really measure all the paths		

		norm_penalty = self.advertisement_cost(a)
		latency_benefit = self.latency_benefit_from_bgp(a)

		# print("Actual: NP: {}, LB: {}".format(norm_penalty,latency_benefit))
		return self.mu * norm_penalty - latency_benefit

class Sparse_Advertisement_Eval(Sparse_Advertisement_Wrapper):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.sas = Sparse_Advertisement_Solver(**kwargs)

	def compare_peer_value(self,minmu=.01,maxmu=5,make_plots=True):
		soln_keys = ['ours','oracle']#,'sparse_greedy']
		soln_peer_alives = {k:[] for k in soln_keys}
		linestyles = ['-','*','^']
		colors = ['r','b','k','orange','fuchsia','sandybrown']
		mus = np.logspace(np.log10(minmu),np.log10(maxmu),num=50)
		mus = np.flip(mus)
		init_advs = []
		if self.n_peers <= 1:
			return None
		for mu in mus:
			if make_plots:
				print(mu)
			self.mu = mu; self.sas.mu = mu
			self.compare_different_solutions(which='twopart',n_run=1,verbose=False)
			if self.oracle is None:
				# This only works if we can calculate the oracle solution
				return None

			init_advs.append(self.sas.metrics['twopart_advertisements'][0])

			our_adv = self.sas.get_last_advertisement()
			oracle_adv = self.oracle['l0_advertisement']
			sparse_greedy_adv = self.sparse_greedy['advertisement']

			for k,adv in zip(soln_keys, [our_adv, oracle_adv, sparse_greedy_adv]):
				peers_alive = (np.sum(self.threshold_a(adv),axis=1) > 0).astype(np.int32)
				soln_peer_alives[k].append(peers_alive)

		if make_plots:
			plt.rcParams["figure.figsize"] = (10,5)
			plt.rcParams.update({'font.size': 22})
			f,ax = plt.subplots(2,1)
		for si,k in enumerate(soln_keys):
			soln_peer_alives[k] = np.array(soln_peer_alives[k])
			if make_plots:
				if k == 'oracle': continue
				for i in range(self.n_peers):
					ax[0].semilogx(mus, soln_peer_alives[k][:,i], 
						linestyles[si+1], c=colors[i], label="{}".format(i))
		if make_plots:
			ax[0].legend(fontsize=14,ncol=2)
			ax[0].set_ylim([-.1,1.1])
			ax[0].set_xlabel("Lambda")
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
				if make_plots:
					if k == 'oracle': continue
					ax[1].semilogx(mus[0:-1], soln_peer_alives_smoothed[k][i,:], linestyles[si+1], 
						c=colors[i], label="Peer {}".format(i))
		
		rankings = {k:{} for k in soln_keys}
		for peer_i in range(self.n_peers):
			for k in soln_keys:
				crit_mu = np.where(np.flip(soln_peer_alives_smoothed[k][peer_i,:]) < self.advertisement_threshold)[0]
				if len(crit_mu) == 0:
					crit_mu = minmu
				else:
					crit_mu = np.flip(mus)[crit_mu[0]]
				rankings[k][peer_i] = crit_mu

		if make_plots:
			ax[1].legend(fontsize=14,ncol=2)
			ax[1].set_xlabel("Lambda")
			ax[1].set_ylim([0,1.0])
			ax[0].set_ylim([-.1,1.1])
			ax[1].set_ylabel("Peer On/Off")
			ax[1].set_aspect('equal', adjustable='datalim')
			for peer_i in rankings['ours']:
				# Circle the critical values
				mu_j = rankings['ours'][peer_i]
				try:
					where_mu_j = np.where(mus==mu_j)[0][0]
					if where_mu_j == len(mus) - 1:
						where_mu_j = len(mus) - 2
					ax[1].plot(mu_j,soln_peer_alives_smoothed['ours'][peer_i,where_mu_j],
						marker='o',ms=12,mfc=(1.,0.,0.,.05),mec='red')
				except IndexError:
					print(np.where(mus==mu_j))
					print(mu_j)
					print(mus)
					exit(0)
			save_fig("peer_value_demonstration.pdf")

		# Quantify number of pairwise disagreements they have
		peer_rankings = {}
		for k in rankings:
			peer_rankings[k] = []
			crit_mus = sorted(list(set(rankings[k].values())))
			for m in crit_mus:
				these_peers = [self.peers[p] for p in rankings[k] if rankings[k][p] == m]
				peer_rankings[k].append(these_peers)
		tmp = {}
		for k in peer_rankings:
			tmp[k] = {}
			for i in range(len(peer_rankings[k])):
				for p in peer_rankings[k][i]:
					tmp[k][p] = i
		peer_rankings = tmp
		disagreement = 0
		for i,peeri in enumerate(self.peers):
			for j,peerj in enumerate(self.peers):
				if j >= i: break
				orderings = {}
				for k in peer_rankings:
					if peer_rankings[k][peeri] > peer_rankings[k][peerj]:
						orderings[k] = 1
					elif peer_rankings[k][peeri] == peer_rankings[k][peerj]:
						orderings[k] = 0
					else:
						orderings[k] = -1
				if len(list(set(orderings.values()))) > 1:
					# disagreement
					disagreement += 1
		total_n = self.n_peers * (self.n_peers - 1) / 2 
		frac_disagree = disagreement / total_n

		# Quantify percent of peers that have monotone tendencies
		n_monotone, n_monotone_not_transit, lambda_j_distances, lambda_j_distances_not_transit = 0, 0, [], []
		for peer_i in range(self.n_peers):
			this_peer_vals = np.flip(soln_peer_alives_smoothed['ours'][peer_i,:])
			# monotonicity check
			d = np.diff(this_peer_vals) # derivative
			is_monotone = sum(d > 0) == 0

			# lambda_j distances
			cross = ((this_peer_vals - .5) < 0).astype(np.int32)
			dcross = np.diff(cross)
			lambda_j_locs = np.where(dcross == 1)[0]
			if len(lambda_j_locs) <= 1:
				lambda_j_distance = 0
			else:
				first_occurence = lambda_j_locs[0]
				last_occurence = lambda_j_locs[1]
				lambda_j_distance = np.log10(np.flip(mus)[last_occurence]/np.flip(mus)[first_occurence])
			lambda_j_distances.append(lambda_j_distance)
			if self.peers[peer_i] not in self.transit_providers:
				lambda_j_distances_not_transit.append(lambda_j_distance)
			if is_monotone:
				n_monotone += 1
			if self.peers[peer_i] not in self.transit_providers and is_monotone:
				n_monotone_not_transit += 1
		frac_monotone = n_monotone / self.n_peers
		frac_monotone_not_transit = n_monotone_not_transit / len(get_difference(self.peers, self.transit_providers))

		return {
			'frac_disagree': frac_disagree,
			'frac_monotone': frac_monotone,
			'frac_monotone_not_transit': frac_monotone_not_transit,
			'lambda_j_distances': lambda_j_distances,
			'lambda_j_distances_not_transit': lambda_j_distances_not_transit,
		}

	def solve_extremes(self, verbose=True):
		# Find minimal and maximal set of advertisements
		# maximal is just everything turned on
		maximal_advertisement = np.ones((self.n_peers, self.n_prefixes))
		maximal_objective = self.outer_objective(maximal_advertisement)

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
			scores.append(self.outer_objective(advertisement))

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
					deltas[peer_i,pref_i] = pre_obj - self.actual_objective(advertisement)
					advertisement[peer_i,pref_i] = 1 - advertisement[peer_i,pref_i]
					n_measures += 1
			# largest additional latency benefit
			best_delta = np.argmax(deltas.flatten())
			best_peer_i = best_delta // self.n_prefixes
			best_pref_i = best_delta % self.n_prefixes
			advertisement[best_peer_i,best_pref_i] = 1 - advertisement[best_peer_i,best_pref_i]

			stop = self.stopping_condition([i,np.abs(deltas[best_peer_i, best_pref_i]), deltas[best_peer_i, best_pref_i]])

			i += 1
		if self.verbose:
			print("Sparse greedy solution measured {} advertisements".format(n_measures))
		self.sparse_greedy = {
			'objective': self.actual_objective(advertisement),
			'advertisement': advertisement,
			'n_adv': n_measures,
		}

	def solve_greedy(self, init_adv, verbose=True):
		# at each iteration, toggle the entry that yields the largest delta latency benefit

		advertisement = copy.copy(self.threshold_a(init_adv))
		stop = False
		i = 0
		while not stop:
			pre_lat_ben = self.latency_benefit_from_bgp(advertisement)
			deltas = np.zeros((self.n_peers, self.n_prefixes))
			for peer_i in range(self.n_peers):
				for pref_i in range(self.n_prefixes):
					# toggle this entry
					advertisement[peer_i,pref_i] = 1 - advertisement[peer_i,pref_i]
					# should really measure paths, but whatever
					deltas[peer_i,pref_i] = self.latency_benefit_from_bgp(advertisement) - pre_lat_ben
					advertisement[peer_i,pref_i] = 1 - advertisement[peer_i,pref_i]
			# largest additional latency benefit
			best_delta = np.argmax(deltas.flatten())
			best_peer_i = best_delta // self.n_prefixes
			best_pref_i = best_delta % self.n_prefixes
			advertisement[best_peer_i,best_pref_i] = 1 - advertisement[best_peer_i,best_pref_i]

			stop = self.stopping_condition([i,np.abs(deltas[best_peer_i, best_pref_i]),-1])

			i += 1

		self.greedy = {
			'objective': self.actual_objective(advertisement),
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
			b = str(bin(i))[2:]
			ib = "0" * (n_arr - len(b)) + b
			for j,el in enumerate(ib):
				a[j] = int(el) * 1.0
			objs[i] = self.outer_objective(a.reshape((self.n_peers, self.n_prefixes)))
			actual_objs[i] = self.actual_objective(a.reshape((self.n_peers,self.n_prefixes)))
			all_as.append(copy.deepcopy(a))

		# Approx
		oracle_objective = np.min(objs)
		approx_oracle_adv = all_as[np.argmin(objs)].reshape(self.n_peers, self.n_prefixes)

		# Actual
		actual_oracle_objective = np.min(actual_objs)
		l0_oracle_adv = all_as[np.argmin(actual_objs)].reshape(self.n_peers, self.n_prefixes)

		if verbose:
			print("Oracle advertisements\n Approx for C(a) ({}): {}\nWith L0 ({}): {}".format(
				round(oracle_objective,2), approx_oracle_adv,
				round(actual_oracle_objective,2), l0_oracle_adv))

		self.oracle = {
			'approx_objective': oracle_objective,
			'l0_objective': actual_oracle_objective,
			'approx_advertisement': approx_oracle_adv,
			'l0_advertisement': l0_oracle_adv,
		}

	def compare_different_solutions(self, n_run=10, verbose=True, init_adv=None, which='twopart'):
		if verbose:
			print("Solving oracle")
		## oracle
		self.solve_oracle(verbose=verbose)
		## Extremes
		self.solve_extremes(verbose=verbose)
		objective_vals = {'ours': [], 'greedy': [], 'sparse greedy': []}
		n_advs = {'ours': [], 'greedy': [], 'sparse_greedy': []}
		our_advs = []
		if verbose:
			print(self.peers)
		for i in range(n_run):
			if verbose:
				print(i)
			# Initialize advertisement
			if init_adv is None:
				adv = self.init_advertisement()
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
			final_a = self.threshold_a(self.sas.get_last_advertisement())
			our_objective = self.actual_objective(final_a)
			our_advs.append(self.sas.get_last_advertisement())
			n_advs['ours'].append(self.sas.path_measures)
			if verbose:
				print("solving greedy")
			# Greedy solution
			self.solve_greedy(init_adv=adv,verbose=verbose)

			# Sparse greedy solution
			self.solve_sparse_greedy(init_adv=adv,verbose=verbose)
			objective_vals['ours'].append(our_objective)
			objective_vals['greedy'].append(self.greedy['objective'])
			objective_vals['sparse greedy'].append(self.sparse_greedy['objective'])
			n_advs['sparse_greedy'].append(self.sparse_greedy['n_adv'])

		if verbose:
			plt.rcParams["figure.figsize"] = (10,5)
			plt.rcParams.update({'font.size': 22})
			for v, a in zip(objective_vals['ours'], our_advs):
				print("{} ({}) -- {}".format(np.round(a,2).flatten(),self.threshold_a(a.flatten()),v))

			for k in objective_vals:
				if self.oracle is None: 
					v = 1
				else:
					v = self.oracle['approx_objective']
				x,cdf_x = get_cdf_xy(objective_vals[k] + .01 * np.abs(self.extreme['maximal_objective'] - v) *\
					 np.random.normal(size=(len(objective_vals[k]),)))
				plt.plot(x,cdf_x,label=k.capitalize())
			if self.oracle is not None:
				plt.axvline(x=self.oracle['approx_objective'],c='k')
				plt.annotate("L1 (Approx) \nOracle", (self.oracle['approx_objective'],.5))
				plt.axvline(x=self.oracle['l0_objective'],c='k')
				plt.annotate("L0 (Actual) \nOracle", (self.oracle['l0_objective'],.8))
			plt.axvline(x=self.extreme['minimal_objective'],c='turquoise')
			plt.annotate("Minimal", (self.extreme['minimal_objective'],.5))
			plt.axvline(x=self.extreme['maximal_objective'],c='darkorchid')
			plt.annotate("Maximal", (self.extreme['maximal_objective'],.5))
			plt.ylim([0,1.0])
			plt.legend()
			plt.xlabel("Final Objective Function Value")
			plt.ylabel("CDF of Trials")
			save_fig("comparison_to_strategies_demonstration.pdf")

		objective_vals['maximal'] = self.extreme['maximal_objective']
		objective_vals['minimal'] = self.extreme['minimal_objective']
		advs['maximal'] = self.extreme['maximal_advertisement']
		advs['minimal'] = self.extreme['minimal_advertisement']
		advs = {
			'ours': our_advs,
			'sparse greedy': self.sparse_greedy['advertisement'],
		}
		if self.oracle is None:  # we can't always solve for the oracle, infeasible
			objective_vals['approx_oracle'] = None
			objective_vals['l0_oracle'] = None
			advs['l0_oracle'] = None
			advs['approx_oracle'] = None
		else:
			objective_vals['approx_oracle'] = self.oracle['approx_objective']
			objective_vals['l0_oracle'] = self.oracle['l0_objective']
			advs['approx_oracle'] = self.oracle['approx_advertisement']
			advs['l0_oracle'] = self.oracle['l0_advertisement']
		return {
			'objectives': objective_vals,
			'advertisements': advs,
			'n_advs': n_advs,
		}

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

	def heaviside_gradient_sigmoid(self, a):
		x = a - self.advertisement_threshold
		grad = self.sigmoid_cost_k * np.exp(-self.sigmoid_cost_k*x) / (1 + np.exp(-self.sigmoid_cost_k*x))**2
		return grad

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

	def gradients(self, a, add_metrics=True):
		# gradient is the proximal gradient of the L1 norm
		# minus mu times gradient of L 
		# gradient of L is calculated via a continuous approximation
		L_grad = self.grad_latency_benefit(a)
		if add_metrics:
			self.metrics['twopart_benefit_grads'].append(L_grad)
			self.metrics['twopart_cost_grads'].append(self.mu * self.alpha * np.ones(L_grad.shape))

		return -1 * L_grad

	def gradients_sigmoid(self, a):
		# LB Grad
		l_grad = self.gradients(a, add_metrics=False)
		## calculate gradient of peer cost
		# old -- sum
		# peer_grad = self.heaviside_gradient_sigmoid(np.sum(a,axis=1))
		# peer_grad = np.tile(np.expand_dims(peer_grad,1),(1,self.n_prefixes))
		# pref_grad = self.heaviside_gradient_sigmoid(np.sum(a,axis=0))
		# pref_grad = np.tile(np.expand_dims(pref_grad,0),(self.n_peers,1))

		# prod cost
		S = self.heaviside_gradient_sigmoid(a)
		peer_prod = np.prod( 1 -  1 / (1 + np.exp(-1 * self.sigmoid_cost_k * (a - self.advertisement_threshold) )), axis=1)
		peer_prod = np.tile(np.expand_dims(peer_prod,axis=1),(1,self.n_prefixes))
		peer_grad = peer_prod * S

		pref_prod = np.prod( 1 -  1 / (1 + np.exp(-1 * self.sigmoid_cost_k * (a - self.advertisement_threshold) )), axis=0)
		pref_prod = np.tile(np.expand_dims(pref_prod,axis=0),(self.n_peers, 1))
		pref_grad = pref_prod * S		

		cost_grad = self.mu * (peer_grad + pref_grad)

		self.metrics['twopart_cost_grads'].append(cost_grad)
		self.metrics['twopart_benefit_grads'].append(l_grad)

		return cost_grad + l_grad

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
		f,ax = plt.subplots(4,2)

		# General convergence metrics plot
		which = args[0]

		all_as = np.array(self.metrics['{}_advertisements'.format(which)])
		all_grads = np.array(self.metrics['{}_grads'.format(which)])
		all_cost_grads = np.array(self.metrics['{}_cost_grads'.format(which)])
		all_benefit_grads = np.array(self.metrics['{}_benefit_grads'.format(which)])
		linestyles = ['-','*','^']
		colors = ['orange','brown','aqua','deeppink','peru','grey','k','tan']
		for pref_i in range(self.n_prefixes):
			pref_sty = linestyles[pref_i]
			for peer_i in range(self.n_peers):
				if 'xlimupper' in kwargs:
					ax[0,0].plot(kwargs['xlimupper'],all_as[:,peer_i,pref_i], pref_sty, c=colors[peer_i%len(colors)], label="Peer {} Prefix {}".format(self.peers[peer_i], pref_i))
				else:
					ax[0,0].plot(all_as[:,peer_i,pref_i][::5], pref_sty, c=colors[peer_i%len(colors)], label="Peer {} Prefix {}".format(self.peers[peer_i], pref_i))
				ax[1,0].plot(all_grads[:,peer_i,pref_i], c=colors[peer_i%len(colors)], label="Peer {} Prefix {}".format(self.peers[peer_i], pref_i))
				ax[2,0].plot(all_cost_grads[:,peer_i,pref_i], c=colors[peer_i%len(colors)], label="Peer {} Prefix {}".format(self.peers[peer_i], pref_i))
				ax[3,0].plot(all_benefit_grads[:,peer_i,pref_i], c=colors[peer_i%len(colors)], label="Peer {} Prefix {}".format(self.peers[peer_i], pref_i))
		ax[0,0].legend(fontsize=6)
		ax[0,0].set_ylabel("a")
		ax[1,0].set_ylabel("Net Grad")
		ax[2,0].set_ylabel("Cost Grad")
		ax[3,0].set_ylabel("LB Grad")

		all_objectives = self.metrics['{}_actual_objectives'.format(which)]
		all_pseudo_objectives = self.metrics['{}_pseudo_objectives'.format(which)]
		all_effective_ojectives = self.metrics['{}_effective_objectives'.format(which)]
		ax[1,1].plot(all_pseudo_objectives)
		ax[1,1].set_ylabel("Believed Objective")
		ax[0,1].plot(all_objectives)
		ax[0,1].set_ylabel("GT Objective")
		ax[2,1].plot(all_effective_ojectives)
		ax[2,1].set_ylabel("GT Effective Objective")

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
		self.alpha = .05

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

		def value_func(u):
			benefits,probs = u
			if self.explore == 'positive_benefit':
				if np.sum(probs[benefits>=current_benefit]) > .99: return 0
				v = np.sum(benefits[benefits>current_benefit] * probs[benefits>current_benefit])
			elif self.explore == 'entropy':
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
				_, u = self.latency_benefit(a, calc_uncertainty=True,plotit=False)
				if n_flips > 1: # we can't afford to look through anything but nearest neighbors
					if value_func(u) > MIN_POTENTIAL_VALUE:
						return a
				uncertainties[flips] = u
				for flip in flips: # flip back
					a[flip] = 1 - a[flip]
				# if time.time() - t_start > max_time: return None

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
						print('woops')
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
		for k in ['actual_objectives', 'advertisements', 'effective_objectives', 
			'pseudo_objectives', 'grads', 'cost_grads', 'benefit_grads', 
			'latency_benefit_grads', 'prox_l1_grads', 'path_likelihoods', 'EL_difference']:
			self.metrics["twopart_" + k] = []

		rolling_delta = 10
		delta_alpha = .7
		delta_dot_alpha = .9
		rolling_delta_dot = -10
		current_objective = self.outer_objective(advertisement)
		current_pseudo_objective = self.objective(advertisement)
		last_objective = current_pseudo_objective

		# Add to metrics
		self.metrics['twopart_pseudo_objectives'].append(current_pseudo_objective)
		self.metrics['twopart_actual_objectives'].append(current_objective)
		self.metrics['twopart_effective_objectives'].append(self.outer_objective(self.threshold_a(advertisement)))
		self.metrics['twopart_advertisements'].append(copy.copy(advertisement))

		while not stop:
			# calculate gradients
			grads = self.gradient_fn(advertisement)
			# update advertisement by taking a gradient step with momentum and then applying the proximal gradient for L1
			a_k = advertisement
			w_k = a_k - self.alpha * grads + self.beta * (a_k - a_km1)
			if self.proximal:
				advertisement = self.apply_prox_l1(w_k)
			else:
				advertisement = w_k
			a_km1 = a_k

			# another constraint we may want is 0 <= a_ij <= 1
			# the solution is just clipping to be in the set
			# clipping can mess with gradient descent
			advertisement = self.impose_advertisement_constraint(advertisement)

			self.metrics['twopart_advertisements'].append(copy.copy(advertisement))
			self.metrics['twopart_grads'].append(advertisement - a_k)

			# Stop when the objective doesn't change, but use an EWMA to track the change so that we don't spuriously exit
			rolling_delta = (1 - delta_alpha) * rolling_delta + delta_alpha * np.abs(current_pseudo_objective - last_objective)
			stop = self.stopping_condition([self.iter,rolling_delta,rolling_delta_dot])
			self.iter += 1

			# Take a gradient step and update measured paths + probabilities
			if not np.array_equal(self.threshold_a(advertisement), self.threshold_a(a_km1)):
				self.measure_paths(advertisement)
				self.calculate_path_probabilities()

			# re-calculate objective
			last_objective = current_pseudo_objective
			current_pseudo_objective = self.objective(advertisement)
			self.metrics['twopart_pseudo_objectives'].append(current_pseudo_objective)
			self.metrics['twopart_actual_objectives'].append(self.outer_objective(advertisement))
			self.metrics['twopart_effective_objectives'].append(self.outer_objective(copy.copy(self.threshold_a(advertisement))))

			# Calculate, advertise & measure information about the prefix that would 
			# give us the most new information
			maximally_informative_advertisement = self.solve_max_information(advertisement)
			if maximally_informative_advertisement is not None:
				self.measure_paths(maximally_informative_advertisement)
				self.calculate_path_probabilities()

			# Add to metrics
			self.latency_benefit(np.ones(advertisement.shape))
			self.metrics['twopart_path_likelihoods'].append(copy.copy(self.path_probabilities))

			self.t_per_iter = (time.time() - t_start) / self.iter

			if self.iter % 100 == 0 and self.verbose:
				print("Optimizing, iter: {}, t_per_iter : {}".format(self.iter, self.t_per_iter))
		if self.verbose:
			print("Stopped train loop on {}, t per iter: {}, {} path measures, O:{}".format(
				self.iter, self.t_per_iter, self.path_measures, current_pseudo_objective))
		self.metrics['twopart_t_per_iter'] = self.t_per_iter


N_SIM = 200

def do_eval_compare_peer_value():
	metrics = {}
	metrics_fn = os.path.join(CACHE_DIR, 'compare_peer_value.pkl')
	if os.path.exists(metrics_fn):
		metrics = pickle.load(open(metrics_fn,'rb'))
	for i in range(N_SIM):
		print("CPV: {}".format(i))
		if metrics != {}:
			if len([metrics[k] for k in metrics][0]) >= N_SIM:
				break
		ret  = None
		while ret is None:
			_gen_random_graph('test_graph',n_transit=2,n_user=5+np.random.randint(5))
			mu = .001
			sae = Sparse_Advertisement_Eval(graph_fn="test_graph.csv", 
				graph_md_fn="test_graph_md.json",mu=mu,verbose=False)
			ret = sae.compare_peer_value(make_plots=False)
		for k in ret:
			try:
				metrics[k].append(ret[k])
			except KeyError:
				metrics[k] = [ret[k]]
	pickle.dump(metrics, open(metrics_fn,'wb'))

	plt.rcParams["figure.figsize"] = (10,5)
	plt.rcParams.update({'font.size': 22})
	f,ax=plt.subplots(1,1)
	labs = {"frac_disagree": "Disagreement with Oracle", "lambda_j_distances": "Uncertainty", 
		"lambda_j_distances_not_transit": "Not-Transit Uncertainty"}
	for k in ['frac_disagree']:
		x,cdf_x = get_cdf_xy(metrics[k])
		ax.plot(x,cdf_x,label=labs[k])
	ax.set_ylim([0,1.0])
	ax.set_xlabel("Fraction of Peer Importance Agreement")
	ax.set_ylabel("CDF of Trials")
	save_fig("peer_value_disagreement.pdf")


	plt.rcParams["figure.figsize"] = (10,5)
	plt.rcParams.update({'font.size': 22})
	f,ax=plt.subplots(1,1)
	for k in ['frac_monotone', 'frac_monotone_not_transit']:
		v = [np.max(arr) for arr in metrics[k]]
		x,cdf_x = get_cdf_xy(v)
		ax.plot(x,cdf_x,label=labs[k])
	ax.set_ylim([0,1.0])
	ax.set_xlabel("Lambda_j Uncertainty")
	ax.legend()
	ax.set_ylabel("CDF of Trials")
	save_fig("lambda_j_uncertainty.pdf")


def err_adv(adv1,adv2):
	# advertisements are said to be equal if their columns are a permutation of each other
	# so find the minimum error over all permutations of columns

	n_col = adv1.shape[1]
	errs = []
	for p in itertools.permutations(list(range(n_col)), n_col):
		tmp = adv1[:,p]
		errs.append(np.sum(np.abs(tmp.flatten() - adv2.flatten())))
	return min(errs)


def do_eval_compare_initializations():
	mu = .1
	inits = [{'type': 'uniform'}, {'type': 'ones'}, {'type': 'zeros'}, {'type':'random_binary'},
		{'type': 'normal', 'var': .01}, {'type': 'normal', 'var': .001}]
	hr_labs = ["Uniform","All On","All Off","Random","N(.5,.01)","N(.5,.001)"]
	
	metrics = {}
	metrics_fn = os.path.join(CACHE_DIR, 'compare_inits.pkl')
	if os.path.exists(metrics_fn):
		metrics = pickle.load(open(metrics_fn,'rb'))

	for init_i, initialization in enumerate(inits):
		if init_i in metrics: continue
		metrics[init_i] = {
			'n_off_adv_approx': [],
			'n_off_adv_l0': [],
			'delta_obj_approx': [],
			'delta_obj_l0': [],
			'delta_obj_l0_greedy': [],
		}
		for _i in range(N_SIM):
			ret_oracle = None
			while ret_oracle is None:
				_gen_random_graph('test_graph',n_transit=1,n_user=4)
				sae = Sparse_Advertisement_Eval(graph_fn="test_graph.csv", 
					graph_md_fn="test_graph_md.json", mu=mu,
					verbose=False,cont_grads=False, init=initialization)
				ret = sae.compare_different_solutions(n_run=1,verbose=False)
				ret_oracle = ret['objectives']['approx_oracle']
			our_adv = sae.threshold_a(ret['advertisements']['ours'][0])
			metrics[init_i]['n_off_adv_approx'].append(err_adv(our_adv,ret['advertisements']['approx_oracle']))
			metrics[init_i]['n_off_adv_l0'].append(err_adv(our_adv,ret['advertisements']['l0_oracle']))
			metrics[init_i]['delta_obj_approx'].append(np.abs(ret['objectives']['ours'][0] - \
				ret['objectives']['approx_oracle']))
			metrics[init_i]['delta_obj_l0'].append(np.abs(ret['objectives']['ours'][0] - \
				ret['objectives']['l0_oracle']))
			metrics[init_i]['delta_obj_l0_greedy'].append(np.abs(ret['objectives']['sparse greedy'][0] - \
				ret['objectives']['l0_oracle']))
	pickle.dump(metrics,open(metrics_fn,'wb'))
	
	plt.rcParams["figure.figsize"] = (10,5)
	plt.rcParams.update({'font.size': 22})
	f,ax=plt.subplots(1,1)
	for i in metrics:
		# # % of runs we arrive within N entries of optimal advertisement
		# n_off_approx_optimal = metrics[i]['n_off_adv_approx']
		# x,cdf_x = get_cdf_xy(n_off_approx_optimal)
		# ax[0].plot(x,cdf_x,c=cols[2*i],label=hr_labs[i] + " Approx")
		n_off_optimal = metrics[i]['n_off_adv_l0']
		x,cdf_x = get_cdf_xy(n_off_optimal)
		ax.plot(x,cdf_x,c=cols[2*i+1],label=hr_labs[i].capitalize())
	ax.legend()
	ax.set_xlabel("Number of Entries off Optimal")
	ax.set_ylabel("CDF of Trials")
	save_fig("compare_initializations_delta_advertisement.pdf")

	plt.rcParams["figure.figsize"] = (10,5)
	plt.rcParams.update({'font.size': 22})
	f,ax=plt.subplots(1,1)
	for i in metrics:
		# # % of runs we arrive within epsilon of optimal cost-beenfit
		# epsilon_off_approx_optimal = metrics[i]['delta_obj_approx']
		# x,cdf_x = get_cdf_xy(epsilon_off_approx_optimal)
		# ax[1].plot(x,cdf_x,c=cols[2*i],label=hr_labs[i] + " Approx")
		epsilon_off_optimal = metrics[i]['delta_obj_l0']
		x,cdf_x = get_cdf_xy(epsilon_off_optimal)
		ax.plot(x,cdf_x,c=cols[2*i+1],label=hr_labs[i].capitalize())
	ax.legend()
	ax.set_ylim([0,1.0])
	ax.set_xlabel("Objective Function Error")
	ax.set_ylabel("CDF of Trials")
	save_fig("compare_initializations_delta_objective.pdf")

def do_eval_compare_strategies():
	mu = 1
	metrics = {}
	metrics_fn = os.path.join(CACHE_DIR, 'compare_strategies.pkl')
	if os.path.exists(metrics_fn):
		metrics = pickle.load(open(metrics_fn,'rb'))
	for cost_strat in ['sigmoid', 'l1']:
		if cost_strat in metrics: continue
		metrics[cost_strat] = {'p_ours_best': []}
		for _i in range(N_SIM):
			ret = None
			while ret is None:
				_gen_random_graph('test_graph',n_transit=2,n_user=10+np.random.randint(5))
				sae = Sparse_Advertisement_Eval(graph_fn="test_graph.csv", 
					graph_md_fn="test_graph_md.json", mu=mu,
					verbose=False,advertisement_cost=cost_strat,
					n_prefixes=1+np.random.randint(3))
				ret = sae.compare_different_solutions(n_run=30,verbose=False)
			our_obj = ret['objectives']['ours']
			ours_best = np.ones((len(our_obj)))
			print(ret['objectives'])
			for k in ret['objectives']:
				if k == 'ours' or 'oracle' in k: continue
				if type(ret['objectives'][k]) == np.float64:
					for i, obj in enumerate(our_obj):
						if obj > ret['objectives'][k]:
							ours_best[i] = 0
				else:
					i=0
					for ours, theirs in zip(our_obj, ret['objectives'][k]):
						if ours > theirs:
							ours_best[i] = 0
						i += 1
			metrics[cost_strat]['p_ours_best'].append(np.sum(ours_best) / len(our_obj))
	
	pickle.dump(metrics,open(metrics_fn,'wb'))
	
	plt.rcParams["figure.figsize"] = (10,5)
	plt.rcParams.update({'font.size': 22})
	f,ax=plt.subplots(1,1)
	labs = {'sigmoid': "Sigmoid Cost", 'l1': "L1 Cost"}
	print(metrics['l1']['p_ours_best'])
	for k in metrics:
		x, cdf_x = get_cdf_xy(1 - np.array(metrics[k]['p_ours_best']))
		ax.plot(x,cdf_x,label=labs[k])
	ax.legend()
	ax.set_ylim([0,1.0])
	ax.set_ylabel("CDF of Trial Batches")
	ax.set_xlabel("Fraction of Trials Our Algorithm Was Not The Best")
	save_fig("compare_across_strategies.pdf")

def do_eval_scale():
	metrics = {
		't_per_iter': [],
		'n_iter': [],
		'n_users': [],
		'n_peers': [],
		'n_advs': {
			'sparse_greedy': [],
			'ours_entropy': [],
			'ours_positive_benefit': [],
		},
	}
	metrics_fn = os.path.join(CACHE_DIR, 'scale_eval.pkl')
	if os.path.exists(metrics_fn):
		metrics = pickle.load(open(metrics_fn,'rb'))
	mu = .1
	# for n_user in [5,10,15,20,25,30,35,40]:
	# 	print("NU: {}".format(n_user))
	# 	for _i in range(N_SIM):
	# 		print(_i)
	# 		_gen_random_graph('test_graph',n_transit=1,n_user=n_user)
	# 		sae = Sparse_Advertisement_Eval(graph_fn="test_graph.csv", 
	# 			graph_md_fn="test_graph_md.json", mu=mu,
	# 			verbose=False,	explore='entropy')
	# 		ret = sae.compare_different_solutions(n_run=1,verbose=False)
	# 		metrics['t_per_iter'].append(sae.sas.t_per_iter)
	# 		metrics['n_iter'].append(sae.sas.iter)
	# 		metrics['n_advs']['sparse_greedy'].append(ret['n_advs']['sparse_greedy'][0])
	# 		metrics['n_advs']['ours_entropy'].append(ret['n_advs']['ours'][0])
	# 		_gen_random_graph('test_graph',n_transit=1,n_user=n_user)
	# 		sae = Sparse_Advertisement_Eval(graph_fn="test_graph.csv", 
	# 			graph_md_fn="test_graph_md.json", mu=mu,
	# 			verbose=False, explore='positive_benefit')
	# 		ret = sae.compare_different_solutions(n_run=1,verbose=False)
	# 		metrics['n_advs']['ours_positive_benefit'].append(ret['n_advs']['ours'][0])

	# 		metrics['n_users'].append(len(sae.sas.user_networks))
	# 		metrics['n_peers'].append(len(sae.sas.peers))

	# pickle.dump(metrics,open(metrics_fn,'wb'))

	plt.rcParams["figure.figsize"] = (10,5)
	plt.rcParams.update({'font.size': 22})
	f,ax = plt.subplots(1,1)
	
	x = np.array(metrics['n_users']) * np.array(metrics['n_peers'])	
	y = np.array(metrics['t_per_iter'])
		
	z = np.poly1d(np.polyfit(x,y,2))
	x_sm = np.linspace(x[0],x[-1])
	y_sm = z(x_sm)
	ax.plot(x_sm, y_sm,'r',marker='.')
	ax.tick_params(axis='y', colors='red')
	ax.yaxis.label.set_color('red')
	ax.set_ylabel("Time per Iter (s)")

	axd = ax.twinx()
	y = np.array(metrics['n_iter'])
	z = np.poly1d(np.polyfit(x,y,2))
	y_sm = z(x_sm)
	axd.plot(x_sm, y_sm,'k')
	axd.tick_params(axis='y', colors='black')
	axd.set_ylabel("Number of Iters")

	ax.set_xlabel("Problem Size")

	save_fig("scale_time.pdf")


	plt.rcParams["figure.figsize"] = (10,5)
	plt.rcParams.update({'font.size': 22})
	f,ax = plt.subplots(1,1)

	x = np.array(metrics['n_users']) * np.array(metrics['n_peers'])	
	labs = {
		'sparse_greedy': "Greedy",
		"ours_entropy": "Entropy Exploration",
		"ours_positive_benefit": "Max Benefit Exploration"
	}
	for i,k in enumerate(metrics['n_advs']):
		y = metrics['n_advs'][k]
		z = np.poly1d(np.polyfit(x,y,2))
		y_sm = z(x_sm)
		ax.plot(x_sm,y_sm,c=cols[3*i],label=labs[k])
	ax.legend()
	ax.set_xlabel("Problem Size")
	ax.set_ylabel("Number of Advertisements")
	save_fig("scale_n_advs.pdf")


def main():
	np.random.seed(31413)
	## Generating graphs
	_gen_random_graph('test_graph',n_transit=1,n_user=5)

	# # Comparing different solutions
	# mu = .05
	# sae = Sparse_Advertisement_Eval(graph_fn="multi_prefix_test.csv", verbose=False,
	# 	graph_md_fn="multi_prefix_test_md.json", mu=mu, explore='positive_benefit')
	# print(sae.compare_peer_value())
	# exit(0)

	# # ## Simple test
	# mu = .1
	# sas = Sparse_Advertisement_Solver(graph_fn="test_graph.csv", graph_md_fn="test_graph_md.json", 
	# 	mu=mu,verbose=True)
	# sas.solve_twopart()
	# print(sas.threshold_a(sas.get_last_advertisement()))
	# print(sas.get_last_objective(effective=True))
	# sas.make_plots('twopart')
	# exit(0)

	# # Comparing different solutions
	# mu = .05
	# sae = Sparse_Advertisement_Eval(graph_fn="test_graph.csv", verbose=True,
	# 	graph_md_fn="test_graph_md.json", mu=mu, explore='positive_benefit')
	# sae.compare_different_solutions(which='twopart',n_run=150)
	# exit(0)

	# ## Simple test
	# mu = .1
	# sas = Sparse_Advertisement_Solver(graph_fn="test_graph.csv", graph_md_fn="test_graph_md.json", 
	# 	mu=mu,verbose=True,cont_grads=False,advertisement_cost="sigmoid")
	# sas.solve_twopart()
	# sas.make_plots('twopart')
	# print(sas.threshold_a(sas.get_last_advertisement()))

	# # Comparing different solutions
	# mu = .001
	# sae = Sparse_Advertisement_Eval(graph_fn="test_graph.csv", verbose=False,
	# 	graph_md_fn="test_graph_md.json", mu=mu, cont_grads=False,advertisement_cost="sigmoid")
	# sae.compare_different_solutions(which='twopart')

	# do_eval_scale()
	# do_eval_compare_peer_value()
	# do_eval_compare_initializations()
	do_eval_compare_strategies()

if __name__ == "__main__":
	main()