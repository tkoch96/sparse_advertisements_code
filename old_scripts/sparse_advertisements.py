import numpy as np, copy, json, time, os, itertools
import matplotlib.pyplot as plt
import networkx as nx

from bgpsim import (
	Announcement,
	ASGraph,
)
from helpers import *
from graph_utils import *

KM_TO_MS = .01
bp_cache = {}
lb_cache = {}


GRAPH_DIR = "graphs"
TOY_GRAPH_FN = "toy_graph.csv"
TOY_GRAPH_MD_FN = "toy_graph_md.json"

class Sparse_Advertisement_Wrapper:
	def __init__(self, mu=1.0, graph_fn=TOY_GRAPH_FN, graph_md_fn=TOY_GRAPH_MD_FN, verbose=True, cont_grads=False,
		gaussian_noise_var = .01):
		# (hyper-) parameters
		self.mu = mu
		self.advertisement_threshold = .5
		self.epsilon = .0001
		self.max_n_iter = 500
		self.gaussian_noise_var = gaussian_noise_var # initialization noise variance for gaussian 
		self.stopping_condition = lambda el : el[0] > self.max_n_iter or el[1] < self.epsilon
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
		self.n_peers = len(self.peers)
		self.n_prefixes = 2

		self.verbose = verbose

	# THIS LIBRARY IS NO BUENO
	# def draw_graph(self):
	# 	from networkx.drawing.nx_agraph import graphviz_layout
	# 	G = nx.DiGraph()

	# 	def is_transit(asn):
	# 		return int(asn) > 1 and int(asn) < 1000

	# 	for (as1,as2),rel in self.my_graph.items():
	# 		if is_transit(as1) or is_transit(as2): continue
	# 		if rel == -1:
	# 			G.add_nodes_from([as1,as2])
	# 		elif rel == 0:
	# 			G.add_nodes_from([as1,as2])
	# 			G.add_nodes_from([as2,as1])
	# 		elif rel == 1:
	# 			G.add_nodes_from([as2,as1])
	# 		G.add_edge(as1,as2, length = 1/self.edge_lengths[as1,as2])

	# 	nx.nx_agraph.write_dot(G,'test.dot')


	# 	val_map = {}
	# 	size_map = {}
	# 	transit_providers = []
	# 	for node in G.nodes():
	# 		if node == '1': 
	# 			val_map[node] = 1.5
	# 			size_map[node] = 10000
	# 		elif is_transit(node): # transit provider
	# 			val_map[node] = .55
	# 			transit_providers.append(node)
	# 		elif int(node) < 1e6: # provider
	# 			val_map[node] = .4
	# 			size_map[node] = 500
	# 		else: # user
	# 			val_map[node] = .25
	# 			size_map[node] = 100
	# 	not_transit_providers = get_difference(G.nodes(), transit_providers)
	# 	values = [val_map[node] for node in not_transit_providers]

	# 	# Specify the edges you want here
	# 	red_edges = []
	# 	edge_colours = ['black' if not edge in red_edges else 'red'
	# 	                for edge in G.edges()]
	# 	black_edges = [edge for edge in G.edges() if edge not in red_edges]

	# 	# Need to create a layout when doing
	# 	# separate calls to draw nodes and edges
	# 	pos = {'1': (2,10)}
	# 	for node in not_transit_providers:
	# 		pos[node] = np.random.uniform((2,))
	# 	fixed = ['1']
	# 	pos = nx.spring_layout(G,weight='length',fixed=fixed,pos=pos)
	# 	# nx.nx_agraph.write_dot(G,'test.dot')
	# 	# pos = graphviz_layout(G, prog='dot')
	# 	nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), nodelist=not_transit_providers,
	# 	                       node_color = values, node_size = [size_map[node] for node in not_transit_providers])
	# 	nx.draw_networkx_labels(G, pos)
	# 	nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
	# 	nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
	# 	plt.show()

	def gradients(self, *args, **kwargs):
		pass

	def gradients_continuous(self, *args, **kwargs):
		pass

	def latency_benefit(self, a):
		a_effective = self.threshold_a(a)
		try:
			return lb_cache[tuple(a_effective.flatten())]
		except KeyError:
			pass
		user_latencies = np.inf * np.ones((len(self.user_networks)))
		for prefix_i in range(a.shape[1]):
			if np.sum(a_effective[:,prefix_i]) == 0:
				continue
			best_paths = self.calculate_best_paths(a_effective[:,prefix_i])
			for i,user_network in enumerate(self.user_networks):
				if best_paths[user_network]['best-paths'] == []:
					latency = np.inf
				else:
					best_path = [user_network] + list(best_paths[user_network]['best-paths'][0])
					latency = self.latency_from_path(best_path)
				if latency < user_latencies[i]: user_latencies[i] = latency
		benefit = self.benefit_from_user_latencies(user_latencies)
		lb_cache[tuple(a_effective.flatten())] = benefit
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

	def benefit_from_user_latencies(self, user_latencies):
		# sum of the benefits, simple model for benefits is 1 / latency
		BAD_VALUE = 0 # if a user has no route, we say the benefit is this value
		user_benefits = [1/(lat + .001) if lat != np.inf else BAD_VALUE for lat in user_latencies]
		# average user benefit -- important that this function is not affected by the number of user groups
		return sum(user_benefits) / len(user_benefits)

	def calculate_best_paths(self, a):
		try:
			return bp_cache[tuple(a)]
		except KeyError:
			pass

		# a is a Nx1 vector where N is the number of peers, 
		# a[i] indicates whether we announce to peer i
		t_s = time.time()
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

	def init_advertisement(self, mode='gaussian'):
		if mode == 'simple':
			return np.random.randint(0,2,size=(self.n_peers, self.n_prefixes)) * 1.0
		elif mode == 'gaussian':
			return self.advertisement_threshold + self.gaussian_noise_var * np.random.normal(size=(self.n_peers, self.n_prefixes))
		else:
			raise ValueError("Adv init {} not recognized.".format(mode))

	def latency_from_path(self, path):
		latency = 0
		for edge in range(len(path) - 1):
			latency += self.edge_lengths[path[edge],path[edge+1]]
		return latency * KM_TO_MS

	def threshold_a(self, a):
		return (a > self.advertisement_threshold).astype(np.float32)

	def objective(self, a, v=False):
		"""Approx actual objective with 1 norm."""
		norm_penalty = np.sum(np.abs(a).flatten())
		latency_benefit = self.latency_benefit_fn(a)
		return self.mu * norm_penalty - latency_benefit

	def actual_objective(self, a):
		"""Actual objective is the number of peering relationships and the number of prefixes."""
		c_pref = .5
		c_peer = .5

		has_peer = (np.sum(a,axis=1) > 0).astype(np.int32)
		has_pref = (np.sum(a,axis=0) > 0).astype(np.int32)

		# cost for different peers may be different
		cost_peers = np.sum(np.dot(has_peer,c_peer*np.ones(has_peer.shape)))
		# cost for different prefs may be different
		cost_prefs = np.sum(has_pref) * c_pref
		norm_penalty = cost_peers + cost_prefs
		latency_benefit = self.latency_benefit_fn(a)
		return self.mu * norm_penalty - latency_benefit

class Sparse_Advertisement_Eval(Sparse_Advertisement_Wrapper):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.sas = Sparse_Advertisement_Solver(**kwargs)

	def compare_peer_value(self,minmu=.01,maxmu=4.0):
		soln_keys = ['ours','oracle']#,'sparse_greedy']
		soln_peer_alives = {k:[] for k in soln_keys}
		linestyles = ['-','*','^']
		colors = ['r','b','k','orange','fuchsia','sandybrown']
		mus = np.linspace(maxmu,minmu,num=5)
		init_advs = []
		for mu in mus:
			print(mu)
			self.mu = mu; self.sas.mu = mu
			self.compare_different_solutions(n_run=1,verbose=False)
			init_advs.append(self.sas.metrics['advertisements'][0])

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
				ax[0].plot(mus+i*1.0/self.n_peers, soln_peer_alives[k][:,i] + (-.1  + .1 * si / len(soln_keys)), linestyles[si], c=colors[i], label="{}--{}".format(k,i))

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
				ax[1].plot(mus[0:-1]+i*1.0/self.n_peers, soln_peer_alives_smoothed[k][i,:] + (-.1  + .1 * si / len(soln_keys)), linestyles[si], c=colors[i], label="{}--{}".format(k,i))
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

		# diffs = np.diff(soln_peer_alives['ours'], axis=0)
		# print(diffs)
		# print(self.peers)
		# bad_occurences = np.where(diffs < 0 )[0]
		# for bad_ind in bad_occurences:
		# 	peer_flipped = self.peers[np.where(diffs[bad_ind] < 0)[0][0]]
		# 	print("Peer : {} flipped".format(peer_flipped))
		# 	for i,mu in enumerate(mus[bad_ind-1:bad_ind+1]):
		# 		self.mu = mu; self.sas.mu = mu
		# 		self.compare_different_solutions(n_run=1,verbose=False,init_adv=init_advs[bad_ind-1+i])
		# 		print("Mu : {}".format(self.mu))
		# 		print("Init: {}".format(init_advs[bad_ind-1+i]))
		# 		print(self.oracle['l1_advertisement'])
		# 		print(self.sas.get_last_advertisement())
		# 		self.sas.make_plots()

	def solve_extremes(self, verbose=True):
		# Find minimal and maximal set of advertisements
		# maximal is just everything turned on
		maximal_adv = np.ones((self.n_peers, self.n_prefixes))
		maximal_objective = self.objective(maximal_adv)

		# minimal is pick smallest set s.t. every user has a path
		# look for transit providers, compute customer cones
		rels_by_asn = {}
		for (as1,as2),rel in self.my_graph.items():
			try:
				rels_by_asn[as1].append(rel)
			except KeyError:
				rels_by_asn[as1] = [rel]
		transit_providers = [asn for asn,rels in rels_by_asn.items() if 1 not in rels]
		transit_providers = get_intersection(transit_providers, self.peers)

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
		for transit_provider in transit_providers:
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
			'maximal_advertisement': maximal_adv,
			'maximal_objective': maximal_objective,
		}

	def solve_sparse_greedy(self, init_adv, verbose=True):
		# at each iteration, toggle the entry that yields the largest objective function benefit

		advertisement = self.threshold_a(init_adv)
		stop = False
		i = 0
		while not stop:
			pre_obj = self.actual_objective(advertisement)
			deltas = np.zeros((self.n_peers, self.n_prefixes))
			for peer_i in range(self.n_peers):
				for pref_i in range(self.n_prefixes):
					# toggle this entry
					iter_adv = copy.deepcopy(advertisement)
					iter_adv[peer_i,pref_i] = 1 - iter_adv[peer_i,pref_i]
					deltas[peer_i,pref_i] = pre_obj - self.actual_objective(iter_adv)
			# largest additional latency benefit
			best_delta = np.argmax(deltas.flatten())
			best_peer_i = best_delta // self.n_prefixes
			best_pref_i = best_delta % self.n_prefixes
			advertisement[best_peer_i,best_pref_i] = 1 - advertisement[best_peer_i,best_pref_i]

			stop = self.stopping_condition([i,np.abs(deltas[best_peer_i, best_pref_i])])

			i += 1

		self.sparse_greedy = {
			'objective': self.objective(advertisement),
			'advertisement': advertisement,
		}

	def solve_greedy(self, init_adv, verbose=True):
		# at each iteration, toggle the entry that yields the largest delta latency benefit

		advertisement = self.threshold_a(init_adv)
		stop = False
		i = 0
		while not stop:
			pre_lat_ben = self.latency_benefit_fn(advertisement)
			deltas = np.zeros((self.n_peers, self.n_prefixes))
			for peer_i in range(self.n_peers):
				for pref_i in range(self.n_prefixes):
					# toggle this entry
					iter_adv = copy.deepcopy(advertisement)
					iter_adv[peer_i,pref_i] = 1 - iter_adv[peer_i,pref_i]
					deltas[peer_i,pref_i] = self.latency_benefit_fn(iter_adv) - pre_lat_ben
			# largest additional latency benefit
			best_delta = np.argmax(deltas.flatten())
			best_peer_i = best_delta // self.n_prefixes
			best_pref_i = best_delta % self.n_prefixes
			advertisement[best_peer_i,best_pref_i] = 1 - advertisement[best_peer_i,best_pref_i]

			stop = self.stopping_condition([i,np.abs(deltas[best_peer_i, best_pref_i])])

			i += 1

		self.greedy = {
			'objective': self.objective(advertisement),
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
			objs[i] = self.objective(a.reshape((self.n_peers, self.n_prefixes)))
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

	def compare_different_solutions(self,n_run=40,verbose=True,init_adv=None):
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
			# Our solution
			self.sas.solve(init_adv=adv)
			our_objective = self.sas.get_last_objective(effective=True) # look at objective for thresholded adv
			our_advs.append(self.sas.get_last_advertisement())

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
				x,cdf_x = get_cdf_xy(vals[k] + .01*np.random.normal(size=(len(vals[k]),)))
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
		self.sigmoid_k = 3.0
		self.metrics = {}
		self.beta = .3

		self.gradient_support = [(a_i,a_j) for a_i in range(self.n_peers) for a_j in range(self.n_prefixes)]
		max_support = 8
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
		return self.metrics['advertisements'][-1]

	def get_last_objective(self, effective=False):
		if effective:
			return self.objective(self.threshold_a(self.get_last_advertisement()))
		else:
			return self.objective(self.get_last_advertisement())

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
		self.metrics['latency_benefit_grads'].append(L_grad)

		return -1 * L_grad

	def gradients_continuous(self, a):
		inds = self.gradient_support
		if self.iter % self.gradient_support_settings['calc_every'] == 0:
			# periodically calculate all gradients
			inds = [(a_i,a_j) for a_i in range(a.shape[0]) for a_j in range(a.shape[1])]

		L_grad = np.zeros(a.shape)
		if self.verbose:
			print("Gradient computation, {} inds".format(len(inds)))
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

		self.metrics['latency_benefit_grads'].append(L_grad)
		return -1 * L_grad

	def impose_advertisement_constraint(self, a):
		"""The convex constraint 0 <= a_ij <= 1 has the simple solution to clip."""
		a = np.clip(a,0,1.0)
		return a

	def make_plots(self, *args, **kwargs):
		f,ax = plt.subplots(4,1)
		try:
			all_as = np.array(args[0])
		except IndexError:
			all_as = np.array(self.metrics['advertisements'])
		all_grads = np.array(self.metrics['grads'])
		all_lb_grads = np.array(self.metrics['latency_benefit_grads'])
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
				ax[2].plot(all_lb_grads[:,peer_i,pref_i], c=colors[peer_i%len(colors)], label="Peer {} Prefix {}".format(self.peers[peer_i], pref_i))

		all_objectives = self.metrics['actual_objectives']
		ax[0].legend(fontsize=6)
		ax[0].set_ylabel("a")
		# ax[1].legend()
		ax[1].set_ylabel("Net Grad")
		ax[2].set_ylabel("LB Grad")
		ax[-1].plot(all_objectives)
		ax[-1].set_ylabel("Objective")

		plt.show() 

	def print_adv(self, a):
		for peer_i in range(self.n_peers):
			for pref_i in range(self.n_prefixes):
				print("Peer {} Prefix {}: {}".format(self.peers[peer_i], pref_i, a[peer_i,pref_i]))

	def set_alpha(self):
		self.alpha = .5

	def update_gradient_support(self, gradient):
		gradient = np.abs(gradient)
		inds = [(a_i,a_j) for a_i in range(self.n_peers) for a_j in range(self.n_prefixes)]
		sorted_inds = list(reversed(np.argsort([gradient[a_i,a_j] for a_i,a_j in inds])))
		# Focus on the largest gradients
		self.gradient_support = list([inds[i] for i in sorted_inds[0:self.gradient_support_settings['support_size']]])

	def solve(self, init_adv=None):
		self.set_alpha()
		stop = False
		if init_adv is None:
			advertisement = self.init_advertisement()
		else:
			advertisement = init_adv
		a_km1 = advertisement
		if self.verbose:
			# self.print_adv(advertisement)
			print("Optimizing over {} peers".format(self.n_peers))
		current_objective = self.objective(advertisement)

		self.iter = 0; t_start = time.time()
		# For analysis
		self.metrics['actual_objectives'] = []
		self.metrics['advertisements'] = []
		self.metrics['effective_objectives'] = []
		self.metrics['grads'] = []
		self.metrics['latency_benefit_grads'] = []
		self.metrics['prox_l1_grads'] = []

		rolling_delta = 10
		delta_alpha = .7

		while not stop:
			# calculate gradients
			grads = self.gradient_fn(advertisement)
			# update advertisement by taking a gradient step with momentum and then applying the proximal gradient for L1
			a_k = advertisement
			w_k = a_k - self.alpha * grads + self.beta * (a_k - a_km1)
			advertisement = self.apply_prox_l1(w_k)
			self.metrics['advertisements'].append(copy.copy(advertisement))
			self.metrics['grads'].append(advertisement - a_k)
			a_km1 = a_k

			# project onto constraints
			# one constraint we may want is Ra <= 1 
			# not sure how to do this

			# another constraint we may want is 0 <= a_ij <= 1
			# the solution is just clipping to be in the set
			# clipping can mess with gradient descent
			advertisement = self.impose_advertisement_constraint(advertisement)

			# re-calculate objective
			last_objective = current_objective
			current_objective = self.objective(advertisement,v=False)
			self.metrics['actual_objectives'].append(current_objective)
			self.metrics['effective_objectives'].append(self.objective(self.threshold_a(advertisement)))

			# Stop when the objective doesn't change, but use an EWMA to track the change so that we don't spuriously exit
			rolling_delta = (1 - delta_alpha) * rolling_delta + delta_alpha * np.abs(current_objective - last_objective)
			stop = self.stopping_condition([self.iter,rolling_delta])
			self.iter += 1
		t_per_iter = (time.time() - t_start) / self.iter
		if self.verbose:
			print("Stopped on {}, t per iter: {}".format(self.iter, t_per_iter))
		self.metrics['t_per_iter'] = t_per_iter

def main():
	np.random.seed(31415)
	# # ## Generating graphs
	# gen_random_graph('test_graph',n_transit=2,n_user=20)

	# ## Looking at behavior for one mu (smallish graph)
	# mu = 30
	# # np.random.seed(31415)
	# sas = Sparse_Advertisement_Solver(graph_fn="toy_graph_bigger.csv", graph_md_fn="toy_graph_bigger_md.json", mu=mu,
	# 	verbose=False, cont_grads=True)
	# sas.solve()
	# print("Final advertisement strat is {}".format(sas.threshold_a(sas.metrics['advertisements'][-1])))
	# sas.make_plots()

	# ## Looking at behavior for one mu (biggish graph)
	# mu = .01
	# plt_arr = []
	# for n_user in range(2,50,3):
	# 	gen_random_graph('test_graph',n_transit=2,n_user=n_user)
	# 	sas = Sparse_Advertisement_Solver(graph_fn="test_graph.csv", graph_md_fn="test_graph_md.json", mu=mu,
	# 		cont_grads=True, verbose=True)
	# 	sas.solve()
	# 	if sas.iter > 200:
	# 		sas.make_plots()
	# 	plt_arr.append(sas.metrics['t_per_iter'])
	# plt.plot(plt_arr)
	# plt.show()

	# # Comparing different solutions (biggish graph)
	# np.random.seed(31415)
	# gen_random_graph('test_graph',n_transit=2,n_user=100)
	# mu = .01
	# sae = Sparse_Advertisement_Eval(graph_fn="test_graph.csv", graph_md_fn="test_graph_md.json", mu=mu,
	# 		verbose=True, cont_grads=True)
	# sae.compare_different_solutions()

	# # Comparing different solutions (smallish graph)
	# mu = 30
	# sae = Sparse_Advertisement_Eval(graph_fn="toy_graph_bigger.csv", graph_md_fn="toy_graph_bigger_md.json", mu=mu,
	# 		verbose=False,cont_grads=True)
	# sae.compare_different_solutions()

	# ## Looking at behavior for one mu (two prefix test graph)
	# mu = .1
	# sas = Sparse_Advertisement_Solver(graph_fn="multi_prefix_test.csv", graph_md_fn="multi_prefix_test_md.json", mu=mu,
	# 	verbose=False,cont_grads=True)
	# print(sas.peers)
	# sas.solve()
	# sas.make_plots()

	# # Comparing different solutions (two prefix test graph)
	# mu = .1
	# sae = Sparse_Advertisement_Eval(graph_fn="multi_prefix_test.csv", graph_md_fn="multi_prefix_test_md.json", mu=mu,
	# 		verbose=False,cont_grads=True)
	# sae.compare_different_solutions()


	# # Sweeping mu
	# sae = Sparse_Advertisement_Eval(graph_fn="multi_prefix_test.csv", graph_md_fn="multi_prefix_test_md.json",
	# 		verbose=False,cont_grads=True)
	# sae.compare_peer_value()

	## Real graph
	mu = .1
	sas = Sparse_Advertisement_Solver(graph_fn="problink_rels.csv", graph_md_fn="problink_rels_md.json", mu=mu,
			verbose=True,cont_grads=True)
	sas.solve()
	sas.make_plots()


if __name__ == "__main__":
	main()