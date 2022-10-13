import numpy as np, json, os
from helpers import *
GRAPH_DIR = "graphs"

def _gen_random_graph(*args,**kwargs):
	global bp_cache
	bp_cache = {}
	gen_random_graph(*args,**kwargs)

def create_random_md_from_graph(graph_fn, cp_network, user_networks):
	graph = {}
	with open(os.path.join("graphs", graph_fn + '.csv'),'r') as f:
		for row in f:
			if row.startswith("#"): continue
			as1,as2,_ = row.strip().split('|')
			graph[as1,as2] = None

	metadata_fn = os.path.join("graphs", graph_fn + "_md.json")
	out_data = {}
	out_data['content_provider'] = cp_network
	out_data['user_networks'] = user_networks
	out_data['edge_lengths'] = {}
	for as1,as2 in graph:
		out_data['edge_lengths']["{}-{}".format(as1,as2)] = 800 + 300 * np.random.normal()

	json.dump(out_data, open(metadata_fn, 'w'))

def gen_random_graph(output_fn, n_transit=1, n_user=10):
	node_add_probabilities = {
		'user_has_rp': .8,
		'rp_has_brp': .2,
		'net_has_transit': .3,
	}
	connection_probabilities = {
		'network_to_transit': {
			'user': .1,
			'rp': .3,
			'brp': 1,
		},
		'network_to_cp': { # Peering arrangements
			'user': .2,
			'rp': .5,
			'brp': 1,
		},
		'peering': .1,
	}

	graph = {}
	provider_mult, provider_cnt = int(1e3), 1
	user_mult = int(1e6)
	user_networks = user_mult * np.array(list(range(1,n_user+1)))
	rp_networks = []
	brp_networks = []
	cp_network = 1
	transit_networks = np.array(list(range(2,n_transit+2)))

	# Each transit provider connects to the CP
	for transit_provider in transit_networks:
		graph[transit_provider, cp_network] = -1
	# Each transit provider connects to all other transit providers
	ntwrk_to_type = {cp_network:'cp'}
	for tp in transit_networks:
		ntwrk_to_type[tp] = 'transit'
		for _tp in transit_networks:
			graph[tp,_tp] = 0

	def get_pc(ntwrk, ag):
		pc = [ntwrk]
		try:
			for connected_ntwrk in ag[ntwrk]:
				if graph[connected_ntwrk, ntwrk] == -1:
					pc = pc + get_pc(connected_ntwrk, ag)
		except KeyError:
			pass
		return pc


	# first, randomly spawn the graph nodes
	for i in range(n_user):
		user_network = user_networks[i]
		if np.random.random() < node_add_probabilities['user_has_rp']:
			rp_id = provider_mult * provider_cnt
			provider_cnt += 1
			rp_networks.append(rp_id)
			graph[rp_id,user_network] = -1
			if np.random.random() < node_add_probabilities['rp_has_brp']:
				brp_id = provider_mult * provider_cnt
				provider_cnt += 1
				brp_networks.append(brp_id)
				graph[brp_id,rp_id] = -1
	# Randomly form connections to transit providers and peerings with CP
	# could possibly add random paid peering arrangements, or pseudo-random connects between certain networks
	for ntwrk_type, ntwrk_set in zip(['user','rp','brp'], [user_networks, rp_networks, brp_networks]):
		for ntwrk in ntwrk_set:
			ntwrk_to_type[ntwrk] = ntwrk_type
			if np.random.random() < connection_probabilities['network_to_transit'][ntwrk_type]:
				transit_choice = np.random.choice(transit_networks)
				graph[transit_choice, ntwrk] = -1
			if np.random.random() < connection_probabilities['network_to_cp'][ntwrk_type]:
				graph[ntwrk,cp_network] = 0
	# Check to make sure each network has a transit provider somewhere in their provider cone
	# to guarantee reachability to the wider Internet
	adjacency_graph = {}
	for ntwrk1,ntwrk2 in graph:
		try:
			adjacency_graph[ntwrk1].append(ntwrk2)
		except KeyError:
			adjacency_graph[ntwrk1] = [ntwrk2]
		try:
			adjacency_graph[ntwrk2].append(ntwrk1)
		except KeyError:
			adjacency_graph[ntwrk2] = [ntwrk1]
	for (ntwrk_type, ntwrk_set) in zip(['user','rp','brp'], [user_networks, rp_networks, brp_networks]):
		for ntwrk in ntwrk_set:
			pc = get_pc(ntwrk,adjacency_graph)
			if get_intersection(pc,transit_networks) == []:
				transit_choice = np.random.choice(transit_networks)
				graph[transit_choice, ntwrk] = -1
				adjacency_graph[transit_choice].append(ntwrk)
				try:
					adjacency_graph[ntwrk].append(transit_choice)
				except KeyError:
					adjacency_graph[ntwrk] = [transit_choice]

	## Randomly add peering links to the graph
	# valid peering links are between networks at the same "tier"
	all_ntwrks = set([ntwrk for ntwrk,_ in graph] + [ntwrk for _,ntwrk in graph])
	for ntwrk_i in all_ntwrks:
		for ntwrk_j in all_ntwrks:
			if ntwrk_i == ntwrk_j: continue # lol
			if ntwrk_i in adjacency_graph[ntwrk_j]: continue # link already exits
			if ntwrk_to_type[ntwrk_i] != ntwrk_to_type[ntwrk_j]: continue # need to be same tier
			if np.random.random() < connection_probabilities['peering']:
				graph[ntwrk_i, ntwrk_j] = 0
	for ntwrk1,ntwrk2 in graph:
		try:
			adjacency_graph[ntwrk1].append(ntwrk2)
		except KeyError:
			adjacency_graph[ntwrk1] = [ntwrk2]
		try:
			adjacency_graph[ntwrk2].append(ntwrk1)
		except KeyError:
			adjacency_graph[ntwrk2] = [ntwrk1]
	for ntwrk in adjacency_graph: adjacency_graph[ntwrk] = list(set(adjacency_graph[ntwrk]))

	## Randomly assign capacities to each peering link with the CP, and volumes for the users
	pl_caps = {str(p):.5+np.random.uniform() for p in adjacency_graph[cp_network]}
	user_vols = {str(un): .4 + .3*np.random.uniform() for un in user_networks}


	# Now assign edge lengths to the graph
	edge_lengths = {}
	el_models = { # gaussian models
		('user','rp'): lambda : 300 + 2000 * np.random.uniform(), # could be close, but also could be far
		('user','user'): lambda : 200 + 1000 * np.random.uniform(), # if users peer, probably close
		('user','cp'): lambda : 300 + 300 * np.random.uniform(), # probably close
		('user', 'transit'): lambda : 500 + 3000 * np.random.uniform(), # wide variance
		('rp', 'cp'): lambda : 500 + 1000 * np.random.uniform(),  # probably lowish
		('rp', 'transit'): lambda : 300 + 300 * np.random.uniform(), # probably quite low
		('rp','rp'): lambda : 500 + 4000 * np.random.uniform(), # probably wide variance
		('transit','cp'): lambda : 800 + 2000 + np.random.uniform(), # probably pretty bad
		('transit','transit'): lambda : 1000 + 5000 * np.random.uniform(), # probably pretty far
	}
	for ntwrk1,ntwrk2 in graph:
		t1,t2 = ntwrk_to_type[ntwrk1], ntwrk_to_type[ntwrk2]
		try:
			_mod = el_models[t1,t2]
		except KeyError:
			_mod = el_models[t2,t1]
		edge_lengths["{}-{}".format(ntwrk1,ntwrk2)] = np.maximum(_mod(), 10)

	graph_fn = os.path.join(GRAPH_DIR, "{}.csv".format(output_fn))
	graph_md_fn = os.path.join(GRAPH_DIR, "{}_md.json".format(output_fn))
	with open(graph_fn,'w') as f:
		# Save it all to file
		for (ntwrk1,ntwrk2), rel in graph.items():
			f.write("{}|{}|{}\n".format(ntwrk1,ntwrk2,rel))
	md = {
		"content_provider": str(cp_network),
		"user_networks": [str(el) for el in user_networks],
		"edge_lengths": edge_lengths,
		"peering_link_capacities": pl_caps,
		"user_volumes": user_vols,
	}
	json.dump(md, open(graph_md_fn,'w'))


if __name__ == "__main__":
	create_random_md_from_graph('problink_rels', 8075, [7922,22773])
