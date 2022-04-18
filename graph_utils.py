import numpy as np, json, os
from helpers import *
GRAPH_DIR = "graphs"

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
			'user': .6,
			'rp': .5,
			'brp': 1,
		}
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
	for tp in transit_networks:
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

	# Now assign edge lengths to the graph
	edge_lengths = {}
	el_models = { # gaussian models
		"user_to_rp": (3000,900),
		"rp_to_brp": (100,20),
		"user_to_cp": (500,300),
		"ntwrk_to_cp": (10,1),
		"ntwrk_to_transit": (5000,2000),
	}
	for ntwrk1,ntwrk2 in graph:
		if ntwrk1 in transit_networks or ntwrk2 in transit_networks:
			if ntwrk2 != cp_network:
				m,std = el_models['ntwrk_to_transit']
			else:
				m,std = el_models['ntwrk_to_cp']
		elif ntwrk1 in user_networks or ntwrk2 in user_networks:
			if ntwrk1 in user_networks:
				un, other = ntwrk1, ntwrk2
			else:
				un, other = ntwrk2, ntwrk1
			if other == cp_network:
				m,std = el_models['user_to_cp']
			elif other in rp_networks:
				m,st = el_models['user_to_rp']
			else: # assumes last type is brp
				m,std = el_models['user_to_brp']
		elif ntwrk2 == cp_network:
			m,std = el_models['ntwrk_to_cp']
		else: # rp to to brp
			m,std = el_models['rp_to_brp']

		edge_lengths["{}-{}".format(ntwrk1,ntwrk2)] = np.maximum(m+std*np.random.normal(), 10)

	graph_fn = os.path.join(GRAPH_DIR, "{}.csv".format(output_fn))
	graph_md_fn = os.path.join(GRAPH_DIR, "{}_md.json".format(output_fn))
	with open(graph_fn,'w') as f:
		# Save it all to file
		for (ntwrk1,ntwrk2), rel in graph.items():
			f.write("{}|{}|{}\n".format(ntwrk1,ntwrk2,rel))
	md = {
		"content_provider": str(cp_network),
		"user_networks": [str(el) for el in user_networks],
		"edge_lengths": edge_lengths
	}
	json.dump(md, open(graph_md_fn,'w'))


if __name__ == "__main__":
	create_random_md_from_graph('problink_rels', 8075, [7922,22773])
