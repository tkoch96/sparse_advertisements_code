import tqdm, numpy as np
from constants import *
from helpers import *

def get_random_ingress_priorities(ug_perfs, ug_anycast_perfs, pop_to_loc, metro_loc):
	## Simulate random ingress priorities for each UG
	ingress_priorities = {}
	dist_cache = {}
	for ug in tqdm.tqdm(ug_perfs,desc="Assigning ingress priorities randomly."):
		ingress_priorities[ug] = {}
		## approximate the anycast interface by getting the one with closest latency
		popps = list(ug_perfs[ug])
		perfs = np.array([ug_perfs[ug][popp] for popp in popps])
		## anycast is the most favored ingress
		probably_anycast = popps[np.argmin(np.abs(perfs - ug_anycast_perfs[ug]))]
		priorities = {probably_anycast:0}

		other_peerings = list(get_difference(list(ug_perfs[ug]), probably_anycast))
		if len(other_peerings) > 0:
			associated_dists = []
			for op in other_peerings:
				try:
					d = dist_cache[pop_to_loc[op[0]], metro_loc[ug[0]]]
				except KeyError:
					d = geopy.distance.geodesic(
						pop_to_loc[op[0]], metro_loc[ug[0]]).km
					dist_cache[pop_to_loc[op[0]], metro_loc[ug[0]]] = d
				associated_dists.append(d)
			ranked_peerings_by_dist = sorted(zip(associated_dists,other_peerings), key = lambda el : el[0])
			for i,(_,pi) in enumerate(ranked_peerings_by_dist):
				priorities[pi] = i + 1
			## randomly flip some priorities
			if len(priorities) > 1:
				for pi in list(priorities):
					if np.random.random() < .01:
						other_pi = list(get_difference(list(priorities), [pi]))[np.random.choice(len(priorities)-1)]
						tmp = copy.copy(priorities[pi])
						priorities[pi] = copy.copy(priorities[other_pi])
						priorities[other_pi] = tmp

		for popp,priority in priorities.items():
			ingress_priorities[ug][popp] = priority
	return ingress_priorities

def cluster_actual_users(**kwargs):
	considering_pops = kwargs.get('considering_pops')
	cpstr = "-".join(sorted(considering_pops))
	cluster_cache_fn = os.path.join(CACHE_DIR, 'clustered_perfs_{}.pkl'.format(cpstr))
	if not os.path.exists(cluster_cache_fn):
		anycast_latencies, ug_perfs = load_actual_perfs(**kwargs)

		### Form a matrix of all latencies
		ugs = sorted(list(ug_perfs))
		popps = sorted(list(set(popp for ug in ugs for popp in ug_perfs[ug])))
		print("{} UGs, {} popps".format(len(ugs), len(popps)))

		ug_to_ind = {ug:i for i,ug in enumerate(ugs)}
		popp_to_ind = {popp:i for i, popp in enumerate(popps)}


		IGNORE_LAT = 10*NO_ROUTE_LATENCY
		latencies_mat = IGNORE_LAT * np.ones((len(ugs), len(popps)), dtype=np.float32)
		for ug, perfs in ug_perfs.items():
			for popp,lat in perfs.items():
				latencies_mat[ug_to_ind[ug],popp_to_ind[popp]] = lat

		from sklearn.cluster import Birch
		### threshold would probably be tuned by whatever gets me an appropriate number of clusters
		brc = Birch(threshold=20,n_clusters=None)
		labels = brc.fit_predict(latencies_mat)

		examples_by_label = {}
		for i in range(len((labels))):
			lab = labels[i]
			try:
				examples_by_label[lab].append(ugs[i])
			except KeyError:
				examples_by_label[lab] = [ugs[i]]
		all_labs = sorted(list(examples_by_label))
		np.random.shuffle(all_labs)
		# for lab in all_labs:
		# 	if len(examples_by_label[lab]) == 1: continue
		# 	print("Lab {}, ugs: ".format(lab))
		# 	for ug in examples_by_label[lab]:
		# 		print("{} -- {}".format(ug,ug_perfs[ug]))
		# 	print("\n\n")
		# 	if np.random.random() > .95:break

		clustered_ug_perfs, clustered_anycast_perfs = {},{}
		ug_id = 0
		print("{} subcluster labels but {} different ug labels".format(len(brc.subcluster_labels_),
			len(examples_by_label)))
		for sc_center, lab in zip(brc.subcluster_centers_, brc.subcluster_labels_):
			metro = str("metro-{}".format(ug_id))
			this_lab_ug = (metro, ug_id)
			try:
				these_ugs = examples_by_label[lab]
			except KeyError:
				# print("no UGs found for subcluster label {}".format(lab))
				continue
			avg_anycast_lat = np.mean([anycast_latencies[ug] for ug in these_ugs])
			clustered_anycast_perfs[this_lab_ug] = avg_anycast_lat
			clustered_ug_perfs[this_lab_ug] = {}
			for i,perf in enumerate(sc_center):
				if perf == IGNORE_LAT: continue
				clustered_ug_perfs[this_lab_ug][popps[i]] = perf
			ug_id += 1

		print("Reduced {} Ugs to {}".format(len(ug_perfs), len(clustered_ug_perfs)))

		## todo incorporate this idea into the pipeline, need to add in all the ug -> centroid conversion
		## need to make bs metro,asn -> average?
		## need to make bs anycast latency -> average

		pickle.dump([clustered_ug_perfs,clustered_anycast_perfs], open(cluster_cache_fn,'wb'))
	else:
		clustered_ug_perfs, clustered_anycast_perfs = pickle.load(open(cluster_cache_fn,'rb'))

	return clustered_ug_perfs, clustered_anycast_perfs

def parse_lat(lat_str):
	lat = float(lat_str) * 1000
	lat = np.maximum(MIN_LATENCY, np.minimum(MAX_LATENCY, lat))
	return lat

def load_actual_perfs(considering_pops=list(POP_TO_LOC['vultr']), **kwargs):
	print("Loading performances, only considering pops: {}".format(considering_pops))
	lat_fn = os.path.join(CACHE_DIR, 'vultr_ingress_latencies_by_dst.csv')
	pop_to_loc = {pop:POP_TO_LOC['vultr'][pop] for pop in considering_pops}
	violate_sol = {}
	for row in open(os.path.join(CACHE_DIR, 'addresses_violating_sol.csv'),'r'):
		metro,asn,violates = row.strip().split(',')
		violate_sol[metro,asn] = int(violates)
	pop_dists = {}
	for i,popi in enumerate(considering_pops):
		for j,popj in enumerate(considering_pops):
			if j > i: continue
			if j == i:
				pop_dists[popi,popj] = 0
				pop_dists[popj,popi] = 0
			pop_dists[popi,popj] = geopy.distance.geodesic(pop_to_loc[popi],
				pop_to_loc[popj]).km
			pop_dists[popj,popi] = pop_dists[popi,popj]
	ug_perfs = {}

	for row in tqdm.tqdm(open(lat_fn, 'r'), desc="Parsing per-ingress VULTR measurements."):
		# if np.random.random() > .9999999:break
		fields = row.strip().split(',')
		if fields[2] not in considering_pops: continue
		t,ip,pop,peer,lat = fields
		lat = parse_lat(lat)
		metro = 'tmp'
		asn = ip#ip32_to_24(ip)
		ug = (metro,asn)
		try:
			if violate_sol[ug]:
				continue
		except KeyError:
			pass
		try:
			ug_perfs[ug]
		except KeyError:
			ug_perfs[ug] = {}
		try:
			ug_perfs[ug][pop,peer].append(lat)
		except KeyError:
			ug_perfs[ug][pop,peer] = [lat]
	for ug in ug_perfs:
		for popp, lats in ug_perfs[ug].items():
			ug_perfs[ug][popp] = np.min(lats)
	ugs = sorted(list(ug_perfs))
	popps = sorted(list(set(popp for ug in ugs for popp in ug_perfs[ug])))
	print("{} UGs, {} popps".format(len(ugs), len(popps)))

	anycast_latencies = {}
	anycast_pop = {}
	for row in tqdm.tqdm(open(os.path.join(CACHE_DIR, 'vultr_anycast_latency.csv')
		,'r'),desc="Parsing VULTR anycast latencies"):
		# if np.random.random() > .9999999:break
		_,ip,lat,pop = row.strip().split(',')
		if lat == '-1': continue
		metro = 'tmp'
		asn = ip#ip32_to_24(ip)
		ug = (metro,asn)
		try:
			if violate_sol[ug]:
				continue
		except KeyError:
			pass
		lat = parse_lat(lat)
		try:
			anycast_latencies[ug].append(lat)
		except KeyError:
			anycast_latencies[ug] = [lat]
	for ug in anycast_latencies:
		anycast_latencies[ug] = np.min(anycast_latencies[ug])

	in_both = get_intersection(ug_perfs, anycast_latencies)
	anycast_latencies = {ug:anycast_latencies[ug] for ug in in_both}
	ug_perfs = {ug:ug_perfs[ug] for ug in in_both}

	ugs = sorted(list(ug_perfs))
	popps = sorted(list(set(popp for ug in ugs for popp in ug_perfs[ug])))
	print("{} UGs, {} popps".format(len(ugs), len(popps)))


	### delete any UGs for which latencies don't follow SOL rules
	to_del = []
	changed =False
	for ug in tqdm.tqdm(ug_perfs,desc="Discarding UGs that violate SOL rules"):
		try:
			if violate_sol[ug]:
				to_del.append(ug)
			continue
		except KeyError:
			pass
		valid = True
		perfs_by_pop = {}
		for (pop,peer), lat in ug_perfs[ug].items():
			try:
				perfs_by_pop[pop] = np.minimum(perfs_by_pop[pop],lat)
			except KeyError:
				perfs_by_pop[pop] = lat
		ug_pops = list(perfs_by_pop)
		for popi in ug_pops:
			for popj in ug_pops:
				# add in 5ms of leeway
				if perfs_by_pop[popi] + perfs_by_pop[popj] + 5 <= pop_dists[popi,popj] * .01:
					# print("({}) {}: {} ms and {}: {} ms but pop dist is {} km".format(
					# 	ug,popi,perfs_by_pop[popi],popj,perfs_by_pop[popj],pop_dists[popi,popj]))
					valid = False
					break
			if not valid:
				break
		if not valid:
			to_del.append(ug)
			violate_sol[ug] = 1
			changed=True
		else:
			violate_sol[ug] = 0
			changed=True

	for ug in to_del:
		del ug_perfs[ug]

	ugs = sorted(list(ug_perfs))
	popps = sorted(list(set(popp for ug in ugs for popp in ug_perfs[ug])))
	print("{} UGs, {} popps".format(len(ugs), len(popps)))
	if changed:
		with open(os.path.join(CACHE_DIR, 'addresses_violating_sol.csv'), 'w') as f:
			for ug,violates in violate_sol.items():
				metro,asn = ug
				f.write("{},{},{}\n".format(metro,asn,violates))
	print("{} UGs violate SOL rules".format(sum(violates for violates in violate_sol.values())))

	in_both = get_intersection(ug_perfs, anycast_latencies)
	anycast_latencies = {ug:anycast_latencies[ug] for ug in in_both}
	ug_perfs = {ug:ug_perfs[ug] for ug in in_both}

	ugs = sorted(list(ug_perfs))
	popps = sorted(list(set(popp for ug in ugs for popp in ug_perfs[ug])))
	print("{} UGs, {} popps after intersecting latency".format(len(ugs), len(popps)))



	### Randomly limit to max_n_ug per popp, unless the popp is a provider
	max_n_ug = kwargs.get('n_users_per_peer', 200)
	provider_fn = os.path.join(CACHE_DIR, 'vultr_provider_popps.csv')
	provider_popps = []
	for row in open(provider_fn, 'r'):
		pop,peer = row.strip().split(',')
		if pop not in considering_pops:
			continue
		provider_popps.append((pop,peer))
	popp_to_ug = {popp:[] for popp in popps}
	for ug, perfs in ug_perfs.items():
		for popp in perfs:
			if popp in provider_popps: continue
			popp_to_ug[popp].append(ug)
	for popp,_ugs in popp_to_ug.items():
		if popp in provider_popps: continue
		if len(_ugs) <= max_n_ug: continue
		np.random.shuffle(_ugs)
		popp_to_ug[popp] = _ugs[0:max_n_ug]
	keep_ugs = list(set(ug for popp in popp_to_ug for ug in popp_to_ug[popp] if popp not in provider_popps))
	ug_perfs = {ug:ug_perfs[ug] for ug in keep_ugs}

	ugs = sorted(list(ug_perfs))
	popps = sorted(list(set(popp for ug in ugs for popp in ug_perfs[ug])))
	print("{} UGs, {} popps after limiting users".format(len(ugs), len(popps)))

	return anycast_latencies, ug_perfs

def load_actual_deployment():
	# considering_pops = list(POP_TO_LOC['vultr'])
	considering_pops = ['miami','tokyo', 'amsterdam']
	cpstr = "-".join(sorted(considering_pops))
	deployment_cache_fn = os.path.join(CACHE_DIR, 'actual_deployment_cache_{}.pkl'.format(cpstr))
	if not os.path.exists(deployment_cache_fn):
		pop_to_loc = {pop:POP_TO_LOC['vultr'][pop] for pop in considering_pops}

		# anycast_latencies, ug_perfs = load_actual_perfs(considering_pops=considering_pops)
		ug_perfs, anycast_latencies = cluster_actual_users(considering_pops=considering_pops, n_users_per_peer=50)

		for ug in list(ug_perfs):
			to_del = [popp for popp in ug_perfs[ug] if popp[0] not in considering_pops]
			for popp in to_del:
				del ug_perfs[ug][popp]
			if len(ug_perfs[ug]) == 0:
				del ug_perfs[ug]
		anycast_latencies = {ug:anycast_latencies[ug] for ug in ug_perfs}

		provider_fn = os.path.join(CACHE_DIR, 'vultr_provider_popps.csv')
		provider_popps = []
		for row in open(provider_fn, 'r'):
			pop,peer = row.strip().split(',')
			if pop not in considering_pops:
				continue
			provider_popps.append((pop,peer))

		ugs = sorted(list(ug_perfs))
		popps = sorted(list(set(popp for ug in ugs for popp in ug_perfs[ug])))
		print("{} popps, {} ugs".format(len(popps), len(ugs)))
		provider_popps = get_intersection(provider_popps, popps)
		n_providers = len(set(peer for pop,peer in provider_popps))
		ug_to_vol = {ug:1 for ug in ugs}
		link_capacities = {popp:len(ugs) for popp in popps}

		metros = list(set(metro for metro,asn in ugs))
		metro_loc = {}
		for ug in ug_perfs:
			metro,asn = ug
			ug_popps = list(ug_perfs[ug])
			closest_popp = ug_popps[np.argmin([ug_perfs[ug][popp] for popp in ug_popps])]
			metro_loc[metro] = pop_to_loc[closest_popp[0]]

		ingress_priorities = get_random_ingress_priorities(ug_perfs, anycast_latencies,
			pop_to_loc, metro_loc)

		deployment = {
			'ugs': ugs,
			'ug_perfs': ug_perfs,
			'ug_anycast_perfs': anycast_latencies,
			'ug_to_vol': ug_to_vol,
			'whole_deployment_ugs': ugs,
			'whole_deployment_ug_to_vol': ug_to_vol,
			'link_capacities': link_capacities,
			'ingress_priorities': ingress_priorities,
			'popps': popps,
			'metro_loc': metro_loc,
			'pop_to_loc': pop_to_loc,
			'n_providers': n_providers,
			'provider_popps': provider_popps,
		}
		pickle.dump(deployment, open(deployment_cache_fn,'wb'))

	else:
		deployment = pickle.load(open(deployment_cache_fn,'rb'))
		ug_perfs = deployment['ug_perfs']
		anycast_latencies = deployment['ug_anycast_perfs']
		pop_to_loc = deployment['pop_to_loc']
		metro_loc = deployment['metro_loc']
		# ingress_priorities = get_random_ingress_priorities(ug_perfs, anycast_latencies,
		# 	pop_to_loc, metro_loc)
		# deployment['ingress_priorities'] = ingress_priorities

	return deployment

problem_params = {
	'really_friggin_small': {
		'n_metro': 5,
		'n_asn': 3,
		'n_peer': 20,
		'n_pop': 2, 
		'max_popp_per_ug': 4, 
		'max_peerings_per_pop': 10,
		'min_peerings_per_pop': 4,
		'n_providers': 2,
	},
	'small': {
		'n_metro': 15,
		'n_asn': 15,
		'n_peer': 100,
		'n_pop': 3, 
		'max_popp_per_ug': 10, 
		'max_peerings_per_pop': 30,
		'min_peerings_per_pop': 5,
		'n_providers': 15,
	},
	'decent': {
		'n_metro': 200,
		'n_asn': 20,
		'n_peer': 100,
		'n_pop': 10, 
		'max_popp_per_ug': 20, 
		'max_peerings_per_pop': 40,
		'min_peerings_per_pop': 20,
		'n_providers': 20,
	},
	'med': { # goal of sorts, maybe more metro,asns 
		'n_metro': 20,
		'n_asn': 100,
		'n_peer': 1500,
		'n_pop': 30, 
		'max_popp_per_ug': 30, 
		'max_peerings_per_pop': 70,
		'min_peerings_per_pop': 20,
		'n_providers': 25,
	},
	'large': {
		'n_metro': 40,
		'n_asn': 100,
		'n_peer': 4100,
		'n_pop': 100,
		'max_popp_per_ug': 30,
		'max_peerings_per_pop': 300,
		'min_peerings_per_pop': 30,
		'n_providers': 30,
	},
}

def get_random_deployment(problem_size):
	if problem_size == 'actual':
		return load_actual_deployment()
	else:
		return get_random_deployment_by_size(problem_size)

def get_random_deployment_by_size(problem_size):
	#### Extensions / todos: 
	### make users probabilistically have valid popps by distance
	### we may want popps to be transit providers depending on the pop, randomly
	### ug perfs should not be nonsensical based on distance

	print("----Creating Random Deployment-----")
	sizes = problem_params[problem_size]

	### Probably update this to be a slightly more interesting model later
	random_latency = lambda : np.random.uniform(MIN_LATENCY, MAX_LATENCY)
	random_transit_provider_latency = lambda : np.random.uniform(MIN_LATENCY*1.3, MAX_LATENCY)

	# testing ideas for learning over time
	pops = np.arange(0,sizes['n_pop'])
	def random_loc():
		return (np.random.uniform(-30,30), np.random.uniform(-20,20))
	pop_to_loc = {pop:random_loc() for pop in pops}
	metros = np.arange(0,sizes['n_metro'])
	metro_loc = {metro:random_loc() for metro in metros}
	asns = np.arange(sizes['n_asn'])
	# ug_to_vol = {(metro,asn): np.power(2,np.random.uniform(1,10)) for metro in metros for asn in asns}
	# ug_to_vol = {(metro,asn): np.random.uniform(1,100) for metro in metros for asn in asns}
	ug_to_vol = {(metro,asn): 1 for metro in metros for asn in asns}
	ug_perfs = {ug: {} for ug in ug_to_vol}
	peers = np.arange(0,sizes['n_peer'])
	popps = []
	n_providers = sizes['n_providers']
	for pop in pops:
		some_peers = np.random.choice(peers, size=np.random.randint(sizes['min_peerings_per_pop'],
			sizes['max_peerings_per_pop']),replace=False)
		provs = [p for p in some_peers if p < n_providers]
		if len(provs) == 0: # ensure at least one provider per pop
			some_peers = np.append(some_peers, [np.random.randint(n_providers)])
		for peer in some_peers:
			popps.append((pop,peer))
	provider_popps = [popp for popp in popps if popp[1] < n_providers]
	for ug in ug_to_vol:
		some_popps = np.random.choice(np.arange(len(popps)), size=np.random.randint(3,
			sizes['max_popp_per_ug']), replace=False)
		for popp in some_popps:
			ug_perfs[ug][popps[popp]] = random_latency()
		for popp in provider_popps:
			# All UGs have routes through deployment providers
			# Assume for now that relationships don't depend on the PoP
			# also assume these performances are probably worse
			ug_perfs[ug][popp] = random_transit_provider_latency()
	ugs = list(ug_to_vol)
	ug_anycast_perfs = {ug:1 for ug in ugs}
	ingress_priorities = get_random_ingress_priorities(ug_perfs, ug_anycast_perfs,
		pop_to_loc, metro_loc)

	## Simulate random link capacities
	# links should maybe hold ~ N * max user volume
	max_user_volume = max(list(ug_to_vol.values()))
	mu = len(ug_perfs)/3*max_user_volume
	sig = mu / 10
	link_capacities = {popp: mu + sig * np.random.normal() for popp in popps}
	print("----Done Creating Random Deployment-----")
	print("Deployment has {} users, {} popps, {} pops".format(
		len(ugs), len(popps), len(pop_to_loc)))
	return {
		'ugs': ugs,
		'ug_perfs': ug_perfs,
		'ug_to_vol': ug_to_vol,
		'ug_anycast_perfs': ug_anycast_perfs,
		'whole_deployment_ugs': ugs,
		'whole_deployment_ug_to_vol': ug_to_vol,
		'link_capacities': link_capacities,
		'ingress_priorities': ingress_priorities,
		'popps': popps,
		'metro_loc': metro_loc,
		'pop_to_loc': pop_to_loc,
		'n_providers': n_providers,
		'provider_popps': provider_popps,
	}

if __name__ == "__main__":
	load_actual_deployment()
	# cluster_actual_users()

