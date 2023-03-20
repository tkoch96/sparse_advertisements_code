from constants import *
from helpers import *

def get_random_ingress_priorities(ug_perfs, pop_to_loc, metro_loc):
	## Simulate random ingress priorities for each UG
	ingress_priorities = {}
	for ug in ug_perfs:
		ingress_priorities[ug] = {}
		these_peerings = list(get_difference(list(ug_perfs[ug]), ['anycast']))
		ranked_peerings_by_dist = sorted(these_peerings, key = lambda el : geopy.distance.geodesic(
			pop_to_loc[el[0]], metro_loc[ug[0]]).km)
		priorities = {pi:i for i,pi in enumerate(ranked_peerings_by_dist)}
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

def load_actual_deployment():
	# CACHE_DIR = '/home/ubuntu/peering_measurements/cache'
	CACHE_DIR = 'cache'
	considering_pops = ['miami','atlanta']
	provider_fn = os.path.join(CACHE_DIR, 'vultr_provider_popps.csv')
	provider_popps = []
	for row in open(provider_fn, 'r'):
		pop,peer = row.strip().split(',')
		if pop not in considering_pops:
			continue
		provider_popps.append((pop,peer))
	CACHE_DIR = 'cache'
	lat_fn = os.path.join(CACHE_DIR, 'vultr_ingress_latencies_by_dst.csv')
	ug_perfs = {}
	n_allowed_popps = 30
	tmppopps = {}
	for row in open(lat_fn, 'r'):
		fields = row.strip().split(',')
		if fields[2] not in considering_pops: continue
		t,ip,pop,peer,lat = fields
		try:
			tmppopps[pop,peer]
		except KeyError:
			if len(tmppopps) >= n_allowed_popps:
				continue
			tmppopps[pop,peer] = None
		lat = float(lat)*1000
		lat = np.maximum(MIN_LATENCY, np.minimum(MAX_LATENCY, lat))
		metro = 'tmp'
		asn = ip32_to_24(ip)
		ug = (metro,asn)
		try:
			ug_perfs[ug][pop,peer] = lat
		except KeyError:
			ug_perfs[ug] = {(pop,peer): lat}

	ugs = sorted(list(ug_perfs))
	popps = sorted(list(set(popp for ug in ugs for popp in ug_perfs[ug])))
	print(popps[-3:])
	print("{} popps, {} ugs".format(len(popps), len(ugs)))
	n_providers = len(set(peer for pop,peer in provider_popps if (pop,peer) in popps))
	ug_to_vol = {ug:1 for ug in ugs}
	link_capacities = {popp:len(ugs) for popp in popps}

	metros = list(set(metro for metro,asn in ugs))
	metro_loc = {m:(np.random.random(),np.random.random()) for m in metros}
	pop_to_loc = {pop:POP_TO_LOC['vultr'][pop] for pop in considering_pops}

	ingress_priorities = get_random_ingress_priorities(ug_perfs, pop_to_loc, metro_loc)

	deployment = {
		'ugs': ugs,
		'ug_perfs': ug_perfs,
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
	for k,v in deployment.items():
		print("{} {} ".format(k,v))
		print("\n")
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
	ingress_priorities = get_random_ingress_priorities(ug_perfs, pop_to_loc, metro_loc)

	## Simulate random link capacities
	# links should maybe hold ~ N * max user volume
	max_user_volume = max(list(ug_to_vol.values()))
	mu = len(ug_perfs)/3*max_user_volume
	sig = mu / 10
	link_capacities = {popp: mu + sig * np.random.normal() for popp in popps}
	ugs = list(ug_to_vol)
	print("----Done Creating Random Deployment-----")
	print("Deployment has {} users, {} popps, {} pops".format(
		len(ugs), len(popps), len(pop_to_loc)))
	print(ug_perfs);exit(0)
	return {
		'ugs': ugs,
		'ug_perfs': ug_perfs,
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

