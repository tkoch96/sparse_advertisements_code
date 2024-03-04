import tqdm, numpy as np
from constants import *
from helpers import *

def get_random_ingress_priorities(deployment):
	## Simulate random ingress priorities for each UG
	ug_perfs = deployment['ug_perfs']
	ug_anycast_perfs = deployment['ug_anycast_perfs']
	pop_to_loc = deployment['pop_to_loc']
	metro_loc = deployment['metro_loc']
	provider_popps = deployment['provider_popps']
	provider_ases = list(set(peer for pop,peer in provider_popps))


	ingress_priorities = {}
	dist_cache = {}
	for ug in tqdm.tqdm(ug_perfs,desc="Assigning ingress priorities randomly."):
		ingress_priorities[ug] = {}
		## approximate the anycast interface by getting the one with closest latency
		popps = list(ug_perfs[ug])

		non_provider_popps = get_difference(popps, provider_popps)
		if len(non_provider_popps) > 0:
			perfs = np.array([ug_perfs[ug][popp] for popp in non_provider_popps])
			## anycast is the most favored peer
			probably_anycast = non_provider_popps[np.argmin(np.abs(perfs - ug_anycast_perfs[ug]))]
			priorities = {probably_anycast:0}
		else:
			perfs = np.array([ug_perfs[ug][popp] for popp in popps])
			## anycast is the most favored ingress
			probably_anycast = popps[np.argmin(np.abs(perfs - ug_anycast_perfs[ug]))]
			priorities = {probably_anycast:0}

		other_peerings = list(get_difference(list(ug_perfs[ug]), [probably_anycast]))
		if len(other_peerings) > 0:
			### Model 
			## user has a preferred provider
			## shortest path within that preferred provider
			## user prefers peering connections to providers, random preference in valid peers
			## random violations of the model

			op_by_as = {}
			for pop,peer in other_peerings:
				try:
					op_by_as[peer].append((pop,peer))
				except KeyError:
					op_by_as[peer] = [(pop,peer)]

			priority_counter = 1
			ases = list(op_by_as)
			these_non_provider_ases = get_difference(ases, provider_ases)
			these_provider_ases = get_intersection(ases, provider_ases)
			np.random.shuffle(these_non_provider_ases)
			np.random.shuffle(these_provider_ases)
			ases = these_non_provider_ases + these_provider_ases
			np.random.shuffle(ases) ### Not prefer-peer
			for _as in ases:
				associated_dists = []
				this_as_peerings = op_by_as[_as]
				for popp in this_as_peerings:
					try:
						d = dist_cache[pop_to_loc[popp[0]], metro_loc[ug[0]]]
					except KeyError:
						d = geopy.distance.geodesic(
							pop_to_loc[popp[0]], metro_loc[ug[0]]).km
						# add noise to break ties
						dist_cache[pop_to_loc[popp[0]], metro_loc[ug[0]]] = d
					d += .01 * np.random.uniform()
					associated_dists.append(d)
				ranked_peerings_by_dist = sorted(zip(associated_dists,this_as_peerings), key = lambda el : el[0])
				for _,pi in ranked_peerings_by_dist:
					priorities[pi] = priority_counter
					priority_counter += 1
			# print(ug)
			# print(these_non_provider_ases)
			# print(sorted(ug_perfs[ug].items(), key = lambda el : el[1]))
			# print(print(priorities))
			# exit(0)
			## randomly flip some priorities
			if len(priorities) > 1:
				for pi in list(priorities):
					if np.random.random() < .05:
						other_pi = list(get_difference(list(priorities), [pi]))[np.random.choice(len(priorities)-1)]
						tmp = copy.copy(priorities[pi])
						priorities[pi] = copy.copy(priorities[other_pi])
						priorities[other_pi] = tmp
						# print("Randomly flipping {}, popp {} and {}".format(ug,pi,other_pi))

		for popp,priority in priorities.items():
			ingress_priorities[ug][popp] = priority
		# if np.random.random() > .999:
		# 	print("{} -- {}".format(ug, ingress_priorities[ug]))
	return ingress_priorities

def get_link_capacities(deployment):

	methodology = ['my_heuristic', 'jiangchen-sigcomm'][1]


	# controls backup volume we have, therefore how hard the resilience
	# problem is to solve
	EASYNESS_MULT = { 
		'easy': 1,
		'medium': .5,
		'hard': .01,
	}[RESILIENCE_DIFFICULTY]

	popps = deployment['popps']
	ug_perfs = deployment['ug_perfs']
	ug_to_vol = deployment['ug_to_vol']
	provider_popps = deployment['provider_popps']
	ugs = deployment['ugs']

	all_pops = list(set([popp[0] for popp in popps]))

	# vol best is the client volume per popp if everyone went to their lowest latency link
	# vol popp is reachable volume for a popp
	vol_best,vol_popp = {popp: 0 for popp in popps}, {popp:0 for popp in get_difference(popps,provider_popps)}
	popp_to_ug = {}
	
	if True:
		import matplotlib.pyplot as plt
		n_peers_by_ug = {ug:len([el for el in ug_perfs[ug] if el not in provider_popps]) for ug in ug_perfs}
		x,cdf_x = get_cdf_xy(list(n_peers_by_ug.values()))
		plt.semilogx(x,cdf_x,label='Just Peers')
		n_peers_by_ug = {ug:len(ug_perfs[ug]) for ug in ug_perfs}
		x,cdf_x = get_cdf_xy(list(n_peers_by_ug.values()))
		plt.semilogx(x,cdf_x,label='All')
		plt.legend()
		plt.xlabel("Number reachable peers")
		plt.ylabel("cdf of UGS")
		plt.grid(True)
		plt.savefig('figures/number_peers_by_ug-{}.pdf'.format(DPSIZE))
		plt.clf();plt.close()


	for ug in ug_perfs:
		these_popps = list(ug_perfs[ug])
		best_performer = these_popps[np.argmin([ug_perfs[ug][popp] for popp in these_popps])]
		vol_best[best_performer] += deployment['ug_to_vol'][ug]
		for popp in ug_perfs[ug]:
			if popp in provider_popps:
				continue
			vol_popp[popp] += deployment['ug_to_vol'][ug]
			try:
				popp_to_ug[popp].append(ug)
			except KeyError:
				popp_to_ug[popp] = [ug]

	## Maximum volume a peer should be expected to handle
	max_peer_volume = np.max(list([v for popp,v in vol_best.items() if popp not in provider_popps]))
	## Typical volume we would expect to flow over transit
	# proportional to transit providers fairly sharing all of user load
	baseline_transit_volume = EASYNESS_MULT * sum(list(ug_to_vol.values())) / len(provider_popps)

	## Compute anycast load on each link
	ingress_priorities = deployment['ingress_priorities']
	anycast_load = {}
	all_loads_by_popp_across_pops = {pop:{} for pop in all_pops}
	for ug in ugs:
		ranked_prefs = sorted(ingress_priorities[ug].items(), key = lambda el : el[1])
		best_popp = ranked_prefs[0][0]
		try:
			anycast_load[best_popp] += ug_to_vol[ug]
		except KeyError:
			anycast_load[best_popp] = ug_to_vol[ug]
		for pop in all_pops:
			these_ip = {popp:i for popp,i in ingress_priorities[ug].items() if popp[0] == pop}
			try:
				best_popp = sorted(these_ip.items(), key = lambda el : el[1])[0][0]
			except IndexError:
				continue
			try:
				all_loads_by_popp_across_pops[pop][best_popp].append(ug_to_vol[ug])
			except KeyError:
				all_loads_by_popp_across_pops[pop][best_popp] = [ug_to_vol[ug]]

	print(anycast_load)
	link_capacities = {}
	if methodology == 'my_heuristic':
		for popp,v in vol_best.items():
			if popp not in provider_popps:
				## Set capacity roughly as the amount of client traffic you'd expect to receive
				link_capacities[popp] = .05 * vol_popp[popp] # kind of easy
			else:
				## Set capacity as some baseline + resilience
				## resilience should be proportional to max peer volume
				link_capacities[popp] = baseline_transit_volume
	else:
		#### increase capacitity so that we can handle 1.x times anycast load
		scale_factor = 1.3
		small_value = np.mean(list(ug_to_vol.values()))
		for popp in popps:
			link_capacities[popp] = scale_factor * anycast_load.get(popp, small_value)
	n_prov,n_tot=0,0
	anycast_link_utils = []
	for popp in anycast_load:
		n_tot += anycast_load[popp]
		if popp in provider_popps:
			n_prov += anycast_load[popp]
		if methodology == 'my_heuristic':
			tmp = copy.copy(link_capacities[popp])
			#### increase capacitity so that we can handle 1.x times anycast load
			scale_factor = 1.1
			link_capacities[popp] = scale_factor * np.maximum(anycast_load[popp], link_capacities[popp])
			if tmp != link_capacities[popp]:
				print("Increased {} link cap by a factor of {} to handle anycast load, provider: {}".format(
					popp, link_capacities[popp]/tmp, popp in provider_popps))
		anycast_link_utils.append(anycast_load[popp] / link_capacities[popp])


	print("provider volume makes up {} of total anycast".format(round(100*n_prov/n_tot,2)))
	vol_by_metro = {}
	for metro,asn in ugs:
		try:
			vol_by_metro[metro] += ug_to_vol[(metro,asn)]
		except KeyError:
			vol_by_metro[metro] = ug_to_vol[(metro,asn)]
	oversubscriptions = {ug:None}
	for ug in ug_perfs:
		total_v = ug_to_vol[ug]
		total_available_v = sum([link_capacities[popp] for popp in ug_perfs[ug]])
		oversubscriptions[ug] = total_available_v / total_v
	print(vol_by_metro)


		### For each pop, what is the distribution of load along its pops assuming 1pp scheme?
	swings_by_pop = []
	for pop in all_loads_by_popp_across_pops:
		this_arr = {popp: sum(all_loads_by_popp_across_pops[pop][popp]) for popp in all_loads_by_popp_across_pops[pop]}
		vthis_arr = {popp: np.var(all_loads_by_popp_across_pops[pop][popp])/np.mean(all_loads_by_popp_across_pops[pop][popp]) for popp in all_loads_by_popp_across_pops[pop]}
		ol_this_arr = {popp: sum(all_loads_by_popp_across_pops[pop][popp])/link_capacities[popp] for popp in all_loads_by_popp_across_pops[pop]}
		# print(pop + '\n')
		# print(ol_this_arr)
		# print(vthis_arr)
		# print(list(this_arr.values()))
		swings_by_pop = swings_by_pop + list(ol_this_arr.values())

	if True:
		import matplotlib.pyplot as plt
		f,ax=plt.subplots(2,1)
		f.set_size_inches(6,12)
		
		x,cdf_x = get_cdf_xy(list(vol_best.values()))
		ax[0].semilogx(x,cdf_x,label='Best PoPP Volume')
		x,cdf_x = get_cdf_xy(list(oversubscriptions.values()))
		ax[0].semilogx(x,cdf_x,label='Oversubscriptions')
		x,cdf_x = get_cdf_xy(list(vol_popp.values()))
		ax[0].semilogx(x,cdf_x,label="Reachable PoPP Volume")
		x,cdf_x = get_cdf_xy(list(ug_to_vol.values()))
		ax[0].semilogx(x,cdf_x,label="User Volumes")
		x,cdf_x = get_cdf_xy(list(swings_by_pop))
		ax[0].semilogx(x,cdf_x,label="PoP Swings")
		x,cdf_x = get_cdf_xy(list(link_capacities.values()))
		ax[0].semilogx(x,cdf_x,label="All Link Caps")
		x,cdf_x = get_cdf_xy(list(vol_by_metro.values()))
		ax[0].semilogx(x,cdf_x,label="All Metro Volumes")
		x,cdf_x = get_cdf_xy(anycast_link_utils)
		ax[0].semilogx(x, cdf_x, label="Anycast Link Utils")
		ax[0].legend()
		ax[0].set_xlabel('Volume')
		ax[0].set_ylabel("CDF of Users/PoPPs/Metros")
		ax[0].grid(True)

		ax[0].set_xticks([.001,.01,.1,1,10,100,1000,10000])
		plt.savefig("figures/link_capacity_summary-{}.pdf".format(DPSIZE))
		plt.clf()
		plt.close()
	
	return link_capacities

def cluster_actual_users(**kwargs):
	considering_pops = kwargs.get('considering_pops')
	cpstr = "-".join(sorted(considering_pops))
	cluster_cache_fn = os.path.join(CACHE_DIR, 'clustered_perfs_{}.pkl'.format(cpstr))
	if not os.path.exists(cluster_cache_fn):
		anycast_latencies, ug_perfs = load_actual_perfs(**kwargs)
		ug_to_ip = {}

		### Form a matrix of all latencies
		ugs = sorted(list(ug_perfs))
		popps = sorted(list(set(popp for ug in ugs for popp in ug_perfs[ug])))
		print("{} UGs, {} popps".format(len(ugs), len(popps)))

		ug_to_ind = {ug:i for i,ug in enumerate(ugs)}
		popp_to_ind = {popp:i for i, popp in enumerate(popps)}


		IGNORE_LAT = 10*NO_ROUTE_LATENCY
		latencies_mat = IGNORE_LAT * np.ones((len(ugs), len(popps)), dtype=np.float32)
		best_pop_by_ug, best_lat_by_ug = {}, {}
		for ug, perfs in ug_perfs.items():
			for popp,lat in perfs.items():
				latencies_mat[ug_to_ind[ug],popp_to_ind[popp]] = lat
				try:
					if best_lat_by_ug[ug] > lat:
						best_lat_by_ug[ug] = lat
						best_pop_by_ug[ug] = popp[0]
				except KeyError:
					best_lat_by_ug[ug] = lat
					best_pop_by_ug[ug] = popp[0]

		from sklearn.cluster import Birch
		### threshold would probably be tuned by whatever gets me an appropriate number of clusters
		brc = Birch(threshold=1,n_clusters=None)
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
		for lab in all_labs:
			if len(examples_by_label[lab]) == 1: continue
			# print("Lab {}, ugs: ".format(lab))
			# for ug in examples_by_label[lab]:
			# 	print("{} -- {}".format(ug,ug_perfs[ug]))
			# print("\n\n")
			if np.random.random() > .95:break

		clustered_ug_perfs, clustered_anycast_perfs = {},{}
		ug_id = 0
		print("{} subcluster labels but {} different ug labels".format(len(brc.subcluster_labels_),
			len(examples_by_label)))
		for sc_center, lab in zip(brc.subcluster_centers_, brc.subcluster_labels_):
			try:
				these_ugs = examples_by_label[lab]
			except KeyError:
				# print("no UGs found for subcluster label {}".format(lab))
				continue
			## This doesn't happen
			# all_fingerprints = {}
			# for ug in these_ugs:
			# 	fp = tuple(sorted([popp_to_ind[popp] for popp in ug_perfs[ug]]))
			# 	try:
			# 		all_fingerprints[fp].append(ug)
			# 	except KeyError:
			# 		all_fingerprints[fp] = [ug]
			# print(len(all_fingerprints))

			# if len(all_fingerprints) > 1:
			# 	print("UGs clustered together but fingerprints are different")
			# 	ind_to_popp = {i:popp for popp,i in popp_to_ind.items()}
			# 	for i,fpi in enumerate(all_fingerprints):
			# 		poppsi = [ind_to_popp[poppi] for poppi in fpi]
			# 		for j,fpj in enumerate(all_fingerprints):
			# 			if j >= i:
			# 				break
			# 			poppsj = [ind_to_popp[poppj] for poppj in fpj]

			# 			amb = get_difference(poppsi,poppsj)
			# 			bma = get_difference(poppsj,poppsi)
			# 			print("{} {} {} {}".format(all_fingerprints[fpi], all_fingerprints[fpj],
			# 				amb,bma))
			# 	exit(0)


			pops_these_ugs = list([best_pop_by_ug[ug] for ug in these_ugs])
			most_popular_pop = max(set(pops_these_ugs), key=pops_these_ugs.count)
			metro = most_popular_pop
			this_lab_ug = (metro, ug_id)

			for _,client_ip in these_ugs:
				try:
					ug_to_ip[this_lab_ug].append(client_ip)
				except KeyError:
					ug_to_ip[this_lab_ug] = [client_ip]

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

		pickle.dump([clustered_ug_perfs,clustered_anycast_perfs, ug_to_ip], open(cluster_cache_fn,'wb'))
	else:
		clustered_ug_perfs, clustered_anycast_perfs, ug_to_ip = pickle.load(open(cluster_cache_fn,'rb'))

	return clustered_ug_perfs, clustered_anycast_perfs, ug_to_ip

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
		t,ip,pop,peer,_,lat = fields
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
	to_del = []
	for ug in ug_perfs:
		if len(ug_perfs[ug]) == 1:
			to_del.append(ug)
			continue
		for popp, lats in ug_perfs[ug].items():
			ug_perfs[ug][popp] = np.min(lats)
		if any([lat == 1 for lat in ug_perfs[ug].values()]):
			# trivial
			to_del.append(ug)
			continue
	for ug in to_del: del ug_perfs[ug]
	ugs = sorted(list(ug_perfs))
	popps = sorted(list(set(popp for ug in ugs for popp in ug_perfs[ug])))
	print("{} UGs, {} popps".format(len(ugs), len(popps)))

	anycast_latencies = {}
	anycast_pop = {}
	for row in tqdm.tqdm(open(os.path.join(CACHE_DIR, 'vultr_anycast_latency_smaller.csv')
		,'r'),desc="Parsing VULTR anycast latencies"):
		# if np.random.random() > .999999:break
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
	changed = False
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

	if kwargs.get('do_filter', True):

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
				if kwargs.get('focus_on_peers',True):
					if popp in provider_popps: continue
				popp_to_ug[popp].append(ug)
		
		import matplotlib.pyplot as plt
		x,cdf_x = get_cdf_xy(list([len(popp_to_ug[popp]) for popp in popp_to_ug]))
		plt.plot(x,cdf_x)
		plt.xlabel("Number of UGs per Ingress")
		plt.ylabel("CDF of Ingresses")
		plt.grid(True)
		plt.savefig('figures/n_ugs_per_ingress.pdf')

		n_total_users, n_peer_was_best, n_provider_was_best = 0,0,0
		for popp,_ugs in popp_to_ug.items():
			if kwargs.get('focus_on_peers',True):
				if popp in provider_popps: continue

			### Favor users whose best popp is not a provider
			peer_ugs, provider_ugs = [],[]
			for _ug in _ugs:
				these_popps = list(ug_perfs[_ug])
				perfs = np.array([ug_perfs[_ug][_popp] for _popp in these_popps])
				best_popp = these_popps[np.argmin(perfs)]
				if best_popp in provider_popps:
					provider_ugs.append(_ug)
				else:
					peer_ugs.append(_ug)
			if len(peer_ugs) > 0:
				np.random.shuffle(peer_ugs)
			if len(provider_ugs) > 0:
				np.random.shuffle(provider_ugs)

			n_keeping_peer = np.minimum(len(peer_ugs), max_n_ug)
			n_peer_was_best += n_keeping_peer
			n_keeping_provider = np.minimum(max_n_ug - n_keeping_peer, len(provider_ugs))
			n_provider_was_best += n_keeping_provider
			n_keep = np.minimum(len(_ugs), max_n_ug)
			n_total_users += n_keep

			_ugs = peer_ugs + provider_ugs
			popp_to_ug[popp] = _ugs[0:n_keep]

		print("Out of {} UGs, {} ({} pct) peer was best, {} ({} pct) provider was best.".format(
			n_total_users, n_peer_was_best, round(n_peer_was_best*100/n_total_users,2),
			n_provider_was_best,round(n_provider_was_best*100/n_total_users,2)))

		if kwargs.get('focus_on_peers',True):
			keep_ugs = list(set(ug for popp in popp_to_ug for ug in popp_to_ug[popp] if popp not in provider_popps))
		else:
			keep_ugs = list(set(ug for popp in popp_to_ug for ug in popp_to_ug[popp]))
		ug_perfs = {ug:ug_perfs[ug] for ug in keep_ugs}
		print("{} UGs after limiting to those with a peer measurement".format(len(ug_perfs)))

		## Remove providers who have very few users
		n_ugs_by_provider = {provider:0 for provider in provider_popps}
		n_providers_by_ug = {ug:0 for ug in ug_perfs}
		for ug in ug_perfs:
			for provider in provider_popps:
				try:
					ug_perfs[ug][provider]
					n_ugs_by_provider[provider] += 1
					n_providers_by_ug[ug] += 1
				except KeyError:
					continue
		plt.clf()
		plt.close()
		x,cdf_x = get_cdf_xy(list([n/len(ug_perfs) for n in n_ugs_by_provider.values()]))
		plt.plot(x,cdf_x,label="Frac UGs by Provider")
		x,cdf_x = get_cdf_xy(list([n/len(provider_popps) for n in n_providers_by_ug.values()]))
		plt.plot(x,cdf_x,label="Frac Providers by UG")
		plt.legend()
		plt.savefig('figures/provider_meas_description_preprune.pdf')
		plt.clf()
		plt.close()

		to_del_popps = []
		for popp, n in sorted(n_ugs_by_provider.items(), key = lambda el : el[1]):
			if n < 2:
				to_del_popps.append(popp)
			else:
				break
		print("Removing providers : {} since they don't have enough measurements.".format(
			to_del_popps))
		ug_perfs = {ug: {popp: ug_perfs[ug][popp] for popp in get_difference(ug_perfs[ug], to_del_popps)}
			for ug in ug_perfs}

		provider_popps = get_difference(provider_popps,to_del_popps)

		## Remove providers who have very few users
		n_ugs_by_provider = {provider:0 for provider in provider_popps}
		n_providers_by_ug = {ug:0 for ug in ug_perfs}
		for ug in ug_perfs:
			for provider in provider_popps:
				try:
					ug_perfs[ug][provider]
					n_ugs_by_provider[provider] += 1
					n_providers_by_ug[ug] += 1
				except KeyError:
					continue
		plt.clf()
		plt.close()
		x,cdf_x = get_cdf_xy(list([n/len(ug_perfs) for n in n_ugs_by_provider.values()]))
		plt.plot(x,cdf_x,label="Frac UGs by Provider")
		x,cdf_x = get_cdf_xy(list([n/len(provider_popps) for n in n_providers_by_ug.values()]))
		plt.plot(x,cdf_x,label="Frac Providers by UG")
		plt.legend()
		plt.savefig('figures/provider_meas_description_postprune.pdf')
		plt.clf()
		plt.close()
		
		## Remove users have measurements to too few providers
		cutoff_frac = .35
		to_del = list([ug for ug,n in n_providers_by_ug.items() if n/len(provider_popps) < cutoff_frac])
		print("Removing {} out of {} UGs since they don't have measurements to enough providers.".format(
			len(to_del), len(ug_perfs)))
		for ug in to_del:
			del ug_perfs[ug]

		n_ugs_by_provider = {provider:0 for provider in provider_popps}
		n_providers_by_ug = {ug:0 for ug in ug_perfs}
		for ug in ug_perfs:
			for provider in provider_popps:
				try:
					ug_perfs[ug][provider]
					n_ugs_by_provider[provider] += 1
					n_providers_by_ug[ug] += 1
				except KeyError:
					continue
		plt.clf()
		plt.close()
		x,cdf_x = get_cdf_xy(list([n/len(ug_perfs) for n in n_ugs_by_provider.values()]))
		plt.plot(x,cdf_x,label="Frac UGs by Provider")
		x,cdf_x = get_cdf_xy(list([n/len(provider_popps) for n in n_providers_by_ug.values()]))
		plt.plot(x,cdf_x,label="Frac Providers by UG")
		plt.legend()
		plt.savefig('figures/provider_meas_description_postprune2.pdf')
		plt.clf()
		plt.close()

		
		# for ug in list(ug_perfs):
		# 	if len(ug_perfs[ug]) < 2:
		# 		del ug_perfs[ug]
	anycast_latencies = {ug:anycast_latencies[ug] for ug in ug_perfs}

	ugs = sorted(list(ug_perfs))
	popps = sorted(list(set(popp for ug in ugs for popp in ug_perfs[ug])))
	print("{} UGs, {} popps after limiting users".format(len(ugs), len(popps)))

	return anycast_latencies, ug_perfs

# def export_interesting_measurements():
# 	# considering_pops = list(POP_TO_LOC['vultr'])#['miami', 'atlanta']
# 	considering_pops = ['miami', 'atlanta']
# 	anycast_latencies, ug_perfs = load_actual_perfs(considering_pops = considering_pops, n_users_per_peer=300000,
# 		focus_on_peers=False)
# 	# ug_perfs = get_random_deployment_by_size('really_friggin_small')['ug_perfs']

# 	all_popps = list(set(popp for ug in ug_perfs for popp in ug_perfs[ug]))
# 	profiles = {}
# 	for ug in ug_perfs:
# 		all_perfs = np.array(list(ug_perfs[ug].values()))
# 		ml = np.min(all_perfs)
# 		popp_key = tuple(sorted(list(ug_perfs[ug].keys()), key = lambda el : "-".join([str(_el) for _el in el])))
# 		try:
# 			profiles[popp_key].append((ug,ml))
# 		except KeyError:
# 			profiles[popp_key] = [(ug, ml)]
# 	print("{} profiles".format(len(profiles)))
# 	n_to_keep = 400
# 	keep_ugs = []
# 	for p in profiles:
# 		keep_ugs = keep_ugs + [el[0] for el in sorted(profiles[p], key = lambda el : el[1])[0:n_to_keep]]
# 	print("Keeping {} Ugs".format(len(keep_ugs)))

# 	with open(os.path.join(CACHE_DIR, 'interesting_targets_to_probe.csv'),'w') as f:
# 		for ug in keep_ugs:
# 			all_perfs = np.array(list(ug_perfs[ug].values()))
# 			if np.min(all_perfs) > 50:
# 				# this target would not likely use one of these popps
# 				continue
# 			if np.random.random() > .99:
# 				print("UG {} perfs {} anycast {}".format(ug,ug_perfs[ug],anycast_latencies[ug]))
# 			_, ip = ug
# 			popps = list(ug_perfs[ug])
# 			popps = [str(popp[0]) + "|" + str(popp[1]) for popp in popps]
# 			popps_str = "-".join(popps)
# 			f.write("{},{}\n".format(ip,popps_str))

def get_random_ug_to_vol(deployment):
	## Set UG vols to balance non-provider expected volume
	ugs = deployment['ugs']
	popps = deployment['popps']
	provider_popps = deployment['provider_popps']
	ug_perfs = deployment['ug_perfs']


	ug_to_vol = {ug:.5 + 1000 * np.random.random() for ug in ugs}
	non_provider_popps = get_difference(popps,provider_popps)
	popp_to_ug = {popp:[] for popp in non_provider_popps}
	for ug in ug_perfs:
		for popp in ug_perfs[ug]:
			if popp in provider_popps:
				continue
			popp_to_ug[popp].append(ug)
	def calc_popp_vol(ugv):
		popp_vol = {popp:sum(ugv[ug] for ug in popp_to_ug[popp]) for popp in popp_to_ug}
		return popp_vol

	last_r = 100000
	max_n_iter = 1
	end = False
	_iter = 0
	vol_by_metro = {}
	for metro,asn in ugs:
		try:
			vol_by_metro[metro] += ug_to_vol[(metro,asn)]
		except KeyError:
			vol_by_metro[metro] = ug_to_vol[(metro,asn)]
	print(vol_by_metro)
	while not end:
		## Tries to increase low-volume ingress expected volume by inflating user volumes
		## I.e., attempts to deal with low target counts in certain ingresses
		popp_vols = calc_popp_vol(ug_to_vol)
		all_vols = list([popp_vols[popp] for popp in non_provider_popps])

		ranked_vols = np.argsort(all_vols)
		min_popp,max_popp = non_provider_popps[ranked_vols[0]],non_provider_popps[ranked_vols[-1]]
		curr_min_ind = 1
		while len(get_difference(popp_to_ug[min_popp], popp_to_ug[max_popp])) == 0:
			min_popp = non_provider_popps[ranked_vols[curr_min_ind]]
			curr_min_ind += 1
			if min_popp == max_popp:
				end = True
				break

		min_vol,max_vol = popp_vols[min_popp],popp_vols[max_popp]
		
		this_r = max_vol / min_vol
		if this_r < 1.1:
			end = True
		for ug in popp_to_ug[min_popp]:
			if ug in popp_to_ug[max_popp]: continue
			# print("multiplying {}".format(ug))
			ug_to_vol[ug] = ug_to_vol[ug] * 1.2
		
		if _iter == max_n_iter:
			break
			
		if _iter > 1000 and last_r <= this_r:
			break

		last_r = copy.copy(this_r)
		_iter += 1
	vol_by_metro = {}
	for metro,asn in ugs:
		try:
			vol_by_metro[metro] += ug_to_vol[(metro,asn)]
		except KeyError:
			vol_by_metro[metro] = ug_to_vol[(metro,asn)]
	### normalize each metro to be random between 1 and 10
	normal_factor = {metro: 1+9*np.random.random() for metro in vol_by_metro}
	for ug in ugs:
		ug_to_vol[ug] = ug_to_vol[ug] * normal_factor[ug[0]] / vol_by_metro[ug[0]]


	# Normalize
	max_vol = np.max(list(ug_to_vol.values()))
	for ug in ug_to_vol:
		ug_to_vol[ug] = ug_to_vol[ug] / max_vol

	return ug_to_vol

def load_actual_deployment(deployment_size):
	if deployment_size == 'actual-large':
		considering_pops = list(POP_TO_LOC['vultr'])
	# elif deployment_size == 'actual':
	# 	considering_pops = ['miami','amsterdam','newyork','atlanta','saopaulo','singapore','tokyo']
	elif deployment_size == 'actual':
		pops = list(POP_TO_LOC['vultr'])
		considering_pops = np.random.choice(pops, size=N_POPS_ACTUAL_DEPLOYMENT, replace=False)
	elif deployment_size == "actual_first_prototype":
		considering_pops = ['newyork', 'atlanta']
	else:
		raise ValueError("Deployment size {} not supported".format(deployment_size))
	print("Considering pops : {}".format(considering_pops))
	cpstr = "-".join(sorted(considering_pops))
	deployment_cache_fn = os.path.join(CACHE_DIR, 'actual_deployment_cache_{}.pkl'.format(cpstr))
	if not os.path.exists(deployment_cache_fn):
		pop_to_loc = {pop:POP_TO_LOC['vultr'][pop] for pop in considering_pops}

		# anycast_latencies, ug_perfs = load_actual_perfs(considering_pops=considering_pops)
		ug_perfs, anycast_latencies, ug_to_ip = cluster_actual_users(considering_pops=considering_pops, 
			n_users_per_peer=300)

		## add sub-ms latency noise to arbitrarily break ties
		for ug in ug_perfs:
			for popp,lat in ug_perfs[ug].items():
				ug_perfs[ug][popp] = lat + .1 * np.random.uniform()

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
		pop_to_n = {}
		print("{} popps, {} ugs".format(len(popps), len(ugs)))
		for pop,peer in popps:
			try:
				pop_to_n[pop] += 1
			except KeyError:
				pop_to_n[pop] = 1
		print("Ingress counts by PoP: {}".format(pop_to_n))
		provider_popps = get_intersection(provider_popps, popps)
		n_providers = len(set(peer for pop,peer in provider_popps))
	
		metros = list(set(metro for metro,asn in ugs))
		metro_loc = {}
		for ug in ug_perfs:
			metro,asn = ug
			ug_popps = list(ug_perfs[ug])
			closest_popp = ug_popps[np.argmin([ug_perfs[ug][popp] for popp in ug_popps])]
			metro_loc[metro] = pop_to_loc[closest_popp[0]]

		deployment = {
			'ugs': ugs,
			'ug_to_ip': ug_to_ip,
			'ug_perfs': ug_perfs,
			'ug_anycast_perfs': anycast_latencies,
			'whole_deployment_ugs': ugs,
			'whole_deployment_ug_perfs': ug_perfs,
			'popps': popps,
			'metro_loc': metro_loc,
			'pop_to_loc': pop_to_loc,
			'n_providers': n_providers,
			'provider_popps': provider_popps,
		}

		ug_to_vol = get_random_ug_to_vol(deployment)
		deployment['ug_to_vol'] = ug_to_vol
		deployment['whole_deployment_ug_to_vol'] = ug_to_vol

		ingress_priorities = get_random_ingress_priorities(deployment)
		deployment['ingress_priorities'] = ingress_priorities
		deployment['whole_deployment_ingress_priorities'] = copy.deepcopy(deployment['ingress_priorities'])

		link_capacities = get_link_capacities(deployment)
		deployment['link_capacities'] = link_capacities

		pickle.dump(deployment, open(deployment_cache_fn,'wb'))

	else:
		deployment = pickle.load(open(deployment_cache_fn,'rb'))

		ug_to_vol = get_random_ug_to_vol(deployment)
		deployment['ug_to_vol'] = ug_to_vol
		deployment['whole_deployment_ug_to_vol'] = ug_to_vol

		ingress_priorities = get_random_ingress_priorities(deployment)
		deployment['ingress_priorities'] = ingress_priorities
		deployment['whole_deployment_ingress_priorities'] = copy.deepcopy(deployment['ingress_priorities'])

		link_capacities = get_link_capacities(deployment)
		deployment['link_capacities'] = link_capacities

	return deployment

problem_params = {
	'really_friggin_small': {
		'n_metro': 5,
		'n_asn': 15,
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
		'max_popp_per_ug': 5, 
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
	if 'actual' in problem_size:
		return load_actual_deployment(problem_size)
	else:
		return get_random_deployment_by_size(problem_size)

def get_random_deployment_by_size(problem_size):
	#### Extensions / todos: 

	print("----Creating Random Deployment-----")
	sizes = problem_params[problem_size]

	### Probably update this to be a slightly more interesting model later
	random_latency = lambda : np.random.uniform(1,10) #lambda : np.random.uniform(MIN_LATENCY, MAX_LATENCY)
	random_transit_provider_latency = lambda : np.random.uniform(3,10) #lambda : np.random.uniform(MIN_LATENCY*1.3, MAX_LATENCY)

	# testing ideas for learning over time
	pops = [str(el) for el in np.arange(0,sizes['n_pop'])]
	def random_loc():
		return (np.random.uniform(-30,30), np.random.uniform(-20,20))
	pop_to_loc = {pop:random_loc() for pop in pops}
	metros = np.arange(0,sizes['n_metro'])
	metro_loc = {metro:random_loc() for metro in metros}
	asns = np.arange(sizes['n_asn'])
	# ug_to_vol = {(metro,asn): np.power(2,np.random.uniform(1,10)) for metro in metros for asn in asns}
	# ug_to_vol = {(metro,asn): np.random.uniform(1,100) for metro in metros for asn in asns}
	ug_to_vol = {(metro,asn): 1 + 10 * np.random.random() for metro in metros for asn in asns}
	ug_perfs = {ug: {} for ug in ug_to_vol}
	peers = np.arange(0,sizes['n_peer'])
	popps = []
	n_providers = sizes['n_providers']
	for pop in pops:
		some_peers = np.random.choice(peers, size=np.random.randint(sizes['min_peerings_per_pop'],
			sizes['max_peerings_per_pop']), replace=False)
		provs = [p for p in some_peers if p < n_providers]
		if len(provs) == 0: # ensure at least one provider per pop
			some_peers = np.append(some_peers, [np.random.randint(n_providers)])
		for peer in some_peers:
			popps.append((str(pop),str(peer)))
	provider_popps = [popp for popp in popps if int(popp[1]) < n_providers]
	for ug in ug_to_vol:
		some_poppsi = np.random.choice(np.arange(len(popps)), size=np.random.randint(3,sizes['max_popp_per_ug']), replace=False)
		some_popps = [popps[i] for i in some_poppsi]
		sorted_dists = sorted(pops, key = lambda pop : geopy.distance.geodesic(pop_to_loc[pop], metro_loc[ug[0]]).km )
		for popp in some_popps:
			base_lat = [i for i,pop in enumerate(sorted_dists) if pop == popp[0]][0] * 10
			ug_perfs[ug][popp] = base_lat + random_latency()
		for popp in provider_popps:
			# All UGs have routes through deployment providers
			# Assume for now that relationships don't depend on the PoP
			# also assume these performances are probably worse
			base_lat = [i for i,pop in enumerate(sorted_dists) if pop == popp[0]][0] * 10
			ug_perfs[ug][popp] = base_lat + random_transit_provider_latency()
	ugs = list(ug_to_vol)
	ug_anycast_perfs = {ug:np.random.choice(list(ug_perfs[ug].values())) for ug in ugs}
		
	ug_to_ip = {ug:[str(i)] for i,ug in enumerate(ugs)}

	deployment = {
		'ugs': ugs,
		'ug_to_ip': ug_to_ip,
		'ug_perfs': ug_perfs,
		'ug_to_vol': ug_to_vol,
		'ug_anycast_perfs': ug_anycast_perfs,
		'whole_deployment_ugs': ugs,
		'whole_deployment_ug_perfs': ug_perfs,
		'whole_deployment_ug_to_vol': ug_to_vol,
		'popps': popps,
		'metro_loc': metro_loc,
		'pop_to_loc': pop_to_loc,
		'n_providers': n_providers,
		'provider_popps': provider_popps,
	}
	deployment['ingress_priorities'] = get_random_ingress_priorities(deployment)
	deployment['whole_deployment_ingress_priorities'] = copy.deepcopy(deployment['ingress_priorities'])
	deployment['link_capacities'] = get_link_capacities(deployment)

	print("----Done Creating Random Deployment-----")
	print("Deployment has {} users, {} popps, {} pops".format(
		len(ugs), len(popps), len(pop_to_loc)))

	return deployment

if __name__ == "__main__":
	export_interesting_measurements()
	# cluster_actual_users()

