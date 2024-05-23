import os, numpy as np, time, glob, json, copy, tqdm
# from advertisement_experiments import Advertisement_Experiments
from helpers import *
from constants import *

from peering_measurements.advertisement_experiments import Advertisement_Experiments
from Resilient_Advertisement.start_measurement import Ilgar_Prober

PROPAGATE_TIME = 60#60*10

TO_POP = False # True = write pops to file (testing), False = write ingresses to file (working)

CLIENTS_OF_INTEREST = ['170.39.226.128', '170.39.226.4']

def measurement_noise_factor():
	return 10 * np.random.uniform()

def is_interesting_ug(ug):
	return (ug[0], int(ug[1])) in UGS_OF_INTEREST

def is_deletable_ug(ug):
	return (ug[0], int(ug[1])) in UGS_TO_DELETE


def popps_to_hash(popps):
	popps = sorted(popps)
	popps_str = ",".join(["-".join(el) for el in popps])
	return popps_str

def hash_to_popps(popps_hash):
	popps = popps_hash.split(',')
	popps = [tuple(popp.split('-')) for popp in popps]
	return popps


class Advertisement_Experiments_Wrapper:
	### Wrapper class to interface with advertisement experiments and other things
	def __init__(self, system, mode, run_dir, deployment_info):
		self.ugs = copy.deepcopy(deployment_info['ugs'])
		self.ug_to_ip = copy.deepcopy(deployment_info['ug_to_ip'])
		self.ip_to_ug = {ip:ug for ug,ips in self.ug_to_ip.items() for ip in ips}
		self.popps = copy.deepcopy(deployment_info['popps'])
		self.calc_pop_to_popps()
		self.ug_perfs = copy.deepcopy(deployment_info['ug_perfs'])

		self.uid = 0 

		# self.conduct_measurements_to_prefix_popps = self.conduct_measurements_to_prefix_popps_fake
		self.conduct_measurements_to_prefix_popps = self.conduct_measurements_to_prefix_popps_ripe

		self.rau = RIPE_Atlas_Utilities(deployment_info['dpsize'])

		## Tracks whether each RIPE Atlas probe has at least one measurement that day
		self.has_had_measurement = {}
		## Tracks whether each RIPE Atlas probe has had a bad traceroute
		self.has_had_bad_traceroute = {}

		self.run_dir = run_dir

	def print_client_description(self, client):
		print("Client: {} ASN : {} UG: {}".format(client, self.ae.utils.parse_asn(client, with_siblings=False), self.ip_to_ug[client]))

	def ip_path_to_asn_path(self, ip_path):
		prev_hop = None
		as_path = []
		for hop in ip_path:
			if hop == "*": continue
			asn = self.ae.utils.parse_asn(hop, with_siblings=False)
			# asn = self.sibling_to_peer.get(asn,asn) # convert siblings of known peers to just those known peers
			if asn is None or asn == prev_hop: continue
			as_path.append(asn)
			prev_hop = asn
		return as_path

	def ip_path_to_asn_path_full(self, ip_path):
		as_path = []
		for hop in ip_path:
			if hop == "*": 
				as_path.append("*")
				continue
			asn = self.ae.utils.parse_asn(hop, with_siblings=False)
			# asn = self.sibling_to_peer.get(asn,asn) # convert siblings of known peers to just those known peers
			as_path.append(asn)
		return as_path

	def check_init_ae(self):
		try:
			self.ae
		except AttributeError:
			self.init_ae()

	def init_ae(self):
		self.ae = Advertisement_Experiments('vultr','null', quickinit=False)
		self.all_peers_in_vultr = list(set([peer for pop,peer in self.ae.popps]))
		self.ae.ilgar_prober = Ilgar_Prober()
		## ASNs corresponding to PEERING and VUltr
		self.IGNORE_ASNS = ['47065','61574', '61575', '61576', '263842', '263843', '263844', '33207', '20473']
		self.disallowed_asns = self.IGNORE_ASNS
		self.peer_to_siblings = {}
		peers = sorted(list(set(peer for pop,peer in self.popps)))
		## Lots of peers are siblings of each other, which can cause confusion when we try to see
		## which ingress something comes in on. So just tabulate all the possibilities
		for peeri in peers:
			orgi = self.ae.utils.parse_asn(peeri)
			for peerj in peers:
				if peeri==peerj: break
				orgj = self.ae.utils.parse_asn(peerj)
				if orgi == orgj:
					try:
						self.peer_to_siblings[peeri].append(peerj)
					except KeyError:
						self.peer_to_siblings[peeri] = [peerj]
					try:
						self.peer_to_siblings[peerj].append(peeri)
					except KeyError:
						self.peer_to_siblings[peerj] = [peeri]
		for peer,siblings in self.peer_to_siblings.items():
			self.peer_to_siblings[peer] = sorted(list(set(siblings)))

		# self.sibling_to_peer = {} ## easy for converting observed ASNs to our peers
		# for popp in self.ae.popps:
		# 	pop,peer = popp
		# 	org = self.ae.utils.parse_asn(peer)
		# 	for sibling_asn in self.ae.utils.org_to_as.get(org,[]):
		# 		self.sibling_to_peer[sibling_asn] = peer

	def add_popp_to_ae(self, popp):
		if DEBUG_CLIENT_INFO_ADDING:
			print("Adding PoPP: {} that we missed entirely to Advertisement Experiments data".format(popp))
		self.ae.add_popp(popp)
		self.init_ae()

	def add_ug_perf(self, ug, popp, latency, addcall=True):
		### Add performance to UG
		if DEBUG_CLIENT_INFO_ADDING:
			print("Adding {}->{} : {}ms".format(ug,popp,latency))
		if is_interesting_ug(ug):
			print("Adding {}->{} : {}ms".format(ug,popp,latency))

		self.ug_perfs[ug][popp] = latency
		if not popp in self.ae.popps:
			self.add_popp_to_ae(popp)
		pop,ingress_peer = popp
		with open(os.path.join(DATA_DIR, 'vultr_ingress_latencies_by_dst.csv'), 'a') as f2:
			## write -10000 in a field to indicate for the future tha this was a "filled in" measurement
			for client in self.ug_to_ip[ug]:
				f2.write("1709263120,{},{},{},-10000,{}\n".format(client, pop, ingress_peer, round(latency,4)))
		if addcall:
			self.calls_to_update_deployment.append(('add_ug_perf', ug, popp, latency))

	def del_ug_perf(self, ug, popp, addcall=True):
		if DEBUG_CLIENT_INFO_ADDING or is_interesting_ug(ug):
			print("Deleting {}->{}".format(ug,popp))
		del self.ug_perfs[ug][popp]
		if addcall:
			self.calls_to_update_deployment.append(('del_ug_perf', ug, popp))

		if len(self.ug_perfs[ug]) <= 1:
			if DEBUG_CLIENT_INFO_ADDING:
				print("Deleted {} for UG, now has <= 1 performances. Deleting UG".format(ug))
			self.del_ug(ug,addcall=Fale)

	def del_ug(self, ug, addcall=True):
		if DEBUG_CLIENT_INFO_ADDING or is_interesting_ug(ug):
			print("Deleting UG {}, clients: {}".format(ug, self.ug_to_ip[ug]))
		del self.ug_perfs[ug]
		if addcall:
			self.calls_to_update_deployment.append(('del_ug', ug))

	def del_client(self, client, addcall=True):
		if DEBUG_CLIENT_INFO_ADDING:
			print("Deleting client: {}".format(client))
		relevant_ugs = []
		if addcall:
			self.calls_to_update_deployment.append(('del_client', client))
		for ug,clients in self.ug_to_ip.items():
			if client in clients:
				relevant_ugs.append(ug)
		for this_ug in list(set(relevant_ugs)):
			self.ug_to_ip[this_ug] = get_difference(self.ug_to_ip[this_ug], [client])
			if len(self.ug_to_ip[this_ug]) == 0:
				self.del_ug(this_ug,addcall=False)

	def remove_probe(self, probe_id):
		if DEBUG_CLIENT_INFO_ADDING:
			print("Removing probe: {}".format(probe_id))		
		self.rau.ignore_probes.append(probe_id)
		self.rau.init_vars()
		
		for ug in list(self.ugs):
			for client in self.ug_to_ip.get(ug,[]):
				try:
					self.client_to_ripe_probe(client)
				except KeyError:
					self.del_client(client)

	def check_update_deployment_upon_modify(self):
		### When we modify UG perfs, it could be that many data structures need to change, check that these are changed
		self.ugs = sorted(list(self.ug_perfs))
		self.ug_to_ip = {ug:self.ug_to_ip[ug] for ug in self.ugs}
		self.ip_to_ug = {ip:ug for ug,ips in self.ug_to_ip.items() for ip in ips}
		self.popps = sorted(list(set(popp for ug in self.ug_perfs for popp in self.ug_perfs[ug])))
		self.calc_pop_to_popps()
		if TO_POP:
			new_ip = {}
			for ug in self.ugs:
				new_ip[ug] = {}
				popps = list(self.ug_perfs[ug])
				sorted_prefs = sorted(self.ingress_priorities[ug].items(), key = lambda el : el[1])
				priority_counter = 0
				for popp, priority in sorted_prefs:
					if popp in popps:
						new_ip[ug][popp] = priority_counter
						priority_counter += 1
			self.ingress_priorities = new_ip

	def calc_pop_to_popps(self):
		self.pop_to_popps = {}
		for pop,peer in self.popps:
			try:
				self.pop_to_popps[pop].append((pop,peer))
			except KeyError:
				self.pop_to_popps[pop] = [(pop,peer)]

	def infer_ingress(self, possible_popps, ug):
		"""No inference, just pretend we know them."""
		preferences = np.array([self.ingress_priorities[ug][popp] for popp in possible_popps])
		return possible_popps[np.argmin(preferences)]

	def conduct_measurements_to_prefix_popps_fake(self, prefix_popps, every_client_of_interest, 
		popp_lat_fn, **kwargs):
		### Idea is to mimic what actual measurements from Vultr would output, just for playing around
		### I.e., we conducted the measurements and these would be the outputs
		### the key is that the only output we know is the PoP, not the PoPP (toggleable setting (TO_POP))

		with open(popp_lat_fn, 'w') as f:
			for popps, client_set in zip(prefix_popps, every_client_of_interest):
				pref = "184.164.{}.0/24".format(self.uid%256)
				popps_str = "--".join(["-".join(popp)for popp in popps])
				f.write("{},{}\n".format(pref,popps_str))
				
				self.uid += 1
				for client in client_set:
					ug = self.ip_to_ug[client]
					possible_popps = get_intersection(popps, self.ug_perfs[ug])
					if len(possible_popps) == 0: continue
					winning_popp = self.infer_ingress(possible_popps,self.ip_to_ug[client])
					rtt = self.ug_perfs[self.ip_to_ug[client]][winning_popp] + measurement_noise_factor()
					if TO_POP:
						winning_pop = winning_popp[0]
						f.write("{},{},{},{},{},{}\n".format(pref_to_ip(pref),int(time.time()),
							client, winning_pop, 0, self.ug_perfs[self.ip_to_ug[client]][winning_popp]))
					else:
						f.write("{},{},{},{},{},{},{}\n".format(pref_to_ip(pref),int(time.time()),
							client, winning_popp[0], winning_popp[1], 0, self.ug_perfs[self.ip_to_ug[client]][winning_popp]))

	def client_to_ripe_probe(self, client_addr):
		return int(self.rau.atlas_24_to_probe_ids[ip32_to_24(client_addr)][0])

	def remove_dups(self, call_prefix_popps, call_every_client_of_interest):
		pruned_prefix_popps, pruned_every_client_of_interest = [], []
		seen_already = {}
		for pp, c in zip(call_prefix_popps, call_every_client_of_interest):
			pp_hash = popps_to_hash(pp)
			try:
				seen_already[pp_hash]
				continue
			except KeyError:
				pass
			pruned_prefix_popps.append(pp)
			pruned_every_client_of_interest.append(c)
			seen_already[pp_hash] = None
		return pruned_prefix_popps, pruned_every_client_of_interest

	def conduct_measurements_to_prefix_popps_ripe(self, call_prefix_popps, call_every_client_of_interest, 
		popp_lat_fn, **kwargs):
		### Conducts measurements by instrumenting RIPE Atlas probes

		prefix_popps, every_client_of_interest = self.remove_dups(call_prefix_popps, call_every_client_of_interest)

		
		self.check_init_ae()
		if kwargs.get('measurement_file_prefix', None) is not None:
			cache_fn = os.path.join(self.run_dir, "{}_ripe_results.pkl".format(kwargs.get('measurement_file_prefix')))
		else:
			cache_fn = os.path.join(self.run_dir, 'tmp_ripe_results-{}.pkl'.format(int(time.time())))

		if not (os.path.exists(popp_lat_fn) and os.path.exists(cache_fn)):
			atlas_probes = list([[self.client_to_ripe_probe(addr) for addr in sub_every_client_of_interest] for sub_every_client_of_interest in every_client_of_interest])
			total_n_probe_measurements = sum(len(probe_set) for probe_set in atlas_probes)
			print("About to conduct measurements using {} probes...".format(total_n_probe_measurements))
			time.sleep(20) # give me a chance to exit if this seems wrong


			# print(len(prefix_popps))
			# print("ABOUT TO MEASURE and store results in {}, {}".format(popp_lat_fn, cache_fn))
			# exit(0)
			ripe_results_info, ripe_results, pinger_results = self.ae.conduct_measurements_to_prefix_popps_ripe(prefix_popps, every_client_of_interest, atlas_probes,
				popp_lat_fn, **kwargs)
			pickle.dump([prefix_popps, every_client_of_interest, ripe_results_info, ripe_results, pinger_results], open(cache_fn,'wb'))

		self.calls_to_update_deployment = []
		self.parse_from_results_fn(popp_lat_fn, cache_fn)
		return self.calls_to_update_deployment
	
	def parse_from_results_fn(self, popp_lat_fn, cache_fn):
		prefix_popps, every_client_of_interest, ripe_results_info, ripe_results, pinger_results = pickle.load(open(cache_fn, 'rb'))		
		def get_rtts_by_pop(res_set):
			rtts_by_pop = {}
			for res in res_set:
				_pop = res[0]
				_rtt = res[1]
				try:
					rtts_by_pop[_pop].append(_rtt)
				except KeyError:
					rtts_by_pop[_pop] = [_rtt]
			return rtts_by_pop
		self.check_init_ae()
		with open(popp_lat_fn, 'w') as f:
			for popps, client_set, ripe_results_set, pinger_results_set in zip(prefix_popps, every_client_of_interest, ripe_results, pinger_results):
				if DEBUG_CLIENT_INFO_ADDING:
					print(popps)
				pref = "184.164.{}.0/24".format(self.uid%256)
				popps_str = "--".join(["-".join(popp)for popp in popps])
				f.write("{},{}\n".format(pref,popps_str))

				popps_with_siblings = list(sorted(list([(pop,sibling) for pop,peer in popps for sibling in self.peer_to_siblings.get(peer,[peer])])))

				popps_by_pop = {}
				for popp in popps:
					try:
						popps_by_pop[popp[0]].append(popp)
					except KeyError:
						popps_by_pop[popp[0]] = [popp]
				
				self.uid += 1
				for client in sorted(client_set):
					try:
						ug = self.ip_to_ug[client]
					except KeyError:
						# Since been removed as a client
						continue
					if is_deletable_ug(ug): #### TMP!!!#*&!^*&$^!*&#^*&!^#*&!*
						print("DELETING CLIENT FOR NO REASON --- SHOULD BE TEMPORARY")
						self.del_client(client)
						continue
					try:
						possible_popps = get_intersection(popps, self.ug_perfs[ug])
					except KeyError:
						# since been removed as a ug
						continue
					try:
						ripe_probe = self.client_to_ripe_probe(client)
					except KeyError:
						# since been deleted, continue
						continue
					try:
						ripe_trace_result = ripe_results_set[0][ripe_probe]
						self.has_had_measurement[ripe_probe] = None
					except KeyError:
						try:
							self.has_had_measurement[ripe_probe]
							## It's possibly active, so just guess based on the popp
							ripe_trace_result = ["*"]
							if is_interesting_ug(ug):
								print("Missed traceroute result for {} on {}, skipping...".format(ug, popps))

						except KeyError:
							# print("No results for probe {}, possibly remove".format(ripe_probe))
							self.remove_probe(ripe_probe)
							continue
					if client not in pinger_results_set:
						try:
							self.has_had_measurement[client]
							if is_interesting_ug(ug):
								print("Missed pinger result for {} on {}, skipping...".format(ug, popps))
							continue
						except KeyError:
							self.del_client(client)
							continue
					else:
						self.has_had_measurement[client] = None

					## add the client to the traceroute in case the initial address is private
					ripe_trace_result = [client] + ripe_trace_result

					### Parse the pop from the pinger results
					## todo -- vote on the most common pop
					try:
						pinger_results_set[client] = sorted(pinger_results_set[client], key = lambda el : el[2]) # sort by time
					except IndexError:
						# backward compatability, tmp
						pass
					rtts_by_pop = get_rtts_by_pop(pinger_results_set[client])
					if len(rtts_by_pop) == 1:
						pop = list(rtts_by_pop)[0]
						latency = parse_lat(np.min(rtts_by_pop[pop]))
					else:
						last_n = 3 ## last N packets have to be to the same pop
						rtts_by_pop = get_rtts_by_pop(pinger_results_set[client][-last_n:])
						if len(rtts_by_pop) == 1:
							pop = list(rtts_by_pop)[0]
							latency = parse_lat(np.min(rtts_by_pop[pop]))
						else:
							print("Oooooooops, not one PoP for client {}: {}.".format(client, pinger_results_set[client]))
							self.del_client(client)
							continue
					this_pop_possible_popps = list([popp for popp in possible_popps if popp[0] == pop])

					## First, try a less-strict way of parsing the peer
					confident_inference = False
					asn_path = self.ip_path_to_asn_path(ripe_trace_result)
					full_asn_path = self.ip_path_to_asn_path_full(ripe_trace_result)
					try:
						asn_path[0]
					except IndexError:
						try:
							self.has_had_bad_traceroute[ripe_probe]
							self.del_client(client) # too many bad traceroutes
						except KeyError:
							self.has_had_bad_traceroute[ripe_probe] = None # you get one chance
						if is_interesting_ug(ug):
							print("Bad traceroute result for {} on {}, skipping...".format(ug, popps))
						continue
					if asn_path[0] in self.disallowed_asns:
						self.del_client(client)
						continue
					ingress_peer = None
					try:
						prev_asn = asn_path[0]
						for asn in asn_path[1:]:
							## if the next hop is PEERING/Vultr and the previous hop was an ASN, take that ASN to be ingress peer
							if asn in self.IGNORE_ASNS and prev_asn not in ['None','*']:
								ingress_peer = prev_asn
								break
							prev_asn = asn
					except IndexError:
						pass
					if (pop, ingress_peer) not in popps_with_siblings: ## we wouldn't expect this peer, for whatever reason.  So be more strict
						## Convert Traceroute to peer ASN. We are very strict here, only allowing direct observable connections. Absolutely no unresponsive hops
						ingress_peer = None
						prev_asn = full_asn_path[0]
						for asn in full_asn_path[1:]:
							if asn in self.IGNORE_ASNS and prev_asn not in ['None','*',None]:
								ingress_peer = prev_asn
								confident_inference = True ### Two consecutive hops that we could map to an ASN, we're pretty confident here
								break
							elif asn in self.IGNORE_ASNS:
								break
							prev_asn = asn
					winning_popp = (pop, ingress_peer)

					if winning_popp not in popps and winning_popp in popps_with_siblings:
						## Need to convert to corresponding sibling we expect
						print("Converting {}".format(winning_popp))
						winning_org = self.ae.utils.parse_asn(winning_popp[1])
						winning_popp = list([popp for popp in popps_with_siblings if popp[0] == winning_popp[0] and self.ae.utils.parse_asn(popp[1]) == winning_org])[0]
						print("Converted to {}".format(winning_popp))

					if len(popps_by_pop[pop]) == 1 and not confident_inference:
						## If there's only one possible option, then we know the answer
						## This assumes that our BGP limiting works perfectly, but that's ok because I think it does 99.9% of the time
						winning_popp = popps_by_pop[pop][0]

					if winning_popp not in popps_with_siblings and winning_popp in self.ae.popps and confident_inference:
						# if winning_popp[1] not in ['199524', '137409']:
						# 	print("Note --- ingress peer was {}, but we shouldn't be advertising to them. You need to update BGP limiter.".format(
						# 		winning_popp))
						# 	print("Probe: {}, Client: {}, UG: {}".format(ripe_probe, client, ug))
						# 	for ip_hop,asn_hop in zip(ripe_trace_result, full_asn_path):
						# 		print("{}->{}".format(ip_hop,asn_hop))
						if len(get_intersection(popps,self.popps)) == len(self.popps):
							self.del_client(client)
						# else:
						# 	print("Annoying traceroute for ug {}".format(ug))
						if is_interesting_ug(ug):
							print("Weird oddly confident traceroute result for {} on {}, skipping...".format(ug, popps))
							print("Note --- ingress peer was {}, but we shouldn't be advertising to them. You need to update BGP limiter.".format(
								winning_popp))
							print("Probe: {}, Client: {}, UG: {}".format(ripe_probe, client, ug))
							for ip_hop,asn_hop in zip(ripe_trace_result, full_asn_path):
								print("{}->{}".format(ip_hop,asn_hop))
						continue

					# if is_interesting_ug(self.ip_to_ug[client]) and len(popps) > 90:
					# 	self.print_client_description(client)
					# 	print(winning_popp)
					# 	print(popps_by_pop[pop])
					# 	print(confident_inference)
					# 	for ip_hop,asn_hop in zip(ripe_trace_result, full_asn_path):
					# 		print("{}->{}".format(ip_hop,asn_hop))


					if len(this_pop_possible_popps) == 0:
						if winning_popp in self.ae.popps:
							### we probably just missed this one, so add it
							# if self.ip_to_ug[client] in UGS_OF_INTEREST:
							# 	self.print_client_description(client)
							# 	print(winning_popp)
							# 	print(popps_by_pop[pop])
							# 	for ip_hop,asn_hop in zip(ripe_trace_result, full_asn_path):
							# 		print("{}->{}".format(ip_hop,asn_hop))
							self.add_ug_perf(ug, winning_popp, latency)
						else:
							if confident_inference:
								# print("\n")
								# print("Note --- confident inference, we might just be missing a popp entirely: {}".format(winning_popp))
								# for ip_hop,asn_hop in zip(ripe_trace_result, full_asn_path):
								# 	print("{} -> {}".format(ip_hop,asn_hop))
								# print("\n")
								pass
							## Probably an uninformative traceroute. Just guess ingress link based on latency
							most_likely_peer = possible_popps[np.argmin([np.abs(self.ug_perfs[ug][popp] - latency) for popp in possible_popps])]
							winning_popp = most_likely_peer
					else:
						if winning_popp not in possible_popps: ## In the advertisement, but not in the user's performances
							if winning_popp in self.ae.popps:
								### we probably just missed this ingress for this user, so add it
								# print(winning_popp)
								# for ip_hop,asn_hop in zip(ripe_trace_result, full_asn_path):
								# 	print("{}->{}".format(ip_hop,asn_hop))
								# if self.ip_to_ug[client] in UGS_OF_INTEREST:
								# 	self.print_client_description(client)
								# 	print(winning_popp)
								# 	print(popps_by_pop[pop])
								# 	for ip_hop,asn_hop in zip(ripe_trace_result, full_asn_path):
								# 		print("{}->{}".format(ip_hop,asn_hop))
								self.add_ug_perf(ug, winning_popp, latency)
							else:
								if confident_inference:
									# print("\n")
									# print("Note --- confident inference, we might just be missing a popp entirely: {}".format(winning_popp))
									# for ip_hop,asn_hop in zip(ripe_trace_result, full_asn_path):
									# 	print("{} -> {}".format(ip_hop,asn_hop))
									# print("\n")
									pass
								## Probably an uninformative traceroute. Just guess ingress link based on latency
								most_likely_peer = this_pop_possible_popps[np.argmin([np.abs(self.ug_perfs[ug][popp] - latency) for popp in this_pop_possible_popps])]
								winning_popp = most_likely_peer
						else:
							## perfect scenario
							pass
					f.write("{},{},{},{},{},{},{}\n".format(pref_to_ip(pref),int(time.time()),
						client, winning_popp[0], winning_popp[1], 0, self.ug_perfs[self.ip_to_ug[client]][winning_popp]))
		self.check_update_deployment_upon_modify()
		return self.calls_to_update_deployment

	def check_info_consistent(self):
		self.calls_to_update_deployment = []
		for ug in list(self.ug_perfs):
			for popp in list(self.ug_perfs[ug]):
				try:
					self.ae.popps[popp]
				except KeyError:
					try:
						self.del_ug_perf(ug,popp)
						print("Deleted {}->{} because it doesnt exist as a PoPP".format(ug,popp))
					except:
						pass
		
		return self.calls_to_update_deployment

	def load_all_info(self, runfile, tmpoutfn):
		self.calls_to_update_deployment = []
		
		# print("Loading from runfile: {}".format(runfile))
		self.parse_from_results_fn(tmpoutfn, runfile)

		return self.calls_to_update_deployment


class RealWorld_Measure_Wrapper:
	#### Actually measure prefix advertisements in the wild, as opposed to simulating them
	def __init__(self, run_dir, deployment_info, **kwargs):
		self.dpsize = deployment_info['dpsize']
		self.run_dir = run_dir
		if self.run_dir is not None:
			if not os.path.exists(self.run_dir):
				from subprocess import call
				call("mkdir {}".format(self.run_dir), shell=True)
		self.ugs = sorted(copy.deepcopy(deployment_info['ugs']))
		self.n_ug = len(self.ugs)
		self.ug_to_ind = {ug:i for i,ug in enumerate(self.ugs)}
		self.ug_to_ip = copy.deepcopy(deployment_info['ug_to_ip'])
		self.ip_to_ug = {ip:ug for ug,ips in self.ug_to_ip.items() for ip in ips}
		self.popps = copy.deepcopy(sorted(deployment_info['popps']))
		self.popp_to_ind = {popp:i for i,popp in enumerate(self.popps)}
		self.calc_pop_to_popps()
		self.ug_perfs = copy.deepcopy(deployment_info['ug_perfs'])

		self.calc_popp_to_clients()

		## Stores information about which PoPPs beat other PoPPs for each user
		self.measured_prefs = {ug: {popp: Ing_Obj(popp) for popp in \
			self.ug_perfs[ug]} for ug in self.ugs}
		self.prefix_popps_to_catchments = {} ## simple caching mechanism

		self.past_rundirs = kwargs.get("past_rundirs", [])


	def calc_popp_to_clients(self):
		self.popp_to_clients = {}
		for ug in self.ug_perfs:
			for popp in self.ug_perfs[ug]:
				for client in self.ug_to_ip[ug]:
					try:
						self.popp_to_clients[popp].append(client)
					except KeyError:
						self.popp_to_clients[popp] = [client]

	def calc_pop_to_popps(self):
		self.pop_to_popps = {}
		for pop,peer in self.popps:
			try:
				self.pop_to_popps[pop].append((pop,peer))
			except KeyError:
				self.pop_to_popps[pop] = [(pop,peer)]

	def add_ug_perf(self, ug, popp, latency, addcall=True):
		### Add performance to UG
		if DEBUG_CLIENT_INFO_ADDING:
			print("RWMW, Adding {}->{}:{}".format(ug,popp,round(latency)))
		self.ug_perfs[ug][popp] = latency
		self.measured_prefs[ug][popp] = Ing_Obj(popp)
		if addcall:
			self.calls_to_update_deployment.append(('add_ug_perf', ug, popp, latency))
		# self.check_update_deployment_upon_modify()

	def del_ug_perf(self, ug, popp, addcall=True):
		if DEBUG_CLIENT_INFO_ADDING:
			print("RWMW, Deleting {}->{}".format(ug,popp))
		del self.ug_perfs[ug][popp]
		del self.measured_prefs[ug][popp]
		if addcall:
			self.calls_to_update_deployment.append(('del_ug_perf', ug, popp))

		if len(self.ug_perfs[ug]) <= 1:
			self.del_ug(ug,addcall=False)
		# self.check_update_deployment_upon_modify()

	def del_ug(self, ug, addcall=True):
		if DEBUG_CLIENT_INFO_ADDING or is_interesting_ug(ug):
			print("RWMW, Deleting {}".format(ug))
		del self.ug_perfs[ug]
		del self.measured_prefs[ug]
		if addcall:
			self.calls_to_update_deployment.append(('del_ug', ug))
		# self.check_update_deployment_upon_modify()

	def del_client(self, client, addcall=True):
		if DEBUG_CLIENT_INFO_ADDING:
			print("RWMW, Deleting client {}".format(client))
		relevant_ugs = []
		if addcall:
			self.calls_to_update_deployment.append(('del_client', client))
		for ug,clients in self.ug_to_ip.items():
			if client in clients:
				relevant_ugs.append(ug)
		for this_ug in list(set(relevant_ugs)):
			self.ug_to_ip[this_ug] = get_difference(self.ug_to_ip[this_ug], [client])
			if len(self.ug_to_ip[this_ug]) == 0:
				self.del_ug(this_ug,addcall=False)

	def check_update_deployment_upon_modify(self):
		### When we modify UG perfs, it could be that many data structures need to change, check that these are changed
		self.ugs = sorted(list(self.ug_perfs))
		self.n_ug = len(self.ugs)
		self.ug_to_ind = {ug:i for i,ug in enumerate(self.ugs)}
		self.ug_to_ip = {ug:self.ug_to_ip[ug] for ug in self.ugs}
		self.ip_to_ug = {ip:ug for ug,ips in self.ug_to_ip.items() for ip in ips}
		self.popps = sorted(list(set(popp for ug in self.ug_perfs for popp in self.ug_perfs[ug])))
		self.popp_to_ind = {popp:i for i,popp in enumerate(self.popps)}
		self.calc_pop_to_popps()
		if TO_POP:
			new_ip = {}
			for ug in self.ugs:
				new_ip[ug] = {}
				popps = list(self.ug_perfs[ug])
				sorted_prefs = sorted(self.ingress_priorities[ug].items(), key = lambda el : el[1])
				priority_counter = 0
				for popp, priority in sorted_prefs:
					if popp in popps:
						new_ip[ug][popp] = priority_counter
						priority_counter += 1
				for unincluded_popp in get_difference(popps, new_ip[ug]):
					new_ip[ug][unincluded_popp] = priority_counter
					priority_counter += 1
			self.ingress_priorities = new_ip


		new_mp = {}
		for ug in self.ug_perfs:
			new_mp[ug] = {}
			for popp,obj in self.measured_prefs.get(ug, {}).items():
				if popp not in self.ug_perfs[ug]:
					continue
				obj.parents = {prnt: obj.parents[prnt] for prnt in get_intersection(self.ug_perfs[ug], obj.parents)}
				obj.children = {child: obj.children[child] for child in get_intersection(self.ug_perfs[ug], obj.children)}
				new_mp[ug][popp] = obj
			for popp in get_difference(self.ug_perfs[ug], self.measured_prefs.get(ug,{})):
				new_mp[ug][popp] = Ing_Obj(popp)
		self.measured_prefs = new_mp

		## Possibly update our results cache to new keys if we've removed popps
		for popps_hash in list(self.prefix_popps_to_catchments):
			popps = hash_to_popps(popps_hash)
			new_popps = get_intersection(self.popps, popps)
			new_popps_hash = popps_to_hash(new_popps)
			if new_popps_hash != popps_hash:
				self.prefix_popps_to_catchments[new_popps_hash] = self.prefix_popps_to_catchments[popps_hash]
				del self.prefix_popps_to_catchments[popps_hash]

		self.calc_pop_to_popps()
		self.calc_popp_to_clients()

	def get_catchments(self, popps):
		### Get's UG catchments assuming we advertise popps, or returns None if we don't exactly know the answer
		popps_hash = popps_to_hash(popps)
		try:
			routed_through_ingress = self.prefix_popps_to_catchments[popps_hash]
			for ug in self.ugs:
				if routed_through_ingress.get(ug) is None:
					print("Cached, in get catchemnts, zero possible options for {}".format(ug))
		except KeyError:
			routed_through_ingress = {}
			for ug in self.ugs:
				possible_peers = self.limit_potential_popps(popps, ug)
				if len(possible_peers) > 1:
					return None
				elif len(possible_peers) == 1:
					routed_through_ingress[ug] = possible_peers[0]
				else:
					print("Not cached, in get catchments, zero possible options for {}".format(ug))
		if len(routed_through_ingress) > 0:
			return routed_through_ingress
		else:
			return None

	def limit_potential_popps(self, advertised_popps, ug):
		## Remove impossible options based on our known information
		# print(available_popp_inds)

		advertised_and_has_route = get_intersection(self.ug_perfs[ug], advertised_popps)
		if len(advertised_and_has_route) == 1: ## only 1 possibility
			return advertised_and_has_route

		for i,popp in enumerate(sorted(advertised_and_has_route)):
			node = self.measured_prefs[ug][popp]
			if node.has_parent(advertised_popps):
				node.kill()
			check_all_possible_popps = list([_popp for _popp, node in self.measured_prefs[ug].items() if node.alive and _popp in advertised_popps])
			if len(check_all_possible_popps) == 1:
				break
		all_possible_popps = [popp for popp, node in self.measured_prefs[ug].items() if node.alive]
		for popp in list(self.measured_prefs[ug]):
			self.measured_prefs[ug][popp].alive = True
		intersect_possible_popps = get_intersection(all_possible_popps, advertised_popps)

		# if is_interesting_ug(ug):
		# 	print("Advertised popps: {}, \nhas route and advertised popps: {}, \nacccounting for preferences popps: {}".format(
		# 		advertised_popps, advertised_and_has_route, intersect_possible_popps))
		# 	print('\n')


		return intersect_possible_popps

	def infer_ingress(self, possible_popps, ug, rtt, method='performance'):
		"""
		Ideally we would know the ingress, but we don't. 
		Infer the ingress as the one closest in latency.
		And then other pseudo-methods for testing purposes.
		"""

		possible_popps = self.limit_potential_popps(possible_popps, ug)
		these_perfs = np.array([self.ug_perfs[ug][popp] for popp in possible_popps])
		if len(possible_popps) == 1:
			return possible_popps[0]
		if len(possible_popps) == 0:
			### Likely an incongruence in our logic which needs to be fixed (would throw an error)
			pass

		if method == 'performance':
			ui = self.ug_to_ind[ug]
			return possible_popps[np.argmin(np.abs(these_perfs - rtt))]
		elif method == 'use_ground_truth':
			### just for testing purposes
			preferences = np.array([self.ingress_priorities[ug][popp] for popp in possible_popps])
			return possible_popps[np.argmin(preferences)]
		else:
			raise ValueError("Method {} for infering ingress not implemented".format(method))

	def check_load_ae(self):
		try:
			self.ae
		except AttributeError:
			## Object to interface with executing advertisements
			deployment_info = {
				'ugs': self.ugs,
				'ug_to_ip': self.ug_to_ip,
				'popps': self.popps,
				'ug_perfs': self.ug_perfs,
				'dpsize': self.dpsize,
			}
			self.ae = Advertisement_Experiments_Wrapper('vultr', 'null', self.run_dir, deployment_info)

	def check_need_measure(self, advs):
		self.check_load_ae()

		self.check_update_deployment_upon_modify()

		filtered_advs = []
		for prefix_i, all_available_options in advs:
			need_meas = False
			try:
				self.prefix_popps_to_catchments[popps_to_hash(all_available_options)]
			except KeyError:
				for ug in self.ugs:
					possible_peers = self.limit_potential_popps(all_available_options, ug)
					if len(possible_peers) > 1:
						need_meas = True
						break
			if need_meas:
				filtered_advs.append((prefix_i, all_available_options))
		return filtered_advs

	def measure_advs(self, advs, **kwargs):
		#### advs is a list of (prefix_i, integer) tuples, each integer tuple corresponds to a PoPP tuple
		#### return prefix_i -> ug -> PoPP for all relevant UGs

		self.check_load_ae()

		self.check_update_deployment_upon_modify()

		routed_through_ingress = {}
		self.calls_to_update_deployment = []

		if kwargs.get('measurement_file_prefix', None) is not None:
			popp_lat_fn = os.path.join(self.run_dir, '{}_measure_output.csv'.format(kwargs.get('measurement_file_prefix')))
			verb=True
		else:
			popp_lat_fn = os.path.join(self.run_dir, 'measure-output-{}.csv'.format(
				int(time.time())))
			verb=False

		### First, filter advs to remove any cases for which we already know the answer
		### We might already know the answer if it's a singleton advertisement
		### or if we've measured everything there is to know in a past run
		filtered_advs = []
		if DEBUG_CLIENT_INFO_ADDING:
			print(list(self.prefix_popps_to_catchments))
		ugs_need_meas_by_prefix = {}
		for prefix_i, all_available_options in advs:
			routed_through_ingress[prefix_i] = {}
			ugs_need_meas_by_prefix[prefix_i] = {}
			need_meas = False
			if DEBUG_CLIENT_INFO_ADDING:
				print(popps_to_hash(all_available_options))
			try:
				routed_through_ingress[prefix_i] = self.prefix_popps_to_catchments[popps_to_hash(all_available_options)]
				for ug in self.ugs:
					if routed_through_ingress[prefix_i].get(ug) is None:
						possible_peers = self.limit_potential_popps(all_available_options, ug)
						if len(possible_peers) == 1:
							routed_through_ingress[prefix_i][ug] = possible_peers[0]
					if is_interesting_ug(ug):
						print("Pulled from cache for {} ( {} available ): {}".format(all_available_options,
							get_intersection(all_available_options, self.ug_perfs[ug]), routed_through_ingress[prefix_i].get(ug)))
			except KeyError:
				if DEBUG_CLIENT_INFO_ADDING or verb:
					dists = {}
					for existing_popps_hash in self.prefix_popps_to_catchments:
						amb = get_difference(hash_to_popps(existing_popps_hash),all_available_options)
						bma = get_difference(all_available_options, hash_to_popps(existing_popps_hash))
						dists[existing_popps_hash] = len(amb) + len(bma)
					sdists = sorted(dists.items(), key = lambda el : el[1])
					closest_option,_ = sdists[0]
					amb = get_difference(hash_to_popps(closest_option),all_available_options)
					bma = get_difference(all_available_options, hash_to_popps(closest_option))
					print("Popps: {}".format(all_available_options))
					print("Closest option difference A-B: {} B-A: {}".format(amb,bma))
					print("Not in cache...")
				for ug in self.ugs:
					possible_peers = self.limit_potential_popps(all_available_options, ug)
					if len(possible_peers) > 1:
						need_meas = True
						ugs_need_meas_by_prefix[prefix_i][ug] = None
					elif len(possible_peers) == 1:
						routed_through_ingress[prefix_i][ug] = possible_peers[0]
			# print(ugs_need_meas_by_prefix[prefix_i])
			if need_meas:
				filtered_advs.append((prefix_i, all_available_options))

		if len(filtered_advs) == 0:
			## we knew everything
			return routed_through_ingress, self.calls_to_update_deployment

		### Measure!!
		prefix_popps = []
		prefix_to_popps = {}
		every_client_of_interest = []
		for prefix_i, all_available_options in filtered_advs:
			prefix_popps.append(all_available_options)
			prefix_to_popps[prefix_i] = all_available_options

			these_clients = set()
			for ug in ugs_need_meas_by_prefix[prefix_i]:
				these_clients = these_clients.union(set(self.ug_to_ip[ug]))
			every_client_of_interest.append(these_clients)
		
		calls_to_update_deployment = self.ae.conduct_measurements_to_prefix_popps(prefix_popps, every_client_of_interest, 
			popp_lat_fn, using_manual_clients = True, propagate_time = PROPAGATE_TIME,
			logcomplete = False, **kwargs)

		self.check_update_calls_to_update_deployment(calls_to_update_deployment)


		## Read results and update our model
		this_ug_to_popp = self.load_and_add_info(popp_lat_fn)

		## Now load the answers from cache
		for prefix_i, all_available_options in filtered_advs:
			ptoh = popps_to_hash(all_available_options)
			try:
				routed_through_ingress[prefix_i]
			except KeyError:
				routed_through_ingress[prefix_i] = {}
			if DEBUG_CLIENT_INFO_ADDING or verb:
				dists = {}
				for existing_popps_hash in self.prefix_popps_to_catchments:
					amb = get_difference(hash_to_popps(existing_popps_hash),all_available_options)
					bma = get_difference(all_available_options, hash_to_popps(existing_popps_hash))
					dists[existing_popps_hash] = len(amb) + len(bma)
				sdists = sorted(dists.items(), key = lambda el : el[1])
				closest_option,_ = sdists[0]
				amb = get_difference(hash_to_popps(closest_option),all_available_options)
				bma = get_difference(all_available_options, hash_to_popps(closest_option))
				print("Popps: {}".format(all_available_options))
				print("Closest option difference A-B: {} B-A: {}".format(amb,bma))
				print("Not in cache...")
			for ug,popp in self.prefix_popps_to_catchments.get(ptoh, {}).items():
				routed_through_ingress[prefix_i][ug] = popp
			## Make sure the cash is also up to date on the users
			for ug,popp in routed_through_ingress[prefix_i].items():
				try:
					self.prefix_popps_to_catchments[ptoh]
				except KeyError:
					self.prefix_popps_to_catchments[ptoh] = {}
				self.prefix_popps_to_catchments[ptoh][ug] = popp
		
		self.check_update_deployment_upon_modify()
		return routed_through_ingress, self.calls_to_update_deployment

	def check_update_calls_to_update_deployment(self, calls_to_update_deployment):
		for call in calls_to_update_deployment:
			args = call[1:]
			getattr(self, call[0])(*args)

	def load_and_add_info(self, popp_lat_fn):
		ret = {} ## output is global poppset -> ug -> popp
		uid_to_popps = {}
		pref_to_uid = {}
		tmp_parse_by_ug = {}
		uid = 0
		### Iterate twice through the files. Once to group things by UG (as opposed to)
		### destination IP, and once to determine the winning ingress per UG (voting)
		for row in open(popp_lat_fn, 'r'):
			fields = row.strip().split(',')
			if len(fields) == 2:
				pref,popps_str = row.strip().split(',')
				popps = sorted([tuple(popp.split('-')) for popp in popps_str.split('--')])
				uid += 1
				pref_to_uid[pref_to_ip(pref)] = uid
				uid_to_popps[uid] = popps

				try:
					tmp_parse_by_ug[tuple(sorted(popps))]
				except KeyError:
					tmp_parse_by_ug[tuple(sorted(popps))] = {}
			else:
				if TO_POP:
					pref,t_meas,client_dst,dst_pop,_,rtt = fields
				else:
					pref,t_meas,client_dst,dst_pop,dst_peer,_,rtt = fields
				rtt = float(rtt)

				try:
					ug = self.ip_to_ug[client_dst]
				except KeyError:
					continue
				uid = pref_to_uid[pref]
				popps_this_experiment = uid_to_popps[uid]
				if TO_POP:
					## Possible ingresses given the PoP it came in at
					popps_this_pop = get_intersection(popps_this_experiment, self.pop_to_popps[dst_pop])
				else:
					popps_this_pop = [(dst_pop,dst_peer)]
					if (dst_pop,dst_peer) not in self.ug_perfs[ug]:
						self.add_ug_perf(ug, (dst_pop,dst_peer), rtt)
				## Possible ingresses given the user
				possible_popps = get_intersection(self.ug_perfs[ug], popps_this_pop)
				try:
					tmp_parse_by_ug[tuple(sorted(popps_this_experiment))][ug].append((possible_popps, rtt))
				except KeyError:
					tmp_parse_by_ug[tuple(sorted(popps_this_experiment))][ug] = [(possible_popps, rtt)]

		for global_popp_set in tmp_parse_by_ug:

			# ## TMP CHECK
			# for ug in self.ugs:
			# 	if is_interesting_ug(ug):
			# 		has_route_and_advertised = get_intersection(self.ug_perfs[ug], global_popp_set)
			# 		if len(has_route_and_advertised) > 0:
			# 			possible_options = self.limit_potential_popps(global_popp_set, ug)
			# 			if len(possible_options) > 1:
			# 				try:
			# 					tmp_parse_by_ug[global_popp_set][ug]
			# 				except KeyError:
			# 						print(popp_lat_fn)
			# 						print("UG {} no measured route to {} but should be measuring !!!".format(ug, global_popp_set))
			# 						exit(0)	
			# 			elif len(possible_options) == 1:
			# 				# easy
			# 				continue
			# 			else:
			# 				print("UG {} Nonzero options for {} ({}), but 0 in possible options...".format(ug,
			# 					global_popp_set,  has_route_and_advertised))


			ret[global_popp_set] = {}
			for ug, this_ug_results in tmp_parse_by_ug[global_popp_set].items():
				## Due to our clustering, it's possible that two clients have different routing
				## so do a vote on the winner
				### TODO -- if this causes problems, we might want to delete clients that don't win the vote
				## but hopefully these cases are rare
				
				n_by_res_set, rtt_by_res_set = {}, {}
				for popps,rtt in this_ug_results:
					try:
						n_by_res_set[tuple(sorted(popps))] += 1
						rtt_by_res_set[tuple(sorted(popps))] = np.minimum(rtt_by_res_set[tuple(sorted(popps))], rtt)
					except KeyError:
						n_by_res_set[tuple(sorted(popps))] = 1
						rtt_by_res_set[tuple(sorted(popps))] = rtt
				possible_popps = sorted(n_by_res_set.items(), key = lambda el : -1 * el[1])[0][0]
				rtt = rtt_by_res_set[possible_popps]
				possible_popps = list(possible_popps)

				## Infer winning PoPP out of the set of valid PoPPs
				### If wrong information, options seem to be:
				### (a) throw an error here, and infer on the global set (ignoring the PoP it came in on)
				### this option effectively makes us stick with incorrect information
				### (b) greedily search for a fix to the problem by deleting information, backpropagate that to the main thread
				### could also just delete all information about the user

				## if winning popp here not in potential winning popps on full set, that's how we know we have an incongruence
				##
				winning_popp = self.infer_ingress(possible_popps, ug, rtt)
				ret[global_popp_set][ug] = winning_popp

				# if is_interesting_ug(ug):
				# 	print("{} {} {} {}".format(self.ug_to_ip[ug], len(global_popp_set), get_intersection(global_popp_set,self.ug_perfs[ug]), winning_popp))

				### Update the model
				routed_ingress_obj = self.measured_prefs[ug].get(winning_popp)
				this_ug_all_possible_popps = get_intersection(self.ug_perfs[ug], global_popp_set)
				for beaten_ingress in get_difference(this_ug_all_possible_popps, [winning_popp]):
					beaten_ingress_obj = self.measured_prefs[ug].get(beaten_ingress)
					beaten_ingress_obj.add_parent(routed_ingress_obj)
					routed_ingress_obj.add_child(beaten_ingress_obj)
			
			# # cache for later
			# if np.abs(len(global_popp_set) - len(self.popps)) < 10:
			# 	print(ret[global_popp_set])
			ptoh = popps_to_hash(global_popp_set)
			try:
				self.prefix_popps_to_catchments[ptoh]
			except KeyError:
				self.prefix_popps_to_catchments[ptoh] = {}
			for ug in ret[global_popp_set]:
				self.prefix_popps_to_catchments[ptoh][ug] = ret[global_popp_set][ug]

			# for ug in self.ugs:
			# 	if is_interesting_ug(ug):
			# 		possible_options = self.limit_potential_popps(global_popp_set, ug)
			# 		print("In here, {} routable and {} possible options".format(len(get_intersection(self.ug_perfs[ug], global_popp_set)), len(possible_options)))
			# 		if len(possible_options) == 1:
			# 			print(possible_options)
			# 			print(self.prefix_popps_to_catchments[popps_to_hash(global_popp_set)].get(ug))
			# 		if len(possible_options) > 1:
			# 			if self.prefix_popps_to_catchments[popps_to_hash(global_popp_set)].get(ug) is None:
			# 				print("{} {} {}".format(len(global_popp_set), global_popp_set, possible_options))
			# 				print("{} available but None stored".format(possible_options))
			# 				exit(0)


		return ret

	def reload_info(self, **kwargs):
		### Bootstrap information about priorities that we know from previous experiments / measurements
		
		self.calls_to_update_deployment = []
		uid = 0 
		actives = {}
		routed_through_ingress = {}
		all_calls_update_deployment = []
		called_ae = False

		match_patterns = kwargs.get('match_file_patterns', ['tmp_ripe_results*.pkl'])

		for rundir in self.past_rundirs:
			self.check_load_ae()
			for match_pattern in match_patterns:
				these_runfiles = sorted(glob.glob(os.path.join(rundir, match_pattern)))
				print("Loading information from {} runfiles in {}".format(len(these_runfiles), rundir))
				for runfile in these_runfiles:
					all_calls_update_deployment = all_calls_update_deployment + self.ae.load_all_info(runfile, 'tmp.txt')
					called_ae = True

					this_ug_to_popp = self.load_and_add_info('tmp.txt')
					for global_poppset, ret in this_ug_to_popp.items():
						routed_through_ingress[uid] = ret
						actives[uid] = global_poppset
						uid += 1

		if called_ae:
			self.check_update_calls_to_update_deployment(all_calls_update_deployment)
			self.check_update_deployment_upon_modify()

			calls_to_update_deployment = self.ae.check_info_consistent()
			self.check_update_calls_to_update_deployment(calls_to_update_deployment)
			self.check_update_deployment_upon_modify()


		return routed_through_ingress, actives, self.calls_to_update_deployment

class RIPE_Atlas_Utilities:
	def __init__(self, dpsize):
		self.deployment_of_interest = dpsize
		self.ignore_probes = []
		self.init_vars()


	def init_vars(self):
		self.atlas_24s = {}
		self.atlas_addr_to_probeid = {}
		self.atlas_24_to_country = {}
		self.atlas_24_to_asn = {}
		self.atlas_24_to_probe_ids = {}


		if self.deployment_of_interest == 'actual_second_prototype':
			probes_fn = 'active_probes_20240331.json'
		elif self.deployment_of_interest == 'actual_third_prototype':
			probes_fn = 'active_probes_20240426.json'

		### From here: https://ftp.ripe.net/ripe/atlas/probes/archive
		for row in json.load(open(os.path.join(CACHE_DIR, probes_fn),'r'))['objects']:
			if row['address_v4'] is None or row['status_name'] != "Connected": continue

			probe_id = int(row['id'])
			addr = row['address_v4']
			country = row['country_code']
			asn = row['asn_v4']

			if probe_id in self.ignore_probes: continue

			self.atlas_24s[ip32_to_24(addr)] = None
			self.atlas_addr_to_probeid[addr] = probe_id
			try:
				self.atlas_24_to_probe_ids[ip32_to_24(addr)].append(probe_id)
			except KeyError:
				self.atlas_24_to_probe_ids[ip32_to_24(addr)] = [probe_id]
			self.atlas_24_to_country[ip32_to_24(addr)] = country
			self.atlas_24_to_asn[ip32_to_24(addr)] = asn

	def load_probe_perfs(self, **kwargs):
		cache_fn = os.path.join(CACHE_DIR, 'ripe_atlas_perfs_{}.pkl'.format(self.deployment_of_interest))
		if not os.path.exists(cache_fn):

			pingable_targets = {}
			pingable_24s = {}
			import tqdm
			for row in tqdm.tqdm(open(os.path.join("..", "peering_measurements", "data", "addresses_that_respond_to_ping.csv"),'r'), 
				desc="Reading pingable addresses..."):
				addr,responds = row.strip().split(',')
				if responds == '0': continue
				addr_24 = ip32_to_24(addr)
				try:
					self.atlas_24s[addr_24]
				except KeyError:
					continue
				pingable_targets[addr] = None
				pingable_24s[addr_24] = None
			print("Of {} Atlas /24s, {} have a pingable target, {} addresses".format(len(self.atlas_24s), len(pingable_24s), len(pingable_targets)))

			these_addresses_perfs = {}
			for row in tqdm.tqdm(open(os.path.join(CACHE_DIR, 'vultr_ingress_latencies_by_dst.csv'), 'r'),
				desc="Reading Vultr measurements..."):
				fields = row.strip().split(',')
				t,ip,pop,peer,_,lat = fields
				try:
					pingable_targets[ip]
				except KeyError:
					continue
				lat = parse_lat(lat)
				try:
					these_addresses_perfs[ip][pop,peer] = lat
				except KeyError:
					these_addresses_perfs[ip] = {(pop,peer) : lat}

			these_addresses_anycast_perfs = {}
			for row in open(os.path.join(CACHE_DIR, 'vultr_anycast_latency_smaller.csv')):
				_,ip,lat,_ = row.strip().split(',')
				try:
					these_addresses_perfs[ip]
				except KeyError:
					continue
				these_addresses_anycast_perfs[ip] = parse_lat(lat)

			pickle.dump([these_addresses_perfs, these_addresses_anycast_perfs], open(cache_fn,'wb'))
		else:
			these_addresses_perfs, these_addresses_anycast_perfs = pickle.load(open(cache_fn, 'rb'))
		all_n_ingresses = list([len(v) for v in these_addresses_perfs.values()])

		#### To get a sense of how many it should have
		# x,cdf_x = get_cdf_xy(all_n_ingresses)
		# plt.plot(x,cdf_x)
		# plt.savefig('tmp.pdf')

		include_pops = CONSIDERING_POPS_ACTUAL_DEPLOYMENT[self.deployment_of_interest]
		pruned_these_addresses_perfs = {}
		already_used = {}
		for k,popps in these_addresses_perfs.items():
			try:
				these_addresses_anycast_perfs[k]
			except KeyError:
				continue
			country = self.atlas_24_to_country[ip32_to_24(k)]
			asn = self.atlas_24_to_asn[ip32_to_24(k)]
			if country != 'US':
				try:
					already_used[country,asn]
					continue
				except KeyError:
					pass
			for popp in popps:
				if popp[0] in include_pops:
					ug = (country,asn,k)
					try:
						pruned_these_addresses_perfs[ug][popp] = these_addresses_perfs[k][popp]
					except KeyError:
						pruned_these_addresses_perfs[ug] = {popp:these_addresses_perfs[k][popp]}
						already_used[country,asn] = None

		cutoff_n = 25 ## needs enough measurements
		pruned_these_addresses_perfs = {k:v for k,v in pruned_these_addresses_perfs.items() if len(v) > cutoff_n}
		min_lat = 15 ## has to be near enough
		pruned_these_addresses_perfs = {k:v for k,v in pruned_these_addresses_perfs.items() if min(list(pruned_these_addresses_perfs[k].values())) < min_lat}
		for k,v in pruned_these_addresses_perfs.items():
			print("{} (Probe ID {}) -- {}".format(k,self.atlas_24_to_probe_ids[ip32_to_24(k[2])], v))
			if np.random.random() > .99: break
		print("{} interesting probes".format(len(pruned_these_addresses_perfs)))

		for ug in list(pruned_these_addresses_perfs):
			if len(pruned_these_addresses_perfs[ug]) <= 1:
				del pruned_these_addresses_perfs[ug]
		print("{} interesting probes after removing probes with 1 measurement".format(len(pruned_these_addresses_perfs)))


		asns = list(set(asn for country,asn,_ in pruned_these_addresses_perfs))
		countries = list(set(country for country,asn,_ in pruned_these_addresses_perfs))
		# Probes are in 1992 ASNs and 121 countries
		print("Probes are in {} ASNs and {} countries, have measurements to {} ingresses".format(len(asns), len(countries),
			len(set(popp for ug in pruned_these_addresses_perfs for popp in pruned_these_addresses_perfs[ug]))))

		these_addresses_anycast_perfs = {k:these_addresses_anycast_perfs[k[2]] for k in pruned_these_addresses_perfs}

		return these_addresses_anycast_perfs, pruned_these_addresses_perfs


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--dpsize", default=None, required=True)
	args = parser.parse_args()
	rau = RIPE_Atlas_Utilities(args.dpsize)
	rau.load_probe_perfs()





