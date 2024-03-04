import os, numpy as np, time, glob
# from advertisement_experiments import Advertisement_Experiments
from helpers import *
from constants import *


PROPAGATE_TIME = 60*10

TO_POP = True

def measurement_noise_factor():
	return 10 * np.random.uniform()

class Advertisement_Experiments_Dummy:
	### Dummy class to get things working
	### Idea is to mimic what actual measurements from Vultr would output, just for playing around
	def __init__(self, system, mode, deployment_info):
		self.ugs = deployment_info['ugs']
		self.ug_to_ips = deployment_info['ug_to_ip']
		self.ip_to_ug = {ip:ug for ug,ips in self.ug_to_ips.items() for ip in ips}
		self.popps = deployment_info['popps']
		self.pop_to_popps = {}
		for pop,peer in self.popps:
			try:
				self.pop_to_popps[pop].append((pop,peer))
			except KeyError:
				self.pop_to_popps[pop] = [(pop,peer)]
		self.ug_perfs = deployment_info['ug_perfs']
		self.ingress_priorities = deployment_info['ingress_priorities']

		self.uid = 0 

	def infer_ingress(self, possible_popps, ug):
		"""No inference, just pretend we know them."""
		preferences = np.array([self.ingress_priorities[ug][popp] for popp in possible_popps])
		return possible_popps[np.argmin(preferences)]

	def conduct_measurements_to_prefix_popps(self, prefix_popps, every_client_of_interest, 
		popp_lat_fn, **kwargs):
		### I.e., we conducted the measurements and these would be the outputs
		### the key is that the only output we know is the PoP, not the PoPP

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

class RealWorld_Measure_Wrapper:
	#### Actually measure prefix advertisements in the wild, as opposed to simulating them
	def __init__(self, run_dir, deployment_info, **kwargs):

		self.run_dir = run_dir

		self.ugs = sorted(deployment_info['ugs'])
		self.n_ug = len(self.ugs)
		self.ug_to_ind = {ug:i for i,ug in enumerate(self.ugs)}
		self.ug_to_ips = deployment_info['ug_to_ip']
		self.ip_to_ug = {ip:ug for ug,ips in self.ug_to_ips.items() for ip in ips}
		self.popps = sorted(deployment_info['popps'])
		self.popp_to_ind = {popp:i for i,popp in enumerate(self.popps)}
		self.pop_to_popps = {}
		for pop,peer in self.popps:
			try:
				self.pop_to_popps[pop].append((pop,peer))
			except KeyError:
				self.pop_to_popps[pop] = [(pop,peer)]
		self.ug_perfs = deployment_info['ug_perfs']
		self.ingress_priorities = deployment_info['ingress_priorities']

		self.popp_to_clients = {}
		for ug in self.ug_perfs:
			for popp in self.ug_perfs[ug]:
				for client in self.ug_to_ips[ug]:
					try:
						self.popp_to_clients[popp].append(client)
					except KeyError:
						self.popp_to_clients[popp] = [client]

		# self.ae = Advertisement_Experiments('vultr', 'null')
		self.ae = Advertisement_Experiments_Dummy('vultr', 'null', deployment_info)

		## Stores information about which PoPPs beat other PoPPs for each user
		self.measured_prefs = {ui: {self.popp_to_ind[popp]: Ing_Obj(self.popp_to_ind[popp]) for popp in \
			self.ug_perfs[self.ugs[ui]]} for ui in range(self.n_ug)}


		### Bootstrap information about priorities that we know from previous experiments / measurements
		for past_rundir in kwargs.get("past_rundirs", []):
			for runfile in glob.glob(os.path.join(past_rundir ,'measure-output*.csv')):
				self.load_and_add_info(runfile)

	def limit_potential_popps(self, possible_popps, ug):
		## Remove impossible options based on our known information
		ui = self.ug_to_ind[ug]
		available_popp_inds = [self.popp_to_ind[popp] for popp in possible_popps]
		# print(available_popp_inds)
		for nodeid in get_intersection(self.measured_prefs[ui], available_popp_inds):
			node = self.measured_prefs[ui][nodeid]
			if node.has_parent(available_popp_inds):
				node.kill()
		all_possible_popps = [self.popps[nodeid] for nodeid, node in self.measured_prefs[ui].items() if node.alive]
		for nodeid in list(self.measured_prefs[ui]):
			self.measured_prefs[ui][nodeid].alive = True
		possible_popps = get_intersection(all_possible_popps, possible_popps)
		return possible_popps

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
			### Likely an incongruence in our logic which needs to be fixed
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


	def measure_advs(self, advs):
		#### advs is a list of (prefix_i, integer) tuples, each integer tuple corresponds to a PoPP tuple
		#### return prefix_i -> ug -> PoPP for all relevant UGs

		routed_through_ingress = {}

		### First, filter advs to remove any cases for which we already know the answer
		### We might already know the answer if it's a singleton advertisement
		### or if we've measured everything there is to know in a past run
		filtered_advs = []
		for prefix_i, adv in advs:
			routed_through_ingress[prefix_i] = {}
			all_available_options = list([self.popps[poppi] for poppi in adv])
			need_meas = False
			for ug in self.ugs:
				possible_peers = self.limit_potential_popps(all_available_options, ug)
				if len(possible_peers) > 1:
					need_meas = True
					break
				elif len(possible_peers) == 1:
					routed_through_ingress[prefix_i][ug] = self.popp_to_ind[possible_peers[0]]
			if need_meas:
				filtered_advs.append((prefix_i, adv))

		if len(filtered_advs) == 0:
			## we knew everything
			return routed_through_ingress

		### Measure!!
		prefix_popps = []
		prefix_to_popps = {}
		every_client_of_interest = []
		for prefix_i, adv in filtered_advs:
			popps = sorted(list([self.popps[poppi] for poppi in adv]))
			prefix_popps.append(popps)
			prefix_to_popps[prefix_i] = popps

			these_clients = set()
			for popp in popps:
				these_clients = these_clients.union(set(self.popp_to_clients[popp]))
			every_client_of_interest.append(these_clients)

		popp_lat_fn = os.path.join(self.run_dir, 'measure-output-{}.csv'.format(
			int(time.time())))
		i=0
		while os.path.exists(path_lat_fn):
			popp_lat_fn = os.path.join(self.run_dir, 'measure-output-{}-{}.csv'.format(
				int(time.time()), i))
			i+=1
		self.ae.conduct_measurements_to_prefix_popps(prefix_popps, every_client_of_interest, 
			popp_lat_fn, using_manual_clients = True, propagate_time = PROPAGATE_TIME,
			logcomplete = False)

		## Read results and update our model
		this_ug_to_popp = self.load_and_add_info(popp_lat_fn)
		for global_poppset, ret in this_ug_to_popp.items():
			prefix_i = [prefix_i for prefix_i, adv in filtered_advs if \
				set([self.popps[poppi] for poppi in adv]) == set(global_poppset)][0]
			routed_through_ingress[prefix_i] = ret

		return routed_through_ingress

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
			else:
				if TO_POP:
					pref,t_meas,client_dst,dst_pop,_,rtt = fields
				else:
					pref,t_meas,client_dst,dst_pop,dst_peer,_,rtt = fields
				rtt = float(rtt)
				ug = self.ip_to_ug[client_dst]
				uid = pref_to_uid[pref]
				popps_this_experiment = uid_to_popps[uid]
				if TO_POP:
					## Possible ingresses given the PoP it came in at
					popps_this_pop = get_intersection(popps_this_experiment, self.pop_to_popps[dst_pop])
				else:
					popps_this_pop = [(dst_pop,dst_peer)]
				## Possible ingresses given the user
				possible_popps = get_intersection(self.ug_perfs[ug], popps_this_pop)
				try:
					tmp_parse_by_ug[tuple(sorted(popps_this_experiment))]
				except KeyError:
					tmp_parse_by_ug[tuple(sorted(popps_this_experiment))] = {}
				try:
					tmp_parse_by_ug[tuple(sorted(popps_this_experiment))][ug].append((possible_popps, rtt))
				except KeyError:
					tmp_parse_by_ug[tuple(sorted(popps_this_experiment))][ug] = [(possible_popps, rtt)]


		for global_popp_set in tmp_parse_by_ug:
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
				ret[global_popp_set][ug] = self.popp_to_ind[winning_popp]

				### Update the model
				winning_poppi = self.popp_to_ind[winning_popp]
				routed_ingress_obj = self.measured_prefs[self.ug_to_ind[ug]].get(winning_poppi)
				this_ug_all_possible_pops = get_intersection(self.ug_perfs[ug], global_popp_set)
				for beaten_ingress in get_difference(this_ug_all_possible_pops, [winning_popp]):
					beaten_ingressi = self.popp_to_ind[beaten_ingress]
					beaten_ingress_obj = self.measured_prefs[self.ug_to_ind[ug]].get(beaten_ingressi)
					beaten_ingress_obj.add_parent(routed_ingress_obj)
					routed_ingress_obj.add_child(beaten_ingress_obj)

		return ret











