import numpy as np, pickle, copy, zmq, time, random
from collections import defaultdict
random.seed(31415)
from constants import *
from helpers import *
from test_polyphase import *
from optimal_adv_wrapper import Optimal_Adv_Wrapper

from solve_lp_assignment import solve_generic_lp_with_failure_catch, get_paths_by_ug, NO_PATH_INGRESS

import gurobipy as gp
from scipy.sparse import csr_matrix


remeasure_a = None
try:
	remeasure_a = pickle.load(open('remeasure_a.pkl','rb'))
except:
	pass

USER_OF_INTEREST = None

# dont make this too big or you'll break the VM
# 2000 for lots of workers / smaller VMs. 15000 for fewer workers / large VMs
MAX_CACHE_SIZE = 8000

PROB_TOLERANCE = .05 ## if probabilities differ by more than this much, we have to recalculate things

def get_a_cache_rep(a):
	tups = []
	a = threshold_a(a)
	for x,y in zip(*np.where(a)):
		tups.append((x,y))
	return tuple(sorted(tups))

class Path_Distribution_Computer(Optimal_Adv_Wrapper):
	def __init__(self, worker_i, base_port, **kwargs):
		self.worker_i = worker_i
		self.port = base_port
		self.logging_iter = 0
		self.timing = { k:0 for k in ['solve_unified_lp_not_optimize', 'optimize', 'get_paths_by_ug','organizing_results','get_ingress_probabilities_by_dict_generic', 'sim_rti']}
		self.updated_ingress_probabilities = True

		self.MC_NUM = 5 ## monte carlo simulations to determine distributions

		if kwargs.get('debug', False):
			self.n_prefixes = None
			return
		args, kwargs = self.start_connection()
		super().__init__(*args, **kwargs)

		with open(os.path.join(LOG_DIR, 'worker_{}_log-{}.txt'.format(self.worker_i, self.dpsize)),'w') as f:
			pass
		self.init_all_vars()
		self.run()
		# print('started in worker {}'.format(self.worker_i))


	def summarize_timing(self):
		total_time = sum(list(self.timing.values()))
		print("\n\n===============\nWorker {} timing summary".format(self.worker_i))
		for k in sorted(list(self.timing), key = lambda el : self.timing[el]):
			pct = round(self.timing[k] * 100.0 / total_time, 2)
			print("{} - {} pct".format(k, pct))
		print("==================\n\n")

	def init_persistent_lp(self):
		"""Sets up the persistent Gurobi shell with static Volumes and Capacities."""
		self.model = gp.Model(f"Worker_{self.worker_i}_Persistent")
		self.model.Params.LogToConsole = 0
		self.model.Params.Method = 1  
		self.model.Params.Threads = 1 

		# 1. Permanent Dummy Variable for MLU (Y)
		self.mlu_dummy = self.model.addVar(lb=0.0, obj=0.0, name="mlu_Y")

		# 2. Permanent Rows (Constraints)
		self.vol_constrs = {}
		for ugi, ug in enumerate(self.whole_deployment_ugs):
			# IMPORTANT: Gurobi addConstr needs an expression. 
			# Since we haven't added columns yet, we set LHS to 0.0 explicitly
			target_vol = float(self.whole_deployment_ug_vols[ugi])
			
			# We use a Linear Expression placeholder (0.0) 
			# so Gurobi knows this is a constraint to be filled later.
			self.vol_constrs[ug] = self.model.addLConstr(0.0, gp.GRB.EQUAL, target_vol, name=f"vol_{ug}")

		self.static_caps = np.concatenate([self.link_capacities_arr.flatten(), [1000000.0]])
		self.cap_constrs = {}
		for pi in range(len(self.static_caps)):
			target_cap = float(self.static_caps[pi])
			# sum(paths) <= cap
			self.cap_constrs[pi] = self.model.addLConstr(0.0, gp.GRB.LESS_EQUAL, target_cap, name=f"cap_{pi}")

		self.var_pool = {} # Key: (ug, poppi) -> Gurobi Var Object

	def solve_unified_lp(self, available_paths, obj_coeffs, using_mlu=False):
		"""Core solve logic. Toggles between Standard and MLU."""
		# 1. Deactivate all path variables
		ts = time.time()
		all_vars = list(self.var_pool.values())
		self.model.setAttr("UB", all_vars, [0.0] * len(all_vars))

		# 2. Configure MLU variable and Capacity RHS
		if using_mlu:
			self.mlu_dummy.Obj = 1.0 / ALPHA
			self.mlu_dummy.UB = gp.GRB.INFINITY
			for pi, constr in self.cap_constrs.items():
				self.model.chgCoeff(constr, self.mlu_dummy, -1.0 * self.static_caps[pi])
				constr.RHS = 0.0 # MLU mode uses RHS of 0
		else:
			self.mlu_dummy.Obj = 0.0
			self.mlu_dummy.UB = 0.0
			for pi, constr in self.cap_constrs.items():
				self.model.chgCoeff(constr, self.mlu_dummy, 0.0)
				constr.RHS = self.static_caps[pi] # Restore static capacity

		# 3. Activate/Discover Columns (Volume RHS is already static)
		for (ug, poppi), latency in zip(available_paths, obj_coeffs):
			key = (ug, poppi)
			if key not in self.var_pool:
				col = gp.Column()
				col.addTerms(1.0, self.vol_constrs[ug])
				col.addTerms(1.0, self.cap_constrs[poppi])
				self.var_pool[key] = self.model.addVar(lb=0.0, obj=latency, column=col)

		active_vars = [self.var_pool[ug, poppi] for ug, poppi in available_paths]
		self.model.setAttr("UB", active_vars, [gp.GRB.INFINITY] * len(active_vars))
		self.model.setAttr("Obj", active_vars, obj_coeffs)

		self.timing['solve_unified_lp_not_optimize'] = time.time() - ts
		ts = time.time()
		self.model.optimize()
		self.timing['optimize'] = time.time() - ts
		if self.model.status == 2:
			return self.model
		return None

	def solve_generic_lp_persistent(self, routed_through_ingress, **kwargs):
		"""The high-level wrapper that tries Standard first, then MLU."""
		ts = time.time()
		available_paths, _ = get_paths_by_ug(self, routed_through_ingress)
		self.timing['get_paths_by_ug'] = time.time() - ts
		
		# Pre-calculate objective (latencies)
		obj_coeffs = []
		for ug, poppi in available_paths:
			if poppi == NO_PATH_INGRESS(self):
				obj_coeffs.append(NO_ROUTE_LATENCY)
			else:
				obj_coeffs.append(self.whole_deployment_ug_perfs[ug][self.popps[poppi]])

		# 1. Try Standard Solve
		model_res = self.solve_unified_lp(available_paths, obj_coeffs, using_mlu=False)

		# 2. Fallback to MLU if Standard is Infeasible
		if model_res is None:
			model_res = self.solve_unified_lp(available_paths, obj_coeffs, using_mlu=True)

		if model_res is None:
			print("Infeasible problem, exiting")
			exit(0)
			return {'solved': False}

		## Distribution is the amount of volume (not percent) placed on each path
		## a path is specified by a <user, popp>
		raw_x = {key: var.X for key, var in self.var_pool.items() if var.X > 1e-7}
	
		# Initialize result containers
		lats_by_ug_arr = np.zeros(self.whole_deployment_n_ug)
		vols_by_poppi = {pi: 0.0 for pi in range(len(self.static_caps))}
		paths_by_ug_res = {}
		congested_vol, total_vol = 0.0, 0.0

		ts = time.time()
		for (ug, poppi), vol_amt in raw_x.items():
			ugi = self.whole_deployment_ug_to_ind[ug]
			vols_by_poppi[poppi] += vol_amt
			total_vol += vol_amt
			
			# Check congestion against STATIC caps
			if vol_amt > self.static_caps[poppi] + 1e-6 and poppi != len(self.static_caps)-1:
				congested_vol += vol_amt
			
			if ugi not in paths_by_ug_res:
				paths_by_ug_res[ugi] = []
			paths_by_ug_res[ugi].append((poppi, vol_amt / self.whole_deployment_ug_to_vol[ug]))

			# Calculate latency for this specific <user, path> allocation
			if poppi == NO_PATH_INGRESS(self):
				path_lat = NO_ROUTE_LATENCY
			else:
				path_lat = self.whole_deployment_ug_perfs[ug][self.popps[poppi]]
			
			# Weighted average latency contribution
			lats_by_ug_arr[ugi] += path_lat * (vol_amt / self.whole_deployment_ug_to_vol[ug])

		obj_norm = np.sum(self.whole_deployment_ug_vols)
		self.timing['organizing_results'] = time.time()-ts
		return {
			"objective": -1 * model_res.objVal / obj_norm, # Framing 'benefit' as positive
			"raw_solution": raw_x,
			"paths_by_ug": paths_by_ug_res,
			"lats_by_ug": lats_by_ug_arr,
			"available_paths": available_paths,
			"solved": True,
			"vols_by_poppi": vols_by_poppi,
			"fraction_congested_volume": congested_vol / (total_vol + 1e-9)
		}

	def init_all_vars(self):
		## Latency benefit for each user is -1 * MAX_LATENCY -> -1 MIN_LATENCY
		## divided by their contribution to the total volume (i.e., multiplied by a weight)
		## important that every worker has the same lbx
		min_vol,max_vol = np.min(self.ug_vols), np.max(self.ug_vols)
		total_deployment_volume = np.sum(self.ug_vols)
		if self.simulated:
			min_lbx = np.maximum(-.1,-1 * NO_ROUTE_LATENCY * max_vol / total_deployment_volume)
		else:
			min_lbx = np.maximum(-.1,-1 * NO_ROUTE_LATENCY * max_vol / total_deployment_volume)

		max_lbx = 0

		self.lbx = np.linspace(min_lbx, max_lbx,num=LBX_DENSITY)
		self.big_lbx = np.zeros((LBX_DENSITY, self.n_ug))
		for i in range(self.n_ug):
			self.big_lbx[:,i] = copy.copy(self.lbx)
		self.lb_range_trackers = {ui: [min_lbx,max_lbx] for ui in range(self.n_ug)}
		self.lb_range_alpha = .005 ## EWMA for update LB range definitions

		self.stop = False
		self.calc_cache = Calc_Cache()
		self.this_time_ip_cache = {}

		self.iter = 0
		self.init_persistent_lp() # Setup the Gurobi shell

	def start_connection(self):
		context = zmq.Context()
		print("Worker {} starting on port {}".format(self.worker_i,self.worker_i+self.port))
		self.main_socket = context.socket(zmq.REP)
		self.main_socket.setsockopt(zmq.RCVTIMEO, 1000)
		self.main_socket.bind('tcp://*:{}'.format(self.worker_i+self.port))
		while True:
			try:
				init_msg = self.main_socket.recv()
				break
			except zmq.error.Again:
				time.sleep(.01)
		self.main_socket.send(pickle.dumps('ACK'))
		msg_decoded = pickle.loads(init_msg)
		_, data = msg_decoded
		return data

	def increment_iter(self):
		self.iter += 1

	def get_limited_cap_latency_multiplier(self):
		if self.simulated:
			LIMITED_CAP_LATENCY_MULTIPLIER = 1.1
			power = 1.02
			return np.minimum(2.4, np.power(power,self.iter+1) * LIMITED_CAP_LATENCY_MULTIPLIER)
		else:
			LIMITED_CAP_LATENCY_MULTIPLIER = 1.3
			power = 1.04
			return np.minimum(10.0, np.power(power,((self.iter+1)/3)) * LIMITED_CAP_LATENCY_MULTIPLIER)
		# LIMITED_CAP_LATENCY_MULTIPLIER = 5
		# power = 1.05
		# return np.minimum(20, np.power(power,self.iter+1) * LIMITED_CAP_LATENCY_MULTIPLIER)

	def clear_caches(self):
		self.this_time_ip_cache = {}
		self.calc_cache.clear_all_caches()

	def get_ingress_probabilities_by_dict_generic(self, a, verb=False, **kwargs):
		## Uses dictionaries to do the job
		a_log = threshold_a(a).astype(bool)

		sum_a = np.sum(a,axis=0)

		timers = {
			'cache_hits': 0,
			'cache_lookups': 0,
			'api': 0,
			'dpi': 0,
			'apugi': 0,
			'vpugi': 0,
			'sort_calc': 0,
			'final_calc': 0,
			'total': 0,
		}

		self.ingress_probabilities = {ui:{} for ui in range(self.whole_deployment_n_ug)}
		for pref_i in np.where(sum_a)[0]:
			ts_loop = time.time()
			tloga = tuple(a_log[:,pref_i].flatten())
			if np.sum(a[:,pref_i]) == 0:
				continue
			try:
				for (poppi,ui), prob in self.this_time_ip_cache[tloga].items():
					# will need a more complicated caching mechanism if ever non-uniform
					self.ingress_probabilities[ui][poppi,pref_i] = 1.0/prob 
				timers['cache_hits'] += 1
				timers['cache_lookups'] += time.time() - ts_loop
				continue
			except KeyError:
				pass

			## i.e, for each user and for each popp. compute whether a parent of that popp is currently active
			active_parent_indicator = {}
			poppis_active = {poppi:None for poppi in np.where(a_log[:,pref_i])[0]}
			for ug,child,parent in self.parent_tracker: ### we should modify parent tracker to map parent to children
				ui = self.whole_deployment_ug_to_ind[ug]
				parenti = self.popp_to_ind[parent]
				childi = self.popp_to_ind[child]
				try:
					poppis_active[parenti]
					active_parent_indicator[ui,childi] = 1
				except KeyError:
					continue
			timers['api'] += time.time() - ts_loop; ts_loop=time.time()

			## For active poppi in active_poppis, for user in poppi to users, if not parent active for poppi -> tabulate

			## Group by user
			self.this_time_ip_cache[tloga] = {}
			cacheref = self.this_time_ip_cache[tloga]
			for ui in range(self.whole_deployment_n_ug):
				these_poppis = []
				ref = self.whole_deployment_ui_to_poppi[ui]
				for poppi in poppis_active:
					try:
						ref[poppi]
					except KeyError:
						continue ### user doesn't have this popp
					try:
						active_parent_indicator[ui,poppi] ### We have an active parent, ignore
						continue
					except KeyError:
						these_poppis.append(poppi)

				if len(these_poppis) == 0:
					continue
				npoppis = len(these_poppis)
				likelihood = 1.0 / npoppis
				for poppi in these_poppis:
					self.ingress_probabilities[ui][poppi,pref_i] = likelihood
					### Cache the entries that have non-zero probability
					cacheref[poppi,ui] = npoppis
			timers['final_calc'] += time.time() - ts_loop; ts_loop=time.time()

		# if np.random.random() > 0 and self.worker_i == 0:
		#   print('\n')
		#   for k,v in timers.items():
		#       print("{} -- {} s".format(k,round(v,5)))

	def get_ingress_probabilities_by_dict(self, a, verb=False, **kwargs):
		## Uses dictionaries to do the job
		a_log = threshold_a(a).astype(bool)

		sum_a = np.sum(a,axis=0)

		timers = {
			'cache_hits': 0,
			'cache_lookups': 0,
			'api': 0,
			'dpi': 0,
			'apugi': 0,
			'vpugi': 0,
			'sort_calc': 0,
			'final_calc': 0,
			'total': 0,
		}

		self.ingress_probabilities = {ui:{} for ui in range(self.n_ug)}
		for pref_i in np.where(sum_a)[0]:
			ts_loop = time.time()
			tloga = tuple(a_log[:,pref_i].flatten())
			if np.sum(a[:,pref_i]) == 0:
				continue
			try:
				for (poppi,ui), prob in self.this_time_ip_cache[tloga].items():
					# will need a more complicated caching mechanism if ever non-uniform
					self.ingress_probabilities[ui][poppi,pref_i] = 1.0/prob 
				timers['cache_hits'] += 1
				timers['cache_lookups'] += time.time() - ts_loop
				continue
			except KeyError:
				pass

			## i.e, for each user and for each popp. compute whether a parent of that popp is currently active
			active_parent_indicator = {}
			poppis_active = {poppi:None for poppi in np.where(a_log[:,pref_i])[0]}
			for ug,child,parent in self.parent_tracker: ### we should modify parent tracker to map parent to children
				ui = self.ug_to_ind[ug]
				parenti = self.popp_to_ind[parent]
				childi = self.popp_to_ind[child]
				try:
					poppis_active[parenti]
					active_parent_indicator[ui,childi] = 1
				except KeyError:
					continue
			timers['api'] += time.time() - ts_loop; ts_loop=time.time()

			## For active poppi in active_poppis, for user in poppi to users, if not parent active for poppi -> tabulate

			## Group by user
			self.this_time_ip_cache[tloga] = {}
			cacheref = self.this_time_ip_cache[tloga]
			for ui in range(self.n_ug):
				these_poppis = []
				ref = self.ui_to_poppi[ui]
				for poppi in poppis_active:
					try:
						ref[poppi]
					except KeyError:
						continue ### user doesn't have this popp
					try:
						active_parent_indicator[ui,poppi] ### We have an active parent, ignore
						continue
					except KeyError:
						these_poppis.append(poppi)

				if len(these_poppis) == 0:
					continue
				npoppis = len(these_poppis)
				likelihood = 1.0 / npoppis
				for poppi in these_poppis:
					self.ingress_probabilities[ui][poppi,pref_i] = likelihood
					### Cache the entries that have non-zero probability
					cacheref[poppi,ui] = npoppis
			timers['final_calc'] += time.time() - ts_loop; ts_loop=time.time()

		# if np.random.random() > 0 and self.worker_i == 0:
		#   print('\n')
		#   for k,v in timers.items():
		#       print("{} -- {} s".format(k,round(v,5)))

	def summarize_cache_size(self):
		## 
		for obj,nm in zip([self.user_ip_cache, self.calc_cache, self.this_time_ip_cache], 
			['user ip', 'calc cache', 'this time ip']):
			self.print("{} cache -- {} size".format(nm,round(len(pickle.dumps(obj))/1e6)))

	def lfs_to_penalty(self, lfs):
		"""LFS is the greedy-allocation volume divided by the link capacity. Want to compute a
			multiplicative latency penalty to encourage people to not inundate links. 
			But can't be too rough so that it's unstable."""
		return np.power(lfs, .1)

	def sim_rti_better(self):
		if self.updated_ingress_probabilities:
			# 1. Map (ui, pref_i) pairs to flat indices and build distributions
			# This prepares the data for a single vectorized operation
			self.all_uis_prefixes_poppis = [] # Stores (ui, pref_i, list_of_poppis)
			all_probs = []
			
			# We group by (ui, pref_i) to form the distribution for each user-prefix pair
			# If your input structure is ui -> {(poppi, pref_i): prob}
			for ui, data in self.ingress_probabilities.items():
				# Temporary group by prefix for this specific user
				prefs_for_ui = defaultdict(list)
				for (poppi, pref_i), p in data.items():
					prefs_for_ui[pref_i].append((poppi, p))
					
				for pref_i, distributions in prefs_for_ui.items():
					poppis = [d[0] for d in distributions]
					probs = [d[1] for d in distributions]
					self.all_uis_prefixes_poppis.append((ui, pref_i, poppis))
					all_probs.append(probs)

			num_dists = len(all_probs)
			if num_dists == 0:
				return {}

			# 2. Create a Padded CDF Matrix
			# Vectorization requires a rectangular matrix, but PoP counts may vary.
			# We pad the rows with 1.0.
			max_pops = max(len(p) for p in all_probs)
			self.routing_cdf_matrix = np.zeros((num_dists, max_pops))
			
			for i, p in enumerate(all_probs):
				row_cdf = np.cumsum(p)
				row_cdf[-1] = 1.0  # Force exact 1.0 to prevent floating point drift
				self.routing_cdf_matrix[i, :len(row_cdf)] = row_cdf
				if len(row_cdf) < max_pops:
					self.routing_cdf_matrix[i, len(row_cdf):] = 1.0

			# 3. Generate ALL Randomness Apriori
			# One single call to the random engine
			# Shape: (number of distributions, number of simulations)
			self.random_mc_routing_samples = np.random.random((num_dists, self.MC_NUM))
			self.updated_ingress_probabilities = False
		else:
			self.random_mc_routing_samples = np.random.random(self.random_mc_routing_samples.shape)

		# 4. Vectorized Sampling
		# np.searchsorted finds which "bucket" the random number falls into.
		sampled_indices = np.empty(self.random_mc_routing_samples.shape, dtype=int)
		for i in range(self.random_mc_routing_samples.shape[0]):
			sampled_indices[i] = np.searchsorted(self.routing_cdf_matrix[i], self.random_mc_routing_samples[i])

		# 5. Reconstruct the Results
		routed_through_ingress = {mci: defaultdict(dict) for mci in range(self.MC_NUM)}


		for row_idx, (ui, pref_i, poppis) in enumerate(self.all_uis_prefixes_poppis):
			ug = self.whole_deployment_ugs[ui]
			# Get the poppi for every simulation for this (ui, pref_i)
			chosen_poppi_indices = sampled_indices[row_idx]
			
			for mci in range(self.MC_NUM):
				poppi = poppis[chosen_poppi_indices[mci]]
				routed_through_ingress[mci][pref_i][ug] = self.popps[poppi]
				try:
					if not (self.ingress_probabilities[ui][poppi,pref_i] > 0):
						print("{},{},{} has route but {} probability".format(ui,poppi,pref_i,self.ingress_probabilities[ui][poppi,pref_i]))
				except KeyError:
					if ui not in self.ingress_probabilities:
						print("UI in not ingress probs")
					elif (poppi,pref_i) not in self.ingress_probabilities[ui]:
						print("{},{} not in inress probs for {}".format(poppi,pref_i,ui))
					exit(0)

		return routed_through_ingress	

	def sim_rti(self):
		### Randomly simulates routes and returns them according to our model of ingress probabilities
		## routed_through_ingress: prefix -> ug -> popp

		## helpful object to precompute
		ts = time.time()
		self.pmat_by_prefix = {}
		for ui in self.ingress_probabilities:
			self.pmat_by_prefix[ui] = {}
			for (poppi, pref_i), p in self.ingress_probabilities[ui].items():
				try:
					self.pmat_by_prefix[ui][pref_i] 
				except KeyError:
					self.pmat_by_prefix[ui][pref_i] = [[],[]]
				self.pmat_by_prefix[ui][pref_i][0].append(poppi)
				self.pmat_by_prefix[ui][pref_i][1].append(p)
		self.timing['pmat_organize'] = time.time() - ts

		## Aggregate by prefix (since we're simulating routes)
		routed_through_ingress = {}
		for ui in self.ingress_probabilities:
			## randomly simulate routing
			choices_by_simi = {}
			for pref_i in self.pmat_by_prefix[ui]:
				poppis, probs = self.pmat_by_prefix[ui][pref_i]
				random_poppi = np.random.choice(poppis, size=self.MC_NUM, replace=True, p=probs)
				for mci in range(self.MC_NUM):
					try:
						routed_through_ingress[mci]
					except KeyError:
						routed_through_ingress[mci] = {}
					try:
						routed_through_ingress[mci][pref_i][self.whole_deployment_ugs[ui]] = self.popps[random_poppi[mci]]
					except KeyError:
						routed_through_ingress[mci][pref_i] = {self.whole_deployment_ugs[ui]: self.popps[random_poppi[mci]]}

		return routed_through_ingress

	def generic_objective_pdf(self, obj, **kwargs):
		"""
			Solves self.MC_NUM traffic assignment problems, assuming that user routes are distributed
			according to distribution self.ingress_probabilities.
		"""

		### TODO -- maybe implement subset of users, but not really essential
		ts = time.time()
		# all_routed_through_ingress = self.sim_rti_better()
		all_routed_through_ingress = self.sim_rti()
		self.timing['sim_rti'] = time.time() - ts
		objs = np.zeros(self.MC_NUM)
		for i in range(self.MC_NUM):
			routed_through_ingress = all_routed_through_ingress[i]
			total_obj = self.solve_generic_lp_persistent(routed_through_ingress)["objective"]
			# total_obj = solve_generic_lp_with_failure_catch(self, routed_through_ingress, obj)['objective']
			objs[i] = total_obj
		### return x and distribution of x
		## numpy histogram returns all bin edges which is of length len(x) + 1
		## so cut off the last edge
		if max(objs) - min(objs) < .001:
			## trivial distribution
			x = np.linspace(objs[0],objs[0]+1, num=LBX_DENSITY)
			pdfx = np.zeros(LBX_DENSITY)
			pdfx[0] = 1.0
		else:
			pdfx, x = np.histogram(objs, bins=LBX_DENSITY, density=True)
			x = x[:-1]
			pdfx = pdfx / np.sum(pdfx)
		return x, pdfx

	def generic_benefit(self, a, f_w, **kwargs):
		"""

		Calculates average and distributional estimate of benefit, where benefit
		is a function of the (joint) routing distribution.
		Works by MC-sampling routing distribution and computing a histogram of benefits.

		"""

		a_effective = threshold_a(a)
		verb = kwargs.get('verbose_workers')

		### We may choose to compute expected benefit over a subset of all users
		### when we do this, the key thing is to remember to turn off caching
		subset_ugs = False
		which_ugs = kwargs.get('ugs', None)
		if which_ugs is not None:
			subset_ugs = True

		if not verb and not subset_ugs:
			## don't rely on caching if we want to log / print statistics
			try:
				cache_rep = get_a_cache_rep(a_effective)
				benefit, (xsumx_cache_rep, psumx_cache_rep) = self.calc_cache.all_caches['lb'][cache_rep]
				xsumx = np.linspace(xsumx_cache_rep[0], xsumx_cache_rep[1], num=LBX_DENSITY)
				psumx = np.zeros(LBX_DENSITY)
				for i,d in psumx_cache_rep.items():
					psumx[i] = d
				ret = (benefit, (xsumx,psumx))

				return ret
			except KeyError:
				pass

		## Dims are path, prefix, user
		ts = time.time()
		self.get_ingress_probabilities_by_dict_generic(a_effective, **kwargs) ## populates self.ingress_probabilities
		self.timing['get_ingress_probabilities_by_dict_generic'] = time.time() - ts

		if subset_ugs: ##### REVISIT
			which_ugs_this_worker = get_intersection(which_ugs, self.whole_deployment_ugs)
			if len(which_ugs_this_worker) == 0:
				pdf = np.zeros(self.lbx.shape)
				pdf[-1] = 1
				return 0, (self.lbx.flatten(), pdf.flatten())
			which_ugs_i = np.array([self.whole_deployment_ug_to_ind[ug] for ug in which_ugs_this_worker])


		## Calculate pdf of the generic objective
		if subset_ugs:
			xsumx, psumx = self.generic_objective_pdf(f_w, which_ugs_i=which_ugs_i)
		else:
			xsumx, psumx = self.generic_objective_pdf(f_w)

		xsumx = xsumx.flatten(); psumx = psumx.flatten()
		benefit = np.sum(xsumx * psumx)

		if not subset_ugs:
			### Store compressed versions of these variables
			cache_rep = get_a_cache_rep(a_effective)
			xsumx_cache_rep = (xsumx[0], xsumx[-1])
			psumx_cache_rep = {}
			for i in np.where(psumx)[0]:
				psumx_cache_rep[i] = psumx[i]

			self.calc_cache.all_caches['lb'][cache_rep] = (benefit, (xsumx_cache_rep, psumx_cache_rep))
		# print("Returning benefit : {}".format(benefit))
		return benefit, (xsumx, psumx)  

	def latency_benefit(self, a, **kwargs):
		"""Calculates distribution of latency benefit at a given advertisement. Benefit is the sum of 
			benefits across all users. Closed form calculation."""
		return self.generic_benefit(a, kwargs.get('generic_obj'))

	def log(self,s):
		self.log_ptr = open(os.path.join(LOG_DIR, 'worker_{}_log-{}.txt'.format(self.worker_i, self.dpsize)),'a')
		self.log_ptr.write(s)
		self.log_ptr.close()

	def print(self, s):
		print("Worker {} -- {}".format(self.worker_i, s))

	def check_clear_cache(self):
		cache_to_clear = self.calc_cache.all_caches['lb']

		if len(cache_to_clear) > MAX_CACHE_SIZE:
			# order of lbx_density + n_popps*n_prefixes per entry
			# self.print("Clearing calc cache, currently size {}".format(
			#   len(pickle.dumps(self.calc_cache))/1e6))
			# self.print("Clearing calc cache, current len {}".format(len(cache_to_clear)))
			self.this_time_ip_cache = {}
			self.calc_cache.all_caches['lb'] = {}
		if not self.simulated:
			if np.random.random() > .9999:
				## Just randomly clear it since we don't measure often
				self.this_time_ip_cache = {}
				self.calc_cache.all_caches['lb'] = {}


	def clear_new_meas_caches(self):
		# print("Clearing caches in worker {}".format(self.worker_i))
		self.this_time_ip_cache = {}
		self.calc_cache.clear_new_measurement_caches()

	def check_for_commands(self):
		# print("checking for commands in worker {}".format(self.worker_i))
		try:
			msg = self.main_socket.recv()
			# print("Received message in worker with length : {}".format(len(msg)))
		except zmq.error.Again:
			return
		try:
			msg = pickle.loads(msg)
		except:
			print(msg)
			print("Failed parsing message of length : {}, sending back error message".format(len(msg)))
			pickle.dump(msg, open('error_{}_{}.pkl'.format(int(time.time()), self.worker_i),'wb'))
			self.main_socket.send(pickle.dumps("ERROR")) # should hopefully generate an error in the main thread
			return
		cmd, data = msg
		# print("received command {} in worker {}".format(cmd, self.worker_i))
		if cmd == 'calc_lb':
			ret = []
			self.this_time_ip_cache = {}
			for (args,kwargs) in data:
				ret.append(self.latency_benefit(*args, **kwargs))
			del self.this_time_ip_cache

		elif cmd == "solve_lp":
			ret = []
			ts = time.time()
			n_iters,t_per_iter = 0,0
			self.check_load_rw_measure_wrapper()
			for fields in sorted(data, key = lambda el : el[0]):
				if len(fields) == 4:
					adv_i, adv, deployment, update_dep = fields
				else:
					adv_i, adv, opt_adv, deployment, update_dep = fields
				if update_dep:
					deployment_save = self.output_deployment()
					self.clear_caches()
					self.update_deployment(deployment,quick_update=True,verb=False,exit_on_impossible=False)
				self.check_load_rw_measure_wrapper()

				rti, _ = self.calculate_ground_truth_ingress(adv, do_cache=False)
				this_ret = solve_generic_lp_with_failure_catch(self, rti, deployment.get('generic_objective'))
				if update_dep:
					self.update_deployment(deployment_save,quick_update=True,verb=False,exit_on_impossible=False)
					self.check_load_rw_measure_wrapper()
				ret.append((adv_i, this_ret))
				n_iters += 1

				t_per_iter = round((time.time() - ts)/n_iters,2)
		elif cmd == 'calc_compressed_lb':
			ts = time.time()
			tlp = time.time()
			ret = []
			base_args,base_kwa = data[0]
			base_adv, = base_args
			base_adv = base_adv.astype(bool)
			ret.append({'ans': self.latency_benefit(base_adv, **base_kwa), 'job_id': base_kwa.get('job_id', -1)})
			i=0
			last_timing_summary = 0
			for diff, kwa in data[1:]:
				kwa['verbose_workers'] = base_kwa.get('verbose_workers',False) or kwa.get('verbose_workers',False)
				for ind in zip(*diff):
					base_adv[ind] = not base_adv[ind]
				ret.append({'ans': self.latency_benefit(base_adv, **kwa), 'job_id': kwa.get('job_id',-1)})
				for ind in zip(*diff):
					base_adv[ind] = not base_adv[ind]
				i += 1
				# kwa['verb'] = True
				if time.time() - tlp > 100:
					self.print("{} pct. done calcing latency benefits, {}ms per iter".format( 
						round(i * 100.0 /len(data),1), round(1000*(time.time() - ts) / i)))
					tlp = time.time()
				if i % 50 == 0 and i > 0 and time.time() - last_timing_summary > 20:
					self.summarize_timing()
					last_timing_summary = time.time()
				self.check_clear_cache()
			# if len(data)>10:
			#   print("Worker {} calcs took {}s".format(self.worker_i, int(time.time() - ts)))
		
		elif cmd == 'reset_new_meas_cache':
			self.clear_new_meas_caches()
			ret = "ACK"
		elif cmd == 'update_parent_tracker':
			self.updated_ingress_probabilities = True
			parents_on = data
			for ug in parents_on:
				for beaten_ingress, routed_ingress in parents_on[ug]:
					self.parent_tracker[ug, beaten_ingress, routed_ingress] = True
			if len(parents_on) > 0:
				self.clear_new_meas_caches()
			ret = "ACK"
		elif cmd == 'update_deployment':
			deployment, kwargs = data
			self.update_deployment(deployment, **kwargs)
			ret = "ACK"
		elif cmd == 'update_kwa':
			new_kwa = data
			if new_kwa.get('n_prefixes') is not None:
				self.n_prefixes = new_kwa.get('n_prefixes')
			if new_kwa.get('gamma') is not None:
				self.gamma = new_kwa.get('gamma')
			if new_kwa.get('with_capacity') is not None:
				self.with_capacity = new_kwa.get('with_capacity')
			ret = 'ACK'
		elif cmd == 'increment_iter':
			self.increment_iter()
			ret = "ACK"
		elif cmd == 'set_iter':
			self.iter = data
			ret = "ACK"
		elif cmd == 'reset_cache':
			self.clear_caches()
			ret = "ACK"
		elif cmd == 'init':
			self.start_connection()
			return
		elif cmd == 'end':
			print("Received end command in worker {}, stopping".format(self.worker_i))
			self.stop = True
			self.main_socket.close()
			return
		else:
			print("Invalid CMD in worker {} : {}".format(self.worker_i, cmd))
			exit(0)
		self.main_socket.send(pickle.dumps(ret))

	def run(self):
		while not self.stop:
			self.check_for_commands()
			time.sleep(.01)
		print("Ended run loop in worker {}".format(self.worker_i))

if __name__ == "__main__":
	worker_i = int(sys.argv[1])
	base_port = int(sys.argv[2])
	pdc = Path_Distribution_Computer(worker_i, base_port)