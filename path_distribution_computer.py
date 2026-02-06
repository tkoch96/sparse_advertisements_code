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

gp.setParam("OutputFlag", 0)


remeasure_a = None
try:
	remeasure_a = pickle.load(open('remeasure_a.pkl','rb'))
except:
	pass

USER_OF_INTEREST = None

## TODO -- just remove the old version once we're more confident it works well
TEST_BETTER_VERSION = True

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
		self.timing = { k:0 for k in ['solve_unified_lp_not_optimize', 'optimize', 'get_paths_by_ug','organizing_results',
		'get_ingress_probabilities_by_dict_generic', 'sim_rti', 
		'solve_generic_lp_persistent', 'solve_generic_lp_not_persistent']}
		self.rti_data = {}

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
		return
		total_time = sum(list(self.timing.values()))
		print("\n\n===============\nWorker {} timing summary".format(self.worker_i))
		for k in sorted(list(self.timing), key = lambda el : self.timing[el]):
			pct = round(self.timing[k] * 100.0 / total_time, 2)
			print("{} - {} pct ({} ms)".format(k, pct, round(self.timing[k]*1000,2)))
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

	def solve_generic_lp_persistent(self, routed_through_ingress, obj, **kwargs):
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
				if obj == "avg_latency":
					obj_coeffs.append(self.whole_deployment_ug_perfs[ug][self.popps[poppi]])
				elif obj == "per_site_cost":
					pop, _ = self.popps[poppi]
					site_cost = self.site_costs[pop]
					obj_coeffs.append(self.whole_deployment_ug_perfs[ug][self.popps[poppi]] + DEFAULT_SITE_COST * site_cost)
				else:
					raise ValueError("obj {} not supported in solve_generic_lp_persistent".format(obj))

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

	def clear_caches(self):
		self.this_time_ip_cache = {}
		self.calc_cache.clear_all_caches()

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

	def sim_rti_better(self):
		ts = time.time()
		
		# --- Step 1: Flatten Data & Prepare Matrices ---
		# We need to map the nested dict structure into linear arrays for vectorization.
		# We will track metadata to reconstruct the dictionary later.
		self.rti_data["meta_data"] = [] # List of tuples: (user_index, prefix_index, user_group_name)
		self.rti_data["all_probs"] = [] # List of probability arrays
		self.rti_data["all_poppis"] = [] # List of ingress index arrays
		
		# Iterate through your existing structure to flatten it
		for ui, entries in self.ingress_probabilities.items():
			# Temporary storage to group by prefix for this user
			# (Your original code grouped by prefix, so we must too)
			temp_group = {} 
			
			for (poppi, pref_i), p in entries.items():
				if pref_i not in temp_group:
					temp_group[pref_i] = {'pops': [], 'probs': []}
				temp_group[pref_i]['pops'].append(poppi)
				temp_group[pref_i]['probs'].append(p)
				
			# Add these grouped entries to our master lists
			ug_name = self.whole_deployment_ugs[ui]
			for pref_i, data in temp_group.items():
				self.rti_data["meta_data"].append((ui, pref_i, ug_name))
				self.rti_data["all_probs"].append(data['probs'])
				self.rti_data["all_poppis"].append(data['pops'])

		self.timing['pmat_organize'] = time.time() - ts

		# --- Step 2: Memory-Efficient Vectorized Selection ---
		
		self.rti_data["num_scenarios"] = len(self.rti_data["all_probs"])
		self.rti_data["max_choices"] = max(len(p) for p in self.rti_data["all_probs"])
		
		# Create Padded Matrix and CDF as before
		# Memory: O(N_scenarios * Max_Choices) - Very small
		P_matrix = np.zeros((self.rti_data["num_scenarios"], self.rti_data["max_choices"]))
		self.rti_data["choices_matrix"] = np.full((self.rti_data["num_scenarios"], self.rti_data["max_choices"]), -1, dtype=int)
		
		for i, (probs, pops) in enumerate(zip(self.rti_data["all_probs"], self.rti_data["all_poppis"])):
			n = len(probs)
			P_matrix[i, :n] = probs
			self.rti_data["choices_matrix"][i, :n] = pops

		cdf = np.cumsum(P_matrix, axis=1)
		cdf[:, -1] = 1.0
		# 1. Create offsets. shape: (N_scenarios,)
		# Each row 'i' is shifted by i. 
		# This ensures values in row 0 are in range [0, 1], row 1 in [1, 2], etc.
		self.rti_data["offsets"] = np.arange(self.rti_data["num_scenarios"])
		
		# 2. Add offsets to the CDF
		# shape: (N_scenarios, Max_Choices)
		self.rti_data["cdf_offset"] = cdf + self.rti_data["offsets"][:, None]
		# Generate Random numbers
		# Memory: O(N_scenarios * MC_NUM)
		rand_vals = np.random.rand(self.rti_data["num_scenarios"], self.MC_NUM)
		
		# 3. Add offsets to the random values
		# shape: (N_scenarios, MC_NUM)
		rand_offset = rand_vals + self.rti_data["offsets"][:, None]

		# 4. Flatten both
		# cdf_flat size: N_scenarios * Max_Choices
		# rand_flat size: N_scenarios * MC_NUM
		cdf_flat = self.rti_data["cdf_offset"].ravel()
		rand_flat = rand_offset.ravel()

		# 5. Perform one giant binary search
		# This returns the insertion index in the flattened array
		insert_indices = np.searchsorted(cdf_flat, rand_flat)

		# 6. Map back to 2D indices
		# The 'insert_indices' are indices into the FLATTENED cdf.
		# We need to know which column (which choice) that corresponds to.
		# Since cdf_flat is row-major, modulo Max_Choices gives us the column index.
		# However, because we offset the VALUES, not the indices, simply taking modulo
		# of the result index works because 'cdf_flat' is effectively sorted globally.
		idx_selections_flat = insert_indices % self.rti_data["max_choices"]
		# Reshape back to (N_scenarios, MC_NUM)
		idx_selections = idx_selections_flat.reshape(self.rti_data["num_scenarios"], self.MC_NUM)
		# Map indices back to actual POPPIs
		row_indices = np.arange(self.rti_data["num_scenarios"])[:, None]
		selected_poppis = self.rti_data["choices_matrix"][row_indices, idx_selections]

		# --- Construct Output Dictionary ---
		routed_through_ingress = {}
		# We iterate over the scenarios (rows) and their generated results
		for i, (ui, pref_i, ug_name) in enumerate(self.rti_data["meta_data"]):
			simulated_routes = selected_poppis[i] # Array of size MC_NUM
			
			for mci, poppi in enumerate(simulated_routes):
				# Access dictionary structure only once per MC index if possible
				if mci not in routed_through_ingress:
					routed_through_ingress[mci] = {}
				
				mc_dict = routed_through_ingress[mci]
				
				# Map the integer index back to the real object (self.popps)
				real_pop_obj = self.popps[poppi]
				
				# Assign deep in the structure
				if pref_i not in mc_dict:
					mc_dict[pref_i] = {}
				
				mc_dict[pref_i][ug_name] = real_pop_obj
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
	
	def get_ingress_probabilities_and_sim(self, a, verb=False, **kwargs):
		"""
	    Combined and optimized version of get_ingress_probabilities + sim_rti_better.
	    Directly produces the routed_through_ingress dictionary using pattern caching.
	    """
		ts_total = time.time()

		# --- 1. Initialize Containers ---
		# Instead of nested dicts, we build the flat lists required for vectorization directly.
		self.rti_data = {
			"meta_data": [],  # List of tuples: (ui, pref_i, ug_name)
			"all_probs": [],  # List of probability lists: [0.5, 0.5]
			"all_poppis": []  # List of choice lists: [pop_A, pop_B]
		}

		# Ensure persistent cache exists (persist this across function calls)
		if not hasattr(self, 'pattern_cache'):
			self.pattern_cache = {}

		# Local variable speedups to avoid self lookups in loop
		ugs = self.whole_deployment_ugs
		# Assumed to be {ui: [poppi, poppi...]} or {ui: {poppi: data}}
		ui_to_poppi = self.whole_deployment_ui_to_poppi 

		# --- 2. Process Availability Matrix (a) ---
		# Assuming threshold_a logic is effectively: > 1e-6 means active
		a_log = (a > 1e-6) 

		# Iterate over prefixes (columns of a)
		for pref_i in range(a.shape[1]):
			col = a_log[:, pref_i]
			
			# Optimization: If no POPs are active for this prefix, skip entirely
			if not np.any(col):
				continue

			# Create a hashable signature for this availability state
			tloga = tuple(col)

			# --- CACHE CHECK ---
			if tloga in self.pattern_cache:
				# HIT: We have seen this network state before.
				# cached_entries is a list of: (ui, valid_pops_list, probs_list)
				cached_entries = self.pattern_cache[tloga]
				
				# Fast append to master lists
				# We reuse the logic (pops/probs), but update the prefix index (pref_i)
				for ui, pops, probs in cached_entries:
					self.rti_data["meta_data"].append((ui, pref_i, ugs[ui]))
					self.rti_data["all_probs"].append(probs)
					self.rti_data["all_poppis"].append(pops)
				continue

			# --- CACHE MISS: Calculate Logic ---
			# This block only runs when we encounter a UNIQUE network failure state
			
			# 1. Identify active POPs indices
			active_poppis = np.where(col)[0]
			active_poppis_set = set(active_poppis)

			# 2. Identify Blocked (User, Child) pairs due to Active Parents
			# blocked_user_child stores (ui, child_poppi) that are FORBIDDEN
			blocked_user_child = set()
			for ug, child, parent in self.parent_tracker:
				parenti = self.popp_to_ind[parent]
				# If the parent is active in this specific state 'tloga', the child is blocked
				if parenti in active_poppis_set:
					ui = self.whole_deployment_ug_to_ind[ug]
					childi = self.popp_to_ind[child]
					blocked_user_child.add((ui, childi))

			# 3. Build Routing for this State
			entries_for_cache = [] # To store (ui, pops, probs) for future reuse

			for ui in range(self.whole_deployment_n_ug):
				valid_pops = []
				
				# Get potentially available POPs for this user (static config)
				potential_pops = ui_to_poppi[ui]
				
				for poppi in potential_pops:
					# Condition 1: POP must be physically UP
					if poppi not in active_poppis_set:
						continue
					
					# Condition 2: POP must not be blocked by an active parent
					if (ui, poppi) in blocked_user_child:
						continue
					
					valid_pops.append(poppi)

				if not valid_pops:
					continue

				# Compute Uniform Probability
				n = len(valid_pops)
				probs = [1.0 / n] * n
				
				# Append to current run
				self.rti_data["meta_data"].append((ui, pref_i, ugs[ui]))
				self.rti_data["all_probs"].append(probs)
				self.rti_data["all_poppis"].append(valid_pops)

				# Append to Cache
				entries_for_cache.append((ui, valid_pops, probs))

			# Save this state's logic to cache so we never calculate it again for this pattern
			self.pattern_cache[tloga] = entries_for_cache

		self.timing['pmat_organize'] = time.time() - ts_total

		# --- 3. Vectorized Simulation (Previously sim_rti_better) ---
		# Now self.rti_data is fully populated. We proceed with the vectorized selection.

		self.rti_data["num_scenarios"] = len(self.rti_data["all_probs"])
		if self.rti_data["num_scenarios"] == 0:
			return {}

		self.rti_data["max_choices"] = max(len(p) for p in self.rti_data["all_probs"])

		# Create Padded Matrix
		P_matrix = np.zeros((self.rti_data["num_scenarios"], self.rti_data["max_choices"]))
		self.rti_data["choices_matrix"] = np.full((self.rti_data["num_scenarios"], self.rti_data["max_choices"]), -1, dtype=int)

		for i, (probs, pops) in enumerate(zip(self.rti_data["all_probs"], self.rti_data["all_poppis"])):
			n = len(probs)
			P_matrix[i, :n] = probs
			self.rti_data["choices_matrix"][i, :n] = pops

		# CDF Construction
		cdf = np.cumsum(P_matrix, axis=1)
		cdf[:, -1] = 1.0 # Force sum to 1.0 to avoid float precision issues

		# Offset Trick for Vectorized Search
		# Shifts the values of every row so we can search a single flattened array
		offsets = np.arange(self.rti_data["num_scenarios"])
		cdf_offset = cdf + offsets[:, None]

		# Generate Random Numbers
		rand_vals = np.random.rand(self.rti_data["num_scenarios"], self.MC_NUM)
		rand_offset = rand_vals + offsets[:, None]

		# Flatten for searchsorted
		cdf_flat = cdf_offset.ravel()
		rand_flat = rand_offset.ravel()

		# Binary Search (Finds insertion point in flattened CDF)
		insert_indices = np.searchsorted(cdf_flat, rand_flat)

		# Map back to 2D indices
		idx_selections_flat = insert_indices % self.rti_data["max_choices"]
		idx_selections = idx_selections_flat.reshape(self.rti_data["num_scenarios"], self.MC_NUM)

		# Retrieve selected POP indices
		row_indices = np.arange(self.rti_data["num_scenarios"])[:, None]
		selected_poppis = self.rti_data["choices_matrix"][row_indices, idx_selections]

		# --- 4. Construct Final Output Dictionary ---
		routed_through_ingress = {}

		for i, (ui, pref_i, ug_name) in enumerate(self.rti_data["meta_data"]):
			simulated_routes = selected_poppis[i] # Array of size MC_NUM
			
			for mci, poppi in enumerate(simulated_routes):
				if mci not in routed_through_ingress:
					routed_through_ingress[mci] = {}
				
				# Ensure structure exists
				if pref_i not in routed_through_ingress[mci]:
					routed_through_ingress[mci][pref_i] = {}
				
				# Assuming self.popps is a list/dict of actual POP objects
				routed_through_ingress[mci][pref_i][ug_name] = self.popps[poppi]

		self.timing['total_rti_calc'] = time.time() - ts_total

		return routed_through_ingress

	def generic_objective_pdf(self, obj, a, **kwargs):
		"""
			Solves self.MC_NUM traffic assignment problems, assuming that user routes are distributed
			according to distribution self.ingress_probabilities.
		"""

		### TODO -- maybe implement subset of users, but not really essential
		ts = time.time()
		if not TEST_BETTER_VERSION:
			all_routed_through_ingress = self.sim_rti_better()
		else:
			all_routed_through_ingress = self.get_ingress_probabilities_and_sim(a)
		self.timing['sim_rti'] = time.time() - ts
		objs = np.zeros(self.MC_NUM)
		for i in range(self.MC_NUM):
			routed_through_ingress = all_routed_through_ingress[i]
			if obj == "avg_latency" or obj == "per_site_cost":
				ts = time.time()
				total_obj = self.solve_generic_lp_persistent(routed_through_ingress, obj)["objective"]
				self.timing['solve_generic_lp_persistent'] = time.time() - ts
			else:
				ts = time.time()
				total_obj = solve_generic_lp_with_failure_catch(self, routed_through_ingress, obj)['objective']
				self.timing['solve_generic_lp_not_persistent'] = time.time() - ts
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
		if not TEST_BETTER_VERSION:
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
			xsumx, psumx = self.generic_objective_pdf(f_w, a_effective, which_ugs_i=which_ugs_i)
		else:
			xsumx, psumx = self.generic_objective_pdf(f_w, a_effective)

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
		self.pattern_cache = {}
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