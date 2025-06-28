import matplotlib.pyplot as plt, copy, time, numpy as np, itertools, pickle, warnings, tqdm, glob
import gurobipy as gp
from subprocess import call, check_output
np.setbufsize(262144*8)
np.set_printoptions(precision=3)
# np.random.seed(31415)
# import random
# random.seed(31416)
import scipy.stats
import sys
# from sklearn.mixture import GaussianMixture
# from sklearn.exceptions import ConvergenceWarning
# import warnings
np.set_printoptions(threshold=sys.maxsize)
from helpers import *
from constants import *
from painter import Painter_Adv_Solver
from anyopt import Anyopt_Adv_Solver
from optimal_adv_wrapper import Optimal_Adv_Wrapper
from worker_comms import Worker_Manager
from generic_objective import Generic_Objective

try:
	from eval_latency_failure import plot_lats_from_adv
except ImportError:
	pass

from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning

from deployment_setup import *


def violates(ordering, bigger, smallers):
	# ordering - list of ingresses
	# bigger - ingress that won
	# smallers - ingresses that lost

	# returns True if bigger comes after any of the smallers in the ordering

	if bigger not in ordering: return False

	ordering = np.array(ordering)
	smallers = np.array(smallers)
	wb = np.where(bigger == ordering)[0]
	for s in get_intersection(smallers, ordering):
		if wb > np.where(ordering == s)[0]:
			return True
	return False


def compare_estimated_actual_per_user(dpsize):
	modeled_user_lats = {}
	for worker_log in glob.glob(os.path.join(LOG_DIR, 'worker*log-{}.txt'.format(dpsize))):
		for row in open(worker_log,'r'):
			if 'benefit_estimate' not in row: continue
			_,_,ui,bi,lb,p,popps_str,itr = row.strip().split(',')
			itr,ui,bi,lb,p = int(itr),int(ui),int(bi),float(lb),float(p)
			try:
				modeled_user_lats[itr]
			except KeyError:
				modeled_user_lats[itr] = {}
			try:
				modeled_user_lats[itr][ui].append((bi,lb,p))
			except KeyError:
				modeled_user_lats[itr][ui] = [(bi,lb,p)]
	actual_user_lats = {}
	for row in open(os.path.join(LOG_DIR, 'main_thread_log-{}.txt'.format(dpsize)),'r'):
		if 'benefit_estimate' not in row: continue
		_,itr,ui,poppi,pct,b = row.strip().split(',')
		itr,ui,poppi,pct,b = int(itr),int(ui),int(poppi),float(pct),float(b)
		try:
			actual_user_lats[itr]
		except KeyError:
			actual_user_lats[itr] = {}
		try:
			actual_user_lats[itr][ui].append((poppi,b,pct))
		except KeyError:
			actual_user_lats[itr][ui] = [(poppi,b,pct)]

	itrs = list(sorted(list(actual_user_lats)))
	uis = list(sorted(list(actual_user_lats[itrs[0]])))

	plt.rcParams["figure.figsize"] = (8,6)
	f,ax = plt.subplots()
	current_itr = itrs[-1]
	all_deltas = np.zeros((len(uis), len(itrs)))
	for ui in uis:
		for itr in itrs:
			modeled_user_lats[itr][ui] = sum(lb*p for _,lb,p in set(modeled_user_lats[itr][ui]))
			actual_user_lats[itr][ui] = sum(lb*p for _,lb,p in set(actual_user_lats[itr][ui]))
			if np.abs(modeled_user_lats[itr][ui] - actual_user_lats[itr][ui]) > .3 and itr == current_itr:
				print("ITR: {} User {} has modeled benefit {} but actual {}".format(itr, ui, modeled_user_lats[itr][ui], actual_user_lats[itr][ui]))
		these_modeled_lats = np.array([modeled_user_lats[itr][ui] for itr in itrs])
		these_actual_lats = np.array([actual_user_lats[itr][ui] for itr in itrs])
		deltas = np.abs(these_actual_lats - these_modeled_lats)
		all_deltas[ui,:] = deltas


	ax.plot(itrs, np.min(all_deltas, axis=0),label='Min')
	ax.plot(itrs, np.median(all_deltas, axis=0),label='Median')
	ax.plot(itrs, np.max(all_deltas, axis=0),label='Max')
	ax.legend()

	ax.set_xlabel("Iteration")
	ax.set_ylabel("Actual - Modeled Benefit")
	plt.savefig("figures/benefit_modeling_error.pdf")
	plt.clf()
	plt.close()

def investigate_congestion_events():
	return
	import glob
	link_failure_events = {}
	for worker_log in glob.glob(os.path.join(CACHE_DIR, 'worker*log.txt')):
		for row in open(worker_log,'r'):
			if 'link_fail_report' not in row: continue
			_,itr,ingress_i,failing_poppi,link_cap,vol_users,uis,p_fails = row.strip().split(',')
			if failing_poppi != 'none':
				failing_poppi = int(failing_poppi)
			itr,ingress_i,link_cap,vol_users,p_fails = int(itr),int(ingress_i),float(link_cap),float(vol_users),float(p_fails)
			uis = [int(el) for el in uis.split('-')]

			uid = (ingress_i,failing_poppi)

			try:
				link_failure_events[uid].append((itr, vol_users - link_cap))
			except KeyError:
				link_failure_events[uid] = [(itr,vol_users - link_cap)]
			max_itr = itr
	# for row in open(os.path.join(CACHE_DIR, 'main_thread_log.txt'),'r'):
	# 	if 'link_fail_report' not in row: continue
	# 	_,itr,ingress_i,failing_poppi,link_cap,vol_users,uis,p_fails = row.strip().split(',')
	# 	if failing_poppi != 'none':
	# 		failing_poppi = int(failing_poppi)
	# 	itr,ingress_i,link_cap,vol_users,p_fails = int(itr),int(ingress_i),float(link_cap),float(vol_users),float(p_fails)
	# 	uis = [int(el) for el in uis.split('-')]

	# 	uid = (ingress_i,failing_poppi)

	# 	try:
	# 		link_failure_events[uid].append((itr, vol_users - link_cap))
	# 	except KeyError:
	# 		link_failure_events[uid] = [(itr,vol_users - link_cap)]
	# 	max_itr = itr

	plt.rcParams["figure.figsize"] = (8,6)
	f,ax = plt.subplots()
	for uid in link_failure_events:
		x,y = [el[0] for el in link_failure_events[uid]],\
			[el[1] for el in link_failure_events[uid]]
		if max_itr - x[-1] < 5:
			# only label recent issues
			ax.plot(x,y,label='{} over, {} fails'.format(uid[0],uid[1]))
		else:
			ax.plot(x,y)

	ax.set_xlabel("Iteration")
	ax.set_ylabel("Excess Link Load")
	# ax.legend(fontsize=6)
	plt.savefig("figures/reported_failure_events_during_training.pdf")
	plt.clf()
	plt.close()

class Sparse_Advertisement_Wrapper(Optimal_Adv_Wrapper):
	def __init__(self, *args, init={'type':'using_objective'}, explore='bimodality',
			using_resilience_benefit=False, **kwargs):
		super().__init__(*args, **kwargs)
		# (hyper-) parameters

		self.iter = 0
		self.initialization = init
		self.explore = explore
		self.stopping_condition = lambda el : el[0] > self.max_n_iter or (el[3] < self.rolling_adv_eps and el[1] < self.epsilon and np.abs(el[2]) < self.epsilon)
		## Whether to incorporate capacity into the objective function
		self.with_capacity = kwargs.get('with_capacity', False)
		### We might vary these functions depending on settings from time to time
		### but always aim to unify them after dev
		self.latency_benefit_fn = self.latency_benefit
		self.gradient_fn = self.gradients
		## Whether to incorporate resilience into the objective function
		# (Note if gamma = 0, this won't matter anyway)
		self.using_resilience_benefit = using_resilience_benefit
		if using_resilience_benefit:
			assert self.gamma > 0
			self.resilience_benefit_fn = self.resilience_benefit
			self.gradients_resilience_benefit_fn = self.gradients_resilience_benefit
		else:
			self.resilience_benefit_fn = lambda a : 0
			self.gradients_resilience_benefit_fn = lambda a : np.zeros(a.shape)

		self.proximal = True

		self.reset_metrics()
		if self.verbose:
			print("Creating problem with {} peers, {} prefixes, {} UGs.".format(self.n_popp, self.n_prefixes, len(self.ugs)))

		## Queue up calls to individual workers
		self.lb_args_queue = []

	def get_init_kwa(self):
		kwa =  {
			'lambduh': self.lambduh, 
			'gamma': self.gamma, 
			'with_capacity': self.with_capacity,
			'verbose': False,
			'init': self.initialization,
			'explore': self.explore,
			'using_resilience_benefit': self.using_resilience_benefit,
			'n_prefixes': self.n_prefixes,
			'save_run_dir': self.save_run_dir,
		}
		if self.using_generic_objective:
			kwa['generic_objective'] = self.generic_objective.obj
		return kwa

	def reset_metrics(self):
		# For analysis
		self.metrics = {}
		for k in ['actual_nonconvex_objective', 'advertisements', 'effective_objectives', 
			'pseudo_objectives', 'grads', 'cost_grads', 'l_benefit_grads', 'res_benefit_grads',
			'path_likelihoods', 'EL_difference', 'resilience_benefit', 'latency_benefit',
			'gt_latency_benefit','gt_resilience_benefit', 'effective_gammas', 'link_utilizations', 'frac_latency_benefit_calls', 'frac_resilience_benefit_calls',
				'popp_rb_sample_probabilities']:
			self.metrics[k] = []

	def gradients(self, *args, **kwargs):
		pass

	def gradients_resilience_benefit(self,*args, **kwargs):
		pass

	def get_n_most_likely_peers_justsort(self, ug, available_peers, n=5, verb=False):
		sorted_available_peers = sorted(available_peers, key = rank_f)
		return sorted_available_peers[0:n]

	def compress_lb_args_queue(self, **kwargs):
		### Idea: first adv is the base, rest are deltas from the base
		### transmit the base and the deltas
		ugs = kwargs.get('ugs', None)
		is_verb = kwargs.get('verbose_workers', False)
		base_args, base_kwa = self.lb_args_queue[0]
		if ugs is not None:
			base_kwa['ugs'] = ugs
		base_kwa['verbose_workers'] = is_verb or base_kwa.get('verbose_workers',False)
		if self.using_generic_objective:
			base_kwa['generic_obj'] = self.generic_objective.obj
		base_adv, = base_args

		base_adv = threshold_a(base_adv)
		base_args = (base_adv,)

		self.compressed_lb_args_queue = [(base_args, base_kwa)]
		for other_args, kwa in self.lb_args_queue[1:]:
			other_adv, = other_args
			other_adv = threshold_a(other_adv)
			if ugs is not None:
				kwa['ugs'] = ugs
			kwa['verbose_workers'] = is_verb or kwa.get('verbose_workers',False)
			if self.using_generic_objective:
				kwa['generic_obj'] = self.generic_objective.obj
			self.compressed_lb_args_queue.append((np.where(base_adv!=other_adv), kwa))

	def flush_latency_benefit_queue_generic(self, **kwargs):
		"""
			For the generic objective, it makes more sense to split jobs among workers rather than
			create smaller jobs for each worker. So, easier splitting process here.
			But we need to do a slightly custom args creation process.
		"""

		### Idea: first adv is the base, rest are deltas from the base
		### transmit the base and the deltas
		n_workers = min(self.get_n_workers(), len(self.lb_args_queue))

		for i,(a,kwa) in enumerate(self.lb_args_queue):
			kwa['job_id'] = i

		ugs = kwargs.get('ugs', None)
		is_verb = kwargs.get('verbose_workers', False)
		base_args, base_kwa = self.lb_args_queue[0]
		if ugs is not None:
			base_kwa['ugs'] = ugs
		base_kwa['verbose_workers'] = is_verb or base_kwa.get('verbose_workers',False)
		base_kwa['generic_obj'] = self.generic_objective.obj
		base_adv, = base_args

		base_adv = threshold_a(base_adv)
		base_args = (base_adv,)

		all_worker_jobs_seq = split_seq(self.lb_args_queue[1:], n_workers)
		
		all_workers_jobs = [[(base_args, base_kwa)] for _ in range(n_workers)]

		for i,job_set in enumerate(all_worker_jobs_seq):
			for other_args, kwa in job_set:
				other_adv, = other_args
				other_adv = threshold_a(other_adv)
				if ugs is not None:
					kwa['ugs'] = ugs
				kwa['verbose_workers'] = is_verb or kwa.get('verbose_workers',False)
				kwa['generic_obj'] = self.generic_objective.obj
				all_workers_jobs[i].append((np.where(base_adv!=other_adv), kwa))

		msgs = list([pickle.dumps(['calc_compressed_lb', subset]) for subset in all_workers_jobs])
		# print(list([['calc_compressed_lb', subset] for subset in all_workers_jobs]))
		rets = self.worker_manager.send_receive_messages_workers(msgs, n_workers=n_workers)
		
		### just append all the jobs, in order. it's important that these things happen in order
		### since that's how we ID the job
		n_to_flush = len(self.lb_args_queue)
		ret_to_call = [None for _ in range(n_to_flush)]

		all_rets = []
		for worker_i in range(n_workers):
			if worker_i > 0:
				all_rets = all_rets + rets[worker_i][1:]
			else: ## get the base answer from worker 0
				all_rets = all_rets + rets[worker_i]
		for adv_ret_i,ret in enumerate(all_rets):
			# print(x.flatten())
			# print(px.flatten())
			# print(mean)
			mean,(x,px) = ret['ans']
			ret_to_call[adv_ret_i] = (mean, (x.flatten(), px.flatten()))

		self.lb_args_queue = []
		self.get_cache()
		# if len(all_rets) > 1:
		# 	exit(0)
		return ret_to_call

	def flush_latency_benefit_queue(self, **kwargs):

		if self.using_generic_objective:
			return self.flush_latency_benefit_queue_generic(**kwargs)

		self.compress_lb_args_queue(**kwargs)

		msg = pickle.dumps(['calc_compressed_lb', self.compressed_lb_args_queue])
		rets = self.send_receive_workers(msg)
		n_workers = len(rets) 
		### combine pdf rets across sub-deployments
		n_to_flush = len(self.lb_args_queue)
		ret_to_call = [None for _ in range(n_to_flush)]
		pdfs = np.zeros((n_to_flush, LBX_DENSITY, n_workers))
		lbxs = np.zeros((n_to_flush, LBX_DENSITY, n_workers))
		vals_by_worker = {}
		for worker_i,ret in enumerate(rets.values()): # n workers times
			vals_by_worker[worker_i] = {}
			for adv_ret_i in range(n_to_flush): # n calls times
				mean, (vals,pdf) = ret[adv_ret_i]['ans']
				lbxs[adv_ret_i, :, worker_i] = vals
				pdfs[adv_ret_i, :, worker_i] = pdf

		### Convert all pdfs to be at the same scale
		GLOBAL_LBX_DENSITY = 10 * LBX_DENSITY
		inds = np.arange(LBX_DENSITY)
		new_lbxs = np.zeros((n_to_flush,GLOBAL_LBX_DENSITY,n_workers))
		new_pdfs = np.zeros((n_to_flush,GLOBAL_LBX_DENSITY,n_workers))
		for adv_ret_i in range(n_to_flush):
			new_max, new_min = np.max(lbxs[adv_ret_i,:,:].flatten()), np.min(lbxs[adv_ret_i,:,:].flatten())
			if new_max == new_min: # trivial
				new_min = new_max - 1
			for worker_i in range(n_workers):
				old_min, old_max = lbxs[adv_ret_i,0,worker_i],lbxs[adv_ret_i,-1,worker_i]
				rescaled_pdf = np.zeros((GLOBAL_LBX_DENSITY,))
				remap_arr = (old_min + inds * (old_max - old_min) / LBX_DENSITY - new_min) * GLOBAL_LBX_DENSITY / (new_max - new_min)
				remap_arr = np.round(remap_arr).astype(np.int32).clip(0,GLOBAL_LBX_DENSITY-1)
				for ind in np.where(pdfs[adv_ret_i,:,worker_i] > 0)[0]:
					rescaled_pdf[remap_arr[ind]] += pdfs[adv_ret_i,ind,worker_i]
				new_lbxs[adv_ret_i,:,worker_i] = np.linspace(new_min, new_max, GLOBAL_LBX_DENSITY)
				new_pdfs[adv_ret_i,:,worker_i] = rescaled_pdf

		for adv_ret_i in range(n_to_flush):
			## point density x number of cores
			if n_workers > 1:
				x,px = self.pdf_sum_function(new_lbxs[adv_ret_i,...],new_pdfs[adv_ret_i,...],**kwargs)
			else:
				x,px = np.expand_dims(new_lbxs[adv_ret_i,:,0],axis=1), np.expand_dims(new_pdfs[adv_ret_i,:,0],axis=1)
			mean = np.sum(px.flatten()*x.flatten())
			ret_to_call[adv_ret_i] = (mean, (x.flatten(), px.flatten()))

		self.lb_args_queue = []
		self.get_cache()
		return ret_to_call

	def latency_benefit(self, *args, **kwargs):
		self.lb_args_queue.append((copy.deepcopy(args), copy.deepcopy(kwargs)))
		if kwargs.get('retnow', False): # we want an immediate calculation
			return self.flush_latency_benefit_queue(**kwargs)[0]

	def get_gamma(self):
		return self.gamma

	def resilience_benefit(self, a, **kwargs):
		""" sum over peers of E(delta benefit when that peer is knocked out)."""
		# want to maximize resilience beneift, so want to maximize new benefits
		# when peers are knocked out
		if not self.simulated:
			return 0
		tmp = np.ones(a.shape)
		cpkwargs = copy.deepcopy(kwargs)
		cpkwargs['retnow'] = False
		self.latency_benefit_fn(a, **cpkwargs)
		for popp in self.popps:
			# we don't know for sure where users are going
			# so we have to compute over all users
			tmp[self.popp_to_ind[popp],:] = 0
			cpkwargs['failing_popp'] = popp
			self.latency_benefit_fn(a * tmp, **cpkwargs)
			tmp[self.popp_to_ind[popp],:] = 1
		rets = self.flush_latency_benefit_queue()

		benefit = 0
		for b,_ in rets[1:]:
			benefit += b

		return benefit

	def init_advertisement(self):
		print("Initializing advertisement...")
		mode = self.initialization['type']
		if mode == 'random_binary':
			return np.random.randint(0,2,size=(self.n_popp, self.n_prefixes)) * 1.0
		elif mode == 'normal':
			return ADVERTISEMENT_THRESHOLD+ np.sqrt(self.initialization['var']) \
				* np.random.normal(size=(self.n_popp, self.n_prefixes))
		elif mode == 'ones':
			return np.ones((self.n_popp, self.n_prefixes))
		elif mode == 'zeros':
			return np.zeros((self.n_popp, self.n_prefixes))
		elif mode == 'uniform':
			return np.random.random(size=(self.n_popp, self.n_prefixes))
		elif mode == 'pseudo_anycast':
			a = np.zeros((self.n_popp, self.n_prefixes))
			a[:,0] = 1
			return a
		if mode == 'using_objective':
			### idea ~ 1 anycast prefix
			### 1 prefix motivated by objective
			### rest set completely randomly near .5, but with aim of lambduh * norm penalty = LB
			### expected latency benefit is roughly -1 * (MAX_LATENCY - MIN_LATENCY) / 2
			### so number of entries on should be (MAX_LATENCY - MIN_LATENCY) / 2 out of lambduh * n_popp * (n_pref - 2)
			### max of .05


			# everything off, to start, with some jitter
			a = .35 * np.ones((self.n_popp, self.n_prefixes)) + (.2 * (np.random.uniform(size=(self.n_popp, self.n_prefixes)) - .5 ))
			a[:,0] = .55 # anycast on the first prefix
			for i in range(self.n_pops):
				these_popps = np.array([self.popp_to_ind[popp] for popp in self.popps if popp[0] == self.pops[i]])
				a[these_popps,i+1] = .55
			## linear decrease to the end
			start_ind = self.n_pops + 1
			prob_ons = np.linspace(.05,.005,num=(self.n_prefixes-start_ind))
			for i in range(self.n_prefixes-start_ind):
				prob_on = prob_ons[i]
				is_on = np.random.random(size=(self.n_popp)) < prob_on
				a[is_on,start_ind+i] = .55
			a += .02 * (np.random.uniform(size=a.shape) - .5) # noise
			print("Done Initializing")
			print("Initial numbers of popps on per prefix.")
			print(np.sum(threshold_a(a),axis=0))

			# self.solve_lp_assignment(threshold_a(a))
			return a
		else:
			raise ValueError("Adv init {} not recognized.".format(mode))

	def modeled_objective(self, a, **kwargs):
		"""Approx actual objective with our belief."""
		if self.verbose:
			print("Calculating modeled objective")
		norm_penalty = self.advertisement_cost(a)
		kwargs['retnow'] = True
		latency_benefit, u = self.latency_benefit_fn(threshold_a(a), **kwargs)

		if self.using_resilience_benefit:
			resilience_benefit = self.resilience_benefit_fn(a, **kwargs)
		else:
			resilience_benefit = 0

		if self.verbose:
			benefits,probs = u
			ex = np.average(benefits,weights=probs+1e-8)
			exsq = np.average(np.power(benefits,2),weights=probs+1e-8)
			var = exsq - np.power(ex,2)
			std = np.sqrt(var)
			print("Believed: NP: {}, LB: {} ({} std dev), RB: {}".format(round(norm_penalty,3),
				round(latency_benefit,3), round(std,3), round(resilience_benefit,3)))

		# gamma = self.get_gamma()
		gamma = self.gamma
		if gamma <= 1:
			benefit = latency_benefit + gamma * resilience_benefit
		else:
			benefit = 1 / gamma * latency_benefit + resilience_benefit

		return self.lambduh * norm_penalty - (benefit)

class Sparse_Advertisement_Eval(Sparse_Advertisement_Wrapper):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def solve_anycast(self, verbose=True, **kwargs):

		self.solution_type = 'sparse' ## just use sparse measurements
		self.get_realworld_measure_wrapper()

		## Simple anycast
		anycast_advertisement = np.zeros((self.n_popp, self.n_prefixes))
		anycast_advertisement[:,0] = 1

		optimization_advertisement_representation = {}
		for poppi,prefi in zip(*np.where(threshold_a(anycast_advertisement))):
			optimization_advertisement_representation[self.popps[poppi], prefi] = None

		self.solutions['anycast'] = {
			'objective': self.measured_objective(anycast_advertisement),
			'advertisement': anycast_advertisement,
			'advertisement_representation': optimization_advertisement_representation,
			'latency_benefit':  self.get_ground_truth_latency_benefit(anycast_advertisement),
			'prefix_cost': self.prefix_cost(anycast_advertisement),
			'norm_penalty': self.advertisement_cost(anycast_advertisement),
			'n_advs': 1,
		}

		self.solution_type = "None"
		self.get_realworld_measure_wrapper()

		return "ok"

	def solve_one_per_peering(self, **kwargs):
		## One per peering
		one_per_peering_adv = np.eye(self.n_popps)
		optimization_advertisement_representation = {}
		for poppi,prefi in zip(*np.where(threshold_a(one_per_peering_adv))):
			optimization_advertisement_representation[self.popps[poppi], prefi] = None
		self.solutions['one_per_peering'] = {
			'objective': self.measured_objective(one_per_peering_adv),
			'advertisement': one_per_peering_adv,
			'latency_benefit':  self.get_ground_truth_latency_benefit(one_per_peering_adv),
			'prefix_cost': self.prefix_cost(one_per_peering_adv),
			'norm_penalty': self.advertisement_cost(one_per_peering_adv),
			'advertisement_representation': optimization_advertisement_representation,
			'n_advs': 1,
		}

		return "ok"

	def solve_random(self, verbose=True, **kwargs):
		# To get the random solution, try every possible combination of advertisements
		# Not possible for problems that are too large
		n_arr = self.n_popp * self.n_prefixes
		logn_possibilities = n_arr
		all_as = []
		
		n_possibilities = int(10)
		objs = np.zeros((n_possibilities,))
		actual_objs = np.zeros((n_possibilities,))
		vcpy = copy.copy(self.verbose)
		self.verbose = False

		for i in tqdm.tqdm(range(n_possibilities), desc="Finding a decent random solution."):
			a = np.random.randint(0,high=2,size=(self.n_popp,self.n_prefixes))
			objs[i] = self.measured_objective(a)
			actual_objs[i] = self.actual_nonconvex_objective(a)
			all_as.append(copy.deepcopy(a))
		self.verbose = vcpy

		# Approx according to L1 norm
		random_objective = np.min(objs)
		approx_random_adv = all_as[np.argmin(objs)].reshape(self.n_popp, self.n_prefixes)

		# Actual
		actual_random_objective = np.min(actual_objs)
		l0_random_adv = all_as[np.argmin(actual_objs)].reshape(self.n_popp, self.n_prefixes)

		self.solutions['random'] = {
			'objective': random_objective,
			'advertisement': approx_random_adv,
			'latency_benefit':  self.get_ground_truth_latency_benefit(approx_random_adv),
			'prefix_cost': self.prefix_cost(approx_random_adv),
			'norm_penalty': self.advertisement_cost(approx_random_adv),
			'n_advs': 1,
		}

		return "ok"

	def solve_anyopt(self, **kwargs):
		deployment = self.output_deployment()
		self.anyopt = Anyopt_Adv_Solver(deployment, **self.get_init_kwa())
		self.anyopt.set_worker_manager(self.get_worker_manager())
		self.anyopt.solve()

		anyopt_adv = self.anyopt.advs
		anyopt_obj = self.measured_objective(anyopt_adv)

		self.solutions['anyopt'] = {
			'objective': anyopt_obj,
			'advertisement': anyopt_adv,
			'latency_benefit':  self.get_ground_truth_latency_benefit(anyopt_adv),
			'prefix_cost': self.prefix_cost(anyopt_adv),
			'norm_penalty': self.advertisement_cost(anyopt_adv),
			'advertisement_representation': self.anyopt.optimization_advertisement_representation,
			'n_advs': self.anyopt.path_measures,
		}
		self.clear_caches()

	def solve_sparse(self, **kwargs):
		deployment = self.output_deployment()
		self.sas = Sparse_Advertisement_Solver(deployment, **self.get_init_kwa())
		self.sas.set_worker_manager(self.get_worker_manager())
		self.sas.compute_one_per_peering_solution()

		try:
			self.sas.painter_solution = self.solutions['painter']
		except KeyError:
			pass
		self.sas.verbose=True
		self.sas.solve(**kwargs)
		try:
			self.sas.make_plots()
		except:
			pass
		final_adv = self.sas.optimization_advertisement
		self.sas.reset_metrics()
		self.sas.metrics['advertisements'].append(final_adv)


		sparse_adv = threshold_a(final_adv)
		sparse_objective = self.sas.measured_objective(sparse_adv)
		print("In outer LB {}".format(self.sas.get_ground_truth_latency_benefit(sparse_adv)))
		# pickle.dump(self.sas.output_deployment(), open('saving_working_sparse_deployment.pkl','wb'))
		self.solutions['sparse'] = {
			'objective': sparse_objective,
			'latency_benefit':  self.sas.get_ground_truth_latency_benefit(sparse_adv),
			'norm_penalty': self.sas.advertisement_cost(sparse_adv),
			'prefix_cost': self.sas.prefix_cost(sparse_adv),
			'advertisement': sparse_adv,
			'advertisement_representation': self.sas.optimization_advertisement_representation,
			'n_advs': self.sas.path_measures,
		}

		self.clear_caches()

	def solve_one_per_pop(self, **kwargs):
		# Solve for the one per pop solution
		deployment = self.output_deployment()
		self.one_per_pop = Painter_Adv_Solver(deployment, **self.get_init_kwa())
		self.one_per_pop.set_worker_manager(self.get_worker_manager())

		self.one_per_pop.one_per_pop()
		one_per_pop_adv = self.one_per_pop.painter_advs_to_sparse_advs(self.one_per_pop.advs)
		one_per_pop_obj = self.one_per_pop.measured_objective(one_per_pop_adv)
		self.solutions['one_per_pop'] = {
			'objective': one_per_pop_obj,
			'latency_benefit':  self.one_per_pop.get_ground_truth_latency_benefit(one_per_pop_adv),
			'norm_penalty': self.one_per_pop.advertisement_cost(one_per_pop_adv),
			'prefix_cost': self.one_per_pop.prefix_cost(one_per_pop_adv),
			'advertisement': one_per_pop_adv,
			'advertisement_representation': self.one_per_pop.optimization_advertisement_representation,
			'n_advs': 1,
		}

		self.clear_caches()

	def solve_painter(self, **kwargs):
		## Solve for the painter solution
		# painter is an improvement over anycast, so it has one less prefix to work with
		# and its assumed the first column will be anycast
		deployment = self.output_deployment()
		self.painter = Painter_Adv_Solver(deployment, **self.get_init_kwa())
		self.painter.set_worker_manager(self.get_worker_manager())

		self.painter.painter_v5(cd=5000)
		painter_adv = self.painter.painter_advs_to_sparse_advs(self.painter.advs)
		print('doing painter')
		painter_obj = self.painter.measured_objective(painter_adv)

		# print("Painter Adv, obj: {} {}".format(painter_adv, painter_obj))
		self.solutions['painter'] = {
			'objective': painter_obj,
			'latency_benefit':  self.painter.get_ground_truth_latency_benefit(painter_adv),
			'norm_penalty': self.painter.advertisement_cost(painter_adv),
			'prefix_cost': self.painter.prefix_cost(painter_adv),
			'advertisement': painter_adv,
			'advertisement_representation': self.painter.optimization_advertisement_representation,
			'n_advs': self.painter.path_measures,
		}

		self.clear_caches()

	def painter_objective(self, a, **kwargs):
		## Improvement over anycast
		user_latencies = self.get_ground_truth_user_latencies(a, **kwargs)
		improves = np.array([self.ug_anycast_perfs[ug] - user_latencies[self.ug_to_ind[ug]] for \
			ug in self.ugs])
		mean_improve = np.sum(improves * self.ug_vols) / np.sum(self.ug_vols)
		return -1 * mean_improve

	def anyopt_objective(self, a):
		## Latency benefit
		return -1 * self.get_ground_truth_latency_benefit(a)

	def compare_different_solutions(self, **kwargs):
		verbose = kwargs.get('verbose', True)
		
		if kwargs.get('soln_types') is not None:
			solution_types = kwargs.get('soln_types')
		else:
			solution_types = ['anyopt', 'painter', 'sparse', 'one_per_pop', 'anycast', 'random', 'one_per_peering']
		metrics = {
			'sparse_objective_vals': {k:[] for k in solution_types},
			'painter_objective_vals': {k:[] for k in solution_types},
			'anyopt_objective_vals': {k:[] for k in solution_types},
			'normalized_sparse_benefit': {k:[] for k in solution_types},
			'latency_benefits': {k: [] for k in solution_types},
			'norm_penalties': {k: [] for k in solution_types},
			'prefix_cost': {k: [] for k in solution_types},
			'objective_diffs': {k:[] for k in solution_types},
			'latency_benefit_diffs': {k:[]for k in solution_types},
			'n_advs': {k:[] for k in solution_types},
			'adv_solns': {k:[] for k in solution_types},
			'adv_representation_solns': {k:[] for k in solution_types},
			'max_sparse_benefits': [],
			'max_painter_benefits': [],
		}
		solve_fns = {'sparse': self.solve_sparse, 'painter': self.solve_painter, 'anyopt': self.solve_anyopt,
			'one_per_pop': self.solve_one_per_pop, 'anycast': self.solve_anycast, 'random': self.solve_random,
			'one_per_peering': self.solve_one_per_peering,}
		self.solutions = {}
		if not self.simulated:
			self.get_realworld_measure_wrapper()
		for i in range(kwargs.get('n_run', 50)):
			if verbose:
				print("Comparing different solutions iteration {}".format(i))
			for solution_type in solution_types:
				if verbose:
					print("\n---solving {}---\n".format(solution_type))
				if not self.simulated:
					try:
						solve_fns[solution_type](**kwargs)
						metrics['sparse_objective_vals'][solution_type].append(self.solutions[solution_type]['objective'])
						metrics['n_advs'][solution_type].append(self.solutions[solution_type]['n_advs'])
						metrics['adv_solns'][solution_type].append(self.solutions[solution_type]['advertisement'])
						metrics['adv_representation_solns'][solution_type].append(self.solutions[solution_type]['advertisement_representation'])
						metrics['latency_benefits'][solution_type].append(self.solutions[solution_type]['latency_benefit'])
						metrics['norm_penalties'][solution_type].append(self.solutions[solution_type]['norm_penalty'])
						metrics['prefix_cost'][solution_type].append(self.solutions[solution_type]['prefix_cost'])


						# adv = self.solutions[solution_type]['advertisement']
						# metrics['painter_objective_vals'][solution_type].append(self.painter_objective(adv))
						# metrics['anyopt_objective_vals'][solution_type].append(self.anyopt_objective(adv))
					except:
						import traceback
						traceback.print_exc()
				else:
					solve_fns[solution_type](**kwargs)
					metrics['sparse_objective_vals'][solution_type].append(self.solutions[solution_type]['objective'])
					metrics['n_advs'][solution_type].append(self.solutions[solution_type]['n_advs'])
					metrics['adv_solns'][solution_type].append(self.solutions[solution_type]['advertisement'])
					metrics['latency_benefits'][solution_type].append(self.solutions[solution_type]['latency_benefit'])
					metrics['norm_penalties'][solution_type].append(self.solutions[solution_type]['norm_penalty'])
					metrics['prefix_cost'][solution_type].append(self.solutions[solution_type]['prefix_cost'])


					adv = self.solutions[solution_type]['advertisement']
					metrics['painter_objective_vals'][solution_type].append(self.painter_objective(adv))
					metrics['anyopt_objective_vals'][solution_type].append(self.anyopt_objective(adv))

			if not kwargs.get('dont_update_deployment', False):
				## Update to new random deployment
				new_deployment = get_random_deployment(self.dpsize)
				self.update_deployment(new_deployment)
			if verbose:
				print(metrics['sparse_objective_vals'])

		return metrics

class Sparse_Advertisement_Solver(Sparse_Advertisement_Wrapper):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.solution_type = 'sparse'
		self.beta = .3 # gradient momentum
		self.sigmoid_k = 5.0 # heavisside gradient parameter

		self.gradient_support = [(a_i,a_j) for a_i in range(self.n_popp) for a_j in range(self.n_prefixes)]
		if self.simulated:
			self.gradient_support_settings = {
				'lb_support_size': 20*self.n_pops,
				'popp_rb_support_size': 60*self.n_pops,
				'info_support_size': 5*self.n_pops,
			}
			if self.gamma == 0:
				self.gradient_support_settings['lb_support_size'] *= 4
		else:
			## we are severely rate limited by measurement speed, so we should aim to compute as much as possible
			self.gradient_support_settings = {
				'lb_support_size': int(.3*(self.n_popps * self.n_prefixes)),
				'popp_rb_support_size': int(.5*(self.n_popps * self.n_prefixes)),
				'info_support_size': 10*self.n_pops,
			}

		self.uncertainty_factor = 10
		self.n_max_info_iter = 1

		self.optimization_var_names = ['rolling_delta', 'rolling_delta_eff', 'rolling_adv_delta', 'rolling_adv_eps', 'last_objective',
			'current_pseudo_objective', 'iter', 'uncertainty_factor', 'stop', 'alpha', 'path_measures', 'current_effective_objective',
			'current_objective', 'calc_times', 'current_latency_benefit', 'current_resilience_benefit']
		if self.simulated:
			self.save_state_every = 20 # how often to save our optimization state
		else:
			self.save_state_every = 1

	def apply_prox_l1(self, w_k):
		"""Applies proximal gradient method to updated variable. Proximal gradient
			for L1 norm is a soft-threshold at the learning rate."""
		return np.sign(w_k) * np.maximum(np.abs(w_k) - self.lambduh * self.alpha, np.zeros(w_k.shape))

	def heaviside_gradient(self, before, after, a_ij):
		# Gradient of sigmoid function
		# when a_ij goes from zero to one, latency benefit value goes from before to after
		# we approx. that as the continuous function before + (after - before) / (1 + exp(-k * a_ij))
		# return the derivative of this function evaluated at a_ij
		x = a_ij - ADVERTISEMENT_THRESHOLD
		return (after - before) * self.sigmoid_k * np.exp(-self.sigmoid_k * x) / (1 + np.exp(-self.sigmoid_k * x))**2

	def heaviside_gradient_sigmoid(self, a):
		x = a - ADVERTISEMENT_THRESHOLD
		grad = self.sigmoid_cost_k * np.exp(-self.sigmoid_cost_k*x) / (1 + np.exp(-self.sigmoid_cost_k*x))**2
		return grad

	def get_last_advertisement(self):
		return self.metrics['advertisements'][-1]

	def get_last_objective(self, effective=False):
		if effective:
			return self.measured_objective(threshold_a(self.get_last_advertisement()))
		else:
			return self.measured_objective(self.get_last_advertisement())
	
	def gradients_latency_benefit(self, a):
		L_grad = np.zeros(a.shape)
		a_effective = threshold_a(a).astype(bool)


		total_n_grad_calc = self.gradient_support_settings['lb_support_size']
		
		pct_explore = 60 # pct of gradient calculation budget dedicated to exploring
		N_EXPLORE = int(total_n_grad_calc * pct_explore/100)
		# number of gradient calcs that re-calc previously high gradients
		N_REMEASURE = total_n_grad_calc - N_EXPLORE

		calls = []
		n_significant = 0
		try:
			best_from_last_time = sorted(self.last_lb_calls_results.items(), key = lambda el : 
				-1 * np.abs(el[1]))
			for ind,val in best_from_last_time:
				if (ind,'ba') in calls or (ind,'ab') in calls: 
					continue
				if np.abs(val) < self.lambduh or np.abs(val) < .01:
					# if it's not important enough to warrant the cost, don't bother
					continue
				if np.abs(ADVERTISEMENT_THRESHOLD - a[ind]) > \
					ADVERTISEMENT_THRESHOLD * 7 / 10: 
					# advertismeent is almost completely on or completely off
					continue
				a_i,a_j = ind
				a_ij = a_effective[a_i,a_j]
				if not a_ij: # off
					self.latency_benefit(a_effective)
					a_effective[a_i,a_j] = True
					self.latency_benefit(a_effective)
					calls.append(((a_i,a_j), 'ba'))
				else: # on
					self.latency_benefit(a_effective)
					a_effective[a_i,a_j] = False
					self.latency_benefit(a_effective)
					calls.append(((a_i,a_j), 'ab'))
				a_effective[a_i,a_j] = a_ij
				n_significant += 1
				if n_significant >= N_REMEASURE:
					break
			print("Last LB call, {} were significant".format(n_significant))
		except AttributeError: # there are no last calls on the first iteration
			pass

		N_REMEASURE = len(calls)
		N_EXPLORE = total_n_grad_calc - N_REMEASURE

		self.last_lb_calls_results = {}

		all_inds = [(a_i,a_j) for a_i in range(self.n_popps) for a_j in range(self.n_prefixes)]
		already_in_calls = [ind for ind,_ in calls]
		possible_choices = get_difference(all_inds, already_in_calls)
		if len(possible_choices) > 0:
			possible_choice_inds = np.arange(len(possible_choices))

			N_EXPLORE = np.minimum(N_EXPLORE, len(possible_choices))
			choice_probs = np.array([ADVERTISEMENT_THRESHOLD - np.abs(a[ind] - ADVERTISEMENT_THRESHOLD) + .01 \
				for ind in possible_choices])
			choice_probs = choice_probs / np.sum(choice_probs)
			explore_inds = np.random.choice(possible_choice_inds, size = N_EXPLORE, 
				replace = False, p = choice_probs)
			explore_inds = [possible_choices[i] for i in explore_inds]

			for ind in explore_inds:
				if (ind,'ba') in calls or (ind,'ab') in calls: 
					continue
				a_ij = a_effective[ind]
				if not a_ij: # off
					self.latency_benefit(a_effective)
					a_effective[ind] = True
					self.latency_benefit(a_effective)
					calls.append((ind, 'ba'))
				else: # on
					self.latency_benefit(a_effective)
					a_effective[ind] = False
					self.latency_benefit(a_effective)
					calls.append((ind, 'ab'))
				a_effective[ind] = a_ij

		all_lb_rets = self.flush_latency_benefit_queue()
		for i, call_ind in enumerate(calls):
			ind, before_then_after = call_ind

			if before_then_after == 'ba':
				before,_ = all_lb_rets[2*i]
				after, _ = all_lb_rets[2*i+1]
			else:
				after,_ = all_lb_rets[2*i]
				before, _ = all_lb_rets[2*i+1]
			this_grad = self.heaviside_gradient(before, after, a[ind])
			if np.abs(this_grad) > 5:
				print('\n\n')
				print("WEIRDNESS Before: {}, After: {}".format(before,after))
				try:
					a_ij = a_effective[ind]
					if not a_ij:
						doublecheck_before = self.latency_benefit(a_effective, verbose_workers=True, retnow=True)
						a_effective[ind] = True
						doublecheck_after = self.latency_benefit(a_effective, verbose_workers=True, retnow=True)
					else:
						doublecheck_after = self.latency_benefit(a_effective, verbose_workers=True, retnow=True)
						a_effective[ind] = False
						doublecheck_before = self.latency_benefit(a_effective, verbose_workers=True, retnow=True)
					print("Double check before: {}, after: {}".format(doublecheck_before[0], doublecheck_after[0]))
					a_effective[ind] = a_ij

					benefit_by_ug = {}
					for worker in range(2):
						for row in open('logs/worker_{}_log-{}.txt'.format(worker,self.dpsize),'r'):
							fields = row.strip().split(',')
							if fields[0] != "benefit_estimate": continue
							ug = fields[2]
							benefit = float(fields[4])
							p_benefit = float(fields[5])
							itr = int(fields[-1])

							try:
								benefit_by_ug[ug]
							except KeyError:
								benefit_by_ug[ug] = {}
							try:
								benefit_by_ug[ug][itr] += (benefit*p_benefit)
							except KeyError:
								benefit_by_ug[ug][itr] = benefit*p_benefit


					all_iters = sorted(list(set(itr for ug in benefit_by_ug for itr in benefit_by_ug[ug])))
					last_two = all_iters[-2:]
					benefit_diffs = {ug:benefit_by_ug[ug][last_two[0]] - benefit_by_ug[ug][last_two[1]] for ug in benefit_by_ug}
					sorted_benefit_diffs = sorted(benefit_diffs.items(), key = lambda el : -1*np.abs(el[1]))
					print(sorted_benefit_diffs)
					print('\n\n')
				except:
					pass

			self.last_lb_calls_results[ind] = this_grad
			L_grad[ind] = this_grad
		

		L_grad = L_grad.clip(-GRAD_CLIP_VAL,GRAD_CLIP_VAL)

		self.all_lb_calls_results.append(L_grad)

		for (popp,pref), _ in calls:
			try:
				self.n_latency_benefit_calls[popp,pref] += 1
			except KeyError:
				self.n_latency_benefit_calls[popp,pref] = 1

		return L_grad

	def gradients(self, a, add_metrics=True):
		# gradient is the proximal gradient of the L1 norm
		# minus lambduh times gradient of L 
		# gradient of L is calculated via a continuous approximation
		if self.verbose:
			ts = time.time()
		L_grad = self.gradients_latency_benefit(a)
		if self.verbose:
			print("Calcing latency benefit grad took {}s".format(int(time.time() - ts)))
		if self.verbose:
			ts = time.time()
		res_grad = self.gradients_resilience_benefit_fn(a)
		if self.verbose:
			print("Calcing resilience benefit grad took {}s".format(int(time.time() - ts)))
		
		gamma = self.get_gamma()
		# gamma specifies a tradeoff between LB and RB, so shouldn't really be > 1
		# to encourage stability
		if gamma <= 1: 
			net_grad = L_grad + gamma * res_grad
			if add_metrics:
				self.metrics['l_benefit_grads'].append(L_grad)
				self.metrics['res_benefit_grads'].append(gamma * res_grad)
				self.metrics['cost_grads'].append(self.lambduh * self.alpha * np.ones(L_grad.shape))
		else:
			net_grad = 1 / gamma * L_grad + res_grad
			if add_metrics:
				self.metrics['l_benefit_grads'].append(1 / gamma * L_grad)
				self.metrics['res_benefit_grads'].append(res_grad)
				self.metrics['cost_grads'].append(self.lambduh * self.alpha * np.ones(L_grad.shape))

		DESIRED_MAX_VAL = 5.0
		max_val = np.max(np.abs(net_grad.flatten()))
		if max_val < DESIRED_MAX_VAL and max_val > 0:
			## try to flip at least one index
			## check to make sure this rescale wouldn't flip multiple advertisement indices at once
			## we might not flip any, or flip multiple because of momentum, however
			
			inds = np.abs(net_grad)>1e-3

			alphas = (ADVERTISEMENT_THRESHOLD - a[inds]) / (self.alpha * net_grad[inds])
			alphas = alphas[alphas>0]
			if len(alphas) > 0:
				limiting_alpha = np.min(alphas)
				mult = np.minimum(limiting_alpha, (DESIRED_MAX_VAL / max_val)) * 1.0001
			else:
				mult = 1.0

			net_grad = net_grad * mult
			print("Modified gradient by a factor of {} to ensure approximately one flip".format(mult))
		else:
			print("WARNING -- gradient is very large, max val is {}".format(max_val))
			net_grad = net_grad * .1 / max_val

		
		return -1 * net_grad

	def gradients_resilience_benefit_popp(self, advertisement):

		## want to test popp,pref 
		## turn it off, fail a popp. measure LB (a)
		## turn it on, fail same popp. measure LB (b)
		## should turn popp,pref on if (b) > (a)



		### Positive resilience benefit gradient means turning a popp
		## on will increase resilience
		### increasing resilience means maximizing benefit under popp failures


		grad_rb = np.zeros(advertisement.shape)
		a_effective = threshold_a(advertisement).astype(bool)
		calls = []


		### We monte-carlo sample the full space
		total_n_grad_calc = self.gradient_support_settings['popp_rb_support_size']
		
		pct_explore = 80 # pct of gradient calculation budget dedicated to exploring
		N_EXPLORE = int(total_n_grad_calc * pct_explore/100)
		# number of gradient calcs that re-calc previously high gradients
		N_REMEASURE = total_n_grad_calc - N_EXPLORE
		gamma = self.get_gamma()
		try:
			best_from_last_time = sorted(self.last_rb_calls_results_popp.items(), key = lambda el : 
				-1 * np.abs(el[1]))
			n_significant = 0
			for (popp,rand_kill_popp,rand_outer_prefix),val in best_from_last_time:
				if (popp,rand_kill_popp,rand_outer_prefix) in calls: 
					continue
				if gamma * np.abs(val) < self.lambduh or np.abs(val) < .01:
					# if it's not important enough to warrant the cost, don't bother
					continue
				if np.abs(ADVERTISEMENT_THRESHOLD - advertisement[self.popp_to_ind[popp], rand_outer_prefix]) > \
					ADVERTISEMENT_THRESHOLD * 7 / 10: 
					# advertisment is almost completely on or completely off
					continue

				tmp_a = copy.copy(a_effective)
				this_popp_random_kill = self.popp_to_ind[rand_kill_popp]
				tmp_a[this_popp_random_kill,:] = False # kill this random popp
				this_killed_popp_ugs = self.popp_to_users.get(this_popp_random_kill, [])
				if len(this_killed_popp_ugs) == 0:
					continue

				poppi = self.popp_to_ind[popp]
				tmp_a[poppi,rand_outer_prefix] = True # Turn this popp on
				self.latency_benefit(tmp_a, ugs=this_killed_popp_ugs)
				tmp_a[poppi,rand_outer_prefix] = False # turn this popp off
				self.latency_benefit(tmp_a, ugs=this_killed_popp_ugs)

				calls.append((popp, rand_kill_popp, rand_outer_prefix, this_killed_popp_ugs))

				n_significant += 1
				if n_significant >= N_REMEASURE:
					break
			print("Last RB call, {} were significant".format(n_significant))

		except AttributeError: # there are no last calls on the first iteration
			pass

		N_REMEASURE = len(calls)
		N_EXPLORE = total_n_grad_calc - N_REMEASURE


		all_popps = np.arange(self.n_popp)


		try:
			raise AttributeError
			## Sample popps that need more help, more
			rand_popp_choices = np.random.choice(all_popps, p=self.popp_rb_sample_probabilities, 
				size=N_EXPLORE)
		except AttributeError:
			rand_popp_choices = np.random.randint(low=0,high=self.n_popps,
				size=N_EXPLORE)

		# associated prefix distribution should be biased towards prefixes that are far from 1 and 0
		possible_prefix_choices = np.arange(self.n_prefixes)
		prob_each_pref = np.ones(self.n_prefixes) / self.n_prefixes

		for rand_kill_poppi in rand_popp_choices:
			rand_kill_popp = self.popps[rand_kill_poppi]
			
			poppi_helper = np.random.choice(all_popps,
				 p=self.popp_backup_sample_probs[rand_kill_poppi,:]) 
			popp_helper = self.popps[poppi_helper] # popp ij testing gradient is poppi,rand_outer_prefix (should we turn this on/off to help out?)

			rand_outer_prefix = int(np.random.choice(possible_prefix_choices, p=prob_each_pref))

			if (popp_helper, rand_kill_popp, rand_outer_prefix) in calls: continue
			
			tmp_a = copy.copy(a_effective)
			tmp_a[rand_kill_poppi,:] = False # kill this random popp
			this_killed_popp_ugs = self.popp_to_users.get(rand_kill_poppi, [])
			if len(this_killed_popp_ugs) == 0:
				continue

			tmp_a[poppi_helper,rand_outer_prefix] = True # Turn this popp on
			self.latency_benefit(tmp_a, ugs=this_killed_popp_ugs)
			tmp_a[poppi_helper,rand_outer_prefix] = False # turn this popp off
			self.latency_benefit(tmp_a, ugs=this_killed_popp_ugs)
			calls.append((popp_helper, rand_kill_popp, rand_outer_prefix, this_killed_popp_ugs))

		all_lb_rets = self.flush_latency_benefit_queue()
		self.last_rb_calls_results_popp = {}
		ind = 0

		for call_popp, killed_popp, rand_outer_prefix, this_killed_popp_ugs in calls:
			poppi = self.popp_to_ind[call_popp]
			
			failed_off,_ = all_lb_rets[ind] ## popp failed, random popp,prefix under consideration off
			failed_on,_ = all_lb_rets[ind+1] ## popp failed, random popp,prefix under consideration on


			this_grad = self.heaviside_gradient(
				failed_on, failed_off, 
				advertisement[poppi,rand_outer_prefix])

			grad_rb[poppi,rand_outer_prefix] += this_grad

			# if np.abs(this_grad) > .3:
			# 	print("RGRAD Entry {},{}, a value {}, n_ugs: {}".format(self.popp_to_ind[call_popp],rand_outer_prefix,
			# 		advertisement[self.popp_to_ind[call_popp],rand_outer_prefix], len(this_killed_popp_ugs)))
			# 	print("before_heavisside {} after_heavisside {}, grad: {}".format(
			# 		failed_on,failed_off,this_grad))

			self.last_rb_calls_results_popp[call_popp,killed_popp,rand_outer_prefix] = this_grad
			self.all_rb_calls_results_popps[self.popp_to_ind[killed_popp]].append((self.iter, poppi, rand_outer_prefix, this_grad))

			ind += 2

		### Track which calls are being made
		for poppi,poppj,pref,_ in calls:
			try:
				self.n_resilience_benefit_popp_calls[poppi,poppj,pref] += 1
			except KeyError:
				self.n_resilience_benefit_popp_calls[poppi,poppj,pref] = 1

		if not self.simulated:
			max_val = np.max(np.abs(grad_rb.flatten()))
			if max_val > 1:
				grad_rb = grad_rb / max_val

		grad_rb = grad_rb.clip(-GRAD_CLIP_VAL,GRAD_CLIP_VAL)

		return grad_rb

	def gradients_resilience_benefit_pop(self, advertisement):

		## want to test popp,pref 
		## turn it off, fail a PoP. measure LB (a)
		## turn it on, fail same PoP. measure LB (b)
		## should turn popp,pref on if (b) > (a)



		### Positive resilience benefit gradient means turning a popp
		## on will increase resilience
		### increasing resilience means maximizing benefit under PoP failures


		grad_rb = np.zeros(advertisement.shape)
		return grad_rb
		a_effective = threshold_a(advertisement).astype(bool)
		calls = []


		total_n_grad_calc = self.gradient_support_settings['pop_rb_support_size']
		
		pct_explore = 80 # pct of gradient calculation budget dedicated to exploring
		N_EXPLORE = int(total_n_grad_calc * pct_explore/100)
		# number of gradient calcs that re-calc previously high gradients
		N_REMEASURE = total_n_grad_calc - N_EXPLORE
		gamma = self.get_gamma()
		try:
			best_from_last_time = sorted(self.last_rb_calls_results_pop.items(), key = lambda el : 
				-1 * np.abs(el[1]))
			n_significant = 0
			for (popp,rand_kill_pop,rand_outer_prefix),val in best_from_last_time:
				if (popp,rand_kill_pop,rand_outer_prefix) in calls: 
					continue
				if gamma * np.abs(val) < self.lambduh:
					# if it's not important enough to warrant the cost, don't bother
					continue
				if np.abs(ADVERTISEMENT_THRESHOLD - advertisement[self.popp_to_ind[popp],rand_outer_prefix]) > \
					ADVERTISEMENT_THRESHOLD * 7 / 10: 
					# advertisment is almost completely on or completely off
					continue

				tmp_a = copy.copy(a_effective)
				tmp_a[self.pop_to_popp_inds[rand_kill_pop],:] = False # kill this random pop

				poppi = self.popp_to_ind[popp]
				tmp_a[poppi,rand_outer_prefix] = True # Turn this popp on
				self.latency_benefit(tmp_a)
				tmp_a[poppi,rand_outer_prefix] = False # turn this popp off
				self.latency_benefit(tmp_a)

				calls.append((popp, rand_kill_pop, rand_outer_prefix))

				n_significant += 1
				if n_significant >= N_REMEASURE:
					break
			print("Last RB call, {} were significant".format(n_significant))

		except AttributeError: # there are no last calls on the first iteration
			pass

		N_REMEASURE = len(calls)
		N_EXPLORE = total_n_grad_calc - N_REMEASURE


		### Popps for which we're testing if we want to turn them on/off
		rand_popp_choices = np.random.randint(low=0,high=self.n_popps,
			size=N_EXPLORE) 
		### associated prefixes for the rand_popp_choices
		# associated prefix distribution should be biased towards prefixes that are far from 1 and 0
		random_prefix_choices = np.zeros(N_EXPLORE,dtype=np.int32)
		possible_choices = np.arange(self.n_prefixes)
		for i in range(N_EXPLORE):
			prob_each_pref = ADVERTISEMENT_THRESHOLD - np.abs(advertisement[rand_popp_choices[i],:] - ADVERTISEMENT_THRESHOLD) + .1
			prob_each_pref = prob_each_pref / np.sum(prob_each_pref)
			prob_each_pref = np.ones(self.n_prefixes) / self.n_prefixes
			random_prefix_choices[i] = int(np.random.choice(possible_choices, p=prob_each_pref))


		for poppi, rand_outer_prefix in zip(rand_popp_choices,random_prefix_choices):
			popp = self.popps[poppi] # popp ij testing gradient is poppi,rand_outer_prefix

			## random kill PoP
			this_popp_random_kill = np.random.choice(np.arange(self.n_popp),
				 p=self.popp_backup_sample_probs[poppi,:]) 
			rand_kill_pop = self.popps[this_popp_random_kill][0]
			if (popp, rand_kill_pop, rand_outer_prefix) in calls: continue
			
			tmp_a = copy.copy(a_effective)
			tmp_a[self.pop_to_popp_inds[rand_kill_pop],:] = False # kill this random pop

			tmp_a[poppi,rand_outer_prefix] = True # Turn this popp on
			self.latency_benefit(tmp_a)
			tmp_a[poppi,rand_outer_prefix] = False # turn this popp off
			self.latency_benefit(tmp_a)
			calls.append((popp, rand_kill_pop, rand_outer_prefix))

		all_lb_rets = self.flush_latency_benefit_queue()
		self.last_rb_calls_results_pop = {}
		ind = 0
		for call_popp, killed_pop, rand_outer_prefix in calls:
			poppi = self.popp_to_ind[call_popp]
			
			failed_off,_ = all_lb_rets[ind] ## popp failed, random popp,prefix under consideration off
			failed_on,_ = all_lb_rets[ind+1] ## popp failed, random popp,prefix under consideration on


			this_grad = self.heaviside_gradient(
				failed_on, failed_off, 
				advertisement[poppi,rand_outer_prefix])

			grad_rb[poppi,rand_outer_prefix] += this_grad

			self.last_rb_calls_results_pop[call_popp,killed_pop,rand_outer_prefix] = this_grad
			self.all_rb_calls_results_pops[self.pop_to_ind[killed_pop]].append((self.iter, poppi, rand_outer_prefix, this_grad))

			ind += 2

		grad_rb = grad_rb.clip(-GRAD_CLIP_VAL,GRAD_CLIP_VAL)

		return grad_rb

	def gradients_resilience_benefit(self, advertisement):

		grad_link_failure = self.gradients_resilience_benefit_popp(advertisement)
		grad_pop_failure = 0#self.gradients_resilience_benefit_pop(advertisement) ### This hurts convergence

		alpha = 0 ## PoP failures are harder to plan for, so matter less
		return grad_link_failure + alpha * grad_pop_failure

	def impose_advertisement_constraint(self, a):
		"""The convex constraint 0 <= a_ij <= 1 has the simple solution to clip."""
		a = np.clip(a,0,1.0)
		return a

	def make_plots(self, *args, **kwargs):

		## Takes a while (plots from logs)
		try:
			compare_estimated_actual_per_user(self.dpsize)
			investigate_congestion_events()
		except:
			import traceback
			traceback.print_exc()
		

		n_sp = 9
		plt.rcParams["figure.figsize"] = (10,4*n_sp)
		plt.rcParams.update({'font.size': 14})
		f,ax = plt.subplots(n_sp,2)

		soln = self.get_last_advertisement()

		# General convergence metrics plot
		i=0
		while True:
			try:
				all_as = np.array(self.metrics['advertisements'][i:])
				all_grads = np.array(self.metrics['grads'][i:])
				all_cost_grads = np.array(self.metrics['cost_grads'][i:])
				all_l_benefit_grads = np.array(self.metrics['l_benefit_grads'][i:])
				all_res_benefit_grads = np.array(self.metrics['res_benefit_grads'][i:])
				linestyles = ['-','*','^','>','v']
				colors = ['orange','brown','aqua','deeppink','peru','grey','k','tan']
				for pref_i in range(self.n_prefixes):
					pref_sty = linestyles[pref_i%len(linestyles)]
					for popp_i in range(self.n_popp):
						if self.dpsize == 'small':
							ax[0,0].plot(all_as[:,popp_i,pref_i], 
								c=colors[popp_i%len(colors)])	
						ax[1,0].plot(all_grads[:,popp_i,pref_i], 
							c=colors[popp_i%len(colors)])
						ax[2,0].plot(all_cost_grads[:,popp_i,pref_i], 
							c=colors[popp_i%len(colors)])
						ax[3,0].plot(all_l_benefit_grads[:,popp_i,pref_i], 
							c=colors[popp_i%len(colors)])
						ax[4,0].plot(all_res_benefit_grads[:,popp_i,pref_i], 
							c=colors[popp_i%len(colors)])
				ax[0,0].set_ylabel("a")
				ax[1,0].set_ylabel("Net Grad")
				ax[2,0].set_ylabel("Cost Grad")
				ax[3,0].set_ylabel("LB Grad")
				ax[4,0].set_ylabel("Res Grad")
			except:
				import traceback
				traceback.print_exc()
				i += 1
				if i >= len(self.metrics['grads']):
					break
				continue
			break

		start_iter = 0
		if self.iter > 10:
			start_iter = 10

		all_objectives = self.metrics['actual_nonconvex_objective']
		all_pseudo_objectives = self.metrics['pseudo_objectives']
		all_effective_ojectives = self.metrics['effective_objectives']
		all_resilience_benefits = self.metrics['resilience_benefit']
		all_latency_benefits = self.metrics['latency_benefit']
		all_gt_latency_benefits = self.metrics['gt_latency_benefit']
		all_gt_resilience_benefits = self.metrics['gt_resilience_benefit']
		all_gammas = self.metrics['effective_gammas']
		ax[1,1].plot(list(range(start_iter,len(all_pseudo_objectives))), all_pseudo_objectives[start_iter:])
		ax[1,1].set_ylabel("Believed Objective")
		ax[0,1].plot(all_objectives)
		ax[0,1].set_ylabel("GT Objective")
		ax[2,1].plot(all_effective_ojectives)
		ax[2,1].set_ylabel("GT Effective Objective")
		ax[3,1].plot(list(range(start_iter,len(all_resilience_benefits))), all_resilience_benefits[start_iter:])
		ax[3,1].set_ylabel("Res Ben")
		ax[4,1].plot(list(range(start_iter,len(all_latency_benefits))), all_latency_benefits[start_iter:])
		ax[4,1].set_ylabel("Lat Ben")

		ax[5,0].plot(all_gt_latency_benefits)
		ax[5,0].set_ylabel("GT Lat Ben")
		ax[5,1].plot(all_gt_resilience_benefits)
		ax[5,1].set_ylabel("GT Res Ben")
		
		#### Add in optimal lines
		####### 
		try:
			ax[5,0].hlines(y=self.optimal_expensive_solution['latency'], xmin=0, xmax=self.iter, linewidth=2, color='k')
			ax[5,0].text(0,self.optimal_expensive_solution['latency'],"One per Peering")
		except AttributeError:
			pass
		try:
			ax[5,1].hlines(y=self.optimal_expensive_solution['resilience'], xmin=0, xmax=self.iter, linewidth=2, color='k')
			ax[5,1].text(0,self.optimal_expensive_solution['resilience'],"One per Peering")
		except AttributeError:
			pass
		try:
			ax[0,1].hlines(y=self.optimal_expensive_solution['overall'], xmin=0, xmax=self.iter, linewidth=2, color='k')
			ax[0,1].text(0,self.optimal_expensive_solution['overall'],"One per Peering")
		except AttributeError:
			pass

		#### ADD IN PAINTER LINES IF APPROPRIATE
		try:
			ax[5,0].hlines(y=self.painter_solution['latency_benefit'], xmin=0, xmax=self.iter, linewidth=2, color='r')
		except AttributeError:
			pass
		try:
			self.painter_gt_resilience_benefit
		except AttributeError:
			try:
				self.painter_gt_resilience_benefit = self.get_ground_truth_resilience_benefit(self.painter_solution['advertisement'])
			except AttributeError:
				pass
		try:
			ax[5,1].hlines(y=self.painter_gt_resilience_benefit, xmin=0, xmax=self.iter, linewidth=2, color='r')
		except AttributeError:
			pass
		try:
			ax[0,1].hlines(y=self.painter_solution['objective'], xmin=0, xmax=self.iter, linewidth=2, color='r')
			ax[0,1].text(0,self.painter_solution['objective'], "PAINTER")
		except AttributeError:
			pass


		ax[6,0].plot(all_gammas)
		ax[6,0].set_ylabel("Effective Gamma")

		try:
			all_link_utilizations = np.array(self.metrics['link_utilizations'])
			for poppi in range(self.n_popps):
				ax[7,0].plot(all_link_utilizations[:,poppi])
			ax[7,0].set_ylabel("Link Utilizations")
		except:
			pass

		save_fig(os.path.join(self.save_run_dir, 'convergence_over_iterations.pdf'), abs_path=True)

	def print_adv(self, a):
		for popp_i in range(self.n_popp):
			for pref_i in range(self.n_prefixes):
				print("PoPP {} Prefix {}: {}".format(self.popps[popp_i], pref_i, a[popp_i,pref_i]))

	def set_alpha(self):
		assert self.lambduh < 10
		if self.lambduh < 10 and self.lambduh > 1:
			self.alpha = .00005
		elif self.lambduh <= 1 and self.lambduh > .1:
			self.alpha = .0005
		elif self.lambduh <= .1 and self.lambduh > .01:
			self.alpha = .001
		elif self.lambduh <= .01:
			self.alpha = .01

	def get_gamma(self):
		### Idea is to increase gamma to our desired value as we become more confident about adjacent strategies
		if self.simulated:
			uncertainty_factor = np.maximum(1,np.abs(self.uncertainty_factor))
			divider = uncertainty_factor * (1 / (1 + 3 / np.sqrt((self.iter+1))))
		else:
			## no uncertainty factor since we don't do max info (for now)
			divider = (1 + 5 / np.sqrt((self.iter+1)))

		return self.gamma / divider

	def solve_max_information(self, current_advertisement):
		"""Search through neighbors of a, calculate maximum uncertainty."""
		uncertainties = {}

		a = np.copy(threshold_a(current_advertisement))
		current_benefit,_ = self.latency_benefit_fn(a, retnow=True)
		awful_benefit = -1000000
		uncertainty_alpha = .25
		# f,ax = plt.subplots(5)
		# self.plti=0

		def get_range(u):
			benefits,probs = u
			significant_prob = np.where(probs>.01)[0]
			if len(significant_prob) == 0:
				return 0
			min_benefit, max_benefit = benefits[significant_prob[0]], benefits[significant_prob[-1]]
			range_benefit = np.abs(max_benefit - min_benefit)
			return range_benefit

		def value_func(u,**kwargs):
			benefits,probs = u
			if len(probs) == 1:
				return awful_benefit
			setting_up = kwargs.get('setting_up',False)

			range_benefit = get_range(u)
			if range_benefit == 0 and not setting_up:
				return awful_benefit
			explore = kwargs.get('force', self.explore)
			if explore == 'positive_benefit':
				if np.sum(probs[benefits>=current_benefit]) > .99: return awful_benefit
				v = np.abs(np.sum(benefits[benefits>current_benefit] * probs[benefits>current_benefit]))
			elif explore == 'entropy':
				v = scipy.stats.entropy(probs+1e-8)
			elif explore == 'bimodality':
				# discussion here https://digitalcommons.wayne.edu/cgi/viewcontent.cgi?article=1120&context=jmasm
				benefits = benefits.flatten()
				probs = probs.flatten()
				ex = np.average(benefits,weights=probs+1e-8)
				exsq = np.average(np.power(benefits,2),weights=probs+1e-8)
				var = exsq - np.power(ex,2)
				std = np.sqrt(var)
				skew = np.average(np.power((benefits - ex) / (std+1e-8), 3), weights = probs+1e-8)
				kurt = np.average(np.power((benefits - ex) / (std+1e-8) , 4), weights = probs+1e-8)
				# maximizing v is maximizing bimodality
				v = -1 * (kurt - np.power(skew,2))
			elif explore == 'other_bimodality':
				negative_part = np.where(benefits <= current_benefit)[0]

				positive_part = np.where(benefits > current_benefit)[0]
				positive_mass = np.sum(probs[positive_part] * (benefits[positive_part] - current_benefit))
				negative_mass = np.sum(probs[negative_part] * (current_benefit - benefits[negative_part]))
				v = positive_mass * negative_mass
			elif explore == 'gmm':
				### idea is maximize distance between bimodal peaks
				### we find bimodal peaks by fitting gmm
				probs = np.array(probs).flatten()
				x_samp = np.random.choice(benefits,size=(1000,1),p=probs/np.sum(probs))
				with warnings.catch_warnings():
					warnings.filterwarnings('error')
					try:
						gmm_model = GaussianMixture(n_components=2).fit(x_samp)
					except ConvergenceWarning:
						return awful_benefit
				gm_means = gmm_model.means_
				v = np.abs(np.diff(gm_means.flatten()))[0]

			significant_prob = np.where(probs>.01)[0]
			min_benefit, max_benefit = benefits[significant_prob[0]], benefits[significant_prob[-1]]
			range_benefit = np.abs(max_benefit - min_benefit)

			if not setting_up:
				if range_benefit == 0:
					return v
				else:
					return v * range_benefit
			else:
				return v
		# if any of these gives a decent signal, measure that
		ranked_explore_methodologies = ['entropy']#['entropy', 'bimodality', 'other_bimodality']

		try:
			self.min_explore_value
		except AttributeError:
			# maybe not the best way?
			x = np.linspace(-1*MAX_LATENCY,-1*MIN_LATENCY,num=LBX_DENSITY)
			methods = ['positive_benefit','entropy','bimodality','other_bimodality','gmm']
			max_min_vals = {m:awful_benefit for m in methods}
			for i in range(len(x)):
				px = np.zeros(x.shape)
				px[i] = 1.0
				px = px + .0001*np.random.uniform(size=px.shape)
				px = px/np.sum(px)
				for method in methods:
					max_min_vals[method] = np.maximum(value_func((x,px), setting_up=True,
						force=method) * 3, max_min_vals[method])
			self.min_explore_value = max_min_vals
			print("Min explore values!")
			print(self.min_explore_value)
			print("\n")

		n_flips = 1
		max_time = 2 # seconds
		t_start = time.time()
		while True:
			# dedicate some percent to exploring permutations specific to transit providers
			pct_resilience = 50
			pct_transit = 30

			n_total = self.gradient_support_settings['info_support_size']
			n_rb = int(n_total * pct_resilience / 100)
			n_lb = n_total - n_rb
			
			perms,perm_labs = [],[]

			## Latency benefit exploration
			all_inds = [(i,j) for i in range(self.n_popp) for j in range(self.n_prefixes)]
			all_perms = sorted(list(itertools.permutations(all_inds, n_flips)))
			np.random.shuffle(all_perms)

			max_n_transit_lb = int(n_lb * pct_transit / 100)
			max_n_nontransit_lb = n_lb - max_n_transit_lb

			transit_perms, nontransit_perms = [], []
			for perm in all_perms:
				if any(poppi in self.provider_popp_inds for poppi,prefi in perm):
					if len(transit_perms) < max_n_transit_lb:
						transit_perms.append(perm)
				else:
					if len(nontransit_perms) < max_n_nontransit_lb:
						nontransit_perms.append(perm)
				if len(nontransit_perms) >= max_n_nontransit_lb and len(transit_perms) >= max_n_transit_lb:
					break
			perms += (transit_perms + nontransit_perms)
			perm_labs += (['LBtransit' for _ in range(len(transit_perms))] + ['LBnontransit' for 
				_ in range(len(nontransit_perms))])

			## Resilience benefit exploration
			max_n_transit_rb = int(n_rb * pct_transit / 100)
			max_n_nontransit_rb = n_rb - max_n_transit_rb

			popp_inds = np.arange(self.n_popps)
			np.random.shuffle(popp_inds)
			transit_perms, nontransit_perms = [], []
			for popp_ind in popp_inds:
				# try to acquire confidence in predictions in cases where popps fail
				perm = tuple([(popp_ind,prefi) for prefi in range(self.n_prefixes) if a[popp_ind,prefi]])
				if len(perm) == 0:
					continue
				if any(poppi in self.provider_popp_inds for poppi,prefi in perm):
					if len(transit_perms) < max_n_transit_rb:
						transit_perms.append(perm)
				else:
					if len(nontransit_perms) < max_n_nontransit_rb:
						nontransit_perms.append(perm)
				if len(nontransit_perms) >= max_n_nontransit_rb and len(transit_perms) >= max_n_transit_rb:
					break
			perms += (transit_perms + nontransit_perms)
			perm_labs += (['RBtransit' for _ in range(len(transit_perms))] + ['RBnontransit' for 
				_ in range(len(nontransit_perms))])

			## add in a couple pop failures for good measure
			pops = np.random.choice(self.pops, size=2)
			for pop in pops:
				corresponding_popps = list([self.popp_to_ind[popp] for popp in self.popps if popp[0] == pop])
				# check to see this PoP is on at all
				perm = tuple([(popp_ind,prefi) for prefi in range(self.n_prefixes) \
					for popp_ind in corresponding_popps if a[popp_ind,prefi]])
				if len(perm) == 0:
					continue
				perms.append(perm)
				perm_labs.append("RBPoP")


			## done searching for perms
			if len(perms) < n_total:
				n_left = self.gradient_support_settings['info_support_size'] - len(perms)
				not_in = get_difference(all_perms, perms)
				n_left = np.minimum(len(not_in), n_left)
				perms = perms + not_in[0:n_left]
				perm_labs += ["random" for _ in range(n_left)]


			for flips in perms:
				for flip in flips:
					a[flip] = 1 - a[flip]
				self.latency_benefit_fn(a)
				for flip in flips:
					a[flip] = 1 - a[flip]
			all_lb_rets = self.flush_latency_benefit_queue()

			for flipi, flips in enumerate(perms):
				_,u = all_lb_rets[flipi]
				uncertainties[flips] = {
					'distribution': u,
					'label': perm_labs[flipi],
				}

			potential_value_measure = {m:{} for m in ranked_explore_methodologies}
			max_benefit = {m:-1 * np.inf for m in ranked_explore_methodologies}
			best_flips = {m: None for m in ranked_explore_methodologies}
			# print(a)
			for flips,vals in uncertainties.items():
				u = vals['distribution']
				inds = np.where(u[1]>.01)[0]
				# print(flips)
				# for i in inds:
				# 	print("LB {} with prob {}".format(round(u[0][i],2), round(u[1][i],2)))
				for m in ranked_explore_methodologies:
					potential_value_measure[m][flips] = value_func(u, force=m)
					# print("Flip type {} had value {} for method {}".format(
					# 	vals['label'], round(potential_value_measure[m][flips],2), m))
					if potential_value_measure[m][flips] >= max_benefit[m]:
						best_flips[m] = flips
						max_benefit[m] = potential_value_measure[m][flips]
				# print("\n")
			# if max_benefit > awful_benefit:
			# 	print("{} -- {} {}".format(self.explore, max_benefit, best_flips))
			for m in ranked_explore_methodologies:
				if best_flips[m] is not None:
					print("Best explore value was {} for {}".format(potential_value_measure[m][best_flips[m]],m))
					if potential_value_measure[m][best_flips[m]] > self.min_explore_value[m]:
						for flip in best_flips[m]:
							a[flip] = 1 - a[flip]
						_,u = self.latency_benefit_fn(a, retnow=True)
						inds = np.where(u[1]>.01)[0]
						# print("explore methodology best flips: {}".format(m))
						# for i in inds:
						# 	print("LB {} with prob {}".format(round(u[0][i],2), round(u[1][i],2)))
						if tuple(a.flatten()) in self.measured:
							print("Re-measuring {}".format(a))
							print(potential_value_measure[m][best_flips[m]])
							pickle.dump(a,open('remeasure_a.pkl','wb'))
							print('woops')
							_,u = self.latency_benefit_fn(a, verbose_workers=True,retnow=True)
							print("This flips had value: {}".format(value_func(u,force=m)))
							print(u)
							exit(0)
						# print("Best flips was: {}".format(best_flips[m]))

						try:
							self.typical_high_uncertainty
						except AttributeError:
							self.typical_high_uncertainty = get_range(u) / 2
							print("Typical High Uncertainty is {}".format(self.typical_high_uncertainty))

						if 'RB' in uncertainties[best_flips[m]]['label']:
							uncertainty_measure = 1 + 10/self.typical_high_uncertainty*get_range(u) * (value_func(u,setting_up=True,force=m) - self.min_explore_value[m])
							self.uncertainty_factor = (1 - uncertainty_alpha) * \
								self.uncertainty_factor + uncertainty_alpha * uncertainty_measure
						else:
							self.uncertainty_factor *= (1 - uncertainty_alpha)
						self.uncertainty_factor = np.maximum(1, self.uncertainty_factor)
						print("New uncertainty factor is {}".format(self.uncertainty_factor))

						return a
					# else:
					# 	tmpa = copy.copy(a)
					# 	for flip in best_flips[m]:
					# 		tmpa[flip] = 1 - tmpa[flip]
					# 	_,u = self.latency_benefit_fn(tmpa,retnow=True)
					# 	inds = np.where(u[1]>.01)[0]
					# 	print("explore methodology: {}".format(m))
					# 	for i in inds:
					# 		print("uniformative LB {} with prob {}".format(round(u[0][i],2), round(u[1][i],2)))
			n_flips += 1
			if n_flips == 2:
				break
		self.uncertainty_factor *= (1 - uncertainty_alpha)
		print("New uncertainty factor is {}".format(self.uncertainty_factor))
		return None		
		# plt.close()

	def stop_tracker(self, advertisement, skip_measuring=False):
		# Stop when the objective doesn't change, 
		# but use an EWMA to track the change so that we don't spuriously exit
		delta_alpha = .2
		delta_eff_alpha = .2



		if not self.simulated:
			if self.iter == 0:
				### Save optimization state, just in case 
				self.output_optimization_state()

		# re-calculate objective
		self.last_objective = self.current_pseudo_objective
		self.last_effective_objective = self.current_effective_objective
		self.metrics['effective_gammas'].append(self.get_gamma())
		if not skip_measuring or len(self.metrics['gt_latency_benefit']) == 0:
			#### This takes the most time, probably because we always step to a new advertisement and so reset our caches
			self.metrics['actual_nonconvex_objective'].append(self.measured_objective(advertisement, verb=True, save_metrics=True))
			self.metrics['gt_latency_benefit'].append(self.get_ground_truth_latency_benefit(advertisement, verb=True, save_ug_ingress_decisions=True))
			self.metrics['gt_resilience_benefit'].append(self.get_ground_truth_resilience_benefit(advertisement,
				store_metrics=True))
			self.metrics['effective_objectives'].append(self.measured_objective(copy.copy(threshold_a(advertisement))))
		else:
			for k in ['actual_nonconvex_objective', 'gt_latency_benefit', 'gt_resilience_benefit', 'effective_objectives']:
				self.metrics[k].append(self.metrics[k][-1])
		self.current_objective = self.metrics['actual_nonconvex_objective'][-1]
		self.current_latency_benefit = self.metrics['gt_latency_benefit'][-1]
		self.current_resilience_benefit = self.metrics['gt_resilience_benefit'][-1]

		self.current_pseudo_objective = self.modeled_objective(advertisement,verbose_workers=True,verbose=True) 
		self.current_effective_objective = self.modeled_objective(threshold_a(advertisement))
		self.metrics['pseudo_objectives'].append(self.current_pseudo_objective)
		rb = self.resilience_benefit(advertisement)
		self.metrics['resilience_benefit'].append(rb)
		lb_model = self.latency_benefit_fn(advertisement,retnow=True)
		self.metrics['latency_benefit'].append(lb_model[0])

		## Add to metrics
		self.metrics['frac_latency_benefit_calls'].append(len(self.n_latency_benefit_calls) / (self.n_popps * self.n_prefixes))
		self.metrics['frac_resilience_benefit_calls'].append(len(self.n_resilience_benefit_popp_calls) / (self.n_popps * self.n_popps * self.n_prefixes))

		### Notify workers of new training iteration
		for worker, worker_socket in self.worker_manager.worker_sockets.items():
			msg = pickle.dumps(('increment_iter', "meep"))
			worker_socket.send(msg)
			worker_socket.recv()

		self.rolling_delta = (1 - delta_alpha) * self.rolling_delta + delta_alpha * np.abs(self.current_pseudo_objective - self.last_objective)
		self.rolling_delta_eff = (1 - delta_eff_alpha) * self.rolling_delta_eff + \
			delta_eff_alpha * np.abs(self.current_effective_objective - self.last_effective_objective)
		adv_delta = np.max(np.abs((advertisement - self.last_advertisement).flatten()))
		self.rolling_adv_delta = (1 - delta_alpha) * self.rolling_adv_delta + delta_alpha * adv_delta
		print("RAD: {}".format(self.rolling_adv_delta))
		self.stop = self.stopping_condition([self.iter,self.rolling_delta,self.rolling_delta_eff,self.rolling_adv_delta])

		if self.iter % self.save_state_every == 0:
			### Save optimization state 
			self.output_optimization_state()
		self.output_small_stats()

	def get_init_kwa(self):
		kwa = {
			'lambduh': self.lambduh, 
			'gamma': self.gamma, 
			'with_capacity': self.with_capacity,
			'verbose': False,
			'init': self.initialization,
			'explore': self.explore,
			'using_resilience_benefit': self.using_resilience_benefit,
			'n_prefixes': self.n_prefixes,
			'save_run_dir': self.save_run_dir,
		}
		if self.using_generic_objective:
			kwa['generic_objective'] = self.generic_objective.obj
		return kwa

	def output_small_stats(self):
		print("Saving smaller stats every iteration, dont exit...")
		self.optimization_vars = {}
		for k in self.optimization_var_names:
			self.optimization_vars[k] = getattr(self, k)
		out_fn = os.path.join(self.save_run_dir, 'small-stats-{}.pkl'.format(self.iter))
		save_state = {
			'optimization_advertisement_representation': self.optimization_advertisement_representation,
			'optimization_vars': self.optimization_vars, # related to when we're going to stop
		}
		pickle.dump(save_state, open(out_fn, 'wb'))
		print("Done saving")

	def output_optimization_state(self):
		print("Saving optimization state, dont exit...")
		self.optimization_vars = {}
		for k in self.optimization_var_names:
			self.optimization_vars[k] = getattr(self, k)
		out_fn = os.path.join(self.save_run_dir, 'state-{}.pkl'.format(self.iter))
		save_state = {
			'deployment': self.og_deployment, # link caps, user performance, etc.
			'optimization_advertisement_representation': self.optimization_advertisement_representation,
			'ug_modified_deployment': self.output_deployment(), # link caps, user performance, etc.
			'all_rb_calls_results_popps': self.all_rb_calls_results_popps,
			'last_gti': self.last_gti,
			'advertisement': self.get_last_advertisement(), # the optimization variable
			'last_advertisement': self.last_advertisement,
			'optimization_vars': self.optimization_vars, # related to when we're going to stop
			'parent_tracker': self.parent_tracker, # measured ingress preferences
			'measured': self.measured,
			'measured_prefs': self.measured_prefs,
			'metrics': self.metrics,
		}
		try:
			# may or may not have these variables
			save_state['old_optimal_expensive_solution'] = self.old_optimal_expensive_solution
		except AttributeError:
			pass
		pickle.dump(save_state, open(out_fn, 'wb'))
		print("Done saving")

	def load_optimization_state(self, specific_iter=None):
		self.clear_caches()
		save_port = copy.deepcopy(self.port)
		if specific_iter is None:
			import glob
			all_states = glob.glob(os.path.join(self.save_run_dir, '*'))
			all_iters = [int(re.search("state\-(.+)\.pkl", fn).group(1)) for fn in all_states if "state" in fn]
			specific_iter = np.max(all_iters)
		print("Loading save state {}".format(specific_iter))
		save_state = pickle.load(open(os.path.join(self.save_run_dir, 'state-{}.pkl'.format(specific_iter)),'rb'))

		## update advertisement
		self.metrics = save_state['metrics']
		self.optimization_advertisement = save_state['advertisement']
		self.optimization_advertisement_representation = save_state.get('optimization_advertisement_representation', {})
		self.last_gti = save_state.get('last_gti',{})
		self.last_advertisement = save_state['last_advertisement']
		self.n_prefixes = self.optimization_advertisement.shape[1]
		print(np.sum(self.optimization_advertisement>.5,axis=0))

		## update deployment from the OG deployment (currently loaded) to the pseudo-UG deployment
		self.og_deployment = save_state['deployment']
		new_deployment_with_pseudo_users = save_state['ug_modified_deployment']
		self.og_deployment['port'] = save_port
		new_deployment_with_pseudo_users['port'] = save_port
		self.old_optimal_expensive_solution = save_state['old_optimal_expensive_solution']
		self.update_deployment(new_deployment_with_pseudo_users)
		print(np.sum(self.optimization_advertisement>.5,axis=0))

		## various optimization variables
		for k,v in save_state['optimization_vars'].items():
			print("{} {}".format(k,v))
			setattr(self, k, v)

		## information about learned preferneces
		self.measured_prefs = save_state['measured_prefs']
		self.parent_tracker = save_state['parent_tracker']
		for (ui,beaten_ingress,routed_ingress), tf in self.parent_tracker.items():
			if not tf: continue
			try:
				self.calc_cache.all_caches['parents_on'][ui][beaten_ingress,routed_ingress] = None
			except KeyError:
				self.calc_cache.all_caches['parents_on'][ui] = {(beaten_ingress,routed_ingress): None}
		self.update_parent_tracker_workers()
		print(np.sum(self.optimization_advertisement>.5,axis=0))

		self.measured = save_state['measured']
		self.all_rb_calls_results_popps = save_state['all_rb_calls_results_popps']

		if self.simulated:
			self.calculate_user_choice(self.optimization_advertisement, get_ug_catchments=True)
			self.get_ground_truth_latency_benefit(self.optimization_advertisement)
		else:
			self.update_ug_ingress_decisions()
		print(np.sum(self.optimization_advertisement>.5,axis=0))

		### Notify workers of current training iteration
		for worker, worker_socket in self.worker_manager.worker_sockets.items():
			msg = pickle.dumps(('set_iter', self.iter))
			worker_socket.send(msg)
			worker_socket.recv()

		self.stop = self.stopping_condition([self.iter,self.rolling_delta,self.rolling_delta_eff,self.rolling_adv_delta])
		print(np.sum(self.optimization_advertisement>.5,axis=0))

	def init_optimization_vars(self):
		self.clear_caches()
		
		self.set_alpha() # momentum parameter

		self.calc_times = []
		self.measured = {}
		self.path_measures = 0 
		self.last_gti = None

		## Track which popps are on/off, helpful for use in the actual deployment
		opt_adv_on_off = threshold_a(self.optimization_advertisement)
		self.optimization_advertisement_representation = {}
		for poppi,prefi in zip(*np.where(opt_adv_on_off)):
			self.optimization_advertisement_representation[self.popps[poppi], prefi] = None
		
		if self.verbose:
			# self.print_adv(advertisement)
			print("Optimizing over {} peers and {} ugs".format(self.n_popp, self.n_ug))

		self.iter = 0

		self.reset_metrics()
		self.stop = False

		# Add to metrics / init vars
		self.current_objective = self.measured_objective(self.optimization_advertisement, save_ug_ingress_decisions=True)
		self.current_latency_benefit = self.get_ground_truth_latency_benefit(self.optimization_advertisement, verb=True, save_ug_ingress_decisions=True)
		self.current_resilience_benefit = self.get_ground_truth_resilience_benefit(self.optimization_advertisement, store_metrics=True)

		self.current_pseudo_objective = self.modeled_objective(self.optimization_advertisement)
		self.current_effective_objective = self.modeled_objective(threshold_a(self.optimization_advertisement))
		self.last_objective = self.current_pseudo_objective
		self.last_effective_objective = self.current_effective_objective
		self.rolling_delta = 10
		self.rolling_delta_eff = 10
		self.rolling_adv_delta = 10
		self.rolling_adv_eps = .01

		self.metrics['pseudo_objectives'].append(self.current_pseudo_objective)
		self.metrics['actual_nonconvex_objective'].append(self.current_objective)
		self.metrics['effective_objectives'].append(self.measured_objective(threshold_a(self.optimization_advertisement)))
		self.metrics['advertisements'].append(copy.copy(self.optimization_advertisement))

	def modify_ugs(self):
		try:
			### See if we've already computed the modified deployment
			self.og_deployment
			return
		except AttributeError:
			pass
		## create a pseudo deployment modeled after the optimal solution
		## make a user's optimally assigned popp their lowest-latency popp
		## split users by volume
		self.og_deployment = self.output_deployment()

		### This shortcut only makes sense if we're using heuristic speedup approximations
		### We don't use heuristic speedups if we're doing the generic objective
		### So return on using generic objective
		if self.using_generic_objective:
			print("Not modifying deployment to use pseudo-UGs because not using heuristic approximations.")
			return
		else:
			print("Modfying UGs to be pseudo-UGs so that our heuristic approximation works better")

		self.old_optimal_expensive_solution = self.optimal_expensive_solution

		e = gp.Env()
		e.setParam('OutputFlag', 0)
		e.setParam('TimeLimit', 3)

		TOO_BIG = .001
		def close_enough(pathvols_a, pathvols_b):
			## test if two pathvol dists are close enough
			# do L1 diff
			pathvols_a = {popp:v for popp,v in pathvols_a}
			pathvols_b = {popp:v for popp,v in pathvols_b}
			for popp in set(pathvols_a).union(set(pathvols_b)):
				if np.abs(pathvols_a.get(popp,0) - pathvols_b.get(popp,0)) > TOO_BIG:
					return False
			return True

		def solve_sub_ug(steadystate_pathvols, failure_pathvols):
			### Create an object that summarizes our target
			verb = False#min_n_ugs > 1

			all_popps = list([self.popps[el[0]] for el in steadystate_pathvols])
			all_vols = list([el[1] for el in steadystate_pathvols])
			for popp in failure_pathvols:
				for _poppi,v in failure_pathvols[popp]:
					all_popps.append(self.popps[_poppi])
					all_vols.append(v)
			all_popps = list(sorted(set(all_popps)))
			all_poppis = list(sorted([self.popp_to_ind[popp] for popp in all_popps]))
			popp_to_ind = {popp:i for i,popp in enumerate(all_popps)}
			n_popps = len(all_popps)
			all_vols = sorted(list(set(all_vols)))
			## Description of all the states we need to consider
			# ones array where 1/0 indicates active or not at the time
			# associated pathvols
			scenarios = [(np.ones(n_popps), steadystate_pathvols)]
			for failed_popp, path_vols in failure_pathvols.items():
				activity_indicator = np.ones(n_popps)
				activity_indicator[popp_to_ind[failed_popp]] = 0
				scenarios.append((activity_indicator, path_vols))

			n_scenarios = len(scenarios)
			## Ensure the data is correctly formatted
			for i in range(n_scenarios): 
				## Volumes in a scenario sum to 1
				pathvols = scenarios[i][1]
				sum_pv = sum([el[1] for el in pathvols])
				scenarios[i] = list(scenarios[i])
				scenarios[i][1] = [(popp,v/sum_pv) for popp,v in pathvols]
				scenarios[i] = tuple(scenarios[i])


			if len(failure_pathvols) == 0:
				print("giving up...")
			# else:
			# 	print(scenarios)
			### All the listings of latencies
			if verb:
				print(all_popps)
				print(steadystate_pathvols)
				print(failure_pathvols)

			winning_users = {}
			
			# print("Scenarios: {}".format(scenarios))
			# print("ALl PoPPs: {}".format(all_popps))
			def solve_model(n_xi):
				model = gp.Model("mip1",env=e)
				model.Params.LogToConsole = 0
				x = model.addMVar(n_xi, name='volume_each_user', lb=0)
				model.addConstr(np.ones(n_xi) @ x == 1)
				

				all_sums = []
				all_vars = {}
				for k,(actives,constraint_set) in enumerate(scenarios):
					running_sum = np.zeros(n_xi)
					constraint_set = {popp:vol for popp,vol in constraint_set}
					all_vars[k] = {}
					for j,popp in enumerate(all_poppis):
						vol = constraint_set.get(popp,0)
						a_jk = model.addMVar(n_xi, vtype=gp.GRB.BINARY, name="a_{}_{}".format(j,k))
						if vol > 0:
							obj = model.addMVar((1,), lb=-10000)
							model.addConstr(obj == ((a_jk @ np.eye(n_xi) @ x) - vol))
							tmp_obj_object = model.addMVar((1,), lb=0)
							model.addConstr(tmp_obj_object[0] == gp.norm(obj,2))
							all_sums.append(tmp_obj_object)
						else:
							model.addConstr((a_jk @ np.eye(n_xi) @ x) == vol)
						running_sum += a_jk
						all_vars[k][j] = a_jk
					model.addConstr(running_sum == np.ones(n_xi))

				## Add preference constraints
				popp_combs = list(itertools.combinations(list(range(n_popps)), 2))
				scenario_combs = list(itertools.combinations(list(range(n_scenarios)), 2))
				for popp_combi in popp_combs:
					if popp_combi[0] == popp_combi[1]: continue
					poppa,poppb = popp_combi
					for scenario_comb in scenario_combs:
						if scenario_comb[0] == scenario_comb[1]: continue
						k1,k2 = scenario_comb
						if scenarios[k1][0][poppa] == 1 and scenarios[k1][0][poppb] == 1 and \
							scenarios[k2][0][poppa] == 1 and scenarios[k2][0][poppb] == 1:
							# print("Adding constraint for k1: {} k2: {} poppa: {} ({}) poppb: {} ({})".format(k1,k2,poppa,all_popps[poppa],poppb,all_popps[poppb]))
							model.addConstr((all_vars[k1][poppa] - all_vars[k1][poppb]) * (all_vars[k2][poppa] - all_vars[k2][poppb]) >= np.zeros(n_xi))


				model.setObjective(gp.quicksum(all_sums))
				model.optimize()
				return model, all_vars, x

			min_n_ugs = 1
			last_obj = 50000
			while True:
				if verb:
					print("\n\nNUM UGS : {}\n\n".format(min_n_ugs))
					print(scenarios)
				model, all_vars, x = solve_model(min_n_ugs)
				if model.status != 2:
					if len(failure_pathvols) == 0:
						print(scenarios)
						print("Even with no failure scenarios, still impossible")
						exit(0)
					return solve_sub_ug(steadystate_pathvols, {})

				obj_value = model.getObjective().getValue()
				# print("{}: Squared sum of errors: {}".format(min_n_ugs, obj_value))
				if obj_value == 0: ## perfect solution immediately
					break
				elif np.abs(obj_value - last_obj) < 0.01 or min_n_ugs >= 7:
					min_n_ugs -=1
					model, all_vars, x = solve_model(min_n_ugs)
					break
				else:
					last_obj = obj_value
					min_n_ugs += 1

			ug_vols = x.X
			if verb:
				print("Testing vols: {}".format(ug_vols))
			## Parse routes from optimization output
			routes_np = {k:np.zeros((n_popps, min_n_ugs)) for k in range(n_scenarios)}
			for k in range(n_scenarios):
				# print('Active PoPPs: {}'.format(scenarios[k][0]))
				for j in range(n_popps):
					routes_np[k][j,:] = all_vars[k][j].X
				# print("Routed: \n{}".format(routes_np[k]))
			### Convert routes to preferences
			ui_popp_to_parents = {}
			base_actives = np.ones(n_popps) ## all on
			base_k = [k for k,(actives,constraint_set) in enumerate(scenarios) if np.array_equal(actives,base_actives)][0]
			base_mapping = all_vars[base_k]
			for k,(actives,constraint_set) in enumerate(scenarios):
				if np.array_equal(actives,base_actives): continue
				popps_on = list([all_popps[i] for i in np.where(actives)[0]])
				routes = routes_np[k]
				for ui in range(min_n_ugs):
					winning_popp = all_popps[np.where(routes[:,ui])[0][0]]
					for losing_popp in get_difference(popps_on, [winning_popp]):
						try:
							ui_popp_to_parents[ui,losing_popp].append(winning_popp)
						except KeyError:
							ui_popp_to_parents[ui,losing_popp] = [winning_popp]
			# for ui,popp in sorted(ui_popp_to_parents):
			# 	print("UI : {}, PoPP: {}, Parents: {}".format(ui,self.popp_to_ind[popp], [self.popp_to_ind[_popp] for _popp in ui_popp_to_parents[ui,popp]]))
			# Initialize the winning popp as routed popp in the base case with all popps on
			ui_to_ranked_popps = {ui:[all_popps[np.where(routes_np[base_k][:,ui])[0][0]]] for ui in range(min_n_ugs)}
			for ui in range(min_n_ugs):
				while len(ui_to_ranked_popps[ui]) < n_popps:
					unassigned_popps = {popp:None for popp in get_difference(all_popps, ui_to_ranked_popps[ui])}
					for popp in unassigned_popps:
						beaten = False ## if no parents are currently unassigned, assign this as the next-best popp
						for parent_popp in ui_popp_to_parents.get((ui, popp), []):
							try:
								unassigned_popps[parent_popp]
								beaten = True
								break
							except KeyError:
								pass
						if not beaten:
							ui_to_ranked_popps[ui].append(popp)
							break
			# print({ui:[self.popp_to_ind[popp] for popp in popps] for ui,popps in ui_to_ranked_popps.items()})
			# print(ug_vols)
			if obj_value == 0: ## if we can solve it perfectly, do a sanity check
				for actives, path_vols in scenarios:
					# print(actives)
					actives = list([all_popps[i] for i,a in enumerate(actives) if a])
					vol_assignments = {}
					for ui, ranked_popps in ui_to_ranked_popps.items(): # for each user
						# if verb:
						# 	print("User pref: {}, zipped: {}".format(user_pref, list(zip(user_pref, active_popps))))
						### Determine user mappings under this scenario and preference model
						for popp in ranked_popps:
							if popp in actives:
								try:
									vol_assignments[self.popp_to_ind[popp]] += ug_vols[ui]
								except KeyError:
									vol_assignments[self.popp_to_ind[popp]] = ug_vols[ui]
								break
					vol_assignments = sorted(list(vol_assignments.items()))
					if verb:
						print("Vol assignments: {}".format(vol_assignments))
					if not close_enough(vol_assignments, path_vols):
						print("Something didn't work as expected")
						print("{} vs {}".format(vol_assignments, path_vols))
						exit(0)
			## Winner!
			for ui, sorted_popp_list in ui_to_ranked_popps.items():
				winning_users[ui] = (ug_vols[ui], sorted_popp_list)

			# print(winning_users)
				
			return winning_users

		### Get the optimal solution during failure as well
		if not os.path.exists(self.optimal_under_failure_cache_fn):
			from wrapper_eval import assess_failure_resilience_one_per_peering
			adv = np.eye(self.n_popps)
			ret = assess_failure_resilience_one_per_peering(self, adv, which='popps')
			self.optimal_under_failure_ug_pathvols = {}
			for _, _, ug, failed_popp, _, _, new_pathvols in ret['mutable']['latency_delta_specific']:
				try:
					self.optimal_under_failure_ug_pathvols[ug][failed_popp] = new_pathvols
				except KeyError:
					self.optimal_under_failure_ug_pathvols[ug] = {failed_popp: new_pathvols}
			pickle.dump(self.optimal_under_failure_ug_pathvols, open(self.optimal_under_failure_cache_fn,'wb'))
		else:
			self.optimal_under_failure_ug_pathvols = pickle.load(open(self.optimal_under_failure_cache_fn, 'rb'))


		optimal_solution = self.optimal_expensive_solution['obj']
		new_deployment = self.output_deployment()
		
		### TMP
		lat_to_vol = {}
		for ug in self.ugs:
			for l in self.ug_perfs[ug].values():
				try:
					lat_to_vol[l] += self.ug_to_vol[ug]
				except KeyError:
					lat_to_vol[l] = self.ug_to_vol[ug]

		for ugi, pathvols in tqdm.tqdm(optimal_solution['paths_by_ug'].items(),
				desc="Solving for pseudo-UGS..."):
			ug = self.ugs[ugi]

			best_popp = sorted(self.ug_perfs[ug].items(), key = lambda el : el[1])[0][0]
			best_popp_latency = self.ug_perfs[ug][best_popp]
			## current dist is {best_popp: 1.0}, desired dist is {poppis: vols}

			new_ugs = solve_sub_ug(pathvols, self.optimal_under_failure_ug_pathvols[ug])
			# new_ugs = solve_sub_ug(pathvols, {})
			npvs = len(new_ugs)
			for newugi in new_ugs:
				vol, popp_prefs = new_ugs[newugi]
				### create a new pseudo user with this much volume and sorted performances according to these preferences
				### lowest preference = MAXIMUM preference = lowest latency
				og_lats = sorted([self.ug_perfs[ug][popp] for popp in popp_prefs])
				newug = (ug[0],ug[1] + round((newugi+1) * 1.0/(npvs+2),2))
				new_deployment['ug_perfs'][newug] = {}
				for i,popp in enumerate(popp_prefs): ## permute the performances according to the desired preferences
					new_deployment['ug_perfs'][newug][popp] = og_lats[i]
				new_deployment['ug_to_vol'][newug] = vol * self.ug_to_vol[ug]
				new_deployment['ug_to_bulk_vol'][newug] = vol * self.ug_to_bulk_vol[ug]
				if self.simulated:
					new_deployment['ingress_priorities'][newug] = new_deployment['ingress_priorities'][ug]
				for k in get_difference(new_deployment['ug_perfs'][ug], new_deployment['ug_perfs'][newug]):
					new_deployment['ug_perfs'][newug][k] = new_deployment['ug_perfs'][ug][k]
				new_deployment['ug_to_ip'][newug] = new_deployment['ug_to_ip'][ug]
				new_deployment['ug_anycast_perfs'][newug] = new_deployment['ug_anycast_perfs'][ug]

			del new_deployment['ug_to_bulk_vol'][ug]
			del new_deployment['ug_perfs'][ug]
			del new_deployment['ug_to_vol'][ug]
			if self.simulated:
				del new_deployment['ingress_priorities'][ug]
			del new_deployment['ug_to_ip'][ug]
			del new_deployment['ug_anycast_perfs'][ug]
		new_deployment['ugs'] = list(sorted(list(new_deployment['ug_perfs'])))
		for k in ['ug_to_vol', 'ug_to_bulk_vol', 'ug_perfs', 'ugs']:
			new_deployment["whole_deployment_" + k] = copy.deepcopy(new_deployment[k])
		if self.simulated:
			new_deployment['whole_deployment_ingress_priorities'] = copy.deepcopy(new_deployment['ingress_priorities'])
		
		self.update_deployment(new_deployment)


		print("Modified deployment from {} UGs to {} UGs".format(len(self.og_deployment['ugs']), len(new_deployment['ugs'])))

	def reset_ugs(self):
		self.update_deployment(self.og_deployment)
		print(np.sum(self.optimization_advertisement>.5,axis=0))
		# if not self.simulated:
		# 	self.get_realworld_measure_wrapper()

	def solve(self, **kwargs):

		try:
			## If we're hot-starting, load the optimization state. But this will throw an error if we're not
			self.load_optimization_state()
			if self.iter >= self.max_n_iter:
				self.reset_ugs() 
				return
		except ValueError:
			self.modify_ugs()
			self.optimization_advertisement = self.init_advertisement()
			self.last_advertisement = copy.copy(self.optimization_advertisement)
			if not self.simulated:
				## This is our first measurement
				self.calculate_ground_truth_ingress(self.optimization_advertisement)
				
			self.init_optimization_vars()

			# Measure where we start, update model of path probabilities
			self.measure_ingresses(self.optimization_advertisement)


		t_start = time.time()
		self.t_per_iter = 0

		if not self.simulated:
			self.last_measured_advertisement = self.optimization_advertisement

		while not self.stop:

			timers = []
			t_last = time.time()

			if self.verbose:
				print("\n\n")
				print("LEARNING ITERATION : {}".format(self.iter))
				print("\n\n")
			self.ts_loop = time.time()

			# calculate gradients
			if self.verbose:
				print("calcing grads")
			grads = self.gradient_fn(self.optimization_advertisement)

			## grads
			timers.append(time.time() - t_last)
			t_last = time.time()

			self.recent_grads = grads
			# update advertisement by taking a gradient step with momentum and then applying the proximal gradient for L1
			a_k = self.optimization_advertisement
			w_k = a_k - self.alpha * grads + self.beta * (a_k - self.last_advertisement)
			if self.proximal:
				self.optimization_advertisement = self.apply_prox_l1(w_k)
			else:
				self.optimization_advertisement = w_k
			self.last_advertisement = copy.copy(a_k)

			# another constraint we may want is 0 <= a_ij <= 1
			# the solution is just clipping to be in the set
			# clipping can mess with gradient descent
			self.optimization_advertisement = self.impose_advertisement_constraint(self.optimization_advertisement)
 
			self.metrics['advertisements'].append(copy.copy(self.optimization_advertisement))
			self.metrics['grads'].append(self.optimization_advertisement - a_k)

			if self.simulated:
				# Take a gradient step and update measured paths + probabilities
				if not np.array_equal(threshold_a(self.optimization_advertisement), threshold_a(self.last_advertisement)):
					if self.verbose:
						print("Gradient stepped to a new advertisement, issuing measurement.")
						print("Changed Indices: {}".format(np.where(np.abs(threshold_a(self.optimization_advertisement) - threshold_a(self.last_advertisement)))))
					self.measure_ingresses(self.optimization_advertisement)
					opt_adv_on_off = threshold_a(self.optimization_advertisement)
					self.optimization_advertisement_representation = {}
					for poppi,prefi in zip(*np.where(opt_adv_on_off)):
						self.optimization_advertisement_representation[self.popps[poppi], prefi] = None
			else:
				NUM_PREFS_TRIGGER_CHANGE = 2 # N prefixes change
				NUM_TOTAL_TRIGGER_CHANGE = 4 # N popps change
				measured_this_round = False
				differences = np.where(np.abs(threshold_a(self.optimization_advertisement) - threshold_a(self.last_measured_advertisement)))
				print("Differences in advertisement so far : {}".format(differences))
				prefs_changed = {}
				if len(differences) > 0:
					## Change in the advertisement, update the actual-deployment-specific tracker
					opt_adv_on_off = threshold_a(self.optimization_advertisement)
					self.optimization_advertisement_representation = {}
					for poppi,prefi in zip(*np.where(opt_adv_on_off)):
						self.optimization_advertisement_representation[self.popps[poppi], prefi] = None

				for poppi,prefi in zip(*differences):
					prefs_changed[prefi] = None
				if len(prefs_changed) >= NUM_PREFS_TRIGGER_CHANGE or len(differences[0]) >= NUM_TOTAL_TRIGGER_CHANGE:
					print("Indices: {}, Prefixes : {} changed, so measuring now...".format(differences, list(prefs_changed)))
					self.measure_ingresses(self.optimization_advertisement)
					self.last_gti = self.calculate_ground_truth_ingress(self.optimization_advertisement)
					measured_this_round = True
					self.last_measured_advertisement = self.optimization_advertisement.copy()
				
			## measure
			timers.append(time.time() - t_last)
			t_last = time.time()

			# Calculate, advertise & measure information about the prefix that would 
			# give us the most new information
			if self.verbose:
				tsmaxinfo = time.time()
			if self.simulated: ## maybe tmp
				for maxinfoi in range(self.n_max_info_iter):
					maximally_informative_advertisement = self.solve_max_information(self.optimization_advertisement)
					if maximally_informative_advertisement is not None:
						print("Found an interesting advertisement on iteration {}, so measuring...".format(maxinfoi))
						self.measure_ingresses(maximally_informative_advertisement)
					else:
						if self.verbose:
							print("No further maximally informative advertisement to measure.")
						break
				if self.verbose:
					print("finding max info took {}s ".format(round(time.time() - tsmaxinfo,2)))
			
			## info
			timers.append(time.time() - t_last)
			t_last = time.time()
			
			if self.simulated:
				## Check stopping conditions
				self.stop_tracker(self.optimization_advertisement)
			else:
				## Check stopping conditions if we measured this round, to avoid excessive measurements
				if measured_this_round:
					self.stop_tracker(self.optimization_advertisement)
				else:
					self.stop_tracker(self.optimization_advertisement, skip_measuring=True)


			self.iter += 1

			## stop
			timers.append(time.time() - t_last)
			t_last = time.time()

			self.t_per_iter = (time.time() - t_start) / self.iter
			if self.iter % PRINT_FREQUENCY(self.dpsize) == 0 and self.verbose:
				print("Optimizing, iter: {}, t_per_iter : {}s, GTO: {}, RD: {}, RDE: {}, {} path measures".format(
					self.iter, round(self.t_per_iter,2), 
					self.metrics['actual_nonconvex_objective'][-1],self.rolling_delta, self.rolling_delta_eff,
					self.path_measures))

				try:
					self.make_plots()
				except:
					import traceback
					traceback.print_exc()

			for t,lab in zip(timers, ['grads','measure','info','stop']):
				print("Timer: {} -- {} s".format(lab, round(t,2)))
			self.calc_times = list(zip(timers, ['grads','measure','info','stop']))

			print("Updated numbers of popps on per prefix.")
			print(np.sum(threshold_a(self.optimization_advertisement),axis=0))

		self.reset_ugs()

		if self.verbose:
			print("Stopped train loop on {}, t per iter: {}s, {} path measures, O:{}, RD: {}, RDE: {}".format(
				self.iter, round(self.t_per_iter,2), self.path_measures, 
				self.current_pseudo_objective, self.rolling_delta, self.rolling_delta_eff))
		self.metrics['t_per_iter'] = self.t_per_iter

def main():
	try:
		import sys
		np.random.seed(31415)
		dpsize = sys.argv[1]
		deployment = get_random_deployment(dpsize)

		## useful for fixing the deployment between testing various settings
		# deployment = pickle.load(open('runs/1710776224-small-sparse/state-0.pkl','rb'))['deployment']

		lambduh = .0001
		gamma = 2.0
		n_prefixes = deployment_to_prefixes(deployment)
		sas = Sparse_Advertisement_Solver(deployment, 
			lambduh=lambduh,verbose=True,with_capacity=True,n_prefixes=n_prefixes,
			using_resilience_benefit=True, gamma=gamma)
		wm = Worker_Manager(sas.get_init_kwa(), deployment)
		wm.start_workers()
		sas.set_worker_manager(wm)
		sas.solve()
		print(sas.get_ground_truth_latency_benefit(sas.optimization_advertisement))
		soln = sas.get_last_advertisement()
		sas.make_plots()
		plot_lats_from_adv(sas, soln, 'basic_run_demo_{}.pdf'.format(sas.dpsize))

		compare_estimated_actual_per_user()

	except:
		import traceback
		traceback.print_exc()
	finally:
		wm.stop_workers()


if __name__ == "__main__":
	main()
