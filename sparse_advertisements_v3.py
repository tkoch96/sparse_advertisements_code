import matplotlib.pyplot as plt, copy, time, numpy as np, itertools, pickle, geopy.distance, warnings, tqdm
from subprocess import call, check_output
np.setbufsize(262144*8)
np.set_printoptions(precision=2)
np.random.seed(31414)
import random
random.seed(31415)
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

class Sparse_Advertisement_Wrapper(Optimal_Adv_Wrapper):
	def __init__(self, *args, init={'type':'using_objective'}, explore='bimodality',
			using_resilience_benefit=False, **kwargs):
		super().__init__(*args, **kwargs)
		# (hyper-) parameters
		self.iter = 0
		self.initialization = init
		self.explore = explore
		self.stopping_condition = lambda el : el[0] > self.max_n_iter or el[1] < self.epsilon or np.abs(el[2]) < self.epsilon
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
		return {
			'lambduh': self.lambduh, 
			'gamma': self.gamma, 
			'with_capacity': self.with_capacity,
			'verbose': False,
			'init': self.initialization,
			'explore': self.explore,
			'using_resilience_benefit': self.using_resilience_benefit,
			'n_prefixes': self.n_prefixes,
		}

	def reset_metrics(self):
		# For analysis
		self.metrics = {}
		for k in ['actual_nonconvex_objective', 'advertisements', 'effective_objectives', 
			'pseudo_objectives', 'grads', 'cost_grads', 'l_benefit_grads', 'res_benefit_grads',
			'path_likelihoods', 'EL_difference', 'resilience_benefit', 'latency_benefit']:
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
		base_args, base_kwa = self.lb_args_queue[0]
		if ugs is not None:
			base_kwa['ugs'] = ugs
		base_adv, = base_args
		self.compressed_lb_args_queue = [(base_args, base_kwa)]
		for other_args, kwa in self.lb_args_queue[1:]:
			other_adv, = other_args
			if ugs is not None:
				kwa['ugs'] = ugs
			self.compressed_lb_args_queue.append((np.where(base_adv!=other_adv), kwa))

	def flush_latency_benefit_queue(self, **kwargs):

		self.compress_lb_args_queue(**kwargs)

		# msg = pickle.dumps(['calc_lb', self.lb_args_queue])
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
				mean, (vals,pdf) = ret[adv_ret_i]
				lbxs[adv_ret_i, :, worker_i] = vals
				pdfs[adv_ret_i, :, worker_i] = pdf

		### Convert all pdfs to be at the same scale
		inds = np.arange(LBX_DENSITY)
		for adv_ret_i in range(n_to_flush):
			new_max, new_min = np.max(lbxs[adv_ret_i,:,:].flatten()), np.min(lbxs[adv_ret_i,:,:].flatten())
			for worker_i in range(n_workers):
				old_min, old_max = lbxs[adv_ret_i,0,worker_i],lbxs[adv_ret_i,-1,worker_i]
				if old_min == new_min and old_max == new_max:
					continue
				rescaled_pdf = np.zeros(pdfs[adv_ret_i,:,worker_i].shape)
				remap_arr = (old_min + inds * (old_max - old_min) / LBX_DENSITY - new_min) * LBX_DENSITY / (new_max - new_min)
				remap_arr = np.round(remap_arr).astype(np.int32).clip(0,LBX_DENSITY-1)
				for ind in np.where(pdfs[adv_ret_i,:,worker_i] > 0)[0]:
					rescaled_pdf[remap_arr[ind]] += pdfs[adv_ret_i,ind,worker_i]
				lbxs[adv_ret_i,:,worker_i] = np.linspace(new_min, new_max, LBX_DENSITY)
				pdfs[adv_ret_i,:,worker_i] = rescaled_pdf

		for adv_ret_i in range(n_to_flush):
			## point density x number of cores
			if n_workers > 1:
				x,px = self.pdf_sum_function(lbxs[adv_ret_i,...],pdfs[adv_ret_i,...],**kwargs)
			else:
				x,px = lbxs[adv_ret_i,:,0], pdfs[adv_ret_i,:,0]
			mean = np.sum(px.flatten()*x.flatten())
			ret_to_call[adv_ret_i] = (mean, (x.flatten(), px.flatten()))

			if kwargs.get('verbose'):
				for _z,_x,_y in zip(*(np.where(pdfs>.01))):
					print("{},{}-> {},{}".format(_x,_y,lbxs[adv_ret_i,_x,_y],pdfs[_z,_x,_y]))
				for _x,_y in zip(*(np.where(px>.01))):
					print("{},{}-> {},{}".format(_x,_y,x[_x],px[_x,_y]))
				exit(0)

		self.lb_args_queue = []
		self.get_cache()
		return ret_to_call

	def latency_benefit(self, *args, **kwargs):
		self.lb_args_queue.append((copy.deepcopy(args),copy.deepcopy(kwargs)))
		if kwargs.get('retnow', False):
			return self.flush_latency_benefit_queue(**kwargs)[0]

	def resilience_benefit(self, a):
		"""1 / n_popp * sum over peers of E(benefit when that peer is knocked out)."""

		tmp = np.ones(a.shape)
		self.latency_benefit_fn(a)
		for peer in self.popps:
			tmp[self.popp_to_ind[peer],:] = 0
			self.latency_benefit_fn(a * tmp)
			tmp[self.popp_to_ind[peer],:] = 1
		rets = self.flush_latency_benefit_queue()
		benefit = 0
		base_b, _ = rets[0]
		for b,_ in rets[1:]:
			benefit += b
		return base_b - benefit / self.n_popp

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

			save_setting = copy.copy(self.using_resilience_benefit)
			self.using_resilience_benefit = False
			a = np.ones((self.n_popp, self.n_prefixes))
			obj_on = self.modeled_objective(a)
			advs = {}
			exclude = []

			### Cache results
			# need to turn off resilience benefit if its on, because
			# that would make this init O(n_popp^2) computations which 
			# isn't practical / worth it
			for i in range(self.n_popp):
				a[i,:] = 0
				self.latency_benefit_fn(a)
				a[i,:] = 1
				
			self.flush_latency_benefit_queue()
			for i in range(self.n_popp):
				a[i,:] = 0
				if self.modeled_objective(a) < obj_on:
					exclude.append(i)
				a[i,:] = 1
			self.using_resilience_benefit = save_setting

			# everything off, to start, with some jitter
			a = .35 * np.ones((self.n_popp, self.n_prefixes)) + (.2 * (np.random.uniform(size=a.shape) - .5 ))
			a[:,0] = .55 # anycast on the first prefix
			a[:,1] = .55 # our belief about which popps are generally helpful on the second prefix
			if len(exclude) > 0:
				a[np.array(exclude),1] = .45
			## then just generally very little on otherwise
			prob_on = np.minimum((MAX_LATENCY - MIN_LATENCY ) / 2 / (self.lambduh * self.n_popp * (self.n_prefixes - 2)), .05)
			is_on = np.random.random(size=(self.n_popp, self.n_prefixes)) < prob_on
			is_on[:,0:2] = False
			a[is_on] = .55
			a += .02 * (np.random.uniform(size=a.shape) - .5)
			print("Done Initializing")

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
			resilience_benefit = self.resilience_benefit_fn(a)
		else:
			resilience_benefit = 0

		if self.verbose:
			benefits,probs = u
			ex = np.average(benefits,weights=probs+1e-8)
			exsq = np.average(np.power(benefits,2),weights=probs+1e-8)
			var = exsq - np.power(ex,2)
			std = np.sqrt(var)
			print("Believed: NP: {}, LB: {} ({} std dev), RB: {}".format(norm_penalty,
				latency_benefit, std, resilience_benefit))
		return self.lambduh * norm_penalty - (latency_benefit + self.gamma * resilience_benefit)

class Sparse_Advertisement_Eval(Sparse_Advertisement_Wrapper):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.sas = Sparse_Advertisement_Solver(*args, **kwargs)

	def solve_extremes(self, verbose=True):
		# Find minimal and maximal set of advertisements
		# maximal is a single anycast advertisement, minimal is transit providers
		maximal_advertisement = np.zeros((self.n_popp, self.n_prefixes))
		maximal_advertisement[:,0] = 1
		maximal_objective = self.measured_objective(maximal_advertisement)


		minimal_advertisement = np.zeros((self.n_popp, self.n_prefixes))
		for tpopp in self.provider_popps:
			minimal_advertisement[self.popp_to_ind[tpopp],0] = 1
		### TODO
		minimal_objective = self.measured_objective(minimal_advertisement)
		minimal_advertisement = minimal_advertisement

		self.extreme = {
			'minimal_advertisement': minimal_advertisement,
			'minimal_objective': minimal_objective,
			'maximal_advertisement': maximal_advertisement,
			'maximal_objective': maximal_objective,
		}

	def solve_oracle(self,verbose=True):
		# To get the oracle solution, try every possible combination of advertisements
		# Not possible for problems that are too large
		n_arr = self.n_popp * self.n_prefixes ### TODO -- only calculate up to a permutation?
		logn_possibilities = n_arr
		if logn_possibilities >= np.log2(1e6):
			print("Note -- too complex to get oracle solution. Skipping")
			self.oracle = None
			return
		n_possibilities = 2**n_arr
		a = np.zeros((n_arr,))
		all_as = []
		objs = np.zeros((n_possibilities,))
		actual_objs = np.zeros((n_possibilities,))
		for i in range(n_possibilities):
			b = str(bin(i))[2:]
			ib = "0" * (n_arr - len(b)) + b
			for j,el in enumerate(ib):
				a[j] = int(el) * 1.0
			objs[i] = self.measured_objective(a.reshape((self.n_popp, self.n_prefixes)))
			actual_objs[i] = self.actual_nonconvex_objective(a.reshape((self.n_popp,self.n_prefixes)))
			all_as.append(copy.deepcopy(a))

		# Approx
		oracle_objective = np.min(objs)
		approx_oracle_adv = all_as[np.argmin(objs)].reshape(self.n_popp, self.n_prefixes)

		# Actual
		actual_oracle_objective = np.min(actual_objs)
		l0_oracle_adv = all_as[np.argmin(actual_objs)].reshape(self.n_popp, self.n_prefixes)

		if verbose:
			print("Oracle advertisements\n Approx for C(a) ({}): {}\nWith L0 ({}): {}".format(
				round(oracle_objective,2), approx_oracle_adv,
				round(actual_oracle_objective,2), l0_oracle_adv))

		self.oracle = {
			'approx_objective': oracle_objective,
			'l0_objective': actual_oracle_objective,
			'approx_advertisement': approx_oracle_adv,
			'l0_advertisement': l0_oracle_adv,
		}

	def solve_anyopt(self, **kwargs):
		self.anyopt = Anyopt_Adv_Solver(self.deployment, **self.get_init_kwa())
		self.anyopt.set_worker_manager(self.get_worker_manager())
		self.anyopt.solve()

		anyopt_adv = self.anyopt.advs
		anyopt_obj = self.measured_objective(anyopt_adv)

		self.anyopt_solution = {
			'objective': anyopt_obj,
			'advertisement': anyopt_adv,
			'latency_benefit':  self.get_ground_truth_latency_benefit(anyopt_adv),
			'prefix_cost': self.prefix_cost(anyopt_adv),
			'norm_penalty': self.advertisement_cost(anyopt_adv),
			'n_advs': self.anyopt.path_measures,
		}
		self.clear_caches()

	def solve_sparse(self, **kwargs):

		## Initialize advertisement, solve sparse
		init_adv = kwargs.get('init_adv')
		if init_adv is None:
			adv = self.init_advertisement()
		else:
			adv = init_adv
		self.sas.update_deployment(self.deployment)			
		self.sas.set_worker_manager(self.get_worker_manager())
		self.sas.solve(init_adv=adv)
		self.sas.make_plots()

		sparse_adv = threshold_a(self.sas.get_last_advertisement())
		sparse_objective = self.measured_objective(sparse_adv)

		self.sparse_solution = {
			'objective': sparse_objective,
			'latency_benefit':  self.get_ground_truth_latency_benefit(sparse_adv),
			'norm_penalty': self.advertisement_cost(sparse_adv),
			'prefix_cost': self.prefix_cost(sparse_adv),
			'advertisement': sparse_adv,
			'n_advs': self.sas.path_measures,
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

	def solve_one_per_pop(self, **kwargs):
		# Solve for the one per pop solution
		deployment = copy.deepcopy(self.deployment)
		self.one_per_pop = Painter_Adv_Solver(deployment, **self.get_init_kwa())
		self.one_per_pop.set_worker_manager(self.get_worker_manager())

		self.one_per_pop.one_per_pop()
		one_per_pop_adv = self.one_per_pop.painter_advs_to_sparse_advs(self.one_per_pop.advs)
		one_per_pop_obj = self.measured_objective(one_per_pop_adv)
		self.one_per_pop_solution = {
			'objective': one_per_pop_obj,
			'latency_benefit':  self.get_ground_truth_latency_benefit(one_per_pop_adv),
			'norm_penalty': self.advertisement_cost(one_per_pop_adv),
			'prefix_cost': self.prefix_cost(one_per_pop_adv),
			'advertisement': one_per_pop_adv,
			'n_advs': 1,
		}

		self.clear_caches()

	def solve_painter(self, **kwargs):
		## Solve for the painter solution
		# painter is an improvement over anycast, so it has one less prefix to work with
		# and its assumed the first column will be anycast
		deployment = copy.deepcopy(self.deployment)
		self.painter = Painter_Adv_Solver(deployment, **self.get_init_kwa())
		self.painter.set_worker_manager(self.get_worker_manager())

		self.painter.painter_v5(cd=2000)
		painter_adv = self.painter.painter_advs_to_sparse_advs(self.painter.advs)
		painter_obj = self.measured_objective(painter_adv)
		# print("Painter Adv, obj: {} {}".format(painter_adv, painter_obj))
		self.painter_solution = {
			'objective': painter_obj,
			'latency_benefit':  self.get_ground_truth_latency_benefit(painter_adv),
			'norm_penalty': self.advertisement_cost(painter_adv),
			'prefix_cost': self.prefix_cost(painter_adv),
			'advertisement': painter_adv,
			'n_advs': self.painter.path_measures,
		}

		self.clear_caches()

	def compare_different_solutions(self, **kwargs):
		verbose = kwargs.get('verbose', True)
		init_adv = kwargs.get('init_adv')
		
		solution_types = ['sparse', 'painter','anyopt', 'one_per_pop', 'maximal', 'minimal','oracle']
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
			'max_sparse_benefits': [],
			'max_painter_benefits': [],
		}
		for i in range(kwargs.get('n_run', 50)):
			if verbose:
				print("Comparing different solutions iteration {}".format(i))

			## Anyopt
			if verbose:
				print("solving anyopt")
			self.solve_anyopt(**kwargs)
			metrics['sparse_objective_vals']['anyopt'].append(self.anyopt_solution['objective'])
			metrics['n_advs']['anyopt'].append(self.anyopt_solution['n_advs'])
			metrics['adv_solns']['anyopt'].append(self.anyopt_solution['advertisement'])
			metrics['latency_benefits']['anyopt'].append(self.anyopt_solution['latency_benefit'])
			metrics['norm_penalties']['anyopt'].append(self.anyopt_solution['norm_penalty'])
			metrics['prefix_cost']['anyopt'].append(self.anyopt_solution['prefix_cost'])

			## One per PoP
			if verbose:
				print("solving one per pop")
			self.solve_one_per_pop(**kwargs)
			metrics['sparse_objective_vals']['one_per_pop'].append(self.one_per_pop_solution['objective'])
			metrics['n_advs']['one_per_pop'].append(self.one_per_pop_solution['n_advs'])
			metrics['adv_solns']['one_per_pop'].append(self.one_per_pop_solution['advertisement'])
			metrics['latency_benefits']['one_per_pop'].append(self.one_per_pop_solution['latency_benefit'])
			metrics['norm_penalties']['one_per_pop'].append(self.one_per_pop_solution['norm_penalty'])
			metrics['prefix_cost']['one_per_pop'].append(self.one_per_pop_solution['prefix_cost'])			

			## Painter
			if verbose:
				print("solving painter")
			self.solve_painter(**kwargs)
			metrics['sparse_objective_vals']['painter'].append(self.painter_solution['objective'])
			metrics['n_advs']['painter'].append(self.painter_solution['n_advs'])
			metrics['adv_solns']['painter'].append(self.painter_solution['advertisement'])
			metrics['latency_benefits']['painter'].append(self.painter_solution['latency_benefit'])
			metrics['norm_penalties']['painter'].append(self.painter_solution['norm_penalty'])
			metrics['prefix_cost']['painter'].append(self.painter_solution['prefix_cost'])

			## Sparse
			if verbose:
				print("solving sparse")
			self.solve_sparse(**kwargs)

			metrics['sparse_objective_vals']['sparse'].append(self.sparse_solution['objective'])
			metrics['n_advs']['sparse'].append(self.sparse_solution['n_advs'])
			metrics['adv_solns']['sparse'].append(self.sparse_solution['advertisement'])
			metrics['latency_benefits']['sparse'].append(self.sparse_solution['latency_benefit'])
			metrics['norm_penalties']['sparse'].append(self.sparse_solution['norm_penalty'])
			metrics['prefix_cost']['sparse'].append(self.sparse_solution['prefix_cost'])

			## Extremes
			self.solve_extremes(verbose=verbose)
			metrics['sparse_objective_vals']['maximal'].append(self.extreme['maximal_objective'])
			metrics['sparse_objective_vals']['minimal'].append(self.extreme['minimal_objective'])
			metrics['latency_benefits']['maximal'].append(self.get_ground_truth_latency_benefit(self.extreme['maximal_advertisement']))
			metrics['latency_benefits']['minimal'].append(self.get_ground_truth_latency_benefit(self.extreme['minimal_advertisement']))

			### Add old objective values to see how differences in what we optimize shape optimality
			for k,adv in zip(['sparse', 'painter', 'anyopt', 'minimal', 'maximal'], [self.sparse_solution['advertisement'], self.painter_solution['advertisement'],
				self.anyopt_solution['advertisement'], self.extreme['minimal_advertisement'], self.extreme['maximal_advertisement']]):
				metrics['painter_objective_vals'][k].append(self.painter_objective(adv))
				metrics['anyopt_objective_vals'][k].append(self.anyopt_objective(adv))
				metrics['normalized_sparse_benefit'][k].append(self.get_normalized_benefit(adv))
			if verbose:
				print("Solving oracle")
			## oracle
			self.solve_oracle(verbose=verbose)

			for k in solution_types:
				if k in ['sparse', 'oracle']: continue
				metrics['objective_diffs'][k].append(metrics['sparse_objective_vals'][k][-1] - \
					metrics['sparse_objective_vals']['sparse'][-1])
				metrics['latency_benefit_diffs'][k].append(metrics['latency_benefits'][k][-1] - \
					metrics['latency_benefits']['sparse'][-1])

			### Best improvement over anycast for every user
			max_painter_benefit = self.get_max_painter_benefit()
			metrics['max_painter_benefits'].append(max_painter_benefit)

			max_benefit = self.get_max_overall_benefit()
			metrics['max_sparse_benefits'].append(max_benefit)

			if not kwargs.get('dont_update_deployment', False):
				## Update to new random deployment
				new_deployment = get_random_deployment(kwargs.get('deployment_size','small'))
				self.update_deployment(new_deployment)
			if verbose:
				print(metrics['sparse_objective_vals'])
		pickle.dump(metrics, open(os.path.join(CACHE_DIR, 'method_comparison_metrics.pkl'),'wb'))
		if verbose:
			plt.rcParams["figure.figsize"] = (10,5)
			plt.rcParams.update({'font.size': 22})
			### Just compare (sparse) objective function values
			for k in solution_types:
				v = 1
				try:
					x,cdf_x = get_cdf_xy(metrics['sparse_objective_vals'][k])
					plt.plot(x,cdf_x,label=k.capitalize())
				except IndexError:
					continue
			plt.ylim([0,1.0])
			plt.legend()
			plt.xlabel("Final Objective Function Value")
			plt.ylabel("CDF of Trials")
			save_fig("comparison_to_strategies_demonstration.pdf")

			### Difference between objective function of sparse and the other method
			plt.rcParams["figure.figsize"] = (10,15)
			plt.rcParams.update({'font.size': 22}) 
			f,ax = plt.subplots(3,1)  
			for i, alg in enumerate(['sparse', 'painter', 'anyopt']):
				base_arr = np.array(metrics['{}_objective_vals'.format(alg)][alg])
				for k in solution_types:
					if k == alg or k == 'oracle': continue
					v = 1
					try:
						x,cdf_x = get_cdf_xy(base_arr - np.array(metrics['{}_objective_vals'.format(alg)][k]))
						ax[i].plot(x,cdf_x,label=k.capitalize())
					except IndexError:
						continue
				ax[i].set_ylim([0,1.0])
				ax[i].grid(True)
				ax[i].set_xlabel("{} - Other Methods, Objective Values".format(alg))
				ax[i].set_ylabel("CDF of Trials")
				ax[i].legend(fontsize=12)
			save_fig("all_objective_comparisons.pdf")

			### Difference between latency benefit of sparse and the other method
			plt.rcParams["figure.figsize"] = (10,5)
			plt.rcParams.update({'font.size': 22})
			for k in solution_types:
				v = 1
				try:
					x,cdf_x = get_cdf_xy(metrics['latency_benefit_diffs'][k])
					plt.plot(x,cdf_x,label=k.capitalize())
				except IndexError:
					continue
			plt.ylim([0,1.0])
			plt.legend()
			plt.xlabel("Latency Benefit Difference Between Sparse and Method")
			plt.ylabel("CDF of Trials")
			save_fig("latency_benefit_comparison_to_strategies_vs_sparse_demonstration.pdf")

			## scatter adv difference vs objective function difference
			plt.rcParams["figure.figsize"] = (10,5)
			plt.rcParams.update({'font.size': 22})
			compares = [('painter', 'sparse')]
			for compare in compares:
				n_adv_differences, obj_fn_differences = [], []
				for i in range(len(metrics['sparse_objective_vals'][compare[0]])):
					obj_fn_differences.append(metrics['sparse_objective_vals'][compare[0]][i] - \
						metrics['sparse_objective_vals'][compare[1]][i])
					n_adv_differences.append(metrics['n_advs'][compare[0]][i] - \
						metrics['n_advs'][compare[1]][i])
				plt.rcParams["figure.figsize"] = (10,5)
				plt.rcParams.update({'font.size': 22})
				plt.scatter(obj_fn_differences,n_adv_differences)
				plt.xlabel("Objective Function Difference")
				plt.ylabel("Number Advertisements Delta")
				save_fig("nadv_vs_objective--{}_vs_{}.pdf".format(compare[0],compare[1]))


		return metrics

class Sparse_Advertisement_Solver(Sparse_Advertisement_Wrapper):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.beta = .3 # gradient momentum
		self.sigmoid_k = 5.0 # heavisside gradient parameter

		self.gradient_support = [(a_i,a_j) for a_i in range(self.n_popp) for a_j in range(self.n_prefixes)]
		max_support = 200#self.n_popp * self.n_prefixes
		self.gradient_support_settings = {
			'calc_every': 5,
			'support_size': np.minimum(self.n_popp * self.n_prefixes,max_support), # setting this value to size(a) turns it off
		}


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
	
	def grad_latency_benefit(self, a):
		L_grad = np.zeros(a.shape)
		a_effective = threshold_a(a).astype(bool)

		inds = self.gradient_support
		if self.iter % self.gradient_support_settings['calc_every'] == 0:
			# periodically calculate all gradients
			inds = [(a_i,a_j) for a_i in range(a.shape[0]) for a_j in range(a.shape[1])]
		dont_calc_inds = get_difference([(a_i,a_j) for a_i in range(a.shape[0]) for a_j in range(a.shape[1])], inds)
		
		calls = []
		for a_i,a_j in inds:
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
		all_lb_rets = self.flush_latency_benefit_queue()
		for i, call_ind in enumerate(calls):
			ind, before_then_after = call_ind
			if before_then_after == 'ba':
				before,_ = all_lb_rets[2*i]
				after, _ = all_lb_rets[2*i+1]
			else:
				after,_ = all_lb_rets[2*i]
				before, _ = all_lb_rets[2*i+1]
			L_grad[ind] = self.heaviside_gradient(before, after, a[ind])
		
		L_grad = L_grad.clip(-1,1)

		if self.iter % self.gradient_support_settings['calc_every'] == 0:
			self.update_gradient_support(L_grad)
			self.last_full_grad = L_grad
		else:
			# Carry through these inds
			for ind in dont_calc_inds:
				L_grad[ind] = self.last_full_grad[ind]

		return L_grad

	def gradients(self, a, add_metrics=True):
		# gradient is the proximal gradient of the L1 norm
		# minus lambduh times gradient of L 
		# gradient of L is calculated via a continuous approximation
		if self.verbose:
			ts = time.time()
		L_grad = self.grad_latency_benefit(a)
		if self.verbose:
			print("Calcing latency benefit grad took {}s".format(int(time.time() - ts)))
		if self.verbose:
			ts = time.time()
		res_grad = self.gradients_resilience_benefit_fn(a)
		if self.verbose:
			print("Calcing resilience benefit grad took {}s".format(int(time.time() - ts)))
		if add_metrics:
			self.metrics['l_benefit_grads'].append(L_grad)
			self.metrics['res_benefit_grads'].append(self.gamma * res_grad)
			self.metrics['cost_grads'].append(self.lambduh * self.alpha * np.ones(L_grad.shape))
		return -1 * (L_grad + self.gamma * res_grad)

	def gradients_sigmoid(self, a):
		# LB Grad
		l_grad = self.gradients(a, add_metrics=False)
		## calculate gradient of peer cost
		# old -- sum
		# peer_grad = self.heaviside_gradient_sigmoid(np.sum(a,axis=1))
		# peer_grad = np.tile(np.expand_dims(peer_grad,1),(1,self.n_prefixes))
		# pref_grad = self.heaviside_gradient_sigmoid(np.sum(a,axis=0))
		# pref_grad = np.tile(np.expand_dims(pref_grad,0),(self.n_popp,1))

		# prod cost
		S = self.heaviside_gradient_sigmoid(a)
		peer_prod = np.prod( 1 -  1 / (1 + np.exp(-1 * self.sigmoid_cost_k * (a - ADVERTISEMENT_THRESHOLD) )), axis=1)
		peer_prod = np.tile(np.expand_dims(peer_prod,axis=1),(1,self.n_prefixes))
		peer_grad = peer_prod * S

		pref_prod = np.prod( 1 -  1 / (1 + np.exp(-1 * self.sigmoid_cost_k * (a - ADVERTISEMENT_THRESHOLD) )), axis=0)
		pref_prod = np.tile(np.expand_dims(pref_prod,axis=0),(self.n_popp, 1))
		pref_grad = pref_prod * S		

		cost_grad = self.lambduh * (peer_grad + pref_grad)

		self.metrics['cost_grads'].append(cost_grad)
		self.metrics['l_benefit_grads'].append(l_grad)

		return cost_grad + l_grad

	def gradients_resilience_benefit(self, a):
		### Positive resilience benefit gradient means turning a popp
		## on will increase resilience
			
		### Heuristic way of calculating resilience benefit gradients

		### need to somehow model the effect of
		## if this is off, and that other thing dies, you're fucked
		## i.e., resilience means encouraging turning on popp,prefixes that will "save the day"
		## when something gets overloaded
		## 1 idea is to kill a popp or 2 randomly, twiddle a random (prefix,popp) pair and see what happens
		
		#### PROBLEM: resilience is a popp x popp problem, so rare for a uniform sample to hit it right
		#### Q is, how can we sample this space intelligently?

		grad_rb = np.zeros(a.shape)
		a_effective = threshold_a(a).astype(bool)
		calls = []

		## get current latency benefit
		lb_default,_ = self.latency_benefit(a_effective, retnow=True)

		total_n_grad_calc = self.gradient_support_settings['support_size']
		
		pct_explore = 20 # pct of gradient calculation budget dedicated to exploring
		N_KILLS_FOR_POPPS = int(total_n_grad_calc * pct_explore/100)
		# number of gradient calcs that re-calc previously high gradients
		N_REMEASURE = total_n_grad_calc - N_KILLS_FOR_POPPS

		try:
			n_significant = sum(1 for el in self.last_calls_results.values() if \
				np.abs(el) > .05)
			print("Last call, {} were significant".format(n_significant))
			best_from_last_time = sorted(self.last_calls_results.items(), key = lambda el : 
				-1 * np.abs(el[1]))[0:N_REMEASURE]

			for (popp,rand_popp,rand_outer_prefix),val in best_from_last_time:
				if (popp,rand_popp,rand_outer_prefix) in calls: 
					continue
				if self.gamma * np.abs(val) < self.lambduh:
					# if it's not important enough to warrant the cost, don't bother
					continue
				tmp_a = copy.copy(a_effective)
				tmp_a[self.popp_to_ind[popp],rand_outer_prefix] = True # Turn this popp on
				tmp_a[self.popp_to_ind[rand_popp],:] = False # Also kill this random popp
				self.latency_benefit(tmp_a)
				tmp_a[self.popp_to_ind[popp],rand_outer_prefix] = False
				self.latency_benefit(tmp_a)
				calls.append((popp, rand_popp,rand_outer_prefix))
		except AttributeError: # there are no last calls on the first iteration
			pass

		N_REMEASURE = len(calls)
		N_KILLS_FOR_POPPS = total_n_grad_calc - N_REMEASURE

		## turn the popp under consideration on
		## turn a random popp,prefix that was previously on, off
		## notice LB(now) - LB default (a) to get a comparison to how useful it is on
		## turn the popp under consideration off
		## notice LB(now) - LB default (b) to measure how bad it would be if this popp failed

		## if (b) is much more negative than (a), we should encourage resilience

		## to encourage resilience we do heavisside(b,a)

		# TODO : could bias towards popps that 'matter' more
		rand_popp_choices = np.random.randint(low=0,high=self.n_popps,
			size=N_KILLS_FOR_POPPS) 
		 # associated prefix distribution should be biased towards prefixes that are far from 1 and 0
		random_prefix_choices = np.zeros(N_KILLS_FOR_POPPS,dtype=np.int32)
		choices = np.arange(self.n_prefixes)
		### TODO -- reassess
		for i in range(N_KILLS_FOR_POPPS):
			prob_each_pref = .5 - np.abs(a[rand_popp_choices[i],:] - ADVERTISEMENT_THRESHOLD) + .001
			prob_each_pref = prob_each_pref / np.sum(prob_each_pref)
			prob_each_pref = np.ones(self.n_prefixes) / self.n_prefixes
			random_prefix_choices[i] = int(np.random.choice(choices, p=prob_each_pref))
		# pairing popps with popps
		# popp -> users -> benefit
		# but also users -> popps
		# support (popp1 -> popp2) is sum_over_users( potential backup popp1 can provide to popp2 for that user)
		# backup is ~ how good that option is compared to user's other options
		# lots of backup for very good relative option
		_, ug_catchments = self.calculate_user_choice(a_effective)
		self.popp_support = np.zeros((self.n_popps, self.n_popps))
		for popp1 in self.popps:
			popp1i = self.popp_to_ind[popp1]
			for popp2 in self.popps:
				if popp1 == popp2: continue
				for ug in self.ug_perfs:
					if ug_catchments[self.ug_to_ind[ug]] != self.popp_to_ind[popp1]:
						continue
					try:
						# support popp2 provides to popp1
						# if popp1 fails, popp2 should be really close to the latency
						if self.ug_perfs[ug][popp2] < self.ug_perfs[ug][popp1]:
							backup = 1
						else:
							# should multiply this by probability of user using this ingress anyway
							backup = np.minimum(1/(self.ug_perfs[ug][popp2] - self.ug_perfs[ug][popp1]+.00001),1)
					except KeyError:
						continue
					popp2i = self.popp_to_ind[popp2]
					self.popp_support[popp1i,popp2i] += backup

		self.popp_support += .0001
		self.popp_sample_probs = (self.popp_support.T / np.sum(self.popp_support, axis=1)).T
		# print(np.round(self.popp_sample_probs,2))

		for poppi, rand_outer_prefix in zip(rand_popp_choices,random_prefix_choices):
			popp = self.popps[poppi]
			this_popp_random_kill = np.random.choice(np.arange(self.n_popp) ,
				 p=self.popp_sample_probs[poppi,:])
			tmp_a = copy.copy(a_effective)
			tmp_a[self.popp_to_ind[popp],rand_outer_prefix] = True # Turn this popp on
			rand_popp = self.popps[this_popp_random_kill]
			tmp_a[this_popp_random_kill,:] = False # Also kill this random popp
			self.latency_benefit(tmp_a)
			tmp_a[self.popp_to_ind[popp],rand_outer_prefix] = False
			self.latency_benefit(tmp_a)
			calls.append((popp, rand_popp, rand_outer_prefix))
		


		all_lb_rets = self.flush_latency_benefit_queue()

		self.last_calls_results = {}
		ind = 0
		for call_popp, killed_popp, rand_outer_prefix in calls:
			before,_ = all_lb_rets[ind] # popp under consideration on
			after, _ = all_lb_rets[ind+1] # popp under consideration off

			delta_a = before - lb_default
			delta_b = after - lb_default

			this_grad = self.heaviside_gradient(
				delta_b, delta_a, a[self.popp_to_ind[call_popp],rand_outer_prefix])
			self.last_calls_results[call_popp,killed_popp,rand_outer_prefix] = this_grad
			# print("identifying resilience of entry {} ({}),pref {}, killing {} ({})".format(
			# 	call_popp, self.popp_to_ind[call_popp], rand_outer_prefix,
			# 	killed_popp, self.popps[killed_popp]))
			# print("delta_b {} delta_a {}, grad: {}".format(delta_b,delta_a,this_grad))
			grad_rb[self.popp_to_ind[call_popp],rand_outer_prefix] += this_grad

			ind += 2

		return grad_rb.clip(-1,1)

	def impose_advertisement_constraint(self, a):
		"""The convex constraint 0 <= a_ij <= 1 has the simple solution to clip."""
		a = np.clip(a,0,1.0)
		return a

	def make_plots(self, *args, **kwargs):
		plt.rcParams["figure.figsize"] = (10,10)
		plt.rcParams.update({'font.size': 14})
		f,ax = plt.subplots(5,2)

		pickle.dump(self.metrics, open(os.path.join(CACHE_DIR, 'metrics.pkl'),'wb'))

		# General convergence metrics plot
		all_as = np.array(self.metrics['advertisements'])
		all_grads = np.array(self.metrics['grads'])
		all_cost_grads = np.array(self.metrics['cost_grads'])
		all_l_benefit_grads = np.array(self.metrics['l_benefit_grads'])
		all_res_benefit_grads = np.array(self.metrics['res_benefit_grads'])
		linestyles = ['-','*','^','>','v']
		colors = ['orange','brown','aqua','deeppink','peru','grey','k','tan']
		for pref_i in range(self.n_prefixes):
			pref_sty = linestyles[pref_i%len(linestyles)]
			for popp_i in range(self.n_popp):
				if 'xlimupper' in kwargs:
					ax[0,0].plot(kwargs['xlimupper'],all_as[:,popp_i,pref_i], pref_sty, 
						c=colors[popp_i%len(colors)], label="PoPP {} Prefix {}".format(self.popps[popp_i], pref_i))
				else:
					ax[0,0].plot(all_as[:,popp_i,pref_i][::5], pref_sty, 
						c=colors[popp_i%len(colors)], label="PoPP {} Prefix {}".format(self.popps[popp_i], pref_i))
				ax[1,0].plot(all_grads[:,popp_i,pref_i], 
					c=colors[popp_i%len(colors)], label="PoPP {} Prefix {}".format(self.popps[popp_i], pref_i))
				ax[2,0].plot(all_cost_grads[:,popp_i,pref_i], 
					c=colors[popp_i%len(colors)], label="PoPP {} Prefix {}".format(self.popps[popp_i], pref_i))
				ax[3,0].plot(all_l_benefit_grads[:,popp_i,pref_i], 
					c=colors[popp_i%len(colors)], label="PoPP {} Prefix {}".format(self.popps[popp_i], pref_i))
				ax[4,0].plot(all_res_benefit_grads[:,popp_i,pref_i], 
					c=colors[popp_i%len(colors)], label="PoPP {} Prefix {}".format(self.popps[popp_i], pref_i))
		if self.n_popp * self.n_prefixes < 20:
			ax[0,0].legend(fontsize=6)
		ax[0,0].set_ylabel("a")
		ax[1,0].set_ylabel("Net Grad")
		ax[2,0].set_ylabel("Cost Grad")
		ax[3,0].set_ylabel("LB Grad")
		ax[4,0].set_ylabel("Res Grad")

		all_objectives = self.metrics['actual_nonconvex_objective']
		all_pseudo_objectives = self.metrics['pseudo_objectives']
		all_effective_ojectives = self.metrics['effective_objectives']
		all_resilience_benefits = self.metrics['resilience_benefit']
		all_latency_benefits = self.metrics['latency_benefit']
		ax[1,1].plot(all_pseudo_objectives)
		ax[1,1].set_ylabel("Believed Objective")
		ax[0,1].plot(all_objectives)
		ax[0,1].set_ylabel("GT Objective")
		ax[2,1].plot(all_effective_ojectives)
		ax[2,1].set_ylabel("GT Effective Objective")
		ax[3,1].plot(all_resilience_benefits)
		ax[3,1].set_ylabel("Res Ben")
		ax[4,1].plot(all_latency_benefits)
		ax[4,1].set_ylabel("Lat Ben")

		save_fig('convergence_over_iterations.pdf')
		return
		# # Make probabilities plot
		# n_users = len(self.ugs)
		# n_rows = int(np.ceil(n_users/3))
		# f,ax = plt.subplots(n_rows, 3)
		# for ug in self.ugs:
		# 	ui = self.ug_to_ind[ug]
		# 	row_i,col_i = ui // 3, ui % 3
		# 	if n_rows == 1:
		# 		access = col_i
		# 	else:
		# 		access = (row_i,col_i)
		# 	user_path_probs = np.array([P[:,0,ui] for P in self.metrics['path_likelihoods']])
		# 	for popp_i in range(self.n_popp):
		# 		ax[access].plot(user_path_probs[:,popp_i],c=colors[popp_i%len(colors)],
		# 			label="PoPP {}".format(self.popps[popp_i]))
		# plt.show()
		# # Make latency estimate error plot
		# for ug in self.ugs:
		# 	ui = self.ug_to_ind[ug]
		# 	latency_estimate_errors = np.array([EL[ui] for EL in self.metrics['EL_difference']])
		# 	plt.plot(latency_estimate_errors,label="User {}".format(ug))
		# plt.show()

	def print_adv(self, a):
		for popp_i in range(self.n_popp):
			for pref_i in range(self.n_prefixes):
				print("PoPP {} Prefix {}: {}".format(self.popps[popp_i], pref_i, a[popp_i,pref_i]))

	def set_alpha(self):
		assert self.lambduh < 10
		if self.lambduh < 10 and self.lambduh > 1:
			self.alpha = .0005
		elif self.lambduh <= 1 and self.lambduh > .1:
			self.alpha = .005
		elif self.lambduh <= .1 and self.lambduh > .01:
			self.alpha = .01
		elif self.lambduh <= .01:
			self.alpha = .1

	def update_gradient_support(self, gradient):
		gradient = np.abs(gradient)
		inds = [(a_i,a_j) for a_i in range(self.n_popp) for a_j in range(self.n_prefixes)]
		sorted_inds = list(reversed(np.argsort([gradient[a_i,a_j] for a_i,a_j in inds])))
		# Focus on the largest gradients
		self.gradient_support = list([inds[i] for i in sorted_inds[0:self.gradient_support_settings['support_size']]])

	def solve_max_information(self, current_advertisement):
		"""Search through neighbors of a, calculate maximum uncertainty."""
		uncertainties = {}

		a = np.copy(threshold_a(current_advertisement))
		current_benefit,_ = self.latency_benefit_fn(a, retnow=True)
		awful_benefit = -100000
		# f,ax = plt.subplots(5)
		# self.plti=0
		def value_func(u):
			benefits,probs = u
			if len(probs) == 1:
				return awful_benefit
			if self.explore == 'positive_benefit':
				if np.sum(probs[benefits>=current_benefit]) > .99: return awful_benefit
				v = np.abs(np.sum(benefits[benefits>current_benefit] * probs[benefits>current_benefit]))
			elif self.explore == 'entropy':
				v = scipy.stats.entropy(probs+1e-8)
			elif self.explore == 'bimodality':
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
			elif self.explore == 'other_bimodality':
				negative_part = np.where(benefits <= current_benefit)[0]

				positive_part = np.where(benefits > current_benefit)[0]
				positive_mass = np.sum(probs[positive_part] * (benefits[positive_part] - current_benefit))
				negative_mass = np.sum(probs[negative_part] * (current_benefit - benefits[negative_part]))
				v = positive_mass * negative_mass
			elif self.explore == 'gmm':
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
				
			return v

		MIN_POTENTIAL_VALUE = {
			'positive_benefit': .01,
			'entropy': .01,
			'bimodality': -5,
			'other_bimodality': .01,
			'gmm': 1.5,
		}[self.explore]

		n_flips = 1
		max_time = 2 # seconds
		t_start = time.time()
		while True:
			all_inds = [(i,j) for i in range(self.n_popp) for j in range(self.n_prefixes)]
			perms = sorted(list(itertools.permutations(all_inds, n_flips)))
			np.random.shuffle(perms)
			perms = perms[0:self.gradient_support_settings['support_size']]
			for flips in perms:
				for flip in flips:
					a[flip] = 1 - a[flip]
				self.latency_benefit_fn(a)
				for flip in flips:
					a[flip] = 1 - a[flip]
			all_lb_rets = self.flush_latency_benefit_queue()

			for flipi, flips in enumerate(perms):
				_,u = all_lb_rets[flipi]
				uncertainties[flips] = u

			potential_value_measure = {}
			max_benefit = -1 * np.inf
			best_flips = None
			for flips,u in uncertainties.items():
				potential_value_measure[flips] = value_func(u)
				if potential_value_measure[flips] >= max_benefit:
					best_flips = flips
					max_benefit = potential_value_measure[flips]
			# if max_benefit > awful_benefit:
			# 	print("{} -- {} {}".format(self.explore, max_benefit, best_flips))
			if best_flips is not None:
				if potential_value_measure[best_flips] > MIN_POTENTIAL_VALUE:
					for flip in best_flips:
						a[flip] = 1 - a[flip]
					if tuple(a.flatten()) in self.measured:
						print("Re-measuring {}".format(a))
						pickle.dump(a,open('remeasure_a.pkl','wb'))
						print('woops')
						print(np.where(self.parent_tracker[7,:,:]))
						_,u = self.latency_benefit_fn(a, plotit=True,retnow=True)
						print("This flips had value: {}".format(value_func(u)))
						print(u)
						exit(0)
					return a
			n_flips += 1
			if n_flips == 2:
				return None
		# plt.close()

	def stop_tracker(self, advertisement):
		# Stop when the objective doesn't change, 
		# but use an EWMA to track the change so that we don't spuriously exit
		delta_alpha = .7
		delta_eff_alpha = .25

		# re-calculate objective
		self.last_objective = self.current_pseudo_objective
		self.last_effective_objective = self.current_effective_objective
		self.metrics['actual_nonconvex_objective'].append(self.measured_objective(advertisement, verb=True))
		self.current_pseudo_objective = self.modeled_objective(advertisement,verb=True)
		self.current_effective_objective = self.modeled_objective(threshold_a(advertisement))
		self.metrics['pseudo_objectives'].append(self.current_pseudo_objective)
		self.metrics['effective_objectives'].append(self.measured_objective(copy.copy(threshold_a(advertisement))))
		self.metrics['resilience_benefit'].append(self.resilience_benefit(advertisement))
		self.metrics['latency_benefit'].append(self.latency_benefit_fn(advertisement,retnow=True)[0])

		self.rolling_delta = (1 - delta_alpha) * self.rolling_delta + delta_alpha * np.abs(self.current_pseudo_objective - self.last_objective)
		self.rolling_delta_eff = (1 - delta_eff_alpha) * self.rolling_delta_eff + \
			delta_eff_alpha * np.abs(self.current_effective_objective - self.last_effective_objective)
		self.stop = self.stopping_condition([self.iter,self.rolling_delta,self.rolling_delta_eff])

	def get_init_kwa(self):
		return {
			'lambduh': self.lambduh, 
			'gamma': self.gamma, 
			'with_capacity': self.with_capacity,
			'verbose': False,
			'init': self.initialization,
			'explore': self.explore,
			'using_resilience_benefit': self.using_resilience_benefit,
			'n_prefixes': self.n_prefixes,
		}

	def solve(self, init_adv=None):
		self.clear_caches()
		self.set_alpha()
		self.measured = {}
		# Initialize model of path probabilities
		if init_adv is None:
			advertisement = self.init_advertisement()
		else:
			advertisement = init_adv
		self.path_measures = 0 
		# Measure where we start, update model of path probabilities
		self.measure_ingresses(advertisement)
		a_km1 = advertisement
		if self.verbose:
			# self.print_adv(advertisement)
			print("Optimizing over {} peers and {} ugs".format(self.n_popp, self.n_ug))

		self.iter = 0

		t_start = time.time()
		self.reset_metrics()
		self.stop = False

		# Add to metrics / init vars
		self.current_objective = self.measured_objective(advertisement)
		self.current_pseudo_objective = self.modeled_objective(advertisement)
		self.current_effective_objective = self.modeled_objective(threshold_a(advertisement))
		self.last_objective = self.current_pseudo_objective
		self.last_effective_objective = self.current_effective_objective
		self.rolling_delta = 10
		self.rolling_delta_eff = 10

		self.metrics['pseudo_objectives'].append(self.current_pseudo_objective)
		self.metrics['actual_nonconvex_objective'].append(self.current_objective)
		self.metrics['effective_objectives'].append(self.measured_objective(threshold_a(advertisement)))
		self.metrics['advertisements'].append(copy.copy(advertisement))
		while not self.stop:
			self.ts_loop = time.time()
			# calculate gradients
			if self.verbose:
				print("calcing grads")
			grads = self.gradient_fn(advertisement)
			self.recent_grads = grads
			# update advertisement by taking a gradient step with momentum and then applying the proximal gradient for L1
			a_k = advertisement
			w_k = a_k - self.alpha * grads + self.beta * (a_k - a_km1)
			if self.proximal:
				advertisement = self.apply_prox_l1(w_k)
			else:
				advertisement = w_k
			a_km1 = a_k

			# another constraint we may want is 0 <= a_ij <= 1
			# the solution is just clipping to be in the set
			# clipping can mess with gradient descent
			advertisement = self.impose_advertisement_constraint(advertisement)
 
			self.metrics['advertisements'].append(copy.copy(advertisement))
			self.metrics['grads'].append(advertisement - a_k)

			# Take a gradient step and update measured paths + probabilities
			if not np.array_equal(threshold_a(advertisement), threshold_a(a_km1)):
				self.measure_ingresses(advertisement)

			# Calculate, advertise & measure information about the prefix that would 
			# give us the most new information
			if self.verbose:
				print("finding max info {} ".format(time.time() - self.ts_loop))
			maximally_informative_advertisement = self.solve_max_information(advertisement)
			if maximally_informative_advertisement is not None:
				if self.verbose:
					print("measuring ingresses {}".format(time.time() - self.ts_loop))
				self.measure_ingresses(maximally_informative_advertisement)
			else:
				if self.verbose:
					print("No maximally informative advertisement to measure.")
			# Check stopping conditions
			self.stop_tracker(advertisement)
			self.iter += 1

			# Add to metrics
			if self.verbose:
				self.summarize_user_latencies(threshold_a(advertisement))

			self.t_per_iter = (time.time() - t_start) / self.iter

			if self.iter % PRINT_FREQUENCY == 0 and self.verbose:
				print("Optimizing, iter: {}, t_per_iter : {}, GTO: {}, RD: {}, RDE: {}".format(self.iter, self.t_per_iter, 
					self.metrics['actual_nonconvex_objective'][-1],self.rolling_delta, self.rolling_delta_eff))

				if DPSIZE in ['decent', 'large', 'actual']:
					self.make_plots()

			print(self.iter)
			# if self.iter == 20:
			# 	break

		if self.verbose:
			print("Stopped train loop on {}, t per iter: {}, {} path measures, O:{}, RD: {}, RDE: {}".format(
				self.iter, self.t_per_iter, self.path_measures, 
				self.current_pseudo_objective, self.rolling_delta, self.rolling_delta_eff))
		self.metrics['t_per_iter'] = self.t_per_iter

def main():
	try:

		deployment = get_random_deployment(DPSIZE)


		# ## Comparing different solutions
		# lambduh = .001
		# sas = Sparse_Advertisement_Eval(deployment, verbose=True,
		# 	lambduh=lambduh,with_capacity=False)
		# wm = Worker_Manager(sas.get_init_kwa(), deployment)
		# wm.start_workers()
		# sas.set_worker_manager(wm)
		# sas.compare_different_solutions(deployment_size=DPSIZE,n_run=5)
		# exit(0)


		# ## Simple test
		# lambduh = .0001
		# gamma = 1.0
		# sas = Sparse_Advertisement_Solver(deployment, 
		# 	lambduh=lambduh,verbose=True,with_capacity=False,n_prefixes=20,
		# 	using_resilience_benefit=True, gamma=gamma)#n_prefixes=len(deployment['popps'])-1)
		# wm = Worker_Manager(sas.get_init_kwa(), deployment)
		# wm.start_workers()
		# sas.set_worker_manager(wm)
		# sas.solve()
		# sas.make_plots()

		### With VULTR measurements test
		lambduh = .0001
		gamma = 1.0
		sas = Sparse_Advertisement_Solver(deployment, 
			lambduh=lambduh,verbose=True,with_capacity=False,n_prefixes=4,
			using_resilience_benefit=True, gamma=gamma)#n_prefixes=len(deployment['popps'])-1)
		wm = Worker_Manager(sas.get_init_kwa(), deployment)
		wm.start_workers()
		sas.set_worker_manager(wm)
		sas.solve()
		sas.make_plots()
		print(threshold_a(sas.get_last_advertisement()))


		# ## Capacity Test ( I think this works, but not positive )
		# lambduh = .1
		# sas = Sparse_Advertisement_Solver(get_random_deployment('small'), 
		# 	lambduh=lambduh,verbose=True,with_capacity=True)
		# sas.solve()
		# sas.make_plots()
	except:
		import traceback
		traceback.print_exc()
	finally:
		wm.stop_workers()


if __name__ == "__main__":
	main()
