"""Generic_Objective: runtime dispatch from objective name to LP function.

Used by Sparse_Advertisement_Wrapper to evaluate the per-iter objective
inside the SGD loop. The objective name (e.g. 'avg_latency', 'site_failure')
is set via the deployment dict or kwargs, and Generic_Objective looks it
up in `solve_lp_assignment.generic_lp_functions` to find the matching
LP function.

This is a thin shim. The heavy lifting is in solve_lp_assignment.py.
"""
from core.solve_lp_assignment import solve_generic_lp_with_failure_catch
from helpers.helpers import *
from helpers.constants import *
import numpy as np
import copy
import os
import os as _os        # moved machinery uses the sparse_adv alias
import pickle
import time

class Generic_Objective:
	"""Base objective. Owns the TRAINING POLICY for an objective (Tom
	2026-08-25): which gradient components exist, the gamma trade-off and
	its annealing, and whether resilience participates. The solver keeps
	the machinery (worker flushes, prox, NaN guards) and consults the
	objective object -- fully implementing a new objective means
	subclassing this and overriding the few methods below, then
	registering in OBJECTIVE_CLASSES."""

	# True iff this objective's training computes resilience-benefit
	# gradients and values (the popp-failure mega-batches, ~210s/iter at
	# actual-32). Base: no.
	uses_resilience_gradient = False

	def __init__(self, sas, obj, **kwargs):
		self.sas = sas # SAS object
		self.obj = obj # string identifying the objective. e.g., avg_latency
		# configured auxiliary-term weight (0 for base objectives) and
		# whether the auxiliary term participates at all
		self.gamma_target = kwargs.get('gamma', 0) or 0
		self.enabled = bool(kwargs.get('using_resilience_benefit', False))
		# Extra kwargs (e.g., site_cost_alpha, bulk_cap_limit) forwarded to every
		# LP call. Set by the experiment driver from the ObjectiveSpec.lp_kwargs.
		# Empty {} preserves prior behavior since the LP functions all use
		# kwargs.get('foo', <existing_default>).
		self.lp_kwargs = kwargs.get('lp_kwargs', {}) or {}

	def get_latency_benefit_adv(self, a):
		routed_through_ingress, _ = self.sas.calculate_ground_truth_ingress(a)
		# Pass `adv` through to the LP function so multi-LP objectives (e.g.,
		# static_failure) that need to simulate adv-row failures can recover
		# `a` without reconstructing it from routed_through_ingress + actives.
		# Existing single-LP objectives just ignore the kwarg.
		ret = solve_generic_lp_with_failure_catch(
			self.sas, routed_through_ingress, self.obj,
			adv=a, **self.lp_kwargs)
		return ret

	def get_ground_truth_objective_value(self, a, **kwargs):
		"""
			Ground-truth value of this objective: whatever pops out of the
			generic LP's objective function. (Renamed from
			get_ground_truth_latency_benefit, 2026-08-25 -- 'latency
			benefit' was historical naming from the avg_latency-only era.)
		"""
		
		ret = self.get_latency_benefit_adv(a)
		# defensive: unsolved/sentinel rets may carry no path map
		ug_ingress_decisions = ret.get('paths_by_ug') or {}
		if kwargs.get('save_ug_ingress_decisions'):
			self.sas.popp_to_users = {}
			for ugi in ug_ingress_decisions:
				for poppi,v in ug_ingress_decisions[ugi]:
					try:
						self.sas.popp_to_users[poppi].append(self.sas.ugs[ugi])
					except KeyError:
						self.sas.popp_to_users[poppi] = [self.sas.ugs[ugi]]

		return ret['objective']

	# back-compat alias
	get_ground_truth_latency_benefit = get_ground_truth_objective_value

	def get_ground_truth_resilience_benefit(self, a, **kwargs):
		"""Resilience is NOT part of a base objective (Tom 2026-08-25) --
		it is latency+gamma*resilience policy and lives on
		LatencyPlusResilienceObjective. Kept as a zero-returning stub
		because generic call sites (modeled/measured objective
		accounting) invoke it on every objective."""
		return 0

	# ------------------------- training policy (override per objective) --
	def get_gamma(self):
		"""LB/RB trade-off weight during training. Base objectives train
		purely on their own gradient."""
		return 0.0

	def component_gradients(self, a):
		"""(L_grad, res_grad) for the SGD step. The heavy lifting stays in
		the solver's flush machinery; this method decides WHICH components
		this objective pays for."""
		import time as _t
		sas = self.sas
		ts = _t.time()
		L_grad = sas.gradients_latency_benefit(a)
		if sas.verbose:
			print(_t.strftime("[%H:%M:%SZ] ", _t.gmtime())
				  + "Calcing latency benefit grad took {}s".format(
					  int(_t.time() - ts)))
			print(_t.strftime("[%H:%M:%SZ] ", _t.gmtime())
				  + "RB grad skipped (objective {} does not train with "
				  "resilience)".format(self.obj))
		return L_grad, np.zeros(np.shape(L_grad))

	# ---------------------- metrics & convergence-plot policy ----------
	def record_gradient_metrics(self, metrics, L_grad, res_grad, gamma,
								cost_grad):
		"""Per-iteration gradient components for the convergence plots.
		Base objectives track their own gradient plus the cost term."""
		metrics['l_benefit_grads'].append(L_grad)
		metrics['res_benefit_grads'].append(0 * L_grad)
		metrics['cost_grads'].append(cost_grad)

	def plot_convergence_extras(self, sas, ax, metrics, start_iter=0):
		"""Objective-specific convergence panels. `ax` is the solver's
		panel grid; rows 3-5 are reserved for the objective. Base
		objectives get their value tracked by the generic believed/GT
		objective panels, so nothing extra is required here."""
		return

	def record_iteration_metrics(self, sas, metrics, advertisement,
								 skip_measuring):
		"""Per-iteration stop-tracker metrics that are OBJECTIVE POLICY.
		Base objectives have no auxiliary term: neutral zeros keep the
		metrics schema stable for downstream aggregation/plots without
		invoking any resilience machinery."""
		metrics['effective_gammas'].append(0)
		metrics['gt_resilience_benefit'].append(0)
		metrics['resilience_benefit'].append(0)
		metrics['frac_resilience_benefit_calls'].append(0)
		sas.current_resilience_benefit = 0

	def modeled_objective_value(self, a, **kwargs):
		sas = self.sas
		"""Approx actual objective with our belief."""
		if sas.verbose:
			print("Calculating modeled objective")
		norm_penalty = sas.advertisement_cost(a)
		latency_benefit = None
		# popped here so it never leaks into worker job kwa on the
		# gamma==0 / no-resilience paths
		_startup_rb = kwargs.pop('startup_rb', False)
		if self.enabled:
			# combined flush: the base-adv LB rides as job 0 of the
			# resilience fan-out (it was already queued and discarded there)
			# instead of a separate 1-job flush that serializes on a single
			# worker while the rest of the pool idles
			resilience_benefit, _base = self.resilience_benefit(
				a, with_lb=True, startup_rb=_startup_rb, **kwargs)
			if _base is not None:
				latency_benefit, u = _base
		else:
			resilience_benefit = 0
		if latency_benefit is None:
			kwargs['retnow'] = True
			latency_benefit, u = sas.latency_benefit_fn(a, **kwargs)

		if sas.verbose:
			benefits,probs = u
			ex = np.average(benefits,weights=probs+1e-8)
			exsq = np.average(np.power(benefits,2),weights=probs+1e-8)
			var = exsq - np.power(ex,2)
			std = np.sqrt(var)
			print("Believed: NP: {}, LB: {} ({} std dev), RB: {}".format(round(norm_penalty,3),
				round(latency_benefit,3), round(std,3), round(resilience_benefit,3)))

		# gamma = self.get_gamma()
		gamma = self.gamma_target
		if gamma <= 1:
			benefit = latency_benefit + gamma * resilience_benefit
		else:
			benefit = 1 / gamma * latency_benefit + resilience_benefit

		return sas.lambduh * norm_penalty - (benefit)


	def resilience_benefit(self, a, with_lb=False, **kwargs):
		"""Base objectives have no auxiliary resilience term."""
		return (0, None) if with_lb else 0


class LatencyPlusResilienceObjective(Generic_Objective):
	"""latency + gamma * resilience -- the one objective whose training
	pays for resilience gradients. Also owns the gamma ANNEALING schedule
	(moved from Sparse_Advertisement_Solver.get_gamma, 2026-08-25)."""

	uses_resilience_gradient = True

	def __init__(self, sas, obj, **kwargs):
		super().__init__(sas, obj, **kwargs)
		# mirror the original Wrapper default: the aux term participates
		# only when the caller passes using_resilience_benefit=True
		# (workers/baseline helpers construct without it, gamma 0)
		if self.enabled:
			assert self.gamma_target > 0

	def get_gamma(self):
		# Increase gamma toward its configured value as confidence about
		# adjacent strategies grows.
		sas = self.sas
		if sas.simulated:
			uncertainty_factor = np.maximum(
				1, np.abs(sas.uncertainty_factor))
			divider = uncertainty_factor * (
				1 / (1 + 3 / np.sqrt((sas.iter + 1))))
		else:
			# no uncertainty factor since we don't do max info (for now)
			divider = (1 + 5 / np.sqrt((sas.iter + 1)))
		return sas.gamma / divider

	def get_ground_truth_resilience_benefit(self, a, **kwargs):
		# moved from the base class 2026-08-25: resilience is THIS
		# objective's policy, not generic behavior.
		"""
			Computes an abstraction of 'resilience benefit'. Which just means 
			the average of whatever pops out of the generic LP's objective function 
			when you fail each popp.
			(quite slow)
		"""
		benefit = 0
		return 0 #### TMPPPPPPPP
		if self.sas.gamma == 0:
			return benefit

		tmp = np.ones(a.shape)
		a = threshold_a(a)

		dep = self.sas.output_deployment()
		dep['generic_objective'] = self.obj
		args = []
		for popp in self.sas.popps:
			tmp[self.sas.popp_to_ind[popp],:] = 0
			# ret = self.get_latency_benefit_adv(copy.copy(a * tmp))
			# benefit += ret['objective']
			args.append((copy.copy(a * tmp), dep, False))
			tmp[self.sas.popp_to_ind[popp],:] = 1

		all_rets = self.sas.solve_lp_with_failure_catch_mp(args)

		for ret in all_rets:
			benefit += ret['objective']
		return benefit


	def component_gradients(self, a):
		import time as _t
		sas = self.sas
		ts = _t.time()
		L_grad = sas.gradients_latency_benefit(a)
		if sas.verbose:
			print(_t.strftime("[%H:%M:%SZ] ", _t.gmtime())
				  + "Calcing latency benefit grad took {}s".format(
					  int(_t.time() - ts)))
		ts = _t.time()
		if not sas.simulated or self.get_gamma() == 0:
			res_grad = np.zeros(np.shape(L_grad))
		else:
			res_grad = self.gradients_resilience_benefit(a)
			if sas.verbose:
				print(_t.strftime("[%H:%M:%SZ] ", _t.gmtime())
					  + "Calcing resilience benefit grad took {}s".format(
						  int(_t.time() - ts)))
		return L_grad, res_grad

	def record_gradient_metrics(self, metrics, L_grad, res_grad, gamma,
								cost_grad):
		# gamma <= 1 blends toward LB; > 1 toward RB (stability convention
		# moved verbatim from the pre-2026-08-25 solver code)
		if gamma <= 1:
			metrics['l_benefit_grads'].append(L_grad)
			metrics['res_benefit_grads'].append(gamma * res_grad)
		else:
			metrics['l_benefit_grads'].append(1 / gamma * L_grad)
			metrics['res_benefit_grads'].append(res_grad)
		metrics['cost_grads'].append(cost_grad)

	def plot_convergence_extras(self, sas, ax, metrics, start_iter=0):
		"""LB/RB decomposition panels: resilience benefit, latency
		benefit, their ground truths, and reference lines -- meaningful
		only for latency + gamma*resilience."""
		rng = list(range(start_iter, len(metrics['resilience_benefit'])))
		ax[3, 1].plot(rng, metrics['resilience_benefit'][start_iter:])
		ax[3, 1].set_ylabel("Res Ben")
		rng = list(range(start_iter, len(metrics['latency_benefit'])))
		ax[4, 1].plot(rng, metrics['latency_benefit'][start_iter:])
		ax[4, 1].set_ylabel("Lat Ben")
		ax[5, 0].plot(metrics['gt_latency_benefit'])
		ax[5, 0].set_ylabel("GT Lat Ben")
		ax[5, 1].plot(metrics['gt_resilience_benefit'])
		ax[5, 1].set_ylabel("GT Res Ben")
		for axis, key, label in (
				(ax[5, 0], 'latency', 'One per Peering'),
				(ax[5, 1], 'resilience', 'One per Peering')):
			try:
				y = sas.optimal_expensive_solution[key]
				axis.hlines(y=y, xmin=0, xmax=sas.iter, linewidth=2,
							color='k')
				axis.text(0, y, label)
			except (AttributeError, KeyError):
				pass
		try:
			ax[5, 0].hlines(y=sas.painter_solution['latency_benefit'],
							xmin=0, xmax=sas.iter, linewidth=2, color='r')
		except (AttributeError, KeyError, TypeError):
			pass
		try:
			if not hasattr(sas, 'painter_gt_resilience_benefit'):
				sas.painter_gt_resilience_benefit = \
					self.get_ground_truth_resilience_benefit(
						sas.painter_solution['advertisement'])
			ax[5, 1].hlines(y=sas.painter_gt_resilience_benefit, xmin=0,
							xmax=sas.iter, linewidth=2, color='r')
		except (AttributeError, KeyError, TypeError):
			pass
		# annealed-gamma trajectory: only meaningful for this objective
		ax[6, 0].plot(metrics['effective_gammas'])
		ax[6, 0].set_ylabel("Effective Gamma")

	def record_iteration_metrics(self, sas, metrics, advertisement,
								 skip_measuring):
		"""Gamma trajectory + resilience-benefit tracking -- the auxiliary
		term only this objective trains with (moved verbatim from
		stop_tracker, 2026-08-25)."""
		import time as _t
		metrics['effective_gammas'].append(self.get_gamma())
		perf_t = _t.time()
		if not skip_measuring or len(metrics['gt_resilience_benefit']) == 0:
			metrics['gt_resilience_benefit'].append(
				self.get_ground_truth_resilience_benefit(
					advertisement, store_metrics=True))
			print(f"[Timing] get_ground_truth_resilience_benefit: "
				  f"{_t.time() - perf_t:.5f}s")
		else:
			metrics['gt_resilience_benefit'].append(
				metrics['gt_resilience_benefit'][-1])
		sas.current_resilience_benefit = metrics['gt_resilience_benefit'][-1]
		perf_t = _t.time()
		rb = sas.resilience_benefit(advertisement)
		metrics['resilience_benefit'].append(rb)
		print(f"[Timing] resilience_benefit: {_t.time() - perf_t:.5f}s")
		metrics['frac_resilience_benefit_calls'].append(
			len(sas.n_resilience_benefit_popp_calls)
			/ (sas.n_popps * sas.n_popps * sas.n_prefixes))

	def resilience_benefit(self, a, with_lb=False, **kwargs):
		sas = self.sas
		""" sum over peers of E(delta benefit when that peer is knocked out).
		with_lb=True also returns the base-adv (latency_benefit, u) tuple
		(job 0 of the flush, previously discarded) so the caller can share
		this flush instead of paying a separate 1-job flush; returns
		(benefit, None) when a gate below skips the flush entirely."""
		# want to maximize resilience beneift, so want to maximize new benefits
		# when peers are knocked out
		if (not sas.simulated
				or not self.uses_resilience_gradient
				or sas.gamma == 0):
			return (0, None) if with_lb else 0
		# Under headroom mode (SCULPTOR_CAPACITY_HEADROOM>0), resilience is
		# absorbed into the LP via reserved capacity, so we don't use the
		# RB-grad for optimization. The N_popps+1 LP flush here is purely to
		# populate the "Believed: RB" print and pseudo_objective stopping
		# signal — both can run RB-free without harming convergence. Skipping
		# saves ~18s/iter at actual-10.
		#
		# Gated on _in_training so a future caller that wants the real RB
		# value for reporting (e.g. paper-figure stats) gets it back. Headroom
		# is a training-time approximation only.
		if sas._in_training and float(os.environ.get('SCULPTOR_CAPACITY_HEADROOM', '0')) > 0:
			return (0, None) if with_lb else 0
		# SCULPTOR_STARTUP_RB (Tom 2026-08-22): the FIRST flush of this
		# fan-out runs against cold workers (empty persistent-LP columns,
		# empty caches) and measured 24-28 MINUTES at dpsize=20/25 on the
		# 2026-08-21 ladder -- ~9x a steady-state iteration. Only the
		# startup call (init_optimization_vars passes startup_rb=True) is
		# affected; per-iteration RB is untouched.
		#   full          stock behavior (default)
		#   sample:<f>    deterministic every-k-th popp subset (k=1/f),
		#                 benefit scaled by n_popps/len(subset). Chosen
		#                 deterministic, not random, so A/B arms share the
		#                 training RNG stream.
		#   skip          no RB at startup (base-adv LB job only); the
		#                 first in-training flush then pays the warm-cache
		#                 price, which is the 'defer' arm of the A/B.
		_startup = bool(kwargs.pop('startup_rb', False))
		_mode = os.environ.get('SCULPTOR_STARTUP_RB', 'full') if _startup else 'full'
		popps_to_fail = list(sas.popps)
		_scale = 1.0
		if _mode.startswith('sample'):
			try:
				_frac = float(_mode.split(':', 1)[1])
			except (IndexError, ValueError):
				_frac = 0.1
			_k = max(1, int(round(1.0 / max(_frac, 1e-6))))
			popps_to_fail = popps_to_fail[::_k]
			_scale = float(len(sas.popps)) / max(len(popps_to_fail), 1)
		elif _mode == 'skip':
			popps_to_fail = []
		tmp = np.ones(a.shape)
		cpkwargs = copy.deepcopy(kwargs)
		cpkwargs['retnow'] = False
		_t_rb = time.time()
		sas.latency_benefit_fn(a, **cpkwargs)
		for popp in popps_to_fail:
			# we don't know for sure where users are going
			# so we have to compute over all users
			tmp[sas.popp_to_ind[popp],:] = 0
			cpkwargs['failing_popp'] = popp
			sas.latency_benefit_fn(a * tmp, **cpkwargs)
			tmp[sas.popp_to_ind[popp],:] = 1
		rets = sas.flush_latency_benefit_queue()

		benefit = 0
		for b,_ in rets[1:]:
			benefit += b
		benefit *= _scale
		if _startup:
			print('[startup-rb] mode={} popps={}/{} scale={:.2f} '
				  'benefit={:.3f} took={:.1f}s'.format(
					  _mode, len(popps_to_fail), len(sas.popps), _scale,
					  benefit, time.time() - _t_rb), flush=True)

		if with_lb:
			return benefit, rets[0]
		return benefit


	def gradients_resilience_benefit_popp(self, advertisement):
		sas = self.sas

		## want to test popp,pref 
		## turn it off, fail a popp. measure LB (a)
		## turn it on, fail same popp. measure LB (b)
		## should turn popp,pref on if (b) > (a)



		### Positive resilience benefit gradient means turning a popp
		## on will increase resilience
		### increasing resilience means maximizing benefit under popp failures


		grad_rb = np.zeros(advertisement.shape)
		calls = []


		### We monte-carlo sample the full space
		total_n_grad_calc = sas.gradient_support_settings['popp_rb_support_size']
		
		pct_explore = 80 # pct of gradient calculation budget dedicated to exploring
		N_EXPLORE = int(total_n_grad_calc * pct_explore/100)
		# number of gradient calcs that re-calc previously high gradients
		N_REMEASURE = total_n_grad_calc - N_EXPLORE
		gamma = self.get_gamma()
		try:
			best_from_last_time = sorted(sas.last_rb_calls_results_popp.items(), key = lambda el :
				-1 * np.abs(el[1]))
			# same objective-scale-relative significance cutoff as the LB
			# remeasure filter (SCULPTOR_SIG_CUTOFF, Tom 2026-08-16)
			if _os.environ.get('SCULPTOR_SIG_CUTOFF', 'p5') == 'p5':
				_prev_mags = np.abs(np.array([v for _, v in best_from_last_time]))
				_sig_cut = (max(1e-12, float(np.percentile(_prev_mags, 5)))
							if len(_prev_mags) else 1e-12)
			else:
				_sig_cut = .01
			n_significant = 0
			for (popp,rand_kill_popp,rand_outer_prefix),val in best_from_last_time:
				if (popp,rand_kill_popp,rand_outer_prefix) in calls:
					continue
				if gamma * np.abs(val) < sas.lambduh or np.abs(val) < _sig_cut:
					# if it's not important enough to warrant the cost, don't bother
					continue
				if np.abs(ADVERTISEMENT_THRESHOLD - advertisement[sas.popp_to_ind[popp], rand_outer_prefix]) > \
					ADVERTISEMENT_THRESHOLD * 7 / 10: 
					# advertisment is almost completely on or completely off
					continue

				tmp_a = copy.copy(advertisement)

				this_popp_random_kill = sas.popp_to_ind[rand_kill_popp]
				tmp_a[this_popp_random_kill,:] = 0.0 # kill this random popp
				this_killed_popp_ugs = sas._ensure_popp_to_users().get(this_popp_random_kill, [])
				if len(this_killed_popp_ugs) == 0:
					continue

				poppi = sas.popp_to_ind[popp]
				tmp_a[poppi,rand_outer_prefix] = 1.0 # Turn this popp on
				sas.latency_benefit(tmp_a, ugs=this_killed_popp_ugs)
				tmp_a[poppi,rand_outer_prefix] = 0.0 # turn this popp off
				sas.latency_benefit(tmp_a, ugs=this_killed_popp_ugs)

				calls.append((popp, rand_kill_popp, rand_outer_prefix, this_killed_popp_ugs))

				n_significant += 1
				if n_significant >= N_REMEASURE:
					break
			print("Last RB call, {} were significant".format(n_significant))

		except AttributeError: # there are no last calls on the first iteration
			pass

		N_REMEASURE = len(calls)
		N_EXPLORE = total_n_grad_calc - N_REMEASURE


		all_popps = np.arange(sas.n_popp)


		try:
			raise AttributeError
			## Sample popps that need more help, more
			rand_popp_choices = np.random.choice(all_popps, p=sas.popp_rb_sample_probabilities, 
				size=N_EXPLORE)
		except AttributeError:
			rand_popp_choices = np.random.randint(low=0,high=sas.n_popps,
				size=N_EXPLORE)

		# associated prefix distribution should be biased towards prefixes that are far from 1 and 0
		possible_prefix_choices = np.arange(sas.n_prefixes)
		prob_each_pref = np.ones(sas.n_prefixes) / sas.n_prefixes

		for rand_kill_poppi in rand_popp_choices:
			rand_kill_popp = sas.popps[rand_kill_poppi]
			
			poppi_helper = np.random.choice(all_popps,
				 p=sas.popp_backup_sample_probs[rand_kill_poppi,:]) 
			popp_helper = sas.popps[poppi_helper] # popp ij testing gradient is poppi,rand_outer_prefix (should we turn this on/off to help out?)

			rand_outer_prefix = int(np.random.choice(possible_prefix_choices, p=prob_each_pref))

			if (popp_helper, rand_kill_popp, rand_outer_prefix) in calls: continue
			
			tmp_a = copy.copy(advertisement)
			tmp_a[rand_kill_poppi,:] = 0.0 # kill this random popp
			this_killed_popp_ugs = sas._ensure_popp_to_users().get(rand_kill_poppi, [])
			if len(this_killed_popp_ugs) == 0:
				continue

			tmp_a[poppi_helper,rand_outer_prefix] = 1.0 # Turn this popp on
			sas.latency_benefit(tmp_a, ugs=this_killed_popp_ugs)
			tmp_a[poppi_helper,rand_outer_prefix] = 0.0 # turn this popp off
			sas.latency_benefit(tmp_a, ugs=this_killed_popp_ugs)
			calls.append((popp_helper, rand_kill_popp, rand_outer_prefix, this_killed_popp_ugs))

		all_lb_rets = sas.flush_latency_benefit_queue()
		grad_rb = sas._assemble_rb_popp_gradients(calls, all_lb_rets, advertisement, grad_rb)

		### Track which calls are being made
		for poppi,poppj,pref,_ in calls:
			try:
				sas.n_resilience_benefit_popp_calls[poppi,poppj,pref] += 1
			except KeyError:
				sas.n_resilience_benefit_popp_calls[poppi,poppj,pref] = 1

		if not sas.simulated:
			max_val = np.max(np.abs(grad_rb.flatten()))
			if max_val > 1:
				grad_rb = grad_rb / max_val

		grad_rb = grad_rb.clip(-GRAD_CLIP_VAL,GRAD_CLIP_VAL)

		return grad_rb


	def gradients_resilience_benefit_pop(self, advertisement):
		sas = self.sas

		## want to test popp,pref 
		## turn it off, fail a PoP. measure LB (a)
		## turn it on, fail same PoP. measure LB (b)
		## should turn popp,pref on if (b) > (a)



		### Positive resilience benefit gradient means turning a popp
		## on will increase resilience
		### increasing resilience means maximizing benefit under PoP failures


		grad_rb = np.zeros(advertisement.shape)
		#### Previously disabled ("unused, too noisy"). Re-enabled May-30 to test
		#### whether the pop-failure gradient term closes the site-failure
		#### painter-beats-sparse gap. Gated externally via SCULPTOR_ALPHA_POP
		#### (called only when alpha > 0 in gradients_resilience_benefit).
		a_effective = threshold_a(advertisement).astype(bool)
		calls = []


		total_n_grad_calc = sas.gradient_support_settings['pop_rb_support_size']
		
		pct_explore = 80 # pct of gradient calculation budget dedicated to exploring
		N_EXPLORE = int(total_n_grad_calc * pct_explore/100)
		# number of gradient calcs that re-calc previously high gradients
		N_REMEASURE = total_n_grad_calc - N_EXPLORE
		gamma = self.get_gamma()
		try:
			best_from_last_time = sorted(sas.last_rb_calls_results_pop.items(), key = lambda el : 
				-1 * np.abs(el[1]))
			n_significant = 0
			for (popp,rand_kill_pop,rand_outer_prefix),val in best_from_last_time:
				if (popp,rand_kill_pop,rand_outer_prefix) in calls: 
					continue
				if gamma * np.abs(val) < sas.lambduh:
					# if it's not important enough to warrant the cost, don't bother
					continue
				if np.abs(ADVERTISEMENT_THRESHOLD - advertisement[sas.popp_to_ind[popp],rand_outer_prefix]) > \
					ADVERTISEMENT_THRESHOLD * 7 / 10: 
					# advertisment is almost completely on or completely off
					continue

				tmp_a = copy.copy(a_effective)
				tmp_a[sas.pop_to_popp_inds[rand_kill_pop],:] = False # kill this random pop

				poppi = sas.popp_to_ind[popp]
				tmp_a[poppi,rand_outer_prefix] = True # Turn this popp on
				sas.latency_benefit(tmp_a)
				tmp_a[poppi,rand_outer_prefix] = False # turn this popp off
				sas.latency_benefit(tmp_a)

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
		rand_popp_choices = np.random.randint(low=0,high=sas.n_popps,
			size=N_EXPLORE) 
		### associated prefixes for the rand_popp_choices
		# associated prefix distribution should be biased towards prefixes that are far from 1 and 0
		random_prefix_choices = np.zeros(N_EXPLORE,dtype=np.int32)
		possible_choices = np.arange(sas.n_prefixes)
		for i in range(N_EXPLORE):
			prob_each_pref = ADVERTISEMENT_THRESHOLD - np.abs(advertisement[rand_popp_choices[i],:] - ADVERTISEMENT_THRESHOLD) + .1
			prob_each_pref = prob_each_pref / np.sum(prob_each_pref)
			prob_each_pref = np.ones(sas.n_prefixes) / sas.n_prefixes
			random_prefix_choices[i] = int(np.random.choice(possible_choices, p=prob_each_pref))


		for poppi, rand_outer_prefix in zip(rand_popp_choices,random_prefix_choices):
			popp = sas.popps[poppi] # popp ij testing gradient is poppi,rand_outer_prefix

			## random kill PoP
			this_popp_random_kill = np.random.choice(np.arange(sas.n_popp),
				 p=sas.popp_backup_sample_probs[poppi,:]) 
			rand_kill_pop = sas.popps[this_popp_random_kill][0]
			if (popp, rand_kill_pop, rand_outer_prefix) in calls: continue
			
			tmp_a = copy.copy(a_effective)
			tmp_a[sas.pop_to_popp_inds[rand_kill_pop],:] = False # kill this random pop

			tmp_a[poppi,rand_outer_prefix] = True # Turn this popp on
			sas.latency_benefit(tmp_a)
			tmp_a[poppi,rand_outer_prefix] = False # turn this popp off
			sas.latency_benefit(tmp_a)
			calls.append((popp, rand_kill_pop, rand_outer_prefix))

		all_lb_rets = sas.flush_latency_benefit_queue()
		grad_rb = sas._assemble_rb_pop_gradients(calls, all_lb_rets, advertisement, grad_rb)

		grad_rb = grad_rb.clip(-GRAD_CLIP_VAL,GRAD_CLIP_VAL)

		return grad_rb


	def gradients_resilience_benefit(self, advertisement):
		sas = self.sas
		# Under SCULPTOR_CAPACITY_HEADROOM>0 the inner LP already reserves
		# capacity for failures, so the SGD-RB gradient is unnecessary.
		# Symmetric with resilience_benefit() (the value), which also
		# short-circuits under headroom. Gated on _in_training so this is
		# strictly a training-time approximation -- non-training callers
		# get the real gradient.
		if sas._in_training and float(os.environ.get('SCULPTOR_CAPACITY_HEADROOM', '0')) > 0:
			return np.zeros(advertisement.shape)

		grad_link_failure = self.gradients_resilience_benefit_popp(advertisement)
		# SCULPTOR_ALPHA_POP weights the pop-failure resilience gradient.
		# Default 0 reproduces the prior "popp only" behaviour (the pop
		# gradient was historically disabled with the note "hurts convergence").
		# Test setting alpha=0.05..0.5 to see if it closes the painter-beats-
		# sparse gap on site-failure latency that shows up at dp15 / dp25.
		#
		# SCULPTOR_ALPHA_POP_ANNEAL_END_ITER (int, default 0):
		#   If >0, linearly ramp alpha from 0 -> SCULPTOR_ALPHA_POP across the
		#   first N iters of sparse training. Lets the latency-benefit term
		#   converge first before the noisier pop-failure gradient kicks in.
		alpha_max = float(os.environ.get('SCULPTOR_ALPHA_POP', '0'))
		anneal_end = int(os.environ.get('SCULPTOR_ALPHA_POP_ANNEAL_END_ITER', '0'))
		if anneal_end > 0:
			it = max(0, int(getattr(self, 'iter', 0)))
			alpha = alpha_max * min(1.0, it / float(anneal_end))
		else:
			alpha = alpha_max
		if alpha > 0:
			grad_pop_failure = self.gradients_resilience_benefit_pop(advertisement)
		else:
			grad_pop_failure = 0
		return grad_link_failure + alpha * grad_pop_failure

# objective name -> objective class; anything unregistered gets the base
# (pure own-gradient training). A new objective with special training
# behavior registers its subclass here.
OBJECTIVE_CLASSES = {
	'avg_latency': LatencyPlusResilienceObjective,
}


def make_objective(sas, obj, **kwargs):
	return OBJECTIVE_CLASSES.get(obj, Generic_Objective)(sas, obj, **kwargs)
