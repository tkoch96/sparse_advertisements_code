from solve_lp_assignment import solve_generic_lp_with_failure_catch
from helpers import *
from constants import *
import numpy as np

class Generic_Objective:
	def __init__(self, sas, obj, **kwargs):
		self.sas = sas # SAS object
		self.obj = obj # string identifying the objective. e.g., avg_latency
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

	def get_ground_truth_latency_benefit(self, a, **kwargs):
		"""
			Computes an abstraction of 'latency benefit'. Which just means 
			whatever pops out of the generic LP's objective function.
		"""
		
		ret = self.get_latency_benefit_adv(a)
		ug_ingress_decisions = ret['paths_by_ug']
		if kwargs.get('save_ug_ingress_decisions'):
			self.sas.popp_to_users = {}
			for ugi in ug_ingress_decisions:
				for poppi,v in ug_ingress_decisions[ugi]:
					try:
						self.sas.popp_to_users[poppi].append(self.sas.ugs[ugi])
					except KeyError:
						self.sas.popp_to_users[poppi] = [self.sas.ugs[ugi]]

		return ret['objective']

	def get_ground_truth_resilience_benefit(self, a, **kwargs):
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



