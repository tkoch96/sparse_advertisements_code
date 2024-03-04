### path preference changes
### latency changes
### capacity changes

from wrapper_eval import *
from constants import *
from helpers import *
from solve_lp_assignment import *

import pickle, numpy as np, matplotlib.pyplot as plt, copy, itertools, time
from sparse_advertisements_v3 import *

def change_model_preferences(sas,adv,deployment):
	"""Goal is to assess resilience to changes in which ingresses users prefer."""
	ret = []
	pref_cpy = copy.deepcopy(deployment['ingress_priorities'])
	for i in range(5):
		## randomly change path preferences for the users
		for ug in sas.ugs:
			# randomly change some path preferences
			popps = list(sas.ug_perfs[ug])
			flip_popps = [popps[pi] for pi in np.random.choice(list(range(len(popps))), size=np.random.randint(1,7))]
			for popp in flip_popps:
				rand_choice_flip = popps[np.random.choice(list(range(len(popps))))]
				cp_tmp = copy.copy(deployment['ingress_priorities'][ug][rand_choice_flip])
				deployment['ingress_priorities'][ug][rand_choice_flip] = copy.copy(deployment['ingress_priorities'][ug][popp])
				deployment['ingress_priorities'][ug][popp] = cp_tmp
		deployment['whole_deployment_ingress_priorities'] = deployment['ingress_priorities']
		sas.update_deployment(deployment)
		## Look at latency
		ret = assess_failure_resilience(sas, adv, which='popps')
		for rval in ret['latency_delta_optimal']:
			ret.append(rval)

		deployment['ingress_priorities'] = copy.deepcopy(pref_cpy)
		deployment['whole_deployment_ingress_priorities'] = deployment['ingress_priorities']

	sas.update_deployment(deployment)

	return ret

def change_model_capacities(sas,adv,deployment):
	"""Goal is to assess resilience to changes in path capacities. Capacities may be smaller than we think, if there are cross-traffic bottlenecks."""
	ret = []

	CUT_FACTOR = 3

	lc_copy = copy.deepcopy(deployment['link_capacities'])
	for i in range(10):
		## cut random path capacities by a factor of something
		new_lc = copy.deepcopy(lc_copy)
		change_popps = [sas.popps[pi] for pi in np.random.choice(list(range(len(sas.popps))), size=5)]
		for popp in change_popps:
			new_lc[popp] = new_lc[popp] / CUT_FACTOR
		deployment['link_capacities'] = new_lc
		sas.update_deployment(deployment)

		## Look at latency
		ret = assess_failure_resilience(sas, adv, which='popps')
		for rval in ret['latency_delta_optimal']:
			ret.append(rval)

	deployment['link_capacities'] = lc_copy
	sas.update_deployment(deployment)

	return ret

def modeling_assumption_comparisons():
	wm = None
	sas = None

	if not os.path.exists(performance_metrics_fn):
		raise FileNotFoundError("You need to populate model file first : {}".format(performance_metrics_fn))

	modeling_metrics_fn = os.path.join(CACHE_DIR, 'modeling_assumption_comparisons_{}.pkl'.format(DPSIZE))
	default_metrics_model_assumptions = {
		'path_preference_changes_latency': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'path_capacity_changes_latency': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'path_latency_changes_latency': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
	}
	metrics = copy.deepcopy(default_metrics_model_assumptions)
	if os.path.exists(modeling_metrics_fn):
		metrics = pickle.load(open(modeling_metrics_fn,'rb'))
	d = pickle.load(open(performance_metrics_fn,'rb'))
	for k in d:
		metrics[k] = d[k]
	for k in default_metrics_model_assumptions: # dynamically add new metrics
		if k not in metrics:
			print(k)
			metrics[k] = copy.deepcopy(default_metrics_model_assumptions[k])
		for i in range(N_TO_SIM):
			for k in metrics:
				if i not in metrics[k] and i in default_metrics_model_assumptions[k]:
					print("{} {}".format(k, i))
					metrics[k][i] = copy.deepcopy(default_metrics_model_assumptions[k][i])

	RECALC_PREFERENCE_CHANGES = False
	try:
		for random_iter in range(N_TO_SIM):
			k_of_interest = 'path_preference_changes_latency'
			havent_calced_everything = check_calced_everything(metrics,random_iter, k_of_interest)
			if RECALC_PREFERENCE_CHANGES or havent_calced_everything:
				if sas is None:
					deployment = metrics['deployment'][random_iter]

					n_prefixes = deployment_to_prefixes(deployment)
					sas = Sparse_Advertisement_Eval(deployment, verbose=True,
						lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
						using_resilience_benefit=True, gamma=gamma, n_prefixes=n_prefixes)
					if wm is None:
						wm = Worker_Manager(sas.get_init_kwa(), deployment)
						wm.start_workers()
					sas.set_worker_manager(wm)
					sas.update_deployment(deployment)
				else:
					deployment = metrics['deployment'][random_iter]
					sas.update_deployment(deployment)

				for solution in soln_types:
					if not RECALC_PREFERENCE_CHANGES:
						if metrics[k_of_interest][random_iter][solution]  != \
							default_metrics[k_of_interest][random_iter][solution]:
							print("Already calced {}".format(solution))
							continue
					print(solution)
					adv = metrics['adv'][random_iter][solution]
					if len(adv) == 0:
						print("No solution for {}".format(solution))
						continue
					metrics['path_preference_changes_latency'][random_iter][solution] = change_model_preferences(sas,adv,deployment)

				pickle.dump(metrics, open(modeling_metrics_fn,'wb'))
	except:
		import traceback
		traceback.print_exc()

	RECALC_CAPACITY_CHANGES = False
	try:
		for random_iter in range(N_TO_SIM):
			k_of_interest = 'path_capacity_changes_latency'
			havent_calced_everything = check_calced_everything(metrics,random_iter, k_of_interest)
			if RECALC_CAPACITY_CHANGES or havent_calced_everything:
				if sas is None:
					deployment = metrics['deployment'][random_iter]

					n_prefixes = deployment_to_prefixes(deployment)
					sas = Sparse_Advertisement_Eval(deployment, verbose=True,
						lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
						using_resilience_benefit=True, gamma=gamma, n_prefixes=n_prefixes)
					if wm is None:
						wm = Worker_Manager(sas.get_init_kwa(), deployment)
						wm.start_workers()
					sas.set_worker_manager(wm)
					sas.update_deployment(deployment)
				else:
					deployment = metrics['deployment'][random_iter]
					sas.update_deployment(deployment)

				for solution in soln_types:
					if not RECALC_CAPACITY_CHANGES:
						if metrics[k_of_interest][random_iter][solution]  != \
							default_metrics[k_of_interest][random_iter][solution]:
							print("Already calced {}".format(solution))
							continue
					print(solution)
					adv = metrics['adv'][random_iter][solution]
					if len(adv) == 0:
						print("No solution for {}".format(solution))
						continue
					metrics['path_capacity_changes_latency'][random_iter][solution] = change_model_capacities(sas,adv,deployment)

				pickle.dump(metrics, open(modeling_metrics_fn,'wb'))
	except:
		import traceback
		traceback.print_exc()


	# RECALC_LATENCY_CHANGES = False
	# try:
	# 	for random_iter in range(N_TO_SIM):
	# 		k_of_interest = 'path_latency_changes_latency'
	# 		havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)
	# 		if RECALC_LATENCY_CHANGES or havent_calced_everything:
	# 			if sas is None:
	# 				deployment = metrics['deployment'][random_iter]

	# 				n_prefixes = deployment_to_prefixes(deployment)
	# 				sas = Sparse_Advertisement_Eval(deployment, verbose=True,
	# 					lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
	# 					using_resilience_benefit=True, gamma=gamma, n_prefixes=n_prefixes)
	# 				if wm is None:
	# 					wm = Worker_Manager(sas.get_init_kwa(), deployment)
	# 					wm.start_workers()
	# 				sas.set_worker_manager(wm)
	# 				sas.update_deployment(deployment)
	# 			else:
	# 				deployment = metrics['deployment'][random_iter]
	# 				sas.update_deployment(deployment)

	# 			for solution in soln_types:
	# 				if not RECALC_LATENCY_CHANGES:
	# 					if metrics[k_of_interest][random_iter][solution]  != \
	# 						default_metrics[k_of_interest][random_iter][solution]:
	# 						print("Already calced {}".format(solution))
	# 						continue
	# 				print(solution)
	# 				adv = metrics['adv'][random_iter][solution]
	# 				if len(adv) == 0:
	# 					print("No solution for {}".format(solution))
	# 					continue
	# 				metrics['path_latency_changes_latency'][random_iter][solution] = change_model_latencies(sas,adv,deployment)

	# 			pickle.dump(metrics, open(modeling_metrics_fn,'wb'))
	# except:
	# 	import traceback
	# 	traceback.print_exc()
	















	finally:
		if wm is not None:
			wm.stop_workers()


	################################
	### PLOTTING
	################################
	
	PATH_PREFERENCE_CHANGES_I = 0
	POPP_FAILURE_LATENCY_I = 1
	POPP_FAILURE_CONGESTION_I = 2
	POP_FAILURE_LATENCY_I = 3
	POP_FAILURE_CONGESTION_I = 4
	FLASH_CROWD_SINGLE_X_I = 5
	FLASH_CROWD_LATENCY_VARY_X_I = 6
	FLASH_CROWD_CONGESTION_VARY_X_I = 7
	FLASH_CROWD_PREFIX_I = 8
	VOLUME_GROWTH_I = 9

	n_subs = 2
	f,ax=plt.subplots(n_subs,1)
	f.set_size_inches(6,4*n_subs)

	for solution in soln_types:
		print(solution)
		try:
			#### Resilience to path preference changes
			## path preference changes
			all_differences = [metrics['path_preference_changes_latency'][ri][solution][i] for ri in range(N_TO_SIM) for 
				i in range(len(metrics['path_preference_changes_latency'][ri][solution]))]
			x,cdf_x = get_cdf_xy(all_differences,weighted=True)
			ax[PATH_PREFERENCE_CHANGES_I].plot(x,cdf_x,label=solution)
		except:
			import traceback
			traceback.print_exc()
			continue

	ax[PATH_PREFERENCE_CHANGES_I].legend(fontsize=8)
	ax[PATH_PREFERENCE_CHANGES_I].grid(True)
	ax[PATH_PREFERENCE_CHANGES_I].set_xlabel("Latency Change Under Path Pref. Change (best - actual) (ms)")
	ax[PATH_PREFERENCE_CHANGES_I].set_ylabel("CDF of Changes,UGs")
	ax[PATH_PREFERENCE_CHANGES_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	save_fig("modeling_assumptions_eval-{}.pdf".format(DPSIZE))



if __name__ == "__main__":
	modeling_assumption_comparisons()