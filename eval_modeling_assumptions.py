### path preference changes
### latency changes
### capacity changes

from wrapper_eval import *
from constants import *
from helpers import *
from solve_lp_assignment import *

import pickle, numpy as np, matplotlib.pyplot as plt, copy, itertools, time
from sparse_advertisements_v3 import *

def modeling_assumption_comparisons():
	wm = None
	sas = None

	if not os.path.exists(performance_metrics_fn):
		raise FileNotFoundError("You need to populate model file first : {}".format(performance_metrics_fn))

	modeling_metrics_fn = os.path.join(CACHE_DIR, 'modeling_assumption_comparisons_{}.pkl'.format(DPSIZE))
	default_metrics = {
		'path_preference_changes_latency': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
	}
	metrics = copy.deepcopy(default_metrics)
	if os.path.exists(modeling_metrics_fn):
		metrics = pickle.load(open(modeling_metrics_fn,'rb'))
	d = pickle.load(open(performance_metrics_fn,'rb'))
	for k in d:
		metrics[k] = d[k]
	for k in default_metrics: # dynamically add new metrics
		if k not in metrics:
			print(k)
			metrics[k] = copy.deepcopy(default_metrics[k])
		for i in range(N_TO_SIM):
			for k in metrics:
				if i not in metrics[k] and i in default_metrics[k]:
					print("{} {}".format(k, i))
					metrics[k][i] = copy.deepcopy(default_metrics[k][i])

	RECALC_PREFERENCE_CHANGES = False
	try:
		for random_iter in range(N_TO_SIM):
			if RECALC_PREFERENCE_CHANGES or metrics['path_preference_changes_latency'][random_iter]['sparse']  == \
				default_metrics['path_preference_changes_latency'][random_iter]['sparse']:
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
					print(solution)
					adv = metrics['adv'][random_iter][solution]
					if len(adv) == 0:
						print("No solution for {}".format(solution))
						continue
					metrics['path_preference_changes_latency'][random_iter][solution] = []
					deployment = sas.output_deployment()
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
						ret1 = assess_failure_resilience(sas, adv, which='popps')
						sas.update_deployment(deployment, quick_update=False)
						## Look at latency
						ret = assess_failure_resilience(sas, adv, which='popps')
						for x,y in zip(ret1['latency_delta'],ret['latency_delta']):
							print(x)
							if x[0] - y[0] != 0:
								print(x[0] - y[0])
						exit(0)
						for rval in ret['latency_delta']:
							metrics['path_preference_changes_latency'][random_iter][solution].append(rval)
	
						deployment['ingress_priorities'] = copy.deepcopy(pref_cpy)

				pickle.dump(metrics, open(modeling_metrics_fn,'wb'))

	except:
		import traceback
		traceback.print_exc()
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