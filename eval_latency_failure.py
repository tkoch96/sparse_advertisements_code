from constants import *
from helpers import *

import pickle, numpy as np, matplotlib.pyplot as plt
from sparse_advertisements_v3 import *

def popp_failure_latency_comparisons():
	# for each deployment, get different advertisement strategies
	# look at latency under popp and link failures compared to optimal

	# goal is to hopefully demonstrate that we're better on these simulated topologies
	# so that we can be confident we'll find resilient strategies in the wild

	np.random.seed(31414)
	metrics = {}
	N_TO_SIM = 1

	lambduh = .1
	
	wm = None
	
	soln_types = ['sparse', 'one_per_pop', 'painter', 'anyopt']
	metrics_fn = os.path.join(CACHE_DIR, 'popp_failure_latency_comparison_{}.pkl'.format(DPSIZE))
	metrics = {'popp_failures': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)}}
	if os.path.exists(metrics_fn):
		metrics = pickle.load(open(metrics_fn,'rb'))

	try:
		for random_iter in range(N_TO_SIM):
			try:
				metrics['popp_failures'][random_iter]
				if len(metrics['popp_failures'][random_iter]['sparse']) > 0: 
					continue
			except KeyError:
				pass
			print("-----Deployment number = {} -------".format(random_iter))
			metrics['popp_failures'][random_iter] = {k:[] for k in soln_types}
			deployment = get_random_deployment(DPSIZE)
			sas = Sparse_Advertisement_Eval(deployment, verbose=False,
				lambduh=lambduh,with_capacity=False,explore=DEFAULT_EXPLORE, 
				using_resilience_benefit=True, gamma=.1)
			if wm is None:
				wm = Worker_Manager(sas.get_init_kwa(), deployment)
				wm.start_workers()
			sas.set_worker_manager(wm)
			sas.update_deployment(deployment)
			ret = sas.compare_different_solutions(deployment_size=DPSIZE,n_run=1, verbose=False,
				 dont_update_deployment=True)

			for solution in soln_types:
				adv = threshold_a(ret['adv_solns'][solution][0])
				print("{} {} ".format(solution,adv))

				_, ug_catchments = sas.calculate_user_choice(adv)
				for popp in sas.popps:
					these_ugs = [ug for ug in sas.ugs if \
						sas.popp_to_ind[popp] == ug_catchments[sas.ug_to_ind[ug]]]
					if len(these_ugs) == 0: 
						continue
					adv_cpy = np.copy(adv)
					adv_cpy[sas.popp_to_ind[popp]] = 0
					## q: what is latency experienced for these ugs compared to optimal?
					_, ug_catchments = sas.calculate_user_choice(adv_cpy)
					for ug in these_ugs:
						other_available = [sas.ug_perfs[ug][u] for u in sas.ug_perfs[ug] if u != popp]
						if len(other_available) == 0:
							best_perf = MAX_LATENCY
						else:
							best_perf = np.min(other_available)
						
						actual_ingress = sas.popps[ug_catchments[sas.ug_to_ind[ug]]]
						actual_perf = sas.ug_perfs[ug][actual_ingress]
						if solution == 'sparse' and best_perf + 10 < actual_perf:
							print("UG {} popp {} ({}) actual {} best {}".format(
								ug,popp,sas.popp_to_ind[popp],actual_perf, best_perf))
							for k,v in sas.ug_perfs[ug].items():
								print("{} ({}) --> {} ms".format(k,sas.popp_to_ind[k],v))
							print("\n")
						metrics['popp_failures'][random_iter][solution].append(best_perf - actual_perf)
			# exit(0)
			pickle.dump(metrics, open(metrics_fn,'wb'))
	except:
		import traceback
		traceback.print_exc()
		exit(0)
	finally:
		if wm is not None:
			wm.stop_workers()
	f,ax=plt.subplots(1,1)
	print(metrics)

	for solution in soln_types:
		all_differences = np.array([el for ri in range(N_TO_SIM) for el in metrics['popp_failures'][ri][solution] ])
		x,cdf_x = get_cdf_xy(all_differences)
		ax.plot(x,cdf_x,label=solution)

	ax.legend()
	ax.grid(True)
	ax.set_xlabel("Best - Actual Latency (ms)")
	ax.set_ylabel("CDF of UGs under Ingress Failures")
	save_fig("popp_latency_failure_comparison.pdf")

if __name__ == "__main__":
	popp_failure_latency_comparisons()
