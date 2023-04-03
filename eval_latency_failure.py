from constants import *
from helpers import *

import pickle, numpy as np, matplotlib.pyplot as plt
from sparse_advertisements_v3 import *

def popp_failure_latency_comparisons():
	# for each deployment, get different advertisement strategies
	# look at latency under popp and link failures compared to optimal

	# goal is to hopefully demonstrate that we're better on these simulated topologies
	# so that we can be confident we'll find resilient strategies in the wild

	np.random.seed(31413)
	metrics = {}
	N_TO_SIM = 3

	lambduh = .01

	wm = None
	
	soln_types = ['sparse', 'one_per_pop', 'painter', 'anyopt']
	metrics_fn = os.path.join(CACHE_DIR, 'popp_failure_latency_comparison_{}.pkl'.format(DPSIZE))
	metrics = {
		'popp_failures': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'latencies': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'best_latencies': {},
		'adv': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'deployment': {i:None for i in range(N_TO_SIM)},
		'settings': {i:None for i in range(N_TO_SIM)},
	}
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
			metrics['deployment'][random_iter] = deployment

			n_prefixes = 20
			sas = Sparse_Advertisement_Eval(deployment, verbose=True,
				lambduh=lambduh,with_capacity=False,explore=DEFAULT_EXPLORE, 
				using_resilience_benefit=True, gamma=1.5,n_prefixes=n_prefixes)
			metrics['settings'][random_iter] = sas.get_init_kwa()
			if wm is None:
				wm = Worker_Manager(sas.get_init_kwa(), deployment)
				wm.start_workers()
			sas.set_worker_manager(wm)
			sas.update_deployment(deployment)
			ret = sas.compare_different_solutions(deployment_size=DPSIZE,n_run=1, verbose=True,
				 dont_update_deployment=True)

			adv = sas.sas.get_last_advertisement()
			print("sparse {}".format(np.round(adv,2)))

			for solution in soln_types:
				adv = ret['adv_solns'][solution][0]
				print("{} {} ".format(solution,np.round(adv,2)))

				user_latencies, ug_catchments = sas.calculate_user_choice(adv)
				metrics['adv'][random_iter][solution] = adv
				metrics['latencies'][random_iter][solution] = user_latencies
				metrics['best_latencies'][random_iter] = np.zeros(user_latencies.shape)
				for ug in sas.ugs:
					metrics['best_latencies'][random_iter][sas.ug_to_ind[ug]] = np.min(
						list(sas.ug_perfs[ug].values()))
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
						best_perf = np.min(list(sas.ug_perfs[ug].values()))
						other_available = [sas.ug_perfs[ug][u] for u in sas.ug_perfs[ug] if u != popp]
						if len(other_available) == 0:
							best_perf = MAX_LATENCY
						else:
							best_perf = np.min(other_available)
						poppi = ug_catchments[sas.ug_to_ind[ug]]
						if poppi is None:
							actual_perf = NO_ROUTE_LATENCY
						else:
							actual_ingress = sas.popps[poppi]
							actual_perf = sas.ug_perfs[ug][actual_ingress]
						if solution == 'sparse' and best_perf + 2 < actual_perf:
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
	f,ax=plt.subplots(2,1)
	f.set_size_inches(6,12)

	for solution in soln_types:
		print("Solution {}".format(solution))
		for i in range(N_TO_SIM):
			adv = metrics['adv'][i][solution]
			deployment = metrics['deployment'][i]
			popps = sorted(list(set(deployment['popps'])))
			for pref in range(adv.shape[1]):
				if np.sum(adv[:,pref]) == adv.shape[0]:
					print("Prefix {} is anycast".format(pref))
				else:
					for poppi in np.where(adv[:,pref])[0]:
						print("Prefix {} has {}".format(pref, popps[poppi]))
				print("\n")
			print("\n")
			print("\n")

	for solution in soln_types:
		all_differences = np.array([el for ri in range(N_TO_SIM) for el in metrics['popp_failures'][ri][solution] ])
		x,cdf_x = get_cdf_xy(all_differences)
		ax[0].plot(x,cdf_x,label=solution)

		diffs = []
		for random_iter in range(N_TO_SIM):
			diffs = diffs + list(metrics['best_latencies'][random_iter] - metrics['latencies'][random_iter][solution])

		x,cdf_x = get_cdf_xy(diffs)
		ax[1].plot(x,cdf_x,label=solution)

	ax[0].legend(fontsize=8)
	ax[0].grid(True)
	ax[0].set_xlabel("Best - Actual Latency Under Failure (ms)")
	ax[0].set_ylabel("CDF of UGs")

	ax[1].legend(fontsize=8)
	ax[1].grid(True)
	ax[1].set_xlabel("Best - Actual Latency (ms)")
	ax[1].set_ylabel("CDF of UGs")

	save_fig("popp_latency_failure_comparison_{}.pdf".format(DPSIZE))

def plot_lats_from_adv(sas, advertisement, fn):
	f,ax=plt.subplots(2,1)
	f.set_size_inches(6,12)

	labs = ['yours', 'anycast']

	for i in range(2):
		metrics = {}
		if i ==0:
			adv = advertisement
		else: # anycast
			adv = np.zeros(advertisement.shape)
			adv[:,0] = 1
		user_latencies, ug_catchments = sas.calculate_user_choice(adv)
		metrics['latencies'] = user_latencies
		metrics['popp_failures'] = []
		metrics['best_latencies'] = np.zeros(user_latencies.shape)
		for ug in sas.ugs:
			metrics['best_latencies'][sas.ug_to_ind[ug]] = np.min(
				list(sas.ug_perfs[ug].values()))

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
				best_perf = np.min(list(sas.ug_perfs[ug].values()))
				other_available = [sas.ug_perfs[ug][u] for u in sas.ug_perfs[ug] if u != popp]
				if len(other_available) == 0:
					best_perf = MAX_LATENCY
				else:
					best_perf = np.min(other_available)
				poppi = ug_catchments[sas.ug_to_ind[ug]]
				if poppi is None:
					actual_perf = NO_ROUTE_LATENCY
				else:
					actual_ingress = sas.popps[poppi]
					actual_perf = sas.ug_perfs[ug][actual_ingress]
				metrics['popp_failures'].append(best_perf - actual_perf)



		all_differences = np.array(metrics['popp_failures'])
		x,cdf_x = get_cdf_xy(all_differences)
		ax[0].plot(x,cdf_x, label=labs[i])

		diffs = list(metrics['best_latencies'] - metrics['latencies'])
		x,cdf_x = get_cdf_xy(diffs)
		ax[1].plot(x,cdf_x, label=labs[i])

	ax[0].legend(fontsize=8)
	ax[0].grid(True)
	ax[0].set_xlabel("Best - Actual Latency Under Failure (ms)")
	ax[0].set_ylabel("CDF of UGs")

	ax[1].legend(fontsize=8)
	ax[1].grid(True)
	ax[1].set_xlabel("Best - Actual Latency Normally (ms)")
	ax[1].set_ylabel("CDF of UGs")

	save_fig(fn)

if __name__ == "__main__":
	popp_failure_latency_comparisons()
