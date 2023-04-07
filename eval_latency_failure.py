from constants import *
from helpers import *

import pickle, numpy as np, matplotlib.pyplot as plt, copy
from sparse_advertisements_v3 import *

def adv_summary(popps,adv):
	adv = threshold_a(adv)
	print("\n")
	print("\n")
	print(adv)
	for pref in range(adv.shape[1]):
		if np.sum(adv[:,pref]) == adv.shape[0]:
			print("Prefix {} is anycast".format(pref))
		else:
			for poppi in np.where(adv[:,pref])[0]:
				print("Prefix {} has {}".format(pref, popps[poppi]))
		print("\n")
	print("\n")
	print("\n")



def popp_failure_latency_comparisons():
	# for each deployment, get different advertisement strategies
	# look at latency under popp and link failures compared to optimal
	# sparse should be simultaneously better in both normal and failure scenarios

	np.random.seed(31413)
	metrics = {}
	N_TO_SIM = 1
	gamma = 2
	capacity = True

	lambduh = .01

	wm = None
	
	soln_types = ['sparse', 'one_per_pop', 'painter', 'anyopt', 'oracle']
	metrics_fn = os.path.join(CACHE_DIR, 'popp_failure_latency_comparison_{}.pkl'.format(DPSIZE))
	metrics = {
		'popp_failures': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'latencies': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'best_latencies': {},
		'adv': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'deployment': {i:None for i in range(N_TO_SIM)},
		'settings': {i:None for i in range(N_TO_SIM)},
		'ug_to_vol': {i:None for i in range(N_TO_SIM)},
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
			print(deployment['link_capacities'])
			metrics['deployment'][random_iter] = deployment

			n_prefixes = np.maximum(4,len(deployment['popps'])//10)
			sas = Sparse_Advertisement_Eval(deployment, verbose=True,
				lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
				using_resilience_benefit=True, gamma=gamma, n_prefixes=n_prefixes)
			metrics['settings'][random_iter] = sas.get_init_kwa()
			if wm is None:
				wm = Worker_Manager(sas.get_init_kwa(), deployment)
				wm.start_workers()
			sas.set_worker_manager(wm)
			sas.update_deployment(deployment)
			ret = sas.compare_different_solutions(deployment_size=DPSIZE,n_run=1, verbose=True,
				 dont_update_deployment=True)

			ug_vols = sas.ug_to_vol
			adv = sas.sas.get_last_advertisement()
			print("sparse {}".format(np.round(adv,2)))

			for solution in soln_types:
				try:
					adv = ret['adv_solns'][solution][0]
				except:
					print("No solution for {}".format(solution))
					continue
				print("{} {} ".format(solution,np.round(adv,2)))

				_, pre_ug_catchments = sas.calculate_user_choice(adv)
				pre_user_latencies = sas.get_ground_truth_user_latencies(adv)
				metrics['adv'][random_iter][solution] = adv
				metrics['latencies'][random_iter][solution] = pre_user_latencies
				metrics['best_latencies'][random_iter] = np.zeros(pre_user_latencies.shape)
				metrics['ug_to_vol'][random_iter] = sas.ug_vols
				for ug in sas.ugs:
					metrics['best_latencies'][random_iter][sas.ug_to_ind[ug]] = np.min(
						list(sas.ug_perfs[ug].values()))
				for popp in sas.popps:
					adv_cpy = np.copy(adv)
					adv_cpy[sas.popp_to_ind[popp]] = 0
					## q: what is latency experienced for these ugs compared to optimal?
					_, ug_catchments = sas.calculate_user_choice(adv_cpy)
					user_latencies = sas.get_ground_truth_user_latencies(adv_cpy)

					## Look at users whose catchment has changed
					these_ugs = [ug for ug in sas.ugs if \
						pre_ug_catchments[sas.ug_to_ind[ug]] != ug_catchments[sas.ug_to_ind[ug]]]

					for ug in these_ugs:
						best_perf = np.min(list(sas.ug_perfs[ug].values()))
						other_available = [sas.ug_perfs[ug][u] for u in sas.ug_perfs[ug] if u != popp]
						if len(other_available) == 0:
							best_perf = MAX_LATENCY
						else:
							best_perf = np.min(other_available)
						poppi = ug_catchments[sas.ug_to_ind[ug]]
						# if poppi is None:
						# 	actual_perf = NO_ROUTE_LATENCY
						# else:
						# 	actual_ingress = sas.popps[poppi]
						# 	actual_perf = sas.ug_perfs[ug][actual_ingress]
						actual_perf = user_latencies[sas.ug_to_ind[ug]]
						if solution == 'sparse' and best_perf + 2 < actual_perf:
							print("UG {} popp {} ({}) actual {} best {}".format(
								ug,popp,sas.popp_to_ind[popp],actual_perf, best_perf))
							for k,v in sas.ug_perfs[ug].items():
								print("{} ({}) --> {} ms".format(k,sas.popp_to_ind[k],v))
							print("\n")
						metrics['popp_failures'][random_iter][solution].append((best_perf - actual_perf, ug_vols[ug]))
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
			try:
				adv_summary(popps,adv)
			except:
				continue

	for solution in soln_types:
		try:
			all_differences = [el for ri in range(N_TO_SIM) for el in metrics['popp_failures'][ri][solution]]
			x,cdf_x = get_cdf_xy(all_differences,weighted=True)
			ax[0].plot(x,cdf_x,label=solution)

			diffs = []
			wts = []
			for random_iter in range(N_TO_SIM):
				for i in range(len(metrics['best_latencies'][random_iter])):
					diffs.append(metrics['best_latencies'][random_iter][i] - metrics['latencies'][random_iter][solution][i])
					wts.append(metrics['ug_to_vol'][random_iter][i])

			x,cdf_x = get_cdf_xy(list(zip(diffs,wts)), weighted=True)
			ax[1].plot(x,cdf_x,label=solution)
		except:
			continue

	ax[0].legend(fontsize=8)
	ax[0].grid(True)
	ax[0].set_xlabel("Best - Actual Latency Under Failure (ms)")
	ax[0].set_ylabel("CDF of UGs")
	ax[0].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[0].set_xlim([-50,0])

	ax[1].legend(fontsize=8)
	ax[1].grid(True)
	ax[1].set_xlabel("Best - Actual Latency (ms)")
	ax[1].set_ylabel("CDF of UGs")
	ax[1].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[1].set_xlim([-50,0])

	save_fig("popp_latency_failure_comparison_{}.pdf".format(DPSIZE))

def plot_lats_from_adv(sas, advertisement, fn):
	verb_copy = copy.copy(sas.verbose)
	sas.verbose = False

	f,ax=plt.subplots(2,1)
	f.set_size_inches(6,12)

	labs = ['yours', 'anycast']

	ug_vols = sas.ug_to_vol
	ug_vols_arr = sas.ug_vols

	for i in range(2):
		metrics = {}
		if i ==0:
			adv = threshold_a(advertisement)
			print("Current adv: {}".format(adv_summary(sas.popps, advertisement)))
		else: # anycast
			adv = np.zeros(advertisement.shape)
			adv[:,0] = 1
		_, pre_ug_catchments = sas.calculate_user_choice(adv)
		pre_user_latencies = sas.get_ground_truth_user_latencies(adv)

		metrics['latencies'] = pre_user_latencies
		metrics['popp_failures'] = []
		metrics['best_latencies'] = np.zeros(pre_user_latencies.shape)
		for ug in sas.ugs:
			metrics['best_latencies'][sas.ug_to_ind[ug]] = np.min(
				list(sas.ug_perfs[ug].values()))

		for popp in sas.popps:
			adv_cpy = np.copy(adv)
			adv_cpy[sas.popp_to_ind[popp]] = 0
			## q: what is latency experienced for these ugs compared to optimal?
			_, ug_catchments = sas.calculate_user_choice(adv_cpy)
			user_latencies = sas.get_ground_truth_user_latencies(adv_cpy,
				overloadverb=(i==0), failing=popp)

			## Look at users whose catchment has changed
			these_ugs = [ug for ug in sas.ugs if \
				pre_ug_catchments[sas.ug_to_ind[ug]] != ug_catchments[sas.ug_to_ind[ug]]]

			inundated=False
			for ug in these_ugs:
				other_available = [sas.ug_perfs[ug][u] for u in sas.ug_perfs[ug] if u != popp]
				if len(other_available) == 0:
					best_perf = MAX_LATENCY
				else:
					best_perf = np.min(other_available)
				poppi = ug_catchments[sas.ug_to_ind[ug]]
				# if poppi is None:
				# 	actual_perf = NO_ROUTE_LATENCY
				# else:
				# 	actual_ingress = sas.popps[poppi]
				# 	actual_perf = sas.ug_perfs[ug][actual_ingress]
				actual_perf = user_latencies[sas.ug_to_ind[ug]]
				if actual_perf == NO_ROUTE_LATENCY:
					inundated = True
				metrics['popp_failures'].append((best_perf - actual_perf, ug_vols[ug]))
			if inundated and i==0:
				recent_iter = sas.all_rb_calls_results[sas.popp_to_ind[popp]][-1][0]
				these_rb_calls = [call for call in sas.all_rb_calls_results[sas.popp_to_ind[popp]] if
					call[0] == recent_iter]
				print("{} recent grad calls".format(len(these_rb_calls)))
				recent_rb = these_rb_calls[-30:]
				recent_lb = sas.all_lb_calls_results[-1]
				# iter poppi popp prefix rbgrad lbgrad
				recent_rb = [(i,poppi,sas.popps[poppi],prefi,round(rbgrad,2),round(recent_lb[poppi,prefi],2),
					advertisement[poppi,prefi]) for i,poppi,prefi,rbgrad in 
					recent_rb]
				print("Popp {} fail causes inundation, recent resilience gradient calls were : {}".format(
					popp, recent_rb))


		all_differences = metrics['popp_failures']
		x,cdf_x = get_cdf_xy(all_differences, weighted=True)
		ax[0].plot(x,cdf_x, label=labs[i])

		diffs = list(metrics['best_latencies'] - metrics['latencies'])
		x,cdf_x = get_cdf_xy(list(zip(diffs, ug_vols_arr)), weighted=True)
		ax[1].plot(x,cdf_x, label=labs[i])

		metrics['adv'] = advertisement

		if i == 0:
			pickle.dump(metrics, open('cache/last_run_metrics.pkl','wb'))

	ax[0].legend(fontsize=8)
	ax[0].grid(True)
	ax[0].set_xlabel("Best - Actual Latency Under Failure (ms)")
	ax[0].set_ylabel("CDF of UGs")
	ax[0].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	ax[1].legend(fontsize=8)
	ax[1].grid(True)
	ax[1].set_xlabel("Best - Actual Latency Normally (ms)")
	ax[1].set_ylabel("CDF of UGs")
	ax[1].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	save_fig(fn)
	sas.verbose = verb_copy

if __name__ == "__main__":
	popp_failure_latency_comparisons()
