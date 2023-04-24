from constants import *
from helpers import *

import pickle, numpy as np, matplotlib.pyplot as plt, copy, itertools, time
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



def compute_optimal_prefix_withdrawals(sas, adv, popp, new_link_capacity,**kwargs):
	## for a given popp, there's technically power_set(number of advertisement to popp) that we could turn off
	## so if that's intractable, might need to sample it or something

	_, pre_ug_catchments = sas.calculate_user_choice(adv)
	poppi = sas.popp_to_ind[popp]
	pre_user_latencies = sas.get_ground_truth_user_latencies(adv)

	prefis = np.where(adv[poppi,:])[0]
	if kwargs.get('verb'):
		print(prefis)
	valid_solutions = []
	ts = time.time()
	max_time = 600
	for withdrawal_number in range(1,len(prefis) + 1):
		for prefi_set in itertools.combinations(prefis,withdrawal_number):
			adv[poppi,np.array(prefi_set)] = 0
			_, ug_catchments = sas.calculate_user_choice(adv) # simulates TIPSY

			## Look at users whose catchment has changed
			these_ugs = [ug for ug in sas.ugs if \
				pre_ug_catchments[sas.ug_to_ind[ug]] != ug_catchments[sas.ug_to_ind[ug]]]
			if len(these_ugs) == 0:
				continue
			all_new_link_volumes = np.zeros((sas.n_popps,))
			for ugi,catchi in ug_catchments.items():
				if catchi is None:
					continue
				if kwargs.get('verb'):
					print(ugi)
					print("{} {}".format(catchi,sas.ug_vols[ugi]))
				all_new_link_volumes[catchi] += sas.ug_vols[ugi]
				if kwargs.get('verb'):
					if all_new_link_volumes[poppi] > 0:
						print(all_new_link_volumes)
						print(ugi)
						print(catchi)
			new_link_volume = all_new_link_volumes[poppi]

			if new_link_volume <= new_link_capacity:
				user_latencies = sas.get_ground_truth_user_latencies(adv)
				if not any(user_latencies==NO_ROUTE_LATENCY): ## check to make sure we don't inundate any other link
					these_ugis = np.array([sas.ug_to_ind[ug] for ug in these_ugs])
					new_perfs = np.array([user_latencies[sas.ug_to_ind[ug]] for ug in these_ugs])
					valid_solutions.append((prefi_set, new_perfs - pre_user_latencies[these_ugis],
						sas.ug_vols[these_ugis], all_new_link_volumes))

			adv[poppi,np.array(prefi_set)] = 1
		if len(valid_solutions) > 0 or time.time() - ts > max_time:
			# greedily stop
			break
	if len(valid_solutions) > 0:
		## pick the best one w.r.t. latency
		best_solution,best_deltas,vols,best_new_link_volumes = valid_solutions[0]
		best_metric = np.sum(best_deltas)
		for solution in valid_solutions:
			pref_solution, performance_deltas, vols, new_link_volumes = solution
			if np.sum(performance_deltas * vols) < best_metric:
				best_solution = pref_solution
				best_metric = np.sum(performance_deltas * vols)
				best_deltas = performance_deltas
				best_new_link_volumes = new_link_volumes
		best_solution = {
			'prefix_withdrawals': best_solution,
			'latency_deltas': best_deltas,
			'link_volumes': best_new_link_volumes,
		}
	else:
		best_solution = None

	
	return best_solution

def assess_resilience_to_congestion(sas, adv, solution, X_vals):
	## for painter/TIPSY
	## assume each link is congested by X% (i.e., need to move X% of capacity's traffic off of it)
	## see if there's a solution
	## if there's a solution, do it and note the latency penalty compared to optimal

	# return cdf of latency penalties, possibly as a function of X
	metrics = {X:[] for X in X_vals}
	required_link_caps = None

	adv = threshold_a(adv)

	if solution == 'painter':
		old_caps = copy.deepcopy(sas.link_capacities_by_popp)
		deployment = sas.deployment
		deployment['link_capacities'] = {popp:100000 for popp in sas.popps}
		sas.update_deployment(deployment)

	for X in X_vals:
		_, pre_ug_catchments = sas.calculate_user_choice(adv)
		for popp in sas.popps:
			these_ugis = np.array([ugi for ugi,catchi in pre_ug_catchments.items() if catchi == sas.popp_to_ind[popp]])
			poppi = sas.popp_to_ind[popp]
			if len(these_ugis) == 0 or len(np.where(adv[poppi,:])[0]) == 0:
				# nothing to solve
				continue
			current_link_volume = np.sum(sas.ug_vols[these_ugis])
			new_link_cap = current_link_volume  * (1- X/100)
			soln = compute_optimal_prefix_withdrawals(sas,adv,popp,new_link_cap)
			if soln is None and solution == 'painter':
				print("Didn't get solution for popp {}, soln type {}".format(popp,solution))
				soln = compute_optimal_prefix_withdrawals(sas,adv,popp,new_link_cap,verb=True)
				exit(0)
			elif soln is None:
				## assign badness
				print("Didn't get solution for popp {}, soln type {}".format(popp,solution))
				metrics[X] = metrics[X] + list(NO_ROUTE_LATENCY * np.ones((len(these_ugis))))
				continue

			## summarize solutions
			pfx_withdrawals = soln['prefix_withdrawals']
			adv[poppi,np.array(pfx_withdrawals)] = 0
			_, post_ug_catchments = sas.calculate_user_choice(adv)
			users_of_interest = [ui for ui,catch in post_ug_catchments.items() if catch != pre_ug_catchments[ui]]
			for ui in users_of_interest:
				ug = sas.ugs[ui]
				catch_before, catch_after = sas.popps[pre_ug_catchments[ui]], sas.popps[post_ug_catchments[ui]]
				print("UG {} was visiting popp {} with {} ms, after withdrawal is {} with {} ms, delta {} ms".format(
					ug,catch_before,round(sas.ug_perfs[ug][catch_before]),catch_after,round(sas.ug_perfs[ug][catch_after]),
					round(sas.ug_perfs[ug][catch_after] - sas.ug_perfs[ug][catch_before])))
			adv[poppi,np.array(pfx_withdrawals)] = 1


			metrics[X] = metrics[X] + list(soln['latency_deltas'])
			new_required_link_caps = soln['link_volumes']
			if required_link_caps is None and new_required_link_caps is not None:
				required_link_caps = new_required_link_caps
			else:
				update_capis = new_required_link_caps > required_link_caps
				required_link_caps[update_capis] = new_required_link_caps[update_capis]

	if solution == 'painter':
		deployment['link_capacities'] = old_caps
		sas.update_deployment(deployment)

	if required_link_caps is not None:
		link_capacities = {sas.popps[poppi]:required_link_caps[poppi] for poppi in range(sas.n_popps)}
	else:
		link_capacities = None
	return {
		'link_capacities': link_capacities,
		'metrics': metrics,
	}


def popp_failure_latency_comparisons():
	# for each deployment, get different advertisement strategies
	# look at latency under popp and link failures compared to optimal
	# sparse should be simultaneously better in both normal and failure scenarios

	np.random.seed(31413)
	metrics = {}
	N_TO_SIM = 1
	X_vals = [20]
	gamma = 10
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
		'resilience_to_congestion': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
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
			n_prefixes = np.maximum(4,5 * int(np.log2(len(deployment['popps']))))
			n_prefixes = np.minimum(len(deployment['popps'])//4, n_prefixes)
			sas = Sparse_Advertisement_Eval(deployment, verbose=True,
				lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
				using_resilience_benefit=True, gamma=gamma, n_prefixes=n_prefixes)
			metrics['settings'][random_iter] = sas.get_init_kwa()
			if wm is None:
				wm = Worker_Manager(sas.get_init_kwa(), deployment)
				wm.start_workers()
			sas.set_worker_manager(wm)
			sas.update_deployment(deployment)
			sas.solve_painter()

			## See which capacities are required for PAINTER
			adv = sas.painter_solution['advertisement']
			solution = 'painter'
			ret = assess_resilience_to_congestion(sas, adv, solution, X_vals)
			new_link_capacities = ret['link_capacities']
			m = ret['metrics']
			pickle.dump(m,open('tmp.pkl','wb'))
			f,ax = plt.subplots()
			f.set_size_inches(8,6)
			for X in X_vals:
				x,cdf_x = get_cdf_xy(list([el for el in m[X] ]))
				ax.plot(x,cdf_x,label="{} Drain pct={}".format(solution,X))
			ax.set_xlabel("Latency Change (new - old) (ms)")
			ax.set_ylabel("CDF of UGs")
			ax.grid(True)
			ax.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
			plt.savefig('figures/just_painter_resilience_to_congestion.pdf')
			# exit(0)

			## Given these capacities, solve all the solutions
			for popp in sas.popps:
				if deployment['link_capacities'][popp] < new_link_capacities[popp]:
					deployment['link_capacities'][popp] = new_link_capacities[popp]
			print("NLC : {}".format(deployment['link_capacities']))
			metrics['deployment'][random_iter] = deployment
			sas.update_deployment(deployment)
			ret = sas.compare_different_solutions(deployment_size=DPSIZE,n_run=1, verbose=True,
				 dont_update_deployment=True)

			ug_vols = sas.ug_to_vol

			for solution in soln_types:
				try:
					adv = ret['adv_solns'][solution][0]
				except:
					print("No solution for {}".format(solution))
					continue
				print("{} {} ".format(solution,np.round(adv,2)))
				m = assess_resilience_to_congestion(sas, adv, solution, X_vals)['metrics']
				metrics['resilience_to_congestion'][random_iter][solution] = m

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
						actual_perf = user_latencies[sas.ug_to_ind[ug]]
						metrics['popp_failures'][random_iter][solution].append((best_perf - actual_perf, ug_vols[ug]))
			pickle.dump(metrics, open(metrics_fn,'wb'))
	except:
		import traceback
		traceback.print_exc()
		exit(0)
	finally:
		if wm is not None:
			wm.stop_workers()
	

	for solution in soln_types:
		print("Solution {}".format(solution))
		for i in range(N_TO_SIM):
			adv = metrics['adv'][i][solution]
			deployment = metrics['deployment'][i]
			popps = sorted(list(set(deployment['popps'])))
			try:
				adv_summary(popps,adv)
				n_prefs_by_popp = np.sum(adv,axis=1)
				x,cdf_x = get_cdf_xy(n_prefs_by_popp)
				plt.plot(x,cdf_x,label=solution)
				
			except:
				continue
	plt.grid(True)
	plt.xlabel("Number of Prefixes Being Advertised via Ingress")
	plt.ylabel("CDF of PoPPs")
	plt.legend()
	plt.savefig("n_prefs_by_popp_solutions.pdf")
	plt.clf(); plt.close()

	f,ax=plt.subplots(3,1)
	f.set_size_inches(6,17)


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


			m = metrics['resilience_to_congestion']
			for X in X_vals:
				x,cdf_x = get_cdf_xy(list([el for ri in range(N_TO_SIM) for el in m[ri][solution][X] ]))
				ax[2].plot(x,cdf_x,label="{} Drain pct={}".format(solution,X))


		except:
			import traceback
			traceback.print_exc()
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

	ax[2].set_xlabel("Latency Change (new - old) (ms)")
	ax[2].set_ylabel("CDF of UGs")
	ax[2].grid(True)
	ax[2].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	# ax[2].set_xlim([0,150])
	ax[2].legend(fontsize=8)

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
			# adv_summary(sas.popps, advertisement)
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
			# if inundated and i==0:
			# 	recent_iter = sas.all_rb_calls_results[sas.popp_to_ind[popp]][-1][0]
			# 	these_rb_calls = [call for call in sas.all_rb_calls_results[sas.popp_to_ind[popp]] if
			# 		call[0] == recent_iter]
			# 	print("{} recent grad calls".format(len(these_rb_calls)))
			# 	recent_rb = these_rb_calls[-30:]
			# 	recent_lb = sas.all_lb_calls_results[-1]
			# 	# iter poppi popp prefix rbgrad lbgrad
			# 	recent_rb = [(i,poppi,sas.popps[poppi],prefi,round(rbgrad,2),round(recent_lb[poppi,prefi],2),
			# 		advertisement[poppi,prefi]) for i,poppi,prefi,rbgrad in 
			# 		recent_rb]
			# 	print("Popp {} fail causes inundation, recent resilience gradient calls were : {}".format(
			# 		popp, recent_rb))


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
