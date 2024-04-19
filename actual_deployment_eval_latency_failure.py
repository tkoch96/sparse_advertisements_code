from constants import *
from helpers import *
from wrapper_eval import *
from solve_lp_assignment import *

import pickle, numpy as np, matplotlib.pyplot as plt, copy, itertools, time
from sparse_advertisements_v3 import *
from eval_latency_failure import calc_pct_volume_within_latency

def evaluate_all_metrics(dpsize, port, save_run_dir=None, **kwargs):
	# for each deployment, get different advertisement strategies
	# look at latency under popp and link failures compared to optimal
	# sparse should be simultaneously better in both normal and failure scenarios

	# np.random.seed(31413)
	metrics = {}

	wm = None
	sas = None

	performance_metrics_fn = kwargs.get('use_performance_metrics_fn', global_performance_metrics_fn(dpsize))
	# soln_types = ['sparse', 'painter', 'anycast', 'one_per_pop', 'one_per_peering']
	# soln_types = ['anycast', 'one_per_peering']
	# soln_types = ['anycast', 'painter']
	soln_types = ['sparse', 'anycast', 'painter', 'one_per_pop']

	N_TO_SIM = kwargs.get('nsim',1)
	if N_TO_SIM > 1:
		assert save_run_dir is None


	metrics = copy.deepcopy(default_metrics)
	if os.path.exists(performance_metrics_fn):
		metrics = pickle.load(open(performance_metrics_fn,'rb'))
	for k in list(metrics):
		if k not in default_metrics:
			del metrics[k]
	for k in default_metrics:
		if k not in metrics:
			print(k)
			metrics[k] = copy.deepcopy(default_metrics[k])
		for i in range(N_TO_SIM):
			for k in metrics:
				if i not in metrics[k]:# and i in default_metrics[k]:
					print("{} {}".format(k, i))
					metrics[k][i] = copy.deepcopy(default_metrics[k][0])

	## Add keys unique to the actual deployment
	try:
		metrics['deployment_by_solution']
	except KeyError:
		metrics['deployment_by_solution'] = {i:{solution: None for solution in soln_types} for i in range(N_TO_SIM)}

	try:
		for random_iter in range(N_TO_SIM):
			try:
				if save_run_dir is not None:
					raise TypeError
				metrics['compare_rets'][random_iter]['n_advs']
				continue
			except TypeError:
				pass
			print("-----Deployment number = {} -------".format(random_iter))
			if save_run_dir is not None:
				print("Loading from hotstart dir")
				save_state = pickle.load(open(os.path.join(RUN_DIR, save_run_dir, 'state-0.pkl'), 'rb'))
				deployment = save_state['deployment']
				deployment['port'] = port
				# deployment['link_capacities'] = save_state['ug_modified_deployment']['link_capacities']
			else:
				deployment = get_random_deployment(dpsize, **kwargs)
				deployment['port'] = port
			metrics['deployment'][random_iter] = deployment

			n_prefixes = kwargs.get('n_prefixes', deployment_to_prefixes(deployment))
	
			sas = Sparse_Advertisement_Eval(deployment, verbose=True,
				lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
				using_resilience_benefit=True, gamma=gamma, n_prefixes=n_prefixes, save_run_dir=save_run_dir)

			metrics['settings'][random_iter] = sas.get_init_kwa()
			if wm is None:
				wm = Worker_Manager(sas.get_init_kwa(), sas.output_deployment())
				wm.start_workers()
			sas.set_worker_manager(wm)
			sas.update_deployment(sas.output_deployment())
			sas.get_realworld_measure_wrapper()
			### Solve the problem for each type of solution (sparse, painter, etc...)
			ret = sas.compare_different_solutions(n_run=1, verbose=True,
				 dont_update_deployment=True, soln_types=soln_types, **kwargs)
			metrics['compare_rets'][random_iter] = ret
			ug_vols = sas.ug_to_vol
			metrics['ug_to_vol'][random_iter] = sas.ug_vols
			metrics['best_latencies'][random_iter] = copy.copy(sas.best_lats_by_ug)
			lp_by_solution = {}
			for solution in soln_types:
				try:
					adv = ret['adv_solns'][solution][0]
				except:
					print("No solution for {}".format(solution))
					continue
				print(solution)

				if metrics['deployment_by_solution'][random_iter][solution] is None:
					print("Storing deployment for {}".format(solution))
					metrics['deployment_by_solution'][random_iter][solution] = sas.output_specific_deployment(solution)
				lp_soln = sas.solve_lp_with_failure_catch_actual_deployment(adv, solution)
				lp_by_solution[solution] = lp_soln
				pre_lats_by_ug = lp_soln['lats_by_ug']
				print(pre_lats_by_ug)

				metrics['adv'][random_iter][solution] = adv
				metrics['latencies'][random_iter][solution] = pre_lats_by_ug
			pickle.dump(lp_by_solution, open('lp_by_solution_pre.pkl','wb'))
			pickle.dump(metrics, open(performance_metrics_fn,'wb'))

	except:
		import traceback
		traceback.print_exc()
	print(metrics.keys())
	for solution in soln_types:
		adv = metrics['adv'][0][solution]
		this_solution_deployment = metrics['deployment_by_solution'][0][solution]
		print("{} -- {}".format(solution, np.sum(adv, axis=0)))
		if solution == 'sparse':
			first_adv = adv[:,0]
			not_on = np.where(first_adv==0)[0]
			print("Not on : {}".format(list([this_solution_deployment['popps'][poppi] for poppi in not_on])))


	RECALC_PCT_VOL_IN_LAT_MULTIPLIERS = False
	try:
		for random_iter in range(N_TO_SIM):
			break
			k_of_interest = 'pct_volume_within_latency'
			havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)
			if RECALC_PCT_VOL_IN_LAT_MULTIPLIERS or havent_calced_everything:
				if sas is None:
					deployment = metrics['deployment'][random_iter]

					n_prefixes = kwargs.get('n_prefixes', deployment_to_prefixes(deployment))
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
				ug_vols = sas.ug_to_vol
				for solution in soln_types:
					adv = metrics['adv'][random_iter][solution]
					if len(adv) == 0:
						print("No solution for {}".format(solution))
						continue
					print("Assessing pct volume within latency for {}".format(solution))
					m = calc_pct_volume_within_latency(sas, adv)
					metrics['pct_volume_within_latency'][random_iter][solution] = m

					
				pickle.dump(metrics, open(performance_metrics_fn,'wb'))
	except:
		import traceback
		traceback.print_exc()

	RECALC_FAILURE_METRICS = False
	try:
		for random_iter in range(N_TO_SIM):
			break
			k_of_interest = 'popp_failures_latency_optimal'
			havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)

			if RECALC_FAILURE_METRICS or havent_calced_everything:
				if sas is None:
					deployment = metrics['deployment'][random_iter]

					n_prefixes = kwargs.get('n_prefixes', deployment_to_prefixes(deployment))
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
					if not RECALC_FAILURE_METRICS:
						if metrics[k_of_interest][random_iter][solution]  != \
							default_metrics[k_of_interest][0][solution]:
							print("Already calced {}".format(solution))
							continue
					print(solution)
					adv = metrics['adv'][random_iter][solution]
					if len(adv) == 0:
						print("No solution for {}".format(solution))
						continue
						
					ret = assess_failure_resilience_real_deployment(sas, adv, which='popps')
					metrics['popp_failures_congestion'][random_iter][solution] = ret['mutable']['congestion_delta']
					metrics['popp_failures_latency_optimal'][random_iter][solution] = ret['mutable']['latency_delta_optimal']
					metrics['popp_failures_latency_optimal_specific'][random_iter][solution] = ret['mutable']['latency_delta_specific']
					metrics['popp_failures_latency_before'][random_iter][solution] = ret['mutable']['latency_delta_before']

					metrics['popp_failures_sticky_congestion'][random_iter][solution] = ret['sticky']['congestion_delta']
					metrics['popp_failures_sticky_latency_optimal'][random_iter][solution] = ret['sticky']['latency_delta_optimal']
					metrics['popp_failures_sticky_latency_optimal_specific'][random_iter][solution] = ret['sticky']['latency_delta_specific']
					metrics['popp_failures_sticky_latency_before'][random_iter][solution] = ret['sticky']['latency_delta_before']

					ret = assess_failure_resilience_real_deployment(sas, adv, which='pops')
					metrics['pop_failures_congestion'][random_iter][solution] = ret['mutable']['congestion_delta']
					metrics['pop_failures_latency_optimal'][random_iter][solution] = ret['mutable']['latency_delta_optimal']
					metrics['pop_failures_latency_optimal_specific'][random_iter][solution] = ret['mutable']['latency_delta_specific']
					metrics['pop_failures_latency_before'][random_iter][solution] = ret['mutable']['latency_delta_before']

					metrics['pop_failures_sticky_congestion'][random_iter][solution] = ret['sticky']['congestion_delta']
					metrics['pop_failures_sticky_latency_optimal'][random_iter][solution] = ret['sticky']['latency_delta_optimal']
					metrics['pop_failures_sticky_latency_optimal_specific'][random_iter][solution] = ret['sticky']['latency_delta_specific']
					metrics['pop_failures_sticky_latency_before'][random_iter][solution] = ret['sticky']['latency_delta_before']



					pickle.dump(metrics, open(performance_metrics_fn,'wb'))

	except:
		import traceback
		traceback.print_exc()
	finally:
		if wm is not None:
			wm.stop_workers()

	################################
	### PLOTTING
	################################
	
	i=0

	ADV_VIZ_I = {}
	for solution in soln_types:
		ADV_VIZ_I[solution] = i;i+=1

	LATENCY_I = i;i+=1
	PCT_VOL_WITHIN_LATENCY_I = i; i+= 1
	## Mutable
	POPP_FAILURE_LATENCY_OPTIMAL_I = i;i+=1
	POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I = i;i+=1
	POPP_FAILURE_LATENCY_BEFORE_I = i;i+=1
	POPP_FAILURE_CONGESTION_I = i;i+=1
	POP_FAILURE_LATENCY_OPTIMAL_I = i;i+=1
	POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I = i;i+=1
	POP_FAILURE_LATENCY_BEFORE_I = i;i+=1
	POP_FAILURE_CONGESTION_I = i;i+=1

	POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I = i;i+=1
	POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I = i;i+=1
	POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I = i;i+=1
	POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I = i;i+=1

	## Sticky
	POPP_FAILURE_STICKY_LATENCY_OPTIMAL_I = i;i+=1
	POPP_FAILURE_STICKY_LATENCY_OPTIMAL_SPECIFIC_I = i;i+=1
	POPP_FAILURE_STICKY_LATENCY_BEFORE_I = i;i+=1
	POPP_FAILURE_STICKY_CONGESTION_I = i;i+=1
	POP_FAILURE_STICKY_LATENCY_OPTIMAL_I = i;i+=1
	POP_FAILURE_STICKY_LATENCY_OPTIMAL_SPECIFIC_I = i;i+=1
	POP_FAILURE_STICKY_LATENCY_BEFORE_I = i;i+=1
	POP_FAILURE_STICKY_CONGESTION_I = i;i+=1

	FLASH_CROWD_SINGLE_X_I = i;i+=1
	FLASH_CROWD_LATENCY_VARY_X_I = i;i+=1
	FLASH_CROWD_CONGESTION_VARY_X_I = i;i+=1
	FLASH_CROWD_PREFIX_I = i;i+=1
	FLASH_CROWD_LIMITING_FACTOR_LATENCY_I = i;i+=1
	FLASH_CROWD_LIMITING_FACTOR_CONGESTION_I = i;i+=1
	VOLUME_GROWTH_I = i;i+=1

	n_subs = i
	f,ax=plt.subplots(n_subs,1)
	f.set_size_inches(6,4*n_subs)

	## we plot the performance changes for a single flash crowd volume increase
	SIM_INDS_TO_PLOT = list(range(N_TO_SIM))

	def get_failure_metric_arr(k, solution, verb=False):
		ret = []
		avg_ret = []
		mind,maxd = np.inf,-1*np.inf
		all_vol = 0
		actually_all_vol = 0
		vol_congested = 0
		vol_best_case_congested = 0
		for ri in SIM_INDS_TO_PLOT:
			these_metrics = metrics[k][ri][solution]
			ugs={}

			summaries_by_element = {}

			for fields in these_metrics:
				if len(fields) == 6:
					diff,vol,ug,element,perf1,perf2 = fields
				else:
					diff,vol,ug,element,perf1,perf2,_ = fields
				ugs[ug] = None
				actually_all_vol += vol
				if perf1 == NO_ROUTE_LATENCY: ## the best-case scenario is congested
					vol_best_case_congested += vol
					perf2 = perf1
				else:
					if perf2 != NO_ROUTE_LATENCY and perf1 != NO_ROUTE_LATENCY:
						avg_ret.append((perf1-perf2,vol))
					if perf2 == NO_ROUTE_LATENCY:
						vol_congested += vol
						perf2=perf2*100
					all_vol += vol
					if diff > maxd:
						maxd=diff
					if diff < mind:
						mind=diff
				
				ret.append((perf1-perf2, vol))

			# for element,vol in summaries_by_element.items():
			# 	print("{} not perfect, {} pct of calls, vol: {}".format(element, n_calls[element]*100.0/ sum(list(n_calls.values())), vol))
		x=np.linspace(mind,maxd,num=200)
		if vol_congested > 0:
			x[0] = -1*100*NO_ROUTE_LATENCY

		avg_latency_difference = np.average([el[0] for el in avg_ret], weights=[el[1] for el in avg_ret])
		print("Average latency difference {},{}: {}".format(solution, k, avg_latency_difference))
		print("{} pct. volume congested".format(round(100 * vol_congested / actually_all_vol, 2)))
		print("{} pct. optimally congested, all volume: {}".format(round(100 * vol_best_case_congested / actually_all_vol, 2), actually_all_vol))

		return ret, x, {'avg_latency_difference': avg_latency_difference, 
			'frac_vol_congested': vol_congested / all_vol, 
			'frac_vol_bestcase_congested': vol_best_case_congested / actually_all_vol}

	#### Plotting everything
	for k in list(metrics):
		if 'latency' in k or 'latencies' in k:
			metrics['stats_' + k] = {}

	for solution in soln_types:
		print(solution)
		if len(metrics['adv'][0][solution]) > 0:
			ax[ADV_VIZ_I[solution]].imshow(metrics['adv'][0][solution])
		try:
			#### Changes in latency
			diffs = []
			wts = []
			for random_iter in SIM_INDS_TO_PLOT:
				for i in range(len(metrics['best_latencies'][random_iter])):
					diffs.append(metrics['best_latencies'][random_iter][i] - metrics['latencies'][random_iter][solution][i])
					wts.append(metrics['ug_to_vol'][random_iter][i])

			x,cdf_x = get_cdf_xy(list(zip(diffs,wts)), weighted=True)
			ax[LATENCY_I].plot(x,cdf_x,label=solution)
			avg_latency_diff = np.average(diffs, weights=wts)
			print("Average latency compared to optimal : {}".format(avg_latency_diff))
			metrics['stats_best_latencies'][solution] = avg_latency_diff

			#### PCT of Volume within a Certainty Latency Threshold
			for random_iter in SIM_INDS_TO_PLOT:
				m = metrics['pct_volume_within_latency'][random_iter][solution]
				ax[PCT_VOL_WITHIN_LATENCY_I].plot(m['latencies'], m['volume_fractions'], label="{} -- Sim {}".format(solution, random_iter))
			
			#### Resilience to PoP and PoPP failures
			### MUTABLE
			## popp failures (latency)
			all_differences, x, stats = get_failure_metric_arr('popp_failures_latency_optimal', solution)
			metrics['stats_' + 'popp_failures_latency_optimal'][solution] = stats
			x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POPP_FAILURE_LATENCY_OPTIMAL_I].plot(x,cdf_x,label=solution)
			all_differences, x, stats = get_failure_metric_arr('popp_failures_latency_optimal_specific', solution)
			metrics['stats_' + 'popp_failures_latency_optimal_specific'][solution] = stats
			x, cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)

			# all_differences, x, stats = get_failure_metric_arr('popp_failures_high_cap_latency_optimal', solution)
			# metrics['stats_' + 'popp_failures_high_cap_latency_optimal'][solution] = stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# ax[POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I].plot(x,cdf_x,label=solution)
			# all_differences, x, stats = get_failure_metric_arr('popp_failures_high_cap_latency_optimal_specific', solution, verb=solution=='sparse')
			# metrics['stats_' + 'popp_failures_high_cap_latency_optimal_specific'][solution] = stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# ax[POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)

			all_differences, x, stats = get_failure_metric_arr('popp_failures_latency_before', solution)
			metrics['stats_' + 'popp_failures_latency_before'][solution] = stats
			x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POPP_FAILURE_LATENCY_BEFORE_I].plot(x,cdf_x,label=solution)
			## popp failures (congestion)
			all_differences = [metrics['popp_failures_congestion'][ri][solution][i] for ri in SIM_INDS_TO_PLOT for 
				i in range(len(metrics['popp_failures_congestion'][ri][solution]))]
			print("{} links had any congestion".format(sum(1 for el in all_differences if el != 0)))
			x,cdf_x = get_cdf_xy(all_differences,weighted=False)
			ax[POPP_FAILURE_CONGESTION_I].plot(x,cdf_x,label=solution)
			
			## pop failures (latency)
			all_differences, x, stats = get_failure_metric_arr('pop_failures_latency_optimal', solution)
			metrics['stats_' + 'pop_failures_latency_optimal'][solution] = stats
			x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POP_FAILURE_LATENCY_OPTIMAL_I].plot(x,cdf_x,label=solution)
			all_differences, x, stats = get_failure_metric_arr('pop_failures_latency_optimal_specific', solution)
			metrics['stats_' + 'pop_failures_latency_optimal_specific'][solution] = stats
			x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)

			all_differences, x, stats = get_failure_metric_arr('pop_failures_high_cap_latency_optimal', solution)
			metrics['stats_' + 'pop_failures_high_cap_latency_optimal'][solution] = stats
			x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I].plot(x,cdf_x,label=solution)
			all_differences, x, stats = get_failure_metric_arr('pop_failures_high_cap_latency_optimal_specific', solution)
			metrics['stats_' + 'pop_failures_high_cap_latency_optimal_specific'][solution] = stats
			x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)

			## pop failures (latency)
			all_differences, x, stats = get_failure_metric_arr('pop_failures_latency_before', solution)
			metrics['stats_' + 'pop_failures_latency_before'][solution] = stats
			x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POP_FAILURE_LATENCY_BEFORE_I].plot(x,cdf_x,label=solution)
			## pop failures (congestion)
			all_differences = [metrics['pop_failures_congestion'][ri][solution][i] for ri in SIM_INDS_TO_PLOT for 
				i in range(len(metrics['pop_failures_congestion'][ri][solution]))]
			x,cdf_x = get_cdf_xy(all_differences,weighted=False)
			ax[POP_FAILURE_CONGESTION_I].plot(x,cdf_x,label=solution)


			#### Resilience to flash crowds
			m = metrics['resilience_to_congestion']
			cm = metrics['fraction_congested_volume']
			

			## Want to track where latency and congestion start to impact performance
			latency_limits_over_Y = []
			congestion_limits_over_Y = []

			for Y_val in Y_vals:
				latency_delta_meds = []			
				congestion_meds = []
				for X_val in X_vals:
					all_lats, all_vols, all_congestions = [], [], []
					for ri in SIM_INDS_TO_PLOT:
						for latdeltaset, latoldset, latnewset, volset in m[ri][solution][Y_val][X_val]:
							for deltalat, oldlat,newlat, vol in zip(latdeltaset,latoldset,latnewset,volset):
								if oldlat == NO_ROUTE_LATENCY or newlat == NO_ROUTE_LATENCY: continue
								all_lats.append(deltalat)
								all_vols.append(vol)
						for congestion in cm[ri][solution][Y_val][X_val]:
							all_congestions.append(congestion)
					if X_val == single_X_of_interest and Y_val == single_Y_of_interest:
						## Plot CDF for this specific X val and Y val
						x,cdf_x = get_cdf_xy(list(zip(all_lats,all_vols)), weighted=True)
						ax[FLASH_CROWD_SINGLE_X_I].plot(x,cdf_x,label="{} Metro increase pct. = {}".format(solution,single_X_of_interest))
						
					lat_med = np.average(all_lats, weights=all_vols)
					con_med = np.average(all_congestions)

					latency_delta_meds.append(lat_med)
					congestion_meds.append(con_med)
				latency_delta_meds = np.array(latency_delta_meds)
				congestion_meds = np.array(congestion_meds)
				if Y_val == single_Y_of_interest:
					## Plot variance over X for this specific Y val
					ax[FLASH_CROWD_LATENCY_VARY_X_I].plot(X_vals, latency_delta_meds, label=solution)
					ax[FLASH_CROWD_CONGESTION_VARY_X_I].plot(X_vals, congestion_meds, label=solution)
				try:
					latency_limits_over_Y.append(X_vals[np.where(latency_delta_meds > 1)[0][0]])
				except IndexError:
					latency_limits_over_Y.append(X_vals[-1])
				try:
					congestion_limits_over_Y.append(X_vals[np.where(congestion_meds > 0)[0][0]])
				except IndexError:
					congestion_limits_over_Y.append(X_vals[-1])
			ax[FLASH_CROWD_LIMITING_FACTOR_LATENCY_I].plot(Y_vals, latency_limits_over_Y, label=solution)
			ax[FLASH_CROWD_LIMITING_FACTOR_CONGESTION_I].plot(Y_vals, congestion_limits_over_Y, label=solution)


			
			##### Number of advertisements changed with flash crowds
			m = metrics['prefix_withdrawals']
			all_withdrawal_ns = []
			for ri in SIM_INDS_TO_PLOT:
				for metro_set in m[ri][solution][single_Y_of_interest][single_X_of_interest]:
					total_withdrawals = sum(len(el) for el in metro_set)
					all_withdrawal_ns.append(total_withdrawals)

			x,cdf_x = get_cdf_xy(all_withdrawal_ns)
			ax[FLASH_CROWD_PREFIX_I].plot(x,cdf_x,label="{} Metro increase pct. = {}".format(solution,single_X_of_interest))


			#### Ability to grow with volume
			all_multipliers = []
			m = metrics['volume_multipliers']
			for ri in SIM_INDS_TO_PLOT:
				try:
					all_multipliers.append(m[ri][solution]['volume_multipliers'][-1])
				except IndexError: # no ability at all
					all_multipliers.append(1)
			x,cdf_x = get_cdf_xy(all_multipliers)
			ax[VOLUME_GROWTH_I].plot(x,cdf_x,label=solution)

		except:
			import traceback
			traceback.print_exc()
			continue

	ax[LATENCY_I].legend(fontsize=8)
	ax[LATENCY_I].grid(True)
	ax[LATENCY_I].set_xlabel("Best - Actual Latency (ms)")
	ax[LATENCY_I].set_ylabel("CDF of Traffic")
	ax[LATENCY_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[LATENCY_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[PCT_VOL_WITHIN_LATENCY_I].legend(fontsize=8)
	ax[PCT_VOL_WITHIN_LATENCY_I].grid(True)
	ax[PCT_VOL_WITHIN_LATENCY_I].set_xlabel("Latency (ms)")
	ax[PCT_VOL_WITHIN_LATENCY_I].set_ylabel("Fraction Ingresses Reachable")
	ax[PCT_VOL_WITHIN_LATENCY_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	### FAILURE PLOTS
	## MUTABLE
	ax[POPP_FAILURE_LATENCY_OPTIMAL_I].legend(fontsize=8)
	ax[POPP_FAILURE_LATENCY_OPTIMAL_I].grid(True)
	ax[POPP_FAILURE_LATENCY_OPTIMAL_I].set_xlabel("Latency Change Under Single-Link Failure (best - actual) (ms)")
	ax[POPP_FAILURE_LATENCY_OPTIMAL_I].set_ylabel("CDF of Traffic,Links")
	ax[POPP_FAILURE_LATENCY_OPTIMAL_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POPP_FAILURE_LATENCY_OPTIMAL_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].legend(fontsize=8)
	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].grid(True)
	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_xlabel("Latency Change Under Single-Link Failure (best - actual) (ms)")
	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_ylabel("CDF of Affected Traffic,Links")
	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I].legend(fontsize=8)
	ax[POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I].grid(True)
	ax[POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I].set_xlabel("HIGH CAP Latency Change Under Single-Link Failure (best - actual) (ms)")
	ax[POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I].set_ylabel("CDF of Traffic,Links")
	ax[POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I].legend(fontsize=8)
	ax[POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I].grid(True)
	ax[POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I].set_xlabel("HIGH CAP Latency Change Under Single-Link Failure (best - actual) (ms)")
	ax[POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I].set_ylabel("CDF of Affected Traffic,Links")
	ax[POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POPP_FAILURE_LATENCY_BEFORE_I].legend(fontsize=8)
	ax[POPP_FAILURE_LATENCY_BEFORE_I].grid(True)
	ax[POPP_FAILURE_LATENCY_BEFORE_I].set_xlabel("Latency Change Under Single-Link Failure (old - new) (ms)")
	ax[POPP_FAILURE_LATENCY_BEFORE_I].set_ylabel("CDF of Traffic,Links")
	ax[POPP_FAILURE_LATENCY_BEFORE_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POPP_FAILURE_LATENCY_BEFORE_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POPP_FAILURE_CONGESTION_I].legend(fontsize=8)
	ax[POPP_FAILURE_CONGESTION_I].grid(True)
	ax[POPP_FAILURE_CONGESTION_I].set_xlabel("Fraction Congested Volume Single-Link Failure (best - actual) (ms)")
	ax[POPP_FAILURE_CONGESTION_I].set_ylabel("CDF of Traffic,Links")
	ax[POPP_FAILURE_CONGESTION_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	ax[POP_FAILURE_LATENCY_OPTIMAL_I].legend(fontsize=8)
	ax[POP_FAILURE_LATENCY_OPTIMAL_I].grid(True)
	ax[POP_FAILURE_LATENCY_OPTIMAL_I].set_xlabel("Latency Change Under Single-PoP Failure (best - actual) (ms)")
	ax[POP_FAILURE_LATENCY_OPTIMAL_I].set_ylabel("CDF of Traffic,PoPs")
	ax[POP_FAILURE_LATENCY_OPTIMAL_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POP_FAILURE_LATENCY_OPTIMAL_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].legend(fontsize=8)
	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].grid(True)
	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_xlabel("Latency Change Under Single-PoP Failure (best - actual) (ms)")
	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_ylabel("CDF of Affected Traffic,PoPs")
	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I].legend(fontsize=8)
	ax[POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I].grid(True)
	ax[POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I].set_xlabel("HIGH CAP Latency Change Under Single-PoP Failure (best - actual) (ms)")
	ax[POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I].set_ylabel("CDF of Traffic,PoPs")
	ax[POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I].legend(fontsize=8)
	ax[POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I].grid(True)
	ax[POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I].set_xlabel("HIGH CAP Latency Change Under Single-PoP Failure (best - actual) (ms)")
	ax[POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I].set_ylabel("CDF of Affected Traffic,PoPs")
	ax[POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POP_FAILURE_LATENCY_BEFORE_I].legend(fontsize=8)
	ax[POP_FAILURE_LATENCY_BEFORE_I].grid(True)
	ax[POP_FAILURE_LATENCY_BEFORE_I].set_xlabel("Latency Change Under Single-PoP Failure (old - new) (ms)")
	ax[POP_FAILURE_LATENCY_BEFORE_I].set_ylabel("CDF of Traffic,PoPs")
	ax[POP_FAILURE_LATENCY_BEFORE_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POP_FAILURE_LATENCY_BEFORE_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	## STICKY
	ax[POPP_FAILURE_STICKY_LATENCY_OPTIMAL_I].legend(fontsize=8)
	ax[POPP_FAILURE_STICKY_LATENCY_OPTIMAL_I].grid(True)
	ax[POPP_FAILURE_STICKY_LATENCY_OPTIMAL_I].set_xlabel("Latency Change Under STICKY Single-Link Failure (best - actual) (ms)")
	ax[POPP_FAILURE_STICKY_LATENCY_OPTIMAL_I].set_ylabel("CDF of Traffic,Links")
	ax[POPP_FAILURE_STICKY_LATENCY_OPTIMAL_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POPP_FAILURE_STICKY_LATENCY_OPTIMAL_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POPP_FAILURE_STICKY_LATENCY_OPTIMAL_SPECIFIC_I].legend(fontsize=8)
	ax[POPP_FAILURE_STICKY_LATENCY_OPTIMAL_SPECIFIC_I].grid(True)
	ax[POPP_FAILURE_STICKY_LATENCY_OPTIMAL_SPECIFIC_I].set_xlabel("Latency Change Under STICKY Single-Link Failure (best - actual) (ms)")
	ax[POPP_FAILURE_STICKY_LATENCY_OPTIMAL_SPECIFIC_I].set_ylabel("CDF of Affected Traffic,Links")
	ax[POPP_FAILURE_STICKY_LATENCY_OPTIMAL_SPECIFIC_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POPP_FAILURE_STICKY_LATENCY_OPTIMAL_SPECIFIC_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POPP_FAILURE_STICKY_LATENCY_BEFORE_I].legend(fontsize=8)
	ax[POPP_FAILURE_STICKY_LATENCY_BEFORE_I].grid(True)
	ax[POPP_FAILURE_STICKY_LATENCY_BEFORE_I].set_xlabel("Latency Change Under STICKY Single-Link Failure (old - new) (ms)")
	ax[POPP_FAILURE_STICKY_LATENCY_BEFORE_I].set_ylabel("CDF of Traffic,Links")
	ax[POPP_FAILURE_STICKY_LATENCY_BEFORE_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POPP_FAILURE_STICKY_LATENCY_BEFORE_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POPP_FAILURE_STICKY_CONGESTION_I].legend(fontsize=8)
	ax[POPP_FAILURE_STICKY_CONGESTION_I].grid(True)
	ax[POPP_FAILURE_STICKY_CONGESTION_I].set_xlabel("Fraction Congested Volume STICKY Single-Link Failure (best - actual) (ms)")
	ax[POPP_FAILURE_STICKY_CONGESTION_I].set_ylabel("CDF of Traffic,Links")
	ax[POPP_FAILURE_STICKY_CONGESTION_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	ax[POP_FAILURE_STICKY_LATENCY_OPTIMAL_I].legend(fontsize=8)
	ax[POP_FAILURE_STICKY_LATENCY_OPTIMAL_I].grid(True)
	ax[POP_FAILURE_STICKY_LATENCY_OPTIMAL_I].set_xlabel("Latency Change Under STICKY Single-PoP Failure (best - actual) (ms)")
	ax[POP_FAILURE_STICKY_LATENCY_OPTIMAL_I].set_ylabel("CDF of Traffic,PoPs")
	ax[POP_FAILURE_STICKY_LATENCY_OPTIMAL_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POP_FAILURE_STICKY_LATENCY_OPTIMAL_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POP_FAILURE_STICKY_LATENCY_OPTIMAL_SPECIFIC_I].legend(fontsize=8)
	ax[POP_FAILURE_STICKY_LATENCY_OPTIMAL_SPECIFIC_I].grid(True)
	ax[POP_FAILURE_STICKY_LATENCY_OPTIMAL_SPECIFIC_I].set_xlabel("Latency Change Under STICKY Single-PoP Failure (best - actual) (ms)")
	ax[POP_FAILURE_STICKY_LATENCY_OPTIMAL_SPECIFIC_I].set_ylabel("CDF of Affected Traffic,PoPs")
	ax[POP_FAILURE_STICKY_LATENCY_OPTIMAL_SPECIFIC_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POP_FAILURE_STICKY_LATENCY_OPTIMAL_SPECIFIC_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POP_FAILURE_STICKY_LATENCY_BEFORE_I].legend(fontsize=8)
	ax[POP_FAILURE_STICKY_LATENCY_BEFORE_I].grid(True)
	ax[POP_FAILURE_STICKY_LATENCY_BEFORE_I].set_xlabel("Latency Change Under STICKY Single-PoP Failure (old - new) (ms)")
	ax[POP_FAILURE_STICKY_LATENCY_BEFORE_I].set_ylabel("CDF of Traffic,PoPs")
	ax[POP_FAILURE_STICKY_LATENCY_BEFORE_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POP_FAILURE_STICKY_LATENCY_BEFORE_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])


	### FLASH CROWDS

	ax[POP_FAILURE_CONGESTION_I].legend(fontsize=8)
	ax[POP_FAILURE_CONGESTION_I].grid(True)
	ax[POP_FAILURE_CONGESTION_I].set_xlabel("Fraction Congested Volume Single-PoP Failure (best - actual) (ms)")
	ax[POP_FAILURE_CONGESTION_I].set_ylabel("CDF of Traffic,PoPs")
	ax[POP_FAILURE_CONGESTION_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	ax[FLASH_CROWD_SINGLE_X_I].set_xlabel("Latency Change under Flash Crowd (new - old) (ms)")
	ax[FLASH_CROWD_SINGLE_X_I].set_ylabel("CDF of Traffic")
	ax[FLASH_CROWD_SINGLE_X_I].grid(True)
	ax[FLASH_CROWD_SINGLE_X_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[FLASH_CROWD_SINGLE_X_I].legend(fontsize=8)

	ax[FLASH_CROWD_PREFIX_I].set_xlabel("Number of Prefix Withdrawals for Flash Crowd")
	ax[FLASH_CROWD_PREFIX_I].set_ylabel("CDF of Metros")
	ax[FLASH_CROWD_PREFIX_I].grid(True)
	ax[FLASH_CROWD_PREFIX_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[FLASH_CROWD_PREFIX_I].legend(fontsize=8)

	ax[FLASH_CROWD_LATENCY_VARY_X_I].set_xlabel("Increase in Traffic per Metro for Flash Crowd (pct.)")
	ax[FLASH_CROWD_LATENCY_VARY_X_I].set_ylabel("Average Latency Change \nunder Flash Crowd (new - old) (ms)")
	ax[FLASH_CROWD_LATENCY_VARY_X_I].grid(True)
	ax[FLASH_CROWD_LATENCY_VARY_X_I].legend(fontsize=8)

	ax[FLASH_CROWD_CONGESTION_VARY_X_I].set_xlabel("Increase in Traffic per Metro for Flash Crowd (pct.)")
	ax[FLASH_CROWD_CONGESTION_VARY_X_I].set_ylabel("Average Delta Fraction Congested \n Traffic under Flash Crowd (new - old)")
	ax[FLASH_CROWD_CONGESTION_VARY_X_I].grid(True)
	ax[FLASH_CROWD_CONGESTION_VARY_X_I].legend(fontsize=8)

	ax[FLASH_CROWD_LIMITING_FACTOR_LATENCY_I].set_xlabel("Link Capacity Overprovisioning (pct)")
	ax[FLASH_CROWD_LIMITING_FACTOR_LATENCY_I].set_ylabel("Flash Crowd Latency Resilience")
	ax[FLASH_CROWD_LIMITING_FACTOR_LATENCY_I].grid(True)
	ax[FLASH_CROWD_LIMITING_FACTOR_LATENCY_I].legend(fontsize=8)

	ax[FLASH_CROWD_LIMITING_FACTOR_CONGESTION_I].set_xlabel("Link Capacity Overprovisioning (pct)")
	ax[FLASH_CROWD_LIMITING_FACTOR_CONGESTION_I].set_ylabel("Flash Crowd Congestion Resilience")
	ax[FLASH_CROWD_LIMITING_FACTOR_CONGESTION_I].grid(True)
	ax[FLASH_CROWD_LIMITING_FACTOR_CONGESTION_I].legend(fontsize=8)

	

	ax[VOLUME_GROWTH_I].legend(fontsize=8)
	ax[VOLUME_GROWTH_I].grid(True)
	ax[VOLUME_GROWTH_I].set_xlabel("Volume Growth Capability")
	ax[VOLUME_GROWTH_I].set_ylabel("CDF of Simulations")
	ax[VOLUME_GROWTH_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	save_fig_fn = kwargs.get('save_fig_fn', "popp_latency_failure_comparison_{}.pdf".format(dpsize))

	save_fig(save_fig_fn)

	return metrics

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_run_dir", default=None)
	parser.add_argument("--dpsize", default=None, required=True)
	parser.add_argument("--port", default=None, required=True)
	args = parser.parse_args()

	np.random.seed(31415)
	if args.save_run_dir is not None:
		## we could specify an array of hotstart dirs otherwise, but that's a task for another day
		assert N_TO_SIM == 1
		evaluate_all_metrics(args.dpsize, int(args.port), save_run_dir=args.save_run_dir)
	else:
		evaluate_all_metrics(args.dpsize, int(args.port))
