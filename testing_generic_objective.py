import numpy as np, sys, glob, re, copy

from deployment_setup import get_random_deployment
from sparse_advertisements_v3 import Sparse_Advertisement_Solver, Sparse_Advertisement_Eval
from worker_comms import Worker_Manager
from wrapper_eval import *
from helpers import *
from constants import *
from paper_plotting_functions import *
from solve_lp_assignment import *


def compare_from_trained_sparse(run_dir, **kwargs):
	### train painter, etc.. (once, if needed) then conduct all the evals on the last advertisement
	run_dir = os.path.join(RUN_DIR, run_dir)
	all_small_stats = glob.glob(os.path.join(run_dir, 'small-stats-*.pkl'))
	max_small_stats = max([int(re.search('stats\-(.+)\.pkl', fn).group(1)) for fn in all_small_stats])

	max_small_stats_fn = os.path.join(run_dir, 'small-stats-{}.pkl'.format(max_small_stats))
	current_sparse_advertisement_rep = pickle.load(open(max_small_stats_fn, 'rb'))['optimization_advertisement_representation']

	large_save_state = pickle.load(open(os.path.join(run_dir, 'state-0.pkl'),'rb'))
	deployment = large_save_state['deployment']


	dpsize = sys.argv[1]
	port = int(sys.argv[2])
	deployment['port'] = port
	lambduh = .00001 ## unused more or less
	gamma = 2.0
	wm = None
	sas = None

	performance_metrics_fn = os.path.join(CACHE_DIR, 'full_sized_full_objective-{}.pkl'.format(dpsize))

	N_TO_SIM = 1

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

	soln_types = get_difference(global_soln_types, ['sparse'])

	try:
		for random_iter in range(N_TO_SIM):
				## just check to make sure that we've computed painter etc. once
			if len(metrics['adv'][random_iter]['painter']) > 0:
				continue
			print("-----Deployment number = {} -------".format(random_iter))
			metrics['deployment'][random_iter] = deployment
	
			n_prefixes = kwargs.get('n_prefixes', deployment_to_prefixes(deployment))
	
			sas = Sparse_Advertisement_Eval(deployment, verbose=True,
				lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
				using_resilience_benefit=True, gamma=gamma, n_prefixes=n_prefixes)

			metrics['settings'][random_iter] = sas.get_init_kwa()
			if wm is None:
				wm = Worker_Manager(sas.get_init_kwa(), deployment)
				wm.start_workers()
			sas.set_worker_manager(wm)
			sas.update_deployment(deployment)
			### Solve the problem for each type of solution (sparse, painter, etc...)
			ret = sas.compare_different_solutions(n_run=1, verbose=True,
				 dont_update_deployment=True, soln_types=soln_types, **kwargs)
			metrics['compare_rets'][random_iter] = ret
			ug_vols = sas.ug_to_vol
			metrics['ug_to_vol'][random_iter] = sas.ug_vols
			metrics['best_latencies'][random_iter] = copy.copy(sas.best_lats_by_ug)
			for solution in soln_types:
				try:
					adv = ret['adv_solns'][solution][0]
				except:
					print("No solution for {}".format(solution))
					continue
				pre_lats_by_ug = sas.solve_lp_with_failure_catch(adv)['lats_by_ug']

				metrics['adv'][random_iter][solution] = adv
				metrics['latencies'][random_iter][solution] = pre_lats_by_ug

			## populate the sparse adv

			metrics['adv'][random_iter]['sparse'] = np.zeros(metrics['adv'][random_iter]['painter'].shape)
			for popp, pref in current_sparse_advertisement_rep:
				metrics['adv'][random_iter]['sparse'][sas.popp_to_ind[popp], pref] = 1.0
			adv = metrics['adv'][random_iter]['sparse']
			pre_lats_by_ug = sas.solve_lp_with_failure_catch(adv)['lats_by_ug']
			metrics['latencies'][random_iter]['sparse'] = pre_lats_by_ug


			pickle.dump(metrics, open(performance_metrics_fn,'wb'))

	except:
		import traceback
		traceback.print_exc()

	soln_types = global_soln_types

	RECALC_FAILURE_METRICS = False
	try:
		changed=False
		for random_iter in range(N_TO_SIM):
			k_of_interest = 'popp_failures_latency_optimal_specific'
			havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)

			if RECALC_FAILURE_METRICS or havent_calced_everything:
				print("-----Failure calc for deployment number = {} -------".format(random_iter))
				if sas is None:
					deployment = metrics['deployment'][random_iter]
					deployment['port'] = port

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
					deployment['port'] = port
					sas.update_deployment(deployment)
				changed=True

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
					try:
						ret = assess_failure_resilience(sas, adv, which='popps')
						metrics['popp_failures_congestion'][random_iter][solution] = ret['mutable']['congestion_delta']
						metrics['popp_failures_latency_optimal'][random_iter][solution] = ret['mutable']['latency_delta_optimal']
						metrics['popp_failures_latency_optimal_specific'][random_iter][solution] = ret['mutable']['latency_delta_specific']
						metrics['popp_failures_latency_before'][random_iter][solution] = ret['mutable']['latency_delta_before']

						ret = assess_failure_resilience(sas, adv, which='pops')
						metrics['pop_failures_congestion'][random_iter][solution] = ret['mutable']['congestion_delta']
						metrics['pop_failures_latency_optimal'][random_iter][solution] = ret['mutable']['latency_delta_optimal']
						metrics['pop_failures_latency_optimal_specific'][random_iter][solution] = ret['mutable']['latency_delta_specific']
						metrics['pop_failures_latency_before'][random_iter][solution] = ret['mutable']['latency_delta_before']

					except:
						import traceback
						traceback.print_exc()
						continue


		if changed:
			pickle.dump(metrics, open(performance_metrics_fn,'wb'))

	except:
		import traceback
		traceback.print_exc()

	RECALC_DIURNAL = False
	diurnal_multipliers = [25,50,65,70,75,85,95,105,115,125,150]
	try:
		changed=False
		for random_iter in range(N_TO_SIM):
			k_of_interest = 'diurnal'
			havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)
			if RECALC_DIURNAL or havent_calced_everything:
				print("-----Diurnal calc for deployment number = {} -------".format(random_iter))
				if sas is None:
					deployment = metrics['deployment'][random_iter]
					deployment['port'] = port

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
					deployment['port'] = port
					sas.update_deployment(deployment)
				diurnal_deployments = get_diurnal_deployments(sas, diurnal_multipliers)
				for solution in soln_types:
					if not RECALC_DIURNAL:
						if metrics[k_of_interest][random_iter][solution]  != \
							default_metrics[k_of_interest][0][solution]:
							print("Already calced {}".format(solution))
							continue
					try:
						adv = metrics['adv'][random_iter][solution]
					except:
						print("No solution for {}".format(solution))
						continue
					try:
						print("Assessing diurnal effect for {}, sim number {}".format(solution,random_iter))
						## can reuse the flash crowd function since its the same
						## "X_vals ~ hours of day 0 - 23"
						## "Y_vals" ~ intensities
						metrics[k_of_interest][random_iter][solution] = assess_resilience_to_flash_crowds_mp(sas, adv, solution, list(range(24)), diurnal_multipliers, diurnal_deployments)
						changed=True
					except:
						import traceback
						traceback.print_exc()

		if changed:
			pickle.dump(metrics, open(performance_metrics_fn,'wb'))
	except:
		import traceback
		traceback.print_exc()


	### Calculates some measure of practical resilience for each strategy
	### current resilience measure is flash crowd / DDoS attack in a region
	RECALC_RESILIENCE = False
	X_vals = np.linspace(10,500,num=20)#[10,40,80,100,130,150,180,200,210,220,250]#[10,20,30,40,50,60,70,80,90,100]
	Y_vals = [1.3]
	try:
		changed=False
		for random_iter in range(N_TO_SIM):
			k_of_interest = 'resilience_to_congestion'
			havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)
			if RECALC_RESILIENCE or havent_calced_everything:
				print("-----Flash crowd calc for deployment number = {} -------".format(random_iter))
				if sas is None:
					deployment = metrics['deployment'][random_iter]
					deployment['port'] = port

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
					deployment['port'] = port
					sas.update_deployment(deployment)
				inflated_deployments = get_inflated_metro_deployments(sas, X_vals, Y_vals)
				ug_vols = sas.ug_to_vol
				for solution in soln_types:
					if not RECALC_RESILIENCE:
						if metrics[k_of_interest][random_iter][solution]  != \
							default_metrics[k_of_interest][0][solution]:
							print("Already calced {}".format(solution))
							continue
					try:
						adv = metrics['adv'][random_iter][solution]
					except:
						print("No solution for {}".format(solution))
						continue
					try:
						print("Assessing resilience to congestion for {}, sim number {}".format(solution,random_iter))
						print("Baseline congestion is {}".format(solve_lp_with_failure_catch(sas, adv)['fraction_congested_volume']))
						# m = assess_resilience_to_congestion(sas, adv, solution, X_vals)['metrics']
						m = assess_resilience_to_flash_crowds_mp(sas, adv, solution, X_vals, Y_vals, inflated_deployments)
						metrics['resilience_to_congestion'][random_iter][solution] = m['metrics']
						metrics['prefix_withdrawals'][random_iter][solution] = m['prefix_withdrawals']
						metrics['fraction_congested_volume'][random_iter][solution] = m['fraction_congested_volume']
						changed=True
					except:
						import traceback
						traceback.print_exc()

		if changed:
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
	LATENCY_I = i;i+=1
	LATENCY_PENALTY_I = i;i+=1
	## Mutable
	POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I = i;i+=1
	POPP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I = i;i+=1
	POPP_FAILURE_CONGESTION_I = i;i+=1
	POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I = i;i+=1
	POP_FAILURE_CONGESTION_I = i;i+=1

	FLASH_CROWD_SINGLE_X_I = i;i+=1
	FLASH_CROWD_LATENCY_VARY_X_I = i;i+=1
	FLASH_CROWD_CONGESTION_VARY_X_I = i;i+=1
	FLASH_CROWD_PREFIX_I = i;i+=1
	FLASH_CROWD_LIMITING_FACTOR_LATENCY_I = i;i+=1
	FLASH_CROWD_LIMITING_FACTOR_CONGESTION_I = i;i+=1
	
	DIURNAL_LATENCY_VARY_I = i;i+=1
	DIURNAL_CONGESTION_VARY_I = i;i+=1
	DIURNAL_ASSIGNMENT_DELTA_VARY_I = i;i+=1


	n_subs = i
	f,ax=plt.subplots(n_subs,1)
	f.set_size_inches(6,4*n_subs)

	## we plot the performance changes for a single flash crowd volume increase
	single_X_of_interest = X_vals[len(X_vals)//2]
	single_Y_of_interest = Y_vals[len(Y_vals)//2]
	SIM_INDS_TO_PLOT = list(range(N_TO_SIM))

	#### Plotting everything
	for k in list(metrics):
		if 'latency' in k or 'latencies' in k:
			metrics['stats_' + k] = {}
	interesting_latency_suboptimalities = [-10,-50,-100]
	for add_str in ["", "_penalty", "_lagrange"]:
		for k in ['stats_latency{}_thresholds_normal'.format(add_str), 'stats_latency{}_thresholds_fail_popp'.format(add_str), 
			'stats_latency{}_thresholds_fail_pop'.format(add_str)]:
			metrics[k] = {solution: {i:{} for i in SIM_INDS_TO_PLOT} for solution in soln_types}
	metrics['stats_resilience_to_congestion'] = {solution: {i:{} for i in SIM_INDS_TO_PLOT} for solution in soln_types}
	metrics['stats_volume_multipliers'] = {solution:None for solution in soln_types}
	metrics['stats_diurnal'] = {solution: {i:{} for i in SIM_INDS_TO_PLOT} for solution in soln_types}

	def get_failure_metric_arr(k, solution, verb=False):
		ret = []
		avg_ret = []
		mind,maxd = np.inf,-1*np.inf
		all_vol = 0
		actually_all_vol = 0
		vol_congested = 0
		vol_best_case_congested = 0
		## storing latency threshold statistics
		threshold_stats = {i:{} for i in SIM_INDS_TO_PLOT}

		for ri in SIM_INDS_TO_PLOT:
			these_metrics = metrics[k][ri][solution]
			ugs={}

			summaries_by_element = {}
			this_diffs,this_vols = [],[]
			this_sim_total_volume_congested = 0
			this_sim_total_volume = 0
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
					this_sim_total_volume += vol
					if perf2 != NO_ROUTE_LATENCY and perf1 != NO_ROUTE_LATENCY:
						avg_ret.append((perf1-perf2,vol))
						this_diffs.append(perf1-perf2)
						this_vols.append(vol)
					if perf2 == NO_ROUTE_LATENCY:
						vol_congested += vol
						this_sim_total_volume_congested += vol
						perf2=perf2*100
					all_vol += vol
					if diff > maxd:
						maxd=diff
					if diff < mind:
						mind=diff
				ret.append((perf1-perf2, vol))

			### Store the fraction of users that DONT satisfy a latency objective
			this_sim_fraction_volume_congested = this_sim_total_volume_congested / (this_sim_total_volume + .0000001)
			try:
				this_x,this_cdf_x = get_cdf_xy(list(zip(this_diffs,this_vols)), weighted=True)
				for lat_threshold in interesting_latency_suboptimalities:
					xi = np.argmin(np.abs(this_x-lat_threshold))
					threshold_stats[ri][lat_threshold] = (1-this_sim_fraction_volume_congested) * this_cdf_x[xi] + this_sim_fraction_volume_congested
			except IndexError: # no good data
				for lat_threshold in interesting_latency_suboptimalities:
					threshold_stats[ri][lat_threshold] = this_sim_fraction_volume_congested ## all users are within the latency

			


		x=np.linspace(mind,maxd,num=200)
		if vol_congested > 0:
			x[0] = -1*100*NO_ROUTE_LATENCY

		try:
			avg_latency_difference = np.average([el[0] for el in avg_ret], weights=[el[1] for el in avg_ret])
		except ZeroDivisionError:
			print("Problem doing {} {}".format(k,solution))
			avg_latency_difference = NO_ROUTE_LATENCY
		print("Average latency difference {},{}: {}".format(solution, k, avg_latency_difference))
		print("{} pct. volume congested".format(round(100 * vol_congested / (actually_all_vol + .00001), 2)))
		print("{} pct. optimally congested, all volume: {}".format(round(100 * vol_best_case_congested / (actually_all_vol+.00001), 2), actually_all_vol))

		return ret, x, {
			'avg_latency_difference': avg_latency_difference, 
			'frac_vol_congested': vol_congested / (all_vol+.0000001), 
			'frac_vol_bestcase_congested': vol_best_case_congested / (actually_all_vol+.0000001),
		}, threshold_stats


	for solution in soln_types:
		print(solution)
		try:
			#### Changes in latency
			diffs = []
			wts = []
			for random_iter in SIM_INDS_TO_PLOT:
				this_diffs = []
				this_wts = []
				for i in range(len(metrics['best_latencies'][random_iter])):
					diffs.append(metrics['best_latencies'][random_iter][i] - metrics['latencies'][random_iter][solution][i])
					this_diffs.append(metrics['best_latencies'][random_iter][i] - metrics['latencies'][random_iter][solution][i])
					wts.append(metrics['ug_to_vol'][random_iter][i])
					this_wts.append(metrics['ug_to_vol'][random_iter][i])

				this_x,this_cdf_x = get_cdf_xy(list(zip(this_diffs,this_wts)), weighted=True)
				for lat_threshold in interesting_latency_suboptimalities:
					xi = np.argmin(np.abs(this_x-lat_threshold))
					metrics['stats_latency_thresholds_normal'][solution][random_iter][lat_threshold] = this_cdf_x[xi]
			for lat_threshold in interesting_latency_suboptimalities:
				avg_suboptimality = np.mean(list([metrics['stats_latency_thresholds_normal'][solution][random_iter][lat_threshold] for random_iter in SIM_INDS_TO_PLOT]))
				print("({}) {} pct of traffic within {} ms of optimal for normal LP".format(solution, 100*round(1-avg_suboptimality,4), lat_threshold))
			x,cdf_x = get_cdf_xy(list(zip(diffs,wts)), weighted=True)
			ax[LATENCY_I].plot(x,cdf_x,label=solution)
			avg_latency_diff = np.average(diffs, weights=wts)
			print("Average latency compared to optimal : {}".format(avg_latency_diff))
			metrics['stats_best_latencies'][solution] = avg_latency_diff

			#### Resilience to PoP and PoPP failures
			
			all_differences, x, stats, threshold_stats = get_failure_metric_arr('popp_failures_latency_optimal_specific', solution)
			metrics['stats_' + 'popp_failures_latency_optimal_specific'][solution] = stats
			metrics['stats_latency_thresholds_fail_popp'][solution] = threshold_stats
			x, cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)

			all_differences, x, stats, threshold_stats = get_failure_metric_arr('pop_failures_latency_optimal_specific', solution)
			metrics['stats_' + 'pop_failures_latency_optimal_specific'][solution] = stats
			metrics['stats_latency_thresholds_fail_pop'][solution] = threshold_stats
			x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)


			#### Resilience to flash crowds
			m = metrics['resilience_to_congestion']
			cm = metrics['fraction_congested_volume']
			

			## Want to track where latency and congestion start to impact performance
			latency_limits_over_Y = []
			congestion_limits_over_Y = []

			for Y_val in Y_vals:
				latency_delta_meds = []			
				congestion_meds = []
				avg_congestion_by_X_val_sim = {ri:[] for ri in SIM_INDS_TO_PLOT}
				for X_val in X_vals:
					all_lats, all_congestions = [], []
					for ri in SIM_INDS_TO_PLOT:
						try:
							this_all_congestions = []
							for avg_lat_deltas in m[ri][solution][Y_val][X_val]:
								all_lats.append(avg_lat_deltas)
							for congestion in cm[ri][solution][Y_val][X_val]:
								all_congestions.append(congestion)
								this_all_congestions.append(congestion)

							avg_congestion_by_X_val_sim[ri].append(np.average(this_all_congestions))
						except:
							continue

					# if X_val == single_X_of_interest and Y_val == single_Y_of_interest:
					# 	## Plot CDF for this specific X val and Y val
					# 	x,cdf_x = get_cdf_xy(list(zip(all_lats,all_vols)), weighted=True)
					# 	ax[FLASH_CROWD_SINGLE_X_I].plot(x,cdf_x,label="{} Metro increase pct. = {}".format(solution,single_X_of_interest))
						
					lat_med = np.average(all_lats)
					con_med = np.average(all_congestions)

					latency_delta_meds.append(lat_med)
					congestion_meds.append(con_med)
				
				## High level stats
				for ri in SIM_INDS_TO_PLOT:
					try:
						critical_X = X_vals[np.where(np.array(avg_congestion_by_X_val_sim[ri]) > 0)[0][0]]
					except IndexError:
						## either never or always is congested
						critical_X = X_vals[0]
					metrics['stats_resilience_to_congestion'][solution][ri] = critical_X


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


			#### Diurnal Resilience
			hours_of_day = np.array(list(range(24)))
			## Want to track what diurnal multiplier causes congestion
			avg_latency_by_sim_Y_val = {ri: [] for ri in SIM_INDS_TO_PLOT}
			avg_congestion_by_sim_Y_val = {ri: [] for ri in SIM_INDS_TO_PLOT}

			avg_latency_by_Y_val_sim = {dm: [] for dm in diurnal_multipliers}
			avg_congestion_by_Y_val_sim = {dm: [] for dm in diurnal_multipliers}
			avg_churn_by_Y_val_sim = {dm: [] for dm in diurnal_multipliers}

			for Y_val in diurnal_multipliers:
				all_lats, all_congestions = {ri:[] for ri in SIM_INDS_TO_PLOT}, {ri:[] for ri in SIM_INDS_TO_PLOT}
				all_churns = {ri:[] for ri in SIM_INDS_TO_PLOT}
				for ri in SIM_INDS_TO_PLOT:
					for X_val in hours_of_day:
						try:
							all_lats[ri].append(metrics['diurnal'][ri][solution]['metrics'][Y_val][X_val][0][0])
							all_churns[ri].append(metrics['diurnal'][ri][solution]['metrics'][Y_val][X_val][0][1])
							all_congestions[ri].append(metrics['diurnal'][ri][solution]['fraction_congested_volume'][Y_val][X_val])
						except KeyError:	
							continue

					lat_med = np.average(all_lats[ri])
					con_med = np.average(all_congestions[ri])
					churn_med = 100 * np.average(all_churns[ri])


					avg_latency_by_sim_Y_val[ri].append(lat_med)
					avg_congestion_by_sim_Y_val[ri].append(con_med)

					avg_latency_by_Y_val_sim[Y_val].append(lat_med)
					avg_congestion_by_Y_val_sim[Y_val].append(con_med)
					avg_churn_by_Y_val_sim[Y_val].append(churn_med)

			ax[DIURNAL_LATENCY_VARY_I].plot(diurnal_multipliers, list([np.mean(avg_latency_by_Y_val_sim[dm]) for dm in diurnal_multipliers]), label=solution)
			ax[DIURNAL_CONGESTION_VARY_I].plot(diurnal_multipliers, list([np.mean(avg_congestion_by_Y_val_sim[dm]) for dm in diurnal_multipliers]), label=solution)
			ax[DIURNAL_ASSIGNMENT_DELTA_VARY_I].plot(diurnal_multipliers, list([np.mean(avg_churn_by_Y_val_sim[dm]) for dm in diurnal_multipliers]), label=solution)


			## High level stats
			for ri in SIM_INDS_TO_PLOT:
				try:
					critical_Y = diurnal_multipliers[np.where(np.array(avg_congestion_by_sim_Y_val[ri]) > 0)[0][0]]
				except IndexError:
					## either never or always is congested
					critical_Y = diurnal_multipliers[0]
				metrics['stats_diurnal'][solution][ri] = critical_Y
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

	### FAILURE PLOTS
	## MUTABLE
	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].legend(fontsize=8)
	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].grid(True)
	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_xlabel("Latency Change Under Single-Link Failure (best - actual) (ms)")
	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_ylabel("CDF of Affected Traffic,Links")
	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POPP_FAILURE_CONGESTION_I].legend(fontsize=8)
	ax[POPP_FAILURE_CONGESTION_I].grid(True)
	ax[POPP_FAILURE_CONGESTION_I].set_xlabel("Fraction Congested Volume Single-Link Failure (best - actual) (ms)")
	ax[POPP_FAILURE_CONGESTION_I].set_ylabel("CDF of Traffic,Links")
	ax[POPP_FAILURE_CONGESTION_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].legend(fontsize=8)
	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].grid(True)
	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_xlabel("Latency Change Under Single-PoP Failure (best - actual) (ms)")
	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_ylabel("CDF of Affected Traffic,PoPs")
	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

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

	ax[DIURNAL_LATENCY_VARY_I].legend(fontsize=8)
	ax[DIURNAL_LATENCY_VARY_I].grid(True)
	ax[DIURNAL_LATENCY_VARY_I].set_xlabel("Diurnal Multiplier Amount (pct)")
	ax[DIURNAL_LATENCY_VARY_I].set_ylabel("Average Latency Increase (ms)")

	ax[DIURNAL_CONGESTION_VARY_I].legend(fontsize=8)
	ax[DIURNAL_CONGESTION_VARY_I].grid(True)
	ax[DIURNAL_CONGESTION_VARY_I].set_xlabel("Diurnal Multiplier Amount (pct)")
	ax[DIURNAL_CONGESTION_VARY_I].set_ylabel("Average Congestion (ms)")

	ax[DIURNAL_ASSIGNMENT_DELTA_VARY_I].legend(fontsize=8)
	ax[DIURNAL_ASSIGNMENT_DELTA_VARY_I].grid(True)
	ax[DIURNAL_ASSIGNMENT_DELTA_VARY_I].set_xlabel("Diurnal Multiplier Amount (pct)")
	ax[DIURNAL_ASSIGNMENT_DELTA_VARY_I].set_ylabel("Average Daily Traffic Churn (pct)")

	save_fig_fn = "full_sized_full_objective{}.pdf".format(dpsize)

	save_fig(save_fig_fn)

	return metrics



def compare_speedup_heuristic():
	"""
		Compare wall clock time and goodness of solution between using linear programming solver 
		with monte-carlo and using our speedup heuristic.
	"""
	### TODO -- could also vary the number of monte carlo, the deployment size, etc..

	np.random.seed(31410)
	try:
		dpsize = sys.argv[1]
		lambduh = .00001 ## unused more or less
		gamma = 2.0
		obj = 'avg_latency'
		using_generic_objectives = ['not_using', 'using']
		# using_generic_objectives = ['using', 'not_using']

		n_random_sim = 10
		performance_metrics_fn = os.path.join(CACHE_DIR, "assessing_speedup_heuristic_metrics_{}.pkl".format(dpsize))
		if os.path.exists(performance_metrics_fn):
			metrics = pickle.load(open(performance_metrics_fn, 'rb'))
		else:
			metrics = {tf:{i:{'done':False} for i in range(n_random_sim)} for tf in using_generic_objectives}
		wm = None

		if False:
			port = int(sys.argv[2])
			for random_iter in range(n_random_sim):
				try:
					this_iter_deployment = metrics[random_iter]['deployment']
				except KeyError:
					this_iter_deployment = get_random_deployment(dpsize)
				this_iter_deployment['port'] = port
				for using_generic_objective in using_generic_objectives:
					print("Random deployment for objective {}, number {}/{}".format(using_generic_objective,random_iter+1,n_random_sim))
					# try:
					# 	if metrics[using_generic_objective][random_iter]['done']: continue
					# except KeyError:
					# 	metrics[using_generic_objective][random_iter] = {'done': False}
					deployment = copy.deepcopy(this_iter_deployment)
					metrics[random_iter] = {'deployment': deployment}
					n_prefixes = deployment_to_prefixes(deployment)

					if using_generic_objective == 'using':
						sas = Sparse_Advertisement_Solver(deployment, 
							lambduh=lambduh,verbose=True,with_capacity=True,n_prefixes=n_prefixes,
							using_resilience_benefit=True, gamma=gamma, generic_objective=obj)
					else:
						sas = Sparse_Advertisement_Solver(deployment, 
							lambduh=lambduh,verbose=True,with_capacity=True,n_prefixes=n_prefixes,
							using_resilience_benefit=True, gamma=gamma)
					if wm is None:
						wm = Worker_Manager(sas.get_init_kwa(), deployment)
						wm.start_workers()
					sas.set_worker_manager(wm)
					sas.update_deployment(sas.output_deployment())
					ts = time.time()
					sas.solve()
					te = time.time()
					metrics[using_generic_objective][random_iter]['settings'] = sas.get_init_kwa()
					metrics[using_generic_objective][random_iter]['adv'] = sas.get_last_advertisement()
					metrics[using_generic_objective][random_iter]['t_convergence'] = ts - te
					metrics[using_generic_objective][random_iter]['final_objective'] = sas.metrics['actual_nonconvex_objective'][-1]
					metrics[using_generic_objective][random_iter]['final_latency_objective'] = sas.metrics['latency_benefit'][-1]
					metrics[using_generic_objective][random_iter]['optimal_objective'] = sas.optimal_expensive_solution
					metrics[using_generic_objective][random_iter]['save_run_dir'] = sas.save_run_dir

					metrics[using_generic_objective][random_iter]['done'] = True
					pickle.dump(metrics, open(performance_metrics_fn, 'wb'))


		# ugo -> itr -> [(opp obj - achieved obj, wall time)]
		font = {'size'   : 14}
		matplotlib.rc('font', **font)
		f,ax = plt.subplots(2)
		f.set_size_inches(4,2)
		high_level_metrics = {}
		skip_every=3
		labels = {
			'using': 'No Heuristic', 
			'not_using': 'Speedup Heuristic'
		}
		max_itr = 0
		for ugoi, ugo in enumerate(using_generic_objectives):
			high_level_metrics[ugo] = {}
			for ri in metrics[ugo]:
				if ri != 0: continue
				if not metrics[ugo][ri].get('done',False): continue
				srd = metrics[ugo][ri]['save_run_dir']

				# opp_obj = metrics[ugo][ri]['optimal_objective']['overall']
				opp_obj = metrics[ugo][ri]['optimal_objective']['latency']
				tstart = np.inf
				for fn in glob.glob(os.path.join(srd, 'small-stats-*')):
					tstart = min(tstart, os.path.getmtime(fn))
				for fn in glob.glob(os.path.join(srd, 'small-stats-*')):
					these_small_stats = pickle.load(open(fn, 'rb'))['optimization_vars']
					itr = these_small_stats['iter']
					# dlta = (these_small_stats['current_objective'] - opp_obj) / opp_obj
					dlta = -1 * (these_small_stats['current_latency_benefit'] - opp_obj) 

					try:
						wall_time = (these_small_stats['calc_times'][0][0]+these_small_stats['calc_times'][1][0]+\
							these_small_stats['calc_times'][2][0]) / 3600
					except IndexError:
						wall_time = 0
					try:	
						high_level_metrics[ugo][itr].append((dlta, wall_time))
					except KeyError:
						high_level_metrics[ugo][itr] = [(dlta, wall_time)]

			itrs = sorted(list(high_level_metrics[ugo]))
			max_itr = max(max(itrs), max_itr)
			dltas = list([np.mean([el[0] for el in high_level_metrics[ugo][itr]]) for itr in itrs])
			wcts = np.array(list([np.mean([el[1] for el in high_level_metrics[ugo][itr]]) for itr in itrs]))
			wcts = np.cumsum(wcts)
			ax[0].plot(itrs[::skip_every], dltas[::skip_every], label=labels[ugo], color=some_colors[ugoi], marker=markers[ugoi])
			ax[1].plot(itrs[::skip_every], wcts[::skip_every], label=labels[ugo], color=some_colors[ugoi], marker=markers[ugoi])
		ax[1].set_xlabel("Training Iteration", fontsize=16)
		ax[0].set_ylabel("Objective\n (ms)", fontsize=14)
		ax[1].set_ylabel("Time\n (h)", fontsize=15)

		ax[0].set_xlim([0,max_itr+10])
		ax[1].set_xlim([0,max_itr+10])
		# ax[0].set_xlim([0,40])
		ax[0].set_xticks([])
		# ax[1].set_xlim([0,40])
		ax[1].set_yticks([0,20.0])
		ax[1].set_ylim([0,25.0])
		
		ax[0].set_ylim([0,9])
		
		# ax[0].legend(loc='upper right', fontsize=12)
		ax[1].legend(loc='upper left', fontsize=12)

		save_fig('objective_heuristic_comparison_{}.pdf'.format(dpsize))


	except:
		import traceback
		traceback.print_exc()
	finally:
		try:
			wm.stop_workers()
		except:
			pass


def train_models_for_obj(obj):
	np.random.seed(31415)
	try:
		dpsize = sys.argv[1]
		port = int(sys.argv[2])
		lambduh = .00001 ## unused more or less
		gamma = 2.0

		n_random_sim = 1
		performance_metrics_fn = os.path.join(CACHE_DIR, "testing_objective_metrics_{}-{}.pkl".format(dpsize, obj))
		if os.path.exists(performance_metrics_fn):
			metrics = pickle.load(open(performance_metrics_fn, 'rb'))
		else:
			metrics = {i:{'done':False} for i in range(n_random_sim)}
		wm = None
		for random_iter in range(n_random_sim):
			try:
				if metrics[random_iter]['done']: continue
			except KeyError:
				metrics[random_iter] = {'done': False}
			print("Random deployment for objective {}, number {}/{}".format(obj,random_iter,n_random_sim))
			deployment = get_random_deployment(dpsize, n_users_per_peer=3000,focus_on_peers=False)
			deployment['port'] = port
			metrics[random_iter]['deployment'] = deployment
			n_prefixes = deployment_to_prefixes(deployment)

			sas = Sparse_Advertisement_Solver(deployment, 
				lambduh=lambduh,verbose=True,with_capacity=True,n_prefixes=n_prefixes,
				using_resilience_benefit=True, gamma=gamma, generic_objective=obj)
			if wm is None:
				wm = Worker_Manager(sas.get_init_kwa(), deployment)
				wm.start_workers()
			sas.set_worker_manager(wm)
			sas.update_deployment(sas.output_deployment())
			ts = time.time()
			sas.solve()
			te = time.time()
			metrics[random_iter]['settings'] = sas.get_init_kwa()
			metrics[random_iter]['adv'] = sas.get_last_advertisement()
			metrics[random_iter]['t_convergence'] = ts - te
			metrics[random_iter]['final_objective'] = sas.metrics['actual_nonconvex_objective'][-1]
			metrics[random_iter]['final_latency_objective'] = sas.metrics['latency_benefit'][-1]
			metrics[random_iter]['optimal_objective'] = sas.optimal_expensive_solution
			metrics[random_iter]['save_run_dir'] = sas.save_run_dir

			metrics[random_iter]['done'] = True
			pickle.dump(metrics, open(performance_metrics_fn, 'wb'))

	except:
		import traceback
		traceback.print_exc()
	finally:
		try:
			wm.stop_workers()
		except:
			pass

def gen_plots_over_objs():
	for metrics_fn in glob.glob(os.path.join(CACHE_DIR, 'testing_objective_metrics_*.pkl')):
		which_metrics = re.search('metrics\_.+\-(.+)\.pkl', metrics_fn).group(1)
		metrics = pickle.load(open(metrics_fn, 'rb'))
		n_sims = len(metrics)


		t_convergences = list([metrics[i]['t_convergence'] for i in range(n_sims)])
		print("Mean convergence time for {} : {}".format(which_metrics, np.mean(t_convergences)))

		objective_efficiencies = list([np.abs(metrics[i]['final_objective'] - metrics[i]['optimal_objective']['overall']) / metrics[i]['optimal_objective']['overall'] \
			for i in range(n_sims)])
		print("Mean objective efficiency time for {} : {}".format(which_metrics, np.mean(objective_efficiencies)))


if __name__ == "__main__":
	# compare_speedup_heuristic()

	# train_models_for_obj('avg_latency')
	compare_from_trained_sparse('1726599827-actual-32-sparse')
	# compare_from_trained_sparse('1726672233-small-sparse')



	# train_models_for_obj('squaring')
	# train_models_for_obj('square_rooting')
	# gen_plots_over_objs()
