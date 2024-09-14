from constants import *
from helpers import *
from wrapper_eval import *
from solve_lp_assignment import *

import pickle, numpy as np, matplotlib.pyplot as plt, copy, itertools, time
from sparse_advertisements_v3 import *

def calc_pct_volume_within_latency(sas, adv):
	routed_through_ingress, _ = sas.calculate_ground_truth_ingress(adv)
	## want latency -> amt of volume that can reach popp within that latency / amt of volume that could possibly reach it
	all_lats = list([l for ug in sas.ug_perfs for l in sas.ug_perfs[ug].values()])
	min_overall_lat = np.min(all_lats)
	max_overall_lat = np.max(all_lats)
	lats = np.linspace(min_overall_lat, max_overall_lat, num=100)
	ret = np.zeros((len(lats)))
	for i,lat in tqdm.tqdm(enumerate(lats), desc="Finding volume within each latency threshold..."):
		vol_this_lat_possible = 0
		for ug in sas.ug_perfs:
			for popp,l in sas.ug_perfs[ug].items():
				if l <= lat:
					vol_this_lat_possible += sas.ug_to_vol[ug]
		vol_this_lat_achieved = 0
		counted={}
		for prefix_i in routed_through_ingress:
			for ug,popp in routed_through_ingress[prefix_i].items():
				l = sas.ug_perfs[ug][popp]
				if l <= lat:
					try:
						counted[ug,popp]
					except KeyError:
						vol_this_lat_achieved += sas.ug_to_vol[ug]
						counted[ug,popp] = None
		ret[i] = vol_this_lat_achieved / vol_this_lat_possible
	return {'latencies':lats, 'volume_fractions':ret}

def evaluate_all_metrics(dpsize, port, save_run_dir=None, **kwargs):
	# for each deployment, get different advertisement strategies
	# look at latency under popp and link failures compared to optimal
	# sparse should be simultaneously better in both normal and failure scenarios

	### save_run_dir can be None (just start from scratch), single directory (in which case nsim must be 1), or list of directories of length
	### nsim, some of which may be None 

	# np.random.seed(31413)
	metrics = {}

	### scale individual metro/thing volume
	X_vals = np.linspace(10,500,num=20)#[10,40,80,100,130,150,180,200,210,220,250]#[10,20,30,40,50,60,70,80,90,100]
	### overprovisioning factor (1.3 = 30%)
	Y_vals = [1.3]

	wm = None
	sas = None

	performance_metrics_fn = kwargs.get('use_performance_metrics_fn', global_performance_metrics_fn(dpsize))
	soln_types = kwargs.get('soln_types', global_soln_types)
	if 'soln_types' in kwargs:
		del kwargs['soln_types']

	## Format save_run_dirs
	N_TO_SIM = kwargs.get('nsim',1)
	if N_TO_SIM > 1:
		if save_run_dir is not None:
			assert type(save_run_dir) == list and len(save_run_dir) == N_TO_SIM
			save_run_dirs = save_run_dir
		else:
			save_run_dirs = [None for _ in range(N_TO_SIM)]
	else:
		save_run_dirs = [save_run_dir]

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
	try:
		for random_iter in range(N_TO_SIM):
			try:
				if save_run_dirs[random_iter] is not None: ## we want to hotstart on a save run dir and continue training
					raise TypeError
				metrics['compare_rets'][random_iter]['n_advs'] ## if this field is populated, we've already computed this iteration's solution
				continue
			except TypeError:
				pass
			print("-----Deployment number = {} -------".format(random_iter))
			if save_run_dirs[random_iter] is not None:
				print("Loading from hotstart dir")
				deployment = pickle.load(open(os.path.join(RUN_DIR, save_run_dirs[random_iter], 'state-0.pkl'), 'rb'))['deployment']
				deployment['port'] = port
			elif kwargs.get('prefix_deployment') is not None:
				print("Prefixing deployment")
				deployment = kwargs.get('prefix_deployment')
				deployment['port'] = port
			else:
				while True:
					try:
						deployment = get_random_deployment(dpsize, **kwargs)
						deployment['port'] = port
						if len(deployment['popps']) < 20:
							continue
						break
					except:
						## It could be that this function fails because our random PoP selection isn't great
						## Just keep trying and it will eventually work
						import traceback
						traceback.print_exc()
			metrics['deployment'][random_iter] = deployment
	
			n_prefixes = kwargs.get('n_prefixes', deployment_to_prefixes(deployment))
	
			sas = Sparse_Advertisement_Eval(deployment, verbose=True,
				lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
				using_resilience_benefit=True, gamma=gamma, n_prefixes=n_prefixes,
				save_run_dir=save_run_dirs[random_iter])

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
			metrics['save_run_dir'][random_iter] = sas.sas.save_run_dir # sparse's save run dir
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

			pickle.dump(metrics, open(performance_metrics_fn,'wb'))

	except:
		import traceback
		traceback.print_exc()

	RECALC_LATENCY_WITH_PENALTY = False
	try:
		changed=False
		for random_iter in range(N_TO_SIM):
			k_of_interest = 'latencies_penalty'
			havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)
			if RECALC_LATENCY_WITH_PENALTY or havent_calced_everything:
				print("-----Latency with penalty calc for deployment number = {} -------".format(random_iter))
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
				ug_vols = sas.ug_to_vol
				ret = metrics['compare_rets'][random_iter]
				for solution in soln_types:
					try:
						adv = ret['adv_solns'][solution][0]
					except:
						print("No solution for {}".format(solution))
						continue

					print("Assessing latency with penalty for {}".format(solution))
					one_per_peer_adv = np.eye(sas.n_popps)
					penalty_lats_by_ug = sas.solve_lp_with_failure_catch_weighted_penalty(adv, one_per_peer_adv)['lats_by_ug']
					metrics['latencies_penalty'][random_iter][solution] = penalty_lats_by_ug
					changed=True

		if changed:
			pickle.dump(metrics, open(performance_metrics_fn,'wb'))
	except:
		import traceback
		traceback.print_exc()

	RECALC_PCT_VOL_IN_LAT_MULTIPLIERS = False
	try:
		for random_iter in range(N_TO_SIM):
			k_of_interest = 'pct_volume_within_latency'
			havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)
			if RECALC_PCT_VOL_IN_LAT_MULTIPLIERS or havent_calced_everything:
				print("-----Volume calc for deployment number = {} -------".format(random_iter))
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
				ug_vols = sas.ug_to_vol
				ret = metrics['compare_rets'][random_iter]
				for solution in soln_types:
					try:
						adv = ret['adv_solns'][solution][0]
					except:
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

						metrics['popp_failures_sticky_congestion'][random_iter][solution] = ret['sticky']['congestion_delta']
						metrics['popp_failures_sticky_latency_optimal'][random_iter][solution] = ret['sticky']['latency_delta_optimal']
						metrics['popp_failures_sticky_latency_optimal_specific'][random_iter][solution] = ret['sticky']['latency_delta_specific']
						metrics['popp_failures_sticky_latency_before'][random_iter][solution] = ret['sticky']['latency_delta_before']

						ret = assess_failure_resilience(sas, adv, which='pops')
						metrics['pop_failures_congestion'][random_iter][solution] = ret['mutable']['congestion_delta']
						metrics['pop_failures_latency_optimal'][random_iter][solution] = ret['mutable']['latency_delta_optimal']
						metrics['pop_failures_latency_optimal_specific'][random_iter][solution] = ret['mutable']['latency_delta_specific']
						metrics['pop_failures_latency_before'][random_iter][solution] = ret['mutable']['latency_delta_before']

						metrics['pop_failures_sticky_congestion'][random_iter][solution] = ret['sticky']['congestion_delta']
						metrics['pop_failures_sticky_latency_optimal'][random_iter][solution] = ret['sticky']['latency_delta_optimal']
						metrics['pop_failures_sticky_latency_optimal_specific'][random_iter][solution] = ret['sticky']['latency_delta_specific']
						metrics['pop_failures_sticky_latency_before'][random_iter][solution] = ret['sticky']['latency_delta_before']
					except:
						import traceback
						traceback.print_exc()
						continue


		if changed:
			pickle.dump(metrics, open(performance_metrics_fn,'wb'))

	except:
		import traceback
		traceback.print_exc()

	# RECALC_FAILURE_LATENCY_PENALTY_METRICS = False
	# try:
	# 	changed=False
	# 	for random_iter in range(N_TO_SIM):
	# 		k_of_interest = 'popp_failures_latency_penalty_optimal_specific'
	# 		havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)

	# 		if RECALC_FAILURE_LATENCY_PENALTY_METRICS or havent_calced_everything:
	# 			print("-----Failure calc with latency penalty for deployment number = {} -------".format(random_iter))
	# 			if sas is None:
	# 				deployment = metrics['deployment'][random_iter]
	# 				deployment['port'] = port

	# 				n_prefixes = kwargs.get('n_prefixes', deployment_to_prefixes(deployment))
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
	# 				deployment['port'] = port
	# 				sas.update_deployment(deployment)
	# 			changed=True

	# 			for solution in soln_types:
	# 				if not RECALC_FAILURE_LATENCY_PENALTY_METRICS:
	# 					if metrics[k_of_interest][random_iter][solution]  != \
	# 						default_metrics[k_of_interest][0][solution]:
	# 						print("Already calced {}".format(solution))
	# 						continue
	# 				print(solution)
	# 				adv = metrics['adv'][random_iter][solution]
	# 				if len(adv) == 0:
	# 					print("No solution for {}".format(solution))
	# 					continue
	# 				try:
	# 					ret = assess_failure_resilience(sas, adv, which='popps', penalty=True)
	# 					metrics['popp_failures_congestion_penalty'][random_iter][solution] = ret['mutable']['congestion_delta']
	# 					metrics['popp_failures_latency_penalty_optimal_specific'][random_iter][solution] = ret['mutable']['latency_delta_specific']

					
	# 					ret = assess_failure_resilience(sas, adv, which='pops', penalty=True)
	# 					metrics['pop_failures_congestion_penalty'][random_iter][solution] = ret['mutable']['congestion_delta']
	# 					metrics['pop_failures_latency_penalty_optimal_specific'][random_iter][solution] = ret['mutable']['latency_delta_specific']

	# 				except:
	# 					import traceback
	# 					traceback.print_exc()
	# 					continue


	# 	if changed:
	# 		pickle.dump(metrics, open(performance_metrics_fn,'wb'))

	# except:
	# 	import traceback
	# 	traceback.print_exc()

	# RECALC_FAILURE_LATENCY_LAGRANGE_METRICS = False
	# try:
	# 	changed=False
	# 	for random_iter in range(N_TO_SIM):
	# 		break
	# 		k_of_interest = 'popp_failures_latency_lagrange_optimal_specific'
	# 		havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)

	# 		if RECALC_FAILURE_LATENCY_LAGRANGE_METRICS or havent_calced_everything:
	# 			print("-----Failure calc with latency w/ Lagrange for deployment number = {} -------".format(random_iter))
	# 			if sas is None:
	# 				deployment = metrics['deployment'][random_iter]
	# 				deployment['port'] = port

	# 				n_prefixes = kwargs.get('n_prefixes', deployment_to_prefixes(deployment))
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
	# 				deployment['port'] = port
	# 				sas.update_deployment(deployment)
	# 			changed=True

	# 			for solution in soln_types:
	# 				if not RECALC_FAILURE_LATENCY_LAGRANGE_METRICS:
	# 					if metrics[k_of_interest][random_iter][solution]  != \
	# 						default_metrics[k_of_interest][0][solution]:
	# 						print("Already calced {}".format(solution))
	# 						continue
	# 				print(solution)
	# 				adv = metrics['adv'][random_iter][solution]
	# 				if len(adv) == 0:
	# 					print("No solution for {}".format(solution))
	# 					continue
	# 				try:
	# 					ret = assess_failure_resilience(sas, adv, which='popps', lagrange=True)
	# 					metrics['popp_failures_congestion_lagrange'][random_iter][solution] = ret['mutable']['congestion_delta']
	# 					metrics['popp_failures_latency_lagrange_optimal_specific'][random_iter][solution] = ret['mutable']['latency_delta_specific']

					
	# 					ret = assess_failure_resilience(sas, adv, which='pops', lagrange=True)
	# 					metrics['pop_failures_congestion_lagrange'][random_iter][solution] = ret['mutable']['congestion_delta']
	# 					metrics['pop_failures_latency_lagrange_optimal_specific'][random_iter][solution] = ret['mutable']['latency_delta_specific']

	# 				except:
	# 					import traceback
	# 					traceback.print_exc()
	# 					continue


	# 	if changed:
	# 		pickle.dump(metrics, open(performance_metrics_fn,'wb'))

	# except:
	# 	import traceback
	# 	traceback.print_exc()

	RECALC_VOL_MULTIPLIERS = False
	volume_multiply_values = np.linspace(0,29,num=20)
	try:
		changed=False
		for random_iter in range(N_TO_SIM):
			k_of_interest = 'volume_multipliers'
			havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)
			if RECALC_VOL_MULTIPLIERS or havent_calced_everything:
				print("-----Volume multiplier calc for deployment number = {} -------".format(random_iter))
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
				inflated_deployments = get_inflated_total_deployments(sas, volume_multiply_values)
				ug_vols = sas.ug_to_vol
				ret = metrics['compare_rets'][random_iter]
				for solution in soln_types:
					if not RECALC_VOL_MULTIPLIERS:
						if metrics[k_of_interest][random_iter][solution]  != \
							default_metrics[k_of_interest][0][solution]:
							print("Already calced {}".format(solution))
							continue
					try:
						adv = ret['adv_solns'][solution][0]
					except:
						print("No solution for {}".format(solution))
						continue
					try:
						print("Assessing volume multiplier for {}, sim number {}".format(solution,random_iter))
						# m = assess_resilience_to_congestion(sas, adv, solution, X_vals)['metrics']
						m = assess_volume_multipliers(sas, adv, solution, inflated_deployments)
						metrics['volume_multipliers'][random_iter][solution] = m['metrics']
						changed=True
					except:
						import traceback
						traceback.print_exc()

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
				ret = metrics['compare_rets'][random_iter]
				for solution in soln_types:
					if not RECALC_DIURNAL:
						if metrics[k_of_interest][random_iter][solution]  != \
							default_metrics[k_of_interest][0][solution]:
							print("Already calced {}".format(solution))
							continue
					try:
						adv = ret['adv_solns'][solution][0]
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
				ret = metrics['compare_rets'][random_iter]
				for solution in soln_types:
					if not RECALC_RESILIENCE:
						if metrics[k_of_interest][random_iter][solution]  != \
							default_metrics[k_of_interest][0][solution]:
							print("Already calced {}".format(solution))
							continue
					try:
						adv = ret['adv_solns'][solution][0]
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
	PCT_VOL_WITHIN_LATENCY_I = i; i+= 1
	## Mutable
	POPP_FAILURE_LATENCY_OPTIMAL_I = i;i+=1
	POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I = i;i+=1
	POPP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I = i;i+=1
	POPP_FAILURE_LATENCY_LAGRANGE_OPTIMAL_SPECIFIC_I = i;i+=1
	POPP_FAILURE_LATENCY_BEFORE_I = i;i+=1
	POPP_FAILURE_CONGESTION_I = i;i+=1
	POP_FAILURE_LATENCY_OPTIMAL_I = i;i+=1
	POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I = i;i+=1
	POP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I = i;i+=1
	POP_FAILURE_LATENCY_LAGRANGE_OPTIMAL_SPECIFIC_I = i;i+=1
	POP_FAILURE_LATENCY_BEFORE_I = i;i+=1
	POP_FAILURE_CONGESTION_I = i;i+=1

	POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I = i;i+=1
	POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I = i;i+=1
	POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I = i;i+=1
	POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I = i;i+=1

	FLASH_CROWD_SINGLE_X_I = i;i+=1
	FLASH_CROWD_LATENCY_VARY_X_I = i;i+=1
	FLASH_CROWD_CONGESTION_VARY_X_I = i;i+=1
	FLASH_CROWD_PREFIX_I = i;i+=1
	FLASH_CROWD_LIMITING_FACTOR_LATENCY_I = i;i+=1
	FLASH_CROWD_LIMITING_FACTOR_CONGESTION_I = i;i+=1
	
	VOLUME_GROWTH_I = i;i+=1

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



			#### Changes in latency (with weighted penalty)
			diffs = []
			wts = []
			for random_iter in SIM_INDS_TO_PLOT:
				this_diffs = []
				this_wts = []
				for i in range(len(metrics['best_latencies'][random_iter])):
					diffs.append(metrics['best_latencies'][random_iter][i] - metrics['latencies_penalty'][random_iter][solution][i])
					this_diffs.append(metrics['best_latencies'][random_iter][i] - metrics['latencies_penalty'][random_iter][solution][i])
					wts.append(metrics['ug_to_vol'][random_iter][i])
					this_wts.append(metrics['ug_to_vol'][random_iter][i])

				this_x,this_cdf_x = get_cdf_xy(list(zip(this_diffs,this_wts)), weighted=True)
				for lat_threshold in interesting_latency_suboptimalities:
					xi = np.argmin(np.abs(this_x-lat_threshold))
					metrics['stats_latency_penalty_thresholds_normal'][solution][random_iter][lat_threshold] = this_cdf_x[xi]
			for lat_threshold in interesting_latency_suboptimalities:
				avg_suboptimality = np.mean(list([metrics['stats_latency_penalty_thresholds_normal'][solution][random_iter][lat_threshold] for random_iter in SIM_INDS_TO_PLOT]))
				print("({}) {} pct of traffic within {} ms of optimal for latency penalty LP".format(solution, 100*round(1-avg_suboptimality,4), lat_threshold))
			
			x,cdf_x = get_cdf_xy(list(zip(diffs,wts)), weighted=True)
			ax[LATENCY_PENALTY_I].plot(x,cdf_x,label=solution)
			avg_latency_diff = np.average(diffs, weights=wts)
			print("Average latency compared to optimal with penalty : {}".format(avg_latency_diff))
			metrics['stats_latencies_penalty'][solution] = avg_latency_diff			


			#### PCT of Volume within a Certainty Latency Threshold
			for random_iter in SIM_INDS_TO_PLOT:
				m = metrics['pct_volume_within_latency'][random_iter][solution]
				ax[PCT_VOL_WITHIN_LATENCY_I].plot(m['latencies'], m['volume_fractions'], label="{} -- Sim {}".format(solution, random_iter))
			
			#### Resilience to PoP and PoPP failures
			
			all_differences, x, stats, threshold_stats = get_failure_metric_arr('popp_failures_latency_optimal_specific', solution)
			metrics['stats_' + 'popp_failures_latency_optimal_specific'][solution] = stats
			metrics['stats_latency_thresholds_fail_popp'][solution] = threshold_stats
			x, cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)

			all_differences, x, stats, threshold_stats = get_failure_metric_arr('popp_failures_latency_penalty_optimal_specific', solution)
			metrics['stats_' + 'popp_failures_latency_penalty_optimal_specific'][solution] = stats
			metrics['stats_latency_penalty_thresholds_fail_popp'][solution] = threshold_stats
			x, cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POPP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)

			all_differences, x, stats, threshold_stats = get_failure_metric_arr('popp_failures_latency_lagrange_optimal_specific', solution)
			metrics['stats_' + 'popp_failures_latency_lagrange_optimal_specific'][solution] = stats
			metrics['stats_latency_lagrange_thresholds_fail_popp'][solution] = threshold_stats
			x, cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POPP_FAILURE_LATENCY_LAGRANGE_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)

			all_differences, x, stats, threshold_stats = get_failure_metric_arr('pop_failures_latency_optimal_specific', solution)
			metrics['stats_' + 'pop_failures_latency_optimal_specific'][solution] = stats
			metrics['stats_latency_thresholds_fail_pop'][solution] = threshold_stats
			x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)

			all_differences, x, stats, threshold_stats = get_failure_metric_arr('pop_failures_latency_penalty_optimal_specific', solution)
			metrics['stats_' + 'pop_failures_latency_penalty_optimal_specific'][solution] = stats
			metrics['stats_latency_penalty_thresholds_fail_pop'][solution] = threshold_stats
			x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)

			all_differences, x, stats, threshold_stats = get_failure_metric_arr('pop_failures_latency_lagrange_optimal_specific', solution)
			metrics['stats_' + 'pop_failures_latency_lagrange_optimal_specific'][solution] = stats
			metrics['stats_latency_lagrange_thresholds_fail_pop'][solution] = threshold_stats
			x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POP_FAILURE_LATENCY_LAGRANGE_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)



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




			#### Ability to grow with volume
			latency_increases_by_X = {}
			m = metrics['volume_multipliers']
			for ri in SIM_INDS_TO_PLOT:
				for X_val,avg_lat in m[ri][solution].items():
					try:
						# try:
						# 	latency_increases_by_X[X_val].append(avg_lat - m[ri]['one_per_peering'][X_val])
						# except KeyError:
						# 	latency_increases_by_X[X_val] = [avg_lat  - m[ri]['one_per_peering'][X_val]]
						try:
							latency_increases_by_X[X_val].append(avg_lat)
						except KeyError:
							latency_increases_by_X[X_val] = [avg_lat]
					except TypeError:
						print("Weird error trying to access {} on {}".format(m[ri]['one_per_peering'], X_val))
			avg_latency_increases_by_X = {X_val: np.mean(latency_increases_by_X[X_val]) for X_val in latency_increases_by_X}
			sorted_ks = sorted(list(avg_latency_increases_by_X))
			plot_mean_volume_multipliers = list([avg_latency_increases_by_X[X_val] for X_val in sorted_ks])
			ax[VOLUME_GROWTH_I].plot(sorted_ks, plot_mean_volume_multipliers, label=solution)

			metrics['stats_volume_multipliers'][solution] = plot_mean_volume_multipliers



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


	ax[LATENCY_PENALTY_I].legend(fontsize=8)
	ax[LATENCY_PENALTY_I].grid(True)
	ax[LATENCY_PENALTY_I].set_xlabel("(PENALTY) Best - Actual Latency (ms)")
	ax[LATENCY_PENALTY_I].set_ylabel("CDF of Traffic")
	ax[LATENCY_PENALTY_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[LATENCY_PENALTY_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

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

	ax[POPP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I].legend(fontsize=8)
	ax[POPP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I].grid(True)
	ax[POPP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I].set_xlabel("(PENALTY) Latency Change Under Single-Link Failure (best - actual) (ms)")
	ax[POPP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I].set_ylabel("CDF of Affected Traffic,Links")
	ax[POPP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POPP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POPP_FAILURE_LATENCY_LAGRANGE_OPTIMAL_SPECIFIC_I].legend(fontsize=8)
	ax[POPP_FAILURE_LATENCY_LAGRANGE_OPTIMAL_SPECIFIC_I].grid(True)
	ax[POPP_FAILURE_LATENCY_LAGRANGE_OPTIMAL_SPECIFIC_I].set_xlabel("(LAGRANGE) Latency Change Under Single-Link Failure (best - actual) (ms)")
	ax[POPP_FAILURE_LATENCY_LAGRANGE_OPTIMAL_SPECIFIC_I].set_ylabel("CDF of Affected Traffic,Links")
	ax[POPP_FAILURE_LATENCY_LAGRANGE_OPTIMAL_SPECIFIC_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POPP_FAILURE_LATENCY_LAGRANGE_OPTIMAL_SPECIFIC_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

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

	ax[POP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I].legend(fontsize=8)
	ax[POP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I].grid(True)
	ax[POP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I].set_xlabel("(PENALTY) Latency Change Under Single-PoP Failure (best - actual) (ms)")
	ax[POP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I].set_ylabel("CDF of Affected Traffic,PoPs")
	ax[POP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POP_FAILURE_LATENCY_PENALTY_OPTIMAL_SPECIFIC_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

	ax[POP_FAILURE_LATENCY_LAGRANGE_OPTIMAL_SPECIFIC_I].legend(fontsize=8)
	ax[POP_FAILURE_LATENCY_LAGRANGE_OPTIMAL_SPECIFIC_I].grid(True)
	ax[POP_FAILURE_LATENCY_LAGRANGE_OPTIMAL_SPECIFIC_I].set_xlabel("(LAGRANGE) Latency Change Under Single-PoP Failure (best - actual) (ms)")
	ax[POP_FAILURE_LATENCY_LAGRANGE_OPTIMAL_SPECIFIC_I].set_ylabel("CDF of Affected Traffic,PoPs")
	ax[POP_FAILURE_LATENCY_LAGRANGE_OPTIMAL_SPECIFIC_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POP_FAILURE_LATENCY_LAGRANGE_OPTIMAL_SPECIFIC_I].set_xlim([-1*NO_ROUTE_LATENCY/2,0])

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
	ax[VOLUME_GROWTH_I].set_xlabel("Volume Inflation Amount (pct)")
	ax[VOLUME_GROWTH_I].set_ylabel("Average Latency over \nOne per Ingress (ms)")

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

	save_fig_fn = kwargs.get('save_fig_fn', "popp_latency_failure_comparison_{}.pdf".format(dpsize))

	save_fig(save_fig_fn)

	return metrics

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_run_dir", default=None)
	parser.add_argument("--use_cache_deployment", action='store_true', default=False)
	parser.add_argument("--dpsize", default=None, required=True)
	parser.add_argument("--port", required=True)
	args = parser.parse_args()

	port = args.port

	np.random.seed(31415)
	if args.save_run_dir is not None:
		## we could specify an array of hotstart dirs otherwise, but that's a task for another day
		assert N_TO_SIM == 1
		evaluate_all_metrics(args.dpsize, int(port), save_run_dir=args.save_run_dir)
	elif args.use_cache_deployment:
		deployment = pickle.load(open(global_performance_metrics_fn(dpsize), 'rb'))['deployment'][0]
		evaluate_all_metrics(args.dpsize, int(port), prefix_deployment=deployment)
	else:
		evaluate_all_metrics(args.dpsize, int(port))
