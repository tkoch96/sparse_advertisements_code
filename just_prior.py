from constants import *
from helpers import *
from wrapper_eval import *
from solve_lp_assignment import *

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

def assess_volume_multipliers(sas, adv, solution, multiply_values):
	ret = {
		'metrics': {'volume_multipliers': []},
	}
	for multiply_value in tqdm.tqdm(multiply_values,desc="Assessing multiply values for {}".format(solution)):
		deployment = sas.output_deployment()
		save_deployment = copy.deepcopy(deployment)
		for ug,v in deployment['ug_to_vol'].items():
			deployment['ug_to_vol'][ug] = v * multiply_value
		try:
			sas.update_deployment(deployment, verb=False)
		except ValueError:
			### not enough capacity globally
			sas.update_deployment(save_deployment, verb=False)
			break
		soln = sas.solve_lp_assignment(adv)
		if soln['solved']:
			# print("{} is solved at multiply {}".format(solution,multiply_value))
			ret['metrics']['volume_multipliers'].append(multiply_value)
		else:
			# print("{} NOT solved at multiply {}".format(solution,multiply_value))
			sas.update_deployment(save_deployment, verb=False)
			break

		sas.update_deployment(save_deployment, verb=False)
	return ret

def compute_optimal_prefix_withdrawals(sas, adv, popp, pre_soln, **kwargs):

	#### Computes the set of prefix withdrawals that results in the least amount of latency hit while 
	#### solving the congestion problem

	## for a given popp, there's technically power_set(number of advertisement to popp) that we could turn off
	## so if that's intractable, might need to sample it or something
	# print("SOLVING PRE SOLN")
	# pre_soln = sas.solve_lp_with_failure_catch(adv)


	pre_lats_by_ug = pre_soln['lats_by_ug']
	pre_paths_by_ug = pre_soln['paths_by_ug']

	# save_cap = copy.copy(sas.link_capacities_arr[sas.popp_to_ind[popp]])
	# sas.link_capacities_arr[sas.popp_to_ind[popp]] = new_link_capacity


	if kwargs.get('allow_congestion', False):
		solvlpfn = sas.solve_lp_with_failure_catch
	else:
		solvlpfn = sas.solve_lp_assignment


	poppi = sas.popp_to_ind[popp]
	prefis = np.where(adv[poppi,:])[0]
	valid_solutions = []
	ts = time.time()
	max_time = 600

	## First, try solving without withdrawing any prefix
	soln = solvlpfn(adv)
	if soln['solved'] and kwargs.get('can_withdraw_none', True): 
		valid_solutions.append(([], soln['lats_by_ug'] - pre_lats_by_ug,
		 sas.ug_vols, 0))
	else:
		for withdrawal_number in range(1,len(prefis) + 1):
			if withdrawal_number > 2: break
			for prefi_set in itertools.combinations(prefis,withdrawal_number):
				adv[poppi,np.array(prefi_set)] = 0
				## unclear exactly how to handle this --- we should allow > 1 util, just the best one
				soln = solvlpfn(adv)
				if not soln['solved']: 
					adv[poppi,np.array(prefi_set)] = 1
					continue

				# ## Get UG inds of interest (i.e., those who've changed)
				# ugis_of_interest = np.array([ugi for ugi in soln['paths_by_ug'] if 
				# 	soln['paths_by_ug'][ugi] != pre_paths_by_ug[ugi]])

				# if len(ugis_of_interest) == 0: 
				# 	## nothing changed, doesn't matter
				# 	continue

				# valid_solutions.append((prefi_set, soln['lats_by_ug'][ugis_of_interest] - pre_lats_by_ug[ugis_of_interest],
				#  sas.ug_vols[ugis_of_interest]))
				valid_solutions.append((prefi_set, soln['lats_by_ug'] - pre_lats_by_ug,
				 	sas.ug_vols, soln['fraction_congested_volume']))

				adv[poppi,np.array(prefi_set)] = 1
			if len(valid_solutions) > 0 or time.time() - ts > max_time or np.sum(adv.flatten()) == 0:
				# greedily stop
				break
	if len(valid_solutions) > 0:
		## pick the best one w.r.t. latency
		best_solution,best_deltas,best_vols,best_cv = valid_solutions[0]
		best_metric = np.sum(best_deltas * best_vols)
		for solution in valid_solutions:
			pref_solution, performance_deltas, vols, cv = solution
			if np.sum(performance_deltas * vols) < best_metric:
				best_solution = pref_solution
				best_metric = np.sum(performance_deltas * vols)
				best_deltas = performance_deltas
				best_vols = vols
				best_cv = cv
		best_solution = {
			'prefix_withdrawals': best_solution,
			'latency_deltas': (best_deltas, best_vols),
			'congested_volume':  best_cv,
		}
	else:
		best_solution = None
	
	return best_solution

def popp_failure_latency_comparisons():
	# for each deployment, get different advertisement strategies
	# look at latency under popp and link failures compared to optimal
	# sparse should be simultaneously better in both normal and failure scenarios

	# np.random.seed(31413)
	metrics = {}
	performance_metrics_fn = os.path.join(CACHE_DIR, 'just_prior_comparison_{}.pkl'.format(DPSIZE))

	X_vals = [40,60,80,100,120,150,180,200,250,300,350,400]#[10,40,80,100,130,150,180,200,210,220,250]#[10,20,30,40,50,60,70,80,90,100]
	if DPSIZE == 'really_friggin_small':
		X_vals = [10,15,30,35,40,45,50,55,60,65,70]

	N_TO_SIM = 1
	soln_types = ['anycast', 'one_per_pop', 'one_per_peering']


	wm = None
	sas = None
	
	default_metrics = {
		'compare_rets': {i:None for i in range(N_TO_SIM)},
		'adv': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'deployment': {i:None for i in range(N_TO_SIM)},
		'ug_to_vol': {i:None for i in range(N_TO_SIM)},
		'settings': {i:None for i in range(N_TO_SIM)},
		'popp_failures_latency': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'popp_failures_congestion': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'pop_failures_latency': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'pop_failures_congestion': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'latencies': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'best_latencies': {},
		'resilience_to_congestion': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'prefix_withdrawals': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'fraction_congested_volume': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'volume_multipliers': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
	}
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
				if i not in metrics[k] and i in default_metrics[k]:
					print("{} {}".format(k, i))
					metrics[k][i] = copy.deepcopy(default_metrics[k][i])
	try:
		for random_iter in range(N_TO_SIM):
			try:
				metrics['compare_rets'][random_iter]['n_advs']
				continue
			except TypeError:
				pass
			print("-----Deployment number = {} -------".format(random_iter))
			deployment = get_random_deployment(DPSIZE)
			metrics['deployment'][random_iter] = deployment
			n_prefixes = deployment_to_prefixes(deployment)
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
			ret = sas.compare_different_solutions(deployment_size=DPSIZE, n_run=1, verbose=True,
				 dont_update_deployment=True, only_do=soln_types)
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

			pickle.dump(metrics, open(performance_metrics_fn,'wb'))

	except:
		import traceback
		traceback.print_exc()

	RECALC_FAILURE_METRICS = False
	try:
		for random_iter in range(N_TO_SIM):
			break
			if RECALC_FAILURE_METRICS or metrics['pop_failures_latency'][random_iter][soln_types[0]]  == \
				default_metrics['pop_failures_latency'][random_iter][soln_types[0]]:
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
						
					ret = assess_failure_resilience(sas, adv, which='popps')
					metrics['popp_failures_congestion'][random_iter][solution] = ret['congestion_delta']
					metrics['popp_failures_latency'][random_iter][solution] = ret['latency_delta']

					ret = assess_failure_resilience(sas, adv, which='pops')
					metrics['pop_failures_congestion'][random_iter][solution] = ret['congestion_delta']
					metrics['pop_failures_latency'][random_iter][solution] = ret['latency_delta']

				pickle.dump(metrics, open(performance_metrics_fn,'wb'))

	except:
		import traceback
		traceback.print_exc()

	RECALC_VOL_MULTIPLIERS = False
	multiply_values = np.linspace(5,10)
	try:
		for random_iter in range(N_TO_SIM):
			break
			if RECALC_VOL_MULTIPLIERS or metrics['volume_multipliers'][random_iter][soln_types[0]]  == \
				default_metrics['volume_multipliers'][random_iter][soln_types[0]]:
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
				ug_vols = sas.ug_to_vol
				ret = metrics['compare_rets'][random_iter]
				for solution in soln_types:
					try:
						adv = ret['adv_solns'][solution][0]
					except:
						print("No solution for {}".format(solution))
						continue
					print("Assessing volume multiplier for {}".format(solution))
					m = assess_volume_multipliers(sas, adv, solution, multiply_values)
					metrics['volume_multipliers'][random_iter][solution] = m['metrics']

					
				pickle.dump(metrics, open(performance_metrics_fn,'wb'))
	except:
		import traceback
		traceback.print_exc()

	### Calculates some measure of practical resilience for each strategy
	### current resilience measure is flash crowd / DDoS attack in a region
	RECALC_RESILIENCE = False
	try:
		for random_iter in range(N_TO_SIM):
			break
			if RECALC_RESILIENCE or metrics['resilience_to_congestion'][random_iter][soln_types[0]]  == \
				default_metrics['resilience_to_congestion'][random_iter][soln_types[0]]:
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
				ug_vols = sas.ug_to_vol
				ret = metrics['compare_rets'][random_iter]
				for solution in soln_types:
					try:
						adv = ret['adv_solns'][solution][0]
					except:
						print("No solution for {}".format(solution))
						continue
					print("Assessing resilience to congestion for {}, sim number {}".format(solution,random_iter))
					print("Baseline congestion is {}".format(solve_lp_with_failure_catch(sas, adv)['fraction_congested_volume']))
					# m = assess_resilience_to_congestion(sas, adv, solution, X_vals)['metrics']
					m = assess_resilience_to_flash_crowds_mp(sas, adv, solution, X_vals)
					metrics['resilience_to_congestion'][random_iter][solution] = m['metrics']
					metrics['prefix_withdrawals'][random_iter][solution] = m['prefix_withdrawals']
					metrics['fraction_congested_volume'][random_iter][solution] = m['fraction_congested_volume']

					
				pickle.dump(metrics, open(performance_metrics_fn,'wb'))
	except:
		import traceback
		traceback.print_exc()
	finally:
		if wm is not None:
			wm.stop_workers()
	

	if False:
		for solution in soln_types:
			try:
				all_n_prefs_by_popp = []
				for i in range(N_TO_SIM):
					adv = metrics['adv'][i][solution]
					deployment = metrics['deployment'][i]
					popps = sorted(list(set(deployment['popps'])))
					# print("Solution {}".format(solution))
					# adv_summary(popps,adv)
					n_prefs_by_popp = np.sum(adv,axis=1)
					all_n_prefs_by_popp = list(n_prefs_by_popp) + all_n_prefs_by_popp
				
				x,cdf_x = get_cdf_xy(all_n_prefs_by_popp)
				plt.plot(x,cdf_x,label=solution)
			except:
				continue
		plt.grid(True)
		plt.xlabel("Number of Prefixes Being Advertised via Ingress")
		plt.ylabel("CDF of PoPPs,Random Iters")
		plt.legend()
		plt.xlim([0,30])
		plt.savefig("figures/n_prefs_by_popp_solutions.pdf")
		plt.clf(); plt.close()

	################################
	### PLOTTING
	################################
	
	LATENCY_I = 0
	POPP_FAILURE_LATENCY_I = 1
	POPP_FAILURE_CONGESTION_I = 2
	POP_FAILURE_LATENCY_I = 3
	POP_FAILURE_CONGESTION_I = 4
	FLASH_CROWD_SINGLE_X_I = 5
	FLASH_CROWD_LATENCY_VARY_X_I = 6
	FLASH_CROWD_CONGESTION_VARY_X_I = 7
	FLASH_CROWD_PREFIX_I = 8
	VOLUME_GROWTH_I = 9

	n_subs = 10
	f,ax=plt.subplots(n_subs,1)
	f.set_size_inches(6,4*n_subs)

	## we plot the performance changes for a single flash crowd volume increase
	single_X_of_interest = 350

	#### Plotting everything
	poppi_inds_to_plot = {ri:{} for ri in range(N_TO_SIM)}
	popi_inds_to_plot = {ri:{} for ri in range(N_TO_SIM)}
	limit_plot_ms = 0
	print("NOTE --- ONLY PLOTTING FAILURES FOR > {} ms".format(limit_plot_ms))
	for solution in soln_types:
		for ri in range(N_TO_SIM):
			for i,el in enumerate(metrics['popp_failures_latency'][ri][solution]):
				if np.abs(el[0]) >= limit_plot_ms:
					poppi_inds_to_plot[ri][i] = None
	for solution in soln_types:
		for ri in range(N_TO_SIM):
			for i,el in enumerate(metrics['pop_failures_latency'][ri][solution]):
				if np.abs(el[0]) >= limit_plot_ms:
					popi_inds_to_plot[ri][i] = None

	for solution in soln_types:
		print(solution)
		try:
			#### Changes in latency
			diffs = []
			wts = []
			for random_iter in range(N_TO_SIM):
				for i in range(len(metrics['best_latencies'][random_iter])):
					diffs.append(metrics['best_latencies'][random_iter][i] - metrics['latencies'][random_iter][solution][i])
					wts.append(metrics['ug_to_vol'][random_iter][i])

			x,cdf_x = get_cdf_xy(list(zip(diffs,wts)), weighted=True)
			ax[LATENCY_I].plot(x,cdf_x,label=solution)

			#### Resilience to PoP and PoPP failures
			
			## pop failures (latency)
			all_differences = [metrics['pop_failures_latency'][ri][solution][i] for ri in range(N_TO_SIM) for 
				i in popi_inds_to_plot[ri]]
			x,cdf_x = get_cdf_xy(all_differences,weighted=True)
			ax[POP_FAILURE_LATENCY_I].plot(x,cdf_x,label=solution)
			## pop failures (congestion)
			all_differences = [metrics['pop_failures_congestion'][ri][solution][i] for ri in range(N_TO_SIM) for 
				i in range(len(metrics['pop_failures_congestion'][ri][solution]))]
			print("{} pops had any congestion".format(sum(1 for el in all_differences if el != 0)))
			x,cdf_x = get_cdf_xy(all_differences,weighted=False)
			ax[POP_FAILURE_CONGESTION_I].plot(x,cdf_x,label=solution)

			## popp failures (latency)
			all_differences = [metrics['popp_failures_latency'][ri][solution][i] for ri in range(N_TO_SIM) for 
				i in poppi_inds_to_plot[ri]]
			x,cdf_x = get_cdf_xy(all_differences,weighted=True)
			ax[POPP_FAILURE_LATENCY_I].plot(x,cdf_x,label=solution)
			## popp failures (congestion)
			all_differences = [metrics['popp_failures_congestion'][ri][solution][i] for ri in range(N_TO_SIM) for 
				i in range(len(metrics['popp_failures_congestion'][ri][solution]))]
			print("{} links had any congestion".format(sum(1 for el in all_differences if el != 0)))
			x,cdf_x = get_cdf_xy(all_differences,weighted=False)
			ax[POPP_FAILURE_CONGESTION_I].plot(x,cdf_x,label=solution)
			

			#### Resilience to flash crowds
			m = metrics['resilience_to_congestion']
			cm = metrics['fraction_congested_volume']
			latency_delta_meds = []			
			latency_stds = []
			congestion_meds = []
			congestion_stds = []
			for X_val in X_vals:
				all_lats, all_vols, all_congestions = [], [], []
				for ri in range(N_TO_SIM):
					for latset,volset in m[ri][solution][X_val]:
						for lat, vol in zip(latset,volset):
							# if lat == 0: continue
							all_lats.append(lat)
							all_vols.append(vol)
					for congestion in cm[ri][solution][X_val]:
						all_congestions.append(congestion)
				x,cdf_x = get_cdf_xy(list(zip(all_lats,all_vols)), weighted=True)
				if X_val == single_X_of_interest:
					ax[FLASH_CROWD_SINGLE_X_I].plot(x,cdf_x,label="{} Metro increase pct. = {}".format(solution,single_X_of_interest))

				x = np.array(x)
				cdf_x = np.array(cdf_x)
				# try:
				# 	med = x[np.where(cdf_x > .5)[0][0]]
				# except IndexError:
				# 	med = x[-1]
				med = np.average(all_lats, weights=all_vols)
				latency_delta_meds.append(med)
				# latency_stds.append((x[np.where(cdf_x > .25)[0][0]], x[np.where(cdf_x > .75)[0][0]]))

				# print(all_congestions)
				# exit(0)
				x,cdf_x = get_cdf_xy(all_congestions)
				x = np.array(x)
				cdf_x = np.array(cdf_x)
				try:
					congestion_meds.append(x[np.where(cdf_x > .5)[0][0]])
					# congestion_stds.append((x[np.where(cdf_x > .25)[0][0]], x[np.where(cdf_x > .75)[0][0]]))
				except IndexError:
					congestion_meds.append(x[0])

			ax[FLASH_CROWD_LATENCY_VARY_X_I].plot(X_vals, latency_delta_meds, label=solution)
			ax[FLASH_CROWD_CONGESTION_VARY_X_I].plot(X_vals, congestion_meds, label=solution)

			# axtwin.plot(X_vals, congestion_meds, linestyle='--')
			
			##### Number of advertisements changed with flash crowds
			m = metrics['prefix_withdrawals']
			all_withdrawal_ns = []
			for ri in range(N_TO_SIM):
				for metro_set in m[ri][solution][single_X_of_interest]:
					total_withdrawals = sum(len(el) for el in metro_set)
					all_withdrawal_ns.append(total_withdrawals)

			x,cdf_x = get_cdf_xy(all_withdrawal_ns)
			ax[FLASH_CROWD_PREFIX_I].plot(x,cdf_x,label="{} Metro increase pct. = {}".format(solution,single_X_of_interest))


			#### Ability to grow with volume
			all_multipliers = []
			m = metrics['volume_multipliers']
			for ri in range(N_TO_SIM):
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

	ax[POPP_FAILURE_LATENCY_I].legend(fontsize=8)
	ax[POPP_FAILURE_LATENCY_I].grid(True)
	ax[POPP_FAILURE_LATENCY_I].set_xlabel("Latency Change Under Single-Link Failure (old - new) (ms)")
	ax[POPP_FAILURE_LATENCY_I].set_ylabel("CDF of UGs,Links")
	ax[POPP_FAILURE_LATENCY_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	# ax[POPP_FAILURE_LATENCY_I].set_xlim([-100,0])

	ax[POPP_FAILURE_CONGESTION_I].legend(fontsize=8)
	ax[POPP_FAILURE_CONGESTION_I].grid(True)
	ax[POPP_FAILURE_CONGESTION_I].set_xlabel("Fraction Congested Volume Single-Link Failure (old - new) (ms)")
	ax[POPP_FAILURE_CONGESTION_I].set_ylabel("CDF of UGs,Links")
	ax[POPP_FAILURE_CONGESTION_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	ax[POP_FAILURE_LATENCY_I].legend(fontsize=8)
	ax[POP_FAILURE_LATENCY_I].grid(True)
	ax[POP_FAILURE_LATENCY_I].set_xlabel("Latency Change Under Single-PoP Failure (old - new) (ms)")
	ax[POP_FAILURE_LATENCY_I].set_ylabel("CDF of UGs,PoPs")
	ax[POP_FAILURE_LATENCY_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	# ax[POP_FAILURE_LATENCY_I].set_xlim([-100,0])

	ax[POP_FAILURE_CONGESTION_I].legend(fontsize=8)
	ax[POP_FAILURE_CONGESTION_I].grid(True)
	ax[POP_FAILURE_CONGESTION_I].set_xlabel("Fraction Congested Volume Single-PoP Failure (old - new) (ms)")
	ax[POP_FAILURE_CONGESTION_I].set_ylabel("CDF of UGs,PoPs")
	ax[POP_FAILURE_CONGESTION_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	ax[LATENCY_I].legend(fontsize=8)
	ax[LATENCY_I].grid(True)
	ax[LATENCY_I].set_xlabel("Best - Actual Latency (ms)")
	ax[LATENCY_I].set_ylabel("CDF of UGs")
	ax[LATENCY_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	ax[FLASH_CROWD_SINGLE_X_I].set_xlabel("Latency Change under Flash Crowd (new - old) (ms)")
	ax[FLASH_CROWD_SINGLE_X_I].set_ylabel("CDF of UGs")
	ax[FLASH_CROWD_SINGLE_X_I].grid(True)
	ax[FLASH_CROWD_SINGLE_X_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[FLASH_CROWD_SINGLE_X_I].legend(fontsize=8)

	ax[FLASH_CROWD_PREFIX_I].set_xlabel("Number of Prefix Withdrawals for Flash Crowd")
	ax[FLASH_CROWD_PREFIX_I].set_ylabel("CDF of Metros")
	ax[FLASH_CROWD_PREFIX_I].grid(True)
	ax[FLASH_CROWD_PREFIX_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[FLASH_CROWD_PREFIX_I].legend(fontsize=8)

	ax[FLASH_CROWD_LATENCY_VARY_X_I].set_xlabel("Increase in Traffic per Metro for Flash Crowd (pct.)")
	ax[FLASH_CROWD_LATENCY_VARY_X_I].set_ylabel("Median Latency Change \nunder Flash Crowd (new - old) (ms)")
	ax[FLASH_CROWD_LATENCY_VARY_X_I].grid(True)
	ax[FLASH_CROWD_LATENCY_VARY_X_I].legend(fontsize=8)

	ax[FLASH_CROWD_CONGESTION_VARY_X_I].set_xlabel("Increase in Traffic per Metro for Flash Crowd (pct.)")
	ax[FLASH_CROWD_CONGESTION_VARY_X_I].set_ylabel("Median Delta Fraction Congested \n Traffic under Flash Crowd (new - old)")
	ax[FLASH_CROWD_CONGESTION_VARY_X_I].grid(True)
	ax[FLASH_CROWD_CONGESTION_VARY_X_I].legend(fontsize=8)

	

	ax[VOLUME_GROWTH_I].legend(fontsize=8)
	ax[VOLUME_GROWTH_I].grid(True)
	ax[VOLUME_GROWTH_I].set_xlabel("Volume Growth Capability")
	ax[VOLUME_GROWTH_I].set_ylabel("CDF of Simulations")
	ax[VOLUME_GROWTH_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	save_fig("just_prior_failure_comparison_{}.pdf".format(DPSIZE))


if __name__ == "__main__":
	popp_failure_latency_comparisons()
