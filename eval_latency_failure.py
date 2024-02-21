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

def assess_resilience_to_flash_crowds(sas, adv, solution, X_vals):
	## !!!!!!for painter/TIPSY!!!!!!
	## assume each metro's volume increases by X times on average
	## see if there's a solution
	## if there's a solution, do it and note the latency penalty compared to optimal

	# return cdf of latency penalties, possibly as a function of X
	metrics = {X:[] for X in X_vals}
	prefix_withdrawals = {X:[] for X in X_vals}
	fraction_congested_volumes = {X:[] for X in X_vals}

	adv = threshold_a(adv)

	base_soln = sas.solve_lp_with_failure_catch(adv)
	for X in X_vals:
		vol_by_metro = {}
		for metro,asn in sas.ugs:
			try:
				vol_by_metro[metro] += sas.ug_to_vol[(metro,asn)]
			except KeyError:
				vol_by_metro[metro] = sas.ug_to_vol[(metro,asn)]

		for metro,vol in tqdm.tqdm(vol_by_metro.items(), 
			desc="Assessing resilience to flash crowds by metro for {},X={}".format(solution,X)):
			prefix_withdrawals[X].append([])
			deployment = sas.output_deployment()
			save_deployment = copy.deepcopy(deployment)
			for ug,v in deployment['ug_to_vol'].items():
				if ug[0] == metro:
					deployment['ug_to_vol'][ug] = v * (1 + X/100)
			# print(deployment['ug_to_vol'])
			# print(deployment['link_capacities'])
			# exit(0)
			sas.update_deployment(deployment, verb=False, exit_on_impossible=False,
				quick_update=True)
			pre_soln = sas.solve_lp_with_failure_catch(adv)

			inundated_popps = [sas.popps[poppi] for poppi,v in pre_soln['vols_by_poppi'].items() if v > 1]
			# doprnt = False
			# if len(inundated_popps) > 0:
			# 	doprnt = True
			# 	print("{} inundated popps".format(len(inundated_popps)))
			soln = None
			soln_adv = copy.deepcopy(adv)
			ts, max_time, max_n_tries = time.time(), 10, 100 # seconds
			n = 0
			if DPSIZE == 'really_friggin_small':
				max_time = 1
			if False:

				#### Would make multiprocesing quite annoying
				while len(inundated_popps) > 0:
					### Greedily search to find a solution to the problem with TIPSY
					popp = inundated_popps[np.random.choice(list(range(len(inundated_popps))))]
					# print("trying drawing to withdraw a prefix from {}".format(popp))
					soln = compute_optimal_prefix_withdrawals(sas, soln_adv, popp, pre_soln,
						allow_congestion=True, can_withdraw_none=False)
					n+=1
					if soln is not None:
						# print("withdrawing {}".format(soln['prefix_withdrawals']))
						soln_adv[sas.popp_to_ind[popp], soln['prefix_withdrawals']] = 0
						prefix_withdrawals[X][-1].append(soln['prefix_withdrawals'])
						new_soln = sas.solve_lp_with_failure_catch(soln_adv)
						inundated_popps = [sas.popps[poppi] for poppi,v in new_soln['vols_by_poppi'].items() if v > 1]
						# print("{} inundated popps after withdrawal".format(len(inundated_popps)))
					if time.time() - ts > max_time or n > max_n_tries:
						break
			# if doprnt:
			# 	print("After termination, {} inundated popps".format(len(inundated_popps)))
			## characterize net latency changes after all the prefix withdrawals,
			## note that there is potentially no change at all
			## affected users get really bad latency
			latency_deltas = []
			vols = []
			new_soln = sas.solve_lp_with_failure_catch(soln_adv)
			ugi=0
			# print(new_soln['vols_by_poppi'][79])
			for old_lat, new_lat, vol in zip(base_soln['lats_by_ug'], new_soln['lats_by_ug'], sas.ug_vols):
				latency_deltas.append(new_lat - old_lat)
				# if new_lat - old_lat > 200:
				# 	print("{} {} {} {}".format(sas.ugs[ugi], new_lat - old_lat, base_soln['paths_by_ug'][ugi],
				# 		new_soln['paths_by_ug'][ugi]))
				vols.append(vol)
				ugi += 1
			fraction_congested_volumes[X].append(new_soln['fraction_congested_volume'] - base_soln['fraction_congested_volume'])
			metrics[X].append((latency_deltas, vols))
			# print(fraction_congested_volumes[X])

			sas.update_deployment(save_deployment, verb=False, exit_on_impossible=False,
				quick_update=True)

	return {
		'metrics': metrics,
		'prefix_withdrawals':prefix_withdrawals, 
		'fraction_congested_volume': fraction_congested_volumes,
	}

def assess_resilience_to_congestion(sas, adv, solution, X_vals):
	## !!!!!!for painter/TIPSY!!!!!!
	## assume each link is congested by X% (i.e., need to move X% of capacity's traffic off of it)
	## see if there's a solution
	## if there's a solution, do it and note the latency penalty compared to optimal

	# return cdf of latency penalties, possibly as a function of X
	metrics = {X:[] for X in X_vals}

	adv = threshold_a(adv)

	for X in X_vals:
		_, pre_ug_catchments = sas.calculate_user_choice(adv)
		# pre_lats_by_ug = solve_lp_with_failure_catch(sas, adv)['lats_by_ug']
		popp_to_ug_vols = {}
		for ugi, catchivols in pre_ug_catchments.items():
			for catchi,vol in catchivols:
				try:
					popp_to_ug_vols[catchi].append((ugi,vol))
				except KeyError:
					popp_to_ug_vols[catchi] = [(ugi,vol)]
		for popp in sas.popps:
			poppi = sas.popp_to_ind[popp]
			ugivols = popp_to_ug_vols.get(poppi,[])
			these_ugis = np.array(list(set(ugi for ugi,_ in ugivols)))
			if len(these_ugis) == 0 or len(np.where(adv[poppi,:])[0]) == 0:
				# nothing to solve
				continue
			current_link_volume = sum(vol for _,vol in ugivols)
			new_link_cap = current_link_volume * (1- X/100)
			
			pre_soln = sas.solve_lp_with_failure_catch(adv)
			save_cap = copy.copy(sas.link_capacities_arr[sas.popp_to_ind[popp]])
			sas.link_capacities_arr[sas.popp_to_ind[popp]] = new_link_capacity
			soln = compute_optimal_prefix_withdrawals(sas, adv, popp, pre_soln)
			sas.link_capacities_arr[sas.popp_to_ind[popp]] = save_cap

			if soln is None:
				print("Didn't get solution for popp {}, soln type {}, allowing overutilization".format(popp,solution))
				soln = compute_optimal_prefix_withdrawals(sas,adv,popp,new_link_cap,
					allow_congestion=True)
				if soln is None:
					raise ValueError("Still no solution even when we allow overutilization..")
			elif soln is None:
				## assign badness
				print("Didn't get solution for popp {}, soln type {}".format(popp,solution))
				metrics[X].append(list(NO_ROUTE_LATENCY * np.ones((len(these_ugis)))))
				continue

			## summarize solutions
			pfx_withdrawals = soln['prefix_withdrawals']
			if len(pfx_withdrawals) > 0:
				adv[poppi,np.array(pfx_withdrawals)] = 0
				_, post_ug_catchments = sas.calculate_user_choice(adv)
				users_of_interest = [ui for ui,catch in post_ug_catchments.items() if catch != pre_ug_catchments[ui]]
				adv[poppi,np.array(pfx_withdrawals)] = 1

			#### Each element of the list is (delta, volume) 
			metrics[X].append(list(soln['latency_deltas']))

	return {
		'metrics': metrics,
	}

def popp_failure_latency_comparisons():
	# for each deployment, get different advertisement strategies
	# look at latency under popp and link failures compared to optimal
	# sparse should be simultaneously better in both normal and failure scenarios

	# np.random.seed(31413)
	metrics = {}

	X_vals = [40,60,80,100,120,150,180,200,250,300,350,400,450,500,600,650]#[10,40,80,100,130,150,180,200,210,220,250]#[10,20,30,40,50,60,70,80,90,100]
	# if DPSIZE == 'really_friggin_small':
		# X_vals = [10,15,30,35,40,45,50,55,60,65,70]

	wm = None
	sas = None
	
	default_metrics = {
		'compare_rets': {i:None for i in range(N_TO_SIM)},
		'adv': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'deployment': {i:None for i in range(N_TO_SIM)},
		'ug_to_vol': {i:None for i in range(N_TO_SIM)},
		'settings': {i:None for i in range(N_TO_SIM)},
		'popp_failures_latency_optimal': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'popp_failures_latency_before': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'popp_failures_congestion': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'pop_failures_latency_optimal': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'pop_failures_latency_before': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
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
				 dont_update_deployment=True)
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

	# these_ris = [0]
	# try:
	# 	for random_iter in these_ris:
	# 		if sas is None:
	# 			deployment = metrics['deployment'][random_iter]

	# 			n_prefixes = deployment_to_prefixes(deployment)
	# 			sas = Sparse_Advertisement_Eval(deployment, verbose=True,
	# 				lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
	# 				using_resilience_benefit=True, gamma=gamma, n_prefixes=n_prefixes)
	# 			if wm is None:
	# 				wm = Worker_Manager(sas.get_init_kwa(), deployment)
	# 				wm.start_workers()
	# 			sas.set_worker_manager(wm)
	# 			sas.update_deployment(deployment)
	# 		else:
	# 			deployment = metrics['deployment'][random_iter]
	# 			sas.update_deployment(deployment)
	# 		for solution in soln_types:
	# 			adv = metrics['adv'][random_iter][solution]
	# 			if len(adv) == 0:
	# 				print("No solution for {}".format(solution))
	# 				continue
	# 			obj = sas.modeled_objective(adv)
	# 			print("Solution {} obj {}".format(solution, obj))
	# 		n_ug_to_prnt = 2
	# 		top_ugs = {ug: None for ug in sorted(sas.ugs, key = lambda el : -1 * sas.ug_to_vol[el])[0:n_ug_to_prnt]}
	# 		print(top_ugs)
	# 		for ug in top_ugs:
	# 			for infltn,v,ug,popp in metrics['popp_failures_latency'][random_iter]['sparse']:
	# 				try:
	# 					top_ugs[ug]
	# 				except KeyError:
	# 					continue
	# 				print("UG {}, vol {}, popp: {} inflation {}".format(ug, v, popp, infltn))
	# except:
	# 	import traceback
	# 	traceback.print_exc()

	RECALC_FAILURE_METRICS = False
	try:
		for random_iter in range(N_TO_SIM):
			break
			havent_calced_everything = False
			k_of_interest = 'popp_failures_latency_optimal'
			for solution in soln_types:
				if metrics[k_of_interest][random_iter][solution]  == \
					default_metrics[k_of_interest][random_iter][solution]:
					havent_calced_everything = True
					break

			if RECALC_FAILURE_METRICS or havent_calced_everything:
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
					if not RECALC_FAILURE_METRICS:
						if metrics[k_of_interest][random_iter][solution]  != \
							default_metrics[k_of_interest][random_iter][solution]:
							print("Already calced {}".format(solution))
							continue
					print(solution)
					adv = metrics['adv'][random_iter][solution]
					if len(adv) == 0:
						print("No solution for {}".format(solution))
						continue
						
					ret = assess_failure_resilience(sas, adv, which='popps')
					metrics['popp_failures_congestion'][random_iter][solution] = ret['congestion_delta']
					metrics['popp_failures_latency_optimal'][random_iter][solution] = ret['latency_delta_optimal']
					metrics['popp_failures_latency_before'][random_iter][solution] = ret['latency_delta_before']

					ret = assess_failure_resilience(sas, adv, which='pops')
					metrics['pop_failures_congestion'][random_iter][solution] = ret['congestion_delta']
					metrics['pop_failures_latency_optimal'][random_iter][solution] = ret['latency_delta_optimal']
					metrics['pop_failures_latency_before'][random_iter][solution] = ret['latency_delta_before']

					pickle.dump(metrics, open(performance_metrics_fn,'wb'))

	except:
		import traceback
		traceback.print_exc()

	RECALC_VOL_MULTIPLIERS = False
	multiply_values = np.linspace(1.1,10)
	try:
		for random_iter in range(N_TO_SIM):
			break
			if RECALC_VOL_MULTIPLIERS or metrics['volume_multipliers'][random_iter]['sparse']  == \
				default_metrics['volume_multipliers'][random_iter]['sparse']:
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
			havent_calced_everything = False
			for solution in soln_types:
				if metrics['resilience_to_congestion'][random_iter][solution]  == \
					default_metrics['resilience_to_congestion'][random_iter][solution]:
					havent_calced_everything = True
					break
			if RECALC_RESILIENCE or havent_calced_everything:
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
					if not RECALC_RESILIENCE:
						if metrics['resilience_to_congestion'][random_iter][solution]  != \
							default_metrics['resilience_to_congestion'][random_iter][solution]:
							print("Already calced {}".format(solution))
							continue
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
	
	i=0
	LATENCY_I = i;i+=1
	POPP_FAILURE_LATENCY_OPTIMAL_I = i;i+=1
	POPP_FAILURE_LATENCY_BEFORE_I = i;i+=1
	POPP_FAILURE_CONGESTION_I = i;i+=1
	POP_FAILURE_LATENCY_OPTIMAL_I = i;i+=1
	POP_FAILURE_LATENCY_BEFORE_I = i;i+=1
	POP_FAILURE_CONGESTION_I = i;i+=1
	FLASH_CROWD_SINGLE_X_I = i;i+=1
	FLASH_CROWD_LATENCY_VARY_X_I = i;i+=1
	FLASH_CROWD_CONGESTION_VARY_X_I = i;i+=1
	FLASH_CROWD_PREFIX_I = i;i+=1
	VOLUME_GROWTH_I = i;i+=1

	n_subs = i
	f,ax=plt.subplots(n_subs,1)
	f.set_size_inches(6,4*n_subs)

	## we plot the performance changes for a single flash crowd volume increase
	single_X_of_interest = 350
	SIM_INDS_TO_PLOT = [0]

	#### Plotting everything
	for solution in soln_types:
		print(solution)
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
			print("Average latency compared to optimal : {}".format(np.average(diffs, weights=wts)))
			
			#### Resilience to PoP and PoPP failures
			## popp failures (latency)
			all_differences = [(metrics['popp_failures_latency_optimal'][ri][solution][i][0],metrics['popp_failures_latency_optimal'][ri][solution][i][1]) for ri in SIM_INDS_TO_PLOT for 
				i in range(len(metrics['popp_failures_latency_optimal'][ri][solution]))]
			x,cdf_x = get_cdf_xy(all_differences,weighted=True)
			ax[POPP_FAILURE_LATENCY_OPTIMAL_I].plot(x,cdf_x,label=solution)
			print("Average latency compared to optimal under single-link failure : {}".format(np.average([el[0] for el in all_differences], weights=[el[1] for el in all_differences])))
			all_differences = [(metrics['popp_failures_latency_before'][ri][solution][i][0],metrics['popp_failures_latency_before'][ri][solution][i][1]) for ri in SIM_INDS_TO_PLOT for 
				i in range(len(metrics['popp_failures_latency_before'][ri][solution]))]
			x,cdf_x = get_cdf_xy(all_differences,weighted=True)
			ax[POPP_FAILURE_LATENCY_BEFORE_I].plot(x,cdf_x,label=solution)
			print("Average latency compared to before under single-link failure : {}".format(np.average([el[0] for el in all_differences], weights=[el[1] for el in all_differences])))
			## popp failures (congestion)
			all_differences = [metrics['popp_failures_congestion'][ri][solution][i] for ri in SIM_INDS_TO_PLOT for 
				i in range(len(metrics['popp_failures_congestion'][ri][solution]))]
			print("{} links had any congestion".format(sum(1 for el in all_differences if el != 0)))
			x,cdf_x = get_cdf_xy(all_differences,weighted=False)
			ax[POPP_FAILURE_CONGESTION_I].plot(x,cdf_x,label=solution)
			## pop failures (latency)
			all_differences = [(metrics['pop_failures_latency_optimal'][ri][solution][i][0],metrics['pop_failures_latency_optimal'][ri][solution][i][1]) for ri in SIM_INDS_TO_PLOT for 
				i in range(len(metrics['pop_failures_latency_optimal'][ri][solution]))]
			x,cdf_x = get_cdf_xy(all_differences,weighted=True)
			ax[POP_FAILURE_LATENCY_OPTIMAL_I].plot(x,cdf_x,label=solution)
			print("Average latency compared to optimal under single-PoP failure : {}".format(np.average([el[0] for el in all_differences], weights=[el[1] for el in all_differences])))
			## pop failures (latency)
			all_differences = [(metrics['pop_failures_latency_before'][ri][solution][i][0],metrics['pop_failures_latency_before'][ri][solution][i][1]) for ri in SIM_INDS_TO_PLOT for 
				i in range(len(metrics['pop_failures_latency_before'][ri][solution]))]
			x,cdf_x = get_cdf_xy(all_differences,weighted=True)
			ax[POP_FAILURE_LATENCY_BEFORE_I].plot(x,cdf_x,label=solution)
			print("Average latency compared to before under single-PoP failure : {}".format(np.average([el[0] for el in all_differences], weights=[el[1] for el in all_differences])))
			## pop failures (congestion)
			all_differences = [metrics['pop_failures_congestion'][ri][solution][i] for ri in SIM_INDS_TO_PLOT for 
				i in range(len(metrics['pop_failures_congestion'][ri][solution]))]
			x,cdf_x = get_cdf_xy(all_differences,weighted=False)
			ax[POP_FAILURE_CONGESTION_I].plot(x,cdf_x,label=solution)

			#### Resilience to flash crowds
			m = metrics['resilience_to_congestion']
			cm = metrics['fraction_congested_volume']
			latency_delta_meds = []			
			latency_stds = []
			congestion_meds = []
			congestion_stds = []
			for X_val in X_vals:
				all_lats, all_vols, all_congestions = [], [], []
				for ri in SIM_INDS_TO_PLOT:
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
				try:
					# med = x[np.where(cdf_x > .5)[0][0]]
					med = np.average(all_lats, weights=all_vols)
				except IndexError:
					med = x[-1]
					import traceback
					traceback.print_exc()
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
			for ri in SIM_INDS_TO_PLOT:
				for metro_set in m[ri][solution][single_X_of_interest]:
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
	ax[LATENCY_I].set_ylabel("CDF of UGs")
	ax[LATENCY_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[LATENCY_I].set_xlim([-100,0])

	ax[POPP_FAILURE_LATENCY_OPTIMAL_I].legend(fontsize=8)
	ax[POPP_FAILURE_LATENCY_OPTIMAL_I].grid(True)
	ax[POPP_FAILURE_LATENCY_OPTIMAL_I].set_xlabel("Latency Change Under Single-Link Failure (best - actual) (ms)")
	ax[POPP_FAILURE_LATENCY_OPTIMAL_I].set_ylabel("CDF of UGs,Links")
	ax[POPP_FAILURE_LATENCY_OPTIMAL_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POPP_FAILURE_LATENCY_OPTIMAL_I].set_xlim([-100,0])

	ax[POPP_FAILURE_LATENCY_BEFORE_I].legend(fontsize=8)
	ax[POPP_FAILURE_LATENCY_BEFORE_I].grid(True)
	ax[POPP_FAILURE_LATENCY_BEFORE_I].set_xlabel("Latency Change Under Single-Link Failure (old - new) (ms)")
	ax[POPP_FAILURE_LATENCY_BEFORE_I].set_ylabel("CDF of UGs,Links")
	ax[POPP_FAILURE_LATENCY_BEFORE_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[POPP_FAILURE_LATENCY_BEFORE_I].set_xlim([-100,0])

	ax[POPP_FAILURE_CONGESTION_I].legend(fontsize=8)
	ax[POPP_FAILURE_CONGESTION_I].grid(True)
	ax[POPP_FAILURE_CONGESTION_I].set_xlabel("Fraction Congested Volume Single-Link Failure (best - actual) (ms)")
	ax[POPP_FAILURE_CONGESTION_I].set_ylabel("CDF of UGs,Links")
	ax[POPP_FAILURE_CONGESTION_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	ax[POP_FAILURE_LATENCY_OPTIMAL_I].legend(fontsize=8)
	ax[POP_FAILURE_LATENCY_OPTIMAL_I].grid(True)
	ax[POP_FAILURE_LATENCY_OPTIMAL_I].set_xlabel("Latency Change Under Single-PoP Failure (best - actual) (ms)")
	ax[POP_FAILURE_LATENCY_OPTIMAL_I].set_ylabel("CDF of UGs,PoPs")
	ax[POP_FAILURE_LATENCY_OPTIMAL_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	ax[POP_FAILURE_LATENCY_BEFORE_I].legend(fontsize=8)
	ax[POP_FAILURE_LATENCY_BEFORE_I].grid(True)
	ax[POP_FAILURE_LATENCY_BEFORE_I].set_xlabel("Latency Change Under Single-PoP Failure (old - new) (ms)")
	ax[POP_FAILURE_LATENCY_BEFORE_I].set_ylabel("CDF of UGs,PoPs")
	ax[POP_FAILURE_LATENCY_BEFORE_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	ax[POP_FAILURE_CONGESTION_I].legend(fontsize=8)
	ax[POP_FAILURE_CONGESTION_I].grid(True)
	ax[POP_FAILURE_CONGESTION_I].set_xlabel("Fraction Congested Volume Single-PoP Failure (best - actual) (ms)")
	ax[POP_FAILURE_CONGESTION_I].set_ylabel("CDF of UGs,PoPs")
	ax[POP_FAILURE_CONGESTION_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])



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

		pre_soln = sas.solve_lp_with_failure_catch(adv)
		pre_lats_by_ug = pre_soln['lats_by_ug']
		pre_ug_decisions = pre_soln['paths_by_ug']

		metrics['latencies'] = pre_lats_by_ug
		if np.median(metrics['latencies']) > 300:
			pickle.dump([adv, pre_soln], open('cache/weird_no_route.pkl','wb'))
		metrics['best_latencies'] = copy.copy(sas.best_lats_by_ug)
		metrics['popp_failures'] = []

		diffs = list(metrics['best_latencies'] - metrics['latencies'])
		x,cdf_x = get_cdf_xy(list(zip(diffs, ug_vols_arr)), weighted=True)
		ax[1].plot(x,cdf_x, label=labs[i])

		for popp in tqdm.tqdm(sas.popps, desc="Plotting lats from advs..."):
			adv_cpy = np.copy(adv)
			adv_cpy[sas.popp_to_ind[popp]] = 0
			## q: what is latency experienced for these ugs compared to optimal?
			one_per_ingress_adv = np.identity(sas.n_popps)
			one_per_ingress_adv[sas.popp_to_ind[popp]] = 0
			ret = sas.solve_lp_with_failure_catch(one_per_ingress_adv,
				computing_best_lats=True)
			failed_best_lats_by_ug = ret['lats_by_ug']

			fail_soln = sas.solve_lp_with_failure_catch(adv_cpy)
			post_ug_decisions = fail_soln['paths_by_ug']
			post_lats_by_ug = fail_soln['lats_by_ug']

			# ## Look at users whose catchment has changed
			# these_ugs = [ugi for ugi in post_ug_decisions if \
			# 	pre_ug_decisions[ugi] != post_ug_decisions[ugi]]
			these_ugs = [sas.ug_to_ind[ug] for ug in sas.ugs]

			### TODO -- what to do about failed users?

			inundated=False
			for ugi in list(set(these_ugs)):
				metrics['popp_failures'].append((failed_best_lats_by_ug[ugi] - \
					post_lats_by_ug[ugi], sas.ug_vols[ugi]))

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
	np.random.seed(31415)
	popp_failure_latency_comparisons()
