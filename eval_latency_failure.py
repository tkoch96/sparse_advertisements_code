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
			if dpsize == 'really_friggin_small':
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

def evaluate_all_metrics(dpsize, save_run_dir=None, **kwargs):
	# for each deployment, get different advertisement strategies
	# look at latency under popp and link failures compared to optimal
	# sparse should be simultaneously better in both normal and failure scenarios

	# np.random.seed(31413)
	metrics = {}

	X_vals = np.linspace(10,500,num=10)#[10,40,80,100,130,150,180,200,210,220,250]#[10,20,30,40,50,60,70,80,90,100]
	Y_vals = np.linspace(1.01,1.5,num=10)
	# if dpsize == 'really_friggin_small':
		# X_vals = [10,15,30,35,40,45,50,55,60,65,70]

	wm = None
	sas = None

	performance_metrics_fn = kwargs.get('use_performance_metrics_fn', global_performance_metrics_fn(dpsize))
	soln_types = kwargs.get('soln_types', global_soln_types)
	if 'soln_types' in kwargs:
		del kwargs['soln_types']

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
				deployment = pickle.load(open(os.path.join(RUN_DIR, save_run_dir, 'state-0.pkl'), 'rb'))['deployment']
			elif kwargs.get('prefix_deployment') is not None:
				print("Prefixing deployment")
				deployment = kwargs.get('prefix_deployment')
			else:
				while True:
					try:
						deployment = get_random_deployment(dpsize, **kwargs)
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
				save_run_dir=save_run_dir)

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

	RECALC_FAILURE_METRICS = True
	try:
		changed=False
		for random_iter in range(N_TO_SIM):
			k_of_interest = 'popp_failures_latency_optimal_specific'
			havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)

			if RECALC_FAILURE_METRICS or havent_calced_everything:
				print("-----Failure calc for deployment number = {} -------".format(random_iter))
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

	# RECALC_FAILURE_METRICS_HIGH_CAP = False
	# try:
	# 	for random_iter in range(N_TO_SIM):
	# 		k_of_interest = 'popp_failures_high_cap_latency_optimal'
	# 		havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)

	# 		if RECALC_FAILURE_METRICS or havent_calced_everything:
	# 			if sas is None:
	# 				deployment = metrics['deployment'][random_iter]

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
	# 				sas.update_deployment(deployment)

	# 			for solution in soln_types:
	# 				if not RECALC_FAILURE_METRICS:
	# 					if metrics[k_of_interest][random_iter][solution]  != \
	# 						default_metrics[k_of_interest][0][solution]:
	# 						print("Already calced {}".format(solution))
	# 						continue
	# 				print("Solving failure {} with infinite cap.".format(solution))
	# 				save_caps = deployment['link_capacities'].copy()
	# 				## solve the problem with infinite capacity
	# 				deployment['link_capacities'] = 100000*np.ones(len(deployment['popps']))
	# 				adv = metrics['adv'][random_iter][solution]
	# 				if len(adv) == 0:
	# 					print("No solution for {}".format(solution))
	# 					continue
						
	# 				ret = assess_failure_resilience(sas, adv, which='popps')
	# 				metrics['popp_failures_high_cap_latency_optimal'][random_iter][solution] = ret['mutable']['latency_delta_optimal']
	# 				metrics['popp_failures_high_cap_latency_optimal_specific'][random_iter][solution] = ret['mutable']['latency_delta_specific']

	# 				ret = assess_failure_resilience(sas, adv, which='pops')
	# 				metrics['pop_failures_high_cap_latency_optimal'][random_iter][solution] = ret['mutable']['latency_delta_optimal']
	# 				metrics['pop_failures_high_cap_latency_optimal_specific'][random_iter][solution] = ret['mutable']['latency_delta_specific']

	# 				deployment['link_capacities'] = save_caps

	# 				pickle.dump(metrics, open(performance_metrics_fn,'wb'))

	# except:
	# 	import traceback
	# 	traceback.print_exc()

	RECALC_VOL_MULTIPLIERS = False
	multiply_values = np.linspace(1.1,10)
	try:
		for random_iter in range(N_TO_SIM):
			break
			k_of_interest = 'volume_multipliers'
			havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)
			if RECALC_VOL_MULTIPLIERS or havent_calced_everything:
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
			k_of_xinterest = 'resilience_to_congestion'
			havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)
			if RECALC_RESILIENCE or havent_calced_everything:
				print("-----Flash crowd calc for deployment number = {} -------".format(random_iter))
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
					print("Assessing resilience to congestion for {}, sim number {}".format(solution,random_iter))
					print("Baseline congestion is {}".format(solve_lp_with_failure_catch(sas, adv)['fraction_congested_volume']))
					# m = assess_resilience_to_congestion(sas, adv, solution, X_vals)['metrics']
					m = assess_resilience_to_flash_crowds_mp(sas, adv, solution, X_vals, Y_vals, inflated_deployments)
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
	single_X_of_interest = X_vals[len(X_vals)//2]
	single_Y_of_interest = Y_vals[len(Y_vals)//2]
	SIM_INDS_TO_PLOT = list(range(N_TO_SIM))

	#### Plotting everything
	for k in list(metrics):
		if 'latency' in k or 'latencies' in k:
			metrics['stats_' + k] = {}
	interesting_latency_suboptimalities = [-10,-50,-100]
	for k in ['stats_latency_thresholds_normal', 'stats_latency_thresholds_fail_popp', 'stats_latency_thresholds_fail_pop']:
		metrics[k] = {solution: {i:{} for i in SIM_INDS_TO_PLOT} for solution in soln_types}

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
						this_diffs.append(perf1-perf2)
						this_vols.append(vol)
					if perf2 == NO_ROUTE_LATENCY:
						vol_congested += vol
						perf2=perf2*100
					all_vol += vol
					if diff > maxd:
						maxd=diff
					if diff < mind:
						mind=diff
				ret.append((perf1-perf2, vol))

			try:
				this_x,this_cdf_x = get_cdf_xy(list(zip(this_diffs,this_wts)), weighted=True)
				for lat_threshold in interesting_latency_suboptimalities:
					xi = np.argmin(np.abs(this_x-lat_threshold))
					threshold_stats[ri][lat_threshold] = this_cdf_x[xi]
			except IndexError:
				for lat_threshold in interesting_latency_suboptimalities:
					threshold_stats[ri][lat_threshold] = 0 ## no users are within any good latency


		x=np.linspace(mind,maxd,num=200)
		if vol_congested > 0:
			x[0] = -1*100*NO_ROUTE_LATENCY

		avg_latency_difference = np.average([el[0] for el in avg_ret], weights=[el[1] for el in avg_ret])
		print("Average latency difference {},{}: {}".format(solution, k, avg_latency_difference))
		print("{} pct. volume congested".format(round(100 * vol_congested / actually_all_vol, 2)))
		print("{} pct. optimally congested, all volume: {}".format(round(100 * vol_best_case_congested / actually_all_vol, 2), actually_all_vol))

		return ret, x, {
			'avg_latency_difference': avg_latency_difference, 
			'frac_vol_congested': vol_congested / all_vol, 
			'frac_vol_bestcase_congested': vol_best_case_congested / actually_all_vol,
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
			# ## popp failures (latency)
			# all_differences, x, stats, _ = get_failure_metric_arr('popp_failures_latency_optimal', solution)
			# metrics['stats_' + 'popp_failures_latency_optimal'][solution] = stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# ax[POPP_FAILURE_LATENCY_OPTIMAL_I].plot(x,cdf_x,label=solution)
			all_differences, x, stats, threshold_stats = get_failure_metric_arr('popp_failures_latency_optimal_specific', solution)
			metrics['stats_' + 'popp_failures_latency_optimal_specific'][solution] = stats
			metrics['stats_latency_thresholds_fail_popp'][solution] = threshold_stats
			x, cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POPP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)

			# all_differences, x, stats, _ = get_failure_metric_arr('popp_failures_high_cap_latency_optimal', solution)
			# metrics['stats_' + 'popp_failures_high_cap_latency_optimal'][solution] = stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# ax[POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I].plot(x,cdf_x,label=solution)
			# all_differences, x, stats, _ = get_failure_metric_arr('popp_failures_high_cap_latency_optimal_specific', solution, verb=solution=='sparse')
			# metrics['stats_' + 'popp_failures_high_cap_latency_optimal_specific'][solution] = stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# ax[POPP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)

			# all_differences, x, stats, threshold_stats = get_failure_metric_arr('popp_failures_latency_before', solution)
			# metrics['stats_' + 'popp_failures_latency_before'][solution] = stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# ax[POPP_FAILURE_LATENCY_BEFORE_I].plot(x,cdf_x,label=solution)
			# ## popp failures (congestion)
			# all_differences = [metrics['popp_failures_congestion'][ri][solution][i] for ri in SIM_INDS_TO_PLOT for 
			# 	i in range(len(metrics['popp_failures_congestion'][ri][solution]))]
			# print("{} links had any congestion".format(sum(1 for el in all_differences if el != 0)))
			# x,cdf_x = get_cdf_xy(all_differences,weighted=False)
			# ax[POPP_FAILURE_CONGESTION_I].plot(x,cdf_x,label=solution)
			
			## pop failures (latency)
			# all_differences, x, stats, threshold_stats = get_failure_metric_arr('pop_failures_latency_optimal', solution)
			# metrics['stats_' + 'pop_failures_latency_optimal'][solution] = stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# ax[POP_FAILURE_LATENCY_OPTIMAL_I].plot(x,cdf_x,label=solution)
			all_differences, x, stats, threshold_stats = get_failure_metric_arr('pop_failures_latency_optimal_specific', solution)
			metrics['stats_' + 'pop_failures_latency_optimal_specific'][solution] = stats
			metrics['stats_latency_thresholds_fail_pop'][solution] = threshold_stats
			x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			ax[POP_FAILURE_LATENCY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)

			# all_differences, x, stats, threshold_stats = get_failure_metric_arr('pop_failures_high_cap_latency_optimal', solution)
			# metrics['stats_' + 'pop_failures_high_cap_latency_optimal'][solution] = stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# ax[POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_I].plot(x,cdf_x,label=solution)
			# all_differences, x, stats, threshold_stats = get_failure_metric_arr('pop_failures_high_cap_latency_optimal_specific', solution)
			# metrics['stats_' + 'pop_failures_high_cap_latency_optimal_specific'][solution] = stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# ax[POP_FAILURE_HIGH_CAP_LATENCY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)

			# ## pop failures (latency)
			# all_differences, x, stats, threshold_stats = get_failure_metric_arr('pop_failures_latency_before', solution)
			# metrics['stats_' + 'pop_failures_latency_before'][solution] = stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# ax[POP_FAILURE_LATENCY_BEFORE_I].plot(x,cdf_x,label=solution)
			# ## pop failures (congestion)
			# all_differences = [metrics['pop_failures_congestion'][ri][solution][i] for ri in SIM_INDS_TO_PLOT for 
			# 	i in range(len(metrics['pop_failures_congestion'][ri][solution]))]
			# x,cdf_x = get_cdf_xy(all_differences,weighted=False)
			# ax[POP_FAILURE_CONGESTION_I].plot(x,cdf_x,label=solution)

			# ### STICKY
			# ## popp failures (latency)
			# all_differences, x, stats, threshold_stats = get_failure_metric_arr('popp_failures_sticky_latency_optimal', solution)
			# metrics['stats_' + 'popp_failures_sticky_latency_optimal'][solution] = stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# ax[POPP_FAILURE_STICKY_LATENCY_OPTIMAL_I].plot(x,cdf_x,label=solution)
			# all_differences, x, stats, threshold_stats = get_failure_metric_arr('popp_failures_sticky_latency_optimal_specific', solution)
			# metrics['stats_' + 'popp_failures_sticky_latency_optimal_specific'][solution] = stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# ax[POPP_FAILURE_STICKY_LATENCY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)
			# all_differences, x, stats, threshold_stats = get_failure_metric_arr('popp_failures_sticky_latency_before', solution)
			# metrics['stats_' + 'popp_failures_sticky_latency_before'][solution] = stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# ax[POPP_FAILURE_STICKY_LATENCY_BEFORE_I].plot(x,cdf_x,label=solution)
			# ## popp failures (congestion)
			# all_differences = [metrics['popp_failures_sticky_congestion'][ri][solution][i] for ri in SIM_INDS_TO_PLOT for 
			# 	i in range(len(metrics['popp_failures_sticky_congestion'][ri][solution]))]
			# print("{} links had any congestion (STICKY)".format(sum(1 for el in all_differences if el != 0)))
			# x,cdf_x = get_cdf_xy(all_differences,weighted=False)
			# ax[POPP_FAILURE_STICKY_CONGESTION_I].plot(x,cdf_x,label=solution)
			
			# ## pop failures (latency)
			# all_differences, x, stats, threshold_stats = get_failure_metric_arr('pop_failures_sticky_latency_optimal', solution)
			# metrics['stats_' + 'pop_failures_sticky_latency_optimal'][solution] = stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# ax[POP_FAILURE_STICKY_LATENCY_OPTIMAL_I].plot(x,cdf_x,label=solution)
			# all_differences, x, stats, threshold_stats = get_failure_metric_arr('pop_failures_sticky_latency_optimal_specific', solution)
			# metrics['stats_' + 'pop_failures_sticky_latency_optimal_specific'][solution] = stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# ax[POP_FAILURE_STICKY_LATENCY_OPTIMAL_SPECIFIC_I].plot(x,cdf_x,label=solution)
			# ## pop failures (latency)
			# all_differences, x, stats, threshold_stats = get_failure_metric_arr('pop_failures_sticky_latency_before', solution)
			# metrics['stats_' + 'pop_failures_sticky_latency_before'][solution] = stats
			# x,cdf_x = get_cdf_xy(all_differences,weighted=True,x=x)
			# ax[POP_FAILURE_STICKY_LATENCY_BEFORE_I].plot(x,cdf_x,label=solution)
			# ## pop failures (congestion)
			# all_differences = [metrics['pop_failures_sticky_congestion'][ri][solution][i] for ri in SIM_INDS_TO_PLOT for 
			# 	i in range(len(metrics['pop_failures_sticky_congestion'][ri][solution]))]
			# x,cdf_x = get_cdf_xy(all_differences,weighted=False)
			# ax[POP_FAILURE_STICKY_CONGESTION_I].plot(x,cdf_x,label=solution)


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
					all_lats, all_congestions = [], []
					for ri in SIM_INDS_TO_PLOT:
						for avg_lat_deltas in m[ri][solution][Y_val][X_val]:
							all_lats.append(avg_lat_deltas)
						for congestion in cm[ri][solution][Y_val][X_val]:
							all_congestions.append(congestion)
					# if X_val == single_X_of_interest and Y_val == single_Y_of_interest:
					# 	## Plot CDF for this specific X val and Y val
					# 	x,cdf_x = get_cdf_xy(list(zip(all_lats,all_vols)), weighted=True)
					# 	ax[FLASH_CROWD_SINGLE_X_I].plot(x,cdf_x,label="{} Metro increase pct. = {}".format(solution,single_X_of_interest))
						
					lat_med = np.average(all_lats)
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
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_run_dir", default=None)
	parser.add_argument("--use_cache_deployment", action='store_true', default=False)
	parser.add_argument("--dpsize", default=None, required=True)
	args = parser.parse_args()

	np.random.seed(31415)
	if args.save_run_dir is not None:
		## we could specify an array of hotstart dirs otherwise, but that's a task for another day
		assert N_TO_SIM == 1
		evaluate_all_metrics(args.dpsize, save_run_dir=args.save_run_dir)
	elif args.use_cache_deployment:
		deployment = pickle.load(open(global_performance_metrics_fn(dpsize), 'rb'))['deployment'][0]
		evaluate_all_metrics(args.dpsize,prefix_deployment=deployment)
	else:
		evaluate_all_metrics(args.dpsize)
