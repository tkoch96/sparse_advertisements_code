from deployment_setup import get_random_deployment, get_bulk_vol, get_link_capacities
from sparse_advertisements_v3 import Sparse_Advertisement_Eval
from worker_comms import Worker_Manager
from helpers import *
from constants import *
from paper_plotting_functions import *
from wrapper_eval import *


def gen_paper_plots():
	# metrics_fn = os.path.join(CACHE_DIR, 'joint_latency_bulk_download_metrics_small.pkl')
	metrics_fn = os.path.join(CACHE_DIR, 'joint_latency_bulk_download_metrics_actual-32.pkl')
	metrics = pickle.load(open(metrics_fn, 'rb'))

	f,ax = get_figure(l=3.5)
	b4_bv_of_interest = 6.0
	swan_bv_of_interest = 4.0
	bv_of_interest = 5.0
	solutions = ['anycast', 'anyopt', 'one_per_pop', 'painter', 'sparse', 'one_per_peering']
	SIM_INDS_TO_PLOT = [0]
	bulk_vals = sorted(list(metrics[0]['one_per_peering']['congestion_over_bulk_vals']))
	for solution in solutions:
		congestions = {bv:[] for bv in bulk_vals}
		for random_iter in SIM_INDS_TO_PLOT:
			m = metrics[random_iter][solution]
			for bv in bulk_vals:
				congestions[bv].append(m['congestion_over_bulk_vals'][bv])
		ax.plot(bulk_vals[::3], 
			list([np.mean(congestions[bv]) for bv in bulk_vals])[::3], marker=solution_to_marker[solution], 
			color=solution_to_line_color[solution],
			label=solution_to_plot_label[solution])
	# ax.legend(fontsize=10)
	ax.set_xlabel("LPrio/HPrio Ratio", fontsize=18)
	ax.set_ylabel("Fraction HPrio\nTraffic Congested", fontsize=18)
	ax.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])

	ax.text(b4_bv_of_interest-.75, .7, "B4\nRatio",fontsize=18)
	ax.axvline(bv_of_interest, 0,1, linestyle='--',color='black')

	ax.text(swan_bv_of_interest-2.7, .6, "SWAN\nRatio",fontsize=18)
	ax.axvline(swan_bv_of_interest, 0,1, linestyle='--',color='black')

	ax.grid(True)

	save_figure('priority_bulk_traffic_congestion.pdf')


	f,ax = get_figure(l=3.5)
	for solution in solutions[::-1]:
		diffs = []
		wts = []
		for random_iter in SIM_INDS_TO_PLOT:
			bvs = np.array(list(metrics[random_iter][solution]['bulk_latencies_over_bulk_vals']))
			bv = bvs[np.argmin(np.abs(bvs-bv_of_interest))]

			lp_solution = metrics[random_iter][solution]['bulk_latencies_over_bulk_vals'][bv]
			best_solution = metrics[random_iter]['one_per_peering']['bulk_latencies_over_bulk_vals'][bv]
			for opt_lat, achieved_lat, vol in zip(best_solution, lp_solution, metrics[random_iter]['ug_to_vol']):
				# diffs.append(opt_lat - achieved_lat)
				diffs.append(achieved_lat)
				wts.append(vol)
		if solution == 'anycast':
			x=np.linspace(0,250,num=100)
			cdf_x = np.zeros(100)
		else:
			x,cdf_x = get_cdf_xy(list(zip(diffs,wts)), weighted=True)
		ax.plot(x[::10], cdf_x[::10], marker=solution_to_marker[solution], 
			color=solution_to_line_color[solution],
			label=solution_to_plot_label[solution])

	ax.legend(fontsize=11, loc='lower right')
	# ax.set_xlabel("Optimal - Actual Latency (ms)")
	ax.set_xlabel("HPrio Latency (ms)", fontsize=18)
	ax.set_ylabel("CDF of Traffic", fontsize=18)
	ax.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	# ax.set_xlim([-100,20])
	ax.set_xlim([0,250])
	ax.grid(True)

	save_figure('priority_bulk_traffic_latency.pdf')



def check_calced_everything(metrics, random_iter, k_of_interest):
	soln_types = global_soln_types
	for solution in soln_types:
		try:
			metrics[random_iter][solution][k_of_interest]
		except KeyError:
			return False
	return True

def get_inflated_bulk_volumes(sas, bulk_volumes):
	out_deployments = []
	for bv in bulk_volumes:
		quick_deployment = sas.output_deployment(copykeys=['ug_to_vol','whole_deployment_ug_to_vol', 'ug_to_bulk_vol', 'whole_deployment_ug_to_bulk_vol'])
		quick_deployment['ug_to_bulk_vol'] = {ug:v*bv for ug,v in quick_deployment['ug_to_vol'].items()}
		quick_deployment['whole_deployment_ug_to_bulk_vol'] = {ug:v*bv for ug,v in quick_deployment['whole_deployment_ug_to_vol'].items()}
		quick_deployment['generic_objective'] = sas.generic_objective.obj
		out_deployments.append(quick_deployment)
	return out_deployments

def assess_performance_over_bulk_volumes(sas, adv, solution, bulk_volumes):
	## bulk volumes is [bv1, ...] where amount of bulk traffic is bvi * ug_vols

	# return cdf of latency penalties, possibly as a function of X
	congestion_over_bulk_vals = {bv: None for bv in bulk_volumes}
	bulk_latencies_over_bulk_vals = {bv: None for bv in bulk_volumes}

	adv = threshold_a(adv)
	inflated_deployments = get_inflated_bulk_volumes(sas, bulk_volumes)

	call_args = []
	for bv, d in zip(bulk_volumes, inflated_deployments):
		## always clear the deployment cache (True on third arg)
		call_args.append((adv, d, True))

	### Call all the solutions with multiprocessing
	all_rets = sas.solve_lp_with_failure_catch_mp(call_args)
	i=0
	print("Done, parsing return values from workers")
	for bv in bulk_volumes:
		congestion_over_bulk_vals[bv] = all_rets[i]['fraction_congested_volume_with_bulk']
		bulk_latencies_over_bulk_vals[bv] = all_rets[i]['bulk_lats_by_ug']
		i += 1
	return {
		'congestion_over_bulk_vals': congestion_over_bulk_vals,
		'bulk_latencies_over_bulk_vals': bulk_latencies_over_bulk_vals,
	}

def testing_priorities_one_pass(**kwargs):
	"""
		Tests minimization of LL + alpha * Bulk download in a one-pass optimization
	"""

	np.random.seed(31411)
	lambduh = .00001 ## unused more or less
	gamma = 0#2.0

	n_random_sim = 1
	dpsize = sys.argv[1]
	obj = 'joint_latency_bulk_download'
	performance_metrics_fn = os.path.join(CACHE_DIR, "{}_metrics_{}.pkl".format(obj,dpsize))
	if os.path.exists(performance_metrics_fn):
		metrics = pickle.load(open(performance_metrics_fn, 'rb'))
	else:
		metrics = {i:{'done':False} for i in range(n_random_sim)}
	soln_types = global_soln_types

	try:
		wm = None
		sas = None

		if True:
			port = int(sys.argv[2])
			for random_iter in range(n_random_sim):
				try:
					if metrics[random_iter]['done']: continue
				except KeyError:
					metrics[random_iter] = {'done': False}

				try:
					this_iter_deployment = metrics[random_iter]['deployment']
				except KeyError:
					this_iter_deployment = get_random_deployment(dpsize)
				this_iter_deployment['port'] = port
				print("Random deployment for joint latency bulk download, number {}/{}".format(random_iter+1,n_random_sim))
				
				deployment = copy.deepcopy(this_iter_deployment)
				metrics[random_iter] = {'deployment': deployment}
				pickle.dump(metrics, open(performance_metrics_fn, 'wb'))

				n_prefixes = deployment_to_prefixes(deployment)

				sas = Sparse_Advertisement_Eval(deployment, verbose=True,
					lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
					using_resilience_benefit=False, gamma=gamma, n_prefixes=n_prefixes,
					generic_objective=obj)

				metrics[random_iter]['settings'] = sas.get_init_kwa()
				if wm is None:
					wm = Worker_Manager(sas.get_init_kwa(), deployment)
					wm.start_workers()
				sas.set_worker_manager(wm)
				sas.update_deployment(deployment)
				### Solve the problem for each type of solution (sparse, painter, etc...)
				ret = sas.compare_different_solutions(n_run=1, verbose=True,
					dont_update_deployment=True, soln_types=soln_types)
				metrics[random_iter]['settings'] = sas.get_init_kwa()
				metrics[random_iter]['optimal_objective'] = sas.optimal_expensive_solution
				metrics[random_iter]['compare_rets'] = ret
				metrics[random_iter]['ug_to_vol'] = sas.ug_vols

				metrics[random_iter]['save_run_dir'] = sas.sas.save_run_dir # sparse's save run dir
				for solution in soln_types:
					metrics[random_iter][solution] = {}
					try:
						adv = ret['adv_solns'][solution][0]
					except:
						print("No solution for {}".format(solution))
						continue
					### compute the allocation with the generic objective, even with non-sparse solutions
					lp_solution = sas.sas.generic_objective.get_latency_benefit_adv(adv) 

					metrics[random_iter][solution]['adv'] = adv
					metrics[random_iter][solution]['lp_solution'] = lp_solution

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

	### Calculates how our metrics change as we vary the amount of bulk traffic we place
	RECALC_BULK_TRAFFIC_VARY = True
	bulk_vals = np.linspace(.1,9,num=60)
	try:
		changed=False
		for random_iter in range(N_TO_SIM):
			k_of_interest = 'bulk_traffic_vary'
			havent_calced_everything = check_calced_everything(metrics, random_iter, k_of_interest)
			if RECALC_BULK_TRAFFIC_VARY or havent_calced_everything:
				print("-----Bulk traffic varying calc for deployment number = {} -------".format(random_iter))
				if sas is None:
					deployment = metrics[random_iter]['deployment']
					deployment['port'] = port

					## tmp code
					link_capacities = get_link_capacities(deployment, scale_factor=5.0)
					deployment['link_capacities'] = link_capacities

					n_prefixes = kwargs.get('n_prefixes', deployment_to_prefixes(deployment))
					sas = Sparse_Advertisement_Eval(deployment, verbose=True,
						lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
						using_resilience_benefit=False, gamma=gamma, n_prefixes=n_prefixes,
						generic_objective=obj)
					if wm is None:
						wm = Worker_Manager(sas.get_init_kwa(), deployment)
						wm.start_workers()
					sas.set_worker_manager(wm)
					sas.update_deployment(deployment)
				else:
					deployment = metrics[random_iter]['deployment']
					deployment['port'] = port
					sas.update_deployment(deployment)
				
				ret = metrics[random_iter]['compare_rets']
				for solution in soln_types:
					try:
						adv = ret['adv_solns'][solution][0]
					except:
						print("No solution for {}".format(solution))
						continue
					### compute the allocation with the generic objective, even with non-sparse solutions
					lp_solution = sas.generic_objective.get_latency_benefit_adv(adv) 
					metrics[random_iter][solution]['lp_solution'] = lp_solution
					try:
						print("Bulk calc for {}...".format(solution))
						m = assess_performance_over_bulk_volumes(sas, adv, solution, bulk_vals)
						metrics[random_iter][solution]['congestion_over_bulk_vals'] = m['congestion_over_bulk_vals']
						metrics[random_iter][solution]['bulk_latencies_over_bulk_vals'] = m['bulk_latencies_over_bulk_vals']

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

	# (for each solution)
	### look at average latency for low-latency traffic
	### look at amount of low-latency traffic that ends up on congested links
	SIM_INDS_TO_PLOT = list(range(n_random_sim))

	################################
	### PLOTTING
	################################
	
	i=0
	LATENCY_I = i;i+=1
	LATENCY_WITH_CONGESTION = i; i+=1
	CONGESTION_OVER_BULK_TRAFFIC = i; i+=1

	n_subs = i
	f,ax=plt.subplots(n_subs,1)
	f.set_size_inches(6,4*n_subs)

	#### Plotting everything
	for k in list(metrics[0]):
		if 'latency' in k or 'latencies' in k:
			metrics['stats_' + k] = {}
	interesting_latency_suboptimalities = [-10,-50,-100]
	for add_str in [""]:
		for k in ['stats_latency{}_thresholds_normal'.format(add_str), 'stats_latency{}_thresholds_fail_popp'.format(add_str), 
			'stats_latency{}_thresholds_fail_pop'.format(add_str), 'stats_latency{}_thresholds_normal_with_bulk'.format(add_str)]:
			metrics[k] = {solution: {i:{} for i in SIM_INDS_TO_PLOT} for solution in soln_types}

	for solution in soln_types:
		print(solution)
		try:
			#### Changes in latency

			### for each user: compute low latency allocation without any bulk traffic
			### then assume all traffic with bulk traffic is also congested ;; compute % of low latency traffic that gets congested

			diffs = []
			wts = []
			for random_iter in SIM_INDS_TO_PLOT:
				this_diffs = []
				this_wts = []

				lp_solution = metrics[random_iter][solution]['lp_solution']
				best_solution = metrics[random_iter]['optimal_objective']['obj']

				for opt_lat, achieved_lat, vol in zip(best_solution['lats_by_ug'], lp_solution['lats_by_ug'], metrics[random_iter]['ug_to_vol']):
					diffs.append(opt_lat - achieved_lat)
					this_diffs.append(opt_lat - achieved_lat)
					wts.append(vol)
					this_wts.append(vol)

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

			diffs = []
			wts = []
			for random_iter in SIM_INDS_TO_PLOT:
				this_diffs = []
				this_wts = []

				lp_solution = metrics[random_iter][solution]['lp_solution']
				best_solution = metrics[random_iter]['optimal_objective']['obj']
				print(best_solution['fraction_congested_volume_with_bulk'])

				for opt_lat, achieved_lat, vol in zip(best_solution['bulk_lats_by_ug'], lp_solution['bulk_lats_by_ug'], metrics[random_iter]['ug_to_vol']):
					diffs.append(opt_lat - achieved_lat)
					this_diffs.append(opt_lat - achieved_lat)
					wts.append(vol)
					this_wts.append(vol)

				this_x,this_cdf_x = get_cdf_xy(list(zip(this_diffs,this_wts)), weighted=True)
				for lat_threshold in interesting_latency_suboptimalities:
					xi = np.argmin(np.abs(this_x-lat_threshold))
					metrics['stats_latency_thresholds_normal_with_bulk'][solution][random_iter][lat_threshold] = this_cdf_x[xi]
			for lat_threshold in interesting_latency_suboptimalities:
				avg_suboptimality = np.mean(list([metrics['stats_latency_thresholds_normal_with_bulk'][solution][random_iter][lat_threshold] for random_iter in SIM_INDS_TO_PLOT]))
				print("({}) {} pct of traffic within {} ms of optimal for normal LP with bulk traffic".format(solution, 100*round(1-avg_suboptimality,4), lat_threshold))
			x,cdf_x = get_cdf_xy(list(zip(diffs,wts)), weighted=True)
			ax[LATENCY_WITH_CONGESTION].plot(x,cdf_x,label=solution)
			avg_latency_diff = np.average(diffs, weights=wts)
			print("Average latency compared to optimal with bulk traffic : {}".format(avg_latency_diff))

			congestions = {bv:[] for bv in bulk_vals}
			for random_iter in SIM_INDS_TO_PLOT:
				m = metrics[random_iter][solution]
				for bv in bulk_vals:
					congestions[bv].append(m['congestion_over_bulk_vals'][bv])
			ax[CONGESTION_OVER_BULK_TRAFFIC].plot(bulk_vals, 
				list([np.mean(congestions[bv]) for bv in bulk_vals]), label=solution)


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


	ax[LATENCY_WITH_CONGESTION].legend(fontsize=8)
	ax[LATENCY_WITH_CONGESTION].grid(True)
	ax[LATENCY_WITH_CONGESTION].set_xlabel("Best - Actual Latency (ms)")
	ax[LATENCY_WITH_CONGESTION].set_ylabel("CDF of Traffic")
	ax[LATENCY_WITH_CONGESTION].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[LATENCY_WITH_CONGESTION].set_xlim([-1*NO_ROUTE_LATENCY/2,0])


	ax[CONGESTION_OVER_BULK_TRAFFIC].set_xlabel("Bulk Traffic Intensity")
	ax[CONGESTION_OVER_BULK_TRAFFIC].set_ylabel("Latency-Sensitive\n Traffic Congestion")
	ax[CONGESTION_OVER_BULK_TRAFFIC].grid(True)
	ax[CONGESTION_OVER_BULK_TRAFFIC].legend(fontsize=8)

	save_fig_fn = kwargs.get('save_fig_fn', "{}_{}.pdf".format(obj,dpsize))

	save_fig(save_fig_fn)

	return metrics




if __name__ == "__main__":
	testing_priorities_one_pass()
	gen_paper_plots()


