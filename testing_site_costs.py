from deployment_setup import get_random_deployment, get_bulk_vol, get_link_capacities
from sparse_advertisements_v3 import Sparse_Advertisement_Eval
from worker_comms import Worker_Manager
from helpers import *
from constants import *
from paper_plotting_functions import *
from wrapper_eval import *
import json


def weighted_cdf_xy(values, weights):
	"""Return x,y for a weighted CDF where y is cumulative weight / total_weight."""
	values = np.asarray(values, dtype=float)
	weights = np.asarray(weights, dtype=float)
	assert values.shape == weights.shape

	order = np.argsort(values)
	x = values[order]
	w = weights[order]
	cw = np.cumsum(w)
	if cw[-1] <= 0:
		return x, np.zeros_like(x)
	y = cw / cw[-1]
	return x, y

def gen_paper_plots(dpsize):
	metrics_fn = os.path.join(CACHE_DIR, 'per_site_cost_metrics_{}.pkl'.format(dpsize))
	
	if not os.path.exists(metrics_fn):
		print("Metrics file not found: {}".format(metrics_fn))
		return

	metrics = pickle.load(open(metrics_fn, 'rb'))
	
	# Filter for standard solutions
	solutions = ['anycast', 'anyopt', 'one_per_pop', 'painter', 'sparse', 'one_per_peering']

	print(f"\nEvaluation Results for Deployment Size: {dpsize}")
	print("="*80)

	# Iterate over each random simulation stored in the metrics
	for random_iter in metrics:
		if not isinstance(metrics[random_iter], dict) or not metrics[random_iter].get('done', False):
			continue

		print(f"\n--- Simulation Iteration {random_iter} ---")
		
		deployment = metrics[random_iter]['deployment']
		site_costs = deployment['site_costs'] # Dictionary mapping site -> cost
		ug_vols = np.array(metrics[random_iter]['ug_to_vol'])
		total_vol = sum(ug_vols)
		
		# Header for the table
		print(f"{'Solution':<20} | {'Avg Latency (ms)':<20} | {'Total Site Cost':<20}")
		print("-" * 70)

		cdf_latency_data = {}
		site_cost_totals_data = {}

		avg_latency_data = {}
		latency_parts = []
		cost_parts = []

		for solution in solutions:
			if solution not in metrics[random_iter]:
				continue
			
			# Retrieve the solution data
			sol_data = metrics[random_iter][solution]
			lp_solution = sol_data['lp_solution']

			# 1. Calculate Average Latency (Weighted by User Group Volume)
			# lats_by_ug is the latency for each user group
			lats = np.array(lp_solution['lats_by_ug'])
			avg_latency = np.average(lats, weights=ug_vols)

			# store CDF of latency across traffic
			x, y = weighted_cdf_xy(lats, ug_vols)
			cdf_latency_data[solution] = (x, y)

			# --- 2. Calculate Total Site Cost using vols_by_poppi ---
			# vols_by_poppi keys are likely sites or (site, peer) tuples
			vols_by_poppi = lp_solution.get('vols_by_poppi', {})
			total_site_cost = 0.0

			for poppi, vol in vols_by_poppi.items():
				site, peer = deployment['popps'][poppi]
				# Check if this site exists in our cost map
				if site in site_costs:
					cost_per_unit = site_costs[site]
					total_site_cost += vol * cost_per_unit
				else:
					# Optional: Print warning if a site is receiving traffic but has no cost data
					# print(f"Warning: Site {site} not in site_costs dict")
					pass
			
			site_cost_totals_data[solution] = total_site_cost

			latency_parts.append(avg_latency)
			cost_parts.append((DEFAULT_SITE_COST * total_site_cost) / total_vol)

			# --- 3. Get Congestion info (optional but helpful) ---
			frac_congested = lp_solution.get('fraction_congested_volume', 0.0)

			print(f"{solution:<20} | {avg_latency:<20.4f} | {total_site_cost:<20.4f} | {(total_vol*avg_latency+DEFAULT_SITE_COST*total_site_cost)/total_vol:<20.4f} | {frac_congested:.4f}")
			
		print("-" * 95)

		def _to_jsonable(x):
			if isinstance(x, (np.integer, np.int64, np.int32)):
				return int(x)
			if isinstance(x, (np.floating, np.float64, np.float32)):
				return float(x)
			if isinstance(x, np.ndarray):
				return x.tolist()
			return x

		payload = {
			"dpsize": dpsize,
			"random_iter": random_iter,
			"solutions": solutions,  # the solutions actually present in this iter (same order used in plots)
			"cdf_latency_data": {
				sol: {
					"x": _to_jsonable(x),
					"y": _to_jsonable(y),
				}
				for sol, (x, y) in cdf_latency_data.items()
			},
			"site_cost_totals_data": {sol: _to_jsonable(val) for sol, val in site_cost_totals_data.items()},
			"latency_parts": {sol: _to_jsonable(v) for sol, v in zip(sols, latency_parts)},
			"cost_parts": {sol: _to_jsonable(v) for sol, v in zip(sols, cost_parts)},
		}

		out_fn = f"site_cost_plot_data_{dpsize}_iter{random_iter}.json"
		with open(out_fn, "w") as f:
			json.dump(payload, f, indent=2, sort_keys=True)

		cdf_colors = {}  # solution -> color
		plt.figure()
		for solution, (x, y) in cdf_latency_data.items():
			line, = plt.plot(x, y, label=solution)
			cdf_colors[solution] = line.get_color() 
		plt.xlabel("Latency (ms)")
		plt.ylabel("CDF (fraction of traffic)")
		plt.title(f"Traffic-weighted latency CDF (iter={random_iter})")
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.tight_layout()
		plt.savefig('site_cost_latency_{}.png'.format(dpsize))
		
		plt.figure()
		sols = list(site_cost_totals_data.keys())
		costs = [site_cost_totals_data[s] for s in sols]
		colors = [cdf_colors.get(s, None) for s in sols]
		plt.bar(sols, costs, color=colors)
		plt.ylabel("Total site cost")
		plt.title(f"Total site cost by solution (iter={random_iter})")
		plt.xticks(rotation=30, ha="right")
		plt.grid(True, axis="y", alpha=0.3)
		plt.tight_layout()
		plt.savefig('site_cost_totals_{}.png'.format(dpsize))

		lat_colors = [cdf_colors[s] for s in sols]

		# Make cost colors slightly lighter (same hue)
		def lighten(color, factor=0.5):
			import matplotlib.colors as mc

			c = np.array(mc.to_rgb(color))
			return tuple(1 - factor * (1 - c))

		cost_colors = [lighten(cdf_colors[s], 0.6) for s in sols]

		# Plot
		plt.figure()
		plt.bar(sols, latency_parts, color=lat_colors, label="Latency")
		plt.bar(
			sols,
			cost_parts,
			bottom=latency_parts,
			color=cost_colors,
			label="Site cost",
		)

		plt.ylabel("Objective value")
		plt.title(f"Final objective breakdown (iter={random_iter})")
		plt.xticks(rotation=30, ha="right")
		plt.grid(True, axis="y", alpha=0.3)
		plt.legend()
		plt.tight_layout()
		plt.savefig("site_cost_final_objective_breakdown_{}.png".format(dpsize))

		# Break after printing the first valid iteration to avoid spamming 
		# (remove break if you want to see all random seeds)
		break

def check_calced_everything(metrics, random_iter, k_of_interest):
	soln_types = global_soln_types
	for solution in soln_types:
		try:
			metrics[random_iter][solution][k_of_interest]
		except KeyError:
			return False
	return True

def testing_site_cost(dpsize, **kwargs):
	"""
		Tests minimization of LL + alpha * site_cost of traffic
	"""

	np.random.seed(31411)
	lambduh = .00001 ## unused more or less
	gamma = 0#2.0

	n_random_sim = 1
	obj = 'per_site_cost'
	performance_metrics_fn = os.path.join(CACHE_DIR, "{}_metrics_{}.pkl".format(obj, dpsize))
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
				print("Random deployment for joint latency site cost, number {}/{}".format(random_iter+1,n_random_sim))
				
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

	try:
		if True:
			port = int(sys.argv[2])
			for random_iter in range(n_random_sim):
				try:
					this_iter_deployment = metrics[random_iter]['deployment']
				except KeyError:
					this_iter_deployment = get_random_deployment(dpsize)
				this_iter_deployment['port'] = port
				print("Random deployment for joint latency site cost, number {}/{}".format(random_iter+1,n_random_sim))
				
				deployment = copy.deepcopy(this_iter_deployment)
				metrics[random_iter] = {'deployment': deployment}

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

	# (for each solution)
	#### TBD --- think of things to plot

	SIM_INDS_TO_PLOT = list(range(n_random_sim))

	################################
	### PLOTTING
	################################
	
	i=0
	LATENCY_I = i;i+=1
	LATENCY_WITH_CONGESTION = i; i+=1

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

	save_fig_fn = kwargs.get('save_fig_fn', "{}_{}.pdf".format(obj,dpsize))

	save_fig(save_fig_fn)

	return metrics


if __name__ == "__main__":
	dpsize = sys.argv[1]
	testing_site_cost(dpsize)
	gen_paper_plots(dpsize)


