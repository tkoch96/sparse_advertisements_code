from constants import *
from helpers import *
from eval_latency_failure import evaluate_all_metrics
import numpy as np, os, pickle
np.random.seed(31701)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

from wrapper_eval import metro_to_diurnal_factor

from paper_plotting_functions import *

def apply_con_latency(lat,con):
	return (1-con) * lat + con * NO_ROUTE_LATENCY

def make_hotnets_plots(cache_fn):
	### for thesis defense
	metrics = pickle.load(open(cache_fn, 'rb'))

	################################
	### PLOTTING
	################################


	SIM_INDS_TO_PLOT = list(sorted(list(metrics['best_latencies'])))
	soln_types = list(metrics['latencies'][SIM_INDS_TO_PLOT[0]])

	plot_every = 2

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
				ret.append((-1 * (perf1-perf2), vol))

			### Store the fraction of users that DONT satisfy a latency objective
			this_sim_fraction_volume_congested = this_sim_total_volume_congested / (this_sim_total_volume + .0000001)
			try:
				this_x,this_cdf_x = get_cdf_xy(list(zip(this_diffs,this_vols)), weighted=True)
				for lat_threshold in interesting_latency_suboptimalities:
					xi = np.argmin(np.abs(this_x- (-1*lat_threshold)))
					threshold_stats[ri][lat_threshold] = (1-this_sim_fraction_volume_congested) * this_cdf_x[xi] + this_sim_fraction_volume_congested
			except IndexError: # no good data
				for lat_threshold in interesting_latency_suboptimalities:
					threshold_stats[ri][lat_threshold] = this_sim_fraction_volume_congested ## all users are within the latency

			


		x=np.linspace(mind,maxd,num=200)
		if vol_congested > 0:
			x[-1] = 100*NO_ROUTE_LATENCY

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


	###### STEADY STATE LATENCY
	soln_types = ['sparse', 'painter', 'anycast', 'one_per_peering']
	solution_to_plot_label = {
		'sparse': 'Gradient Descent Prototype',
		'anycast': "Anycast",
		'painter':'PAINTER',
		'one_per_peering':"Configure All Routes"
	}
	f,ax = get_figure()
	for solution in soln_types:
		diffs = []
		wts = []
		for random_iter in SIM_INDS_TO_PLOT:
			this_diffs = []
			this_wts = []
			if solution == 'one_per_peering': 
				metrics['latencies'][random_iter][solution] = metrics['best_latencies'][random_iter]
			for ug in metrics['best_latencies'][random_iter]:
				try:
					metrics['latencies'][random_iter][solution][ug]
				except KeyError:
					## deleted, don't include
					continue
				vol = metrics['deployment'][random_iter]['ug_to_vol'][ug]
				diffs.append(-1 * (metrics['best_latencies'][random_iter][ug] - metrics['latencies'][random_iter][solution][ug]))
				this_diffs.append(-1 * (metrics['best_latencies'][random_iter][ug] - metrics['latencies'][random_iter][solution][ug]))
				wts.append(vol)
				this_wts.append(vol)

			this_x,this_cdf_x = get_cdf_xy(list(zip(this_diffs,this_wts)), weighted=True)
			for lat_threshold in interesting_latency_suboptimalities:
				xi = np.argmin(np.abs(this_x- (-1*lat_threshold)))
				metrics['stats_latency_thresholds_normal'][solution][random_iter][lat_threshold] = this_cdf_x[xi]
		for lat_threshold in interesting_latency_suboptimalities:
			avg_suboptimality = np.mean(list([metrics['stats_latency_thresholds_normal'][solution][random_iter][lat_threshold] for random_iter in SIM_INDS_TO_PLOT]))
			print("({}) {} pct of traffic within {} ms of optimal for normal LP".format(solution, 100*round(1-avg_suboptimality,4), lat_threshold))
		x,cdf_x = get_cdf_xy(list(zip(diffs,wts)), weighted=True)
		ax.plot(x[::plot_every*5],cdf_x[::plot_every*5], label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
		avg_latency_diff = np.average(diffs, weights=wts)
		print("Average latency compared to optimal : {}".format(avg_latency_diff))
		metrics['stats_best_latencies'][solution] = avg_latency_diff
	ax.legend()
	ax.grid(True)
	ax.set_xlabel("Actual - Optimal Latency (ms)")
	ax.set_ylabel("CDF of Traffic")
	ax.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax.set_xlim([0,30])
	save_figure('hotnets/steady_state_latency_actual_deployment.pdf')


	####### SINGLE INGRESS/SITE FAILURES
	for metric_k, lab in zip(['popp', 'pop'], ['Link', 'Site']):
		f,ax = get_figure()
		for solution in soln_types:
			all_differences, x, stats, threshold_stats = get_failure_metric_arr('{}_failures_latency_optimal_specific'.format(metric_k), solution)
			metrics['stats_' + '{}_failures_latency_optimal_specific'.format(metric_k)][solution] = stats
			metrics['stats_latency_thresholds_fail_{}'.format(metric_k)][solution] = threshold_stats
			x, cdf_x = get_cdf_xy(all_differences, weighted=True, x=x)
			ax.plot(x[::plot_every],cdf_x[::plot_every],label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])

		ax.legend()
		ax.grid(True)
		ax.set_xlabel("Actual - Optimal Latency Single-{} Failure (ms)".format(lab))
		ax.set_ylabel("CDF of Traffic")
		ax.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
		ax.set_xlim([0,100])

		save_figure('hotnets/{}_failure_latency_actual_deployment.pdf'.format(lab.lower()))


def make_ppt_plots(cache_fn):
	### for thesis defense
	metrics = pickle.load(open(cache_fn, 'rb'))

	################################
	### PLOTTING
	################################


	SIM_INDS_TO_PLOT = list(sorted(list(metrics['best_latencies'])))
	soln_types = list(metrics['latencies'][SIM_INDS_TO_PLOT[0]])

	plot_every = 2

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
				ret.append((-1 * (perf1-perf2), vol))

			### Store the fraction of users that DONT satisfy a latency objective
			this_sim_fraction_volume_congested = this_sim_total_volume_congested / (this_sim_total_volume + .0000001)
			try:
				this_x,this_cdf_x = get_cdf_xy(list(zip(this_diffs,this_vols)), weighted=True)
				for lat_threshold in interesting_latency_suboptimalities:
					xi = np.argmin(np.abs(this_x- (-1*lat_threshold)))
					threshold_stats[ri][lat_threshold] = (1-this_sim_fraction_volume_congested) * this_cdf_x[xi] + this_sim_fraction_volume_congested
			except IndexError: # no good data
				for lat_threshold in interesting_latency_suboptimalities:
					threshold_stats[ri][lat_threshold] = this_sim_fraction_volume_congested ## all users are within the latency

			


		x=np.linspace(mind,maxd,num=200)
		if vol_congested > 0:
			x[-1] = 100*NO_ROUTE_LATENCY

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


	###### STEADY STATE LATENCY
	soln_types = ['anycast', 'one_per_pop', 'painter', 'sparse', 'one_per_peering']
	f,ax = get_figure()
	for solution in soln_types:
		diffs = []
		wts = []
		for random_iter in SIM_INDS_TO_PLOT:
			this_diffs = []
			this_wts = []
			if solution == 'one_per_peering': 
				metrics['latencies'][random_iter][solution] = metrics['best_latencies'][random_iter]
			for ug in metrics['best_latencies'][random_iter]:
				try:
					metrics['latencies'][random_iter][solution][ug]
				except KeyError:
					## deleted, don't include
					continue
				vol = metrics['deployment'][random_iter]['ug_to_vol'][ug]
				diffs.append(-1 * (metrics['best_latencies'][random_iter][ug] - metrics['latencies'][random_iter][solution][ug]))
				this_diffs.append(-1 * (metrics['best_latencies'][random_iter][ug] - metrics['latencies'][random_iter][solution][ug]))
				wts.append(vol)
				this_wts.append(vol)

			this_x,this_cdf_x = get_cdf_xy(list(zip(this_diffs,this_wts)), weighted=True)
			for lat_threshold in interesting_latency_suboptimalities:
				xi = np.argmin(np.abs(this_x- (-1*lat_threshold)))
				metrics['stats_latency_thresholds_normal'][solution][random_iter][lat_threshold] = this_cdf_x[xi]
		for lat_threshold in interesting_latency_suboptimalities:
			avg_suboptimality = np.mean(list([metrics['stats_latency_thresholds_normal'][solution][random_iter][lat_threshold] for random_iter in SIM_INDS_TO_PLOT]))
			print("({}) {} pct of traffic within {} ms of optimal for normal LP".format(solution, 100*round(1-avg_suboptimality,4), lat_threshold))
		x,cdf_x = get_cdf_xy(list(zip(diffs,wts)), weighted=True)
		ax.plot(x[::plot_every*5],cdf_x[::plot_every*5], label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
		avg_latency_diff = np.average(diffs, weights=wts)
		print("Average latency compared to optimal : {}".format(avg_latency_diff))
		metrics['stats_best_latencies'][solution] = avg_latency_diff
	ax.legend()
	ax.grid(True)
	ax.set_xlabel("Actual - Optimal Latency (ms)")
	ax.set_ylabel("CDF of Traffic")
	ax.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax.set_xlim([0,100])
	save_figure('ppt/steady_state_latency_actual_deployment.pdf')


	####### SINGLE INGRESS/SITE FAILURES
	for metric_k, lab in zip(['popp', 'pop'], ['Link', 'Site']):
		f,ax = get_figure()
		for solution in soln_types:
			all_differences, x, stats, threshold_stats = get_failure_metric_arr('{}_failures_latency_optimal_specific'.format(metric_k), solution)
			metrics['stats_' + '{}_failures_latency_optimal_specific'.format(metric_k)][solution] = stats
			metrics['stats_latency_thresholds_fail_{}'.format(metric_k)][solution] = threshold_stats
			x, cdf_x = get_cdf_xy(all_differences, weighted=True, x=x)
			ax.plot(x[::plot_every],cdf_x[::plot_every],label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])

		ax.legend()
		ax.grid(True)
		ax.set_xlabel("Actual - Optimal Latency Single-{} Failure (ms)".format(lab))
		ax.set_ylabel("CDF of Traffic")
		ax.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
		ax.set_xlim([0,100])

		save_figure('ppt/{}_failure_latency_actual_deployment.pdf'.format(lab.lower()))

	
def make_paper_plots(cache_fn):
	metrics = pickle.load(open(cache_fn, 'rb'))

	################################
	### PLOTTING
	################################


	SIM_INDS_TO_PLOT = list(sorted(list(metrics['best_latencies'])))
	soln_types = list(metrics['latencies'][SIM_INDS_TO_PLOT[0]])

	plot_every = 2

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


	###### STEADY STATE LATENCY
	soln_types = ['anycast', 'one_per_pop', 'painter', 'sparse', 'one_per_peering']
	f,ax = get_figure()
	for solution in soln_types:
		diffs = []
		wts = []
		for random_iter in SIM_INDS_TO_PLOT:
			this_diffs = []
			this_wts = []
			if solution == 'one_per_peering': 
				metrics['latencies'][random_iter][solution] = metrics['best_latencies'][random_iter]
			for ug in metrics['best_latencies'][random_iter]:
				try:
					metrics['latencies'][random_iter][solution][ug]
				except KeyError:
					## deleted, don't include
					continue
				vol = metrics['deployment'][random_iter]['ug_to_vol'][ug]
				diffs.append(metrics['best_latencies'][random_iter][ug] - metrics['latencies'][random_iter][solution][ug])
				this_diffs.append(metrics['best_latencies'][random_iter][ug] - metrics['latencies'][random_iter][solution][ug])
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
		ax.plot(x[::plot_every*5],cdf_x[::plot_every*5], label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
		avg_latency_diff = np.average(diffs, weights=wts)
		print("Average latency compared to optimal : {}".format(avg_latency_diff))
		metrics['stats_best_latencies'][solution] = avg_latency_diff
	ax.legend()
	ax.grid(True)
	ax.set_xlabel("Optimal - Actual Latency (ms)")
	ax.set_ylabel("CDF of Traffic")
	ax.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax.set_xlim([-100,0])
	save_figure('steady_state_latency_actual_deployment.pdf')


	####### SINGLE INGRESS/SITE FAILURES
	for metric_k, lab in zip(['popp', 'pop'], ['Link', 'Site']):
		f,ax = get_figure()
		for solution in soln_types:
			all_differences, x, stats, threshold_stats = get_failure_metric_arr('{}_failures_latency_optimal_specific'.format(metric_k), solution)
			metrics['stats_' + '{}_failures_latency_optimal_specific'.format(metric_k)][solution] = stats
			metrics['stats_latency_thresholds_fail_{}'.format(metric_k)][solution] = threshold_stats
			x, cdf_x = get_cdf_xy(all_differences, weighted=True, x=x)
			ax.plot(x[::plot_every],cdf_x[::plot_every],label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])

		ax.legend()
		ax.grid(True)
		ax.set_xlabel("Optimal - Achieved Latency Single-{} Failure (ms)".format(lab))
		ax.set_ylabel("CDF of Traffic")
		ax.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
		ax.set_xlim([-100,0])

		save_figure('{}_failure_latency_actual_deployment.pdf'.format(lab.lower()))


	#### Resilience to flash crowds
	m = metrics['resilience_to_congestion']
	cm = metrics['fraction_congested_volume']
	latencies_over_X_by_solution, congestions_over_X_by_solution = {}, {}
	for solution in soln_types:
		Y_val = list(m[SIM_INDS_TO_PLOT[0]][solution])[0]
		X_vals = sorted(list(m[SIM_INDS_TO_PLOT[0]][solution][Y_val]))
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

			latency_delta_meds.append(apply_con_latency(lat_med,con_med))
			congestion_meds.append(con_med)

		## High level stats
		for ri in SIM_INDS_TO_PLOT:
			try:
				critical_X = X_vals[np.where(np.array(avg_congestion_by_X_val_sim[ri]) > 0)[0][0]]
			except IndexError:
				## either never or always is congested
				critical_X = X_vals[0]
			print("Critical X val for {} is {}".format(solution, critical_X))
			metrics['stats_resilience_to_congestion'][solution][ri] = critical_X

		latencies_over_X_by_solution[solution] = np.array(latency_delta_meds)
		congestions_over_X_by_solution[solution] = np.array(congestion_meds)


	for arr,ylab,fnsave in zip([latencies_over_X_by_solution, congestions_over_X_by_solution],
		['Average Latency \n Flash Crowd (ms)', 'Fraction Congested Traffic\n Flash Crowd'],
		['latency', 'congestion']):
		f,ax = get_figure()
		for solution in soln_types:
			# crit_X = metrics['stats_resilience_to_congestion'][solution][0]
			plot_until_i = -1#np.where(np.array(X_vals) == crit_X)[0][0]
			ax.plot(X_vals[0:plot_until_i], arr[solution][0:plot_until_i], label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
			# if fnsave == 'latency':
			# 	## Mark the critical X with a vertical line
			# 	max_lat = max(list([np.max(arr[s]) for s in soln_types]))
			# 	ax.axvline(, 0, max_lat, color=solution_to_line_color[solution])

		ax.set_xlabel("Increase in Traffic per Region for Flash Crowd (pct.)")
		if fnsave == 'latency':
			ax.set_ylim([0,50])
		ax.set_ylabel(ylab)
		ax.grid(True)
		ax.legend()
		save_figure('flash_crowd_{}_increase_actual_deployment.pdf'.format(fnsave))

	# f,ax = get_figure()
	# for solution in soln_types:
	# 	ax.plot(latencies_over_X_by_solution[solution], congestions_over_X_by_solution[solution], label=solution_to_plot_label[solution], 
	# 		marker=solution_to_marker[solution], color=solution_to_line_color[solution])
	# ax.set_xlabel("Average Latency Change Flash Crowd (new - old) (ms)")
	# ax.set_ylabel("Fraction Traffic\nCongested")
	# ax.legend()
	# save_figure('flash_crowd_latency_vs_congestion_actual_deployment.pdf')

	#### DIURNAL 
	latencies_over_X_by_solution, congestions_over_X_by_solution = {}, {}
	diurnal_multipliers = list(sorted(list(metrics['diurnal'][SIM_INDS_TO_PLOT[0]][soln_types[0]]['metrics'])))
	for solution in soln_types:
		#### Diurnal Resilience
		hours_of_day = np.array(list(range(24)))
		## Want to track what diurnal multiplier causes congestion
		avg_latency_by_sim_Y_val = {ri: [] for ri in SIM_INDS_TO_PLOT}
		avg_congestion_by_sim_Y_val = {ri: [] for ri in SIM_INDS_TO_PLOT}

		avg_latency_by_Y_val_sim = {dm: [] for dm in diurnal_multipliers}
		avg_congestion_by_Y_val_sim = {dm: [] for dm in diurnal_multipliers}

		for Y_val in diurnal_multipliers:
			all_lats, all_congestions = {ri:[] for ri in SIM_INDS_TO_PLOT}, {ri:[] for ri in SIM_INDS_TO_PLOT}
			for ri in SIM_INDS_TO_PLOT:
				for X_val in hours_of_day:
					try:
						all_lats[ri].append(metrics['diurnal'][ri][solution]['metrics'][Y_val][X_val])
						all_congestions[ri].append(metrics['diurnal'][ri][solution]['fraction_congested_volume'][Y_val][X_val])
					except KeyError:	
						continue
				lat_med = np.average(all_lats[ri])
				con_med = np.average(all_congestions[ri])

				avg_latency_by_sim_Y_val[ri].append(apply_con_latency(lat_med,con_med))
				avg_congestion_by_sim_Y_val[ri].append(con_med)

				avg_latency_by_Y_val_sim[Y_val].append(apply_con_latency(lat_med,con_med))
				avg_congestion_by_Y_val_sim[Y_val].append(con_med)

		latencies_over_X_by_solution[solution] = list([np.mean(avg_latency_by_Y_val_sim[dm]) for dm in diurnal_multipliers])
		congestions_over_X_by_solution[solution] = list([np.mean(avg_congestion_by_Y_val_sim[dm]) for dm in diurnal_multipliers])

		## High level stats
		for ri in SIM_INDS_TO_PLOT:
			try:
				critical_X = diurnal_multipliers[np.where(np.array(avg_congestion_by_sim_Y_val[ri]) > 0)[0][0]]
			except IndexError:
				## either never or always is congested
				critical_X = diurnal_multipliers[0]
			print("Critical diurnal val for {} is {}".format(solution, critical_X))
			metrics['stats_diurnal'][solution][ri] = critical_X

	for arr,ylab,fnsave in zip([latencies_over_X_by_solution, congestions_over_X_by_solution],
		['Average Latency \nunder Diurnal Effect (ms)', 'Fraction Congested \n Traffic under Diurnal Effect'],
		['latency', 'congestion']):
		f,ax = get_figure()
		for solution in soln_types:
			# crit_X = metrics['stats_diurnal'][solution][0]
			plot_until_i = -1#np.where(np.array(diurnal_multipliers) == crit_X)[0][0]
			ax.plot(diurnal_multipliers[0:plot_until_i], arr[solution][0:plot_until_i], label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
			# if fnsave == 'latency':
			# 	## Mark the critical X with a vertical line
			# 	max_lat = max(list([np.max(arr[s]) for s in soln_types]))
			# 	ax.axvline(crit_X, 0, max_lat, color=solution_to_line_color[solution])
		ax.set_xlabel("Increase in Traffic per Region for Diurnal Effect (pct.)")
		if fnsave == 'latency':
			ax.set_ylim([0,50])
		ax.set_ylabel(ylab)
		ax.grid(True)
		ax.legend()
		save_figure('diurnal_{}_increase_actual_deployment.pdf'.format(fnsave))

	# f,ax = get_figure()
	# for solution in soln_types:
	# 	ax.plot(latencies_over_X_by_solution[solution], congestions_over_X_by_solution[solution], label=solution_to_plot_label[solution], 
	# 		marker=solution_to_marker[solution], color=solution_to_line_color[solution])
	# ax.set_xlabel("Average Latency Change Diurnal Effect (new - old) (ms)")
	# ax.set_ylabel("Fraction Traffic\nCongested")
	# ax.legend()
	# save_figure('diurnal_latency_vs_congestion_actual_deployment.pdf')


	
	### FLASH CROWDS

	# ax[FLASH_CROWD_SINGLE_X_I].set_xlabel("Latency Change under Flash Crowd (new - old) (ms)")
	# ax[FLASH_CROWD_SINGLE_X_I].set_ylabel("CDF of Traffic")
	# ax[FLASH_CROWD_SINGLE_X_I].grid(True)
	# ax[FLASH_CROWD_SINGLE_X_I].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	# ax[FLASH_CROWD_SINGLE_X_I].legend(fontsize=8)

def create_diurnal_shape_figure():
	metro = 'vtrlondon'
	hours = np.array(list(range(24)))
	diurnal_curve = np.array(list([metro_to_diurnal_factor(metro,hour) for hour in range(24)]))	
	f,ax = get_figure()
	ax.plot(hours,diurnal_curve)

	ax.set_yticks([.1, .25, .5, .75, 1.0], ['0.1M', '0.25M', '0.5M', '0.75M', 'M'])
	ax.set_xlabel("Hour of the Day (GMT)")
	ax.set_ylabel("Regional Diurnal\nMultiplier")
	save_figure('diurnal_shape.pdf')

def create_uncertainty_over_convergence_figure():
	import re, glob
	run_dir = os.path.join(RUN_DIR, '1714508601-actual-32-sparse')
	# run_dir = os.path.join(RUN_DIR, '1712338080-actual-3-sparse')
	all_run_nums = list([int(re.search("small\-stats\-(.+)\.pkl", fn).group(1)) for fn in glob.glob(os.path.join(run_dir, "*")) if 'small' in fn])
	all_run_nums = list(sorted(all_run_nums))
	
	iters = []
	path_measures = []
	uncertainty_measures = []

	for run_num in all_run_nums:
		if run_num == 0: continue
		this_state = pickle.load(open(os.path.join(run_dir, 'small-stats-{}.pkl'.format(run_num)), 'rb'))
		mtrcs = this_state['optimization_vars']
		iters.append(run_num)
		path_measures.append(mtrcs['path_measures'])
		uncertainty_measures.append(mtrcs['uncertainty_factor'])
	uncertainty_measures = np.array(uncertainty_measures)
	uncertainty_measures = uncertainty_measures / np.max(uncertainty_measures)

	f,ax = get_figure(h=2)
	ax.plot(iters,uncertainty_measures,color='red')
	ax2 = ax.twinx()
	ax2.plot(iters,path_measures,color='blue')

	ax.set_ylabel("Normalized\nEntropy")
	ax2.set_ylabel("Advertisements")
	ax.set_xlabel("Gradient Step Iteration")


	ax.annotate("Uncertainty", (8,.6), color='red')
	ax2.annotate("Advertisements", (80,58), color='blue')
	ax.yaxis.label.set_color('red')
	ax.tick_params(axis='y', colors='red')

	ax2.yaxis.label.set_color('blue')
	ax2.tick_params(axis='y', colors='blue')

	save_figure('uncertainty_path_measures_over_iterations.pdf')


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--dpsize", default=None, required=True)
	args = parser.parse_args()
	dpsize = args.dpsize


	cache_fn = os.path.join(CACHE_DIR, 'popp_failure_latency_comparison_{}.pkl'.format(dpsize))

	make_hotnets_plots(cache_fn)
	# make_ppt_plots(cache_fn)
	# create_diurnal_shape_figure()
	# make_paper_plots(cache_fn)
	# create_uncertainty_over_convergence_figure()

