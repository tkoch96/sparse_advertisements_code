from constants import *
from eval_latency_failure import evaluate_all_metrics
import numpy as np, os, pickle
np.random.seed(31705)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

from paper_plotting_functions import *

def pull_results(cache_fn):
	cache_fn = os.path.join(CACHE_DIR, 'evaluate_over_deployment_sizes_cache_fn.pkl')

	only_recalc = [25] ## recalc these deployment sizes
	metrics_by_dpsize = {}
	if os.path.exists(cache_fn):
		metrics_by_dpsize = pickle.load(open(cache_fn, 'rb'))
	if not os.path.exists(cache_fn) or only_recalc is not None:
		import argparse
		parser = argparse.ArgumentParser()
		parser.add_argument("--port", required=True)
		args = parser.parse_args()
		n_sim_by_dpsize = [15,20,10,16,10,10,6] # running now
		# n_sim_by_dpsize = [15,20,10,16,8,5,6] # done
		for dpsize, nsim in zip([3,5,10,15,20,25,len(POP_TO_LOC['vultr'])], n_sim_by_dpsize):
			if only_recalc is not None:
				if dpsize not in only_recalc: continue
			print("Evaluating over deployment size {} Sites".format(dpsize))
			dpsize_str = "actual-{}".format(dpsize)
			# if dpsize == 32:
			# 	save_run_dir = [None, None, None, None, None, '1713700098-actual-32-sparse']
			# else:
			# 	save_run_dir = None
			save_run_dir = None
			metrics = evaluate_all_metrics(dpsize_str, int(args.port), save_run_dir=save_run_dir, nsim=nsim)
			metrics_by_dpsize[dpsize] = {}
			for k in metrics:
				if 'stats' in k:
					metrics_by_dpsize[dpsize][k] = metrics[k]
		pickle.dump(metrics_by_dpsize, open(cache_fn, 'wb'))

def make_paper_plots(cache_fn, **kwargs):
	metrics_by_dpsize = pickle.load(open(cache_fn, 'rb'))
	dpsizes = sorted(list(metrics_by_dpsize))
	solutions = sorted(list(metrics_by_dpsize[dpsizes[0]]['stats_best_latencies']))

	solutions = ['anycast', 'anyopt', 'one_per_pop', 'painter', 'sparse', 'one_per_peering']

	xlab = kwargs.get('xlab', "Deployment Size (Num Sites)")
	evaluate_over = kwargs.get('evaluate_over', 'deployment_size')

	f,ax = get_figure()
	metric_by_solution = {}
	for solution in solutions:
		avg_latency_diff_normal = []
		for dpsize in dpsizes:
			avg_latency_diff_normal.append(-1*metrics_by_dpsize[dpsize]['stats_best_latencies'][solution])
		ax.plot(dpsizes,avg_latency_diff_normal, label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
		metric_by_solution[solution] = np.array(avg_latency_diff_normal)
	
	print("Painter - Sparse Normal: {}".format(metric_by_solution['painter'] - metric_by_solution['sparse']))
	print("Sparse - OPP Normal: {}".format(metric_by_solution['sparse'] - metric_by_solution['one_per_peering']))
	for solution in solutions:
		print("{} average over all deployments: {}".format(solution,np.average(metric_by_solution[solution])))
	print('\n')

	ax.set_xlabel(xlab)
	ax.set_ylabel("Avg Suboptimality\nNormal Operation (ms)")
	ax.legend(fontsize=12)
	save_figure('average_latency_over_{}_normal.pdf'.format(evaluate_over))

	for lp_tp, tp, tp_k in zip(['mlu','mlu','lagrange','lagrange'], ['Ingress', 'Site', 'Ingress', 'Site'],['stats_popp_failures_latency_optimal_specific', 'stats_pop_failures_latency_optimal_specific',
		'stats_popp_failures_latency_lagrange_optimal_specific', 'stats_pop_failures_latency_lagrange_optimal_specific']):
		for metric_k, outer_k, ylab in zip(['avg_latency_difference', 'frac_vol_congested'], ['latency', 'congestion'], 
			['Avg Suboptimality\n During {} Failure (ms)'.format(tp), 'Pct Volume Congested\nDuring {} Failure'.format(tp)]):
			f,ax = get_figure()
			fig_fn = 'average_{}_over_{}_fail_{}_{}.pdf'.format(outer_k, evaluate_over, tp.lower(), lp_tp)
			metric_by_solution = {}
			for solution in solutions:
				agg_metrics =[]
				for dpsize in dpsizes:
					try:
						these_metrics = metrics_by_dpsize[dpsize][tp_k][solution]
						if outer_k == 'congestion':
							agg_metrics.append(100*these_metrics[metric_k])
						else:
							agg_metrics.append(-1*these_metrics[metric_k])
					except KeyError:
						continue
				ax.plot(dpsizes[0:len(agg_metrics)], agg_metrics, label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
				metric_by_solution[solution] = np.array(agg_metrics)
			ax.set_xlabel(xlab)
			ax.set_ylabel(ylab)
			ax.legend(fontsize=12)
			save_figure(fig_fn)

			if lp_tp == 'mlu':
				print("{} -- {}".format(tp, outer_k))
				print("Painter - Sparse: {}".format(metric_by_solution['painter'] - metric_by_solution['sparse']))
				print("Sparse - OPP: {}".format(metric_by_solution['sparse'] - metric_by_solution['one_per_peering']))
				for solution in solutions:
					print("{} average over all deployments: {}".format(solution,np.average(metric_by_solution[solution])))
				print('\n')


	metric_access_ks = ['stats_latency_thresholds_normal', 'stats_latency_thresholds_fail_popp', 'stats_latency_thresholds_fail_pop',
		'stats_latency_penalty_thresholds_normal', 'stats_latency_penalty_thresholds_fail_popp', 'stats_latency_penalty_thresholds_fail_pop',
		'stats_latency_lagrange_thresholds_normal', 'stats_latency_lagrange_thresholds_fail_popp', 'stats_latency_lagrange_thresholds_fail_pop']

	solutions = list(reversed(solutions))
	plot_metrics= {solution:{k:{} for k in metric_access_ks} for solution in solutions}
	parsed_metrics = {}
	lat_thresholds = {}
	for solution in solutions:
		agg_metrics = {k:{dpsize:{} for dpsize in dpsizes} for k in metric_access_ks}
		for dpsize in dpsizes:
			for k in metric_access_ks:
				try:
					these_metrics = metrics_by_dpsize[dpsize][k][solution]
				except KeyError:
					pass					
				for soln_i in these_metrics:
					for lat_threshold in these_metrics[soln_i]:
						lat_thresholds[lat_threshold] = None
						try:
							agg_metrics[k][dpsize][lat_threshold].append(these_metrics[soln_i][lat_threshold])
						except KeyError:
							agg_metrics[k][dpsize][lat_threshold] = [these_metrics[soln_i][lat_threshold]]
		parsed_metrics[solution] = agg_metrics
		n_lat_thresholds = len(lat_thresholds)
		for i,k in enumerate(metric_access_ks):
			for j,lat_threshold in enumerate(sorted(lat_thresholds,reverse=True)):
				try:
					arr = list([100-100*np.mean(parsed_metrics[solution][k][dpsize][lat_threshold]) for dpsize in dpsizes])		
					plot_metrics[solution][k][lat_threshold] = arr
				except KeyError:
					continue

	for lat_threshold in sorted(lat_thresholds):
		all_ylabs = ['Pct Traffic Within {} ms\n of Optimal (Normally)'.format(int(np.abs(lat_threshold))), 'Pct Traffic Within {} ms\n of Optimal (Link Failure)'.format(int(np.abs(lat_threshold))),
			 'Pct Traffic Within {} ms\n of Optimal (Site Failure)'.format(int(np.abs(lat_threshold))),
			 'Pct Traffic Within {} ms\n of Optimal (Normally) (Penalty)'.format(int(np.abs(lat_threshold))), 'Pct Traffic Within {} ms\n of Optimal (Link Failure) (Penalty)'.format(int(np.abs(lat_threshold))),
			 'Pct Traffic Within {} ms\n of Optimal (Site Failure) (Penalty)'.format(int(np.abs(lat_threshold))),
			 'Pct Traffic Within {} ms\n of Optimal (Normally)'.format(int(np.abs(lat_threshold))), 'Pct Traffic Within {} ms\n of Optimal (Link Failure)'.format(int(np.abs(lat_threshold))),
			 'Pct Traffic Within {} ms\n of Optimal (Site Failure)'.format(int(np.abs(lat_threshold))),]
		fns = ['normal','link_failure', 'site_failure', 'normal_penalty', 'link_failure_penalty', 'site_failure_penalty', 'normal_lagrange', 'link_failure_lagrange',' site_failure_lagrange']
		for k, ylab, fn in zip(metric_access_ks, all_ylabs, fns):
			if 'penalty' in fn: continue ## ignore for now
			f,ax = get_figure()
			metric_by_solution = {}
			for solution in solutions:
				try:
					metric_by_solution[solution] = np.array(plot_metrics[solution][k][lat_threshold])
					ax.plot(dpsizes, plot_metrics[solution][k][lat_threshold], label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
				except KeyError:
					continue
			ax.set_xlabel(xlab)
			ax.set_ylabel(ylab)
			ax.legend(fontsize=12)
			fig_fn = "percent_traffic_within_{}_ms_{}_over_{}.pdf".format(int(np.abs(lat_threshold)), fn, evaluate_over)
			save_figure(fig_fn)

			if 'lagrange' not in fn and 'penalty' not in fn:
				print("Within {} ms during {}".format(int(np.abs(lat_threshold)), fn))
				print("OPP - Painter: {}".format(np.round(100 - (metric_by_solution['one_per_peering'] - metric_by_solution['painter']),2)))
				print("OPP - Sparse: {}".format(np.round(100-(metric_by_solution['one_per_peering'] - metric_by_solution['sparse']),2)))
				for solution in solutions:
					print("{} average over all deployments: {}".format(solution,100-(np.average(metric_by_solution['one_per_peering']) - np.average(metric_by_solution[solution]))))
				print('\n')


	f,ax = get_figure()
	metric_by_solution = {}
	for solution in solutions:
		this_resiliences = []
		for dpsize in dpsizes:
			try:
				avg_resilience = np.average(list(metrics_by_dpsize[dpsize]['stats_resilience_to_congestion'][solution].values()))
				this_resiliences.append(avg_resilience)
			except KeyError:
				pass
		ax.plot(dpsizes[0:len(this_resiliences)], this_resiliences, label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
		metric_by_solution[solution] = np.array(this_resiliences)
	ax.set_xlabel(xlab)
	ax.set_ylabel("Flash Crowd Intensity (M)\nBefore Congestion (Pct.)",fontsize=12)
	ax.set_ylim([0,300])
	ax.set_yticks([0,60,120,180,240,300])
	ax.legend(fontsize=12)
	save_figure('flash_crowd_blowup_before_congestion_over_{}.pdf'.format(evaluate_over))
	print("--Flash Crowd--")
	print("Sparse / Painter: {}".format(100 * metric_by_solution['sparse'] / metric_by_solution['painter']))
	print("OPP - Sparse: {}".format(metric_by_solution['one_per_peering'] - metric_by_solution['sparse']))
	print("------------------------")
	for solution in solutions:
		print("{} : {}".format(solution,metric_by_solution[solution]))
	print('\n')


	solutions = ['anycast', 'anyopt', 'one_per_pop', 'painter', 'sparse', 'one_per_peering']
	f,ax = get_figure()
	for solution in solutions:
		this_resiliences = list([metrics_by_dpsize[dpsize]['stats_volume_multipliers'][solution][-1] - metrics_by_dpsize[dpsize]['stats_volume_multipliers'][solution][0] for dpsize in dpsizes])
		ax.plot(dpsizes, this_resiliences, label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
	ax.set_xlabel(xlab)
	ax.set_ylabel("Rate of\nLatency Increase (ms/byte)")
	ax.legend(fontsize=12)
	save_figure('latency_increase_up_to_threshold_over_{}.pdf'.format(evaluate_over))

	solutions = list(reversed(solutions))
	f,ax = get_figure()
	metric_by_solution = {}
	for solution in solutions:
		this_resiliences = []
		for dpsize in dpsizes:
			try:
				avg_resilience = np.average(list(metrics_by_dpsize[dpsize]['stats_diurnal'][solution].values()))
				this_resiliences.append(avg_resilience)
			except KeyError:
				pass
		ax.plot(dpsizes[0:len(this_resiliences)], this_resiliences, label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
		metric_by_solution[solution] = np.array(this_resiliences)
	ax.set_xlabel(xlab)
	ax.set_ylabel("Diurnal Intensity (M)\nBefore Congestion (Pct.)",fontsize=12)
	ax.set_ylim([0,120])
	ax.set_yticks([0,40,80,120])
	ax.legend(fontsize=12)
	save_figure('diurnal_blowup_before_congestion_over_{}.pdf'.format(evaluate_over))
	print("--Diurnal--")
	print("sparse / painter: {}".format(100 * metric_by_solution['sparse'] / metric_by_solution['painter']))
	print("OPP - Sparse: {}".format(metric_by_solution['one_per_peering'] - metric_by_solution['sparse']))
	print("------------------------")
	for solution in solutions:
		print("{} : {}".format(solution,metric_by_solution[solution]))
	print('\n')




if __name__ == '__main__':
	cache_fn = os.path.join(CACHE_DIR, 'evaluate_over_deployment_sizes_cache_fn.pkl')
	pull_results(cache_fn)
	# make_paper_plots(cache_fn)




