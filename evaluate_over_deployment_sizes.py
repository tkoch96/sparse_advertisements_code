from constants import *
from eval_latency_failure import evaluate_all_metrics
import numpy as np, os, pickle
np.random.seed(31700)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

# todo -- selectively compute each element of metrics_by_dpsize

def pull_results(cache_fn):
	cache_fn = os.path.join(CACHE_DIR, 'evaluate_over_deployment_sizes_cache_fn.pkl')

	only_recalc = [3] ## recalc these deployment sizes
	metrics_by_dpsize = {}
	if os.path.exists(cache_fn):
		metrics_by_dpsize = pickle.load(open(cache_fn, 'rb'))
	if not os.path.exists(cache_fn) or only_recalc is not None:
		import argparse
		parser = argparse.ArgumentParser()
		parser.add_argument("--port", required=True)
		args = parser.parse_args()
		# n_sim_by_dpsize = [15,20,10,16,4,5,6] # running now
		# n_sim_by_dpsize = [15,20,10,16,4,2,4] # already done
		n_sim_by_dpsize = [1,20,10,16,4,2,4] # testing
		for dpsize, nsim in zip([3,5,10,15,20,25,len(POP_TO_LOC['vultr'])], n_sim_by_dpsize):
			if only_recalc is not None:
				if dpsize not in only_recalc: continue
			print("Evaluating over deployment size {} Sites".format(dpsize))
			dpsize_str = "actual-{}".format(dpsize)
			if dpsize == 32:
				save_run_dir = [None, None, None, None, '1713454873-actual-32-sparse', None]
			elif dpsize == 25:
				save_run_dir = [None, None, '1713486580-actual-25-sparse', None, None]
			else:
				save_run_dir = None
			metrics = evaluate_all_metrics(dpsize_str, int(args.port), save_run_dir=save_run_dir, nsim=nsim)
			metrics_by_dpsize[dpsize] = {}
			for k in metrics:
				if 'stats' in k:
					metrics_by_dpsize[dpsize][k] = metrics[k]
		pickle.dump(metrics_by_dpsize, open(cache_fn, 'wb'))


	## second one is 4 keys * 3 latency thresholds per key
	n_subs = 5 + 6*3 + 1 + 1
	f,ax = plt.subplots(n_subs)
	f.set_size_inches(6,4*n_subs)
	dpsizes = sorted(list(metrics_by_dpsize))
	solutions = sorted(list(metrics_by_dpsize[dpsizes[0]]['stats_best_latencies']))

	ks = ['stats_popp_failures_latency_optimal_specific', 'stats_pop_failures_latency_optimal_specific']

	for solution in solutions:
		avg_latency_diff_normal = []
		agg_metrics = {k:{lab:[] for lab in ['latency','congestion']} for k in ks}
		for dpsize in dpsizes:
			avg_latency_diff_normal.append(-1*metrics_by_dpsize[dpsize]['stats_best_latencies'][solution])

			for k in ks:
				these_metrics = metrics_by_dpsize[dpsize][k][solution]
				agg_metrics[k]['latency'].append(-1*these_metrics['avg_latency_difference'])
				agg_metrics[k]['congestion'].append(these_metrics['frac_vol_congested'])
		ax[0].plot(dpsizes,avg_latency_diff_normal, label=solution)
		
		for i,k in enumerate(ks):
			ax[1+2*i].plot(dpsizes,agg_metrics[k]['latency'], label=solution)
			ax[1+2*i+1].plot(dpsizes,agg_metrics[k]['congestion'], label=solution)
	
	ax[0].set_xlabel("Deployment Size (Num Sites)")
	ax[0].set_ylabel("Average Latency Suboptimality\nNormal Operation (ms)")
	ax[0].legend()

	for i,lab in enumerate(['Ingress', 'Site']):
		ax[1+2*i].set_xlabel("Deployment Size (Num Sites)")
		ax[1+2*i].set_ylabel("Average Latency Suboptimality\n During {} Failure (ms)".format(lab))
		ax[1+2*i].legend()
		
		ax[1+2*i+1].set_xlabel("Deployment Size (Num Sites)")
		ax[1+2*i+1].set_ylabel("Fraction Volume Congested\nDuring {} Failure".format(lab))
		ax[1+2*i+1].legend()

	ax_start = 5
	ks = ['stats_latency_thresholds_normal', 'stats_latency_thresholds_fail_popp', 'stats_latency_thresholds_fail_pop',
		'stats_latency_penalty_thresholds_normal', 'stats_latency_penalty_thresholds_fail_popp', 'stats_latency_penalty_thresholds_fail_pop']
	parsed_metrics = {}
	for solution in solutions:
		lat_thresholds = {}
		agg_metrics = {k:{dpsize:{} for dpsize in dpsizes} for k in ks}
		for dpsize in dpsizes:
			for k in ks:
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
	for i,k in enumerate(ks):
		for j,lat_threshold in enumerate(sorted(lat_thresholds,reverse=True)):
			for solution in solutions:
				arr = list([100-100*np.mean(parsed_metrics[solution][k][dpsize][lat_threshold]) for dpsize in dpsizes])
				ax[ax_start+i*n_lat_thresholds+j].plot(dpsizes, arr, label="{}".format(solution))
			lab = ['Percent of Traffic Within\n {} ms of Optimal Normally'.format(int(np.abs(lat_threshold))), 'Percent of Traffic Within\n {} ms of Optimal Link Failure'.format(int(np.abs(lat_threshold))),
				 'Percent of Traffic Within\n {} ms of Optimal Site Failure'.format(int(np.abs(lat_threshold))),
				 'Percent of Traffic Within\n {} ms of Optimal Normally (Penalty)'.format(int(np.abs(lat_threshold))), 'Percent of Traffic Within\n {} ms of Optimal Link Failure (Penalty)'.format(int(np.abs(lat_threshold))),
				 'Percent of Traffic Within\n {} ms of Optimal Site Failure (Penalty)'.format(int(np.abs(lat_threshold))),][i]
			ax[ax_start+i*n_lat_thresholds+j].set_xlabel("Deployment Size (Num Sites)")
			ax[ax_start+i*n_lat_thresholds+j].set_ylabel(lab)
			ax[ax_start+i*n_lat_thresholds+j].legend()

	ax_start = 5 + 6*3
	for solution in solutions:
		this_resiliences = []
		for dpsize in dpsizes:
			try:
				avg_resilience = np.average(list(metrics_by_dpsize[dpsize]['stats_resilience_to_congestion'][solution].values()))
				this_resiliences.append(avg_resilience)
			except KeyError:
				pass
		ax[ax_start].plot(dpsizes[0:len(this_resiliences)], this_resiliences, label=solution)
	ax[ax_start].set_xlabel("Deployment Size (Num Sites)")
	ax[ax_start].set_ylabel("Flash Crowd Volume Blowup\nBefore Congestion")
	ax[ax_start].legend()



	ax_start = 5 + 6*3 + 1
	for solution in solutions:
		this_resiliences = list([metrics_by_dpsize[dpsize]['stats_volume_multipliers'][solution] for dpsize in dpsizes])
		ax[ax_start].plot(dpsizes, this_resiliences, label=solution)
	ax[ax_start].set_xlabel("Deployment Size (Num Sites)")
	ax[ax_start].set_ylabel("Average Latency over \nOne per Ingress (ms)")
	ax[ax_start].legend()



	plt.savefig('figures/stats_over_deployment_size.pdf')


def get_figure(l=7,h=3):
	plt.clf()
	plt.close()

	font = {'size'   : 14}
	matplotlib.rc('font', **font)
	f,ax = plt.subplots(1)
	f.set_size_inches(l,h)
	return f,ax

def save_figure(fn):
	plt.savefig(os.path.join('figures', 'paper', fn), bbox_inches='tight')
	plt.clf()
	plt.close()

def make_paper_plots(cache_fn):
	metrics_by_dpsize = pickle.load(open(cache_fn, 'rb'))
	dpsizes = sorted(list(metrics_by_dpsize))
	solutions = sorted(list(metrics_by_dpsize[dpsizes[0]]['stats_best_latencies']))

	solution_to_plot_label = {
		'sparse': 'SCULPTOR',
		'painter': 'PAINTER',
		'anyopt': 'AnyOpt',
		'anycast': 'Anycast',
		'one_per_pop': 'Unicast',
		'one_per_peering': 'One per Peering'
	}

	solution_to_line_color = {
		'sparse': 'magenta',
		'painter': 'black',
		'anyopt': 'orange',
		'anycast': 'midnightblue',
		'one_per_pop': 'red', 
		'one_per_peering': 'lawngreen',
	}
	solution_to_marker = {
		'sparse': '*',
		'painter': 'o',
		'anyopt': '>',
		'anycast': 'D',
		'one_per_pop': '+',
		'one_per_peering': '_',
	}


	solutions = ['anycast', 'anyopt', 'one_per_pop', 'painter', 'sparse', 'one_per_peering']
	f,ax = get_figure()
	for solution in solutions:
		avg_latency_diff_normal = []
		for dpsize in dpsizes:
			avg_latency_diff_normal.append(-1*metrics_by_dpsize[dpsize]['stats_best_latencies'][solution])
		ax.plot(dpsizes,avg_latency_diff_normal, label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
	ax.set_xlabel("Deployment Size (Num Sites)")
	ax.set_ylabel("Avg Suboptimality\nNormal Operation (ms)")
	ax.legend(fontsize=12)
	save_figure('average_latency_over_deployment_size_normal.pdf')
	

	xlab = "Deployment Size (Num Sites)"

	for tp, tp_k in zip(['Ingress', 'Site'],['stats_popp_failures_latency_optimal_specific', 'stats_pop_failures_latency_optimal_specific']):
		for metric_k, outer_k, ylab in zip(['avg_latency_difference', 'frac_vol_congested'], ['latency', 'congestion'], 
			['Avg Suboptimality\n During {} Failure (ms)'.format(tp), 'Pct Volume Congested\nDuring {} Failure'.format(tp)]):
			f,ax = get_figure()
			fig_fn = 'average_{}_over_deployment_size_fail_{}.pdf'.format(outer_k, tp.lower())
			for solution in solutions:
				agg_metrics =[]
				for dpsize in dpsizes:
					these_metrics = metrics_by_dpsize[dpsize][tp_k][solution]
					agg_metrics.append(-1*these_metrics[metric_k])
					if outer_k == 'congestion':
						agg_metrics[-1] *= 100
				ax.plot(dpsizes, agg_metrics, label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
			ax.set_xlabel(xlab)
			ax.set_ylabel(ylab)
			ax.legend(fontsize=12)
			save_figure(fig_fn)


	metric_access_ks = ['stats_latency_thresholds_normal', 'stats_latency_thresholds_fail_popp', 'stats_latency_thresholds_fail_pop',
		'stats_latency_penalty_thresholds_normal', 'stats_latency_penalty_thresholds_fail_popp', 'stats_latency_penalty_thresholds_fail_pop']

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
				arr = list([100-100*np.mean(parsed_metrics[solution][k][dpsize][lat_threshold]) for dpsize in dpsizes])		
				plot_metrics[solution][k][lat_threshold] = arr

	for lat_threshold in sorted(lat_thresholds):
		all_ylabs = ['Pct Traffic Within {} ms\n of Optimal Normally'.format(int(np.abs(lat_threshold))), 'Pct Traffic Within {} ms\n of Optimal Link Failure'.format(int(np.abs(lat_threshold))),
			 'Pct Traffic Within {} ms\n of Optimal Site Failure'.format(int(np.abs(lat_threshold))),
			 'Pct Traffic Within {} ms\n of Optimal Normally (Penalty)'.format(int(np.abs(lat_threshold))), 'Pct Traffic Within {} ms\n of Optimal Link Failure (Penalty)'.format(int(np.abs(lat_threshold))),
			 'Pct Traffic Within {} ms\n of Optimal Site Failure (Penalty)'.format(int(np.abs(lat_threshold)))]
		fns = ['normal','link_failure', 'site_failure', 'normal_penalty', 'link_failure_penalty', 'site_failure_penalty']
		for k, ylab, fn in zip(metric_access_ks, all_ylabs, fns):
			if 'penalty' in fn: continue ## ignore for now
			f,ax = get_figure()
			for solution in solutions:
				ax.plot(dpsizes, plot_metrics[solution][k][lat_threshold], label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
			ax.set_xlabel(xlab)
			ax.set_ylabel(ylab)
			ax.legend(fontsize=12)
			fig_fn = "percent_traffic_within_{}_ms_{}.pdf".format(int(np.abs(lat_threshold)), fn)
			save_figure(fig_fn)


	f,ax = get_figure()
	for solution in solutions:
		this_resiliences = []
		for dpsize in dpsizes:
			try:
				avg_resilience = np.average(list(metrics_by_dpsize[dpsize]['stats_resilience_to_congestion'][solution].values()))
				this_resiliences.append(avg_resilience)
			except KeyError:
				pass
		ax.plot(dpsizes[0:len(this_resiliences)], this_resiliences, label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
	ax.set_xlabel(xlab)
	ax.set_ylabel("Flash Crowd Volume Blowup\nBefore Congestion (Pct.)")
	ax.legend(fontsize=12)
	save_figure('flash_crowd_blowup_before_congestion.pdf')


	f,ax = get_figure()
	for solution in solutions:
		this_resiliences = list([metrics_by_dpsize[dpsize]['stats_volume_multipliers'][solution] for dpsize in dpsizes])
		ax.plot(dpsizes, this_resiliences, label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
	ax.set_xlabel(xlab)
	ax.set_ylabel("Worst Latency Compared to \nOne per Peering (ms)")
	ax.legend(fontsize=12)
	save_figure('latency_increase_up_to_threshold.pdf')




if __name__ == '__main__':
	cache_fn = os.path.join(CACHE_DIR, 'evaluate_over_deployment_sizes_cache_fn.pkl')
	pull_results(cache_fn)
	# make_paper_plots(cache_fn)




