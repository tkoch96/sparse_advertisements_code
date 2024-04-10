from constants import *
from eval_latency_failure import evaluate_all_metrics
import numpy as np, os, pickle
np.random.seed(31455)

# todo -- selectively compute each element of metrics_by_dpsize

if __name__ == '__main__':
	cache_fn = os.path.join(CACHE_DIR, 'evaluate_over_deployment_sizes_cache_fn.pkl')

	only_recalc = None #[3,20,25,len(POP_TO_LOC['vultr'])] ## recalc these deployment sizes
	metrics_by_dpsize = {}
	if os.path.exists(cache_fn):
		metrics_by_dpsize = pickle.load(open(cache_fn, 'rb'))
	if not os.path.exists(cache_fn) or only_recalc is not None:
		# n_sim_by_dpsize = [15,10,10,3,2,2,2]
		# n_sim_by_dpsize = [20,20,15,3,2,2,2]
		n_sim_by_dpsize = [15,20,10,3,2,2,2]
		for dpsize, nsim in zip([3,5,10,15,20,25,len(POP_TO_LOC['vultr'])], n_sim_by_dpsize):
			if dpsize < 15: continue
			if only_recalc is not None:
				if dpsize not in only_recalc: continue
			print("Evaluating over deployment size {} PoPs".format(dpsize))
			dpsize_str = "actual-{}".format(dpsize)
			metrics = evaluate_all_metrics(dpsize_str, nsim=nsim)
			metrics_by_dpsize[dpsize] = {}
			for k in metrics:
				if 'stats' in k:
					metrics_by_dpsize[dpsize][k] = metrics[k]
		pickle.dump(metrics_by_dpsize, open(cache_fn, 'wb'))


	import matplotlib.pyplot as plt
	n_subs = 8
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
				agg_metrics[k]['congestion'].append(these_metrics['frac_vol_congested'] - these_metrics['frac_vol_bestcase_congested'])
		ax[0].plot(dpsizes,avg_latency_diff_normal, label=solution)
		
		for i,k in enumerate(ks):
			ax[1+2*i].plot(dpsizes,agg_metrics[k]['latency'], label=solution)
			ax[1+2*i+1].plot(dpsizes,agg_metrics[k]['congestion'], label=solution)
	
	ax[0].set_xlabel("Deployment Size (Num PoPs)")
	ax[0].set_ylabel("Average Latency Suboptimality\nNormal Operation (ms)")
	ax[0].legend()

	for i,lab in enumerate(['PoPP', 'PoP']):
		ax[1+2*i].set_xlabel("Deployment Size (Num PoPs)")
		ax[1+2*i].set_ylabel("Average Latency Suboptimality\n During {} Failure (ms)".format(lab))
		ax[1+2*i].legend()
		
		ax[1+2*i+1].set_xlabel("Deployment Size (Num PoPs)")
		ax[1+2*i+1].set_ylabel("Fraction Volume \nCongested")
		ax[1+2*i+1].legend()

	ax_start = 5
	ks = ['stats_latency_thresholds_normal', 'stats_latency_thresholds_fail_popp', 'stats_latency_thresholds_fail_pop']
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
		for i,k in enumerate(ks):
			for lat_threshold in sorted(lat_thresholds):
				if lat_threshold != -10: continue
				arr = list([np.mean(agg_metrics[k][dpsize][lat_threshold]) for dpsize in dpsizes])
				ax[ax_start+i].plot(dpsizes, arr, label="{} - {}".format(solution, lat_threshold))
	
	labs = ['Percent of Users Within\n Latency Threshold', 'Percent of Users Within\n Latency Threshold PoPP Failure', 'Percent of Users Within\n Latency Threshold PoP Failure']
	for i,lab in enumerate(labs):
		ax[ax_start+i].set_xlabel("Deployment Size (Num PoPs)")
		ax[ax_start+i].set_ylabel(lab)
		ax[ax_start+i].legend()





	plt.savefig('figures/stats_over_deployment_size.pdf')

