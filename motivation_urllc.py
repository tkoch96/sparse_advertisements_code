from constants import *
from helpers import *

import pickle, numpy as np, matplotlib.pyplot as plt, copy, itertools, time
from sparse_advertisements_v3 import *


def simple_motivation_urllc(sas):
	def lat_bw_to_high_level_metric(lat, bw):
		lats_of_interest = np.array([0,1,5,20,50,100,300,1000])
		lat = np.minimum(lats_of_interest[-1], lat)
		lat_region = np.where(lat - lats_of_interest <= 0)[0][0]

		bws_of_interest = np.array([0, .6, .8, .9, .95, 1.0])
		bw = np.minimum(bws_of_interest[-1],bw)
		bw_region = np.where(bw - bws_of_interest <= 0)[0][0]

		# want to maximize reward
		reward = -1 * (lat_region + bw_region)
		return reward

	adv = sas.painter_solution['advertisement']
	routed_through_ingress, _ = sas.calculate_ground_truth_ingress(adv)

	user_latencies, ug_ingress_decisions = sas.calculate_user_choice(adv)
	link_volumes = np.zeros(sas.n_popp)
	ingress_to_users = {}
	for ugi, ingress_i in ug_ingress_decisions.items():
		if ingress_i is None: continue
		link_volumes[ingress_i] += sas.ug_vols[ugi]
	utilizations = (link_volumes / sas.link_capacities_arr).clip(0,1.0)
	print(utilizations)

	perfs_by_pfx = NO_ROUTE_LATENCY * np.ones((sas.n_ug, sas.n_prefixes))
	for pfxi in range(sas.n_prefixes):
		for ugi,ug in enumerate(sas.ugs):
			routed_ingress = routed_through_ingress[pfxi].get(ug)
			if routed_ingress is None: continue
			popp = sas.popps[routed_ingress]
			perfs_by_pfx[ugi,pfxi] = sas.ug_perfs[ug][popp]
	ranked_prefs = np.argsort(perfs_by_pfx,axis=1)

	best_perfs = {}
	for ug in sas.ug_perfs:
		best_perf = np.min(list(sas.ug_perfs[ug].values()))
		best_perfs[sas.ug_to_ind[ug]] = best_perf

	# simple solution -- switch traffic to backup at a utilization threshold
	threshold = .7 

	metrics,optimals = {},{}

	for ui in range(sas.n_ug): ### TODO -- fill this in
		primary = np.where(ranked_prefs[ui,:] == 0)[0][0]
		backup = np.where(ranked_prefs[ui,:] == 1)[0][0]

		## util primary assumes ui already using the link
		backup_switch = lat_bw_to_high_level_metric(perfs_by_pfx[ui,backup] - best_perfs[ui], utilizations[backup] + sas.ug_vols[ui] / sas.link_capacities_arr[backup])
		primary_switch = lat_bw_to_high_level_metric(perfs_by_pfx[ui,primary] - best_perfs[ui], utilizations[primary])
		if utilizations[primary] > threshold:
			# do backup
			metrics[ui] = backup_switch
		else:
			metrics[ui] = primary_switch

		## then compute optimal decision
		optimals[ui] = np.maximum(backup_switch,primary_switch)

	# positive delta means optimal high
	# metric is a measure of reward, so higher is better
	deltas = list([optimals[ui] - metrics[ui] for ui in metrics])
	x,cdf_x = get_cdf_xy(deltas)
	plt.plot(x,cdf_x)
	plt.grid(True)
	plt.xlabel("Optimal Metric - Straw Man Metric")
	plt.ylabel("CDF of Users")
	plt.savefig('figures/urllc_motivation.pdf')
	plt.clf(); plt.close()
	exit(0)


def motivation_urllc():
	np.random.seed(31413)
	metrics = {}
	N_TO_SIM = 1
	X_vals = [20]
	gamma = 10
	capacity = True

	lambduh = .000001

	wm = None
	
	soln_types = ['painter']
	metrics_fn = os.path.join(CACHE_DIR, 'motivation_urllc_{}.pkl'.format(DPSIZE))
	metrics = {
		'high_level_metric': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'adv': {i:{k:[] for k in soln_types} for i in range(N_TO_SIM)},
		'deployment': {i:None for i in range(N_TO_SIM)},
		'settings': {i:None for i in range(N_TO_SIM)},
		'ug_to_vol': {i:None for i in range(N_TO_SIM)},
	}
	if os.path.exists(metrics_fn):
		metrics = pickle.load(open(metrics_fn,'rb'))
	try:
		for random_iter in range(N_TO_SIM):
			try:
				metrics['high_level_metric'][random_iter]
				if len(metrics['high_level_metric'][random_iter][soln_types[0]]) > 0: 
					continue
			except KeyError:
				pass
			print("-----Deployment number = {} -------".format(random_iter))

			deployment = get_random_deployment(DPSIZE)
			n_prefixes = np.maximum(4,5 * int(np.log2(len(deployment['popps']))))
			n_prefixes = np.minimum(len(deployment['popps'])//4, n_prefixes)
			sas = Sparse_Advertisement_Eval(deployment, verbose=True,
				lambduh=lambduh,with_capacity=capacity,explore=DEFAULT_EXPLORE, 
				using_resilience_benefit=True, gamma=gamma, n_prefixes=n_prefixes)
			metrics['settings'][random_iter] = sas.get_init_kwa()
			if wm is None:
				wm = Worker_Manager(sas.get_init_kwa(), deployment)
				wm.start_workers()
			sas.set_worker_manager(wm)
			sas.update_deployment(deployment)
			sas.solve_painter()

			simple_motivation_urllc(sas)

			pickle.dump(metrics, open(metrics_fn,'wb'))
	except:
		import traceback
		traceback.print_exc()
		exit(0)
	finally:
		if wm is not None:
			wm.stop_workers()
	

	for solution in soln_types:
		print("Solution {}".format(solution))
		for i in range(N_TO_SIM):
			adv = metrics['adv'][i][solution]
			deployment = metrics['deployment'][i]
			popps = sorted(list(set(deployment['popps'])))
			try:
				adv_summary(popps,adv)
				n_prefs_by_popp = np.sum(adv,axis=1)
				x,cdf_x = get_cdf_xy(n_prefs_by_popp)
				plt.plot(x,cdf_x,label=solution)
				
			except:
				continue
	plt.grid(True)
	plt.xlabel("Number of Prefixes Being Advertised via Ingress")
	plt.ylabel("CDF of PoPPs")
	plt.legend()
	plt.savefig("n_prefs_by_popp_solutions.pdf")
	plt.clf(); plt.close()

	f,ax=plt.subplots(3,1)
	f.set_size_inches(6,17)


	for solution in soln_types:
		try:
			all_differences = [el for ri in range(N_TO_SIM) for el in metrics['popp_failures'][ri][solution]]
			x,cdf_x = get_cdf_xy(all_differences,weighted=True)
			ax[0].plot(x,cdf_x,label=solution)

			diffs = []
			wts = []
			for random_iter in range(N_TO_SIM):
				for i in range(len(metrics['best_latencies'][random_iter])):
					diffs.append(metrics['best_latencies'][random_iter][i] - metrics['latencies'][random_iter][solution][i])
					wts.append(metrics['ug_to_vol'][random_iter][i])

			x,cdf_x = get_cdf_xy(list(zip(diffs,wts)), weighted=True)
			ax[1].plot(x,cdf_x,label=solution)


			m = metrics['resilience_to_congestion']
			for X in X_vals:
				x,cdf_x = get_cdf_xy(list([el for ri in range(N_TO_SIM) for el in m[ri][solution][X] ]))
				ax[2].plot(x,cdf_x,label="{} Drain pct={}".format(solution,X))


		except:
			import traceback
			traceback.print_exc()
			continue

	ax[0].legend(fontsize=8)
	ax[0].grid(True)
	ax[0].set_xlabel("Best - Actual Latency Under Failure (ms)")
	ax[0].set_ylabel("CDF of UGs")
	ax[0].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[0].set_xlim([-50,0])

	ax[1].legend(fontsize=8)
	ax[1].grid(True)
	ax[1].set_xlabel("Best - Actual Latency (ms)")
	ax[1].set_ylabel("CDF of UGs")
	ax[1].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	ax[1].set_xlim([-50,0])

	ax[2].set_xlabel("Latency Change (new - old) (ms)")
	ax[2].set_ylabel("CDF of UGs")
	ax[2].grid(True)
	ax[2].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	# ax[2].set_xlim([0,150])
	ax[2].legend(fontsize=8)

	save_fig("popp_latency_failure_comparison_{}.pdf".format(DPSIZE))


if __name__ == '__main__':
	motivation_urllc()




