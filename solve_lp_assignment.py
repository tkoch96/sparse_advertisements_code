import numpy as np, matplotlib.pyplot as plt
from helpers import *

def solve_lp_assignment(sas, adv):
	## get vector of latency of all users to all (available) ingresses
	ret,_ = sas.calculate_ground_truth_ingress(adv)
	all_ug_lat_ingresses = {}
	for prefixi in ret:
		for ug in ret[prefixi]:
			poppi = ret[prefixi][ug]
			all_ug_lat_ingresses[ug,poppi] = None
	available_paths = sorted(list(all_ug_lat_ingresses), key = lambda el : el[0])
	available_latencies = np.array([sas.ug_perfs[ug][sas.popps[poppi]] for 
		ug,poppi in available_paths])
	n_paths = len(available_paths)

	cap_constraint_A = np.zeros((sas.n_popps, n_paths))
	caps = np.reshape(sas.link_capacities_arr, (sas.n_popps,1))
	print(cap_constraint_A.shape)
	print(caps.shape)

	volume_conservation_A = np.zeros((sas.n_ug, n_paths))
	conservation_b = np.ones((sas.n_ug,1))

	for pli in range(n_paths):
		poppi = available_paths[pli][1]
		ugi = sas.ug_to_ind[available_paths[pli][0]]
		cap_constraint_A[poppi,pli] = 1
		volume_conservation_A[ugi,pli] = 1
	
	import scipy
	res = scipy.optimize.linprog(available_latencies, 
		A_ub=cap_constraint_A, b_ub=caps,
		A_eq = volume_conservation_A, b_eq = conservation_b)
	distribution = res.x.round(2)

	paths_by_ug = {}
	for (ug,poppi),vol_pct in zip(available_paths, distribution):
		if vol_pct > 0:
			try:
				paths_by_ug[ug].append((poppi, vol_pct))
			except KeyError:
				paths_by_ug[ug] = [(poppi, vol_pct)]

	inflations = []
	for ug, pathvols in paths_by_ug.items():
		best_lat = sas.ug_perfs[ug][sas.best_popp_by_ug[ug]]
		these_deltas = []
		cum_vol = 0
		for poppi,vol in pathvols:
			popp = sas.popps[poppi]
			these_deltas.append((sas.ug_perfs[ug][popp] - best_lat, vol))
			cum_vol += vol
		avg_inflation = np.sum([el[0] * el[1] for el in these_deltas]) / cum_vol
		inflations.append((avg_inflation, sas.ug_to_vol[ug]))

	# x,cdf_x = get_cdf_xy(inflations, weighted=True)
	# plt.plot(x,cdf_x)
	# plt.xlabel("Inflation (ms)")
	# plt.ylabel("CDF of Traffic")
	# plt.grid(True)
	# plt.show()

	# exit(0)