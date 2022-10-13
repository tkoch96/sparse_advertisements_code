from constants import *

import pickle, numpy as np, matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True
})

N_SIM = 50

import multiprocessing
from sparse_advertisements_v3 import *


def do_eval_compare_peer_value(args):
	worker_n, = args

	metrics = {}
	if worker_n > 0:
		metrics_fns = [os.path.join(CACHE_DIR, 'compare_peer_value_{}.pkl'.format(worker_n))]
	else:
		import glob
		metrics_fns = glob.glob(os.path.join(CACHE_DIR, 'compare_peer_value*.pkl'))
	for metrics_fn in metrics_fns:
		if os.path.exists(metrics_fn):
			tmp = pickle.load(open(metrics_fn,'rb'))
			for k,v in tmp.items():
				try:
					metrics[k] = metrics[k] + v
				except KeyError:
					metrics[k] = v
	for i in range(N_SIM):
		print("Worker: {} CPV: {}".format(worker_n,i))
		if metrics != {}:
			if len([metrics[k] for k in metrics][0]) >= N_SIM:
				break
		ret  = None
		while ret is None:
			gen_random_graph_('compare_pv_test_graph_{}'.format(worker_n),n_transit=2,n_user=5+np.random.randint(5))
			lambduh = .001
			sae = Sparse_Advertisement_Eval(graph_fn='compare_pv_test_graph_{}.csv'.format(worker_n), 
				graph_md_fn='compare_pv_test_graph_{}_md.json'.format(worker_n),lambduh=lambduh,verbose=False,
				init={'type':'using_objective'})
			ret = sae.compare_peer_value(make_plots=False)
		for k in ret:
			try:
				metrics[k].append(ret[k])
			except KeyError:
				metrics[k] = [ret[k]]
		pickle.dump(metrics, open(metrics_fn,'wb'))

	plt.rcParams["figure.figsize"] = (10,5)
	plt.rcParams.update({'font.size': 22})
	f,ax=plt.subplots(1,1)
	labs = {"frac_disagree": "Disagreement with Oracle", "lambda_j_distances": "Uncertainty", 
		"lambda_j_distances_not_transit": "Not-Transit Uncertainty"}
	for k in ['frac_disagree']:
		x,cdf_x = get_cdf_xy(metrics[k])
		ax.plot(x,cdf_x,label=labs[k])
	ax.set_ylim([0,1.0])
	ax.set_xlabel("Fraction of Peer Importance Disagreement with Oracle")
	ax.grid(True)
	ax.set_ylabel("CDF of Trials")
	save_fig("peer_value_disagreement.pdf")


	plt.rcParams["figure.figsize"] = (10,5)
	plt.rcParams.update({'font.size': 22})
	f,ax=plt.subplots(1,1)
	for k in ['lambda_j_distances', 'lambda_j_distances_not_transit']:
		v = [np.max(arr) for arr in metrics[k]]
		x,cdf_x = get_cdf_xy(v)
		ax.plot(x,cdf_x,label=labs[k])
	ax.set_ylim([0,1.0])
	ax.set_xlabel("$\lambda_j$ Uncertainty")
	ax.legend()
	ax.grid(True)
	ax.set_ylabel("CDF of Trials")
	save_fig("lambda_j_uncertainty.pdf")


def err_adv(adv1,adv2):
	# advertisements are said to be equal if their columns are a permutation of each other
	# so find the minimum error over all permutations of columns

	n_col = adv1.shape[1]
	errs = []
	for p in itertools.permutations(list(range(n_col)), n_col):
		tmp = adv1[:,p]
		errs.append(np.sum(np.abs(tmp.flatten() - adv2.flatten())))
	return min(errs)


def do_eval_compare_explores():
	N_SIM = 10
	lambduh = .1
	explores = ['other_bimodality','gmm','positive_benefit', 'entropy', 'bimodality']
	hr_labs = ['other_bimodality','gmm',"Positive Benefit", "Entropy", "Bimodality"]
	
	metrics = {}
	metrics_fn = os.path.join(CACHE_DIR, 'compare_explores.pkl')
	if os.path.exists(metrics_fn):
		metrics = pickle.load(open(metrics_fn,'rb'))

	for explore_i, explore in enumerate(explores):
		if explore_i in metrics: continue
		metrics[explore_i] = {
			'n_advs': [],
			'obj': [],
		}
		for _i in range(N_SIM):
			sae = Sparse_Advertisement_Eval(get_random_deployment('really_friggin_small'), 
				lambduh=lambduh,verbose=False,with_capacity=False, explore=explore,n_prefixes=2)
			ret = sae.compare_different_solutions(n_run=1,verbose=False)
			our_adv = sae.threshold_a(ret['advertisements']['sparse'][0])
			metrics[explore_i]['obj'].append(ret['objectives']['sparse'][0])
			metrics[explore_i]['n_advs'].append(sae.sas.path_measures)
	pickle.dump(metrics,open(metrics_fn,'wb'))

	plt.rcParams["figure.figsize"] = (10,5)
	plt.rcParams.update({'font.size': 22})
	f,ax=plt.subplots(1,1)
	for i in metrics:
		obj = metrics[i]['obj']
		x,cdf_x = get_cdf_xy(obj)
		ax.plot(x,cdf_x,c=cols[2*i+1],label=hr_labs[i].capitalize())
	ax.legend()
	ax.set_ylim([0,1.0])
	ax.grid(True)
	ax.set_xlabel("Objective Function Value")
	ax.set_ylabel("CDF of Trials")
	save_fig("compare_explores_objective.pdf")

	plt.rcParams["figure.figsize"] = (10,5)
	plt.rcParams.update({'font.size': 22})
	f,ax=plt.subplots(1,1)
	for i in metrics:
		obj,n_adv = metrics[i]['obj'],metrics[i]['n_advs']
		ax.scatter(obj,n_adv,c=cols[2*i+1],label=hr_labs[i].capitalize())
	ax.legend()
	ax.grid(True)
	ax.set_xlabel("Objective Function")
	ax.set_ylabel("Number of Advs")
	save_fig("compare_explores_nadv_obj_scatter.pdf")

def do_eval_compare_initializations():
	lambduh = .1
	inits = [{'type': 'uniform'}, {'type': 'ones'}, {'type': 'zeros'}, {'type':'random_binary'},
		{'type': 'normal', 'var': .01}, {'type': 'normal', 'var': .001}, {'type': 'using_objective'}]
	hr_labs = ["Uniform","All On","All Off","Random","N(.5,.01)","N(.5,.001)",'Custom']
	
	metrics = {}
	metrics_fn = os.path.join(CACHE_DIR, 'compare_inits.pkl')
	if os.path.exists(metrics_fn):
		metrics = pickle.load(open(metrics_fn,'rb'))

	for init_i, initialization in enumerate(inits):
		if init_i in metrics: continue
		metrics[init_i] = {
			'n_off_adv_approx': [],
			'n_off_adv_l0': [],
			'delta_obj_approx': [],
			'delta_obj_l0': [],
			'delta_obj_l0_greedy': [],
		}
		for _i in range(N_SIM):
			ret_oracle = None
			while ret_oracle is None:
				gen_random_graph_('test_graph',n_transit=1,n_user=4)
				sae = Sparse_Advertisement_Eval(graph_fn="test_graph.csv", 
					graph_md_fn="test_graph_md.json", lambduh=lambduh,
					verbose=False,cont_grads=False, init=initialization)
				ret = sae.compare_different_solutions(n_run=1,verbose=False)
				ret_oracle = ret['objectives']['approx_oracle']
			our_adv = sae.threshold_a(ret['advertisements']['ours'][0])
			metrics[init_i]['n_off_adv_approx'].append(err_adv(our_adv,ret['advertisements']['approx_oracle']))
			metrics[init_i]['n_off_adv_l0'].append(err_adv(our_adv,ret['advertisements']['l0_oracle']))
			metrics[init_i]['delta_obj_approx'].append(np.abs(ret['objectives']['ours'][0] - \
				ret['objectives']['approx_oracle']))
			metrics[init_i]['delta_obj_l0'].append(np.abs(ret['objectives']['ours'][0] - \
				ret['objectives']['l0_oracle']))
			metrics[init_i]['delta_obj_l0_greedy'].append(np.abs(ret['objectives']['sparse greedy'][0] - \
				ret['objectives']['l0_oracle']))
	pickle.dump(metrics,open(metrics_fn,'wb'))
	
	plt.rcParams["figure.figsize"] = (10,5)
	plt.rcParams.update({'font.size': 22})
	f,ax=plt.subplots(1,1)
	for i in metrics:
		# # % of runs we arrive within N entries of optimal advertisement
		# n_off_approx_optimal = metrics[i]['n_off_adv_approx']
		# x,cdf_x = get_cdf_xy(n_off_approx_optimal)
		# ax[0].plot(x,cdf_x,c=cols[2*i],label=hr_labs[i] + " Approx")
		n_off_optimal = metrics[i]['n_off_adv_l0']
		x,cdf_x = get_cdf_xy(n_off_optimal)
		ax.plot(x,cdf_x,c=cols[2*i+1],label=hr_labs[i].capitalize())
	ax.legend()
	ax.grid(True)
	ax.set_xlabel("Number of Entries off Optimal")
	ax.set_ylabel("CDF of Trials")
	save_fig("compare_initializations_delta_advertisement.pdf")

	plt.rcParams["figure.figsize"] = (10,5)
	plt.rcParams.update({'font.size': 22})
	f,ax=plt.subplots(1,1)
	for i in metrics:
		# # % of runs we arrive within epsilon of optimal cost-beenfit
		# epsilon_off_approx_optimal = metrics[i]['delta_obj_approx']
		# x,cdf_x = get_cdf_xy(epsilon_off_approx_optimal)
		# ax[1].plot(x,cdf_x,c=cols[2*i],label=hr_labs[i] + " Approx")
		epsilon_off_optimal = metrics[i]['delta_obj_l0']
		x,cdf_x = get_cdf_xy(epsilon_off_optimal)
		ax.plot(x,cdf_x,c=cols[2*i+1],label=hr_labs[i].capitalize())
	ax.legend()
	ax.set_ylim([0,1.0])
	ax.grid(True)
	ax.set_xlim([0,1.0])
	ax.set_xlabel("Objective Function Error")
	ax.set_ylabel("CDF of Trials")
	save_fig("compare_initializations_delta_objective.pdf")

def do_eval_compare_strategies():
	lambduh = .1
	metrics = {}
	metrics_fn = os.path.join(CACHE_DIR, 'compare_strategies.pkl')
	if os.path.exists(metrics_fn):
		metrics = pickle.load(open(metrics_fn,'rb'))
	for cost_strat in ['sigmoid', 'l1']:
		if cost_strat not in metrics: 
			metrics[cost_strat] = {'p_ours_best': []}
		for _i in range(N_SIM):
			if len(metrics[cost_strat]) > _i: continue
			ret = None
			while ret is None:
				gen_random_graph_('test_graph_cs',n_transit=2,n_user=5+np.random.randint(5))
				sae = Sparse_Advertisement_Eval(graph_fn="test_graph_cs.csv", 
					graph_md_fn="test_graph_cs_md.json", lambduh=lambduh,
					verbose=False,advertisement_cost=cost_strat,
					n_prefixes=1+np.random.randint(3),
					init={'type':'using_objective'})
				ret = sae.compare_different_solutions(n_run=30,verbose=False)
			our_obj = ret['objectives']['ours']
			ours_best = np.ones((len(our_obj)))
			print(ret['objectives'])
			for k in ret['objectives']:
				if k == 'ours' or 'oracle' in k: continue
				if type(ret['objectives'][k]) == np.float64:
					for i, obj in enumerate(our_obj):
						if obj > ret['objectives'][k]:
							ours_best[i] = 0
				else:
					i=0
					for ours, theirs in zip(our_obj, ret['objectives'][k]):
						if ours > theirs:
							ours_best[i] = 0
						i += 1
			metrics[cost_strat]['p_ours_best'].append(np.sum(ours_best) / len(our_obj))
	
			pickle.dump(metrics,open(metrics_fn,'wb'))
	
	plt.rcParams["figure.figsize"] = (10,5)
	plt.rcParams.update({'font.size': 22})
	f,ax=plt.subplots(1,1)
	labs = {'sigmoid': "Sigmoid Cost", 'l1': "L1 Cost"}
	print(metrics['l1']['p_ours_best'])
	print(metrics['sigmoid']['p_ours_best'])
	for k in metrics:
		x, cdf_x = get_cdf_xy(1 - np.array(metrics[k]['p_ours_best']))
		ax.plot(x,cdf_x,label=labs[k])
	ax.legend()
	ax.set_ylim([0,1.0])
	ax.set_ylabel("CDF of Trial Batches")
	ax.set_xlabel("Fraction of Trials Our Algorithm Was Not The Best")
	save_fig("compare_across_strategies.pdf")

def do_eval_scale():
	metrics = {
		't_per_iter': [],
		'n_iter': [],
		'n_users': [],
		'n_peers': [],
		'n_advs': {
			'sparse_greedy': [],
			'ours_entropy': [],
			'ours_positive_benefit': [],
		},
	}
	metrics_fn = os.path.join(CACHE_DIR, 'scale_eval.pkl')
	if os.path.exists(metrics_fn):
		metrics = pickle.load(open(metrics_fn,'rb'))
	lambduh = .1
	# for n_user in [5,10,15,20,25,30,35,40]:
	# 	print("NU: {}".format(n_user))
	# 	for _i in range(N_SIM):
	# 		print(_i)
	# 		gen_random_graph_('test_graph',n_transit=1,n_user=n_user)
	# 		sae = Sparse_Advertisement_Eval(graph_fn="test_graph.csv", 
	# 			graph_md_fn="test_graph_md.json", lambduh=lambduh,
	# 			verbose=False,	explore='entropy')
	# 		ret = sae.compare_different_solutions(n_run=1,verbose=False)
	# 		metrics['t_per_iter'].append(sae.sas.t_per_iter)
	# 		metrics['n_iter'].append(sae.sas.iter)
	# 		metrics['n_advs']['sparse_greedy'].append(ret['n_advs']['sparse_greedy'][0])
	# 		metrics['n_advs']['ours_entropy'].append(ret['n_advs']['ours'][0])
	# 		gen_random_graph_('test_graph',n_transit=1,n_user=n_user)
	# 		sae = Sparse_Advertisement_Eval(graph_fn="test_graph.csv", 
	# 			graph_md_fn="test_graph_md.json", lambduh=lambduh,
	# 			verbose=False, explore='positive_benefit')
	# 		ret = sae.compare_different_solutions(n_run=1,verbose=False)
	# 		metrics['n_advs']['ours_positive_benefit'].append(ret['n_advs']['ours'][0])

	# 		metrics['n_users'].append(len(sae.sas.user_networks))
	# 		metrics['n_peers'].append(len(sae.sas.peers))

	# pickle.dump(metrics,open(metrics_fn,'wb'))

	plt.rcParams["figure.figsize"] = (10,3)
	plt.rcParams.update({'font.size': 22})
	f,ax = plt.subplots(1,1)
	
	x = np.array(metrics['n_users']) * np.array(metrics['n_peers'])	
	y = np.array(metrics['t_per_iter'])
		
	z = np.poly1d(np.polyfit(x,y,2))
	x_sm = np.linspace(x[0],x[-1])
	y_sm = z(x_sm)
	ax.plot(x_sm, y_sm,'r',marker='.')
	ax.tick_params(axis='y', colors='red')
	ax.yaxis.label.set_color('red')
	ax.set_ylabel("Time per Iter (s)")

	axd = ax.twinx()
	y = np.array(metrics['n_iter'])
	z = np.poly1d(np.polyfit(x,y,2))
	y_sm = z(x_sm)
	axd.plot(x_sm, y_sm,'k')
	axd.tick_params(axis='y', colors='black')
	axd.set_ylabel("Number of Iters")

	ax.set_xlabel("Problem Size")

	save_fig("scale_time.pdf")


	plt.rcParams["figure.figsize"] = (10,3)
	plt.rcParams.update({'font.size': 22})
	f,ax = plt.subplots(1,1)

	x = np.array(metrics['n_users']) * np.array(metrics['n_peers'])	
	labs = {
		'sparse_greedy': "Greedy",
		"ours_entropy": "Entropy Exploration",
		"ours_positive_benefit": "Max Benefit Exploration"
	}
	for i,k in enumerate(metrics['n_advs']):
		y = metrics['n_advs'][k]
		z = np.poly1d(np.polyfit(x,y,2))
		y_sm = z(x_sm)
		ax.plot(x_sm,y_sm,c=cols[3*i],label=labs[k])
	ax.legend()
	ax.set_xlabel("Problem Size")
	ax.set_ylabel("Number of Advertisements")
	save_fig("scale_n_advs.pdf")

if __name__ == "__main__":
	# all_args = []
	# n_workers = multiprocessing.cpu_count() // 2
	# for i in range(n_workers):
	# 	all_args.append((i,))
	# ppool = multiprocessing.Pool(processes=n_workers)
	# print("Launching workers")
	# all_rets = ppool.map(do_eval_compare_peer_value, all_args)
	# do_eval_compare_peer_value((-1,))
	# do_eval_compare_strategies()
	do_eval_compare_explores()