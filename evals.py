from constants import *
from helpers import *

import pickle, numpy as np, matplotlib.pyplot as plt

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
	for lambduh in [.01,.1]:
		N_SIM = 40
		dpsize = 'really_friggin_small'
		explores = ['other_bimodality','gmm','positive_benefit', 'entropy', 'bimodality']
		hr_labs = ['other_bimodality','gmm',"Positive Benefit", "Entropy", "Bimodality"]
		
		metrics = {}
		metrics_fn = os.path.join(CACHE_DIR, 'compare_explores_{}_{}.pkl'.format(lambduh,dpsize))
		if os.path.exists(metrics_fn):
			metrics = pickle.load(open(metrics_fn,'rb'))

		for explore_i, explore in enumerate(explores):
			if explore_i in metrics: continue
			metrics[explore_i] = {
				'n_advs': [],
				'obj': [],
			}
			for _i in range(N_SIM):
				deployment = get_random_deployment(dpsize)
				sas = Sparse_Advertisement_Eval(deployment, verbose=False,
					lambduh=lambduh,with_capacity=False,explore=explore)
				try:
					wm = Worker_Manager(sas.get_init_kwa(), deployment)
					wm.start_workers()
					sas.set_worker_manager(wm)
					ret = sas.compare_different_solutions(deployment_size=dpsize,n_run=1, verbose=False)
				except:
					import traceback
					traceback.print_exc()
				finally:
					wm.stop_workers()
				our_adv = threshold_a(ret['adv_solns']['sparse'][0])
				metrics[explore_i]['obj'].append(ret['sparse_objective_vals']['sparse'][0])
				metrics[explore_i]['n_advs'].append(ret['n_advs']['sparse'][0])
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
		save_fig("compare_explores_objective_{}.pdf".format(lambduh))

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
		save_fig("compare_explores_nadv_obj_scatter_{}.pdf".format(lambduh))

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

def do_eval_improvement_over_budget_single_deployment():
	np.random.seed(31414)
	metrics = {}
	N_TO_SIM = 1
	explore='bimodality'
	# lambduhs = list(reversed(np.logspace(-2,.9,num=20))) #RFS
	lambduhs = list(reversed(np.logspace(-2,.3)))
	solution_types = ['sparse', 'anyopt', 'painter']
	
	wm = None
	
	metrics_fn = os.path.join(CACHE_DIR, 'improvement_over_budget_{}.pkl'.format(DPSIZE))
	if os.path.exists(metrics_fn):
		metrics = pickle.load(open(metrics_fn,'rb'))
	deployment = get_random_deployment(DPSIZE)

	# import pprint
	# pp = pprint.PrettyPrinter(indent=2)
	# pp.pprint(deployment)

	metric_keys = ['painter_benefit','cost','prefix_cost',
		'max_painter_benefit', 'sparse_benefit', 'max_sparse_benefit']

	try:
		for li,lambduh in enumerate(lambduhs):
			print("-----LAMBDUH = {} ({}th iter) -------".format(lambduh, li))
			metrics[lambduh] = metrics.get(lambduh, {m: {st:[] for st in solution_types} for m in metric_keys})
			for random_iter in range(N_TO_SIM):
				try:
					metrics[lambduh]['sparse_benefit']['sparse'][random_iter]
					print("Already have metrics for {} {}, continuing".format(lambduh, random_iter))
					continue
				except:
					pass
				sas = Sparse_Advertisement_Eval(deployment, verbose=True,
					lambduh=lambduh,with_capacity=False,explore=explore,n_prefixes=len(deployment['popps'])-1)
				if wm is None:
					wm = Worker_Manager(sas.get_init_kwa(), deployment)
					wm.start_workers()
				sas.set_worker_manager(wm)
				sas.update_deployment(deployment)
				ret = sas.compare_different_solutions(deployment_size=DPSIZE,n_run=1, verbose=True)
				for st in solution_types:
					metrics[lambduh]['painter_benefit'][st].append(-1*ret['painter_objective_vals'][st][0])
					metrics[lambduh]['cost'][st].append(ret['norm_penalties'][st][0])
					metrics[lambduh]['prefix_cost'][st].append(ret['prefix_cost'][st][0])
					metrics[lambduh]['max_painter_benefit'][st].append(ret['max_painter_benefits'][0])
					metrics[lambduh]['sparse_benefit'][st].append(ret['normalized_sparse_benefit'][st][0])
					metrics[lambduh]['max_sparse_benefit'][st].append(ret['max_sparse_benefits'][0])
				pickle.dump(metrics, open(metrics_fn,'wb'))
			# f,ax=plt.subplots(1,1)
			# for st in solution_types:
			# 	these_costs = [np.median(metrics[lambduh]['cost'][st]) for lambduh in sorted(metrics)]
			# 	these_benefits = [np.median(np.array(metrics[lambduh]['painter_benefit'][st]) * 100.0 \
			# 		/ np.array(metrics[lambduh]['max_painter_benefit'][st])) for lambduh in sorted(metrics)]
			# 	ax.scatter(these_costs, these_benefits,label=st + " PB",marker='x')
			# 	these_benefits = [np.median(np.array(metrics[lambduh]['sparse_benefit'][st]) * 100.0 \
			# 		/ np.array(metrics[lambduh]['max_sparse_benefit'][st])) for lambduh in sorted(metrics)]
			# 	ax.scatter(these_costs, these_benefits,label=st + " SB")

			# ax.legend()
			# ax.grid(True)
			# ax.set_xlabel("Cost")
			# ax.set_ylabel("Pct. Benefit")
			# save_fig("cost_vs_benefit.pdf")
	except:
		import traceback
		traceback.print_exc()
		exit(0)
	finally:
		if wm is not None:
			wm.stop_workers()
	for cost_type in ['cost', 'prefix_cost']:
		f,ax=plt.subplots(1,1)
		for st in solution_types:
			these_costs = [np.median(metrics[lambduh][cost_type][st]) for lambduh in lambduhs]
			these_benefits = [np.median(np.array(metrics[lambduh]['painter_benefit'][st]) * 100.0 \
				/ np.array(metrics[lambduh]['max_painter_benefit'][st])) for lambduh in lambduhs]
			print(st)
			print(these_benefits)
			print(these_costs)
			ax.scatter(these_costs, these_benefits,label=st + " PB")

			these_benefits = [np.median(np.array(metrics[lambduh]['sparse_benefit'][st]) * 100.0 \
				/ np.array(metrics[lambduh]['max_sparse_benefit'][st])) for lambduh in lambduhs]
			ax.scatter(these_costs, these_benefits,label=st + " SB")

		ax.legend()
		ax.grid(True)
		ax.set_xlabel(cost_type)
		ax.set_ylabel("Pct. Benefit")
		save_fig("{}_vs_benefit.pdf".format(cost_type))

def do_eval_improvement_over_budget_multi_deployment():
	np.random.seed(31414)
	dpsize = 'really_friggin_small'
	metrics = {}
	N_TO_SIM = 100
	explore='bimodality'
	lambduhs = list(reversed(np.logspace(-2,.3,num=10)))
	solution_types = ['sparse', 'anyopt', 'painter']
	
	wm = None
	
	pcntles_of_interest = [70,90,95]
	metrics_fn = os.path.join(CACHE_DIR, 'improvement_over_budget_multideployment_{}.pkl'.format(dpsize))
	metrics = {'comparisons':{bt: {st: {p:[] for p in pcntles_of_interest} for st in solution_types} for bt in\
		['sparse_benefit', 'painter_benefit']}}
	if os.path.exists(metrics_fn):
		metrics = pickle.load(open(metrics_fn,'rb'))
	N_TO_SIM = np.maximum(len(metrics) - 2, N_TO_SIM)
	metrics['comparisons'] = {bt: {st: {p:[None for _ in range(N_TO_SIM)] for p in pcntles_of_interest} for st in solution_types} for bt in\
		['sparse_benefit', 'painter_benefit']}

	metric_keys = ['painter_benefit','cost','max_painter_benefit', 'sparse_benefit', 'max_sparse_benefit']



	try:
		for random_iter in range(N_TO_SIM):
			print("-----Deployment number = {} -------".format(random_iter))
			deployment = get_random_deployment(dpsize)
			metrics[random_iter] = metrics.get(random_iter, {lambduh: {m: {st:None for st in solution_types} for \
				m in metric_keys} for lambduh in lambduhs})
			for lambduh in lambduhs:
				if metrics[random_iter][lambduh]['sparse_benefit']['sparse'] is not None:
					print("Already have metrics for {} {}, continuing".format(lambduh, random_iter))
					continue
				sas = Sparse_Advertisement_Eval(deployment, verbose=True,
					lambduh=lambduh,with_capacity=False,explore=explore,n_prefixes=len(deployment['popps'])-1)
				if wm is None:
					wm = Worker_Manager(sas.get_init_kwa(), deployment)
					wm.start_workers()
				sas.set_worker_manager(wm)
				sas.update_deployment(deployment)
				ret = sas.compare_different_solutions(deployment_size=dpsize,n_run=1, verbose=True)
				for st in solution_types:
					metrics[random_iter][lambduh]['painter_benefit'][st] = -1*ret['painter_objective_vals'][st][0]
					metrics[random_iter][lambduh]['cost'][st] = ret['norm_penalties'][st][0]
					metrics[random_iter][lambduh]['max_painter_benefit'][st] = ret['max_painter_benefits'][0]
					metrics[random_iter][lambduh]['sparse_benefit'][st] = ret['normalized_sparse_benefit'][st][0]
					metrics[random_iter][lambduh]['max_sparse_benefit'][st] = ret['max_sparse_benefits'][0]
				pickle.dump(metrics, open(metrics_fn,'wb'))
			for st in solution_types:
				for bt in ['sparse','painter']:
					these_costs = [metrics[random_iter][lambduh]['cost'][st] for lambduh in lambduhs]
					these_benefits = np.array([metrics[random_iter][lambduh]['{}_benefit'.format(bt)][st] * 100.0 \
						/ metrics[random_iter][lambduh]['max_{}_benefit'.format(bt)][st] for lambduh in lambduhs])
					
					for p in pcntles_of_interest:
						try:
							metrics['comparisons']['{}_benefit'.format(bt)][st][p][random_iter] = \
								these_costs[np.where(these_benefits >= p)[0][0]]
						except IndexError:
							pass
			pickle.dump(metrics, open(metrics_fn,'wb'))
	except:
		import traceback
		traceback.print_exc()
		exit(0)
	finally:
		if wm is not None:
			wm.stop_workers()
	f,ax=plt.subplots(1,1)
	## Do CDF of cost [method] / [sparse] for benefit p% 
	# do for both sparse and painter benefits

	for p in pcntles_of_interest:
		for mt,compare_method in zip(['<','.'],['painter', 'anyopt']):
			for bt in ['painter', 'sparse']:
				comps = []
				these_costs = metrics['comparisons']['{}_benefit'.format(bt)]
				for j in range(N_TO_SIM):
					if these_costs[compare_method][p][j] != None and these_costs['sparse'][p][j] != None:
						comps.append(these_costs[compare_method][p][j] / these_costs['sparse'][p][j])
				if len(comps) > 1:
					x,cdf_x = get_cdf_xy(comps,n_points=50)
					ax.plot(x,cdf_x,marker=mt,markersize=3,label="{} {} {}".format(p,compare_method,bt))
	ax.legend()
	ax.grid(True)
	ax.set_xlabel("Method / Sparse Cost at Each Benefit Pile")
	ax.set_ylabel("CDF of Deployments")
	save_fig("sparse_cost_savings_over_deployments_cdf.pdf")

def do_eval_whatifs():
	# for each deployment, solve problem
	# fail each link and pop, for each user calc latency range and pdf, get ground truth latency
	# look at some measure of the degree to which our model helps us get closer to ground truth latencies

	np.random.seed(31414)
	metrics = {}
	N_TO_SIM = 1

	lambduh = .1
	
	wm = None
	
	metrics_fn = os.path.join(CACHE_DIR, 'whatifs_{}.pkl'.format(DPSIZE))
	metrics = {'popp_failures': {i:{} for i in range(N_TO_SIM)}}
	if os.path.exists(metrics_fn):
		metrics = pickle.load(open(metrics_fn,'rb'))

	try:
		for random_iter in range(N_TO_SIM):
			try:
				metrics['popp_failures'][random_iter]
				if len(metrics['popp_failures'][random_iter]) > 0: 
					continue
			except KeyError:
				pass
			metrics['popp_failures'][random_iter] = {}
			print("-----Deployment number = {} -------".format(random_iter))
			deployment = get_random_deployment(DPSIZE)
			sas = Sparse_Advertisement_Eval(deployment, verbose=False,
				lambduh=lambduh,with_capacity=False,explore=DEFAULT_EXPLORE)
			if wm is None:
				wm = Worker_Manager(sas.get_init_kwa(), deployment)
				wm.start_workers()
			sas.set_worker_manager(wm)
			sas.update_deployment(deployment)
			ret = sas.compare_different_solutions(deployment_size=DPSIZE,n_run=1, verbose=False,
				 dont_update_deployment=True)

			adv = threshold_a(ret['adv_solns']['sparse'][0])

			_, ug_catchments = sas.calculate_user_choice(adv)
			for popp in sas.popps:
				these_ugs = [ug for ug in sas.ugs if \
					sas.popp_to_ind[popp] == ug_catchments[sas.ug_to_ind[ug]]]
				if len(these_ugs) == 0: 
					continue
				adv_cpy = np.copy(adv)
				adv_cpy[sas.popp_to_ind[popp]] = 0

				avg_benefit, (x,pdf) = sas.latency_benefit_fn(adv_cpy, retnow=True, ugs=these_ugs)
				actual_lb = sas.get_ground_truth_latency_benefit(adv_cpy, ugs=these_ugs)
				naive_range = sas.get_naive_range(adv_cpy, ugs=these_ugs)

				metrics['popp_failures'][random_iter][popp] = {
					'actual': actual_lb,
					'expected': avg_benefit,
					'range': naive_range
				}


			pickle.dump(metrics, open(metrics_fn,'wb'))
	except:
		import traceback
		traceback.print_exc()
		exit(0)
	finally:
		if wm is not None:
			wm.stop_workers()
	f,ax=plt.subplots(1,1)
	print(metrics)


	all_predicted = np.array([metrics['popp_failures'][ri][popp]['expected'] for ri in range(N_TO_SIM) for popp \
		in metrics['popp_failures'][ri]])
	all_actual = np.array([metrics['popp_failures'][ri][popp]['actual'] for ri in range(N_TO_SIM) for popp \
		in metrics['popp_failures'][ri]])
	all_avg = np.array([metrics['popp_failures'][ri][popp]['range']['avg'] for ri in range(N_TO_SIM) for popp \
		in metrics['popp_failures'][ri]])

	x1,cdf_predicted_diffs = get_cdf_xy(all_predicted - all_actual)
	x2,cdf_naive_diffs = get_cdf_xy(all_avg - all_actual)

	ax.plot(x1,cdf_predicted_diffs,label="Our Prediction")
	ax.plot(x2,cdf_naive_diffs,label="Average Prediction")

	ax.legend()
	ax.grid(True)
	ax.set_xlabel("Difference Between Actual and Predicted (ms)")
	ax.set_ylabel("CDF of Deployments")
	save_fig("whatifs.pdf")

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
	# do_eval_compare_explores()
	do_eval_whatifs()
