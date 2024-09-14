import numpy as np, sys, glob, re, copy

from deployment_setup import get_random_deployment
from sparse_advertisements_v3 import Sparse_Advertisement_Solver
from worker_comms import Worker_Manager
from helpers import *
from constants import *
from paper_plotting_functions import *


def compare_speedup_heuristic():
	"""
		Compare wall clock time and goodness of solution between using linear programming solver 
		with monte-carlo and using our speedup heuristic.
	"""
	### TODO -- could also vary the number of monte carlo, the deployment size, etc..

	np.random.seed(31410)
	try:
		dpsize = sys.argv[1]
		lambduh = .00001 ## unused more or less
		gamma = 2.0
		obj = 'avg_latency'
		using_generic_objectives = ['not_using', 'using']
		# using_generic_objectives = ['using', 'not_using']

		n_random_sim = 10
		performance_metrics_fn = os.path.join(CACHE_DIR, "assessing_speedup_heuristic_metrics_{}.pkl".format(dpsize))
		if os.path.exists(performance_metrics_fn):
			metrics = pickle.load(open(performance_metrics_fn, 'rb'))
		else:
			metrics = {tf:{i:{'done':False} for i in range(n_random_sim)} for tf in using_generic_objectives}
		wm = None

		if False:
			port = int(sys.argv[2])
			for random_iter in range(n_random_sim):
				try:
					this_iter_deployment = metrics[random_iter]['deployment']
				except KeyError:
					this_iter_deployment = get_random_deployment(dpsize)
				this_iter_deployment['port'] = port
				for using_generic_objective in using_generic_objectives:
					print("Random deployment for objective {}, number {}/{}".format(using_generic_objective,random_iter+1,n_random_sim))
					# try:
					# 	if metrics[using_generic_objective][random_iter]['done']: continue
					# except KeyError:
					# 	metrics[using_generic_objective][random_iter] = {'done': False}
					deployment = copy.deepcopy(this_iter_deployment)
					metrics[random_iter] = {'deployment': deployment}
					n_prefixes = deployment_to_prefixes(deployment)

					if using_generic_objective == 'using':
						sas = Sparse_Advertisement_Solver(deployment, 
							lambduh=lambduh,verbose=True,with_capacity=True,n_prefixes=n_prefixes,
							using_resilience_benefit=True, gamma=gamma, generic_objective=obj)
					else:
						sas = Sparse_Advertisement_Solver(deployment, 
							lambduh=lambduh,verbose=True,with_capacity=True,n_prefixes=n_prefixes,
							using_resilience_benefit=True, gamma=gamma)
					if wm is None:
						wm = Worker_Manager(sas.get_init_kwa(), deployment)
						wm.start_workers()
					sas.set_worker_manager(wm)
					sas.update_deployment(sas.output_deployment())
					ts = time.time()
					sas.solve()
					te = time.time()
					metrics[using_generic_objective][random_iter]['settings'] = sas.get_init_kwa()
					metrics[using_generic_objective][random_iter]['adv'] = sas.get_last_advertisement()
					metrics[using_generic_objective][random_iter]['t_convergence'] = ts - te
					metrics[using_generic_objective][random_iter]['final_objective'] = sas.metrics['actual_nonconvex_objective'][-1]
					metrics[using_generic_objective][random_iter]['final_latency_objective'] = sas.metrics['latency_benefit'][-1]
					metrics[using_generic_objective][random_iter]['optimal_objective'] = sas.optimal_expensive_solution
					metrics[using_generic_objective][random_iter]['save_run_dir'] = sas.save_run_dir

					metrics[using_generic_objective][random_iter]['done'] = True
					pickle.dump(metrics, open(performance_metrics_fn, 'wb'))


		# ugo -> itr -> [(opp obj - achieved obj, wall time)]
		font = {'size'   : 14}
		matplotlib.rc('font', **font)
		f,ax = plt.subplots(2)
		f.set_size_inches(4,2)
		high_level_metrics = {}
		skip_every=3
		labels = {
			'using': 'No Heuristic', 
			'not_using': 'Speedup Heuristic'
		}
		max_itr = 0
		for ugoi, ugo in enumerate(using_generic_objectives):
			high_level_metrics[ugo] = {}
			for ri in metrics[ugo]:
				if ri != 0: continue
				if not metrics[ugo][ri].get('done',False): continue
				srd = metrics[ugo][ri]['save_run_dir']

				# opp_obj = metrics[ugo][ri]['optimal_objective']['overall']
				opp_obj = metrics[ugo][ri]['optimal_objective']['latency']
				tstart = np.inf
				for fn in glob.glob(os.path.join(srd, 'small-stats-*')):
					tstart = min(tstart, os.path.getmtime(fn))
				for fn in glob.glob(os.path.join(srd, 'small-stats-*')):
					these_small_stats = pickle.load(open(fn, 'rb'))['optimization_vars']
					itr = these_small_stats['iter']
					# dlta = (these_small_stats['current_objective'] - opp_obj) / opp_obj
					dlta = -1 * (these_small_stats['current_latency_benefit'] - opp_obj) 

					try:
						wall_time = (these_small_stats['calc_times'][0][0]+these_small_stats['calc_times'][1][0]+\
							these_small_stats['calc_times'][2][0]) / 3600
					except IndexError:
						wall_time = 0
					try:	
						high_level_metrics[ugo][itr].append((dlta, wall_time))
					except KeyError:
						high_level_metrics[ugo][itr] = [(dlta, wall_time)]

			itrs = sorted(list(high_level_metrics[ugo]))
			max_itr = max(max(itrs), max_itr)
			dltas = list([np.mean([el[0] for el in high_level_metrics[ugo][itr]]) for itr in itrs])
			wcts = np.array(list([np.mean([el[1] for el in high_level_metrics[ugo][itr]]) for itr in itrs]))
			wcts = np.cumsum(wcts)
			ax[0].plot(itrs[::skip_every], dltas[::skip_every], label=labels[ugo], color=some_colors[ugoi], marker=markers[ugoi])
			ax[1].plot(itrs[::skip_every], wcts[::skip_every], label=labels[ugo], color=some_colors[ugoi], marker=markers[ugoi])
		ax[1].set_xlabel("Training Iteration")
		ax[0].set_ylabel("Objective\n (ms)", fontsize=14)
		ax[1].set_ylabel("Time\n (h)", fontsize=15)

		ax[0].set_xlim([0,max_itr+10])
		ax[1].set_xlim([0,max_itr+10])
		# ax[0].set_xlim([0,40])
		ax[0].set_xticks([])
		# ax[1].set_xlim([0,40])
		ax[1].set_yticks([0,20.0])
		ax[1].set_ylim([0,20.0])
		
		ax[0].set_ylim([0,7])
		
		ax[0].legend(loc='upper right', fontsize=11)

		save_fig('objective_heuristic_comparison_{}.pdf'.format(dpsize))


	except:
		import traceback
		traceback.print_exc()
	finally:
		try:
			wm.stop_workers()
		except:
			pass


def train_models_for_obj(obj):
	np.random.seed(31415)
	try:
		dpsize = sys.argv[1]
		port = int(sys.argv[2])
		lambduh = .00001 ## unused more or less
		gamma = 2.0

		n_random_sim = 1
		performance_metrics_fn = os.path.join(CACHE_DIR, "testing_objective_metrics_{}-{}.pkl".format(dpsize, obj))
		if os.path.exists(performance_metrics_fn):
			metrics = pickle.load(open(performance_metrics_fn, 'rb'))
		else:
			metrics = {i:{'done':False} for i in range(n_random_sim)}
		wm = None
		for random_iter in range(n_random_sim):
			try:
				if metrics[random_iter]['done']: continue
			except KeyError:
				metrics[random_iter] = {'done': False}
			print("Random deployment for objective {}, number {}/{}".format(obj,random_iter,n_random_sim))
			deployment = get_random_deployment(dpsize)
			deployment['port'] = port
			metrics[random_iter]['deployment'] = deployment
			n_prefixes = deployment_to_prefixes(deployment)

			sas = Sparse_Advertisement_Solver(deployment, 
				lambduh=lambduh,verbose=True,with_capacity=True,n_prefixes=n_prefixes,
				using_resilience_benefit=True, gamma=gamma, generic_objective=obj)
			if wm is None:
				wm = Worker_Manager(sas.get_init_kwa(), deployment)
				wm.start_workers()
			sas.set_worker_manager(wm)
			sas.update_deployment(sas.output_deployment())
			ts = time.time()
			sas.solve()
			te = time.time()
			metrics[random_iter]['settings'] = sas.get_init_kwa()
			metrics[random_iter]['adv'] = sas.get_last_advertisement()
			metrics[random_iter]['t_convergence'] = ts - te
			metrics[random_iter]['final_objective'] = sas.metrics['actual_nonconvex_objective'][-1]
			metrics[random_iter]['final_latency_objective'] = sas.metrics['latency_benefit'][-1]
			metrics[random_iter]['optimal_objective'] = sas.optimal_expensive_solution
			metrics[random_iter]['save_run_dir'] = sas.save_run_dir

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

def gen_plots_over_objs():
	for metrics_fn in glob.glob(os.path.join(CACHE_DIR, 'testing_objective_metrics_*.pkl')):
		which_metrics = re.search('metrics\_.+\-(.+)\.pkl', metrics_fn).group(1)
		metrics = pickle.load(open(metrics_fn, 'rb'))
		n_sims = len(metrics)


		t_convergences = list([metrics[i]['t_convergence'] for i in range(n_sims)])
		print("Mean convergence time for {} : {}".format(which_metrics, np.mean(t_convergences)))

		objective_efficiencies = list([np.abs(metrics[i]['final_objective'] - metrics[i]['optimal_objective']['overall']) / metrics[i]['optimal_objective']['overall'] \
			for i in range(n_sims)])
		print("Mean objective efficiency time for {} : {}".format(which_metrics, np.mean(objective_efficiencies)))


if __name__ == "__main__":
	compare_speedup_heuristic()


	# train_models_for_obj('avg_latency')
	# train_models_for_obj('squaring')
	# train_models_for_obj('square_rooting')
	# gen_plots_over_objs()
