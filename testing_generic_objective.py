import numpy as np, sys, glob, re

from deployment_setup import get_random_deployment
from sparse_advertisements_v3 import Sparse_Advertisement_Solver
from worker_comms import Worker_Manager
from helpers import *
from constants import *


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
	# train_models_for_obj('avg_latency')
	# train_models_for_obj('squaring')
	train_models_for_obj('square_rooting')

	gen_plots_over_objs()
