from constants import *
from helpers import *
from eval_latency_failure import evaluate_all_metrics
import numpy as np, os, pickle, glob
np.random.seed(31700)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from paper_plotting_functions import *

def pull_results(dpsize):

	n_workers = 1
	dpsize_str = "actual-{}".format(dpsize)
	cache_fn = os.path.join(CACHE_DIR, 'evaluate_over_prefix_numbers_cache_fn_{}.pkl'.format(dpsize_str))
	workers_per_deployment = get_n_workers(dpsize_str) + 1


	only_recalc = [45,50,55,60,65,70,75,80] ## recalc these prefix numbers
	metrics_by_prefix_number = {}
	if os.path.exists(cache_fn):
		metrics_by_prefix_number = pickle.load(open(cache_fn, 'rb'))

	metrics = pickle.load(open(os.path.join(CACHE_DIR, 'popp_failure_latency_comparison_actual-{}.pkl'.format(dpsize)) ,'rb'))
	deployment = metrics['deployment'][3]

	if not os.path.exists(cache_fn) or only_recalc is not None:
		import argparse
		parser = argparse.ArgumentParser()
		parser.add_argument("--port", required=True)
		parser.add_argument("--worker_n", required=True)
		args = parser.parse_args()
		worker_n = int(args.worker_n)
		port = int(args.port) + workers_per_deployment * worker_n
		prefixes_to_calc = [30,35,40,45,50,55,60,65,70,75,80,85,90,95,100][worker_n::n_workers]
		for prefix_num in prefixes_to_calc:
			if only_recalc is not None:
				if prefix_num not in only_recalc: continue
			print("Worker {} evaluating over {} prefixes".format(worker_n, prefix_num))
			inner_metrics_fn = os.path.join(CACHE_DIR, '{}_over_prefixes-{}.pkl'.format(dpsize_str,prefix_num))
			metrics = evaluate_all_metrics(dpsize_str, int(args.port), prefix_deployment=deployment, n_prefixes=prefix_num, nsim=1, use_performance_metrics_fn=inner_metrics_fn)
			metrics_by_prefix_number[prefix_num] = {}
			for k in metrics:
				if 'stats' in k:
					metrics_by_prefix_number[prefix_num][k] = metrics[k]
		pickle.dump(metrics_by_prefix_number, open(cache_fn, 'wb'))

def grab_what_we_have(dpsize):
	## just tabulate stats for the things we have done
	print("Grabbing what we have computed already for {} sites".format(dpsize))
	dpsize_str = "actual-{}".format(dpsize)
	cache_fn = os.path.join(CACHE_DIR, 'evaluate_over_prefix_numbers_cache_fn_{}.pkl'.format(dpsize_str))

	metrics_by_prefix_number = {}
	port = 40000 ## shouldn't ever be needed
	
	done_files = glob.glob(os.path.join(CACHE_DIR, '{}_over_prefixes*.pkl'.format(dpsize_str)))
	done_prefixes = sorted(list([int(re.search("over\_prefixes\-(.+)\.pkl", done_fn).group(1)) for done_fn in done_files]))

	for prefix_num in done_prefixes:
		print("Loading results for {} prefixes".format(prefix_num))
		inner_metrics_fn = os.path.join(CACHE_DIR, '{}_over_prefixes-{}.pkl'.format(dpsize_str,prefix_num))
		metrics = evaluate_all_metrics(dpsize_str, port, nsim=1, use_performance_metrics_fn=inner_metrics_fn)
		metrics_by_prefix_number[prefix_num] = {}
		for k in metrics:
			if 'stats' in k:
				metrics_by_prefix_number[prefix_num][k] = metrics[k]
	pickle.dump(metrics_by_prefix_number, open(cache_fn, 'wb'))


if __name__ == '__main__':
	dpsize = "32"
	# pull_results(dpsize)
	grab_what_we_have(dpsize)
	from evaluate_over_deployment_sizes import make_paper_plots
	dpsize_str = "actual-{}".format(dpsize)
	cache_fn = os.path.join(CACHE_DIR, 'evaluate_over_prefix_numbers_cache_fn_{}.pkl'.format(dpsize_str))
	make_paper_plots(cache_fn, xlab="Prefix Budget", evaluate_over="prefix_budget")
