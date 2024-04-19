import pickle
from wrapper_eval import *
from constants import *
from deployment_setup import *
from eval_latency_failure import evaluate_all_metrics

def eval_specific_deployment(hotstart_dir):
	# deployment = pickle.load(open(os.path.join(RUN_DIR, hotstart_dir, 'state-0.pkl'),'rb'))['deployment']

	performance_metrics_fn = os.path.join(CACHE_DIR, 'pseudo_deployment_comparison_{}.pkl'.format(DPSIZE))
	save_fig_fn = "pseudo_deployment_failure_comparison_{}.pdf".format(DPSIZE)
	# metrics = evaluate_all_metrics(use_performance_metrics_fn=performance_metrics_fn, save_fig_fn=save_fig_fn,
	# 	prefix_deployment=deployment)
	metrics = evaluate_all_metrics(use_performance_metrics_fn=performance_metrics_fn, save_fig_fn=save_fig_fn)

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--hotstart_dir", required=True)
	args = parser.parse_args()

	assert N_TO_SIM == 1
	eval_specific_deployment(args.hotstart_dir)
