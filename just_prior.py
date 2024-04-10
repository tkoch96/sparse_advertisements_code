import pickle
from constants import *
from deployment_setup import *
from eval_latency_failure import evaluate_all_metrics

def eval_just_prior_metrics():
	# for each deployment, get different advertisement strategies
	# look at latency under popp and link failures compared to optimal
	# sparse should be simultaneously better in both normal and failure scenarios

	# np.random.seed(31413)
	metrics = {}
	soln_types = ['painter', 'anycast', 'one_per_pop', 'one_per_peering']
	
	deployment = pickle.load(open('runs/1711510437-actual-large-sparse/state-0.pkl','rb'))['deployment']
	performance_metrics_fn = os.path.join(CACHE_DIR, 'just_prior_comparison_{}.pkl'.format(DPSIZE))
	save_fig_fn = "just_prior_failure_comparison_{}.pdf".format(DPSIZE)
	metrics = evaluate_all_metrics(soln_types=soln_types, use_performance_metrics_fn=performance_metrics_fn, save_fig_fn=save_fig_fn,
		prefix_deployment=deployment)


if __name__ == "__main__":
	eval_just_prior_metrics()
