"""Sweep + plot: how does SCULPTOR compare to baselines as number of prefixes varies?

Mirrors `evaluate_over_deployment_sizes.py` but the swept axis is
`n_prefixes` instead of dpsize. Used for the paper plot showing that
SCULPTOR's advantage holds until n_prefixes exceeds the number of sites.
"""

# run-as-script bootstrap: this module lives in a package now,
# so put the repo root on sys.path before importing siblings.
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from helpers.constants import *
from helpers.helpers import *
from evaluations.eval_all_solution_types import evaluate_all_metrics
import numpy as np, os, pickle, glob
np.random.seed(31700)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from helpers.paper_plotting_functions import *

def dpsize_to_str(dpsize):
	"""'small'/'decent' pass through; a bare number means actual-N."""
	d = str(dpsize)
	return d if not d.isdigit() else "actual-{}".format(d)


def pull_results(dpsize, port=None, prefixes=None, nsim=1, worker_n=0,
	n_workers=1, cache_fn=None, max_iter=None, deployment_idx=3):
	"""Evaluate over a set of prefix budgets at one deployment size.

	Args (2026-08-21 -- all optional; omitting them reproduces the previous
	hardcoded behaviour, so existing invocations are unaffected):
	  dpsize          'small', 'decent', or a PoP count meaning actual-N.
	  prefixes        list of prefix budgets, e.g. [1,2,3,4].
	  nsim            sims per prefix budget.
	  max_iter        sets SCULPTOR_MAX_ITER.
	  deployment_idx  which cached deployment to reuse (default 3, as before;
	                  small caches usually only have index 0).
	"""
	dpsize_str = dpsize_to_str(dpsize)
	if cache_fn is None:
		cache_fn = os.path.join(CACHE_DIR,
			'evaluate_over_prefix_numbers_cache_fn_{}.pkl'.format(dpsize_str))
	workers_per_deployment = get_n_workers(dpsize_str) + 1

	default_prefixes = [30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
	only_recalc = None if prefixes else [45,50,55,60,65,70,75,80]
	metrics_by_prefix_number = {}
	if os.path.exists(cache_fn):
		metrics_by_prefix_number = pickle.load(open(cache_fn, 'rb'))

	src = os.path.join(CACHE_DIR,
		'popp_failure_latency_comparison_{}.pkl'.format(dpsize_str))
	src_metrics = pickle.load(open(src, 'rb'))
	deps = src_metrics['deployment']
	# the classic path hardcoded index 3; small/decent caches often hold only
	# index 0, so fall back rather than KeyError (Tom 2026-08-21).
	deployment = deps.get(deployment_idx, deps[sorted(deps)[0]])

	if port is None:
		import argparse
		parser = argparse.ArgumentParser()
		parser.add_argument("--port", default=DEFAULT_PORT, type=int)
		parser.add_argument("--worker_n", required=True)
		args = parser.parse_args()
		worker_n = int(args.worker_n)
		port = int(args.port)
	port = int(port)
	if max_iter is not None:
		os.environ['SCULPTOR_MAX_ITER'] = str(max_iter)

	prefixes_to_calc = (list(prefixes) if prefixes
						else default_prefixes[worker_n::n_workers])
	for prefix_num in prefixes_to_calc:
		if only_recalc is not None and prefix_num not in only_recalc:
			continue
		print("Worker {} evaluating over {} prefixes".format(worker_n, prefix_num))
		# inner per-prefix pickles namespace NEXT TO cache_fn (Tom
		# 2026-08-30: the global-CACHE_DIR path silently short-circuited
		# hermetic reruns and starved the depstore layer)
		_inner_dir = os.path.dirname(cache_fn) or CACHE_DIR
		os.makedirs(_inner_dir, exist_ok=True)
		inner_metrics_fn = os.path.join(_inner_dir,
			'{}_over_prefixes-{}.pkl'.format(dpsize_str, prefix_num))
		metrics = evaluate_all_metrics(dpsize_str, port,
			prefix_deployment=deployment, n_prefixes=prefix_num, nsim=nsim,
			use_performance_metrics_fn=inner_metrics_fn)
		metrics_by_prefix_number[prefix_num] = {}
		for k in metrics:
			if 'stats' in k:
				metrics_by_prefix_number[prefix_num][k] = metrics[k]
	pickle.dump(metrics_by_prefix_number, open(cache_fn, 'wb'))
	return cache_fn


def grab_what_we_have(dpsize):
	## just tabulate stats for the things we have done
	print("Grabbing what we have computed already for {} sites".format(dpsize))
	dpsize_str = "actual-{}".format(dpsize)
	cache_fn = os.path.join(CACHE_DIR, 'evaluate_over_prefix_numbers_cache_fn_{}.pkl'.format(dpsize_str))

	metrics_by_prefix_number = {}
	port = 40000 ## shouldn't ever be needed
	
	done_files = glob.glob(os.path.join(CACHE_DIR, '{}_over_prefixes*.pkl'.format(dpsize_str)))
	done_prefixes = sorted(list([int(re.search(r"over_prefixes-(.+)\.pkl", done_fn).group(1)) for done_fn in done_files]))

	for prefix_num in done_prefixes:
		print("Loading results for {} prefixes".format(prefix_num))
		inner_metrics_fn = os.path.join(CACHE_DIR, '{}_over_prefixes-{}.pkl'.format(dpsize_str,prefix_num))
		metrics = evaluate_all_metrics(dpsize_str, port, nsim=1, use_performance_metrics_fn=inner_metrics_fn)
		metrics_by_prefix_number[prefix_num] = {}
		for k in metrics:
			if 'stats' in k:
				metrics_by_prefix_number[prefix_num][k] = metrics[k]
	pickle.dump(metrics_by_prefix_number, open(cache_fn, 'wb'))


def _cli():
	import argparse
	ap = argparse.ArgumentParser(description=pull_results.__doc__)
	ap.add_argument('--port', type=int, default=DEFAULT_PORT,
					help='vestigial under Ray; nothing binds it')
	ap.add_argument('--dpsize', default='32',
					help="'small', 'decent', or a PoP count (actual-N)")
	ap.add_argument('--prefixes', default=None,
					help='comma-separated prefix budgets, e.g. 1,2,3,4')
	ap.add_argument('--nsim', type=int, default=1)
	ap.add_argument('--max-iter', type=int, default=None)
	ap.add_argument('--cache-fn', default=None)
	ap.add_argument('--plot', action='store_true')
	ap.add_argument('--figures-subdir', default=None,
					help="namespace this run's figures: figures/<subdir>/... (e.g. --figures-subdir real_sweep_2026_08). Sets SCULPTOR_FIG_SUBDIR.")
	a = ap.parse_args()
	if a.figures_subdir:
		os.environ['SCULPTOR_FIG_SUBDIR'] = a.figures_subdir
	prefixes = [int(x) for x in a.prefixes.split(',')] if a.prefixes else None
	cache_fn = pull_results(a.dpsize, port=a.port, prefixes=prefixes,
							nsim=a.nsim, cache_fn=a.cache_fn,
							max_iter=a.max_iter)
	if a.plot:
		from evaluations.evaluate_over_deployment_sizes import make_paper_plots
		make_paper_plots(cache_fn, xlab="Prefix Budget",
						 evaluate_over="prefix_budget")
	return cache_fn


if __name__ == '__main__':
	import sys as _sys
	if len(_sys.argv) > 1:
		_cli(); raise SystemExit(0)
	dpsize = "32"
	grab_what_we_have(dpsize)
	from evaluations.evaluate_over_deployment_sizes import make_paper_plots
	dpsize_str = "actual-{}".format(dpsize)
	cache_fn = os.path.join(CACHE_DIR,
		'evaluate_over_prefix_numbers_cache_fn_{}.pkl'.format(dpsize_str))
	make_paper_plots(cache_fn, xlab="Prefix Budget", evaluate_over="prefix_budget")
