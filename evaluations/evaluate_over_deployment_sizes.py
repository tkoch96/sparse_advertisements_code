"""Sweep + plot: how does SCULPTOR compare to baselines as deployment size varies?

For each dpsize in a default list (or env-var override), calls
`evaluate_all_metrics` to train sparse + all baseline strategies, then
aggregates per-strategy results across dpsizes into the paper plots
(`figures/paper/average_latency_over_deployment_size_*.pdf` and
percent-within-X-ms variants).

`pull_results_new()` is the per-dpsize execution loop;
`make_paper_plots()` is the plotting entry-point.

For cluster-friendly invocation with env-var config, see
`experiments/deployment_sizes_full_timing_investigation/run_deployment_sweep.py` (newer, recommended).
"""

# run-as-script bootstrap: this module lives in a package now,
# so put the repo root on sys.path before importing siblings.
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from helpers.constants import *
from evaluations.eval_all_solution_types import evaluate_all_metrics
import numpy as np, os, pickle, json, time, traceback
np.random.seed(31705)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

from helpers.paper_plotting_functions import *


def _log_mem(tag, **extra):
	"""Driver memory + wall-clock marker, in the `[mem]` format that
	cluster/plot_phase_timings.py and cluster/cluster_dashboard.py already
	parse. Ported from the run_deployment_sweep fork (2026-08-21) so the
	timing story lives in the evaluation Tom actually runs.

	No-ops off Linux (the Mac has no /proc), which is why it is safe to
	call unconditionally.
	"""
	rss_kb = sys_avail_kb = -1
	try:
		with open('/proc/self/status') as f:
			for line in f:
				if line.startswith('VmRSS:'):
					rss_kb = int(line.split()[1]); break
		with open('/proc/meminfo') as f:
			for line in f:
				if line.startswith('MemAvailable:'):
					sys_avail_kb = int(line.split()[1]); break
	except (FileNotFoundError, PermissionError):
		return
	extras = ' '.join('{}={}'.format(k, v) for k, v in extra.items())
	print('[mem] tag={} rss_mb={} sys_avail_mb={} pid={} t={:.2f} {}'.format(
		tag, rss_kb // 1024, sys_avail_kb // 1024, os.getpid(), time.time(),
		extras), flush=True)


def _disk_free_gb(path='.'):
	try:
		st = os.statvfs(path)
		return st.f_bavail * st.f_frsize / (1024.0 ** 3)
	except OSError:
		return None


def _progress(state):
	"""Write the sweep's machine-readable progress, if asked to.

	SCULPTOR_SWEEP_PROGRESS_JSON is set by cluster/expctl.py; the dashboard
	tab reads the harvested copy. Best-effort by construction -- a sweep
	must never die because its progress file could not be written.
	"""
	path = os.environ.get('SCULPTOR_SWEEP_PROGRESS_JSON')
	if not path:
		return
	try:
		os.makedirs(os.path.dirname(path), exist_ok=True)
		tmp = path + '.tmp'
		with open(tmp, 'w') as fh:
			json.dump(state, fh, indent=1, default=str)
		os.replace(tmp, path)
	except OSError:
		pass

def pull_results_new(cache_fn, port=None, dpsizes=None, n_sim_by_dpsize=None,
	only_recalc=None, max_iter=None):
	"""Sweep deployment sizes and cache the per-size stats.

	Args (2026-08-21 -- all optional; omitting them reproduces the previous
	hardcoded behaviour exactly, so existing invocations are unaffected):
	  port              LP port. Falls back to --port on the command line.
	  dpsizes           list of PoP counts, e.g. [3,4,5,6].
	  n_sim_by_dpsize   list of nsim parallel to dpsizes, or a single int
	                    applied to all of them.
	  only_recalc       subset of dpsizes to actually recompute.
	  max_iter          sets SCULPTOR_MAX_ITER for the sweep.

	These exist so integration tests (and quick laptop runs) can drive the
	real sweep at small sizes. experiments/deployment_sizes_full_timing_-
	investigation/run_deployment_sweep.py reimplemented this loop to get the
	same three knobs via env vars; prefer this.
	"""
	if only_recalc is None:
		only_recalc = [3,5,10,15,20,25,32] ## recalc these deployment sizes
	# --cache-fn may name a directory that does not exist yet (cluster runs
	# namespace it per run id). Creating it here rather than at the caller
	# means the sweep cannot burn a size's worth of compute and then die on
	# the write -- which is exactly what happened on 2026-08-21, eight
	# minutes into the first cluster run.
	_cache_dir = os.path.dirname(cache_fn)
	if _cache_dir:
		os.makedirs(_cache_dir, exist_ok=True)
	metrics_by_dpsize = {}
	if os.path.exists(cache_fn):
		metrics_by_dpsize = pickle.load(open(cache_fn, 'rb'))
	if not os.path.exists(cache_fn) or only_recalc is not None:
		if port is None:
			import argparse
			parser = argparse.ArgumentParser()
			parser.add_argument("--port", default=DEFAULT_PORT, type=int)
			args = parser.parse_args()
			port = int(args.port)
		port = int(port)
		if max_iter is not None:
			os.environ['SCULPTOR_MAX_ITER'] = str(max_iter)
		if dpsizes is None:
			dpsizes = [3,5,10,15,20,25,len(POP_TO_LOC['vultr'])]
			if n_sim_by_dpsize is None:
				n_sim_by_dpsize = [15,20,10,16,15,15,12] # running now
		if n_sim_by_dpsize is None:
			n_sim_by_dpsize = 1
		if isinstance(n_sim_by_dpsize, int):
			n_sim_by_dpsize = [n_sim_by_dpsize] * len(dpsizes)
		if only_recalc is True:
			only_recalc = list(dpsizes)

		todo = [(dp, ns) for dp, ns in zip(dpsizes, n_sim_by_dpsize)
				if only_recalc is None or dp in only_recalc]
		overall_start = time.time()
		per_size = {}
		_log_mem('sweep_start', n_sizes=len(todo))
		print("[sweep] START sizes={} nsim={} cache_fn={} disk_free_gb={}".format(
			[d for d, _ in todo], [n for _, n in todo], cache_fn,
			'{:.1f}'.format(_disk_free_gb() or -1)), flush=True)
		_progress({'phase': 'start', 'started': overall_start,
				   'sizes': [d for d, _ in todo],
				   'nsim': {str(d): n for d, n in todo},
				   'done': {}, 'failed': {}, 'cache_fn': cache_fn})

		for dpsize, nsim in todo:
			dp_start = time.time()
			_log_mem('dpsize_start', dpsize=dpsize, nsim=nsim)
			print("Evaluating over deployment size {} Sites".format(dpsize))
			# Non-numeric tokens (e.g. 'small') are named deployments and
			# pass through as-is; ints keep the testing_feature mapping.
			dpsize_str = (dpsize if isinstance(dpsize, str)
						  else "testing_feature-actual-{}".format(dpsize))
			# Exact wording matters: cluster/plot_phase_timings.py's SWEEP_RE
			# keys its per-phase timing off this line.
			print("[sweep] === dpsize={} dpsize_str={} nsim={} ===".format(
				dpsize, dpsize_str, nsim), flush=True)
			print("[sweep] disk_free_gb={:.1f}".format(
				_disk_free_gb() or -1), flush=True)
			_progress({'phase': 'running', 'started': overall_start,
					   'current': dpsize, 'current_started': dp_start,
					   'sizes': [d for d, _ in todo],
					   'nsim': {str(d): n for d, n in todo},
					   'done': per_size, 'failed': {}, 'cache_fn': cache_fn})
			save_run_dir = None
			# Was this size's result already on disk? If so evaluate_all_metrics
			# will load it and return in about a second WITHOUT training, and
			# its "wall time" is a cache-read, not a measurement. Recording
			# that distinction is the difference between a timing table and a
			# fiction -- launch with expctl --nocache to force real work.
			try:
				from evaluations.wrapper_eval import global_performance_metrics_fn
				_was_cached = os.path.exists(global_performance_metrics_fn(dpsize_str))
			except Exception:
				_was_cached = None
			# One size failing must not cost the sizes already computed:
			# before 2026-08-21 the pickle was written only after the whole
			# loop, so a crash at 25 threw away 3/5/10/15/20 as well.
			try:
				metrics = evaluate_all_metrics(dpsize_str, port,
					save_run_dir=save_run_dir, nsim=nsim)
			except KeyboardInterrupt:
				print("[sweep] interrupted during dpsize={}".format(dpsize), flush=True)
				raise
			except Exception:
				traceback.print_exc()
				dp_wall = time.time() - dp_start
				print("[sweep] dpsize={} FAILED after {:.1f}s; continuing "
					  "to the next size".format(dpsize, dp_wall), flush=True)
				_log_mem('dpsize_failed', dpsize=dpsize, wall_s=int(dp_wall))
				per_size[str(dpsize)] = {'wall_s': dp_wall, 'ok': False}
				_progress({'phase': 'running', 'started': overall_start,
						   'sizes': [d for d, _ in todo],
						   'nsim': {str(d): n for d, n in todo},
						   'done': per_size, 'cache_fn': cache_fn})
				continue
			metrics_by_dpsize[dpsize] = {}
			for k in metrics:
				if 'stats' in k:
					metrics_by_dpsize[dpsize][k] = metrics[k]
			pickle.dump(metrics_by_dpsize, open(cache_fn, 'wb'))
			dp_wall = time.time() - dp_start
			per_size[str(dpsize)] = {'wall_s': dp_wall, 'ok': True,
									 'nsim': nsim, 'cached': _was_cached,
									 'sec_per_sim': dp_wall / max(nsim, 1)}
			print("[sweep] dpsize={} done in {:.1f}s ({:.1f} min, {:.1f}s/sim); "
				  "cumulative {:.1f}s; disk_free_gb={}{}".format(
					dpsize, dp_wall, dp_wall / 60.0, dp_wall / max(nsim, 1),
					time.time() - overall_start,
					'{:.1f}'.format(_disk_free_gb() or -1),
					"  [CACHE HIT -- this wall time is NOT a timing "
					"measurement]" if _was_cached else ""), flush=True)
			_log_mem('dpsize_done', dpsize=dpsize, wall_s=int(dp_wall))
			_progress({'phase': 'running', 'started': overall_start,
					   'sizes': [d for d, _ in todo],
					   'nsim': {str(d): n for d, n in todo},
					   'done': per_size, 'cache_fn': cache_fn})

		pickle.dump(metrics_by_dpsize, open(cache_fn, 'wb'))
		overall = time.time() - overall_start
		_log_mem('sweep_done', wall_s=int(overall))
		n_ok = sum(1 for v in per_size.values() if v.get('ok'))
		# The completion banner is the contract the harvest/monitor tools
		# check for -- rc==0 alone has lied before (2026-08-20: a cell
		# exited 0 in six seconds after a silently-failed hot-start).
		print("\n[sweep] ALL DONE in {:.1f}s ({:.1f} min). {}/{} sizes ok, "
			  "wrote {} dpsizes to {}".format(
				overall, overall / 60.0, n_ok, len(todo),
				len(metrics_by_dpsize), cache_fn), flush=True)
		_progress({'phase': 'done', 'started': overall_start,
				   'finished': time.time(), 'wall_s': overall,
				   'sizes': [d for d, _ in todo],
				   'nsim': {str(d): n for d, n in todo},
				   'done': per_size, 'n_ok': n_ok, 'n_total': len(todo),
				   'cache_fn': cache_fn})

def pull_results_old(cache_fn):
	only_recalc = [3,5,10,15,20,25,32] ## recalc these deployment sizes
	metrics_by_dpsize = {}
	if os.path.exists(cache_fn):
		metrics_by_dpsize = pickle.load(open(cache_fn, 'rb'))
	if not os.path.exists(cache_fn) or only_recalc is not None:
		import argparse
		parser = argparse.ArgumentParser()
		parser.add_argument("--port", required=True)
		args = parser.parse_args()
		dpsizes = [3,5,10,15,20,25,len(POP_TO_LOC['vultr'])]
		n_sim_by_dpsize = [15,20,10,16,9,9,6] # done

		for dpsize, nsim in zip(dpsizes, n_sim_by_dpsize):
			if only_recalc is not None:
				if dpsize not in only_recalc: continue
			print("Evaluating over deployment size {} Sites".format(dpsize))
			dpsize_str = "actual-{}".format(dpsize)
			# if dpsize == 32:
			# 	save_run_dir = [None, None, None, None, None, '1713700098-actual-32-sparse']
			# else:
			# 	save_run_dir = None
			save_run_dir = None
			metrics = evaluate_all_metrics(dpsize_str, port, save_run_dir=save_run_dir, nsim=nsim)
			metrics_by_dpsize[dpsize] = {}
			for k in metrics:
				if 'stats' in k:
					metrics_by_dpsize[dpsize][k] = metrics[k]
		pickle.dump(metrics_by_dpsize, open(cache_fn, 'wb'))

def make_paper_plots(cache_fn, **kwargs):

	print_metrics = {}

	metrics_by_dpsize = pickle.load(open(cache_fn, 'rb'))
	dpsizes = sorted(list(metrics_by_dpsize))
	solutions = sorted(list(metrics_by_dpsize[dpsizes[0]]['stats_best_latencies']))

	solutions = ['anycast', 'anyopt', 'one_per_pop', 'painter', 'sparse', 'one_per_peering']

	# Robustness: the cross-size plots/prints assume every solution has data at
	# every dpsize (arrays are subtracted elementwise). Some historical runs are
	# incomplete -- e.g. dp10/dp15 from the dep_sweep run lack 'sparse' in
	# stats_best_latencies. Drop such dpsizes so arrays stay aligned, and report
	# the gap rather than crashing.
	_complete = [dp for dp in dpsizes
		if all(sol in metrics_by_dpsize[dp].get('stats_best_latencies', {})
			   for sol in solutions)]
	_dropped = [dp for dp in dpsizes if dp not in _complete]
	if _dropped:
		print('[plots] WARNING: dropping dpsizes with incomplete solution data '
			  '(missing >=1 solution in stats_best_latencies): {}'.format(_dropped))
	dpsizes = _complete

	xlab = kwargs.get('xlab', "Deployment Size (Num Sites)")
	evaluate_over = kwargs.get('evaluate_over', 'deployment_size')

	f,ax = get_figure()
	metric_by_solution = {}
	for solution in solutions:
		avg_latency_diff_normal = []
		for dpsize in dpsizes:
			print(dpsize)
			print(metrics_by_dpsize[dpsize]['stats_best_latencies'].keys())
			avg_latency_diff_normal.append(-1*metrics_by_dpsize[dpsize]['stats_best_latencies'][solution])
		ax.plot(dpsizes,avg_latency_diff_normal, label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
		metric_by_solution[solution] = np.array(avg_latency_diff_normal)
	
	print("Painter - Sparse Normal: {}".format(metric_by_solution['painter'] - metric_by_solution['sparse']))
	print("Sparse - OPP Normal: {}".format(metric_by_solution['sparse'] - metric_by_solution['one_per_peering']))
	for solution in solutions:
		print("{} average over all deployments: {}".format(solution,np.average(metric_by_solution[solution])))
	print_metrics['normal_latency'] = metric_by_solution['sparse']
	print('\n')

	ax.set_xlabel(xlab)
	ax.set_ylabel("Avg Suboptimality\nNormal Operation (ms)")
	ax.legend(fontsize=12)
	save_figure('average_latency_over_{}_normal.pdf'.format(evaluate_over))

	for lp_tp, tp, tp_k in zip(['mlu','mlu'], ['Ingress', 'Site'],
		['stats_popp_failures_latency_optimal_specific', 'stats_pop_failures_latency_optimal_specific']):
		for metric_k, outer_k, ylab in zip(['avg_latency_difference', 'frac_vol_congested'], ['latency', 'congestion'], 
			['Avg Suboptimality\n During {} Failure (ms)'.format(tp), 'Pct Volume Congested\nDuring {} Failure'.format(tp)]):
			f,ax = get_figure()
			fig_fn = 'average_{}_over_{}_fail_{}_{}.pdf'.format(outer_k, evaluate_over, tp.lower(), lp_tp)
			metric_by_solution = {}
			for solution in solutions:
				agg_metrics =[]
				for dpsize in dpsizes:
					try:
						these_metrics = metrics_by_dpsize[dpsize][tp_k][solution]
						if outer_k == 'congestion':
							agg_metrics.append(100*these_metrics[metric_k])
						else:
							agg_metrics.append(-1*these_metrics[metric_k])
					except KeyError:
						continue
				ax.plot(dpsizes[0:len(agg_metrics)], agg_metrics, label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
				metric_by_solution[solution] = np.array(agg_metrics)
			ax.set_xlabel(xlab)
			ax.set_ylabel(ylab)
			ax.legend(fontsize=12)
			save_figure(fig_fn)

			if lp_tp == 'mlu':
				print("{} -- {}".format(tp, outer_k))
				print("Painter - Sparse: {}".format(metric_by_solution['painter'] - metric_by_solution['sparse']))
				print("Sparse - OPP: {}".format(metric_by_solution['sparse'] - metric_by_solution['one_per_peering']))
				for solution in solutions:
					print("{} average over all deployments: {}".format(solution,np.average(metric_by_solution[solution])))
				print_metrics['single_{}_failure'.format(tp)] = metric_by_solution['sparse']
				print('\n')


	metric_access_ks = ['stats_latency_thresholds_normal', 'stats_latency_thresholds_fail_popp', 'stats_latency_thresholds_fail_pop',
		'stats_latency_penalty_thresholds_normal', 'stats_latency_penalty_thresholds_fail_popp', 'stats_latency_penalty_thresholds_fail_pop',
		]

	solutions = list(reversed(solutions))
	plot_metrics= {solution:{k:{} for k in metric_access_ks} for solution in solutions}
	parsed_metrics = {}
	lat_thresholds = {}
	for solution in solutions:
		agg_metrics = {k:{dpsize:{} for dpsize in dpsizes} for k in metric_access_ks}
		for dpsize in dpsizes:
			for k in metric_access_ks:
				try:
					these_metrics = metrics_by_dpsize[dpsize][k][solution]
				except KeyError:
					pass					
				for soln_i in these_metrics:
					for lat_threshold in these_metrics[soln_i]:
						lat_thresholds[lat_threshold] = None
						try:
							agg_metrics[k][dpsize][lat_threshold].append(these_metrics[soln_i][lat_threshold])
						except KeyError:
							agg_metrics[k][dpsize][lat_threshold] = [these_metrics[soln_i][lat_threshold]]
		parsed_metrics[solution] = agg_metrics
		n_lat_thresholds = len(lat_thresholds)
		for i,k in enumerate(metric_access_ks):
			for j,lat_threshold in enumerate(sorted(lat_thresholds,reverse=True)):
				try:
					arr = list([100-100*np.mean(parsed_metrics[solution][k][dpsize][lat_threshold]) for dpsize in dpsizes])		
					plot_metrics[solution][k][lat_threshold] = arr
				except KeyError:
					continue

	for lat_threshold in sorted(lat_thresholds):
		all_ylabs = ['Pct Traffic Within {} ms\n of Optimal (Normally)'.format(int(np.abs(lat_threshold))), 'Pct Traffic Within {} ms\n of Optimal (Link Failure)'.format(int(np.abs(lat_threshold))),
			 'Pct Traffic Within {} ms\n of Optimal (Site Failure)'.format(int(np.abs(lat_threshold))),
			 'Pct Traffic Within {} ms\n of Optimal (Normally) (Penalty)'.format(int(np.abs(lat_threshold))), 'Pct Traffic Within {} ms\n of Optimal (Link Failure) (Penalty)'.format(int(np.abs(lat_threshold))),
			 'Pct Traffic Within {} ms\n of Optimal (Site Failure) (Penalty)'.format(int(np.abs(lat_threshold))),
			 'Pct Traffic Within {} ms\n of Optimal (Normally)'.format(int(np.abs(lat_threshold))), 'Pct Traffic Within {} ms\n of Optimal (Link Failure)'.format(int(np.abs(lat_threshold))),
			 'Pct Traffic Within {} ms\n of Optimal (Site Failure)'.format(int(np.abs(lat_threshold))),]
		fns = ['normal','link_failure', 'site_failure', 'normal_penalty', 'link_failure_penalty', 'site_failure_penalty']
		for k, ylab, fn in zip(metric_access_ks, all_ylabs, fns):
			if 'penalty' in fn: continue ## ignore for now
			f,ax = get_figure()
			metric_by_solution = {}
			for solution in solutions:
				try:
					metric_by_solution[solution] = np.array(plot_metrics[solution][k][lat_threshold])
					ax.plot(dpsizes, plot_metrics[solution][k][lat_threshold], label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
				except KeyError:
					continue
			ax.set_xlabel(xlab)
			ax.set_ylabel(ylab)
			ax.legend(fontsize=12)
			fig_fn = "percent_traffic_within_{}_ms_{}_over_{}.pdf".format(int(np.abs(lat_threshold)), fn, evaluate_over)
			save_figure(fig_fn)

			if 'penalty' not in fn:
				print("Within {} ms during {}".format(int(np.abs(lat_threshold)), fn))
				print("OPP - Painter: {}".format(np.round(100 - (metric_by_solution['one_per_peering'] - metric_by_solution['painter']),2)))
				print("OPP - Sparse: {}".format(np.round(100-(metric_by_solution['one_per_peering'] - metric_by_solution['sparse']),2)))
				for solution in solutions:
					print("{} average over all deployments: {}".format(solution,100-(np.average(metric_by_solution['one_per_peering']) - np.average(metric_by_solution[solution]))))
				print_metrics["{}_{}".format(lat_threshold, fn)] = metric_by_solution['sparse']
				print('\n')


	f,ax = get_figure()
	metric_by_solution = {}
	for solution in solutions:
		this_resiliences = []
		for dpsize in dpsizes:
			try:
				avg_resilience = np.average(list(metrics_by_dpsize[dpsize]['stats_resilience_to_congestion'][solution].values()))
				this_resiliences.append(avg_resilience)
			except KeyError:
				pass
		ax.plot(dpsizes[0:len(this_resiliences)], this_resiliences, label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
		metric_by_solution[solution] = np.array(this_resiliences)
	ax.set_xlabel(xlab)
	ax.set_ylabel("Flash Crowd Intensity (M)\nBefore Congestion (Pct.)",fontsize=12)
	ax.set_ylim([0,300])
	ax.set_yticks([0,60,120,180,240,300])
	ax.legend(fontsize=12)
	save_figure('flash_crowd_blowup_before_congestion_over_{}.pdf'.format(evaluate_over))
	print("--Flash Crowd--")
	print("Sparse / Painter: {}".format(100 * metric_by_solution['sparse'] / metric_by_solution['painter']))
	print("OPP - Sparse: {}".format(metric_by_solution['one_per_peering'] - metric_by_solution['sparse']))
	print("------------------------")
	for solution in solutions:
		print("{} : {}".format(solution,metric_by_solution[solution]))
	print_metrics['flash_crowd'] = metric_by_solution['one_per_peering'] - metric_by_solution['sparse']
	print('\n')


	solutions = ['anycast', 'anyopt', 'one_per_pop', 'painter', 'sparse', 'one_per_peering']
	f,ax = get_figure()
	for solution in solutions:
		ax.plot(dpsizes, this_resiliences, label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
	ax.set_xlabel(xlab)
	ax.set_ylabel("Rate of\nLatency Increase (ms/byte)")
	ax.legend(fontsize=12)
	save_figure('latency_increase_up_to_threshold_over_{}.pdf'.format(evaluate_over))

	solutions = list(reversed(solutions))
	f,ax = get_figure()
	metric_by_solution = {}
	for solution in solutions:
		this_resiliences = []
		for dpsize in dpsizes:
			try:
				avg_resilience = np.average(list(metrics_by_dpsize[dpsize]['stats_diurnal'][solution].values()))
				this_resiliences.append(avg_resilience)
			except KeyError:
				pass
		ax.plot(dpsizes[0:len(this_resiliences)], this_resiliences, label=solution_to_plot_label[solution], marker=solution_to_marker[solution], color=solution_to_line_color[solution])
		metric_by_solution[solution] = np.array(this_resiliences)
	ax.set_xlabel(xlab)
	ax.set_ylabel("Diurnal Intensity (M)\nBefore Congestion (Pct.)",fontsize=12)
	ax.set_ylim([0,120])
	ax.set_yticks([0,40,80,120])
	ax.legend(fontsize=12)
	save_figure('diurnal_blowup_before_congestion_over_{}.pdf'.format(evaluate_over))
	print("--Diurnal--")
	print("sparse / painter: {}".format(100 * metric_by_solution['sparse'] / metric_by_solution['painter']))
	print("OPP - Sparse: {}".format(metric_by_solution['one_per_peering'] - metric_by_solution['sparse']))
	print("------------------------")
	for solution in solutions:
		print("{} : {}".format(solution,metric_by_solution[solution]))
	print_metrics['diurnal'] = metric_by_solution['one_per_peering'] - metric_by_solution['sparse']
	print('\n')

	return print_metrics

def compare_ab(old_metrics,new_metrics):
	for k in old_metrics:
		print(k)
		print(old_metrics[k] < new_metrics[k])
		print("{}\n{}".format(old_metrics[k],new_metrics[k]))
		print("{} vs {}".format(np.mean(old_metrics[k]),np.mean(new_metrics[k])))
		print('\n')


def _cli():
	"""CLI so the sweep can be driven without editing the file."""
	import argparse
	ap = argparse.ArgumentParser(description=pull_results_new.__doc__)
	ap.add_argument('--port', type=int, default=DEFAULT_PORT,
					help='vestigial under Ray; nothing binds it')
	ap.add_argument('--dpsizes', default=None,
					help='comma-separated PoP counts, e.g. 3,4,5,6')
	ap.add_argument('--nsim', default=None,
					help='single int, or comma list parallel to --dpsizes')
	ap.add_argument('--max-iter', type=int, default=None)
	ap.add_argument('--cache-fn', default=None,
					help='override the results cache (namespaces a test run)')
	ap.add_argument('--probe-n', default=None,
					help='measurement budget per solve() (SCULPTOR_PROBE_N). '
						 'An integer, or the literal "prefixes" to use each '
						 "deployment's own prefix count (which varies with "
						 'size). Implies --probe-mode smart unless one is '
						 'given. Default is DEFAULT_PROBE_N in constants.py.')
	ap.add_argument('--probe-mode', default=None,
					choices=['post_step', 'scheduled', 'slotted', 'gated',
							 'smart'],
					help='WHEN-probing policy. post_step = stock (measure '
						 'after every step that moved the advertisement, '
						 'no budget).')
	ap.add_argument('--plot', action='store_true', help='also run make_paper_plots')
	ap.add_argument('--figures-subdir', default=None,
					help="namespace this run's figures: figures/<subdir>/... (e.g. --figures-subdir real_sweep_2026_08). Sets SCULPTOR_FIG_SUBDIR.")
	a = ap.parse_args()
	# SCULPTOR_EVAL_SEED: seed the global numpy RNG so different arms of an
	# A/B draw IDENTICAL random deployments per sim (2026-08-22: unseeded,
	# the 4 startup-RB arms each drew different sim-0 deployments -- an
	# unpaired comparison at nsim=5 is noise). Deliberately env-gated:
	# production sweeps stay unseeded.
	_es = os.environ.get('SCULPTOR_EVAL_SEED')
	if _es:
		import numpy as _np
		_np.random.seed(int(_es))
		print('[sweep] SCULPTOR_EVAL_SEED={} -- deployment draws pinned'.format(_es), flush=True)
	if a.figures_subdir:
		os.environ['SCULPTOR_FIG_SUBDIR'] = a.figures_subdir
	# Measurement budget. Set in the environment because the solver reads it
	# at construction, and the sweep builds one solver per dpsize per sim.
	if a.probe_n is not None:
		if str(a.probe_n).strip().lower() not in ('prefixes', 'n_prefixes',
												  'prefix'):
			int(a.probe_n)          # fail fast on a typo, not 40 minutes in
		os.environ['SCULPTOR_PROBE_N'] = str(a.probe_n)
		if a.probe_mode is None:
			# a budget with no policy would silently do nothing: post_step
			# (the default) has no budget at all.
			a.probe_mode = 'smart'
	if a.probe_mode is not None:
		os.environ['SCULPTOR_PROBE_MODE'] = a.probe_mode
	print('[sweep] probing: mode={} budget={}'.format(
		os.environ.get('SCULPTOR_PROBE_MODE', 'post_step (stock)'),
		os.environ.get('SCULPTOR_PROBE_N', 'n/a')), flush=True)
	dpsizes = ([int(x) if x.strip().isdigit() else x.strip()
				for x in a.dpsizes.split(',')] if a.dpsizes else None)
	nsim = None
	if a.nsim:
		nsim = [int(x) for x in a.nsim.split(',')] if ',' in a.nsim else int(a.nsim)
	cache_fn = a.cache_fn or os.path.join(CACHE_DIR, 'testing_feature_cache_fn.pkl')
	pull_results_new(cache_fn, port=a.port, dpsizes=dpsizes,
					 n_sim_by_dpsize=nsim,
					 only_recalc=True if dpsizes else None,
					 max_iter=a.max_iter)
	if a.plot:
		make_paper_plots(cache_fn)
	return cache_fn


if __name__ == '__main__':
	import sys as _sys
	if len(_sys.argv) > 1:
		_cli(); raise SystemExit(0)
	# cache_fn = os.path.join(CACHE_DIR, 'evaluate_over_deployment_sizes_cache_fn.pkl')
	# pull_results_old(cache_fn)
	# high_level_metrics_old = make_paper_plots(cache_fn)

	cache_fn = os.path.join(CACHE_DIR, 'testing_feature_cache_fn.pkl')
	pull_results_new(cache_fn)
	# high_level_metrics_new = make_paper_plots(cache_fn)



	# compare_ab(high_level_metrics_old, high_level_metrics_new)


