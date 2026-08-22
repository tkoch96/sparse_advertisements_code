"""Single-objective CLI driver.

Usage:
    python -m experiments.run_objective --obj <name> --dpsize <size> --port <port> \
        [--max-iter N] [--n-workers N] [--seed S] [--save-run-dir DIR]

Looks up the ObjectiveSpec from experiments/objectives.py, builds a fresh
deployment, trains all strategies (sparse, painter, anyopt, anycast,
one_per_pop, one_per_peering) via compare_different_solutions, and dumps:
  - a pickle with per-strategy advertisements + LP solutions
  - a markdown table (per-strategy latency/congestion/objective-specific cols)

Stays narrow on purpose: doesn't run the heavy eval phases (failure
resilience, diurnal, flash crowd). Those still live in eval_all_solution_types
and can be invoked separately when wanted.
"""
import argparse
import os
import pickle
import sys
import time
import numpy as np

# Stay backwards-compatible with the unconfigured script-style imports the
# rest of this codebase uses.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.objectives import get as get_spec, all_specs
# Late-load specs so their register() runs even when run --list-only.
import experiments.site_failure  # noqa: F401


def _format_table(rows, headers):
	"""Render a list of dicts as a github-markdown table."""
	# Stringify
	str_rows = [[str(r.get(h, '')) for h in headers] for r in rows]
	widths = [max(len(h), *(len(row[i]) for row in str_rows)) for i, h in enumerate(headers)]
	def line(cells):
		return '| ' + ' | '.join(c.ljust(widths[i]) for i, c in enumerate(cells)) + ' |'
	out = [line(headers)]
	out.append('|' + '|'.join('-' * (w + 2) for w in widths) + '|')
	for row in str_rows:
		out.append(line(row))
	return '\n'.join(out)


def _summarize(metrics, spec, dpsize):
	"""Produce a markdown summary of the strategy_compare result.

	Pulled fields per strategy (from compare_different_solutions output):
	  - avg_latency_ms : traffic-weighted average user latency
	  - frac_congested : fraction of volume placed on congested links
	  - lp_objective   : raw LP objective value (objective-specific units)
	  - site_cost_total: site cost summed across (vol * cost_per_unit), only
	                     populated when site_costs is in the deployment
	"""
	deployment = metrics.get('deployment', {})
	ug_vols = np.asarray(metrics.get('ug_to_vol', []), dtype=float)
	total_vol = ug_vols.sum() if ug_vols.size else 1.0
	site_costs = deployment.get('site_costs', {}) if isinstance(deployment, dict) else {}

	strategies = sorted(metrics.get('per_strategy', {}).keys())
	rows = []
	for s in strategies:
		entry = metrics['per_strategy'][s]
		lp = entry.get('lp_solution', {}) or {}
		lats = np.asarray(lp.get('lats_by_ug', []), dtype=float)
		if lats.size and ug_vols.size:
			avg_lat = float(np.average(lats, weights=ug_vols))
		else:
			avg_lat = float('nan')
		frac_cong = float(lp.get('fraction_congested_volume', float('nan')))
		obj_val = lp.get('objective', float('nan'))

		# Site cost (only meaningful for site-cost objectives but cheap to compute)
		site_cost_total = float('nan')
		if site_costs:
			vols_by_poppi = lp.get('vols_by_poppi', {}) or {}
			site_cost_total = 0.0
			popps = deployment.get('popps') or []
			for poppi, vol in vols_by_poppi.items():
				try:
					site, _ = popps[poppi]
				except (IndexError, TypeError):
					continue
				site_cost_total += float(vol) * float(site_costs.get(site, 0.0))

		rows.append({
			'strategy': s,
			'avg_latency_ms': '{:.3f}'.format(avg_lat),
			'frac_congested': '{:.4f}'.format(frac_cong),
			'lp_objective': '{:.4f}'.format(obj_val) if isinstance(obj_val, (int, float, np.floating)) else str(obj_val),
			'site_cost_total': '{:.3f}'.format(site_cost_total) if site_cost_total == site_cost_total else 'n/a',
		})
	headers = ['strategy', 'avg_latency_ms', 'frac_congested', 'lp_objective', 'site_cost_total']
	table = _format_table(rows, headers)
	out = (
		"## Objective: `{}` (dpsize={})\n\n".format(spec.name, dpsize)
		+ "{}\n\n".format(spec.description)
		+ "LP kwargs: `{}`. gamma={}. using_resilience_benefit={}.\n\n".format(
			spec.lp_kwargs, spec.gamma, spec.using_resilience_benefit)
		+ table + "\n"
	)

	# Optional static_failure_resilience section (added when the eval phase
	# was run for this objective).
	sfr = metrics.get('static_failure_resilience')
	if sfr:
		sfr_rows = []
		for sname in sorted(sfr.keys()):
			r = sfr[sname]
			if 'error' in r:
				sfr_rows.append({'strategy': sname, 'note': 'ERROR: ' + r['error']})
				continue
			popp = r.get('popp', {})
			pop = r.get('pop', {})
			sfr_rows.append({
				'strategy': sname,
				'steady_lat_ms': '{:.2f}'.format(popp.get('avg_lat_steady', float('nan'))),
				'popp_fail_lat_ms': '{:.2f}'.format(popp.get('avg_lat_failure', float('nan'))),
				'popp_fail_noroute': '{:.4f}'.format(popp.get('frac_no_route_failure', float('nan'))),
				'pop_fail_lat_ms': '{:.2f}'.format(pop.get('avg_lat_failure', float('nan'))),
				'pop_fail_noroute': '{:.4f}'.format(pop.get('frac_no_route_failure', float('nan'))),
				'worst_popp_lat': '{:.2f}'.format(popp.get('worst_lat_failure', float('nan'))),
			})
		sfr_headers = ['strategy', 'steady_lat_ms', 'popp_fail_lat_ms', 'popp_fail_noroute',
					   'pop_fail_lat_ms', 'pop_fail_noroute', 'worst_popp_lat']
		# Drop rows that only have a 'note' (errors) -- render them after
		if any('note' in r for r in sfr_rows):
			ok_rows = [r for r in sfr_rows if 'note' not in r]
			err_rows = [r for r in sfr_rows if 'note' in r]
		else:
			ok_rows, err_rows = sfr_rows, []
		out += "\n### Static-failure resilience (BGP-fallback semantic, frozen user→prefix)\n\n"
		if ok_rows:
			out += _format_table(ok_rows, sfr_headers) + "\n"
		for r in err_rows:
			out += "- {}: {}\n".format(r['strategy'], r['note'])
		out += "\n"
	return out


def run(spec_name, dpsize, port, max_iter=None, n_workers=None,
		seed=31415, save_run_dir=None, out_dir=None, extra_evals=()):
	# Imports deferred so `--list` works in a fresh checkout without firing
	# the heavy SAS / Gurobi import chain.
	from helpers.constants import CACHE_DIR, DEFAULT_EXPLORE
	from evaluations.wrapper_eval import capacity  # module-level True; matches existing testing_*.py
	from core.deployment_setup import get_random_deployment
	from core.sparse_advertisements_v3 import Sparse_Advertisement_Eval
	from core.worker_comms import Worker_Manager
	from helpers.helpers import deployment_to_prefixes

	spec = get_spec(spec_name)

	# Honor caller's worker cap by setting env BEFORE Worker_Manager spawns. This
	# is the same lever the Ray Worker_Manager already respects; we extended the
	# non-Ray path in worker_comms.py to match.
	if n_workers is not None:
		os.environ['SCULPTOR_N_WORKERS'] = str(n_workers)
	if max_iter is not None:
		os.environ['SCULPTOR_MAX_ITER'] = str(max_iter)
	# Apply spec.train_env (e.g., SCULPTOR_CAPACITY_HEADROOM=0.2 for
	# static_failure). Save previous values so we restore on exit.
	prior_env = {}
	for k, v in spec.train_env.items():
		prior_env[k] = os.environ.get(k)
		os.environ[k] = str(v)
		print("[spec env] {}={!r} (was {!r})".format(k, str(v), prior_env[k]))

	print("=" * 70)
	print("Objective: {}    dpsize={}    port={}".format(spec.name, dpsize, port))
	print("LP kwargs: {}".format(spec.lp_kwargs))
	print("gamma={}    using_resilience_benefit={}".format(spec.gamma, spec.using_resilience_benefit))
	print("max_iter={}    n_workers={}    seed={}".format(
		os.environ.get('SCULPTOR_MAX_ITER', '<default>'),
		os.environ.get('SCULPTOR_N_WORKERS', '<default>'),
		seed))
	print("=" * 70)

	np.random.seed(seed)

	# Build deployment with spec.deployment_kwargs (e.g., cost_type='carbon')
	deployment = get_random_deployment(dpsize, **spec.deployment_kwargs)
	deployment['port'] = port
	n_prefixes = deployment_to_prefixes(deployment)

	# Train all strategies under this objective
	sas = Sparse_Advertisement_Eval(
		deployment,
		verbose=True,
		lambduh=0,
		with_capacity=capacity,
		explore=DEFAULT_EXPLORE,
		using_resilience_benefit=spec.using_resilience_benefit,
		gamma=spec.gamma,
		n_prefixes=n_prefixes,
		generic_objective=spec.lp_obj_string,
		lp_kwargs=dict(spec.lp_kwargs),  # forwarded to Generic_Objective
		save_run_dir=save_run_dir,
	)

	wm = Worker_Manager(sas.get_init_kwa(), deployment)
	# Adaptive resize: if SCULPTOR_N_WORKERS_DURING_PARALLEL is set, start
	# with the reduced pool; compare_different_solutions's watcher will
	# request_add_workers up to the SCULPTOR_N_WORKERS target when the
	# concurrent parallel-strategy subprocesses finish.
	_dp_env = os.environ.get('SCULPTOR_N_WORKERS_DURING_PARALLEL')
	_dp_initial = None
	if _dp_env is not None:
		try:
			_dp_initial = int(_dp_env)
		except ValueError:
			print("WARNING: SCULPTOR_N_WORKERS_DURING_PARALLEL={!r} is not an int; ignoring".format(_dp_env))
	wm.start_workers(n_workers_override=_dp_initial)
	try:
		sas.set_worker_manager(wm)
		sas.update_deployment(deployment)

		print("Starting compare_different_solutions ...")
		t0 = time.time()
		ret = sas.compare_different_solutions(
			n_run=1, verbose=True, dont_update_deployment=True,
		)
		train_secs = time.time() - t0
		print("compare_different_solutions done in {:.1f}s".format(train_secs))

		# Re-evaluate each strategy's adv under the configured generic objective
		# so that the reported LP-solution row uses the *objective being studied*,
		# not whatever the strategy happened to optimize internally.
		per_strategy = {}
		for sname, advs in ret.get('adv_solns', {}).items():
			if not advs:
				continue
			adv = advs[0]
			lp_solution = sas.sas.generic_objective.get_latency_benefit_adv(adv)
			per_strategy[sname] = {'adv': adv, 'lp_solution': lp_solution}

		metrics = {
			'objective': spec.name,
			'dpsize': dpsize,
			'deployment': sas.output_deployment(),
			'ug_to_vol': sas.ug_vols,
			'compare_rets': ret,
			'per_strategy': per_strategy,
			'optimal_objective': getattr(sas, 'optimal_expensive_solution', None),
			'spec_lp_kwargs': dict(spec.lp_kwargs),
			'train_secs': train_secs,
			'save_run_dir': sas.sas.save_run_dir,
		}

		# Extra eval phases beyond strategy_compare are dispatched here.
		# 'strategy_compare' itself is already done above; this loop runs any
		# additional named phases. Stays narrow: silent skip for unknown phases
		# (existing eval files like eval_all_solution_types.py handle them).
		# Caller can append phases via extra_evals (e.g., to run
		# static_failure_resilience against a non-static_failure trained adv
		# for apples-to-apples comparison).
		phases_to_run = tuple(spec.eval_phases) + tuple(extra_evals)
		for phase in phases_to_run:
			if phase in ('strategy_compare',):
				continue
			if phase == 'static_failure_resilience':
				print("[eval] static_failure_resilience: per-popp BGP-fallback ...")
				from core.static_failure_eval import assess_static_failure_resilience
				phase_out = {}
				for sname, entry in per_strategy.items():
					try:
						r_popp = assess_static_failure_resilience(sas.sas, entry['adv'], which='popps')
						r_pop = assess_static_failure_resilience(sas.sas, entry['adv'], which='pops')
						phase_out[sname] = {'popp': r_popp, 'pop': r_pop}
						print("  {:>18s}  popp: steady={:.2f} fail={:.2f} no_route={:.3f}  pop: fail={:.2f} no_route={:.3f}".format(
							sname,
							r_popp['avg_lat_steady'], r_popp['avg_lat_failure'], r_popp['frac_no_route_failure'],
							r_pop['avg_lat_failure'], r_pop['frac_no_route_failure'],
						))
					except Exception as e:
						import traceback; traceback.print_exc()
						phase_out[sname] = {'error': str(e)}
				metrics['static_failure_resilience'] = phase_out
			# Other named phases (e.g., 'failure_resilience', 'diurnal',
			# 'flash_crowd', 'priority_bulk_sweep', 'site_cost_summary') can
			# be added here in time; they would call into the existing eval
			# code in eval_all_solution_types.py / testing_*.py.

	finally:
		try:
			wm.stop_workers()
		except Exception as e:
			print("warning: wm.stop_workers raised {}".format(e))
		# Restore any env vars we mutated for spec.train_env.
		for k, prev in prior_env.items():
			if prev is None:
				os.environ.pop(k, None)
			else:
				os.environ[k] = prev

	out_dir = out_dir or os.path.join(CACHE_DIR, 'experiments')
	os.makedirs(out_dir, exist_ok=True)
	pkl_fn = os.path.join(out_dir, 'run_obj_{}_{}.pkl'.format(spec.name, dpsize))
	with open(pkl_fn, 'wb') as f:
		pickle.dump(metrics, f)
	md_fn = os.path.join(out_dir, 'run_obj_{}_{}.md'.format(spec.name, dpsize))
	summary = _summarize(metrics, spec, dpsize)
	with open(md_fn, 'w') as f:
		f.write(summary)
	print("\n" + summary)
	print("Wrote {}".format(pkl_fn))
	print("Wrote {}".format(md_fn))
	return metrics, summary


def _list_objectives():
	for s in all_specs():
		print("- {:<20s} lp={:<32s} {}".format(s.name, s.lp_obj_string, s.description[:80]))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--obj', required=False, help='Objective name (see --list)')
	ap.add_argument('--dpsize', default='small', help='Deployment size (default: small)')
	ap.add_argument('--port', type=int, default=31510, help='Worker port (default: 31510)')
	ap.add_argument('--max-iter', type=int, default=None, help='Override SCULPTOR_MAX_ITER')
	ap.add_argument('--n-workers', type=int, default=None, help='Override SCULPTOR_N_WORKERS')
	ap.add_argument('--seed', type=int, default=31415)
	ap.add_argument('--save-run-dir', default=None, help='Existing run dir for hot-start')
	ap.add_argument('--out-dir', default=None, help='Where to write pkl+md (default: $CACHE_DIR/experiments)')
	ap.add_argument('--list', action='store_true', help='List available objectives and exit')
	ap.add_argument('--extra-evals', default='', help='Comma-separated extra eval phase names to run on top of spec.eval_phases')
	args = ap.parse_args()

	if args.list or not args.obj:
		_list_objectives()
		return
	extra_evals = tuple(e for e in args.extra_evals.split(',') if e.strip())
	run(args.obj, args.dpsize, args.port,
		max_iter=args.max_iter, n_workers=args.n_workers,
		seed=args.seed, save_run_dir=args.save_run_dir, out_dir=args.out_dir,
		extra_evals=extra_evals)


if __name__ == '__main__':
	main()
