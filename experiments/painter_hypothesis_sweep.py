"""Painter-degradation hypothesis sweep.

Hypothesis: painter's avg-latency gap above sparse widens as link headroom
shrinks AND per-user volume variance grows. Theory: painter's greedy add
ignores capacity, so when overprovisioning is tight and a few users carry
disproportionate volume, painter's choice can pile load on a popp that
sparse's LP would route around.

Grid axes:
  - scale_factor (link-cap multiplier over anycast load), driven via
    SCULPTOR_SCALE_FACTOR env var hooked in deployment_setup.get_link_capacities
  - vol_spread (log-uniform spread parameter for per-UG volumes), driven
    via SCULPTOR_VOL_SPREAD env var hooked in
    deployment_setup.get_random_deployment_by_size. vol = exp(s * U), so
    s=0 -> uniform 1.0, s=6 -> CV ~ 1.42.

Per cell: builds a fresh `small` deployment with these env vars, trains all
strategies under avg_latency, and records the per-strategy avg latency.

Output:
  - cache/experiments/painter_hypothesis_sweep/cell_sf{sf}_vs{vs}/run_obj_*.{pkl,md}
    per cell (the unmodified run_objective.run output)
  - cache/experiments/painter_hypothesis_sweep/grid_summary.pkl
  - cache/experiments/painter_hypothesis_sweep/grid_summary.md  (the headline table)

Local-only; 1-2 workers per cell (sessions are NOT a constraint -- see
WLS policy note in experiments/ablation/README.md; sized for a laptop)
concurrent.
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default grid. 5 x 5 = 25 cells.
DEFAULT_SCALE_FACTORS = (1.0, 1.05, 1.1, 1.2, 1.3)
DEFAULT_VOL_SPREADS = (0.0, 1.0, 2.0, 4.0, 6.0)


def _format_table(rows, headers):
	str_rows = [[str(r.get(h, '')) for h in headers] for r in rows]
	widths = [max(len(h), *(len(row[i]) for row in str_rows)) for i, h in enumerate(headers)]
	def line(cells):
		return '| ' + ' | '.join(c.ljust(widths[i]) for i, c in enumerate(cells)) + ' |'
	out = [line(headers)]
	out.append('|' + '|'.join('-' * (w + 2) for w in widths) + '|')
	for row in str_rows:
		out.append(line(row))
	return '\n'.join(out)


def _grid_md(grid, scale_factors, vol_spreads, metric_key, title):
	headers = ['vol_spread \\ scale_factor'] + ['{:.2f}'.format(sf) for sf in scale_factors]
	rows = []
	for vs in vol_spreads:
		row = {'vol_spread \\ scale_factor': '{:.2f}'.format(vs)}
		for sf in scale_factors:
			cell = grid.get((sf, vs), {})
			val = cell.get(metric_key)
			row['{:.2f}'.format(sf)] = '{:.3f}'.format(val) if isinstance(val, (int, float)) else 'n/a'
		rows.append(row)
	return '### ' + title + '\n\n' + _format_table(rows, headers) + '\n'


def _per_strategy_avg_latency(metrics):
	"""Extract avg traffic-weighted latency per strategy from a run_objective result."""
	import numpy as np
	ug_vols = np.asarray(metrics.get('ug_to_vol', []), dtype=float)
	out = {}
	for sname, entry in (metrics.get('per_strategy') or {}).items():
		lp = entry.get('lp_solution', {}) or {}
		lats = np.asarray(lp.get('lats_by_ug', []), dtype=float)
		if lats.size and ug_vols.size:
			out[sname] = float(np.average(lats, weights=ug_vols))
		else:
			out[sname] = float('nan')
	return out


def run_cell(scale_factor, vol_spread, *, dpsize, max_iter, n_workers, seed,
			 port, out_root):
	from experiments import run_objective

	cell_dir = os.path.join(out_root, 'cell_sf{:.2f}_vs{:.2f}'.format(scale_factor, vol_spread))
	os.makedirs(cell_dir, exist_ok=True)

	# Pin RNG so the same base deployment skeleton is used per cell.
	# get_random_deployment also honors SCULPTOR_DEPLOYMENT_SEED.
	os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(seed)
	os.environ['SCULPTOR_SCALE_FACTOR'] = '{:.6f}'.format(scale_factor)
	os.environ['SCULPTOR_VOL_SPREAD'] = '{:.6f}'.format(vol_spread)

	print('\n' + '#' * 70)
	print('# cell sf={} vs={}  port={} max_iter={} n_workers={} seed={}'.format(
		scale_factor, vol_spread, port, max_iter, n_workers, seed))
	print('#' * 70)

	t0 = time.time()
	metrics, _summary = run_objective.run(
		'avg_latency', dpsize, port,
		max_iter=max_iter, n_workers=n_workers,
		seed=seed, out_dir=cell_dir,
		extra_evals=('static_failure_resilience',),
	)
	dt = time.time() - t0

	lats = _per_strategy_avg_latency(metrics)
	painter = lats.get('painter', float('nan'))
	sparse = lats.get('sparse', float('nan'))
	gap = painter - sparse  # +ve -> painter worse than sparse

	# Failure-eval extraction. metrics['static_failure_resilience'][sname] =
	# {'popp': {avg_lat_steady, avg_lat_failure, frac_no_route_failure, ...},
	#  'pop':  {... same shape ...}}
	# We pull steady + mean-over-single-failure latencies and no_route fracs.
	sfr = metrics.get('static_failure_resilience') or {}
	failure_metrics = {}
	for sname, r in sfr.items():
		if 'error' in r:
			failure_metrics[sname] = {'error': r['error']}
			continue
		failure_metrics[sname] = {
			'steady_lat': float(r.get('popp', {}).get('avg_lat_steady', float('nan'))),
			'popp_fail_lat': float(r.get('popp', {}).get('avg_lat_failure', float('nan'))),
			'popp_fail_noroute': float(r.get('popp', {}).get('frac_no_route_failure', float('nan'))),
			'pop_fail_lat': float(r.get('pop', {}).get('avg_lat_failure', float('nan'))),
			'pop_fail_noroute': float(r.get('pop', {}).get('frac_no_route_failure', float('nan'))),
			'popp_worst_lat': float(r.get('popp', {}).get('worst_lat_failure', float('nan'))),
		}

	return {
		'scale_factor': scale_factor,
		'vol_spread': vol_spread,
		'wall_secs': dt,
		'per_strategy_lat_ms': lats,
		'painter_lat_ms': painter,
		'sparse_lat_ms': sparse,
		'painter_minus_sparse_ms': gap,
		'failure_metrics': failure_metrics,
	}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--dpsize', default='small')
	ap.add_argument('--max-iter', type=int, default=50)
	ap.add_argument('--n-workers', type=int, default=1)
	ap.add_argument('--seed', type=int, default=31415)
	ap.add_argument('--port-base', type=int, default=32100,
					help='Starting port; each cell uses port_base + i.')
	ap.add_argument('--scale-factors', default=','.join('{}'.format(s) for s in DEFAULT_SCALE_FACTORS),
					help='Comma-separated scale_factor values.')
	ap.add_argument('--vol-spreads', default=','.join('{}'.format(s) for s in DEFAULT_VOL_SPREADS),
					help='Comma-separated vol_spread values.')
	ap.add_argument('--out-root', default=None,
					help='Where to put the sweep dir. Defaults under $CACHE_DIR/experiments/painter_hypothesis_sweep.')
	ap.add_argument('--only', default=None,
					help='Run only the (sf,vs) cell given as "sf=1.0,vs=2.0". Smoke-test shortcut.')
	args = ap.parse_args()

	from helpers.constants import CACHE_DIR
	out_root = args.out_root or os.path.join(CACHE_DIR, 'experiments', 'painter_hypothesis_sweep')
	os.makedirs(out_root, exist_ok=True)

	sfs = [float(s) for s in args.scale_factors.split(',') if s.strip()]
	vss = [float(s) for s in args.vol_spreads.split(',') if s.strip()]

	if args.only:
		parts = dict(p.split('=') for p in args.only.split(','))
		sfs = [float(parts['sf'])]
		vss = [float(parts['vs'])]

	cells = [(sf, vs) for sf in sfs for vs in vss]
	print('Sweep: {} cells, sfs={}, vss={}'.format(len(cells), sfs, vss))

	grid = {}
	t0 = time.time()
	for i, (sf, vs) in enumerate(cells):
		port = args.port_base + i
		try:
			rec = run_cell(sf, vs,
						   dpsize=args.dpsize,
						   max_iter=args.max_iter,
						   n_workers=args.n_workers,
						   seed=args.seed,
						   port=port,
						   out_root=out_root)
		except Exception as e:
			import traceback; traceback.print_exc()
			rec = {
				'scale_factor': sf, 'vol_spread': vs,
				'error': '{}: {}'.format(type(e).__name__, e),
				'painter_lat_ms': float('nan'),
				'sparse_lat_ms': float('nan'),
				'painter_minus_sparse_ms': float('nan'),
			}
		grid[(sf, vs)] = rec
		# Persist incrementally so a crash mid-sweep is recoverable.
		with open(os.path.join(out_root, 'grid_summary.pkl'), 'wb') as f:
			pickle.dump({'grid': grid, 'scale_factors': sfs, 'vol_spreads': vss,
						 'args': vars(args)}, f)
		elapsed = time.time() - t0
		remaining = elapsed / (i + 1) * (len(cells) - i - 1)
		print('  >> cell {}/{} done. sf={} vs={} painter-sparse={:.3f}ms.  '
			  '{:.0f}s elapsed, ~{:.0f}s remaining'.format(
			i + 1, len(cells), sf, vs,
			rec.get('painter_minus_sparse_ms', float('nan')),
			elapsed, remaining,
		))

	# Discover the set of strategies that ran at least once (cells may error).
	STRATS = set()
	for rec in grid.values():
		STRATS.update((rec.get('failure_metrics') or {}).keys())
	# Stable order for tables.
	STRAT_ORDER = [s for s in
				   ('sparse', 'painter', 'one_per_peering', 'one_per_pop',
					'anyopt', 'anycast', 'random') if s in STRATS]

	def _failure_grid_md(metric_key, title):
		"""Grid (rows=vs, cols=sf) of one strategy's failure metric."""
		headers = ['vol_spread \\ scale_factor'] + ['{:.2f}'.format(sf) for sf in sfs]
		rows = []
		for vs in vss:
			row = {'vol_spread \\ scale_factor': '{:.2f}'.format(vs)}
			for sf in sfs:
				rec = grid.get((sf, vs)) or {}
				fm = (rec.get('failure_metrics') or {}).get(strat_name, {})
				v = fm.get(metric_key)
				row['{:.2f}'.format(sf)] = '{:.3f}'.format(v) if isinstance(v, (int, float)) and v == v else 'n/a'
			rows.append(row)
		return '#### ' + title + '\n\n' + _format_table(rows, headers) + '\n'

	md = '# Painter-degradation hypothesis sweep\n\n'
	md += '- dpsize: `{}`\n- max_iter: {}\n- n_workers: {}\n- seed: {}\n'.format(
		args.dpsize, args.max_iter, args.n_workers, args.seed)
	md += '- {} cells, total wall {:.1f}s\n\n'.format(len(cells), time.time() - t0)
	md += '## Painter vs sparse, no failure (training LP)\n\n'
	md += _grid_md(grid, sfs, vss, 'painter_minus_sparse_ms',
				   'painter avg_lat - sparse avg_lat (ms)  -- larger ↔ painter worse')
	md += '\n' + _grid_md(grid, sfs, vss, 'painter_lat_ms', 'painter avg_lat (ms)')
	md += '\n' + _grid_md(grid, sfs, vss, 'sparse_lat_ms', 'sparse avg_lat (ms)')

	# Per-strategy steady / popp_fail / pop_fail tables (BGP-fallback semantic,
	# from static_failure_resilience). Mean over single popp (or pop) failures
	# of the volume-weighted avg user latency.
	md += '\n## Per-strategy failure resilience (BGP-fallback)\n\n'
	md += 'Latency reported is volume-weighted avg over users. '
	md += 'popp_fail = mean over single popp failures. pop_fail = mean over single pop failures. '
	md += 'no_route fractions are mean volume share routed to no_route under failure.\n\n'
	for strat_name in STRAT_ORDER:
		md += '### Strategy: `{}`\n\n'.format(strat_name)
		md += _failure_grid_md('steady_lat', 'steady avg_lat (ms)')
		md += '\n' + _failure_grid_md('popp_fail_lat', 'popp_fail mean avg_lat (ms)')
		md += '\n' + _failure_grid_md('popp_fail_noroute', 'popp_fail mean no_route frac')
		md += '\n' + _failure_grid_md('pop_fail_lat', 'pop_fail mean avg_lat (ms)')
		md += '\n' + _failure_grid_md('pop_fail_noroute', 'pop_fail mean no_route frac')
		md += '\n'
	with open(os.path.join(out_root, 'grid_summary.md'), 'w') as f:
		f.write(md)
	print('\n' + md)
	print('\nWrote grid_summary.{pkl,md} under', out_root)


if __name__ == '__main__':
	main()
