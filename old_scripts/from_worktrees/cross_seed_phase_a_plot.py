"""Cross-seed aggregate of Phase A (small × N=3, headroom 0.2).

Loads 3 per-seed metrics pickles and produces a single PDF with 3 panels:
  - Normal LP (no failure)
  - Popp-failure (single-popp drop)
  - Pop-failure (whole-pop drop)
Each panel plots volume-weighted CDFs of `best_perf - new_perf` per solver,
with one line per seed (lighter weight) plus a thicker median line. Sign:
0 = solution matches per-UG-best; negative = solution worse than best.
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.expanduser('~/Documents/sparse_advertisements_code/benchmarks/out/phase_a_pdfs')
SEEDS = [1, 2, 3]
SOLVERS = ['sparse', 'painter', 'anyopt', 'anycast', 'one_per_pop', 'one_per_peering']
COLORS = {
	'sparse': 'C0',
	'painter': 'C3',
	'anyopt': 'C2',
	'anycast': 'C4',
	'one_per_pop': 'C5',
	'one_per_peering': 'k',
}


def load(seed):
	p = os.path.join(BASE, f'popp_failure_latency_comparison_small_hd_seed{seed}_small.pkl')
	return pickle.load(open(p, 'rb'))


def vol_weighted_cdf(diffs, vols):
	"""Volume-weighted CDF: returns (sorted_diffs, cumulative_volume_fraction)."""
	if not len(diffs):
		return np.array([0.0]), np.array([0.0])
	diffs = np.asarray(diffs, dtype=float)
	vols = np.asarray(vols, dtype=float)
	order = np.argsort(diffs)
	diffs = diffs[order]
	vols = vols[order]
	cum = np.cumsum(vols)
	cum = cum / cum[-1]
	return diffs, cum


def extract(metrics_pkl, key, solver):
	"""key like 'popp_failures_latency_optimal_specific'; returns (diffs, vols) flattened across all UGs/failures."""
	if key == 'normal':
		# Compare deployment latency under no failure: best - new per UG
		best_lats = metrics_pkl['best_latencies'][0]   # dict ug_idx -> latency
		new_lats = metrics_pkl['latencies'][0][solver]
		ug_to_vol = metrics_pkl['ug_to_vol'][0]
		diffs, vols = [], []
		# best_latencies and latencies and ug_to_vol all keyed the same way
		for ugi, blat in enumerate(best_lats):
			try:
				nlat = new_lats[ugi]
				v = ug_to_vol[ugi]
			except (KeyError, IndexError):
				continue
			if blat < 0 or nlat < 0:
				continue
			diffs.append(blat - nlat)
			vols.append(v)
		return diffs, vols
	tuples = metrics_pkl[key][0][solver]
	# each tuple is (latency_delta, vol, ug, element, perf1, perf2, [paths])
	diffs = [float(t[0]) for t in tuples]
	vols = [float(t[1]) for t in tuples]
	return diffs, vols


def plot_panel(ax, metrics_list, key, title):
	for solver in SOLVERS:
		all_seed_xy = []
		for seed_idx, m in enumerate(metrics_list):
			diffs, vols = extract(m, key, solver)
			if not diffs:
				continue
			x, cdf = vol_weighted_cdf(diffs, vols)
			# Sample CDF at common x-grid for median
			all_seed_xy.append((x, cdf))
			ax.plot(x, cdf, color=COLORS[solver], alpha=0.30, linewidth=1.0,
					label=None)
		# Compute median CDF on a shared x-grid
		if all_seed_xy:
			x_all = np.concatenate([x for x, _ in all_seed_xy])
			x_lo, x_hi = float(np.percentile(x_all, 2)), float(np.percentile(x_all, 98))
			x_grid = np.linspace(x_lo, x_hi, 200)
			interp_cdfs = []
			for x, cdf in all_seed_xy:
				interp_cdfs.append(np.interp(x_grid, x, cdf, left=0.0, right=1.0))
			med_cdf = np.median(interp_cdfs, axis=0)
			ax.plot(x_grid, med_cdf, color=COLORS[solver], linewidth=2.5,
					label=f'{solver}')
	ax.set_title(title)
	ax.set_xlabel('best_perf - new_perf (ms; 0 = matches per-UG-best, more negative = worse)')
	ax.set_ylabel('volume-weighted CDF')
	ax.grid(True, alpha=0.3)
	ax.legend(loc='lower right', fontsize=9)


def main():
	metrics_list = [load(s) for s in SEEDS]
	print(f'loaded seeds {SEEDS}')
	fig, axes = plt.subplots(1, 3, figsize=(18, 5))
	# Note: 'normal' key path may not exist if eval didn't populate `latencies`
	# field cleanly. Skip if errors.
	try:
		plot_panel(axes[0], metrics_list, 'normal', 'Normal LP (no failure)')
	except Exception as e:
		axes[0].text(0.5, 0.5, f'normal panel skipped: {e}',
			transform=axes[0].transAxes, ha='center')
	plot_panel(axes[1], metrics_list, 'popp_failures_latency_optimal_specific',
		'Single popp failure')
	plot_panel(axes[2], metrics_list, 'pop_failures_latency_optimal_specific',
		'Single pop failure')

	fig.suptitle('Phase A cross-seed CDF (small × N=3, headroom 0.2). '
				 'Thin lines = per-seed; thick = cross-seed median.',
				 fontsize=11)
	plt.tight_layout()
	out = os.path.join(BASE, 'phase_a_cross_seed_comparison.pdf')
	plt.savefig(out, bbox_inches='tight')
	print(f'wrote {out}')


if __name__ == '__main__':
	main()
