"""
Stochastic LP study: speedup vs problem size (plot a) + convergence vs
sub-sample size (plot b).

Produces:
  benchmarks/out/stochastic_lp_speedup.csv
  benchmarks/out/stochastic_lp_speedup.png
  benchmarks/out/stochastic_lp_convergence.csv
  benchmarks/out/stochastic_lp_convergence.png

Run: python benchmarks/stochastic_lp_study.py
"""
import os
import sys
import csv
import time
import random
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))   # for sibling util imports

# Headless plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), 'out')
os.makedirs(OUT_DIR, exist_ok=True)


def _build_worker(size, seed=31415):
	"""Build one _LocalPathDistributionComputer at a deployment size.

	Works with both synthetic sizes ('small', 'decent', etc.) and 'actual-N'
	deployments. The latter use SCULPTOR_DEPLOYMENT_SEED-equivalent seeding
	via np.random.seed; if a per-seed deployment cache exists at
	cache/deployments/, it'll be reused (instant). Otherwise the 4.5GB
	CSV parse happens (~5 min).
	"""
	random.seed(seed)
	np.random.seed(seed)
	if 'actual' in size:
		# Mirror SCULPTOR_DEPLOYMENT_SEED behavior so cache filenames match
		os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(seed)

	from deployment_setup import get_random_deployment
	from helpers import split_deployment_by_ug
	from path_distribution_computer_ray import _LocalPathDistributionComputer

	dep = get_random_deployment(size, port=31800)
	subdep = split_deployment_by_ug(dep, n_chunks=1)[0]
	init_kwa = {
		'lambduh': 1.0, 'gamma': 0, 'verbose': False,
		'n_prefixes': None, 'with_capacity': False,
		'save_run_dir': None, 'generic_objective': 'avg_latency',
	}
	worker = _LocalPathDistributionComputer(
		worker_i=0, subdeployment=subdep, init_kwargs=init_kwa)
	n_popps = len(dep['popps'])
	# Multi-prefix all-on advertisement so popp failure can be re-routed via
	# alternate (popp, prefix) entries.
	n_prefixes = min(4, max(2, n_popps // 4))
	adv = np.ones((n_popps, n_prefixes))
	return worker, dep, adv


# -------------------- (a) SPEEDUP STUDY -------------------- #

def run_speedup_study():
	"""For each problem size, time warm vs cold on the same scenario set.

	Single-popp failures + nominal as scenario set; |S| = n_popps + 1.
	Repeat each measurement TRIALS times for stability; report median.
	"""
	from stochastic_lp import solve_stochastic_lp, single_popp_scenarios

	# Skip 'decent' locally — it can trip a transient SSL cert error against
	# Gurobi's WLS license server when the model rebuild forces a fresh token
	# request. For cluster runs this is fine. Synthetic 'small' is enough to
	# demonstrate the speedup direction; magnitudes scale with problem size.
	sizes = ['really_friggin_small', 'small']
	TRIALS = 3

	rows = []
	print("\n=== SPEEDUP STUDY ===")
	print(f"{'size':>22}  {'n_popps':>8}  {'|S|':>4}  {'warm (s)':>10}  {'cold (s)':>10}  {'speedup':>8}")
	for size in sizes:
		worker, dep, adv = _build_worker(size, seed=31415)
		n_popps = len(dep['popps'])
		scenarios = single_popp_scenarios(dep, p_any_fail=0.5)

		warm_times = []
		cold_times = []
		for _ in range(TRIALS):
			# Warm
			warm_res = solve_stochastic_lp(worker, adv, scenarios, method='warm')
			warm_times.append(warm_res.wall_time)
			# Cold (worker is rebuilt internally per scenario)
			cold_res = solve_stochastic_lp(worker, adv, scenarios, method='cold')
			cold_times.append(cold_res.wall_time)

		warm_med = float(np.median(warm_times))
		cold_med = float(np.median(cold_times))
		speedup = cold_med / warm_med if warm_med > 0 else float('nan')

		print(f"{size:>22}  {n_popps:>8d}  {len(scenarios):>4d}  "
			  f"{warm_med:>10.3f}  {cold_med:>10.3f}  {speedup:>8.2f}x")

		# Check agreement
		eps_disagree = abs(warm_res.expected_latency - cold_res.expected_latency)
		rows.append({
			'size': size, 'n_popps': n_popps, 'n_scenarios': len(scenarios),
			'warm_seconds_med': warm_med, 'cold_seconds_med': cold_med,
			'speedup_x': speedup,
			'expected_latency_warm': warm_res.expected_latency,
			'expected_latency_cold': cold_res.expected_latency,
			'agreement_eps': eps_disagree,
		})

	csv_path = os.path.join(OUT_DIR, 'stochastic_lp_speedup.csv')
	with open(csv_path, 'w', newline='') as f:
		w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
		w.writeheader()
		w.writerows(rows)
	print(f"  wrote {csv_path}")
	return rows


def plot_speedup(rows):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

	n_popps_arr = np.array([r['n_popps'] for r in rows])
	warm = np.array([r['warm_seconds_med'] for r in rows])
	cold = np.array([r['cold_seconds_med'] for r in rows])
	speedup = np.array([r['speedup_x'] for r in rows])

	# Left: log-log wall time vs n_popps
	ax1.loglog(n_popps_arr, cold, 'o-', label='cold (rebuild model)', color='C3', linewidth=2, markersize=8)
	ax1.loglog(n_popps_arr, warm, 'o-', label='warm (basis reuse)', color='C0', linewidth=2, markersize=8)
	ax1.set_xlabel('# popps')
	ax1.set_ylabel('total wall time (s), |S| = n_popps+1')
	ax1.set_title('Stochastic LP wall time vs problem size')
	ax1.legend()
	ax1.grid(True, which='both', alpha=0.3)
	for x, cw, cc in zip(n_popps_arr, warm, cold):
		ax1.annotate(f"{cw:.2f}s", (x, cw), textcoords='offset points', xytext=(5, -10), fontsize=8, color='C0')
		ax1.annotate(f"{cc:.2f}s", (x, cc), textcoords='offset points', xytext=(5, 7), fontsize=8, color='C3')

	# Right: speedup as a function of problem size
	ax2.semilogx(n_popps_arr, speedup, 'o-', color='C2', linewidth=2, markersize=10)
	ax2.axhline(1, color='gray', linestyle='--', alpha=0.5, label='no speedup')
	ax2.set_xlabel('# popps')
	ax2.set_ylabel('speedup (cold time / warm time)')
	ax2.set_title('Warm vs cold speedup')
	ax2.legend()
	ax2.grid(True, which='both', alpha=0.3)
	for x, s in zip(n_popps_arr, speedup):
		ax2.annotate(f"{s:.2f}×", (x, s), textcoords='offset points', xytext=(5, 5), fontsize=10)

	plt.tight_layout()
	png_path = os.path.join(OUT_DIR, 'stochastic_lp_speedup.png')
	plt.savefig(png_path, dpi=120)
	print(f"  wrote {png_path}")
	plt.close()


# -------------------- (b) CONVERGENCE STUDY -------------------- #

def run_convergence_study():
	"""Fix problem size; sample K scenarios uniformly from S_full, M trials per K.

	Report (mean, std, max abs deviation from L_full) for each K. This tells
	us how many scenarios we need to draw to get a stable estimate of the true
	expected latency.
	"""
	from stochastic_lp import (
		solve_stochastic_lp, single_popp_scenarios, subsample_scenarios,
	)

	worker, dep, adv = _build_worker('small', seed=31415)
	scenarios_full = single_popp_scenarios(dep, p_any_fail=0.5)
	full = solve_stochastic_lp(worker, adv, scenarios_full, method='warm')
	L_full = full.expected_latency
	print(f"\n=== CONVERGENCE STUDY ===")
	print(f"  problem size: 'small', |S_full|={len(scenarios_full)}, L_full={L_full:.4f}")

	rng = np.random.default_rng(31415)
	Ks = [1, 2, 4, 8, 16, min(32, len(scenarios_full)), len(scenarios_full)]
	Ks = sorted(set(Ks))
	M = 30  # trials per K
	rows = []
	print(f"  {'K':>4}  {'mean':>10}  {'std':>10}  {'|mean-Lfull|':>14}  {'max|dev|':>10}")
	for K in Ks:
		objs = []
		for _ in range(M):
			sub = subsample_scenarios(scenarios_full, K, rng)
			r = solve_stochastic_lp(worker, adv, sub, method='warm')
			objs.append(r.expected_latency)
		mean = float(np.mean(objs))
		std = float(np.std(objs))
		bias = abs(mean - L_full)
		max_dev = float(np.max(np.abs(np.array(objs) - L_full)))
		print(f"  {K:>4d}  {mean:>10.4f}  {std:>10.4f}  {bias:>14.4f}  {max_dev:>10.4f}")
		rows.append({
			'K': K, 'M_trials': M, 'mean': mean, 'std': std,
			'bias_from_Lfull': bias, 'max_abs_deviation': max_dev,
			'L_full': L_full,
		})

	csv_path = os.path.join(OUT_DIR, 'stochastic_lp_convergence.csv')
	with open(csv_path, 'w', newline='') as f:
		w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
		w.writeheader()
		w.writerows(rows)
	print(f"  wrote {csv_path}")
	return rows, L_full


def plot_convergence(rows, L_full):
	Ks = np.array([r['K'] for r in rows])
	means = np.array([r['mean'] for r in rows])
	stds = np.array([r['std'] for r in rows])

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

	# Left: mean ± std vs K (linear)
	ax1.errorbar(Ks, means, yerr=stds, fmt='o-', capsize=4, linewidth=2, markersize=8, color='C0', label='estimate (mean ± std over trials)')
	ax1.axhline(L_full, color='C3', linestyle='--', alpha=0.6, label=f'full LP: {L_full:.3f}')
	ax1.set_xlabel('K (sub-sample size)')
	ax1.set_ylabel('expected latency estimate')
	ax1.set_title('Stochastic LP estimate vs sub-sample size K')
	ax1.legend()
	ax1.grid(True, alpha=0.3)

	# Right: std vs K on log-log; compare to 1/sqrt(K) reference
	ax2.loglog(Ks, stds, 'o-', linewidth=2, markersize=8, color='C0', label='measured std')
	# 1/sqrt(K) reference line, anchored at the smallest K
	ref = stds[0] * np.sqrt(Ks[0]) / np.sqrt(Ks)
	ax2.loglog(Ks, ref, ':', linewidth=2, color='gray', label='1/√K reference')
	ax2.set_xlabel('K (sub-sample size)')
	ax2.set_ylabel('std of estimate (over trials)')
	ax2.set_title('Convergence: std vs K')
	ax2.legend()
	ax2.grid(True, which='both', alpha=0.3)

	plt.tight_layout()
	png_path = os.path.join(OUT_DIR, 'stochastic_lp_convergence.png')
	plt.savefig(png_path, dpi=120)
	print(f"  wrote {png_path}")
	plt.close()


def run_scenario_scaling_study():
	"""Hold problem size fixed; vary the number of scenarios |S| we feed in.

	This directly exercises the basis-reuse advantage: with K scenarios, warm
	does K sequential solves with the basis carrying forward; cold pays the
	model build cost K times. Speedup should grow with K.
	"""
	from stochastic_lp import solve_stochastic_lp, single_popp_scenarios

	worker, dep, adv = _build_worker('small', seed=31415)
	scenarios_full = single_popp_scenarios(dep, p_any_fail=0.5)
	n_popps = len(dep['popps'])
	TRIALS = 3

	rows = []
	print(f"\n=== SCENARIO-SCALING STUDY  (size='small', n_popps={n_popps}) ===")
	print(f"  {'|S|':>4}  {'warm (s)':>10}  {'cold (s)':>10}  {'speedup':>8}")
	Ks = [2, 4, 8, 16, len(scenarios_full)]
	Ks = sorted(set(Ks))
	for K in Ks:
		sub = scenarios_full[:K]
		warm_times, cold_times = [], []
		for _ in range(TRIALS):
			w = solve_stochastic_lp(worker, adv, sub, method='warm')
			c = solve_stochastic_lp(worker, adv, sub, method='cold')
			warm_times.append(w.wall_time)
			cold_times.append(c.wall_time)
		warm_med = float(np.median(warm_times))
		cold_med = float(np.median(cold_times))
		speedup = cold_med / warm_med if warm_med > 0 else float('nan')
		print(f"  {K:>4d}  {warm_med:>10.3f}  {cold_med:>10.3f}  {speedup:>8.2f}x")
		rows.append({'K_scenarios': K, 'n_popps': n_popps,
					 'warm_seconds_med': warm_med, 'cold_seconds_med': cold_med,
					 'speedup_x': speedup})
	return rows


def plot_scenario_scaling(rows):
	Ks = np.array([r['K_scenarios'] for r in rows])
	warm = np.array([r['warm_seconds_med'] for r in rows])
	cold = np.array([r['cold_seconds_med'] for r in rows])
	speedup = np.array([r['speedup_x'] for r in rows])

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

	ax1.plot(Ks, cold, 'o-', label='cold (rebuild model)', color='C3', linewidth=2, markersize=8)
	ax1.plot(Ks, warm, 'o-', label='warm (basis reuse)', color='C0', linewidth=2, markersize=8)
	ax1.set_xlabel('|S| (number of scenarios)')
	ax1.set_ylabel('total wall time (s)')
	ax1.set_title("Wall time vs |S|  (problem size 'small', 45 popps)")
	ax1.legend()
	ax1.grid(True, alpha=0.3)

	ax2.plot(Ks, speedup, 'o-', color='C2', linewidth=2, markersize=10)
	ax2.axhline(1, color='gray', linestyle='--', alpha=0.5)
	ax2.set_xlabel('|S|')
	ax2.set_ylabel('speedup (cold / warm)')
	ax2.set_title('Speedup grows with |S|')
	ax2.grid(True, alpha=0.3)
	for x, s in zip(Ks, speedup):
		ax2.annotate(f"{s:.2f}×", (x, s), textcoords='offset points', xytext=(5, 5), fontsize=10)

	plt.tight_layout()
	png_path = os.path.join(OUT_DIR, 'stochastic_lp_scenario_scaling.png')
	plt.savefig(png_path, dpi=120)
	print(f"  wrote {png_path}")
	plt.close()


def run_actual_deployment_study(size='actual-10', s_max=30, seed=1):
	"""Run scenario-scaling + convergence on a real-data deployment.

	Bounded |S| via s_max so we don't sit on the cluster for hours.
	"""
	from stochastic_lp import (
		solve_stochastic_lp, single_popp_scenarios, subsample_scenarios,
	)

	print(f"\n=== ACTUAL-DEPLOYMENT STUDY ({size}, seed={seed}) ===")
	worker, dep, adv = _build_worker(size, seed=seed)
	scenarios_full = single_popp_scenarios(dep, p_any_fail=0.5)
	if len(scenarios_full) > s_max + 1:
		# Keep nominal + s_max single-popp failures (uniform sampled deterministically)
		rng = np.random.default_rng(seed)
		idx = rng.choice(len(scenarios_full) - 1, size=s_max, replace=False) + 1
		scenarios_full = [scenarios_full[0]] + [scenarios_full[int(i)] for i in idx]
		# renormalise
		total = sum(p for _, p in scenarios_full)
		scenarios_full = [(s, p / total) for s, p in scenarios_full]
	n_popps = len(dep['popps'])
	print(f"  n_popps={n_popps}, n_ugs={len(dep['ugs'])}, |S_full|={len(scenarios_full)}")

	# Scenario-scaling speedup
	TRIALS = 2
	scaling_rows = []
	Ks = [2, 4, 8, 16, len(scenarios_full)]
	Ks = sorted(set(K for K in Ks if K <= len(scenarios_full)))
	print(f"  {'|S|':>4}  {'warm (s)':>10}  {'cold (s)':>10}  {'speedup':>8}")
	for K in Ks:
		sub = scenarios_full[:K]
		warm_times, cold_times = [], []
		for _ in range(TRIALS):
			w = solve_stochastic_lp(worker, adv, sub, method='warm')
			c = solve_stochastic_lp(worker, adv, sub, method='cold')
			warm_times.append(w.wall_time)
			cold_times.append(c.wall_time)
		warm_med = float(np.median(warm_times))
		cold_med = float(np.median(cold_times))
		speedup = cold_med / warm_med if warm_med > 0 else float('nan')
		print(f"  {K:>4d}  {warm_med:>10.3f}  {cold_med:>10.3f}  {speedup:>8.2f}x")
		scaling_rows.append({'size': size, 'K_scenarios': K, 'n_popps': n_popps,
							 'warm_seconds_med': warm_med,
							 'cold_seconds_med': cold_med, 'speedup_x': speedup})

	# Convergence on this deployment
	full = solve_stochastic_lp(worker, adv, scenarios_full, method='warm')
	L_full = full.expected_latency
	print(f"\n  convergence: L_full={L_full:.4f}")
	rng = np.random.default_rng(seed)
	M = 12
	conv_Ks = [1, 2, 4, 8, min(16, len(scenarios_full)), len(scenarios_full)]
	conv_Ks = sorted(set(conv_Ks))
	conv_rows = []
	print(f"  {'K':>4}  {'mean':>10}  {'std':>10}  {'|mean-Lf|':>10}")
	for K in conv_Ks:
		objs = []
		for _ in range(M):
			sub = subsample_scenarios(scenarios_full, K, rng)
			r = solve_stochastic_lp(worker, adv, sub, method='warm')
			objs.append(r.expected_latency)
		mean = float(np.mean(objs))
		std = float(np.std(objs))
		bias = abs(mean - L_full)
		print(f"  {K:>4d}  {mean:>10.4f}  {std:>10.4f}  {bias:>10.4f}")
		conv_rows.append({'size': size, 'K': K, 'M_trials': M, 'mean': mean, 'std': std,
						  'bias_from_Lfull': bias, 'L_full': L_full})

	# CSVs
	for stub, rows in [('actual_scaling', scaling_rows), ('actual_convergence', conv_rows)]:
		path = os.path.join(OUT_DIR, f'stochastic_lp_{stub}.csv')
		with open(path, 'w', newline='') as f:
			w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
			w.writeheader()
			w.writerows(rows)
		print(f"  wrote {path}")

	# Plots
	plot_actual_scaling(scaling_rows, size)
	plot_actual_convergence(conv_rows, L_full, size)
	return scaling_rows, conv_rows, L_full


def plot_actual_scaling(rows, size):
	Ks = np.array([r['K_scenarios'] for r in rows])
	warm = np.array([r['warm_seconds_med'] for r in rows])
	cold = np.array([r['cold_seconds_med'] for r in rows])
	speedup = np.array([r['speedup_x'] for r in rows])
	n_popps = rows[0]['n_popps']
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
	ax1.plot(Ks, cold, 'o-', label='cold', color='C3', linewidth=2, markersize=8)
	ax1.plot(Ks, warm, 'o-', label='warm', color='C0', linewidth=2, markersize=8)
	ax1.set_xlabel('|S|'); ax1.set_ylabel('total wall time (s)')
	ax1.set_title(f"Stochastic LP wall on {size}  ({n_popps} popps)")
	ax1.legend(); ax1.grid(True, alpha=0.3)
	ax2.plot(Ks, speedup, 'o-', color='C2', linewidth=2, markersize=10)
	ax2.axhline(1, color='gray', linestyle='--', alpha=0.5)
	ax2.set_xlabel('|S|'); ax2.set_ylabel('speedup (cold/warm)')
	ax2.set_title(f"Speedup on {size}")
	ax2.grid(True, alpha=0.3)
	for x, s in zip(Ks, speedup):
		ax2.annotate(f"{s:.2f}×", (x, s), textcoords='offset points', xytext=(5, 5), fontsize=10)
	plt.tight_layout()
	path = os.path.join(OUT_DIR, f'stochastic_lp_{size}_scaling.png')
	plt.savefig(path, dpi=120)
	print(f"  wrote {path}")
	plt.close()


def plot_actual_convergence(rows, L_full, size):
	Ks = np.array([r['K'] for r in rows])
	means = np.array([r['mean'] for r in rows])
	stds = np.array([r['std'] for r in rows])
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
	ax1.errorbar(Ks, means, yerr=stds, fmt='o-', capsize=4, linewidth=2, markersize=8, color='C0', label='estimate ± std')
	ax1.axhline(L_full, color='C3', linestyle='--', alpha=0.6, label=f'full LP: {L_full:.3f}')
	ax1.set_xlabel('K'); ax1.set_ylabel('expected latency')
	ax1.set_title(f'Convergence on {size}')
	ax1.legend(); ax1.grid(True, alpha=0.3)
	ax2.loglog(Ks, stds, 'o-', linewidth=2, markersize=8, color='C0', label='measured std')
	ref = stds[0] * np.sqrt(Ks[0]) / np.sqrt(Ks)
	ax2.loglog(Ks, ref, ':', linewidth=2, color='gray', label='1/√K reference')
	ax2.set_xlabel('K'); ax2.set_ylabel('std of estimate')
	ax2.set_title(f'Std vs K  ({size})')
	ax2.legend(); ax2.grid(True, which='both', alpha=0.3)
	plt.tight_layout()
	path = os.path.join(OUT_DIR, f'stochastic_lp_{size}_convergence.png')
	plt.savefig(path, dpi=120)
	print(f"  wrote {path}")
	plt.close()


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--synthetic-only', action='store_true',
		help='Only run synthetic-size benchmarks (skip actual-N)')
	parser.add_argument('--actual-size', default='actual-10',
		help='Which actual-N deployment to use (default: actual-10)')
	parser.add_argument('--s-max', type=int, default=30,
		help='Cap |S_full| for actual deployments to bound wall time (default: 30)')
	parser.add_argument('--seed', type=int, default=1,
		help='Deployment seed for actual-N (default: 1)')
	args = parser.parse_args()

	t0 = time.time()
	speedup_rows = run_speedup_study()
	plot_speedup(speedup_rows)

	scaling_rows = run_scenario_scaling_study()
	plot_scenario_scaling(scaling_rows)

	conv_rows, L_full = run_convergence_study()
	plot_convergence(conv_rows, L_full)

	if not args.synthetic_only:
		run_actual_deployment_study(size=args.actual_size, s_max=args.s_max, seed=args.seed)

	print(f"\nTotal wall: {time.time()-t0:.1f}s")
