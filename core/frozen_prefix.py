"""frozen_prefix objective: one JOINT LP over per-(user, prefix) volume
variables, scored across the normal scenario plus a set of single-popp
failure scenarios (Tom 2026-09-05, SCRAPBOOK_FROZEN_PREFIX.md).

The decision variable is w(ug, prefix) -- strictly which PREFIX each user's
volume is steered to, never which popp. Routes are coefficient data: for
the normal case the caller's routed_through_ingress (an MC realization
during training, ground truth during eval) maps (prefix, ug) -> popp; for
each failure scenario the map is recomputed with the failed popp's adv row
zeroed (BGP ingress-priority fallback via calculate_ground_truth_ingress,
same precedent as solve_lp_popp_failure_congestion / site_failure).

Because ONE w is shared by every scenario, the "freeze user->prefix after
failure" semantic is structural -- there is nothing to pin or invert. w is
jointly optimized: a single static allocation hedged across normal +
failure operation (the no-reactive-DNS story).

Objective (minimize; returned negated per the repo's benefit convention):

  normal_term + gamma * mean_over_failures(failure_term)

  term_s = sum_live w * lat_s / V
         + P_nr * (no-route volume in s) / V
         + P_c  * (over-capacity excess volume in s) / V

Soft throughout: no NO_ROUTE_LATENCY-scale coefficients anywhere in the
scalar (gradient-stability rule); P_nr > P_c, both moderate. Congestion is
priced on EXCESS volume above capacity (linear via per-scenario overflow
variables), not the volume-on-congested-links paper metric -- the eval
suite reports the latter post-hoc.

Levers are the objective's own (core/objective_registry.py frozen_prefix
lp_defaults -> Generic_Objective.lp_kwargs -> every LP call, and stamped to
workers per flush); the env names below are per-run OVERRIDES resolved
there. Direct callers may still pass kwargs (kwargs > env > default):
  frozen_kill_popps / (deterministic stride)  popp indices to fail
  frozen_gamma  / SCULPTOR_FROZEN_PREFIX_GAMMA               (1.0)
  frozen_n_fail / SCULPTOR_FROZEN_PREFIX_N_FAIL              (20)
  frozen_no_route_penalty / SCULPTOR_FROZEN_PREFIX_NO_ROUTE_PENALTY  (50.0)
  frozen_congestion_penalty / SCULPTOR_FROZEN_PREFIX_CONGESTION_PENALTY (25.0)

The per-iteration explore/exploit kill list is chosen driver-side
(FrozenPrefixObjective in core/generic_objective.py) and ships to workers
via the flush's lp_kwargs_extra; when absent (driver-side eval calls) the
deterministic stride default keeps evaluation stable and reproducible.
"""
import os
import time

import numpy as np
from scipy.sparse import csr_matrix

from helpers.constants import NO_ROUTE_LATENCY
from helpers.helpers import threshold_a
import core.gpshim as gp


def _clear_highs_model(model):
	"""Best-effort release of a gpshim model's native solver resources. On
	the HiGHS backend the model wraps a highspy.Highs() (`_h`); clearing it
	frees the native model and detaches from the global scheduler. No-op /
	harmless on the gurobi backend or if the facade changes."""
	try:
		h = getattr(model, '_h', None)
		if h is not None:
			h.clear()
	except Exception:
		pass


def _knob(kwargs, key, env, default, cast=float):
	v = kwargs.get(key)
	if v is not None:
		return cast(v)
	return cast(os.environ.get(env, default))


def default_kill_popps(n_popps, n_fail):
	"""Deterministic stride sample over ALL popp indices (advertised or
	not -- killing an unadvertised popp is a no-op scenario, but keying off
	the adv would make the default set drift between the two halves of a
	finite-difference probe pair)."""
	if n_fail >= n_popps:
		return list(range(n_popps))
	step = max(1, n_popps // n_fail)
	return list(range(n_popps))[::step][:n_fail]


def solve_lp_frozen_prefix(sas, routed_through_ingress, obj, **kwargs):
	# The generic-LP dispatcher wraps the registered-function call in
	# `except KeyError: pass` (solve_lp_assignment.py) -- so a KeyError
	# raised INSIDE here is silently misrouted to solve_generic_lp, which
	# then dies with "Objective frozen_prefix not implemented". Convert any
	# unexpected internal error into a loud RuntimeError (not swallowed) so
	# it surfaces with a real traceback instead of that misleading message.
	try:
		return _solve_lp_frozen_prefix_impl(sas, routed_through_ingress, obj, **kwargs)
	except KeyError as e:
		import traceback
		traceback.print_exc()
		raise RuntimeError('frozen_prefix LP raised KeyError {!r} (would have '
						   'been swallowed by the dispatcher)'.format(e))


def _solve_lp_frozen_prefix_impl(sas, routed_through_ingress, obj, **kwargs):
	adv = kwargs.get('adv')
	gamma = _knob(kwargs, 'frozen_gamma', 'SCULPTOR_FROZEN_PREFIX_GAMMA', 1.0)
	n_fail = _knob(kwargs, 'frozen_n_fail', 'SCULPTOR_FROZEN_PREFIX_N_FAIL', 20, int)
	p_nr = _knob(kwargs, 'frozen_no_route_penalty',
				 'SCULPTOR_FROZEN_PREFIX_NO_ROUTE_PENALTY', 50.0)
	p_c = _knob(kwargs, 'frozen_congestion_penalty',
				'SCULPTOR_FROZEN_PREFIX_CONGESTION_PENALTY', 25.0)
	# Latency weight in the OBJECTIVE SCALAR only (Tom 2026-09-06): scaling
	# latency down by 10x while holding P_nr/P_c is the same argmin as
	# penalties x10, but keeps the penalties readable in ms-equivalents and
	# the scalar small (no gradient blow-up). Reported *_lat diagnostics stay
	# in raw ms.
	lat_scale = _knob(kwargs, 'frozen_lat_scale',
					  'SCULPTOR_FROZEN_PREFIX_LAT_SCALE', 1.0)
	# Capacity headroom INSIDE the LP (Tom 2026-09-06 A/B finding): the LP
	# prices congestion on EXCESS volume, but the paper metric flags ALL
	# volume on any popp over cap -- so loading a popp to exactly its cap is
	# free to the LP and maximally fragile under failure (every A/B arm put
	# 70.0 on a 70.0-cap popp). Solving against caps*headroom leaves slack
	# for displaced traffic. 1.0 = off.
	cap_headroom = _knob(kwargs, 'frozen_cap_headroom',
						 'SCULPTOR_FROZEN_PREFIX_CAP_HEADROOM', 1.0)

	from core.solve_lp_assignment import obj_round

	n_popps = sas.n_popps
	popp_to_ind = sas.popp_to_ind
	ug_perfs = sas.whole_deployment_ug_perfs
	ug_to_vol = sas.whole_deployment_ug_to_vol
	ug_to_ind = sas.whole_deployment_ug_to_ind
	n_ug = sas.whole_deployment_n_ug
	caps = np.asarray(sas.link_capacities_arr, dtype=float).flatten()
	total_vol = float(sum(ug_to_vol.values())) or 1.0

	kill_popps = kwargs.get('frozen_kill_popps')
	if kill_popps is None:
		kill_popps = default_kill_popps(n_popps, n_fail)
	kill_popps = [int(k) for k in kill_popps]
	K = len(kill_popps)
	weights = [1.0] + ([gamma / K] * K if K else [])

	# ---- pairs: every (ug, prefix) routable in the NORMAL scenario. A pair
	# unroutable normally stays unroutable under failure (failures only
	# remove options), so these are all the columns that can carry volume.
	pair_ug = []       # ug tuple per pair
	pair_ugi = []      # whole-deployment ug index per pair
	pair_prefix = []   # prefix index per pair
	base_winner = []   # normal-scenario popp index per pair
	base_lat = []      # normal-scenario latency per pair
	for prefix_i, ug_to_popp in sorted(routed_through_ingress.items()):
		for ug, popp_tuple in ug_to_popp.items():
			poppi = popp_to_ind.get(popp_tuple)
			if poppi is None:
				continue
			lat = ug_perfs.get(ug, {}).get(popp_tuple)
			if lat is None or lat >= NO_ROUTE_LATENCY:
				continue
			pair_ug.append(ug)
			pair_ugi.append(ug_to_ind[ug])
			pair_prefix.append(prefix_i)
			base_winner.append(poppi)
			base_lat.append(float(lat))
	n_pairs = len(pair_ug)

	routable_ugis = set(pair_ugi)
	unroutable_vol = float(sum(v for ug, v in ug_to_vol.items()
							   if ug_to_ind[ug] not in routable_ugis))
	# Users with no route on any prefix are no-route in EVERY scenario:
	# constant bounded penalty (never a marker-scale scalar).
	const_term = (1.0 + gamma) * p_nr * unroutable_vol / total_vol

	lats_by_ug_arr = np.zeros(n_ug)
	for ug, vol in ug_to_vol.items():
		if ug_to_ind[ug] not in routable_ugis:
			lats_by_ug_arr[ug_to_ind[ug]] = NO_ROUTE_LATENCY

	if n_pairs == 0:
		return {
			'objective': obj_round(-const_term),
			'solved': True,
			'paths_by_ug': {},
			'lats_by_ug': lats_by_ug_arr,
			'available_paths': [],
			'vols_by_poppi': {poppi: 0 for poppi in range(n_popps)},
			'fraction_congested_volume': 0.0,
			'frozen_prefix_n_fail': K,
			'frozen_prefix_unroutable_frac': unroutable_vol / total_vol,
			'frozen_prefix_pairs': [],
		}

	pair_ugi = np.asarray(pair_ugi, dtype=int)
	base_winner = np.asarray(base_winner, dtype=int)
	base_lat = np.asarray(base_lat, dtype=float)

	# ---- per-failure-scenario winners: only pairs whose NORMAL winner is
	# the killed popp change; their fallback comes from ground-truth
	# ingress on the row-zeroed adv (per-prefix-column cache makes the
	# repeated calls cheap). -1 marks a dead pair (no surviving popp on
	# that user's prefix).
	a_thresh = threshold_a(np.asarray(adv, dtype=float)) if adv is not None else None
	scen_winner = [base_winner]
	scen_lat = [base_lat]
	for k in kill_popps:
		w_s = base_winner.copy()
		l_s = base_lat.copy()
		affected = np.where(base_winner == k)[0]
		if len(affected) and a_thresh is not None:
			a_fail = a_thresh.copy()
			a_fail[k, :] = 0
			fail_rti, _ = sas.calculate_ground_truth_ingress(a_fail)
			for i in affected:
				popp_tuple = fail_rti.get(pair_prefix[i], {}).get(pair_ug[i])
				poppi = popp_to_ind.get(popp_tuple) if popp_tuple is not None else None
				if poppi is None or poppi == k:
					w_s[i] = -1
					l_s[i] = 0.0
				else:
					w_s[i] = poppi
					l_s[i] = float(ug_perfs[pair_ug[i]].get(popp_tuple, NO_ROUTE_LATENCY))
					if l_s[i] >= NO_ROUTE_LATENCY:
						w_s[i] = -1
						l_s[i] = 0.0
		elif len(affected):
			# no adv context: cannot compute fallback -> treat as dead
			w_s[affected] = -1
			l_s[affected] = 0.0
		scen_winner.append(w_s)
		scen_lat.append(l_s)

	n_scen = 1 + K

	# ---- objective coefficients on x: sum over scenarios of
	# weight_s * (lat if live else P_nr) / V
	c_x = np.zeros(n_pairs)
	for s in range(n_scen):
		live = scen_winner[s] >= 0
		c_x += weights[s] * np.where(live, scen_lat[s] * lat_scale, p_nr) / total_vol

	# ---- constraints
	# volume conservation: one row per routable ug
	cons_rows = pair_ugi
	cons_A = csr_matrix((np.ones(n_pairs), (cons_rows, np.arange(n_pairs))),
						shape=(n_ug, n_pairs))
	cons_b = np.zeros(n_ug)
	for ug, vol in ug_to_vol.items():
		if ug_to_ind[ug] in routable_ugis:
			cons_b[ug_to_ind[ug]] = vol
	keep = np.asarray(sorted(routable_ugis), dtype=int)
	cons_A = cons_A[keep, :]
	cons_b = cons_b[keep]

	# per-(scenario, popp) load rows: A x - o <= caps (soft congestion)
	load_rows, load_cols = [], []
	for s in range(n_scen):
		live = np.where(scen_winner[s] >= 0)[0]
		load_rows.append(s * n_popps + scen_winner[s][live])
		load_cols.append(live)
	load_rows = np.concatenate(load_rows)
	load_cols = np.concatenate(load_cols)
	load_A = csr_matrix((np.ones(len(load_rows)), (load_rows, load_cols)),
						shape=(n_scen * n_popps, n_pairs))
	caps_tiled = np.tile(caps * cap_headroom, n_scen)

	c_o = np.repeat(np.asarray(weights) * p_c / total_vol, n_popps)

	# Single stacked variable vector z = [x ; o] -- the gpshim facade
	# supports csr @ MVar (in)equalities but not MVar-expression algebra
	# like `A@x - o`, so the overflow identity block lives in the matrix.
	from scipy.sparse import hstack as sp_hstack, identity as sp_identity
	n_o = n_scen * n_popps
	zeros_cons = csr_matrix((cons_A.shape[0], n_o))
	cons_A_z = sp_hstack([cons_A, zeros_cons], format='csr')
	load_A_z = sp_hstack([load_A, -sp_identity(n_o, format='csr')], format='csr')
	c_z = np.concatenate([c_x, c_o])

	ts = time.time()
	model = gp.Model()
	model.Params.LogToConsole = 0
	model.Params.TimeLimit = 30.0
	# Single-threaded: this joint model (up to n_pairs + n_scen*n_popps vars)
	# is much larger than the other LPs, and engaging HiGHS's GLOBAL parallel
	# scheduler leaves process-wide solver state that falsely reports a later,
	# genuinely-feasible LP as infeasible (confirmed 2026-09-05: identical adv
	# solves clean in a fresh process, infeasible after a frozen solve in the
	# same one). Threads=1 keeps the solve off the shared scheduler.
	model.Params.Threads = 1
	z = model.addMVar(n_pairs + n_o, name='vol_ug_prefix_plus_overflow', lb=0)
	model.addConstr(cons_A_z @ z == cons_b)
	model.addConstr(load_A_z @ z <= caps_tiled)
	model.setObjective(c_z @ z)
	model.optimize()

	if model.status != 2:
		_clear_highs_model(model)
		return {'solved': False}

	zv = np.asarray(z.X).flatten()
	raw_objval = float(model.objVal)
	xv = zv[:n_pairs]
	ov = zv[n_pairs:]
	# Release the native HiGHS resources for this large model promptly rather
	# than waiting on Python GC (belt-and-suspenders with Threads=1 above).
	_clear_highs_model(model)

	# ---- standard-contract reporting from the NORMAL scenario
	paths_by_ug = {}
	vols_by_poppi = {poppi: 0.0 for poppi in range(n_popps)}
	lat_sum_by_ugi = np.zeros(n_ug)
	vol_sum_by_ugi = np.zeros(n_ug)
	for i in range(n_pairs):
		v = xv[i]
		if v <= 0:
			continue
		ugi = pair_ugi[i]
		poppi = base_winner[i]
		vols_by_poppi[poppi] += v
		lat_sum_by_ugi[ugi] += v * base_lat[i]
		vol_sum_by_ugi[ugi] += v
		vol_pct = v / ug_to_vol[pair_ug[i]]
		paths_by_ug.setdefault(int(ugi), []).append((int(poppi), vol_pct))
	routed = vol_sum_by_ugi > 0
	lats_by_ug_arr[routed] = lat_sum_by_ugi[routed] / vol_sum_by_ugi[routed]

	normal_loads = np.zeros(n_popps)
	for poppi, v in vols_by_poppi.items():
		normal_loads[poppi] = v
	congested_popps = normal_loads > caps + 1e-9
	cong_vol = float(sum(v for poppi, v in vols_by_poppi.items()
						 if congested_popps[poppi]))
	fraction_congested_volume = cong_vol / total_vol

	# per-scenario diagnostics (vectorized off the solved x)
	fail_lat, fail_nr, fail_ovf = [], [], []
	for s in range(1, n_scen):
		live = scen_winner[s] >= 0
		lat_mass = float(np.sum(xv[live] * scen_lat[s][live]))
		live_vol = float(np.sum(xv[live]))
		nr_vol = float(np.sum(xv[~live])) + unroutable_vol
		fail_lat.append(lat_mass / live_vol if live_vol > 0 else 0.0)
		fail_nr.append(nr_vol / total_vol)
		fail_ovf.append(float(np.sum(ov[s * n_popps:(s + 1) * n_popps])) / total_vol)

	obj_val = raw_objval + const_term

	ret = {
		'objective': obj_round(-obj_val),
		'raw_solution': xv,
		'paths_by_ug': paths_by_ug,
		'lats_by_ug': lats_by_ug_arr,
		'available_paths': [(pair_ug[i], int(base_winner[i])) for i in range(n_pairs)],
		'solved': model.status,
		'vols_by_poppi': {poppi: round(v / float(caps[poppi]), 2)
						  for poppi, v in vols_by_poppi.items()},
		'fraction_congested_volume': fraction_congested_volume,
		'frozen_prefix_n_fail': K,
		'frozen_prefix_kill_popps': kill_popps,
		'frozen_prefix_gamma': gamma,
		'frozen_prefix_no_route_penalty': p_nr,
		'frozen_prefix_congestion_penalty': p_c,
		'frozen_prefix_lat_scale': lat_scale,
		'frozen_prefix_cap_headroom': cap_headroom,
		'frozen_prefix_normal_lat': (float(np.sum(xv * base_lat)) /
									 max(float(np.sum(xv)), 1e-9)),
		'frozen_prefix_fail_lat_mean': float(np.mean(fail_lat)) if fail_lat else 0.0,
		'frozen_prefix_fail_no_route_frac': float(np.mean(fail_nr)) if fail_nr else 0.0,
		'frozen_prefix_fail_overflow_frac': float(np.mean(fail_ovf)) if fail_ovf else 0.0,
		'frozen_prefix_normal_overflow_frac': float(np.sum(ov[:n_popps])) / total_vol,
		'frozen_prefix_unroutable_frac': unroutable_vol / total_vol,
		'frozen_prefix_lp_secs': time.time() - ts,
		# the frozen allocation itself, per (ug, prefix) with volume > 0 --
		# the eval suite pins on exactly this (no lossy popp->prefix
		# inversion; core/frozen_prefix_eval.py)
		'frozen_prefix_pairs': [(pair_ug[i], int(pair_prefix[i]), float(xv[i]))
								for i in np.where(xv > 1e-12)[0]],
	}
	return ret
