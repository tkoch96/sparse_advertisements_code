"""Frozen-prefix FAILURE metrics for the paper table (Tom 2026-09-06).

For a strategy's advertisement, evaluate single-popp (or single-site)
failures under the frozen semantic -- the per-(ug, prefix) allocation is
FIXED, BGP fallback within the pinned prefix is the only post-failure
adaptation, nothing is re-optimized -- and report THREE metrics kept
deliberately distinct (the older static_failure_eval priced congestion AND
no-route at NO_ROUTE_LATENCY, conflating them):

  fail_latency_ms   volume-weighted avg latency of ROUTED, UNCONGESTED volume
  fail_frac_cong    fraction of volume landing on an over-capacity popp
  fail_frac_no_route fraction of volume whose pinned prefix has no surviving popp

Pin = the frozen_prefix LP's own allocation for that advertisement (exact
per-(ug, prefix) pairs it returns; no popp->prefix inversion). The pin is
hedged against EVERY single-popp failure by default (the lifted LP makes
that affordable; SCULPTOR_FROZEN_PREFIX_PIN_N_FAIL=k samples k instead,
which was the pre-2026-09-07 behaviour with k=20). Small-deployment A/B
2026-09-07: stride-20 pin left 2.5-2.8% of volume on over-cap popps under
the exhaustive sweep -- failures OUTSIDE the pin's sample landed on popps
the pin had loaded to exactly cap; the exhaustive pin took that to
0.0-0.08% at +0.3-1.3 ms. The measurement sweep is exhaustive unless
SCULPTOR_FROZEN_PREFIX_EVAL_N_FAIL caps it.

`reactive_optimal_metrics` is the upper ANCHOR: the same three metrics for
a full-availability advertisement (one-per-peering by default) with the
assignment RE-OPTIMIZED after each failure. It bounds any frozen scheme and,
unlike frozen one-per-peering (single popp per prefix -> no backup ->
stranded users), never strands. It is LP-per-failure, so it samples
failures (SCULPTOR_FROZEN_PREFIX_ANCHOR_N_FAIL, default 50; 0 = all).
"""
import os

import numpy as np

from helpers.constants import NO_ROUTE_LATENCY
from helpers.helpers import threshold_a


def _scenarios(sas, which, cap):
	n_popps = sas.n_popps
	if which == 'popps':
		scen = [[i] for i in range(n_popps)]
	elif which == 'pops':
		pop_to_inds = {}
		for pi, (pop, _) in enumerate(sas.popps):
			pop_to_inds.setdefault(pop, []).append(pi)
		scen = list(pop_to_inds.values())
	else:
		raise ValueError("which must be 'popps' or 'pops', got {!r}".format(which))
	if cap and 0 < cap < len(scen):
		step = max(1, len(scen) // cap)
		scen = scen[::step][:cap]
	return scen


def pin_pairs(sas, adv, routed_through_ingress, pin_kill_popps=None):
	"""[(ug, prefix_i, vol)] -- the frozen_prefix LP's allocation for adv."""
	from core.frozen_prefix import solve_lp_frozen_prefix, default_kill_popps
	from core.objective_registry import lp_kwargs_for
	levers = lp_kwargs_for('frozen_prefix')   # the objective's own tunables
	if pin_kill_popps is None:
		pin_n = int(os.environ.get('SCULPTOR_FROZEN_PREFIX_PIN_N_FAIL', '0'))
		pin_kill_popps = (default_kill_popps(sas.n_popps, pin_n) if pin_n > 0
						  else list(range(sas.n_popps)))
	levers = {k: v for k, v in levers.items() if k not in ('frozen_n_fail',)}
	# one pin per strategy per sim: let the exhaustive model finish
	levers.setdefault('frozen_time_limit',
					  float(os.environ.get('SCULPTOR_FROZEN_PREFIX_PIN_TIME_LIMIT', '1800')))
	ret = solve_lp_frozen_prefix(sas, routed_through_ingress, 'frozen_prefix',
								 adv=adv, frozen_kill_popps=list(pin_kill_popps),
								 **levers)
	if not ret.get('solved'):
		raise ValueError('frozen_prefix pin LP unsolved')
	return ret.get('frozen_prefix_pairs') or []


def frozen_failure_metrics(sas, adv, which='popps', pairs=None,
						   routed_through_ingress=None, pin_kill_popps=None,
						   eval_n_fail=None, use_gti=False):
	"""use_gti=True forces the per-failure calculate_ground_truth_ingress
	path (the pre-2026-09-09 implementation, exact but O(n_popps) Python
	ingress recomputations: hours at size 32). Default: single-popp
	failures use core.frozen_prefix.frozen_fallbacks -- the same exact
	BGP-fallback rule, vectorized once per prefix column. Site ('pops')
	failures still use the gti path."""
	adv = threshold_a(np.asarray(adv, dtype=float))
	popp_to_ind = sas.popp_to_ind
	ug_perfs = sas.whole_deployment_ug_perfs
	caps = np.asarray(sas.link_capacities_arr, dtype=float).flatten()
	n_popps = sas.n_popps
	total_vol = float(sum(sas.whole_deployment_ug_to_vol.values())) or 1.0
	if eval_n_fail is None:
		eval_n_fail = int(os.environ.get('SCULPTOR_FROZEN_PREFIX_EVAL_N_FAIL', '0'))

	if routed_through_ingress is None:
		routed_through_ingress, _ = sas.calculate_ground_truth_ingress(adv)
	if pairs is None:
		pairs = pin_pairs(sas, adv, routed_through_ingress, pin_kill_popps)

	# entry arrays (one per (ug, prefix) share)
	e_ug = [p[0] for p in pairs]
	e_prefix = np.asarray([int(p[1]) for p in pairs], dtype=int)
	e_vol = np.asarray([float(p[2]) for p in pairs], dtype=float)
	n_e = len(e_ug)
	e_popp = np.full(n_e, -1, dtype=int)
	e_lat = np.zeros(n_e)
	for i in range(n_e):
		popp_tuple = routed_through_ingress.get(e_prefix[i], {}).get(e_ug[i])
		pi = popp_to_ind.get(popp_tuple) if popp_tuple is not None else None
		if pi is None:
			continue
		lat = ug_perfs.get(e_ug[i], {}).get(popp_tuple, NO_ROUTE_LATENCY)
		if lat >= NO_ROUTE_LATENCY:
			continue
		e_popp[i] = pi
		e_lat[i] = float(lat)
	unroutable_vol = max(0.0, total_vol - float(e_vol.sum()))

	def _score(w_popp, w_lat):
		live = w_popp >= 0
		loads = np.bincount(w_popp[live], weights=e_vol[live], minlength=n_popps)
		cong_popp = loads > caps + 1e-9
		on_cong = live & cong_popp[np.clip(w_popp, 0, None)]
		good = live & ~on_cong
		cong_vol = float(e_vol[on_cong].sum())
		nr_vol = float(e_vol[~live].sum()) + unroutable_vol
		gv = float(e_vol[good].sum())
		lat = float(np.sum(e_vol[good] * w_lat[good]) / gv) if gv > 0 else float('nan')
		return lat, cong_vol / total_vol, nr_vol / total_vol

	steady_lat, steady_cong, steady_nr = _score(e_popp, e_lat)

	# Vectorized single-popp path: every pair has exactly one fallback
	# (the best remaining ingress in its prefix when its winner dies), so
	# the whole sweep is a per-prefix top-2 computation plus bincounts.
	fb = None
	if which == 'popps' and not use_gti and n_e:
		from core.frozen_prefix import frozen_fallbacks
		ug_to_ind = sas.whole_deployment_ug_to_ind
		e_ugi = np.asarray([ug_to_ind[u] for u in e_ug], dtype=int)
		fb = frozen_fallbacks(sas, adv, e_prefix, e_ugi, e_popp)
		fb_lat = np.zeros(n_e)
		for i in np.where(fb >= 0)[0]:
			lat = ug_perfs.get(e_ug[i], {}).get(sas.popps[fb[i]], NO_ROUTE_LATENCY)
			if lat >= NO_ROUTE_LATENCY:
				fb[i] = -1
			else:
				fb_lat[i] = float(lat)
		fb[e_popp < 0] = -1          # unrouted entries stay unrouted

	lats, congs, nrs = [], [], []
	for killed in _scenarios(sas, which, eval_n_fail):
		killed_set = set(int(k) for k in killed)
		w_popp = e_popp.copy()
		w_lat = e_lat.copy()
		affected = np.where(np.isin(e_popp, list(killed_set)))[0]
		if len(affected) and fb is not None:
			w_popp[affected] = fb[affected]
			w_lat[affected] = np.where(fb[affected] >= 0, fb_lat[affected], 0.0)
		elif len(affected):
			a_fail = adv.copy()
			for k in killed_set:
				a_fail[k, :] = 0
			fail_rti, _ = sas.calculate_ground_truth_ingress(a_fail)
			for i in affected:
				popp_tuple = fail_rti.get(e_prefix[i], {}).get(e_ug[i])
				pi = popp_to_ind.get(popp_tuple) if popp_tuple is not None else None
				if pi is None or pi in killed_set:
					w_popp[i] = -1; w_lat[i] = 0.0
					continue
				lat = ug_perfs.get(e_ug[i], {}).get(popp_tuple, NO_ROUTE_LATENCY)
				if lat >= NO_ROUTE_LATENCY:
					w_popp[i] = -1; w_lat[i] = 0.0
				else:
					w_popp[i] = pi; w_lat[i] = float(lat)
		l, c, nr = _score(w_popp, w_lat)
		lats.append(l); congs.append(c); nrs.append(nr)

	return {
		'steady_latency_ms': steady_lat,
		'steady_frac_cong': steady_cong,
		'steady_frac_no_route': steady_nr,
		'fail_latency_ms': float(np.nanmean(lats)) if lats else float('nan'),
		'fail_frac_cong': float(np.mean(congs)) if congs else 0.0,
		'fail_frac_no_route': float(np.mean(nrs)) if nrs else 0.0,
		'worst_frac_cong': float(np.max(congs)) if congs else 0.0,
		'worst_frac_no_route': float(np.max(nrs)) if nrs else 0.0,
		'n_failures': len(lats),
		'n_pairs': n_e,
	}


def reactive_optimal_metrics(sas, adv=None, which='popps', n_fail=None):
	"""Upper anchor: assignment RE-OPTIMIZED (avg_latency LP) after each
	failure on a full-availability advertisement (one-per-peering by
	default). Same three metrics as frozen_failure_metrics."""
	from core.solve_lp_assignment import solve_generic_lp_with_failure_catch
	n_popps = sas.n_popps
	if adv is None:
		adv = np.eye(n_popps)
	adv = threshold_a(np.asarray(adv, dtype=float))
	if n_fail is None:
		n_fail = int(os.environ.get('SCULPTOR_FROZEN_PREFIX_ANCHOR_N_FAIL', '50'))
	vols = np.asarray([sas.whole_deployment_ug_to_vol[u]
					   for u in sas.whole_deployment_ugs], dtype=float)
	total = float(vols.sum()) or 1.0

	def _m(ret):
		lats = np.asarray(ret['lats_by_ug'], dtype=float)
		nr = lats >= NO_ROUTE_LATENCY - 1e-9
		routed = ~nr
		lat = (float(np.average(lats[routed], weights=vols[routed]))
			   if routed.any() else float('nan'))
		return (lat, float(ret.get('fraction_congested_volume', 0.0) or 0.0),
				float(vols[nr].sum() / total))

	rti, _ = sas.calculate_ground_truth_ingress(adv)
	st = solve_generic_lp_with_failure_catch(sas, rti, 'avg_latency')
	steady = _m(st) if st.get('solved') else (float('nan'), float('nan'), float('nan'))
	lats, congs, nrs = [], [], []
	for killed in _scenarios(sas, which, n_fail):
		a = adv.copy()
		for k in killed:
			a[int(k), :] = 0
		rf, _ = sas.calculate_ground_truth_ingress(a)
		ret = solve_generic_lp_with_failure_catch(sas, rf, 'avg_latency')
		if not ret.get('solved'):
			lats.append(float('nan')); congs.append(1.0); nrs.append(1.0)
			continue
		l, c, nr = _m(ret)
		lats.append(l); congs.append(c); nrs.append(nr)
	return {
		'steady_latency_ms': steady[0],
		'steady_frac_cong': steady[1],
		'steady_frac_no_route': steady[2],
		'fail_latency_ms': float(np.nanmean(lats)) if lats else float('nan'),
		'fail_frac_cong': float(np.mean(congs)) if congs else 0.0,
		'fail_frac_no_route': float(np.mean(nrs)) if nrs else 0.0,
		'worst_frac_cong': float(np.max(congs)) if congs else 0.0,
		'worst_frac_no_route': float(np.max(nrs)) if nrs else 0.0,
		'n_failures': len(lats),
	}
