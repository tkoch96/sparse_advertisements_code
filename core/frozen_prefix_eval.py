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
	from core.frozen_prefix import solve_lp_frozen_prefix, default_kill_popps, site_groups
	from core.objective_registry import lp_kwargs_for
	levers = lp_kwargs_for('frozen_prefix')   # the objective's own tunables
	if pin_kill_popps is None:
		pin_n = int(os.environ.get('SCULPTOR_FROZEN_PREFIX_PIN_N_FAIL', '0'))
		pin_kill_popps = (default_kill_popps(sas.n_popps, pin_n) if pin_n > 0
						  else list(range(sas.n_popps)))
		# an objective trained against site failures is pinned against
		# EVERY site failure too (Tom 2026-09-11)
		if float(levers.get('frozen_site_fail_frac', 0.0) or 0.0) > 0:
			pin_kill_popps = list(pin_kill_popps) + site_groups(sas)
	levers = {k: v for k, v in levers.items() if k not in ('frozen_n_fail',)}
	# one pin per strategy per sim: let the exhaustive model finish (the
	# registry's frozen_time_limit is the TRAINING-probe cap, not this)
	levers['frozen_time_limit'] = float(os.environ.get('SCULPTOR_FROZEN_PREFIX_PIN_TIME_LIMIT', '1800'))
	ret = solve_lp_frozen_prefix(sas, routed_through_ingress, 'frozen_prefix',
								 adv=adv, frozen_kill_popps=list(pin_kill_popps),
								 **levers)
	if not ret.get('solved'):
		raise ValueError('frozen_prefix pin LP unsolved')
	return ret.get('frozen_prefix_pairs') or []


def frozen_failure_metrics(sas, adv, which='popps', pairs=None,
						   routed_through_ingress=None, pin_kill_popps=None,
						   eval_n_fail=None, use_gti=False, steady_only=False):
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

	def _score(w_popp, w_lat, subset=None, masks=False):
		"""(latency of routed non-congested volume, congested fraction,
		no-route fraction). subset: index array restricting the accounting to
		those pairs (the 'affected' view, Tom 2026-09-11: the users whose
		pinned prefix was on the failed link/site); fractions are then of the
		subset's volume and no-route excludes the never-routable volume."""
		live = w_popp >= 0
		loads = np.bincount(w_popp[live], weights=e_vol[live], minlength=n_popps)
		cong_popp = loads > caps + 1e-9
		on_cong = live & cong_popp[np.clip(w_popp, 0, None)]
		good = live & ~on_cong
		if subset is not None:
			m = np.zeros(n_e, dtype=bool); m[subset] = True
			live, on_cong, good = live & m, on_cong & m, good & m
			denom = float(e_vol[m].sum()) or 1.0
			nr_vol = float(e_vol[m & ~live].sum())
		else:
			denom = total_vol
			nr_vol = float(e_vol[~live].sum()) + unroutable_vol
		cong_vol = float(e_vol[on_cong].sum())
		gv = float(e_vol[good].sum())
		lat = float(np.sum(e_vol[good] * w_lat[good]) / gv) if gv > 0 else float('nan')
		if masks:
			return lat, cong_vol / denom, nr_vol / denom, good, live
		return lat, cong_vol / denom, nr_vol / denom

	steady_lat, steady_cong, steady_nr = _score(e_popp, e_lat)
	if steady_only:
		# reactive_optimal_metrics prices one re-optimized scenario this way;
		# the entry arrays let it score the affected-users subset too.
		_, _, _, good, live = _score(e_popp, e_lat, masks=True)
		return {'steady_latency_ms': steady_lat, 'steady_frac_cong': steady_cong,
				'steady_frac_no_route': steady_nr, 'n_pairs': n_e,
				'entries': (e_ug, e_popp, e_lat, e_vol, good, live)}

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
	alats, acongs, anrs = [], [], []      # affected-users view
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
		if len(affected):
			al, ac, anr = _score(w_popp, w_lat, subset=affected)
			alats.append(al); acongs.append(ac); anrs.append(anr)

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
		# affected users only (their pinned prefix was on the failed element)
		'fail_affected_latency_ms': float(np.nanmean(alats)) if alats else float('nan'),
		'fail_affected_frac_cong': float(np.mean(acongs)) if acongs else 0.0,
		'fail_affected_frac_no_route': float(np.mean(anrs)) if anrs else 0.0,
	}


def reactive_optimal_metrics(sas, adv=None, which='popps', n_fail=None):
	"""Upper anchor: the user->prefix assignment RE-OPTIMIZED after each
	failure on a full-availability advertisement (one-per-peering by
	default). Same three metrics, same accounting as frozen_failure_metrics:
	every scenario's surviving advertisement is pinned by the frozen LP with
	an EMPTY kill set (no failure hedging -- the assignment is chosen for
	that scenario alone) and scored with _score (latency of the routed
	non-congested volume; a link over capacity congests all of its volume;
	no-route = the unroutable volume). The pre-2026-09-11 version priced
	scenarios with the avg_latency failure-catch LP, whose fallback marks
	congested users with the NO_ROUTE sentinel -- so congested volume was
	counted as stranded too (site failures: 14% congested AND 14% no-route
	for one-per-peering at size 32, next to 0.0% / 3.9% on its frozen row)."""
	n_popps = sas.n_popps
	if adv is None:
		adv = np.eye(n_popps)
	adv = threshold_a(np.asarray(adv, dtype=float))
	if n_fail is None:
		n_fail = int(os.environ.get('SCULPTOR_FROZEN_PREFIX_ANCHOR_N_FAIL', '50'))

	def _steady(a):
		if a.sum() == 0:
			return float('nan'), 1.0, 1.0, None
		m = frozen_failure_metrics(sas, a, pin_kill_popps=[], steady_only=True)
		return m['steady_latency_ms'], m['steady_frac_cong'], m['steady_frac_no_route'], m['entries']

	def _affected(base_entries, killed_set, entries):
		# The users whose STEADY-STATE pinned ingress was on the failed element
		# (the same population frozen_failure_metrics' affected view scores),
		# weighted by that volume; scored on their RE-OPTIMIZED entries:
		# latency = volume-weighted mean of their routed non-congested volume,
		# congested / no-route = fractions of their volume on over-capacity
		# links / left unrouted (unroutable users count as no-route).
		b_ug, b_popp, _, b_vol, _, _ = base_entries
		w = {}
		for i in range(len(b_ug)):
			if b_popp[i] in killed_set:
				w[b_ug[i]] = w.get(b_ug[i], 0.0) + float(b_vol[i])
		if not w:
			return None
		if entries is None:
			return float('nan'), 1.0, 1.0
		e_ug, e_popp, e_lat, e_vol, good, live = entries
		tot = {u: 0.0 for u in w}; gvol = {u: 0.0 for u in w}; glat = {u: 0.0 for u in w}
		cvol = {u: 0.0 for u in w}
		for i in range(len(e_ug)):
			u = e_ug[i]
			if u not in w:
				continue
			v = float(e_vol[i])
			tot[u] += v
			if good[i]:
				gvol[u] += v; glat[u] += v * float(e_lat[i])
			elif live[i]:
				cvol[u] += v
		W = float(sum(w.values()))
		ug_vol = sas.whole_deployment_ug_to_vol
		lat_num = lat_den = 0.0
		cong = nr = 0.0
		for u, wu in w.items():
			full = float(ug_vol.get(u, tot[u])) or 1.0
			if gvol[u] > 0:
				lat_num += wu * (glat[u] / gvol[u]); lat_den += wu
			cong += wu * cvol[u] / full
			nr += wu * max(0.0, full - gvol[u] - cvol[u]) / full
		return ((lat_num / lat_den) if lat_den > 0 else float('nan'), cong / W, nr / W)

	steady = _steady(adv)
	base_entries = steady[3]
	lats, congs, nrs = [], [], []
	alats, acongs, anrs = [], [], []
	for killed in _scenarios(sas, which, n_fail):
		killed_set = set(int(k) for k in killed)
		a = adv.copy()
		for k in killed_set:
			a[k, :] = 0
		try:
			l, c, nr, entries = _steady(a)
		except ValueError:          # pin LP unsolved for this scenario
			l, c, nr, entries = float('nan'), 1.0, 1.0, None
		lats.append(l); congs.append(c); nrs.append(nr)
		if base_entries is not None:
			aff = _affected(base_entries, killed_set, entries)
			if aff is not None:
				alats.append(aff[0]); acongs.append(aff[1]); anrs.append(aff[2])
	return {
		'fail_affected_latency_ms': float(np.nanmean(alats)) if alats and not all(np.isnan(alats)) else float('nan'),
		'fail_affected_frac_cong': float(np.mean(acongs)) if acongs else 0.0,
		'fail_affected_frac_no_route': float(np.mean(anrs)) if anrs else 0.0,
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


def reactive_objective(sas, adv=None, **lp_kwargs):
  """The frozen_prefix objective SCALAR for an advertisement whose
  assignment is RE-OPTIMIZED after each failure (Tom 2026-09-10: the
  ablation's 100% anchor for frozen_prefix is 'the OPP that can move
  around' -- unrealistic, a bound; frozen one-per-peering strands every
  pinned user of a failed popp and scores below painter).

  Same composition and weights as the lifted frozen LP
  (core/frozen_prefix.py _solve_lp_frozen_prefix_lifted), same evaluation
  kill set (default_kill_popps(n_popps, frozen_n_fail), what the ground-
  truth measured_objective uses), same levers (registry lp_defaults + env):
    -( lat_scale * [lat_mass_normal + (gamma/K) * sum_k lat_mass_k] / V
       + P_nr * [nr_normal + w_nr * sum_k nr_k] / V
       + P_c  * [ovf_normal + w_c * sum_k ovf_k] / V )
  with w_nr = gamma (penalty_sum no_route|both) else gamma/K, w_c = gamma
  (both) else gamma/K -- but every scenario's routing is RE-OPTIMIZED on
  the surviving advertisement by the frozen LP itself with an empty kill
  set (same pricing: latency at lat_scale, overflow at P_c with soft
  capacities, unroutable users at P_nr), so lat_mass sums vol*latency over
  routed users, nr is the unroutable volume and ovf the overflow volume of
  that scenario. Returns the LP-convention BENEFIT (negative cost, like
  ret['objective']); measured_objective negates it."""
  from core.frozen_prefix import kill_scenarios_from_kwargs, _knob, _penalty_sum_mode
  from core.objective_registry import lp_kwargs_for
  kw = dict(lp_kwargs_for('frozen_prefix'), **lp_kwargs)
  gamma = _knob(kw, 'frozen_gamma', 'SCULPTOR_FROZEN_PREFIX_GAMMA', 1.0)
  n_fail = _knob(kw, 'frozen_n_fail', 'SCULPTOR_FROZEN_PREFIX_N_FAIL', 20, int)
  p_nr = _knob(kw, 'frozen_no_route_penalty', 'SCULPTOR_FROZEN_PREFIX_NO_ROUTE_PENALTY', 50.0)
  p_c = _knob(kw, 'frozen_congestion_penalty', 'SCULPTOR_FROZEN_PREFIX_CONGESTION_PENALTY', 25.0)
  lat_scale = _knob(kw, 'frozen_lat_scale', 'SCULPTOR_FROZEN_PREFIX_LAT_SCALE', 1.0)
  penalty_sum = _penalty_sum_mode(kw)
  n_popps = sas.n_popps
  adv = np.eye(n_popps) if adv is None else threshold_a(np.asarray(adv, dtype=float))
  kill = kill_scenarios_from_kwargs(sas, kw, n_fail)   # single peerings and/or sites
  K = len(kill)
  w_k = gamma / K if K else 0.0
  w_nr = (gamma if penalty_sum in ('no_route', 'both') else w_k) if K else 0.0
  w_c = (gamma if penalty_sum == 'both' else w_k) if K else 0.0
  vols = np.asarray([sas.whole_deployment_ug_to_vol[u] for u in sas.whole_deployment_ugs], dtype=float)
  V = float(vols.sum()) or 1.0

  def _scenario(a):
    # Re-optimized routing for ONE scenario, priced exactly as the frozen
    # LP prices its normal scenario: the lifted frozen LP with an EMPTY
    # kill set (K=0) on the surviving advertisement -- latency at
    # lat_scale, overflow at P_c (soft capacities), unroutable users at
    # P_nr. (The avg_latency failure-catch LP is NOT usable here: its
    # fallback marks congested users with the NO_ROUTE sentinel, which
    # would charge them as stranded -- anycast scored 310 vs 7 frozen on
    # the small check, 2026-09-10.)
    if a.sum() == 0:
      return 0.0, V, 0.0
    from core.frozen_prefix import solve_lp_frozen_prefix
    rti, _ = sas.calculate_ground_truth_ingress(a)
    kw0 = {k: v for k, v in kw.items() if k.startswith('frozen_') and k != 'frozen_kill_popps'}
    kw0['frozen_kill_popps'] = []
    ret = solve_lp_frozen_prefix(sas, rti, 'frozen_prefix', adv=a, **kw0)
    if not ret.get('solved'):
      return 0.0, V, 0.0
    unr = float(ret.get('frozen_prefix_unroutable_frac', 0.0) or 0.0)
    lat = float(ret.get('frozen_prefix_normal_lat', 0.0) or 0.0)      # mean over routed volume
    ovf = float(ret.get('frozen_prefix_normal_overflow_frac', 0.0) or 0.0)
    return lat * (1.0 - unr) * V, unr * V, ovf * V

  lm0, nr0, ov0 = _scenario(adv)
  lat_term, nr_term, c_term = lm0, nr0, ov0
  per = []
  for sc in kill:
    a = adv.copy()
    a[list(sc), :] = 0
    lm, nr, ov = _scenario(a)
    lat_term += w_k * lm
    nr_term += w_nr * nr
    c_term += w_c * ov
    per.append((sc[0] if len(sc) == 1 else list(sc), lm / V, nr / V, ov / V))
  cost = (lat_scale * lat_term + p_nr * nr_term + p_c * c_term) / V
  return {'objective': -float(cost), 'cost': float(cost),
          'normal': (lm0 / V, nr0 / V, ov0 / V), 'per_failure': per,
          'kill_popps': [sc[0] if len(sc) == 1 else list(sc) for sc in kill],
          'gamma': gamma, 'lat_scale': lat_scale,
          'no_route_penalty': p_nr, 'congestion_penalty': p_c, 'penalty_sum': penalty_sum}
