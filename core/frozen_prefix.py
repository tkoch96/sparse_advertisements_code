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
  frozen_site_fail_frac / SCULPTOR_FROZEN_PREFIX_SITE_FAIL_FRAC (0.0)  share of the
        n_fail slots that fail a WHOLE SITE (all its peerings at once)
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


# per-process column-generation warm start: n_popps -> (ug, prefix) keys of
# the last converged solve's support (see _solve_lp_frozen_prefix_lifted)
_COLGEN_WARM = {}


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


def _penalty_sum_mode(kwargs):
	"""frozen_penalty_sum: which failure-scenario PENALTIES are summed over
	the K sampled failures (weight gamma each) instead of averaged (gamma/K):
	'no_route' (default; Tom 2026-09-07 "stranding is worse"), 'both'
	(no-route + overflow), 'none' (pre-2026-09-07 mean semantics). Latency
	is always a mean. Rationale: with the mean, stranding one unit in one of
	K failures cost P_nr/K while one unit of excess in normal operation cost
	P_c undiluted, so the assignment LP preferred stranding users who had
	safe alternatives (small ÷100 arm: 6 users; P_nr*K -> 0.000% no-route)."""
	v = kwargs.get('frozen_penalty_sum')
	if v is None:
		v = os.environ.get('SCULPTOR_FROZEN_PREFIX_PENALTY_SUM', 'no_route')
	v = str(v).strip().lower()
	if v in ('1', 'true', 'yes', 'on'):
		return 'no_route'
	if v in ('0', 'false', 'no', 'off', 'none', 'mean'):
		return 'none'
	if v not in ('no_route', 'both'):
		raise ValueError("frozen_penalty_sum must be none|no_route|both, got {!r}".format(v))
	return v


def default_kill_popps(n_popps, n_fail):
	"""Deterministic stride sample over ALL popp indices (advertised or
	not -- killing an unadvertised popp is a no-op scenario, but keying off
	the adv would make the default set drift between the two halves of a
	finite-difference probe pair)."""
	if n_fail >= n_popps:
		return list(range(n_popps))
	step = max(1, n_popps // n_fail)
	return list(range(n_popps))[::step][:n_fail]


def site_groups(sas):
	"""[(popp indices of one site), ...] in order of first appearance."""
	groups = {}
	for i, popp in enumerate(sas.popps):
		groups.setdefault(popp[0], []).append(i)
	return [tuple(v) for v in groups.values()]


def normalize_kill_scenarios(kill):
	"""frozen_kill_popps entries -> list of sorted popp-index tuples. An int
	is a single-peering failure, an iterable a GROUP failing together (a
	site: every peering at that site, Tom 2026-09-11). Order kept, exact
	duplicates dropped, empty groups dropped."""
	out, seen = [], set()
	for k in kill:
		if isinstance(k, (int, np.integer)):
			t = (int(k),)
		else:
			t = tuple(sorted(set(int(x) for x in k)))
		if t and t not in seen:
			seen.add(t)
			out.append(t)
	return out


def default_kill_scenarios(sas, n_fail, site_frac=0.0):
	"""Deterministic default kill set: default_kill_popps' stride sample of
	single peerings, with round(site_frac * n_fail) of the n_fail slots spent
	on whole-site failures (a stride over the sites). site_frac == 0 is
	exactly the pre-2026-09-11 default."""
	n_popps = sas.n_popps
	n_site = int(round(float(site_frac) * n_fail)) if site_frac and site_frac > 0 else 0
	sites = []
	if n_site > 0:
		groups = site_groups(sas)
		n_site = min(n_site, len(groups))
		step = max(1, len(groups) // n_site)
		sites = groups[::step][:n_site]
	singles = default_kill_popps(n_popps, max(int(n_fail) - len(sites), 0))
	return [int(k) for k in singles] + [tuple(g) for g in sites]


def kill_scenarios_from_kwargs(sas, kwargs, n_fail):
	"""The scenario list an LP call prices: frozen_kill_popps if given (ints
	and/or groups), else the deterministic default with the objective's
	site share (frozen_site_fail_frac / SCULPTOR_FROZEN_PREFIX_SITE_FAIL_FRAC)."""
	kill = kwargs.get('frozen_kill_popps')
	if kill is None:
		site_frac = _knob(kwargs, 'frozen_site_fail_frac',
						  'SCULPTOR_FROZEN_PREFIX_SITE_FAIL_FRAC', 0.0)
		kill = default_kill_scenarios(sas, n_fail, site_frac)
	return normalize_kill_scenarios(kill)


def kill_scenarios_as_list(scenarios):
	"""Result-dict form: an int per single failure, a list per group."""
	return [s[0] if len(s) == 1 else list(s) for s in scenarios]


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
	form = kwargs.get('frozen_formulation') or os.environ.get(
		'SCULPTOR_FROZEN_PREFIX_FORMULATION', 'lifted')
	if form == 'stacked':
		return _solve_lp_frozen_prefix_stacked(sas, routed_through_ingress, obj, **kwargs)
	if form == 'lifted':
		# persistent per-process model with a basis warm start across probes
		# (core/frozen_prefix_persistent.py, Tom 2026-09-11); the exhaustive
		# eval pin (K ~ n_popps) and K=0 calls stay on the lifted builder
		from core.frozen_prefix_persistent import persistent_enabled, solve_persistent
		n_fail = _knob(kwargs, 'frozen_n_fail', 'SCULPTOR_FROZEN_PREFIX_N_FAIL', 20, int)
		if kwargs.get('adv') is not None and persistent_enabled(kwargs, len(kill_scenarios_from_kwargs(sas, kwargs, n_fail))):
			return solve_persistent(sas, routed_through_ingress, obj, **kwargs)
	return _solve_lp_frozen_prefix_lifted(sas, routed_through_ingress, obj, **kwargs)


def frozen_fallbacks(sas, adv, pair_prefix, pair_ugi, base_winner):
	"""Per pair: the popp that carries it when its NORMAL winner fails --
	the best remaining BGP ingress (highest popp_by_ug_indicator pref, first
	index on ties, exactly calculate_ground_truth_ingress's rule) among the
	prefix's advertised popps other than the pair's winner; -1 if none.

	Single-popp failures only ever remove the pair's own winner, so every
	pair has exactly ONE fallback regardless of which popps are in the kill
	set -- this is the whole reason the failure scenarios are near-copies of
	the normal one (see _solve_lp_frozen_prefix_lifted).

	Vectorized per prefix column (top-2 by pref); equals a per-scenario
	calculate_ground_truth_ingress(adv with row k zeroed) lookup, which is
	what the stacked reference does and what test_frozen_prefix_lifted.py
	checks. Falls back to that lookup when the indicator matrix is missing.
	"""
	n_pairs = len(pair_prefix)
	fb = np.full(n_pairs, -1, dtype=int)
	if n_pairs == 0 or adv is None:
		return fb
	a = threshold_a(np.asarray(adv, dtype=float))
	ind = getattr(sas, 'popp_by_ug_indicator', None)
	if ind is None:
		return _frozen_fallbacks_via_gti(sas, a, pair_prefix, pair_ugi, base_winner)
	pair_prefix = np.asarray(pair_prefix, dtype=int)
	pair_ugi = np.asarray(pair_ugi, dtype=int)
	base_winner = np.asarray(base_winner, dtype=int)
	order = np.argsort(pair_prefix, kind='stable')
	bounds = np.searchsorted(pair_prefix[order], np.arange(a.shape[1] + 1))
	for prefix_i in range(a.shape[1]):
		lo, hi = bounds[prefix_i], bounds[prefix_i + 1]
		if hi <= lo:
			continue
		active = np.where(a[:, prefix_i] > 0)[0]
		if len(active) < 2:
			continue    # nothing to fall back to
		idx = order[lo:hi]
		ugis = pair_ugi[idx]
		D = ind[active][:, ugis].toarray()          # (n_active, n_pairs_in_prefix)
		cols = np.arange(D.shape[1])
		top1 = np.argmax(D, axis=0)                 # first index on ties, like gti
		D2 = D.copy()
		D2[top1, cols] = -1.0
		top2 = np.argmax(D2, axis=0)
		top2_ok = D2[top2, cols] > 0
		top1_popp = active[top1]
		# winner == best -> fallback is 2nd best; winner is some other
		# (MC-realized) popp -> the ground-truth best is what remains
		use_second = base_winner[idx] == top1_popp
		f = np.where(use_second, np.where(top2_ok, active[top2], -1), top1_popp)
		f = np.where(f == base_winner[idx], -1, f)   # never its own winner
		fb[idx] = f
	return fb


def _frozen_fallbacks_via_gti(sas, a, pair_prefix, pair_ugi, base_winner):
	"""Slow exact path (one calculate_ground_truth_ingress per distinct
	winner) for deployments without popp_by_ug_indicator."""
	popp_to_ind = sas.popp_to_ind
	ugs = sas.whole_deployment_ugs
	n_pairs = len(pair_prefix)
	fb = np.full(n_pairs, -1, dtype=int)
	base_winner = np.asarray(base_winner, dtype=int)
	for k in np.unique(base_winner):
		affected = np.where(base_winner == k)[0]
		a_fail = a.copy()
		a_fail[k, :] = 0
		fail_rti, _ = sas.calculate_ground_truth_ingress(a_fail)
		for i in affected:
			popp_tuple = fail_rti.get(int(pair_prefix[i]), {}).get(ugs[pair_ugi[i]])
			poppi = popp_to_ind.get(popp_tuple) if popp_tuple is not None else None
			if poppi is not None and poppi != k:
				fb[i] = poppi
	return fb


def frozen_fallbacks_killed(sas, adv, pair_prefix, pair_ugi, base_winner, killed):
	"""Fallbacks for a GROUP failure (several popps -- a site -- die
	together): for every pair whose normal winner is in `killed`, the best
	remaining BGP ingress among the prefix's advertised popps OUTSIDE
	`killed` (frozen_fallbacks' rule with the whole group removed; -1 if
	none). Returns (moved_idx, fb) with fb aligned to moved_idx. For a
	singleton group this equals frozen_fallbacks on those pairs."""
	killed = np.asarray(sorted(set(int(k) for k in killed)), dtype=int)
	base_winner = np.asarray(base_winner, dtype=int)
	moved_idx = np.where(np.isin(base_winner, killed))[0]
	fb = np.full(len(moved_idx), -1, dtype=int)
	if len(moved_idx) == 0 or adv is None:
		return moved_idx, fb
	a = threshold_a(np.asarray(adv, dtype=float))
	ind = getattr(sas, 'popp_by_ug_indicator', None)
	pp = np.asarray(pair_prefix, dtype=int)[moved_idx]
	pu = np.asarray(pair_ugi, dtype=int)[moved_idx]
	if ind is None:
		a_fail = a.copy()
		a_fail[killed, :] = 0
		fail_rti, _ = sas.calculate_ground_truth_ingress(a_fail)
		ugs = sas.whole_deployment_ugs
		popp_to_ind = sas.popp_to_ind
		kset = set(int(k) for k in killed)
		for j in range(len(moved_idx)):
			popp_tuple = fail_rti.get(int(pp[j]), {}).get(ugs[pu[j]])
			poppi = popp_to_ind.get(popp_tuple) if popp_tuple is not None else None
			if poppi is not None and poppi not in kset:
				fb[j] = poppi
		return moved_idx, fb
	kill_mask = np.zeros(a.shape[0], dtype=bool)
	kill_mask[killed] = True
	order = np.argsort(pp, kind='stable')
	bounds = np.searchsorted(pp[order], np.arange(a.shape[1] + 1))
	for prefix_i in range(a.shape[1]):
		lo, hi = bounds[prefix_i], bounds[prefix_i + 1]
		if hi <= lo:
			continue
		active = np.where((a[:, prefix_i] > 0) & ~kill_mask)[0]
		if len(active) == 0:
			continue
		idx = order[lo:hi]
		D = ind[active][:, pu[idx]].toarray()
		top1 = np.argmax(D, axis=0)
		ok = D[top1, np.arange(D.shape[1])] > 0
		fb[idx] = np.where(ok, active[top1], -1)
	return moved_idx, fb


def _solve_lp_frozen_prefix_lifted(sas, routed_through_ingress, obj, **kwargs):
	"""LIFTED formulation (2026-09-07). Same optimum as the stacked
	reference, but the LP size no longer scales with the number of failure
	scenarios.

	Observation: a single-popp failure k only moves the pairs whose normal
	winner is k, each to a fixed BGP fallback. So scenario k's load on popp j
	is  L_j + (volume falling from k onto j),  i.e. the NORMAL load plus a
	sparse delta. Writing the normal load once as an auxiliary variable L_j
	and adding a row only where a delta exists gives:

	  vars: x (pairs), L (popps), o0 (normal overflow), o_kj (one per
	        (scenario, receiving popp) with non-empty delta)
	  rows: conservation; L_j = sum x on j; L_j - o0_j <= cap_j;
	        L_j + delta_kj(x) - o_kj <= cap_j  for affected (k, j)
	  nnz:  ~2 * n_pairs + O(rows), independent of K

	Popps unaffected in scenario k have the identical row to the normal one,
	so they share o0_j (its objective weight absorbs those scenarios' gamma/K
	-- exact, the optimal overflow is max(0, L_j - cap) either way). The
	killed popp carries nothing in its own scenario (no row, no cost).
	Latency / no-route costs are per-pair coefficients as before.

	Because K no longer costs anything in the LP, n_fail >= n_popps
	(exhaustive single-popp failures) becomes affordable; the sampled
	explore/exploit kill set stays available as the default lever.
	"""
	adv = kwargs.get('adv')
	gamma = _knob(kwargs, 'frozen_gamma', 'SCULPTOR_FROZEN_PREFIX_GAMMA', 1.0)
	n_fail = _knob(kwargs, 'frozen_n_fail', 'SCULPTOR_FROZEN_PREFIX_N_FAIL', 20, int)
	p_nr = _knob(kwargs, 'frozen_no_route_penalty',
				 'SCULPTOR_FROZEN_PREFIX_NO_ROUTE_PENALTY', 50.0)
	p_c = _knob(kwargs, 'frozen_congestion_penalty',
				'SCULPTOR_FROZEN_PREFIX_CONGESTION_PENALTY', 25.0)
	lat_scale = _knob(kwargs, 'frozen_lat_scale',
					  'SCULPTOR_FROZEN_PREFIX_LAT_SCALE', 1.0)
	cap_headroom = _knob(kwargs, 'frozen_cap_headroom',
						 'SCULPTOR_FROZEN_PREFIX_CAP_HEADROOM', 1.0)
	# Solver wall-clock cap. 30 s is right for training probes; the eval PIN
	# (exhaustive failure set at size 32) legitimately needs longer.
	time_limit = _knob(kwargs, 'frozen_time_limit',
					   'SCULPTOR_FROZEN_PREFIX_TIME_LIMIT', 30.0)
	penalty_sum = _penalty_sum_mode(kwargs)

	from core.solve_lp_assignment import obj_round
	from scipy.sparse import hstack as sp_hstack, vstack as sp_vstack

	n_popps = sas.n_popps
	popp_to_ind = sas.popp_to_ind
	popps = sas.popps
	ug_perfs = sas.whole_deployment_ug_perfs
	ug_to_vol = sas.whole_deployment_ug_to_vol
	ug_to_ind = sas.whole_deployment_ug_to_ind
	n_ug = sas.whole_deployment_n_ug
	caps = np.asarray(sas.link_capacities_arr, dtype=float).flatten()
	total_vol = float(sum(ug_to_vol.values())) or 1.0

	# scenarios: single peerings and/or GROUPS (sites) failing together
	# (Tom 2026-09-11); a group scenario is priced exactly like a single one
	# (latency of the displaced pairs at their fallback, P_nr for the
	# stranded ones, P_c on the overflow its fallback volume causes).
	scenarios = kill_scenarios_from_kwargs(sas, kwargs, n_fail)
	K = len(scenarios)
	w_k = gamma / K if K else 0.0                 # latency weight per failure scenario
	w_nr = (gamma if penalty_sum in ('no_route', 'both') else w_k) if K else 0.0
	w_c = (gamma if penalty_sum == 'both' else w_k) if K else 0.0
	kill_count = np.zeros(n_popps, dtype=int)     # scenarios in which popp j is dead
	for sc in scenarios:
		kill_count[list(sc)] += 1
	kpos1 = np.full(n_popps, -1, dtype=int)       # popp -> its SINGLE-failure scenario
	for si, sc in enumerate(scenarios):
		if len(sc) == 1:
			kpos1[sc[0]] = si
	group_scen = [(si, sc) for si, sc in enumerate(scenarios) if len(sc) > 1]

	# ---- pairs: every (ug, prefix) routable in the NORMAL scenario
	pair_ug, pair_ugi, pair_prefix, base_winner, base_lat = [], [], [], [], []
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
			pair_prefix.append(int(prefix_i))
			base_winner.append(poppi)
			base_lat.append(float(lat))
	n_pairs = len(pair_ug)

	routable_ugis = set(pair_ugi)
	unroutable_vol = float(sum(v for ug, v in ug_to_vol.items()
							   if ug_to_ind[ug] not in routable_ugis))
	# users no-route in EVERY scenario: normal + each failure scenario's weight
	const_term = (1.0 + w_nr * K) * p_nr * unroutable_vol / total_vol

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
	pair_prefix = np.asarray(pair_prefix, dtype=int)
	base_winner = np.asarray(base_winner, dtype=int)
	base_lat = np.asarray(base_lat, dtype=float)

	# ---- moved entries: one per (pair, scenario) in which the pair's winner
	# dies, with where it lands. Single-popp scenarios: the vectorized
	# top-2 fallback (every pair has one). Group scenarios: the best
	# remaining ingress outside the whole group.
	fb1 = frozen_fallbacks(sas, adv, pair_prefix, pair_ugi, base_winner)
	mv_pair, mv_scen, mv_fb = [], [], []
	m1 = np.where(kpos1[base_winner] >= 0)[0]
	if len(m1):
		mv_pair.append(m1); mv_scen.append(kpos1[base_winner[m1]]); mv_fb.append(fb1[m1])
	for si, sc in group_scen:
		idx, fbs = frozen_fallbacks_killed(sas, adv, pair_prefix, pair_ugi, base_winner, sc)
		if len(idx):
			mv_pair.append(idx); mv_scen.append(np.full(len(idx), si, dtype=int)); mv_fb.append(fbs)
	if mv_pair:
		mv_pair = np.concatenate(mv_pair).astype(int)
		mv_scen = np.concatenate(mv_scen).astype(int)
		mv_fb = np.concatenate(mv_fb).astype(int)
	else:
		mv_pair = np.zeros(0, dtype=int); mv_scen = np.zeros(0, dtype=int); mv_fb = np.zeros(0, dtype=int)
	mv_fb_lat = np.zeros(len(mv_pair))
	for j in np.where(mv_fb >= 0)[0]:
		lat = ug_perfs[pair_ug[mv_pair[j]]].get(popps[mv_fb[j]], NO_ROUTE_LATENCY)
		if lat >= NO_ROUTE_LATENCY:
			mv_fb[j] = -1
		else:
			mv_fb_lat[j] = float(lat)
	mv_live = mv_fb >= 0
	mv_dead = ~mv_live
	n_moved = np.bincount(mv_pair, minlength=n_pairs)   # scenarios displacing each pair

	# ---- objective on x: normal weight 1 + gamma/K per scenario where the
	# pair keeps its winner; each scenario displacing it prices the fallback
	n_keep = K - n_moved
	c_x = (1.0 + w_k * n_keep) * base_lat * lat_scale / total_vol
	if mv_live.any():
		c_x = c_x + np.bincount(mv_pair[mv_live], weights=w_k * mv_fb_lat[mv_live] * lat_scale,
								minlength=n_pairs) / total_vol
	if mv_dead.any():
		c_x = c_x + np.bincount(mv_pair[mv_dead], weights=np.full(int(mv_dead.sum()), w_nr * p_nr),
								minlength=n_pairs) / total_vol

	# ---- rows (row layout of the model: cons | loaddef | normcap | aff)
	keep = np.asarray(sorted(routable_ugis), dtype=int)
	ug_row = np.searchsorted(keep, pair_ugi)            # pair -> its conservation row
	cons_b = np.asarray([ug_to_vol[sas.whole_deployment_ugs[u]] for u in keep], dtype=float)
	if mv_live.any():
		ml_e = np.where(mv_live)[0]              # moved-entry indices
		ml = mv_pair[ml_e]                        # their pair (LP column)
		key = mv_scen[ml_e] * n_popps + mv_fb[ml_e]
		uniq, row_of = np.unique(key, return_inverse=True)
		n_aff = len(uniq)
		aff_scen = uniq // n_popps
		aff_popp = uniq % n_popps
	else:
		ml = np.zeros(0, dtype=int); row_of = np.zeros(0, dtype=int)
		n_aff = 0
		aff_scen = np.zeros(0, dtype=int); aff_popp = np.zeros(0, dtype=int)

	# ---- overflow costs. o0_j stands in for every scenario in which popp j
	# is untouched: all K sampled scenarios minus its own failures minus the
	# m_j scenarios in which it receives fallback volume.
	m_j = np.bincount(aff_popp, minlength=n_popps) if n_aff else np.zeros(n_popps, dtype=int)
	c_o0 = (1.0 + w_c * (K - kill_count - m_j)) * p_c / total_vol
	c_oa = np.full(n_aff, w_c * p_c / total_vol)
	n_cons = len(keep)

	def _solve_cols(active, limit=None):
		"""Solve the lifted LP over the x columns `active` (sorted pair
		indices; every other pair fixed at 0). All rows are kept (the aff rows
		belong to (scenario, popp) keys, not to pairs), so the row duals price
		any excluded pair exactly. Returns (status, z, objval, row_duals)."""
		from scipy.sparse import hstack as sp_hstack, vstack as sp_vstack
		n_a = len(active)
		pos = np.full(n_pairs, -1, dtype=int)
		pos[active] = np.arange(n_a)
		iL = n_a
		iO0 = iL + n_popps
		iOA = iO0 + n_popps
		n_z = iOA + n_aff

		def _sp(rows, cols, vals, nrows):
			return csr_matrix((np.asarray(vals, dtype=float), (rows, cols)), shape=(nrows, n_z))
		# conservation: sum_{i in ug} x_i = vol_ug
		cons_A = _sp(ug_row[active], np.arange(n_a), np.ones(n_a), n_cons)
		# L_j - sum_{i: winner j} x_i = 0
		loaddef_A = _sp(np.concatenate([base_winner[active], np.arange(n_popps)]),
						np.concatenate([np.arange(n_a), iL + np.arange(n_popps)]),
						np.concatenate([-np.ones(n_a), np.ones(n_popps)]), n_popps)
		# L_j - o0_j <= cap_j
		normcap_A = _sp(np.concatenate([np.arange(n_popps), np.arange(n_popps)]),
						np.concatenate([iL + np.arange(n_popps), iO0 + np.arange(n_popps)]),
						np.concatenate([np.ones(n_popps), -np.ones(n_popps)]), n_popps)
		eq_A = sp_vstack([cons_A, loaddef_A], format='csr')
		eq_b = np.concatenate([cons_b, np.zeros(n_popps)])
		if n_aff:
			# L_j + sum_{i falls onto j in scenario k} x_i - o_kj <= cap_j
			r = np.arange(n_aff)
			sel = pos[ml] >= 0
			aff_A = _sp(np.concatenate([row_of[sel], r, r]),
						np.concatenate([pos[ml[sel]], iL + aff_popp, iOA + r]),
						np.concatenate([np.ones(int(sel.sum())), np.ones(n_aff), -np.ones(n_aff)]), n_aff)
			le_A = sp_vstack([normcap_A, aff_A], format='csr')
			le_b = np.concatenate([caps * cap_headroom, caps[aff_popp] * cap_headroom])
		else:
			le_A = normcap_A
			le_b = caps * cap_headroom
		c_z = np.concatenate([c_x[active], np.zeros(n_popps), c_o0, c_oa])
		model = gp.Model()
		model.Params.LogToConsole = 0
		model.Params.TimeLimit = time_limit if limit is None else float(limit)
		model.Params.Threads = 1     # see the stacked reference: shared-scheduler hazard
		z = model.addMVar(n_z, name='vol_ug_prefix_load_overflow', lb=0)
		model.addConstr(eq_A @ z == eq_b)
		model.addConstr(le_A @ z <= le_b)
		model.setObjective(c_z @ z)
		model.optimize()
		if model.status != 2:
			_clear_highs_model(model)
			return model.status, None, None, None, (int(eq_A.shape[0] + le_A.shape[0]), int(n_z), int(eq_A.nnz + le_A.nnz))
		zv = np.asarray(z.X).flatten()
		objval = float(model.objVal)
		try:
			y = np.asarray(model.getRowDuals(), dtype=float)
		except Exception:
			y = None
		_clear_highs_model(model)
		return 2, zv, objval, y, (int(eq_A.shape[0] + le_A.shape[0]), int(n_z), int(eq_A.nnz + le_A.nnz))

	def _reduced_costs(y):
		"""c_i - A_i^T y for EVERY pair (rows: cons | loaddef | normcap | aff)."""
		y_cons = y[:n_cons]
		y_load = y[n_cons:n_cons + n_popps]
		y_aff = y[n_cons + 2 * n_popps:]
		rc = c_x - y_cons[ug_row] + y_load[base_winner]
		if n_aff:
			rc = rc - np.bincount(ml, weights=y_aff[row_of], minlength=n_pairs)
		return rc

	# ---- column generation (Tom 2026-09-11: "prune the LP"). The optimum
	# uses ~1.1 pairs per user out of ~40 routable ones (size 32: 5.7k of
	# 217k), but a static latency-based prune is badly wrong (capacity relief
	# needs far-away prefixes). So: solve on a small candidate set, price
	# every excluded pair with the row duals, add the violators, repeat --
	# exact at convergence (all reduced costs >= 0), and 5-10x fewer columns
	# per solve. Off: SCULPTOR_FROZEN_PREFIX_COLGEN=0 / frozen_colgen=0.
	colgen = _knob(kwargs, 'frozen_colgen', 'SCULPTOR_FROZEN_PREFIX_COLGEN', 1, int)
	colgen_k = _knob(kwargs, 'frozen_colgen_k', 'SCULPTOR_FROZEN_PREFIX_COLGEN_K', 6, int)
	colgen_min_pairs = _knob(kwargs, 'frozen_colgen_min_pairs', 'SCULPTOR_FROZEN_PREFIX_COLGEN_MIN_PAIRS', 5000, int)
	ts = time.time()
	colgen_rounds = 0
	if colgen and n_pairs > colgen_min_pairs:
		# initial set: per user, the colgen_k cheapest pairs by objective cost
		order = np.lexsort((c_x, pair_ugi))
		rank = np.empty(n_pairs, dtype=int)
		first = np.r_[0, np.flatnonzero(np.diff(pair_ugi[order])) + 1]
		rank[order] = np.arange(n_pairs) - np.repeat(first, np.diff(np.r_[first, n_pairs]))
		active_mask = rank < colgen_k
		# warm start: the pairs the PREVIOUS solve in this process ended with
		# (a worker prices many probes around one advertisement; the optimal
		# support barely moves between them) -- keyed by (ug, prefix), so it
		# survives the pair set changing with the adv
		pair_key = pair_ugi.astype(np.int64) * int(adv.shape[1] if adv is not None else 1 << 20) + pair_prefix.astype(np.int64)
		_warm = _COLGEN_WARM.get(n_popps)
		if _warm is not None:
			active_mask |= np.isin(pair_key, _warm)
		warm = kwargs.get('frozen_colgen_warm_pairs')
		if warm is not None and len(warm):
			active_mask[np.asarray(warm, dtype=int)] = True
		status, converged = None, False
		# A pair only counts as improving when its reduced cost is negative
		# beyond the solver's own dual tolerance in the COST's scale (costs
		# here are ~1e-6..1e-5 per unit volume: an absolute 1e-9 test, the
		# first version, treated dual noise as violators and cycled to the
		# round cap, then fell back to a full solve -- 100 s probes at
		# iteration 90 of deployment 2, 2026-09-12).
		c_scale = float(np.abs(c_x).max()) if n_pairs else 1.0
		rc_tol = 1e-6 * max(c_scale, 1e-12)
		max_rounds = _knob(kwargs, 'frozen_colgen_max_rounds', 'SCULPTOR_FROZEN_PREFIX_COLGEN_MAX_ROUNDS', 12, int)
		# per-round solver cap: a hard restricted LP must not turn one probe into
		# max_rounds x time_limit (12 x 120 s straggler probes stretched
		# iterations 90/110 of the 2026-09-12 size-32 run to 12-30 min while
		# 158 workers sat idle); a capped round falls back to the last optimal
		# round's solution, which is feasible and near-optimal
		round_limit = min(time_limit, _knob(kwargs, 'frozen_colgen_round_limit',
											 'SCULPTOR_FROZEN_PREFIX_COLGEN_ROUND_LIMIT', 60.0))
		last_good = None
		for colgen_rounds in range(1, max_rounds + 1):
			active = np.flatnonzero(active_mask)
			status, zv, raw_objval, y, dims = _solve_cols(active, round_limit)
			if status != 2 or y is None:
				break
			last_good = (active, zv, raw_objval, y, dims)
			rc = _reduced_costs(y)
			rc[active_mask] = 0.0
			viol = np.flatnonzero(rc < -rc_tol)
			if len(viol) == 0:
				converged = True
				break
			if len(viol) > 20000:
				viol = viol[np.argsort(rc[viol])[:20000]]
			active_mask[viol] = True
		if status != 2 or y is None:
			if last_good is not None:
				active, zv, raw_objval, y, dims = last_good
			else:
				# the very first restricted solve failed (time limit / no duals):
				# the exact full solve is the last resort
				active = np.arange(n_pairs)
				status, zv, raw_objval, y, dims = _solve_cols(active)
				if status != 2:
					return {'solved': False}
				converged = True
		# not converged within max_rounds: the restricted solution is feasible
		# and within sum(|rc| * vol) of optimal -- accept it (a near-exact
		# probe beats a 100 s full solve or an unsolved fallback price)
		# remember the support (pairs carrying volume) for the next probe
		n_a0 = len(active)
		_COLGEN_WARM[n_popps] = pair_key[active[zv[:n_a0] > 1e-12]]
		colgen_converged = converged
	else:
		active = np.arange(n_pairs)
		status, zv, raw_objval, y, dims = _solve_cols(active)
		if status != 2:
			return {'solved': False}
		colgen_converged = True
	n_a = len(active)
	iO0 = n_a + n_popps
	iOA = iO0 + n_popps
	xv = np.zeros(n_pairs)
	xv[active] = zv[:n_a]
	o0 = zv[iO0:iOA]
	oa = zv[iOA:]

	# ---- standard-contract reporting from the NORMAL scenario
	paths_by_ug = {}
	vols_by_poppi = {poppi: 0.0 for poppi in range(n_popps)}
	lat_sum_by_ugi = np.zeros(n_ug)
	vol_sum_by_ugi = np.zeros(n_ug)
	for i in np.where(xv > 0)[0]:
		v = xv[i]
		ugi = pair_ugi[i]
		poppi = base_winner[i]
		vols_by_poppi[poppi] += v
		lat_sum_by_ugi[ugi] += v * base_lat[i]
		vol_sum_by_ugi[ugi] += v
		paths_by_ug.setdefault(int(ugi), []).append((int(poppi), v / ug_to_vol[pair_ug[i]]))
	routed = vol_sum_by_ugi > 0
	lats_by_ug_arr[routed] = lat_sum_by_ugi[routed] / vol_sum_by_ugi[routed]

	normal_loads = np.bincount(base_winner, weights=xv, minlength=n_popps)
	congested_popps = normal_loads > caps + 1e-9
	fraction_congested_volume = float(normal_loads[congested_popps].sum()) / total_vol

	# ---- per-scenario diagnostics, vectorized by scenario index
	fail_lat, fail_nr, fail_ovf = [], [], []
	if K:
		tot_lat_mass = float(np.sum(xv * base_lat))
		tot_x = float(np.sum(xv))
		lost_lat = np.bincount(mv_scen, weights=(xv * base_lat)[mv_pair], minlength=K)
		gain_lat = np.bincount(mv_scen[mv_live], weights=xv[mv_pair[mv_live]] * mv_fb_lat[mv_live], minlength=K)
		dead_vol = np.bincount(mv_scen[mv_dead], weights=xv[mv_pair[mv_dead]], minlength=K)
		live_vol = tot_x - dead_vol
		lat_mass = tot_lat_mass - lost_lat + gain_lat
		fail_lat = list(np.where(live_vol > 0, lat_mass / np.maximum(live_vol, 1e-12), 0.0))
		fail_nr = list((dead_vol + unroutable_vol) / total_vol)
		ovf_aff = np.bincount(aff_scen, weights=oa, minlength=K) if n_aff else np.zeros(K)
		o0_recv = np.bincount(aff_scen, weights=o0[aff_popp], minlength=K) if n_aff else np.zeros(K)
		o0_killed = np.asarray([float(o0[list(sc)].sum()) for sc in scenarios])
		fail_ovf = list((ovf_aff + float(np.sum(o0)) - o0_killed - o0_recv) / total_vol)

	obj_val = raw_objval + const_term
	return {
		'objective': obj_round(-obj_val),
		'raw_solution': xv,
		'paths_by_ug': paths_by_ug,
		'lats_by_ug': lats_by_ug_arr,
		'available_paths': [(pair_ug[i], int(base_winner[i])) for i in range(n_pairs)],
		'solved': status,
		'vols_by_poppi': {poppi: round(v / float(caps[poppi]), 2)
						  for poppi, v in vols_by_poppi.items()},
		'fraction_congested_volume': fraction_congested_volume,
		'frozen_prefix_n_fail': K,
		'frozen_prefix_kill_popps': kill_scenarios_as_list(scenarios),
		'frozen_prefix_n_site_fail': len(group_scen),
		'frozen_prefix_gamma': gamma,
		'frozen_prefix_no_route_penalty': p_nr,
		'frozen_prefix_congestion_penalty': p_c,
		'frozen_prefix_lat_scale': lat_scale,
		'frozen_prefix_cap_headroom': cap_headroom,
		'frozen_prefix_formulation': 'lifted',
		'frozen_prefix_penalty_sum': penalty_sum,
		'frozen_prefix_n_rows': dims[0],
		'frozen_prefix_n_vars': dims[1],
		'frozen_prefix_nnz': dims[2],
		'frozen_prefix_colgen_rounds': colgen_rounds,
		'frozen_prefix_colgen_converged': bool(colgen_converged),
		'frozen_prefix_active_pairs': int(n_a),
		'frozen_prefix_normal_lat': (float(np.sum(xv * base_lat)) /
									 max(float(np.sum(xv)), 1e-9)),
		'frozen_prefix_fail_lat_mean': float(np.mean(fail_lat)) if len(fail_lat) else 0.0,
		'frozen_prefix_fail_no_route_frac': float(np.mean(fail_nr)) if len(fail_nr) else 0.0,
		'frozen_prefix_fail_overflow_frac': float(np.mean(fail_ovf)) if len(fail_ovf) else 0.0,
		'frozen_prefix_normal_overflow_frac': float(np.sum(o0)) / total_vol,
		'frozen_prefix_unroutable_frac': unroutable_vol / total_vol,
		'frozen_prefix_lp_secs': time.time() - ts,
		'frozen_prefix_pairs': [(pair_ug[i], int(pair_prefix[i]), float(xv[i]))
								for i in np.where(xv > 1e-12)[0]],
	}


def _solve_lp_frozen_prefix_stacked(sas, routed_through_ingress, obj, **kwargs):
	"""REFERENCE formulation (2026-09-05..07): one dense load block per
	scenario, (K+1)*n_popps overflow vars. Exact but ~(K+1)x the nonzeros of
	the base LP (18.7 s/probe at actual-32). Kept for equivalence tests and
	as the `frozen_formulation='stacked'` escape hatch; production path is
	_solve_lp_frozen_prefix_lifted."""
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
	penalty_sum = _penalty_sum_mode(kwargs)

	from core.solve_lp_assignment import obj_round

	n_popps = sas.n_popps
	popp_to_ind = sas.popp_to_ind
	ug_perfs = sas.whole_deployment_ug_perfs
	ug_to_vol = sas.whole_deployment_ug_to_vol
	ug_to_ind = sas.whole_deployment_ug_to_ind
	n_ug = sas.whole_deployment_n_ug
	caps = np.asarray(sas.link_capacities_arr, dtype=float).flatten()
	total_vol = float(sum(ug_to_vol.values())) or 1.0

	scenarios = kill_scenarios_from_kwargs(sas, kwargs, n_fail)
	kill_popps = kill_scenarios_as_list(scenarios)
	K = len(scenarios)
	weights = [1.0] + ([gamma / K] * K if K else [])            # latency
	nr_weights = [1.0] + ([gamma if penalty_sum in ('no_route', 'both') else gamma / K] * K if K else [])
	c_weights = [1.0] + ([gamma if penalty_sum == 'both' else gamma / K] * K if K else [])

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
	const_term = sum(nr_weights) * p_nr * unroutable_vol / total_vol

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
	for sc in scenarios:
		w_s = base_winner.copy()
		l_s = base_lat.copy()
		kset = set(int(k) for k in sc)
		affected = np.where(np.isin(base_winner, list(kset)))[0]
		if len(affected) and a_thresh is not None:
			a_fail = a_thresh.copy()
			a_fail[list(kset), :] = 0
			fail_rti, _ = sas.calculate_ground_truth_ingress(a_fail)
			for i in affected:
				popp_tuple = fail_rti.get(pair_prefix[i], {}).get(pair_ug[i])
				poppi = popp_to_ind.get(popp_tuple) if popp_tuple is not None else None
				if poppi is None or poppi in kset:
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
		c_x += (weights[s] * np.where(live, scen_lat[s] * lat_scale, 0.0)
				+ nr_weights[s] * np.where(live, 0.0, p_nr)) / total_vol

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

	c_o = np.repeat(np.asarray(c_weights) * p_c / total_vol, n_popps)

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
