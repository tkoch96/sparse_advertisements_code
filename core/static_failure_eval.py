"""BGP-fallback failure evaluation: simulate single popp/pop failures under
the "frozen user->prefix mapping" semantic for the static_failure objective.

This is intentionally NOT an LP: under static_failure the post-failure routing
is *not* re-optimized -- BGP simply picks the next-most-preferred surviving
popp announcing that user's pinned prefix, and a user whose prefix has no
surviving popp goes to NO_ROUTE.

Capacity: after rerouting, we check per-popp total volume. Users on a popp
whose post-failure load exceeds capacity get NO_ROUTE_LATENCY (same penalty
the existing LP-based eval applies for congested links).

Public entry point: assess_static_failure_resilience(sas, adv, which='popps'|'pops').
Returns a dict with summary stats compatible with the markdown summary writer.

NOTE: `routed_through_ingress[prefix_i][ug]` returns a popp *tuple*
`(pop, peer)`, not an index, despite what the upstream docstring claims.
This module converts via `sas.popp_to_ind` where indices are needed.
"""
import numpy as np

from helpers.constants import NO_ROUTE_LATENCY
from helpers.helpers import threshold_a


def _pin_user_to_prefix(sas, adv):
	"""For each user, return the prefix they would use in steady state, and
	the popp they'd route through on that prefix.

	A user picks the prefix whose BGP-winning popp gives them best latency.
	"""
	a = threshold_a(adv)
	routed, _ = sas.calculate_ground_truth_ingress(a)
	ug_perfs = sas.whole_deployment_ug_perfs

	user_to_prefix = {}
	user_to_popp_tuple = {}
	user_to_lat = {}

	for ug in sas.whole_deployment_ugs:
		best_pref = None
		best_popp = None
		best_lat = NO_ROUTE_LATENCY
		for prefix_i, ug_to_popp in routed.items():
			popp_tuple = ug_to_popp.get(ug)
			if popp_tuple is None:
				continue
			lat = ug_perfs[ug].get(popp_tuple, NO_ROUTE_LATENCY)
			if lat < best_lat:
				best_lat = lat
				best_popp = popp_tuple
				best_pref = prefix_i
		user_to_prefix[ug] = best_pref
		user_to_popp_tuple[ug] = best_popp
		user_to_lat[ug] = best_lat
	return user_to_prefix, user_to_popp_tuple, user_to_lat


def _route_under_failure(sas, adv, failed_popp_inds, user_to_prefix):
	"""Compute per-user latency when popp indices in failed_popp_inds are down,
	users stay on their pinned prefix, BGP picks next-best surviving popp.

	Returns (per_user_lat dict, inundated_popp_inds set).
	"""
	failed_set = set(failed_popp_inds)
	a_fail = adv.copy()
	for popp_i in failed_set:
		a_fail[popp_i, :] = 0
	a_fail = threshold_a(a_fail)
	routed_after, _ = sas.calculate_ground_truth_ingress(a_fail)

	popp_to_ind = sas.popp_to_ind
	ug_perfs = sas.whole_deployment_ug_perfs
	ug_to_vol = sas.whole_deployment_ug_to_vol
	caps_arr = sas.link_capacities_arr.flatten()

	per_user_lat = {}
	per_user_popp_ind = {}
	popp_vol = np.zeros(sas.n_popps)
	for ug in sas.whole_deployment_ugs:
		pref = user_to_prefix.get(ug)
		if pref is None:
			per_user_lat[ug] = NO_ROUTE_LATENCY
			continue
		ug_to_popp = routed_after.get(pref, {})
		popp_tuple = ug_to_popp.get(ug)
		if popp_tuple is None:
			per_user_lat[ug] = NO_ROUTE_LATENCY
			continue
		popp_i = popp_to_ind.get(popp_tuple)
		if popp_i is None or popp_i in failed_set:
			# Belt-and-suspenders: calculate_ground_truth_ingress already
			# zeroed the failed popp row, so we shouldn't see it back; this
			# guards against any caching surprises.
			per_user_lat[ug] = NO_ROUTE_LATENCY
			continue
		per_user_popp_ind[ug] = popp_i
		per_user_lat[ug] = ug_perfs[ug].get(popp_tuple, NO_ROUTE_LATENCY)
		popp_vol[popp_i] += ug_to_vol.get(ug, 0.0)

	# Capacity: users on a popp whose post-failure total > cap get NO_ROUTE.
	inundated = {i for i in range(sas.n_popps) if popp_vol[i] > caps_arr[i]}
	for ug, popp_i in per_user_popp_ind.items():
		if popp_i in inundated:
			per_user_lat[ug] = NO_ROUTE_LATENCY
	return per_user_lat, inundated


def assess_static_failure_resilience(sas, adv, which='popps', max_failures=None):
	"""Average over single popp (or pop) failures of the BGP-fallback latency.

	Args:
	  sas: SAS instance with calculate_ground_truth_ingress.
	  adv: advertisement matrix (popp x prefix).
	  which: 'popps' (per-popp failures) or 'pops' (kill all popps in one pop).
	  max_failures: optional cap for quick smoke tests.

	Returns dict with:
	  avg_lat_steady         : steady-state weighted average latency.
	  avg_lat_failure        : mean over failures of weighted-avg latency.
	  frac_no_route_failure  : mean over failures of NO_ROUTE volume fraction.
	  worst_lat_failure      : max over failures of weighted-avg latency.
	  per_failure            : list of {failed, avg_lat, frac_no_route}.
	  n_failures             : count.
	"""
	user_to_prefix, _, user_to_lat_steady = _pin_user_to_prefix(sas, adv)
	ug_to_vol = sas.whole_deployment_ug_to_vol
	ugs = list(sas.whole_deployment_ugs)
	vols = np.array([ug_to_vol.get(u, 0.0) for u in ugs])
	total_vol = vols.sum() if vols.sum() > 0 else 1.0

	steady_lats = np.array([user_to_lat_steady[u] for u in ugs])
	avg_lat_steady = float(np.average(steady_lats, weights=vols)) if total_vol > 0 else float('nan')

	# Build list of failure scenarios as (label, [popp_inds_to_fail]) tuples.
	if which == 'popps':
		scenarios = [(i, [i]) for i in range(sas.n_popps)]
	elif which == 'pops':
		pop_to_popp_inds = {}
		for popp_i, (pop, _) in enumerate(sas.popps):
			pop_to_popp_inds.setdefault(pop, []).append(popp_i)
		scenarios = [(pop, inds) for pop, inds in pop_to_popp_inds.items()]
	else:
		raise ValueError("which must be 'popps' or 'pops', got {!r}".format(which))

	if max_failures is not None:
		scenarios = scenarios[:max_failures]

	per_failure = []
	lats_over_failures = []
	no_route_fracs = []
	for label, popp_inds in scenarios:
		per_user_lat, _ = _route_under_failure(sas, adv, popp_inds, user_to_prefix)
		lats = np.array([per_user_lat[u] for u in ugs])
		avg_lat = float(np.average(lats, weights=vols)) if total_vol > 0 else float('nan')
		no_route_mask = (lats >= NO_ROUTE_LATENCY - 1e-9)
		frac_no_route = float(vols[no_route_mask].sum() / total_vol) if total_vol > 0 else 0.0
		per_failure.append({
			'failed': label, 'avg_lat': avg_lat, 'frac_no_route': frac_no_route,
		})
		lats_over_failures.append(avg_lat)
		no_route_fracs.append(frac_no_route)

	return {
		'avg_lat_steady': avg_lat_steady,
		'avg_lat_failure': float(np.mean(lats_over_failures)) if lats_over_failures else float('nan'),
		'frac_no_route_failure': float(np.mean(no_route_fracs)) if no_route_fracs else 0.0,
		'worst_lat_failure': float(np.max(lats_over_failures)) if lats_over_failures else float('nan'),
		'per_failure': per_failure,
		'n_failures': len(per_failure),
	}
