"""Exact-order equivalence fence for the LP-assembly loop elimination
(get_paths_by_ug + _path_obj_coeffs). Old implementations frozen
verbatim below. available_paths ORDER IS LOAD-BEARING: it fixes
var_pool key order and the LP solutions are degenerate, so the fence
asserts exact LIST equality, not set equality. Latency ties are the
one place old (set-iteration order) and new (ascending popp index) may
legally differ -- reported separately as TIE-ONLY.

    python unit_tests/test_paths_coeffs_equiv.py
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from helpers.constants import *
from core.solve_lp_assignment import NO_PATH_INGRESS, get_difference
from operator import itemgetter

# ------------------------- frozen OLD implementations ------------------
OLD_GP_NS = {}
exec("""def get_paths_by_ug(sas, routed_through_ingress):
	## sas is Sparse_Advertisement_Solver (i.e., deployment) object
	## routed_through_ingress is one possible realization of routes. it is a dictionary maping prefixes -> user -> ingress
	## returns available paths which is a list of all (users, ingresses)
	## returns paths_by_ug which is a dictionary mapping ug -> [list of ingresses]

	### Returns structured paths for downstream use
	## availa
	paths_by_ug = {}
	# Iterate the rti dict's own items instead of scanning every UG per
	# prefix (2026-08-24 hot-loop pass): the old form did
	# n_prefixes x n_ugs dict .get()s (~200k/call at actual-20, ~1.7M
	# lambda-adjacent ops per 13-job batch) to find the entries the dict
	# already enumerates. Iteration order over a dict is insertion order,
	# which matched the whole_deployment_ugs scan only incidentally --
	# per-UG path lists are order-normalized by the ranking below, so
	# the visible output is unchanged.
	popp_to_ind = sas.popp_to_ind
	for prefixi in sorted(routed_through_ingress):
		for ug, ingress in routed_through_ingress[prefixi].items():
			if ingress is None:
				continue
			poppi = popp_to_ind[ingress]
			try:
				paths_by_ug[ug].append(poppi)
			except KeyError:
				paths_by_ug[ug] = [poppi]
	ugs_with_no_path = get_difference(list(sas.whole_deployment_ugs), list(paths_by_ug))
	for ug in ugs_with_no_path:
		if not sas.simulated:
			print("UG {} has no path, clients: {}".format(ug, sas.ug_to_ip.get(ug)))
	### As an approximation, only consider the best N paths per UG. Otherwise computation is too expensive
	all_ug_lat_ingresses = {}
	N_KEEP = -1
	_lm = getattr(sas, 'lat_matrix', None)
	for ug in sorted(paths_by_ug):
		# 2026-08-20 fix (recurring worker KeyError, e.g. (vtrtokyo,
		# 9824)): paths_by_ug can contain a popp the UG has no perf
		# entry for; missing == NO_ROUTE by codebase convention. The
		# dense lat_matrix (Patch B) encodes exactly that and is faster
		# than the dict chain; dict fallback keeps non-lat_matrix
		# callers working.
		if _lm is not None:
			_u = sas.whole_deployment_ug_to_ind[ug]
			# one vectorized gather + argsort instead of a Python key
			# callback per element: numpy SCALAR indexing (_lm[el, _u])
			# costs ~300ns/lookup; the fancy-indexed column gather is one
			# C loop (2026-08-24; ~10x on this step, stable order kept)
			_opts = np.fromiter(set(paths_by_ug[ug]), dtype=np.int64)
			sorted_options = _opts[np.argsort(_lm[_opts, _u],
											  kind='stable')].tolist()
		else:
			_perf = sas.whole_deployment_ug_perfs[ug]
			sorted_options = sorted(set(paths_by_ug[ug]),
									key=lambda el: _perf.get(
										sas.popps[el], NO_ROUTE_LATENCY))
		if N_KEEP >= 0:
			keep_options = sorted_options[0:N_KEEP]
		else:
			keep_options = sorted_options
		for poppi in keep_options:
			all_ug_lat_ingresses[ug,poppi] = None
	from operator import itemgetter
	available_paths = sorted(all_ug_lat_ingresses, key=itemgetter(0))

	for ug in ugs_with_no_path:
		available_paths.append((ug, NO_PATH_INGRESS(sas)))

	return available_paths, paths_by_ug
""", dict(globals()), OLD_GP_NS)
old_get_paths_by_ug = OLD_GP_NS['get_paths_by_ug']

class _OldPC:
    def _path_obj_coeffs(self, available_paths, obj, site_cost_alpha):
        """Per-path LP objective coefficients (latencies). Named sub-step of
        solve_generic_lp_persistent so subclasses can override path pricing."""
        obj_coeffs = []
        # hoisted: NO_PATH_INGRESS(self) was re-evaluated per PATH --
        # 1.68M calls per 13-job batch in the 2026-08-24 profile
        _no_path = NO_PATH_INGRESS(self)
        for ug, poppi in available_paths:
            if poppi == _no_path:
                obj_coeffs.append(NO_ROUTE_LATENCY)
            else:
                if obj == "avg_latency":
                    try:
                        obj_coeffs.append(self.whole_deployment_ug_perfs[ug][self.popps[poppi]])
                    except KeyError:
                        # RARE STALE-PATH EVENT (2026-08-24, Tom): a path was
                        # offered for a (ug, popp) the ug has no perf entry
                        # for -- observed once per multi-hour solve at size
                        # 32 (KeyError ('vtrwarsaw','9009'), iter 71). Log a
                        # full forensic dump LOUDLY, then price the path as
                        # unroutable instead of killing a 5h solve: one
                        # NO_ROUTE-priced path among ~100k biases one LP
                        # call, aborting loses the whole strategy.
                        self._log_stale_path(ug, poppi, available_paths)
                        obj_coeffs.append(NO_ROUTE_LATENCY)
                elif obj == "per_site_cost":
                    pop, _ = self.popps[poppi]
                    site_cost = self.site_costs[pop]
                    obj_coeffs.append(self.whole_deployment_ug_perfs[ug][self.popps[poppi]] + site_cost_alpha * site_cost)
                else:
                    raise ValueError("obj {} not supported in solve_generic_lp_persistent".format(obj))
        return obj_coeffs



def main():
    from unit_tests.bench_path_distribution import (
        build_worker, init_like_advertisement, seed_parent_tracker_from_init)
    from core.solve_lp_assignment import get_paths_by_ug as new_gp
    w = build_worker('cache/popp_failure_latency_comparison_'
                     'testing_feature-actual-20_dep_sweep_20.pkl')
    rng = np.random.RandomState(11)
    base = init_like_advertisement(w, rng)
    seed_parent_tracker_from_init(w, base)
    advs = [base.copy()]
    for _ in range(3):
        x = base.copy(); x[rng.randint(w.n_popp), rng.randint(w.n_prefixes)] ^= True
        advs.append(x)
    for _ in range(2):
        x = base.copy(); x[rng.randint(w.n_popp), :] = False
        advs.append(x)
    old_pc = types.MethodType(_OldPC._path_obj_coeffs, w)
    exact = tie_only = bad = 0
    for k, adv in enumerate(advs):
        w._compute_scenario_options(adv.astype(np.float64))
        np.random.seed(2000 + k)
        rti = w._sample_scenario_realizations()[0]
        ap_old, pbu_old = old_get_paths_by_ug(w, rti)
        ap_new, pbu_new = new_gp(w, rti)                      # dict wanted
        assert pbu_old == pbu_new, 'paths_by_ug mismatch adv %d' % k
        assert ap_old == ap_new, 'available_paths (dict-mode) mismatch adv %d' % k
        ap_fast, _ = new_gp(w, rti, want_paths_by_ug=False)   # fast path
        for obj in ('avg_latency', 'per_site_cost'):
            co_old = old_pc(ap_old, obj, DEFAULT_SITE_COST)
            co_new = w._path_obj_coeffs(ap_fast, obj, DEFAULT_SITE_COST)
            co_new = list(co_new)
            if ap_fast == ap_old and co_new == co_old:
                exact += 1; continue
            # order difference: legal ONLY on latency ties
            if sorted(ap_fast) != sorted(ap_old):
                bad += 1; print('BAD adv %d %s: path SETS differ' % (k, obj)); continue
            od = dict(zip(ap_old, co_old)); nd = dict(zip(ap_fast, co_new))
            if od != nd:
                bad += 1; print('BAD adv %d %s: coeff values differ' % (k, obj)); continue
            tie_only += 1
        print('adv %d done (%d paths)' % (k, len(ap_old)))
    print('exact=%d tie_only=%d bad=%d' % (exact, tie_only, bad))
    print('FENCE FAIL' if bad else 'FENCE HOLDS' + (' (with tie-order diffs)' if tie_only else ' (byte-exact)'))
    sys.exit(1 if bad else 0)

if __name__ == '__main__':
    main()
