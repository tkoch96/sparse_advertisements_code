"""Two-path frozen oracle (Tom 2026-09-06).

A resilient upper anchor for the frozen-failure setting that one-per-peering
FAILS to be (OPP gives each prefix a single popp -> no backup -> strands
users on failure). With perfect knowledge, give EACH USER a dedicated prefix
advertising a PRIMARY popp (good latency) and a BACKUP popp at a DIFFERENT
site (ok latency). Under any single popp/site failure the user's volume that
was on the primary fails over to the backup -- so NO user is ever stranded
(0% no-route by construction for single failures). Choose the (primary,
backup) assignment to minimize steady latency subject to capacity in the
NORMAL scenario AND every single-failure scenario.

This is an oracle (one prefix per user is even more prefixes than OPP), used
only as an achievable-ceiling reference for the frozen-failure columns.

LP (relaxation of the assignment; each user fractionally split over its
candidate ordered (primary, backup) pairs):

  vars   y[u, (p1, p2)] >= 0            p1 != p2, different pops, both
                                        reachable by u; restricted to the
                                        user's top-K lowest-latency popps
  min    sum  y * lat(u, p1)            + P_c/V * sum_f overflow_f          # steady latency + soft congestion
  s.t.   sum_{p2} y[u,(p1,p2)] summed over pairs == vol_u                   # all volume placed
         normal:      load_p^0 <= cap_p                                     # p1 traffic
         failure f:   load_p^f <= cap_p                                     # p1 (if p1!=f) + p2 (if p1==f)

Congestion is priced softly (overflow vars) rather than hard-constrained so
the LP stays feasible under tight capacity and reports a real % congested.
"""
import numpy as np
from scipy.sparse import csr_matrix, hstack as sp_hstack, identity as sp_identity

from helpers.constants import NO_ROUTE_LATENCY
import core.gpshim as gp


def _clear(model):
	try:
		h = getattr(model, '_h', None)
		if h is not None:
			h.clear()
	except Exception:
		pass


def two_path_oracle(sas, topk=6, congestion_penalty=25.0, which='popps',
					include_failures=True):
	"""Return frozen-failure metrics for the two-path oracle, matching the
	shape of core.frozen_prefix_eval.frozen_failure_metrics:
	  steady_latency_ms, fail_latency_ms, fail_frac_cong, fail_frac_no_route,
	  worst_frac_cong, worst_frac_no_route, n_failures.
	"""
	ug_perfs = sas.whole_deployment_ug_perfs
	ug_to_vol = sas.whole_deployment_ug_to_vol
	popp_to_ind = sas.popp_to_ind
	popps = sas.popps
	n_popps = sas.n_popps
	caps = np.asarray(sas.link_capacities_arr, dtype=float).flatten()
	total_vol = float(sum(ug_to_vol.values())) or 1.0

	pop_of = np.array([popps[i][0] for i in range(n_popps)])

	# failure scenarios (indices grouped)
	if which == 'popps':
		scen = [[i] for i in range(n_popps)]
	elif which == 'pops':
		pop_to_inds = {}
		for pi, (pop, _) in enumerate(popps):
			pop_to_inds.setdefault(pop, []).append(pi)
		scen = list(pop_to_inds.values())
	else:
		raise ValueError(which)
	n_scen = len(scen)
	# map each popp -> which scenario index kills it (for constraint building)
	popp_killed_in = {}
	for si, inds in enumerate(scen):
		for pi in inds:
			popp_killed_in.setdefault(pi, []).append(si)

	# Build candidate ordered pairs per user from top-K lowest-latency popps.
	pair_u = []      # ug index (into keep list)
	pair_p1 = []
	pair_p2 = []
	pair_lat = []
	pair_vol = []
	ug_list = []
	unroutable_vol = 0.0
	for ug, perfs in ug_perfs.items():
		vol = ug_to_vol.get(ug, 0.0)
		if vol <= 0:
			continue
		cands = []
		for popp_tuple, lat in perfs.items():
			pi = popp_to_ind.get(popp_tuple)
			if pi is None or lat >= NO_ROUTE_LATENCY:
				continue
			cands.append((float(lat), pi))
		cands.sort()
		cands = cands[:topk]
		if len(cands) < 1:
			unroutable_vol += vol
			continue
		ui = len(ug_list)
		ug_list.append(ug)
		made = False
		for a_i in range(len(cands)):
			la, p1 = cands[a_i]
			# backup: best-latency candidate at a DIFFERENT pop
			for b_i in range(len(cands)):
				if b_i == a_i:
					continue
				lb, p2 = cands[b_i]
				if pop_of[p2] == pop_of[p1]:
					continue
				pair_u.append(ui); pair_p1.append(p1); pair_p2.append(p2)
				pair_lat.append(la); pair_vol.append(vol)
				made = True
		if not made:
			# no distinct-pop backup available: single-popp primary only
			# (this user CAN be stranded -- counts toward no-route)
			la, p1 = cands[0]
			pair_u.append(ui); pair_p1.append(p1); pair_p2.append(-1)
			pair_lat.append(la); pair_vol.append(vol)
	n_pairs = len(pair_u)
	n_users = len(ug_list)
	if n_pairs == 0:
		return {'steady_latency_ms': float('nan'), 'fail_latency_ms': float('nan'),
				'fail_frac_cong': 0.0, 'fail_frac_no_route': unroutable_vol/total_vol,
				'worst_frac_cong': 0.0, 'worst_frac_no_route': unroutable_vol/total_vol,
				'n_failures': n_scen}

	pair_u = np.asarray(pair_u); pair_p1 = np.asarray(pair_p1)
	pair_p2 = np.asarray(pair_p2); pair_lat = np.asarray(pair_lat)
	user_vol = np.array([ug_to_vol[u] for u in ug_list])

	# ---- conservation: each user's pairs sum to its volume
	consA = csr_matrix((np.ones(n_pairs), (pair_u, np.arange(n_pairs))),
					   shape=(n_users, n_pairs))
	consb = user_vol

	# ---- load rows: normal (scenario 0) + optionally one per failure scenario
	# scenario s occupies rows [ (s)*n_popps , (s+1)*n_popps ), s=0 normal.
	n_cons_scen = n_scen if include_failures else 0
	rows = []; cols = []
	# normal: pair contributes to p1
	rows.append(pair_p1); cols.append(np.arange(n_pairs))
	# failures: for scenario si, pair contributes to p1 unless p1 killed;
	# if p1 killed -> contributes to p2 (if p2 alive), else nowhere (no-route)
	if include_failures:
		for si, killed in enumerate(scen):
			killed_set = set(killed)
			base = (si + 1) * n_popps
			r = []; c = []
			for k in range(n_pairs):
				p1 = pair_p1[k]; p2 = pair_p2[k]
				if p1 not in killed_set:
					r.append(base + p1); c.append(k)
				elif p2 >= 0 and p2 not in killed_set:
					r.append(base + p2); c.append(k)
				# else: stranded this scenario (counts as no-route later)
			if r:
				rows.append(np.asarray(r)); cols.append(np.asarray(c))
	rows = np.concatenate(rows); cols = np.concatenate(cols)
	nload = (1 + n_cons_scen) * n_popps
	loadA = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nload, n_pairs))
	caps_tiled = np.tile(caps, 1 + n_cons_scen)
	c_o = np.full(nload, congestion_penalty / total_vol)
	# steady latency objective on y
	c_y = pair_lat * pair_vol / total_vol  # note: pair_vol==user vol per pair
	# actually latency should weight by ALLOCATED volume (y), and y is volume
	# (not fraction); latency contribution = y * lat / total_vol
	c_y = pair_lat / total_vol

	# stack z = [y ; o]
	consA_z = sp_hstack([consA, csr_matrix((n_users, nload))], format='csr')
	loadA_z = sp_hstack([loadA, -sp_identity(nload, format='csr')], format='csr')
	c_z = np.concatenate([c_y, c_o])

	model = gp.Model()
	model.Params.LogToConsole = 0
	model.Params.TimeLimit = 60.0
	model.Params.Threads = 1
	z = model.addMVar(n_pairs + nload, lb=0)
	model.addConstr(consA_z @ z == consb)
	model.addConstr(loadA_z @ z <= caps_tiled)
	model.setObjective(c_z @ z)
	model.optimize()
	if model.status != 2:
		_clear(model)
		return {'steady_latency_ms': float('nan'), 'fail_latency_ms': float('nan'),
				'fail_frac_cong': float('nan'), 'fail_frac_no_route': float('nan'),
				'worst_frac_cong': float('nan'), 'worst_frac_no_route': float('nan'),
				'n_failures': n_scen, 'solved': False}
	zv = np.asarray(z.X).flatten()
	yv = zv[:n_pairs]
	_clear(model)

	# ---- report the three metrics per scenario from the solved y
	def scen_metrics(killed_set):
		load = np.zeros(n_popps)
		lat_num = 0.0; routed = 0.0; noroute = 0.0
		# first pass: loads
		dest = np.full(n_pairs, -1)
		for k in range(n_pairs):
			p1 = pair_p1[k]; p2 = pair_p2[k]
			if p1 not in killed_set:
				dest[k] = p1
			elif p2 >= 0 and p2 not in killed_set:
				dest[k] = p2
			else:
				noroute += yv[k]
		for k in range(n_pairs):
			if dest[k] >= 0:
				load[dest[k]] += yv[k]
		cong_popp = load > caps + 1e-9
		cong = 0.0
		for k in range(n_pairs):
			d = dest[k]
			if d < 0:
				continue
			if cong_popp[d]:
				cong += yv[k]
			else:
				lat_num += yv[k] * pair_lat[k]
				routed += yv[k]
		return (lat_num/routed if routed > 0 else float('nan'),
				cong/total_vol, (noroute + unroutable_vol)/total_vol)

	steady_lat, _, _ = scen_metrics(set())
	lats=[]; congs=[]; nrs=[]
	for killed in scen:
		l,c,nr = scen_metrics(set(killed))
		lats.append(l); congs.append(c); nrs.append(nr)
	return {
		'steady_latency_ms': steady_lat,
		'fail_latency_ms': float(np.nanmean(lats)),
		'fail_frac_cong': float(np.mean(congs)),
		'fail_frac_no_route': float(np.mean(nrs)),
		'worst_frac_cong': float(np.max(congs)),
		'worst_frac_no_route': float(np.max(nrs)),
		'n_failures': n_scen,
		'n_users': n_users,
		'n_pairs': n_pairs,
	}
