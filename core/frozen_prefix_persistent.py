"""Persistent frozen_prefix LP with a basis WARM START across probes (Tom
2026-09-11: "implement both" -- this and the column generation in
core/frozen_prefix.py).

A worker prices hundreds of gradient probes per iteration, each one
advertisement entry away from the last. The lifted LP (core/frozen_prefix.py)
rebuilds and cold-solves ~200k columns per probe at size 32. Here one HiGHS
model lives per worker process and kill set; a probe edits only what it
changed and the dual simplex restarts from the incumbent basis.

Fixed row set (so edits never touch rows):
  cons    one per user: sum of its pair volumes = its volume (0 if unroutable)
  loaddef one per popp: L_j - sum_{pairs won by j} x = 0
  normcap one per popp: L_j - o0_j <= cap_j
  aff     one per (scenario k, popp j): L_j + (volume falling onto j in k) - o_kj <= cap_j
          disabled (upper = +inf, o_kj cost 0) where j is dead in k
Every alive scenario prices its own overflow o_kj (weight w_c p_c), so the
normal overflow o0 carries only the normal scenario -- same optimum as the
lifted model's shared-o0 bookkeeping, without per-probe row churn.

Columns: [L | o0 | o_aff | prefix-0 block | prefix-1 block | ...]. A prefix
block holds one column per user (unroutable pairs: bounds 0, no entries). A
probe changes the routed ingress of one or two prefixes: those blocks are
deleted and re-added at the end (HiGHS keeps the basis of every other column;
new columns enter nonbasic), block offsets are re-derived, and the model is
re-run. Objective / diagnostics are the same quantities the lifted model
reports.

Used for training probes only: the exhaustive evaluation pin (K ~ n_popps
scenarios) would need K x n_popps aff rows and goes through the lifted model.
Levers: frozen_persistent (SCULPTOR_FROZEN_PREFIX_PERSISTENT, default 1),
frozen_persistent_max_k (max scenarios; default 64).
"""
import os
import time

import numpy as np

from helpers.constants import NO_ROUTE_LATENCY
from helpers.helpers import threshold_a
from core.frozen_prefix import (frozen_fallbacks, frozen_fallbacks_killed,
								kill_scenarios_from_kwargs, kill_scenarios_as_list,
								_knob, _penalty_sum_mode)

_MODELS = {}      # cache key -> _PersistentFrozenLP (one per process)
_MAX_MODELS = 2   # a worker alternates between at most a couple of kill sets


def _lever_values(kwargs):
	return dict(
		gamma=_knob(kwargs, 'frozen_gamma', 'SCULPTOR_FROZEN_PREFIX_GAMMA', 1.0),
		n_fail=_knob(kwargs, 'frozen_n_fail', 'SCULPTOR_FROZEN_PREFIX_N_FAIL', 20, int),
		p_nr=_knob(kwargs, 'frozen_no_route_penalty', 'SCULPTOR_FROZEN_PREFIX_NO_ROUTE_PENALTY', 50.0),
		p_c=_knob(kwargs, 'frozen_congestion_penalty', 'SCULPTOR_FROZEN_PREFIX_CONGESTION_PENALTY', 25.0),
		lat_scale=_knob(kwargs, 'frozen_lat_scale', 'SCULPTOR_FROZEN_PREFIX_LAT_SCALE', 1.0),
		cap_headroom=_knob(kwargs, 'frozen_cap_headroom', 'SCULPTOR_FROZEN_PREFIX_CAP_HEADROOM', 1.0),
		time_limit=_knob(kwargs, 'frozen_time_limit', 'SCULPTOR_FROZEN_PREFIX_TIME_LIMIT', 30.0),
		penalty_sum=_penalty_sum_mode(kwargs),
	)


def persistent_enabled(kwargs, n_scenarios):
	on = _knob(kwargs, 'frozen_persistent', 'SCULPTOR_FROZEN_PREFIX_PERSISTENT', 1, int)
	max_k = _knob(kwargs, 'frozen_persistent_max_k', 'SCULPTOR_FROZEN_PREFIX_PERSISTENT_MAX_K', 64, int)
	if not on or n_scenarios > max_k or n_scenarios == 0:
		return False
	try:
		import highspy  # noqa: F401
	except Exception:
		return False
	import core.gpshim as gp
	return getattr(gp, 'BACKEND', 'highs') == 'highs'


def solve_persistent(sas, routed_through_ingress, obj, **kwargs):
	"""Entry point used by core.frozen_prefix._solve_lp_frozen_prefix_impl."""
	adv = kwargs.get('adv')
	lv = _lever_values(kwargs)
	scenarios = kill_scenarios_from_kwargs(sas, kwargs, lv['n_fail'])
	adv = threshold_a(np.asarray(adv, dtype=float))
	key = (id(sas), sas.n_popps, adv.shape[1], tuple(scenarios),
		   tuple(sorted((k, v) for k, v in lv.items() if k != 'time_limit')))
	m = _MODELS.get(key)
	if m is None:
		if len(_MODELS) >= _MAX_MODELS:
			_MODELS.pop(next(iter(_MODELS)))
		m = _PersistentFrozenLP(sas, adv.shape[1], scenarios, lv)
		_MODELS[key] = m
	m.time_limit = lv['time_limit']
	return m.solve(routed_through_ingress, adv)


class _PersistentFrozenLP:
	def __init__(self, sas, n_prefixes, scenarios, lv):
		import highspy as hp
		self.hp = hp
		self.sas = sas
		self.lv = lv
		self.n_prefixes = int(n_prefixes)
		self.scenarios = list(scenarios)
		self.n_popps = n_popps = sas.n_popps
		self.n_ug = n_ug = sas.whole_deployment_n_ug
		self.ugs = sas.whole_deployment_ugs
		self.ug_to_ind = sas.whole_deployment_ug_to_ind
		self.ug_perfs = sas.whole_deployment_ug_perfs
		self.popp_to_ind = sas.popp_to_ind
		self.popps = sas.popps
		self.vols = np.asarray([sas.whole_deployment_ug_to_vol[u] for u in self.ugs], dtype=float)
		self.total_vol = float(self.vols.sum()) or 1.0
		self.caps = np.asarray(sas.link_capacities_arr, dtype=float).flatten() * lv['cap_headroom']
		K = self.K = len(self.scenarios)
		gamma, ps = lv['gamma'], lv['penalty_sum']
		self.w_k = gamma / K if K else 0.0
		self.w_nr = (gamma if ps in ('no_route', 'both') else self.w_k) if K else 0.0
		self.w_c = (gamma if ps == 'both' else self.w_k) if K else 0.0
		self.kill_count = np.zeros(n_popps, dtype=int)
		self.kpos1 = np.full(n_popps, -1, dtype=int)
		self.group_scen = []
		dead = np.zeros((K, n_popps), dtype=bool)
		for si, sc in enumerate(self.scenarios):
			self.kill_count[list(sc)] += 1
			dead[si, list(sc)] = True
			if len(sc) == 1:
				self.kpos1[sc[0]] = si
			else:
				self.group_scen.append((si, sc))
		self.dead = dead
		self.time_limit = lv['time_limit']
		V = self.total_vol
		# ---- rows
		self.r_cons, self.r_load, self.r_cap, self.r_aff = 0, n_ug, n_ug + n_popps, n_ug + 2 * n_popps
		n_rows = self.r_aff + K * n_popps
		lower = np.empty(n_rows); upper = np.empty(n_rows)
		lower[self.r_cons:self.r_load] = 0.0; upper[self.r_cons:self.r_load] = 0.0   # set per update
		lower[self.r_load:self.r_cap] = 0.0; upper[self.r_load:self.r_cap] = 0.0
		lower[self.r_cap:self.r_aff] = -hp.kHighsInf; upper[self.r_cap:self.r_aff] = self.caps
		aff_upper = np.tile(self.caps, K)
		aff_upper[dead.reshape(-1)] = hp.kHighsInf
		lower[self.r_aff:] = -hp.kHighsInf; upper[self.r_aff:] = aff_upper
		h = self.h = hp.Highs()
		h.setOptionValue('output_flag', False)
		h.setOptionValue('threads', 1)
		h.setOptionValue('solver', 'simplex')
		h.setOptionValue('simplex_strategy', 1)     # dual: warm-start friendly
		h.addRows(n_rows, lower, upper, 0, np.zeros(n_rows + 1, dtype=np.int32),
				  np.array([], dtype=np.int32), np.array([], dtype=np.float64))
		# ---- structural columns: L | o0 | o_aff
		j = np.arange(n_popps)
		cols = []   # (cost, lb, ub, rows, vals)
		alive_rows_by_j = [self.r_aff + np.flatnonzero(~dead[:, jj]) * n_popps + jj for jj in range(n_popps)]
		for jj in range(n_popps):
			rows = np.concatenate([[self.r_load + jj, self.r_cap + jj], alive_rows_by_j[jj]]).astype(np.int32)
			cols.append((0.0, 0.0, hp.kHighsInf, rows, np.ones(len(rows))))
		for jj in range(n_popps):
			cols.append((lv['p_c'] / V, 0.0, hp.kHighsInf, np.array([self.r_cap + jj], dtype=np.int32), np.array([-1.0])))
		for si in range(K):
			for jj in range(n_popps):
				# dead (k, j): disabled row, and the overflow variable pinned at 0 so
				# a free zero-cost column can't pollute the per-scenario overflow sums
				if dead[si, jj]:
					cols.append((0.0, 0.0, 0.0, np.array([self.r_aff + si * n_popps + jj], dtype=np.int32), np.array([-1.0])))
				else:
					cols.append((self.w_c * lv['p_c'] / V, 0.0, hp.kHighsInf, np.array([self.r_aff + si * n_popps + jj], dtype=np.int32), np.array([-1.0])))
		self._add_cols(cols)
		self.c_base = 2 * n_popps + K * n_popps
		self.iL, self.iO0, self.iOA = 0, n_popps, 2 * n_popps
		# ---- pair blocks
		self.block_start = {}          # prefix -> first model column
		self.block_order = []          # prefixes in model order
		self.rti_cache = {}            # prefix -> dict ug -> popp tuple (identity of the block)
		self.pdata = {}                # prefix -> per-block arrays for diagnostics
		self.n_routable = np.zeros(n_ug, dtype=int)
		self.cons_set = np.zeros(n_ug, dtype=bool)
		self.n_solves = 0
		self.n_blocks_rebuilt = 0

	# ---------------------------------------------------------------- columns
	def _add_cols(self, cols):
		n = len(cols)
		costs = np.asarray([c[0] for c in cols], dtype=float)
		lbs = np.asarray([c[1] for c in cols], dtype=float)
		ubs = np.asarray([c[2] for c in cols], dtype=float)
		lens = np.asarray([len(c[3]) for c in cols], dtype=np.int64)
		starts = np.zeros(n + 1, dtype=np.int32); starts[1:] = np.cumsum(lens)
		idx = np.concatenate([c[3] for c in cols]).astype(np.int32) if n else np.array([], dtype=np.int32)
		val = np.concatenate([c[4] for c in cols]).astype(np.float64) if n else np.array([])
		self.h.addCols(n, costs, lbs, ubs, int(starts[-1]), starts, idx, val)

	def _add_block(self, costs, ubs, starts, idx, val):
		n = len(costs)
		self.h.addCols(n, costs, np.zeros(n), ubs, int(starts[-1]), starts.astype(np.int32),
					   idx.astype(np.int32), val.astype(np.float64))

	def _delete_block(self, p):
		start = self.block_start.pop(p)
		n = self.n_ug
		self.h.deleteCols(n, np.arange(start, start + n, dtype=np.int32))
		self.block_order.remove(p)
		for q in self.block_order:
			if self.block_start[q] > start:
				self.block_start[q] -= n

	# ----------------------------------------------------------- pair data
	def _prefix_data(self, p, rti_p, adv):
		"""Per-user arrays for prefix p: winner popp (-1 unroutable), normal
		latency, column cost, and the (row, coef) entries + moved entries."""
		n_ug, n_popps, K, lv = self.n_ug, self.n_popps, self.K, self.lv
		winner = np.full(n_ug, -1, dtype=int)
		lat = np.zeros(n_ug)
		for ug, popp_tuple in rti_p.items():
			pi = self.popp_to_ind.get(popp_tuple)
			if pi is None:
				continue
			l = self.ug_perfs.get(ug, {}).get(popp_tuple)
			if l is None or l >= NO_ROUTE_LATENCY:
				continue
			ugi = self.ug_to_ind[ug]
			winner[ugi] = pi; lat[ugi] = float(l)
		routable = np.flatnonzero(winner >= 0)
		n_r = len(routable)
		# moved entries (pair, scenario, fallback) for this block
		mv_pair, mv_scen, mv_fb = [], [], []
		if n_r:
			pp = np.full(n_r, p, dtype=int)
			bw = winner[routable]
			fb1 = frozen_fallbacks(self.sas, adv, pp, routable, bw)
			m1 = np.flatnonzero(self.kpos1[bw] >= 0)
			if len(m1):
				mv_pair.append(routable[m1]); mv_scen.append(self.kpos1[bw[m1]]); mv_fb.append(fb1[m1])
			for si, sc in self.group_scen:
				idx, fbs = frozen_fallbacks_killed(self.sas, adv, pp, routable, bw, sc)
				if len(idx):
					mv_pair.append(routable[idx]); mv_scen.append(np.full(len(idx), si, dtype=int)); mv_fb.append(fbs)
		if mv_pair:
			mv_pair = np.concatenate(mv_pair); mv_scen = np.concatenate(mv_scen); mv_fb = np.concatenate(mv_fb)
		else:
			mv_pair = np.zeros(0, dtype=int); mv_scen = np.zeros(0, dtype=int); mv_fb = np.zeros(0, dtype=int)
		mv_fb_lat = np.zeros(len(mv_pair))
		for e in np.flatnonzero(mv_fb >= 0):
			l = self.ug_perfs[self.ugs[mv_pair[e]]].get(self.popps[mv_fb[e]], NO_ROUTE_LATENCY)
			if l >= NO_ROUTE_LATENCY:
				mv_fb[e] = -1
			else:
				mv_fb_lat[e] = float(l)
		mv_live = mv_fb >= 0
		n_moved = np.bincount(mv_pair, minlength=n_ug)
		V = self.total_vol
		cost = np.zeros(n_ug)
		cost[routable] = (1.0 + self.w_k * (K - n_moved[routable])) * lat[routable] * lv['lat_scale'] / V
		if mv_live.any():
			cost += np.bincount(mv_pair[mv_live], weights=self.w_k * mv_fb_lat[mv_live] * lv['lat_scale'], minlength=n_ug) / V
		if (~mv_live).any():
			cost += np.bincount(mv_pair[~mv_live], weights=np.full(int((~mv_live).sum()), self.w_nr * lv['p_nr']), minlength=n_ug) / V
		# column entries: cons row (+1), loaddef row of winner (-1), aff rows (+1) for live moved entries
		ent_col, ent_row, ent_val = [], [], []
		ent_col.append(routable); ent_row.append(self.r_cons + routable); ent_val.append(np.ones(n_r))
		ent_col.append(routable); ent_row.append(self.r_load + winner[routable]); ent_val.append(-np.ones(n_r))
		if mv_live.any():
			ent_col.append(mv_pair[mv_live]); ent_row.append(self.r_aff + mv_scen[mv_live] * n_popps + mv_fb[mv_live]); ent_val.append(np.ones(int(mv_live.sum())))
		ent_col = np.concatenate(ent_col); ent_row = np.concatenate(ent_row); ent_val = np.concatenate(ent_val)
		order = np.argsort(ent_col, kind='stable')
		ent_col, ent_row, ent_val = ent_col[order], ent_row[order], ent_val[order]
		counts = np.bincount(ent_col, minlength=n_ug)
		starts = np.zeros(n_ug + 1, dtype=np.int64); starts[1:] = np.cumsum(counts)
		ubs = np.where(winner >= 0, self.hp.kHighsInf, 0.0)
		return dict(winner=winner, lat=lat, cost=cost, ubs=ubs, starts=starts, idx=ent_row, val=ent_val,
					mv_pair=mv_pair, mv_scen=mv_scen, mv_fb=mv_fb, mv_fb_lat=mv_fb_lat, mv_live=mv_live)

	# --------------------------------------------------------------- update
	def update(self, rti, adv):
		changed = 0
		for p in range(self.n_prefixes):
			rti_p = rti.get(p, {})
			old = self.rti_cache.get(p)
			if old is not None and old == rti_p:
				continue
			d = self._prefix_data(p, rti_p, adv)
			prev = self.pdata.get(p)
			if prev is not None:
				self.n_routable -= (prev['winner'] >= 0)
				self._delete_block(p)
			self.n_routable += (d['winner'] >= 0)
			self.block_start[p] = self.h.getNumCol()
			self.block_order.append(p)
			self._add_block(d['cost'], d['ubs'], d['starts'], d['idx'], d['val'])
			self.pdata[p] = d
			self.rti_cache[p] = dict(rti_p)
			changed += 1
		# conservation rhs: users with a routable pair carry their volume
		want = self.n_routable > 0
		flip = np.flatnonzero(want != self.cons_set)
		if len(flip):
			rhs = np.where(want[flip], self.vols[flip], 0.0)
			self.h.changeRowsBounds(len(flip), (self.r_cons + flip).astype(np.int32), rhs, rhs)
			self.cons_set[flip] = want[flip]
		self.n_blocks_rebuilt += changed
		return changed

	# ----------------------------------------------------------------- solve
	def solve(self, rti, adv):
		from core.solve_lp_assignment import obj_round
		hp = self.hp
		ts = time.time()
		changed = self.update(rti, adv)
		unroutable_vol = float(self.vols[self.n_routable == 0].sum())
		K, V, lv = self.K, self.total_vol, self.lv
		const_term = (1.0 + self.w_nr * K) * lv['p_nr'] * unroutable_vol / V
		self.h.setOptionValue('time_limit', float(self.time_limit))
		self.h.run()
		self.n_solves += 1
		st = self.h.getModelStatus()
		if st != hp.HighsModelStatus.kOptimal:
			return {'solved': False, 'frozen_prefix_persistent_status': str(st)}
		sol = self.h.getSolution()
		z = np.asarray(sol.col_value)
		raw_objval = float(self.h.getInfo().objective_function_value)
		n_popps, n_ug = self.n_popps, self.n_ug
		o0 = z[self.iO0:self.iO0 + n_popps]
		oa = z[self.iOA:self.iOA + K * n_popps].reshape(K, n_popps) if K else np.zeros((0, n_popps))
		# ---- standard-contract reporting from the NORMAL scenario
		lats_by_ug_arr = np.zeros(n_ug)
		lats_by_ug_arr[self.n_routable == 0] = NO_ROUTE_LATENCY
		paths_by_ug = {}
		vols_by_poppi = np.zeros(n_popps)
		lat_sum = np.zeros(n_ug); vol_sum = np.zeros(n_ug)
		pairs_out = []
		tot_lat_mass = 0.0; tot_x = 0.0
		lost_lat = np.zeros(K); gain_lat = np.zeros(K); dead_vol = np.zeros(K)
		for p in self.block_order:
			d = self.pdata[p]
			xv = z[self.block_start[p]:self.block_start[p] + n_ug]
			nz = np.flatnonzero(xv > 1e-12)
			if len(nz) == 0:
				continue
			w = d['winner'][nz]
			vols_by_poppi += np.bincount(w, weights=xv[nz], minlength=n_popps)
			lat_sum[nz] += xv[nz] * d['lat'][nz]; vol_sum[nz] += xv[nz]
			for i in nz:
				paths_by_ug.setdefault(int(i), []).append((int(d['winner'][i]), float(xv[i] / self.vols[i])))
				pairs_out.append((self.ugs[i], int(p), float(xv[i])))
			tot_lat_mass += float(np.sum(xv[nz] * d['lat'][nz])); tot_x += float(xv[nz].sum())
			if K and len(d['mv_pair']):
				xm = xv[d['mv_pair']]
				lost_lat += np.bincount(d['mv_scen'], weights=xm * d['lat'][d['mv_pair']], minlength=K)
				live = d['mv_live']
				gain_lat += np.bincount(d['mv_scen'][live], weights=xm[live] * d['mv_fb_lat'][live], minlength=K)
				dead_vol += np.bincount(d['mv_scen'][~live], weights=xm[~live], minlength=K)
		routed = vol_sum > 0
		lats_by_ug_arr[routed] = lat_sum[routed] / vol_sum[routed]
		congested = vols_by_poppi > self.caps / max(lv['cap_headroom'], 1e-9) + 1e-9
		fraction_congested_volume = float(vols_by_poppi[congested].sum()) / V
		if K:
			live_vol = tot_x - dead_vol
			lat_mass = tot_lat_mass - lost_lat + gain_lat
			fail_lat = np.where(live_vol > 0, lat_mass / np.maximum(live_vol, 1e-12), 0.0)
			fail_nr = (dead_vol + unroutable_vol) / V
			fail_ovf = oa.sum(axis=1) / V
		else:
			fail_lat = fail_nr = fail_ovf = np.zeros(0)
		obj_val = raw_objval + const_term
		return {
			'objective': obj_round(-obj_val),
			'raw_solution': None,
			'paths_by_ug': paths_by_ug,
			'lats_by_ug': lats_by_ug_arr,
			'available_paths': [],
			'solved': 2,
			'vols_by_poppi': {pi: round(float(vols_by_poppi[pi]) / float(self.caps[pi] / max(lv['cap_headroom'], 1e-9)), 2)
							  for pi in range(n_popps)},
			'fraction_congested_volume': fraction_congested_volume,
			'frozen_prefix_n_fail': K,
			'frozen_prefix_kill_popps': kill_scenarios_as_list(self.scenarios),
			'frozen_prefix_n_site_fail': len(self.group_scen),
			'frozen_prefix_gamma': lv['gamma'],
			'frozen_prefix_no_route_penalty': lv['p_nr'],
			'frozen_prefix_congestion_penalty': lv['p_c'],
			'frozen_prefix_lat_scale': lv['lat_scale'],
			'frozen_prefix_cap_headroom': lv['cap_headroom'],
			'frozen_prefix_formulation': 'persistent',
			'frozen_prefix_penalty_sum': lv['penalty_sum'],
			'frozen_prefix_n_rows': int(self.h.getNumRow()),
			'frozen_prefix_n_vars': int(self.h.getNumCol()),
			'frozen_prefix_blocks_rebuilt': changed,
			'frozen_prefix_persistent_solves': self.n_solves,
			'frozen_prefix_normal_lat': tot_lat_mass / max(tot_x, 1e-9),
			'frozen_prefix_fail_lat_mean': float(np.mean(fail_lat)) if len(fail_lat) else 0.0,
			'frozen_prefix_fail_no_route_frac': float(np.mean(fail_nr)) if len(fail_nr) else 0.0,
			'frozen_prefix_fail_overflow_frac': float(np.mean(fail_ovf)) if len(fail_ovf) else 0.0,
			'frozen_prefix_normal_overflow_frac': float(np.sum(o0)) / V,
			'frozen_prefix_unroutable_frac': unroutable_vol / V,
			'frozen_prefix_lp_secs': time.time() - ts,
			'frozen_prefix_pairs': pairs_out,
		}
