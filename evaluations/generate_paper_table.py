"""generate_paper_table: THE paper table, one command, three pickle layers.

    python generate_paper_table.py --dpsize small --num_training_iter 10 \
        --number_of_deployments 1 --run_id demo1

Does, in order, reusing everything already on disk:
  (a) optimize every solution type under every objective (5 total:
      avg_latency, per_site_cost, max_util, frac_beyond_optimal,
      joint_priority), for the requested deployments x training iters
  (b) compute every evaluation metric (failure, sticky, flash/diurnal
      via batched bisection, priority placement, site-cost loads, ...)
  (c) emit the methods x metrics supersection table: terminal + LaTeX
      (booktabs, ready to paste) + CSV, best-real-method bolded,
      one-per-peering pinned on top as the optimal reference
  (d) if EVERYTHING is already computed: pure pickle loads -> table in
      seconds. Pause-and-resume works at every layer.

PICKLE LAYERS
  L1  per (objective, size, run_id): solves + advertisements + metrics
      cache/popp_failure_latency_comparison_<size>_<run_id>[_<obj>].pkl
      (written incrementally by the eval driver; resume-safe)
  L2  metric families inside L1, individually resumable/forcible
      (SCULPTOR_RECALC=failure,flash,diurnal,volume,pct_vol)
  L3  condensed table cells {group: {method: (mean, std, n)}}:
      cache/paper_table_condensed_<size>_<run_id>.pkl -- the 5-second
      path. Auto-invalidated when any L1 pickle is newer.

FORCE FLAGS (set here, or override via env of the same name)
"""

# ---------------------------------------------------------------- flags --
import os as _os
# re-run solves even if L1 pickles exist (uses a fresh sub-tag; never
# deletes old L1 pickles)
FORCE_RESOLVE = _os.environ.get('FORCE_RESOLVE', '0') == '1'
# comma list of metric families to force (maps to SCULPTOR_RECALC):
# failure, flash, diurnal, volume, pct_vol, or 'all'
FORCE_RECOMPUTE_METRICS = _os.environ.get('FORCE_RECOMPUTE_METRICS', '')
# rebuild the condensed L3 pickle even if fresh
FORCE_REAGGREGATE = _os.environ.get('FORCE_REAGGREGATE', '0') == '1'

import argparse
import os
import pickle
import re
import subprocess
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

CACHE_DIR = os.path.join(_REPO, 'cache')

# ----------------------------------------------------------------- rows --
METHODS = [
    # row order (Tom 2026-08-30): optimal reference, then methods from
    # most to least sophisticated
    ('one_per_peering', 'One-per-peering'),
    ('sparse',          'SCULPTOR'),
    ('painter',         'PAINTER'),
    ('one_per_pop',     'Unicast'),
    ('anyopt',          'AnyOpt'),
    ('anycast',         'Anycast'),
]
# The optimal reference: advertises every prefix, so it bounds what any
# real method can achieve. It renders as a reference row and is EXCLUDED
# from best-in-column marking (Tom 2026-08-22: "opp is optimal so dont
# color that one green, color next most optimal green").
REFERENCE_METHODS = {'One-per-peering'}

OBJECTIVE_ALIASES = {
    'priorities': 'joint_priority',
    'latency': 'avg_latency', 'latency_resilience': 'avg_latency',
    'site_cost': 'per_site_cost', 'mlu': 'max_util',
}
DEFAULT_OBJECTIVES = ['avg_latency', 'per_site_cost', 'max_util',
                      'frac_beyond_optimal', 'joint_priority']
BLOCKED_OBJECTIVES = {}  # joint_priority registered 2026-08-22; none blocked

# -------------------------------------------------------------- columns --
# (objective, label, direction, extractor). direction '<' = lower better.
# Extractors read keys the per-objective eval modules already write; an
# absent key renders '-' rather than inventing a number.


def _vol_weighted_latency(m, sim, soln):
    lats = (m.get('latencies') or {}).get(sim, {}).get(soln)
    vols = (m.get('ug_to_vol') or {}).get(sim)
    if lats is None:
        return None
    lats = np.asarray(lats, dtype=float)
    if not lats.size:
        return None
    if vols is not None and np.size(vols) == np.size(lats):
        vols = np.asarray(vols, dtype=float)
        return float(np.sum(lats * vols) / (np.sum(vols) + 1e-9))
    return float(np.mean(lats))


def _fail_delta(key):
    # entries are (diff, vol, ug, element, perf1, perf2[, paths]) tuples
    # (the canonical consumer is get_failure_metric_arr in
    # evaluations_for_latency_plus_resilience) -> volume-weighted diff
    def _x(m, sim, soln):
        v = (m.get(key) or {}).get(sim, {}).get(soln)
        if not v:
            return None
        num = den = 0.0
        try:
            for fields in v:
                diff, vol = float(fields[0]), float(fields[1])
                num += diff * vol
                den += vol
        except (TypeError, ValueError, IndexError):
            return None
        return num / den if den > 0 else None
    return _x


def _mean_key(key):
    def _x(m, sim, soln):
        v = (m.get(key) or {}).get(sim, {}).get(soln)
        if v is None:
            return None
        try:
            arr = np.asarray(v, dtype=float)
            return float(np.mean(arr)) if arr.size else None
        except (TypeError, ValueError):
            return None
    return _x


def _flash_crowd(m, sim, soln):
    # shape: {Y_val: {X_surge: [(benefit_delta, frac_congested), ...]}}
    # (assess_resilience_to_flash_crowds_mp) -> mean over the surge sweep
    # of the mean congested fraction
    v = (m.get('resilience_to_congestion') or {}).get(sim, {}).get(soln)
    if not isinstance(v, dict):
        return None
    per_x = []
    try:
        for _y, xs in v.items():
            if not isinstance(xs, dict):
                continue
            for _x_, entries in xs.items():
                fr = [float(e[1]) for e in entries if e is not None]
                if fr:
                    per_x.append(float(np.mean(fr)))
    except (TypeError, ValueError, IndexError):
        return None
    return float(np.mean(per_x)) if per_x else None


def _diurnal(m, sim, soln):
    # {'fraction_congested_volume': {capacity_pct: {hour: [frac,...]}}}
    # -> mean congested fraction over every (capacity, hour) leaf
    v = (m.get('diurnal') or {}).get(sim, {}).get(soln)
    if isinstance(v, dict):
        v = v.get('fraction_congested_volume')
    if not isinstance(v, dict):
        return None
    leaves = []
    try:
        for _cap, hours in v.items():
            if not isinstance(hours, dict):
                continue
            for _h, fr in hours.items():
                a = np.asarray(fr, dtype=float)
                if a.size:
                    leaves.append(float(np.mean(a)))
    except (TypeError, ValueError):
        return None
    return float(np.mean(leaves)) if leaves else None


def _final_obj(m, sim, soln):
    """Final objective value the solve reported -- under
    SCULPTOR_GENERIC_OBJECTIVE=X this is X's own scalar, so reading it
    from each objective's cell pickle gives 'final objective value'
    columns for the non-latency objectives (Tom 2026-08-22). Lower is
    better (objectives are minimized)."""
    cr = (m.get('compare_rets') or {}).get(sim) or {}
    v = (cr.get('sparse_objective_vals') or {}).get(soln)
    if not v:
        return None
    try:
        return float(v[0])
    except (TypeError, ValueError, IndexError):
        return None


def _mlu_cell_latency(m, sim, soln):
    """Vol-weighted latency of the MLU-trained advertisement (the MLU
    group's (a) sub-column) -- read from the max_util cell's own pickle."""
    return _vol_weighted_latency(m, sim, soln)


def _site_cost_stat(agg):
    """max/avg cost of the sites the advertisement activates. Computed
    from stored adv + deployment['site_costs'] -- pure aggregation."""
    def _x(m, sim, soln):
        adv = (m.get('adv') or {}).get(sim, {}).get(soln)
        dep = (m.get('deployment') or {}).get(sim) or {}
        costs = dep.get('site_costs') or {}
        popps = dep.get('popps')
        if adv is None or not costs or not popps:
            return None
        a = np.asarray(adv)
        if a.size == 0:
            return None
        on = a > 0.5
        pops_on = sorted({popps[i][0] for i in range(min(len(popps), on.shape[0]))
                          if on[i].any()})
        vals = [float(costs[p]) for p in pops_on if p in costs]
        return float(agg(vals)) if vals else None
    return _x


# Supersection layout (Tom 2026-08-23): row 1 = objective supersections,
# row 2 = that objective's metrics. GROUPS: (supersection label, objective
# whose pickle feeds it, [(sub label, direction, extractor), ...]).
def _lat_split(part):
    """(a)+(c) (Tom 2026-08-23): split each solution's steady-state lats
    into clean / congested / stranded. Sentinel-valued UGs in 'latencies'
    are congested OR stranded; fraction_congested_volume (recorded by the
    solve) identifies the congested share; stranded = sentinel - congested.
    'clean' = vol-weighted latency over NON-sentinel UGs only."""
    def _x(m, sim, soln):
        lats = (m.get('latencies') or {}).get(sim, {}).get(soln)
        vols = (m.get('ug_to_vol') or {}).get(sim)
        if lats is None:
            return None
        lats = np.asarray(lats, dtype=float).flatten()
        if not lats.size:
            return None
        if vols is not None and np.size(vols) == np.size(lats):
            vols = np.asarray(vols, dtype=float).flatten()
        else:
            vols = np.ones_like(lats)
        tv = float(np.sum(vols))
        bad = lats >= 0.99 * 30000.0
        bad_frac = float(np.sum(vols[bad]) / max(tv, 1e-9))
        if part == 'clean':
            ok = ~bad
            if not ok.any():
                return None
            return float(np.sum(lats[ok] * vols[ok]) / max(np.sum(vols[ok]), 1e-9))
        cong = (m.get('fraction_congested_volume') or {}).get(sim, {}).get(soln)
        try:
            # nanmean: stored fractions can carry NaN entries (an unsolved
            # scenario) -- plain mean propagated NaN and the table showed
            # '-' where the true value is 0 (Tom 2026-08-23: dash must
            # mean not-applicable, zeros must print 0)
            cong = float(np.nanmean(np.asarray(cong, dtype=float))) if cong is not None else 0.0
        except (TypeError, ValueError):
            cong = 0.0
        if not np.isfinite(cong):
            cong = 0.0
        cong = min(cong, bad_frac)
        if part == 'congested':
            return cong
        return max(0.0, bad_frac - cong)      # stranded
    return _x


def _stats_key(key, inner=None, scale=1.0):
    """Top-level per-solution stats the suite aggregates ACROSS sims
    (what make_paper_plots reads) -- sim-independent, so every sim
    returns the same value and std collapses to 0."""
    def _x(m, sim, soln):
        v = (m.get(key) or {}).get(soln)
        if v is None:
            return None
        if inner is not None:
            if not isinstance(v, dict):
                return None
            v = v.get(inner)
            if v is None:
                return None
        try:
            if isinstance(v, dict):
                vals = [float(np.mean(np.asarray(x, dtype=float)))
                        for x in v.values()]
                return float(np.mean(vals)) * scale if vals else None
            return float(np.mean(np.asarray(v, dtype=float))) * scale
        except (TypeError, ValueError):
            return None
    return _x


def _lat_res_objective(m, sim, soln):
    """(a) (Tom 2026-08-23): THE latency+resilience objective =
    steady avg latency + gamma * mean over single-popp failures of the
    avg latency DURING that failure, congested/unrouted charged at the
    sentinel. Aggregated from the stored failure entries
    (diff, vol, ug, element, perf_optimal, perf_actual): perf_actual is
    already sentinel-valued for congested/unrouted volume."""
    lats = (m.get('latencies') or {}).get(sim, {}).get(soln)
    vols = (m.get('ug_to_vol') or {}).get(sim)
    if lats is None:
        return None
    lats = np.asarray(lats, dtype=float).flatten()
    if not lats.size:
        return None
    if vols is not None and np.size(vols) == np.size(lats):
        w = np.asarray(vols, dtype=float).flatten()
    else:
        w = np.ones_like(lats)
    steady = float(np.sum(lats * w) / max(np.sum(w), 1e-9))
    entries = (m.get('popp_failures_latency_optimal_specific') or {}).get(
        sim, {}).get(soln)
    if not entries:
        return None
    per_el = {}
    for f in entries:
        el = f[3]
        try:
            per_el.setdefault(el, []).append((float(f[5]), float(f[1])))
        except (TypeError, ValueError, IndexError):
            return None
    fail_avgs = []
    for el, pv in per_el.items():
        tv = sum(v for _p, v in pv)
        if tv > 0:
            fail_avgs.append(sum(p * v for p, v in pv) / tv)
    if not fail_avgs:
        return None
    gamma = float((m.get('settings') or {}).get('gamma', 10.0) or 10.0)
    return steady + gamma * float(np.mean(fail_avgs))


# ----------------------------------------------------------- key table --
# The SECOND, condensed table ("key metrics"): (group, sub-label) pairs
# pulled from GROUPS, in print order. Edit this list to change what the
# key table shows -- labels must match GROUPS exactly.
KEY_COLUMNS = [
    ('Latency + g*Resilience', 'Latency (ms)'),
    ('Latency + g*Resilience', '% cong PoPP-fail'),
    ('Latency + g*Resilience', '% cong PoP-fail'),
    ('Latency + g*Resilience', 'Flash-crowd resilience'),
    ('Latency + g*Resilience', 'Diurnal resilience'),
    ('High + Low Priority Traffic', 'HPrio latency (ms)'),
    ('High + Low Priority Traffic', 'Crit bulk ratio'),
    ('Frac beyond optimal', 'Objective'),
    ('MLU', 'MLU'),
    ('MLU', 'Latency (ms)'),
    ('Site cost', 'Wgt avg site cost'),
]

# Display-text overrides for BOTH tables (stdout, LaTeX and CSV): map an
# original group or sub label to the text you want printed. Examples:
#   'Latency + g*Resilience': r'Latency $+ \gamma\cdot$Resilience',
#   '% cong PoPP-fail': '% congested (PoPP fail)',
HEADER_TEXT = {
}


GROUPS = [
    ('Latency + g*Resilience', 'avg_latency', [
        ('Latency (ms)',            '<', _lat_split('clean')),
        ('Congested vol',           '<', _lat_split('congested')),
        ('Stranded vol',            '<', _lat_split('stranded')),
        ('Subopt normal (ms)',      '<', _stats_key('stats_best_latencies', scale=-1.0)),
        ('Subopt PoPP-fail (ms)',   '<', _stats_key('stats_popp_failures_latency_optimal_specific', 'avg_latency_difference', scale=-1.0)),
        ('% cong PoPP-fail',        '<', _stats_key('stats_popp_failures_latency_optimal_specific', 'frac_vol_congested', scale=100.0)),
        ('Subopt PoP-fail (ms)',    '<', _stats_key('stats_pop_failures_latency_optimal_specific', 'avg_latency_difference', scale=-1.0)),
        ('% cong PoP-fail',         '<', _stats_key('stats_pop_failures_latency_optimal_specific', 'frac_vol_congested', scale=100.0)),
        ('Flash-crowd resilience',  '>', _stats_key('stats_resilience_to_congestion')),
        ('Diurnal resilience',      '>', _stats_key('stats_diurnal')),
        ('Objective (lat+g*RB)',    '<', _lat_res_objective),
    ]),
    ('High + Low Priority Traffic', 'joint_priority', [
        ('Frac HPrio routed',     '>', _mean_key('hprio_frac_routed_by_strategy')),
        ('HPrio latency (ms)',    '<', _mean_key('hprio_latency_by_strategy')),
        ('Crit bulk ratio',       '>', _mean_key('critical_bulk_ratio_by_strategy')),
        ('HPrio cong @SWAN',      '<', _mean_key('hprio_cong_swan_by_strategy')),
        ('Congested vol',         '<', _lat_split('congested')),
        ('Stranded vol',          '<', _lat_split('stranded')),
        ('Objective',             '>', _mean_key('objective_value_by_strategy')),
    ]),
    ('Frac beyond optimal', 'frac_beyond_optimal', [
        ('Objective',             '>', _mean_key('objective_value_by_strategy')),
        ('Congested vol',         '<', _lat_split('congested')),
        ('Stranded vol',          '<', _lat_split('stranded')),
    ]),
    ('MLU', 'max_util', [
        ('Latency (ms)',          '<', _mlu_cell_latency),
        ('MLU',                   '<', _mean_key('mlu_by_strategy')),
        ('Congested vol',         '<', _lat_split('congested')),
        ('Stranded vol',          '<', _lat_split('stranded')),
        ('Objective',             '>', _mean_key('objective_value_by_strategy')),
    ]),
    ('Site cost', 'per_site_cost', [
        ('Wgt max site cost',     '<', _mean_key('max_site_cost_load_by_strategy')),
        ('Wgt avg site cost',     '<', _mean_key('weighted_site_cost_by_strategy')),
        ('Congested vol',         '<', _lat_split('congested')),
        ('Stranded vol',          '<', _lat_split('stranded')),
        ('Objective',             '>', _mean_key('objective_value_by_strategy')),
    ]),
]
# flat view for coverage/build compatibility
COLUMNS = [(obj, '{}|{}'.format(g, lab), d, fn)
           for g, obj, subs in GROUPS for lab, d, fn in subs]


# --------------------------------------------------------------- pickle --

def normalize_dpsize(raw):
    r = str(raw).strip()
    if r.startswith('actual-'):
        r = r[len('actual-'):]
    return r      # '3'/'32' (driver maps to testing_feature-actual-N) or 'small'


def dpsize_str(dpsize):
    return dpsize if isinstance(dpsize, str) and not dpsize.isdigit() \
        else 'testing_feature-actual-{}'.format(dpsize)


def pickle_path(dpsize, run_tag=None):
    base = 'popp_failure_latency_comparison_{}'.format(dpsize_str(dpsize))
    if run_tag:
        base += '_{}'.format(run_tag)
    return os.path.join(CACHE_DIR, base + '.pkl')


# a pickle only counts for an objective if it actually holds that
# objective's metric key -- otherwise the shared avg_latency pickle
# satisfies every fallback and coverage lies "fully covered"
OBJECTIVE_REQUIRED_KEY = {
    'per_site_cost': 'active_sites_by_strategy',
    'max_util': 'mlu_by_strategy',
    'lat_plus_max_util': 'mlu_by_strategy',
    'frac_beyond_optimal': 'frac_within_threshold_by_strategy',
    'joint_priority': 'priority_by_strategy',
}


def load_metrics(dpsize, objective, run_tag=None):
    candidates = []
    if objective != 'avg_latency' and run_tag:
        candidates.append(pickle_path(dpsize, '{}_{}'.format(run_tag, objective)))
    if objective != 'avg_latency':
        candidates.append(pickle_path(dpsize, objective))
    candidates.append(pickle_path(dpsize, run_tag))
    if not run_tag:
        # the untagged pickle may be MONTHS old; only trust it when the
        # caller explicitly opted out of namespacing
        candidates.append(pickle_path(dpsize, None))
    need = OBJECTIVE_REQUIRED_KEY.get(objective)
    for p in candidates:
        if os.path.exists(p):
            try:
                m = pickle.load(open(p, 'rb'))
            except Exception as e:
                print('  !! unreadable {}: {}'.format(p, e))
                continue
            if need and not m.get(need):
                continue        # right file family, wrong objective
            return m, p
    return None, candidates[0]


# ------------------------------------------------------------- coverage --

def coverage(dpsize, objectives, nsim_target, run_tag, tag_overrides=None):
    out = {}
    for obj in objectives:
        _full = (tag_overrides or {}).get(obj)
        if _full:
            # FORCE_RESOLVE renamed this objective's tag mid-run; load the
            # exact pickle the cell just wrote (2026-08-25)
            p = pickle_path(dpsize, _full)
            try:
                m = pickle.load(open(p, 'rb'))
            except (FileNotFoundError, EOFError):
                m = None
        else:
            m, p = load_metrics(dpsize, obj, run_tag)
        n = 0
        failed = []
        if m:
            advs = m.get('adv') or {}
            for sim, soldict in (advs.items() if isinstance(advs, dict)
                                 else enumerate(advs)):
                if isinstance(soldict, dict) and any(
                        v is not None and np.size(v)
                        for v in soldict.values()):
                    n += 1
            for sim, cr in (m.get('compare_rets') or {}).items():
                fs = cr.get('failed_strategies') if isinstance(cr, dict) else None
                if fs:
                    failed.append((sim, sorted(set(fs))))
        status = 'MISSING' if not m else '{} sim(s){}'.format(
            n, ' [FAILED strategies in sims: {}]'.format(
                [s for s, _ in failed]) if failed else '')
        print('  {:<22s} {:<8s} {}'.format(
            obj, 'ok' if m and n >= nsim_target else
            ('partial' if m else 'MISSING'), status))
        print('      {}'.format(p))
        out[obj] = (m, p, n, failed)
    return out


def plan(dpsize, objectives, nsim_target, run_tag, cov):
    todo = [(o, c) for o, c in cov.items()
            if (c[0] is None or c[2] < nsim_target)
            and o not in BLOCKED_OBJECTIVES]
    if not todo:
        print('\n  nothing missing -- table is fully covered.')
        return
    print('\n  commands that produce the missing cells (NOT executed):')
    for obj, (_m, _p, n, _f) in todo:
        tag = run_tag if obj == 'avg_latency' else '{}_{}'.format(run_tag, obj)
        print('\n  # {}: have {} sim(s), want {}'.format(obj, n, nsim_target))
        print('  python -m cluster.expctl launch head --preset dpsweep \\')
        print('      --label {} --dpsizes {} --nsim {} --max-iter 200 \\'.format(
            tag, dpsize, nsim_target))
        print('      --probe-n prefixes --nocache --objsize \\')
        print('      --env SCULPTOR_GENERIC_OBJECTIVE={} \\'.format(obj))
        print('      --env SCULPTOR_RUN_TAG={}'.format(tag))


# ------------------------------------------------------------ execution --

def run_objective_cell(obj, dpsize, nsim, iters, tag, env_extra=None):
    """One objective's full evaluation in a fresh subprocess (isolates
    Ray + env per objective). Judged on the ALL DONE banner, never rc."""
    env = dict(os.environ)
    # Ray isolation: back-to-back cells raced -- the next subprocess
    # attached to the PREVIOUS cell's dying local Ray ("Connecting to
    # existing Ray cluster") and its actor pool died at construction
    # (the 2026-08-22 per_site_cost cell scored an unsolved run). A
    # per-cell RAY_TMPDIR isolates cluster discovery; the wait below is
    # belt-and-braces for the raylet actually exiting. Path kept SHORT
    # (unix-socket length limit).
    import hashlib as _h
    _rt = '/tmp/rtg_{}'.format(_h.md5(tag.encode()).hexdigest()[:6])
    for _ in range(30):
        _r = subprocess.run(['pgrep', '-f', 'raylet'],
                            capture_output=True)
        if _r.returncode != 0:
            break
        time.sleep(2)
    env.update({
        'RAY_TMPDIR': _rt,
        'PYTHONUNBUFFERED': '1',
        'SCULPTOR_GENERIC_OBJECTIVE': obj,
        'SCULPTOR_RUN_TAG': tag,
        # extension objectives (max_util, frac_beyond_optimal, ...)
        # register at import time in EVERY process (driver + Ray
        # workers); without this the solve dies with 'Objective X not
        # implemented in solve_lp_assignment' in each strategy
        'SCULPTOR_XOBJS': '1',
        # pin deployment draws so every objective's column is computed
        # over the SAME nsim deployments -- comparable, not confounded
        'SCULPTOR_EVAL_SEED': env.get('SCULPTOR_EVAL_SEED', '31415'),
    })
    # env_extra LAST so per-cell overrides (SCULPTOR_HOTSTART_RUN_DIR,
    # FORCE_* recalc flags) actually reach the subprocess -- the
    # parameter was accepted and silently DROPPED until 2026-08-25;
    # every --hotstart papertable cell trained from scratch.
    if env_extra:
        env.update(env_extra)
    log_fn = os.path.join(CACHE_DIR, 'table_generate_{}.log'.format(tag))
    argv = [sys.executable, '-u',
            os.path.join(_REPO, 'evaluations',
                         'evaluate_over_deployment_sizes.py'),
            '--dpsizes', dpsize, '--nsim', str(nsim),
            '--max-iter', str(iters),
            '--cache-fn', os.path.join(
                CACHE_DIR, 'table_generate_{}.pkl'.format(tag)),
            '--figures-subdir', 'table_generate/{}'.format(tag)]
    print('\n  [{}] running ({} deployments x {} iters; log {})'.format(
        obj, nsim, iters, log_fn))
    t0 = time.time()
    with open(log_fn, 'w') as fh:
        rc = subprocess.call(argv, cwd=_REPO, env=env, stdout=fh,
                             stderr=subprocess.STDOUT)
    txt = open(log_fn, errors='replace').read()
    m_ok = re.search(r'ALL DONE in .*?(\d+)/(\d+) sizes ok', txt)
    # the banner prints even when 0/1 sizes succeeded -- demand full ok
    ok = bool(m_ok) and m_ok.group(1) == m_ok.group(2) != '0'
    print('  [{}] {} in {:.1f} min (rc={}; the banner is the judge)'.format(
        obj, 'DONE' if ok else 'NO COMPLETION BANNER -- read the log',
        (time.time() - t0) / 60, rc))
    return ok


# ---------------------------------------------------------------- table --

def build_table(cov):
    cols = [(obj, lab, d, fn) for obj, lab, d, fn in COLUMNS
            if obj in cov and cov[obj][0] is not None]
    raw = {}
    for obj, lab, d, fn in cols:
        m = cov[obj][0]
        sims = sorted((m.get('adv') or {}).keys())
        for key, disp in METHODS:
            vals = []
            for sim in sims:
                v = fn(m, sim, key)
                if v is not None and np.isfinite(v):
                    vals.append(v)
            raw.setdefault(lab, {})[disp] = vals
    labels = [lab for _o, lab, _d, _f in cols]
    dirs = {lab: d for _o, lab, d, _f in cols}
    rows = {disp: [] for _k, disp in METHODS}
    for lab in labels:
        means = {}
        for _key, disp in METHODS:
            vals = raw.get(lab, {}).get(disp, [])
            means[disp] = float(np.mean(vals)) if vals else None
        finite = {k: v for k, v in means.items()
                  if v is not None and k not in REFERENCE_METHODS}
        best = None
        if len(finite) > 1:
            best = (min if dirs[lab] == '<' else max)(finite, key=finite.get)
            # ties go to SCULPTOR (Tom 2026-08-23: an all-equal column,
            # e.g. everyone at 0, should color SCULPTOR as the winner)
            if ('SCULPTOR' in finite and best != 'SCULPTOR'
                    and finite['SCULPTOR'] == finite[best]):
                best = 'SCULPTOR'
        for _key, disp in METHODS:
            vals = raw.get(lab, {}).get(disp, [])
            rows[disp].append((
                means[disp],
                float(np.std(vals)) if len(vals) > 1 else None,
                len(vals),
                disp == best and len(finite) > 1))
    return labels, rows


def _fmt(cell, latex=False, prec=2):
    mean, std, n, best = cell
    if mean is None:
        return '-'
    s = '{:.{p}f}'.format(mean, p=prec)
    if std is not None:
        s += (('\\pm{:.{p}f}' if latex else '+/-{:.{p}f}')
              .format(std, p=prec))
    if best:
        # \mathbf inside math -- \textbf drops to text mode and a \pm
        # inside it dies with 'missing $' (found compiling the pasted
        # table in the paper, 2026-08-30)
        _bf = '\\mathbf' if '\\pm' in s else '\\textbf'
        s = ('{}{{{}}}'.format(_bf, s)) if latex else '*{}*'.format(s)
    return ('${}$'.format(s) if (latex and '\\pm' in s) else s)



# ---- LaTeX display maps (Tom's paper wording, 2026-08-30) -------------
# Applied ONLY in the .tex emit: stored labels / CSV headers / dash keys
# stay stable so caches and merge tooling never re-key.
TEX_GROUP_DISPLAY = {
    'Latency + g*Resilience': 'Failure Robustness',
    'High + Low Priority Traffic': 'Traffic Classes',
    'Frac beyond optimal': 'Latency Sensitive Services',
    'Site cost': 'Traffic Cost Across Sites',
}
TEX_SUB_DISPLAY = {
    'Subopt PoPP-fail (ms)': 'Subopt Ingress-fail (ms)',
    '% cong PoPP-fail': '% cong Ingress-fail',
    'HPrio cong @SWAN': 'HPrio cong @SWAN',
}
# house macros defined in the paper's macros.tex
TEX_METHOD_DISPLAY = {
    'One-per-peering': '\\expensive',
    'SCULPTOR': '\\sparse',
    'PAINTER': '\\painter',
    'AnyOpt': '\\anyopt',
    'Anycast': '\\acast',
    'Unicast': '\\ucast',
}

def emit(labels, rows, fmt, out_dir, basename='paper_table'):
    os.makedirs(out_dir, exist_ok=True)
    # labels are 'Group|Sub'; derive the two header rows. HEADER_TEXT
    # renames for display only -- DIRECTION lookup keeps original labels.
    groups = []
    for l in labels:
        g = HEADER_TEXT.get(l.split('|')[0], l.split('|')[0])
        if groups and groups[-1][0] == g:
            groups[-1][1] += 1
        else:
            groups.append([g, 1])
    subs = [HEADER_TEXT.get(l.split('|', 1)[1], l.split('|', 1)[1])
            for l in labels]
    # per-column precision (Tom 2026-08-30): site-cost differences live
    # in the 3rd/4th decimal
    _precs = [4 if 'Wgt avg site cost' in l else 2 for l in labels]
    w = max(len(d) for _k, d in METHODS) + 2
    line = '  ' + ' ' * w
    for g, n in groups:
        line += '| {:^{gw}s} '.format(g[:22 * n], gw=24 * n - 2)
    print('\n' + line)
    print('  ' + ' ' * w + ' | '.join('{:>22s}'.format(x[:22]) for x in subs))
    for _key, disp in METHODS:
        print('  {:<{w}s}'.format(disp, w=w)
              + ' | '.join('{:>22s}'.format(_fmt(c, prec=p))
                           for c, p in zip(rows[disp], _precs)))
    if fmt in ('latex', 'all'):
        pth = os.path.join(out_dir, basename + '.tex')
        with open(pth, 'w') as f:
            f.write('% generated by evaluations/generate_paper_table.py\n')
            # table*: spans both columns of a twocolumn paper, pinned to
            # the top of a page; resizebox compresses the wide tabular
            # to full text width (Tom 2026-08-30: inline tabular rendered
            # in one column and ran off the page)
            f.write('\\begin{table*}[!t]\n\\centering\n'
                    '\\setlength{\\tabcolsep}{2pt}\n'
                    '\\resizebox{\\textwidth}{!}{%\n')
            f.write('\\begin{tabular}{l' + 'r' * len(labels) + '}\n\\toprule\n')
            f.write(' & ' + ' & '.join(
                '\\multicolumn{{{}}}{{c}}{{{}}}'.format(
                    n, TEX_GROUP_DISPLAY.get(g, g))
                for g, n in groups) + ' \\\\\n')
            # cmidrules under each supersection
            col = 2
            rules = []
            for _g, n in groups:
                rules.append('\\cmidrule(lr){{{}-{}}}'.format(col, col + n - 1))
                col += n
            f.write(''.join(rules) + '\n')
            # LaTeX-escape header labels (a bare % in '% cong ...'
            # comments out the row terminator -- found compiling the
            # pasted table in the paper, 2026-08-30)
            _esc = [TEX_SUB_DISPLAY.get(x, x)
                    .replace('%', '\\%').replace('&', '\\&')
                    for x in subs]
            f.write('Method & ' + ' & '.join(_esc) + ' \\\\\n\\midrule\n')
            for _key, disp in METHODS:
                f.write(TEX_METHOD_DISPLAY.get(disp, disp) + ' & '
                        + ' & '.join(_fmt(c, latex=True, prec=p)
                                     for c, p in zip(rows[disp], _precs))
                        + ' \\\\\n')
            f.write('\\bottomrule\n\\end{tabular}\n')
            f.write('}\n\\caption{Methods versus metrics across '
                    'objectives (edit caption in the doc).}\n'
                    '\\label{tab:' + basename + '}\n'
                    '\\end{table*}\n')
        print('\n  wrote {}'.format(pth))
    if fmt in ('csv', 'all'):
        pth = os.path.join(out_dir, basename + '.csv')
        with open(pth, 'w') as f:
            disp_labels = ['{}|{}'.format(
                HEADER_TEXT.get(l.split('|')[0], l.split('|')[0]),
                HEADER_TEXT.get(l.split('|', 1)[1], l.split('|', 1)[1]))
                for l in labels]
            f.write('method,' + ','.join(disp_labels) + '\n')
            # direction row: the renderer reads this instead of keeping a
            # (stale-prone) duplicate heuristic -- 2026-08-23, after the
            # dash marked most-negative Obj as best
            dircol = {l: d for _o, l, d, _f in COLUMNS}
            f.write('DIRECTION,' + ','.join(dircol.get(l, '<') for l in labels) + '\n')
            for _key, disp in METHODS:
                f.write(disp + ',' + ','.join(
                    '' if c[0] is None else '{:.4f}'.format(c[0])
                    for c in rows[disp]) + '\n')
        print('  wrote {}'.format(pth))




def emit_key(labels, rows, fmt, out_dir):
    """The condensed second table: KEY_COLUMNS only, same machinery."""
    want = ['{}|{}'.format(g, sub) for g, sub in KEY_COLUMNS]
    idx = [labels.index(l) for l in want if l in labels]
    missing = [l for l in want if l not in labels]
    if missing:
        print('  [key table] not in GROUPS (check KEY_COLUMNS): {}'.format(
            missing))
    if not idx:
        return
    klabels = [labels[i] for i in idx]
    krows = {disp: [rows[disp][i] for i in idx] for _k, disp in METHODS}
    print('\n  -- key metrics --')
    emit(klabels, krows, fmt, out_dir, basename='paper_table_key')

def _condensed_path(dpsize, run_tag):
    return os.path.join(CACHE_DIR, 'paper_table_condensed_{}_{}.pkl'.format(
        dpsize_str(dpsize), run_tag or 'default'))


def _l1_paths(dpsize, objectives, run_tag):
    out = []
    for obj in objectives:
        tag = run_tag if obj == 'avg_latency' else '{}_{}'.format(run_tag, obj)
        out.append(pickle_path(dpsize, tag))
    return out


def _condensed_fresh(dpsize, objectives, run_tag, nsim=None):
    cp = _condensed_path(dpsize, run_tag)
    if not os.path.exists(cp):
        return False
    # the L3 key must cover the REQUEST, not just the artifacts: a
    # follow-on asking for MORE sims or different objectives is not
    # 'identical' (2026-08-26: the nsim=3 resume loaded the nsim=1
    # table and exited in 0.0s without training anything)
    try:
        d = pickle.load(open(cp, 'rb'))
        if nsim is not None and (d.get('nsim') or 0) < nsim:
            return False
        if objectives and set(sorted(objectives)) - set(d.get('objectives')
                                                        or []):
            return False
    except Exception:
        return False
    ct = os.path.getmtime(cp)
    for p in _l1_paths(dpsize, objectives, run_tag):
        if os.path.exists(p) and os.path.getmtime(p) > ct:
            return False
    return True


def _save_condensed(dpsize, run_tag, labels, rows, nsim=None,
                    objectives=None):
    with open(_condensed_path(dpsize, run_tag), 'wb') as f:
        pickle.dump({'labels': labels, 'rows': rows,
                     'methods': METHODS, 'nsim': nsim,
                     'objectives': sorted(objectives or [])}, f)


def _load_condensed(dpsize, run_tag):
    d = pickle.load(open(_condensed_path(dpsize, run_tag), 'rb'))
    return d['labels'], d['rows']


def main():
    import time as _t
    t0 = _t.time()
    ap = argparse.ArgumentParser(
        description='methods x metrics paper table, one command')
    ap.add_argument('--dpsize', default='small',
                    help="'small', '3', 'actual-3', '32', ...")
    ap.add_argument('--number_of_deployments', '--num_deployments', '--nsim',
                    dest='nsim', type=int, default=3)
    ap.add_argument('--num_training_iter', '--iters', '--max-iter',
                    dest='iters', type=int, default=150)
    ap.add_argument('--run_id', '--run-tag', dest='run_tag',
                    default='papertable')
    ap.add_argument('--objectives', default=','.join(DEFAULT_OBJECTIVES))
    ap.add_argument('--hotstart', default='',
                    help="resume sparse solves from state-N checkpoints: "
                         "'obj:runs_dir[,obj:runs_dir]', e.g. "
                         "'avg_latency:1787505247-testing_feature-actual-32-"
                         "sparse'. Each dir must hold state-0.pkl (the "
                         "deployment) plus the checkpoint to resume from; "
                         "sparse writes them every 5 iters automatically.")
    ap.add_argument('--plan-only', action='store_true',
                    help='print coverage + commands, execute nothing')
    ap.add_argument('--format', default='all',
                    choices=['text', 'latex', 'csv', 'all'])
    ap.add_argument('--out',
                    default=os.path.join(_REPO, 'figures', 'paper_table'))
    a = ap.parse_args()
    objectives = [OBJECTIVE_ALIASES.get(o.strip(), o.strip())
                  for o in a.objectives.split(',') if o.strip()]
    hotstart = {}
    for tok in (t for t in a.hotstart.split(',') if t.strip()):
        if ':' not in tok:
            raise SystemExit("--hotstart entries are 'objective:runs_dir' "
                             "(got {!r})".format(tok))
        obj, d = tok.split(':', 1)
        obj = OBJECTIVE_ALIASES.get(obj.strip(), obj.strip())
        if obj not in objectives:
            raise SystemExit('--hotstart objective {!r} not in --objectives'
                             .format(obj))
        sd = os.path.join(_REPO, 'runs', d.strip())
        if not os.path.exists(os.path.join(sd, 'state-0.pkl')):
            raise SystemExit('--hotstart {}: no state-0.pkl under {} -- '
                             'state-0 is REQUIRED for hot-start'
                             .format(obj, sd))
        hotstart[obj] = d.strip()
    dpsize = normalize_dpsize(a.dpsize)
    run_tag = a.run_tag

    # ---- 5-SECOND PATH: fresh condensed pickle and nothing forced ----
    if (not a.plan_only and not FORCE_RESOLVE and not FORCE_REAGGREGATE
            and not FORCE_RECOMPUTE_METRICS
            and _condensed_fresh(dpsize, objectives, run_tag,
                                 nsim=a.nsim)):
        labels, rows = _load_condensed(dpsize, run_tag)
        emit(labels, rows, a.format, a.out)
        emit_key(labels, rows, a.format, a.out)
        print('\n  [condensed] table from L3 pickle in {:.1f}s'.format(
            _t.time() - t0))
        return

    print('== coverage (dpsize={} nsim>={}) =='.format(dpsize, a.nsim))
    cov = coverage(dpsize, objectives, a.nsim, run_tag)

    if a.plan_only:
        plan(dpsize, objectives, a.nsim, run_tag, cov)
    else:
        env_extra = {}
        _forced_tags = {}
        if FORCE_RECOMPUTE_METRICS:
            env_extra['SCULPTOR_RECALC'] = FORCE_RECOMPUTE_METRICS
        for obj in objectives:
            m, _p, n, _f = cov[obj]
            if obj in BLOCKED_OBJECTIVES:
                print('\n  [{}] SKIPPED: {}'.format(
                    obj, BLOCKED_OBJECTIVES[obj]))
                continue
            covered = (m is not None and n >= a.nsim
                       and not FORCE_RESOLVE
                       and not FORCE_RECOMPUTE_METRICS)
            if covered:
                print('\n  [{}] covered ({} sims) -- reusing'.format(obj, n))
                continue
            tag = run_tag if obj == 'avg_latency' \
                else '{}_{}'.format(run_tag, obj)
            if FORCE_RESOLVE:
                tag = '{}_r{}'.format(tag, int(_t.time()) % 100000)
                # remember the FULL renamed tag: coverage/emit must
                # re-check THESE pickles, not the base-tag ones
                # (2026-08-25 -- --no-cache runs solved everything then
                # said 'no pickles found' and emitted no table)
                _forced_tags[obj] = tag
            cell_env = dict(env_extra)
            if obj in hotstart:
                cell_env['SCULPTOR_HOTSTART_RUN_DIR'] = hotstart[obj]
            run_objective_cell(obj, dpsize, a.nsim, a.iters, tag,
                               env_extra=cell_env)
        print('\n== re-checking coverage ==')
        cov = coverage(dpsize, objectives, a.nsim, run_tag,
                       tag_overrides=_forced_tags)

    if any(c[0] is not None for c in cov.values()):
        labels, rows = build_table(cov)
        if labels:
            emit(labels, rows, a.format, a.out)
            emit_key(labels, rows, a.format, a.out)
            if not a.plan_only:
                _save_condensed(dpsize, run_tag, labels, rows,
                                nsim=a.nsim, objectives=objectives)
                print('  [condensed] L3 pickle saved; next identical call '
                      'loads the table in seconds')
    else:
        print('\n  no pickles found -- nothing to tabulate yet.')
    print('  total {:.1f}s'.format(_t.time() - t0))


if __name__ == '__main__':
    main()
