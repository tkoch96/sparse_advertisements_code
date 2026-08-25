#!/usr/bin/env python
"""Latency-breakdown bench for Path_Distribution_Computer (Tom 2026-08-24).

    python unit_tests/bench_path_distribution.py                # all scenarios
    python unit_tests/bench_path_distribution.py --candidates 60
    python unit_tests/bench_path_distribution.py --scenarios warm,cold

Instantiates ONE _LocalPathDistributionComputer directly (no Ray) on a
real harvested deployment and drives the actual hot path
(latency_benefit -> generic_objective_pdf -> solve_generic_lp_persistent)
with a stream of single-popp-flip candidates -- the same shape as the
compressed-LB mega-batches. Reports the worker's own phase timing table,
per-solve wall, HiGHS model size, and RSS per scenario:

  warm       defaults (incremental UB diffs, basis preserved)
  cold       SCULPTOR_LP_INCREMENTAL=0 (full sweep + cold simplex)
  mlu_off    alternate avg_latency/max_util, SCULPTOR_LP_INCR_MLU=0
  mlu_on     same alternation with SCULPTOR_LP_INCR_MLU=1
  rebuild    warm + init_persistent_lp() rebuild every --rebuild-every

The questions this answers (2026-08-24): (a) is optimize warm-started;
(b) is a periodic model rebuild worth it for RAM/latency; (c) where the
per-call milliseconds actually go.
"""
import argparse
import copy
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SCULPTOR_XOBJS', '1')
os.makedirs('logs', exist_ok=True)

import numpy as np


def rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024 if sys.platform == 'darwin' else 1024)


def build_worker(pickle_fn):
    import pickle
    m = pickle.load(open(pickle_fn, 'rb'))
    deployment = copy.deepcopy(m['deployment'][0])
    deployment['port'] = 0
    kwa = dict(m['settings'][0] or {})
    kwa.pop('save_run_dir', None)
    kwa['verbose'] = False
    from core.path_distribution_computer import _LocalPathDistributionComputer
    t0 = time.time()
    w = _LocalPathDistributionComputer(0, deployment, kwa)
    w._bench_deployment = deployment   # for seed_parent_tracker_from_init
    print('worker init: {:.1f}s | popps={} ugs={} prefixes={}'.format(
        time.time() - t0, w.n_popp, w.n_ug, w.n_prefixes))
    return w


def gradient_step_load(w, n_steps=4, flips_per_step=12, seed=1234,
                       congested=False):
    """The realistic hot-loop shape (Tom 2026-08-24): a handful of
    gradient-step ITERATIONS, each = one base advertisement (the step's
    optimization point) plus its compressed-LB flip batch; the base
    moves a few popps between steps like an applied gradient update.
    Yields (step_idx, adv); per-step accounting exposes variance."""
    rng = np.random.RandomState(seed)
    base = np.zeros((w.n_popp, w.n_prefixes))
    base[:, 0] = 1                                # anycast prefix
    # congested=True models the belief-phase regime: few popps on, caps
    # exceeded, so the standard LP goes infeasible and the persistent
    # solver falls back to MLU mode -- the real source of the
    # standard<->MLU alternation that disables incrementality.
    density = (w.n_popp // 20) if congested else (w.n_popp // 6)
    for j in range(1, w.n_prefixes):
        base[rng.choice(w.n_popp, size=max(2, density),
                        replace=False), j] = 1
    if congested:
        base[rng.choice(w.n_popp, size=int(w.n_popp * .8),
                        replace=False), 0] = 0
    for step in range(n_steps):
        yield step, base.copy()
        for _ in range(flips_per_step):
            a = base.copy()
            i, j = rng.randint(w.n_popp), rng.randint(w.n_prefixes)
            a[i, j] = 1 - a[i, j]
            yield step, a
        # apply a "gradient step": toggle a few entries of the base
        for _ in range(3):
            i, j = rng.randint(w.n_popp), rng.randint(w.n_prefixes)
            base[i, j] = 1 - base[i, j]


def run_scenario(w, name, n, env, objs, rebuild_every=0):
    for k in ('SCULPTOR_LP_FORCE_NONPERSISTENT', 'SCULPTOR_LP_INCR_MLU'):
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = v
    for k in w.timing:
        w.timing[k] = 0
    # distinct candidates defeat the lb memo, but clear it anyway so no
    # scenario inherits another's cache
    try:
        w.calc_cache.all_caches['lb'] = {}
    except Exception:
        pass
    rss0, t0 = rss_mb(), time.time()
    n_done = 0
    step_walls = {}
    for i, (step, a) in enumerate(gradient_step_load(
            w, n_steps=n[0], flips_per_step=n[1],
            congested=(len(objs) > 1 or objs[0] == 'congested'))):
        if rebuild_every and i and i % rebuild_every == 0:
            trb = time.time()
            for attr in ('_active_keys', '_last_active_vars', '_last_mlu'):
                if hasattr(w, attr):
                    delattr(w, attr)
            w.var_pool = {}
            w.init_persistent_lp()
            print('    [rebuild] at candidate {} took {:.2f}s'.format(
                i, time.time() - trb))
        _tc = time.time()
        w.latency_benefit(a, generic_obj='avg_latency')
        step_walls.setdefault(step, []).append(time.time() - _tc)
        n_done += 1
    wall = time.time() - t0
    tt = {k: v for k, v in sorted(w.timing.items(), key=lambda kv: -kv[1])
          if v > 0.01}
    print('\n== {} ==  {} solves in {:.1f}s -> {:.0f} ms/solve | '
          'var_pool={} | RSS {:.0f}->{:.0f} MB'.format(
              name, n_done, wall, 1000 * wall / max(n_done, 1),
              len(getattr(w, 'var_pool', {})), rss0, rss_mb()))
    for k, v in tt.items():
        print('    {:38s} {:7.2f}s  ({:4.0f}%)'.format(k, v, 100 * v / wall))
    for step in sorted(step_walls):
        d = step_walls[step]
        print('    step {:>2d}: {:5.0f} ms/solve mean, {:5.0f} min, '
              '{:6.0f} max over {} solves'.format(
                  step, 1000 * sum(d) / len(d), 1000 * min(d),
                  1000 * max(d), len(d)))
    return wall / max(n_done, 1)



def scenario_table(a):
    lat = 'avg_latency'
    return {
        'warm':    (dict(SCULPTOR_LP_INCREMENTAL='1'), [lat], 0),
        'cold':    (dict(SCULPTOR_LP_INCREMENTAL='0'), [lat], 0),
        'mlu_off': (dict(SCULPTOR_LP_INCREMENTAL='1',
                         SCULPTOR_LP_INCR_MLU='0'), ['congested'], 0),
        'mlu_on':  (dict(SCULPTOR_LP_INCREMENTAL='1',
                         SCULPTOR_LP_INCR_MLU='1'), ['congested'], 0),
        'rebuild': (dict(SCULPTOR_LP_INCREMENTAL='1'), [lat],
                    getattr(a, 'rebuild_every', 15)),
        'fresh':   (dict(SCULPTOR_LP_FORCE_NONPERSISTENT='1'), [lat], 0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pickle', default='cache/popp_failure_latency_'
                    'comparison_testing_feature-actual-20_dep_sweep_20.pkl')
    ap.add_argument('--steps', type=int, default=4,
                    help='gradient-step iterations (3-5 typical)')
    ap.add_argument('--flips-per-step', type=int, default=12)
    ap.add_argument('--rebuild-every', type=int, default=15)
    ap.add_argument('--scenarios', default='warm,cold,mlu_off,mlu_on,rebuild')
    a = ap.parse_args()
    w = build_worker(a.pickle)
    S = scenario_table(a)
    results = {}
    names = a.scenarios.split(',')
    # pass 0 warms the var_pool/model so scenario ORDER does not decide
    # the winner (the pool is built once and shared); report pass 1.
    for rep in range(2):
        if rep:
            print('\n======== steady-state pass ========')
        for name in names:
            env, objs, reb = S[name.strip()]
            r = run_scenario(w, name, (a.steps, a.flips_per_step),
                             env, objs, reb)
            if rep:
                results[name] = r
    print('\n== summary (ms/solve) ==')
    base = results.get('warm')
    for k, v in results.items():
        print('  {:10s} {:7.0f}{}'.format(k, v * 1000,
              '   ({:+.0f}% vs warm)'.format(100 * (v - base) / base)
              if base and k != 'warm' else ''))


if __name__ == '__main__':
    main()


def build_batch_payload(w, n_jobs=24, rb_frac=0.6, seed=99):
    """A calc_compressed_lb payload shaped like a production RB/LB flush:
    data[0] = ((base_adv,), base_kwa); data[1:] = (diff, kwa) where diff
    = np.where(base != candidate). RB-style jobs zero one popp's whole
    row (a failure); LB-style jobs flip one (popp, prefix) entry."""
    rng = np.random.RandomState(seed)
    base = np.zeros((w.n_popp, w.n_prefixes))
    base[:, 0] = 1
    for j in range(1, w.n_prefixes):
        base[rng.choice(w.n_popp, size=max(2, w.n_popp // 6),
                        replace=False), j] = 1
    kwa = {'generic_obj': 'avg_latency'}
    payload = [((base.copy(),), dict(kwa))]
    for i in range(n_jobs):
        cand = base.copy()
        if rng.random() < rb_frac:
            cand[rng.randint(w.n_popp), :] = 0          # popp failure
        else:
            r, c = rng.randint(w.n_popp), rng.randint(w.n_prefixes)
            cand[r, c] = 1 - cand[r, c]                 # gradient flip
        payload.append((np.where(base != cand), dict(kwa)))
    return payload


def replay_batch(w, n_jobs=24, profile=False):
    """Drive _cmd_calc_compressed_lb exactly like a VM flush; print the
    per-batch timing table (same [wt] accounting as production) and,
    with profile=True, a cProfile top-25 to split pmat internals."""
    payload = build_batch_payload(w, n_jobs=n_jobs)
    try:
        w.calc_cache.all_caches['lb'] = {}
    except Exception:
        pass
    rss0, t0 = rss_mb(), time.time()
    if profile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()
    ret = w._cmd_calc_compressed_lb(payload)
    if profile:
        pr.disable()
    wall = time.time() - t0
    n = len(payload)
    print('\n== batch replay ==  {} jobs in {:.1f}s -> {:.0f} ms/job | '
          'RSS {:.0f}->{:.0f} MB'.format(n, wall, 1000 * wall / n,
                                         rss0, rss_mb()))
    for k, v in sorted(w.timing.items(), key=lambda kv: -kv[1]):
        if v > 0.01:
            print('    {:38s} {:7.2f}s  ({:4.0f}%)'.format(
                k, v, 100 * v / wall))
    if profile:
        sio = io.StringIO()
        ps = pstats.Stats(pr, stream=sio).sort_stats('cumulative')
        ps.print_stats(25)
        print('\n== cProfile top-25 (cumulative) ==')
        for line in sio.getvalue().splitlines():
            if line.strip():
                print('   ', line[:150])
    return wall / n


def init_like_advertisement(w, rng):
    """Thresholded shape of the production 'using_objective' init: anycast
    prefix 0 fully on, one prefix per pop carrying that pop's popps, the
    remaining prefixes sparsely on with linearly-decaying density
    (init_advertisement in core/sparse_advertisements_v3.py)."""
    a = np.zeros((w.n_popp, w.n_prefixes), dtype=bool)
    a[:, 0] = True
    pops = sorted(set(p for p, _ in w.popps))
    n_assigned = min(len(pops), w.n_prefixes - 1)
    for i in range(n_assigned):
        rows = [j for j, (p, _) in enumerate(w.popps) if p == pops[i]]
        a[rows, i + 1] = True
    start = n_assigned + 1
    if w.n_prefixes > start:
        prob_ons = np.linspace(.05, .005, num=w.n_prefixes - start)
        for i, prob in enumerate(prob_ons):
            a[rng.random_sample(w.n_popp) < prob, start + i] = True
    return a


def realistic_rounds(w, n_indices=12, n_rounds=6, pct_new=0.2, drift=3,
                     seed=7, obj='avg_latency', rb_rows=0, seed_pt=True,
                     clear_meas=True):
    """Tom's A/B/C protocol (2026-08-25): measure the STEADY-STATE batch,
    not a synthetic one.

      (A) base advertisement shaped like the production init
      (B) one warm-up batch: n_indices probed entries, each enqueued as
          the same single-flip PAIR the driver builds (entry at 0 and at
          1, both as diffs vs base) -- plus rb_rows popp-row-zero jobs
          if the batch should look like an RB fan-out slice
      (C) repeat the batch with ~pct_new of the probed indices replaced
          by fresh ones (the explore/remeasure mix), the base drifted by
          `drift` toggled entries (the applied gradient step -- keeps
          the lb memo honest while pattern caches stay warm)

    Rounds 2+ of (C) are the realistic measurement. Batches go through
    _cmd_calc_compressed_lb so the driver->worker payload shape, the
    per-batch timing reset, and the [wt] summary are all the production
    ones."""
    rng = np.random.RandomState(seed)
    base = init_like_advertisement(w, rng)
    if seed_pt:
        seed_parent_tracker_from_init(w, base)
    all_inds = [(i, j) for i in range(w.n_popp) for j in range(w.n_prefixes)]

    def _fresh(k, exclude):
        picks = []
        while len(picks) < k:
            ind = all_inds[rng.randint(len(all_inds))]
            if ind not in exclude and ind not in picks:
                picks.append(ind)
        return picks

    probed = _fresh(n_indices, set())
    rb_popps = list(rng.choice(w.n_popp, size=rb_rows, replace=False)) \
        if rb_rows else []
    results = []
    for rnd in range(n_rounds):
        # build the exact production payload: job 0 = base, then for each
        # probed index the off/on pair as diffs vs base, then RB rows
        kwa0 = {'generic_obj': obj, 'job_id': 0}
        data = [((base.copy(),), kwa0)]
        jid = 1
        for ind in probed:
            for setting in (False, True):
                other = base.copy()
                other[ind] = setting
                data.append((np.where(base != other),
                             {'generic_obj': obj, 'job_id': jid}))
                jid += 1
        for poppi in rb_popps:
            other = base.copy()
            other[poppi, :] = False
            data.append((np.where(base != other),
                         {'generic_obj': obj, 'job_id': jid}))
            jid += 1

        lb_cache = w.calc_cache.all_caches.get('lb', {})
        hits0, t0 = len(lb_cache), time.time()
        w._cmd_calc_compressed_lb(data)   # resets w.timing itself
        wall = time.time() - t0
        n_jobs = len(data)
        # exclusive (self) times -- the raw dict nests (lp_persistent
        # wraps optimize etc.) and summing it double-counts
        tt = {k: v for k, v in w._self_timing().items() if v > 0}
        timed = sum(tt.values())
        shares = ' '.join('{}={:.0f}%'.format(k, 100 * v / timed)
                          for k, v in sorted(tt.items(),
                                             key=lambda kv: -kv[1])
                          if v / timed >= .01)
        new_entries = len(lb_cache) - hits0
        print('round {:>2d}{}: {:6.1f}s  {:5.0f} ms/job  jobs={}  '
              'memo_hits={}  {}'.format(
                  rnd, ' (warm-up)' if rnd == 0 else '',
                  wall, 1000 * wall / n_jobs, n_jobs,
                  n_jobs - new_entries, shares), flush=True)
        results.append((wall, n_jobs, dict(tt)))

        # (C) evolve: replace ~pct_new of the probed indices, drift base
        if clear_meas:
            # production measures the stepped advertisement every
            # iteration; enforce_measured_prefs -> clear_new_meas_caches
            # cold-starts the pattern cache for the NEXT batch. This is
            # what makes pmat_organize dominate live [wt] lines.
            w.clear_new_meas_caches()
        n_new = max(1, int(round(pct_new * n_indices)))
        keep = probed[n_new:]
        probed = keep + _fresh(n_new, set(keep))
        for _ in range(drift):
            ind = all_inds[rng.randint(len(all_inds))]
            base[ind] = ~base[ind]

    # steady state = rounds 2+ (round 0 warms, round 1 still mixes)
    ss = results[2:] or results[1:]
    tot_w = sum(r[0] for r in ss); tot_j = sum(r[1] for r in ss)
    agg = {}
    for _, _, tt in ss:
        for k, v in tt.items():
            agg[k] = agg.get(k, 0) + v
    timed = sum(agg.values()) or 1
    print('\n== steady state (rounds 2+) ==  {:.0f} ms/job over {} jobs'
          .format(1000 * tot_w / max(tot_j, 1), tot_j))
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        if v / timed >= .01:
            print('    {:38s} {:7.2f}s  ({:4.0f}%)'.format(k, v, 100 * v / timed))
    return results


def seed_parent_tracker_from_init(w, base):
    """Emulate the STARTUP MEASUREMENT of the init advertisement (Tom
    2026-08-25): for every (ug, prefix col of base), the ground-truth
    routed ingress (min priority rank among active popps the ug can
    reach) beats every other active reachable popp -- exactly the pairs
    enforce_measured_prefs learns when production measures init. Fed to
    the worker through BOTH production channels: the compact CSR
    (_cmd_update_parent_tracker_csr, the SCULPTOR_COMPACT_PT default)
    and the legacy dict command."""
    dep = w._bench_deployment
    ip = dep.get('whole_deployment_ingress_priorities') \
        or dep['ingress_priorities']
    ug_perfs = dep.get('whole_deployment_ug_perfs') or dep['ug_perfs']
    ugs = list(w.whole_deployment_ugs)
    popp_to_ind = {p: i for i, p in enumerate(w.popps)}
    ug_to_ind = {ug: i for i, ug in enumerate(ugs)}
    trips, parents_on = set(), {}
    for pref_i in range(base.shape[1]):
        active = set(np.where(base[:, pref_i])[0])
        if not active:
            continue
        for ug in ugs:
            prefs = ip.get(ug, {})
            cand = [popp_to_ind[p] for p in ug_perfs.get(ug, [])
                    if popp_to_ind.get(p) in active]
            if len(cand) < 2:
                continue
            routed = min(cand, key=lambda pi: prefs.get(w.popps[pi], 1e9))
            ui = ug_to_ind[ug]
            for beaten in cand:
                if beaten != routed:
                    trips.add((routed, ui, beaten))
                    parents_on.setdefault(ug, {})[
                        (w.popps[beaten], w.popps[routed])] = None
    t = np.asarray(sorted(trips), dtype=np.int64)
    if t.shape[0]:
        parents, starts = np.unique(t[:, 0], return_index=True)
        offsets = np.concatenate([starts, [t.shape[0]]]).astype(np.int64)
        payload = (parents.astype(np.int32), offsets,
                   t[:, 1:].astype(np.int32), True)
    else:
        payload = (np.zeros(0, dtype=np.int32), np.zeros(1, dtype=np.int64),
                   np.zeros((0, 2), dtype=np.int32), False)
    w._cmd_update_parent_tracker(parents_on)
    w._cmd_update_parent_tracker_csr(payload)
    print('seeded parent tracker from init measurement: {} (ug,beaten,'
          'routed) pairs over {} ugs'.format(len(trips), len(parents_on)),
          flush=True)
    return len(trips)
