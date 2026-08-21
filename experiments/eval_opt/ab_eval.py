"""Eval-tail optimization A/B (Tom 2026-08-20): frozen advertisement
solutions, evals recomputed twice (legacy vs SCULPTOR_EVAL_VOLSCEN=1),
LP results must be exactly identical.

Stages (driven by argv[1]):
  fixture  - full legacy evaluate_all_metrics at --dpsize small with
             trivial solutions -> fixture metrics pkl (advs + evals)
  arm      - copy fixture, DELETE eval sections, recompute with the
             current env (SCULPTOR_EVAL_VOLSCEN set by caller), save as
             argv[2] label
  diff     - deep-compare two labeled metrics pkls on eval keys

Run from a scaffolded ws dir (cwd) with PYTHONPATH=repo.
"""
import os, sys, copy, pickle, time

import numpy as np

WS = os.getcwd()
FIX = os.path.join(WS, 'fixture_metrics.pkl')

EVAL_KEYS = [
    'diurnal', 'resilience_to_congestion',
    'pct_volume_within_latency',
    'popp_failures_latency_optimal_specific',
    'popp_failures_latency_before', 'popp_failures_latency_optimal',
    'popp_failures_sticky_latency_before',
    'popp_failures_sticky_latency_optimal',
    'popp_failures_sticky_latency_optimal_specific',
    'popp_failures_high_cap_latency_optimal',
    'popp_failures_high_cap_latency_optimal_specific',
    'pop_failures_latency_optimal_specific',
    'pop_failures_latency_before', 'pop_failures_latency_optimal',
    'pop_failures_sticky_latency_before',
    'pop_failures_sticky_latency_optimal',
    'pop_failures_sticky_latency_optimal_specific',
    'pop_failures_high_cap_latency_optimal',
    'pop_failures_high_cap_latency_optimal_specific',
    'latency_penalty_thresholds_normal',
]


def run_evals(metrics_fn):
    from evaluations.eval_latency_failure import evaluate_all_metrics
    t0 = time.time()
    m = evaluate_all_metrics('small', int(os.environ.get('ABPORT', '41900')),
                             nsim=1, use_performance_metrics_fn=metrics_fn,
                             soln_types=['one_per_pop', 'anycast',
                                         'one_per_peering'])
    return m, time.time() - t0


def deep_diff(a, b, path='', diffs=None, max_report=20):
    if diffs is None:
        diffs = []
    if len(diffs) >= max_report:
        return diffs
    if type(a) != type(b) and not (isinstance(a, (int, float, np.floating))
                                   and isinstance(b, (int, float, np.floating))):
        diffs.append((path, 'type', str(type(a)), str(type(b))))
    elif isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                diffs.append((path + '/' + str(k), 'missing', k in a, k in b))
            else:
                deep_diff(a[k], b[k], path + '/' + str(k), diffs)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            diffs.append((path, 'len', len(a), len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                deep_diff(x, y, path + '[{}]'.format(i), diffs)
    elif isinstance(a, np.ndarray):
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
            mx = (float(np.max(np.abs(a.astype(float) - b.astype(float))))
                  if a.shape == b.shape and a.size else 'shape')
            diffs.append((path, 'ndarray', a.shape, mx))
    elif isinstance(a, (int, float, np.floating)):
        if not (a == b or (np.isnan(a) and np.isnan(b))):
            diffs.append((path, 'val', a, b))
    elif a != b:
        diffs.append((path, 'val', str(a)[:50], str(b)[:50]))
    return diffs


def main():
    stage = sys.argv[1]
    if stage == 'fixture':
        m, wall = run_evals(FIX)
        print('[ab_eval] fixture built in {:.0f}s'.format(wall))
    elif stage == 'arm':
        label = sys.argv[2]
        arm_fn = os.path.join(WS, 'arm_{}_metrics.pkl'.format(label))
        m = pickle.load(open(FIX, 'rb'))
        # DELETE eval keys outright: evaluate_all_metrics refills missing
        # keys from default_metrics, which is the only reset shape its
        # check_calced_everything treats as "not calced"
        for k in EVAL_KEYS:
            m.pop(k, None)
        for k in [x for x in m if str(x).startswith('stats_')]:
            del m[k]
        pickle.dump(m, open(arm_fn, 'wb'))
        m2, wall = run_evals(arm_fn)
        print('[ab_eval] arm {} recomputed in {:.0f}s (volscen={})'.format(
            label, wall, os.environ.get('SCULPTOR_EVAL_VOLSCEN', '0')))
    elif stage == 'diff':
        a = pickle.load(open(os.path.join(WS, 'arm_{}_metrics.pkl'.format(sys.argv[2])), 'rb'))
        b = pickle.load(open(os.path.join(WS, 'arm_{}_metrics.pkl'.format(sys.argv[3])), 'rb'))
        total = 0
        for k in EVAL_KEYS + [x for x in set(a) & set(b) if str(x).startswith('stats_')]:
            if k in a and k in b:
                d = deep_diff(a[k], b[k], path=str(k))
                if d:
                    total += len(d)
                    print('[diff] {}: {} diffs, first: {}'.format(k, len(d), d[0]))
        print('[ab_eval] TOTAL DIFFS: {}'.format(total))
        print('[ab_eval] ' + ('EXACT MATCH ✓' if total == 0 else 'MISMATCH ✗'))
    else:
        raise SystemExit('stage?')


if __name__ == '__main__':
    main()
