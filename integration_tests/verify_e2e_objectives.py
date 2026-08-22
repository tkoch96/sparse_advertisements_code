#!/usr/bin/env python
"""(c) One evaluation at dpsize=small per objective.

    python integration_tests/verify_e2e_objectives.py
    python integration_tests/verify_e2e_objectives.py --quick
    python integration_tests/verify_e2e_objectives.py --objectives max_util

Objectives exercised:
  avg_latency          baseline -- minimize traffic-weighted average latency
  per_site_cost        minimize site cost
  max_util             minimize MLU (straight, not the lat+MLU blend)
  frac_beyond_optimal  maximize traffic within 10ms of optimal

The last three are extension objectives registered by `core/hard_objectives.py`,
so the runs set SCULPTOR_XOBJS=1 (see _common.env_for). Two things that used to
make this impossible and were fixed on 2026-08-21:

* `eval_all_solution_types.py` hardcoded `generic_objective='avg_latency'`. It now
  takes --objective.
* Registration was import-order dependent -- importing core.hard_objectives
  before core.solve_lp_assignment hit the cycle mid-init and silently logged
  "registration FAILED", leaving max_util and frac_beyond_optimal
  undispatchable. hard_objectives now self-registers as well.

Each objective gets its own SCULPTOR_RUN_TAG so the metrics pickles do not
collide -- without it, `evaluate_all_metrics` resume-skips the second objective
off the first one's results and silently evaluates nothing.
"""
import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

_LABEL = 'objectives'   # figures/integration_tests/<_LABEL>/

# These verify the pipeline RUNS, not that it converges -- keep them in
# minutes. Override with --iters when you actually want convergence.
DEFAULT_ITERS = 5

OBJECTIVES = [
    ('avg_latency',         'baseline: minimize average latency'),
    ('per_site_cost',       'minimize site cost'),
    ('max_util',            'minimize MLU (straight)'),
    ('frac_beyond_optimal', 'maximize traffic within 10ms of optimal'),
]


def run_case(root, iters, objectives, dpsize='small'):
    res = C.Result('objectives ({}, {} objectives)'.format(dpsize, len(objectives)))
    ws = C.workspace(root, 'objectives')
    t0 = time.time()
    extras = []
    for obj, label in objectives:
        tag = 'e2eobj_' + obj
        log = os.path.join(root, 'obj_{}.log'.format(obj))
        started = time.time()
        rc = C.run([sys.executable, '-u', C.driver('eval_all_solution_types.py'),
                    '--dpsize', dpsize, '--objective', obj],
                   ws, C.env_for(iters, {'SCULPTOR_RUN_TAG': tag}, label=_LABEL), log)
        res.check(rc == 0, '{}: exit 0'.format(obj),
                  '{} -- rc={}'.format(label, rc))
        C.scan_log(log, res)
        # SCULPTOR_RUN_TAG suffixes the metrics filename (wrapper_eval.
        # _run_tag_suffix), so look for the tagged name -- checking the
        # untagged one silently finds nothing and skips every content check.
        pkl = os.path.join(ws, 'cache',
                           'popp_failure_latency_comparison_{}_{}.pkl'.format(dpsize, tag))
        C.check_metrics(pkl, res, prefix='{}: '.format(obj), started_at=started)
        extras.append(pkl)
    res.wall_s = time.time() - t0
    C.collect(ws, res, 'objectives', extras)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--objectives', default=None,
                    help='comma list; default runs all four')
    ap.add_argument('--dpsize', default='small')
    ap.add_argument('--quick', action='store_true', help='3 iters, 2 objectives')
    ap.add_argument('--iters', type=int, default=None)
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()

    print('=' * 74 + '\ne2e: one evaluation per objective\n' + '=' * 74)
    if not C.preflight():
        return 2
    if a.objectives:
        want = set(a.objectives.split(','))
        objs = [(o, l) for o, l in OBJECTIVES if o in want]
        if not objs:
            sys.exit('no objective matches {!r}; have: {}'.format(
                a.objectives, ', '.join(o for o, _ in OBJECTIVES)))
    else:
        objs = OBJECTIVES[:2] if a.quick else OBJECTIVES
    iters = 3 if a.quick else (a.iters or DEFAULT_ITERS)
    root = tempfile.mkdtemp(prefix='verify_e2e_objectives_')
    print('objectives  : {}\niters       : {}\nscratch     : {}'.format(
        ', '.join(o for o, _ in objs), iters, root))
    print('=' * 74)
    res = run_case(root, iters, objs, a.dpsize)
    print('  -> {} in {:.0f}s'.format('PASS' if res.passed else 'FAIL', res.wall_s))
    return C.finish([res], root, a.keep)


if __name__ == '__main__':
    sys.exit(main())
