"""Startup call-graph timing instrumentation (Tom 2026-08-20).

Activate with SCULPTOR_STARTUP_TIMELOG=1 (idempotent; called from
run_eods_cell for the driver and from the Ray actor __init__ for
workers). Wraps a curated list of methods along the startup path —
init -> first gradients -> end of iteration 1 — and prints one line per
call that takes >= SCULPTOR_TIMELOG_MIN seconds (default 0.5):

    [tl] w=<worker_i|drv> pid=<pid> <Class.method> took <dur>s

Entry lines ([tl-enter]) are printed only for the coarse phase list so
logs stay readable. get_ground_truth_resilience_benefit additionally
logs entry/exit unconditionally with gamma — Tom's hypothesis is that
it short-circuits and never actually computes.
"""
import functools
import os
import time

_MIN = float(os.environ.get('SCULPTOR_TIMELOG_MIN', '0.5'))
_ACTIVATED = False

# always announce entry for these (coarse phases)
_ENTER_ANNOUNCE = {
    'init_optimization_vars', 'measured_objective',
    'get_ground_truth_latency_benefit', 'get_ground_truth_resilience_benefit',
    'modeled_objective', 'update_deployment', 'compute_one_per_peering_solution',
    'update_ug_ingress_decisions', 'start_workers', 'solve_sparse',
    'compare_different_solutions', 'init_persistent_lp', 'init_all_vars',
}


def _who(args):
    if args and hasattr(args[0], 'worker_i'):
        return 'w={}'.format(getattr(args[0], 'worker_i'))
    return 'w=drv'


def _wrap(cls, name):
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, '_tl_wrapped', False):
        return
    @functools.wraps(orig)
    def wrapped(*args, **kwargs):
        who = _who(args)
        announce = name in _ENTER_ANNOUNCE
        if announce:
            print('[tl-enter] {} pid={} {}.{}'.format(
                who, os.getpid(), cls.__name__, name), flush=True)
        t0 = time.time()
        try:
            return orig(*args, **kwargs)
        finally:
            dur = time.time() - t0
            if announce or dur >= _MIN:
                print('[tl] {} pid={} {}.{} took {:.2f}s'.format(
                    who, os.getpid(), cls.__name__, name, dur), flush=True)
    wrapped._tl_wrapped = True
    setattr(cls, name, wrapped)


def activate():
    global _ACTIVATED
    if _ACTIVATED or os.environ.get('SCULPTOR_STARTUP_TIMELOG', '0') != '1':
        return
    _ACTIVATED = True

    import optimal_adv_wrapper as oaw
    import sparse_advertisements_v3 as sas3
    import path_distribution_computer as pdc

    for name in ['update_deployment', 'compute_one_per_peering_solution',
                 'update_ug_ingress_decisions', 'solve_lp_with_failure_catch',
                 'update_parent_tracker_workers', 'solve_lp_volscen_mp',
                 'solve_lp_with_failure_catch_mp']:
        _wrap(oaw.Optimal_Adv_Wrapper, name)

    for cls_name in ['Sparse_Advertisement_Wrapper', 'Sparse_Advertisement_Solver',
                     'Sparse_Advertisement_Eval']:
        cls = getattr(sas3, cls_name, None)
        if cls is None:
            continue
        for name in ['init_optimization_vars', 'measured_objective',
                     'get_ground_truth_latency_benefit',
                     'get_ground_truth_resilience_benefit', 'modeled_objective',
                     'solve_sparse', 'compare_different_solutions', 'solve',
                     '_solve_setup', '_solve_calc_grads', '_solve_apply_step',
                     'stop_tracker', 'calculate_ground_truth_ingress',
                     'latency_benefit_fn', 'gradient_fn',
                     'get_ground_truth_user_latencies']:
            _wrap(cls, name)

    for name in ['init_all_vars', 'init_persistent_lp', 'generic_benefit',
                 'generic_objective_pdf', '_compute_scenario_options',
                 '_sample_scenario_realizations', 'solve_generic_lp_persistent',
                 'update_deployment', 'calculate_ground_truth_ingress']:
        _wrap(pdc.Path_Distribution_Computer, name)

    # RB short-circuit hypothesis probe: log gamma + wall unconditionally
    def _rb_probe(orig):
        @functools.wraps(orig)
        def wrapped(self, *a, **k):
            t0 = time.time()
            try:
                return orig(self, *a, **k)
            finally:
                print('[tl-rb] {} pid={} gamma={} using_rb={} took {:.3f}s'.format(
                    _who((self,)), os.getpid(), getattr(self, 'gamma', '?'),
                    getattr(self, 'using_resilience_benefit', '?'),
                    time.time() - t0), flush=True)
        return wrapped
    for cls_name in ['Sparse_Advertisement_Wrapper', 'Sparse_Advertisement_Solver',
                     'Sparse_Advertisement_Eval']:
        cls = getattr(sas3, cls_name, None)
        if cls is not None and 'get_ground_truth_resilience_benefit' in cls.__dict__:
            cls.get_ground_truth_resilience_benefit = _rb_probe(
                cls.__dict__['get_ground_truth_resilience_benefit'])

    try:
        import worker_comms_ray as wcr
        for name in ['start_workers', 'send_receive_messages_workers']:
            _wrap(wcr.Worker_Manager, name)
    except Exception:
        pass
    print('[tl] instrumentation ACTIVE pid={} min={}s'.format(os.getpid(), _MIN),
          flush=True)
