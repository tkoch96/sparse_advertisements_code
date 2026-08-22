"""Shim between the solution-type comparison and the per-objective evaluation.

`eval_all_solution_types.evaluate_all_metrics` does the objective-INDEPENDENT
work: build the deployment, stand up the workers, and solve every solution type
(SCULPTOR, PAINTER, AnyOpt, anycast, one_per_pop, one_per_peering). Nothing in
that half cares which objective was optimised.

Everything after it does. The failure-resilience, flash-crowd, diurnal and
volume phases -- and every panel of the comparison figure -- are written
against latency and resilience specifically: they read `latency_delta_*`,
weight by UG volume, and plot milliseconds. Running them after an MLU or
priority optimisation produces numbers that look valid and mean nothing.

So each objective gets its own module, and this file routes to it:

    evaluations_for_latency_plus_resilience.py   avg_latency  (the default)
    evaluations_for_mlu.py                       max_util, lat_plus_max_util
    evaluations_for_priorities.py                joint_priority
    evaluations_for_site_cost.py                 per_site_cost
    evaluations_for_frac_beyond_optimal.py       frac_beyond_optimal

Each module exposes:

    OBJECTIVES        tuple of objective names it handles
    run(ctx)          eval phases + figures; returns the metrics dict

`ctx` is an EvalContext carrying everything the phases used to read out of
evaluate_all_metrics' local scope.
"""
import importlib
import os


class EvalContext(object):
    """Everything the objective-dependent phases need from the driver.

    These were locals inside evaluate_all_metrics before the 2026-08-21 split;
    the attribute names are unchanged so the moved code reads the same.
    """

    def __init__(self, **kw):
        # driver state
        self.sas = kw.get('sas')
        self.wm = kw.get('wm')
        self.metrics = kw.get('metrics')
        self.soln_types = kw.get('soln_types')
        self.dpsize = kw.get('dpsize')
        self.port = kw.get('port')
        self.kwargs = kw.get('kwargs') or {}
        self.N_TO_SIM = kw.get('N_TO_SIM', 1)
        self.performance_metrics_fn = kw.get('performance_metrics_fn')
        self.valid_iters = kw.get('valid_iters')
        self.default_metrics = kw.get('default_metrics')
        # solver settings the phases re-instantiate a wrapper with
        self.lambduh = kw.get('lambduh')
        self.gamma = kw.get('gamma')
        self.capacity = kw.get('capacity')
        # flash-crowd / diurnal sweep points (driver locals before the split)
        self.X_vals = kw.get('X_vals')
        self.Y_vals = kw.get('Y_vals')
        # figure destination
        self.save_fig_fn = kw.get('save_fig_fn')
        # the objective actually optimised, for provenance in figures/logs
        self.objective = kw.get('objective', 'avg_latency')


# objective name -> module basename (all under evaluations/)
_ROUTES = {
    'avg_latency':         'evaluations_for_latency_plus_resilience',
    'max_util':            'evaluations_for_mlu',
    'lat_plus_max_util':   'evaluations_for_mlu',
    'joint_priority':      'evaluations_for_priorities',
    'per_site_cost':       'evaluations_for_site_cost',
    'frac_beyond_optimal': 'evaluations_for_frac_beyond_optimal',
}

DEFAULT_ROUTE = 'evaluations_for_latency_plus_resilience'


def module_name_for(objective):
    return _ROUTES.get(objective, DEFAULT_ROUTE)


def for_objective(objective):
    """Import and return the evaluation module handling `objective`.

    Unknown objectives fall back to the latency+resilience suite with a warning
    rather than raising -- a new objective should still get *some* comparison
    out of a run, but you should see that its evaluation is borrowed.
    """
    name = module_name_for(objective)
    if objective not in _ROUTES:
        print("[objective_hooks] no evaluation module registered for "
              "objective {!r}; falling back to {} -- its phases are written "
              "for latency + resilience, so read the numbers with that in "
              "mind.".format(objective, name))
    return importlib.import_module('evaluations.' + name)


def resolve_objective(explicit=None):
    """The objective this run optimised: explicit arg, else env, else default."""
    return (explicit
            or os.environ.get('SCULPTOR_GENERIC_OBJECTIVE')
            or 'avg_latency')
