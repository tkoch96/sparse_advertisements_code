"""Evaluation for the joint_priority objective.

joint_priority solves a strict-priority-like LP: the high-priority traffic
class is served first, then the remainder. The comparison that matters is
therefore how much high-priority traffic each solution type places well, not
average latency across all UGs.

NOTE (2026-08-21): `joint_priority` is documented in
core/solve_lp_assignment.py's header, core/gpshim.py and
evaluations/testing_priorities.py, but it is **not in
generic_lp_functions and not registered by core/hard_objectives**, so it is
currently not dispatchable. This module is the evaluation half, ready for when
the objective is wired up; `run` says so plainly rather than producing a
figure from a solve that did not happen.

evaluations/testing_priorities.py is the existing standalone driver for this
objective and is the place to look for the intended semantics.
"""
from evaluations._objective_eval_base import announce

OBJECTIVES = ('joint_priority',)


def run(ctx):
    announce(ctx, 'evaluations_for_priorities',
             'joint_priority evaluation')
    print("[priorities] joint_priority is documented but NOT dispatchable: it "
          "is absent from core.solve_lp_assignment.generic_lp_functions and "
          "from core.hard_objectives.REGISTERED_OBJECTIVES. Nothing was "
          "optimised for priorities, so no priority comparison is produced.")
    print("[priorities] wire the LP up first (see evaluations/"
          "testing_priorities.py for the intended semantics), then fill in "
          "scoring here.")
    return ctx.metrics
