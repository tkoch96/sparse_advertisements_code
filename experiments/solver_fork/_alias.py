"""sys.modules alias: route bare `solve_lp_assignment` imports to the fork.

Imported at the top of the forked path_distribution_computer BEFORE it
imports optimal_adv_wrapper, because optimal_adv_wrapper does
`from solve_lp_assignment import *` — in a fresh ray worker process this
alias guarantees that star-import binds the FORK (gpshim-backed) module.

Loud contamination guard: if the mainline module was already imported in
this process, aliasing now would leave two divergent copies live — refuse.
"""
import sys

from experiments.solver_fork import solve_lp_assignment as _fork_sla

_existing = sys.modules.get('solve_lp_assignment')
if _existing is not None and _existing is not _fork_sla:
    raise RuntimeError(
        'solver_fork contamination: mainline solve_lp_assignment was '
        'imported before the fork alias could be installed (importer: '
        '{!r}). Import experiments.solver_fork.* (or run_equivalence) '
        'FIRST in this process.'.format(getattr(_existing, '__file__', '?')))
sys.modules['solve_lp_assignment'] = _fork_sla
