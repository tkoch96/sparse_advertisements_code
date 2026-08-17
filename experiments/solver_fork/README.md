# solver_fork — backend equivalence harness (shim now MAINLINE)

As of 2026-08-17 the gurobipy-subset facade lives at the repo root
(`gpshim.py`) and the core LP callers (`solve_lp_assignment.py`,
`path_distribution_computer.py`, `sparse_advertisements_v3.py`) import
it directly — there are no forked module copies and no sys.modules
aliases anymore. Backend selection is one env var, everywhere:

    SCULPTOR_LP_BACKEND=gurobi   # default; passthrough to gurobipy
    SCULPTOR_LP_BACKEND=highs    # license-free HiGHS (highspy)

Extra verification machinery:

    SCULPTOR_GPSHIM_AUDIT=1      # re-solve every facade solve with
                                 # scipy on the LP pulled back from
                                 # HiGHS; hard-raise on mismatch

What remains in this directory:

- `test_gpshim_unit.py` — facade battery vs independently assembled
  scipy references (12 checks; run under either backend).
- `run_equivalence.py` — the A/B acceptance harness: same deployment +
  init per seed, gurobi vs highs arms, `--report` for the paired table.
  The gurobi arm auto-skips while the license is dead.

History: the original fork copies + regen/alias machinery were used to
prove the shim end-to-end without touching mainline (equiv_v1 highs arm
10/10, v4-grid HiGHS campaign, per-solve scipy audits) before merging;
see git history of this directory for the fork-era layout.
