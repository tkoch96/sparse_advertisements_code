# solver_fork — backend equivalence harness (shim now MAINLINE)

As of 2026-08-17 the gurobipy-subset facade lives at the repo root
(`core/gpshim.py`) and the core LP callers (`core/solve_lp_assignment.py`,
`core/path_distribution_computer.py`, `core/sparse_advertisements_v3.py`) import
it directly — there are no forked module copies and no sys.modules
aliases anymore. Backend selection is one env var, everywhere:

    SCULPTOR_LP_BACKEND=highs    # DEFAULT (since 2026-08-20); license-free HiGHS (highspy)
    SCULPTOR_LP_BACKEND=gurobi   # opt-in passthrough to gurobipy (quadratic objectives only)

(Default flipped gurobi->highs on 2026-08-20: Gurobi had WLS scaling
issues — multi-machine sessions sustained over the license baseline get
killed after ~30 min, which took down the eods32 fleet run.)

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
