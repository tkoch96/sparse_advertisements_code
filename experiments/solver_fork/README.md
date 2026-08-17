# solver_fork — Gurobi-optional SCULPTOR stack (Tom + agent, 2026-08-17)

Production fork of the LP-touching call chain, with all gurobipy access
routed through a **backend-pluggable facade** so the whole SCULPTOR
pipeline can run on a license-free solver.

    forked callers                      shim                 backends
    ------------------------------      ----------------     -----------------
    path_distribution_computer.py  \                       / gurobipy (passthrough,
    solve_lp_assignment.py          >--  gpshim.py  ------<   default; zero delta)
    (pdc_ray / worker_comms* carry /                        \ highspy / HiGHS
     the fork imports)                                        (SCULPTOR_LP_BACKEND=highs)

## Files

- `gpshim.py` — gurobipy-subset facade. Backend picked at import via
  `SCULPTOR_LP_BACKEND` (`gurobi` default | `highs`). The highs side
  implements the exact API subset the repo uses (inventoried 2026-08-17):
  one-shot matrix LPs (`addMVar`, `csr @ x`, slicing, `objVal`) and the
  persistent incremental worker model (`addLConstr` placeholders,
  `Column` col-gen, `chgCoeff`, batch `setAttr`/`getAttr`, `RHS`
  mutation) with HiGHS basis hot-starts standing in for Gurobi dual
  simplex warm starts.
- `regen_fork.py` — regenerates the five forked copies from mainline
  with exactly-once import transforms. **Never hand-edit the forked
  copies**; fix mainline and re-run.
- `_alias.py` — routes bare `solve_lp_assignment` imports to the fork
  inside worker processes (imported by the forked pdc before
  optimal_adv_wrapper).
- `run_equivalence.py` — the acceptance harness: same deployment + init
  per seed, gurobi vs highs arms, queue-style per-slot workspaces;
  `--report` prints the paired table.
- `test_gpshim_unit.py` — facade battery vs independently-assembled
  scipy.linprog (12 checks, both backends).

## Scope (Tom-ratified 2026-08-17)

- highs backend covers every **linear** objective (avg_latency, mlu,
  per_site_cost, joint_priority via the soft-bounded scalar). The
  quadratic objectives (`squaring`, `square_rooting`) raise
  `NotImplementedError` on highs; gurobi passthrough retains them.
- Equivalence is judged at the **objective level** (degenerate LPs ⇒
  alternate optima; same-seed runs are additionally gradient-RNG noisy).

## Running

    # unit battery
    SCULPTOR_LP_BACKEND=highs python -m experiments.solver_fork.test_gpshim_unit

    # 10-seed equivalence campaign (skips gurobi arm if license dead)
    python -m experiments.solver_fork.run_equivalence \
        --backends highs,gurobi --seeds 1-10 --parallel 8 \
        --out-dir cache/solver_fork/equiv_v1

## Gotchas

- Any process using the fork must install the sys.modules aliases BEFORE
  importing mainline modules (run_equivalence does this; see
  `install_aliases`). A mainline import sneaking in first raises the
  contamination guard in `_alias.py`.
- `gpshim.MVar` deliberately has no `__len__` (scipy sparse `@` dispatch
  requirement; see comment in class).
- Ray workers inherit `SCULPTOR_LP_BACKEND` from the driver env
  (single-node clusters; same mechanism as every other SCULPTOR_* knob).
