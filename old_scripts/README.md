# old_scripts/

Retired code. **Nothing here is imported by the running pipeline** — that is
the entry criterion.

Kept rather than deleted because git history alone does not help you find
something you have forgotten the name of. If you are looking for a module that
used to be at the repo root or under `experiments/`, check here and the
"Where things went" table in `experiments/README.md`.

Contents include the pre-v3 solvers (`sparse_advertisements.py`, `_v2.py`),
retired one-off drivers, plotters unwired from any dashboard
(`plot_stepalpha.py`, `plot_mesh.py`), the historical
solver-equivalence harnesses, and `from_worktrees/` — work salvaged from five
abandoned agent worktrees before they were removed, with its own README
explaining each file's provenance.

Two things in here are worth knowing about rather than assuming dead:

- `optimal_adv_wrapper_ray.py` — a 1,589-line duplicate of
  `core/optimal_adv_wrapper.py` with **zero unique definitions**. It was still
  being edited as late as 2026-08-19, and every one of those edits was a
  runtime no-op because nothing imported it.
- `from_worktrees/offline_failure_eval_actual32.py` — the offline failure-eval
  script the recovered actual-32 result needed to fill in its empty
  failure-Δ rows.
