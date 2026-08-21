# Salvage from `.claude/worktrees/` (removed 2026-08-21)

Five abandoned agent worktrees were removed. Their **branches survive in git**
(`claude/busy-wu-4feb02`, `claude/dreamy-gauss-475467`,
`claude/jolly-clarke-f235f3`, `claude/practical-babbage-4b76fb`) — only the
working directories are gone. What could NOT survive removal is saved here:
untracked files, and uncommitted diffs on top of those branches.

| file | from | why it was at risk |
|---|---|---|
| `offline_failure_eval_actual32.py` | dreamy-gauss | Untracked. Offline failure-eval against saved advertisements — the script the recovered actual-32 result needed to fill in its empty failure-Δ rows. |
| `dump_gurobi_lps.py` | dreamy-gauss | Untracked. Dumps Gurobi LP models for inspection. |
| `VERIFICATION_RESULT_stop_tracker.md` | dreamy-gauss | Untracked. Local verification that dropping `verbose_workers=True` at `sparse_advertisements_v3.py:1877` doesn't move the Believed/GTO/eval signal. Its four raw `.log` companions were dropped. |
| `cross_seed_phase_a_plot.py` | practical-babbage | Untracked. |
| `stochastic_lp.py` | practical-babbage | Tracked on that branch only — **does not exist on `main`** — and carried 104 uncommitted lines. This copy is the full working version (base + uncommitted). |
| `test_stochastic_lp.py` | practical-babbage | Same: branch-only, +174 uncommitted lines. Full working version. |
| `ray-cluster.jolly-clarke.yaml` | jolly-clarke | Untracked, and **differs from the root `ray-cluster.yaml`**. Kept in case the difference is the fleet config you want. |
| `*.patch` | busy-wu, jolly-clarke, practical-babbage | Uncommitted diffs against files that DO exist on `main` (`optimal_adv_wrapper.py` x2, `sculptor_3way_runner.sh`). Apply with `git apply` from the repo root. |

`elated-blackburn` held only a file deletion, but its HEAD was **detached at
`7a6a91b` with no branch containing it**. That commit is preserved by the tag
`salvage/elated-blackburn-7a6a91b`; delete the tag if you don't want it.
