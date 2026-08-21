# Handoff — 2026-08-21

All AWS VMs are STOPPED (verified via boto3). Nothing is running. Nothing is
billing compute. Do not start any VM without asking me first.

## What happened overnight

1. **eods32 (actual-32 production run) died and its training state was lost.**
   It reached iteration 125 in 10.9h (~5.3 min/iter at 64 workers), then the
   head's 49G disk hit 100% — the run writes a state pickle every 5 iters,
   never deletes old ones, and each is bigger than the last (130MB at iter 5,
   481MB at iter 120). During recovery the agent pruned checkpoints including
   `state-0.pkl`, which the hot-start path needs for the deployment. The
   hot-start then failed silently, the cell exited 0 after 6 seconds, and the
   queue treated that as success and deleted the run dir — taking the two
   intact checkpoints with it. Survivors: convergence + model-error PDFs in
   `cache/eods/v1_artifacts/figs`, and the timing numbers above.

2. **A RAM-opt smoke run on a separate 24xl VM crashed at ~03:13Z** (sparse
   strategy, 2 tracebacks). The traceback bodies were never captured. The full
   log `~/ramopt_smoke.log` is still on stopped VM `i-04d7439fa93efaf2a`.

3. **An MC fastpath optimization was built and proven** for
   `path_distribution_computer` — bit-identical results vs the old code across
   decent / actual-10 / actual-15, ~1.1-1.16x faster on converged workloads.
   It lives ONLY in the `/tmp/ramopt_wt` worktree on branch `ramopt` (commit
   `ab44ef4`). It is NOT in the main repo and has never run in a real training
   loop.

## State of the code

- Main repo has ONE uncommitted fix: `experiments/eods/run_eods_cell.py` — a
  gate that validates the hot-start dir and exits nonzero if it's unusable,
  so a broken hot-start can never again masquerade as success. Tested against
  3 fixtures, all pass. Review the diff before committing.
- `experiments/dashboard/generate.py` modified + `cost_calibration.py`
  untracked (from earlier work, unrelated).

## Open decisions — ask me, don't assume

1. **Restart eods32?** It would start from scratch (~17h+). Before any
   restart: delete the garbage `cache/eods/v1/actual-32/N1/seed_1_eods.json`
   (a fake-success artifact that will make the run skip itself), cap
   checkpoint retention, and free disk.
2. **Disk.** Home is 38G of 49G. ~16G is finished-campaign workspaces
   (`eods25_ws`, `v3grid_ws`, `mesh_ws`, `a10*`, `mhr4_ws`, `fixed_ws`,
   `eods25_backup_1738`). Probably all harvested already, but confirm with me
   per-directory before deleting anything.
3. **The stranded smoke log** on `i-04d7439fa93efaf2a` — ~3 min of VM time to
   retrieve, would explain the crash in #2.
4. **The MC fastpath** needs an end-to-end integration run (a 30-iter Mac A/B
   mirroring what commit `f3d7d1f` got) before it's merged anywhere.

## Rules

- One VM at a time, and only with my explicit go-ahead. Stop it when done.
- Update `~/.sculptor_cluster_alert/active_cluster.json` on every VM
  start/stop.
- Never auto-kill the production head — alert me instead.
- Deletions must be approved by me, not assumed.
