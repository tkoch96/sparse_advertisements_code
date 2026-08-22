# Handoff — 2026-08-21 (late)

**A VM IS RUNNING**: `i-0428c395787bc3ca0` (`head`), retyped to
**c8g.16xlarge**, ~$2.55/hr, at `107.22.173.189`. It is running the
timing smoke `20260821_132822-timing_smoke`. Stop it with
`python -m cluster.vmctl stop head` when done.

---

## What now exists: the cluster interface

The goal from the last handoff is built. Read `cluster/README.md`; the
loop is five commands.

```bash
python -m cluster.vmctl  list                      # what exists, what it burns
python -m cluster.vmctl  start head --disk 300     # start + grow the disk
python -m cluster.expctl push   head               # rsync the code
python -m cluster.expctl launch head --preset dpsweep --label smoke \
        --dpsizes 3,5,10 --nsim 1 --max-iter 10 --plot
python -m cluster.expctl watch  <run_id>           # pulls every tick
python -m cluster.vmctl  stop   head               # harvest-gated
```

New files: `cluster/vmctl.py`, `cluster/expctl.py`, `cluster/vmlib.py`,
`cluster/harvest_all.py`, `cluster/vms.json`, `cluster/hooks/`,
`dashboard/cluster_runs.py`, `dashboard/plot_cluster_timing.py`.

### The two contracts

**A log is never left on a box that gets stopped.** The pull happens on
`status`, on every `watch` tick, after `kill`, from the dashboard refresh
loop every 180 s unattended, and inside `vmctl stop` — which **refuses to
stop** when bytes are still only on the VM. Two Claude Code hooks close
the remaining gaps: one blocks `aws ec2 stop-instances` / `ray down` /
`teardown.sh` and redirects to the gated door, the other won't let a turn
end quietly while the alert JSON says a VM is up.

**`rc == 0` is not success.** `expctl.verdict()` judges from the log:
`done` / `done-dirty` / **`suspect` (rc 0 with no completion banner — the
2026-08-20 shape)** / `failed` / `running`. The dashboard shows the
verdict, never the exit code.

### Dashboards are automatic

Every launched run gets a section under the **Cluster runs** tab (first
tab, http://localhost:8643) with no registry edit: status card, per-size
progress, timing figures, log tail. `dashboard/cluster_runs.py` discovers
whatever `expctl` has harvested.

---

## Things learned today that the tooling now encodes

- **Stopped instances are pinned to their AZ.** `c8g.24xlarge` capacity
  in `us-east-1f` was exhausted and no amount of retrying helped.
  `vmctl start` now walks *down* the size ladder within the family,
  retyping the stopped instance (which keeps the volume, the instance id
  and the Elastic IP). `--type X` pins it and fails loudly instead.
- **`head` has an Elastic IP** (`107.22.173.189`), so its address
  survives stop/start — the Apache dash URL is stable. It bills
  ~$0.005/hr while stopped.
- **Disk growth is nearly free and now automatic.** `vmctl start` grew
  the root volume 50 → 300 GB and extended the filesystem live: **12 GB
  free became 254 GB** for ~$0.03/hr. EBS allows one resize per volume
  per 6 hours.
- **macOS ships openrsync**, which rejects `--info=` outright. Keep
  rsync flags in the portable 2.6.9 set.
- **Register the run before launching it.** The first real launch had the
  ssh channel stay open after the job detached; the launcher timed out on
  a run that was in fact healthy, and the manifest was never written — an
  unregistered run is an unharvested run. `expctl launch` now saves the
  manifest first, `run.sh` writes its own pidfile, and a launch timeout is
  non-fatal.
- **`calendar.timegm`, not `mktime` minus `time.timezone`** — the latter
  ignores DST and aged a four-minute-old run by an hour.

## Changes to the evaluation itself

`evaluations/evaluate_over_deployment_sizes.py` (Tom's own sweep, not the
`run_deployment_sweep.py` fork) gained the fork's timing instrumentation,
which was the last thing keeping the fork alive — open decision #2 from
the previous handoff is now mostly closed:

- `[mem]` and `[sweep] === dpsize=N dpsize_str=... nsim=N ===` markers, in
  the exact format `cluster/plot_phase_timings.py` already parses;
- per-size wall / sec-per-sim, and a `[sweep] ALL DONE` completion banner
  (the thing the harvest tools check instead of the exit code);
- `SCULPTOR_SWEEP_PROGRESS_JSON` → machine-readable progress for the dash;
- **the pickle is now written after every size.** It used to be written
  only after the whole loop, so a crash at size 25 threw away 3/5/10/15/20
  as well. One size failing no longer costs the others.

## Open decisions — ask Tom, don't assume

1. **Commit and push.** Still nothing on `main`. The branch
   `repo-restructure-and-eval-cleanup` has three commits; everything from
   the dashboard/cluster reorg onward — including all of today's tooling —
   is uncommitted.
2. **Retire `run_deployment_sweep.py`.** Its knobs and instrumentation are
   both ported now. What remains only in the fork is the NSIM-aware
   hot-start logic (`_count_random_iters_done`, `_find_save_run_dir_for_dpsize`).
   Port that and the 321-line fork can go.
3. **`joint_priority` is still not dispatchable** (documented in three
   places, absent from `generic_lp_functions` and
   `hard_objectives.REGISTERED_OBJECTIVES`).
4. **eods32 restart** — unchanged from the last handoff. The unbounded
   checkpoint growth that caused the original crash is still unfixed.
5. **`core/` dead code** — 13 unused imports, ~10 unreferenced functions.

## Rules

- One VM at a time, and only with Tom's explicit go-ahead. Stop it when done.
- `vmctl` writes `~/.sculptor_cluster_alert/active_cluster.json` on every
  start/stop; nothing else should need to.
- Never auto-kill the production head — alert instead.
- Deletions must be approved by Tom, not assumed.
- Run experiments in the cloud, not on the Mac.
