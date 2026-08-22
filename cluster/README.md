# cluster/

VM and cluster management: start a box, push the repo, run an experiment,
watch it, pull everything back, stop the box.

## The interface

Four commands cover the whole loop. Everything else in this directory is
either a helper of these or an older campaign-specific script.

```bash
python -m cluster.vmctl  list                       # what exists, what it costs
python -m cluster.vmctl  start head --disk 300      # start + grow the disk
python -m cluster.expctl push   head                # rsync the code
python -m cluster.expctl launch head --preset dpsweep --label smoke \
        --dpsizes 3,5,10 --nsim 1 --max-iter 10 --plot \
        --probe-n prefixes            # optional measurement budget
python -m cluster.expctl watch  <run_id>            # pulls every tick
python -m cluster.vmctl  stop   head                # harvest-gated
```

`head`, `preflight`, `oldhead` are aliases from `vms.json`. Two instances
share the Name tag `ray-sculptor-head`, which is why the alias file
exists.

## The two contracts these tools enforce

**1. A log is never left on a box that gets stopped.**

Too many experiments have been paid for and then never read. So the pull
happens on every path, not on the path someone remembers to take:

| when | what pulls |
|---|---|
| `expctl status` | pulls before it reports (`--no-pull` opts out) |
| `expctl watch` | every tick, plus once more when the process exits |
| `expctl kill` | after the kill |
| the dashboard refresh loop | `cluster.harvest_all` every 180 s, unattended |
| `vmctl stop` | pulls all live runs, then **refuses to stop** if any bytes are still only on the VM |
| `.claude` PreToolUse hook | blocks `aws ec2 stop-instances`, `ray down`, `teardown.sh` and points at `vmctl stop` |
| `.claude` Stop hook | won't let a turn end silently while the alert JSON says a VM is up |

`vmctl stop --force` still exists for the genuine "yes, abandon it" case,
and prints exactly what it is abandoning.

`vmctl stop` distinguishes two cases, because a live log grows between the
harvest and the byte check — comparing bytes against a running process
always shows a gap, and a gate that always blocks just teaches everyone to
type `--force`:

- **process alive** → refuse on those grounds (stopping the VM kills a
  running experiment). `--kill` ends it deliberately, then harvests.
- **process dead** → harvest, compare bytes, retry once to absorb a
  process that exited mid-harvest. A gap then is real.

The Stop hook can be *satisfied* rather than merely repeated:

```bash
python -m cluster.expctl ack <run_id> --minutes 30 --reason "why"
```

It expires, it is tied to one run id, and it is void the moment the run
stops — a finished or failed run alarms immediately regardless.

Do **not** pipe `expctl watch` (`watch ... | tail`): the pipe hands you
`tail`'s exit code, which reported 0 for a FAILED run on 2026-08-21.


**2. `rc == 0` is not success.**

On 2026-08-20 a cell exited 0 in six seconds after a silently-failed
hot-start; the queue read that as success and `rmtree`'d eleven hours of
checkpoints. `expctl.verdict()` therefore judges from the log:

| verdict | meaning |
|---|---|
| `done` | completion banner present, rc 0, nothing in the log to read |
| `done-dirty` | finished, but there are tracebacks / a disk warning / a failed size |
| `suspect` | **rc 0 with no completion banner** — the 08-20 shape. Do not trust the numbers |
| `failed` | nonzero rc |
| `running` | still going |

The dashboard card shows the verdict, not the exit code.

## Disk

`vmctl start` grows the root volume to `SCULPTOR_VM_MIN_DISK_GB` (default
300) and extends the partition and filesystem, so the space is usable
rather than merely allocated. gp3 is $0.08/GB-month: 50 → 300 GB costs
about **$0.03/hr against a $3.83/hr box**. A full disk killed an 11-hour
actual-32 run at iteration 125 and destroyed its checkpoints on the way
down — this is a default, not a flag to remember.

Note EBS allows one resize per volume per 6 hours; `vmctl` reports that
and carries on rather than retrying.

The launcher also samples free disk every 60 s into `sysmon.jsonl` and
shouts `DISKLOW` into the log once when it drops under `--disk-floor`
(25 GB default). It does not kill anything — it makes the death
predictable, and the dashboard plots the slope.

## Running the actual experiment

`expctl launch --preset dpsweep` drives
`evaluations/evaluate_over_deployment_sizes.py`. Its flags, the
result-cache trap, and the env knobs that change what you measure are
documented in
[../evaluations/RUNNING_THE_DEPLOYMENT_SWEEP.md](../evaluations/RUNNING_THE_DEPLOYMENT_SWEEP.md).

## Dashboards

Every launched run gets a dashboard section automatically. There is no
per-experiment registry entry to write: `dashboard/cluster_runs.py`
discovers whatever `expctl` has harvested and builds the **Cluster runs**
tab (first tab) on the next refresh cycle.

Each section shows the status card (verdict, VM, elapsed, cost so far,
disk, last-harvest time), a per-deployment-size progress table with wall
time and sec/sim, the timing figures, and the tail of the log — so a
failure is readable in the browser instead of needing an ssh.

Served by the loop already running: `python -m dashboard.refresh --loop 180`
→ http://localhost:8643 (the `Cluster runs` tab).

## Files

| file | role |
|---|---|
| `vmctl.py` | VM lifecycle: list/start/stop/terminate/df/grow-disk/ssh/status |
| `expctl.py` | experiment lifecycle: push/launch/status/watch/pull/kill/finish |
| `vmlib.py` | shared: EC2 lookup, ssh/rsync, run registry, alert-JSON contract |
| `harvest_all.py` | pull every live run; safe from cron, no-ops when nothing runs |
| `vms.json` | instance aliases + notes (which box has what) |
| `hooks/` | Claude Code hooks: block unguarded stops, don't end a turn on a live VM |
| `ray-cluster.yaml` | Ray autoscaler config. **Last touched 2026-05-23** — verify before any `ray up`. The tools above do not use it |
| `teardown.sh` | last-resort cost control; only sees `project=sculptor`-tagged boxes |
| `fleet/` | multi-VM share-nothing bootstrap (SCALE-500) |
| `cluster_dashboard.py` | older ops dashboard under `~/sculptor_dashboard/`; cron every 10 min |
| `plot_phase_timings.py` | parses `[mem]` tags into per-phase plots; helper of the above |
| `manifests/` | 19 campaign manifests driving the ablation queue |
| `kill_zombie_workers.py`, `watch_v3.sh`, `chain_v3.sh`, `send_report.py` | campaign-era helpers |

## The boxes

| alias | instance | notes |
|---|---|---|
| `head` | `i-0428c395787bc3ca0` | **currently c7g.16xlarge** (64 vCPU, 123 GB, **$2.32/hr**) — `vmctl list` is authoritative; this table was stale at c8g/$3.83 until 2026-08-21. Retyped down from c8g.24xlarge because us-east-1f had no 24xlarge capacity; `vmctl start head --type c8g.24xlarge` puts it back when there is. **Has an Elastic IP (107.22.173.189)** so its address survives stop/start. Carries the repo, the 4.5 GB latency CSV, eods25 results. The EIP bills ~$0.005/hr while stopped — the price of a stable address |
| `oldhead` | `i-09a6ff2823b0bb304` | m7g.4xlarge. Too small for actual-25+ drivers (OOM); fine for cheap smokes |
| `preflight` | `i-04d7439fa93efaf2a` | c8g.24xlarge, 200 GB. **Untagged** — `teardown.sh` will not see it |

## Live cron

```
*/10 * * * * .../cluster/sculptor_dashboard_refresh.sh >> ~/sculptor_dashboard/refresh.log
*/3  * * * * ~/sculptor_dashboard/eods_dash_autocheck.sh
*/10 * * * * ~/.sculptor_cluster_alert/liveness_check.py
0 */3 * * * ~/.sculptor_cluster_alert/heartbeat.py
```

The 3-minute autocheck restarts `python -m dashboard.refresh --loop 180`
if it dies, so **the refresh loop self-heals** — stop the cron before
doing surgery on dashboard code, or it will resurrect a half-migrated loop.

`~/.sculptor_cluster_alert/active_cluster.json` is the contract the
liveness cron watches: it pages Tom when the file disagrees with AWS.
`vmctl` writes it on every start and stop, so nothing else should need to.

## SSH

The boxes are reachable **directly from the Mac** — the subnet
auto-assigns a public IP on start and the security group allows `:22`.
The two-hop-via-head recipe in older notes is a fallback, not a
requirement. Public IPs change on stop/start (except `head`, which holds
an Elastic IP); the tools always re-resolve, and nothing caches an
address across a stop.
