# Cluster runbook: SCULPTOR on AWS Ray

Operational reference for getting SCULPTOR running on AWS via Ray, end-to-end.
Written 2026-05-19 after the overnight session that stood up the first
working setup. Every step below is what we actually did and verified —
nothing aspirational.

If you're a new Claude agent picking this up: read this file, then
`OVERNIGHT_SUMMARY.md`, then `RESEARCH_ROADMAP.md`. The cluster yaml and
teardown script are ready to use — `ray up ray-cluster.yaml -y` works
unmodified.

---

## TL;DR — happy path from zero

```
# 0. Prereqs (one-time, do once per AWS account):
#    - Create IAM user with AmazonEC2FullAccess + IAMFullAccess
#    - aws configure on local Mac with that user's keys
#    - Have ~/gurobi.lic (WLS, not node-locked)
#    - Have Drive data files in:
#       data/vultr_peers_inferred.csv
#       cache/vultr_ingress_latencies_by_dst.csv  (~4.5GB)
#       cache/vultr_anycast_latency_smaller.csv   (~52MB)
#       cache/vultr_provider_popps.csv            (~2KB)
#    - pip install "ray[default]" boto3 in your local venv

# 1. Spin cluster (5-30 min depending on whether 4.5GB rsync is fresh)
~/Documents/venv312/bin/ray up ray-cluster.yaml -y

# 2. Run something
~/Documents/venv312/bin/ray exec ray-cluster.yaml '
  ts=$(date +%Y%m%d_%H%M%S)
  cd ~/sparse_advertisements_code
  SCULPTOR_MAX_ITER=10 SCULPTOR_N_WORKERS=32 nohup /home/ubuntu/venv312/bin/python \
    eval_latency_failure.py --port 31415 --dpsize actual-32 \
    > /tmp/cluster_runs/${ts}.log 2>&1 < /dev/null &
  echo $! > /tmp/cluster_runs/${ts}.pid
  ln -sfn ${ts}.log /tmp/cluster_runs/latest.log
  ln -sfn ${ts}.pid /tmp/cluster_runs/latest.pid
'

# 3. Watch progress (any of):
~/Documents/venv312/bin/ray exec ray-cluster.yaml 'tail -30 /tmp/cluster_runs/latest.log'
~/Documents/venv312/bin/ray exec ray-cluster.yaml 'ray status'

# 4. Tear down (UNCONDITIONAL AT END OF EACH SESSION)
./teardown.sh
```

## 📊 Local timing dashboard

A cron job (every 10 min) pulls the cluster logs to the laptop, persists timing
stats into a local SQLite DB (so they survive teardown), and regenerates a
refreshing dashboard. Set up session 10 (2026-05-28).

- **Tool:** [tools/cluster_dashboard.py](tools/cluster_dashboard.py) — pull → ingest → plot → html.
- **Wrapper / cron entry:** `tools/sculptor_dashboard_refresh.sh`, run by
  `*/10 * * * *`.
- **State dir:** `~/sculptor_dashboard/` — `sculptor_timings.db` (SQLite),
  `raw/` (mirrored logs, never-lose), `plots/*.png`, `index.html`.
- **View:** `open ~/sculptor_dashboard/index.html` (auto-refreshes every 60s).
- **What it tracks:**
  - Driver per-iter phase timing (grad/measure/stop) for the active run, via the
    reused [tools/plot_phase_timings.py](tools/plot_phase_timings.py) parser.
  - **Per-worker (worker 0) per-computation LP timing over time** — parsed from
    the `Worker N timing summary` blocks the actor prints each gradient batch
    (`solve_generic_lp_persistent`, `sim_rti`, `total_rti_calc`, …). Each block
    is a per-batch snapshot, so this is the "did my change move sub-step X"
    view. Only worker 0 emits (Ray dedups the rest), so it is the PER-WORKER
    time; the right axis shows cluster-total (× N_active). For a true
    sum-over-all-workers ÷ N (to expose shard imbalance) the workers would need
    to write per-worker timing files like the mem logs already do.
  - **Active worker count over time** (`workers_active.png`) + a `worker_count`
    table. Constant within a run today (no adaptive ramp), but N varies across
    runs/sizes (e.g. dep_sweep used 5 workers at dp3/5, 32 at dp10-32; ext200
    used 48) — so normalizing timing by N matters for cross-size comparison.
- **DB keeps ALL runs** (keyed by run tag) for cross-run comparison. The
  driver dashboard shows **all dpsizes** (latest run per size = the cross-size
  scaling view); the worker plot shows the active run only.
- **Off-cluster history:** drop older logs (e.g. session-9 `boost` logs with
  dpsize 3-20) into `~/sculptor_dashboard/extra_logs/*.log` — they're ingested
  on every refresh, so the cross-size view isn't limited to what's on the live
  head (which only carries the current dpsize 25/32 sweeps).
- Manual: `python tools/cluster_dashboard.py --plot-only` (replot from DB),
  `--no-pull` (re-ingest mirrored logs), `--ingest LOG` (one local log).

## 📡 Local liveness monitoring

A cron-based liveness checker runs every 10 min on the local laptop and
alerts via ntfy + macOS notification if the cluster looks dead, stuck,
or orphaned. Set up in session 10 (2026-05-27).

- **Config:** `~/.sculptor_cluster_alert/active_cluster.json` — agents
  MUST update this on every cluster lifecycle event (bring up, IP change
  via stop+start, sweep relaunch, tear down).
- **Script:** `~/.sculptor_cluster_alert/liveness_check.py` — what cron runs.
- **Crontab entry:** `*/10 * * * * /Users/tomkoch/Documents/venv312/bin/python /Users/tomkoch/.sculptor_cluster_alert/liveness_check.py >> /Users/tomkoch/.sculptor_cluster_alert/cron.log 2>&1`
- **Alert channels** (in order of how reliably they reach the user):
  1. **SMS via Gmail email-to-SMS gateway** (primary) — same pattern as
     `~/Documents/budgeter`. Reuses `~/Documents/budgeter/emailer.py`'s
     `send_notification()` so the Gmail app password lives in one place.
     Phone vibrates whether or not the Mac is online.
  2. ntfy.sh push to topic `sculptor-tk-95c9decb99ed7220` (fallback).
  3. macOS desktop notification via `osascript` (only seen at the Mac).
- **Alert tags:**
  - `vm_crashed` — config active=true but AWS shows instance stopped/terminated (CRIT)
  - `orphan_vm` — config active=false but AWS shows instance running (agent forgot to tear down OR forgot to update config)
  - `stale_config` — config active=true and last_updated > 24h ago
  - `vm_unreachable` — SSH fails while AWS shows instance running (CRIT)
  - `sweep_pid_dead` — kill -0 says the sweep process is gone (CRIT).
    Skipped automatically when ANY of:
      (a) log contains `[sweep] ALL DONE` → fires `sweep_completed`
          (INFO) instead -- sweep finished normally
      (b) pidfile mtime < 180s ago → we're catching a transition
          between runs (kill + relaunch); next tick will see new PID
      (c) log mtime < 60s ago → process just exited, will reassess
          next tick
  - `sweep_completed` (INFO) — `[sweep] ALL DONE` in the log + PID
    dead. Fires once per completion (60 min dedup).
  - `sweep_stalled` — log hasn't grown by `min_log_growth_per_check_lines`
    for **3 consecutive checks** (~30 min). The consecutive threshold
    avoids false positives during deployment warm-up (loading the 4.5 GB
    latency CSV easily takes one 10-min cron tick without log growth).
- **Alert dedup:** 60 min per tag, so a persistent issue doesn't spam.
- **Heartbeat cron (additional):** every 3 hours on the hour, sends a
  proof-of-life SMS with iter / dpsize / pool size / driver RSS /
  sys_avail / autoscale state / mem-worker count / error count /
  recent-alert tally. Stays silent when config `active=false`. Script:
  `~/.sculptor_cluster_alert/heartbeat.py`. Cron line:
  `0 */3 * * * /Users/tomkoch/Documents/venv312/bin/python /Users/tomkoch/.sculptor_cluster_alert/heartbeat.py >> /Users/tomkoch/.sculptor_cluster_alert/cron.log 2>&1`
- **Telemetry:** `~/.sculptor_cluster_alert/alert.log` (alert history),
  `~/.sculptor_cluster_alert/state.json` (last-check delta state),
  `~/.sculptor_cluster_alert/cron.log` (cron stderr).

**Disabling alerts when no cluster is up:** set `active: false` in the
config — the script will still detect orphan VMs but won't alert on
missing process/log.

---

## AWS account prereqs (one-time per account)

### 1. IAM user with right perms

In AWS console → IAM → Users → Create user `ray-cluster`. Attach BOTH:

- `AmazonEC2FullAccess` (provision EC2)
- `IAMFullAccess` (Ray creates an instance profile for cluster nodes; without
  this `ray up` errors with `AccessDenied iam:GetInstanceProfile`)

Yes, `IAMFullAccess` is broad. For a single-user research account it's the
operational simplicity tradeoff. To scope down, the specific perms Ray uses
are: `iam:GetInstanceProfile`, `iam:CreateInstanceProfile`,
`iam:AddRoleToInstanceProfile`, `iam:RemoveRoleFromInstanceProfile`,
`iam:DeleteInstanceProfile`, `iam:CreateRole`, `iam:GetRole`,
`iam:DeleteRole`, `iam:AttachRolePolicy`, `iam:DetachRolePolicy`,
`iam:PassRole`.

Create access keys for the user, save the CSV. The secret is only shown
once.

### 2. Configure aws CLI locally

```
pip install awscli   # if not already
aws configure        # paste access key ID, secret, region us-east-1
aws sts get-caller-identity   # must return your user's ARN
```

**Don't paste real AWS credentials into chat with Claude.** Anthropic logs
the conversation. If you ever leak one, rotate immediately.

### 3. Check (and possibly request) spot quota

AWS console → Service Quotas → EC2 → "All Standard (A, C, D, H, I, M, R,
T, Z) Spot Instance Requests". Default is often 5-64 vCPU which doesn't fit
even one c7g.16xlarge (64 vCPU). Request increase to **256 or 512 vCPU**
(supports 4-8 spot workers). Takes hours to ~1 day.

Tom's account: already at **640 vCPU** as of 2026-05-18 → 10× c7g.16xlarge.

### 4. Gurobi WLS license

Must be a WLS (Web License Service) file, not node-locked. Lives at
`~/gurobi.lic` on the local Mac. Ray syncs it to `/home/ubuntu/gurobi.lic`
on each cluster node via `file_mounts`. License is account-bound, not
machine-bound, so the same file works on N cluster nodes.

If you have an older node-locked Gurobi license, get a WLS one from
gurobi.com/academia.

---

## Repo layout

```
sparse_advertisements_code/
├── HANDOFF.md                         ← original handoff (pre-AWS work)
├── OVERNIGHT_SUMMARY.md               ← what session-2 (this one) did
├── CLUSTER_RUNBOOK.md                 ← this file
├── RESEARCH_ROADMAP.md                ← next-steps + experiment plans
├── ray-cluster.yaml                   ← Ray cluster config (use as-is)
├── teardown.sh                        ← end-of-session script (use as-is)
├── sparse_advertisements_v3.py        ← SCULPTOR algorithm
├── eval_latency_failure.py            ← primary driver
├── worker_comms_ray.py                ← Ray Worker_Manager
├── path_distribution_computer_ray.py  ← Ray actor wrapper
├── solve_lp_assignment.py             ← LP dispatch (with headroom helper)
├── deployment_setup.py                ← deployment builder
├── constants.py                       ← APNIC_VOLUME=False, etc.
└── tests/                             ← pytest suite
```

Data files (you provide):
```
data/vultr_peers_inferred.csv           (~500KB from Drive)
cache/vultr_ingress_latencies_by_dst.csv (~4.5GB from Drive)
cache/vultr_anycast_latency_smaller.csv  (~52MB from Drive)
cache/vultr_provider_popps.csv           (~2KB from Drive)
# cache/addresses_violating_sol.csv      auto-touched by setup_commands
```

Source: shared Google Drive folder. README.md mentions the link.

---

## ray-cluster.yaml: what's in it and why

The yaml in the repo is already validated. **Don't rewrite it from scratch
without reading this section** — there are 8 distinct gotchas baked in that
we discovered the hard way.

### Cluster shape

- Head: `m7g.4xlarge` on-demand. 16 vCPU, 64 GB RAM. **Don't downsize to
  m7g.large** — the actual-32 driver loads the 4.5GB CSV into Python dicts
  and peaks at ~30 GB RAM. m7g.large (8 GB) OOMs.
- Worker: `c7g.16xlarge` spot. 64 vCPU, 128 GB RAM. ARM64 (matches AMI).
  Cheap on spot (~$0.55-0.85/hr).
- `min_workers: 0`, `idle_timeout_minutes: 10`. Workers auto-terminate
  when idle. Idle-only worker for cost: head $0.16/hr (~$4/day).
- All instances tagged `project=sculptor` for cost allocation and
  teardown verification.

### file_mounts (~/Documents/sparse_advertisements_code → cluster)

Mirror the whole repo to `/home/ubuntu/sparse_advertisements_code`. Plus
the Gurobi license to `/home/ubuntu/gurobi.lic`.

**`rsync_filter` is intentionally NOT set.** The default would respect
.gitignore which excludes `cache/`. But cache/ contains the 4.5GB latency
file we MUST ship. Use explicit `rsync_exclude` instead (the yaml does).

### setup_commands

Six categories of mandatory items. **All present and tested** in the yaml.
For reference if rebuilding from scratch:

1. **Ubuntu 22.04 ARM64 + Python 3.12** via deadsnakes PPA.
2. **Venv at `/home/ubuntu/venv312`** to mirror local Mac path layout.
3. **Pip install**: ray[default], numpy, matplotlib, scipy, gurobipy,
   tqdm, geopy, pyzmq, cymruwhois, scikit-learn, pytest, **boto3**.
   - `boto3` is mandatory on the head node. Ray autoscaler runs there and
     uses boto3 to call EC2 RunInstances. Without it autoscaler crashes
     silently — visible only in `/tmp/ray/session_latest/logs/monitor.err`.
4. **`sudo ln -sf /home/ubuntu/venv312/bin/ray /usr/local/bin/ray`**.
   Ray autoscaler internally invokes a bare `ray stop` from non-interactive
   SSH. Doesn't see venv bin even with --login. Symlink fixes it.
5. **`ln -sfn /home/ubuntu/venv312 /home/ubuntu/venv`** and same for
   `cache`, `data`, `figures`, `logs`, `runs`. The codebase has
   hardcoded paths (`worker_comms.py:11-16` checks specific venv paths;
   `path_distribution_computer_ray.py:66` opens relative `logs/...`).
   Symlinks make these resolve regardless of CWD.
6. **`mkdir -p` runtime directories** (figures, logs, runs, cache,
   cache/deployments) and **`touch cache/addresses_violating_sol.csv`**.
   Codebase auto-populates this file but raises FileNotFoundError if
   missing entirely.
7. **`export GRB_LICENSE_FILE=/home/ubuntu/gurobi.lic`** in .bashrc and
   .profile.

### head_start_ray_commands / worker_start_ray_commands

Standard, with `ulimit -n 65536` bump required at scale.

### AMI

Currently pinned to `ami-06683ebc6ba468d04` (Ubuntu 22.04 ARM64 in
us-east-1, dated 2026-05-03). Canonical publishes new images regularly.
To refresh:

```
aws ec2 describe-images --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-arm64-server-*" \
  --query 'sort_by(Images, &CreationDate)[-1].[ImageId,Name]' \
  --output text
```

Paste new ImageId into yaml at TWO places (head + worker definitions).

---

## Running a SCULPTOR job — the detached pattern

`ray up` and `ray attach` keep an SSH session open. Long SCULPTOR runs need
to survive wifi flakes, laptop sleep, etc. The pattern we settled on:

```bash
~/Documents/venv312/bin/ray exec ray-cluster.yaml '
  mkdir -p /tmp/cluster_runs
  ts=$(date +%Y%m%d_%H%M%S)
  cd ~/sparse_advertisements_code
  <ENV_VARS> nohup /home/ubuntu/venv312/bin/python \
    eval_latency_failure.py --port 31415 --dpsize <dpsize> \
    > /tmp/cluster_runs/${ts}_<name>.log 2>&1 < /dev/null &
  pid=$!
  echo $pid > /tmp/cluster_runs/${ts}_<name>.pid
  ln -sfn ${ts}_<name>.log /tmp/cluster_runs/latest.log
  ln -sfn ${ts}_<name>.pid /tmp/cluster_runs/latest.pid
  echo "started <name> pid=$pid"
'
```

Key elements:
- `nohup ... &` detaches from the SSH session
- `< /dev/null` to avoid hanging on input
- `> file 2>&1` redirects both streams
- pid file + latest symlinks so polling scripts can find it without
  knowing the timestamp

### Useful runtime env vars (all default-off; all gated behind env var checks)

| Env var | Default | Effect |
|---|---|---|
| `SCULPTOR_MAX_ITER` | (uses dpsize-based default: 20 for small, 150 otherwise) | Override max SAS iterations |
| `SCULPTOR_N_WORKERS` | min(multiprocessing.cpu_count(), get_n_workers(dpsize)) | Override actor count; needed because the default reads driver's cpu_count |
| `SCULPTOR_RB_NO_UGS_SUBSET` | unset | Drop `ugs=` from RB-grad calls (cache + warm-start eligible). NEUTRAL at actual-10 scale; untested at actual-32 |
| `SCULPTOR_CAPACITY_HEADROOM` | "0" | Multiply LP capacities by (1 - this). 0.2 = 20% headroom. Currently also affects eval LP (bug — see RESEARCH_ROADMAP) |
| `SCULPTOR_SKIP_RB_GRAD` | unset | Returns zeros from gradients_resilience_benefit. Use WITH HEADROOM. |

### Monitoring patterns

**Quick status:**
```bash
~/Documents/venv312/bin/ray exec ray-cluster.yaml '
  pid=$(cat /tmp/cluster_runs/latest.pid)
  if ps -p $pid > /dev/null; then echo RUNNING; else echo STOPPED; fi
  ray status | head -20
  tail -30 /tmp/cluster_runs/latest.log
'
```

**Iter timing extraction:**
```bash
~/Documents/venv312/bin/ray exec ray-cluster.yaml '
  grep -E "Timer: grads|LEARNING ITERATION|Calcing.*grad took" /tmp/cluster_runs/latest.log
'
```

**Heartbeat (every 5 min) for long-running jobs:**

Use Claude's `Monitor` tool with a loop that ssh's once per tick to grab
status and exits when the process dies. Pattern from session 2 saved at
the bottom of OVERNIGHT_SUMMARY.md if useful, but it's a 20-line shell
construct nothing fancy.

### tqdm and \r progress bars

Worker code uses `tqdm` heavily. Progress bars use `\r` to overwrite within
a single log line. `tail -1` of the log gets the LATEST tqdm update only
if you strip control chars with `tr '\r' '\n' | tail -1`. Otherwise you
see the FIRST update (the "0%|..." part of the line) and think it's stuck.

This burned us multiple times — heartbeats reported "0/5206" when actually
it was 100% complete and the next phase had started.

---

## Teardown — the hard rule

**Every session that brings up cluster MUST end with teardown.** Cost
discipline. Do not trust yourself to remember.

```
./teardown.sh
```

What it does:
1. `ray down -y ray-cluster.yaml` (terminates head; workers cascade)
2. Verifies no project=sculptor EC2 instances are still running
3. Checks for unattached EBS volumes
4. Checks for orphan Elastic IPs

Exits non-zero if any orphans found, with explicit terminate commands you
can paste. If `ray down` fails for any reason, the verification step is
your safety net.

If for some reason `teardown.sh` itself fails:

```bash
# Manual nuclear option
~/Documents/venv312/bin/aws ec2 describe-instances --region us-east-1 \
  --filters "Name=tag:project,Values=sculptor" "Name=instance-state-name,Values=running,pending,stopping" \
  --query 'Reservations[].Instances[].InstanceId' --output text \
  | xargs -r ~/Documents/venv312/bin/aws ec2 terminate-instances --region us-east-1 --instance-ids
```

---

## Cost reference

Cluster running 1 head + 1 c7g.16xlarge worker:
- Head: $0.16/hr (m7g.4xlarge on-demand)
- Worker: ~$0.60-0.85/hr (c7g.16xlarge spot, varies by capacity)
- Total: ~$0.80-1.00/hr active

Cluster running 1 head + 0 workers (autoscaled down):
- Just head: $0.16/hr (~$4/day if forgotten)

Per-run cost estimate for actual-32 at MAX_ITER=200 (single c7g.16xlarge):
- ~16 min/iter × 200 = ~55 hours wall, ~$50-60 spot

After the headroom optimization (untested at scale but plausible 6×):
- ~3 min/iter × 200 = ~10 hours wall, ~$10-15 spot

**Cap your AWS Cost Explorer alert at $50/day** if you don't already.
This is a research account on advisor's bill — better than discovering a
surprise later.

---

## Known issues you'll hit eventually

These are non-fatal but show up in logs and confuse new readers.

### Tracebacks from SAS top-of-file diagnostic code

`sparse_advertisements_v3.py:64` (`compare_estimated_actual_per_user`) and
`:1224` (`make_plots`) raise IndexError on most runs. Old code, non-fatal,
SCULPTOR continues. Just ignore.

### Diurnal-eval np.int64 KeyError on synthetic deployments

`wrapper_eval.py:741` `metro_to_diurnal_factor` tries to look up
`POP2TIMEZONE[metro]` but for synthetic dpsizes (`decent`, `med`),
metros are np.int64 not city strings. Doesn't fire for actual-N runs
(real city names). Non-fatal — caught somewhere upstream.

### `tuple - tuple` TypeError in flash-crowd eval

`eval_latency_failure.py:480` `assess_resilience_to_flash_crowds_mp` raises
`TypeError: unsupported operand type(s) for -: 'tuple' and 'tuple'`
repeatedly. Eval results for flash-crowd are partially lost but other
phases continue.

### Ray log deduplication hides repeated worker prints

By default Ray deduplicates identical log lines across actors. So if 32
workers all print "Worker N -- X% done", you might see only one. Set
`RAY_DEDUP_LOGS=0` env var if you need to see all of them (rarely needed).

### SSH dropouts on long ad-hoc commands

`ray exec` opens an SSH session for each command. If your local wifi drops
for >15s, the session dies. Most of the time this is harmless because the
detached process keeps running. But if you were running an inline
`ray exec '... slow command ...'`, that command dies.

Mitigation: keep ad-hoc commands short. Use the detached pattern for
anything that needs to survive.

### Worker takes 5-10 min to spawn first time

When the autoscaler sees demand and provisions a new c7g.16xlarge spot:
- ~30s for AWS to allocate spot
- ~30s for cloud-init
- ~5-8 min for `setup_commands` (apt + pip)
- ~30s for Ray daemon start

So a SCULPTOR call that needs a new worker waits ~7-10 min before its
first actor task runs. Subsequent calls reuse the existing worker
(no setup again) until idle timeout fires.

If you'll be doing back-to-back runs, bump `idle_timeout_minutes` to 30 or
keep at least one worker alive by setting `min_workers: 1` (costs
$0.60-0.85/hr always-on).

---

## What's in `OVERNIGHT_SUMMARY.md` (don't duplicate, point to)

Session-2's empirical results:
- Per-iter timing measurements (actual-10, actual-32)
- The 7 setup hurdles in chronological order with diagnostic detail
- The headroom optimization A/B test results
- Cost model for paper grid-runs

If something in this runbook contradicts OVERNIGHT_SUMMARY.md, this
runbook is wrong — update it.
