# SCALE-500 PLAN — 500+ core fleets for SCULPTOR (2026-08-17)

Goal: run `evaluate_over_deployment_sizes` (and grid campaigns) in ~a
day instead of 1-2 weeks. HiGHS removes the license constraint entirely
— solver capacity now scales with raw vCPUs and RAM, nothing else.

## 1. Architecture: share-nothing VM fleet (recommended)

Two candidate shapes:

**(A) One big Ray cluster spanning VMs** — single scheduler, elastic.
Rejected as the primary path: the whole pipeline is built around
per-cell LOCAL ray instances (RAY_ADDRESS=local, SCULPTOR_RAY_NUM_CPUS
cap), and attaching evals to a shared busy cluster has crashed us
before (rescore_fork doctrine). Cross-node deployment pickling would
put the 5-15 min worker-broadcast on the network per cell. High
refactor risk for zero throughput gain over (B).

**(B) Share-nothing fleet** — N independent VMs, each running today's
proven single-node stack (run_n_sweep_queue + RAM governor + per-slot
workspaces + local ray per cell) on a SHARD of the manifest. A thin
fleet layer handles provision/bootstrap/monitor/collect/teardown.
Cells are already idempotent (result-JSON = done, .inprog markers,
multi-pass re-scan), which makes this trivially spot-safe and
failure-isolated. **This is the plan.**

## 2. Components to build

**P0 — durability + fleet registry (~half day)**
- S3 results bucket; per-VM sync loop (out_root -> s3://…/store/, every
  ~2 min) so spot reclaims never lose landed cells.
- Fleet registry: extend the alert-JSON contract from one `head` to a
  `fleet` list (instance_id, ip, shard, state). All watchers/tickers
  iterate it. (Contract memory: update on every lifecycle event.)
- fleet_tick: per-VM progress.json aggregation -> one dash widget
  (sum done_cells/inflight, per-VM RAM/CPU rows, fleet ETA via the
  existing regression ticker).

**P1 — provision/bootstrap/teardown (~1 day)**
- Launch template (or EC2 Fleet API) + instance profile with S3 access.
- Bootstrap user-data: git clone + venv + pip (highspy, boto3) + the
  KNOWN NEW-HOST TRAPS baked in:
  * worker_comms asserts on hardcoded venv paths -> symlink fix
  * `ray` symlinked into /usr/local/bin (autoscaler bare `ray stop`)
  * short RAY_TMPDIR dirs pre-created
- fleet_up.py / fleet_down.py (boto3): request N spot instances
  (capacity-diversified across types/AZs), assign manifest shards
  (deployment-major so same-seed cells share a VM's deployment cache),
  launch queue per VM, register in fleet JSON. fleet_down drains +
  final S3 sync + terminates.
- Spot-interruption handler on each VM: poll IMDS 2-min notice ->
  final S3 sync + mark shard cells' .inprog stale. Replacement VM
  re-runs the shard; re-scan skips landed cells.

**P2 — evaluate_over_deployment_sizes as a manifest (~1 day)**
- Today it is a sequential per-dpsize loop calling
  evaluate_all_metrics(dpsize, nsim) — dpsizes [3,5,10,15,20,25,32] x
  nsim [15,20,10,16,15,15,12] with per-strategy subprocesses inside.
- Refactor into work units (dpsize, sim_seed, strategy) emitted as a
  manifest; a merge step (existing checkpoint-pickle convention, same
  as eval_ladder_metrics' per-seed merge) aggregates on one node.
  NO metric logic changes — orchestration only.
- Prerequisite for dpsize>=20 units (HANDOFF actual-32 scoping):
  parent_tracker uint32 packing (-96% RAM, measured in
  experiments/profile_worker_memory.py) and the
  measured_latency_benefits growth check. Without packing, big-dpsize
  cells need r-family RAM (see §4).

**P3 — burn-in (~half day + a campaign)**
- Pilot: 2 VMs / 192 cores on a maxhard grid shard; verify shard
  hand-off, S3 sync, fleet dash, interruption drill (terminate one VM
  mid-run, confirm zero loss + re-run).
- Then the full evaluate_over_deployment_sizes fleet run.

## 3. Cross-VM monitoring

- Dash: one fleet panel (per-VM rows: cells running, RAM, CPU, last
  heartbeat age; fleet totals + ETA). Sources: per-VM progress.json
  pulled by fleet_tick over ssh (IPs from fleet registry).
- Alerting: existing Mac cron liveness_check extended to iterate the
  fleet list; alert on stale heartbeat / dead queue / RAM>92%.
- Logs: stay per-VM (workspace logs); harvested figures sync via S3
  alongside results.

## 4. Scale limitations (known, with mitigations)

| Limit | Where it bites | Mitigation |
|---|---|---|
| RAM/cell at dpsize>=20 | 64GB head OOM'd at dpsize 25 (history) | uint32 packing (P2); r7i for big-dpsize shards; governor already gates admission |
| Deployment build+pickle 5-15 min/cell | startup dominates short cells | deployment-major sharding reuses per-VM cache; pre-bake deployment pickles to S3 per (dpsize, seed) |
| EC2 spot vCPU quota | default often 640/region | Service Quotas raise BEFORE burn-in (action item; takes ~a day) |
| IAM for fleet APIs | ray-up memory: needs more than EC2 | extend the instance role once, in P1 |
| Disk (runs/ ~35MB/cell) | long campaigns | existing retention GC per workspace + S3 offload |
| Dash/public sync fan-in | one Mac pulling N VMs | fleet_tick batches; figures via S3, not rsync-per-VM |
| HiGHS threads | 1 thread/LP (like our Gurobi settings) | already sized: cores ~= concurrent LPs; no change |
| Same-code discipline | worker code drift across VMs | git-pull-only doctrine; bootstrap pins the commit SHA |

## 5. Pricing (live spot quotes, us-east-1, 2026-08-17 ~20:00Z)

| Instance | vCPU | RAM | Spot $/hr (median) | $/vCPU-hr | Notes |
|---|---|---|---|---|---|
| c7g.16xlarge (Graviton) | 64 | 128G | $0.70 | $0.011 | cheapest compute; aarch64 wheels all fine (prior head was Graviton) |
| c7i.24xlarge | 96 | 192G | $1.37 | $0.014 | current-head-like; 2G/vCPU |
| c7a.24xlarge | 96 | 192G | $1.68 | $0.017 | AMD alternative for spot diversification |
| m7i.24xlarge | 96 | 384G | $1.42 | $0.015 | 4G/vCPU — mid dpsizes |
| r7i.24xlarge | 96 | 768G | $2.49 | $0.026 | 8G/vCPU — dpsize 25/32 without packing |

On-demand is ~3x spot (c7i.24xlarge ~$4.28/hr). Spot varies by hour/AZ;
fleet_up should diversify across >=3 types x >=3 AZs.

**Fleet examples**
- 512 cores Graviton: 8x c7g.16xlarge ~= **$5.6/hr** spot
- 576 cores Intel: 6x c7i.24xlarge ~= **$8.2/hr** spot
- 960 cores mixed: 10x 96-vCPU ~= **$14/hr** spot

**Campaign forecast** — anchor: the 96-core head does full
evaluate_over_deployment_sizes in ~10 days ~= 23,000 core-hours.
- 576 spot cores: ~40h wall, **~$330** total
- 960 spot cores: ~24h wall, **~$330** total (same core-hours;
  parallelism buys wall-clock, not dollars)
- Graviton variants: ~25% cheaper (~$250)
- Rule of thumb: **spot solver time ~= $0.011-0.015 per core-hour**;
  a "one-day" full eval run is a ~$300 decision, not a $3k one.

## 6. Risks / open questions

- evaluate_all_metrics internal coupling: how cleanly (dpsize, seed,
  strategy) units separate needs a half-day spike (the checkpoint-
  pickle resume convention suggests: cleanly).
- Spot pool volatility on 96-vCPU types: mitigate with 64-vCPU
  Graviton as ballast.
- Mac as fleet controller: fine for campaigns we watch; for fully
  unattended runs move fleet_tick + alerting to a $0.01/hr t4g.micro
  controller so the Mac can sleep.
- Cross-VM clock/version skew in stores: stamp every result JSON with
  commit SHA + world env (cheap, catches drift forensically).
