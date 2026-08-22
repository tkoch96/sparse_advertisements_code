> 📜 **HISTORICAL** — session-8 snapshot (titled "Handoff to session 9"
> because that's where it was pointing). For current state see
> [README.md](README.md) and the highest-numbered `HANDOFF_SESSION_*.md`.
> File:line references in this document may be stale against current code
> (ZMQ Worker_Manager described here was removed in session 10).

# Handoff to session 9

Picking up a SCULPTOR research codebase mid-stride. **Read this doc first**;
older handoffs (`HANDOFF.md`, `HANDOFF_SESSION_6.md`, `HANDOFF_SESSION_7.md`,
`SESSION_4_SUMMARY.md`, `SESSION_5_SUMMARY.md`) are kept for reference but are
not required reading — the gist is summarized below.

## State at handoff (one-paragraph version)

A boost sweep is running on the cluster (head `i-09a6ff2823b0bb304`, public IP
**`44.220.249.176`** — changed from the previous handoff after an AWS
stop+start). It is filling `random_iter=1,2,3,4` for dpsizes 3, 5, 10, 15, 20
on top of the previously-completed `random_iter=0` for those sizes plus
dpsize 25. `dpsize=32` was OOM-killed at the start of session 8 and is
**skipped** in the boost. Local objective-registry refactor is **complete and
committed-quality but uncommitted on disk**; 4 objectives keep (avg_latency,
per_site_cost, joint_priority, site_failure), 5 retired. **The worker-side
`adv` kwarg plumbing bug was the biggest unlock** — without it, every
multi-LP objective (static_failure, backup_capacity) was training on noise.

## 🚨 ACTIVE: boost deployment-size sweep

Cluster boost is running. **Don't blow it away.** Status check command (works
post stop+start since the IP changed):

```bash
~/Documents/venv312/bin/ray exec ~/Documents/sparse_advertisements_code/cluster/ray-cluster.yaml \
    'ps -eo pid,etime,cmd | awk "/[r]un_deployment_sweep.py/ && /python/" | head -3; echo ---; \
     grep -E "^\[sweep\]|Stopped train loop|Traceback|GurobiError|Overage|Killed" \
       /home/ubuntu/sweep_latest.log | tail -10; echo ---; \
     ls -la /home/ubuntu/sparse_advertisements_code/cache/popp_failure_latency_comparison_testing_feature-actual-*_dep_sweep_*.pkl 2>&1 | \
       awk "{print \$5, \$6, \$7, \$8, \$9}"; \
     free -m | head -2; date -u'
```

Boost env vars (active on PID 4090, launched 2026-05-23 10:59 UTC):

```
SCULPTOR_MAX_ITER=200
SCULPTOR_N_WORKERS=32
SCULPTOR_CAPACITY_HEADROOM=0.2
SCULPTOR_DEPLOYMENT_SWEEP_NSIM=5
SCULPTOR_DEPLOYMENT_SWEEP_SIZES=3,5,10,15,20
SCULPTOR_WORKER_INIT_STAGGER_SEC=2   ← new; offsets worker spawn times
PYTHONUNBUFFERED=1
```

ETA: ~18 hours wall time for 4 new random_iters × 5 dpsizes. Cluster wall time
already incurred in previous (now-completed) sweep:
dpsize 3 ~5min, 5 ~11min, 10 ~39min, 15 ~38min, 20 ~2.7hr, 25 ~7.8hr.

**Pickles already pulled locally** (these are random_iter=0 and survive
across all sweeps): `cache/popp_failure_latency_comparison_testing_feature-actual-{3,5,10,15,20,25}_dep_sweep_*.pkl`,
plus `cache/testing_feature_cache_fn.pkl`. The boost run writes back into the
same pickles (random_iter=1..4) in place.

**dpsize=32 is intentionally skipped in the boost** — it OOM'd during init
(see "Why dpsize 32 died" below). To get dpsize 32 results, either upgrade
the head to m7g.8xlarge (128GB) or use the stagger-init + per-worker mem
log to find the spike.

## What session 8 did

### 1. Objective registry + driver consolidation (the "infrastructure" ask)

Replaces the prior pattern of one `testing_*.py` file per objective (each
with its own `main()`).

- `experiments/objectives.py` — `ObjectiveSpec` dataclass + registry
- `experiments/run_objective.py` — single CLI driver, `python -m experiments.run_objective --obj <name> --dpsize <size> --port <p> ...`
- `experiments/site_failure.py` — the new objective's spec
- `experiments/static_failure_eval.py` — BGP-fallback eval phase (used by site_failure's `eval_phases`)
- `tests/test_lp_correctness.py` — 13 unit tests for LP correctness + objective math (was 9 before; +4 for site_failure)

**Generic_Objective** now forwards `lp_kwargs` and passes `adv=a` to LP calls
(was: discarded everything except routed_through_ingress). Both worker code
paths (`path_distribution_computer.py:814` and `path_distribution_computer.py:138`)
now pass `adv` through — **this was the silent bug** that broke gradient flow
for static_failure and backup_capacity objectives.

### 2. Objective audit + cleanup

Reduced from 9 specs to 4 keepers. Retired (deleted): `static_failure`,
`static_failure_k8`, `static_failure_k16`, `backup_capacity`,
`backup_capacity_only`. All their LP functions + spec files removed.

| keep | semantic |
|---|---|
| `avg_latency` | baseline; headroom (via SCULPTOR_CAPACITY_HEADROOM env) + MLU fallback covers congestion |
| `per_site_cost` | latency + alpha · site_cost weighted sum |
| `joint_priority` | HPrio LP solved first, bulk fills around it (strict-priority-like). NOTE: `bulk_cap_limit=100` debug leftover preserved as default (SIGCOMM 2025 used 3.0); override per-experiment if reproducing paper |
| `site_failure` | steady avg_latency + (1-β)·steady + β·mean over pop failures with **frozen user→prefix mapping**, split penalty (heavy no_route + light congestion) |

### 3. The frozen-prefix objective (the "static prefix assignment" ask)

`site_failure` LP function (in `solve_lp_assignment.py:666`):

1. Solve steady-state avg_latency LP.
2. Derive each user's pinned prefix from steady (primary popp by traffic share).
3. For each PoP (exhaustive — n_pops typically 3-30, no sampling):
   - Zero out all popps at that PoP.
   - Recompute BGP routing under the failure.
   - Constrain LP routing so each user can only use popps on their **pinned**
     prefix — this freezes the user→prefix decision per the no-DNS-update reality.
   - Solve avg_latency LP with the constraint.
4. Combine: `(1-β)·steady_obj + β·mean(failure_obj)`. Failure obj uses
   `_failure_obj_split` which separates `no_route` (heavy penalty, default 50)
   from `congestion` (light, default 2) — see `solve_lp_assignment.py:611`.

**Best local result**: at small dpsize, 75 iters, `β=0.5`, sparse popp-fail
no-route 67.83% → **61.38%** (6.5 pp improvement) at +0.4ms steady cost.
Convergent and stable.

Why other variants failed:
- Sampled-popp `static_failure` was noisier (sampling variance) and never
  matched site_failure's deterministic gradient.
- `backup_capacity` proxy rewards "has any backup popp" without modeling
  post-failure capacity load → SGD drove the proxy to its math max (1.0) but
  the BGP-fallback eval didn't improve.

### 4. Critical bug: worker-side `adv` plumbing (probably high-impact)

`path_distribution_computer.py:814` (and `_ray.py:138`) called
`solve_generic_lp_with_failure_catch(..., obj)` with NO `adv` kwarg. Workers
do most of the LP calls during gradient probing. For objectives that need
the adv matrix (static_failure, backup_capacity), the LP read `adv=None`
from kwargs and returned a degenerate value → every gradient probe got the
same number → `max_val=0` → divide-by-zero. SCULPTOR's `Modified gradient by
a factor` printout in the log was the symptom; the actual values would
oscillate 0.22-12.99 across iters before the fix.

**Fixed for both worker paths**. The fix is small but invalidates all prior
multi-LP-objective convergence experiments (β tuning, penalty tuning, etc.
done before the fix were on broken gradient signal).

### 5. Why dpsize 32 died (the OOM diagnosis)

Not the usual eval-phase OOM (failure_resilience/diurnal accumulating LP
cache — those were the session-6 problems). This was **dpsize 32's worker
fan-out at init**. Last `[mem]` entries before death:

```
dpsize_done dpsize=25            sys_avail_mb=33188
dpsize_start dpsize=32           sys_avail_mb=33188
solve_enter (dpsize=32)          sys_avail_mb=14862
solve_cold_start                 sys_avail_mb=14695
solve_post_modify_ugs            sys_avail_mb=14127
solve_post_init_optim_vars       sys_avail_mb= 3838    ← last log line
```

Driver RSS barely moved (27→31GB). The 29GB hole in system-available came
from **32 workers each simultaneously loading dpsize-32 var_pool + RB-backup
data**. The crash was in the next call (`measure_ingresses` triggering all
workers in parallel).

### 6. Lighter-touch worker-init staggering

`SCULPTOR_WORKER_INIT_STAGGER_SEC` env var added to both
`worker_comms.py` and `worker_comms.py`. When set >0, adds `time.sleep(N)`
between worker spawn calls in the start_workers loop. Workers still init in
parallel (no ACK blocking), only their start times are offset so memory
peaks don't perfectly overlap. Default 0 = preserves original behavior.

Also added `_log_mem_worker(worker_i, tag, **extra)` in
`path_distribution_computer.py` so future post-mortems can see per-worker
RSS (driver `[mem]` lines hid the worker memory previously). Workers log at
`worker_proc_start`, `worker_init_msg_received`, `update_deployment_enter`,
`update_deployment_done`. Grep for `[mem-worker idx=N`.

## 🚨 Uncommitted state

**Nothing in session 8 is committed yet**, including the worker-adv bug fix
and the registry refactor. Run `git status` to see the deltas. Suggested
commit order if you want to land this:

1. `path_distribution_computer*.py` worker-adv plumbing fix (the bug)
2. `worker_comms*.py` SCULPTOR_WORKER_INIT_STAGGER_SEC + mem logging
3. `solve_lp_assignment.py` site_failure LP + retire dead LP funcs
4. `experiments/` registry + driver + site_failure spec
5. `tests/test_lp_correctness.py` site_failure tests; backup_capacity tests removed
6. Delete the now-orphan `testing_site_costs.py` / `testing_priorities.py` /
   `testing_generic_objective.py` etc. main blocks (still in tree, still work,
   but duplicate the registry — low-prio cleanup)

## 🚨 NEXT STEPS (in priority order)

### (Step 0) Monitor the active sweep through completion

Use the ssh status command at the top. The boost is doing 4 additional
random_iter values per dpsize (3, 5, 10, 15, 20). Estimated 18 hours wall.
When it finishes:

1. SCP all per-dpsize pickles + `testing_feature_cache_fn.pkl` locally:
   ```bash
   scp -i ~/.ssh/ray-autoscaler_us-east-1.pem \
       'ubuntu@44.220.249.176:/home/ubuntu/sparse_advertisements_code/cache/popp_failure_latency_comparison_testing_feature-actual-*_dep_sweep_*.pkl' \
       ~/Documents/sparse_advertisements_code/cache/
   scp -i ~/.ssh/ray-autoscaler_us-east-1.pem \
       ubuntu@44.220.249.176:/home/ubuntu/sparse_advertisements_code/cache/testing_feature_cache_fn.pkl \
       ~/Documents/sparse_advertisements_code/cache/
   ```

2. Regenerate plots:
   ```bash
   cd ~/Documents/sparse_advertisements_code && /Users/tomkoch/Documents/venv312/bin/python -c \
   "import sys; sys.path.insert(0, '.'); from evaluate_over_deployment_sizes import make_paper_plots; \
    make_paper_plots('cache/testing_feature_cache_fn.pkl')"
   ```
   Output: `figures/paper/average_latency_over_deployment_size_{normal,fail_ingress_mlu,fail_site_mlu}.pdf`
   and the percent-within-X-ms variants.

3. **Tear down cluster** when done: `ray down -y cluster/ray-cluster.yaml`. Confirm
   no leftover EC2 instances with `aws ec2 describe-instances --filters
   "Name=tag:project,Values=sculptor" "Name=instance-state-name,Values=running"`.

### (Step a) Continue static-prefix-assignment exploration

The current best with `site_failure` is **6.5 pp improvement** in popp-fail
no-route at small dpsize. Knobs not yet swept:

- **`site_failure_beta` > 0.5** with longer warmup — does β=0.7 break
  convergence or push further? Be aware that high β can push SGD off into a
  worse basin (`HANDOFF_SESSION_8.md` history showed β=0.5 gave 64.6% but
  with penalty=100 and clean gradient signal it dropped further).
- **Add popp failures to the objective** (not just pop failures). Cheapest
  way: in `solve_lp_assignment_site_failure`, sample k random popps per call
  in addition to the exhaustive pop sweep. Knob: `site_failure_popp_samples=k`.
- **Test on `decent` dpsize** (n_pop=10). Currently only validated on
  `small`. The "structural ceiling" (sparse can't beat anycast's ~3-7%
  no-route at small) may move differently at larger n_pop.

Smoke test pattern (with the boost still running on the cluster, do NOT bump
local Gurobi sessions beyond ~2):
```bash
/Users/tomkoch/Documents/venv312/bin/python -m experiments.run_objective \
    --obj site_failure --dpsize small --port 31618 --max-iter 50 \
    --n-workers 1 --extra-evals static_failure_resilience --out-dir /tmp/dbg_sf
```

### (Step b) Infrastructure for more objective cases

The registry already supports the additive pattern. To add a new objective:

1. Add an `ObjectiveSpec` to `experiments/<name>.py`
2. Add an LP function to `solve_lp_assignment.py` and register it in `generic_lp_functions`
3. (Optional) Add a custom eval phase under `experiments/<name>_eval.py`
4. Add unit tests in `tests/test_lp_correctness.py` for hand-verifiable cases
5. `import experiments.<name>` at the top of `experiments/run_objective.py`

Existing rough edges worth fixing for the next objectives:
- **No CLI flag to override `lp_kwargs`**. Currently spec defaults are the only
  knob; tuning needs editing the spec file. Adding `--lp-kwarg key=val` to
  `run_objective.py` would let you sweep params from the shell.
- **Legacy `testing_*.py` files still exist** (`testing_site_costs.py`,
  `testing_priorities.py`, `testing_generic_objective.py`,
  `eval_all_solution_types.py`). They work but duplicate the registry. Low-prio
  consolidation.
- **`backup_capacity` failed; consider reviving with a smarter formulation**.
  See "tests/test_lp_correctness.py git history" for the original tests, but
  the proxy was structurally weak (rewards backup existence, ignores
  post-failure capacity load).

### (Step c) Test the painter-degradation hypothesis

**Hypothesis**: painter's performance gap to sparse widens when:
- (i) link-capacity overprovisioning is low (`scale_factor` in
  `deployment_setup.py:228` close to 1.0), AND
- (ii) user-volume variance across the deployment is high.

Theory: greedy adds popps that help low-volume users; can't trade margin off
against capacity. Sparse's LP can route around capacity bottlenecks.

Variables:
- **Overprovisioning**: edit `scale_factor` in `get_link_capacities`
  (currently 1.1 = 10% over anycast load; user previously had 1.3). Sweep
  e.g. {1.0, 1.05, 1.1, 1.2, 1.3}.
- **Volume variance**: in `get_random_deployment`, the per-UG volumes
  currently come from a distribution defined in `deployment_setup.py` —
  look at `ug_to_vol` generation, find the distribution parameter, add a
  knob for variance.
- **Metric**: painter's avg latency minus sparse's avg latency (under
  normal operation AND under popp failures), as functions of (scale_factor,
  volume_variance).

Suggested test plan:
1. Add `volume_variance_factor` kwarg to `get_random_deployment` (default
   preserves current behavior). Multiplies the std of the per-UG volume
   distribution.
2. Add a small wrapper script `experiments/painter_hypothesis_sweep.py`
   that loops over `(scale_factor, vol_variance)` and runs the registry's
   `avg_latency` objective for each. Should run locally on `small` with 100
   iters per cell, ~2-5 min each, total ~1-2 hours for a 5×5 grid.
3. Emit a markdown table: rows = `vol_variance`, cols = `scale_factor`,
   cells = "painter_lat - sparse_lat" (in ms).

This is a clean local-only experiment; no cluster needed.

## Cluster ops cheatsheet

| action | command |
|---|---|
| Bring cluster up (after stop or after `ray down`) | `~/Documents/venv312/bin/ray up -y cluster/ray-cluster.yaml` |
| Tear down (terminates instances, destroys EBS — logs are lost) | `~/Documents/venv312/bin/ray down -y cluster/ray-cluster.yaml` |
| Hard restart (preserves EBS!) | `aws ec2 stop-instances --instance-ids i-09a6ff2823b0bb304 --force` then wait for stopped, then `aws ec2 start-instances --instance-ids i-09a6ff2823b0bb304`. **IP changes after start.** |
| Run a one-shot command on head | `~/Documents/venv312/bin/ray exec cluster/ray-cluster.yaml '<cmd>'` |
| AWS state check | `~/Documents/venv312/bin/aws ec2 describe-instances --filters "Name=tag:project,Values=sculptor" --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' --output table` |

## Hazards & gotchas

- **EC2 `pkill -f "run_deployment_sweep.py"` self-kills the SSH session**
  because the pattern matches its own bash argv. Use `[r]un_deployment_sweep`
  (the bracket trick) or pkill by PID.
- **AWS reboot is "soft"** — if the OS is wedged it does nothing. Use
  stop+start for hard reset.
- **AWS stop+start changes the public IP**. Hardcoded IPs in scripts will
  break. Use `aws ec2 describe-instances` to discover.
- **`SCULPTOR_WORKER_INIT_STAGGER_SEC=N` adds `N×n_workers` seconds to dpsize
  startup**. For dpsize 32 with N=2 → 64s added during init only. Negligible.
- **The `static_failure_resilience` eval phase name is still used** even
  though the `static_failure` *objective* was deleted. The name lives in
  `site_failure`'s `eval_phases` tuple and in `experiments/static_failure_eval.py`.
  Don't rename it casually — `run_objective.py:252` dispatches on the string.
- **Gurobi WLS license has ~3 concurrent-session baseline** (academic). The
  cluster sweep uses many sessions; **avoid running local Gurobi experiments
  while the cluster sweep is active** unless you've confirmed there's slack.
- **`bulk_cap_limit=100` in `solve_lp_assignment.py:179`** is a debug
  leftover (comment says "temporary"). SIGCOMM 2025 paper value was 3.0.
  All `joint_priority` results in the codebase were produced under 100.0,
  so the default is preserved for reproducibility. Override per-experiment
  via spec's `lp_kwargs={'bulk_cap_limit': 3.0}` if reproducing paper.

## Key files modified this session (uncommitted)

```
solve_lp_assignment.py                  # +site_failure LP, -dead LPs, +_failure_obj_split
path_distribution_computer.py           # +adv kwarg, +_log_mem_worker, +mem logging
path_distribution_computer.py       # +adv kwarg
worker_comms.py                         # +SCULPTOR_N_WORKERS env, +stagger
worker_comms.py                     # +stagger
generic_objective.py                    # +lp_kwargs forwarding, +adv=a passthrough
experiments/__init__.py                 # new (empty)
experiments/objectives.py               # new — registry
experiments/run_objective.py            # new — CLI driver
experiments/site_failure.py             # new — spec
experiments/static_failure_eval.py      # new — BGP-fallback eval helper (kept; used by site_failure)
tests/test_lp_correctness.py            # +site_failure tests, -backup_capacity tests
HANDOFF_SESSION_8.md                    # this doc

# Local data added under cache/:
cache/popp_failure_latency_comparison_testing_feature-actual-25_dep_sweep_25.pkl  # 33MB
# (others were already there)
```

## Critical environment

- Local venv: `~/Documents/venv312/bin/python`
- Local repo: `~/Documents/sparse_advertisements_code`
- AWS: instances tagged `project=sculptor`, head `i-09a6ff2823b0bb304`,
  worker `i-08c7b448c78a6f50f`, IP at handoff `44.220.249.176`
- Gurobi: WLS license at `~/gurobi.lic`. Pinned at 3 concurrent sessions
  (academic baseline). Cluster MPS dump owed to Nicholas — pending.
- ntfy topic: `sculptor-tk-95c9decb99ed7220` (from session 3)
