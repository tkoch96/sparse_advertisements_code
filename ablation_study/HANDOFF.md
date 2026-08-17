# Ablation handoff — V4 ERA (2026-08-17, L1-L6 ladder, license-paused)

## READ THIS SECTION ONLY. History lives in old_handoffs/ — consult for
## the why, not the what.

Single source of live state: `~/.sculptor_cluster_alert/active_cluster.json`
(active:false right now — VM is STOPPED; on restart the IP CHANGES,
update that file first, everything re-resolves it per use).

### THE ARC (why this ladder exists)

We are isolating which of SCULPTOR's methodological features actually
earn their complexity, by ablation ladder, while SLOWLY GROWING PROBLEM
SIZE so conclusions survive scale:

    small x 1 seed (smokes)  ->  small x 10 deployments (CURRENT, 67% done)
    ->  actual-10 x 1 deployment (STARTED, 2/6 cells)  ->  actual-32 (future)

Ladder (one capability per rung; probing is ALWAYS pure grounding —
measure the current advertisement):
  L1 budgeted-fixed -> L2 +scheduled spacing -> L3 +belief LP ->
  L4 +memory -> L5 +direction/explore -> L6 +slotted WHEN

### SETTLED THIS ERA (Tom-ratified, all with measured evidence in git)

1. **WHAT is dead forever.** Smart probe targeting (maxinfo L6, then
   decision/expected-regret L6') never beat grounding at the current
   advertisement. Theory: model-based DFO mandates evaluating the
   iterate (sufficient-decrease needs f(x_k)); the current point is the
   hub of the finite-difference star (one measurement corrects n
   decisions' common-mode bias). Measured: mean belief-drift +0.22
   (optimism bias grounding resets), probe sigma did NOT shrink after
   targeted measurement (0.067->0.091). Old rungs retired to
   cache/ablation/RETIRED_*.
2. **L6 = slotted WHEN** (Tom's design): probe k owns slot
   k*period ± period/2 (period=TCONV/N) — mean rate stays the even
   schedule; the last grounding's realized SURPRISE (the one bias-immune
   signal) fires early-in-slot when hot, center when quiet, slot-end
   force-fires; skips retry until slot close (no budget leak).
   PROBE_MODE=slotted. 3-seed A/B: best mean, tightest spread, exact
   budgets. Self-assessed-uncertainty gates (old L7) are dead: ~290 of
   ~400 firings were the dumb backstop — a biased model never
   volunteers that it needs checking.
3. **Step policy = AdaGrad-Norm, alpha0=1** (head DEFAULT via
   SCULPTOR_GRAD_SCALE; fixed/dog/legacy-auto selectable). 5-seed A/B:
   composite +10.1 vs stock's +13.7 vs opp; alpha0 has a floor-not-peak
   effect; DoG is the parameter-free fallback. NOTE fixed alpha=0.5 was
   the outright best at small — scale-fragile, revisit at actual-32.
4. **stop-v2 stopping rule** (head DEFAULT): scale-relative rolling
   delta (< 0.03 x its own initial) + best-patience 20 + GROUNDED gate
   (budget spent / horizon elapsed) + patience-from-last-measurement.
   The legacy rule was dead code (absolute eps vs ~150-scale objectives).
5. **LB cache ON** (default; =0 only for measurement-validity studies).
   Paired 167-cell A/B: quality delta median +0.000 / mean +0.11ms,
   3.1x faster, immune to the cache-off late-run support-growth stalls.
6. **Gurobi WLS sessions are NOT a sizing constraint** (official
   baseline 2, empirically 20-88+ fine; oversubscription just waits).
   Size pools by RAM (90% target, ~/queue_governor.json live-tunable)
   and cores. Cap per-cell ray instances: SCULPTOR_RAY_NUM_CPUS=6
   (uncapped, 31 cells held 2651 idle ray workers = ~111G).
7. **EVERYTHING LIVES IN GIT.** The era-cut exposed that core files
   (solve_lp_assignment objectives, model_error/, mc_off_worker) lived
   only in scp'd working trees — two mass-failure incidents came from
   that. Repo is complete now; never scp code, git pull instead.

### BLOCKED: GUROBI LICENSE (first action on resume)

License 2487370 EXPIRED ~2026-08-17T00:30Z. Tom renews at
portal.gurobi.com (needs his Columbia creds) -> new ~/gurobi.lic on
BOTH machines. NOTHING LP-touching runs until then (train, eval,
rescore — all of it). Do not chase "mysterious" failures before
checking the license.

### SIMULATION STATUS + RESUME CHECKLIST (in order)

State: v4 grid **971/1440 clean cells** in cache/ablation/
{policy_ladder_v3,hardobj_v3} (all L3-L6 x 4 objectives x 10
deployments; L1/L2 = the no_mc rung missing — they crashed twice: first
the untracked-objectives bug, then the sigma-refresh-vs-mc-off bug
(both FIXED, see git log ~baa3967), then the license died). actual-10:
2/6 cells (L1/L2 same story). All data pulled to the Mac and
count-verified; VM stopped.

1. Tom renews license; put ~/gurobi.lic on Mac + VM.
2. Start VM: `python -c "import boto3; boto3.client('ec2',
   region_name='us-east-1').start_instances(InstanceIds=
   ['i-0428c395787bc3ca0'])"` (aws CLI not installed on Mac). Update
   alert JSON: new public IP, active:true.
3. On head: `cd sparse_advertisements_code && git pull`, then
   `setsid nohup bash ~/v4_repair.sh &` — it purges crash remnants,
   reruns the 469 no_mc cells (queue: multi-pass re-scan, .inprog
   markers, 90-min cell timeout, deployment-major), then completes the
   actual-10 follow-up (6 cells x 16 workers). Re-arm ram_watchdog
   (`bash ~/ram_watchdog.sh`, setsid) and a session Monitor.
4. **FIRST ANALYSIS JOB — the scorer dispute (OPEN, blocks trusting
   L6):** in-run cell scores say slotted-L6 is the best rung ever
   (+0.4-0.8 vs opp); the dash fresh-process eval said the same advs
   were bad (seed-1 steady up to 56). One scorer lies; repo doctrine
   says only rescore_fork is authoritative (in-process contamination
   history). Run rescore_fork over the grid (the queue's post-phase
   does it; or per-dir), then re-run steady/failure scoring and compare
   all three. If fresh-eval was wrong, find why (it uses the same code
   now — suspect env/world mismatch in the eval child).
5. Dash refresh loop then repopulates the fresh-eval composite figure
   automatically (the ladder tab currently shows the clearly-labeled
   IN-RUN fallback; plot_ladder_direct auto-switches back when
   policy_steady_v3 store has data). Also run a painter eval for mlu
   (era-comparable painter ref missing on that panel).
6. When grid + a10 land: report per-deployment tables, then the
   actual-32 question: prerequisites already scoped = parent_tracker
   uint32 packing (-96% measured, experiments/profile_worker_memory.py),
   shared ray instance (~90G at 88 cells), and a
   measured_latency_benefits growth check.

### PARKED / OPEN

- **zipf**: 48-cell smoke (z in {0,1.5} x L1-L6 x N{5,20} x 2 seeds,
  L1/L2 pending rerun): z=1.5 FLATTENS the ladder (global-popularity
  skew makes the problem easier; N-value of measurements also shrinks).
  Idea stays parked pending reformulation (per-UG-PRIVATE zipf prefs x
  tight caps).
- **security**: a Gmail app password leaked in commit 3e51bb5 (public).
  Code is clean (env/keyfile now); Tom must REVOKE the old password;
  optional history rewrite staged (git-filter-repo installed in venv;
  needs Tom's go for force-push). GitGuardian incident open.
- prio metric redefinition (linear-joint vs exemption) — still with Tom.

### INFRASTRUCTURE MAP

- **Dash** localhost:8643: server = experiments/dashboard/serve.py
  (no-store headers; NOT bare http.server), generator registry in
  generate.py (tabs: policy ladder / hard objectives v4 / actual-10 /
  Old dashboards). Refresh loop `python -m experiments.dashboard.refresh
  --loop 180` (Mac, nohup; RESTART IT after editing the registry — it
  holds the step list in memory). 30s ticker progress_tick.py writes
  progress.json (iterations + VM RAM/CPU; head-live via ssh). Figures
  are mtime-cache-busted. Scorers refuse to clobber stores with empty
  results (license-outage lesson).
- **Queue** run_n_sweep_queue: manifest mode, deployment-major, RAM
  governor, SCULPTOR_QUEUE_PASSES re-scans, .inprog markers,
  SCULPTOR_CELL_TIMEOUT. Manifests: tools/v3grid_manifest.json (1440),
  tools/a10_manifest.json.
- **Diagnostics per run**: convergence_over_iterations.pdf now carries
  the adaptive-WHEN row (K + surprise-vs-theta); companion
  model_error_over_iterations.pdf (belief vs GT + per-probe WHY
  annotations); result JSONs carry probe_log/gate_hist. Dash grid cells
  link both (superscript m).
- **Watchers**: Mac cron notifiers exist but alert JSON active:false
  keeps them quiet. Dash refresh loop + ticker survive on the Mac.
  Everything session-bound (Monitors, bg watchers) is DEAD — re-arm on
  resume.
