# Ablation study handoff — V3 ERA (2026-08-16, WHAT/WHEN ladder)

## READ THIS SECTION ONLY. History lives in old_handoffs/ — consult for
## the why, not the what.

Single source of live state: `~/.sculptor_cluster_alert/active_cluster.json`
(head IP lives there; EVERYTHING must re-resolve it per-use, never cache).

### CURRENT STATE (2026-08-16 ~12:40Z)

- Head (c8g.24xlarge, i-0428c395787bc3ca0) is UP and IDLE except the
  completed v3 smoke. NO teardown — awaiting Tom's GO on the full grid.
- **v3 SMOKE PASSED**: 7 arms x seed 1 x N=5 x 100 iters in
  `cache/ablation/policy_ladder_v3/` — no errors, budgets respected,
  L1 exits at iter N, L7's min-gap guard held. Visible on the dash
  (policy-ladder tab, big objective-only figure).
- **READY TO LAUNCH**: `cluster/chain_v3.sh` + `cluster/manifests/v3_full_manifest.json`
  (28 specs, 840 cells: {classic g=.1, fracb, mlu, prio} x L1-L7 x
  seeds 1-5 x N{1,2,5,10,20,50}, 100 iters, deployment-major across all
  lanes). Deploy both to the head (~/), then:
  `setsid nohup bash chain_v3.sh > /dev/null 2>&1 &`
  Verdicts in logs/v3_driver.log; watcher `cluster/watch_v3.sh`
  (session Monitor, 30-min beats). Est ~12-16h on the governed pool.
  The 7 smoke cells are reused (queue skips existing JSONs; identical
  config).

### THE V3 LADDER (Tom-ratified 2026-08-16: smart probing decomposed
### into WHAT vs WHEN, pushed to the top of the ladder)

Probing is PURE GROUNDING (probe = measure the CURRENT advertisement,
`SCULPTOR_ABLATION_PROBE_TARGET=current`) on the fixed schedule
through L5; the smart deltas are isolated at L6/L7:

| arm | dir | adds | probe mode / target |
|---|---|---|---|
| L1 | L1_nomc_fixed | budgeted-fixed (measure first N iters, exit) | fixed / current |
| L2 | L2_nomc_sched | evenly-spaced measuring | sched / current |
| L3 | L3_nomem_sched | congestion-aware belief LP | sched / current |
| L4 | L4_nodir_sched | memory (continuous advs, 1-coord steps) | sched / current |
| L5 | L5_full_sched | direction + solver-side explore | sched / current |
| L6 | L6_full_schedwhat | +smart WHAT (max-info targeting) | sched / maxinfo |
| L7 | L7_full_smartcons | +smart WHEN (gate, min-gap-guarded) | smart / maxinfo |

L7 conservatism (Tom: gate "fires a bit too early"): criteria (a/b/c)
cannot fire before `SCULPTOR_ABLATION_SMART_MINGAP_FRAC` (0.7) x the
scheduled spacing since the last measurement; the (s) backstop is
exempt. OLD L7 (Bernoulli-K3 gradient base) is PAUSED — do not rerun;
its flag stack (SCULPTOR_ABLATION_GRAD_BASE/_K) remains in the fork.

### OBJECTIVES (all four lanes, final semantics)

- classic: avg_latency + 0.1*resilience (the only gamma>0 lane).
- fracb: `frac_beyond_optimal` — scalar = HINGE capability metric (min
  achievable excess-ms beyond opt+10ms; opp floor EXACT, via
  solve_lp_assignment.solve_min_hinge_excess). Old assignment-derived
  fraction kept as the 'frac_beyond' component. Revert:
  SCULPTOR_FRACB_SCALAR=assign.
- mlu: STANDALONE `max_util` v2 = -(A*minMLU + routed_lat + 3A*bad_frac),
  A = SCULPTOR_MLU_WEIGHT_MULT(10) x optimal floor (~90% weight on MLU;
  latency = tie-break; stranding provably unprofitable, bounded, never
  the 30s sentinel). minMLU via THE canonical
  `solve_lp_assignment.solve_min_mlu` (Gurobi). Dead ends (never
  resurrect): force_mlu fallback Y (concentration artifact),
  assignment-peak MLU (degenerate ~1.0). `lat_plus_max_util` remains
  registered (the v2-era mlu-redo data used it).
- prio: `joint_latency_bulk_download` (two-stage assignment-derived —
  small opp crossings are metric-structural; capability twin is a
  nonconvex QP; annotated on the dash panel; OPEN DECISION for Tom:
  (a) jointly-linear redefinition (exact opp floor) vs (b) keep with
  the documented exemption).

### POST-FIX SEMANTICS (both defaults, deployed everywhere 2026-08-16)

1. **lambduh == 0 in every file** (call sites + constructor defaults;
   prox = identity; no L1 cost term; set_alpha branch unchanged -> .01).
2. **SCULPTOR_SIG_CUTOFF=p5**: remeasure-significance cutoff = 5th
   percentile of the prior iteration's |gradient| distribution (LB +
   RB-popp sites in sparse_advertisements_v3). 'abs' restores the old
   absolute .01. ROOT CAUSE it fixes: the absolute cutoff wiped
   remeasure persistence on quantized fraction-scale objectives (the
   fracb L6 freeze: 50/100 starved iterations -> 0/100 under p5;
   A/B-proven, prio unaffected — continuous scalar).
   ALSO ESTABLISHED: the min-Y fallback LP does NOT drop paths
   (verified empirically); its NO_ROUTE sentinel marks are DESIGNED
   congestion pricing; classic-lane v2 data was never contaminated.

### INFRASTRUCTURE

- **Scheduler**: `run_n_sweep_queue.py --manifest` = many cell-group
  specs, ONE global slot pool (no static partitions), per-spec env /
  gamma / probe-mode / dpsize / init_src / out_root / max_iter, global
  --launch-stagger (build-thrash guard), per-spec audit+rescore, inline
  per-cell convergence-fig harvest (spec 'artifacts_figs';
  '<label>_<suffix>.pdf' names). Single-spec CLI mode unchanged.
- **RAM governor** (MemGovernor in the queue; config
  `~/queue_governor.json`, LIVE-tunable per decision): admits cells
  toward 90% RAM with EWMA per-cell estimates + spike reserve;
  max_active is sized by RAM/cores ONLY (Tom 2026-08-16: the "~28
  Gurobi session budget" was a myth — official academic-WLS baseline is
  2 sessions yet 20-48+ sustain fine empirically; oversubscription just
  waits. See WLS policy note in experiments/ablation/README.md). Proven
  overnight including throttle/recover cycles at 91-92%.
- **Dashboard** (`dashboard/`, localhost:8643, refresh
  loop `python -m dashboard.refresh --loop 180`, log
  /private/tmp/dashboard_refresh.log): policy-ladder v3 tab (big
  objective-only figure `policy_ladder_v3_5panel_objective.png` + 7-arm
  conv-link grid with per-arm L<k>_ filename prefixes) + hard-objectives
  v2 tab (replace with hardobj_v3 sections when the grid lands — clone
  the v2 registry pattern; scorer OBJ_OF/WORLD_OF need the hardobj_v3
  root wired). Sanity gate `sanity.py::assert_not_better_than_opp` in
  plot_hardb3 (popfail + prio exempt, documented); score_hardb3 prunes
  deleted files + persists objective components; per-seed opp
  normalization EVERYWHERE (absolute axes bury good seeds — opp spans
  30x across seeds); blank panels are data-driven-dropped.
- **PROCESS WATCHERS — session-bound, hand over explicitly.** Three
  layers, only one survives an agent session:
  1. SURVIVES SESSIONS: the Mac cron notifiers
     (`~/.sculptor_cluster_alert/liveness_check.py` every 10 min +
     `heartbeat.py` every 3h -> SMS) and the Mac dashboard refresh loop
     (nohup; check `pgrep -f dashboard.refresh`, restart
     per dashboard/README.md if dead). Head-side setsid
     chains also survive (verdicts in their driver logs).
  2. SESSION-BOUND (dies with the agent): the Monitor wrapping
     `cluster/watch_v3.sh` on 30-min beats. A NEW AGENT MUST RE-ARM THIS
     FIRST, before any other work:
       Monitor(persistent) running:
       `while true; do cluster/watch_v3.sh; sleep 1800; done`
     watch_v3.sh re-resolves the head IP from the alert JSON per beat;
     verdict lines are COMPLETE / FAILED / SSH_ERROR / RUNNING. During
     heavy pools also watch memfree in the beat (the governor protects,
     but eyes on it) and disk (<6G: clean dead /tmp/ray_q_* session
     dirs — date-pinned patterns only, never the live pool's).
  3. Standing rules: completion is COUNT-based (JSONs vs targets) +
     per-spec rc lines, never exit codes alone; cron heartbeats are NOT
     reliable as the sole watchdog for dormant sessions (memory);
     update the alert JSON on EVERY lifecycle event or the SMS crons
     false-alarm.
- **Email**: `python cluster/send_report.py "<subject>" <body.txt>
  [figures...]`.
- **v3 eval stores**: tags policy_{steady,failure}_v3. v2-era ladder
  stores quarantined in `cache/model_error/V2_ERA/` — their L1/L2 dir
  names collide with v3's; never move them back into the live glob.
- **Local-Mac standalone runs**: use the queue's ws recipe (PYTHONPATH,
  short RAY_TMPDIR, figures/logs dirs, cache/data symlinks) — see
  memory local-fork-ws-recipe.

### DATASET TRUST TABLE

- `policy_ladder_v3/` + (pending) `hardobj_v3/`: THE current era
  (post-fix semantics, 100 iters, WHAT/WHEN ladder).
- v2 era (valid under pre-fix semantics + old ladder; superseded for
  the ladder question, keep for reference): `policy_ladder_v2/`,
  `hardB3v2/{fracb,prio}` (L1-L6 complete + clean), `hardB3v2/mlu`
  (lat_plus redo, complete), `policy_ladder_v2_L7K3/` +
  `hardB3v2_L7K3/` (old-L7 grid, killed at 107/120).
- Quarantined: `PUREMLU_STRANDING_ERA_mlu` (head), `PRESKIP_ERA_v2/*`,
  `WRONGWORLD_STOCK_EVAL/*`, `cache/model_error/V2_ERA/`.
- PAUSED ideas: old L7 Bernoulli-K3 (Tom: waste of compute; flags
  remain in the fork); zipf knob (needs reformulation: per-UG-PRIVATE
  zipf prefs x tight caps — current global-popularity form makes the
  problem EASIER; SCULPTOR_ZIPF plumbing works, z=0 bit-identical);
  a10/actual-10 grid (canceled pre-launch; recipe in git history).

### NEXT STEPS

1. Tom's GO -> deploy tools/{chain_v3.sh,v3_full_manifest.json} to
   ~/ on the head, launch (command above). Deployment-major: seed-1's
   full 4-objective block lands first — report per-deployment early.
2. When hardobj_v3 cells land: add dash registry sections for the
   three hard objectives (v2 tab pattern) + scorer root mapping.
3. On V3 COMPLETE: audit (classic 210 + hard 630), email Tom
   tables+figures, then ASK about teardown (never auto-stop while Tom
   is iterating).
4. Open decisions parked with Tom: prio metric redefinition
   (linear-joint vs exemption); zipf reformulation.

---

Historical era notes are archived in old_handoffs/:
- HANDOFF_2026-08-15_16_fix_stack_and_v2_eras.md — the v2 era, old-L7
  Bernoulli-K3, MLU bug forensics, freeze root-cause, fix stack,
  balancer/governor build-out.
- HANDOFF_2026-08-14_stability_and_fix_eras.md — stability stack,
  objective-bug fixes, policy-ladder eras.
