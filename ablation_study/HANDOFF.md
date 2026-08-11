# Ablation study handoff (2026-08-11) — IN PROGRESS

Read this + `experiments/ablation/README.md` (mechanics/agent notes) first.
Supersedes the 2026-08-08 handoff. Worktree branch
`claude/elated-blackburn-13c10e` holds the authoritative code (main repo +
VM are synced copies — ALWAYS md5-verify after deploying anywhere).

## ⚠ Live state right now

- **VM head i-0428c395787bc3ca0 (c7g.16xlarge, 100.54.8.15) is RUNNING
  and has NO auto-teardown** — the finalize watchers were disarmed. Tom
  wants the big instance kept while dev continues; STOPPING IT WHEN WORK
  PAUSES IS A MANUAL RESPONSIBILITY. Update
  `~/.sculptor_cluster_alert/active_cluster.json` on lifecycle events.
- **In flight**: the "minimal extremes" test — no_memory vs full ×
  N∈{1,50} × seeds 1–5 × 200 iters, gated mode, via the queue driver
  (`logs/nsweep_mini_chain.log` on head; out
  `cache/ablation/nsweep_mini`). Question it answers: does full SCULPTOR
  beat no_memory under scarce measurement (N=1) and does the gap vanish
  at N=50? Its verdict decides whether the full N-grid is worth running.
- Tom's standing rules: **test code before using it** (smoke first, every
  time); only validated experiments may occupy the machine; hourly status
  reports during long runs; no multi-seed sweeps without his go.

## The experiment program (Tom's design)

Two threads:
1. **Ladder ablation** (painter → no_mc → no_memory → no_direction →
   expl_none → expl_random → full): quantify each SCULPTOR feature on the
   REAL solver. Fixed-mode 20×200 study DONE (see Results).
2. **Measurement-budget (N) study**: gated probing = measure-XOR-step
   under a TOTAL budget N (stock SCULPTOR measured every step). U =
   composed LB+RB sign-error probability; auto-learned threshold c
   (quantile-targeted, annealed, refractory-doubling with tau scaled to
   intended probe spacing). Hypothesis: small N separates the rungs
   (information efficiency matters), large N collapses the gap.

## Results so far (ALL numbers from rescore_fork — never in-process)

- **Fixed-mode 20×200 clean-v3** (`cache/ablation/fork_small_20x200_v3`,
  rescored, = canonical init source + N=∞ anchor): medians — all rungs
  close ~most of painter's gap (painter median combined +784; rungs
  ~1.1–1.6; no_mc ~5 = bottom link verified). THE TAILS carry the story:
  catastrophic blowups out of 20 — no_direction 5, no_memory 3, no_mc 1,
  full 1, expl_none 0, expl_random 0. Plus one rescored fixed-mode
  replica (`cache/ablation/fixedmode_replica1`, n=14): pooled rates ≈
  no_memory 21%, no_direction 18%, direction-bearing 0–3%. Blowups are
  mostly STOCHASTIC (seed sets differ across replicas; seeds 1 and 12
  look high-risk). Interpretation Tom liked: **monte-carlo/LP modeling
  buys the mean; memory and direction buy the tail; memory WITHOUT
  direction is a noise integrator** (removing the remeasure carryover
  exposed this — see below).
- **Remeasure clearing** (direction-off rungs forget cross-iteration
  gradient info): raised no_direction's blowup rate vs the old 5×200
  study — the carryover had been an accidental stabilizer. Old-semantics
  datasets are labeled in README.
- **In-process eval contamination CONFIRMED** (multi-seed shared-worker
  scoring turned a +21k blowup into +0.99). All scoring must be fresh
  per-seed processes; eval_ladder_metrics has per-seed isolation + a
  canary vs rescore. The repo's own evaluate_all_metrics shares the
  hazard for paper historicals — open investigation for Tom.
- **Gamma bisection** (seed-1, 50-iter smokes): γ=0.1/0.2 stable,
  γ≥0.3 collapses (RB:LB gradient-scale pathology; NOT fixable by the
  NO_ROUTE penalty knob alone — γ=1+penalty1000 goes failure-blind
  instead). `SCULPTOR_NO_ROUTE_LATENCY` env-gates the sentinel for
  training only. Studies run γ=0.1.
- **Gated probing so far**: ratchet bug (permanent c-doubling) capped all
  arms at ~1–2 probes → first N-sweep was invalid (retired). Refractory
  decay fixed it (smoke: N=5 → 4 probes spent, +0.52). Fixed tau=10 then
  capped spending at ~iters/10 (N=50 spent 10, N=100 spent 8) → tau now
  scales with intended spacing (deployed, validated only in the mini test
  in flight). Sleeper finding: ~1–4 measurements ≈ fixed-mode median
  quality at small (fixed mode spends ~125/run) — a big
  measurement-efficiency claim if it survives the validated rerun.
  KNOWN GAP (Tom aware, not yet approved to build): U cannot detect
  confident divergence — a run that diverges keeps low U (precise, wrong
  beliefs). Proposed: objective-worsening trigger (probe/flag when
  believed objective degrades k iters running). Multiple blowups would
  have been caught by it.

## Dataset inventory (cache/ablation/)

TRUSTED: `fork_small_20x200_v3` (fixed clean-v3, rescored) ·
`fixedmode_replica1` (rescored) · `fork_5x200` (old semantics, rescored) ·
`ladder_{small,a10}_eval_stats.pkl` (repo-metrics via eval_ladder_metrics).
RETIRED/QUARANTINED: `nsweep` (ratchet gate + killed cells, AUDIT FAILED)
· `nsweep_STALE_fixedmode_replicas` on head (stale-code deploy; N2..N20
subdirs are UNRESCORED fixed-mode replicas — useful for more blowup-rate
n) · `nsweep_v2_UNVALIDATED_GATE` on head (pre-tau-fix gate).
IN FLIGHT: `nsweep_mini` (the extremes test).
`fork_a10_v2` (actual-10 seeds 1–4, OLD semantics, killed at 25/30):
usable but predates remeasure clearing; actual-N rescoring must run ON
THE TRAINING HOST (deployment builds are measurement-cache-dependent).

## Code map (experiments/ablation/)

`sculptor_fork.py` — all flags (see its docstring; MEMORY / DIRECTION /
EXPLORE / MC / PROBE_MODE gated|fixed / PROBE_N / AUTO_C / TCONV / FRAC /
MULT_TAU auto-scaled), per-iteration binding assertions, probe gate.
`mc_off_worker.py` + ACTOR_CLS seam — no_mc rung. `run_fork_ladder.py` —
one cell; GC now CWD-scoped (was repo-absolute: caused cross-workspace
checkpoint deletion twice). `run_n_sweep_queue.py` — THE harness for
grids: global cell queue, no straggler tail, mandatory pre-seeded inits,
audit gate incl. stale-code guard (probe_mode must match), built-in
rescore. SIZING: slots×workers = Gurobi sessions ≤ ~28; memory ≈
4GB/slot — 28 slots OOM'd a 123GB box alongside other work; 20 is safe.
`run_n_sweep.sh` — legacy lane harness (straggler-prone; superseded).
`rescore_fork.py` (+SCULPTOR_RESCORE_STORE_SCENARIOS) · `table_fork.py`
(incl. painter-anchored quantile table) · `cdf_fork.py` · `plot_n_sweep.py`
· `eval_ladder_metrics.py` · `test_mc_off_unit.py`.

## Operational gotchas (each cost us real time)

1. **Deploy = scp + md5-verify + banner-check.** A stale deploy burned 7h
   producing fixed-mode replicas labeled as an N-sweep. The audit's
   probe_mode guard now catches it — keep that pattern for new flags.
2. **Remote pkill self-match**: bracket patterns (`fork_ladde[r]`) and
   never write the unbracketed name ANYWHERE in the same remote command
   (including echo/env text). Multiple ssh sessions killed themselves.
3. **Never run run_fork_ladder with cwd=repo while anything else runs**
   — use a workspace dir (runs/logs/figures + cache/data symlinks).
4. **RAY_ADDRESS=local + unique RAY_TMPDIR for every side-job** or it
   attaches to a neighbor's Ray and dies with it.
5. Lane/queue exit codes lie ("0 failures" while runs died) —
   ALWAYS audit JSONs: solve_error absent, n_iters ≥ MAX_ITER+1 (the
   off-by-2 is sometimes off-by-1 — don't flag 201), probe_mode correct.
6. Same-seed single trials are noise (probe RNG unpinned); blowups are
   probabilistic — compare distributions/rates, never single runs.
7. Orchestrator scripts: verdict-line-gated waits (process checks
   self-match); detach with setsid/start_new_session (harness-child
   nohup gets reaped).

## Next steps (in order)

1. Collect `nsweep_mini` verdict (watcher task bhnson35h). If the
   extremes trend is interesting → full N grid via run_n_sweep_queue
   (20 slots, all 7 rungs, N grid per Tom, ~4–6h); else redesign with
   Tom (he floated 'scheduled' probing as an alternative arm).
2. Whatever runs next: smoke first (Tom's rule), md5-verified deploy.
3. Pending Tom decisions: divergence trigger (recommended, small);
   rescore remaining fixed-mode replicas for tighter blowup rates
   (cheap); a10 clean-semantics rerun; repo-pipeline contamination
   investigation; gamma/soft-penalty workstream (needs objective
   redesign, not penalty knob).
4. When work pauses: STOP THE INSTANCE (manual!), update alert JSON,
   pull any unpulled results first (rsync dirs listed above).
