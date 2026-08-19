# reducing_iteration_timing

Charter (Tom 2026-08-18 evening): drive down SCULPTOR iteration time,
focused on the L6 (slotted WHEN) path — gradient computation above all
— plus startup beyond deployment creation. Autonomous workstream on
the head (repo copy ~/rit_repo; no source edits under live campaigns).

Ground rules:
- Metrics tracking STAYS (Tom: worth 5-10% for debuggability). Only
  remove provably redundant computation inside it.
- For every idea: smoke at small size, then REAL verification at
  actual-15 (minimum) / actual-25 (ideally). Small-size wins that
  vanish at scale are not wins.
- Gradients dominate: a 10ms/call save x millions of calls is real.

## Idea backlog (Tom's a-d + derived)

a) STARTUP (not deployment creation — that's depsetup_fork, merged):
   object construction, worker start, the 5-15 min "silent phase"
   pickling/broadcasting deployment state to workers. Candidates:
   per-worker construction from baked shards (kills N x pickle),
   lazy attribute construction, ray object-store zero-copy arrays.
b) INTER-NODE COMMS: per-call pickling of job dicts / results between
   driver and pdc workers. Measure serialize sizes + times first.
c) SOLVER CALL SETUP: persist LP model objects across gradient calls
   (gurobi persistent-constraint trick -> HiGHS incremental API:
   keep the Highs instance, changeColsCost/changeCoeff deltas only).
   Possibly warm-start consecutive gradient LPs (pending Tom ruling
   on bitwise-identity requirements).
d) PDC DISPATCH: job receipt -> lower-level solver call (route
   simulation, dict assembly, cache lookups) inside
   path_distribution_computer.

## Measurement protocol

- Phase budget from live logs ([Timing] lines, Timer blocks, [mem]
  iter stamps) at real sizes before touching anything.
- Micro-harness per candidate (bench_<idea>.py) with A/B at fixed RNG.
- Verification: single L6 cell at actual-15 / actual-25, phase budget
  before vs after, same-seed (noise floor caveats apply — compare
  phase timings, not objective trajectories).

## Findings log

- 2026-08-18 F1 (free, from live a10x10 L6 logs, actual-10, 16
  workers, 40 iters): **grads 64.1 s/iter (82%)**, **stop 13.9 s/iter
  (18%)**, measure 0.10, info ~0, worker-notify 0.003, resilience
  0.026, GT-latency-benefit 0.27. => (b) comms is a non-issue at this
  scale; the whole game is INSIDE the gradient computation, with the
  per-iteration stop/believed-objective evaluation a clear second
  target. Worker logs carry NO internal gradient timing — first build
  is env-gated instrumentation (SCULPTOR_RIT_PROF=1) of the pdc
  gradient call chain: job deserialize / route-sim / LP setup / LP
  solve / result serialize per call.
- Head repo copy at ~/rit_repo (cache+data symlinked) for workshop
  code; live campaigns untouched.
- 2026-08-18 F2 (free, live v5full mlu worker batch summary): the
  worker's gradient batch splits **solve_generic_lp_not_persistent
  43%** / **route simulation (sim_rti + total_rti_calc) 38%** /
  pmat_organize 5.6% / unified-LP setup 6% / actual optimize() only
  3.3% / get_paths_by_ug 3%. => Candidate (c) has a precise target:
  the persistent-LP path EXISTS but only avg_latency/per_site_cost
  use it — every family objective (mlu/fracb/prio) rebuilds its LP
  per MC sample per candidate. Since optimize() itself is ~3%, the
  rebuild overhead is most of the 43%. Candidate (d) = the 38%
  route-sim block. Next builds: (1) persistent/incremental generic-LP
  path in rit_repo + A/B smoke; (2) decompose not_persistent
  internals (setup vs solve); (3) sim_rti vectorization pass;
  (4) dissect the driver-side 'stop' 13.9 s/iter; (5) startup budget
  at actual-15/25 once the head frees.

- 2026-08-19 A/B VERDICT (actual-15, seed 201, rung full, max-iter 12,
  head otherwise idle, ~/rit_repo + ~/rit_out_a15): persistent-inner-LP
  candidate wins but NOT big — `Timer: grads` 1227.22 s (base) vs
  1015.49 s (persist) = **-17.3%**; worker not_persistent bucket
  1184.7 s -> 977.5 s (-17.5%, consistent). Driver `stop` 11.7 -> 10.0 s.
  Instrumentation note: the persist arm still logs its LP time under
  `solve_generic_lp_not_persistent` (the `_persistent` bucket reads 0.0
  in BOTH arms) — the wrapper label doesn't move when the persistent
  path engages; time savings are real (wall + grads timer agree), but
  the bucket split is not to be trusted for attribution.
  DECISION: modest win, below the "wins big" bar the handoff set for
  adopting into EODS-25 pre-gate; EODS-25 runs mainline. Candidate stays
  open pending byte-exactness/quality gate + a look at whether the
  persistent path actually engaged for the family objectives (0.0 bucket
  is suspicious in the engage-direction too).
