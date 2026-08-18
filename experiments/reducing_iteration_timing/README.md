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

(appended as measured)
