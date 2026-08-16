# Variable space of the model-uncertainty / measurement-budget study

The study asks one question along five axes: **when does an optimizer
guided by a model need real measurements, and which solver machinery
converts them into advantage?** Every experiment is a point in this
space; name the coordinates when reporting results.

## 1. Estimator quality — how well the solver can evaluate its own belief

| knob | values | meaning |
|---|---|---|
| `SCULPTOR_MC_NUM` | default 5; 1 | Monte-Carlo draws per benefit estimate. 1 = single-draw noisy estimator. It was ONLY EVER 5 — the good estimates never came from big MC. |
| `SCULPTOR_LB_CACHE` | default 1; 0 | Memoization of `benefit(A)` AND its pdf, keyed on the *thresholded* advertisement, invalidated **only by a real measurement**. Under starved probing nothing clears it → beliefs (including the uncertainty the probe gate consumes) freeze at their first draw. `0` = fresh MC every call (~2× slower). Discovered 2026-08-11. |

Interaction to remember: `MC_NUM=1` **with** the cache is a persistently
wrong estimator (one draw frozen per pattern); **without** the cache it
is an honestly noisy one. Different regimes, test both.

## 2. Probing budget — information supply

| knob | meaning |
|---|---|
| `SCULPTOR_ABLATION_PROBE_MODE` | `fixed` = stock (measure every changed step, ~125/run at small); `gated` = measure-XOR-step under total budget |
| `SCULPTOR_ABLATION_PROBE_N` | total measurement budget N (the x-axis of the N-sweep) |
| `SCULPTOR_ABLATION_PROBE_C` / `_AUTO_C` / `_TCONV` | gate threshold; static when AUTO_C=0 (C=2.0 pins the gate shut → zero probes, since U ≤ 1); annealed toward a U-quantile otherwise |

Known: at N=1 the annealing front-loads the probe (~iter 35–43 of 200).
Gate spend is rung-dependent (no_memory exhausts budget, full spends
14–27/50, no_mc can spend 0). Alternatives on the table: scheduled
probing (external clock), objective-worsening trigger (bypasses
self-assessment).

## 3. Deployment base uncertainty — how unpredictable the world is

| knob | values | axis |
|---|---|---|
| `SCULPTOR_LAT_MODEL` | `tiered` (default) / `geo` | toy 3-tier latencies vs geodesic×1.3 ± U(30–50)ms (10% tail to 100), floored at geodesic |
| `SCULPTOR_GEO_NOISE` | default 1 | multiplier on the geo noise spread |
| `SCULPTOR_LAT_SPREAD` | default 1 | tiered-model within-tier noise multiplier (weak lever — tiers absorb it) |
| `SCULPTOR_PREF_MODEL` | `structured` / `random` | ingress preferences: mostly-random-with-anchors vs full random permutation |
| `SCULPTOR_ROUTE_VIOLATION` | default .05 | structured-model priority-swap probability |
| `SCULPTOR_VOL_SPREAD` | unset / s | log-uniform UG volumes, CV up to ~1.4 at s=6 (see axis 4a for why this matters) |
| `SCULPTOR_SCALE_FACTOR` | default 1.1 | capacity headroom over anycast load; 1.0 = zero slack |

Canonical worlds: **stock** (all defaults) · **georand**
(`LAT_MODEL=geo PREF_MODEL=random` — the realistic internet spec) ·
**maxhard** (georand + `GEO_NOISE=2 VOL_SPREAD=6 SCALE_FACTOR=1.0`).

## 4. Objective difficulty — what functional must be estimated

Theory taxonomy (why avg latency is easy and failure is hard):

- **Easy: linear functionals with many light-weight units.** Avg latency
  is linear in per-UG routing probabilities → marginals are sufficient
  (E[f(X)] = f(E[X])) and CLT over 225 UGs self-averages the error.
  Measured: believed-LB error 0.01–0.4 everywhere.
- **(a) Heavy-tailed weights** break the CLT: whale UGs make the mean
  inherit single-UG uncertainty (effective n collapses). `VOL_SPREAD`
  is a mild version; a Zipf/power-law volume knob would be the strong one
  (NOT YET BUILT).
- **(b) Nonlinear functionals** — max/worst-UG, p95/p99, variance and
  fairness: depend on distribution shape, E[g(X)] ≠ g(E[X]), plug-in
  estimates biased, tails dominate. (NOT YET SCORED — cheap to add: p95
  readout beside the mean in probe.py.)
- **(c) Thresholds / rare events** — capacity overload is an indicator
  function; P(overload)≈p needs ~1/p samples. `NO_ROUTE_LATENCY=30000`
  is the extreme discontinuity (the γ≥0.3 collapse couples to it).
  Knob: `SCALE_FACTOR→1.0`.
- **(d) Joint dependence** — the popp-failure/RB term: a failure moves
  all its UGs *together*; whether backups congest depends on where they
  co-land. Marginals cannot answer this; it needs the joint distribution.
  Measured: believed-RB error 41–130 units vs 0.2 for LB (~300×), but
  ~constant per deployment (±1.6–3.6 across advs — the offset cancels in
  comparisons; the residual still rivals rung gaps). γ
  (`SCULPTOR_ABLATION_GAMMA`, studies use 0.1) sets how much this hard
  term steers training.

The current combined objective is easy-LB + hard-RB. The natural "hard
objective" experiment is raising γ / tightening capacity, not inventing
a new functional — (d) already is the canonical hard case.

## 5. Solver arms — what machinery consumes the information

The ladder rungs (see `experiments/ablation/README.md`): painter →
no_mc → no_memory → no_direction → expl_none → expl_random → full,
via `SCULPTOR_ABLATION_{MEMORY,DIRECTION,EXPLORE,MC}`. The study's end
question lives here: does memory/direction/exploration convert scarce
measurements into advantage, or is bounded simple search (no_memory)
enough — and in which region of axes 1–4?

## Measured so far (2026-08-11, cache/model_error/*.json)

- Iteration-0 beliefs: calibrated in every world; latency error tiny
  (structural: axis-4 easiness), all absolute error in RB.
- Per-decision err/signal ≥ 0.5 even at stock; ≥ 2.8 georand; 17 maxhard
  (single flips). Aggregates hide this — always report per-decision.
- Trajectory rot (cached, MC=5): after 20 measurement-free iterations
  the model is ~2.5× overconfident in every world; gradient err/signal
  2–21. Iteration-driven, not distance-driven. Cache freezing is the
  prime suspect; cache-off redo pending.
- Mini extremes (stock world): no_memory ≥ full at N=1 (5/5 paired) and
  ≈ at N=50; full blew up 2/10 under gating. Landscape forgiveness, not
  estimate quality, explains N-insensitivity at stock.
- Cache attribution (2026-08-11 evening, nocache redo): with
  `LB_CACHE=0` the k=20 calibration ratio recovers 0.40→0.99 (stock),
  0.38→0.48→0.73 (maxhard/georand, MC1→MC5), and absolute drift error
  drops up to 4×. Ranking of miscalibration sources: frozen lb-cache >>
  MC=1 pdf narrowness > residual (within n=5 noise). The U-gate's
  input is trustworthy iff the cache is cleared per-iteration (or off)
  in starved-probing regimes.
- Hard-vs-easy re-rank (fixed-mode ladder, 20 seeds): avg_lat barely
  separates the top 5 rungs; MLU separates memory/direction rungs
  (no_memory 0.305 / no_direction 0.327 vs 0.238 direction-bearing);
  popp-failure congestion is categorical (painter 18% mean congested,
  every ladder rung exactly 0) and DEGENERATE at stock capacity for
  trained solutions -- needs SCALE_FACTOR=1.0 worlds to discriminate.
- New objectives implemented + unit-tested (objectives.py):
  'frac_beyond_optimal', 'lat_plus_max_util', 'popp_failure_congestion'
  in the generic-LP contract (register() wires them into
  generic_lp_functions; benefit sign convention).
- DENSE MESH (2026-08-12, cache/ablation/mesh_georand): georand,
  {full,no_direction,no_memory,no_mc}+painter x N{1,2,5,10,20,50} x
  seeds 1-5, 100 iters, LB_CACHE=0, MC_NUM=1, gated. The expl_* rungs
  were RETIRED (traditional exploration is off the table under gated
  probing; probe-target selection folded into 'full'). Findings:
  (1) full is the only MC-arm that converts budget — median 83k at N=1
  -> ~1-5k at N>=2 (its budget threshold is N=2, not N=50); (2) no_mc
  NEVER collapses (0/30) and spends 0 probes (deterministic model ->
  U=0 -> gate never fires): the robustness arm; (3) no_direction is
  the uniform loser (26/30 collapsed) — memory-without-direction toxic;
  (4) seed heterogeneity >> everything: per-seed winners were no_mc
  (s1), no_memory@N50 (s2), full (s3, s4), everyone (s5 easy);
  (5) more budget can HURT crude machinery (probe iterations don't
  step: no_memory s3 healthy N2-10, collapsed N20/50). Config caveat:
  MC=1+nocache is hostile to MC-sampling arms (vs the 200-iter
  cache-on waves where full's ranking differed).

## 2026-08-14/15 — FINAL-SEMANTICS ERA findings (see HANDOFF.md for the fix stack)

- **Info-efficiency RESTORED** (policy_ladder_v2, monotone ladder, final
  gate): paired per-seed, L6 full+smart beats L4 no_mem+smart by ~8
  train-objective units at N=1-2 (5/5 seeds at N=2), parity at N>=20.
  Every earlier "memory hurts" verdict was the remeasure-stop /
  exit-on-budget self-truncation chain (L6 was spending <=11/50 probes
  and stopping by iter ~37).
- **L1 budgeted-fixed (measure-then-exit) is worst everywhere and
  non-monotone in N**: it exits mid-descent wherever iteration ~N/2
  lands (N50 mean 180 vs L6 140; 17% steady stranding at N50).
- **L3 no_mem+sched is the sleeper**: evenly-spaced grounding +
  congestion-aware beliefs ~= most of the smart gate's value at N>=5;
  the gate's edge concentrates at tiny N.
- **Sparse arms legitimately beat one-per-peering on the popp-failure
  composite** (opp advertises all popps so every failure congests
  someone; sparse arms get free no-op failures). Below-opp on STEADY
  latency remains impossible (assert-enforced).
- **Step paradigms** (seeds 1-3, TCONV-matched, figures/
  paradigm_comparison_s123.png): auto-scale vs fixed-alpha endpoint
  deltas within seed noise; alpha=0.2 doubles flips + halves no-op
  iterations (~30% waste at best) at no quality cost. The flip
  "guarantee" in _rescale_gradient is approximate (~43-55% realized;
  momentum/prox interference). Open: decisive-flip variant.
- **Gate census** (pre-windowed-c): L6 ~30% backstop-fired — the
  full-history U-quantile lags nonstationary U; windowed c (TCONV/2)
  now default. Probe bookkeeping (reasons/refractory/surprise) commits
  only on ACTUAL measurements as of 2026-08-15 (skipped probes
  previously polluted probe_reasons + c-trajectories).
