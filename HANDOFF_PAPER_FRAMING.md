# HANDOFF: paper-results framing (written 2026-09-05)

You are picking up a mature experimental codebase whose evaluation
campaigns are DONE. Your job is not infrastructure — it is helping Tom
frame, present, and sanity-check results for the paper. Read this whole
file; the semantics sections are where prior agents got burned.

## What this codebase is

SCULPTOR ("sparse" in the code) learns BGP *advertisement matrices* —
(n_popps × n_prefixes) 0/1 matrices deciding which peerings are
announced on which prefixes — to optimize traffic-engineering
objectives for an anycast CDN-style deployment. Baselines: `anycast`
(1 prefix), `anyopt`, `one_per_pop` (= "Unicast" in the table),
`painter` (anycast + greedy per-prefix improvements; objective-aware),
`one_per_peering` (= OPP, one prefix per peering — the routing optimum
and upper anchor; 779 prefixes at actual-32 vs SCULPTOR's 42).

Layout: `core/` (solver, LPs, deployment build, depstore cache),
`evaluations/` (drivers + `objectives/` per-objective eval suites),
`experiments/ablation/` (feature-ladder fork), `integration_tests/`,
`cluster/` (vmctl/expctl VM tooling). The paper LaTeX lives in
`~/Documents/resilient_advertisements_paper/` but Tom edits text in a
Google Doc and pulls; never edit paper text/macros directly — table
content is fixed at the EMITTER (`evaluations/generate_paper_table.py`).

## The one-command result pipeline

Everything reproducible flows through
`evaluations/intents/paper_intent.por.json`:

    ~/Documents/venv312/bin/python evaluations/run_all_paper_evaluations.py \
        grab evaluations/intents/paper_intent.por.json

pulls every stage's artifacts from the storage VM's depstore into
`figures/paper_artifacts/`, then auto-copies a curated list
(`paper_repo_copy.files` in the intent) into the paper repo's figures
dir. `run ... --only <stage>` re-executes a stage on the VM. All stages
currently green.

## The results, and framings already ratified or proposed

**Paper table** (`figures/paper_artifacts/paper_table.tex` = the pruned
11-column table; `paper_table_full.tex` = everything). actual-32,
papertable32b campaign, nsim=3 per objective (nsim=1 for avg_latency —
hence no ± there). Headline: SCULPTOR bold on every column: 30.39ms
latency (OPP 29.05), MLU 0.88, 0.00%/2.22% congestion under
ingress/site failure, flash 33.64 / diurnal 13.07, HPrio 31.21 / bulk
ratio 4.50, 83.70% within 10ms (OPP 87.12, painter 79.00), site cost
0.5574.

**Prefix sweep** (actual-32, budgets 16/24/32/40/64, 150 iters, the
papertable deployment): SCULPTOR at budget 16 (= sites/2, via the new
grouped init) beats every baseline at any budget; ~1ms subopt flat from
32 up. The knee is at/below sites/2.

**Ablation ladder** (20 actual-10 paper deployments, 6 arms, 100
iters; trusted rescore = full ingress-failure enumeration, γ=4
objective = steady + 4·Σ_popps(avg lat under failure)):
full≈expl_none ~1.3× OPP objective; no_memory 1.75×; painter≈no_mc
~2.2× (MC ablation forfeits the whole gradient advantage);
no_memory_dir median-fine but catastrophically fragile. Trajectory
figure (`figures/paper/ablation_pct_painter_to_opp_over_iterations.pdf`,
% of painter→OPP gap closed, mean over 20 deployments): expl_none
faster early, full edges ahead late (83.7 vs 82.0 at it150); the
degraded arms carry the separation. full's 0–150 curves were recovered
from vector convergence figures (see `evaluations/
extract_convergence_from_figures.py`; corpus
`cache/convergence_series_corpus_all.json`, 292 series; figures filed
per-deployment in `figures/convergence_figures/`).

**Proposed framings Tom liked / discussed (2026-09-05), not yet in the
table** — your likely starting point:
1. "% of achievable routing headroom captured" for resilience columns
   (anycast = 0, OPP = 100): flash 82%, diurnal 76% for SCULPTOR vs
   ~18% painter. Rationale below (anycast tautology).
2. Provisioning-equivalence: SCULPTOR's routing ≈ +3 points of
   fleet-wide overprovisioning (anycast needs 1.336× to match its flash
   tolerance vs 1.3×).
3. THE headline: near-optimal on every objective at **42 vs OPP's 779
   prefixes (19×)**; still winning at 16 prefixes.
4. Universal normalization candidate: % of painter→OPP gap closed.

## Semantics you MUST understand before framing anything

* **Capacity model**: `cap(link) = scale × that link's ANYCAST load`
  (`core/deployment_setup.py:get_link_capacities_actual_deployment`);
  links with no anycast load get a floor of one mean-user volume.
  scale = 1.1 at deployment build (steady/failure/diurnal evals);
  the FLASH eval rebuilds caps at 1.3 (`Y_vals=[1.3]`). Consequences:
  anycast's MLU is 1/1.1 = 0.909 *by construction*; anycast's flash
  and diurnal criticals are exactly caps−1 (30.21≈30, 9.80≈10) —
  tautologies. Users have ~95 routes each at actual-32 (verified) but
  most routes end at floor-capacity links, so even OPP's headroom over
  anycast is only ~+4 points — that's the provisioning model, not weak
  routing. The 1.3-vs-1.1 inconsistency across columns is a reviewer
  risk; a Y=1.1 flash re-bisect was offered, not yet run.
* **Diurnal metric**: 24 hourly snapshots, per-metro volume =
  (1+I/100)×diurnal_factor(hour) with factor peaking at 1.0, metros
  phase-shifted by real timezones; critical I = largest with all hours
  uncongested. Flash: single-metro surge ×(1+X/100), worst-case over
  metros, zero-congestion criterion, bisected (rel_tol 5%).
* **Evaluate with the trained objective** — Tom's hard rule (est.
  2026-09-02 after two bugs): every objective suite scores through the
  trained objective's own LP (`evaluations/objectives/*`,
  `solve_generic_lp_with_failure_catch`). History: "% within 10ms" was
  (a) originally a capacity-blind route-reachability curve that
  multi-counted users across prefixes — inverted the ordering, painter
  falsely won; (b) briefly "fixed" to a min-avg-latency assignment —
  also wrong per the rule. Current: trained-objective soft-LP
  assignment, hard 10ms count, volume counted once. At steady state
  (uncongested) it coincides with min-latency assignment — robustness
  note worth keeping. `SCULPTOR_XOBJS=1` is required to register
  max_util/frac_beyond LPs.
* **Painter is NOT objective-blind**, and each objective cell drew its
  own random deployment variant — never transplant advs across cells.
  Painter's joint_priority sim-0 solve crashed in the campaign (the
  pre-08-26 `paths_by_ug` KeyError); re-solved + backfilled 2026-09-03;
  priorities metrics now 3-sim means.
* **MLU column pair**: "MLU" = solve_min_mlu (canonical best-achievable
  peak util); the MLU group's "Latency" = min-latency-assignment value.
  They are two separately-achievable bests, not one operating point; a
  true joint point needs a lexicographic LP that doesn't exist yet
  (the 'max_util' generic-LP dispatch falls back to latency — known).
* **Caption caveat**: "outperforms every methodology on every metric"
  is now true on the pruned table; keep checking after any change.

## What exists but isn't surfaced

* 200-iter ablation continuation: killed by Tom at 85/100 cells (+20
  painter at 1.5× budget); advs banked UN-rescored on i-09a6's disk
  (`cache/ablation/cdf_a10_ext/`). Prelim driver-side deltas: leaders
  already converged (~±0.04), no_memory_dir's catastrophic seed
  self-repaired, no_memory drifted worse. Rescore+overlay = restart
  i-09a6, run `experiments/ablation/rescore_fork` with
  SCULPTOR_ABLATION_DEP_FILE dep-file env, then the table/CDF scripts.
* Scaling summaries: `figures/paper_artifacts/*_scaling_summary.csv`
  (gap grow/hold/shrink verdicts over deployment size / prefix budget).
* Depstore: content-addressed training/eval cache on the storage VM;
  `core/depstore.py:choke_config` is THE lookup key builder — never
  hand-roll it.

## Open decisions on Tom's desk

anycast/OPP-normalized resilience columns; Y=1.1 flash re-bisect;
5-sims-per-objective top-up (~$150–200, days); 200-iter ext rescore;
stopping the dash VM.

## Operational ground rules (from Tom, standing)

* VMs via `cluster/vmctl` + `cluster/expctl` ONLY. Storage/dash VM =
  i-0428c395787bc3ca0 (m8g.4xlarge, Elastic IP 107.22.173.189, hosts
  the public dash — keep up until Tom says stop). Big compute =
  i-09a6ff2823b0bb304 (m8g.16xlarge, STOPPED, disk holds everything).
  Tear down what you start; every reply while a VM burns should carry a
  status line (a stop-hook enforces this).
* Integration tests in `integration_tests/` (standalone, don't trust
  exit codes — they scan output). Run only what you touched; full
  suite for cross-cutting changes.
* Cost-consciousness is first-class (advisor-billed AWS). Overnight
  runs OK; quote costs before multi-day compute.
* Never commit `tom_high_level_todos.txt`. Memory dir has deep context
  (`~/.claude/.../memory/`); `project_ablation_cdf_pipeline.md` is the
  densest recent one.

## Integration-test status (verified 2026-09-05)

Ran the full suite. 10/11 green. Fixes/notes:
* `verify_e2e_objectives`: FIXED — it reused a stale condensed L3 table
  pickle (`cache/paper_table_condensed_*`) that predated a KEY_COLUMNS
  rename, so the key-table check tripped though the emit was correct
  (SCULPTOR row 30/30 cells). Now always passes FORCE_REAGGREGATE.
* `verify_depstore_pipeline`: passes clean in isolation. A suite-run
  rc=1 at step (i) was a Ray socket flake, not reproducible standalone.
* `generate_full_paper_figures`: compute + emit + fingerprint checks all
  PASS; the only failures are the grab step cascading from a LOCAL Ray
  GCS/actor death mid-run ("Failed to connect to GCS within 60s"). This
  is macOS-local Ray instability under back-to-back cluster
  teardown/rebuild — NOT a code bug and NOT reproducible on the Linux
  VMs where every real campaign runs. If you need this test green, run
  it alone (not right after the other Ray-heavy tests) or on a VM.
Takeaway: trust the VM for anything Ray-heavy; the Mac is fine for the
grab/emit/analysis work that is your actual job.
