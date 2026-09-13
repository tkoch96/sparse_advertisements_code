# Where every paper artifact's pickles live (storage box)

**Box:** `i-0428c395787bc3ca0` ("ray-sculptor-head", m8g.4xlarge, always on, ~$0.70/h),
`ssh -i ~/.ssh/ray-autoscaler_us-east-1.pem ubuntu@<ip>` (IP changes on stop/start;
`python -m cluster.vmctl list` prints it). Repo copy: `/home/ubuntu/sparse_advertisements_code`.
All paths below are relative to that directory. The paper-of-record intent is
`evaluations/intents/paper_intent.por.json`; `run_all_paper_evaluations.py grab` pulls the
finished figures/tables from `paper_artifacts/<stage>/` into the laptop's `figures/paper_artifacts/`.

Naming you need to know:

- `popp_failure_latency_comparison_<dpsize>[_<run_tag>].pkl` is the L1 metrics pickle of one
  campaign at one deployment size: the trained advertisements of every strategy
  (`m['adv'][sim][strategy]`), the deployments (`m['deployment'][sim]`), and every evaluation
  family (`m['stats_*']`, failure sweeps, flash/diurnal). One file holds ALL deployments (sims)
  of that campaign.
- Size 32 has TWO labels: `actual-32` (the deployment-size sweep) and
  `testing_feature-actual-32` (the paper table, the prefix sweep). They are different
  deployment draws. Everything else uses `testing_feature-actual-N`.
- `*_tree/`, `*_artifacts/` next to an ablation study are the harvested per-cell figures and
  states; the study directory itself holds the per-seed result JSONs.

## (a) Evaluate over deployment sizes (`paper_artifacts/eods/*.pdf`)

Late-August campaign, `evaluations/evaluate_over_deployment_sizes.py`, latency objective
(`avg_latency`, gamma 4, 150 iters), canonical untagged pickles:

| Size | Pickle | Trained deployments | Date |
|---|---|---|---|
| 5 | `cache/popp_failure_latency_comparison_testing_feature-actual-5.pkl` | 20 | Sep 11 (re-evaluated) |
| 10 | `cache/popp_failure_latency_comparison_testing_feature-actual-10.pkl` | 20 | Sep 11 (re-evaluated) |
| 15 | `cache/popp_failure_latency_comparison_testing_feature-actual-15.pkl` | 12 | Sep 11 (re-evaluated) |
| 20 | `cache/popp_failure_latency_comparison_testing_feature-actual-20.pkl` | 5 | Aug 30 |
| 25 | `cache/popp_failure_latency_comparison_testing_feature-actual-25.pkl` | 4 | Aug 30 |
| 32 | `cache/popp_failure_latency_comparison_actual-32.pkl` | 3 | Aug 30 training; re-evaluated Sep 12 (current failure families, flash at 1.1) |

Aggregate used by the plots: `cache/cluster_runs/20260822_220131-prefixbudget3/metrics_by_dpsize.pkl`
(the intent's `cache_fn`); figures in `figures/cluster/20260822_220131-prefixbudget3/` and the
curated set in `paper_artifacts/eods/`. `paper_artifacts/scaling_summary/deployment_scaling_summary.csv`
is derived from the same pickles.

Caveats: `cache/popp_failure_latency_comparison_testing_feature-actual-32.pkl` is a SYMLINK (made
Sep 2) to the paper table's avg_latency pickle, which since Sep 12 is a COPY of the 3-deployment
sweep file above (so the symlink and the sweep file now agree). The size-32 point was re-evaluated on
Sep 12 (run `20260912_120728-eods32-reeval` on the study box, mirrored here; the Aug 30 evaluation is
kept as `*actual-32.pre_reeval_20260912.pkl`, and the aggregate's previous state as
`metrics_by_dpsize.pre_reeval_20260912.pkl`). Sizes 20/25 still carry the Aug 30 evaluation (flash
crowd at 1.3 headroom, no all-users failure families); the Sep 11 rerun
(`20260910_220214-paperv1-deployment_sizes`) cache-hit sizes 5/10/15 without recomputing anything and
died at size 20 with rc 137, so only the size-32 point uses the current definitions.

## (b) Paper table (`paper_artifacts/paper_table/`)

Campaign tag `20260823_130342_papertable32b`, size `testing_feature-actual-32`, 150 iters,
one L1 pickle per objective (the avg_latency one carries no suffix):

| Table group | Pickle | Trained deployments |
|---|---|---|
| Dynamic failover, flash/diurnal (`avg_latency`) | `cache/popp_failure_latency_comparison_testing_feature-actual-32_20260823_130342_papertable32b.pkl` | 3 (since Sep 12: a copy of the sweep's `actual-32` pickle, i.e. the Aug 30 sweep deployments, NOT the campaign's seeded draw; the 1-deployment original is `*.pre_eods3_20260912.pkl`) |
| Latency + MLU (`max_util`) | `..._papertable32b_max_util.pkl` | 3 |
| Static failover (`frozen_prefix`, site-failure objective since 2026-09-11) | `..._papertable32b_frozen_prefix.pkl` | 3 |
| Latency-sensitive (`frac_beyond_optimal`) | `..._papertable32b_frac_beyond_optimal.pkl` | 3 |
| Traffic classes (`joint_priority`) | `..._papertable32b_joint_priority.pkl` | 3 (Painter missing in sim 0) |
| Site cost (`per_site_cost`) | `..._papertable32b_per_site_cost.pkl` | 3 |

Backups kept alongside: `*_frozen_prefix.pre_sitefail_20260911.pkl` (the single-peering
objective's training), `*_frac_beyond_optimal.pkl.pre_honest_fracwithin_backup`,
`*_joint_priority.pkl.pre_painter_fill_backup`. Emitted tables: `paper_artifacts/paper_table/`
(`paper_table{,_full,_key}.{csv,tex}`; the CSVs are the record, tex is re-emitted locally with
`generate_paper_table.py --reemit-from-csv`). The L3 condensed pickle
`cache/paper_table_condensed_testing_feature-actual-32_20260823_130342_papertable32b.pkl` is a
cache only (currently absent: the joint_priority "failed strategy" flag stops it from being saved);
rebuild the table from the L1 pickles in seconds with the six-objective `generate_paper_table.py` call.

Overlap with (a): since Sep 12 the avg_latency pickle here IS the sweep's 3-deployment size-32
evaluation (same trainings, same numbers); the other five objectives were trained on the campaign's
own seeded deployments, so the dynamic-failover group averages different deployments than they do.

## (c) Evaluate over number of prefixes (`paper_artifacts/n_prefixes/*.pdf`)

`evaluations/evaluate_over_n_prefixes.py`, `testing_feature-actual-32`, 1 deployment, 150 iters,
latency objective, one pickle per prefix budget:

- `cache/paperv1/testing_feature-actual-32_over_prefixes-{16,24,32,40,64}.pkl` (Sep 2)
- aggregate: `cache/paperv1/prefix_metrics_a32.pkl`; figures `figures/cluster/prefix_sweep_a32/`
- the actual-3 shim used while the 32 sweep was pending: `cache/paperv1/testing_feature-actual-3_over_prefixes-{2,3,4,5}.pkl`, `cache/paperv1/prefix_metrics.pkl`

The five trainings are also in the depstore (`depstore/trainings/<fp>_it152/`, key
`n_prefixes` = 16..64), which is how a re-evaluation finds them without retraining.

## (d) Ablation ladder, all objectives (`paper_artifacts/ablation_cdf/*.pdf` and the % tables)

Studies are `evaluations/ablation.py` output roots under `cache/ablation/`; each holds
`study.json`, per-deployment `N<budget>/seed_<s>_<rung>.json` (+ `init_dep<s>.npy`), and next to it
`<study>_artifacts/` (per-cell figures + `..._state-<N>.pkl`) and `<study>_tree/deployment_NN/L#_<rung>/`.
Rungs: `full, expl_none, no_memory_dir, no_memory, no_mc, painter`. Re-evaluate any study with
`python -m evaluations.ablation evaluate --in-dir cache/ablation/<study> [--train-objective ...]`.

| Study | What | Where |
|---|---|---|
| Paper CDF figure (Sep 2) | latency objective, actual-10, 20 deployment seeds, 100 iters (`run_ablation_cdf.py`) | deployments `cache/ablation/cdf_a10_deps/dep_seed{1..20}.pkl`, inits `/home/ubuntu/abl_cdf10_ws/inits`, manifest `/home/ubuntu/abl_cdf10_ws/cdf_manifest.json`. The per-cell results were NOT retained (that campaign cleaned its run dirs; only the figures survive). |
| 10-deployment ladder (Sep 9) | latency objective, actual-10, 10 deployments, 100 iters, probe budget 0.5x | `cache/ablation/cdf_a10_night{,_artifacts,_tree}` |
| Per-objective ladders (Sep 10) | one actual-10 deployment, 100 iters, 0.5x, for each of `avg_latency, max_util, frac_beyond_optimal, joint_priority, per_site_cost, frozen_prefix` | `cache/ablation/obj_a10d1_<objective>{,_artifacts}` (+ `_tree` for the four that have one) |
| Probing-policy study (Sep 9) | actual-5, smart vs scheduled vs loose, 150 iters | `cache/ablation/probe_a5_150_{smart,sched,loose}{,_artifacts,_tree}` |
| Small/actual-3 smokes | `cache/ablation/smoke_*` | not paper material |

The Sep 9/10 studies were run on the study box `i-04d7439fa93efaf2a` and copied here on
2026-09-12; the study box is not a store, everything it produced that matters is here.

## Other stores on this box

- `depstore/` (500 MB, `index.jsonl`, `trainings/<fp>_it<N>/`, `evals/`): every training keyed by
  the semantic fingerprint (deployment seed, objective, levers). `SCULPTOR_DEPSTORE_MODE=eval_only`
  makes a cache miss fatal instead of retraining.
- `backups/i-09a6_final/cache/`: the Aug 25-28 copies of the paper-table pickles from the retired
  r8g head.
- Real-deployment (campus VM) figures do not live here; see
  `~/Documents/actual_deployment_figures` (git: tkoch96/actual_deployment_figures).
