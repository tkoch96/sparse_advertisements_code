# model_error output formats (for future agents)

All evals write under `cache/model_error/`. CONTAMINATION WARNING: the
plot scripts (plot_policy5.py etc.) select inputs by FILENAME GLOB
(`steady/policy_steady*.json`, `failure/policy_failure*.json`,
`rerank/policy_*`). Never leave outputs from a superseded methodology
era under glob-matching names -- quarantine them (see
`PREFIX_ERA_congestion_bug/`, the pre-congestion-fix outputs) or the
figures silently blend eras (bit us 2026-08-14).

- `steady/<tag>.json` (steady_metrics.py): list of entries
  `{dir, seed, rung, solved, steady_congested_frac, clean_avg_lat,
  routed_frac}`; `clean_avg_lat` excludes sentinel (>=15000ms) volume;
  `dir` encodes arm and N as its last two path components.
- `failure/<tag>.json` (failure_metrics.py): list of entries
  `{dir, seed, rung, solved, popp: {...}, pop: {...}}` where each block
  has `cong_mean/cong_max` (LP re-assignment congestion over single-
  entity failures), `affected_routed_lat_mean` (vol-weighted post-
  failure latency of PRE-failure users of the failed entity, routed
  only), `affected_stranded_frac_mean`, `n_scenarios(_with_affected)`.
- `rerank/<tag>/seed_<s>.json` (rerank_ladder.py): per-seed
  `{rungs: {<rung>: {avg_lat, frac_beyond{10,50,100}, lat_plus_mlu,
  popp_fail{mean,max}}}}`.
- `hardB3_scores.json` (experiments/dashboard/score_hardb3.py): see
  experiments/dashboard/README.md.
- `opp_ref_georand.json`: one-per-peering reference values per seed.

World knobs are NOT recorded inside most outputs -- the tag is the only
provenance. Name tags with era/world (`policy_steady_fixed`,
`hardC_steady`) and record the env in DIMENSIONS.md when adding runs.
