# EODS — evaluate_over_deployment_sizes, modernized (2026-08-17)

The standalone `evaluate_over_deployment_sizes.py` loop becomes
queue-native cells so the grid machinery (queue, RAM governor, dash
progress/ETA, fleet sharding) runs it unchanged:

- **Cell** = one (dpsize, sim): `run_eods_cell.py` speaks the
  run_fork_ladder CLI + result-JSON convention (seed = sim index,
  n_iters = 1) and checkpoints through `evaluate_all_metrics`'s own
  resumable pickle (`use_performance_metrics_fn` per cell).
- **Manifest**: `build_manifest.py` emits queue specs with the new
  `runner` field (`run_n_sweep_queue` change, 2026-08-17); canonical
  grid = dpsizes [3,5,10,15,20,25,32] x nsim [15,20,10,16,15,15,12]
  = 103 cells. Strategy set via `SCULPTOR_SOLN_TYPES` (env knob added
  to actual_deployment_eval_latency_failure — no more hand-editing
  soln_types).
- **Merge**: `merge_eods.py` mechanically concatenates per-sim pickles
  into the classic `metrics_by_dpsize` cache; the existing paper-plot
  path consumes it unchanged.

Run (single VM):

    python -m experiments.eods.build_manifest --out tools/eods_manifest.json \
        --soln-types sparse,painter,anycast,one_per_pop,one_per_peering
    python -m experiments.ablation.run_n_sweep_queue \
        --manifest tools/eods_manifest.json --ws-root ~/eods_ws \
        --slots <N> --workers-per-run <W> --port0 52000 --no-rescore

Fleet run: shard the manifest with experiments/fleet/shard.py; each VM
runs its shard with the same command.

STATUS: skeleton validated by import/manifest smokes; the heavy
validation (one actual-3 cell end-to-end, then merge-vs-legacy parity
on a small grid) is the P2 burn-in step in
ablation_study/SCALE_500_PLAN.md.
