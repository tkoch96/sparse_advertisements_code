# actual_deployment_figures

In-repo (`experiments/actual_deployment_figures/`, Tom 2026-09-15; it started life as the
standalone repo tkoch96/actual_deployment_figures, now frozen with a pointer here) regeneration of the real-deployment paper figures
(`figures/paper/*_actual_deployment.pdf` + `uncertainty_path_measures_over_iterations.pdf`).

Source of truth: the OLD flat-layout repo on the campus VM (ubuntu@77.112.21.176,
`~/sparse_advertisements_code`). Everything here was PULLED read-only (scp/rsync);
nothing was executed there.

Pulled (minimum set):
- `vm_src/` -- make_actual_deployment_plots.py, constants.py, helpers.py,
  paper_plotting_functions.py, wrapper_eval.py, eval_latency_failure.py (code only)
- `cache/popp_failure_latency_comparison_actual_third_prototype.pkl` (2.2 MB) --
  the paper's real deployment metrics (the script's documented --dpsize)
- `runs/1714909593-actual-32-sparse/small-stats-*.pkl` (152 files, 12 MB) --
  for the uncertainty-over-iterations figure

Local modifications (see `make_actual_deployment_plots.py` vs `vm_src/`):
- dropped the unused `from eval_latency_failure import evaluate_all_metrics`
  (would pull in the whole flat solver stack)
- `vm_src/helpers.py`: `cymruwhois` import guarded (only lookup_asn needs it)
- failure-latency CDFs (link/site): x-axis label on two lines, label and tick labels at 13pt (one point under the 14pt default), so it fits the 3.5in figure (Tom 2026-09-15)

Run (from anywhere): `experiments/actual_deployment_figures/run_plots.sh [dpsize]` (default actual_third_prototype). Output in `figures/paper/`.
Other deployments' pickles (actual_second_prototype, actual-32, ...) are still on the VM.

The `cache/`, `runs/` and `figures/` subdirs are re-included in the repo's `.gitignore` on purpose:
the pickles, checkpoints and emitted PDFs ARE the reproducibility record.
`evaluations/actual_deployment_numbers.py` reads the pickle from here for the paper's `internet.*` numbers.
