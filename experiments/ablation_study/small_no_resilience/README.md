# Ablation ladder: small dpsize, resilience OFF (pure latency)

30 seeds x 200 iters, 2026-08-06/07. Fork of the real SCULPTOR solver
(experiments/ablation/sculptor_fork.py) with feature flags; objective =
capacity-aware avg latency only (using_resilience_benefit=False, gamma=0,
lambduh=1e-5). Scored via repo ground-truth LP (fresh-process rescoring;
'rescored': true in every JSON — in-process driver scores are in
'driver_diff_vs_opp' and are known-contaminated).

Ladder medians (final avg latency - one_per_peering, ms):
  painter 5.10 -> monte-carlo 4.35 -> +memory 3.04 -> +direction 1.02
  -> +random-explore 1.00 -> +entropy(=SCULPTOR) 1.01

results_json/ contains per-(seed,rung) JSONs incl. final advertisement
matrices. Figure: fork_ladder_final.png/pdf.
