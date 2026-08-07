"""Objective estimators used by the ablation arms.

  * painter arm uses PreferenceBelief.expected_prefix_latency directly
    (deterministic, capacity-ignoring) -- see arms.run_painter.
  * mc_estimate: monte-carlo over sampled preference outcomes, each
    realization scored through the capacity-aware assignment (SCULPTOR's
    latency_benefit model, MC_NUM=5 in the repo).
  * outcome_distribution + entropy_of_distribution: sampled distribution
    of objectives for one advertisement -- used by the entropic-
    exploration arm to pick maximally-informative measurements
    (solve_max_information analogue with the 'entropy' methodology).
"""
import numpy as np


def mc_estimate(problem, belief, adv_bool, seed, n_mc=5, with_capacity=True):
    """Mean objective (avg latency, ms) over n_mc sampled preference
    realizations. `seed` fixes the draws -- callers use common random
    numbers so before/after flip comparisons see the same realizations."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_mc):
        avail = belief.sample_avail(adv_bool, rng)
        vals.append(problem.score_avail(avail, with_capacity=with_capacity))
    return float(np.mean(vals))


def outcome_distribution(problem, belief, adv_bool, seed, n_samples=15):
    """Sampled objective values (capacity-free, cheap) for entropy scoring."""
    rng = np.random.default_rng(seed)
    return np.array([
        problem.score_avail(belief.sample_avail(adv_bool, rng), with_capacity=False)
        for _ in range(n_samples)
    ])


def entropy_of_distribution(vals, bin_ms=1.0):
    """Shannon entropy of the sampled objective distribution, binned at
    bin_ms granularity (the repo bins LB into a fixed grid similarly)."""
    binned = np.round(np.asarray(vals) / bin_ms).astype(np.int64)
    _, counts = np.unique(binned, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())
