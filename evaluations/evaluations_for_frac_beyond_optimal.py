"""Evaluation for frac_beyond_optimal -- traffic within X ms of optimal.

The objective maximises the fraction of traffic served within a latency
threshold of the best achievable, so the comparison is that fraction per
solution type, not mean latency: two advertisements can share a mean and
differ sharply in how much traffic sits in the tail.

Threshold is 10ms by default, overridable with SCULPTOR_FRAC_BEYOND_MS. It
reuses `calc_pct_volume_within_latency` from eval_all_solution_types, which is
the same volume-weighted computation the latency suite's
pct-volume-within-latency panel uses -- one implementation, so the numbers are
comparable across suites.
"""
import os

import numpy as np

from evaluations._objective_eval_base import (
    score_all_strategies, bar_comparison, announce)
from evaluations.eval_all_solution_types import calc_pct_volume_within_latency

OBJECTIVES = ('frac_beyond_optimal',)

THRESHOLD_MS = float(os.environ.get('SCULPTOR_FRAC_BEYOND_MS', '10'))


def _frac_within(sas, adv):
    m = calc_pct_volume_within_latency(sas, adv)
    lats = np.asarray(m['latencies'], dtype=float)
    fracs = np.asarray(m['volume_fractions'], dtype=float)
    if not len(lats):
        raise ValueError('calc_pct_volume_within_latency returned no points')
    idx = int(np.argmin(np.abs(lats - THRESHOLD_MS)))
    return fracs[idx]


def run(ctx):
    announce(ctx, 'evaluations_for_frac_beyond_optimal',
             'fraction of traffic within {:.0f}ms of optimal'.format(THRESHOLD_MS))
    score_all_strategies(ctx, _frac_within, 'frac_within_threshold_by_strategy')
    bar_comparison(ctx, 'frac_within_threshold_by_strategy',
                   ylabel='fraction of volume within {:.0f}ms'.format(THRESHOLD_MS),
                   title='Traffic within {:.0f}ms of optimal'.format(THRESHOLD_MS),
                   out_name='frac_within_{:.0f}ms_{}.pdf'.format(
                       THRESHOLD_MS, ctx.dpsize),
                   lower_is_better=False)
    return ctx.metrics
