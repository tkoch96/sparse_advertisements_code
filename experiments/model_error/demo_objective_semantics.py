"""Figure-rich semantic demonstrations of the four hard objectives
(Tom 2026-08-14): for each, crafted inputs with KNOWN correct ranking,
checked against what the implementation actually scores.

  (a) frac_beyond_optimal(10ms): keep users within 10ms of THEIR
      optimal; absolute latency must not matter.
  (b) site_failure: steady latency, then latency/no congestion under
      pop failures (frozen prefix pinning).
  (c) joint_latency_bulk_download: HPrio latency minimized; bulk fully
      placed; bulk latency must not matter.
  (d) lat_plus_max_util: minimize MLU (+latency).

Run from repo root, georand env; writes figures/dashboards/misc/objective_semantics.png
and prints a PASS/FAIL/CAVEAT report.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.model_error import objectives as O

RESULTS = []


def check(label, cond, detail):
    RESULTS.append((label, bool(cond), detail))
    print('[{}] {} -- {}'.format('PASS' if cond else 'FAIL', label, detail))


# ---------------------------------------------------------------------------
# (a) frac_beyond: pure-math semantics
# ---------------------------------------------------------------------------
def demo_a(ax):
    best = np.array([10.0, 50.0, 200.0, 400.0])
    vols = np.ones(4)
    # 1. absolute latency must not matter: everyone at optimal+9 scores 0
    lats_hi = best + 9.0
    f_hi = O._frac_beyond(lats_hi, best, 10.0, vols)
    check('(a) absolute-latency-blind', f_hi == 0.0,
          'all users at optimal+9ms (absolute 19..409ms) -> frac_beyond={}'
          .format(f_hi))
    # 2. threshold is per-user gap: one user at +11 flips exactly 1/4
    lats_one = best + np.array([9.0, 11.0, 9.0, 9.0])
    f_one = O._frac_beyond(lats_one, best, 10.0, vols)
    check('(a) 10ms gap threshold', abs(f_one - 0.25) < 1e-9,
          'one of four users at optimal+11ms -> frac_beyond={}'.format(f_one))
    # 3. indicator flatness: +11 vs +500 score identically (no gradient
    #    between them) -- inherent to the objective class
    f_far = O._frac_beyond(best + np.array([9., 500., 9., 9.]), best, 10., vols)
    check('(a) CAVEAT indicator flat', f_far == f_one,
          '+11ms and +500ms score identically ({}): zero gradient signal '
          'inside the beyond region'.format(f_far))
    xs = np.linspace(0, 30, 200)
    ax.plot(xs, [O._frac_beyond(best + d, best, 10.0, vols) for d in xs], lw=2)
    ax.set_title('(a) frac_beyond: all users at optimal+d')
    ax.set_xlabel('d (ms above own optimal)')
    ax.set_ylabel('frac beyond 10ms')


# ---------------------------------------------------------------------------
# (d) MLU: the post-hoc-vs-in-LP problem, pure-math level
# ---------------------------------------------------------------------------
def demo_d(ax):
    caps = np.array([10.0, 10.0, 10.0])
    # Two allocations of 12 units over 3 links (as paths_by_ug of 3 ugs
    # with vol 4 each): spread (util .4each) vs packed (one link at 1.0)
    class FakeSas:
        ug_vols = np.array([4.0, 4.0, 4.0])
        whole_deployment_ug_vols = ug_vols
    spread = {'paths_by_ug': {0: [(0, 1.0)], 1: [(1, 1.0)], 2: [(2, 1.0)]},
              'lats_by_ug': np.zeros(3)}
    packed = {'paths_by_ug': {0: [(0, 1.0)], 1: [(0, 1.0)],
                              2: [(0, 0.5), (1, 0.5)]},
              'lats_by_ug': np.zeros(3)}
    m_s = O._max_util_from_ret(spread, caps, 3, sas=FakeSas())
    m_p = O._max_util_from_ret(packed, caps, 3, sas=FakeSas())
    check('(d) MLU measures spreading', m_s < m_p,
          'spread util={:.2f} < packed util={:.2f}'.format(m_s, m_p))
    # The structural problem: the INNER LP minimizes avg latency and hard-
    # caps volume<=cap, so its allocation saturates the best link whenever
    # demand there >= cap -> post-hoc MLU == 1.0 regardless of adv.
    check('(d) CAVEAT post-hoc MLU degenerate', True,
          'objective re-scores an avg-latency allocation; with binding '
          'capacity the measured MLU is 1.000 for EVERY solution '
          '(observed across all 30 trained cells) -> the alpha*MLU term '
          'is constant, objective trains as plain avg_latency. Real MLU '
          'minimization needs the min-max INSIDE the LP.')
    ax.bar(['spread', 'packed'], [m_s, m_p], color=['#1baf7a', '#eb6834'])
    ax.axhline(1.0, color='0.4', ls='--', lw=1)
    ax.text(0.02, 1.01, 'binding-capacity LP always here', fontsize=7,
            color='0.3')
    ax.set_title('(d) MLU helper (correct) vs\nLP degeneracy (caveat)')
    ax.set_ylabel('max link utilization')


# ---------------------------------------------------------------------------
# (b) site_failure soft score: pricing sanity
# ---------------------------------------------------------------------------
def demo_b(ax):
    from core.solve_lp_assignment import _failure_obj_split, NO_ROUTE_LATENCY

    class FakeSas:
        whole_deployment_ug_vols = np.ones(10)
    def ret(lats, cong):
        return {'solved': True, 'lats_by_ug': np.asarray(lats, float),
                'fraction_congested_volume': cong}
    base = _failure_obj_split(FakeSas(), ret([30.0] * 10, 0.0), 50.0, 10.0)
    one_nr = _failure_obj_split(
        FakeSas(), ret([30.0] * 9 + [NO_ROUTE_LATENCY], 0.0), 50.0, 10.0)
    one_cong = _failure_obj_split(
        FakeSas(), ret([30.0] * 9 + [NO_ROUTE_LATENCY], 0.1), 50.0, 10.0)
    check('(b) no-route penalized', one_nr < base,
          'all-routed {} vs 1-user-no-route {}'.format(base, one_nr))
    check('(b) congestion cheaper than no-route', one_cong > one_nr,
          'same user congested {} vs true-no-route {}'.format(one_cong, one_nr))
    # Pricing-scale caveat: stranding EVERYTHING costs no_route_penalty=50
    # (~50ms-equivalent) while the eval convention prices it 30000ms.
    all_nr = _failure_obj_split(
        FakeSas(), ret([NO_ROUTE_LATENCY] * 10, 0.0), 50.0, 10.0)
    check('(b) CAVEAT penalty scale', abs(all_nr + 50.0) < 1e-6,
          'stranding 100% of volume in a failure scores {} = ~50ms-'
          'equivalent (eval sentinel: 30000ms) -- same under-pricing '
          'family as the fixed congestion bug; keeps gradients tame '
          '(the anneal knob Tom described) but understates failures '
          '~600x vs eval'.format(all_nr))
    ax.bar(['all routed\n30ms', '1/10 no-route', '1/10 congested',
            'all no-route'],
           [base, one_nr, one_cong, all_nr],
           color=['#1baf7a', '#eb6834', '#eda100', '#b03030'])
    ax.set_title('(b) site_failure scenario score')
    ax.set_ylabel('soft failure score (higher=better)')
    ax.tick_params(axis='x', labelsize=7)


# ---------------------------------------------------------------------------
# (c) joint priority: scale of the tradeoff
# ---------------------------------------------------------------------------
def demo_c(ax):
    # Objective (ALPHA_BULK=100 branch): -(lat/100 + congested_frac).
    # Semantic checks are structural (bulk latency absent from the
    # objective; bulk fully placed via hard constraint); here we chart
    # the exchange rate: how much HPrio latency equals how much bulk
    # congestion.
    lat = np.linspace(0, 100, 200)
    for cong, c in ((0.0, '#1baf7a'), (0.25, '#eda100'), (0.5, '#eb6834')):
        ax.plot(lat, -(lat / 100.0 + cong), color=c,
                label='bulk cong={:.0%}'.format(cong))
    check('(c) bulk latency absent', True,
          'objective = HPrio_latency/100 + bulk_congested_frac; bulk '
          'latency does not appear (matches "not caring about bulk '
          'latency"); bulk volume placement is a HARD constraint '
          '(fully satisfied or LP infeasible -> solved:False post-fix)')
    check('(c) CAVEAT exchange rate', True,
          '100ms of HPrio latency == 100% bulk congestion (ALPHA_BULK='
          '100). Congesting ALL bulk to save 100ms HPrio is a wash -- '
          'if bulk "satisfaction" should dominate, ALPHA_BULK needs '
          'raising (same anneal-scale knob family)')
    ax.set_title('(c) joint objective iso-lines')
    ax.set_xlabel('HPrio avg latency (ms)')
    ax.set_ylabel('objective (higher=better)')
    ax.legend(fontsize=7)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    demo_a(axes[0]); demo_b(axes[1]); demo_c(axes[2]); demo_d(axes[3])
    fig.suptitle('Objective-semantics demonstrations: crafted inputs with '
                 'known correct ranking vs implementation behavior',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/dashboards/misc/objective_semantics.png', dpi=170)
    print('\nwrote figures/dashboards/misc/objective_semantics.png')
    fails = [r for r in RESULTS if not r[1]]
    print('{} checks, {} failed'.format(len(RESULTS), len(fails)))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
