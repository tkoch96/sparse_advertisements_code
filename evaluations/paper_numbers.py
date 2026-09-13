"""Named numbers for the paper (Tom 2026-09-12).

The paper's prose lives in a Google Doc; pandoc passes raw LaTeX through, so
a sentence in the doc reads

    \\sparse is \\pn{dtf.popfail.all.lat.mean.sculptor.worsethan.opp} ms worse
    than \\opp under site failure.

and this module writes tables/paper_numbers.tex in the paper repo, defining
every referenced key from the paper-of-record artifacts
(figures/paper_artifacts/*.csv). An undefined key renders red as
"TBD:<key>" -- the outstanding placeholders are whatever is still red.

KEYS ARE COMPOSED, NOT NAMED. A key is a fixed-order path of vocabulary
tokens; every token carries its own one-line doc, and a key's meaning is the
composition of its tokens' docs (printed as a comment above each definition
and by `explain KEY`). Nobody names a number by hand except the few
`headline.*` entries, each of which MUST carry a prose comment.

Families
  <obj>.<scenario>.<population>.<metric>.<stat>.<method>[.<verb>[.<arg>..]][.dN]
      the paper table (paper_table_full.csv)
  sweep.<axis>.<objective-slug>.<metric-slug>.<field>[.dN]
      deployment_scaling_summary.csv / prefix_scaling_summary.csv
  abl.<obj>.<rung>.<field>[.dN]
      ablation ladder summaries (ladder_summary.json per objective)
  internet.<steady|linkfail|sitefail>.<metric>.<method>[.<verb>..][.dN]
      the on-Internet (RIPE Atlas) deployment: actual_deployment_stats.csv
      from evaluations/actual_deployment_numbers.py (metrics gap, overloaded,
      within10/50/100; methods anycast unicast painter sculptor + roles)
  headline.<slug>[.dN]
      hand-named claims: a formula over other keys + a REQUIRED doc
  manual.<slug>[.dN]
      numbers with no automated source yet (on-Internet deployment, costs,
      dates): evaluations/paper_numbers_manual.json, value + doc + where-from

Comparison verbs (direction-aware, from the CSV's DIRECTION row):
  minus.M      self - M, metric units (percentage points for % metrics)
  worsethan.M  raw difference oriented so POSITIVE = self is worse than M
  betterthan.M -worsethan.M (positive = self is better than M)
  pctbetter.M  improvement of self over M as a percent OF M (M is always
               the denominator); undefined when |M| is below the metric's
               floor (no "84% less congestion" from 2.22 vs 14; no divide
               by 0.00) -- use minus (points) instead
  pctworse.M   -pctbetter.M
  over.M       plain ratio self / M
  pctway.LO.HI 100 * (self - LO) / (HI - LO): fraction of the way from LO's
               value to HI's value (direction-agnostic)
  name         the TeX macro of the method (for roles: who it resolved to)
M is a literal method or a ROLE (best / bestpractical / bestbaseline /
worst) resolved per cell. Stat is applied BEFORE the comparison (compare
the means), so every number is checkable from the table.

CLI
  python -m evaluations.paper_numbers emit    [--paper-dir P] [--tex T]
  python -m evaluations.paper_numbers check   [--paper-dir P] [--tex T]
  python -m evaluations.paper_numbers ls      [--grep PAT] [--values]
  python -m evaluations.paper_numbers explain KEY [KEY ...]
  python -m evaluations.paper_numbers vocab
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_ARTIFACTS = os.path.join(REPO, 'figures', 'paper_artifacts')
DEFAULT_PAPER_DIR = os.environ.get('SCULPTOR_PAPER_DIR',
                                   os.path.expanduser('~/Documents/resilient_advertisements_paper'))
MANUAL_JSON = os.path.join(HERE, 'paper_numbers_manual.json')
PN_RE = re.compile(r'\\pn(?:\[[^\]]*\])?\{([^}]*)\}')


# =============================================================================
# Vocabulary. One entry per token; the doc is the ONLY documentation of the
# token and is composed into every key that uses it.
# =============================================================================

@dataclass(frozen=True)
class Tok:
    doc: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, k):
        try:
            return self.extra[k]
        except KeyError:
            raise AttributeError(k)


OBJECTIVES = {
    # token -> registry objective name, stored table group, TeX group macro
    'latmlu': Tok('Latency + Maximum Link Utilization objective (\\latmlu): min-latency assignment '
                  'plus the MLU term; the group\'s latency column is the min-latency assignment value.',
                  dict(objective='max_util', group='MLU', tex='\\latmlu')),
    'dtf': Tok('Dynamic Traffic Failover objective (\\dtf): steady latency + gamma * resilience '
               'benefit; after a failure the LP re-optimizes traffic over the surviving paths.',
               dict(objective='avg_latency', group='Latency + g*Resilience', tex='\\dtf')),
    'stf': Tok('Static Traffic Failover objective (\\stf): ONE user->prefix allocation shared by '
               'the normal scenario and every failure; BGP fallback within the pinned prefix is the '
               'only post-failure adaptation (no DNS/allocation change).',
               dict(objective='frozen_prefix', group='Frozen failover', tex='\\stf')),
    'lss': Tok('Latency Sensitive Services objective (\\lss): maximize traffic within 10 ms of its '
               'one-per-peering (optimal) latency.',
               dict(objective='frac_beyond_optimal', group='Frac beyond optimal', tex='\\lss')),
    'prios': Tok('Traffic Classes objective (\\prios): latency-sensitive (HPrio) traffic solved '
                 'first, bulk (LPrio) traffic fills the remaining capacity.',
                 dict(objective='joint_priority', group='High + Low Priority Traffic', tex='\\prios')),
    'sitecost': Tok('Traffic Cost Across Sites objective (\\sitecost): traffic-weighted latency + '
                    'alpha * per-site cost.',
                    dict(objective='per_site_cost', group='Site cost', tex='\\sitecost')),
}

SCENARIOS = {
    'normal': Tok('normal operation (no failure, steady-state traffic)'),
    'poppfail': Tok('single peering-link (ingress) failure, averaged over the failure scenarios'),
    'popfail': Tok('whole-site (PoP) failure, averaged over the failure scenarios'),
    'flash': Tok('flash crowd: one metro\'s volume multiplied until a link congests'),
    'diurnal': Tok('diurnal traffic swing: every metro follows the measured daily profile'),
}

POPULATIONS = {
    'all': Tok('averaged over ALL users (unaffected users included; congested volume excluded '
               'from latency averages)'),
    'affected': Tok('averaged over only the users whose traffic was on the failed link/site'),
}

# metric token -> doc, units, default decimals, floor for ratio verbs
METRICS = {
    'lat': Tok('average latency', dict(units='ms', prec=1, floor=1.0)),
    'subopt': Tok('average latency ABOVE the one-per-peering optimum (0 = optimal)',
                  dict(units='ms', prec=1, floor=1.0)),
    'cong': Tok('percent of traffic volume on over-capacity links', dict(units='%', prec=2, floor=1.0)),
    'noroute': Tok('percent of traffic volume with NO surviving route (stranded)',
                   dict(units='%', prec=2, floor=1.0)),
    'congvol': Tok('congested volume fraction of the LP solution (0..1)', dict(units='', prec=3, floor=0.01)),
    'strandvol': Tok('stranded volume fraction of the LP solution (0..1)', dict(units='', prec=3, floor=0.01)),
    'obj': Tok('the objective\'s own scalar (soft-bounded LP objective as trained)',
               dict(units='', prec=1, floor=1e-9)),
    'mlu': Tok('maximum link utilization (capacity = 1.1 x anycast load per link, so anycast is 1/1.1)',
               dict(units='', prec=3, floor=0.01)),
    'mluratio': Tok('maximum link utilization RELATIVE to anycast\'s provisioned 1/1.1 -- the table\'s '
                    '"(vs anycast)" column', dict(units='x', prec=3, floor=0.01)),
    'intensity': Tok('traffic multiplier the deployment absorbs before any link congests',
                     dict(units='x', prec=2, floor=0.1)),
    'intensityratio': Tok('absorbed traffic multiplier RELATIVE to anycast\'s -- the table\'s '
                          '"Intensity (vs anycast)" column', dict(units='x', prec=2, floor=0.1)),
    'within10': Tok('percent of traffic within 10 ms of its one-per-peering latency',
                    dict(units='%', prec=1, floor=1.0)),
    'beyond10': Tok('percent of traffic MORE than 10 ms from its one-per-peering latency '
                    '(= 100 - within10; the table shows this)', dict(units='%', prec=1, floor=1.0)),
    'hpriolat': Tok('average latency of the latency-sensitive (HPrio) class', dict(units='ms', prec=1, floor=1.0)),
    'hpriofrac': Tok('fraction of HPrio volume routed (0..1)', dict(units='', prec=3, floor=0.01)),
    'critbulk': Tok('critical bulk ratio: multiple of the bulk volume routable before HPrio congests',
                    dict(units='x', prec=2, floor=0.1)),
    'hpriocongswan': Tok('HPrio congestion at the SWAN-style fill level', dict(units='%', prec=2, floor=1.0)),
    'cost': Tok('traffic-weighted average site cost (carbon cost model)', dict(units='', prec=4, floor=0.01)),
    'costratio': Tok('traffic-weighted average site cost RELATIVE to anycast -- the table\'s '
                     '"(vs anycast)" column', dict(units='x', prec=3, floor=0.01)),
    'maxcost': Tok('traffic-weighted MAXIMUM site cost', dict(units='', prec=4, floor=0.01)),
}

STATS = {
    'mean': Tok('mean over the evaluated deployments (the table value)'),
    'std': Tok('standard deviation over the evaluated deployments (needs paper_table_full_stats.csv)'),
    'n': Tok('number of evaluated deployments (needs paper_table_full_stats.csv)'),
}

# method token -> CSV row name, TeX macro, doc
METHODS = {
    'opp': Tok('One-per-Peering: one prefix per peering, the routing OPTIMUM and an unrealistic '
               'upper bound (779 prefixes at 32 sites vs SCULPTOR\'s 42)',
               dict(csv='One-per-peering', tex='\\opp', sweep='one_per_peering')),
    'sculptor': Tok('SCULPTOR', dict(csv='SCULPTOR', tex='\\sparse', sweep='sparse')),
    'painter': Tok('PAINTER (anycast + greedy per-prefix improvement, objective-aware)',
                   dict(csv='PAINTER', tex='\\painter', sweep='painter')),
    'unicast': Tok('Unicast: one prefix per site', dict(csv='Unicast', tex='\\ucast', sweep='one_per_pop')),
    'anyopt': Tok('AnyOpt', dict(csv='AnyOpt', tex='\\anyopt', sweep='anyopt')),
    'anycast': Tok('Anycast: a single prefix everywhere', dict(csv='Anycast', tex='\\acast', sweep='anycast')),
}
PRACTICAL = ('sculptor', 'painter', 'unicast', 'anyopt', 'anycast')
BASELINES = ('painter', 'unicast', 'anyopt', 'anycast')

ROLES = {
    'best': Tok('the best-performing method INCLUDING one-per-peering, per the metric\'s direction',
                dict(pool=('opp',) + PRACTICAL, pick='best')),
    'bestpractical': Tok('the best-performing deployable method (one-per-peering excluded; SCULPTOR included)',
                         dict(pool=PRACTICAL, pick='best')),
    'bestbaseline': Tok('the best-performing PRIOR approach (one-per-peering and SCULPTOR excluded): '
                        '"the next-best solution"', dict(pool=BASELINES, pick='best')),
    'nextbest': Tok('alias of bestbaseline', dict(pool=BASELINES, pick='best')),
    'worst': Tok('the worst-performing deployable method', dict(pool=PRACTICAL, pick='worst')),
}

VERBS = {
    'minus': Tok('self minus M, in the metric\'s units (percentage POINTS for % metrics)', dict(nargs=1)),
    'worsethan': Tok('raw difference oriented so positive = self is WORSE than M (direction-aware)', dict(nargs=1)),
    'betterthan': Tok('raw difference oriented so positive = self is BETTER than M (direction-aware)', dict(nargs=1)),
    'pctbetter': Tok('percent improvement of self over M, as a percent OF M (direction-aware; '
                     'undefined below the metric floor)', dict(nargs=1, prec=0)),
    'pctworse': Tok('percent degradation of self vs M, as a percent OF M (= -pctbetter)', dict(nargs=1, prec=0)),
    'over': Tok('plain ratio self / M', dict(nargs=1, prec=1)),
    'pctway': Tok('100 * (self - LO) / (HI - LO): percent of the way from LO\'s value to HI\'s',
                  dict(nargs=2, prec=0)),
    'name': Tok('the TeX macro naming the method (for roles: whoever the role resolved to)', dict(nargs=0)),
}

SWEEP_AXES = {
    'dpsize': Tok('sweep over deployment size (number of sites), prefix budget fixed',
                  dict(csv='deployment_scaling_summary.csv', axis='deployment_size')),
    'nprefix': Tok('sweep over prefix budget on the size-32 deployment',
                   dict(csv='prefix_scaling_summary.csv', axis='prefix_budget')),
}
SWEEP_FIELDS = {
    'sculptor-at-max': Tok('SCULPTOR\'s value at the largest axis point', dict(col='sculptor_at_max', prec=2)),
    'best-baseline-at-max': Tok('the best baseline\'s value at the largest axis point',
                                dict(col='best_baseline_at_max', prec=2)),
    'best-baseline-name': Tok('which baseline is best at the largest axis point (TeX macro)',
                              dict(col='best_baseline_name', prec=None)),
    'gap-at-min': Tok('SCULPTOR\'s lead over the best baseline at the smallest axis point',
                      dict(col='gap_at_min', prec=2)),
    'gap-at-max': Tok('SCULPTOR\'s lead over the best baseline at the largest axis point',
                      dict(col='gap_at_max', prec=2)),
    'gap-change-pct': Tok('percent change of SCULPTOR\'s lead from the smallest to the largest axis point '
                          '(positive = the lead grows with scale)', dict(col='gap_change_pct', prec=0)),
    'spearman-rho': Tok('Spearman rank correlation of the lead with the axis', dict(col='spearman_rho', prec=2)),
    'sculptor-leads': Tok('axis points where SCULPTOR beats every baseline, as "k/n"',
                          dict(col='sculptor_leads', prec=None)),
    'verdict': Tok('grow / hold / shrink: the summary script\'s reading of the lead trend',
                   dict(col='verdict', prec=None)),
}

ABL_RUNGS = {
    'painter': Tok('PAINTER, the 0% anchor of the ablation ladder'),
    'no_mc': Tok('L2: gradient descent WITHOUT Monte Carlo (point estimate of routing)'),
    'no_memory': Tok('L3: Monte Carlo, no measurement memory'),
    'no_memory_dir': Tok('L4: Monte Carlo + directed measurements, no memory'),
    'expl_none': Tok('L5: full pipeline with scheduled (non-smart) measurements'),
    'full': Tok('L6: SCULPTOR (smart measurements)'),
    'OPP': Tok('one-per-peering, the 100% anchor'),
}
ABL_FIELDS = {
    'pctbenefit': Tok('percent of the PAINTER-to-OPP objective gap closed (on means over deployments)',
                      dict(col='pct_gap_closed_on_means', prec=0)),
    'increment': Tok('percent of the gap closed by THIS rung beyond the previous rung',
                     dict(col='increment_pct', prec=0)),
    'meanobj': Tok('mean training objective over deployments', dict(col='mean_objective', prec=1)),
    'minusopp': Tok('mean objective minus OPP\'s', dict(col='mean_minus_opp', prec=1)),
}


INTERNET_SCENARIOS = {
    'steady': Tok('on-Internet deployment (10 PEERING/Vultr sites, 12 prefixes, 493 RIPE Atlas user groups), steady state'),
    'linkfail': Tok('on-Internet deployment, single peering-link failure: the users on the failed link, averaged over failed links'),
    'sitefail': Tok('on-Internet deployment, single site failure: the users at the failed site, averaged over failed sites'),
}
INTERNET_METRICS = {
    'gap': Tok('mean latency above the one-per-peering (optimal) latency over NON-overloaded traffic, traffic weighted. '
               'CAUTION: not comparable across methods with different overloaded shares (anycast/PAINTER overload '
               '70-95% under failure); compare failure gaps to unicast, the only prior approach that does not overload',
               dict(units='ms', prec=1, floor=1.0, direction='<')),
    'gapall': Tok('mean latency above optimal over ALL traffic with overloaded users priced at the 450 ms congestion cap '
                  '(the plot script\'s printed average; 300+ ms for anycast/PAINTER under failure)',
                  dict(units='ms', prec=1, floor=1.0, direction='<')),
    'overloaded': Tok('percent of traffic whose achieved latency hit the 450 ms congestion cap (landed on an over-capacity link)',
                      dict(units='%', prec=1, floor=1.0, direction='<')),
    'within10': Tok('percent of traffic within 10 ms of its one-per-peering latency (overloaded traffic counts as not within)',
                    dict(units='%', prec=1, floor=1.0, direction='>')),
    'within50': Tok('percent of traffic within 50 ms of its one-per-peering latency', dict(units='%', prec=1, floor=1.0, direction='>')),
    'within100': Tok('percent of traffic within 100 ms of its one-per-peering latency', dict(units='%', prec=1, floor=1.0, direction='>')),
}
INTERNET_METHODS = ('anycast', 'unicast', 'painter', 'sculptor', 'anyopt')   # anyopt absent from the deployment (empty rows skipped)


# =============================================================================
# Cells: which (objective, scenario, population, metric) exist, and which
# stored column (or derivation) each maps to. THIS is the validity rule.
# =============================================================================

@dataclass(frozen=True)
class Cell:
    column: Optional[str] = None            # 'Group|Sub' in paper_table_full.csv
    derive: Optional[Tuple] = None          # ('ratio_const', base_cell, const) |
                                            # ('ratio_method', base_cell, method) |
                                            # ('complement', base_cell, total)


def _lat_split(obj):
    g = OBJECTIVES[obj].group
    return {
        (obj, 'normal', 'all', 'congvol'): Cell(g + '|Congested vol'),
        (obj, 'normal', 'all', 'strandvol'): Cell(g + '|Stranded vol'),
    }


CELLS: Dict[Tuple[str, str, str, str], Cell] = {}
G = 'MLU'
CELLS.update({
    ('latmlu', 'normal', 'all', 'lat'): Cell(G + '|Latency (ms)'),
    ('latmlu', 'normal', 'all', 'mlu'): Cell(G + '|MLU'),
    ('latmlu', 'normal', 'all', 'mluratio'): Cell(derive=('ratio_const', ('latmlu', 'normal', 'all', 'mlu'), 1.0 / 1.1)),
    ('latmlu', 'normal', 'all', 'obj'): Cell(G + '|Objective'),
})
CELLS.update(_lat_split('latmlu'))
G = 'Latency + g*Resilience'
CELLS.update({
    ('dtf', 'normal', 'all', 'lat'): Cell(G + '|Latency (ms)'),
    ('dtf', 'normal', 'all', 'subopt'): Cell(G + '|Subopt normal (ms)'),
    ('dtf', 'poppfail', 'affected', 'subopt'): Cell(G + '|Subopt PoPP-fail (ms)'),
    ('dtf', 'poppfail', 'all', 'lat'): Cell(G + '|Latency PoPP-fail (ms)'),
    ('dtf', 'poppfail', 'all', 'cong'): Cell(G + '|% cong PoPP-fail'),
    ('dtf', 'poppfail', 'all', 'noroute'): Cell(G + '|% no-route PoPP-fail'),
    ('dtf', 'poppfail', 'affected', 'lat'): Cell(G + '|Affected latency PoPP-fail (ms)'),
    ('dtf', 'popfail', 'affected', 'subopt'): Cell(G + '|Subopt PoP-fail (ms)'),
    ('dtf', 'popfail', 'all', 'lat'): Cell(G + '|Latency PoP-fail (ms)'),
    ('dtf', 'popfail', 'all', 'cong'): Cell(G + '|% cong PoP-fail'),
    ('dtf', 'popfail', 'all', 'noroute'): Cell(G + '|% no-route PoP-fail'),
    ('dtf', 'popfail', 'affected', 'lat'): Cell(G + '|Affected latency PoP-fail (ms)'),
    ('dtf', 'flash', 'all', 'intensity'): Cell(G + '|Flash-crowd resilience'),
    ('dtf', 'flash', 'all', 'intensityratio'): Cell(derive=('ratio_method', ('dtf', 'flash', 'all', 'intensity'), 'anycast')),
    ('dtf', 'diurnal', 'all', 'intensity'): Cell(G + '|Diurnal resilience'),
    ('dtf', 'diurnal', 'all', 'intensityratio'): Cell(derive=('ratio_method', ('dtf', 'diurnal', 'all', 'intensity'), 'anycast')),
    ('dtf', 'normal', 'all', 'obj'): Cell(G + '|Objective (lat+g*RB)'),
})
CELLS.update(_lat_split('dtf'))
G = 'Frozen failover'
CELLS.update({
    ('stf', 'normal', 'all', 'lat'): Cell(G + '|Steady latency (ms)'),
    ('stf', 'poppfail', 'all', 'lat'): Cell(G + '|Latency (ms)'),
    ('stf', 'poppfail', 'all', 'cong'): Cell(G + '|% cong fail'),
    ('stf', 'poppfail', 'all', 'noroute'): Cell(G + '|% no-route fail'),
    ('stf', 'poppfail', 'affected', 'lat'): Cell(G + '|Affected latency (ms)'),
    ('stf', 'popfail', 'all', 'lat'): Cell(G + '|Site latency (ms)'),
    ('stf', 'popfail', 'all', 'cong'): Cell(G + '|% cong site-fail'),
    ('stf', 'popfail', 'all', 'noroute'): Cell(G + '|% no-route site-fail'),
    ('stf', 'popfail', 'affected', 'lat'): Cell(G + '|Site affected latency (ms)'),
    ('stf', 'normal', 'all', 'obj'): Cell(G + '|Objective'),
})
CELLS.update(_lat_split('stf'))
G = 'Frac beyond optimal'
CELLS.update({
    ('lss', 'normal', 'all', 'within10'): Cell(G + '|% within 10ms'),
    ('lss', 'normal', 'all', 'beyond10'): Cell(derive=('complement', ('lss', 'normal', 'all', 'within10'), 100.0)),
    ('lss', 'normal', 'all', 'obj'): Cell(G + '|Objective'),
})
CELLS.update(_lat_split('lss'))
G = 'High + Low Priority Traffic'
CELLS.update({
    ('prios', 'normal', 'all', 'hpriofrac'): Cell(G + '|Frac HPrio routed'),
    ('prios', 'normal', 'all', 'hpriolat'): Cell(G + '|HPrio latency (ms)'),
    ('prios', 'normal', 'all', 'critbulk'): Cell(G + '|Crit bulk ratio'),
    ('prios', 'normal', 'all', 'hpriocongswan'): Cell(G + '|HPrio cong @SWAN'),
    ('prios', 'normal', 'all', 'obj'): Cell(G + '|Objective'),
})
CELLS.update(_lat_split('prios'))
G = 'Site cost'
CELLS.update({
    ('sitecost', 'normal', 'all', 'maxcost'): Cell(G + '|Wgt max site cost'),
    ('sitecost', 'normal', 'all', 'cost'): Cell(G + '|Wgt avg site cost'),
    ('sitecost', 'normal', 'all', 'costratio'): Cell(derive=('ratio_method', ('sitecost', 'normal', 'all', 'cost'), 'anycast')),
    ('sitecost', 'normal', 'all', 'obj'): Cell(G + '|Objective'),
})
CELLS.update(_lat_split('sitecost'))

# direction of derived cells (stored cells read the CSV's DIRECTION row)
DERIVED_DIRECTION = {'mluratio': '<', 'intensityratio': '>', 'costratio': '<', 'beyond10': '<'}


# =============================================================================
# Headline keys: the ONLY hand-named numbers. doc is mandatory.
# =============================================================================

@dataclass(frozen=True)
class Headline:
    doc: str                       # the claim this number backs, in prose
    formula: str                   # a grammar key, or 'py:<expr>' over R(key)
    prec: int = 0


HEADLINES: Dict[str, Headline] = {
    # ---- prefix cost point -------------------------------------------------
    'sculptor_prefixes': Headline(
        'Prefix count SCULPTOR used on the size-32 deployment (the "42 vs 779" cost point).',
        'manual.sculptor_prefixes_a32', prec=0),
    'opp_prefixes': Headline(
        'Prefix count one-per-peering needs on the size-32 deployment.',
        'manual.opp_prefixes_a32', prec=0),
    'prefix_savings_x': Headline(
        'How many times fewer prefixes SCULPTOR uses than one-per-peering at size 32.',
        'py:R("manual.opp_prefixes_a32") / R("manual.sculptor_prefixes_a32")', prec=0),
    # ---- abstract ----------------------------------------------------------
    'abstract_flash_surge_vs_anycast_pct': Headline(
        'Abstract: "absorbs flash crowds X% larger than anycast on the same infrastructure" '
        '(flash-crowd intensity, size-32 emulation, capacity = 1.1x anycast load).',
        'dtf.flash.all.intensity.mean.sculptor.pctbetter.anycast', prec=0),
    'abstract_diurnal_surge_vs_anycast_pct': Headline(
        'Abstract: "absorbs diurnal swings X% larger than anycast" (diurnal intensity, size-32 emulation).',
        'dtf.diurnal.all.intensity.mean.sculptor.pctbetter.anycast', prec=0),
    'abstract_sitefail_overload_reduction_pct': Headline(
        'Abstract: "reduces overloading during site failures by up to X% vs the best prior approach" '
        '(STATIC failover, all users, next-best = whoever has the lowest congestion among prior approaches).',
        'stf.popfail.all.cong.mean.sculptor.pctbetter.bestbaseline', prec=0),
    'abstract_hprio_overload_reduction_pct': Headline(
        'Abstract: "routes high-priority traffic with X% less overloading" (traffic classes, LPrio/HPrio = 5, from the figure).',
        'manual.prios_cong_reduction_at_5x_pct', prec=0),
    # ---- intro, on-Internet ------------------------------------------------
    'intro_internet_steady_gain_ms': Headline(
        'Intro: on-Internet steady-state latency improvement over the best prior approach (its gap to optimal minus ours).',
        'internet.steady.gap.sculptor.betterthan.bestbaseline', prec=1),
    'intro_internet_linkfail_gain_ms': Headline(
        'Intro: on-Internet link-failure latency improvement over Unicast, the only prior approach that does not '
        'overload traffic under failure (PAINTER 76% / anycast 69% of affected traffic overloaded).',
        'internet.linkfail.gap.sculptor.betterthan.unicast', prec=1),
    'intro_internet_sitefail_gain_ms': Headline(
        'Intro: on-Internet site-failure latency improvement over Unicast, the only prior approach that does not '
        'overload traffic under failure (PAINTER 95% / anycast 74% overloaded).',
        'internet.sitefail.gap.sculptor.betterthan.unicast', prec=0),
    # ---- intro, emulated ---------------------------------------------------
    'intro_steady_lat_better_than_nextbest_pct': Headline(
        'Intro: emulated steady-state latency, percent lower than the next-best prior approach (latency+MLU cell).',
        'latmlu.normal.all.lat.mean.sculptor.pctbetter.bestbaseline', prec=0),
    'intro_steady_lat_gap_to_opp_ms': Headline(
        'Intro: emulated steady-state latency SCULPTOR gives up vs the optimum (latency+MLU cell).',
        'latmlu.normal.all.lat.mean.sculptor.worsethan.opp', prec=1),
    'intro_bulk_traffic_vs_nextbest_x': Headline(
        'Intro: traffic classes, multiple of the next-best approach\'s critical bulk ratio ("routes Xx more bulk traffic").',
        'prios.normal.all.critbulk.mean.sculptor.over.bestbaseline', prec=1),
    'intro_lss_more_traffic_within10_points': Headline(
        'Intro: latency-sensitive services, percentage POINTS more traffic within 10 ms of optimal than PAINTER.',
        'lss.normal.all.within10.mean.sculptor.minus.painter', prec=1),
    'intro_lss_less_traffic_beyond10_pct': Headline(
        'Intro: latency-sensitive services, percent less traffic BEYOND 10 ms of optimal than PAINTER '
        '(the relative form of the same fact).',
        'lss.normal.all.beyond10.mean.sculptor.pctbetter.painter', prec=0),
    'intro_sitecost_saving_vs_anycast_pct': Headline(
        'Intro: site cost, percent lower traffic-weighted site cost than anycast (vs the next-best it is ~1%).',
        'sitecost.normal.all.cost.mean.sculptor.pctbetter.anycast', prec=0),
    'intro_sitecost_pctway_anycast_to_opp': Headline(
        'Intro/site-cost section: how far from anycast toward the optimum SCULPTOR gets on site cost.',
        'sitecost.normal.all.cost.mean.sculptor.pctway.anycast.opp', prec=0),
    # ---- failover sections -------------------------------------------------
    'stf_ingress_noroute_pct': Headline(
        'Static failover: SCULPTOR\'s stranded traffic under ingress failure with a frozen allocation.',
        'stf.poppfail.all.noroute.mean.sculptor', prec=2),
    'lss_pctway_anycast_to_opp': Headline(
        'Latency-sensitive services: how far from anycast toward the optimum SCULPTOR gets on % within 10 ms.',
        'lss.normal.all.within10.mean.sculptor.pctway.anycast.opp', prec=0),
}


# =============================================================================
# Sources
# =============================================================================

def _git_sha(path):
    try:
        return subprocess.check_output(['git', '-C', path, 'rev-parse', '--short', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return '?'


class Source:
    """Lazy readers over the artifacts directory."""

    def __init__(self, artifacts=DEFAULT_ARTIFACTS, ablation_dir=None, manual_json=MANUAL_JSON,
                 run_tag=None):
        self.artifacts = artifacts
        self.ablation_dir = ablation_dir or os.path.join(artifacts, 'ablation')
        self.manual_json = manual_json
        self._table = None
        self._stats = None
        self._sweeps = {}
        self._abl = {}
        self._manual = None
        self.run_tag = run_tag or self._intent_run_tag()
        self.sha = _git_sha(REPO)

    @staticmethod
    def _intent_run_tag():
        fn = os.path.join(HERE, 'intents', 'paper_intent.por.json')
        try:
            return json.load(open(fn))['stages']['paper_table']['run_tag']
        except Exception:
            return None

    def _prov(self, fn):
        return {'file': os.path.relpath(fn, REPO) if fn.startswith(REPO) else fn,
                'mtime': time.strftime('%Y-%m-%dT%H:%M', time.localtime(os.path.getmtime(fn)))
                if os.path.exists(fn) else None,
                'run_tag': self.run_tag, 'code_sha': self.sha}

    # -- paper table --------------------------------------------------------
    def table(self):
        """{'dir': {col: '<'|'>'}, 'rows': {csv_method: {col: float|None}}, 'prov': {...}}"""
        if self._table is None:
            fn = os.path.join(self.artifacts, 'paper_table_full.csv')
            if not os.path.exists(fn):
                raise FileNotFoundError('paper table CSV missing: {}'.format(fn))
            rows = list(csv.reader(open(fn)))
            hdr = rows[0][1:]
            assert rows[0][0] == 'method' and rows[1][0] == 'DIRECTION', fn
            self._table = {'dir': dict(zip(hdr, rows[1][1:])), 'rows': {}, 'prov': self._prov(fn)}
            for r in rows[2:]:
                self._table['rows'][r[0]] = {
                    c: (float(v) if v not in ('', '-') else None) for c, v in zip(hdr, r[1:])}
        return self._table

    def stats(self):
        """{csv_method: {col: (mean, std, n)}} from paper_table_full_stats.csv, or None."""
        if self._stats is None:
            fn = os.path.join(self.artifacts, 'paper_table_full_stats.csv')
            if not os.path.exists(fn):
                self._stats = {}
            else:
                rows = list(csv.reader(open(fn)))
                hdr = rows[0][1:]
                self._stats = {}
                for r in rows[1:]:
                    d = {}
                    for c, v in zip(hdr, r[1:]):
                        parts = v.split('|') if v else []
                        if len(parts) == 3 and parts[0] != '':
                            d[c] = (float(parts[0]), float(parts[1]) if parts[1] else None,
                                    int(parts[2]) if parts[2] else None)
                    self._stats[r[0]] = d
        return self._stats

    # -- sweeps ---------------------------------------------------------------
    def sweep(self, axis_tok):
        if axis_tok not in self._sweeps:
            fn = os.path.join(self.artifacts, SWEEP_AXES[axis_tok].csv)
            rows = list(csv.DictReader(open(fn))) if os.path.exists(fn) else []
            self._sweeps[axis_tok] = ({(slug(r['objective']), slug(r['metric'])): r for r in rows},
                                      self._prov(fn))
        return self._sweeps[axis_tok]

    # -- ablation -------------------------------------------------------------
    def ablation(self, obj_tok):
        if obj_tok not in self._abl:
            fn = os.path.join(self.ablation_dir, OBJECTIVES[obj_tok].objective, 'ladder_summary.json')
            if os.path.exists(fn):
                d = json.load(open(fn))
                rungs = {r['rung']: r for r in d.get('rungs', [])}
                rungs.setdefault('painter', {'rung': 'painter', 'pct_gap_closed_on_means': 0.0,
                                             'mean_objective': d.get('mean_zero_anchor_objective'),
                                             'mean_minus_opp': (d.get('mean_zero_anchor_objective', 0) or 0)
                                             - (d.get('mean_opp_objective', 0) or 0)})
                rungs.setdefault('OPP', {'rung': 'OPP', 'pct_gap_closed_on_means': 100.0,
                                         'mean_objective': d.get('mean_opp_objective'), 'mean_minus_opp': 0.0})
                self._abl[obj_tok] = (rungs, dict(self._prov(fn), n_deployments=d.get('n_deployments')))
            else:
                self._abl[obj_tok] = ({}, {'file': fn, 'missing': True})
        return self._abl[obj_tok]

    # -- on-Internet deployment -----------------------------------------------
    def internet(self):
        """{(scenario, method_token): row}, provenance"""
        if not hasattr(self, '_internet'):
            fn = os.path.join(self.artifacts, 'actual_deployment_stats.csv')
            rows = list(csv.DictReader(open(fn))) if os.path.exists(fn) else []
            self._internet = ({(r['scenario'], r['method']): r for r in rows}, self._prov(fn))
        return self._internet

    # -- manual ---------------------------------------------------------------
    def manual(self):
        if self._manual is None:
            d = json.load(open(self.manual_json)) if os.path.exists(self.manual_json) else {}
            self._manual = {k: v for k, v in d.items() if not k.startswith('_')}
        return self._manual


def slug(s):
    s = re.sub(r'[^A-Za-z0-9]+', '-', s.strip().lower()).strip('-')
    return s


# =============================================================================
# Resolution
# =============================================================================

@dataclass
class Result:
    key: str
    value: Any = None            # float | str | None
    text: str = ''               # formatted for TeX
    prec: Optional[int] = None
    doc: str = ''                # composed meaning
    source: str = ''             # where it came from (file / formula)
    prov: Dict[str, Any] = field(default_factory=dict)
    error: str = ''              # why undefined (value is None)
    notes: List[str] = field(default_factory=list)   # sign anomalies etc.

    @property
    def ok(self):
        return self.value is not None and not self.error


def _fmt(value, prec):
    if isinstance(value, str):
        return value
    if prec is None:
        return str(value)
    s = '{:.{p}f}'.format(value, p=prec)
    if s.startswith('-0') and float(s) == 0:
        s = s[1:]   # no "-0.0"
    return s


def _better(direction, a, b):
    return a < b if direction == '<' else a > b


class Resolver:
    def __init__(self, source: Source):
        self.src = source

    # -- public ---------------------------------------------------------------
    def resolve(self, key: str) -> Result:
        toks = key.split('.')
        prec = None
        if len(toks) > 1 and re.fullmatch(r'd\d', toks[-1]):
            prec = int(toks[-1][1:])
            toks = toks[:-1]
        try:
            if toks[0] in OBJECTIVES:
                r = self._table_key(key, toks)
            elif toks[0] == 'sweep':
                r = self._sweep_key(key, toks)
            elif toks[0] == 'abl':
                r = self._abl_key(key, toks)
            elif toks[0] == 'internet':
                r = self._internet_key(key, toks)
            elif toks[0] == 'headline':
                r = self._headline_key(key, toks)
            elif toks[0] == 'manual':
                r = self._manual_key(key, toks)
            else:
                return Result(key, error='unknown family {!r}; first token must be one of {}'.format(
                    toks[0], sorted(OBJECTIVES) + ['sweep', 'abl', 'internet', 'headline', 'manual']))
        except _KeyError as e:
            return Result(key, error=str(e))
        if prec is not None:
            r.prec = prec
        if r.value is not None and not r.error:
            r.text = _fmt(r.value, r.prec)
        return r

    # -- table family ---------------------------------------------------------
    def cell_value(self, cell: Tuple[str, str, str, str], stat: str, method: str):
        """(value|None, direction, provenance-ish source string)"""
        c = CELLS[cell]
        t = self.src.table()
        mcsv = METHODS[method].csv
        if c.derive:
            kind, base, arg = c.derive
            v, d, s = self.cell_value(base, stat, method)
            if v is None:
                return None, DERIVED_DIRECTION[cell[3]], s
            if kind == 'ratio_const':
                return v / arg, DERIVED_DIRECTION[cell[3]], s + ' / {:.4f}'.format(arg)
            if kind == 'ratio_method':
                vref, _, _ = self.cell_value(base, 'mean', arg)
                if vref is None or abs(vref) < 1e-12:
                    return None, DERIVED_DIRECTION[cell[3]], s
                if stat == 'n':
                    return v, DERIVED_DIRECTION[cell[3]], s
                return v / vref, DERIVED_DIRECTION[cell[3]], s + ' / {}'.format(METHODS[arg].csv)
            if kind == 'complement':
                if stat in ('std', 'n'):
                    return v, DERIVED_DIRECTION[cell[3]], s
                return arg - v, DERIVED_DIRECTION[cell[3]], '{} - '.format(arg) + s
            raise ValueError(kind)
        direction = t['dir'].get(c.column, '<')
        if stat == 'mean':
            row = t['rows'].get(mcsv)
            if row is None:
                raise _KeyError('method {!r} not in the table CSV'.format(mcsv))
            return row.get(c.column), direction, 'paper_table_full.csv[{}][{}]'.format(mcsv, c.column)
        st = self.src.stats().get(mcsv, {}).get(c.column)
        if not st:
            raise _KeyError('stat {!r} needs paper_table_full_stats.csv (mean|std|n cells), not in artifacts'.format(stat))
        return (st[1] if stat == 'std' else st[2]), direction, 'paper_table_full_stats.csv[{}][{}].{}'.format(mcsv, c.column, stat)

    def _resolve_method(self, tok, cell, stat):
        """literal or role -> literal method token (None if unresolvable)"""
        if tok in METHODS:
            return tok
        if tok in ROLES:
            role = ROLES[tok]
            vals = []
            for m in role.pool:
                v, d, _ = self.cell_value(cell, 'mean', m)
                if v is not None:
                    vals.append((m, v, d))
            if not vals:
                return None
            d = vals[0][2]
            want_best = role.pick == 'best'
            pick = vals[0]
            for cand in vals[1:]:
                if _better(d, cand[1], pick[1]) == want_best and cand[1] != pick[1]:
                    pick = cand
            return pick[0]
        raise _KeyError('unknown method/role {!r}; methods {} roles {}'.format(
            tok, sorted(METHODS), sorted(ROLES)))

    def _table_key(self, key, toks):
        if len(toks) < 6:
            raise _KeyError('table keys are <obj>.<scenario>.<population>.<metric>.<stat>.<method>[.<verb>..]; got {} tokens'.format(len(toks)))
        obj, scen, pop, met, stat, mtok = toks[:6]
        rest = toks[6:]
        for tok, vocab, name in ((scen, SCENARIOS, 'scenario'), (pop, POPULATIONS, 'population'),
                                 (met, METRICS, 'metric'), (stat, STATS, 'stat')):
            if tok not in vocab:
                raise _KeyError('unknown {} {!r}; valid: {}'.format(name, tok, sorted(vocab)))
        cell = (obj, scen, pop, met)
        if cell not in CELLS:
            valid = sorted('{}.{}.{}'.format(s, p, m) for (o, s, p, m) in CELLS if o == obj)
            raise _KeyError('no such cell {}.{}.{}.{}; valid for {}: {}'.format(obj, scen, pop, met, obj, ', '.join(valid)))
        method = self._resolve_method(mtok, cell, stat)
        cellsrc = 'paper_table_full.csv'
        doc = '{obj}; {scen}; {pop}; {met}; {stat}; {meth}'.format(
            obj=OBJECTIVES[obj].doc.split(':')[0], scen=SCENARIOS[scen].doc, pop=POPULATIONS[pop].doc,
            met=METRICS[met].doc, stat=STATS[stat].doc,
            meth=('{} -> {}'.format(ROLES[mtok].doc, METHODS[method].csv if method else '?') if mtok in ROLES
                  else METHODS[mtok].doc))
        prov = dict(self.src.table()['prov'])
        m = METRICS[met]
        r = Result(key, doc=doc, prov=prov, prec=(0 if stat == 'n' else m.prec))
        if method is None:
            r.error = 'role {!r} could not be resolved (no values in the cell)'.format(mtok)
            return r
        v, direction, src = self.cell_value(cell, stat, method)
        r.source = src
        if not rest:
            if v is None:
                r.error = 'cell is empty in the CSV ("-": not evaluated)'
            r.value = v
            return r
        verb, args = rest[0], rest[1:]
        if verb not in VERBS:
            raise _KeyError('unknown verb {!r}; valid: {}'.format(verb, sorted(VERBS)))
        if len(args) != VERBS[verb].nargs:
            raise _KeyError('verb {} takes {} argument(s), got {}'.format(verb, VERBS[verb].nargs, len(args)))
        r.doc += '; ' + VERBS[verb].doc
        if verb == 'name':
            r.value = METHODS[method].tex
            r.prec = None
            r.doc = 'TeX name of {}'.format(doc)
            return r
        others = []
        for a in args:
            om = self._resolve_method(a, cell, 'mean')
            if om is None:
                r.error = 'comparison method {!r} unresolvable'.format(a)
                return r
            ov, _, osrc = self.cell_value(cell, stat, om)
            others.append((a, om, ov, osrc))
            r.doc += ' [{} -> {}]'.format(a, METHODS[om].csv) if a in ROLES else ''
        r.source = src + ' vs ' + ', '.join(o[3] for o in others)
        if v is None or any(o[2] is None for o in others):
            r.error = 'an operand cell is empty ("-")'
            return r
        m_ = others[0][2]
        if verb == 'minus':
            r.value = v - m_
        elif verb in ('worsethan', 'betterthan'):
            r.value = (v - m_) if direction == '<' else (m_ - v)
            if r.value < 0 and others[0][1] == 'opp':
                r.notes.append('NEGATIVE worsethan.opp: SCULPTOR beats the optimum here?! check the cell')
            if verb == 'betterthan':
                r.value = -r.value
        elif verb in ('pctbetter', 'pctworse'):
            if abs(m_) < m.floor:
                r.error = ('|{}| = {:.4g} is below the {} floor {} for a percent-of comparison; '
                           'quote minus (points) instead'.format(others[0][1], m_, met, m.floor))
                return r
            imp = ((m_ - v) if direction == '<' else (v - m_)) / abs(m_) * 100.0
            r.value = imp if verb == 'pctbetter' else -imp
            r.prec = VERBS[verb].prec
        elif verb == 'over':
            if abs(m_) < m.floor:
                r.error = '|{}| = {:.4g} is below the {} floor {} for a ratio'.format(others[0][1], m_, met, m.floor)
                return r
            r.value = v / m_
            r.prec = VERBS[verb].prec
        elif verb == 'pctway':
            lo, hi = others[0][2], others[1][2]
            if abs(hi - lo) < 1e-9:
                r.error = 'LO and HI coincide'
                return r
            r.value = 100.0 * (v - lo) / (hi - lo)
            r.prec = VERBS[verb].prec
        return r

    # -- sweep family ---------------------------------------------------------
    def _sweep_key(self, key, toks):
        if len(toks) != 5:
            raise _KeyError('sweep keys are sweep.<axis>.<objective-slug>.<metric-slug>.<field>')
        _, axis, oslug, mslug, fld = toks
        if axis not in SWEEP_AXES:
            raise _KeyError('unknown sweep axis {!r}; valid: {}'.format(axis, sorted(SWEEP_AXES)))
        if fld not in SWEEP_FIELDS:
            raise _KeyError('unknown sweep field {!r}; valid: {}'.format(fld, sorted(SWEEP_FIELDS)))
        rows, prov = self.src.sweep(axis)
        row = rows.get((oslug, mslug))
        if row is None:
            raise _KeyError('no sweep row {}.{}; valid: {}'.format(
                oslug, mslug, ', '.join('{}.{}'.format(*k) for k in sorted(rows))))
        f = SWEEP_FIELDS[fld]
        raw = row[f.col]
        r = Result(key, prov=prov, prec=f.prec,
                   doc='{}; objective group "{}"; metric "{}"; {}'.format(
                       SWEEP_AXES[axis].doc, row['objective'], row['metric'], f.doc),
                   source='{}[{}][{}]'.format(SWEEP_AXES[axis].csv, '{}/{}'.format(row['objective'], row['metric']), f.col))
        if fld == 'best-baseline-name':
            tex = {m.sweep: m.tex for m in METHODS.values()}.get(raw, raw)
            r.value = tex
        elif f.prec is None:
            r.value = raw
        else:
            r.value = float(raw)
        return r

    # -- ablation family ------------------------------------------------------
    def _abl_key(self, key, toks):
        if len(toks) != 4:
            raise _KeyError('ablation keys are abl.<obj>.<rung>.<field>')
        _, obj, rung, fld = toks
        if obj not in OBJECTIVES:
            raise _KeyError('unknown objective {!r}'.format(obj))
        if rung not in ABL_RUNGS:
            raise _KeyError('unknown rung {!r}; valid: {}'.format(rung, list(ABL_RUNGS)))
        if fld not in ABL_FIELDS:
            raise _KeyError('unknown field {!r}; valid: {}'.format(fld, list(ABL_FIELDS)))
        rungs, prov = self.src.ablation(obj)
        f = ABL_FIELDS[fld]
        r = Result(key, prov=prov, prec=f.prec, source='{}[{}][{}]'.format(prov.get('file'), rung, f.col),
                   doc='ablation ladder, {} objective; {}; {}'.format(obj, ABL_RUNGS[rung].doc, f.doc))
        if prov.get('missing'):
            r.error = 'no ladder_summary.json at {} (copy <obj>/ladder_summary.json under artifacts/ablation/)'.format(prov['file'])
            return r
        row = rungs.get(rung)
        if row is None or row.get(f.col) is None:
            r.error = 'rung {} / field {} absent from the summary'.format(rung, f.col)
            return r
        r.value = float(row[f.col])
        return r

    # -- on-Internet family ---------------------------------------------------
    def _internet_value(self, scen, met, mtok):
        rows, prov = self.src.internet()
        if mtok in ROLES:
            pool = [m for m in ROLES[mtok].pool if m != 'opp' and (scen, m) in rows]
            if not pool:
                return None, None, prov
            d = INTERNET_METRICS[met].direction
            pick = pool[0]
            for c in pool[1:]:
                a, b = float(rows[(scen, c)][met]), float(rows[(scen, pick)][met])
                if (_better(d, a, b) if ROLES[mtok].pick == 'best' else _better(d, b, a)) and a != b:
                    pick = c
            mtok = pick
        elif mtok == 'opp':
            return (0.0 if met == 'gap' else (0.0 if met == 'overloaded' else 100.0)), 'opp', prov
        elif mtok not in METHODS:
            raise _KeyError('unknown method/role {!r}'.format(mtok))
        row = rows.get((scen, mtok))
        if row is None:
            return None, mtok, prov
        v = row[met]
        return (float(v) if v not in ('', 'nan') else None), mtok, prov

    def _internet_key(self, key, toks):
        if len(toks) < 4:
            raise _KeyError('internet keys are internet.<scenario>.<metric>.<method>[.<verb>.<M>]')
        _, scen, met, mtok = toks[:4]
        rest = toks[4:]
        if scen not in INTERNET_SCENARIOS:
            raise _KeyError('unknown internet scenario {!r}; valid: {}'.format(scen, list(INTERNET_SCENARIOS)))
        if met not in INTERNET_METRICS:
            raise _KeyError('unknown internet metric {!r}; valid: {}'.format(met, list(INTERNET_METRICS)))
        m = INTERNET_METRICS[met]
        v, meth, prov = self._internet_value(scen, met, mtok)
        r = Result(key, prec=m.prec, prov=prov,
                   doc='{}; {}; {}'.format(INTERNET_SCENARIOS[scen].doc, m.doc,
                                           ('{} -> {}'.format(ROLES[mtok].doc, meth) if mtok in ROLES else (METHODS[mtok].doc if mtok in METHODS else mtok))),
                   source='actual_deployment_stats.csv[{}][{}][{}]'.format(scen, meth, met))
        if not rest:
            if v is None:
                r.error = 'no row for {} / {} in actual_deployment_stats.csv (run evaluations/actual_deployment_numbers.py)'.format(scen, meth)
            r.value = v
            return r
        verb, args = rest[0], rest[1:]
        if verb not in VERBS:
            raise _KeyError('unknown verb {!r}; valid: {}'.format(verb, sorted(VERBS)))
        if len(args) != VERBS[verb].nargs:
            raise _KeyError('verb {} takes {} argument(s), got {}'.format(verb, VERBS[verb].nargs, len(args)))
        r.doc += '; ' + VERBS[verb].doc
        if verb == 'name':
            r.value = METHODS[meth].tex if meth in METHODS else meth
            r.prec = None
            return r
        others = []
        for a_ in args:
            ov, om, _ = self._internet_value(scen, met, a_)
            others.append((a_, om, ov))
            if a_ in ROLES:
                r.doc += ' [{} -> {}]'.format(a_, om)
        if v is None or any(o[2] is None for o in others):
            r.error = 'an operand is missing from actual_deployment_stats.csv'
            return r
        d = m.direction
        m_ = others[0][2]
        if verb == 'minus':
            r.value = v - m_
        elif verb in ('worsethan', 'betterthan'):
            r.value = (v - m_) if d == '<' else (m_ - v)
            if verb == 'betterthan':
                r.value = -r.value
        elif verb in ('pctbetter', 'pctworse'):
            if abs(m_) < m.floor:
                r.error = '|{}| = {:.4g} is below the {} floor {}; quote minus (points) instead'.format(others[0][1], m_, met, m.floor)
                return r
            imp = ((m_ - v) if d == '<' else (v - m_)) / abs(m_) * 100.0
            r.value = imp if verb == 'pctbetter' else -imp
            r.prec = VERBS[verb].prec
        elif verb == 'over':
            if abs(m_) < m.floor:
                r.error = '|{}| = {:.4g} is below the {} floor {}'.format(others[0][1], m_, met, m.floor)
                return r
            r.value = v / m_
            r.prec = VERBS[verb].prec
        elif verb == 'pctway':
            lo, hi = others[0][2], others[1][2]
            if abs(hi - lo) < 1e-9:
                r.error = 'LO and HI coincide'
                return r
            r.value = 100.0 * (v - lo) / (hi - lo)
            r.prec = VERBS[verb].prec
        return r

    # -- headline / manual ----------------------------------------------------
    def _headline_key(self, key, toks):
        if len(toks) != 2:
            raise _KeyError('headline keys are headline.<slug>')
        h = HEADLINES.get(toks[1])
        if h is None:
            raise _KeyError('unknown headline {!r}; declared: {}'.format(toks[1], sorted(HEADLINES)))
        if not h.doc.strip():
            raise _KeyError('headline {} has no doc -- refuse to emit an undocumented hand-named number'.format(key))
        r = Result(key, prec=h.prec, doc='HEADLINE: ' + h.doc, source=h.formula)
        deps = []

        def R(k):
            sub = self.resolve(k)
            deps.append(sub)
            if not sub.ok:
                raise _Undefined('{}: {}'.format(k, sub.error))
            return sub.value
        try:
            if h.formula.startswith('py:'):
                r.value = eval(h.formula[3:], {'R': R, 'abs': abs, 'min': min, 'max': max})
            else:
                r.value = R(h.formula)
        except _Undefined as e:
            r.error = 'operand undefined: {}'.format(e)
        if deps:
            r.prov = dict(deps[0].prov)
            r.doc += ' | via: ' + ' ; '.join('{} = {}'.format(d.key, d.doc) for d in deps)
        return r

    def _manual_key(self, key, toks):
        if len(toks) != 2:
            raise _KeyError('manual keys are manual.<slug>')
        ent = self.src.manual().get(toks[1])
        r = Result(key, source=os.path.relpath(self.src.manual_json, REPO))
        if ent is None:
            r.error = 'not in {} (add {{"value":..,"doc":..,"from":..}})'.format(os.path.basename(self.src.manual_json))
            r.doc = 'MANUAL (missing)'
            return r
        if not ent.get('doc') or not ent.get('from'):
            r.error = 'manual entry needs both "doc" and "from"'
            return r
        r.doc = 'MANUAL: {} (from: {})'.format(ent['doc'], ent['from'])
        if ent.get('placeholder') or ent.get('value') is None:
            r.error = 'manual placeholder, no value yet -- {}'.format(ent['from'])
            return r
        r.value = ent['value']
        r.prec = ent.get('prec', 0 if isinstance(ent['value'], int) else 2)
        r.prov = {'file': r.source, 'from': ent['from']}
        return r


class _KeyError(Exception):
    pass


class _Undefined(Exception):
    pass


# =============================================================================
# Enumeration / emission / checking
# =============================================================================

def base_keys():
    """Every literal-method mean cell: what the table shows."""
    out = []
    for (o, s, p, m) in CELLS:
        for meth in METHODS:
            out.append('{}.{}.{}.{}.mean.{}'.format(o, s, p, m, meth))
    return out


def referenced_keys(tex_paths):
    keys = {}
    for p in tex_paths:
        if not os.path.exists(p):
            continue
        for ln, line in enumerate(open(p, errors='replace'), 1):
            for k in PN_RE.findall(line):
                keys.setdefault(k.strip(), []).append('{}:{}'.format(os.path.basename(p), ln))
    return keys


def _tex_escape_comment(s):
    return s.replace('\n', ' ')


def emit(resolver: Resolver, tex_paths, out_dir, write=True, all_base=True):
    refs = referenced_keys(tex_paths)
    keys = list(refs)
    if all_base:
        keys += [k for k in base_keys() if k not in refs]
    keys += ['headline.' + h for h in HEADLINES if 'headline.' + h not in refs]
    keys += ['manual.' + m for m in resolver.src.manual() if 'manual.' + m not in refs]
    results = {k: resolver.resolve(k) for k in keys}

    prev = {}
    snap_fn = os.path.join(out_dir, 'paper_numbers.json')
    if os.path.exists(snap_fn):
        try:
            prev = json.load(open(snap_fn)).get('keys', {})
        except Exception:
            prev = {}

    if write:
        os.makedirs(out_dir, exist_ok=True)
        tex_fn = os.path.join(out_dir, 'paper_numbers.tex')
        with open(tex_fn, 'w') as f:
            f.write('% generated by evaluations/paper_numbers.py -- DO NOT EDIT\n')
            f.write('% code {} ; run_tag {} ; {}\n'.format(resolver.src.sha, resolver.src.run_tag,
                                                          time.strftime('%Y-%m-%d %H:%M')))
            f.write('% a key referenced in the paper but not defined here renders red as TBD:<key>\n\n')
            for section, pred in (('REFERENCED IN THE PAPER', lambda k: k in refs),
                                  ('HEADLINE', lambda k: k.startswith('headline.') and k not in refs),
                                  ('MANUAL', lambda k: k.startswith('manual.') and k not in refs),
                                  ('TABLE CELLS (as shown)', lambda k: k not in refs and not k.startswith(('headline.', 'manual.')))):
                f.write('% ' + '=' * 70 + '\n% ' + section + '\n% ' + '=' * 70 + '\n')
                for k in keys:
                    if not pred(k):
                        continue
                    r = results[k]
                    f.write('% {}\n'.format(k))
                    if k in refs:
                        f.write('%   used at: {}\n'.format(', '.join(refs[k])))
                    f.write('%   {}\n'.format(_tex_escape_comment(r.doc or r.error)))
                    if r.source:
                        f.write('%   source: {}\n'.format(_tex_escape_comment(r.source)))
                    if r.prov:
                        f.write('%   provenance: {}\n'.format(json.dumps(r.prov, sort_keys=True)))
                    for n in r.notes:
                        f.write('%   !! {}\n'.format(n))
                    if r.ok:
                        f.write('\\pndef{{{}}}{{{}}}\n'.format(k, r.text))
                    else:
                        f.write('%   UNDEFINED: {}\n'.format(_tex_escape_comment(r.error)))
                    f.write('\n')
        snap = {'generated': time.strftime('%Y-%m-%dT%H:%M'), 'code_sha': resolver.src.sha,
                'run_tag': resolver.src.run_tag,
                'keys': {k: {'value': r.value, 'text': r.text, 'doc': r.doc, 'source': r.source,
                             'prov': r.prov, 'error': r.error, 'used_at': refs.get(k, [])}
                         for k, r in results.items()}}
        with open(snap_fn, 'w') as f:
            json.dump(snap, f, indent=1, sort_keys=True)
    return results, refs, prev


def report(results, refs, prev, out):
    undefined = [(k, results[k]) for k in refs if not results[k].ok]
    ok_ref = [k for k in refs if results[k].ok]
    changed = []
    for k in refs:
        r = results[k]
        p = prev.get(k)
        if p and r.ok and p.get('text') not in (None, '') and p['text'] != r.text:
            changed.append((k, p['text'], r.text))
    notes = [(k, n) for k, r in results.items() for n in r.notes]
    manual = [k for k in refs if k.startswith('manual.')]
    out('[paper-numbers] {} key(s) referenced in the paper: {} defined, {} UNDEFINED, {} manual'.format(
        len(refs), len(ok_ref), len(undefined), len(manual)))
    if undefined:
        out('\n  UNDEFINED (render red as TBD:<key>):')
        for k, r in undefined:
            out('    {}  @ {}\n      -> {}'.format(k, ', '.join(refs[k]), r.error))
    if changed:
        out('\n  CHANGED since the last emit (re-read the sentence):')
        for k, a, b in changed:
            out('    {}  {} -> {}  @ {}'.format(k, a, b, ', '.join(refs[k])))
    if notes:
        out('\n  NOTES:')
        for k, n in notes:
            out('    {}: {}'.format(k, n))
    unused_head = [k for k in results if k.startswith('headline.') and k not in refs]
    if unused_head:
        out('\n  headline keys declared but not referenced: {}'.format(', '.join(unused_head)))
    return len(undefined)


# =============================================================================
# CLI
# =============================================================================

def _paper_tex_paths(paper_dir, tex):
    if tex:
        return [tex]
    return [os.path.join(paper_dir, 'resilience.tex'), os.path.join(paper_dir, 'resilience.md')]


def cmd_emit(a, write):
    src = Source(a.artifacts, a.ablation_dir, a.manual)
    res = Resolver(src)
    out_dir = os.path.join(a.paper_dir, 'tables')
    results, refs, prev = emit(res, _paper_tex_paths(a.paper_dir, a.tex), out_dir, write=write,
                               all_base=not a.no_base)
    if write:
        print('[paper-numbers] wrote {}/paper_numbers.tex (+ .json snapshot)'.format(out_dir))
    n_undef = report(results, refs, prev, print)
    return 1 if (a.strict and n_undef) else 0


def cmd_ls(a):
    src = Source(a.artifacts, a.ablation_dir, a.manual)
    res = Resolver(src)
    keys = base_keys() + ['headline.' + h for h in HEADLINES] + ['manual.' + m for m in src.manual()]
    for axis in SWEEP_AXES:
        rows, _ = src.sweep(axis)
        for (o, m) in sorted(rows):
            keys += ['sweep.{}.{}.{}.{}'.format(axis, o, m, f) for f in SWEEP_FIELDS]
    irows, _ = src.internet()
    for (scen, meth) in sorted(irows):
        keys += ['internet.{}.{}.{}'.format(scen, met, meth) for met in INTERNET_METRICS]
    for obj in OBJECTIVES:
        rungs, prov = src.ablation(obj)
        if not prov.get('missing'):
            keys += ['abl.{}.{}.{}'.format(obj, r, f) for r in ABL_RUNGS for f in ABL_FIELDS]
    pat = re.compile(a.grep) if a.grep else None
    for k in keys:
        if pat and not pat.search(k):
            continue
        if a.values:
            r = res.resolve(k)
            print('{:<70s} {}'.format(k, r.text if r.ok else '-- ' + r.error))
        else:
            print(k)
    if a.grep and a.expand:
        print('\n(derived keys are not enumerated: append .<verb>.<method|role> to a cell key; '
              'verbs: {}; roles: {})'.format(', '.join(VERBS), ', '.join(ROLES)))


def cmd_explain(a):
    src = Source(a.artifacts, a.ablation_dir, a.manual)
    res = Resolver(src)
    for k in a.keys:
        r = res.resolve(k)
        print('=' * 78)
        print(k)
        print('  value    : {}'.format(r.text if r.ok else 'UNDEFINED -- ' + r.error))
        print('  meaning  : {}'.format(r.doc))
        if r.source:
            print('  source   : {}'.format(r.source))
        if r.prov:
            print('  prov     : {}'.format(json.dumps(r.prov, sort_keys=True)))
        for n in r.notes:
            print('  !!       : {}'.format(n))


def cmd_vocab(a):
    for title, vocab in (('OBJECTIVES', OBJECTIVES), ('SCENARIOS', SCENARIOS), ('POPULATIONS', POPULATIONS),
                         ('METRICS', METRICS), ('STATS', STATS), ('METHODS', METHODS), ('ROLES', ROLES),
                         ('VERBS', VERBS), ('SWEEP AXES', SWEEP_AXES), ('SWEEP FIELDS', SWEEP_FIELDS),
                         ('ABLATION RUNGS', ABL_RUNGS), ('ABLATION FIELDS', ABL_FIELDS),
                         ('INTERNET SCENARIOS', INTERNET_SCENARIOS), ('INTERNET METRICS', INTERNET_METRICS)):
        print('\n' + title)
        for k, t in vocab.items():
            extra = ''
            if 'units' in t.extra:
                extra = '  [{} ; {} decimals ; floor {}]'.format(t.units or 'unitless', t.prec, t.floor)
            print('  {:<18s} {}{}'.format(k, t.doc, extra))
    print('\nVALID CELLS (obj.scenario.population.metric)')
    for (o, s, p, m) in CELLS:
        print('  {}.{}.{}.{}'.format(o, s, p, m))
    print('\nHEADLINES')
    for k, h in HEADLINES.items():
        print('  headline.{:<44s} {}  = {}'.format(k, h.doc, h.formula))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--artifacts', default=DEFAULT_ARTIFACTS)
    ap.add_argument('--ablation-dir', default=None, help='dir holding <objective>/ladder_summary.json '
                                                          '(default <artifacts>/ablation)')
    ap.add_argument('--manual', default=MANUAL_JSON)
    ap.add_argument('--paper-dir', default=DEFAULT_PAPER_DIR)
    ap.add_argument('--tex', default=None, help='scan this file instead of <paper-dir>/resilience.{tex,md}')
    sub = ap.add_subparsers(dest='cmd')
    for name in ('emit', 'check'):
        p = sub.add_parser(name)
        p.add_argument('--strict', action='store_true', help='exit 1 if any referenced key is undefined')
        p.add_argument('--no-base', action='store_true', help='do not emit the unreferenced table cells')
    p = sub.add_parser('ls')
    p.add_argument('--grep', default=None)
    p.add_argument('--values', action='store_true')
    p.add_argument('--expand', action='store_true')
    p = sub.add_parser('explain')
    p.add_argument('keys', nargs='+')
    sub.add_parser('vocab')
    a = ap.parse_args(argv)
    if a.cmd == 'emit':
        return cmd_emit(a, write=True)
    if a.cmd == 'check':
        return cmd_emit(a, write=False)
    if a.cmd == 'ls':
        return cmd_ls(a)
    if a.cmd == 'explain':
        return cmd_explain(a)
    if a.cmd == 'vocab':
        return cmd_vocab(a)
    ap.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main() or 0)
