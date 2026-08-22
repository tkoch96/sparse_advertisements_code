"""Five-panel policy-ladder figure over N (Tom, 2026-08-13): the three
plot_policy panels (steady congested frac, clean routed latency,
popp-failure congestion) plus popp-failure AFFECTED-user routed latency
and pop-failure congestion, one line per ladder arm.

Data: cache/model_error/steady/policy_steady*.json (steady_metrics),
cache/model_error/failure/policy_failure*.json (failure_metrics);
opp refs from cache/model_error/opp_ref_georand.json; painter refs from
the mesh sidecars (panels 1-3 only; no painter failure_metrics run yet).
Env: POLICY_PLOT_STAT=mean|median (default median), POLICY_PLOT_OUT.
Output: figures/<out>.{png,pdf}
"""
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from helpers.figpaths import fig_path  # dashboard figures -> figures/dashboards/<dashboard>/

_STAT = st.mean if os.environ.get('POLICY_PLOT_STAT', 'median') == 'mean' \
    else st.median
_OUT = os.environ.get('POLICY_PLOT_OUT', 'policy_ladder_overN_5panel')

NS = [1, 2, 5, 10, 20, 50]
# Monotone ladder (Tom, 2026-08-14): each arm adds ONE capability.
# Dir names are historical on-disk names; display labels are the ladder
# positions (L3 = dir L4_nomem_sched, L4 = dir L5_nomem_smart,
# L5 = dir L5_nodir_smart). L2->L3 isolates the belief LP (at MC_NUM=1
# the "monte carlo" part is one draw; the real delta is congestion-aware
# routing). L5->L6 adds direction AND explore targeting (bundled).
# v3 ladder (Tom 2026-08-16: WHAT/WHEN decomposition, post-fix
# semantics era -- lambduh=0, p5 cutoff, 100 iters). Probing is pure
# grounding (target=current) through L5; L6 adds smart WHAT on the
# fixed schedule; L7 adds conservative smart WHEN. v2-era ARMS are in
# git history; v2 stores quarantined (cache/model_error/V2_ERA).
# v4 era (2026-08-16 late): WHAT is dead (probes always ground at the
# current advertisement -- DFO-mandated + measured optimism drift); L6 =
# slotted WHEN (even mean rate, surprise-biased within slots); old
# maxinfo-L6 / decision-L6' / smart-gate-L7 all retired.
ARMS = [('L1_nomc_fixed', 'L1 no_mc+fixed', '#2a78d6'),
        ('L2_nomc_sched', 'L2 no_mc+sched', '#eb6834'),
        ('L3_nomem_sched', 'L3 no_mem+sched', '#1baf7a'),
        ('L4_nodir_sched', 'L4 no_dir+sched', '#eda100'),
        ('L5_full_sched', 'L5 full+sched', '#e87ba4'),
        ('L6_full_slotted', 'L6 slotted WHEN', '#4a3aa7')]


def _key(dirpath):
    parts = dirpath.split(os.sep)
    return parts[-2], int(parts[-1][1:])


_GAMMA = float(os.environ.get('POLICY_PLOT_GAMMA', '0.1'))
# store-tag prefix so cloned tabs (e.g. the LB-cache A/B) read ONLY their
# own stores instead of glob-merging with the main ladder's
_TAG_PFX = os.environ.get('POLICY_PLOT_TAG_PREFIX', 'policy')


def load():
    """data[panel][(arm, N)] = list of (seed, value); second return is
    {rung: {seed: train_obj_total}} for the REFS advs (opp, painter)."""
    data = {k: {} for k in
            ('scong', 'clean', 'train_obj', 'popp_cong', 'popp_afflat',
             'pop_cong', 'iters', 'spent')}
    steady_obj, fail_obj = {}, {}
    ref_steady, ref_fail = {}, {}
    # panel_refs[panel][rung][seed] = value, for opp/painter/... — every
    # panel gets its refs from the SAME eval pipeline as the arms.
    panel_refs = {}

    def _pref(panel, rung, seed, val):
        if val is not None:
            panel_refs.setdefault(panel, {}).setdefault(
                rung, {})[seed] = val
    for sf in glob.glob('cache/model_error/steady/{}_steady*.json'.format(_TAG_PFX)):
        for e in json.load(open(sf)):
            if not e.get('solved'):
                continue
            if e['dir'] == 'REFS':
                if e.get('train_obj') is not None:
                    ref_steady.setdefault(
                        e['rung'], {})[e['seed']] = e['train_obj']
                _pref('scong', e['rung'], e['seed'],
                      e['steady_congested_frac'])
                _pref('clean', e['rung'], e['seed'], e['clean_avg_lat'])
                continue
            k, s = _key(e['dir']), e['seed']
            data['scong'].setdefault(k, []).append(
                (s, e['steady_congested_frac']))
            if e['clean_avg_lat'] is not None:
                data['clean'].setdefault(k, []).append(
                    (s, e['clean_avg_lat']))
            if e.get('train_obj') is not None:
                steady_obj.setdefault(k, {})[s] = e['train_obj']
            if e.get('n_iters') is not None:
                data['iters'].setdefault(k, []).append((s, e['n_iters']))
            if e.get('probes_spent') is not None:
                data['spent'].setdefault(k, []).append(
                    (s, e['probes_spent']))
    for ff in glob.glob('cache/model_error/failure/{}_failure*.json'.format(_TAG_PFX)):
        for e in json.load(open(ff)):
            if not e.get('solved'):
                continue
            if e['dir'] == 'REFS':
                if e['popp'].get('obj_cost_sum') is not None:
                    ref_fail.setdefault(
                        e['rung'], {})[e['seed']] = \
                        e['popp']['obj_cost_sum']
                _pref('popp_cong', e['rung'], e['seed'],
                      e['popp']['cong_mean'])
                _pref('popp_afflat', e['rung'], e['seed'],
                      e['popp']['affected_routed_lat_mean'])
                _pref('pop_cong', e['rung'], e['seed'],
                      e['pop']['cong_mean'])
                continue
            k, s = _key(e['dir']), e['seed']
            data['popp_cong'].setdefault(k, []).append(
                (s, e['popp']['cong_mean']))
            if e['popp']['affected_routed_lat_mean'] is not None:
                data['popp_afflat'].setdefault(k, []).append(
                    (s, e['popp']['affected_routed_lat_mean']))
            data['pop_cong'].setdefault(k, []).append(
                (s, e['pop']['cong_mean']))
            if e['popp'].get('obj_cost_sum') is not None:
                fail_obj.setdefault(k, {})[s] = e['popp']['obj_cost_sum']
    # Full in-loop training objective (Tom, 2026-08-14): steady soft
    # cost + gamma * Sum_popps(soft cost under that popp's failure) —
    # the composition modeled_objective() descends (lambduh*norm
    # omitted: ~1e-5). Only cells with BOTH stores land in the panel.
    for k, by_seed in steady_obj.items():
        for s, v in by_seed.items():
            if k in fail_obj and s in fail_obj[k]:
                data['train_obj'].setdefault(k, []).append(
                    (s, v + _GAMMA * fail_obj[k][s]))
    # CANARY: opp evaluated first AND last in each eval child; drift
    # means shared-instance eval contamination is back (2026-08-14:
    # refs-evaluated-last were inflated to the soft bound).
    for store, name in ((ref_steady, 'steady'), (ref_fail, 'failure')):
        for s in store.get('opp', {}):
            v0 = store['opp'][s]
            v1 = store.get('opp_canary', {}).get(s)
            if v1 is not None and abs(v1 - v0) > max(1.0, 0.05 * abs(v0)):
                print('[CANARY-WARNING] {} opp drifted seed {}: first '
                      '{:.2f} vs last {:.2f} — in-process eval '
                      'contamination!'.format(name, s, v0, v1), flush=True)
    train_refs = {}
    for rung in set(ref_steady) & set(ref_fail):
        train_refs[rung] = {
            s: ref_steady[rung][s] + _GAMMA * ref_fail[rung][s]
            for s in set(ref_steady[rung]) & set(ref_fail[rung])}
    panel_refs['train_obj'] = train_refs
    # SANITY GATE (Tom, 2026-08-14): no arm may beat one-per-peering on
    # the STEADY objective per seed -- that below-opp signature is always
    # an eval bug (it caught the primary-LP congestion-blind scalar the
    # night this was added). STEADY ONLY: the gamma-composite can
    # legitimately go below opp, because opp advertises every popp so
    # every single-popp failure congests someone, while sparse advs get
    # free no-op failures on unused popps. SCULPTOR_SANITY_ASSERT=0
    # renders anyway.
    viols = []
    for k, by_seed in steady_obj.items():
        for s, v in by_seed.items():
            ref = ref_steady.get('opp', {}).get(s)
            if ref is not None and v < ref - 0.5:
                viols.append('{} N{} seed{}: steady {:.2f} < opp {:.2f}'
                             .format(k[0], k[1], s, v, ref))
    if viols and os.environ.get('SCULPTOR_SANITY_ASSERT', '1') != '0':
        raise AssertionError(
            'STEADY train_obj below one-per-peering (impossible; eval '
            'bug):\n  ' + '\n  '.join(viols))
    return data, train_refs, panel_refs


def _stat_anchored(pairs, ref_by_seed):
    """Seed-anchored statistic for partially-landed data: pair each
    landed seed against ITS OWN per-seed ref, then re-center on the
    all-seed ref mean. Equals the plain statistic once every seed has
    landed; while partial, it can never show an arm crossing a ref the
    per-seed values never cross (which deployment landed first is not a
    finding)."""
    anchored = [v - ref_by_seed[s] for s, v in pairs if s in ref_by_seed]
    if not anchored:
        return _STAT([v for _, v in pairs])
    return _STAT(anchored) + _STAT(list(ref_by_seed.values()))


def main():
    # ALL refs (opp + painter, per seed, every panel) come from the
    # eval pipeline's REFS entries — no external ref files (Tom,
    # 2026-08-14: one pipeline; refs can never desync from the arms).
    data, train_refs, panel_refs = load()

    def _ref(panel, rung):
        d = panel_refs.get(panel, {}).get(rung)
        return _STAT(list(d.values())) if d else None

    sname = 'mean' if _STAT is st.mean else 'median'
    panels = (
        ('scong', '{} frac congested/stranded, STEADY'.format(sname)),
        ('clean', '{} steady latency of ROUTED traffic (ms)'.format(
            sname)),
        ('train_obj',
         '{} TRAINING objective: steady + {}*sum popp-fail (soft cost)'
         .format(sname, _GAMMA)),
        ('popp_cong', '{} frac congested, popp FAILURES'.format(sname)),
        ('popp_afflat',
         '{} routed latency of AFFECTED users, popp FAILURES (ms)'.format(
             sname)),
        ('pop_cong', '{} frac congested, pop FAILURES'.format(sname)),
    )
    fig, axes = plt.subplots(1, 6, figsize=(28.5, 4.6))
    for ax, (dk, ylab) in zip(axes, panels):
        d = data[dk]
        anchor = panel_refs.get(dk, {}).get('opp')
        ref_opp, ref_painter = _ref(dk, 'opp'), _ref(dk, 'painter')
        for arm, label, color in ARMS:
            ys = [(_stat_anchored(d[(arm, n)], anchor) if anchor else
                   _STAT([v for _, v in d[(arm, n)]]))
                  if (arm, n) in d else np.nan
                  for n in NS]
            ax.plot(NS, ys, 'o-', color=color, label=label, lw=1.8, ms=4.5)
        if ref_opp is not None and np.isfinite(ref_opp):
            ax.axhline(ref_opp, color='0.35', ls='--', lw=1.1)
            ax.text(55, ref_opp, 'opp', va='center', fontsize=8,
                    color='0.35')
        if ref_painter is not None and np.isfinite(ref_painter):
            ax.axhline(ref_painter, color='0.35', ls=':', lw=1.1)
            ax.text(55, ref_painter, 'painter', va='center', fontsize=8,
                    color='0.35')
        ax.set_xscale('log')
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        ax.set_xlabel('measurement budget N')
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
        if (ref_opp is not None and ref_painter is not None
                and np.isfinite(ref_opp) and np.isfinite(ref_painter)):
            # every panel banded opp..painter +/-10% with FIXED
            # orientation (Tom 2026-08-17): opp at the BOTTOM edge,
            # painter at the TOP, regardless of numeric order (the
            # axis inverts itself if a metric's orientation flips);
            # pad floors keep degenerate bands readable
            gap = ref_painter - ref_opp
            pad = max(0.1 * abs(gap),
                      0.05 * max(abs(ref_opp), abs(ref_painter)), 1e-3)
            sgn = 1.0 if gap >= 0 else -1.0
            ax.set_ylim(ref_opp - sgn * pad, ref_painter + sgn * pad)
        else:
            ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        'Policy ladder over budget N — 6 metrics (georand, 100 iters max, '
        'exit-on-budget, cache off, MC=1, congestion-aware objective; '
        '{}s of 5 deployments; dashed=one-per-peering, dotted=painter)'
        .format(sname.upper()), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ('png', 'pdf'):
        fig.savefig(fig_path('{}.{}'.format(_OUT, ext)), dpi=170)
    print('wrote', fig_path('{}.png'.format(_OUT)))

    # BIG single-panel objective figure (Tom 2026-08-16: "I care about
    # the objective -- make that the only plot, big enough to see"):
    # same data/refs/banding as the train_obj panel, rendered alone.
    figO, axO = plt.subplots(figsize=(11.5, 7.5))
    dk = 'train_obj'
    d = data[dk]
    anchorO = panel_refs.get(dk, {}).get('opp')
    ro, rp = _ref(dk, 'opp'), _ref(dk, 'painter')
    for arm, label, color in ARMS:
        ys = [(_stat_anchored(d[(arm, n)], anchorO) if anchorO else
               _STAT([v for _, v in d[(arm, n)]]))
              if (arm, n) in d else np.nan for n in NS]
        axO.plot(NS, ys, 'o-', color=color, label=label, lw=2.2, ms=6)
    if ro is not None and np.isfinite(ro):
        axO.axhline(ro, color='0.35', ls='--', lw=1.3)
        axO.text(55, ro, 'opp', va='center', fontsize=10, color='0.35')
    if rp is not None and np.isfinite(rp):
        axO.axhline(rp, color='0.35', ls=':', lw=1.3)
        axO.text(55, rp, 'painter', va='center', fontsize=10, color='0.35')
    axO.set_xscale('log')
    axO.set_xticks(NS)
    axO.set_xticklabels([str(n) for n in NS])
    axO.set_xlabel('measurement budget N', fontsize=12)
    axO.set_ylabel('{} TRAINING objective: steady + {}*sum popp-fail'
                   .format(sname, _GAMMA), fontsize=12)
    axO.grid(True, alpha=0.25)
    if (ro is not None and rp is not None
            and np.isfinite(ro) and np.isfinite(rp)):
        lo, hi = sorted((ro, rp))
        pad = max(0.1 * (hi - lo), 0.05 * max(abs(hi), abs(lo)), 1e-3)
        axO.set_ylim(lo - pad, hi + pad)
    axO.legend(fontsize=11, frameon=False)
    figO.tight_layout()
    figO.savefig(fig_path('{}_objective.png'.format(_OUT)), dpi=170)
    plt.close(figO)
    print('wrote', fig_path('{}_objective.png'.format(_OUT)))

    # Companion figure (Tom, 2026-08-14): how long each arm actually
    # trains and how much of the budget it actually spends — mean
    # n_iters and probes_spent over N (from the run JSONs via the
    # steady store). The L5-vs-L6 mechanism lives here: remeasure-stops
    # truncate both.
    fig2, axes2 = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, dk, ylab in ((axes2[0], 'iters',
                          '{} total training iterations'.format(sname)),
                         (axes2[1], 'spent',
                          '{} measurements spent'.format(sname))):
        d = data[dk]
        for arm, label, color in ARMS:
            ys = [_STAT([v for _, v in d[(arm, n)]])
                  if (arm, n) in d else np.nan for n in NS]
            ax.plot(NS, ys, 'o-', color=color, label=label, lw=1.8,
                    ms=4.5)
        if dk == 'spent':
            ax.plot(NS, NS, ls='--', color='0.35', lw=1.1)
            ax.text(55, 50, 'N', va='center', fontsize=8, color='0.35')
        ax.set_xscale('log')
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        ax.set_xlabel('measurement budget N')
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)
    axes2[0].legend(fontsize=8, frameon=False)
    fig2.suptitle('Training length and budget spend over N', fontsize=10)
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ('png', 'pdf'):
        fig2.savefig(fig_path('{}_iters.{}'.format(_OUT, ext)), dpi=170)
    print('wrote', fig_path('{}_iters.png'.format(_OUT)))


if __name__ == '__main__':
    main()
