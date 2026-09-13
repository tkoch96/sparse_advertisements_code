"""On-Internet (RIPE Atlas) result numbers for the paper (Tom 2026-09-12).

Writes figures/paper_artifacts/actual_deployment_stats.csv, served by
evaluations/paper_numbers.py as the `internet.*` key family:

    internet.<steady|linkfail|sitefail>.<gap|overloaded|within10|within50|within100>.<method>[.verb.M]

PRIMARY SOURCE: the real-deployment metrics pickle the paper figures were made
from (old_scripts/make_actual_deployment_plots.py --dpsize actual_third_prototype),
pulled from the old campus VM into ~/Documents/actual_deployment_figures/cache/.
The statistics are computed exactly as that script prints them (its run.log
reproduces the prose numbers: SCULPTOR 1.93 ms / 91.8% steady, 7.93 ms link,
13.09 ms site; PAINTER 5.50 ms / 88.0%; Unicast 14.24 / 25.14 ms).

FALLBACK: the figure PDFs (vector CDF polylines). Approximate (+-1 ms on gap
for short-tailed curves, worse for long-tailed ones); used only when the
pickle is absent, and the CSV says which source produced each row.

Metric definitions (x = optimal - achieved latency per user, traffic weighted;
failure rows are over the users affected by each failed element, averaged over
the failure scenarios; one deployment, 10 PEERING/Vultr sites, 493 RIPE Atlas
user groups):
  gapall     mean ms above the one-per-peering latency over ALL traffic with
             overloaded users priced at the 450 ms cap: the plot script's own
             printed "Average latency difference" (anycast/PAINTER: 300+ ms)
  gap        mean ms above the one-per-peering latency over NON-overloaded
             traffic. In this evaluation an overloaded user is priced at the
             450 ms congestion cap, so including them makes anycast/PAINTER's
             failure gaps 300+ ms and meaningless; the script's own printed
             average includes them. `gap` here EXCLUDES them and pairs with
             `overloaded`.
  overloaded percent of (affected) traffic whose achieved latency hit the
             450 ms congestion cap, i.e. landed on an over-capacity link.
             This is the definition that reproduces the prose's "PAINTER
             overloads 95% of traffic during a site failure" (95.0%); under
             link failure PAINTER is 75.8% (the prose's 69.6% is anycast's).
  within10/50/100  percent of traffic within N ms of its one-per-peering
             latency; overloaded traffic counts as NOT within (as the plot
             script does).

Usage:
  python -m evaluations.actual_deployment_numbers [--pickle P] [--figures DIR] [--out CSV]
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_PICKLE = os.path.expanduser(
    '~/Documents/actual_deployment_figures/cache/popp_failure_latency_comparison_actual_third_prototype.pkl')
DEFAULT_FIGURES = os.path.expanduser('~/Documents/resilient_advertisements_paper/figures')
DEFAULT_OUT = os.path.join(REPO, 'figures', 'paper_artifacts', 'actual_deployment_stats.csv')
CONGESTION_CAP_MS = 450.0     # the eval's latency for an overloaded user (perf2 == 450)
NO_ROUTE_LATENCY = 30000
METHOD_TOKEN = {'anycast': 'anycast', 'one_per_pop': 'unicast', 'painter': 'painter',
                'sparse': 'sculptor', 'anyopt': 'anyopt'}
THRESHOLDS = (10, 50, 100)
COLS = ['scenario', 'method', 'gap', 'gapall', 'overloaded', 'within10', 'within50', 'within100', 'n_scenarios', 'source']


# ----------------------------------------------------------------------------
# primary: the metrics pickle
# ----------------------------------------------------------------------------

def _within(diffs, wts, over_frac):
    """percent within N ms, overloaded traffic counted as not within."""
    if not diffs:
        return {t: 0.0 for t in THRESHOLDS}
    d = np.array(diffs); w = np.array(wts)
    o = np.argsort(d); d, w = d[o], w[o]
    F = np.cumsum(w) / w.sum()
    out = {}
    for t in THRESHOLDS:
        # share of non-overloaded traffic with (optimal - achieved) >= -t
        share = 1.0 - F[np.searchsorted(d, -t, side='left') - 1] if np.searchsorted(d, -t, side='left') > 0 else 1.0
        out[t] = 100.0 * (1 - over_frac) * share
    return out


def stats_from_pickle(pkl_path):
    m = pickle.load(open(pkl_path, 'rb'))
    sims = sorted(m['best_latencies'])
    rows = []
    for sol, tok in METHOD_TOKEN.items():
        # -- steady state -----------------------------------------------------
        diffs, wts = [], []
        for ri in sims:
            lat = m['latencies'][ri].get(sol) or {}
            vol = m['deployment'][ri]['ug_to_vol']
            for ug, best in m['best_latencies'][ri].items():
                if ug in lat:
                    diffs.append(best - lat[ug]); wts.append(vol[ug])
        if diffs:
            w = _within(diffs, wts, 0.0)
            rows.append(dict(scenario='steady', method=tok, gap=-np.average(diffs, weights=wts),
                             gapall=-np.average(diffs, weights=wts), overloaded=0.0,
                             within10=w[10], within50=w[50], within100=w[100], n_scenarios=len(sims), source='pickle'))
        # -- failures ---------------------------------------------------------
        for key, scen in (('popp_failures_latency_optimal_specific', 'linkfail'),
                          ('pop_failures_latency_optimal_specific', 'sitefail')):
            diffs, wts, alld, allw, vol_over, vol_all, elems = [], [], [], [], 0.0, 0.0, set()
            for ri in sims:
                for f in (m[key][ri].get(sol) or []):
                    _diff, v, _ug, el, perf1, perf2 = f[:6]
                    if perf1 == NO_ROUTE_LATENCY:
                        continue            # optimal itself has no route: excluded (as the script does)
                    elems.add(el); vol_all += v
                    if perf2 != NO_ROUTE_LATENCY:
                        alld.append(perf1 - perf2); allw.append(v)
                    if perf2 == NO_ROUTE_LATENCY or perf2 >= CONGESTION_CAP_MS:
                        vol_over += v
                    else:
                        diffs.append(perf1 - perf2); wts.append(v)
            if vol_all <= 0:
                continue
            over = vol_over / vol_all
            w = _within(diffs, wts, over)
            rows.append(dict(scenario=scen, method=tok, gap=(-np.average(diffs, weights=wts) if diffs else float('nan')),
                             gapall=(-np.average(alld, weights=allw) if alld else float('nan')), overloaded=100.0 * over, within10=w[10], within50=w[50], within100=w[100],
                             n_scenarios=len(elems), source='pickle'))
    return rows


# ----------------------------------------------------------------------------
# fallback: the figure PDFs
# ----------------------------------------------------------------------------

FIGURES = {'steady': 'steady_state_latency_actual_deployment.pdf',
           'linkfail': 'link_failure_latency_actual_deployment.pdf',
           'sitefail': 'site_failure_latency_actual_deployment.pdf'}
COLOR_TO_METHOD = {(0.1, 0.1, 0.44): 'anycast', (1.0, 0.0, 0.0): 'unicast',
                   (0.0, 0.0, 0.0): 'painter', (1.0, 0.0, 1.0): 'sculptor'}


def stats_from_figures(fig_dir):
    import fitz   # PyMuPDF
    rows = []
    for scen, fn in FIGURES.items():
        p = os.path.join(fig_dir, fn)
        if not os.path.exists(p):
            continue
        page = fitz.open(p)[0]
        xt, yt = {}, {}
        for b in page.get_text('dict')['blocks']:
            for l in b.get('lines', []):
                for s in l['spans']:
                    t = s['text'].strip().replace('−', '-'); bb = s['bbox']
                    try:
                        v = float(t)
                    except ValueError:
                        continue
                    if t in ('-100', '-75', '-50', '-25', '0'):
                        xt[v] = (bb[0] + bb[2]) / 2
                    elif '.' in t and 0 <= v <= 1:
                        yt[v] = (bb[1] + bb[3]) / 2
        xs = sorted(xt); ax_, bx_ = np.polyfit([xt[v] for v in xs], xs, 1)
        ys = sorted(yt); ay_, by_ = np.polyfit([yt[v] for v in ys], ys, 1)
        curves, clip = {}, None
        for d in page.get_drawings():
            if len(d['items']) < 20 or d.get('color') is None:
                continue
            meth = COLOR_TO_METHOD.get(tuple(round(c, 2) for c in d['color']))
            if not meth:
                continue
            pts = np.array([q for it in d['items'] if it[0] == 'l' for q in ((it[1].x, it[1].y), (it[2].x, it[2].y))])
            data = np.c_[ax_ * pts[:, 0] + bx_, ay_ * pts[:, 1] + by_]
            curves[meth] = data
            clip = data[:, 0].min() if clip is None else min(clip, data[:, 0].min())
        for meth, data in curves.items():
            a = data[np.argsort(data[:, 0], kind='stable')]
            x, F = a[:, 0], np.maximum.accumulate(np.clip(a[:, 1], 0, 1))
            vis = x > clip + 0.5
            beyond = F[vis][0] if vis.any() else F[0]
            mass, prev = 0.0, None
            for xi, Fi in zip(x[vis], F[vis]):
                if prev is not None and Fi > prev[1]:
                    mass += 0.5 * (xi + prev[0]) * (Fi - prev[1])
                prev = (xi, Fi)
            top = F[x <= 0.5].max() if (x <= 0.5).any() else F.max()
            vm = top - beyond
            rows.append(dict(scenario=scen, method=meth, gap=(-mass / vm if vm > 1e-6 else float('nan')),
                             gapall='', overloaded=100.0 * beyond,
                             within10=100.0 * (1 - np.interp(-10.0, x, F)), within50=100.0 * (1 - np.interp(-50.0, x, F)),
                             within100=100.0 * (1 - np.interp(-100.0, x, F)), n_scenarios='', source='figure:' + fn))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pickle', default=DEFAULT_PICKLE)
    ap.add_argument('--figures', default=DEFAULT_FIGURES)
    ap.add_argument('--out', default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    if os.path.exists(a.pickle):
        rows = stats_from_pickle(a.pickle)
        print('[actual-deployment] {} rows from {}'.format(len(rows), a.pickle))
    else:
        rows = stats_from_figures(a.figures)
        print('[actual-deployment] pickle missing ({}); {} rows read from the figure PDFs (approximate)'.format(a.pickle, len(rows)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ('{:.3f}'.format(v) if isinstance(v, float) else v) for k, v in r.items()})
    print('[actual-deployment] wrote {}'.format(a.out))
    for r in rows:
        print('  {:<9s} {:<9s} gap {:6.2f} ms (all-traffic {:>6s})  overloaded {:5.1f}%  within10 {:5.1f}%  within50 {:5.1f}%  within100 {:5.1f}%'.format(
            r['scenario'], r['method'], r['gap'], ('{:.1f}'.format(r['gapall']) if isinstance(r['gapall'], float) else '-'),
            r['overloaded'], r['within10'], r['within50'], r['within100']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
