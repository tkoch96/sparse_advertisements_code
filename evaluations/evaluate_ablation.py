"""THE ablation evaluation (Tom 2026-09-08): given a finished (or running)
ladder directory of cell JSONs, emit

  1. the ladder table -- per rung, the mean trusted objective over
     deployments and the % of the painter->OPP gap it closes (on the means
     and as the mean of per-deployment %), written to ladder_summary.{json,csv};
  2. the same percentage OVER ITERATIONS, per rung, from each cell's
     per-iteration ground-truth objective (`gt_objective_series`), written
     to pct_gap_closed_over_iterations.{pdf,json}.

Inputs are the cell JSONs the ladder writes (seed_<s>_<rung>.json): each
carries `repo_objective` (the cell's final ground-truth objective),
`opp_objective` (one-per-peering under the same evaluator; identical across
the cells of one deployment now that seeded deployments are deterministic)
and `gt_objective_series` ([[iter, objective], ...]; painter is one-shot and
has none). NO metric logic lives anywhere else: cdf_fork.main calls run()
after its CDF figure, and run_ablation_cdf.py reaches it through cdf_fork.

    python -m evaluations.evaluate_ablation --in-dir cache/ablation/<study>/Nprefixes
    python -m evaluations.evaluate_ablation --in-dir ... --prelim   # running study

--prelim drops the 'rescored' gate: repo_objective/opp_objective are written
by the cell itself at completion (the rescore only adds latency/failure
columns), so preliminary numbers equal the final ones for finished cells.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# rung order / labels / colors are the ladder's single source of truth
from experiments.ablation.cdf_fork import LADDER  # noqa: E402


def _load_cells(in_dir, require_rescored=True):
    """{(seed, rung): json} for every scored cell in in_dir."""
    cells = {}
    for fn in sorted(glob.glob(os.path.join(in_dir, 'seed_*_*.json'))):
        with open(fn) as f:
            r = json.load(f)
        if (require_rescored and not r.get('rescored')) \
                or r.get('repo_objective') is None:
            continue
        cells[(int(r['seed']), r['rung'])] = r
    return cells


def _anchors(cells):
    """Per seed: painter objective, OPP anchor (mean of the seed's per-cell
    OPP values -- identical when the world is pinned; the spread is reported
    as the anchor's noise floor), and the gap."""
    painter, opp_cells = {}, {}
    for (s, rung), r in cells.items():
        if rung == 'painter':
            painter[s] = float(r['repo_objective'])
        if r.get('opp_objective') is not None:
            opp_cells.setdefault(s, []).append(float(r['opp_objective']))
    opp = {s: float(np.mean(v)) for s, v in opp_cells.items()}
    spread = {s: float(max(v) - min(v)) for s, v in opp_cells.items()}
    return painter, opp, spread


def ladder_summary(in_dir, require_rescored=True):
    """THE headline metric (Tom 2026-09-08): per rung, the mean trusted
    objective over deployments and the cumulative percentage of the
    painter->OPP gap it closes, computed on the MEANS
    (100 * (mean_painter - mean_rung) / (mean_painter - mean_OPP)), plus
    the increment over the previous rung in capability order, the mean of
    the per-deployment percentages (scale-free companion: raw objectives
    mix per-deployment scales) and the per-deployment percentages."""
    cells = _load_cells(in_dir, require_rescored)
    by_rung = {}
    for (s, rung), r in cells.items():
        by_rung.setdefault(rung, {})[s] = float(r['repo_objective'])
    painter, opp, opp_spread = _anchors(cells)
    if 'painter' not in by_rung or not opp:
        return None, ''
    rungs = [r for r, _, _ in LADDER if r in by_rung]
    seeds = sorted(s for s in set(painter) & set(opp)
                   if all(s in by_rung[r] for r in rungs))
    if not seeds:
        return None, ''
    mean = {r: float(np.mean([by_rung[r][s] for s in seeds])) for r in rungs}
    mean_opp = float(np.mean([opp[s] for s in seeds]))
    gap = mean['painter'] - mean_opp
    rows, prev = [], 0.0
    for r in rungs:
        cum = 100.0 * (mean['painter'] - mean[r]) / gap if gap > 0 else float('nan')
        per_seed = {}
        for s in seeds:
            g = painter[s] - opp[s]
            per_seed[s] = (100.0 * (painter[s] - by_rung[r][s]) / g
                           if g > 0 else float('nan'))
        finite = [v for v in per_seed.values() if np.isfinite(v)]
        rows.append({'rung': r, 'mean_objective': mean[r],
                     'mean_minus_opp': mean[r] - mean_opp,
                     'pct_gap_closed_on_means': cum,
                     'increment_pct': cum - prev,
                     'mean_of_per_seed_pct': (float(np.mean(finite))
                                              if finite else float('nan')),
                     'n_seeds_with_positive_gap': len(finite),
                     'pct_gap_closed_per_seed': per_seed})
        prev = cum
    summary = {'seeds': seeds, 'n_deployments': len(seeds),
               'mean_opp_objective': mean_opp,
               'opp_anchor': "mean of the deployment's per-cell OPP values",
               'opp_cell_spread_per_seed': {s: opp_spread[s] for s in seeds},
               'painter_to_opp_gap_on_means': gap,
               'preliminary': not require_rescored, 'rungs': rows}
    hdr = '{:<14}{:>10}{:>12}{:>12}{:>10}{:>14}'.format(
        'rung', 'mean obj', 'mean-OPP', '% gap (cum)', 'incr', 'mean seed-%')
    lines = ['LADDER SUMMARY{} (means over {} deployments; % of painter->OPP '
             'gap closed on the means; OPP mean {:.3f}):'.format(
                 ' [PRELIMINARY, un-rescored cells]' if not require_rescored else '',
                 len(seeds), mean_opp), hdr, '-' * len(hdr)]
    for row in rows:
        lines.append('{:<14}{:>10.3f}{:>12.3f}{:>11.1f}%{:>+9.1f}{:>13.1f}%'.format(
            row['rung'], row['mean_objective'], row['mean_minus_opp'],
            row['pct_gap_closed_on_means'], row['increment_pct'],
            row['mean_of_per_seed_pct']))
    lines.append('{:<14}{:>10.3f}{:>12.3f}{:>11.1f}%'.format(
        'OPP', mean_opp, 0.0, 100.0))
    lines.append('per-deployment % gap closed: ' + '; '.join(
        '{}: {}'.format(row['rung'], ', '.join(
            '{:.0f}'.format(v) for s, v in sorted(row['pct_gap_closed_per_seed'].items())))
        for row in rows if row['rung'] != 'painter'))
    lines.append("OPP anchor = mean of each deployment's per-cell OPP; "
                 'cell-to-cell OPP spread per deployment: ' + ', '.join(
                     '{}: {:.3f}'.format(s, opp_spread[s]) for s in seeds) +
                 ' (painter->OPP gap on means {:.3f})'.format(gap))
    return summary, '\n'.join(lines)


def write_summary_files(summary, out_dir):
    with open(os.path.join(out_dir, 'ladder_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(out_dir, 'ladder_summary.csv'), 'w') as f:
        f.write('rung,mean_objective,mean_minus_opp,pct_gap_closed_on_means,'
                'increment_pct,mean_of_per_seed_pct\n')
        for row in summary['rungs']:
            f.write('{},{:.6f},{:.6f},{:.3f},{:.3f},{:.3f}\n'.format(
                row['rung'], row['mean_objective'], row['mean_minus_opp'],
                row['pct_gap_closed_on_means'], row['increment_pct'],
                row['mean_of_per_seed_pct']))
        f.write('OPP,{:.6f},0,100,,100\n'.format(summary['mean_opp_objective']))


def pct_over_iterations(in_dir, require_rescored=True):
    """Per rung, the % of the painter->OPP gap closed at every iteration:
    per deployment 100*(painter_s - gt_s(it))/(painter_s - OPP_s) from the
    cell's gt_objective_series (held at its final value past the cell's
    last iteration, which is what an early-stopped cell delivers), then
    (a) the mean of the per-deployment % and (b) the means-based %
    100*(mean painter - mean gt(it))/(mean painter - mean OPP)."""
    cells = _load_cells(in_dir, require_rescored)
    painter, opp, _ = _anchors(cells)
    series = {}   # rung -> seed -> np.array indexed by iteration
    for (s, rung), r in cells.items():
        if rung == 'painter' or s not in painter or s not in opp:
            continue
        ser = r.get('gt_objective_series') or []
        if not ser:
            continue
        its = [int(p[0]) for p in ser]
        vals = [float(p[1]) for p in ser]
        arr = np.full(max(its) + 1, np.nan)
        arr[its] = vals
        # forward-fill gaps (series are recorded per iteration; be safe)
        last = vals[0]
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                arr[i] = last
            else:
                last = arr[i]
        series.setdefault(rung, {})[s] = arr
    if not series:
        return None
    rungs = [r for r, _, _ in LADDER if r in series]
    seeds = sorted(s for s in set(painter) & set(opp)
                   if all(s in series[r] for r in rungs) and painter[s] - opp[s] > 0)
    if not seeds:
        return None
    n_it = max(len(series[r][s]) for r in rungs for s in seeds)
    out = {'iterations': list(range(n_it)), 'seeds': seeds, 'rungs': {},
           'preliminary': not require_rescored}
    mean_painter = float(np.mean([painter[s] for s in seeds]))
    mean_opp = float(np.mean([opp[s] for s in seeds]))
    for r in rungs:
        held = np.array([np.concatenate([series[r][s],
                                         np.full(n_it - len(series[r][s]),
                                                 series[r][s][-1])])
                         for s in seeds])                      # seeds x n_it
        per_seed = np.array([100.0 * (painter[s] - held[i]) / (painter[s] - opp[s])
                             for i, s in enumerate(seeds)])
        means_based = 100.0 * (mean_painter - held.mean(axis=0)) / (mean_painter - mean_opp)
        out['rungs'][r] = {
            'mean_of_per_seed_pct': per_seed.mean(axis=0).tolist(),
            'means_based_pct': means_based.tolist(),
            'per_seed_pct': {s: per_seed[i].tolist() for i, s in enumerate(seeds)},
            'n_seeds': len(seeds)}
    return out


def plot_pct_over_iterations(data, out_pdf):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    color = {r: c for r, _, c in LADDER}
    label = {r: l for r, l, _ in LADDER}
    its = data['iterations']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key, title in ((axes[0], 'mean_of_per_seed_pct',
                            'mean of per-deployment %'),
                           (axes[1], 'means_based_pct',
                            '% on the means of the objectives')):
        for r, d in data['rungs'].items():
            ax.plot(its, d[key], color=color.get(r), label=label.get(r, r), lw=1.6)
        ax.axhline(0, color=color['painter'], ls='--', lw=1, label='painter')
        ax.axhline(100, color='k', ls=':', lw=1, label='one-per-peering (OPP)')
        ax.set_title('{} ({} deployments{})'.format(
            title, len(data['seeds']),
            ', PRELIMINARY' if data.get('preliminary') else ''), fontsize=10)
        ax.set_xlabel('iteration')
        ax.grid(True, alpha=.3)
    axes[0].set_ylabel('% of painter -> OPP gap closed')
    axes[0].legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def run(in_dir, out_dir=None, require_rescored=True, plot=True):
    """Table + files + over-iterations figure. Returns (summary, curves)."""
    out_dir = out_dir or in_dir
    os.makedirs(out_dir, exist_ok=True)
    summary, table = ladder_summary(in_dir, require_rescored)
    if summary is None:
        print('[evaluate_ablation] no complete painter/OPP deployment in {} '
              '-> no ladder summary'.format(in_dir), flush=True)
        return None, None
    print('\n' + table + '\n', flush=True)
    write_summary_files(summary, out_dir)
    print('[evaluate_ablation] wrote {}/ladder_summary.{{json,csv}}'.format(out_dir),
          flush=True)
    curves = pct_over_iterations(in_dir, require_rescored) if plot else None
    if curves is not None:
        with open(os.path.join(out_dir, 'pct_gap_closed_over_iterations.json'), 'w') as f:
            json.dump(curves, f)
        pdf = os.path.join(out_dir, 'pct_gap_closed_over_iterations.pdf')
        plot_pct_over_iterations(curves, pdf)
        print('[evaluate_ablation] wrote {} (+ .json; {} rungs x {} iterations x '
              '{} deployments)'.format(pdf, len(curves['rungs']),
                                       len(curves['iterations']),
                                       len(curves['seeds'])), flush=True)
    elif plot:
        print('[evaluate_ablation] no gt_objective_series -> no over-iterations '
              'figure', flush=True)
    return summary, curves


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in-dir', required=True, help='ladder dir of seed_<s>_<rung>.json')
    ap.add_argument('--out-dir', default=None, help='default: --in-dir')
    ap.add_argument('--prelim', action='store_true',
                    help='running study: drop the rescored gate')
    ap.add_argument('--no-plot', action='store_true')
    a = ap.parse_args()
    summary, _ = run(a.in_dir, a.out_dir, require_rescored=not a.prelim,
                     plot=not a.no_plot)
    return 0 if summary is not None else 1


if __name__ == '__main__':
    sys.exit(main())
