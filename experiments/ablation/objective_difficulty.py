"""Objective difficulty vs ladder-arm divergence (Tom 2026-08-26).

Hypothesis: as an objective gets harder (larger ground-truth-minus-
believed gap over training), the performance divergence between ladder
arms widens -- especially top pairs (full vs expl_none, ~L6-L5) versus
mid pairs (no_direction vs no_memory_dir, ~L4-L3).

    python -m experiments.ablation.objective_difficulty \
        [--store cache/ablation/grid_objdim]

Inputs: the run_ablation_grid store (seed_<s>_<rung>.json with the
model_gap block persisted by run_fork_ladder). Outputs: scatter figures
into figures/dashboards/ablation_scout/, an OLS table (printed + JSON
beside the store). Runs/logs stay untouched -- everything here re-derives
from the compact JSONs.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
RANK = {'no_mc': 1, 'no_memory': 2, 'no_memory_dir': 3,
        'no_direction': 4, 'expl_none': 5, 'full': 6}
OBJ_COLOR = {'avg_latency': '#c026a8', 'per_site_cost': '#c9862b',
             'max_util': '#2f9e6e', 'frac_beyond_optimal': '#4a6fa5',
             'joint_priority': '#8b5aa8'}
HIGHLIGHT_PAIRS = [('full', 'expl_none'), ('no_direction', 'no_memory_dir')]


def load(store):
    rows = []
    for fn in glob.glob(os.path.join(store, '*', 'N*', 'seed_*_*.json')):
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        rung = d.get('rung')
        if rung not in RANK:
            continue
        parts = fn.split(os.sep)
        obj, Ndir = parts[-3], parts[-2]
        final = d.get('diff_vs_opp')
        if final is None and d.get('repo_objective') is not None \
                and d.get('opp_objective') is not None:
            final = d['repo_objective'] - d['opp_objective']
        mg = d.get('model_gap') or {}
        rows.append({'obj': obj, 'N': int(Ndir[1:]),
                     'seed': int(d.get('seed', -1)), 'rung': rung,
                     'final': final,
                     'opp_obj': d.get('opp_objective'),
                     'diff_abs': mg.get('mean_abs'),
                     'diff_rel': mg.get('mean_rel'),
                     'gap_pts': mg.get('n_pts', 0)})
    return rows


def build_pairs(rows):
    """(difficulty of the cell's FULL run, divergence) per ordered rung
    pair within each (obj, N, seed) cell."""
    cell = {}
    for r in rows:
        cell.setdefault((r['obj'], r['N'], r['seed']), {})[r['rung']] = r
    out = []
    for (obj, N, seed), by_rung in cell.items():
        ref = by_rung.get('full')
        difficulty = (ref or {}).get('diff_rel')
        if difficulty is None:
            continue
        rungs = [r for r in by_rung if by_rung[r]['final'] is not None]
        for hi in rungs:
            for lo in rungs:
                if RANK[hi] <= RANK[lo]:
                    continue
                out.append({
                    'obj': obj, 'N': N, 'seed': seed,
                    'pair': (hi, lo),
                    'rank_delta': RANK[hi] - RANK[lo],
                    'difficulty': difficulty,
                    'divergence': by_rung[lo]['final']
                                  - by_rung[hi]['final']})
    return out


def ols(y, X, names):
    X = np.column_stack([np.ones(len(y))] + X)
    names = ['const'] + names
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tstat = beta / np.where(se > 0, se, np.nan)
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-12
    r2 = 1 - float(resid @ resid) / ss_tot
    return {'names': names, 'beta': beta.tolist(), 'se': se.tolist(),
            't': tstat.tolist(), 'r2': r2, 'n': len(y)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--store', default=os.path.join(
        REPO, 'cache', 'ablation', 'grid_objdim_maxhard'))
    a = ap.parse_args()
    rows = load(a.store)
    pairs = build_pairs(rows)
    print('[difficulty] {} runs -> {} pair points'.format(
        len(rows), len(pairs)))
    if not pairs:
        return 1
    # SENTINEL GUARD: a handful of finals carry no-route-scale values
    # (|div| ~ 2e4 in joint_priority/max_util); drop |div| > 50x the
    # objective's median |div| and report.
    import collections
    med = {}
    for obj in {p['obj'] for p in pairs}:
        d = np.abs([q['divergence'] for q in pairs if q['obj'] == obj])
        med[obj] = max(float(np.median(d)), 1e-9)
    kept = [p for p in pairs
            if abs(p['divergence']) <= 50 * med[p['obj']]]
    print('[difficulty] sentinel-filtered {} of {} pair points'.format(
        len(pairs) - len(kept), len(pairs)))
    pairs = kept
    # UNIT NORMALIZATION (objectives differ 50x in scale/range):
    #  - divergence -> relative: / objective's median |OPP objective|
    #  - difficulty -> z-scored within objective
    scale = {}
    for obj in med:
        o = [abs(r['opp_obj']) for r in rows
             if r['obj'] == obj and r.get('opp_obj')]
        scale[obj] = max(float(np.median(o)) if o else 1.0, 1e-9)
    dstat = {}
    for obj in med:
        f = np.array([q['difficulty'] for q in pairs if q['obj'] == obj])
        dstat[obj] = (float(f.mean()), max(float(f.std()), 1e-9))
    for p in pairs:
        p['div_rel'] = p['divergence'] / scale[p['obj']]
        m, sd = dstat[p['obj']]
        p['diff_z'] = (p['difficulty'] - m) / sd
    outdir = os.path.join(REPO, 'figures', 'dashboards', 'ablation_scout')
    os.makedirs(outdir, exist_ok=True)

    # -- scatter: highlighted pairs, one panel each + all-pairs panel --
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, pr in zip(axes[:2], HIGHLIGHT_PAIRS):
        sub = [p for p in pairs if p['pair'] == pr]
        for obj in OBJ_COLOR:
            ss = [p for p in sub if p['obj'] == obj]
            if not ss:
                continue
            ax.scatter([p['diff_z'] for p in ss],
                       [p['div_rel'] for p in ss],
                       s=[14 + 3 * p['N'] for p in ss],
                       color=OBJ_COLOR[obj], label=obj, alpha=.75)
        if sub:
            x = np.array([p['diff_z'] for p in sub])
            y = np.array([p['div_rel'] for p in sub])
            if len(sub) > 2 and x.std() > 0:
                b = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 20)
                ax.plot(xs, np.polyval(b, xs), 'k--', lw=1,
                        label='slope {:.2f}'.format(b[0]))
        ax.set_title('{} - {}'.format(*pr), fontsize=9)
        ax.set_xlabel('difficulty (z within objective)')
        ax.set_ylabel('relative divergence\n(worse minus better, / OPP scale)')
        ax.grid(alpha=.25)
        ax.legend(fontsize=6)
    ax = axes[2]
    sc = ax.scatter([p['diff_z'] for p in pairs],
                    [p['div_rel'] for p in pairs],
                    c=[p['rank_delta'] for p in pairs], cmap='viridis',
                    s=[10 + 2.5 * p['N'] for p in pairs], alpha=.7)
    fig.colorbar(sc, ax=ax, label='rank delta')
    ax.set_title('all pairs (color = ladder distance)', fontsize=9)
    ax.set_xlabel('difficulty')
    ax.grid(alpha=.25)
    fig.tight_layout()
    f1 = os.path.join(outdir, 'ablation_scout_difficulty_scatter.png')
    fig.savefig(f1, dpi=130)
    plt.close(fig)

    # -- regressions --
    y = np.array([p['div_rel'] for p in pairs], dtype=float)
    diff = np.array([p['diff_z'] for p in pairs], dtype=float)
    rd = np.array([p['rank_delta'] for p in pairs], dtype=float)
    lN = np.log2(np.array([p['N'] for p in pairs], dtype=float))
    models = {
        'divergence ~ difficulty': ols(y, [diff], ['difficulty']),
        'divergence ~ difficulty + rank_delta + log2N':
            ols(y, [diff, rd, lN], ['difficulty', 'rank_delta', 'log2N']),
        'divergence ~ difficulty * rank_delta + log2N':
            ols(y, [diff, rd, diff * rd, lN],
                ['difficulty', 'rank_delta', 'difficulty:rank_delta',
                 'log2N']),
    }
    # CROSS-OBJECTIVE level (the hypothesis proper): per-objective
    # median difficulty vs median relative divergence
    print('\n== cross-objective (one point per objective) ==')
    xo, yo = [], []
    for obj in sorted(med):
        sub = [p for p in pairs if p['obj'] == obj]
        xo.append(float(np.median([p['difficulty'] for p in sub])))
        yo.append(float(np.median([abs(p['div_rel']) for p in sub])))
        print('   {:>20s} difficulty {:7.3f}  median |rel divergence| '
              '{:7.4f}'.format(obj, xo[-1], yo[-1]))
    if len(xo) > 2:
        _r = np.corrcoef(xo, yo)[0, 1]
        _rs = np.corrcoef(np.argsort(np.argsort(xo)),
                          np.argsort(np.argsort(yo)))[0, 1]
        print('   pearson r = {:+.3f}   spearman rho = {:+.3f}   (n={})'
              .format(_r, _rs, len(xo)))

    per_pair_slopes = {}
    for pr in sorted({p['pair'] for p in pairs}):
        sub = [p for p in pairs if p['pair'] == pr]
        x = np.array([p['diff_z'] for p in sub])
        yy = np.array([p['div_rel'] for p in sub])
        if len(sub) > 2 and x.std() > 0:
            m = ols(yy, [x], ['difficulty'])
            per_pair_slopes['{}-{}'.format(*pr)] = {
                'slope': m['beta'][1], 't': m['t'][1],
                'r2': m['r2'], 'n': m['n']}
    print('\n== OLS ==')
    for label, m in models.items():
        print('\n{}  (n={}, R2={:.3f})'.format(label, m['n'], m['r2']))
        for nm, b, se, t in zip(m['names'], m['beta'], m['se'], m['t']):
            print('   {:26s} {:+9.4f}  (se {:7.4f}, t {:+6.2f})'.format(
                nm, b, se, t))
    print('\n== per-pair slope of divergence on difficulty ==')
    for k, v in sorted(per_pair_slopes.items(),
                       key=lambda kv: -abs(kv[1]['t'])):
        print('   {:28s} slope {:+8.3f}  t {:+6.2f}  R2 {:.2f}  n {}'
              .format(k, v['slope'], v['t'], v['r2'], v['n']))
    out_json = a.store.rstrip('/') + '_difficulty_analysis.json'
    with open(out_json, 'w') as fh:
        json.dump({'models': models, 'per_pair': per_pair_slopes,
                   'n_pairs': len(pairs)}, fh, indent=1)
    print('\nwrote {} and {}'.format(f1, out_json))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
