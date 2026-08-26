"""Normalization lab (Tom 2026-08-26: 'Try all these ideas').

Compares difficulty measures x divergence normalizations on the existing
grid store -- no reruns:

  difficulty:  D1 rel-gap        mean|GT-believed| / GT scale (original)
               D2 rank-fidelity  1 - Kendall tau(GT, believed) over the run
               D3 step-fidelity  1 - sign-agreement of per-step deltas
                                 (does belief move the way GT moves)
  divergence:  V1 OPP-scale      (Y-X) / median|OPP objective|  (original)
               V2 regret share   (Y-X) / (no_mc - full) per cell
                                 (fraction of the cell's ladder span)

Outputs: per-objective difficulty under each measure, the hypothesis
regression under each (D,V) combo, and a scatter for the D2/V2 combo.
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.ablation.objective_difficulty import (
    load, build_pairs, ols, RANK, OBJ_COLOR)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _kendall_tau(x, y):
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    if n < 3:
        return None
    c = d = 0
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        s = np.sign(dx) * np.sign(dy)
        c += int((s > 0).sum())
        d += int((s < 0).sum())
    tot = c + d
    return (c - d) / tot if tot else None


def _series_measures(store):
    """Per (obj, N, seed): D1/D2/D3 from the FULL rung's model_gap series."""
    out = {}
    for fn in glob.glob(os.path.join(store, '*', 'N*', 'seed_*_full.json')):
        d = json.load(open(fn))
        mg = d.get('model_gap') or {}
        ser = mg.get('series') or []
        if len(ser) < 4:
            continue
        obj = fn.split(os.sep)[-3]
        N = int(fn.split(os.sep)[-2][1:])
        seed = int(d.get('seed', -1))
        gt = np.array([r[1] for r in ser], dtype=float)
        bel = np.array([r[2] for r in ser], dtype=float)
        tau = _kendall_tau(gt, bel)
        dgt, dbel = np.diff(gt), np.diff(bel)
        m = (dgt != 0) & (dbel != 0)
        step = float((np.sign(dgt[m]) == np.sign(dbel[m])).mean()) \
            if m.sum() > 2 else None
        out[(obj, N, seed)] = {
            'D1': mg.get('mean_rel'),
            'D2': (1 - tau) if tau is not None else None,
            'D3': (1 - step) if step is not None else None,
            'n_pts': len(ser)}
    return out


def main():
    store = os.path.join(REPO, 'cache', 'ablation', 'grid_objdim')
    rows = load(store)
    pairs = build_pairs(rows)
    med = {}
    for obj in {p['obj'] for p in pairs}:
        dd = np.abs([q['divergence'] for q in pairs if q['obj'] == obj])
        med[obj] = max(float(np.median(dd)), 1e-9)
    pairs = [p for p in pairs if abs(p['divergence']) <= 50 * med[p['obj']]]
    meas = _series_measures(store)
    # V2 denominator: ladder span per cell (no_mc final - full final)
    cell_final = {}
    for r in rows:
        if r['final'] is not None:
            cell_final[(r['obj'], r['N'], r['seed'], r['rung'])] = r['final']
    span = {}
    for (obj, N, seed, rung), v in cell_final.items():
        if rung == 'no_mc':
            f = cell_final.get((obj, N, seed, 'full'))
            if f is not None and abs(v - f) > 1e-9:
                span[(obj, N, seed)] = v - f
    scale = {}
    for obj in med:
        o = [abs(r['opp_obj']) for r in rows
             if r['obj'] == obj and r.get('opp_obj')]
        scale[obj] = max(float(np.median(o)) if o else 1.0, 1e-9)

    # ---- difficulty table under each measure ----
    print('== difficulty per objective (mean over cells) ==')
    print('{:>22s} {:>10s} {:>12s} {:>12s}'.format(
        'objective', 'D1 relgap', 'D2 1-tau', 'D3 1-stepagr'))
    agg = {}
    for obj in sorted({k[0] for k in meas}):
        vs = [m for k, m in meas.items() if k[0] == obj]
        row = tuple(
            (float(np.mean([v[d] for v in vs if v[d] is not None]))
             if any(v[d] is not None for v in vs) else None)
            for d in ('D1', 'D2', 'D3'))
        agg[obj] = row
        print('{:>22s} {:>10s} {:>12s} {:>12s}'.format(
            obj, *[('{:.3f}'.format(x) if x is not None else '-')
                   for x in row]))

    # ---- regressions per (D, V) combo ----
    print('\n== hypothesis regression per combo: divergence ~ difficulty + '
          'rank_delta + log2N ==')
    for dkey in ('D1', 'D2', 'D3'):
        for vkey in ('V1', 'V2'):
            xs, ys, rds, lns = [], [], [], []
            for p in pairs:
                m = meas.get((p['obj'], p['N'], p['seed']))
                if not m or m.get(dkey) is None:
                    continue
                if vkey == 'V1':
                    v = p['divergence'] / scale[p['obj']]
                else:
                    sp = span.get((p['obj'], p['N'], p['seed']))
                    if sp is None:
                        continue
                    v = p['divergence'] / sp
                xs.append(m[dkey]); ys.append(v)
                rds.append(float(p['rank_delta']))
                lns.append(np.log2(float(p['N'])))
            if len(ys) < 20:
                print('  {}/{}: n={} (too few)'.format(dkey, vkey, len(ys)))
                continue
            # standardize difficulty within objective? keep raw here --
            # D2/D3 are already unit-free across objectives
            mres = ols(np.array(ys), [np.array(xs), np.array(rds),
                                      np.array(lns)],
                       ['difficulty', 'rank_delta', 'log2N'])
            b, t = mres['beta'][1], mres['t'][1]
            print('  {}/{}: n={:3d} R2={:.3f}  difficulty {:+8.3f} '
                  '(t {:+5.2f})   rank_delta t {:+5.2f}'.format(
                      dkey, vkey, mres['n'], mres['r2'], b, t,
                      mres['t'][2]))

    # ---- scatter for D2/V2 ----
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for obj in OBJ_COLOR:
        px, py = [], []
        for p in pairs:
            if p['obj'] != obj:
                continue
            m = meas.get((p['obj'], p['N'], p['seed']))
            sp = span.get((p['obj'], p['N'], p['seed']))
            if not m or m.get('D2') is None or sp is None:
                continue
            px.append(m['D2']); py.append(p['divergence'] / sp)
        if px:
            ax.scatter(px, py, color=OBJ_COLOR[obj], label=obj, alpha=.7,
                       s=18)
    ax.set_xlabel('difficulty D2 (1 - Kendall tau of believed vs GT)')
    ax.set_ylabel('divergence V2 (fraction of ladder span)')
    ax.grid(alpha=.25); ax.legend(fontsize=7)
    ax.set_title('rank-fidelity difficulty vs regret-share divergence',
                 fontsize=10)
    out = os.path.join(REPO, 'figures', 'dashboards', 'ablation_scout',
                       'ablation_scout_difficulty_d2v2.png')
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print('\nwrote', out)


if __name__ == '__main__':
    main()
