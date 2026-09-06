"""Recover per-iteration GT-objective series from convergence_over_iterations
vector PDFs (Tom 2026-09-03: "could we get that for all runs all iters?").

The solver's make_plots writes matplotlib VECTOR figures: the GT Objective
panel's curve is an embedded path, so the numbers are recoverable exactly --
no OCR. Calibration comes from the panel's own tick labels; the in-panel
'One per Peering' reference hline is extracted too when present (it anchors
the objective scale and, for campaign figures, identifies the deployment).

    python evaluations/extract_convergence_from_figures.py \
        --roots <dir> [<dir> ...] --out corpus.json

Scans each root recursively for convergence_over_iterations.pdf, emits
{run_dir_name: {path, iters, objs, opp, n_points}}. Figures whose layout
defeats calibration are recorded under 'failed' with a reason -- honest
coverage, no silent drops. First used to recover the 20 paper actual-10
full/L6 trainings (validated: assignment by (OPP, final) matched ladder
anchors with ~0.03 objective-unit error, epoch order == sim order).
"""
import argparse
import glob
import json
import os

import fitz
import numpy as np

BLUE = (0.12156862765550613, 0.46666666865348816, 0.7058823704719543)


def extract_one(pdf):
    pg = fitz.open(pdf)[0]
    nums = []
    for w in pg.get_text('words'):
        try:
            v = float(w[4])
        except ValueError:
            continue
        nums.append((v, fitz.Rect(w[:4])))
    # GT Objective panel = top-right in the fixed make_plots grid
    xt = sorted([(v, (r.x0 + r.x1) / 2) for v, r in nums
                 if 195 < r.y0 < 235 and 375 < r.x0 < 640],
                key=lambda t: t[1])
    yt = sorted([(v, (r.y0 + r.y1) / 2) for v, r in nums
                 if 330 < r.x1 < 395 and 5 < r.y0 < 190],
                key=lambda t: t[1])
    if len(xt) < 2 or len(yt) < 2:
        return None, 'ticks-not-found'
    (xv0, xp0), (xv1, xp1) = xt[0], xt[-1]
    (yv0, yp0), (yv1, yp1) = yt[0], yt[-1]
    if xp1 == xp0 or yp1 == yp0:
        return None, 'degenerate-ticks'
    to_it = lambda x: xv0 + (x - xp0) / (xp1 - xp0) * (xv1 - xv0)
    to_ob = lambda y: yv0 + (y - yp0) / (yp1 - yp0) * (yv1 - yv0)
    ax = fitz.Rect(375, 5, 640, 200)
    curve, opp_y = None, None
    for d in pg.get_drawings():
        col = d.get('color')
        r = d['rect']
        pts = []
        for it in d['items']:
            if it[0] == 'l':
                pts.append((it[1].x, it[1].y))
                pts.append((it[2].x, it[2].y))
        if not pts:
            continue
        inpanel = (ax.x0 - 6 <= r.x0 and r.x1 <= ax.x1 + 6
                   and r.y0 >= ax.y0 - 6 and r.y1 <= ax.y1 + 16)
        if (col and all(abs(a - b) < 1e-3 for a, b in zip(col, BLUE))
                and len(pts) > 40 and inpanel
                and (curve is None or len(pts) > len(curve))):
            curve = pts
        if col == (0.0, 0.0, 0.0):
            for it in d['items']:
                if it[0] != 'l':
                    continue
                p1, p2 = it[1], it[2]
                if (abs(p1.y - p2.y) < 0.5 and abs(p1.x - p2.x) > 100
                        and 375 < p1.x < 640 and 180 < p1.y < 202):
                    opp_y = p1.y
    if curve is None:
        return None, 'curve-not-found'
    arr = sorted(set(curve))
    its = np.array([to_it(x) for x, _ in arr])
    obs = np.array([to_ob(y) for _, y in arr])
    o = np.argsort(its)
    return {'iters': [round(float(v), 2) for v in its[o]],
            'objs': [round(float(v), 4) for v in obs[o]],
            'opp': (round(float(to_ob(opp_y)), 4)
                    if opp_y is not None else None)}, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    corpus, failed = {}, {}
    pdfs = []
    for root in a.roots:
        pdfs.extend(glob.glob(os.path.join(
            root, '**', 'convergence_over_iterations.pdf'), recursive=True))
    print('{} figures found'.format(len(pdfs)))
    for pdf in sorted(pdfs):
        name = os.path.basename(os.path.dirname(pdf))
        try:
            res, err = extract_one(pdf)
        except Exception as e:
            res, err = None, 'exception: {}'.format(e)
        if res is None:
            failed[name] = {'path': pdf, 'reason': err}
        else:
            res['path'] = pdf
            corpus[name] = res
    json.dump({'series': corpus, 'failed': failed},
              open(a.out, 'w'))
    print('extracted {} | failed {} -> {}'.format(
        len(corpus), len(failed), a.out))
    for name, f in list(failed.items())[:10]:
        print('  FAIL {}: {}'.format(name, f['reason']))


if __name__ == '__main__':
    main()
