"""Render the CANONICAL paper-table CSVs to LaTeX.

    python evaluations/emit_paper_table_tex.py

The canonical merged tables live in figures/paper_table/*.csv -- they are
the single source that absorbs every healing/re-eval merge (the per-run
emits from generate_paper_table cover only that run's objectives). This
renders them to booktabs .tex alongside, with the same display rules as
the dash renderer: 4 decimals for Wgt avg site cost, 2 elsewhere,
best-in-column bolded per the DIRECTION row (One-per-peering excluded as
the reference).

KNOWN LIMIT (iteration item): the canonical CSVs carry means only, so
these .tex tables have no +/-std; per-run emitted .tex keeps stds for
its objectives.
"""
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_DIR = os.path.join(_REPO, 'figures', 'paper_table')
REFERENCE = {'One-per-peering'}


def _prec(label):
    return 4 if 'Wgt avg site cost' in label else 2


def render(csv_name):
    path = os.path.join(TABLE_DIR, csv_name)
    rows = list(csv.reader(open(path)))
    header, direction = rows[0], None
    body = []
    for r in rows[1:]:
        if r and r[0] == 'DIRECTION':
            direction = r
        elif r:
            body.append(r)
    ncol = len(header)

    # supersection groups from 'Group|Metric' labels
    groups, subs = [], []
    for lab in header[1:]:
        g, _, m = lab.partition('|')
        subs.append(m or g)
        if groups and groups[-1][0] == g:
            groups[-1][1] += 1
        else:
            groups.append([g, 1])

    # best-in-column per DIRECTION, reference rows excluded
    best = {}
    for j in range(1, ncol):
        want = direction[j].strip() if direction and j < len(direction) else ''
        cands = []
        for r in body:
            if r[0] in REFERENCE:
                continue
            try:
                cands.append((float(r[j]), r[0]))
            except (ValueError, IndexError):
                pass
        if cands and want in ('<', '>'):
            best[j] = (min if want == '<' else max)(cands)[1]

    out = ['% rendered by evaluations/emit_paper_table_tex.py from the',
           '% canonical merged {} (means only; per-run emits keep stds)'
           .format(csv_name),
           '\\begin{tabular}{l' + 'r' * (ncol - 1) + '}', '\\toprule']
    out.append(' & ' + ' & '.join(
        '\\multicolumn{{{}}}{{c}}{{{}}}'.format(n, g)
        for g, n in groups) + ' \\\\')
    col, rules = 2, []
    for _g, n in groups:
        rules.append('\\cmidrule(lr){{{}-{}}}'.format(col, col + n - 1))
        col += n
    out.append(''.join(rules))
    out.append('Method & ' + ' & '.join(subs) + ' \\\\')
    out.append('\\midrule')
    for r in body:
        cells = [r[0]]
        for j in range(1, ncol):
            v = r[j] if j < len(r) else ''
            try:
                disp = '{:.{p}f}'.format(float(v), p=_prec(header[j]))
            except ValueError:
                disp = v or '-'
            if best.get(j) == r[0]:
                disp = '\\textbf{{{}}}'.format(disp)
            cells.append(disp)
        out.append(' & '.join(cells) + ' \\\\')
    out += ['\\bottomrule', '\\end{tabular}', '']
    tex = path[:-4] + '.tex'
    open(tex, 'w').write('\n'.join(out))
    print('wrote {}'.format(tex))
    return tex


def main():
    for name in ('paper_table.csv', 'paper_table_key.csv'):
        if os.path.exists(os.path.join(TABLE_DIR, name)):
            render(name)
        else:
            print('missing {}'.format(name))
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
