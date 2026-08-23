"""Paper-table tab: renders generate_paper_table.py output on the dash.

Tom (2026-08-22, AFK): "put it on the dash" -- the methods x metrics
tables should be readable from a phone via the public dashboard rather
than only in a terminal. This renders every paper_table.csv the table
generator has written (figures/paper_table*/), best-in-column bolded
with the same direction rules as the generator, plus file mtime so a
stale table is visibly stale.

No new pipeline: generate_paper_table.py writes the CSVs; this just shows
them on the next refresh cycle.
"""
import csv
import html
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# direction per column label prefix ('<' lower is better). Mirrors
# generate_paper_table.py COLUMNS; a label not listed defaults to
# '<' with no bolding confidence lost (worst case: no bold).
_HIGHER_BETTER = ('PoPP-fail benefit', 'PoP-fail benefit', 'Sticky',
                  'Frac within', 'Priority')
# opp bounds the achievable; it renders muted as a reference row and never
# takes the green (Tom 2026-08-22)
_REFERENCE = ('One-per-peering',)


def _render_csv(path):
    try:
        rows = list(csv.reader(open(path)))
    except Exception as e:
        return '<p class="note">unreadable {}: {}</p>'.format(
            html.escape(path), html.escape(str(e)))
    if len(rows) < 2:
        return '<p class="note">empty table: {}</p>'.format(html.escape(path))
    header, body = rows[0], rows[1:]
    # explicit DIRECTION row (written by table_generate since 2026-08-23);
    # falling back to the label heuristic only for old CSVs
    dirs = None
    if body and body[0][0] == 'DIRECTION':
        dirs = body[0]
        body = body[1:]
    ncol = len(header)
    best = [None] * ncol
    for j in range(1, ncol):
        vals = []
        for r in body:
            if r[0] in _REFERENCE:
                continue
            try:
                vals.append((float(r[j]), r[0]))
            except (ValueError, IndexError):
                pass
        if len(vals) > 1:
            if dirs is not None and j < len(dirs):
                hi = dirs[j].strip() == '>'
            else:
                sub = header[j].split('|')[-1]
                hi = any(sub.startswith(p) for p in _HIGHER_BETTER)
            win_v, win_n = (max if hi else min)(vals)
            # ties go to SCULPTOR (matches the emitter's rule)
            for v, n in vals:
                if n == 'SCULPTOR' and v == win_v:
                    win_n = 'SCULPTOR'
            best[j] = win_n
    age_min = (time.time() - os.path.getmtime(path)) / 60
    out = ['<p class="note">{} &mdash; written {:.0f} min ago</p>'.format(
        html.escape(os.path.relpath(path, REPO)), age_min)]
    # headers may be 'Group|Sub' (supersection layout, 2026-08-23):
    # render a colspan group row above the sub row
    has_groups = any('|' in h for h in header[1:])
    out.append('<div class="wrap"><table><thead>')
    if has_groups:
        grp = []
        for h in header[1:]:
            g = h.split('|')[0]
            if grp and grp[-1][0] == g:
                grp[-1][1] += 1
            else:
                grp.append([g, 1])
        out.append('<tr><th></th>' + ''.join(
            '<th colspan="{}" style="text-align:center;border-bottom:'
            '1.5px solid var(--fg)">{}</th>'.format(n, html.escape(g))
            for g, n in grp) + '</tr>')
        subs = [header[0]] + [h.split('|', 1)[1] for h in header[1:]]
    else:
        subs = header
    out.append('<tr>' + ''.join('<th>{}</th>'.format(html.escape(h))
                                for h in subs) + '</tr></thead><tbody>')
    for r in body:
        is_ref = r[0] in _REFERENCE
        cells = ['<th>{}{}</th>'.format(
            html.escape(r[0]),
            ' <small style="color:var(--mut);font-weight:400">(optimal '
            'ref)</small>' if is_ref else '')]
        for j in range(1, ncol):
            v = r[j] if j < len(r) else ''
            mark = best[j] == r[0] and v not in ('', None)
            try:
                disp = '{:.2f}'.format(float(v))
            except ValueError:
                disp = html.escape(v) or '-'
            style = ('font-weight:700;color:var(--go)' if mark
                     else ('color:var(--mut)' if is_ref else ''))
            cells.append(
                '<td class="c" style="{}">{}</td>'.format(style, disp))
        out.append('<tr>' + ''.join(cells) + '</tr>')
    out.append('</tbody></table></div>')
    return '\n'.join(out)


def render(exp):
    out = ['<h2>Paper table <small>methods &times; metrics '
           '(generate_paper_table.py)</small></h2>']
    dirs = []
    figs = os.path.join(REPO, 'figures')
    if os.path.isdir(figs):
        for d in sorted(os.listdir(figs)):
            if d.startswith('paper_table'):
                p = os.path.join(figs, d, 'paper_table.csv')
                if os.path.exists(p):
                    dirs.append((d, p))
    if not dirs:
        return out[0] + ('<p class="note">no paper_table.csv yet &mdash; run '
                         '<span class="c">python generate_paper_table.py '
                         '--dpsize actual-3 --num_deployments 3 --iters 150'
                         '</span></p>')
    # newest first so the run in progress tops the page
    dirs.sort(key=lambda t: -os.path.getmtime(t[1]))
    for name, p in dirs:
        out.append('<h3 style="font-size:.95rem;margin:1.4rem 0 .3rem">{}'
                   '</h3>'.format(html.escape(name)))
        out.append(_render_csv(p))
        # the literal LaTeX, copy-pasteable into the paper (Tom 2026-08-23)
        texp = os.path.join(os.path.dirname(p), 'paper_table.tex')
        if os.path.exists(texp):
            out.append('<details><summary class="note">LaTeX source</summary>'
                       '<pre style="font-size:.72rem;overflow-x:auto;'
                       'background:var(--bg2,#00000010);padding:.6rem;'
                       'border-radius:6px">{}</pre></details>'.format(
                           html.escape(open(texp).read())))
    out.append('<p class="note" style="color:var(--warn,#c9862b)">KNOWN '
               'SCORING ARTIFACT (2026-08-23): under the hard objectives '
               '(MLU group), strategies that STRAND traffic are not charged '
               'for it -- Unicast&rsquo;s low Obj there is invalid. Fix '
               'queued: bounded stranding penalty in the benefit scalar.</p>')
    out.append('<p class="note">&ldquo;-&rdquo; = metric not applicable '
               'or not yet computed for that cell; genuine zeros print '
               'as 0.</p>')
    out.append('<p class="note">Bold green = best in column among real '
               'methods; one-per-peering is the optimal reference and is '
               'never marked. '
               '(direction-aware). Failure benefit deltas: closer to zero '
               'is better. joint_priority renders once its LP is '
               'registered.</p>')
    return '\n'.join(out)


def experiment():
    return {'id': 'paper_table', 'title': 'Paper table',
            'sections': [{'id': 'tables', 'title': 'tables',
                          'kind': 'paper_table'}]}
