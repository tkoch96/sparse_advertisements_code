"""Dashboard tab for cluster runs launched by `cluster/expctl.py`.

Every run that `expctl launch` registers under `cache/cluster_runs/<run_id>/`
becomes a section here automatically -- there is nothing to add to a
registry per experiment, which is the whole point (Tom's lower-level todo
#1: "I just want to be able to say 'add a dashboard to this task'").

What a section shows, top to bottom:

* a status card -- VM, state, elapsed, cost so far, disk headroom, and the
  **verdict computed from the log rather than the exit code** (see
  `expctl.verdict`; rc==0 has lied before);
* per-size progress with wall time and sec/sim, from the sweep's
  progress.json;
* the timing figures, if `dashboard/plot_cluster_timing.py` has drawn them;
* the tail of the harvested log, so a failure is readable here instead of
  requiring an ssh.

Everything is read from the LOCAL harvest. If a section looks stale, the
harvest is stale -- run `python -m cluster.expctl status <run_id>`.
"""

from __future__ import annotations

import html
import json
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO, 'cache', 'cluster_runs')

_STATE_COLOR = {
    'done': 'var(--go)', 'running': 'var(--acc)',
    'failed': '#c2544d', 'suspect': '#c9862b', 'done-dirty': '#c9862b',
    'killed': 'var(--mut)', 'died': '#c2544d',
}


def _read(path, default=None):
    try:
        with open(path, errors='replace') as fh:
            return fh.read()
    except (IOError, OSError):
        return default


def _read_json(path, default=None):
    txt = _read(path)
    if txt is None:
        return default
    try:
        return json.loads(txt)
    except ValueError:
        return default


def discover():
    """Every registered run, newest first."""
    if not os.path.isdir(RUNS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(RUNS_DIR), reverse=True):
        m = _read_json(os.path.join(RUNS_DIR, name, 'manifest.json'))
        if m:
            out.append(m)
    return out


def _rate(instance_type):
    """$/hr from the single source in cluster/vmlib.

    Three copies of this table existed; the run moved to c7g.16xlarge,
    which two of them had never heard of, and the dashboard quietly
    reported "~$0.00 of compute" for an hour of a $2.32/hr box. A missing
    price must read as unknown, not as free.
    """
    try:
        from cluster.vmlib import HOURLY_USD
    except Exception:                             # noqa: BLE001
        return 0.0
    return HOURLY_USD.get(instance_type, 0.0)


def _fmt_dt(seconds):
    if seconds is None:
        return '?'
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    mnt, s = divmod(rem, 60)
    return '{}h{:02d}m'.format(h, mnt) if h else '{}m{:02d}s'.format(mnt, s)


def _status_of(m):
    """Recompute the verdict from the harvested log, not from the manifest.

    The manifest's `state` is only written when a run is watched to
    completion; a run whose watcher was interrupted would otherwise show
    'running' forever even though its log ends with a traceback.
    """
    from cluster.expctl import verdict
    d = os.path.join(RUNS_DIR, m['run_id'], 'logs')
    log = _read(os.path.join(d, 'run.log'), '') or ''
    rc_txt = (_read(os.path.join(d, 'run.rc'), '') or '').strip()
    rc = int(rc_txt) if rc_txt.lstrip('-').isdigit() else None
    state, headline, details = verdict(
        log, rc, killed=(m.get('state') == 'killed'))
    # This renderer is offline -- it cannot ping the VM to see whether the
    # process is alive. So when the log alone is inconclusive ('running')
    # but the manifest already records a TERMINAL state, the manifest wins.
    # Otherwise a killed or died run shows as in-flight on the dashboard
    # forever, which is the stale-but-plausible state these tools exist to
    # remove.
    _TERMINAL = ('killed', 'died', 'failed', 'done', 'done-dirty', 'suspect')
    if state == 'running' and m.get('state') in _TERMINAL:
        state = m['state']
        headline = m.get('verdict') or 'recorded as {}'.format(state)
    return state, headline, details, log


def _current_ip(m):
    """The VM's address NOW, not the one it had at launch.

    Public IPs are reassigned on every stop/start, so the manifest's
    launch_ip goes stale. The alert JSON is rewritten by vmctl on each
    lifecycle event, which makes it the freshest thing available offline.
    (`head` happens to hold an Elastic IP, so its address survives -- but
    nothing else does, and hardcoding that assumption would break the day
    we use another box.)
    """
    cfg = _read_json(os.path.expanduser(
        '~/.sculptor_cluster_alert/active_cluster.json'), {}) or {}
    head = cfg.get('head') or {}
    if head.get('instance_id') == m.get('instance_id') and head.get('public_ip'):
        return head['public_ip']
    return m.get('launch_ip') or '<VM_IP>'


def _cmd_box(cmd, label):
    """A one-click-to-copy shell command."""
    esc = html.escape(cmd)
    return ('<div style="display:flex;gap:.4rem;align-items:flex-start;'
            'margin:.15rem 0">'
            '<button class="copybtn" data-cmd="{}" title="copy" '
            'style="font:inherit;font-size:.7rem;padding:.1rem .4rem;'
            'border:1px solid var(--line);background:var(--card);'
            'color:var(--mut);border-radius:5px;cursor:pointer;flex:none">'
            'copy</button>'
            '<span class="c" style="word-break:break-all">{}</span>'
            '</div>'
            '<div class="note" style="margin:0 0 .35rem 2.6rem">{}</div>'
            .format(esc, esc, html.escape(label)))


def _tail_cmds(m):
    ip = _current_ip(m)
    remote = m.get('remote_log', '')
    ssh_cmd = ("ssh -i ~/.ssh/ray-autoscaler_us-east-1.pem "
               "-o StrictHostKeyChecking=accept-new ubuntu@{} "
               "'tail -f {}'".format(ip, remote))
    # Strip \r so tqdm redraws advance instead of repainting one line, and
    # keep the markers that actually matter visible.
    filt = ("ssh -i ~/.ssh/ray-autoscaler_us-east-1.pem ubuntu@{} "
            "'tail -f {}' | tr '\\r' '\\n' | "
            "grep -E '^\\[sweep\\]|^\\[mem\\]|LEARNING ITERATION|Timer:|"
            "^\\[parallel\\]|Traceback|Error'".format(ip, remote))
    local = ('tail -f {}/cache/cluster_runs/{}/logs/run.log'.format(
        REPO, m['run_id']))
    return (_cmd_box(ssh_cmd, 'live from the VM (everything)') +
            _cmd_box(filt, 'live, milestones only (tqdm bars stripped)') +
            _cmd_box(local, 'the harvested copy on this Mac '
                            '(advances each harvest, survives the VM)'))


def _card(m, state, headline, details):
    end = m.get('finished_epoch') or time.time()
    elapsed = end - m.get('started_epoch', end)
    rate = _rate(m.get('instance_type'))
    cost = rate * elapsed / 3600.0
    harvest = _read_json(os.path.join(RUNS_DIR, m['run_id'], 'harvest.json'), {})
    sysmon = _read(os.path.join(RUNS_DIR, m['run_id'], 'logs',
                                'sysmon.jsonl'), '') or ''
    disk = '?'
    for line in reversed(sysmon.strip().splitlines()[-5:] or []):
        try:
            disk = '{} GB'.format(json.loads(line)['disk_avail_gb'])
            break
        except (ValueError, KeyError):
            continue
    rows = [
        ('verdict', '<b style="color:{}">{}</b> &mdash; {}'.format(
            _STATE_COLOR.get(state, 'var(--ink)'), state.upper(),
            html.escape(headline))),
        ('vm', '{} {}'.format(m.get('instance_id', '?'),
                              m.get('instance_type', ''))),
        ('started', m.get('started_utc', '?')),
        ('elapsed', '{} &nbsp; ~${:.2f} of compute'.format(
            _fmt_dt(elapsed), cost)),
        ('disk free', disk),
        ('last harvest', harvest.get('last_harvest_utc', 'never')),
        ('command', '<span class="c">{}</span>'.format(
            html.escape(' '.join(m.get('cmd', []))))),
        ('local', '<span class="c">cache/cluster_runs/{}/</span>'.format(
            m['run_id'])),
        ('live tail', _tail_cmds(m)),
    ]
    out = ['<div class="wrap"><table><tbody>']
    for k, v in rows:
        out.append('<tr><th>{}</th><td>{}</td></tr>'.format(k, v))
    out.append('</tbody></table></div>')
    if details:
        out.append('<p class="note" style="color:#c2544d">' +
                   '<br>'.join('!! ' + html.escape(d) for d in details) +
                   '</p>')
    return '\n'.join(out)


def _progress_table(m):
    p = _read_json(os.path.join(RUNS_DIR, m['run_id'], 'logs',
                                'progress.json'))
    if not p:
        return ('<p class="note">no progress.json harvested yet &mdash; the '
                'sweep writes it at the first size boundary.</p>')
    sizes = p.get('sizes', [])
    done = p.get('done', {})
    nsim = p.get('nsim', {})
    out = ['<div class="wrap"><table><thead><tr><th>deployment size</th>'
           '<th>nsim</th><th>state</th><th>wall</th><th>sec / sim</th>'
           '</tr></thead><tbody>']
    for s in sizes:
        e = done.get(str(s))
        if e is None:
            st = ('<b style="color:var(--acc)">running</b>'
                  if p.get('current') == s else '<span class="mut">queued</span>')
            out.append('<tr><th>actual-{}</th><td>{}</td><td>{}</td>'
                       '<td>-</td><td>-</td></tr>'.format(
                           s, nsim.get(str(s), '?'), st))
        else:
            ok = e.get('ok')
            # A cache hit is NOT a timing measurement: the size returned in
            # about a second without training. Showing it as a plain "ok"
            # with a 1s wall would quietly poison the pricing model, so it
            # gets its own label and its timings are struck through.
            cached = e.get('cached')
            if not ok:
                state, color = 'FAILED', '#c2544d'
            elif cached:
                state, color = 'cached', '#c9862b'
            else:
                state, color = 'ok', 'var(--go)'
            wall = _fmt_dt(e.get('wall_s'))
            sps = ('{:.0f}'.format(e['sec_per_sim'])
                   if e.get('sec_per_sim') else '-')
            if cached:
                wall = '<s class="mut">{}</s>'.format(wall)
                sps = '<s class="mut">{}</s>'.format(sps)
            out.append('<tr><th>actual-{}</th><td>{}</td>'
                       '<td style="color:{}">{}</td><td class="c">{}</td>'
                       '<td class="c">{}</td></tr>'.format(
                           s, nsim.get(str(s), '?'), color, state, wall, sps))
    out.append('</tbody></table></div>')
    if any((done.get(str(s)) or {}).get('cached') for s in sizes):
        out.append('<p class="note" style="color:#c9862b">Struck-through '
                   'rows were <b>cache hits</b> &mdash; the size loaded a '
                   'previous result and did no training, so its wall time '
                   'is a disk read, not a measurement. Relaunch with '
                   '<span class="c">expctl launch --nocache</span> for '
                   'honest timing.</p>')
    if p.get('phase') == 'done':
        out.append('<p class="note">sweep finished: {}/{} sizes ok in {}'
                   '.</p>'.format(p.get('n_ok', '?'), p.get('n_total', '?'),
                                  _fmt_dt(p.get('wall_s'))))
    return '\n'.join(out)


def _phase_table(m):
    """Per-size phase breakdown, in numbers rather than bar heights."""
    from dashboard import cluster_phases as cp
    recs = cp.load(m['run_id']) or cp.parse(m['run_id'])
    recs = {k: v for k, v in recs.items() if v.get('total_s', 0) > 0}
    if not recs:
        return ''
    sizes = sorted(recs)
    stages = [lbl for _t, lbl in cp.EVAL_STAGES]

    def cell(sec, tot):
        pct = (100.0 * sec / tot) if tot else 0
        return '<td class="c">{}<span class="mut"> {:.0f}%</span></td>'.format(
            _fmt_dt(sec), pct)

    out = ['<h3 style="font-size:.9rem;margin:1.2rem 0 .3rem">phase '
           'breakdown</h3>',
           '<div class="wrap"><table><thead><tr><th>size</th><th>total</th>'
           '<th>deployment setup</th><th>sparse init</th>'
           '<th>learning</th><th>evals</th><th>unaccounted</th>'
           '</tr></thead><tbody>']
    for s in sizes:
        r = recs[s]
        tot = r['total_s']
        out.append('<tr><th>actual-{}{}</th><td class="c">{}</td>{}{}{}{}{}'
                   '</tr>'.format(
                       s, ' <span class="mut">(partial)</span>'
                          if r.get('partial') else '',
                       _fmt_dt(tot),
                       cell(r.get('setup', 0), tot),
                       cell(r.get('sparse_init', 0), tot),
                       cell(r.get('learning', 0), tot),
                       cell(r.get('evals_total', 0), tot),
                       cell(r.get('residual', 0), tot)))
    out.append('</tbody></table></div>')

    out.append('<p class="note">learning column covers '
               '{} iterations per size.</p>'.format(
                   ', '.join('actual-{}:{}'.format(s, recs[s].get('n_iters', 0))
                             for s in sizes)))

    out.append('<div class="wrap"><table><thead><tr><th>size</th>' +
               ''.join('<th>{}</th>'.format(html.escape(x)) for x in stages) +
               '</tr></thead><tbody>')
    for s in sizes:
        r = recs[s]
        tot = r.get('evals_total', 0)
        out.append('<tr><th>actual-{}</th>{}</tr>'.format(
            s, ''.join(cell(r['evals'].get(x, 0), tot) for x in stages)))
    out.append('</tbody></table></div>')

    # (d) other solutions -- and the fact that they overlap learning.
    any_others = [r for r in recs.values() if r.get('others')]
    if any_others:
        names = sorted({k for r in any_others for k in r['others']})
        out.append('<div class="wrap"><table><thead><tr><th>size</th>' +
                   ''.join('<th>{}</th>'.format(n) for n in names) +
                   '<th>wall (max)</th><th>CPU (sum)</th></tr></thead>'
                   '<tbody>')
        for s in sizes:
            r = recs[s]
            if not r.get('others'):
                continue
            out.append('<tr><th>actual-{}</th>{}<td class="c">{:.1f}s</td>'
                       '<td class="c">{:.1f}s</td></tr>'.format(
                           s,
                           ''.join('<td class="c">{:.2f}s</td>'.format(
                               r['others'].get(n, 0)) for n in names),
                           r.get('others_wall', 0), r.get('others_cpu', 0)))
        out.append('</tbody></table></div>')
        par = any(r.get('others_parallel') for r in any_others)
        out.append('<p class="note">The five non-sparse solutions run '
                   '<b>{}</b> &mdash; as subprocesses launched alongside '
                   'sparse, so their wall time overlaps the learning column '
                   'and is <b>not</b> additive with it. "wall (max)" is what '
                   'they actually cost if they finish before sparse does; '
                   '"CPU (sum)" is the core-time they consume.</p>'.format(
                       'IN PARALLEL' if par else 'SERIALLY'))
    return '\n'.join(out)


def _init_table(m):
    """sparse init, decomposed. See plot_cluster_timing._INIT_LABELS."""
    from dashboard import cluster_phases as cp
    from dashboard.plot_cluster_timing import _INIT_LABELS
    recs = cp.load(m['run_id']) or cp.parse(m['run_id'])
    recs = {k: v for k, v in recs.items() if v.get('init_breakdown')}
    if not recs:
        return ''
    sizes = sorted(recs)
    keys = []
    for s in sizes:
        for k, _v in sorted(recs[s]['init_breakdown'].items(),
                            key=lambda kv: -kv[1]):
            if k not in keys:
                keys.append(k)
    out = ['<h3 style="font-size:.9rem;margin:1.2rem 0 .3rem">sparse init, '
           'by component</h3>',
           '<div class="wrap"><table><thead><tr><th>component</th>' +
           ''.join('<th>actual-{}</th>'.format(s) for s in sizes) +
           '</tr></thead><tbody>']
    for k in keys:
        cells = ''.join(
            '<td class="c">{}</td>'.format(
                _fmt_dt(recs[s]['init_breakdown'][k])
                if k in recs[s]['init_breakdown']
                else '<span class="mut">-</span>')
            for s in sizes)
        out.append('<tr><th>{}</th>{}</tr>'.format(
            html.escape(_INIT_LABELS.get(k, k)), cells))
    out.append('<tr><th>sparse init TOTAL</th>{}</tr>'.format(
        ''.join('<td class="c"><b>{}</b></td>'.format(
            _fmt_dt(recs[s]['sparse_init'])) for s in sizes)))
    out.append('</tbody></table></div>')
    out.append('<p class="note">A component spans one <span class="c">[mem]'
               '</span> marker to the NEXT, so it is named for what runs '
               '<i>after</i> the marker that opened it. '
               '&ldquo;between measure-ingress rounds&rdquo; is the gap from '
               'one <span class="c">mi_*</span> block ending to the next '
               'beginning &mdash; the measurement itself, which carries no '
               'markers of its own.</p>')
    return '\n'.join(out)


def _ram_table(m):
    """Per-worker RAM by object -- what to shrink to fit a cheaper box."""
    from dashboard import cluster_objsize as co
    d = co.load(m['run_id']) or co.parse(m['run_id'])
    by = {k: v for k, v in (d.get('by_size') or {}).items() if v.get('attrs')}
    if not by:
        return ''
    sizes = sorted(by)
    attrs = []
    for s in sizes:
        for a in by[s]['attrs'][:10]:
            if a['attr'] not in attrs:
                attrs.append(a['attr'])
    out = ['<h3 style="font-size:.9rem;margin:1.2rem 0 .3rem">per-worker '
           'RAM by object</h3>',
           '<div class="wrap"><table><thead><tr><th>object</th>' +
           ''.join('<th>actual-{}</th>'.format(s) for s in sizes) +
           '</tr></thead><tbody>']
    for attr in attrs:
        cells = []
        for s in sizes:
            hit = [a for a in by[s]['attrs'] if a['attr'] == attr]
            cells.append('<td class="c">{}</td>'.format(
                '{:.1f} MB'.format(hit[0]['max_mb']) if hit
                else '<span class="mut">-</span>'))
        out.append('<tr><th class="c">{}</th>{}</tr>'.format(
            html.escape(attr), ''.join(cells)))
    out.append('<tr><th>census TOTAL (attributed)</th>{}</tr>'.format(
        ''.join('<td class="c"><b>{:.1f} MB</b></td>'.format(
            by[s]['worker_total_max_mb']) for s in sizes)))
    # RSS and coverage, so the census can never again read as if it were
    # the whole process. A census covering 24% of RSS with no such row is
    # how "158 MB per worker" stood next to a real 1074 MB.
    if any(by[s].get('worker_rss_max_mb') for s in sizes):
        out.append('<tr><th>peak worker RSS</th>{}</tr>'.format(
            ''.join('<td class="c">{}</td>'.format(
                '{:.0f} MB'.format(by[s]['worker_rss_max_mb'])
                if by[s].get('worker_rss_max_mb')
                else '<span class="mut">-</span>') for s in sizes)))
        out.append('<tr><th class="mut">unattributed (RSS - census)</th>'
                   '{}</tr>'.format(
                       ''.join('<td class="c mut">{}</td>'.format(
                           '{:.0f} MB'.format(by[s]['unattributed_mb'])
                           if by[s].get('unattributed_mb')
                           else '-') for s in sizes)))
        out.append('<tr><th>coverage</th>{}</tr>'.format(
            ''.join('<td class="c" style="color:{}">{}</td>'.format(
                'var(--go)' if (by[s].get('coverage_pct') or 0) >= 50
                else '#c9862b',
                '{:.0f}%'.format(by[s]['coverage_pct'])
                if by[s].get('coverage_pct')
                else '<span class="mut">not reported</span>')
                for s in sizes)))
    out.append('<tr><th class="mut">workers observed</th>{}</tr>'.format(
        ''.join('<td class="c mut">{}</td>'.format(by[s]['n_workers_seen'])
                for s in sizes)))
    out.append('</tbody></table></div>')

    # Call out whole_deployment_* / shard duplicate pairs: on 2026-08-21
    # these were byte-for-byte the same size, i.e. each worker holds a full
    # copy alongside its shard. That is the single clearest RAM lever.
    dupes = []
    biggest = max(sizes)
    rows = {a['attr']: a['max_mb'] for a in by[biggest]['attrs']}
    for attr, mb in rows.items():
        if attr.startswith('whole_deployment_'):
            peer = attr[len('whole_deployment_'):]
            if peer in rows and abs(rows[peer] - mb) < 0.05 * max(mb, 1):
                dupes.append((attr, peer, mb))
    if dupes:
        tot = sum(mb for _a, _p, mb in dupes)
        out.append('<p class="note" style="color:#c9862b">At actual-{}, {} '
                   'whole-deployment/shard pair(s) are the same size &mdash; '
                   '{} &mdash; totalling <b>{:.1f} MB per worker</b> of '
                   'apparent duplication. Each worker appears to hold a full '
                   'copy beside its shard; that is the clearest lever on '
                   'per-process RAM, and therefore on which instance family '
                   'a size fits.</p>'.format(
                       biggest, len(dupes),
                       ', '.join(p for _a, p, _m in dupes), tot))
    out.append('<p class="note">Source: {}. Peak = the largest value seen '
               'for any worker at that size. <b>Coverage is census / RSS</b> '
               '&mdash; it will never reach 100%: RSS also holds the '
               'interpreter, numpy/Ray/HiGHS native allocations and '
               'allocator fragmentation. Watch the unattributed row: if it '
               'stays near a constant baseline while the census grows, the '
               'accounting is sound. Sizes showing &ldquo;not reported&rdquo; '
               'predate the census measuring its own coverage.</p>'.format(
                   html.escape(d.get('source', '?'))))
    return '\n'.join(out)


def _figures(m):
    """Timing figures for this run, if plot_cluster_timing has drawn them."""
    rel = 'figures/dashboards/cluster/{}'.format(m['run_id'])
    d = os.path.join(REPO, rel)
    if not os.path.isdir(d):
        return ''
    imgs = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.png'):
            continue
        p = os.path.join(d, fn)
        imgs.append('<img src="plots/dashboards/cluster/{}/{}?v={}" alt="{}">'
                    .format(m['run_id'], fn, int(os.path.getmtime(p)),
                            html.escape(fn)))
    return '\n'.join(imgs)


def _log_tail(m, n=120):
    log = _read(os.path.join(RUNS_DIR, m['run_id'], 'logs', 'run.log'), '') or ''
    if not log:
        return '<p class="note">no log harvested yet.</p>'
    # tqdm redraws with \r; without this split the tail shows the first
    # frame of a progress bar forever and reads as a hang.
    lines = log.replace('\r', '\n').rstrip().splitlines()
    body = html.escape('\n'.join(lines[-n:]))
    return ('<p class="note">last {} of {} lines &mdash; full log at '
            '<span class="c">cache/cluster_runs/{}/logs/run.log</span></p>'
            '<pre class="c" style="max-height:60vh;overflow:auto;'
            'background:var(--card);border:1px solid var(--line);'
            'padding:.7rem;white-space:pre-wrap">{}</pre>'.format(
                min(n, len(lines)), len(lines), m['run_id'], body))


def render(exp):
    """Renderer for kind 'cluster_run'. `exp['run_id']` selects the run."""
    runs = {m['run_id']: m for m in discover()}
    m = runs.get(exp.get('run_id'))
    if m is None:
        return ('<h2>{}</h2><p class="note">no manifest under '
                'cache/cluster_runs/{}/ &mdash; was it harvested?</p>'.format(
                    html.escape(exp.get('title', 'run')),
                    html.escape(str(exp.get('run_id')))))
    state, headline, details, _log = _status_of(m)
    return '\n'.join([
        '<h2>{} <small>{}</small></h2>'.format(
            html.escape(m.get('label', m['run_id'])), m['run_id']),
        _card(m, state, headline, details),
        '<h3 style="font-size:.9rem;margin:1.2rem 0 .3rem">progress</h3>',
        _progress_table(m),
        _phase_table(m),
        _init_table(m),
        _ram_table(m),
        _figures(m),
        '<h3 style="font-size:.9rem;margin:1.2rem 0 .3rem">log</h3>',
        _log_tail(m),
    ])


def render_index(exp):
    """Roll-up of every run, for the tab's first section."""
    runs = discover()
    if not runs:
        return ('<h2>Cluster runs</h2><p class="note">nothing launched yet. '
                '<span class="c">python -m cluster.expctl launch head '
                '--preset dpsweep --label smoke</span></p>')
    out = ['<h2>Cluster runs</h2>',
           '<div class="wrap"><table><thead><tr><th>run</th><th>verdict</th>'
           '<th>vm</th><th>started</th><th>elapsed</th><th>$</th>'
           '</tr></thead><tbody>']
    total = 0.0
    for m in runs:
        state, headline, _d, _l = _status_of(m)
        end = m.get('finished_epoch') or time.time()
        elapsed = end - m.get('started_epoch', end)
        rate = _rate(m.get('instance_type'))
        cost = rate * elapsed / 3600.0
        total += cost
        out.append('<tr><th>{}</th><td style="color:{}">{}</td><td>{}</td>'
                   '<td class="c">{}</td><td class="c">{}</td>'
                   '<td class="c">${:.2f}</td></tr>'.format(
                       html.escape(m['run_id']),
                       _STATE_COLOR.get(state, 'var(--ink)'), state,
                       m.get('instance_type', '?'),
                       m.get('started_utc', '?')[:16],
                       _fmt_dt(elapsed), cost))
    out.append('</tbody></table></div>')
    out.append('<p class="note">~${:.2f} of compute across {} run(s). '
               'Cost is elapsed &times; on-demand rate &mdash; indicative, '
               'not billing.</p>'.format(total, len(runs)))
    return '\n'.join(out)


def sections():
    """One section per run, newest first, behind an index section."""
    secs = [{'id': 'index', 'title': 'all runs', 'kind': 'cluster_index'}]
    for m in discover()[:12]:
        secs.append({'id': m['run_id'].replace('-', '_'),
                     'title': m.get('label') or m['run_id'],
                     'kind': 'cluster_run', 'run_id': m['run_id']})
    return secs


def experiment():
    return {
        'id': 'cluster_runs', 'title': 'Cluster runs',
        # Both steps are cheap and therefore 'always' (dashboard/README.md:
        # staleness gating is a cost optimization for expensive evals only).
        # The harvest step is the unattended half of the never-lose-a-log
        # contract: it runs every refresh cycle whether or not anyone is
        # watching, and no-ops when nothing is running.
        'refresh': {'steps': [
            {'argv': ['{py}', '-m', 'cluster.harvest_all', '--quiet'],
             'always': True},
            {'argv': ['{py}', '-m', 'dashboard.plot_cluster_timing', '--all'],
             'always': True},
        ]},
        'sections': sections()}
