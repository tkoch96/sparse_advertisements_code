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
import re
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
    # run.rc is cleared REMOTELY at relaunch but the harvested local copy
    # survives -- after a resume it still holds the PREVIOUS segment's rc
    # (43 here, 2026-08-24) and the card called a live run FAILED. Only
    # honor rc when the log's CURRENT segment actually wrote its exit
    # marker; an open segment has no exit code yet, whatever the file says.
    i = log.rfind('[expctl] run_id=')
    if i >= 0 and 'exit_rc=' not in log[i:]:
        rc = None
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


def _pt_card_summary(m):
    """'objectives 1/5 done -- projected ~34h / ~$97 more at current pace'"""
    try:
        states = _pt_cell_states(m)
    except Exception:
        return None
    done = sum(1 for st, _ in states.values() if st in ('done', 'cached'))
    total = len(states)
    # measured pace: wall of completed cells, else the running cell's
    # sec/iter extrapolated to the cap
    drv = _read(os.path.join(RUNS_DIR, m['run_id'], 'logs', 'run.log')) or ''
    i = drv.rfind('[expctl] run_id=')
    walls = [float(x) for x in re.findall(
        r'\] DONE in ([\d.]+) min', drv[i:])] if i >= 0 else []
    per_cell_h = (sum(walls) / len(walls) / 60.0) if walls else None
    if per_cell_h is None:
        for obj, (st, log) in states.items():
            if st == 'running' and log:
                pts = [(float(t), int(n)) for t, n in
                       _PT_ITER_T_RE.findall(_read(log) or '')]
                d = [(b[0] - a[0]) for a, b in zip(pts[-8:], pts[-7:])
                     if a[1] < b[1] and 0 < b[0] - a[0] < 3600]
                if d:
                    per_cell_h = 150 * (sum(d) / len(d)) / 3600.0 + 0.5
    if per_cell_h is None:
        return '{}/{} objectives done'.format(done, total)
    remaining_h = per_cell_h * (total - done)
    rate = _rate(m.get('instance_type'))
    return ('{}/{} objectives done &mdash; projected ~{} / ~${:.0f} more '
            'at current pace'.format(done, total, _fmt_dt(remaining_h * 3600),
                                     remaining_h * rate))


def _card(m, state, headline, details, extra_rows=None):
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
    for r in (extra_rows or []):
        rows.insert(1, r)
    out = ['<div class="wrap"><table><tbody>']
    for k, v in rows:
        out.append('<tr><th>{}</th><td>{}</td></tr>'.format(k, v))
    out.append('</tbody></table></div>')
    if details:
        out.append('<p class="note" style="color:#c2544d">' +
                   '<br>'.join('!! ' + html.escape(d) for d in details) +
                   '</p>')
    return '\n'.join(out)


def _dpsweep_live_line(m):
    """Live iteration + staleness for the size currently training --
    the same burning-money signal the papertable progress table carries."""
    txt = _read(os.path.join(RUNS_DIR, m['run_id'], 'logs', 'run.log')) or ''
    i = txt.rfind('[expctl] run_id=')
    if i > 0:
        txt = txt[i:]
    pts = [(float(t), int(n)) for t, n in _PT_ITER_T_RE.findall(txt)]
    if not pts:
        return ('<p class="note">no training iterations in this segment '
                'yet (deployment setup / baselines).</p>')
    age_min = (time.time() - pts[-1][0]) / 60.0
    d = [(b[0] - a[0]) for a, b in zip(pts[-8:], pts[-7:])
         if a[1] < b[1] and 0 < b[0] - a[0] < 3600]
    spi = (sum(d) / len(d)) if d else None
    terminal = (m.get('state') in
                ('killed', 'died', 'failed', 'done', 'done-dirty', 'suspect')
                or m.get('finished_epoch'))
    if terminal:
        stale = '<span class="mut">(run over)</span>'
    else:
        stale = ('<b style="color:var(--bad,#c0392b)">{:.1f} h ago</b>'.format(
                     age_min / 60) if age_min > 90
                 else '{:.0f} min ago'.format(age_min))
    valves = len(re.findall(r'\[mem-valve\] iter=', txt))
    return ('<p class="note">live: iter <b>{}</b>, last advance {}{}{}</p>'
            .format(pts[-1][1], stale,
                    ', ~{:.0f} s/iter recently'.format(spi) if spi else '',
                    ', {} mem-valve rebirths'.format(valves) if valves
                    else ''))


def _ramp_table(m):
    """Per-size x per-sim status for a multi-deployment sweep segment
    (the 2026-08-25 nsim ramp: 5,10,15,20,25,32 x 20,20,12,5,4,3).
    Sources: progress.json (sizes/nsim/done/current) + the CURRENT
    segment of run.log for which sim is training and its iter."""
    pj = _read_json(os.path.join(RUNS_DIR, m['run_id'], 'logs',
                                 'progress.json'))
    if not pj or not pj.get('nsim'):
        return ''
    txt = _read(os.path.join(RUNS_DIR, m['run_id'], 'logs', 'run.log')) or ''
    i = txt.rfind('[expctl] run_id=')
    if i > 0:
        txt = txt[i:]
    cur = str(pj.get('current'))
    # latest sim marker + iter in the current segment
    # count TRAINING STARTS ('Initializing advertisement' -> iter=0):
    # 'deployment number' lines are family-eval markers that don't fire
    # during training segments at all (Tom 2026-08-26: table said sim
    # 1/20 while 18 sims had already trained). Early stop-v2 exits shown
    # so a fast size reads as progress, not as a stuck iter counter.
    # scope counts to the CURRENT size: slice from the last size banner
    # (otherwise size 5's 18 starts inflate size 10's counter)
    _j = txt.rfind('[sweep] === dpsize=')
    _cur_txt = txt[_j:] if _j > 0 else txt
    starts = len(re.findall(r'Initializing advertisement', _cur_txt))
    early = len(re.findall(r'stop-v2.*EARLY EXIT', _cur_txt))
    itm = re.findall(r'\[it\] t=\S+ iter=(\d+)', _cur_txt if _j > 0 else txt)
    cur_iter = int(itm[-1]) if itm else None
    done = pj.get('done') or {}
    out = ['<table><thead><tr><th>size</th><th>sims</th>'
           '<th>status</th></tr></thead><tbody>']
    for sz in pj.get('sizes', []):
        n = (pj.get('nsim') or {}).get(str(sz), '?')
        d = done.get(str(sz))
        if d and d.get('ok'):
            st = 'done in {:.0f} min'.format(d.get('wall_s', 0) / 60)
        elif str(sz) == cur:
            st = ('<b>{} trainings started / {}</b>'
                  '{}{}'.format(
                      starts, n,
                      ', {} converged early'.format(early) if early else '',
                      ', current iter {}'.format(cur_iter)
                      if cur_iter is not None else ' (setup)'))
        else:
            st = '<span class="mut">queued</span>'
        out.append('<tr><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
            sz, n, st))
    out.append('</tbody></table>')
    return ('<h3>deployment ramp</h3>' + '\n'.join(out))


def _szlabel(s):
    s = str(s)
    return s if s.startswith('actual') else 'actual-{}'.format(s)


def _progress_table(m):
    p = _read_json(os.path.join(RUNS_DIR, m['run_id'], 'logs',
                                'progress.json'))
    if not p:
        return ('<p class="note">no progress.json harvested yet &mdash; the '
                'sweep writes it at the first size boundary.</p>')
    sizes = p.get('sizes', [])
    done = p.get('done', {})
    nsim = p.get('nsim', {})
    # ---- REAL progress (Tom 2026-08-26: 'state and goal' is not
    # progress): elapsed, per-size % via sims completed, measured-only
    # total estimate. Current-size sim counts ride the segment log
    # (training starts), like the ramp table.
    now = time.time()
    started = p.get('current_started') or p.get('started')
    txt = _read(os.path.join(RUNS_DIR, m['run_id'], 'logs', 'run.log')) or ''
    i = txt.rfind('[expctl] run_id=')
    if i > 0:
        txt = txt[i:]
    _j = txt.rfind('[sweep] === dpsize=')
    _cur_txt = txt[_j:] if _j > 0 else txt
    cur_starts = len(re.findall(r'Initializing advertisement', _cur_txt))
    cur = p.get('current')
    cur_n = int(nsim.get(str(cur), 0) or 0)
    cur_elapsed = (now - started) if started else None
    cur_done_sims = max(cur_starts - 1, 0)
    cur_frac = (cur_done_sims / cur_n) if cur_n else 0.0
    cur_est = (cur_elapsed / max(cur_frac, 1.0 / max(cur_n, 1) * 0.5)
               if cur_elapsed and cur_n and cur_starts else None)
    total_done_wall = sum((done.get(str(s2)) or {}).get('wall_s') or 0
                          for s2 in sizes)
    run_started = p.get('started')
    overall_elapsed = (now - run_started) if run_started else None
    n_sizes_done = sum(1 for s2 in sizes if str(s2) in done)
    known_total = total_done_wall + (cur_est or 0)
    n_unmeasured = len(sizes) - n_sizes_done - (1 if cur is not None else 0)
    sims_total = sum(int(nsim.get(str(s2), 0) or 0) for s2 in sizes)
    sims_done = sum(int(nsim.get(str(s2), 0) or 0) for s2 in sizes
                    if str(s2) in done) + cur_done_sims
    pct = 100.0 * sims_done / sims_total if sims_total else 0.0
    hdr = ('<p class="note"><b>{:.0f}% of sims done</b> ({}/{}) &middot; '
           'elapsed {} &middot; measured est &ge; {}{}{}</p>'.format(
               pct, sims_done, sims_total,
               _fmt_dt(overall_elapsed) if overall_elapsed else '?',
               _fmt_dt(known_total) if known_total else '?',
               ' (+{} unmeasured queued sizes)'.format(n_unmeasured)
               if n_unmeasured > 0 else '',
               ' &middot; current size ~{:.0f}% ({} of {} sims, ~{}/sim)'
               .format(100 * cur_frac, cur_done_sims, cur_n,
                       _fmt_dt(cur_elapsed / max(cur_done_sims, 1))
                       if cur_elapsed and cur_done_sims else '?')
               if cur is not None and cur_n else ''))
    out = [hdr,
           '<div class="wrap"><table><thead><tr><th>deployment size</th>'
           '<th>nsim</th><th>state</th><th>wall</th><th>sec / sim</th>'
           '</tr></thead><tbody>']
    for s in sizes:
        e = done.get(str(s))
        if e is None:
            if p.get('current') == s:
                st = ('<b style="color:var(--acc)">running</b> '
                      '&mdash; sim {}/{}, ~{:.0f}% of size'.format(
                          min(cur_starts, cur_n or cur_starts),
                          cur_n or '?', 100 * cur_frac))
                wall_c = _fmt_dt(cur_elapsed) if cur_elapsed else '-'
                est_c = ('~{} total'.format(_fmt_dt(cur_est))
                         if cur_est else '-')
            else:
                st, wall_c, est_c = '<span class="mut">queued</span>', '-', '-'
            out.append('<tr><th>{}</th><td>{}</td><td>{}</td>'
                       '<td class="c">{}</td><td class="c">{}</td></tr>'
                       .format(_szlabel(s), nsim.get(str(s), '?'), st, wall_c, est_c))
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
            out.append('<tr><th>{}</th><td>{}</td>'
                       '<td style="color:{}">{}</td><td class="c">{}</td>'
                       '<td class="c">{}</td></tr>'.format(
                           _szlabel(s), nsim.get(str(s), '?'), color,
                           state, wall, sps))
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


_SIM_SIZE_RE = re.compile(r'\[sweep\] === dpsize=(\w+) ')
_SIM_POPS_RE = re.compile(r"Considering pops : \[(.*?)\], deployment size:")
_SIM_BUDGET_RE = re.compile(r'\[probe-gate\] PROBE_N=prefixes -> budget (\d+)')
_SIM_EXIT_RE = re.compile(
    r'\[probe-budget\] EXITING on (\d+) path measures '
    r'\(= (\d+) setup grounding \+ (\d+) probes\) \| budget N=(\d+) '
    r'mode=(\w+) skipped=(\d+) iters=(\d+)')
_SIM_FAIL_RE = re.compile(r'Strategy sparse failed')
# Per-learner accounting (Tom 2026-08-22): measurements conducted by EACH
# learner + iterations to convergence + deployment dims, per sim.
_SIM_OPT_RE = re.compile(r'Optimizing over (\d+) peers and (\d+) ugs')
_SIM_PAINTER_RE = re.compile(r'PAINTER ITER (\d+)')
_SIM_PCAP_RE = re.compile(r'PAINTER measurement cap hit')
_SIM_ANYOPT_RE = re.compile(r'Measuring anyopt providers\.: 100%.*?\| (\d+)/\d+')


def _sim_table(m):
    """Per-sim rows parsed from the harvested log: with nsim>1 each size is
    MANY deployments (a fresh random PoP draw per sim -- the thing the
    2026-08-22 worker-rebirth bug broke), and the per-size progress table
    hides that entirely. One <details> block per size: sim #, the drawn
    pops, the per-deployment probe budget (PROBE_N=prefixes resolves per
    draw), spend/skips/iters, and whether sparse survived."""
    txt = _read(os.path.join(RUNS_DIR, m['run_id'], 'logs', 'run.log'))
    if not txt:
        return ''
    # Judge the CURRENT segment only (resumes append; same convention as
    # expctl.verdict).
    i = txt.rfind('[expctl] run_id=')
    if i > 0:
        txt = txt[i:]
    sizes = {}          # size -> [sim dicts]
    order = []
    cur_size, cur = None, None
    for line in txt.splitlines():
        mm = _SIM_SIZE_RE.search(line)
        if mm:
            cur_size = mm.group(1)
            if cur_size not in sizes:
                sizes[cur_size] = []
                order.append(cur_size)
            cur = None
            continue
        if cur_size is None:
            continue
        mm = _SIM_POPS_RE.search(line)
        if mm:
            pops = mm.group(1).replace("'", '').replace('vtr', '')
            cur = {'pops': ' '.join(pops.split()), 'budget': None,
                   'probes': None, 'skipped': None, 'iters': None,
                   'popps': None, 'ugs': None, 'painter_it': None,
                   'painter_cap': False, 'anyopt_meas': None,
                   'state': 'running'}
            sizes[cur_size].append(cur)
            continue
        if cur is None:
            continue
        mm = _SIM_BUDGET_RE.search(line)
        if mm and cur['budget'] is None:
            cur['budget'] = mm.group(1)
            continue
        mm = _SIM_EXIT_RE.search(line)
        if mm:
            cur['probes'], cur['budget'] = mm.group(3), mm.group(4)
            cur['skipped'], cur['iters'] = mm.group(6), mm.group(7)
            cur['state'] = 'ok'
            continue
        if _SIM_FAIL_RE.search(line):
            cur['state'] = 'FAILED'
            continue
        mm = _SIM_OPT_RE.search(line)
        if mm and cur['popps'] is None:
            cur['popps'], cur['ugs'] = mm.group(1), mm.group(2)
            continue
        mm = _SIM_PAINTER_RE.search(line)
        if mm:
            cur['painter_it'] = max(int(mm.group(1)),
                                    cur['painter_it'] or 0)
            continue
        if _SIM_PCAP_RE.search(line):
            cur['painter_cap'] = True
            continue
        mm = _SIM_ANYOPT_RE.search(line)
        if mm:
            cur['anyopt_meas'] = mm.group(1)
    if not order:
        return ''
    out = ['<h3 style="font-size:.9rem;margin:1.2rem 0 .3rem">deployments '
           'per size <small class="mut">(each sim is a fresh random PoP '
           'draw; budget resolves per draw)</small></h3>']
    for sz in order:
        sims = sizes[sz]
        n_ok = sum(1 for x in sims if x['state'] == 'ok')
        n_bad = sum(1 for x in sims if x['state'] == 'FAILED')
        summ = 'actual-{} &mdash; {} sim(s): {} sparse-ok'.format(
            sz, len(sims), n_ok)
        if n_bad:
            summ += ', <b style="color:#c2544d">{} FAILED</b>'.format(n_bad)
        is_last = (sz == order[-1])
        out.append('<details{}><summary>{}</summary>'.format(
            ' open' if (is_last or n_bad) else '', summ))
        out.append('<div class="wrap"><table><thead><tr><th>#</th>'
                   '<th>pops</th><th>popps</th><th>ugs</th>'
                   '<th>prefixes</th>'
                   '<th>sculptor meas</th><th>sculptor iters</th>'
                   '<th>painter meas=iters</th><th>anyopt meas</th>'
                   '<th>sparse</th></tr></thead><tbody>')
        for j, x in enumerate(sims):
            color = {'ok': 'var(--go)', 'FAILED': '#c2544d'}.get(
                x['state'], 'var(--acc)')
            # measures = probes + 1 setup grounding (the EXITING contract)
            sc_meas = ('{}/{}'.format(int(x['probes']) + 1, x['budget'])
                       if x['probes'] is not None and x['budget']
                       else (x['budget'] and '?/{}'.format(x['budget'])) or '-')
            p_it = x['painter_it']
            painter = '-' if p_it is None else '{}{}'.format(
                p_it + 1, ' (cap)' if x['painter_cap'] else '')
            out.append(
                '<tr><th>{}</th><td>{}</td><td class="c">{}</td>'
                '<td class="c">{}</td><td class="c">{}</td>'
                '<td class="c">{}</td><td class="c">{}</td>'
                '<td class="c">{}</td><td class="c">{}</td>'
                '<td style="color:{}">{}</td></tr>'.format(
                    j, html.escape(x['pops']), x['popps'] or '-',
                    x['ugs'] or '-', x['budget'] or '-',
                    sc_meas, x['iters'] or '-',
                    painter, x['anyopt_meas'] or '-',
                    color, x['state']))
        out.append('</tbody></table></div></details>')
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
        out.append('<tr><th>{}</th>{}</tr>'.format(
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
            out.append('<tr><th>{}</th>{}<td class="c">{:.1f}s</td>'
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


# ------------------------------------------------- papertable renderer --

_PT_OBJECTIVES = ['avg_latency', 'per_site_cost', 'max_util',
                  'frac_beyond_optimal', 'joint_priority']
_PT_ITER_RE = re.compile(r'\[mem\] tag=iter_start .*? iter=(\d+)')


def _pt_tag(m, obj):
    """The SCULPTOR_RUN_TAG generate_paper_table gives this objective."""
    cmd = m.get('cmd') or []
    base = None
    for i, tok in enumerate(cmd):
        if tok == '--run_id' and i + 1 < len(cmd):
            base = cmd[i + 1]
    if not base:
        base = m['run_id'].replace('-', '_')
    return base if obj == 'avg_latency' else '{}_{}'.format(base, obj)


def _pt_cell_log(m, obj):
    """Harvested cell log: repo cache/ (glob pulls) or run results/."""
    fn = 'table_generate_{}.log'.format(_pt_tag(m, obj))
    import glob as _g
    cands = ([os.path.join(REPO, 'cache', fn)]
             + _g.glob(os.path.join(RUNS_DIR, m['run_id'], 'results',
                                    '**', fn), recursive=True))
    for cand in cands:
        if os.path.exists(cand):
            return cand
    return None


def _pt_cell_states(m):
    """objective -> (state, cell_log_path or None) from the driver log."""
    drv = _read(os.path.join(RUNS_DIR, m['run_id'], 'logs', 'run.log')) or ''
    i = drv.rfind('[expctl] run_id=')
    if i > 0:
        drv = drv[i:]
    out = {}
    for obj in _PT_OBJECTIVES:
        st = 'pending'
        if re.search(r'\[{}\] DONE in'.format(re.escape(obj)), drv):
            st = 'done'
        elif re.search(r'\[{}\] NO COMPLETION BANNER'.format(re.escape(obj)), drv):
            st = 'FAILED'
        elif re.search(r'\[{}\] covered'.format(re.escape(obj)), drv):
            st = 'cached'
        elif re.search(r'\[{}\] running'.format(re.escape(obj)), drv):
            st = 'running'
        out[obj] = (st, _pt_cell_log(m, obj))
    return out


_PT_ITER_T_RE = re.compile(
    r'\[mem\] tag=iter_start rss_mb=\d+ .*? t=([\d.]+) iter=(\d+)')


def _pt_progress_table(m):
    """The burning-money view: per objective -- live iteration, minutes
    since it last advanced, recent sec/iter, ETA to the iteration cap.
    The dpsweep progress table with objective as the size axis."""
    states = _pt_cell_states(m)
    cap = 150
    for i, tok in enumerate(m.get('cmd') or []):
        if tok == '--num_training_iter':
            try:
                cap = int((m.get('cmd'))[i + 1])
            except Exception:
                pass
    out = ['<h3 style="font-size:.9rem;margin:1.2rem 0 .3rem">progress '
           '<small>(cap {} iters/objective)</small></h3>'.format(cap),
           '<div class="wrap"><table><thead><tr><th>objective</th>'
           '<th>state</th><th>iter</th><th>last advance</th>'
           '<th>sec/iter (recent)</th><th>ETA to cap</th><th>popps/ugs</th>'
           '<th>probes/budget</th><th>valve</th><th>sparse</th>'
           '</tr></thead><tbody>']
    now = time.time()
    for obj in _PT_OBJECTIVES:
        st, log = states[obj]
        it_n = last_age = spi = eta = popps = sparse = ''
        probes = ''; valve = '-'
        if log:
            txt = _read(log) or ''
            pts = [(float(t), int(i)) for t, i in _PT_ITER_T_RE.findall(txt)]
            if pts:
                it_n = str(pts[-1][1])
                age_min = (now - pts[-1][0]) / 60.0
                last_age = ('{:.0f} min ago'.format(age_min)
                            if age_min < 90 else
                            '<b style="color:var(--bad,#c0392b)">'
                            '{:.1f} h ago</b>'.format(age_min / 60))
                d = [(b[0] - a[0]) for a, b in zip(pts[-8:], pts[-7:])
                     if a[1] < b[1] and 0 < b[0] - a[0] < 3600]
                if d:
                    v = sum(d) / len(d)
                    spi = '{:.0f}'.format(v)
                    if st == 'running':
                        eta = _fmt_dt((cap - pts[-1][1]) * v)
            mm = _SIM_OPT_RE.search(txt)
            if mm:
                popps = '{}/{}'.format(mm.group(1), mm.group(2))
            if 'ALL DONE' in txt:
                sparse = ('FAILED' if _SIM_FAIL_RE.search(txt) else 'ok')
            elif _SIM_FAIL_RE.search(txt):
                sparse = 'FAILED'
            mm = _SIM_EXIT_RE.search(txt)
            if mm:
                probes = '{}/{}'.format(mm.group(3), mm.group(4))
            else:
                mm = _SIM_BUDGET_RE.search(txt)
                probes = ('?/' + mm.group(1)) if mm else ''
            valves = len(re.findall(r'\[mem-valve\] iter=', txt))
            stale = len(re.findall(r'\[stale-path\]', txt))
            valve = ('{}x'.format(valves) if valves else '-') + \
                    (' / {} stale'.format(stale) if stale else '')
        color = {'done': 'var(--ok,#2c7a2c)', 'cached': 'var(--ok,#2c7a2c)',
                 'running': 'var(--warn,#c9862b)',
                 'FAILED': 'var(--bad,#c0392b)'}.get(st, 'var(--mut)')
        out.append('<tr><th>{}</th><td style="color:{}">{}</td>'
                   '<td class="c">{}</td><td class="c">{}</td>'
                   '<td class="c">{}</td><td class="c">{}</td>'
                   '<td class="c">{}</td><td class="c">{}</td>'
                   '<td class="c">{}</td><td class="c">{}</td></tr>'.format(
                       html.escape(obj), color, st, it_n, last_age,
                       spi, eta, popps, probes, valve, sparse))
    out.append('</tbody></table></div>')
    return '\n'.join(out)


def _pt_phase_table(m):
    """Per-objective wall split: setup+baselines / training / after-train
    (eval), from the cell logs' [mem] markers."""
    states = _pt_cell_states(m)
    rows = []
    for obj in _PT_OBJECTIVES:
        st, log = states[obj]
        if not log:
            continue
        txt = _read(log) or ''
        ts = re.findall(r'\[mem\] tag=\S+ rss_mb=\d+ .*? t=([\d.]+)', txt)
        pts = [(float(t), int(i)) for t, i in _PT_ITER_T_RE.findall(txt)]
        if not ts:
            continue
        t0, tlast = float(ts[0]), float(ts[-1])
        setup = (pts[0][0] - t0) if pts else (tlast - t0)
        train = (pts[-1][0] - pts[0][0]) if pts else 0
        after = max(0, tlast - (pts[-1][0] if pts else t0))
        rows.append((obj, st, setup, train, after))
    if not rows:
        return ''
    out = ['<h3 style="font-size:.9rem;margin:1.2rem 0 .3rem">phases</h3>',
           '<div class="wrap"><table><thead><tr><th>objective</th>'
           '<th>setup + baselines</th><th>sparse training</th>'
           '<th>eval (so far)</th><th>total</th></tr></thead><tbody>']
    for obj, st, a, b, c in rows:
        out.append('<tr><th>{}</th><td class="c">{}</td><td class="c">{}'
                   '</td><td class="c">{}</td><td class="c">{}</td></tr>'
                   .format(html.escape(obj), _fmt_dt(a), _fmt_dt(b),
                           _fmt_dt(c), _fmt_dt(a + b + c)))
    out.append('</tbody></table></div>')
    return '\n'.join(out)


def _pt_log_tail(m, n=100):
    """Tail of the ACTIVE cell's log -- run.log is the orchestrator and
    barely moves; the training lives in the per-objective cell logs
    (Tom 2026-08-24: 'even the tail log looks like it hasn't updated')."""
    states = _pt_cell_states(m)
    pick = None
    for obj in _PT_OBJECTIVES:
        if states[obj][0] == 'running' and states[obj][1]:
            pick = (obj, states[obj][1])
    if pick is None:
        for obj in reversed(_PT_OBJECTIVES):
            if states[obj][1]:
                pick = (obj, states[obj][1])
                break
    if pick is None:
        return _log_tail(m)
    obj, log = pick
    txt = _read(log) or ''
    lines = [l for l in txt.replace('\r', '\n').splitlines()
             if l.strip() and 'it/s' not in l and 's/it' not in l]
    age_min = (time.time() - os.path.getmtime(log)) / 60.0
    return ('<h3 style="font-size:.9rem;margin:1.2rem 0 .3rem">log &mdash; '
            '{} cell <small>(harvested {:.0f} min ago)</small></h3>'
            '<pre style="font-size:.7rem;max-height:24rem;overflow:auto;'
            'background:var(--bg2,#00000010);padding:.6rem;border-radius:'
            '6px">{}</pre>'.format(html.escape(obj), age_min,
                                   html.escape('\n'.join(lines[-n:]))))


def _objective_table(m):
    """Per-OBJECTIVE rows for preset=papertable runs -- the analogue of
    _sim_table where the unit of work is an objective function, not a
    deployment size. Parsed from the driver log (cell status) + each
    objective's harvested cell log (iters, budget, popps, sparse)."""
    drv = _read(os.path.join(RUNS_DIR, m['run_id'], 'logs', 'run.log')) or ''
    i = drv.rfind('[expctl] run_id=')
    if i > 0:
        drv = drv[i:]
    out = ['<h3 style="font-size:.9rem;margin:1.2rem 0 .3rem">objectives'
           '</h3>',
           '<div class="wrap"><table><thead><tr><th>objective</th>'
           '<th>state</th><th>wall</th><th>iters</th><th>popps/ugs</th>'
           '<th>probes / budget</th><th>skipped</th><th>sparse</th>'
           '</tr></thead><tbody>']
    for obj in _PT_OBJECTIVES:
        state, wall = 'pending', ''
        mm = re.search(r'\[{}\] DONE in ([\d.]+) min'.format(
            re.escape(obj)), drv)
        if mm:
            state, wall = 'done', '{} min'.format(mm.group(1))
        elif re.search(r'\[{}\] NO COMPLETION BANNER'.format(
                re.escape(obj)), drv):
            state = 'FAILED'
        elif re.search(r'\[{}\] running'.format(re.escape(obj)), drv):
            state = 'running'
        elif re.search(r'\[{}\] covered'.format(re.escape(obj)), drv):
            state, wall = 'cached', '-'
        iters = popps = ugs = probes = budget = skipped = ''
        sparse = ''
        log = _pt_cell_log(m, obj)
        if log:
            txt = _read(log) or ''
            it = _PT_ITER_RE.findall(txt)
            if it:
                iters = it[-1]
            mm = _SIM_OPT_RE.search(txt)
            if mm:
                popps, ugs = mm.group(1), mm.group(2)
            mm = _SIM_BUDGET_RE.search(txt)
            if mm:
                budget = mm.group(1)
            mm = _SIM_EXIT_RE.search(txt)
            if mm:
                probes, budget = mm.group(3), mm.group(4)
                skipped = mm.group(6)
            if 'ALL DONE' in txt:
                sparse = ('FAILED' if _SIM_FAIL_RE.search(txt)
                          else 'ok')
        color = {'done': 'var(--ok,#2c7a2c)', 'cached': 'var(--ok,#2c7a2c)',
                 'running': 'var(--warn,#c9862b)',
                 'FAILED': 'var(--bad,#c0392b)'}.get(state, 'var(--ink)')
        out.append('<tr><th>{}</th><td style="color:{}">{}</td>'
                   '<td class="c">{}</td><td class="c">{}</td>'
                   '<td class="c">{}</td><td class="c">{}</td>'
                   '<td class="c">{}</td><td class="c">{}</td></tr>'.format(
                       html.escape(obj), color, state, wall, iters,
                       '{}/{}'.format(popps, ugs) if popps else '',
                       '{} / {}'.format(probes, budget) if probes
                       else (budget or ''), skipped, sparse))
    out.append('</tbody></table></div>')
    return '\n'.join(out)


def _pt_table(m):
    """Render the emitted paper table (key + full) if harvested."""
    import glob as _g
    cands = (_g.glob(os.path.join(RUNS_DIR, m['run_id'], 'results', '**',
                                  'paper_table'), recursive=True)
             + [os.path.join(REPO, 'figures', 'cluster', m['run_id'],
                             'paper_table')])
    d = next((c for c in cands if os.path.isdir(c)), '')
    if not os.path.isdir(d):
        return ''
    from dashboard import paper_table as _pt
    out = []
    for name, title in (('paper_table_key.csv', 'key metrics'),
                        ('paper_table.csv', 'full table')):
        fn = os.path.join(d, name)
        if os.path.exists(fn):
            out.append('<h3 style="font-size:.9rem;margin:1.2rem 0 .3rem">'
                       '{}</h3>'.format(title))
            out.append(_pt._render_csv(fn))
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
    # ladder_* (the evaluate_over_deployment_sizes paper figures) go LAST
    # -- they are many and noisy next to the run's own timing plots
    # (Tom 2026-08-23)
    for fn in sorted(os.listdir(d),
                     key=lambda f: (f.startswith('ladder_'), f)):
        if not fn.endswith('.png'):
            continue
        p = os.path.join(d, fn)
        # ladder figures render two-up at ~50% width (Tom 2026-08-23);
        # the run's own timing plots keep full width
        _style = (' style="width:49%;display:inline-block;'
                  'vertical-align:top"' if fn.startswith('ladder_') else '')
        imgs.append('<img src="plots/dashboards/cluster/{}/{}?v={}" '
                    'alt="{}"{}>'
                    .format(m['run_id'], fn, int(os.path.getmtime(p)),
                            html.escape(fn), _style))
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
    if m.get('preset') == 'papertable':
        # objective-keyed run: the size-keyed progress/sim/phase tables
        # would all render empty -- the unit of work here is an objective
        _summary = _pt_card_summary(m)
        return '\n'.join([
            '<h2>{} <small>{}</small></h2>'.format(
                html.escape(m.get('label', m['run_id'])), m['run_id']),
            _card(m, state, headline, details,
                  extra_rows=[('table progress', _summary)] if _summary
                  else None),
            _pt_progress_table(m),
            _pt_phase_table(m),
            _pt_table(m),
            _figures(m),
            _ram_table(m),
            _pt_log_tail(m),
        ])
    return '\n'.join([
        '<h2>{} <small>{}</small></h2>'.format(
            html.escape(m.get('label', m['run_id'])), m['run_id']),
        _card(m, state, headline, details),
        '<h3 style="font-size:.9rem;margin:1.2rem 0 .3rem">progress</h3>',
        _progress_table(m),
        _dpsweep_live_line(m),
        _ramp_table(m),
        _sim_table(m),
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
