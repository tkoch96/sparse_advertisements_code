"""Render the scoping-smoke profiler's status figure for the dash
(Tom 2026-08-17). Pulls from the profiler VM (fleet registry entry with
shard='profiling'): smoke_profile.log START/DONE lines, profiles.json
rows, and the CURRENT cell's [mem] iteration lines -> one PNG with a
completed-picks table + the live cell's RAM/iteration trace.
Run by the dash refresh loop ('always' step).
"""
import json
import os
import re
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cluster.fleet import registry
from helpers.figpaths import fig_path  # -> figures/dashboards/<dashboard>/

KEY = os.path.expanduser('~/.ssh/ray-autoscaler_us-east-1.pem')
SSH = ['ssh', '-i', KEY, '-o', 'StrictHostKeyChecking=no',
       '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes']
MEM_RE = re.compile(r'rss_mb=(\d+) vms_mb=\d+ peak_mb=\d+ '
                    r'sys_avail_mb=(\d+) pid=\d+ t=([0-9.]+) iter=(\d+)')


def profiler_ip():
    for e in registry.fleet():
        if e.get('shard') == 'profiling' and e.get('state') not in (
                'terminated',):
            return e.get('public_ip')
    return None


def fetch(ip):
    cmd = (
        'grep -E "START|DONE|all picks" ~/smoke_profile.log 2>/dev/null; '
        'echo ---PROFILES---; '
        'cat ~/sparse_advertisements_code/cache/eods/profiles/profiles.json '
        '2>/dev/null; echo; echo ---CURRENT---; '
        'L=$(ls -t ~/prof_ws/*/logs/*.log 2>/dev/null | head -1); '
        'echo "$L"; grep "tag=iter_start" "$L" 2>/dev/null | tail -400')
    r = subprocess.run(SSH + ['ubuntu@{}'.format(ip), cmd],
                       capture_output=True, text=True, timeout=30)
    return r.stdout


def main():
    ip = profiler_ip()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 4.4),
        gridspec_kw={'width_ratios': [1.4, 1]})
    ax1.axis('off')
    title = 'Scoping smokes — profiler'
    if not ip:
        ax1.text(0.5, 0.5, 'no profiler VM in fleet registry',
                 ha='center')
    else:
        title += ' @ {} (single-core ladder)'.format(ip)
        try:
            out = fetch(ip)
        except Exception as e:
            out = ''
            ax1.text(0.5, 0.5, 'fetch failed: {}'.format(e), ha='center')
        head, _, rest = out.partition('---PROFILES---')
        pj, _, cur = rest.partition('---CURRENT---')
        profs = []
        try:
            profs = json.loads(pj.strip() or '[]')
        except ValueError:
            pass
        started = [l.split('START ')[1].split(' (')[0]
                   for l in head.splitlines() if 'START' in l]
        rows = [['pick', 'wall', 'startup', 's/iter', 'iters',
                 'rss pk', 'foot pk', '$/cell']]
        for p in profs:
            rows.append([
                p['label'], '{:.0f}m'.format(p.get('wall_s', 0) / 60),
                '{:.0f}s'.format(p.get('startup_s') or 0),
                str(p.get('s_per_iter_p50', '-')),
                str(p.get('iters_seen', '-')),
                '{}G'.format(p.get('driver_rss_gb_peak', '-')),
                '{}G'.format(p.get('sys_footprint_gb_peak', '-')),
                '${}'.format(p.get('cell_usd_spot', '-'))])
        done_labels = {p['label'] for p in profs}
        running = [s for s in started if s not in done_labels]
        tbl = ax1.table(cellText=rows, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.35)
        ax1.set_title('completed picks ({}); running: {}'.format(
            len(profs), ', '.join(running[-1:]) or '—'), fontsize=10)
        # live cell trace
        pts = [(float(m.group(3)), int(m.group(4)), int(m.group(1)))
               for m in (MEM_RE.search(l) for l in cur.splitlines()) if m]
        if pts:
            t0 = pts[0][0]
            ax2.plot([p[0] - t0 for p in pts], [p[2] / 1024 for p in pts],
                     lw=1.4, color='#2a78d6')
            ax2.set_xlabel('s since first iteration')
            ax2.set_ylabel('driver RSS (GB)', color='#2a78d6')
            ax2b = ax2.twinx()
            ax2b.plot([p[0] - t0 for p in pts], [p[1] for p in pts],
                      lw=1.0, color='#2f9e6e', alpha=.7)
            ax2b.set_ylabel('iteration', color='#2f9e6e')
            ax2.set_title('live cell: iter {} '.format(pts[-1][1]),
                          fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'no live iteration data', ha='center')
            ax2.axis('off')
    fig.suptitle(title + '  ({}Z)'.format(
        time.strftime('%H:%M', time.gmtime())), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_fn = fig_path('profiler_status.png')
    os.makedirs(os.path.dirname(out_fn), exist_ok=True)
    fig.savefig(out_fn, dpi=140)
    print('wrote', out_fn)


if __name__ == '__main__':
    main()
