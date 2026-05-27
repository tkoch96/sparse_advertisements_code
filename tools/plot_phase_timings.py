"""Parse [mem] tags from one or more sweep logs and plot per-phase timings.

The driver emits [mem] tags at well-known points inside each training
iter (iter_start, iter_post_grad, iter_post_measure, iter_post_stop_tracker)
and around the per-random_iter init phases (iov_enter, iov_post_*,
solve_*, mi_*, dpsize_start, dpsize_done). The timestamps embedded in
each tag are unix epoch seconds, so we can recover per-phase durations
by taking deltas.

Phases this script computes per training iter:
  grad         : iter_start -> iter_post_grad
  measure      : iter_post_grad -> iter_post_measure
  stop_tracker : iter_post_measure -> iter_post_stop_tracker
  iter_total   : iter_start -> iter_post_stop_tracker (= sum of above)

Phases this script computes per random_iter setup:
  init_optim_vars : iov_enter -> iov_post_gt_resilience_benefit
  first_measure   : solve_post_init_optim_vars -> solve_post_first_measure_ingresses

Output: figures/session_10_phase_timings/*.pdf

Usage:
    python tools/plot_phase_timings.py LOG [LOG ...]
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

MEM_RE = re.compile(
    r'\[mem\] tag=(?P<tag>\S+)\s+'
    r'(?:rss_mb=(?P<rss>-?\d+)\s+)?'
    r'(?:vms_mb=(?P<vms>-?\d+)\s+)?'
    r'(?:peak_mb=(?P<peak>-?\d+)\s+)?'
    r'sys_avail_mb=(?P<avail>-?\d+)\s+'
    r'pid=(?P<pid>\d+)\s+'
    r't=(?P<t>[\d.]+)'
    r'(?P<extra>.*)'
)
SWEEP_RE = re.compile(r'\[sweep\] === dpsize=(\d+)\s+dpsize_str=(\S+)\s+nsim=(\d+) ===')


# ---------------------------------------------------------------------- #
# Parsing                                                                 #
# ---------------------------------------------------------------------- #
def parse_log(path):
    """Return a dict of events:
        'dpsize_phases': list of {dpsize:int, line:int, ts:float} markers
        'mems': list of dicts {tag, t, pid, iter (optional), dpsize (optional)}
    """
    events = []
    dpsize_phases = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            m = SWEEP_RE.search(line)
            if m:
                dpsize_phases.append({
                    'dpsize': int(m.group(1)),
                    'line': lineno,
                })
                continue
            m = MEM_RE.search(line)
            if not m:
                continue
            d = m.groupdict()
            extras_str = d.pop('extra') or ''
            extras = dict(re.findall(r'(\w+)=([\w.+-]+)', extras_str))
            ev = {
                'lineno': lineno,
                'tag': d['tag'],
                't': float(d['t']),
                'pid': int(d['pid']),
                'rss_mb': int(d['rss']) if d.get('rss') else None,
                'avail_mb': int(d['avail']),
            }
            if 'iter' in extras:
                ev['iter'] = int(extras['iter'])
            if 'dpsize' in extras:
                ev['dpsize'] = int(extras['dpsize']) if extras['dpsize'].isdigit() else extras['dpsize']
            events.append(ev)
    return {'mems': events, 'dpsize_phases': dpsize_phases}


def assign_dpsize_to_mems(parsed):
    """Annotate each [mem] event with the dpsize phase it belongs to,
    based on which [sweep] marker preceded it in the log."""
    phases = parsed['dpsize_phases']
    if not phases:
        return parsed
    # Sort phases by lineno (should already be)
    phases = sorted(phases, key=lambda p: p['line'])
    # For each mem event, find the most recent phase whose line < event line.
    phase_idx = 0
    for ev in parsed['mems']:
        while (phase_idx + 1 < len(phases)
               and phases[phase_idx + 1]['line'] <= ev['lineno']):
            phase_idx += 1
        ev['sweep_dpsize'] = phases[phase_idx]['dpsize']
    return parsed


def extract_phase_timings(parsed):
    """Walk mem events for the DRIVER pid (largest event count) and
    produce per-iter phase durations. Returns dict:
        per_iter: list of {dpsize, iter, grad_s, measure_s, stop_s, total_s}
        per_dpsize_meta: {dpsize: {first_iter_ts, last_iter_ts, n_iters}}
    """
    mems = parsed['mems']
    if not mems:
        return {'per_iter': [], 'per_dpsize_meta': {}}
    # Identify driver pid = the pid that produced the most events.
    pid_counts = defaultdict(int)
    for ev in mems:
        pid_counts[ev['pid']] += 1
    driver_pid = max(pid_counts, key=pid_counts.get)

    # Filter to driver events with iter and dpsize info.
    by_pos = defaultdict(dict)   # (dpsize, iter) -> {tag: ts}
    for ev in mems:
        if ev['pid'] != driver_pid:
            continue
        tag = ev['tag']
        if 'iter' not in ev:
            continue
        # Use the sweep-level dpsize annotation
        dp = ev.get('sweep_dpsize')
        if dp is None:
            continue
        key = (dp, ev['iter'])
        if tag in ('iter_start', 'iter_post_grad', 'iter_post_measure',
                   'iter_post_stop_tracker'):
            by_pos[key][tag] = ev['t']

    per_iter = []
    for (dp, it), tags in sorted(by_pos.items()):
        s = tags.get('iter_start')
        g = tags.get('iter_post_grad')
        m = tags.get('iter_post_measure')
        e = tags.get('iter_post_stop_tracker')
        if not all((s, g, m, e)):
            continue
        per_iter.append({
            'dpsize': dp, 'iter': it,
            'grad_s':     g - s,
            'measure_s':  m - g,
            'stop_s':     e - m,
            'total_s':    e - s,
            'iter_start_ts': s,
        })

    # Per-dpsize meta: first/last iter timestamps.
    per_dpsize_meta = defaultdict(lambda: {'first_iter_ts': None,
                                            'last_iter_ts': None,
                                            'n_iters': 0})
    for rec in per_iter:
        dp = rec['dpsize']
        meta = per_dpsize_meta[dp]
        if meta['first_iter_ts'] is None or rec['iter_start_ts'] < meta['first_iter_ts']:
            meta['first_iter_ts'] = rec['iter_start_ts']
        if meta['last_iter_ts'] is None or rec['iter_start_ts'] > meta['last_iter_ts']:
            meta['last_iter_ts'] = rec['iter_start_ts']
        meta['n_iters'] += 1
    return {'per_iter': per_iter,
            'per_dpsize_meta': dict(per_dpsize_meta),
            'driver_pid': driver_pid}


# ---------------------------------------------------------------------- #
# Plotting                                                                #
# ---------------------------------------------------------------------- #
def plot_phase_breakdown_by_dpsize(per_iter_by_log, out_path):
    """One figure: median per-iter phase time (grad / measure / stop) as
    a stacked bar per dpsize, with whiskers showing min/max."""
    # Aggregate across all logs by dpsize.
    by_dp = defaultdict(lambda: {'grad': [], 'measure': [], 'stop': [], 'total': []})
    for log_label, per_iter in per_iter_by_log.items():
        for rec in per_iter:
            d = by_dp[rec['dpsize']]
            d['grad'].append(rec['grad_s'])
            d['measure'].append(rec['measure_s'])
            d['stop'].append(rec['stop_s'])
            d['total'].append(rec['total_s'])
    dpsizes = sorted(by_dp.keys())
    if not dpsizes:
        return False

    xs = np.arange(len(dpsizes))
    grad_meds = [np.median(by_dp[d]['grad']) for d in dpsizes]
    meas_meds = [np.median(by_dp[d]['measure']) for d in dpsizes]
    stop_meds = [np.median(by_dp[d]['stop']) for d in dpsizes]
    totals = [np.median(by_dp[d]['total']) for d in dpsizes]
    p25 = [np.percentile(by_dp[d]['total'], 25) for d in dpsizes]
    p75 = [np.percentile(by_dp[d]['total'], 75) for d in dpsizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(xs, grad_meds, label='grad (gradient probe)', color='#1f77b4')
    ax.bar(xs, meas_meds, bottom=grad_meds, label='measure_ingresses', color='#ff7f0e')
    ax.bar(xs, stop_meds, bottom=np.array(grad_meds) + np.array(meas_meds),
           label='stop_tracker (incl. solve_max_information)', color='#2ca02c')
    # Show iqr of total iter time as whiskers
    ax.errorbar(xs, totals,
                yerr=[np.array(totals) - np.array(p25),
                      np.array(p75) - np.array(totals)],
                fmt='none', color='black', capsize=4,
                label='IQR of total iter time')
    ax.set_xticks(xs)
    ax.set_xticklabels(['actual-{}'.format(d) for d in dpsizes])
    ax.set_ylabel('Median per-iter wall time (s)')
    ax.set_title('SCULPTOR sparse-iter phase breakdown by deployment size')
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    for x, t, n in zip(xs, totals, [len(by_dp[d]['total']) for d in dpsizes]):
        ax.text(x, t * 1.05, '{:.0f}s\nn={}'.format(t, n), ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_iter_time_distribution(per_iter_by_log, out_path):
    """Per-dpsize box/violin of total iter time."""
    by_dp = defaultdict(list)
    for log_label, per_iter in per_iter_by_log.items():
        for rec in per_iter:
            by_dp[rec['dpsize']].append(rec['total_s'])
    dpsizes = sorted(by_dp.keys())
    if not dpsizes:
        return False
    data = [by_dp[d] for d in dpsizes]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=['actual-{}'.format(d) for d in dpsizes],
                     showmeans=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#aec7e8')
    ax.set_ylabel('Total per-iter wall time (s)')
    ax.set_title('SCULPTOR per-iter time distribution by deployment size')
    ax.grid(True, axis='y', alpha=0.3)
    for i, d in enumerate(dpsizes, start=1):
        ax.text(i, ax.get_ylim()[1] * 0.95, 'n={}'.format(len(by_dp[d])),
                ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_per_iter_trace(per_iter_by_log, out_path):
    """Per-iter total time vs iter, colored by dpsize. Shows whether
    iter time changes as training progresses (e.g. cache warmup, drift)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    dpsize_color = {}
    cmap = plt.cm.viridis
    seen_dps = set()
    for log_label, per_iter in per_iter_by_log.items():
        for rec in per_iter:
            seen_dps.add(rec['dpsize'])
    dps_sorted = sorted(seen_dps)
    for i, d in enumerate(dps_sorted):
        dpsize_color[d] = cmap(i / max(1, len(dps_sorted) - 1))
    for log_label, per_iter in per_iter_by_log.items():
        # Plot per-dpsize trajectories
        by_dp = defaultdict(list)
        for rec in per_iter:
            by_dp[rec['dpsize']].append((rec['iter'], rec['total_s']))
        for d, points in sorted(by_dp.items()):
            points.sort()
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, '-o', color=dpsize_color[d], markersize=2,
                    label='actual-{} ({})'.format(d, log_label),
                    alpha=0.7)
    ax.set_xlabel('Training iter')
    ax.set_ylabel('Total iter wall time (s)')
    ax.set_title('SCULPTOR per-iter time trajectory')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_driver_rss_trace(parsed_logs, out_path):
    """Driver RSS over wall time per dpsize -- shows memory growth across
    training."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.viridis
    all_dps = set()
    for parsed in parsed_logs.values():
        for ev in parsed['mems']:
            if 'sweep_dpsize' in ev:
                all_dps.add(ev['sweep_dpsize'])
    dps_sorted = sorted(all_dps)
    dp_color = {d: cmap(i / max(1, len(dps_sorted) - 1)) for i, d in enumerate(dps_sorted)}

    plotted = False
    for log_label, parsed in parsed_logs.items():
        # Driver = pid with most events
        pid_counts = defaultdict(int)
        for ev in parsed['mems']:
            pid_counts[ev['pid']] += 1
        if not pid_counts: continue
        driver_pid = max(pid_counts, key=pid_counts.get)
        by_dp = defaultdict(list)
        for ev in parsed['mems']:
            if ev['pid'] != driver_pid: continue
            if ev.get('rss_mb') is None: continue
            dp = ev.get('sweep_dpsize')
            if dp is None: continue
            by_dp[dp].append((ev['t'], ev['rss_mb']))
        for dp, pts in sorted(by_dp.items()):
            pts.sort()
            if not pts: continue
            t0 = pts[0][0]
            xs = [(p[0] - t0) / 60.0 for p in pts]   # minutes from dpsize_start
            ys = [p[1] / 1024.0 for p in pts]        # GB
            ax.plot(xs, ys, '-', color=dp_color[dp], alpha=0.6,
                    label='actual-{} ({})'.format(dp, log_label))
            plotted = True
    if not plotted:
        return False
    ax.set_xlabel('Minutes since dpsize_start')
    ax.set_ylabel('Driver RSS (GB)')
    ax.set_title('SCULPTOR driver-memory growth by deployment size')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------- #
# Main                                                                    #
# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+', help='Sweep log file(s) to parse')
    ap.add_argument('--out-dir', default='figures/session_10_phase_timings')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    parsed_logs = {}
    per_iter_by_log = {}
    for path in args.logs:
        label = os.path.basename(path).replace('.log', '')
        print('Parsing {} ...'.format(path), file=sys.stderr)
        parsed = parse_log(path)
        parsed = assign_dpsize_to_mems(parsed)
        parsed_logs[label] = parsed
        timings = extract_phase_timings(parsed)
        per_iter_by_log[label] = timings['per_iter']
        print('  {} dpsize markers, {} iter-timing records'.format(
            len(parsed['dpsize_phases']), len(timings['per_iter'])),
            file=sys.stderr)
        # Print dpsize summary
        per_dp = defaultdict(list)
        for rec in timings['per_iter']:
            per_dp[rec['dpsize']].append(rec['total_s'])
        for dp in sorted(per_dp):
            print('  dpsize={:<3} n_iters={:<4} median_total={:.1f}s '
                  '(grad+measure+stop)'.format(
                      dp, len(per_dp[dp]), np.median(per_dp[dp])), file=sys.stderr)

    # Write CSV of all per-iter records for any offline analysis.
    csv_path = os.path.join(args.out_dir, 'per_iter_timings.csv')
    with open(csv_path, 'w') as f:
        f.write('log,dpsize,iter,grad_s,measure_s,stop_s,total_s\n')
        for log_label, per_iter in per_iter_by_log.items():
            for rec in per_iter:
                f.write('{},{},{},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(
                    log_label, rec['dpsize'], rec['iter'],
                    rec['grad_s'], rec['measure_s'], rec['stop_s'], rec['total_s']))
    print('Wrote {}'.format(csv_path), file=sys.stderr)

    # Plots.
    p1 = os.path.join(args.out_dir, 'phase_breakdown_by_dpsize.pdf')
    if plot_phase_breakdown_by_dpsize(per_iter_by_log, p1):
        print('Wrote {}'.format(p1), file=sys.stderr)
    p2 = os.path.join(args.out_dir, 'iter_time_distribution.pdf')
    if plot_iter_time_distribution(per_iter_by_log, p2):
        print('Wrote {}'.format(p2), file=sys.stderr)
    p3 = os.path.join(args.out_dir, 'per_iter_trace.pdf')
    if plot_per_iter_trace(per_iter_by_log, p3):
        print('Wrote {}'.format(p3), file=sys.stderr)
    p4 = os.path.join(args.out_dir, 'driver_rss_trace.pdf')
    if plot_driver_rss_trace(parsed_logs, p4):
        print('Wrote {}'.format(p4), file=sys.stderr)


if __name__ == '__main__':
    main()
