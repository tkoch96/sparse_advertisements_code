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
    python cluster/plot_phase_timings.py LOG [LOG ...]
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
def _dpsize_palette(dpsizes):
    """Stable color per dpsize across all subplots."""
    cmap = plt.cm.viridis
    n = max(1, len(dpsizes) - 1)
    return {d: cmap(i / n) for i, d in enumerate(sorted(dpsizes))}


def _rolling_median(xs, ys, window=5):
    """Rolling median over a uniform iter axis. xs must be sorted."""
    if len(ys) < window:
        return xs, ys
    half = window // 2
    sm = []
    for i in range(len(ys)):
        lo = max(0, i - half)
        hi = min(len(ys), i + half + 1)
        sm.append(float(np.median(ys[lo:hi])))
    return xs, sm


def _ylim_clip(values, lo_quantile=0.0, hi_quantile=0.99, pad=0.05):
    """Compute (ymin, ymax) clipped to percentiles to suppress outliers.
    Negative values are dropped (a negative duration means clock skew or
    parser miscount and is never meaningful)."""
    arr = np.array([v for v in values if v is not None and v >= 0])
    if arr.size == 0:
        return (0, 1)
    lo = max(0.0, float(np.quantile(arr, lo_quantile)))
    hi = float(np.quantile(arr, hi_quantile))
    span = max(1e-6, hi - lo)
    return (max(0, lo - pad * span), hi + pad * span)


def plot_combined_dashboard(per_iter_by_log, out_path, title_suffix='',
                            value_label='wall-clock per iter (s)'):
    """Single combined figure with:
      - Top-left:   stacked bar phase breakdown (median per-iter) by dpsize
      - Top-right:  total iter-time distribution by dpsize (box, outliers off)
      - Mid row:    per-iter trajectories for grad and measure phases
      - Bot row:    per-iter trajectories for stop and total phases
    Trajectories show raw points (small) + rolling median (window=5) per
    dpsize. y-axes clipped to the 99th percentile (per phase) so a few
    long-tail iters don't compress the rest of the trace.
    """
    # Aggregate (filter negatives -- a negative duration means the parser
    # crossed a sweep boundary and is not meaningful).
    by_dp = defaultdict(lambda: {'grad': [], 'measure': [], 'stop': [], 'total': [],
                                  'iter_grad': [], 'iter_meas': [], 'iter_stop': [],
                                  'iter_total': []})
    for per_iter in per_iter_by_log.values():
        for rec in per_iter:
            d = by_dp[rec['dpsize']]
            if rec['grad_s']     >= 0: d['grad'].append(rec['grad_s']);    d['iter_grad'].append(rec['iter'])
            if rec['measure_s']  >= 0: d['measure'].append(rec['measure_s']); d['iter_meas'].append(rec['iter'])
            if rec['stop_s']     >= 0: d['stop'].append(rec['stop_s']);    d['iter_stop'].append(rec['iter'])
            if rec['total_s']    >= 0: d['total'].append(rec['total_s']);  d['iter_total'].append(rec['iter'])
    dpsizes = sorted(by_dp.keys())
    if not dpsizes:
        return False
    color = _dpsize_palette(dpsizes)

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.40, wspace=0.22)

    # ---- top-left: stacked bar by dpsize (median grad/measure/stop) ----
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(dpsizes))
    grad_meds = [np.median(by_dp[d]['grad'])    if by_dp[d]['grad']    else 0 for d in dpsizes]
    meas_meds = [np.median(by_dp[d]['measure']) if by_dp[d]['measure'] else 0 for d in dpsizes]
    stop_meds = [np.median(by_dp[d]['stop'])    if by_dp[d]['stop']    else 0 for d in dpsizes]
    totals    = [np.median(by_dp[d]['total'])   if by_dp[d]['total']   else 0 for d in dpsizes]
    ax.bar(xs, grad_meds, label='grad', color='#1f77b4')
    ax.bar(xs, meas_meds, bottom=grad_meds, label='measure', color='#ff7f0e')
    ax.bar(xs, stop_meds,
           bottom=np.array(grad_meds) + np.array(meas_meds),
           label='stop (incl. max_information)', color='#2ca02c')
    ax.set_xticks(xs)
    ax.set_xticklabels(['{}'.format(d) for d in dpsizes])
    ax.set_xlabel('deployment size (popps)')
    ax.set_ylabel('median ' + value_label)
    ax.set_title('phase breakdown by dpsize (median)')
    ax.set_ylim(0, max(totals) * 1.18 if totals else 1)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    for x, t, n in zip(xs, totals, [len(by_dp[d]['total']) for d in dpsizes]):
        ax.text(x, t + 0.02 * max(totals), '{:.0f}s\nn={}'.format(t, n),
                ha='center', fontsize=8)

    # ---- top-right: boxplot of total iter time (outliers suppressed) ----
    ax = fig.add_subplot(gs[0, 1])
    data = [by_dp[d]['total'] for d in dpsizes]
    bp = ax.boxplot(data, tick_labels=[str(d) for d in dpsizes],
                    showmeans=True, patch_artist=True, showfliers=False)
    for patch, d in zip(bp['boxes'], dpsizes):
        patch.set_facecolor(color[d])
        patch.set_alpha(0.55)
    ax.set_xlabel('deployment size (popps)')
    ax.set_ylabel('total-iter ' + value_label)
    ax.set_title('total iter-time distribution (outliers clipped)')
    all_totals = [v for d in dpsizes for v in by_dp[d]['total']]
    ax.set_ylim(_ylim_clip(all_totals, hi_quantile=0.99))
    ax.grid(True, axis='y', alpha=0.3)

    # ---- mid/bot rows: per-iter trajectories, one panel per phase ----
    phase_specs = [
        ('grad',    'iter_grad',  'gradient probe (grad_s)',           gs[1, 0]),
        ('measure', 'iter_meas',  'measure_ingresses (measure_s)',     gs[1, 1]),
        ('stop',    'iter_stop',  'stop_tracker incl. max_info (stop_s)', gs[2, 0]),
        ('total',   'iter_total', 'total iter time (total_s)',         gs[2, 1]),
    ]
    for key, iter_key, title, gsspec in phase_specs:
        ax = fig.add_subplot(gsspec)
        all_vals = []
        for d in dpsizes:
            vals = by_dp[d][key]
            its = by_dp[d][iter_key]
            if not vals:
                continue
            # Sort by iter for the rolling median line.
            order = np.argsort(its)
            its_s = [its[i] for i in order]
            vals_s = [vals[i] for i in order]
            all_vals.extend(vals_s)
            ax.scatter(its_s, vals_s, color=color[d], s=6, alpha=0.35)
            xs_sm, ys_sm = _rolling_median(its_s, vals_s, window=5)
            ax.plot(xs_sm, ys_sm, color=color[d], linewidth=1.6,
                    label='dpsize={}  (n={})'.format(d, len(vals_s)))
        ax.set_xlabel('training iteration')
        ax.set_ylabel(value_label)
        ax.set_title(title)
        ax.set_ylim(_ylim_clip(all_vals, hi_quantile=0.99))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right', ncol=1)

    fig.suptitle('SCULPTOR phase-timing dashboard '
                 '(points = per-iter; lines = 5-iter rolling median; '
                 'y-axes clipped to p99)' + title_suffix,
                 fontsize=11)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_phases_over_iter_condensed(per_iter_by_log, out_path, title_suffix='',
                                    value_label='wall-clock per iter (s)'):
    """Single-panel-per-phase compact: shows grad/measure/stop/total on a
    single axis with all dpsizes overlaid. Same data as the dashboard's
    bottom four panels but tighter -- one figure, 4 stacked subplots."""
    by_dp = defaultdict(lambda: {'iter': [], 'grad': [], 'measure': [],
                                  'stop': [], 'total': []})
    for per_iter in per_iter_by_log.values():
        for rec in per_iter:
            d = by_dp[rec['dpsize']]
            d['iter'].append(rec['iter'])
            d['grad'].append(max(0, rec['grad_s']))
            d['measure'].append(max(0, rec['measure_s']))
            d['stop'].append(max(0, rec['stop_s']))
            d['total'].append(max(0, rec['total_s']))
    dpsizes = sorted(by_dp.keys())
    if not dpsizes:
        return False
    color = _dpsize_palette(dpsizes)

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    phases = [('grad',    'gradient probe (grad_s)'),
              ('measure', 'measure_ingresses (measure_s)'),
              ('stop',    'stop_tracker incl. max_information (stop_s)'),
              ('total',   'total iter time (total_s)')]
    for ax, (key, title) in zip(axes, phases):
        all_vals = []
        for d in dpsizes:
            its = by_dp[d]['iter']
            vals = by_dp[d][key]
            order = np.argsort(its)
            its_s = [its[i] for i in order]
            vals_s = [vals[i] for i in order]
            all_vals.extend(vals_s)
            ax.scatter(its_s, vals_s, color=color[d], s=6, alpha=0.30)
            xs_sm, ys_sm = _rolling_median(its_s, vals_s, window=5)
            ax.plot(xs_sm, ys_sm, color=color[d], linewidth=1.8,
                    label='dpsize={}'.format(d))
        ax.set_ylabel(title + '\n' + value_label, fontsize=9)
        ax.set_ylim(_ylim_clip(all_vals, hi_quantile=0.99))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=len(dpsizes), loc='upper right')
    axes[-1].set_xlabel('training iter')
    fig.suptitle('SCULPTOR phase time over training iters '
                 '(points = raw per-iter; lines = 5-iter rolling median; '
                 'y-axes clipped to p99)' + title_suffix,
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)
    return True


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

    # Plots. The two new "combined" outputs are the headline figures
    # (per user request: everything on one figure, axes cleaned up,
    # per-phase trajectories shown across iters). The standalone PDFs
    # are kept for backward-compatibility with anyone bookmarking them.
    p_dash = os.path.join(args.out_dir, 'dashboard.pdf')
    if plot_combined_dashboard(per_iter_by_log, p_dash):
        print('Wrote {}'.format(p_dash), file=sys.stderr)
    p_traj = os.path.join(args.out_dir, 'phases_over_iter.pdf')
    if plot_phases_over_iter_condensed(per_iter_by_log, p_traj):
        print('Wrote {}'.format(p_traj), file=sys.stderr)

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
