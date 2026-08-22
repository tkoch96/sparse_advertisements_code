#!/usr/bin/env python3
"""Pull SCULPTOR cluster logs, persist timing stats to a local SQLite DB, and
regenerate a refreshing dashboard of plots. Designed to be run on a cron
(every ~10 min) from the local Mac while a sweep runs on AWS.

What it captures
----------------
1. Driver per-iter phase timing (grad / measure / stop / total), per dpsize.
   Parsed from the `[mem] tag=iter_*` lines via plot_phase_timings.parse_log
   (we reuse the existing, tested parser rather than duplicating it).

2. Per-worker per-computation LP-solve timing. The worker actors print a
   `Worker N timing summary` block once per gradient batch (worker 0 only;
   Ray dedups the rest). Each block is a PER-BATCH snapshot (the ray actor
   zeroes self.timing at the top of each batch -- see
   path_distribution_computer_ray._cmd_calc_compressed_lb), so each summary
   is an independent sample of where that worker's batch time went
   (solve_generic_lp_persistent, sim_rti, total_rti_calc, ...). We store one
   row per (block, key) and plot each key over the batch sequence so you can
   watch how a code change moves a specific sub-step's cost over time.

Persistence: ~/sculptor_dashboard/sculptor_timings.db (SQLite). Logs are
mirrored into ~/sculptor_dashboard/raw/ so they survive a cluster teardown.

Usage:
  python cluster/cluster_dashboard.py                 # pull + ingest + plot + html (cron entry)
  python cluster/cluster_dashboard.py --no-pull       # re-ingest mirrored logs + replot
  python cluster/cluster_dashboard.py --plot-only     # replot from the DB only
  python cluster/cluster_dashboard.py --ingest LOG    # ingest one local log file (dev)
"""
import argparse
import glob
import os
import re
import sqlite3
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')   # headless: must precede any pyplot import (incl. ppt)
import matplotlib.pyplot as plt
import numpy as np

# Reuse the existing, tested driver-phase parser/plotter.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_phase_timings as ppt   # noqa: E402

# --------------------------------------------------------------------------- #
# Paths / config
# --------------------------------------------------------------------------- #
DASH_DIR = os.path.expanduser('~/sculptor_dashboard')
RAW_DIR = os.path.join(DASH_DIR, 'raw')
# Drop any older / off-cluster sweep logs here (e.g. session-9 forensics with
# dpsize 3-20) and they're ingested into the DB on every refresh -- so the
# dashboard's cross-size view isn't limited to whatever is on the live head.
EXTRA_LOGS_DIR = os.path.join(DASH_DIR, 'extra_logs')
DB_PATH = os.path.join(DASH_DIR, 'sculptor_timings.db')
PLOT_DIR = os.path.join(DASH_DIR, 'plots')
HTML_PATH = os.path.join(DASH_DIR, 'index.html')

AWS_BIN = os.path.expanduser('~/Documents/venv312/bin/aws')
SSH_KEY = os.path.expanduser('~/.ssh/ray-autoscaler_us-east-1.pem')
SSH_OPTS = ('-i {key} -o StrictHostKeyChecking=no -o ConnectTimeout=20 '
            '-o BatchMode=yes').format(key=SSH_KEY)

_ANSI = re.compile(r'\x1b\[[0-9;]*m')
_SUMMARY_HEADER = re.compile(
    r'\(_LocalPathDistributionComputer pid=(?P<pid>\d+),\s*ip=(?P<ip>[\d.]+)\)'
    r'\s+Worker (?P<wi>\d+) timing summary')
_SUMMARY_KEY = re.compile(
    r'\(_LocalPathDistributionComputer pid=(?P<pid>\d+),\s*ip=(?P<ip>[\d.]+)\)'
    r'\s+(?P<key>[A-Za-z0-9_]+)\s+[\d.]+%\s+\((?P<ms>[\d.]+) ms\)')
_MEM = re.compile(r'\[mem\] tag=\S+.*?\bt=(?P<t>[\d.]+)(?:.*?\biter=(?P<iter>\d+))?')
_DPSIZE = re.compile(r'\[sweep\] === dpsize=(\d+)')
_TAG = re.compile(r'^\s*tag:\s*(\S+)')
# Worker-count signals: the start_workers override (per dpsize) and any
# mid-run pool rebuilds from the recovery path.
_NW_OVERRIDE = re.compile(r'SCULPTOR_N_WORKERS override active: n_workers=(\d+)')
_NW_REBUILT = re.compile(r'\[ray-recover\] rebuilt pool with (\d+) workers')


def _run(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, '', 'TIMEOUT'


# --------------------------------------------------------------------------- #
# Pull
# --------------------------------------------------------------------------- #
def discover_ips():
    """Return (head_ip, [worker_ips]) from AWS, or (None, []) on failure."""
    rc, out, err = _run(
        '{aws} ec2 describe-instances '
        '--filters "Name=tag:project,Values=sculptor" '
        '"Name=instance-state-name,Values=running" '
        '--query "Reservations[].Instances[].[InstanceType,PublicIpAddress]" '
        '--output text'.format(aws=AWS_BIN))
    if rc != 0:
        print('[pull] AWS describe-instances failed: {}'.format(err.strip()), file=sys.stderr)
        return None, []
    head, workers = None, []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        itype, ip = parts[0], parts[1]
        if ip == 'None':
            continue
        if itype.startswith('m7g.'):
            head = ip
        elif itype.startswith('c7g.'):
            workers.append(ip)
    return head, workers


def pull_logs():
    """Mirror the head sweep logs + per-worker mem files locally. Returns the
    list of local sweep-log paths."""
    os.makedirs(os.path.join(RAW_DIR, 'head'), exist_ok=True)
    head, workers = discover_ips()
    if head is None:
        print('[pull] no head found; using whatever is already mirrored', file=sys.stderr)
    else:
        _run('scp {opts} ubuntu@{ip}:/tmp/cluster_runs/*.log {dst}/ 2>/dev/null'.format(
            opts=SSH_OPTS, ip=head, dst=os.path.join(RAW_DIR, 'head')))
        for wip in workers:
            wdst = os.path.join(RAW_DIR, 'worker_{}'.format(wip.replace('.', '_')))
            os.makedirs(wdst, exist_ok=True)
            _run('scp {opts} ubuntu@{ip}:/tmp/sculptor_worker_*.log {dst}/ 2>/dev/null'.format(
                opts=SSH_OPTS, ip=wip, dst=wdst))
    return sorted(glob.glob(os.path.join(RAW_DIR, 'head', '*.log')))


# --------------------------------------------------------------------------- #
# DB
# --------------------------------------------------------------------------- #
def init_db():
    os.makedirs(DASH_DIR, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
    CREATE TABLE IF NOT EXISTS iter_timing (
        run_tag TEXT, dpsize INT, iter INT,
        grad_s REAL, measure_s REAL, stop_s REAL, total_s REAL, iter_start_ts REAL,
        PRIMARY KEY (run_tag, dpsize, iter)
    );
    CREATE TABLE IF NOT EXISTS worker_timing (
        run_tag TEXT, ip TEXT, pid INT, worker_i INT, block_seq INT,
        nearest_iter INT, ts REAL, dpsize INT, key TEXT, ms REAL,
        PRIMARY KEY (run_tag, ip, pid, worker_i, block_seq, key)
    );
    CREATE INDEX IF NOT EXISTS idx_wt ON worker_timing (run_tag, worker_i, key, block_seq);
    CREATE TABLE IF NOT EXISTS worker_count (
        run_tag TEXT, dpsize INT, n_workers INT,
        PRIMARY KEY (run_tag, dpsize)
    );
    """)
    db.commit()
    return db


def _run_tag_for_log(path):
    """Read the `tag:` line the sweep prints in its header; fall back to the
    filename stem."""
    try:
        with open(path, errors='replace') as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                m = _TAG.match(_ANSI.sub('', line))
                if m:
                    return m.group(1)
    except OSError:
        pass
    return os.path.basename(path).replace('.log', '')


def ingest_driver_iters(db, path, run_tag):
    parsed = ppt.assign_dpsize_to_mems(ppt.parse_log(path))
    timings = ppt.extract_phase_timings(parsed)
    rows = [(run_tag, r['dpsize'], r['iter'], r['grad_s'], r['measure_s'],
             r['stop_s'], r['total_s'], r.get('iter_start_ts'))
            for r in timings['per_iter']]
    db.executemany(
        "INSERT OR REPLACE INTO iter_timing "
        "(run_tag,dpsize,iter,grad_s,measure_s,stop_s,total_s,iter_start_ts) "
        "VALUES (?,?,?,?,?,?,?,?)", rows)
    db.commit()
    return len(rows)


def ingest_worker_count(db, path, run_tag):
    """Capture the active worker count per dpsize. Today this is the
    start_workers override (constant within a run unless the adaptive ramp /
    autoscale / recovery-rebuild changes it). Stored per (run_tag, dpsize) so
    cross-size and cross-run normalization knows N(iter)."""
    cur_dp = None
    seen = {}
    with open(path, errors='replace') as f:
        for raw in f:
            line = _ANSI.sub('', raw)
            m = _DPSIZE.search(line)
            if m:
                cur_dp = int(m.group(1)); continue
            m = _NW_OVERRIDE.search(line) or _NW_REBUILT.search(line)
            if m and cur_dp is not None:
                seen[cur_dp] = int(m.group(1))   # last wins (latest count for dp)
    rows = [(run_tag, dp, n) for dp, n in seen.items()]
    db.executemany("INSERT OR REPLACE INTO worker_count (run_tag,dpsize,n_workers) "
                   "VALUES (?,?,?)", rows)
    db.commit()
    return len(rows)


def ingest_worker_timing(db, path, run_tag):
    """Single pass: track current dpsize/iter/ts from driver [mem] lines, and
    assign each 'Worker N timing summary' block a stable per-(pid,worker) seq."""
    cur_iter, cur_t, cur_dp = None, None, None
    block_counts = {}        # (pid, wi) -> count of summaries seen so far
    cur_block = None         # (pid, wi, seq)
    rows = []
    with open(path, errors='replace') as f:
        for raw in f:
            line = _ANSI.sub('', raw)
            m = _DPSIZE.search(line)
            if m:
                cur_dp = int(m.group(1)); continue
            m = _MEM.search(line)
            if m:
                cur_t = float(m.group('t'))
                if m.group('iter') is not None:
                    cur_iter = int(m.group('iter'))
                continue
            m = _SUMMARY_HEADER.search(line)
            if m:
                pid, ip, wi = int(m.group('pid')), m.group('ip'), int(m.group('wi'))
                seq = block_counts.get((pid, wi), 0)
                block_counts[(pid, wi)] = seq + 1
                cur_block = (pid, ip, wi, seq)
                continue
            m = _SUMMARY_KEY.search(line)
            if m and cur_block is not None:
                pid, ip, wi, seq = cur_block
                if int(m.group('pid')) == pid:
                    rows.append((run_tag, ip, pid, wi, seq, cur_iter, cur_t,
                                 cur_dp, m.group('key'), float(m.group('ms'))))
    db.executemany(
        "INSERT OR IGNORE INTO worker_timing "
        "(run_tag,ip,pid,worker_i,block_seq,nearest_iter,ts,dpsize,key,ms) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    db.commit()
    return len(rows)


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_worker_timing(db, out_path, run_tag=None):
    """Per-computation-key time over the batch sequence, for worker 0 (the
    one that emits). One line per key; x = block sequence (≈ gradient batch /
    iter). This is the 'how does each sub-step's cost evolve' view."""
    where, params = '', []
    if run_tag:
        where, params = 'WHERE run_tag=?', [run_tag]
    cur = db.execute(
        "SELECT worker_i, block_seq, key, ms, nearest_iter FROM worker_timing "
        + where + " ORDER BY block_seq", params)
    rows = cur.fetchall()
    if not rows:
        return False
    # worker_i -> key -> list of (block_seq, ms, nearest_iter)
    by_key = {}
    iters_by_block = {}
    for wi, seq, key, ms, nit in rows:
        by_key.setdefault(key, []).append((seq, ms))
        if nit is not None:
            iters_by_block[seq] = nit
    keys = sorted(by_key, key=lambda k: -sum(v for _, v in by_key[k]))  # biggest first

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    cmap = plt.cm.tab10
    for i, key in enumerate(keys):
        pts = sorted(by_key[key])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, '-o', ms=2.5, lw=1.3, color=cmap(i % 10), label=key)
    ax.set_xlabel('worker-0 batch # (one LP batch per gradient/max-info probe, ≈ per training iter)')
    ax.set_ylabel('worker-0 time spent in that computation, per batch (ms)')
    ax.set_title('Worker-0 per-computation LP time over the run  [run={}]'.format(run_tag or 'all'))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    # Right axis: cluster-total = per-worker × N_active. The worker summary is
    # one worker, so ×N is the whole-cluster compute burned per category.
    n_active = db.execute(
        "SELECT MAX(n_workers) FROM worker_count WHERE run_tag=?", [run_tag]).fetchone()
    n_active = (n_active[0] if n_active else None) or 1
    secax = ax.secondary_yaxis(
        'right', functions=(lambda y: y * n_active, lambda y: y / n_active))
    secax.set_ylabel('cluster total = worker-0 × N_active({}) (ms)'.format(n_active))

    # Stacked-share view: what fraction of each batch goes to each key.
    blocks = sorted({s for v in by_key.values() for s, _ in v})
    keymat = {k: dict(by_key[k]) for k in keys}
    bottoms = np.zeros(len(blocks))
    for i, key in enumerate(keys):
        ys = np.array([keymat[key].get(b, 0.0) for b in blocks])
        ax2.bar(range(len(blocks)), ys, bottom=bottoms, color=cmap(i % 10),
                width=1.0, label=key)
        bottoms += ys
    ax2.set_xlabel('worker-0 batch # (≈ per training iter)')
    ax2.set_ylabel('worker-0 time per batch (ms), stacked by computation')
    ax2.set_title('Worker-0 batch composition (stacked)')
    ax2.legend(fontsize=7, ncol=2, loc='upper right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def _sweep_logs():
    """All logs to ingest: mirrored head logs (minus the latest.log symlink
    dup) plus any off-cluster historical logs dropped in EXTRA_LOGS_DIR."""
    head = [p for p in sorted(glob.glob(os.path.join(RAW_DIR, 'head', '*.log')))
            if os.path.basename(p) != 'latest.log']
    extra = sorted(glob.glob(os.path.join(EXTRA_LOGS_DIR, '*.log')))
    return head + extra


def _driver_per_iter_all_dpsizes(db):
    """One clean per-iter series per dpsize, taken from the most-recent run
    that covers that dpsize (so dp25/32 don't blend the resilience-on and
    resilience-off regimes, and dp3-20 come from whatever historical run has
    them). This is the cross-size scaling view."""
    pairs = db.execute("""
        SELECT dpsize, run_tag FROM iter_timing it
        WHERE COALESCE(iter_start_ts,0) = (
            SELECT MAX(COALESCE(iter_start_ts,0)) FROM iter_timing i2
            WHERE i2.dpsize = it.dpsize)
        GROUP BY dpsize""").fetchall()
    per_iter = []
    for dp, tag in pairs:
        for r in db.execute(
                "SELECT iter,grad_s,measure_s,stop_s,total_s FROM iter_timing "
                "WHERE dpsize=? AND run_tag=? ORDER BY iter", [dp, tag]):
            per_iter.append({'dpsize': dp, 'iter': r[0], 'grad_s': r[1],
                             'measure_s': r[2], 'stop_s': r[3], 'total_s': r[4]})
    return per_iter


def _driver_per_iter_total_worker(db):
    """Same as _driver_per_iter_all_dpsizes but each phase time is MULTIPLIED by
    the active worker count -> total worker-time (aggregate worker-seconds the
    whole cluster spent in that phase). This is the worker-count-invariant
    measure: adding workers shortens wall-clock but leaves the total work ~flat,
    so it's the honest cross-dpsize / cross-run quantity. Skips a dpsize whose
    worker count is unknown."""
    pairs = db.execute("""
        SELECT dpsize, run_tag FROM iter_timing it
        WHERE COALESCE(iter_start_ts,0) = (
            SELECT MAX(COALESCE(iter_start_ts,0)) FROM iter_timing i2
            WHERE i2.dpsize = it.dpsize)
        GROUP BY dpsize""").fetchall()
    per_iter = []
    for dp, tag in pairs:
        nrow = db.execute("SELECT n_workers FROM worker_count "
                          "WHERE run_tag=? AND dpsize=?", [tag, dp]).fetchone()
        if not nrow or not nrow[0]:
            continue
        n = float(nrow[0])
        for r in db.execute(
                "SELECT iter,grad_s,measure_s,stop_s,total_s FROM iter_timing "
                "WHERE dpsize=? AND run_tag=? ORDER BY iter", [dp, tag]):
            per_iter.append({'dpsize': dp, 'iter': r[0], 'grad_s': r[1] * n,
                             'measure_s': r[2] * n, 'stop_s': r[3] * n,
                             'total_s': r[4] * n})
    return per_iter


def _active_tag(db):
    """Active run = the one with the most recent iter timestamp in the DB.
    Robust to scp rewriting local file mtimes (which broke mtime-based
    detection). Falls back to worker_timing's newest ts if no iter rows."""
    row = db.execute("SELECT run_tag FROM iter_timing "
                     "WHERE iter_start_ts=(SELECT MAX(iter_start_ts) FROM iter_timing)").fetchone()
    if row:
        return row[0]
    row = db.execute("SELECT run_tag FROM worker_timing "
                     "WHERE ts=(SELECT MAX(ts) FROM worker_timing)").fetchone()
    return row[0] if row else None


def plot_workers_active(db, out_path, run_tag):
    """Active worker count over the run's iters. For each dpsize the count is
    the start_workers/recovery value (constant within a dpsize today); this
    becomes a real time series once the adaptive ramp / autoscale moves N."""
    rows = db.execute(
        "SELECT it.dpsize, it.iter, wc.n_workers FROM iter_timing it "
        "JOIN worker_count wc ON wc.run_tag=it.run_tag AND wc.dpsize=it.dpsize "
        "WHERE it.run_tag=? ORDER BY it.dpsize, it.iter", [run_tag]).fetchall()
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(12, 3.2))
    cmap = plt.cm.tab10
    for i, dp in enumerate(sorted({r[0] for r in rows})):
        pts = [(r[1], r[2]) for r in rows if r[0] == dp]
        ax.step([p[0] for p in pts], [p[1] for p in pts], where='post',
                color=cmap(i % 10), lw=1.8, label='dpsize={}'.format(dp))
    ax.set_xlabel('training iteration')
    ax.set_ylabel('active Ray workers (N)')
    ax.set_ylim(bottom=0)
    ax.set_title('Active worker count over time  [run={}]'.format(run_tag))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def regenerate_plots(db, active_tag):
    """Plots focus on the ACTIVE run (sourced from the DB) for a clean live
    view; the DB retains all runs for cross-run comparison."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    made = []
    # TOTAL WORKER-TIME driver dashboard FIRST (phase × N_active): aggregate
    # worker-seconds, the worker-count-invariant view for fair cross-dpsize
    # comparison (a run with more workers has shorter wall-clock but ~same
    # total work).
    per_iter_tw = _driver_per_iter_total_worker(db)
    if per_iter_tw:
        bytw = {'total-worker-time': per_iter_tw}
        sfx = '   — TOTAL WORKER-TIME (driver phase × N_active)'
        vlab = 'total worker-time per iter (s) [= wall-clock × N_active]'
        p = os.path.join(PLOT_DIR, 'driver_dashboard_total_worker.png')
        if ppt.plot_combined_dashboard(bytw, p, title_suffix=sfx, value_label=vlab):
            made.append('driver_dashboard_total_worker.png')
        p2 = os.path.join(PLOT_DIR, 'driver_phases_over_iter_total_worker.png')
        if ppt.plot_phases_over_iter_condensed(bytw, p2, title_suffix=sfx, value_label=vlab):
            made.append('driver_phases_over_iter_total_worker.png')
    # Raw (wall-clock) driver dashboard: ALL dpsizes (latest run per size).
    per_iter = _driver_per_iter_all_dpsizes(db)
    if per_iter:
        by_log = {'all-runs': per_iter}
        vlab = 'wall-clock per iter (s)'
        p = os.path.join(PLOT_DIR, 'driver_dashboard.png')
        if ppt.plot_combined_dashboard(by_log, p, title_suffix='   — raw wall-clock', value_label=vlab):
            made.append('driver_dashboard.png')
        p2 = os.path.join(PLOT_DIR, 'driver_phases_over_iter.png')
        if ppt.plot_phases_over_iter_condensed(by_log, p2, title_suffix='   — raw wall-clock', value_label=vlab):
            made.append('driver_phases_over_iter.png')
    # Worker plot: active run only (per-run "did my change help" view).
    if active_tag:
        p = os.path.join(PLOT_DIR, 'worker0_timing.png')
        if plot_worker_timing(db, p, run_tag=active_tag):
            made.append('worker0_timing.png')
        pw = os.path.join(PLOT_DIR, 'workers_active.png')
        if plot_workers_active(db, pw, active_tag):
            made.append('workers_active.png')
    return made


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def write_html(made, status):
    imgs = '\n'.join(
        '<h2>{0}</h2><img src="plots/{0}?_={1}" style="max-width:100%;">'.format(name, int(time.time()))
        for name in made)
    html = """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="60">
<title>SCULPTOR dashboard</title>
<style>body{{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;background:#fafafa;color:#222}}
h2{{margin-top:28px}} .status{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 16px;white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace}}</style>
</head><body>
<h1>SCULPTOR cluster dashboard</h1>
<div class="status">{status}</div>
{imgs}
</body></html>""".format(status=status, imgs=imgs)
    with open(HTML_PATH, 'w') as f:
        f.write(html)


def build_status(db, sweep_logs):
    parts = ['updated {} UTC'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))]
    # Active run = the one with the most recent iter timestamp.
    row = db.execute(
        "SELECT run_tag, dpsize, iter FROM iter_timing "
        "WHERE iter_start_ts = (SELECT MAX(iter_start_ts) FROM iter_timing)").fetchone()
    if row:
        parts.append('active: run={} dpsize={} iter≈{}'.format(row[0], row[1], row[2]))
    n_wt = db.execute("SELECT COUNT(*) FROM worker_timing").fetchone()[0]
    n_it = db.execute("SELECT COUNT(*) FROM iter_timing").fetchone()[0]
    parts.append('db rows: iter_timing={} worker_timing={}'.format(n_it, n_wt))
    parts.append('logs mirrored: {}'.format(', '.join(os.path.basename(p) for p in sweep_logs)))
    return '\n'.join(parts)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-pull', action='store_true', help='Skip the scp pull; ingest mirrored logs.')
    ap.add_argument('--plot-only', action='store_true', help='Only replot from the DB.')
    ap.add_argument('--ingest', metavar='LOG', help='Ingest a single local log and exit (dev).')
    args = ap.parse_args()

    db = init_db()

    if args.ingest:
        tag = _run_tag_for_log(args.ingest)
        ni = ingest_driver_iters(db, args.ingest, tag)
        nw = ingest_worker_timing(db, args.ingest, tag)
        print('[ingest] {}: tag={} iter_rows={} worker_rows={}'.format(args.ingest, tag, ni, nw))
        return 0

    if args.plot_only:
        sweep_logs = _sweep_logs()
        made = regenerate_plots(db, _active_tag(db))
        write_html(made, build_status(db, sweep_logs))
        print('[plot-only] regenerated: {}'.format(made))
        return 0

    if not args.no_pull:
        pull_logs()
    sweep_logs = _sweep_logs()
    for path in sweep_logs:                       # ingest ALL runs -> DB history
        tag = _run_tag_for_log(path)
        ingest_driver_iters(db, path, tag)
        ingest_worker_timing(db, path, tag)
        ingest_worker_count(db, path, tag)
    active_tag = _active_tag(db)
    made = regenerate_plots(db, active_tag)
    write_html(made, build_status(db, sweep_logs))
    print('[refresh] logs={} active_tag={} plots={}'.format(
        len(sweep_logs), active_tag, made))
    return 0


if __name__ == '__main__':
    sys.exit(main())
