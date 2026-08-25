"""Timing figures for a cluster run, drawn from its harvested log.

    python -m dashboard.plot_cluster_timing <run_id>       # one run
    python -m dashboard.plot_cluster_timing --all          # every run

Reads only local files under `cache/cluster_runs/<run_id>/` -- it never
touches the VM, so it is safe to call from the refresh loop at any
cadence. Writes PNGs to `figures/dashboards/cluster/<run_id>/`, which the
dashboard tab picks up by directory listing (no filename registry).

Three panels, each drawn only if its data exists:

* **wall time per deployment size** -- the number the whole timing
  investigation is for, plus sec/sim so sizes with different nsim are
  comparable. This is what seeds the pricing dashboard.
* **driver RSS + system available memory** -- from the `[mem]` markers
  evaluate_over_deployment_sizes emits at each size boundary. RSS climbing
  across sizes is the signature of the leak that OOM-killed dpsize=25.
* **free disk** -- from the launcher's sysmon sampler. A downward slope
  that reaches the floor is the failure that has cost us the most compute.
"""

from __future__ import annotations

import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO, 'cache', 'cluster_runs')
FIG_ROOT = os.path.join(REPO, 'figures', 'dashboards', 'cluster')

# Same shape as cluster/plot_phase_timings.py's MEM_RE: vms_mb/peak_mb are
# optional because the sweep driver's own markers omit them while SAS's
# per-iteration markers carry them. Matching only the short form silently
# dropped 95% of the markers and drew a two-point memory plot.
MEM_RE = re.compile(
    r'\[mem\] tag=(?P<tag>\S+)\s+'
    r'(?:rss_mb=(?P<rss>-?\d+)\s+)?'
    r'(?:vms_mb=(?P<vms>-?\d+)\s+)?'
    r'(?:peak_mb=(?P<peak>-?\d+)\s+)?'
    r'sys_avail_mb=(?P<avail>-?\d+)\s+'
    # the 2026-08-25 instrumentation inserted shm/workers columns here;
    # without these optional groups every new-format marker silently
    # failed to match and the memory plot froze (Tom: "i dont see the
    # memory problems you talk about cause the dash is stale")
    r'(?:shm_mb=(?P<shm>-?\d+)\s+)?'
    r'(?:workers_rss_mb=(?P<wrss>-?\d+)\s+)?'
    r'pid=(?P<pid>\d+)\s+t=(?P<t>[\d.]+)(?P<extra>.*)')

ITER_RE = re.compile(r'LEARNING ITERATION\s*:\s*(\d+)')
TIMER_RE = re.compile(r'Timer:\s+(\w+)\s+--\s+([\d.]+)\s*s')
# Worker-0 per-batch breakdown, e.g.
#   [wt] w=0 total=2.5s lp_persistent=39% paths=16% sim_rti=13% ...
WT_RE = re.compile(r'\[wt\] w=0 total=([\d.]+)s\s+(.*)')
WT_PART_RE = re.compile(r'(\w+)=(\d+)%')
# `lp_incl` is the INCLUSIVE share of the whole LP block (parent + its
# children) and is reported alongside the exclusive parts for readability.
# Stacking it with them would double-count the LP work and push the bar
# past 100%. It is a summary figure, not a slice.
WT_NOT_A_SLICE = {'lp_incl'}
# `mc=` and `lp_solves=` are counts, not percentages; the part regex only
# matches `key=NN%` so they are ignored, but named here for the reader.
# One-time startup accounting, emitted once per worker after the
# deployment lands. Separate from [wt], which is reset every batch.
WTI_RE = re.compile(r'\[wt-init\] w=(\d+) tag=(\S+) total=([\d.]+)s\s+(.*)')
WTI_PART_RE = re.compile(r'(\w+)=([\d.]+)s')
# `[wt-init]` is emitted at post_update_deployment, but rb_backups is built
# LAZILY on the first resilience call -- after that point. So the aggregate
# under-reports startup by exactly the most expensive item (27.9s vs the
# 2.9s aggregate at actual-25). `[wt-init-part]` carries the late pieces;
# merge them in rather than leaving the headline wrong. It has no worker
# index (it is emitted from Optimal_Adv_Wrapper, which does not know one)
# and Ray dedups it, so it is treated as a PER-WORKER cost measured once.
WTI_LATE_RE = re.compile(r'\[wt-init-part\]\s+(\w+)=([\d.]+)s')


def _log(run_id):
    p = os.path.join(RUNS_DIR, run_id, 'logs', 'run.log')
    try:
        return open(p, errors='replace').read()
    except IOError:
        return ''


def _progress(run_id):
    p = os.path.join(RUNS_DIR, run_id, 'logs', 'progress.json')
    try:
        return json.load(open(p))
    except (IOError, ValueError):
        return None


def _sysmon(run_id):
    p = os.path.join(RUNS_DIR, run_id, 'logs', 'sysmon.jsonl')
    rows = []
    try:
        for line in open(p, errors='replace'):
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    except IOError:
        pass
    return rows


def _style(ax):
    ax.grid(alpha=.25, linewidth=.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def plot_sizes(run_id, outdir):
    p = _progress(run_id)
    if not p or not p.get('done'):
        return None
    sizes, wall, per_sim, n_cached = [], [], [], 0
    for s in p.get('sizes', []):
        e = p['done'].get(str(s))
        if not e or not e.get('ok'):
            continue
        if e.get('cached'):
            # Cache hits are excluded, not plotted-and-annotated: a ~1s
            # point at a large size would bend any curve fitted through
            # it, and this figure exists to seed the cost model.
            n_cached += 1
            continue
        sizes.append(s)
        wall.append(e['wall_s'] / 60.0)
        per_sim.append(e.get('sec_per_sim', e['wall_s']) / 60.0)
    if not sizes:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(sizes, wall, 'o-', label='wall per size', color='#31647f')
    if any(a != b for a, b in zip(wall, per_sim)):
        ax.plot(sizes, per_sim, 's--', label='per simulation',
                color='#2f9e6e')
    ax.set_xlabel('deployment size (num sites)')
    ax.set_ylabel('minutes')
    title = '{}: eval wall time by deployment size'.format(run_id)
    if n_cached:
        title += '\n({} cached size(s) excluded -- not measurements)'.format(
            n_cached)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    _style(ax)
    out = os.path.join(outdir, 'wall_by_size.png')
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


_PHASE_COLORS = {'setup': '#6d7478', 'sparse_init': '#8fb8cc',
                 'learning': '#31647f', 'evals_total': '#2f9e6e',
                 'residual': '#d9d5cc'}
_EVAL_COLORS = ['#2f9e6e', '#c9862b', '#31647f', '#a15c8e', '#8b9296']


def plot_phases(run_id, outdir):
    """Where each size's wall-clock went: setup / init / learning / evals.

    Two panels. Left is the whole-size breakdown; right is the eval stage
    split, because on small deployments the evals dominate the size
    entirely and a single stacked bar hides which eval is responsible.
    """
    from dashboard import cluster_phases as cp
    recs = cp.parse(run_id)
    recs = {k: v for k, v in recs.items() if v.get('total_s', 0) > 0}
    if not recs:
        return None
    sizes = sorted(recs)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))

    bottom = [0.0] * len(sizes)
    for key, label in cp.PHASES:
        vals = [recs[s].get(key, 0.0) / 60.0 for s in sizes]
        if not any(vals):
            continue
        ax.bar(range(len(sizes)), vals, bottom=bottom, label=label,
               color=_PHASE_COLORS.get(key, '#999'), width=.7)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(['actual-{}'.format(s) for s in sizes], fontsize=7)
    ax.set_ylabel('minutes')
    ax.set_title('{}: wall-clock by phase'.format(run_id), fontsize=9)
    ax.legend(fontsize=7)
    _style(ax)

    stage_names = [lbl for _t, lbl in cp.EVAL_STAGES]
    bottom = [0.0] * len(sizes)
    for i, lbl in enumerate(stage_names):
        vals = [recs[s]['evals'].get(lbl, 0.0) / 60.0 for s in sizes]
        if not any(vals):
            continue
        ax2.bar(range(len(sizes)), vals, bottom=bottom, label=lbl,
                color=_EVAL_COLORS[i % len(_EVAL_COLORS)], width=.7)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels(['actual-{}'.format(s) for s in sizes], fontsize=7)
    ax2.set_ylabel('minutes')
    ax2.set_title('eval stages', fontsize=9)
    ax2.legend(fontsize=7)
    _style(ax2)

    # The non-sparse strategies run as subprocesses CONCURRENTLY with
    # sparse learning, so their time is not a slice of the bar. Stating it
    # on the figure stops the next reader from looking for a missing
    # segment that was never additive in the first place.
    par = [r for r in recs.values() if r.get('others')]
    if par:
        wall = max(r['others_wall'] for r in par)
        cpu = max(r['others_cpu'] for r in par)
        fig.text(0.5, -0.02,
                 'non-sparse strategies run in PARALLEL subprocesses '
                 'alongside learning (peak {:.1f}s wall, {:.1f}s CPU '
                 'summed) -- not additive with the bars above'.format(
                     wall, cpu),
                 ha='center', fontsize=7, color='#6d7478')
    out = os.path.join(outdir, 'phases.png')
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    cp.write_json(run_id)
    return out


# Human labels for the sparse-init segments. A segment is the interval
# from one [mem] marker to the NEXT, so it is named for what runs after
# the marker that opened it -- not for the marker itself.
_INIT_LABELS = {
    'mi_post_enforce_prefs': 'between measure-ingress rounds',
    'mi_post_clear_caches': 'mi: after cache clear',
    'mi_post_gt_ingress': 'mi: after ground-truth ingress',
    'mi_post_calc_user_choice': 'mi: after user-choice calc',
    'mi_enter': 'mi: enter -> first step',
    'iov_post_gt_resilience_benefit': 'after gt resilience benefit',
    'iov_post_gt_latency_benefit': 'after gt latency benefit',
    'iov_post_measured_objective': 'after measured objective',
    'iov_post_clear_caches': 'iov: after cache clear',
    'iov_enter': 'iov: enter',
    'solve_enter': 'solve: enter',
    'solve_cold_start': 'solve: cold start',
    'solve_post_modify_ugs': 'solve: after modify ugs',
    'solve_post_init_optim_vars': 'solve: after init optim vars',
    'solve_post_first_measure_ingresses': 'solve: after first measure',
}


def _work_panel(axis, pts, power, logy=True):
    """Scatter compute-time against UGs x popps^power, annotating the local
    exponent between adjacent sizes. Returns the list of local exponents."""
    import math
    for ugs, popps, comp, learn, n, partial in pts:
        w = ugs * popps ** power
        axis.scatter([w], [comp], s=75, zorder=3,
                     facecolor='none' if partial else '#31647f',
                     edgecolor='#31647f', linewidth=1.6)
        if learn > 0:
            axis.scatter([w], [learn], s=48, marker='s', zorder=3,
                         facecolor='none' if partial else '#2f9e6e',
                         edgecolor='#2f9e6e', linewidth=1.4)
        axis.annotate('{} sites{}'.format(n, '*' if partial else ''),
                      (w, comp), textcoords='offset points',
                      xytext=(8, 5), fontsize=7.5, color='#22282b')
    axis.set_xscale('log')
    if logy:
        axis.set_yscale('log')
    axis.set_xlabel('UGs x popps$^{}$'.format(power))
    _style(axis)

    solid = [(ugs * popps ** power, comp) for ugs, popps, comp, _l, _n, p
             in pts if not p and comp > 0]
    solid.sort()
    slopes = []
    for i in range(len(solid) - 1):
        w1, c1 = solid[i]
        w2, c2 = solid[i + 1]
        if w2 <= w1 or c1 <= 0 or c2 <= 0:
            continue
        sl = math.log(c2 / c1) / math.log(w2 / w1)
        slopes.append(sl)
        axis.plot([w1, w2], [c1, c2], '-', color='#c2544d', linewidth=1,
                  alpha=.75, zorder=2)
        axis.annotate('{:.2f}'.format(sl), (math.sqrt(w1 * w2),
                                            math.sqrt(c1 * c2)),
                      textcoords='offset points', xytext=(0, -14),
                      fontsize=8, color='#c2544d', ha='center')
    return slopes


def plot_work_scaling(run_id, outdir):
    """Cost vs candidate work metrics: UGs x popps^2 beside UGs x popps^3.

    Site count is a poor x-axis: sites are drawn at random per size, so
    popp count is not monotone in N (actual-15 drew 324 popps against
    actual-10's 287, on FEWER UGs).

    y is COMPUTE-ONLY (total minus deployment setup). Setup switched from
    the serial CSV loop to the shard fast path between actual-15 and
    actual-20, so total wall is not comparable across that boundary and a
    fit through it would be measuring our own optimisation.

    Read it by the LOCAL exponents (red): the work metric that best
    describes the algorithm is the one whose exponents sit nearest 1.0.
    No global fit is shown -- the smallest size is fixed-overhead
    dominated and would drag any single fit toward zero.
    """
    from dashboard import cluster_phases as cp
    recs = cp.parse(run_id)
    pts = [(r['ugs'], r['popps'], r['total_s'] - r.get('setup', 0),
            r['learning'], s, bool(r.get('partial')))
           for s, r in sorted(recs.items())
           if r.get('ugs') and r.get('popps') and r.get('total_s', 0) > 0]
    if len(pts) < 2:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    res = {}
    for axis, power in zip(axes, (2, 3)):
        res[power] = _work_panel(axis, pts, power)
        axis.set_title('UGs x popps$^{}$'.format(power), fontsize=9)
    axes[0].set_ylabel('seconds (compute: total minus setup)')
    axes[0].scatter([], [], s=75, facecolor='#31647f', edgecolor='#31647f',
                    label='compute (total - setup)')
    axes[0].scatter([], [], s=48, marker='s', facecolor='#2f9e6e',
                    edgecolor='#2f9e6e', label='learning only')
    axes[0].legend(fontsize=7)

    def _summary(power):
        sl = res.get(power) or []
        if not sl:
            return '{}: n/a'.format(power)
        return 'popps^{}: local exponents {}'.format(
            power, ', '.join('{:.2f}'.format(x) for x in sl))
    fig.suptitle('{}: cost vs deployment work'.format(run_id), fontsize=9)
    fig.text(0.5, -0.05,
             '{}   |   {}\n'
             'red = local exponent d log(time)/d log(work) between adjacent '
             'sizes; the better work metric is the one nearer 1.00. '
             '* = still running (lower bound); smallest size is '
             'fixed-overhead dominated.'.format(_summary(2), _summary(3)),
             ha='center', fontsize=6.5, color='#6d7478')
    out = os.path.join(outdir, 'work_scaling.png')
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_init_breakdown(run_id, outdir):
    """What sparse init spends its time on, per deployment size.

    sparse init is solve_enter -> the first learning iteration. At
    actual-20 that was 925s, 81% of it in the gaps between the six
    measure-ingress rounds -- which is invisible in the phase chart
    because it is one bar segment there.
    """
    from dashboard import cluster_phases as cp
    recs = cp.parse(run_id)
    recs = {k: v for k, v in recs.items() if v.get('init_breakdown')}
    if not recs:
        return None
    sizes = sorted(recs)
    keys = []
    for s in sizes:
        for k, _v in sorted(recs[s]['init_breakdown'].items(),
                            key=lambda kv: -kv[1]):
            if k not in keys:
                keys.append(k)
    keys = keys[:8]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    cmap = plt.get_cmap('tab20')
    bottom = [0.0] * len(sizes)
    for i, k in enumerate(keys):
        vals = [recs[s]['init_breakdown'].get(k, 0.0) / 60.0 for s in sizes]
        ax.bar(range(len(sizes)), vals, bottom=bottom,
               label=_INIT_LABELS.get(k, k), color=cmap(i % 20), width=.7)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(['actual-{}'.format(s) for s in sizes], fontsize=7)
    ax.set_ylabel('minutes in sparse init')
    ax.set_title('{}: sparse init by component'.format(run_id), fontsize=9)
    ax.legend(fontsize=6)
    _style(ax)

    tot = [recs[s]['sparse_init'] / 60.0 for s in sizes]
    ax2.plot(sizes, tot, 'o-', color='#31647f', label='sparse init total')
    top = keys[0] if keys else None
    if top:
        ax2.plot(sizes,
                 [recs[s]['init_breakdown'].get(top, 0.0) / 60.0
                  for s in sizes], 's--', color='#c2544d',
                 label=_INIT_LABELS.get(top, top))
    ax2.set_xlabel('deployment size')
    ax2.set_ylabel('minutes')
    ax2.set_ylim(bottom=0)
    ax2.set_title('scaling of sparse init', fontsize=9)
    ax2.legend(fontsize=7)
    _style(ax2)
    fig.text(0.5, -0.02,
             'a segment spans one [mem] marker to the NEXT, so it is named '
             'for what runs AFTER the marker that opened it',
             ha='center', fontsize=6.5, color='#6d7478')
    out = os.path.join(outdir, 'sparse_init.png')
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_worker_init(run_id, outdir):
    """One-time worker startup, decomposed.

    Parses `[wt-init]`, which is emitted once per worker after the
    deployment lands. Until this existed, everything a worker did once --
    receiving the deployment, building rb_backups, standing up the
    persistent LP -- was folded into gradient batch #1, which is why the
    first bar of the per-batch chart towers over the rest with no way to
    read it. Returns None (and draws nothing) on runs from before the
    instrumentation, rather than inventing a breakdown.
    """
    import glob as _glob
    texts = []
    wdir = os.path.join(RUNS_DIR, run_id, 'logs', 'workers')
    for fp in sorted(_glob.glob(os.path.join(wdir, '*.log'))):
        try:
            texts.append(open(fp, errors='replace').read())
        except (IOError, OSError):
            continue
    texts.append(_log(run_id))
    rows = {}
    for txt in texts:
        for m in WTI_RE.finditer(txt.replace('\r', '\n')):
            w = int(m.group(1))
            parts = {k: float(v) for k, v in WTI_PART_RE.findall(m.group(4))}
            if parts:
                # keep the largest report per worker (later deployments
                # rebuild more state than the first)
                if sum(parts.values()) > sum(rows.get(w, {}).values() or [0]):
                    rows[w] = parts
    late = {}
    for txt in texts:
        for m in WTI_LATE_RE.finditer(txt.replace('\r', '\n')):
            k, v = m.group(1), float(m.group(2))
            late[k] = max(late.get(k, 0.0), v)
    for w in rows:
        for k, v in late.items():
            rows[w].setdefault(k, v)
    if not rows:
        return None
    workers = sorted(rows)
    keys = []
    for w in workers:
        for k in sorted(rows[w], key=lambda x: -rows[w][x]):
            if k not in keys:
                keys.append(k)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    cmap = plt.get_cmap('tab10')
    bottom = [0.0] * len(workers)
    for i, k in enumerate(keys):
        vals = [rows[w].get(k, 0.0) for w in workers]
        ax.bar(range(len(workers)), vals, bottom=bottom, label=k,
               color=cmap(i % 10), width=1.0)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xlabel('worker')
    ax.set_ylabel('seconds (one-time)')
    ax.set_title('{}: worker startup cost'.format(run_id), fontsize=9)
    ax.legend(fontsize=6.5)
    _style(ax)

    tot = {k: sum(rows[w].get(k, 0.0) for w in workers) for k in keys}
    order = sorted(tot, key=lambda k: tot[k])
    ax2.barh(range(len(order)), [tot[k] for k in order],
             color=[cmap(keys.index(k) % 10) for k in order])
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(order, fontsize=7)
    ax2.set_xlabel('core-seconds summed over {} workers'.format(len(workers)))
    ax2.set_title('where startup goes, cluster-wide', fontsize=9)
    _style(ax2)
    out = os.path.join(outdir, 'worker_init.png')
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def plot_worker_ram(run_id, outdir):
    """Per-worker RAM by object, and how it scales with deployment size.

    Left: stacked top attributes per size (what holds the memory).
    Right: peak worker total vs size, with GB/core lines for the ARM
    families -- the actual decision this data serves is "which instance
    family can hold this size", so the chart draws that line directly.
    """
    from dashboard import cluster_objsize as co
    d = co.parse(run_id)
    by = {k: v for k, v in (d.get('by_size') or {}).items()
          if v.get('attrs')}
    if not by:
        return None
    sizes = sorted(by)
    top = []
    for s in sizes:
        for a in by[s]['attrs'][:8]:
            if a['attr'] not in top:
                top.append(a['attr'])
    top = top[:9]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    cmap = plt.get_cmap('tab10')
    bottom = [0.0] * len(sizes)
    for i, attr in enumerate(top):
        vals = []
        for s in sizes:
            hit = [a for a in by[s]['attrs'] if a['attr'] == attr]
            vals.append(hit[0]['max_mb'] if hit else 0.0)
        ax.bar(range(len(sizes)), vals, bottom=bottom, label=attr,
               color=cmap(i % 10), width=.7)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(['actual-{}'.format(s) for s in sizes], fontsize=7)
    ax.set_ylabel('MB held per worker (peak)')
    ax.set_title('{}: per-worker RAM by object'.format(run_id), fontsize=9)
    ax.legend(fontsize=5.5, ncol=2)
    _style(ax)

    peaks = [by[s]['worker_total_max_mb'] for s in sizes]
    ax2.plot(sizes, peaks, 'o-', color='#31647f', label='peak worker total')
    for fam, gb_per_core in (('c8g/c7g (2 GB/core)', 2.0),
                             ('m8g/m7g (4 GB/core)', 4.0)):
        ax2.axhline(gb_per_core * 1024, linestyle='--', linewidth=.8,
                    color='#8b9296')
        ax2.annotate(fam, (sizes[0], gb_per_core * 1024), fontsize=6,
                     color='#6d7478', va='bottom')
    ax2.set_xlabel('deployment size')
    ax2.set_ylabel('MB per worker')
    ax2.set_ylim(bottom=0)
    ax2.set_title('scaling vs per-core RAM budget', fontsize=9)
    ax2.legend(fontsize=7)
    _style(ax2)
    fig.text(0.5, -0.02, 'source: {}'.format(d.get('source', '?')),
             ha='center', fontsize=6.5, color='#6d7478')
    out = os.path.join(outdir, 'worker_ram.png')
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    co.write_json(run_id)
    return out


def plot_iter_timing(run_id, outdir):
    """Per-iteration driver phase timing, split by deployment size.

    The `Timer: <phase> -- <x> s` lines are what actually answer "where
    does an iteration go", and they are the numbers the pricing model
    needs. Iteration index comes from the preceding LEARNING ITERATION
    banner; deployment size from the preceding [sweep] === marker.
    """
    log = _log(run_id)
    if not log:
        return None
    cur_size, cur_iter = None, None
    # {size: {phase: [(iter, seconds)]}}
    data = {}
    for line in log.replace('\r', '\n').splitlines():
        # banner prints dpsize=32 (old) or dpsize=actual-32 (post-
        # restructure); the numeric-only regex left cur_size stuck on the
        # previous segment's size and misattributed every iteration
        # (Tom 2026-08-25: new actual-32 bars showed in the actual-3 panel)
        ms = re.search(r'\[sweep\] === dpsize=(?:[\w]*?-)?(\d+)', line)
        if ms:
            cur_size, cur_iter = int(ms.group(1)), None
            continue
        mi = ITER_RE.search(line)
        if mi:
            cur_iter = int(mi.group(1))
            continue
        mt = TIMER_RE.search(line)
        if mt and cur_iter is not None:
            data.setdefault(cur_size, {}).setdefault(
                mt.group(1), []).append((cur_iter, float(mt.group(2))))
    sizes = [s for s in sorted(data, key=lambda x: (x is None, x))
             if any(data[s].values())]
    if not sizes:
        return None
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.4 * len(sizes), 3.2),
                             squeeze=False, sharey=True)
    colors = {'grads': '#31647f', 'measure': '#2f9e6e', 'stop': '#c9862b',
              'info': '#8b9296'}
    for ax, s in zip(axes[0], sizes):
        bottom = {}
        for phase in ('grads', 'measure', 'stop', 'info'):
            # resumed segments REUSE iteration numbers; the raw list is in
            # file order, so keep the LAST occurrence per iter -- otherwise
            # a resume's fast bars hide under the old segment's slow ones
            # (Tom 2026-08-25: 'USE UPDATED DATA')
            _dd = {}
            for _it, _v in data[s].get(phase, []):
                _dd[_it] = _v
            pts = sorted(_dd.items())
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            base = [bottom.get(x, 0.0) for x in xs]
            ax.bar(xs, ys, bottom=base, label=phase,
                   color=colors.get(phase, '#999'), width=.8)
            for x, y, b in zip(xs, ys, base):
                bottom[x] = b + y
        ax.set_title('actual-{}'.format(s) if s is not None else 'run',
                     fontsize=9)
        ax.set_xlabel('iteration')
        _style(ax)
    axes[0][0].set_ylabel('seconds in iteration')
    # Outlier rejection on the y-limit (Tom 2026-08-25): one pathological
    # iteration (a straggler or GC stall) blew the shared axis so every
    # normal bar was unreadable. Clip at 1.5x the p95 of per-iteration
    # TOTALS; clipped bars keep drawing off-axis and the count is noted.
    totals = []
    for s_ in sizes:
        per_iter = {}
        for phase_pts in data[s_].values():
            for it, sec in phase_pts:
                per_iter[it] = per_iter.get(it, 0.0) + sec
        totals.extend(per_iter.values())
    if totals:
        totals.sort()
        cap = 1.5 * totals[min(len(totals) - 1, int(.95 * len(totals)))]
        n_clip = sum(1 for t in totals if t > cap)
        if n_clip:
            for ax in axes[0]:
                ax.set_ylim(0, cap)
            axes[0][-1].annotate(
                '{} outlier iter(s) clipped (max {:.0f}s)'.format(
                    n_clip, totals[-1]),
                xy=(0.98, 0.98), xycoords='axes fraction', ha='right',
                va='top', fontsize=7, color='#c9862b')
    axes[0][-1].legend(fontsize=7)
    fig.suptitle('{}: where each iteration goes'.format(run_id), fontsize=9)
    out = os.path.join(outdir, 'iter_timing.png')
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def plot_worker_breakdown(run_id, outdir):
    """Worker-0's per-batch LP/sim breakdown over the run.

    Only worker 0 emits (Ray dedups the rest), so this is per-worker time,
    not cluster total -- multiply by the active worker count for that.
    """
    batches = []
    for line in _log(run_id).replace('\r', '\n').splitlines():
        m = WT_RE.search(line)
        if m:
            parts = {k: int(v) for k, v in WT_PART_RE.findall(m.group(2))
                     if k not in WT_NOT_A_SLICE}
            batches.append((float(m.group(1)), parts))
    if len(batches) < 3:
        return None
    keys, seen = [], set()
    for _tot, parts in batches:
        for k in parts:
            if k not in seen:
                seen.add(k); keys.append(k)
    keys = sorted(keys, key=lambda k: -sum(p.get(k, 0)
                                           for _t, p in batches))[:7]
    xs = list(range(len(batches)))
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    bottom = [0.0] * len(batches)
    cmap = plt.get_cmap('tab10')
    for i, k in enumerate(keys):
        vals = [tot * parts.get(k, 0) / 100.0 for tot, parts in batches]
        ax.bar(xs, vals, bottom=bottom, label=k, color=cmap(i % 10), width=1.0)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xlabel('gradient batch')
    ax.set_ylabel('worker-0 seconds')
    ax.set_title('{}: worker 0 time breakdown'.format(run_id), fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    _style(ax)
    out = os.path.join(outdir, 'worker_breakdown.png')
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def plot_mem(run_id, outdir):
    rows = []
    for m in MEM_RE.finditer(_log(run_id)):
        if m.group('rss') is None:
            continue
        rows.append((float(m.group('t')), int(m.group('rss')),
                     int(m.group('avail')), m.group('tag'),
                     m.group('extra'),
                     int(m.group('wrss')) if m.group('wrss') else None,
                     int(m.group('shm')) if m.group('shm') else None))
    if len(rows) < 2:
        return None
    t0 = rows[0][0]
    hrs = [(r[0] - t0) / 3600.0 for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(hrs, [r[1] / 1024.0 for r in rows], 'o-', color='#31647f',
            label='driver RSS', markersize=2.5)
    _w = [(h, r[5]) for h, r in zip(hrs, rows) if r[5] and r[5] > 0]
    if _w:
        ax.plot([p[0] for p in _w], [p[1] / 1024.0 for p in _w], '^-',
                color='#2f9e6e', label='workers RSS (sum)', markersize=2.5)
    _s = [(h, r[6]) for h, r in zip(hrs, rows) if r[6] and r[6] > 0]
    if _s:
        ax.plot([p[0] for p in _s], [p[1] / 1024.0 for p in _s], 'v-',
                color='#8b5aa8', label='shm (plasma)', markersize=2.5)
    ax.legend(fontsize=6, loc='upper left')
    ax.set_xlabel('hours into run')
    ax.set_ylabel('GB (driver / workers / shm)', color='#31647f')
    _style(ax)
    ax2 = ax.twinx()
    ax2.plot(hrs, [r[2] / 1024.0 for r in rows], 's--', color='#c9862b',
             label='system available')
    ax2.set_ylabel('system available (GB)', color='#c9862b')
    ax2.spines['top'].set_visible(False)
    # Annotate size boundaries -- that is where the leak shows itself.
    for i, r in enumerate(rows):
        mm = re.search(r'dpsize=(\d+)', r[4] or '')
        if mm and r[3] == 'dpsize_start':
            ax.axvline(hrs[i], color='#8b9296', linewidth=.6, alpha=.6)
            ax.annotate(mm.group(1), (hrs[i], ax.get_ylim()[1]),
                        fontsize=6, color='#6d7478', ha='left', va='top')
    ax.set_title('{}: driver memory'.format(run_id), fontsize=9)
    out = os.path.join(outdir, 'memory.png')
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def plot_disk(run_id, outdir):
    rows = _sysmon(run_id)
    rows = [r for r in rows if 'disk_avail_gb' in r and 't' in r]
    if len(rows) < 2:
        return None
    t0 = rows[0]['t']
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.plot([(r['t'] - t0) / 3600.0 for r in rows],
            [r['disk_avail_gb'] for r in rows], color='#2f9e6e')
    mem = [r for r in rows if r.get('mem_avail_mb')]
    if mem:
        ax2 = ax.twinx()
        gb = [r['mem_avail_mb'] / 1024.0 for r in mem]
        ax2.plot([(r['t'] - t0) / 3600.0 for r in mem], gb,
                 color='#c9862b', linewidth=.8, alpha=.8)
        ax2.set_ylabel('mem available (GB)', color='#c9862b')
        ax2.spines['top'].set_visible(False)
        # BOTH axes anchored at zero. Auto-scaling drew a 7% dip in
        # available memory (122 -> 114 GB, with 110 GB still free) as a
        # cliff filling the whole panel, and it read as an imminent OOM
        # (Tom, 2026-08-21). A headroom chart whose axis floats is worse
        # than no chart: it manufactures alarm out of noise.
        ax2.set_ylim(0, max(gb) * 1.08)
    ax.set_xlabel('hours into run')
    ax.set_ylabel('disk free (GB)', color='#2f9e6e')
    ax.set_ylim(0, max(r['disk_avail_gb'] for r in rows) * 1.08)
    ax.set_title('{}: headroom'.format(run_id), fontsize=9)
    _style(ax)
    out = os.path.join(outdir, 'headroom.png')
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out




# ---------------------------------------------- papertable (objectives) --

_PT_OBJS = ['avg_latency', 'per_site_cost', 'max_util',
            'frac_beyond_optimal', 'joint_priority']
_PT_SHORT = {'avg_latency': 'lat+res', 'per_site_cost': 'site cost',
             'max_util': 'MLU', 'frac_beyond_optimal': 'frac-opt',
             'joint_priority': 'priorities'}


def _pt_cell_logs(run_id, manifest):
    """objective -> local cell-log path (harvested into repo cache/)."""
    base = None
    cmd = manifest.get('cmd') or []
    for i, tok in enumerate(cmd):
        if tok == '--run_id' and i + 1 < len(cmd):
            base = cmd[i + 1]
    if not base:
        base = run_id.replace('-', '_')
    out = {}
    for obj in _PT_OBJS:
        tag = base if obj == 'avg_latency' else '{}_{}'.format(base, obj)
        fn = 'table_generate_{}.log'.format(tag)
        import glob as _g
        # glob pulls flatten remote paths into results/<glob-name>/<file>
        cands = ([os.path.join(REPO, 'cache', fn)]
                 + _g.glob(os.path.join(RUNS_DIR, run_id, 'results',
                                        '**', fn), recursive=True))
        for cand in cands:
            if os.path.exists(cand):
                out[obj] = cand
                break
    return out


def _latest_attempt(pts):
    """A killed+relaunched cell appends a SECOND training attempt to the
    same log, restarting the iter counter (or hotstarting lower). Keep
    only the points after the last counter reset -- stats and series then
    describe the run that is actually executing (Tom 2026-08-25)."""
    start = 0
    for i in range(1, len(pts)):
        if pts[i][1] < pts[i - 1][1]:
            start = i
    return pts[start:]


def _pt_parse_cell(fn):
    """(iter_deltas, peak_rss_mb, wall_min, n_iters) from one cell log."""
    txt = open(fn, errors='replace').read()
    pts = [(float(t), int(i)) for t, i in re.findall(
        r'\[mem\] tag=iter_start rss_mb=\d+ .*? t=([\d.]+) iter=(\d+)', txt)]
    pts = _latest_attempt(pts)
    deltas = [b[0] - a[0] for a, b in zip(pts, pts[1:])
              if a[1] < b[1] and 0 < b[0] - a[0] < 3600]
    rss = [int(r) for r in re.findall(r'\[mem\] tag=\S+ rss_mb=(\d+)', txt)]
    wall = None
    try:
        wall = (os.path.getmtime(fn) - pts[0][0]) / 60.0 if pts else None
    except Exception:
        pass
    return deltas, (max(rss) if rss else None), wall, (pts[-1][1] if pts else 0)


def _pt_series(fn):
    """(t_epoch, iter) points + (t, rss_mb) points from one cell log."""
    txt = open(fn, errors='replace').read()
    it = [(float(t), int(i)) for t, i in re.findall(
        r'\[mem\] tag=iter_start rss_mb=\d+ .*? t=([\d.]+) iter=(\d+)', txt)]
    it = _latest_attempt(it)
    rss = [(float(t), int(r)) for r, t in re.findall(
        r'\[mem\] tag=\S+ rss_mb=(\d+) .*? t=([\d.]+)', txt)]
    if it:
        rss = [p for p in rss if p[0] >= it[0][0] - 600]
    # phases: cell start -> first iter (setup+baselines), iter span (train),
    # last iter -> log mtime (eval, if the training finished)
    t0 = rss[0][0] if rss else None
    done = 'ALL DONE' in txt
    return it, rss, t0, done


def plot_papertable(run_id, outdir, manifest):
    """Objective-keyed timing figures -- the papertable analogue of the
    size-keyed dpsweep plots. One bar per OBJECTIVE."""
    logs = _pt_cell_logs(run_id, manifest)
    if not logs:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    objs = [o for o in _PT_OBJS if o in logs]
    parsed = {o: _pt_parse_cell(logs[o]) for o in objs}
    names = [_PT_SHORT[o] for o in objs]
    made = []

    # 1. sec/iter distribution per objective
    fig, ax = plt.subplots(figsize=(7, 3.2))
    data = [parsed[o][0] or [0] for o in objs]
    ax.boxplot(data, labels=names, showfliers=False)
    ax.set_ylabel('sec / training iter')
    ax.set_title('{} -- iteration time by objective'.format(run_id))
    ax.grid(alpha=.25, axis='y')
    f = os.path.join(outdir, 'obj_iter_timing.png')
    fig.tight_layout(); fig.savefig(f, dpi=130); plt.close(fig)
    made.append(f)

    # 2. wall + iterations per objective
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.0))
    a1.bar(names, [(parsed[o][2] or 0) for o in objs], color='#4878a8')
    a1.set_ylabel('wall so far (min)'); a1.grid(alpha=.25, axis='y')
    a2.bar(names, [parsed[o][3] for o in objs], color='#6aa66a')
    a2.set_ylabel('training iters seen'); a2.grid(alpha=.25, axis='y')
    for a in (a1, a2):
        a.tick_params(axis='x', rotation=20)
    fig.suptitle('{} -- wall & progress by objective'.format(run_id),
                 fontsize=10)
    f = os.path.join(outdir, 'obj_wall_iters.png')
    fig.tight_layout(); fig.savefig(f, dpi=130); plt.close(fig)
    made.append(f)

    # 3. peak driver RSS per objective
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar(names, [(parsed[o][1] or 0) / 1024.0 for o in objs],
           color='#a86048')
    ax.set_ylabel('peak driver RSS (GB)')
    ax.set_title('{} -- memory by objective'.format(run_id))
    ax.grid(alpha=.25, axis='y')
    ax.tick_params(axis='x', rotation=20)
    f = os.path.join(outdir, 'obj_memory.png')
    fig.tight_layout(); fig.savefig(f, dpi=130); plt.close(fig)
    made.append(f)

    series = {o: _pt_series(logs[o]) for o in objs}

    # 4. training iterations over wall-clock -- the dpsweep progress
    #    view with objective as the size axis (Tom 2026-08-24)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for o in objs:
        it, _rss, t0, done = series[o]
        if not it or t0 is None:
            continue
        xs = [(t - t0) / 3600.0 for t, _ in it]
        ys = [i for _, i in it]
        ax.plot(xs, ys, label=_PT_SHORT[o] + (' (done)' if done else ''))
    ax.set_xlabel('hours since cell start'); ax.set_ylabel('training iter')
    ax.set_title('{} -- training progress over time'.format(run_id))
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    f = os.path.join(outdir, 'obj_iters_over_time.png')
    fig.tight_layout(); fig.savefig(f, dpi=130); plt.close(fig)
    made.append(f)

    # 5. sec/iter vs iteration (RB-cadence spikes visible)
    fig, ax = plt.subplots(figsize=(8, 3.0))
    for o in objs:
        it = series[o][0]
        d = [(b[1], b[0] - a[0]) for a, b in zip(it, it[1:])
             if a[1] < b[1] and 0 < b[0] - a[0] < 3600]
        if d:
            ax.plot([x for x, _ in d], [y for _, y in d],
                    label=_PT_SHORT[o], alpha=.8)
    ax.set_xlabel('iteration'); ax.set_ylabel('sec / iter')
    ax.set_title('{} -- per-iteration wall time'.format(run_id))
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    f = os.path.join(outdir, 'obj_sec_per_iter.png')
    fig.tight_layout(); fig.savefig(f, dpi=130); plt.close(fig)
    made.append(f)

    # 6. driver RSS over time per objective
    fig, ax = plt.subplots(figsize=(8, 3.0))
    for o in objs:
        _it, rss, t0, _done = series[o]
        if rss and t0 is not None:
            ax.plot([(t - t0) / 3600.0 for t, _ in rss],
                    [r / 1024.0 for _, r in rss], label=_PT_SHORT[o], alpha=.8)
    ax.set_xlabel('hours since cell start'); ax.set_ylabel('driver RSS (GB)')
    ax.set_title('{} -- driver memory over time'.format(run_id))
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    f = os.path.join(outdir, 'obj_mem_over_time.png')
    fig.tight_layout(); fig.savefig(f, dpi=130); plt.close(fig)
    made.append(f)

    # 7. phase split per objective: setup+baselines vs training span
    fig, ax = plt.subplots(figsize=(7, 3.0))
    setup, train = [], []
    for o in objs:
        it, _rss, t0, _done = series[o]
        if it and t0 is not None:
            setup.append((it[0][0] - t0) / 3600.0)
            train.append((it[-1][0] - it[0][0]) / 3600.0)
        else:
            setup.append(0); train.append(0)
    ax.bar(names, setup, label='setup + baselines', color='#8a8a8a')
    ax.bar(names, train, bottom=setup, label='sparse training', color='#4878a8')
    ax.set_ylabel('hours'); ax.legend(fontsize=8)
    ax.set_title('{} -- phase split by objective'.format(run_id))
    ax.grid(alpha=.25, axis='y'); ax.tick_params(axis='x', rotation=20)
    f = os.path.join(outdir, 'obj_phases.png')
    fig.tight_layout(); fig.savefig(f, dpi=130); plt.close(fig)
    made.append(f)
    return made




def plot_run(run_id):
    outdir = os.path.join(FIG_ROOT, run_id)
    os.makedirs(outdir, exist_ok=True)
    try:
        import json as _j
        manifest = _j.load(open(os.path.join(RUNS_DIR, run_id,
                                             'manifest.json')))
    except Exception:
        manifest = {}
    if manifest.get('preset') == 'papertable':
        made = plot_papertable(run_id, outdir, manifest)
        print('{}: {} figure(s)'.format(run_id, len(made)))
        for f in made:
            print('  ' + os.path.relpath(f, REPO))
        return made
    made = [f for f in (plot_sizes(run_id, outdir),
                        plot_phases(run_id, outdir),
                        plot_work_scaling(run_id, outdir),
                        plot_init_breakdown(run_id, outdir),
                        plot_worker_init(run_id, outdir),
                        plot_worker_ram(run_id, outdir),
                        plot_iter_timing(run_id, outdir),
                        plot_worker_breakdown(run_id, outdir),
                        plot_mem(run_id, outdir),
                        plot_disk(run_id, outdir)) if f]
    print('{}: {} figure(s)'.format(run_id, len(made)))
    for f in made:
        print('  ' + os.path.relpath(f, REPO))
    return made


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--all' in argv or not argv:
        if not os.path.isdir(RUNS_DIR):
            print('no runs yet')
            return 0
        ids = [d for d in sorted(os.listdir(RUNS_DIR))
               if os.path.exists(os.path.join(RUNS_DIR, d, 'manifest.json'))]
    else:
        ids = argv
    for run_id in ids:
        plot_run(run_id)
    return 0


def convert_ladder_pdfs(run_id):
    """Pin the evaluate_over_deployment_sizes paper figures into the run's
    dash dir as ladder_*.png (steps contract: tab figures must be step
    outputs -- the Aug-23 set was a one-off manual conversion that then
    froze; Tom 2026-08-25 'i still just see up to 20'). Uses macOS sips,
    falls back to pdftoppm. Only converts when the PDF is newer."""
    import subprocess
    src = os.path.join(REPO, 'figures', 'cluster', run_id)
    dst = os.path.join(FIG_ROOT, run_id)
    if not os.path.isdir(src):
        return
    os.makedirs(dst, exist_ok=True)
    for fn in sorted(os.listdir(src)):
        if not fn.endswith('.pdf'):
            continue
        out = os.path.join(dst, 'ladder_' + fn[:-4] + '.png')
        pdf = os.path.join(src, fn)
        if os.path.exists(out) and \
                os.path.getmtime(out) >= os.path.getmtime(pdf):
            continue
        for cmd in (['sips', '-s', 'format', 'png', '--resampleWidth',
                     '900', pdf, '--out', out],
                    ['pdftoppm', '-png', '-r', '110', '-singlefile',
                     pdf, out[:-4]]):
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=60)
                if r.returncode == 0 and os.path.exists(out):
                    break
            except (OSError, subprocess.TimeoutExpired):
                continue


_orig_main = main
def main():
    rc = _orig_main()
    try:
        import sys as _s
        convert_ladder_pdfs(_s.argv[1])
    except Exception:
        import traceback; traceback.print_exc()
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
