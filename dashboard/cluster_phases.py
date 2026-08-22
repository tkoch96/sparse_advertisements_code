"""Per-deployment-size phase breakdown for a cluster run.

Answers, for each size, where the wall-clock actually went:

  (a) deployment setup   dpsize_start        -> solve_enter
  (b) sparse init        solve_enter         -> iter_start(iter=1)
  (c) learning           iter_start(1)       -> last iter_post_stop_tracker
  (d) other solutions    the five non-sparse strategies, which run as
                         SUBPROCESSES CONCURRENTLY with sparse learning --
                         so their time is NOT additive with (c), and the
                         breakdown says so rather than double-counting it
  (e) evals              eval_volume_calc -> eval_failure_calc ->
                         eval_diurnal -> eval_flash_crowd ->
                         eval_stats_assembly -> dpsize_done

No new instrumentation was needed: every boundary is an existing `[mem]`
marker carrying an epoch `t=`, and the per-strategy subprocess durations
are already printed as `[parallel] <name> finished in <n>s`. That matters
because it means this works on logs already harvested, not only on runs
started after the parser existed.

Phase durations are derived from *timestamps*, never from summing
sub-steps, so they cannot silently disagree with the size's total wall
time. `residual` catches whatever the named phases do not account for --
if that grows, a phase boundary is missing, and the chart says so instead
of quietly absorbing it.
"""

from __future__ import annotations

import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO, 'cache', 'cluster_runs')

_MEM = re.compile(
    r'\[mem\] tag=(?P<tag>\S+)\s+'
    r'(?:rss_mb=(?P<rss>-?\d+)\s+)?'
    r'(?:vms_mb=-?\d+\s+)?(?:peak_mb=-?\d+\s+)?'
    r'sys_avail_mb=-?\d+\s+pid=(?P<pid>\d+)\s+t=(?P<t>[\d.]+)(?P<extra>[^\n]*)')
_SIZE = re.compile(r'\[sweep\] === dpsize=(\d+)')
_PARALLEL_DONE = re.compile(
    r'\[parallel\] (\w+) finished in ([\d.]+)s \(subprocess\)')
_PARALLEL_LAUNCH = re.compile(r'\[parallel\] launching (\d+) non-sparse')
# The FINAL deployment shape, after every filter. Earlier lines in the
# same chunk report pre-filter counts in the hundreds of thousands;
# only 'after limiting users' describes what actually gets solved.
_SHAPE = re.compile(r'(\d+) UGs, (\d+) popps after limiting users')

# Ordered eval stages. Each tag marks the START of its stage; the stage
# ends where the next one begins (and the last at dpsize_done).
EVAL_STAGES = [
    ('eval_volume_calc', 'pct volume within latency'),
    ('eval_failure_calc', 'failure eval'),
    ('eval_diurnal', 'diurnal'),
    ('eval_flash_crowd', 'flash crowd'),
    ('eval_stats_assembly', 'stats assembly'),
]

# (record key, display label). NOTE the evals key is `evals_total`, the
# scalar -- `evals` itself holds the per-stage dict.
PHASES = [
    ('setup', 'deployment setup'),
    ('sparse_init', 'sparse init'),
    ('learning', 'learning'),
    ('evals_total', 'evals'),
    ('residual', 'unaccounted'),
]


def _log_text(run_id):
    p = os.path.join(RUNS_DIR, run_id, 'logs', 'run.log')
    try:
        # tqdm redraws with \r; without this split, markers emitted onto a
        # progress-bar line are still found by finditer, but line-oriented
        # scanning of section headers would miss them.
        return open(p, errors='replace').read().replace('\r', '\n')
    except IOError:
        return ''


def parse(run_id):
    """-> {size: {...phase seconds..., 'evals': {...}, 'others': {...}}}"""
    txt = _log_text(run_id)
    if not txt:
        return {}

    # Chunk on the `dpsize_start` MARKER, not on the `[sweep] === dpsize=`
    # banner. The marker is emitted first and carries both the size and the
    # timestamp; chunking on the banner put each size's dpsize_start at the
    # END of the previous size's chunk, which made setup times negative and
    # totals ~0.
    bounds, sizes = [], []
    for m in _MEM.finditer(txt):
        if m.group('tag') != 'dpsize_start':
            continue
        k = re.search(r'dpsize=(\d+)', m.group('extra') or '')
        if not k:
            continue
        bounds.append(m.start())
        sizes.append(int(k.group(1)))
    if not bounds:
        return {}
    bounds.append(len(txt))

    out = {}
    for i, size in enumerate(sizes):
        chunk = txt[bounds[i]:bounds[i + 1]]
        # PIN THE DRIVER PID. The five non-sparse strategies run as
        # concurrent subprocesses writing into this same log, and they emit
        # the identical [mem] tags. Summing across pids attributed 747s of
        # PAINTER's measure-ingress loop to sparse init (2026-08-21) -- an
        # answer that was wrong by an order of magnitude and looked
        # entirely plausible. The driver is whoever emitted dpsize_start.
        driver_pid = None
        for mm in _MEM.finditer(chunk):
            if mm.group('tag') == 'dpsize_start':
                driver_pid = mm.group('pid')
                break

        marks, iters = {}, {}
        for mm in _MEM.finditer(chunk):
            if driver_pid and mm.group('pid') != driver_pid:
                continue
            tag, t = mm.group('tag'), float(mm.group('t'))
            extra = mm.group('extra') or ''
            if tag == 'iter_start':
                k = re.search(r'iter=(\d+)', extra)
                if k:
                    iters[int(k.group(1))] = t
            # First occurrence wins: these tags mark a phase START.
            marks.setdefault(tag, t)
            if tag in ('iter_post_stop_tracker', 'dpsize_done'):
                marks[tag] = t            # ...except these: last wins

        t0 = marks.get('dpsize_start')
        if t0 is None:
            continue
        rec = {'size': size, 'evals': {}, 'others': {},
               'driver_pid': driver_pid}

        solve = marks.get('solve_enter')
        first_iter = iters.get(min(iters)) if iters else None
        last_iter_end = marks.get('iter_post_stop_tracker')
        first_eval = next((marks[t] for t, _ in EVAL_STAGES if t in marks),
                          None)
        t_end = marks.get('dpsize_done')
        if t_end is None:
            # Still running, or killed: bound the size by the newest marker
            # we saw rather than inventing an end.
            cands = [v for v in list(marks.values()) + list(iters.values())]
            t_end = max(cands) if cands else t0
            rec['partial'] = True

        rec['total_s'] = t_end - t0
        rec['setup'] = (solve - t0) if solve else 0.0
        if solve and first_iter:
            rec['sparse_init'] = first_iter - solve
        else:
            rec['sparse_init'] = 0.0
        if first_iter and last_iter_end and last_iter_end > first_iter:
            rec['learning'] = last_iter_end - first_iter
            rec['n_iters'] = len(iters)
        else:
            rec['learning'] = 0.0
            rec['n_iters'] = len(iters)

        # (b) decomposed. Every [mem] marker between solve_enter and the
        # first iteration opens an interval that closes at the next
        # marker, so a segment is labelled by the marker that STARTED it
        # -- i.e. "mi_post_enforce_prefs: 54s" means 54s elapsed AFTER
        # enforce_prefs finished, before the next marker fired. Derived
        # generically from marker order rather than a hardcoded phase
        # list, so new markers appear here automatically instead of
        # silently inflating whatever segment precedes them.
        rec['init_breakdown'] = {}
        if solve and first_iter and first_iter > solve:
            win = sorted([(float(mm.group('t')), mm.group('tag'))
                          for mm in _MEM.finditer(chunk)
                          if (not driver_pid or mm.group('pid') == driver_pid)
                          and solve <= float(mm.group('t')) <= first_iter])
            for j in range(len(win) - 1):
                tag = win[j][1]
                rec['init_breakdown'][tag] = (
                    rec['init_breakdown'].get(tag, 0.0)
                    + (win[j + 1][0] - win[j][0]))
            rec['init_breakdown'] = {
                k: v for k, v in rec['init_breakdown'].items() if v >= 0.05}

        # Eval stages: each runs until the next one starts.
        stage_ts = [(tag, lbl, marks[tag]) for tag, lbl in EVAL_STAGES
                    if tag in marks]
        for j, (tag, lbl, ts) in enumerate(stage_ts):
            nxt = stage_ts[j + 1][2] if j + 1 < len(stage_ts) else t_end
            rec['evals'][lbl] = max(0.0, nxt - ts)
        rec['evals_total'] = sum(rec['evals'].values())
        if first_eval:
            # Prefer the timestamp span over the sum, so a missing stage
            # boundary shows up as residual rather than shrinking 'evals'.
            rec['evals_total'] = max(rec['evals_total'], t_end - first_eval)

        # Non-sparse strategies: subprocesses launched alongside sparse.
        for mm in _PARALLEL_DONE.finditer(chunk):
            rec['others'][mm.group(1)] = float(mm.group(2))
        rec['others_parallel'] = bool(_PARALLEL_LAUNCH.search(chunk))
        rec['others_wall'] = max(rec['others'].values()) if rec['others'] else 0.0
        rec['others_cpu'] = sum(rec['others'].values())

        named = rec['setup'] + rec['sparse_init'] + rec['learning'] + \
            rec['evals_total']
        # Deployment shape. Sites are chosen at random per size, so popp
        # count is NOT monotone in site count -- actual-15 drew 324 popps
        # against actual-10's 287. Recording both lets cost be plotted
        # against the work actually implied rather than against N.
        shape = _SHAPE.findall(chunk)
        if shape:
            rec['ugs'] = int(shape[-1][0])
            rec['popps'] = int(shape[-1][1])
            rec['work'] = rec['ugs'] * rec['popps'] ** 2

        rec['residual'] = max(0.0, rec['total_s'] - named)

        # A RESUMED log can contain the same size twice: once from a
        # segment that died partway (2026-08-21, the box was stopped
        # during actual-20's setup) and once from the re-run. Blindly
        # letting the later chunk win would show the aborted attempt
        # whenever the retry has not yet reached the same marker. Prefer
        # a completed chunk over a partial one; between two of equal
        # completeness, prefer the later (the retry supersedes).
        old = out.get(size)
        if old is not None:
            old_partial = bool(old.get('partial'))
            new_partial = bool(rec.get('partial'))
            if old_partial != new_partial and old_partial is False:
                continue                   # keep the completed one
        rec['attempts'] = (old.get('attempts', 1) + 1) if old else 1
        out[size] = rec
    return out


def write_json(run_id):
    d = parse(run_id)
    if not d:
        return None
    p = os.path.join(RUNS_DIR, run_id, 'phases.json')
    try:
        with open(p, 'w') as fh:
            json.dump({str(k): v for k, v in d.items()}, fh, indent=1)
    except OSError:
        return None
    return p


def load(run_id):
    p = os.path.join(RUNS_DIR, run_id, 'phases.json')
    try:
        raw = json.load(open(p))
    except (IOError, ValueError):
        return {}
    return {int(k): v for k, v in raw.items()}
