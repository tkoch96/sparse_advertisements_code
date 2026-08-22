"""Per-worker RAM: which objects hold it, and how that scales with size.

`[mem-worker]` already answers "how much RSS does a worker have". This
answers "held by WHAT", which is the question that decides whether a
deployment size fits on a cheaper instance family (Tom's todo #3: reduce
per-process RAM to reduce VM cost).

Source lines, emitted by `core/path_distribution_computer._log_objsize_worker`
under `SCULPTOR_LOG_OBJSIZE=1`:

    [objsize idx=2] tag=post_update_deployment_dp<dpsize> attr=deployment mb=3.2 n=23 t=...
    [objsize idx=2] tag=post_update_deployment_dp<dpsize> TOTAL_mb=14.3 attrs=6 census_s=0.18 t=...

Two sources, in preference order:

1. `logs/workers/sculptor_worker_<i>_<pid>.log` -- the per-worker files.
   AUTHORITATIVE. Ray deduplicates identical worker stdout, so the driver
   log carries one worker's census plus a "[repeated Nx across cluster]"
   marker; the real per-worker spread only exists in these files.
2. the driver log -- fallback for runs launched before the per-worker
   files were routed into the run dir (before 2026-08-21).

Sizes are reported as the deployment size integer parsed out of the tag
(`...dptesting_feature-actual-10` -> 10).
"""

from __future__ import annotations

import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO, 'cache', 'cluster_runs')

_ATTR = re.compile(
    r'\[objsize idx=(?P<idx>\d+)\] tag=(?P<tag>\S+) attr=(?P<attr>\S+) '
    r'mb=(?P<mb>[\d.]+) n=(?P<n>\S*)')
_TOTAL = re.compile(
    r'\[objsize idx=(?P<idx>\d+)\] tag=(?P<tag>\S+) TOTAL_mb=(?P<mb>[\d.]+) '
    r'attrs=(?P<attrs>\d+) census_s=(?P<cs>[\d.]+)')
_SIZE_IN_TAG = re.compile(r'actual-(\d+)')


def _sources(run_id):
    d = os.path.join(RUNS_DIR, run_id, 'logs')
    worker_files = sorted(glob.glob(os.path.join(d, 'workers', '*.log')))
    if worker_files:
        return worker_files, 'per-worker files'
    p = os.path.join(d, 'run.log')
    return ([p] if os.path.exists(p) else []), 'driver log (Ray-deduped)'


def parse(run_id):
    """-> {'by_size': {size: {...}}, 'source': str}"""
    files, source = _sources(run_id)
    if not files:
        return {}
    # {size: {attr: {worker_idx: max_mb}}} and {size: {worker_idx: max_total}}
    attrs, totals, ns = {}, {}, {}
    for fp in files:
        try:
            txt = open(fp, errors='replace').read().replace('\r', '\n')
        except (IOError, OSError):
            continue
        for m in _ATTR.finditer(txt):
            k = _SIZE_IN_TAG.search(m.group('tag'))
            if not k:
                continue
            size, idx = int(k.group(1)), int(m.group('idx'))
            mb = float(m.group('mb'))
            a = attrs.setdefault(size, {}).setdefault(m.group('attr'), {})
            # Peak per worker: the census runs at every update_deployment,
            # and the largest is what has to fit in RAM.
            a[idx] = max(a.get(idx, 0.0), mb)
            if m.group('n'):
                ns.setdefault(size, {})[m.group('attr')] = m.group('n')
        for m in _TOTAL.finditer(txt):
            k = _SIZE_IN_TAG.search(m.group('tag'))
            if not k:
                continue
            size, idx = int(k.group(1)), int(m.group('idx'))
            t = totals.setdefault(size, {})
            t[idx] = max(t.get(idx, 0.0), float(m.group('mb')))

    by_size = {}
    for size in sorted(set(list(attrs) + list(totals))):
        per_worker = totals.get(size, {})
        rows = []
        for attr, per in sorted(attrs.get(size, {}).items(),
                                key=lambda kv: -max(kv[1].values())):
            vals = list(per.values())
            rows.append({'attr': attr,
                         'max_mb': max(vals), 'mean_mb': sum(vals) / len(vals),
                         'workers': len(vals),
                         'n': ns.get(size, {}).get(attr, '')})
        tv = list(per_worker.values())
        by_size[size] = {
            'attrs': rows,
            'n_workers_seen': len(tv),
            'worker_total_max_mb': max(tv) if tv else
                (sum(r['max_mb'] for r in rows) if rows else 0.0),
            'worker_total_mean_mb': (sum(tv) / len(tv)) if tv else 0.0,
        }
    return {'by_size': by_size, 'source': source}


def write_json(run_id):
    d = parse(run_id)
    if not d or not d.get('by_size'):
        return None
    p = os.path.join(RUNS_DIR, run_id, 'objsize.json')
    try:
        with open(p, 'w') as fh:
            json.dump({'source': d['source'],
                       'by_size': {str(k): v
                                   for k, v in d['by_size'].items()}},
                      fh, indent=1)
    except OSError:
        return None
    return p


def load(run_id):
    p = os.path.join(RUNS_DIR, run_id, 'objsize.json')
    try:
        raw = json.load(open(p))
    except (IOError, ValueError):
        return {}
    return {'source': raw.get('source', ''),
            'by_size': {int(k): v for k, v in raw.get('by_size', {}).items()}}
