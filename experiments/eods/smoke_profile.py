"""Scoping smokes (Tom 2026-08-17): run ONE representative cell per
(experiment, size) sequentially — one deployment gets ALL cores — and
harvest timing/RAM profiles so we find the scaling walls BEFORE the
campaigns do.

Per cell, parsed from the run log's [mem] lines (t=epoch, iter, rss_mb,
sys_avail_mb) + wall clock:
  startup_s        proc start -> first training iteration (deployment
                   build + worker pickle broadcast — the known silent phase)
  s_per_iter       median + p90 of successive iter_start deltas
  driver_rss_gb    peak driver RSS
  sys_footprint_gb peak (total system RAM - available) delta from cell
                   start — approximates driver + all ray workers
  wall_s, iters, and $ at --spot-rate per vCPU-hour

    python -m experiments.eods.smoke_profile --manifest cluster/manifests/eods_manifest.json \
        --picks eods_a5:1,eods_a10:1 --workers 60 --out cache/eods/profiles
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MEM_RE = re.compile(
    r'\[mem\] tag=iter_start rss_mb=(\d+) vms_mb=\d+ peak_mb=(\d+) '
    r'sys_avail_mb=(\d+) pid=\d+ t=([0-9.]+) iter=(\d+)')


def profile_log(log_fn, t_start, wall_s):
    rows = []
    with open(log_fn, errors='replace') as f:
        for ln in f:
            m = MEM_RE.search(ln)
            if m:
                rows.append((float(m.group(4)), int(m.group(5)),
                             int(m.group(1)), int(m.group(3))))
    if not rows:
        return {'iters_seen': 0, 'wall_s': wall_s}
    rows.sort()
    ts = [r[0] for r in rows]
    deltas = sorted(ts[i + 1] - ts[i] for i in range(len(ts) - 1)
                    if 0 < ts[i + 1] - ts[i] < 3600)
    avail0 = rows[0][3]
    return {
        'iters_seen': rows[-1][1] + 1,
        'startup_s': round(ts[0] - t_start, 1),
        's_per_iter_p50': round(deltas[len(deltas) // 2], 2) if deltas else None,
        's_per_iter_p90': round(deltas[int(len(deltas) * 0.9)], 2) if deltas else None,
        'driver_rss_gb_peak': round(max(r[2] for r in rows) / 1024.0, 2),
        'sys_footprint_gb_peak': round(
            max(avail0 - r[3] for r in rows) / 1024.0, 2),
        'sys_avail_gb_min': round(min(r[3] for r in rows) / 1024.0, 1),
        'wall_s': wall_s,
    }


def run_pick(sp, seed, args):
    n = str(sp['n_values']).split(',')[0]
    out_dir = os.path.join(_REPO_ROOT, sp['out_root'], 'N{}'.format(n))
    ws = os.path.join(args.ws_root, sp['label'])
    for sub in ('runs', 'logs', 'figures/paper'):
        os.makedirs(os.path.join(ws, sub), exist_ok=True)
    for link in ('cache', 'data'):
        lp = os.path.join(ws, link)
        if not os.path.islink(lp):
            try:
                os.symlink(os.path.join(_REPO_ROOT, link), lp)
            except FileExistsError:
                pass
    env = dict(os.environ)
    env.update(sp.get('env') or {})
    env.update({'PYTHONPATH': _REPO_ROOT, 'MPLBACKEND': 'Agg',
                'RAY_ADDRESS': 'local',
                'RAY_TMPDIR': '/tmp/ray_prof',
                'SCULPTOR_N_WORKERS': str(args.workers),
                'SCULPTOR_RAY_NUM_CPUS': str(args.workers + 4),
                'SCULPTOR_ABLATION_GAMMA': sp.get('gamma', '0'),
                'SCULPTOR_ABLATION_PROBE_MODE': sp.get('probe_mode', 'fixed'),
                'SCULPTOR_ABLATION_PROBE_N': n})
    if args.max_iter:
        # scoping mode (Tom 2026-08-17): a few iterations per deployment
        # — we are scraping timing/RAM, not training. Override both the
        # CLI budget and the min-iter floor some specs pin via env.
        env['SCULPTOR_MAX_ITER'] = str(args.max_iter)
        env['SCULPTOR_ABLATION_MIN_ITER'] = str(args.max_iter)
    runner = sp.get('runner', 'experiments.ablation.run_fork_ladder')
    rung = sp['rungs'].split(',')[0]
    cmd = [sys.executable, '-u', '-m', runner, '--seed', str(seed),
           '--rung', rung, '--port', str(args.port), '--max-iter',
           str(args.max_iter or sp.get('max_iter', 200)),
           '--dpsize', sp['dpsize'], '--out-dir', out_dir]
    log_fn = os.path.join(ws, 'logs', 'profile_{}_s{}.log'.format(
        sp['label'], seed))
    print('[profile] START {} seed {} (workers={})'.format(
        sp['label'], seed, args.workers), flush=True)
    t0 = time.time()
    with open(log_fn, 'w') as lf:
        try:
            rc = subprocess.call(cmd, cwd=ws, env=env, stdout=lf,
                                 stderr=subprocess.STDOUT,
                                 timeout=args.cell_timeout)
        except subprocess.TimeoutExpired:
            # a timeout IS a datum (startup + RAM + partial iters land
            # in the log); it killed the whole ladder on 2026-08-18
            rc = -99
    wall = round(time.time() - t0, 1)
    prof = {'label': sp['label'], 'dpsize': sp['dpsize'], 'seed': seed,
            'runner': runner, 'workers': args.workers, 'rc': rc,
            'log': log_fn, **profile_log(log_fn, t0, wall)}
    prof['cell_core_hours'] = round(wall / 3600.0 * args.vcpus, 2)
    prof['cell_usd_spot'] = round(
        prof['cell_core_hours'] * args.spot_rate, 3)
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True, action='append')
    ap.add_argument('--picks', required=True,
                    help='comma list label:seed')
    ap.add_argument('--workers', type=int, default=60)
    ap.add_argument('--vcpus', type=int, default=96)
    ap.add_argument('--spot-rate', type=float, default=0.0148,
                    help='$/vCPU-hr (c7i spot median 2026-08-17)')
    ap.add_argument('--port', type=int, default=41000)
    ap.add_argument('--ws-root', default=os.path.expanduser('~/prof_ws'))
    ap.add_argument('--out', default='cache/eods/profiles')
    ap.add_argument('--cell-timeout', type=float, default=4 * 3600)
    ap.add_argument('--max-iter', type=int, default=None,
                    help='override every pick to N training iters')
    args = ap.parse_args()

    specs = {}
    for mf in args.manifest:
        for sp in json.load(open(mf)):
            specs[sp['label']] = sp
    out_dir = os.path.join(_REPO_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for pick in args.picks.split(','):
        label, seed = pick.split(':')
        prof = run_pick(specs[label], int(seed), args)
        results.append(prof)
        with open(os.path.join(out_dir, 'profiles.json'), 'w') as f:
            json.dump(results, f, indent=1)
        print('[profile] DONE {}: {}'.format(label, json.dumps(
            {k: prof.get(k) for k in (
                'rc', 'wall_s', 'startup_s', 's_per_iter_p50',
                'iters_seen', 'driver_rss_gb_peak',
                'sys_footprint_gb_peak', 'cell_usd_spot')})), flush=True)
    print('[profile] all picks complete ->', out_dir)


if __name__ == '__main__':
    main()
