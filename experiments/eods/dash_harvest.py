"""Head-side EODS-25 dash harvest (runs via refresh.py remote_harvest
each ~3-min cycle, so it must stay CHEAP): distill the campaign store
into cache/eods/v1_dash/ — the only dir the Mac pulls. The per-cell
metrics pickles are deployment-sized (too big to mirror); merge them
head-side into the stats-only metrics_by_dpsize.pkl, and only when a
cell pickle is actually newer than the merged output.

    python -m experiments.eods.dash_harvest \
        [--store cache/eods/v1] [--ws ~/eods25_ws]
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', default='cache/eods/v1')
    ap.add_argument('--ws', default=os.path.expanduser('~/eods25_ws'))
    args = ap.parse_args()
    store = (args.store if os.path.isabs(args.store)
             else os.path.join(_REPO_ROOT, args.store))
    dash = store.rstrip('/') + '_dash'
    os.makedirs(dash, exist_ok=True)

    # cell result JSONs + inprog markers (small)
    for pat in ('actual-*/N1/seed_*.json', 'actual-*/N1/*.inprog'):
        for fn in glob.glob(os.path.join(store, pat)):
            shutil.copy2(fn, dash)
    # prune stale inprog markers (queue removes them when the cell exits)
    for m in glob.glob(os.path.join(dash, '*.inprog')):
        src = os.path.join(store, 'actual-25', 'N1', os.path.basename(m))
        if not os.path.exists(src):
            os.remove(m)

    # stats-only merge, gated on pickle mtimes (merge loads every big
    # per-cell pickle -- only pay that when a cell actually landed)
    out_pkl = os.path.join(dash, 'metrics_by_dpsize.pkl')
    cells = glob.glob(os.path.join(store, 'actual-*/N1/seed_*_metrics.pkl'))
    newest = max([os.path.getmtime(f) for f in cells], default=0)
    if cells and (not os.path.exists(out_pkl)
                  or newest > os.path.getmtime(out_pkl)):
        subprocess.call([sys.executable, '-m', 'experiments.eods.merge_eods',
                         '--store', store, '--out', out_pkl],
                        cwd=_REPO_ROOT, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

    # smoke profile summary, if the profiler has run on this host
    prof = os.path.join(_REPO_ROOT, 'cache', 'eods', 'smoke25_profiles',
                        'profiles.json')
    if os.path.exists(prof):
        shutil.copy2(prof, os.path.join(dash, 'smoke_profiles.json'))

    # per-cycle system sample: load, free RAM, per-worker RSS (Tom
    # 2026-08-19 run-inspection dash)
    import time as _t
    try:
        la = open('/proc/loadavg').read().split()[0]
        avail = [l for l in open('/proc/meminfo') if 'MemAvailable' in l][0].split()[1]
        r = subprocess.run(['ps', '-eo', 'rss,comm'], capture_output=True, text=True, timeout=15)
        wrss = sorted(int(l.split()[0]) for l in r.stdout.splitlines()
                      if 'ray::' in l and int(l.split()[0]) > 200000)
        drss = max((int(l.split()[0]) for l in r.stdout.splitlines()
                    if l.strip().endswith('python') and int(l.split()[0]) > 2000000), default=0)
        import numpy as _np
        row = [int(_t.time()), la, int(avail)//1024, len(wrss),
               int(_np.percentile(wrss,50))//1024 if wrss else 0,
               int(_np.percentile(wrss,90))//1024 if wrss else 0,
               (max(wrss)//1024) if wrss else 0, drss//1024]
        with open(os.path.join(dash, 'sys_samples.csv'), 'a') as f:
            f.write(','.join(str(x) for x in row) + '\n')
    except Exception:
        pass

    # timing lines for statistics (bounded)
    with open(os.path.join(dash, 'timing_lines.txt'), 'w') as out:
        for fn in sorted(glob.glob(os.path.join(args.ws, '*', 'logs', '*.log'))):
            try:
                r = subprocess.run(['grep', '-hE',
                    'ms per iter|benefit grad took|Timer: |\\[wt\\] |objective',
                    fn], capture_output=True, text=True, timeout=30)
                out.write(''.join(r.stdout.splitlines(True)[-3000:]))
            except Exception:
                pass

    # [mem] telemetry + log tails from the queue workspaces
    logs = sorted(glob.glob(os.path.join(args.ws, '*', 'logs', '*.log')))
    with open(os.path.join(dash, 'mem_iter.txt'), 'w') as out:
        for fn in logs:
            try:
                r = subprocess.run(
                    ['grep', '-h', 'tag=iter_start', fn],
                    capture_output=True, text=True, timeout=30)
                lines = r.stdout.splitlines()[-400:]
                for ln in lines:
                    out.write('{}\t{}\n'.format(os.path.basename(fn), ln))
            except Exception:
                pass
    with open(os.path.join(dash, 'log_tails.txt'), 'w') as out:
        for fn in logs:
            out.write('==> {} <==\n'.format(fn))
            try:
                r = subprocess.run(['tail', '-n', '150', fn],
                                   capture_output=True, text=True,
                                   timeout=15)
                out.write(r.stdout + '\n')
            except Exception:
                pass
    print('[dash_harvest] {} jsons, {} logs -> {}'.format(
        len(glob.glob(os.path.join(dash, 'seed_*.json'))), len(logs), dash))


if __name__ == '__main__':
    main()
