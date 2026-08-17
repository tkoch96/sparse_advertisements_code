"""30-second grid-progress ticker (Tom 2026-08-16: "put that on a 30
second updater").

Every TICK seconds: ssh to the head (IP re-resolved from the cluster
alert JSON per use -- never cached), compute completed learning
iterations vs estimated total straight from the head's manifest + landed
cell JSONs, and write dashboard_site/progress.json. The bar in the
ladder tab fetches that file on the same cadence client-side, so the
display updates without waiting for the 3-minute refresh cycle.

Run (survives like the refresh loop):
    nohup python -m experiments.dashboard.progress_tick --loop > /dev/null 2>&1 &
"""
import argparse
import json
import os
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
ALERT = os.path.expanduser('~/.sculptor_cluster_alert/active_cluster.json')
SSH_KEY = os.path.expanduser('~/.ssh/ray-autoscaler_us-east-1.pem')

REMOTE_PY = r'''
import glob, json, time
# HiGHS-era campaign (Tom 2026-08-17): count the live highs grid; the
# gurobi-era v3 grid is frozen at 971/1440 pending license.
specs = json.load(open("/home/ubuntu/sparse_advertisements_code/"
                       "tools/grid_georand_manifest.json"))
done_it = est = done_cells = total = 0
root = "/home/ubuntu/sparse_advertisements_code/"
for sp in specs:
    ns = [int(n) for n in str(sp["n_values"]).split(",")]
    ss = str(sp.get("seeds", "1-5"))
    seeds = (list(range(int(ss.split("-")[0]), int(ss.split("-")[1]) + 1))
             if "-" in ss else [int(x) for x in ss.split(",")])
    mi = int(sp.get("max_iter", 100))
    for n in ns:
        for s in seeds:
            total += 1
            hit = glob.glob(root + sp["out_root"] +
                            "/N{}/seed_{}_*.json".format(n, s))
            got = None
            if hit:
                try:
                    got = json.load(open(hit[0])).get("n_iters")
                except Exception:
                    got = None
            if got:
                done_it += int(got); est += int(got); done_cells += 1
            else:
                est += (min(n, mi) if sp["label"].endswith("L1")
                        else (100 if mi <= 100 else 150))
# In-flight iterations (Tom 2026-08-17, iteration-rate ticker): each
# RUNNING cell's log carries '[mem] tag=iter_start ... iter=N' lines;
# the last one is the cell's current iteration. Only recently-written
# logs count (dead logs from finished cells are excluded by mtime).
import os
inflight_it = 0
now = time.time()
for lf in glob.glob("/home/ubuntu/hx_ws/S*/logs/*.log"):
    try:
        if now - os.path.getmtime(lf) > 600:
            continue
        last = None
        with open(lf, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 65536))
            for ln in f.read().decode("utf-8", "replace").splitlines():
                i = ln.rfind(" iter=")
                if i != -1:
                    v = ln[i + 6:].split()[0]
                    if v.isdigit():
                        last = int(v)
        if last:
            inflight_it += last
    except Exception:
        pass
mem = {}
for line in open("/proc/meminfo"):
    p = line.split()
    if p[0].rstrip(":") in ("MemTotal", "MemAvailable"):
        mem[p[0].rstrip(":")] = int(p[1])
ram_total_gb = mem["MemTotal"] / 1048576.0
ram_used_gb = (mem["MemTotal"] - mem["MemAvailable"]) / 1048576.0
import os
load1 = float(open("/proc/loadavg").read().split()[0])
cores = os.cpu_count() or 1
cells = 0
for d in glob.glob("/proc/[0-9]*/cmdline"):
    try:
        if b"run_fork_ladder" in open(d, "rb").read():
            cells += 1
    except Exception:
        pass
print(json.dumps({"done_it": done_it, "est_total": est,
                  "done_cells": done_cells, "total_cells": total,
                  "ram_used_gb": round(ram_used_gb, 1),
                  "ram_total_gb": round(ram_total_gb, 1),
                  "ram_pct": round(100 * ram_used_gb / ram_total_gb, 1),
                  "cpu_pct": round(min(100.0, 100 * load1 / cores), 1),
                  "load1": load1, "cores": cores, "cells_running": cells,
                  "inflight_it": inflight_it,
                  "ts": time.strftime("%H:%M:%SZ", time.gmtime())}))
'''


def tick():
    try:
        ip = json.load(open(ALERT))['head']['public_ip']
    except Exception as e:
        return 'no head ip: {}'.format(e)
    try:
        out = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=10', '-i', SSH_KEY,
             'ubuntu@{}'.format(ip),
             # single string so the REMOTE shell keeps -c's arg intact
             '/home/ubuntu/venv/bin/python -c '
             '"import sys; exec(sys.stdin.read())"'],
            input=REMOTE_PY, capture_output=True, text=True, timeout=25)
        line = out.stdout.strip().splitlines()[-1]
        data = json.loads(line)
    except Exception as e:
        return 'tick failed: {}'.format(e)
    _add_rate_and_eta(data)
    with open(os.path.join(REPO, 'dashboard_site', 'progress.json'), 'w') as f:
        json.dump(data, f)
    return 'ok {done_it}/{est_total}'.format(**data)


RATE_STATE = os.path.expanduser('~/sculptor_dashboard/tick_rate_state.json')


def _add_rate_and_eta(data):
    """Iteration rate + full-grid ETA (Tom 2026-08-17). Progress metric =
    landed iterations + in-flight iterations (running cells' current
    iter, from their logs), sampled every tick into a rolling ~30-min
    window. Rate over the window -> fleet it/s, per-cell s/iter, and
    remaining/rate -> ETA. Queue restarts make the metric dip (in-flight
    resets); negative or ~zero rates just blank the ETA for a few ticks
    until the window recovers."""
    now = time.time()
    prog = float(data['done_it'] + data.get('inflight_it', 0))
    try:
        st = json.load(open(RATE_STATE))
        hist, rate_ema = st['hist'], st.get('rate_ema')
    except Exception:
        hist, rate_ema = [], None
    # dip-reset: ONLY for catastrophic drops (a queue restart zeroes all
    # in-flight iterations at once). Ordinary in-flight noise — cells
    # finishing, logs aging out of the mtime window — dips by hundreds
    # routinely, and a 500-threshold reset wiped the window every few
    # minutes (permanent 'warming up', Tom 2026-08-17). The regression
    # absorbs moderate dips; only a multi-thousand step-down means the
    # world actually changed.
    if hist and prog < max(h[1] for h in hist) - 3000:
        hist, rate_ema = [], None
    hist.append([now, prog])
    hist = [h for h in hist if now - h[0] <= 7200][-240:]
    span = now - hist[0][0]
    if len(hist) < 8 or span < 300:
        rate = None
    else:
        # least-squares slope over the FULL window: every 30s sample
        # contributes, so single lumpy ticks (a cell landing, a log
        # rotating) barely move it — the 2-point secant it replaces
        # swung the ETA by hours on 1s/iter of noise (Tom 2026-08-17)
        n = float(len(hist))
        mt = sum(h[0] for h in hist) / n
        mp = sum(h[1] for h in hist) / n
        den = sum((h[0] - mt) ** 2 for h in hist)
        rate = (sum((h[0] - mt) * (h[1] - mp) for h in hist) / den
                if den > 0 else None)
    if rate is not None and rate > 0:
        # light EMA across ticks on top of the windowed slope
        rate_ema = rate if rate_ema is None else 0.85 * rate_ema + 0.15 * rate
    try:
        json.dump({'hist': hist, 'rate_ema': rate_ema},
                  open(RATE_STATE, 'w'))
    except OSError:
        pass
    if not rate_ema or rate_ema <= 0:
        data['it_per_s'] = None
        data['eta_s'] = None
        data['eta_str'] = None
        data['sec_per_iter_cell'] = None
        return
    remaining = max(0.0, float(data['est_total']) - prog)
    eta_s = remaining / rate_ema
    cells = max(1, int(data.get('cells_running') or 0))
    data['it_per_s'] = round(rate_ema, 2)
    data['sec_per_iter_cell'] = round(cells / rate_ema, 2)
    data['eta_s'] = int(eta_s)
    # 10-minute display granularity: the number should breathe, not flicker
    q = int(round(eta_s / 600.0)) * 600
    h, m = q // 3600, (q % 3600) // 60
    data['eta_str'] = ('~{}h{:02d}m'.format(h, m) if h
                       else '~{}m'.format(max(m, 10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--loop', action='store_true')
    ap.add_argument('--tick', type=int, default=30)
    args = ap.parse_args()
    while True:
        msg = tick()
        print(time.strftime('%H:%M:%S'), msg, flush=True)
        if not args.loop:
            break
        time.sleep(args.tick)


if __name__ == '__main__':
    main()
