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
specs = json.load(open("/home/ubuntu/v3grid_manifest.json"))
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
                est += (min(n, mi) if sp["label"] == "L1"
                        else (100 if mi <= 100 else 150))
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
    with open(os.path.join(REPO, 'dashboard_site', 'progress.json'), 'w') as f:
        json.dump(data, f)
    return 'ok {done_it}/{est_total}'.format(**data)


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
