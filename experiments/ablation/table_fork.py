"""Print Tom's preferred ablation tables from trusted rescore JSONs.

combined = diff_vs_opp + gamma * (fail_popp.avg_lat_under_failure_abs
                                  - fail_popp.opp_avg_lat_under_failure_abs)

Only reads seed_<s>_<rung>.json files produced by rescore_fork.py
(rescored=True enforced). Pure stdlib; safe to run while sweeps are live.

Usage: python -m experiments.ablation.table_fork --in-dir cache/ablation/fork_5x200 [--gamma 0.1]
"""
import argparse, glob, json, os, re, statistics, sys

RUNG_ORDER = ["painter", "no_mc", "no_memory", "no_direction", "expl_none", "expl_random", "full"]


def load(in_dir):
    rows = {}
    for fn in sorted(glob.glob(os.path.join(in_dir, "seed_*_*.json"))):
        m = re.match(r"seed_(\d+)_(.+)\.json$", os.path.basename(fn))
        if not m:
            continue
        seed, rung = int(m.group(1)), m.group(2)
        with open(fn) as f:
            d = json.load(f)
        if not d.get("rescored"):
            print(f"[table] SKIPPING {fn}: not a trusted rescore", file=sys.stderr)
            continue
        rows.setdefault(rung, {})[seed] = d
    return rows


def combined(d, gamma):
    fp = d["fail_popp"]
    return d["diff_vs_opp"] + gamma * (
        fp["avg_lat_under_failure_abs"] - fp["opp_avg_lat_under_failure_abs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--gamma", type=float, default=0.1)
    args = ap.parse_args()

    rows = load(args.in_dir)
    if not rows:
        print("no rescored JSONs found"); return
    seeds = sorted({s for r in rows.values() for s in r})
    rungs = [r for r in RUNG_ORDER if r in rows] + sorted(set(rows) - set(RUNG_ORDER))

    print(f"\n{args.in_dir}  (gamma={args.gamma}; combined vs opp; lower better; opp=0)")
    hdr = f"{'rung':<14}" + "".join(f"{'s'+str(s):>12}" for s in seeds) + f"{'median':>12}"
    print(hdr); print("-" * len(hdr))
    for rung in rungs:
        vals = []
        cells = ""
        for s in seeds:
            d = rows[rung].get(s)
            if d is None:
                cells += f"{'—':>12}"
            else:
                c = combined(d, args.gamma)
                vals.append(c)
                cells += f"{c:>12.2f}"
        med = f"{statistics.median(vals):>12.2f}" if vals else f"{'—':>12}"
        print(f"{rung:<14}{cells}{med}")

    print(f"\nsteady-state only (diff_vs_opp):")
    print(hdr); print("-" * len(hdr))
    for rung in rungs:
        vals = []
        cells = ""
        for s in seeds:
            d = rows[rung].get(s)
            if d is None:
                cells += f"{'—':>12}"
            else:
                vals.append(d["diff_vs_opp"])
                cells += f"{d['diff_vs_opp']:>12.2f}"
        med = f"{statistics.median(vals):>12.2f}" if vals else f"{'—':>12}"
        print(f"{rung:<14}{cells}{med}")


if __name__ == "__main__":
    main()
