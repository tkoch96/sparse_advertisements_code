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

    # ---- painter-anchored view: benefit over painter per seed (ms) ------
    # benefit_i = painter_i - rung_i on the SAME seed (positive = better
    # than painter; OPP row shows the max attainable). Quantiles give the
    # distribution across seeds.
    if pain:
        print(f"\nbenefit over painter, per-seed paired (combined, gamma={args.gamma}; "
              f"positive = better; ms):")
        qhdr = (f"{'rung':<14}{'median':>10}{'mean':>10}{'p10':>10}{'p25':>10}"
                f"{'p75':>10}{'p90':>10}{'min':>10}{'max':>10}{'n':>4}")
        print(qhdr); print("-" * len(qhdr))
        import numpy as _np
        for rung in rungs:
            if rung == "painter":
                continue
            vals = []
            for s in seeds:
                d = rows[rung].get(s); p = pain.get(s)
                if d is not None and p is not None:
                    vals.append(combined(p, args.gamma) - combined(d, args.gamma))
            if not vals:
                print(f"{rung:<14}{'—':>10}"); continue
            v = _np.asarray(vals)
            print(f"{rung:<14}{_np.median(v):>10.2f}{v.mean():>10.2f}"
                  f"{_np.quantile(v,.1):>10.2f}{_np.quantile(v,.25):>10.2f}"
                  f"{_np.quantile(v,.75):>10.2f}{_np.quantile(v,.9):>10.2f}"
                  f"{v.min():>10.2f}{v.max():>10.2f}{len(v):>4}")
        # OPP = the maximum attainable benefit (painter's own combined)
        pv = _np.asarray([combined(p, args.gamma) for p in pain.values()])
        print(f"{'OPP (max)':<14}{_np.median(pv):>10.2f}{pv.mean():>10.2f}"
              f"{_np.quantile(pv,.1):>10.2f}{_np.quantile(pv,.25):>10.2f}"
              f"{_np.quantile(pv,.75):>10.2f}{_np.quantile(pv,.9):>10.2f}"
              f"{pv.min():>10.2f}{pv.max():>10.2f}{len(pv):>4}")

    # ---- percentage view: painter = 0%, one_per_peering (OPP) = 100% ----
    # pct = 100 * (1 - combined(rung)/combined(painter)), per seed (same
    # convention as plot_normalized). Blowups go hugely negative on purpose.
    pain = rows.get("painter", {})
    if pain:
        print(f"\n%% of painter->OPP gap closed (combined, gamma={args.gamma}; "
              f"painter=0%, OPP=100%):")
        print(hdr); print("-" * len(hdr))
        for rung in rungs:
            if rung == "painter":
                continue
            vals = []
            cells = ""
            for s in seeds:
                d = rows[rung].get(s)
                p = pain.get(s)
                if d is None or p is None or combined(p, args.gamma) <= 0:
                    cells += f"{'—':>12}"
                else:
                    pct = 100.0 * (1.0 - combined(d, args.gamma) / combined(p, args.gamma))
                    vals.append(pct)
                    cells += f"{pct:>12.1f}"
            med = f"{statistics.median(vals):>12.1f}" if vals else f"{'—':>12}"
            print(f"{rung:<14}{cells}{med}")
    else:
        print("\n(no painter runs in this dir -> percentage view skipped)")


if __name__ == "__main__":
    main()
