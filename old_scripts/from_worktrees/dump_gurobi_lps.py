"""Generate representative MPS + Gurobi log files from SCULPTOR's LP workload.

Loads a deployment + advertisement (from saved eval pickles or synthetic
deployment_setup) and solves one LP via the production code path with
Gurobi's model.write() + LogFile hook turned on. Output goes to a chosen
directory as `<tag>.mps` + `<tag>.log`.

Generates files at multiple representative sizes so the Gurobi support
team can see how the LP structure scales.

Usage:
  cd ~/Documents/sparse_advertisements_code
  ~/Documents/venv312/bin/python benchmarks/dump_gurobi_lps.py \\
    --out-dir /tmp/gurobi_dumps

Output layout:
  /tmp/gurobi_dumps/
    small_sparse.mps        small_sparse.log     # synthetic small (15 metros, 3 PoPs, ~51 popps)
    decent_sparse.mps       decent_sparse.log    # synthetic decent (200 metros, 10 PoPs, ~294 popps)
    actual32_sparse.mps     actual32_sparse.log  # real Vultr-32 (5173 UGs, 779 popps)

Run from the cluster head to avoid burning local WLS sessions if the
sweep is running.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
os.chdir(_REPO)

import worker_comms_ray as _ray_mod  # noqa: E402
sys.modules['worker_comms'] = _ray_mod

import numpy as np  # noqa: E402

# ------------------------------------------------------------------ #
# Monkey-patch gp.Model.optimize to dump .mps + set LogFile before
# every solve. Restored on exit.
# ------------------------------------------------------------------ #
import gurobipy as gp  # noqa: E402

_real_optimize = gp.Model.optimize
_dump_state = {'dir': None, 'tag': None, 'count': 0}


def _patched_optimize(self):
    d = _dump_state.get('dir')
    if d:
        _dump_state['count'] += 1
        n = _dump_state['count']
        tag = _dump_state.get('tag') or 'lp'
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, f"{tag}_{n:03d}")
        # Configure log file BEFORE optimize so the log captures the solve.
        # Also bump LogToConsole=1 momentarily for stdout context, but the
        # production code sets LogToConsole=0 right before this; we don't
        # override that.
        self.Params.LogFile = f"{base}.log"
        try:
            self.write(f"{base}.mps")
            print(f"[mps-dump] wrote {base}.mps + {base}.log "
                  f"(nvars={self.NumVars}, nconstrs={self.NumConstrs})", flush=True)
        except Exception as e:
            print(f"[mps-dump] WRITE FAILED for {base}: {e}", flush=True)
    return _real_optimize(self)


gp.Model.optimize = _patched_optimize


# ------------------------------------------------------------------ #
# After patching, import the SCULPTOR stack
# ------------------------------------------------------------------ #
from sparse_advertisements_v3 import Sparse_Advertisement_Eval  # noqa: E402
from deployment_setup import get_random_deployment  # noqa: E402


def solve_one_representative_lp(deployment, adv, tag, out_dir):
    """Build an SAS, call solve_lp_with_failure_catch on the given adv,
    triggering one Gurobi LP solve that gets dumped as <tag>_001.mps."""
    _dump_state['dir'] = out_dir
    _dump_state['tag'] = tag
    _dump_state['count'] = 0
    n_prefixes = adv.shape[1]
    print(f"\n=== building SAS for tag={tag}, n_popps={len(deployment['popps'])}, "
          f"n_ugs={len(deployment['ug_perfs'])}, n_prefixes={n_prefixes} ===", flush=True)
    sas = Sparse_Advertisement_Eval(
        deployment, verbose=False, lambduh=0.000007, with_capacity=True,
        using_resilience_benefit=True, gamma=1.0,
        n_prefixes=n_prefixes, generic_objective='avg_latency')
    # No worker_manager set → solve_lp_with_failure_catch routes to the
    # LOCAL non-persistent LP via solve_lp_assignment (the simpler LP
    # path; the worker-side persistent LP has a slightly different
    # structure but the constraint set is conceptually the same).
    sas.update_deployment(deployment)
    t0 = time.time()
    ret = sas.solve_lp_with_failure_catch(adv)
    elapsed = time.time() - t0
    print(f"[solve_one_representative_lp] tag={tag} solved in {elapsed:.2f}s, "
          f"{_dump_state['count']} model writes; "
          f"objective={ret.get('objective')}, "
          f"fraction_congested_volume={ret.get('fraction_congested_volume')}", flush=True)
    # Reset for next call
    _dump_state['dir'] = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/tmp/gurobi_dumps")
    parser.add_argument("--sizes", default="small,decent,actual32",
                        help="comma-separated subset of {small, decent, actual32}")
    parser.add_argument("--actual32-pickle",
                        default=os.path.expanduser(
                            "~/Documents/sparse_advertisements_code/recovered_actual32/"
                            "popp_failure_latency_comparison_actual-32_FULL.pkl"),
                        help="Path to the recovered actual-32 pickle (used to "
                             "extract deployment + sparse adv).")
    args = parser.parse_args()
    sizes = [s.strip() for s in args.sizes.split(',') if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    if 'small' in sizes:
        np.random.seed(1)
        dep = get_random_deployment('small')
        # Build a simple sparse-style adv: anycast on prefix 0 + 1 popp per
        # other prefix. n_prefixes = small's deployment_to_prefixes value.
        n_popp = len(dep['popps'])
        n_pref = max(2, min(n_popp, 8))
        adv = np.zeros((n_popp, n_pref))
        adv[:, 0] = 1  # anycast prefix
        for i in range(1, n_pref):
            adv[i % n_popp, i] = 1
        adv = adv.astype(np.float32)
        solve_one_representative_lp(dep, adv, 'small_sparse', args.out_dir)

    if 'decent' in sizes:
        np.random.seed(1)
        dep = get_random_deployment('decent')
        n_popp = len(dep['popps'])
        n_pref = max(2, min(n_popp, 16))
        adv = np.zeros((n_popp, n_pref))
        adv[:, 0] = 1
        for i in range(1, n_pref):
            adv[i % n_popp, i] = 1
        adv = adv.astype(np.float32)
        solve_one_representative_lp(dep, adv, 'decent_sparse', args.out_dir)

    if 'actual32' in sizes:
        if not os.path.exists(args.actual32_pickle):
            print(f"[skip] actual32: pickle not found at {args.actual32_pickle}")
        else:
            p = pickle.load(open(args.actual32_pickle, 'rb'))
            dep = p['deployment'][0]
            adv = p['adv'][0]['sparse']
            if isinstance(adv, list) and len(adv) > 0:
                adv = adv[0]
            if adv is None or (hasattr(adv, '__len__') and len(adv) == 0):
                # Fall back to the per-strategy adv on compare_rets
                adv = p['compare_rets'][0]['adv_solns']['sparse'][0]
            solve_one_representative_lp(dep, adv, 'actual32_sparse', args.out_dir)

    print(f"\n=== done. files in {args.out_dir} ===", flush=True)
    for f in sorted(os.listdir(args.out_dir)):
        p = os.path.join(args.out_dir, f)
        print(f"  {f}  {os.path.getsize(p)/1024:.1f} KB")


if __name__ == '__main__':
    main()
