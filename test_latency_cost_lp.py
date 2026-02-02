# test_latency_cost_lp_pretty.py
import numpy as np
from types import SimpleNamespace

import solve_lp_assignment as sla


def NO_PATH_INGRESS(sas):
    return sas.n_popps


sla.NO_PATH_INGRESS = NO_PATH_INGRESS


if not hasattr(sla, "NO_ROUTE_LATENCY"):
    sla.NO_ROUTE_LATENCY = 1e6

if not hasattr(sla, "N_WORKERS_GENERIC"):
    sla.N_WORKERS_GENERIC = 1


# -----------------------------
# Create a minimum sas object for testing
# fake sas with SimpleNamespace
def make_sas(popps, ug_vols, ug_perfs, caps):
    """
    popps: list of (site, peer) tuples, length = n_popps
    ug_vols: dict ug -> volume
    ug_perfs: dict ug -> dict popp(tuple) -> latency
    caps: list/np array length = n_popps, capacity per popp
    """
    ugs = list(ug_vols.keys())
    ug_to_ind = {ug: i for i, ug in enumerate(ugs)}
    sas = SimpleNamespace()
    sas.popps = popps
    sas.n_popps = len(popps)
    sas.whole_deployment_ugs = ugs
    sas.whole_deployment_ug_to_ind = ug_to_ind
    sas.whole_deployment_n_ug = len(ugs)
    sas.whole_deployment_ug_to_vol = ug_vols
    sas.whole_deployment_ug_vols = np.array([ug_vols[ug] for ug in ugs], dtype=float)
    sas.whole_deployment_ug_perfs = ug_perfs
    sas.link_capacities_arr = np.array(caps, dtype=float)
    return sas


def simple_get_paths_by_ug(sas, routed_through_ingress):
    """
    routed_through_ingress: dict ug -> list of poppi indices feasible for that ug
    Returns:
      available_paths: list[(ug, poppi)]
      paths_by_ug: dict ug -> list[poppi]
    """
    available_paths = []
    paths_by_ug = {}
    for ug in sas.whole_deployment_ugs:
        poppis = sorted(set(routed_through_ingress.get(ug, [])))
        if len(poppis) == 0:
            poppis = [NO_PATH_INGRESS(sas)]
        paths_by_ug[ug] = poppis
        for poppi in poppis:
            available_paths.append((ug, poppi))
    return available_paths, paths_by_ug


def compute_vols_by_poppi_from_solution(available_paths, sol, n_popps_plus_no_path):
    vols = np.zeros(n_popps_plus_no_path, dtype=float)
    for (_ug, poppi), v in zip(available_paths, sol):
        vols[poppi] += float(v)
    return vols


def compute_lat_by_ug_from_solution(sas, available_paths, sol):
    numer = {ug: 0.0 for ug in sas.whole_deployment_ugs}
    denom = {ug: 0.0 for ug in sas.whole_deployment_ugs}
    for (ug, poppi), v in zip(available_paths, sol):
        v = float(v)
        if v <= 0:
            continue
        if poppi == NO_PATH_INGRESS(sas):
            lat = float(sla.NO_ROUTE_LATENCY)
        else:
            popp = sas.popps[poppi]
            lat = float(sas.whole_deployment_ug_perfs[ug][popp])
        numer[ug] += lat * v
        denom[ug] += v
    return {ug: (numer[ug] / denom[ug] if denom[ug] > 0 else 0.0) for ug in sas.whole_deployment_ugs}


def print_problem_and_solution(
    sas,
    routed,
    available_paths,
    sol,
    alpha,
    site_cost,
    title=None,
):
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("-" * 72)

    # Problem setup
    print("UGs (demand):")
    for ug in sas.whole_deployment_ugs:
        print(f"  {ug}: {sas.whole_deployment_ug_to_vol[ug]}")

    print("\nIngresses (poppi -> (site, peer), cap):")
    for i, popp in enumerate(sas.popps):
        print(f"  {i}: {popp}, cap={float(sas.link_capacities_arr[i])}")

    print("\nSite costs (SITE_COST):")
    # print all sites seen in popps, defaulting missing to 0.0
    sites = sorted({site for (site, _peer) in sas.popps})
    for s in sites:
        print(f"  {s}: {float(site_cost.get(s, 0.0))}")

    print("\nUG latencies (ms):")
    for ug in sas.whole_deployment_ugs:
        print(f"  {ug}:")
        for poppi in routed.get(ug, []):
            site, peer = sas.popps[poppi]
            lat = float(sas.whole_deployment_ug_perfs[ug][(site, peer)])
            print(f"    -> popp {poppi} ({site},{peer}): {lat}")

    print(f"\nalpha = {alpha}")

    # Solution aggregates
    vols_by_poppi = compute_vols_by_poppi_from_solution(available_paths, sol, sas.n_popps + 1)
    lat_by_ug = compute_lat_by_ug_from_solution(sas, available_paths, sol)

    # Per-path allocations (non-zero)
    print("\nSolution allocations (volume):")
    any_printed = False
    for (ug, poppi), v in zip(available_paths, sol):
        v = float(v)
        if v > 1e-6:
            site, peer = sas.popps[poppi]
            print(f"  {ug} -> popp {poppi} ({site},{peer}): {v:.4f}")
            any_printed = True
    if not any_printed:
        print("  (all ~0?)")

    print("\nTotal volume per ingress (and utilization):")
    for i in range(sas.n_popps):
        site, peer = sas.popps[i]
        cap = float(sas.link_capacities_arr[i])
        vol = float(vols_by_poppi[i])
        util = (vol / cap) if cap > 0 else float("inf")
        print(f"  popp {i} ({site},{peer}): {vol:.4f} / {cap:.4f}  (util {util:.3f})")

    print("\nLatency by UG (traffic-weighted ms):")
    for ug in sas.whole_deployment_ugs:
        print(f"  {ug}: {float(lat_by_ug[ug]):.4f}")

    print("=" * 72 + "\n")


# replace with simple_get_paths_by_ug
sla.get_paths_by_ug = simple_get_paths_by_ug



# Test 1: Alpha threshold (1 UG, 2 ingresses)

def test_alpha_threshold_flip():
    popps = [("A", "p1"), ("B", "p1")]
    ug_vols = {"UG0": 100.0}
    ug_perfs = {"UG0": {("A", "p1"): 10.0, ("B", "p1"): 20.0}}
    caps = [1000.0, 1000.0]  # no binding

    sas = make_sas(popps, ug_vols, ug_perfs, caps)
    routed = {"UG0": [0, 1]}

    sla.SITE_COST = {"A": 10.0, "B": 0.0}

    # alpha < 1 => all to popp0
    alpha = 0.5
    ret = sla.solve_latency_plus_site_cost_with_capacity(sas, routed, obj="lat_cost", alpha=alpha, threads=1)
    assert ret["solved"]
    sol = np.array(ret["raw_solution"], dtype=float)
    available_paths, _ = simple_get_paths_by_ug(sas, routed)

    print_problem_and_solution(
        sas, routed, available_paths, sol, alpha=alpha, site_cost=sla.SITE_COST,
        title="Test 1a: alpha=0.5 (should choose site A)"
    )

    vols_by_poppi = compute_vols_by_poppi_from_solution(available_paths, sol, sas.n_popps + 1)
    assert abs(vols_by_poppi[0] - 100.0) < 1e-6, vols_by_poppi
    assert abs(vols_by_poppi[1] - 0.0) < 1e-6, vols_by_poppi

    # alpha > 1 => all to popp1
    alpha = 2.0
    ret = sla.solve_latency_plus_site_cost_with_capacity(sas, routed, obj="lat_cost", alpha=alpha, threads=1)
    assert ret["solved"]
    sol = np.array(ret["raw_solution"], dtype=float)

    print_problem_and_solution(
        sas, routed, available_paths, sol, alpha=alpha, site_cost=sla.SITE_COST,
        title="Test 1b: alpha=2.0 (should choose site B)"
    )

    vols_by_poppi = compute_vols_by_poppi_from_solution(available_paths, sol, sas.n_popps + 1)
    assert abs(vols_by_poppi[1] - 100.0) < 1e-6, vols_by_poppi
    assert abs(vols_by_poppi[0] - 0.0) < 1e-6, vols_by_poppi



# Test 2: Capacity binding / spillover (2 UGs, best ingress saturates)

def test_capacity_spillover():
    popps = [("S0", "p1"), ("S1", "p1")]
    ug_vols = {"UG0": 100.0, "UG1": 80.0}

    ug_perfs = {
        "UG0": {("S0", "p1"): 5.0, ("S1", "p1"): 50.0},
        "UG1": {("S0", "p1"): 5.0, ("S1", "p1"): 50.0},
    }

    caps = [120.0, 1000.0]
    sas = make_sas(popps, ug_vols, ug_perfs, caps)
    routed = {"UG0": [0, 1], "UG1": [0, 1]}
    sla.SITE_COST = {"S0": 0.0, "S1": 0.0}

    alpha = 0.0
    ret = sla.solve_latency_plus_site_cost_with_capacity(sas, routed, obj="lat_cost", alpha=alpha, threads=1)
    assert ret["solved"]
    sol = np.array(ret["raw_solution"], dtype=float)

    available_paths, _ = simple_get_paths_by_ug(sas, routed)
    print_problem_and_solution(
        sas, routed, available_paths, sol, alpha=alpha, site_cost=sla.SITE_COST,
        title="Test 2: capacity spillover (popp0 should hit cap=120)"
    )

    vols_by_poppi = compute_vols_by_poppi_from_solution(available_paths, sol, sas.n_popps + 1)
    assert vols_by_poppi[0] <= 120.0 + 1e-6, vols_by_poppi
    assert abs(vols_by_poppi[0] - 120.0) < 1e-4, vols_by_poppi
    assert abs(np.sum(vols_by_poppi[:2]) - 180.0) < 1e-6, vols_by_poppi
    assert abs(vols_by_poppi[1] - 60.0) < 1e-4, vols_by_poppi



# Test 3: Cost-driven steering under equal latency

def test_cost_driven_steering_equal_latency():
    popps = [("EXP", "p1"), ("CHEAP", "p1")]
    ug_vols = {"UG0": 100.0}
    ug_perfs = {"UG0": {("EXP", "p1"): 10.0, ("CHEAP", "p1"): 10.0}}
    caps = [1000.0, 1000.0]

    sas = make_sas(popps, ug_vols, ug_perfs, caps)
    routed = {"UG0": [0, 1]}
    sla.SITE_COST = {"EXP": 10.0, "CHEAP": 0.0}

    alpha = 100.0
    ret = sla.solve_latency_plus_site_cost_with_capacity(sas, routed, obj="lat_cost", alpha=alpha, threads=1)
    assert ret["solved"]
    sol = np.array(ret["raw_solution"], dtype=float)

    available_paths, _ = simple_get_paths_by_ug(sas, routed)
    print_problem_and_solution(
        sas, routed, available_paths, sol, alpha=alpha, site_cost=sla.SITE_COST,
        title="Test 3: equal latency, cost drives to CHEAP"
    )

    vols_by_poppi = compute_vols_by_poppi_from_solution(available_paths, sol, sas.n_popps + 1)
    assert abs(vols_by_poppi[1] - 100.0) < 1e-6, vols_by_poppi
    assert abs(vols_by_poppi[0] - 0.0) < 1e-6, vols_by_poppi



# Test 4: Two UGs, same site has two peers, capacity spill across peers
def test_two_ugs_same_site_two_peers_capacity_spill():
    popps = [
        ("S0", "p1"),
        ("S0", "p2"),
        ("S1", "p1"),
    ]

    ug_vols = {"UG0": 80.0, "UG1": 70.0}
    ug_perfs = {
        "UG0": {("S0", "p1"): 10.0, ("S0", "p2"): 20.0, ("S1", "p1"): 60.0},
        "UG1": {("S0", "p1"): 11.0, ("S0", "p2"): 21.0, ("S1", "p1"): 55.0},
    }

    caps = [100.0, 200.0, 1000.0]
    sas = make_sas(popps, ug_vols, ug_perfs, caps)
    routed = {"UG0": [0, 1, 2], "UG1": [0, 1, 2]}

    sla.SITE_COST = {"S0": 1.0, "S1": 10.0}

    alpha = 1.0
    ret = sla.solve_latency_plus_site_cost_with_capacity(sas, routed, obj="lat_cost", alpha=alpha, threads=1)
    assert ret["solved"]
    sol = np.array(ret["raw_solution"], dtype=float)

    available_paths, _ = simple_get_paths_by_ug(sas, routed)
    print_problem_and_solution(
        sas, routed, available_paths, sol, alpha=alpha, site_cost=sla.SITE_COST,
        title="Test 4: two UGs, two peers at same site, spill to 2nd peer"
    )

    vols_by_poppi = compute_vols_by_poppi_from_solution(available_paths, sol, sas.n_popps + 1)
    assert abs(vols_by_poppi[0] - 100.0) < 1e-4, vols_by_poppi
    assert abs(vols_by_poppi[1] - 50.0) < 1e-4, vols_by_poppi
    assert abs(vols_by_poppi[2] - 0.0) < 1e-6, vols_by_poppi

    assigned_by_ug = {ug: 0.0 for ug in sas.whole_deployment_ugs}
    for (ug, _poppi), v in zip(available_paths, sol):
        assigned_by_ug[ug] += float(v)
    assert abs(assigned_by_ug["UG0"] - 80.0) < 1e-6, assigned_by_ug
    assert abs(assigned_by_ug["UG1"] - 70.0) < 1e-6, assigned_by_ug


# Run tests 
if __name__ == "__main__":
    test_alpha_threshold_flip()
    test_capacity_spillover()
    test_cost_driven_steering_equal_latency()
    test_two_ugs_same_site_two_peers_capacity_spill()
    print("All tests passed.")
