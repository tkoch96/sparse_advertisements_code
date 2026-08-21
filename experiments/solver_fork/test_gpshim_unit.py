"""Unit battery for gpshim (run with SCULPTOR_LP_BACKEND=highs or gurobi).

Every test cross-checks against an independently built scipy.linprog
(also HiGHS underneath, but through a completely different code path —
matrix assembly is independent, so facade lowering bugs can't cancel).

    SCULPTOR_LP_BACKEND=highs python -m experiments.solver_fork.test_gpshim_unit
"""
import os
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import core.gpshim as gp

RNG = np.random.default_rng(0)
TOL = 1e-6
CHECKS = []


def check(name, got, want, tol=TOL):
    ok = abs(got - want) <= tol * max(1.0, abs(want))
    CHECKS.append((name, ok, got, want))
    print('  [{}] {}  got={:.9g} want={:.9g}'.format(
        'ok' if ok else 'FAIL', name, got, want))
    if not ok:
        raise AssertionError(name)


def rand_instance(n_ug=6, n_link=4, seed=1):
    """Random feasible transportation-style LP mirroring the repo shape:
    per-UG volume conservation (eq) + per-link capacity (ub)."""
    rng = np.random.default_rng(seed)
    vols = rng.uniform(1, 5, n_ug)
    # each ug can use 2-3 links
    rows_eq, cols, data_eq = [], [], []
    lat = []
    link_of_path = []
    p = 0
    for u in range(n_ug):
        for l in rng.choice(n_link, rng.integers(2, n_link), replace=False):
            rows_eq.append(u)
            cols.append(p)
            data_eq.append(1.0)
            lat.append(rng.uniform(10, 100))
            link_of_path.append(l)
            p += 1
    n_paths = p
    A_eq = csr_matrix((data_eq, (rows_eq, cols)), shape=(n_ug, n_paths))
    A_ub = csr_matrix((np.ones(n_paths),
                       (link_of_path, np.arange(n_paths))),
                      shape=(n_link, n_paths))
    caps = np.full(n_link, vols.sum())  # generous: always feasible
    return A_eq, vols, A_ub, caps, np.array(lat), n_paths, n_link


def ref_solve(c, A_eq, b_eq, A_ub, b_ub):
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=(0, None), method='highs')
    assert r.status == 0, r.message
    return r.fun, r.x


def test_oneshot_basic():
    print('one-shot: csr@x ==/<= + ndarray@x objective')
    A_eq, b_eq, A_ub, caps, lat, n, _ = rand_instance(seed=2)
    m = gp.Model()
    m.Params.LogToConsole = 0
    m.Params.Threads = 1
    x = m.addMVar(n, name='volume_each_path', lb=0)
    m.addConstr(A_eq @ x == b_eq)
    m.addConstr(A_ub @ x <= caps)
    m.setObjective(lat @ x)
    m.optimize()
    assert m.status == 2 == gp.GRB.OPTIMAL
    want, _ = ref_solve(lat, A_eq, b_eq, A_ub, caps)
    check('basic objVal', m.objVal, want)
    check('basic x-dot', float(lat @ x.X), want)


def test_oneshot_mlu_slice():
    print('one-shot: MLU pattern — addMVar(1+n), slicing, x@w objective')
    A_eq, b_eq, A_ub, caps, lat, n, n_link = rand_instance(seed=3)
    # x = [Y, paths]; A_util = [-caps | A_ub]; conservation on x[1:]
    m = gp.Model()
    m.Params.LogToConsole = 0
    x = m.addMVar(1 + n, name='volume_each_path', lb=0)
    A_util = csr_matrix(np.hstack([-caps.reshape(-1, 1), A_ub.toarray()]))
    vol_A = csr_matrix(np.hstack([np.zeros((A_eq.shape[0], 1)), A_eq.toarray()]))
    m.addConstr(A_util @ x <= np.zeros(n_link))
    m.addConstr(vol_A @ x == b_eq)
    w = np.concatenate([[1000.0], lat])
    m.setObjective(x @ w)
    m.optimize()
    assert m.status == 2
    c_ref = w
    A_ub_ref = A_util.toarray()
    A_eq_ref = vol_A.toarray()
    want, _ = ref_solve(c_ref, A_eq_ref, b_eq, A_ub_ref, np.zeros(n_link))
    check('mlu objVal', m.objVal, want)
    check('mlu slice .X', float(lat @ x.X[1:] + 1000.0 * x.X[0]), want)


def test_oneshot_scalar_sum_obj():
    print('one-shot: setObjective(z[k].sum(), MINIMIZE) [solve_min_mlu shape]')
    A_eq, b_eq, A_ub, caps, lat, n, n_link = rand_instance(seed=4)
    tight = caps * 0.4  # make MLU bind
    m = gp.Model()
    m.Params.LogToConsole = 0
    z = m.addMVar(n + 1, lb=0)
    A_eq2 = csr_matrix(np.hstack([A_eq.toarray(), np.zeros((A_eq.shape[0], 1))]))
    A_ub2 = csr_matrix(np.hstack([A_ub.toarray(), -tight.reshape(-1, 1)]))
    m.addConstr(A_eq2 @ z == b_eq)
    m.addConstr(A_ub2 @ z <= np.zeros(n_link))
    m.setObjective(z[n].sum(), gp.GRB.MINIMIZE)
    m.optimize()
    assert m.status == 2
    c = np.zeros(n + 1)
    c[n] = 1.0
    want, _ = ref_solve(c, A_eq2.toarray(), b_eq, A_ub2.toarray(),
                        np.zeros(n_link))
    check('min-mlu objVal', m.objVal, want)
    check('min-mlu z.X[n]', float(z.X[n]), want)


def test_oneshot_affine_matexpr():
    print('one-shot: csr @ (b + x) <= caps and MatLin @ w objective '
          '[bulk-download shape]')
    A_eq, b_eq, A_ub, caps, lat, n, n_link = rand_instance(seed=5)
    x_frozen = RNG.uniform(0, 0.4, n)  # pre-existing traffic (ndarray)
    m = gp.Model()
    m.Params.LogToConsole = 0
    b = m.addMVar(n, name='bulk', lb=0)
    m.addConstr(A_eq @ b == b_eq)
    m.addConstr(A_ub @ (b + x_frozen) <= caps)
    over = A_ub @ (b + x_frozen) - 0.5 * caps
    sig = RNG.uniform(0.5, 2.0, n_link)
    m.setObjective(over @ sig)
    m.optimize()
    assert m.status == 2
    # reference: minimize sig^T(A_ub(b+x) - .5caps) = (A_ub^T sig)@b + const
    c = A_ub.T @ sig
    const = float(sig @ (A_ub @ x_frozen - 0.5 * caps))
    want_core, _ = ref_solve(c, A_eq.toarray(), b_eq, A_ub.toarray(),
                             caps - A_ub @ x_frozen)
    check('affine objVal', m.objVal, want_core + const)


def test_persistent_pattern():
    print('persistent: placeholder rows + Column addVar + UB/Obj batches + '
          'chgCoeff MLU toggle + RHS mutation, multi-resolve')
    A_eq, b_eq, A_ub, caps, lat, n, n_link = rand_instance(seed=6)
    ug_of_path = A_eq.tocoo().row
    link_of_path = A_ub.tocoo().row
    m = gp.Model('Worker_test_Persistent')
    m.Params.LogToConsole = 0
    m.Params.Method = 1
    m.Params.Threads = 1
    mlu_dummy = m.addVar(lb=0.0, obj=0.0, name='mlu_Y')
    vol_constrs = {u: m.addLConstr(0.0, gp.GRB.EQUAL, float(b_eq[u]),
                                   name='vol_%d' % u)
                   for u in range(A_eq.shape[0])}
    cap_constrs = {l: m.addLConstr(0.0, gp.GRB.LESS_EQUAL, float(caps[l]),
                                   name='cap_%d' % l)
                   for l in range(n_link)}
    var_pool = {}

    def solve_unified(active, coeffs, using_mlu, caps_now):
        all_vars = list(var_pool.values())
        if all_vars:
            m.setAttr('UB', all_vars, [0.0] * len(all_vars))
        if using_mlu:
            mlu_dummy.Obj = 1.0 / 10.0
            mlu_dummy.UB = gp.GRB.INFINITY
            for l, constr in cap_constrs.items():
                m.chgCoeff(constr, mlu_dummy, -1.0 * caps_now[l])
                constr.RHS = 0.0
        else:
            mlu_dummy.Obj = 0.0
            mlu_dummy.UB = 0.0
            for l, constr in cap_constrs.items():
                m.chgCoeff(constr, mlu_dummy, 0.0)
                constr.RHS = caps_now[l]
        for p, c in zip(active, coeffs):
            if p not in var_pool:
                col = gp.Column()
                col.addTerms(1.0, vol_constrs[ug_of_path[p]])
                col.addTerms(1.0, cap_constrs[link_of_path[p]])
                var_pool[p] = m.addVar(lb=0.0, obj=c, column=col)
        av = [var_pool[p] for p in active]
        m.setAttr('UB', av, [gp.GRB.INFINITY] * len(av))
        m.setAttr('Obj', av, coeffs)
        m.optimize()
        return m.status, (m.getAttr('X', av) if m.status == 2 else None)

    def ref(active, coeffs, using_mlu, caps_now):
        # build from scratch: cols = active paths (+Y if mlu)
        na = len(active)
        Ae = np.zeros((A_eq.shape[0], na + 1))
        Au = np.zeros((n_link, na + 1))
        for j, p in enumerate(active):
            Ae[ug_of_path[p], j] = 1.0
            Au[link_of_path[p], j] = 1.0
        c = np.concatenate([coeffs, [0.0]])
        if using_mlu:
            Au[:, na] = -caps_now
            b_ub = np.zeros(n_link)
            c[na] = 1.0 / 10.0
        else:
            b_ub = caps_now.copy()
        want, xr = ref_solve(c, Ae, b_eq, Au, b_ub)
        return want

    # solve 1: subset of paths, standard mode
    active1 = list(range(0, n, 2))
    st, xs = solve_unified(active1, lat[active1], False, caps)
    assert st == 2
    check('persist s1', m.objVal, ref(active1, lat[active1], False, caps))
    # solve 2: all paths, standard
    active2 = list(range(n))
    st, xs = solve_unified(active2, lat[active2], False, caps)
    assert st == 2
    check('persist s2 (grown pool)', m.objVal,
          ref(active2, lat[active2], False, caps))
    # solve 3: MLU mode with tight caps
    tight = caps * 0.3
    st, xs = solve_unified(active2, lat[active2], True, tight)
    assert st == 2
    check('persist s3 (mlu)', m.objVal, ref(active2, lat[active2], True, tight))
    # solve 4: back to standard, headroom-scaled caps (set_training_mode shape)
    scaled = caps * 0.9
    st, xs = solve_unified(active1, lat[active1], False, scaled)
    assert st == 2
    check('persist s4 (back+rhs)', m.objVal,
          ref(active1, lat[active1], False, scaled))
    # batch X vs per-var X
    xa = m.getAttr('X', [var_pool[p] for p in active1])
    xb = [var_pool[p].X for p in active1]
    check('persist batchX==varX', float(np.abs(np.array(xa) - np.array(xb)).max()), 0.0, tol=1e-12)


def test_quadratic_guard():
    if gp.BACKEND != 'highs':
        print('quadratic guard: skipped (gurobi passthrough supports quads)')
        return
    print('quadratic guard: MVar*MVar raises NotImplementedError')
    m = gp.Model()
    x = m.addMVar(3, lb=0)
    try:
        _ = x * x
        raise AssertionError('quadratic expression did not raise')
    except NotImplementedError:
        print('  [ok] raises as scoped')


def main():
    print('gpshim unit battery, backend =', gp.BACKEND)
    test_oneshot_basic()
    test_oneshot_mlu_slice()
    test_oneshot_scalar_sum_obj()
    test_oneshot_affine_matexpr()
    test_persistent_pattern()
    test_quadratic_guard()
    n_ok = sum(1 for _, ok, _, _ in CHECKS if ok)
    print('PASS {}/{} checks, backend={}'.format(n_ok, len(CHECKS), gp.BACKEND))


if __name__ == '__main__':
    main()
