"""gurobipy-subset facade with pluggable LP backends (Tom 2026-08-17;
MAINLINE since the solver_fork merge the same day).

Core LP callers (solve_lp_assignment, path_distribution_computer) do
`import gpshim as gp` instead of `import gurobipy as gp`. Backend
selected ONCE at import via SCULPTOR_LP_BACKEND:

    gurobi (default)  -- passthrough: the names below ARE gurobipy's own
                         objects; zero behavioral change vs mainline.
    highs             -- license-free HiGHS (highspy) implementation of the
                         exact gurobipy subset the repo uses (inventoried
                         2026-08-17; see experiments/solver_fork/README.md).

Scope (Tom-ratified): the highs backend supports every LINEAR objective
(avg_latency / per_site_cost / mlu / joint_priority via the soft-bounded
scalar). The quadratic objectives ('squaring' convex-QP,
'square_rooting' nonconvex) raise NotImplementedError at expression
build time on highs; gurobi passthrough keeps supporting them.

The highs Model is fully incremental (columns, chgCoeff, batch UB/Obj,
RHS mutation) so path_distribution_computer's persistent per-worker
model works identically: HiGHS hot-starts from the incumbent basis on
re-run after modification, playing the role of Gurobi Method=1 dual
simplex warm starts.

Numerical caveat: these LPs are degenerate; HiGHS returns equal-objective
but potentially different vertex solutions than Gurobi. Equivalence is
asserted at the objective level (Tom 2026-08-17), never solution-level.
"""
import os

import numpy as np

BACKEND = os.environ.get('SCULPTOR_LP_BACKEND', 'gurobi').lower()
if BACKEND not in ('gurobi', 'highs'):
    raise ValueError("SCULPTOR_LP_BACKEND must be 'gurobi' or 'highs', got %r" % BACKEND)

QUAD_MSG = ("quadratic expression not supported on the highs backend "
            "(objectives 'squaring'/'square_rooting' are gurobi-only; "
            "Tom-ratified scope 2026-08-17)")


if BACKEND == 'gurobi':
    import gurobipy as _grb
    Model = _grb.Model
    Column = _grb.Column
    GRB = _grb.GRB
    setParam = _grb.setParam

else:
    import highspy as _hp

    _KHINF = _hp.kHighsInf
    _GLOBAL_PARAMS = {}

    def setParam(name, value):
        # gurobipy's global setParam; both forked modules call
        # gp.setParam("OutputFlag", 0) at import. Applied at Model creation.
        _GLOBAL_PARAMS[name.lower()] = value

    class GRB:
        MINIMIZE = 1
        MAXIMIZE = -1
        OPTIMAL = 2
        INFEASIBLE = 3
        UNBOUNDED = 5
        TIME_LIMIT = 9
        INFINITY = 1e100
        EQUAL = '='
        LESS_EQUAL = '<'
        GREATER_EQUAL = '>'

    def _to_highs_bound(v):
        # Gurobi's INFINITY (1e100) exceeds HiGHS's infinite_bound (1e30)
        # threshold; clamp so HiGHS treats it as a proper infinity.
        if v >= 1e30:
            return _KHINF
        if v <= -1e30:
            return -_KHINF
        return float(v)

    class _Params:
        """Write-only Params facade. Maps the params the repo sets onto
        HiGHS options; silently accepts-and-ignores the rest (matching the
        inventory: OutputFlag/LogToConsole/LogFile/TimeLimit/Threads/
        Method/MIPGap are the complete set used)."""

        def __init__(self, model):
            object.__setattr__(self, '_model', model)

        def __setattr__(self, name, value):
            h = self._model._h
            key = name.lower()
            if key in ('outputflag', 'logtoconsole'):
                h.setOptionValue('output_flag', bool(value))
            elif key == 'logfile':
                h.setOptionValue('log_file', str(value))
                if value:
                    h.setOptionValue('output_flag', True)
            elif key == 'timelimit':
                h.setOptionValue('time_limit', float(value))
            elif key == 'threads':
                h.setOptionValue('threads', int(value))
            elif key == 'method':
                # Method=1 (dual simplex, warm-start-critical for the
                # persistent model). HiGHS: force simplex; strategy 1 = dual.
                if int(value) == 1:
                    h.setOptionValue('solver', 'simplex')
                    h.setOptionValue('simplex_strategy', 1)
            elif key == 'mipgap':
                h.setOptionValue('mip_rel_gap', float(value))
            # anything else: accept and ignore (none used in the repo)

    class Column:
        __slots__ = ('_terms',)

        def __init__(self):
            self._terms = []

        def addTerms(self, coeff, constr):
            self._terms.append((float(coeff), constr))

    class Var:
        __slots__ = ('_m', 'idx')
        __array_ufunc__ = None
        __array_priority__ = 10000

        def __init__(self, model, idx):
            self._m = model
            self.idx = idx

        @property
        def X(self):
            return float(self._m._sol[self.idx])

        @property
        def Obj(self):
            return self._m._col_cost[self.idx]

        @Obj.setter
        def Obj(self, v):
            self._m._col_cost[self.idx] = float(v)
            self._m._h.changeColCost(self.idx, float(v))

        @property
        def UB(self):
            return self._m._col_ub[self.idx]

        @UB.setter
        def UB(self, v):
            self._m._col_ub[self.idx] = float(v)
            self._m._h.changeColBounds(
                self.idx, _to_highs_bound(self._m._col_lb[self.idx]),
                _to_highs_bound(v))

        @property
        def LB(self):
            return self._m._col_lb[self.idx]

        @LB.setter
        def LB(self, v):
            self._m._col_lb[self.idx] = float(v)
            self._m._h.changeColBounds(
                self.idx, _to_highs_bound(v),
                _to_highs_bound(self._m._col_ub[self.idx]))

    class Constr:
        __slots__ = ('_m', 'row', '_sense')

        def __init__(self, model, row, sense):
            self._m = model
            self.row = row
            self._sense = sense

        @property
        def RHS(self):
            return self._m._row_rhs[self.row]

        @RHS.setter
        def RHS(self, v):
            v = float(v)
            self._m._row_rhs[self.row] = v
            if self._sense == GRB.EQUAL:
                lo, up = v, v
            elif self._sense == GRB.LESS_EQUAL:
                lo, up = -_KHINF, v
            else:
                lo, up = v, _KHINF
            self._m._h.changeRowBounds(self.row, lo, up)

    class _Lin:
        """Scalar linear expression: sum(coefs * cols) + const."""
        __slots__ = ('_m', 'cols', 'coefs', 'const')
        __array_ufunc__ = None
        __array_priority__ = 10000

        def __init__(self, model, cols, coefs, const=0.0):
            self._m = model
            self.cols = np.asarray(cols, dtype=np.int64)
            self.coefs = np.asarray(coefs, dtype=np.float64)
            self.const = float(const)

        def sum(self):
            return self

        def __add__(self, other):
            if isinstance(other, (int, float)):
                return _Lin(self._m, self.cols, self.coefs, self.const + other)
            if isinstance(other, _Lin):
                return _Lin(self._m,
                            np.concatenate([self.cols, other.cols]),
                            np.concatenate([self.coefs, other.coefs]),
                            self.const + other.const)
            raise NotImplementedError(QUAD_MSG)

        __radd__ = __add__

        def __sub__(self, other):
            if isinstance(other, (int, float)):
                return self + (-other)
            raise NotImplementedError(QUAD_MSG)

        def __mul__(self, other):
            if isinstance(other, (int, float)):
                return _Lin(self._m, self.cols, self.coefs * other,
                            self.const * other)
            raise NotImplementedError(QUAD_MSG)

        __rmul__ = __mul__

        def __eq__(self, rhs):
            return _ConstrSpec.from_scalar(self, GRB.EQUAL, rhs)

        def __le__(self, rhs):
            return _ConstrSpec.from_scalar(self, GRB.LESS_EQUAL, rhs)

        def __ge__(self, rhs):
            return _ConstrSpec.from_scalar(self, GRB.GREATER_EQUAL, rhs)

        __hash__ = None

    class _ScalarView:
        """MVar[int] — a single-variable handle usable in expressions."""
        __slots__ = ('_m', 'idx')
        __array_ufunc__ = None
        __array_priority__ = 10000

        def __init__(self, model, idx):
            self._m = model
            self.idx = int(idx)

        def _lin(self, coef=1.0):
            return _Lin(self._m, [self.idx], [coef])

        def sum(self):
            return self._lin()

        @property
        def X(self):
            return float(self._m._sol[self.idx])

        def __mul__(self, other):
            if isinstance(other, (int, float, np.floating)):
                return self._lin(float(other))
            raise NotImplementedError(QUAD_MSG)

        __rmul__ = __mul__

        def __add__(self, other):
            return self._lin() + other

        __radd__ = __add__

    class _AffineBlock:
        """MVar + ndarray offset; only ever consumed by sparse @ (...)."""
        __slots__ = ('mvar', 'offset')
        __array_ufunc__ = None
        __array_priority__ = 10000

        def __init__(self, mvar, offset):
            self.mvar = mvar
            self.offset = np.asarray(offset, dtype=np.float64)

        def __rmatmul__(self, A):
            ml = self.mvar.__rmatmul__(A)
            ml.const = ml.const + A @ self.offset
            return ml

    class MVar:
        """Contiguous (or index-selected) block of columns."""
        __slots__ = ('_m', 'indices')
        __array_ufunc__ = None
        __array_priority__ = 10000

        # NOTE deliberately no __len__: with a sequence protocol numpy
        # coerces MVar into a 1-d object array and scipy's csr @ MVar
        # tries dense multiply instead of returning NotImplemented (which
        # is what routes to our __rmatmul__). gurobipy sidesteps this via
        # its own numpy integration; we sidestep it by not being sized.
        def __init__(self, model, indices):
            self._m = model
            self.indices = np.asarray(indices, dtype=np.int64)

        @property
        def X(self):
            return self._m._sol[self.indices].copy()

        def sum(self):
            return _Lin(self._m, self.indices, np.ones(len(self.indices)))

        def __getitem__(self, key):
            if isinstance(key, (int, np.integer)):
                return _ScalarView(self._m, self.indices[key])
            return MVar(self._m, self.indices[key])

        def __add__(self, other):
            if isinstance(other, np.ndarray):
                return _AffineBlock(self, other)
            raise NotImplementedError(QUAD_MSG)

        __radd__ = __add__

        def __mul__(self, other):
            raise NotImplementedError(QUAD_MSG)

        __rmul__ = __mul__

        def __matmul__(self, w):
            # MVar @ ndarray -> scalar linear expr
            w = np.asarray(w, dtype=np.float64)
            if w.ndim != 1:
                raise NotImplementedError(QUAD_MSG)
            return _Lin(self._m, self.indices, w)

        def __rmatmul__(self, A):
            # ndarray/sparse @ MVar
            if isinstance(A, np.ndarray) and A.ndim == 1:
                return _Lin(self._m, self.indices, A)
            from scipy import sparse as _sp
            if isinstance(A, np.ndarray):
                A = _sp.csr_matrix(A)
            elif not _sp.issparse(A):
                raise TypeError('unsupported lhs operand for @: %r' % type(A))
            return _MatLin(self._m, A.tocsr(), self.indices,
                           np.zeros(A.shape[0]))

    class _MatLin:
        """Vector linear expression: A @ x[indices] + const (m rows)."""
        __slots__ = ('_m', 'A', 'indices', 'const')
        __array_ufunc__ = None
        __array_priority__ = 10000

        def __init__(self, model, A, indices, const):
            self._m = model
            self.A = A
            self.indices = np.asarray(indices, dtype=np.int64)
            self.const = np.asarray(const, dtype=np.float64)

        def __sub__(self, other):
            return _MatLin(self._m, self.A, self.indices,
                           self.const - np.asarray(other, dtype=np.float64))

        def __add__(self, other):
            return _MatLin(self._m, self.A, self.indices,
                           self.const + np.asarray(other, dtype=np.float64))

        def __matmul__(self, w):
            # (A x + c) @ w -> scalar linear expr: (w^T A) x + w.c
            w = np.asarray(w, dtype=np.float64)
            coefs = self.A.T @ w
            return _Lin(self._m, self.indices, coefs, float(w @ self.const))

        def __eq__(self, rhs):
            return _ConstrSpec.from_matrix(self, GRB.EQUAL, rhs)

        def __le__(self, rhs):
            return _ConstrSpec.from_matrix(self, GRB.LESS_EQUAL, rhs)

        def __ge__(self, rhs):
            return _ConstrSpec.from_matrix(self, GRB.GREATER_EQUAL, rhs)

        __hash__ = None

    class _ConstrSpec:
        """Rows ready to be added by Model.addConstr."""
        __slots__ = ('model', 'A', 'indices', 'lower', 'upper', 'sense', 'rhs')

        @classmethod
        def from_matrix(cls, matlin, sense, rhs):
            rhs = np.broadcast_to(
                np.asarray(rhs, dtype=np.float64),
                (matlin.A.shape[0],)).copy()
            s = cls()
            s.model = matlin._m
            s.A = matlin.A.tocsr()
            s.indices = matlin.indices
            s.sense = sense
            s.rhs = rhs - matlin.const
            if sense == GRB.EQUAL:
                s.lower, s.upper = s.rhs, s.rhs
            elif sense == GRB.LESS_EQUAL:
                s.lower = np.full(len(s.rhs), -_KHINF)
                s.upper = s.rhs
            else:
                s.lower = s.rhs
                s.upper = np.full(len(s.rhs), _KHINF)
            return s

        @classmethod
        def from_scalar(cls, lin, sense, rhs):
            from scipy import sparse as _sp
            m = _sp.csr_matrix(
                (lin.coefs, (np.zeros(len(lin.cols), dtype=np.int64),
                             np.arange(len(lin.cols)))),
                shape=(1, len(lin.cols)))
            ml = _MatLin(lin._m, m, lin.cols, np.array([lin.const]))
            return cls.from_matrix(ml, sense, np.array([float(rhs)]))

    class Model:
        def __init__(self, name=''):
            self._h = _hp.Highs()
            self._name = name
            # global gp.setParam defaults (both fork modules set OutputFlag=0)
            self._h.setOptionValue(
                'output_flag',
                bool(_GLOBAL_PARAMS.get('outputflag', 1)))
            self._params = _Params(self)
            self._ncols = 0
            self._nrows = 0
            self._col_lb = _GrowArray()
            self._col_ub = _GrowArray()
            self._col_cost = _GrowArray()
            self._row_rhs = _GrowArray()
            self._obj_cols = None      # cols referenced by last setObjective
            self._sol = None
            self.status = None

        # -- Params ------------------------------------------------------
        @property
        def Params(self):
            return self._params

        # -- construction ------------------------------------------------
        def addMVar(self, n, name=None, lb=0.0):
            n = int(n)
            base = self._ncols
            lbv = np.full(n, float(lb))
            self._h.addCols(n, np.zeros(n), lbv, np.full(n, _KHINF),
                            0, np.zeros(n + 1, dtype=np.int32),
                            np.array([], dtype=np.int32), np.array([]))
            self._col_lb.extend(lbv)
            self._col_ub.extend(np.full(n, GRB.INFINITY))
            self._col_cost.extend(np.zeros(n))
            self._ncols += n
            return MVar(self, np.arange(base, base + n))

        def addVar(self, lb=0.0, ub=GRB.INFINITY, obj=0.0, name=None,
                   column=None):
            idx = self._ncols
            if column is not None and column._terms:
                rows = np.array([c.row for _, c in column._terms],
                                dtype=np.int32)
                vals = np.array([v for v, _ in column._terms])
            else:
                rows = np.array([], dtype=np.int32)
                vals = np.array([])
            self._h.addCol(float(obj), _to_highs_bound(lb),
                           _to_highs_bound(ub), len(rows), rows, vals)
            self._col_lb.append(float(lb))
            self._col_ub.append(float(ub))
            self._col_cost.append(float(obj))
            self._ncols += 1
            return Var(self, idx)

        def addLConstr(self, lhs, sense, rhs, name=None):
            # Only the placeholder form addLConstr(0.0, sense, float) is
            # used in the repo (persistent model rows filled by columns).
            if not isinstance(lhs, (int, float)) or float(lhs) != 0.0:
                raise NotImplementedError(
                    'gpshim.addLConstr supports only the 0.0-placeholder form')
            rhs = float(rhs)
            if sense == GRB.EQUAL:
                lo, up = rhs, rhs
            elif sense == GRB.LESS_EQUAL:
                lo, up = -_KHINF, rhs
            elif sense == GRB.GREATER_EQUAL:
                lo, up = rhs, _KHINF
            else:
                raise ValueError('bad sense %r' % sense)
            row = self._nrows
            self._h.addRow(lo, up, 0, np.array([], dtype=np.int32),
                           np.array([]))
            self._row_rhs.append(rhs)
            self._nrows += 1
            return Constr(self, row, sense)

        def addConstr(self, spec, name=None):
            if not isinstance(spec, _ConstrSpec):
                raise NotImplementedError(QUAD_MSG)
            A = spec.A
            nrows = A.shape[0]
            # remap local csr column indices -> model column indices
            col_idx = spec.indices[A.indices].astype(np.int32)
            self._h.addRows(nrows, spec.lower, spec.upper,
                            len(A.data), A.indptr.astype(np.int32),
                            col_idx, A.data)
            self._row_rhs.extend(spec.rhs)
            base = self._nrows
            self._nrows += nrows
            return [Constr(self, base + i, spec.sense) for i in range(nrows)]

        def setObjective(self, expr, sense=None):
            if isinstance(expr, _ScalarView):
                expr = expr.sum()
            if not isinstance(expr, _Lin):
                raise NotImplementedError(QUAD_MSG)
            if sense is not None and sense != GRB.MINIMIZE:
                raise NotImplementedError('only MINIMIZE is used in the repo')
            # zero out previous objective, then apply the new one
            if self._obj_cols is not None and len(self._obj_cols):
                self._h.changeColsCost(
                    len(self._obj_cols), self._obj_cols.astype(np.int32),
                    np.zeros(len(self._obj_cols)))
            cols, coefs = _accumulate(expr.cols, expr.coefs)
            self._h.changeColsCost(len(cols), cols.astype(np.int32), coefs)
            self._col_cost.arr[:self._ncols][cols] = coefs
            self._obj_cols = cols
            self._h.changeObjectiveOffset(expr.const)

        # -- incremental mutation (persistent model) ---------------------
        def chgCoeff(self, constr, var, value):
            self._h.changeCoeff(constr.row, var.idx, float(value))

        def setAttr(self, attrname, objs, values):
            idx = np.array([o.idx for o in objs], dtype=np.int32)
            vals = np.asarray(values, dtype=np.float64)
            if attrname == 'UB':
                self._col_ub.arr[:self._ncols][idx] = vals
                self._h.changeColsBounds(
                    len(idx), idx,
                    self._col_lb.arr[:self._ncols][idx],
                    np.array([_to_highs_bound(v) for v in vals]))
            elif attrname == 'Obj':
                self._col_cost.arr[:self._ncols][idx] = vals
                self._h.changeColsCost(len(idx), idx, vals)
            else:
                raise NotImplementedError('setAttr %r' % attrname)

        def getAttr(self, attrname, objs):
            if attrname != 'X':
                raise NotImplementedError('getAttr %r' % attrname)
            sol = self._sol
            return [float(sol[o.idx]) for o in objs]

        # -- solve -------------------------------------------------------
        def optimize(self):
            self._h.run()
            st = self._h.getModelStatus()
            if st == _hp.HighsModelStatus.kOptimal:
                self.status = 2
            elif st == _hp.HighsModelStatus.kInfeasible:
                self.status = 3
            elif st == _hp.HighsModelStatus.kUnbounded:
                self.status = 5
            elif st in (_hp.HighsModelStatus.kTimeLimit,
                        _hp.HighsModelStatus.kIterationLimit):
                self.status = 9
            else:
                self.status = 13
            if self.status == 2:
                self._sol = np.asarray(self._h.getSolution().col_value)
                if os.environ.get('SCULPTOR_GPSHIM_AUDIT') == '1':
                    self._audit()
            return self.status

        def _audit(self):
            """SCULPTOR_GPSHIM_AUDIT=1: re-solve the EXACT LP HiGHS holds
            (pulled back via getLp, so facade state bugs — stale bounds,
            missed cost zeroing, bad chgCoeff — are inside the audited
            object) with scipy's independent HiGHS wrapper and compare
            objectives. Loud [gpshim-audit] line + hard raise on mismatch."""
            from scipy import sparse as _sp
            from scipy.optimize import linprog as _linprog
            lp = self._h.getLp()
            n, m = lp.num_col_, lp.num_row_
            A = _sp.csc_matrix(
                (np.asarray(lp.a_matrix_.value_),
                 np.asarray(lp.a_matrix_.index_),
                 np.asarray(lp.a_matrix_.start_)),
                shape=(m, n)).tocsr()
            c = np.asarray(lp.col_cost_)
            cl = np.asarray(lp.col_lower_)
            cu = np.asarray(lp.col_upper_)
            rl = np.asarray(lp.row_lower_)
            ru = np.asarray(lp.row_upper_)
            eq = rl == ru
            rows_ub, rhs_ub = [], []
            if (~eq).any():
                fin_u = ~eq & (ru < _KHINF)
                fin_l = ~eq & (rl > -_KHINF)
                if fin_u.any():
                    rows_ub.append(A[fin_u])
                    rhs_ub.append(ru[fin_u])
                if fin_l.any():
                    rows_ub.append(-A[fin_l])
                    rhs_ub.append(-rl[fin_l])
            r = _linprog(
                c,
                A_ub=_sp.vstack(rows_ub) if rows_ub else None,
                b_ub=np.concatenate(rhs_ub) if rhs_ub else None,
                A_eq=A[eq] if eq.any() else None,
                b_eq=rl[eq] if eq.any() else None,
                bounds=np.stack([np.where(cl <= -_KHINF, -np.inf, cl),
                                 np.where(cu >= _KHINF, np.inf, cu)], 1),
                method='highs')
            mine = self._h.getObjectiveValue()
            ref = (r.fun + lp.offset_) if r.status == 0 else None
            tol = 1e-5 * max(1.0, abs(mine))
            if ref is None or abs(mine - ref) > tol:
                msg = ('[gpshim-audit] MISMATCH model={!r} facade={!r} '
                       'scipy={!r} (scipy status {})'.format(
                           self._name, mine, ref, r.status))
                print(msg, flush=True)
                raise AssertionError(msg)

        @property
        def objVal(self):
            return float(self._h.getObjectiveValue())

        ObjVal = objVal

        def write(self, path):
            self._h.writeModel(path)

    class _GrowArray:
        """Amortized-growth float array supporting append/extend + slicing
        (var_pool hits 100k-1M cols in production; python lists are fine
        but numpy views keep setAttr batches allocation-free)."""
        __slots__ = ('arr', 'n')

        def __init__(self):
            self.arr = np.zeros(1024)
            self.n = 0

        def _ensure(self, extra):
            need = self.n + extra
            if need > len(self.arr):
                new = np.zeros(max(need, 2 * len(self.arr)))
                new[:self.n] = self.arr[:self.n]
                self.arr = new

        def append(self, v):
            self._ensure(1)
            self.arr[self.n] = v
            self.n += 1

        def extend(self, vals):
            vals = np.asarray(vals, dtype=np.float64)
            self._ensure(len(vals))
            self.arr[self.n:self.n + len(vals)] = vals
            self.n += len(vals)

        def __getitem__(self, i):
            return self.arr[:self.n][i]

        def __setitem__(self, i, v):
            self.arr[:self.n][i] = v

    def _accumulate(cols, coefs):
        """Sum duplicate column entries (LinExpr concatenation can repeat)."""
        cols = np.asarray(cols, dtype=np.int64)
        uniq, inv = np.unique(cols, return_inverse=True)
        out = np.zeros(len(uniq))
        np.add.at(out, inv, coefs)
        return uniq, out
