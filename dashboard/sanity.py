"""Shared dashboard sanity guards (Tom 2026-08-16: "put an assertion in
all of your plotting figures ... that no objective value should ever be
better than one per peering").

The invariant: one-per-peering advertises every popp, so its feasible
routing set is a superset of every sparse advertisement's — for any
CAPABILITY metric (the best value achievable given the option set), an
arm can never beat opp. A violation therefore means an eval bug OR an
assignment-derived metric (a metric computed on the avg-latency-optimal
split rather than optimized directly — the pure-MLU failure class).
Either way it must fail LOUD, not render as a plausible line.

Documented exemption: the popp/site-failure composite — sparse arms
LEGITIMATELY beat opp there (opp advertises every popp, so every failure
congests someone; standing finding 2026-08-14).
"""
import functools

EXEMPT_OBJECTIVES = ('popfail',)


def assert_not_better_than_opp(rows, context, tol=1e-9):
    """rows: iterable of (cell_id, objective_dirname, obj_val, opp_val)
    in the BENEFIT convention (higher = better). Raises AssertionError
    listing every cell that beats its own opp reference."""
    viol = []
    for cell_id, obj, val, opp in rows:
        if obj in EXEMPT_OBJECTIVES or val is None or opp is None:
            continue
        if val > opp + tol:
            viol.append('{} ({:.4f} > opp {:.4f})'.format(cell_id, val, opp))
    assert not viol, (
        '[sanity] {}: {} cell(s) score BETTER than one-per-peering — '
        'impossible for capability metrics; eval bug or assignment-'
        'derived metric. Offenders: {}'.format(
            context, len(viol), '; '.join(viol[:10])
            + (' ...' if len(viol) > 10 else '')))


def guarded_figure(rows_fn, context):
    """Decorator form: wrap a figure-writing function so the invariant is
    checked (via rows_fn() -> rows as above) BEFORE the figure renders."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            assert_not_better_than_opp(rows_fn(), context)
            return fn(*a, **kw)
        return wrapper
    return deco
