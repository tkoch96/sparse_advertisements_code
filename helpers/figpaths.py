"""Where dashboard figures go.

Before 2026-08-21 every dashboard dumped its PNGs straight into `figures/`,
so ~53 unrelated plots from a dozen campaigns sat in one flat directory. They
now live in `figures/dashboards/<dashboard>/`, one subdirectory per dashboard.

Routing is by filename prefix rather than by caller, because the producing
scripts are shared: `plot_ladder_direct.py` renders the v3, v3lbc, a10, a10x10
and HiGHS ladders, and only the output name distinguishes them. Rules are
ordered most-specific-first — `grid_maxhard_v2_gurobi_` must beat
`grid_maxhard_v2_`, which must beat `grid_maxhard_`.

Anything unmatched lands in `dashboards/misc/` rather than raising, so a new
plot never breaks a refresh; add a rule when you notice it there.
"""
import os

# (filename prefix, dashboard subdirectory) -- ORDER MATTERS, longest first.
_RULES = [
    ('grid_objdim_', 'ablation_scout'),
    ('ablation_scout', 'ablation_scout'),
    ('conv_', 'ablation_scout'),
    ('policy_ladder_v3lbc_',        'ablation_v3lbc'),
    ('policy_ladder_v3_',           'ablation_v3'),
    ('policy_ladder_a10x10_',       'a10x10'),
    ('policy_ladder_a10_',          'ladder_a10'),
    ('policy_ladder_highs_',        'ladder_highs'),
    ('grid_maxhard_v2_gurobi_',     'grid_maxhard_v2_gurobi'),
    ('grid_maxhard_v2_',            'grid_maxhard_v2'),
    ('grid_maxhard_r4_',            'grid_maxhard'),
    ('grid_maxhard_',               'grid_maxhard'),
    ('grid_georand_r4_',            'grid_georand'),
    ('grid_georand_',               'grid_georand'),
    ('grid_v5scout_',               'v5scout'),
    ('v5scout_',                    'v5scout'),
    ('hardobj_highs_',              'hardobj_highs'),
    ('hardobj_v4_',                 'hardobj_v4'),
    ('hardB3v2_',                   'hardobj_v2'),
    ('eods25_',                     'eods25'),
    ('eods32_',                     'eods32'),
    ('adagrad_smoke_',              'adagrad_smoke'),
    ('profiler_',                   'profiler'),
    ('depcache_',                   'depcache'),
    ('stepalpha',                   'stepalpha'),
]

REL_ROOT = os.path.join('figures', 'dashboards')


def dashboard_for(name):
    """Dashboard subdirectory owning a figure basename."""
    base = os.path.basename(name)
    for prefix, dash in _RULES:
        if base.startswith(prefix):
            return dash
    return 'misc'


def fig_relpath(name):
    """Repo-relative path for a figure basename, e.g.
    'figures/dashboards/hardobj_v4/hardobj_v4_prio.png'."""
    return os.path.join(REL_ROOT, dashboard_for(name), os.path.basename(name))


def fig_path(name, repo=None, mkdir=True):
    """Absolute path for a figure basename, creating its directory."""
    if repo is None:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(repo, fig_relpath(name))
    if mkdir:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
