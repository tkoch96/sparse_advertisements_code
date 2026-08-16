"""Canonical world definitions — the SINGLE source of truth (Tom,
2026-08-14: everything must resolve the world from one file so nothing
needs remembering/updating in several places).

A "world" is exactly the set of deployment-generation env knobs
(documented in DIMENSIONS.md section 3). Training chains and eval
scripts MUST build deployments under the same world or the numbers are
garbage — the tell is arm latencies BELOW the per-seed opp reference
(caught 2026-08-14: v2 evals silently ran in stock, ~12 ms vs georand's
15-28 ms opp).

Consumers: dashboard refresh steps declare `'world': '<name>'`
(refresh.py merges these knobs into the step env); score_hardb3 calls
apply(); manual eval invocations should use
`python -c "from experiments.model_error import worlds; ..."` or copy
env(name) rather than retyping knobs.
"""

WORLDS = {
    'stock': {},
    'georand': {'SCULPTOR_LAT_MODEL': 'geo',
                'SCULPTOR_PREF_MODEL': 'random'},
    'maxhard': {'SCULPTOR_LAT_MODEL': 'geo',
                'SCULPTOR_PREF_MODEL': 'random',
                'SCULPTOR_GEO_NOISE': '2',
                'SCULPTOR_VOL_SPREAD': '6',
                'SCULPTOR_SCALE_FACTOR': '1.0'},
}


def env(world):
    """The env-knob dict for a canonical world name."""
    return dict(WORLDS[world])


def apply(world):
    """Set a world's knobs in os.environ (child-process pattern)."""
    import os
    os.environ.update(env(world))
