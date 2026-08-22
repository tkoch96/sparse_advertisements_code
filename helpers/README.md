# helpers/

Constants and general utilities. No SCULPTOR logic lives here — if something
needs the solver, it belongs in `core/`.

| file | role |
|---|---|
| `constants.py` | `NO_ROUTE_LATENCY` (30000, the no-route sentinel), `DEFAULT_PORT`, `RUN_DIR`, `CACHE_DIR`, dpsize→PoP-count mapping. The most depended-on file in the repo. |
| `helpers.py` | Generic utilities: logging, `save_fig`, memory snapshots, CDF helpers, deployment splitting. |
| `figpaths.py` | Routes dashboard figures to `figures/dashboards/<dashboard>/` by filename prefix. Rules are ordered most-specific-first (`grid_maxhard_v2_gurobi_` must beat `grid_maxhard_v2_` must beat `grid_maxhard_`); unmatched names land in `dashboards/misc/` rather than raising. |
| `paper_plotting_functions.py` | Plot styling primitives. |
| `timelog.py` | Startup instrumentation, activated by `SCULPTOR_STARTUP_TIMELOG=1`. |

`from helpers.helpers import *` reads awkwardly but is deliberate — the package
is `helpers`, the module inside it is `helpers.py`.

**`NO_ROUTE_LATENCY` is a sentinel, not a latency.** Using it as an axis bound
(`set_xlim([-NO_ROUTE_LATENCY/2, 0])`) made every real curve collapse onto x=0
and read as an empty panel. Fit axes to the data.
