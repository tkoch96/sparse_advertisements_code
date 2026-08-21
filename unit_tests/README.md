# tests/

Unit + timing tests for the worker class and LP-solving code.

## Quick start

```bash
# from the repo root, in your venv:
pip install pytest
pytest                                # run everything
pytest -v -s tests/test_timing_baseline.py   # see timing prints
pytest -m unit                        # fast tests only
pytest -m "not slow"                  # skip the benchmarks
pytest -m "not gurobi"                # skip anything that needs Gurobi
```

## What the scaffolding gives you

The hard part of testing this code is setup: building a deployment dict the
worker accepts, splitting it the way `Worker_Manager` does, and constructing
a worker synchronously (no Ray actor protocol). All of that lives in
`conftest.py`. Once those fixtures exist, individual tests stay tiny.

Key fixtures:

| Fixture | Scope | What it gives you |
|---|---|---|
| `tiny_deployment` | session | Full `really_friggin_small` deployment (2 pops, 20 peers, 75 ugs) |
| `worker_deployment` | session | The full deployment, the shape a worker actually receives |
| `init_kwa` | session | Matches `Worker_Manager.get_init_kwa()` |
| `gurobi_available` | session | `True` iff Gurobi can solve a trivial model on this box |
| `worker` | function | Fresh `_LocalPathDistributionComputer` (the non-Ray class wrapped by `@ray.remote`). Skips automatically if `gurobi_available` is `False`. |
| `worker_session` | session | Same as `worker` but reused. Use for read-only tests. |
| `tiny_advertisement` | function | Valid advertisement matrix for the tiny deployment |
| `lp_timer` | function | `lp_timer(fn, n=10, warmup=2)` -> `TimingStats` |
| `stopwatch` | function | Context manager for ad-hoc timing |

## Writing a new test

```python
import pytest

@pytest.mark.unit
@pytest.mark.gurobi
def test_my_thing(worker, tiny_advertisement, worker_deployment):
    subdep = dict(worker_deployment, generic_objective='avg_latency')
    out = worker._cmd_solve_lp([(0, tiny_advertisement, subdep, False)])
    assert out[0][1]['solved']
    assert out[0][1]['objective'] > 0
```

## Markers (see pytest.ini)

* `unit` – fast, in-process, no Ray.
* `integration` – exercises Ray actors via `Worker_Manager`.
* `gurobi` – needs a working Gurobi license. (HiGHS is the project
  default backend since 2026-08-20 — Gurobi had WLS scaling issues on
  fleets — so these tests are opt-in on boxes with a license.)
* `slow` – takes more than a few seconds; skipped by `-m "not slow"`.

## Notes

* Tests use a tmp dir for `LOG_DIR` (patched by `_isolate_environment` in
  `conftest.py`) so they never write into `./logs/`.
* RNG seeds are fixed at session start so deployments are reproducible.
* `_LocalPathDistributionComputer` is the same class body as the Ray actor;
  it's exposed so tests can instantiate a worker synchronously. The Ray
  actor (`Path_Distribution_Computer`) wraps it via `ray.remote(num_cpus=1)`.
* Ray itself is **not** required for the default test run -- only for tests
  marked `@pytest.mark.integration` (which currently don't exist; add them
  when you want to verify the full Ray fan-out path end to end).
