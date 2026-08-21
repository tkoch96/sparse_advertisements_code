# integration_tests/

End-to-end checks that run the real pipeline as a subprocess and judge it on
what it produced. Unit tests live in `unit_tests/` and run under pytest; these
run standalone.

```bash
python integration_tests/verify_e2e_eval.py           # small@100, actual-5@50
python integration_tests/verify_e2e_eval.py --quick   # both at 3 iters (~1 min for small)
python integration_tests/verify_e2e_eval.py --only small
```

## `verify_e2e_eval.py`

Runs `evaluations/eval_latency_failure.py` at two deployment sizes and asserts
the evaluation actually happened.

**It deliberately does not trust the exit code.** `evaluate_all_metrics` wraps
its strategy loop in a bare `except:` that prints a traceback and continues, so
a run whose solver died in the first second still reaches the plotting section,
still returns a metrics dict, and still exits 0. On 2026-08-21 that cost ~11h
of actual-32 training: a broken hot-start exited 0 after six seconds, the queue
read rc=0 as success, and its harvest step deleted the run directory.

So each case is judged on the metrics pickle it produced — all six strategies
solved, `failed_strategies` empty, per-UG latency vectors populated and finite —
plus a log scan for the failure markers the bare except would have hidden.

Two traps the harness itself had to be fixed for, both worth knowing:

- **Stale artifacts.** The workspace cache is symlinked to the repo, so a
  previous run's `popp_failure_latency_comparison_<dpsize>.pkl` would satisfy
  every content check. Worse, `evaluate_all_metrics` *resume-skips* a
  simulation whose `n_advs` is already populated — so the run became a 9-second
  no-op that passed. The workspace now symlinks cache inputs but excludes the
  result pickles, and the pickle's mtime must post-date the run start.
- **`RAY_TMPDIR` length.** Ray sockets are AF_UNIX, capped at 103 bytes. A
  `tempfile.mkdtemp()` root under `/var/folders/...` blows the cap on macOS, so
  the harness puts Ray's tmpdir under `/tmp/rt_*`.

### Known non-fatal condition

`assess_volume_multipliers` (`evaluations/wrapper_eval.py`) raises `ValueError`
whenever any UG hits `NO_ROUTE_LATENCY` at an inflated volume. On `small` this
fires ~6 times per run, even at the lowest multiplier (X≈10.7), which should
not congest. It is swallowed by the bare except. The harness reports it as a
WARN rather than failing, and fails on any traceback signature *not* on that
list — so this stays visible without blocking, and a new failure mode is caught.

### Ray, and the `--port` red herring

Every case spawns Ray actors — Ray is not the default backend, it is the only
one (`core/worker_comms.py` is a re-export; the ZMQ path was deleted in the
mid-2026 migration). Cases are isolated by `RAY_ADDRESS=local`, which forces a
private cluster each time. Without it the second case attaches to the first
case's still-draining cluster and dies ~2s in, which reads like a code bug.

`--port` is vestigial. `eval_latency_failure.py` requires it, but nothing binds
it under Ray — `path_distribution_computer.py (actor layer)` literally sets
`self.port = 0  # unused under Ray`. Its only effect is avoiding the 5-second
`NO PORT SPECIFIED` sleep in `core/optimal_adv_wrapper.py:936`. Any value works;
distinct values per case isolate nothing.

### Timing

`small` is ~1 min at 3 iters. `actual-5` is much slower on a laptop — it parses
the 4.2 GB measurement CSV on every run and did not finish 3 iterations in 15
minutes. Use `--only small` for a quick gate; run the full default where you
have cores.
