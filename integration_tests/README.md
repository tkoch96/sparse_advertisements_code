# integration_tests/

End-to-end checks that run the real pipeline as a subprocess and judge it on
what it produced. Unit tests live in `unit_tests/` and run under pytest; these
run standalone.

```bash
~/Documents/venv312/bin/python integration_tests/verify_e2e_eval.py
~/Documents/venv312/bin/python integration_tests/verify_e2e_prefixes.py
~/Documents/venv312/bin/python integration_tests/verify_e2e_dpsizes.py
~/Documents/venv312/bin/python integration_tests/verify_e2e_objectives.py
```

Each takes `--quick` (3 iters, half the sweep points), `--iters N` and
`--keep`. **Defaults are 5 iterations** — these check the pipeline *runs*, not
that it converges. Pass `--iters` when you want convergence.

| file | what it verifies |
|---|---|
| `verify_e2e_eval.py` | `evaluate_all_metrics` at `small` and `actual-5`: all six solution types solved, per-UG latency vectors populated and finite. |
| `verify_e2e_prefixes.py` | `evaluate_over_n_prefixes.py` at `small` over four prefix budgets, with `--plot`. |
| `verify_e2e_dpsizes.py` | `evaluate_over_deployment_sizes.py` over 3/4/5/6 PoPs, with `--plot`. |
| `verify_e2e_objectives.py` | One evaluation per objective at `small`: `avg_latency`, `per_site_cost`, `max_util`, `frac_beyond_optimal`. |
| `_common.py` | Shared workspace/env/scan/collect machinery. |

All four drive the real `evaluations/` drivers rather than reimplementing their
loops, so the figures are the ones the paper sweeps produce. Artifacts land in
`figures/integration_tests/<case>/` — deliberately separate from real sweep
output, so a test run can never be mistaken for one of your evaluations.

The drivers themselves take `--figures-subdir` for the same reason: a real
`evaluate_over_deployment_sizes.py` run can put its figures under
`figures/<whatever>/` and keep them together. Both routes go through
`SCULPTOR_FIG_SUBDIR`, which `helpers.save_fig` honours — no plotting call
needs to know about it.

## Why none of them trust the exit code

`evaluate_all_metrics` wraps its strategy loop in a bare `except:` that prints
and continues, so a run whose solver died in the first second still reaches
plotting, returns a metrics dict, and exits 0. On 2026-08-21 the queue read
exactly that as success and its harvest step deleted 11h of actual-32 training.
Each case is judged on the artifacts instead.

## Traps the harness itself had to be fixed for

Worth knowing, because each one made a broken run look green:

- **Stale artifacts.** The workspace cache symlinks the repo, so a previous
  run's result pickle satisfied every content check. Worse,
  `evaluate_all_metrics` *resume-skips* a sim whose `n_advs` is populated, so
  the run became a 9-second no-op that passed. `_common.workspace()` symlinks
  cache inputs but excludes result pickles, and pickle mtimes must post-date
  the run start.
- **`SCULPTOR_RUN_TAG` renames the pickle.** The objectives case was looking
  for the untagged filename, finding nothing, and skipping every content check
  while reporting PASS.
- **`RAY_TMPDIR` length.** Ray sockets are AF_UNIX, capped at 103 bytes; a
  `tempfile.mkdtemp()` root under `/var/folders/...` blows the cap on macOS.
- **Cross-contamination.** A second case attached to the first's
  still-draining Ray cluster and died ~2s in, which reads like a code bug.
  `RAY_ADDRESS=local` forces a private cluster per case.

The failure paths were verified against six synthetic cases, so this is not a
suite that only ever passes.

## Constraints these tests discovered

- **`n_prefixes >= n_pops + 1`.** `init_advertisement` gives prefix 0 to
  anycast and one prefix per PoP. `small` has 3 PoPs, so budgets below 4 make
  sparse fail — as a swallowed IndexError before 2026-08-21, as a clear
  `ValueError` since. Default budgets are `[4, 6, 8, 10]`.
- **`actual-5` is slow.** 2831s at *three* iterations on a laptop; it rebuilds
  from the 4.2 GB CSV and carries 2567 UGs. Its default is 3 iters for that
  reason — 50 is a cluster setting.

## Environment

Use `~/Documents/venv312/bin/python`. `~/Documents/venv` is Python 3.14 without
`highspy`, and `gpshim` defaults `SCULPTOR_LP_BACKEND` to `highs`. Each script
preflights and fails in under a second on the wrong interpreter, printing the
correct re-run command.

`--port` is not passed by any of these: nothing binds it under Ray
(`path_distribution_computer` sets `self.port = 0` outright).
