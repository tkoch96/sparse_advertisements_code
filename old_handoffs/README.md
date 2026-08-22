# old_handoffs/

Point-in-time snapshots and concluded-program records. Useful for archaeology,
**not authoritative for current state** — `HANDOFF_NEXT.md` in the repo root is.

The reason to keep these: they record *which era invalidated which dataset*.
When you find an old result and want to know whether it is still usable, this
is where the answer is.

## Session handoffs

`HANDOFF_SESSION_6/7/8/9.md`, `SESSION_4/5_SUMMARY.md`, `OVERNIGHT_SUMMARY.md`,
`HANDOFF_2026-08-*.md`, `HANDOFF_root_pre_ablation.md`, `RESEARCH_ROADMAP.md`
(stale since 2026-05-27).

## Concluded programs

| file | what it records |
|---|---|
| `DEPLOYMENT_SPEEDUP_PROGRAM.md` | depcache + depsetup_fork, phases 1–3. Production code is now `core/shard_loader.py`, `core/fork_load.py`, `core/convert_latencies.py`; parity gates in `unit_tests/`. |
| `SOLVER_FORK_MIGRATION.md` | gurobi → HiGHS. Shim is `core/gpshim.py`; HiGHS is the default backend since 2026-08-20. Unit battery is `unit_tests/test_gpshim_unit.py`. |
| `MODEL_UNCERTAINTY_DIMENSIONS.md` | `DIMENSIONS.md` + `FORMATS.md` from the model-error program. |
| `UG_DESHARDING_SURVEY.md` | Why UG slices shipped to workers were inert, and the removal that followed. **Not** the same as latency sharding, which is live in `core/shard_loader.py`. |
| `RECOVERED_ACTUAL32_RESULT.md` | A converged actual-32 run (iter 202, sparse 28.96ms vs opp 28.35ms) from 2026-05-21. Its 853 MB of pickles were deleted; these numbers are what survived. Predates several era cuts — treat as historical. |
