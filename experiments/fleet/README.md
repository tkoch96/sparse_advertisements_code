# fleet — SCALE-500 multi-VM machinery (2026-08-17, P0/P1 skeleton)

Share-nothing fleet per ablation_study/SCALE_500_PLAN.md: each VM runs
the standard single-node queue on a manifest SHARD; this package only
provisions, shards, monitors, collects, tears down.

- registry.py — fleet list inside the alert-JSON contract (single-head
  consumers unaffected)
- shard.py — deployment-major manifest sharding
- fleet_up.py — bake AMI from the head / launch spot VMs / bootstrap
  (pinned SHA + known new-host fixes)
- fleet_launch.py — scp shard JSONs (data, not code) + start queues
- fleet_tick.py — fleet-aggregated progress.json (per-VM rows + summed
  ETA via the standard regression ticker)
- fleet_down.py — final result pull + terminate (head never terminated)

v1 durability = Mac-pull every tick (spot reclaim loses at most the
in-flight cells + last-pull delta; cells are idempotent). S3 write-
through is the P0 upgrade once an instance profile exists.

Burn-in checklist before first real fleet (SCALE_500_PLAN P3):
spot vCPU quota raise, bake AMI, 2-VM pilot incl. kill-a-VM drill,
EODS actual-3 cell validation + merge-vs-legacy parity.
