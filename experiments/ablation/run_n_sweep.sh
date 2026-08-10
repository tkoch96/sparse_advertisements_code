#!/bin/bash
# N-budget sweep of the fork ladder under gated probing (Tom's design,
# 2026-08-10): same 20 seeded deployments for EVERY N (canonical inits are
# pre-copied into each N dir from INIT_SRC, so the per-seed init-equality
# assertion enforces cross-N consistency), all rungs, gamma=0.1, canonical
# NO_ROUTE penalty. Lane-isolated workspaces (cwd-relative runs/) so
# concurrent lanes can never collide; post-audit gate refuses to rescore
# corrupted runs.
#
# Env: REPO PY OUT_ROOT INIT_SRC WS_ROOT [N_VALUES] [SEEDS] [MAX_ITER]
#      [LANES] [WORKERS_PER_LANE] [PORT0] [DO_RESCORE]
set -u
REPO=${REPO:?}; PY=${PY:?}; OUT_ROOT=${OUT_ROOT:?}
INIT_SRC=${INIT_SRC:?}; WS_ROOT=${WS_ROOT:?}
N_VALUES=${N_VALUES:-"1 2 5 10 20"}
SEEDS=${SEEDS:-"$(seq 1 20 | tr '\n' ' ')"}
MAX_ITER=${MAX_ITER:-200}
LANES=${LANES:-7}
WORKERS_PER_LANE=${WORKERS_PER_LANE:-4}
PORT0=${PORT0:-53000}
DO_RESCORE=${DO_RESCORE:-1}
RUNGS="full expl_random expl_none no_direction no_memory no_mc painter"

for N in $N_VALUES; do
  mkdir -p "$OUT_ROOT/N$N"
  cp "$INIT_SRC"/init_dep*.npy "$OUT_ROOT/N$N/" 2>/dev/null || true
done

lane () {
  local L=$1
  local WS=$WS_ROOT/L$L
  mkdir -p "$WS/runs" "$WS/logs" "$WS/figures/paper"
  ln -sfn "$REPO/cache" "$WS/cache"
  ln -sfn "$REPO/data" "$WS/data"
  cd "$WS"
  export PYTHONPATH=$REPO
  export SCULPTOR_ABLATION_GAMMA=0.1 SCULPTOR_N_WORKERS=$WORKERS_PER_LANE MPLBACKEND=Agg
  export RAY_ADDRESS=local RAY_TMPDIR=/tmp/ray_nsweep_L$L
  export SCULPTOR_ABLATION_PROBE_MODE=gated SCULPTOR_ABLATION_PROBE_TCONV=200
  local i=0 s N rung
  for s in $SEEDS; do
    i=$((i+1))
    [ $(( (i-1) % LANES + 1 )) -ne $L ] && continue
    for N in $N_VALUES; do
      export SCULPTOR_ABLATION_PROBE_N=$N
      for rung in $RUNGS; do
        $PY -u -m experiments.ablation.run_fork_ladder --seed $s --rung $rung \
          --port $((PORT0 + 20*L)) --max-iter $MAX_ITER --dpsize small \
          --out-dir "$OUT_ROOT/N$N" > "$WS/logs/N${N}_s${s}_${rung}.log" 2>&1 \
          || echo "[lane$L] FAIL N=$N seed=$s rung=$rung" >> "$OUT_ROOT/failures.log"
      done
    done
    echo "[nsweep] lane $L seed $s complete"
  done
}

: > "$OUT_ROOT/failures.log"
for L in $(seq 1 $LANES); do lane $L & done
wait
echo "[nsweep] sweep done; failures: $(grep -c . "$OUT_ROOT/failures.log")"

$PY - <<EOF
import json, glob, sys
bad = 0
for fn in glob.glob("$OUT_ROOT/N*/seed_*_*.json"):
    r = json.load(open(fn))
    if r["rung"] == "painter":
        continue
    # accept MAX_ITER+1 too: the iter counter's known off-by-2 is
    # sometimes an off-by-1 on clean runs (15/120 in the 20x200 reroll)
    if r.get("solve_error") or (r.get("n_iters") or 0) < $MAX_ITER + 1:
        print("[audit] BAD:", fn, r.get("n_iters"), str(r.get("solve_error"))[:40])
        bad += 1
print("[audit]", bad, "bad runs")
sys.exit(1 if bad else 0)
EOF
if [ $? -ne 0 ]; then echo "[nsweep] AUDIT FAILED"; exit 1; fi

if [ "$DO_RESCORE" = "1" ]; then
  for N in $N_VALUES; do
    for s in $SEEDS; do
      ( RAY_ADDRESS=local RAY_TMPDIR=/tmp/ray_nrs_${N}_$s MPLBACKEND=Agg \
        $PY -m experiments.ablation.rescore_fork --in-dir "$OUT_ROOT/N$N" \
        --dpsize small --seed $s > /dev/null 2>&1 ) &
      while [ "$(jobs -r | wc -l)" -ge 8 ]; do sleep 3; done
    done
  done
  wait
  echo "[nsweep] rescore done"
fi
echo "[nsweep] ALL DONE"
