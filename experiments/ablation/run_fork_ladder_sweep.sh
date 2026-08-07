#!/usr/bin/env bash
# Sequential sweep over (seed, rung) for the fork ladder. Sequential on
# purpose: each run holds ~2-3 Gurobi WLS sessions (workers + driver).
# Env overrides: SEEDS, RUNGS, MAX_ITER, OUT_DIR, PORT_BASE.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-~/Documents/venv312/bin/python}
SEEDS=${SEEDS:-"1 2 3"}
RUNGS=${RUNGS:-"full expl_random expl_none no_direction no_memory painter"}
MAX_ITER=${MAX_ITER:-30}
OUT_DIR=${OUT_DIR:-cache/ablation/fork_smoke}
PORT_BASE=${PORT_BASE:-31800}
DPSIZE=${DPSIZE:-small}

mkdir -p "$OUT_DIR" logs
i=0
for seed in $SEEDS; do
  for rung in $RUNGS; do
    port=$((PORT_BASE + i)); i=$((i + 1))
    log="logs/fork_${rung}_s${seed}.log"
    echo "[sweep] seed=$seed rung=$rung port=$port -> $log"
    $PY -u -m experiments.ablation.run_fork_ladder \
        --seed "$seed" --rung "$rung" --port "$port" \
        --max-iter "$MAX_ITER" --dpsize "$DPSIZE" --out-dir "$OUT_DIR" > "$log" 2>&1
    rc=$?
    # run dirs are self-contained: keep the full solve log next to the state
    rdir="runs/ablation-${DPSIZE}-${rung}-dep${seed}"
    [ -d "$rdir" ] && cp "$log" "$rdir/solve.log"
    grep -E "^\[seed" "$log" | tail -1
    [ $rc -ne 0 ] && echo "[sweep] FAIL seed=$seed rung=$rung rc=$rc (see $log)"
  done
done
echo "[sweep] ALL DONE"
