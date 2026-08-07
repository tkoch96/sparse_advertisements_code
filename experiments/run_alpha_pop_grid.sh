#!/usr/bin/env bash
# Parallel grid sweep over (alpha, anneal_end) for SCULPTOR_ALPHA_POP.
#
# Each cell trains sparse + painter + one_per_peering on the *same* seeded
# `small` deployment, then dumps a JSON with the paper-plot pop-failure
# metrics. Run analyze_alpha_pop_grid.py afterwards to compare.
#
# Defaults: NSIM=1 (single-seed scan, cheap), MAX_ITER=100, seed=1.
# Override via env vars: NSIM, MAX_ITER, SEED, DPSIZE, OUT_DIR, PARALLEL.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-~/Documents/venv312/bin/python}
DPSIZE=${DPSIZE:-small}
NSIM=${NSIM:-1}
MAX_ITER=${MAX_ITER:-100}
SEED=${SEED:-1}
OUT_DIR=${OUT_DIR:-/tmp/alpha_pop_grid}
PARALLEL=${PARALLEL:-3}

mkdir -p "$OUT_DIR" "$OUT_DIR/logs"
echo "[$(date)] dpsize=$DPSIZE nsim=$NSIM max_iter=$MAX_ITER seed=$SEED out=$OUT_DIR parallel=$PARALLEL"

# Grid: (alpha, anneal_end, port_offset, tag)
CONFIGS=(
  "0.00 0  0  a000_const"
  "0.05 0  1  a005_const"
  "0.10 0  2  a010_const"
  "0.20 0  3  a020_const"
  "0.30 0  4  a030_const"
  "0.50 0  5  a050_const"
  "0.10 50 6  a010_an50"
  "0.30 50 7  a030_an50"
  "0.50 50 8  a050_an50"
)
PORT_BASE=${PORT_BASE:-31900}

run_cell() {
  local alpha="$1"; local anneal="$2"; local poff="$3"; local tag="$4"
  local port=$((PORT_BASE + poff))
  local out="$OUT_DIR/${tag}.json"
  local log="$OUT_DIR/logs/${tag}.log"

  # Clean cache so we don't accidentally reuse a stale pickle
  rm -f "cache/popp_failure_latency_comparison_${DPSIZE}_${tag}.pkl"

  echo "  [START] tag=$tag alpha=$alpha anneal=$anneal port=$port log=$log"
  $PY -u -m experiments.alpha_pop_search \
      --alpha "$alpha" --anneal-end "$anneal" \
      --dpsize "$DPSIZE" --port "$port" \
      --nsim "$NSIM" --max-iter "$MAX_ITER" \
      --seed "$SEED" --tag "$tag" \
      --train-log "$log" \
      --out "$out" \
      > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "  [DONE]  tag=$tag rc=0"
  else
    echo "  [FAIL]  tag=$tag rc=$rc (see $log)"
  fi
}

# Bounded-parallel runner (batched). macOS bash 3.2 doesn't support `wait -n`,
# so we just batch-and-wait: launch PARALLEL cells, wait for ALL, launch next.
n=${#CONFIGS[@]}
i=0
while [ $i -lt $n ]; do
  batch_end=$((i + PARALLEL))
  [ $batch_end -gt $n ] && batch_end=$n
  for ((j=i; j<batch_end; j++)); do
    # shellcheck disable=SC2086
    run_cell ${CONFIGS[$j]} &
  done
  wait
  i=$batch_end
done

echo "[$(date)] ALL CELLS DONE"
echo "Analyze with: $PY experiments/analyze_alpha_pop_grid.py --in $OUT_DIR"
