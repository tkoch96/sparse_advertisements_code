#!/bin/bash
# 3-condition SCULPTOR comparison at small / 50 iters / same seed.
#
# Runs 3 invocations of run_ray.py eval_latency_failure:
#   A: RB-grad (default)
#   C: headroom (SCULPTOR_CAPACITY_HEADROOM=0.2 + SCULPTOR_SKIP_RB_GRAD=1)
#   D: stochastic LP (SCULPTOR_USE_STOCHASTIC_LP_GRAD=1 + SCULPTOR_STOCHASTIC_LP_K=16)
#
# All three use the same SCULPTOR_DEPLOYMENT_SEED=1 so they see the same
# problem instance and the same initial advertisement.
#
# Usage:
#   ./benchmarks/sculptor_3way_runner.sh [MAX_ITER]
# Default MAX_ITER=50.

set -uo pipefail

MAX_ITER=${1:-50}
SEED=1
N_WORKERS=2
DPSIZE=small
HEADROOM=0.2
K_STOCH=16
PY=/Users/tomkoch/Documents/venv312/bin/python
REPO=/Users/tomkoch/Documents/sparse_advertisements_code
OUT_DIR=${REPO}/benchmarks/out

mkdir -p "${OUT_DIR}"
ts=$(date +%Y%m%d_%H%M%S)

run_one() {
    local cond="$1"
    local extra_env="$2"
    local logname="${OUT_DIR}/3way_${cond}_${ts}.log"
    echo "=== [${cond}] launching MAX_ITER=${MAX_ITER} log=${logname}"
    cd "${REPO}"
    eval "SCULPTOR_DEPLOYMENT_SEED=${SEED} \
        SCULPTOR_RUN_TAG=3way_${cond} \
        SCULPTOR_MAX_ITER=${MAX_ITER} \
        SCULPTOR_N_WORKERS=${N_WORKERS} \
        ${extra_env} \
        ${PY} run_ray.py eval_latency_failure \
            --port 3170${RANDOM:0:1} --dpsize ${DPSIZE} \
            > '${logname}' 2>&1"
    echo "=== [${cond}] done"
    echo "${logname}"
}

# Sequential. (Parallel risks Gurobi WLS contention.)
A_LOG=$(run_one A "")
C_LOG=$(run_one C "SCULPTOR_CAPACITY_HEADROOM=${HEADROOM} SCULPTOR_SKIP_RB_GRAD=1")
D_LOG=$(run_one D "SCULPTOR_USE_STOCHASTIC_LP_GRAD=1 SCULPTOR_STOCHASTIC_LP_K=${K_STOCH}")

echo "ALL RUNS DONE"
echo "A_LOG=${A_LOG}"
echo "C_LOG=${C_LOG}"
echo "D_LOG=${D_LOG}"
