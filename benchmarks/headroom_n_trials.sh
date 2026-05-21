#!/bin/bash
# Headroom (condition C) × N trials, run with up to MAX_CONCURRENT in parallel.
# Each trial uses a different SCULPTOR_DEPLOYMENT_SEED, otherwise identical.
#
# Usage:
#   ./benchmarks/headroom_n_trials.sh <dpsize> <max_iter> <max_concurrent> <seed1> [seed2 ...]
#
# Example:
#   ./benchmarks/headroom_n_trials.sh small 50 3 1 2 3
#   ./benchmarks/headroom_n_trials.sh actual-10 150 3 1 2 3 4 5
#
# Concurrency cap matters for Gurobi WLS license: it has ~3 concurrent
# sessions baseline and hard-cuts sustained overage at 32 min.

set -o pipefail
# Deliberately NOT using `set -u` because we dereference associative arrays
# that may be empty, which would otherwise trip nounset.

DPSIZE="${1:?usage: $0 <dpsize> <max_iter> <max_concurrent> <seeds...>}"
MAX_ITER="${2:?usage: $0 <dpsize> <max_iter> <max_concurrent> <seeds...>}"
MAX_CONCURRENT="${3:?usage: $0 <dpsize> <max_iter> <max_concurrent> <seeds...>}"
shift 3
SEEDS=("$@")

REPO=/home/ubuntu/sparse_advertisements_code
PY=/home/ubuntu/venv312/bin/python
OUT_DIR=/tmp/cluster_runs/headroom_n
HEADROOM=0.2
N_WORKERS=2

mkdir -p "${OUT_DIR}"
ts=$(date +%Y%m%d_%H%M%S)

cd "${REPO}"

launch_one() {
    local seed="$1"
    local tag="hd_seed${seed}_${DPSIZE}"
    local logname="${OUT_DIR}/${tag}_${ts}.log"
    # Clear per-tag metrics so we don't short-circuit training
    rm -f "${REPO}/cache/popp_failure_latency_comparison_${DPSIZE}_${tag}.pkl"
    echo "[$(date +%T)] LAUNCH seed=${seed} log=${logname}"
    SCULPTOR_DEPLOYMENT_SEED=${seed} \
        SCULPTOR_RUN_TAG=${tag} \
        SCULPTOR_MAX_ITER=${MAX_ITER} \
        SCULPTOR_N_WORKERS=${N_WORKERS} \
        SCULPTOR_CAPACITY_HEADROOM=${HEADROOM} \
        SCULPTOR_SKIP_RB_GRAD=1 \
        ${PY} run_ray.py eval_latency_failure \
            --port $((31400 + seed)) --dpsize ${DPSIZE} \
            > "${logname}" 2>&1 &
    echo $!  # pid
}

# Queue with max concurrency
declare -A PID_TO_SEED
for seed in "${SEEDS[@]}"; do
    # Block until we have headroom for another concurrent run
    while [ "${#PID_TO_SEED[@]}" -ge "${MAX_CONCURRENT}" ]; do
        # Wait for any one to finish
        for pid in "${!PID_TO_SEED[@]}"; do
            if ! kill -0 "${pid}" 2>/dev/null; then
                done_seed="${PID_TO_SEED[$pid]}"
                echo "[$(date +%T)] FINISHED seed=${done_seed} pid=${pid}"
                unset 'PID_TO_SEED[$pid]'
            fi
        done
        if [ "${#PID_TO_SEED[@]}" -ge "${MAX_CONCURRENT}" ]; then
            sleep 10
        fi
    done
    new_pid=$(launch_one "${seed}")
    PID_TO_SEED[${new_pid}]="${seed}"
done

# Drain
while [ "${#PID_TO_SEED[@]}" -gt 0 ]; do
    for pid in "${!PID_TO_SEED[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            done_seed="${PID_TO_SEED[$pid]}"
            echo "[$(date +%T)] FINISHED seed=${done_seed} pid=${pid}"
            unset 'PID_TO_SEED[$pid]'
        fi
    done
    if [ "${#PID_TO_SEED[@]}" -gt 0 ]; then
        sleep 10
    fi
done

echo "[$(date +%T)] ALL ${#SEEDS[@]} TRIALS DONE for dpsize=${DPSIZE}"
echo "Logs in: ${OUT_DIR}"
ls -1 "${OUT_DIR}"/hd_seed*_${DPSIZE}_${ts}.log
