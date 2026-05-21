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

DPSIZE="${1:?usage: $0 <dpsize> <max_iter> <max_concurrent> <seeds...>}"
MAX_ITER="${2:?usage: $0 <dpsize> <max_iter> <max_concurrent> <seeds...>}"
MAX_CONCURRENT="${3:?usage: $0 <dpsize> <max_iter> <max_concurrent> <seeds...>}"
shift 3
SEEDS=("$@")

REPO=/home/ubuntu/sparse_advertisements_code
PY=/home/ubuntu/venv312/bin/python
OUT_DIR=/tmp/cluster_runs/headroom_n
HEADROOM=0.2
# N_WORKERS per trial. Higher = more parallelism on LB-grad probes; user
# confirmed they've used 64+ without Gurobi WLS issues. Session 3 saw
# ~37s/iter on actual-10 with N_WORKERS=8 (vs ~120s/iter with N_WORKERS=2).
N_WORKERS=8

mkdir -p "${OUT_DIR}"
ts=$(date +%Y%m%d_%H%M%S)

cd "${REPO}"

# Tracks pid -> seed for active runs.
pids=()
pid_seeds=()

launch_one() {
    local seed="$1"
    local tag="hd_seed${seed}_${DPSIZE}"
    local logname="${OUT_DIR}/${tag}_${ts}.log"
    rm -f "${REPO}/cache/popp_failure_latency_comparison_${DPSIZE}_${tag}.pkl"
    SCULPTOR_DEPLOYMENT_SEED=${seed} \
        SCULPTOR_RUN_TAG=${tag} \
        SCULPTOR_MAX_ITER=${MAX_ITER} \
        SCULPTOR_N_WORKERS=${N_WORKERS} \
        SCULPTOR_CAPACITY_HEADROOM=${HEADROOM} \
        ${PY} run_ray.py eval_latency_failure \
            --port $((31400 + seed)) --dpsize ${DPSIZE} \
            > "${logname}" 2>&1 &
    local new_pid=$!
    pids+=("${new_pid}")
    pid_seeds+=("${seed}")
    echo "[$(date +%T)] LAUNCH seed=${seed} pid=${new_pid} log=${logname}"
}

reap_dead() {
    # Walk current pids, drop ones that are no longer alive.
    local new_pids=()
    local new_seeds=()
    local i=0
    while [ $i -lt ${#pids[@]} ]; do
        local p="${pids[$i]}"
        local s="${pid_seeds[$i]}"
        if kill -0 "${p}" 2>/dev/null; then
            new_pids+=("${p}")
            new_seeds+=("${s}")
        else
            echo "[$(date +%T)] FINISHED seed=${s} pid=${p}"
        fi
        i=$((i+1))
    done
    pids=("${new_pids[@]}")
    pid_seeds=("${new_seeds[@]}")
}

active_count() {
    echo "${#pids[@]}"
}

for seed in "${SEEDS[@]}"; do
    while [ "$(active_count)" -ge "${MAX_CONCURRENT}" ]; do
        sleep 10
        reap_dead
    done
    launch_one "${seed}"
done

while [ "$(active_count)" -gt 0 ]; do
    sleep 10
    reap_dead
done

echo "[$(date +%T)] ALL ${#SEEDS[@]} TRIALS DONE for dpsize=${DPSIZE}"
echo "Logs in: ${OUT_DIR}"
ls -1 "${OUT_DIR}"/hd_seed*_${DPSIZE}_${ts}.log
