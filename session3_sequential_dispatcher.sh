#!/bin/bash
# Sequential dispatcher v2 — stays under Gurobi WLS baseline.
#
# 3 seeds, both conditions (A=baseline, C=headroom), MAX_ITER=100,
# SCULPTOR_N_WORKERS=2 (one persistent-Gurobi session per actor).
# Runs strictly one-at-a-time so total concurrent Gurobi sessions stays
# at 2. The previous parallel attempt exceeded baseline with 80 sessions
# and Gurobi hard-cut everything at T+32min.
#
# Writes /tmp/cluster_runs/heartbeat.tsv after each run so a polling
# session can resume context without parsing logs.
#
# Designed to be invoked via:
#   ssh head 'nohup bash ~/sparse_advertisements_code/session3_sequential_dispatcher.sh > /tmp/cluster_runs/dispatcher_v2.log 2>&1 < /dev/null &'

set -uo pipefail   # no -e: we want to continue even if a single run fails

REPO=/home/ubuntu/sparse_advertisements_code
PY=/home/ubuntu/venv312/bin/python
DPSIZE=actual-10
MAX_ITER=100
N_WORKERS=2
HEADROOM=0.2
SEEDS=(1 2 3)
RUN_DIR=/tmp/cluster_runs
MANIFEST=${RUN_DIR}/session3v2_manifest.tsv
HEARTBEAT=${RUN_DIR}/heartbeat.tsv

mkdir -p "${RUN_DIR}"
cd "${REPO}"

emit_heartbeat() {
    local phase="$1"
    local seed="$2"
    local cond="$3"
    local pid="$4"
    local status="$5"
    local etime="$6"
    local last_iter="$7"
    echo -e "$(date -u +%FT%TZ)\t${phase}\tseed=${seed}\tcond=${cond}\tpid=${pid}\t${status}\tetime=${etime}\tlast_iter=${last_iter}" >> "${HEARTBEAT}"
}

# Reset heartbeat for this dispatcher invocation
> "${HEARTBEAT}"
echo -e "ts\tphase\tseed\tcond\tpid\tstatus\tetime\tlast_iter" > "${MANIFEST}"

emit_heartbeat START - - - DISPATCHER_BEGIN - -

run_one() {
    local seed=$1
    local cond=$2
    local ts=$(date +%Y%m%d_%H%M%S)
    local logname="${RUN_DIR}/v2_seed${seed}_${cond}_${ts}.log"

    local extra_env=""
    if [ "${cond}" = "C" ]; then
        extra_env="SCULPTOR_CAPACITY_HEADROOM=${HEADROOM} SCULPTOR_SKIP_RB_GRAD=1"
    fi

    emit_heartbeat LAUNCH "${seed}" "${cond}" "" QUEUED "" ""

    # Run synchronously (no nohup &) — script blocks until completion
    eval "SCULPTOR_DEPLOYMENT_SEED=${seed} \
        SCULPTOR_RUN_TAG=v2_seed${seed}_${cond} \
        SCULPTOR_MAX_ITER=${MAX_ITER} \
        SCULPTOR_N_WORKERS=${N_WORKERS} \
        ${extra_env} \
        ${PY} run_ray.py eval_latency_failure \
            --port 31415 --dpsize ${DPSIZE} \
            > ${logname} 2>&1 < /dev/null" &
    local pid=$!
    echo -e "$(date -u +%FT%TZ)\tLAUNCHED\t${seed}\t${cond}\t${pid}\tRUNNING\t-\t-\t${logname}" >> "${MANIFEST}"
    emit_heartbeat RUNNING "${seed}" "${cond}" "${pid}" RUNNING 0s 0

    # Wait for completion; emit a heartbeat every 5 min while it runs
    local start_ts=$(date +%s)
    while kill -0 "${pid}" 2>/dev/null; do
        sleep 300
        local elapsed=$(( $(date +%s) - start_ts ))
        local last_iter=$(tr '\r' '\n' < "${logname}" 2>/dev/null | grep -E "LEARNING ITERATION" | tail -1 | tr -d '\n' || echo "")
        emit_heartbeat HEARTBEAT "${seed}" "${cond}" "${pid}" RUNNING "${elapsed}s" "${last_iter}"
    done

    # Capture exit code
    wait "${pid}"
    local rc=$?
    local elapsed=$(( $(date +%s) - start_ts ))
    local last_iter=$(tr '\r' '\n' < "${logname}" 2>/dev/null | grep -E "LEARNING ITERATION" | tail -1 | tr -d '\n' || echo "")
    local final_status="DONE_rc${rc}"
    if grep -q "GurobiError" "${logname}" 2>/dev/null; then
        final_status="DIED_GUROBI"
    fi
    if grep -q "Stopped train loop" "${logname}" 2>/dev/null; then
        final_status="COMPLETED"
    fi
    emit_heartbeat FINISHED "${seed}" "${cond}" "${pid}" "${final_status}" "${elapsed}s" "${last_iter}"
    echo -e "$(date -u +%FT%TZ)\tFINISHED\t${seed}\t${cond}\t${pid}\t${final_status}\t${elapsed}s\t${last_iter}\t${logname}" >> "${MANIFEST}"
}

for seed in "${SEEDS[@]}"; do
    run_one "${seed}" A
    run_one "${seed}" C
done

emit_heartbeat END - - - DISPATCHER_DONE - -
echo "All runs complete. Manifest at ${MANIFEST}"
