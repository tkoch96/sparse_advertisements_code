#!/bin/bash
# One-seed dispatcher: waits for seed=1 cond=A python to finish, then runs
# seed=1 cond=C. Pushes status to ntfy.sh on key events so the user sees
# them on phone independent of any Claude session.
#
# Usage: ./session3_oneseed_dispatcher.sh <pid_of_seed1_A_python>
# Logs to /tmp/cluster_runs/oneseed_dispatcher.log
# Heartbeats to /tmp/cluster_runs/heartbeat.tsv

set -uo pipefail

WAIT_PID="${1:?usage: $0 <pid_of_in_flight_python>}"
TOPIC="${SCULPTOR_NTFY_TOPIC:?env var SCULPTOR_NTFY_TOPIC must be set}"
REPO=/home/ubuntu/sparse_advertisements_code
PY=/home/ubuntu/venv312/bin/python
DPSIZE=actual-10
MAX_ITER=100
N_WORKERS=2
HEADROOM=0.2
SEED=1
RUN_DIR=/tmp/cluster_runs
HEARTBEAT=${RUN_DIR}/heartbeat.tsv

cd "${REPO}"

push_ntfy() {
    local title="$1"
    local msg="$2"
    local priority="${3:-default}"   # default | high | min
    curl -sS \
        -H "Title: ${title}" \
        -H "Priority: ${priority}" \
        -H "Tags: gear" \
        -d "${msg}" \
        "https://ntfy.sh/${TOPIC}" > /dev/null || true
}

emit_hb() {
    echo -e "$(date -u +%FT%TZ)\t$1\tseed=${SEED}\t${2:-}" >> "${HEARTBEAT}"
}

push_ntfy "Session-3 follow-up" "Waiting for seed=1 cond=A (pid ${WAIT_PID}) to finish, then will run seed=1 cond=C."
emit_hb FOLLOWUP_START "waiting on pid=${WAIT_PID}"

# Phase 1: wait for the in-flight A run to finish.
poll_count=0
while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 300
    poll_count=$((poll_count+1))
    if [ $((poll_count % 12)) -eq 0 ]; then
        # Push every hour during the wait so user knows we're alive
        last_iter=$(tr '\r' '\n' < /tmp/cluster_runs/v2_seed1_A_*.log 2>/dev/null | grep -E "LEARNING ITERATION" | tail -1 | tr -d '\n' || echo "")
        push_ntfy "seed=1 A still running" "${last_iter} (poll #${poll_count})" min
        emit_hb WAITING "pid=${WAIT_PID} last_iter=${last_iter}"
    fi
done

emit_hb A_DONE "pid=${WAIT_PID} exited"

# Check whether A completed cleanly or crashed
a_log=$(ls -t /tmp/cluster_runs/v2_seed1_A_*.log 2>/dev/null | head -1)
if [ -z "${a_log}" ]; then
    push_ntfy "seed=1 A: no log found?!" "Cannot find /tmp/cluster_runs/v2_seed1_A_*.log; aborting." high
    exit 1
fi
if grep -q "GurobiError" "${a_log}" 2>/dev/null; then
    push_ntfy "seed=1 A DIED (Gurobi)" "Aborting follow-up; not running cond=C." high
    emit_hb ABORT "seed=1 A died with GurobiError"
    exit 1
fi
if grep -q "Stopped train loop" "${a_log}" 2>/dev/null; then
    push_ntfy "seed=1 A COMPLETED" "Now launching seed=1 cond=C (~3-4h)" default
    emit_hb A_COMPLETED ""
else
    push_ntfy "seed=1 A finished but no Stopped-train-loop marker" "Proceeding to cond=C anyway." default
    emit_hb A_UNCERTAIN ""
fi

# Phase 2: run seed=1 cond=C
ts=$(date +%Y%m%d_%H%M%S)
c_log="${RUN_DIR}/v2_seed1_C_${ts}.log"

push_ntfy "seed=1 C starting" "MAX_ITER=${MAX_ITER}, N_WORKERS=${N_WORKERS}, headroom=${HEADROOM}" default
emit_hb C_LAUNCH "log=${c_log}"

SCULPTOR_DEPLOYMENT_SEED=${SEED} \
    SCULPTOR_RUN_TAG=v2_seed${SEED}_C \
    SCULPTOR_MAX_ITER=${MAX_ITER} \
    SCULPTOR_N_WORKERS=${N_WORKERS} \
    SCULPTOR_CAPACITY_HEADROOM=${HEADROOM} \
    SCULPTOR_SKIP_RB_GRAD=1 \
    ${PY} run_ray.py eval_latency_failure \
        --port 31415 --dpsize ${DPSIZE} \
        > "${c_log}" 2>&1 < /dev/null &
c_pid=$!

# Wait + hourly check-in
poll_count=0
start_ts=$(date +%s)
while kill -0 "${c_pid}" 2>/dev/null; do
    sleep 300
    poll_count=$((poll_count+1))
    if [ $((poll_count % 12)) -eq 0 ]; then
        elapsed=$(( $(date +%s) - start_ts ))
        last_iter=$(tr '\r' '\n' < "${c_log}" 2>/dev/null | grep -E "LEARNING ITERATION" | tail -1 | tr -d '\n' || echo "")
        push_ntfy "seed=1 C alive (${elapsed}s)" "${last_iter}" min
        emit_hb C_RUNNING "etime=${elapsed}s last_iter=${last_iter}"
    fi
done

wait "${c_pid}"
rc=$?
elapsed=$(( $(date +%s) - start_ts ))

# Final classification
status="UNKNOWN_rc${rc}"
if grep -q "GurobiError" "${c_log}" 2>/dev/null; then status=DIED_GUROBI
elif grep -q "Stopped train loop" "${c_log}" 2>/dev/null; then status=COMPLETED
fi

emit_hb C_FINISHED "status=${status} etime=${elapsed}s"

case "${status}" in
    COMPLETED)
        push_ntfy "seed=1 A+C DONE" "Both runs completed. Wall=${elapsed}s on C. Ready to aggregate." high
        ;;
    DIED_GUROBI)
        push_ntfy "seed=1 C DIED (Gurobi)" "Wall=${elapsed}s before crash." high
        ;;
    *)
        push_ntfy "seed=1 C exited unclear" "status=${status} rc=${rc} wall=${elapsed}s — check log" high
        ;;
esac
