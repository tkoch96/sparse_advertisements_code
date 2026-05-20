#!/bin/bash
# Heartbeat status for the Session-3 A/B runs.
# Usage: ssh head 'bash ~/sparse_advertisements_code/session3_status.sh'

RUN_DIR=/tmp/cluster_runs
MANIFEST=${RUN_DIR}/session3_manifest.tsv

if [ ! -f "${MANIFEST}" ]; then
    echo "No manifest yet at ${MANIFEST}"
    exit 1
fi

echo "=== $(date -u +%FT%TZ) ==="

# Skip header line; iterate over the 10 rows
running=0
done_count=0
while IFS=$'\t' read -r seed cond pid log run_dir_hint; do
    if [ "${seed}" = "seed" ]; then continue; fi
    if ps -p "${pid}" > /dev/null 2>&1; then
        status=RUNNING
        running=$((running+1))
    else
        status=STOPPED
        done_count=$((done_count+1))
    fi
    # Latest learning-iteration line (handle tqdm \r within lines)
    last_iter=$(tr '\r' '\n' < "${log}" 2>/dev/null | grep -E "LEARNING ITERATION" | tail -1 | tr -d '\n')
    # Latest exception (if any) – first line of last traceback
    last_err=$(grep -E "Error|Traceback|Exception" "${log}" 2>/dev/null | tail -1 | head -c 100)
    printf "seed=%s cond=%s pid=%s %-7s | %s | err: %s\n" \
        "${seed}" "${cond}" "${pid}" "${status}" "${last_iter}" "${last_err}"
done < "${MANIFEST}"

echo "--- ${running} running, ${done_count} stopped ---"
ray status 2>&1 | grep -E "^ |Active:|Idle:|Pending:|Total" | head -15
