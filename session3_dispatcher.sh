#!/bin/bash
# Session-3 dispatcher for RESEARCH_ROADMAP Tier 1 #1.
#
# 5 trial pairs (10 detached SCULPTOR runs) on actual-10:
#   - Condition A (baseline): SCULPTOR_SKIP_RB_GRAD unset, no headroom
#   - Condition C (headroom): SCULPTOR_CAPACITY_HEADROOM=0.2, SCULPTOR_SKIP_RB_GRAD=1
#
# Each pair shares SCULPTOR_DEPLOYMENT_SEED so they see the same problem
# instance. Run dirs differ via a SCULPTOR_RUN_TAG that we surface in the
# manifest at the end; the run dir name itself uses int(time.time()) which
# is naturally unique since launches are staggered by 2s.
#
# Designed to be invoked via:
#   ray exec ray-cluster.yaml 'bash ~/sparse_advertisements_code/session3_dispatcher.sh'
#
# Or in pieces (recommended for first launch) — see the comments below.

set -euo pipefail

REPO=/home/ubuntu/sparse_advertisements_code
PY=/home/ubuntu/venv312/bin/python
DPSIZE=actual-10
MAX_ITER=100
N_WORKERS=8
HEADROOM=0.2
SEEDS=(1 2 3 4 5)
RUN_DIR=/tmp/cluster_runs
MANIFEST=${RUN_DIR}/session3_manifest.tsv

mkdir -p "${RUN_DIR}"
cd "${REPO}"

# ----------------------------------------------------------------------
# Phase 1: pre-build deployment caches sequentially.
# Each seed's first call to get_random_deployment parses the 4.5GB CSV and
# pickles a cache; running this in parallel would 5x peak memory on the
# m7g.4xlarge head (64GB). Sequential is ~5 min/seed = ~25 min total.
# ----------------------------------------------------------------------
echo "[$(date -u +%FT%TZ)] Phase 1: pre-building 5 deployment caches"
for seed in "${SEEDS[@]}"; do
    echo "[$(date -u +%FT%TZ)] Building cache for seed=${seed}"
    SCULPTOR_DEPLOYMENT_SEED=${seed} "${PY}" -c "
import os
from deployment_setup import get_random_deployment
d = get_random_deployment('${DPSIZE}')
print('seed={} popps={} ugs={}'.format(${seed}, len(d['popps']), len(d['ugs'])))
" 2>&1 | tail -5
done
echo "[$(date -u +%FT%TZ)] Phase 1 complete"

# ----------------------------------------------------------------------
# Phase 2: launch 10 detached SCULPTOR runs.
# Stagger launches by 2s so run-dir timestamps are unique.
# ----------------------------------------------------------------------
echo "[$(date -u +%FT%TZ)] Phase 2: launching 10 detached SCULPTOR runs"
echo -e "seed\tcondition\tpid\tlog\trun_dir_hint" > "${MANIFEST}"

launch_one() {
    local seed=$1
    local cond=$2   # A or C
    local ts=$(date +%Y%m%d_%H%M%S)
    local logname="${RUN_DIR}/seed${seed}_${cond}_${ts}.log"
    local pidname="${RUN_DIR}/seed${seed}_${cond}_${ts}.pid"

    local extra_env=""
    if [ "${cond}" = "C" ]; then
        extra_env="SCULPTOR_CAPACITY_HEADROOM=${HEADROOM} SCULPTOR_SKIP_RB_GRAD=1"
    fi

    # Approx t_start that SCULPTOR will use for save_run_dir naming — for the
    # manifest hint only; the real dir is created inside SAS.
    local t_start_hint=$(date +%s)

    cd "${REPO}"
    eval "SCULPTOR_DEPLOYMENT_SEED=${seed} \
        SCULPTOR_RUN_TAG=seed${seed}_${cond} \
        SCULPTOR_MAX_ITER=${MAX_ITER} \
        SCULPTOR_N_WORKERS=${N_WORKERS} \
        ${extra_env} \
        nohup ${PY} run_ray.py eval_latency_failure \
            --port 31415 --dpsize ${DPSIZE} \
            > ${logname} 2>&1 < /dev/null &"
    local pid=$!
    echo "${pid}" > "${pidname}"
    echo -e "${seed}\t${cond}\t${pid}\t${logname}\t${t_start_hint}-${DPSIZE}-sparse" >> "${MANIFEST}"
    echo "  launched seed=${seed} cond=${cond} pid=${pid} -> ${logname}"
}

for seed in "${SEEDS[@]}"; do
    launch_one "${seed}" A
    sleep 2
    launch_one "${seed}" C
    sleep 2
done

echo "[$(date -u +%FT%TZ)] Phase 2 complete. Manifest:"
cat "${MANIFEST}"
echo
echo "Monitor with:  tail -F ${RUN_DIR}/seed*.log"
echo "Status with:   bash ${REPO}/session3_status.sh   (created separately)"
