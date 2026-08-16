#!/bin/bash
# hardB3 v3 (Tom 2026-08-15 ~02:30Z): monotone ladder x hard objectives,
# FINAL config:
#   - 200 iters (100 left visible room on the hard objectives)
#   - 'mlu' = PURE max_util (no latency term)
#   - windowed-c gate (U_WINDOW default = TCONV/2, new fork deployed)
#   - ALL OBJECTIVES CONCURRENT (Tom: prioritize one deployment x all N
#     x ALL objectives over objective-at-a-time), 9 queues, each
#     deployment-major, 24 slots total, staggered launch
#   - gamma 0, out cache/ablation/hardB3v2/<obj>/<pdir> (same root the
#     dash/scoring read; 100-iter era quarantined)
# Prints HB3V3 COMPLETE / AUDIT FAILED to logs/hb3v3_driver.log.
set -u
REPO=/home/ubuntu/sparse_advertisements_code
PY=/home/ubuntu/venv/bin/python
DLOG=$REPO/logs/hb3v3_driver.log
cd $REPO
BASE="SCULPTOR_LAT_MODEL=geo SCULPTOR_PREF_MODEL=random SCULPTOR_LB_CACHE=0 SCULPTOR_MC_NUM=1 SCULPTOR_MC_NUM_EXPLORE=5 SCULPTOR_ABLATION_KEEP_RUNS=1 SCULPTOR_XOBJS=1"

echo "[hb3v3] SMOKE $(date -u +%H:%M:%SZ)" | tee -a $DLOG
rm -rf cache/ablation/hb3v3_smoke /home/ubuntu/hb3v2_ws_smk3*
env $BASE SCULPTOR_ABLATION_OBJECTIVE=max_util $PY -m experiments.ablation.run_n_sweep_queue \
  --out-root cache/ablation/hb3v3_smoke \
  --init-src cache/ablation/nsweep_v2_inits_georand \
  --ws-root /home/ubuntu/hb3v2_ws_smk3 --n-values 5 --rungs full --seeds 1 \
  --max-iter 2 --slots 1 --workers-per-run 1 --probe-mode smart \
  --port0 65000 --gamma 0 --py $PY >> logs/hb3v3_chain.log 2>&1
n=$(find cache/ablation/hb3v3_smoke -name "seed_1_full.json" | wc -l)
echo "[hb3v3] smoke jsons=$n/1" | tee -a $DLOG
if [ "$n" != "1" ]; then
  echo "[hb3v3] SMOKE FAILED - stopping" | tee -a $DLOG
  echo "AUDIT FAILED" >> $DLOG
  exit 1
fi

echo "[hb3v3] DENSE 540 cells, 200 iters, 3 objectives CONCURRENT, deployment-major $(date -u +%H:%M:%SZ)" | tee -a $DLOG
run_q() {
  obj=$1; objdir=$2; pdir=$3; pmode=$4; rungs=$5; extra=$6; port0=$7; slots=$8
  env $BASE SCULPTOR_ABLATION_OBJECTIVE=$obj $extra \
    $PY -m experiments.ablation.run_n_sweep_queue \
    --out-root cache/ablation/hardB3v2/$objdir/$pdir \
    --init-src cache/ablation/nsweep_v2_inits_georand \
    --ws-root /home/ubuntu/hb3v2_ws_${objdir}_${pdir} --n-values 1,2,5,10,20,50 \
    --rungs $rungs --seeds 1-5 --max-iter 200 --slots $slots \
    --workers-per-run 1 --probe-mode $pmode --port0 $port0 \
    --gamma 0 --py $PY >> logs/hb3v3_chain.log 2>&1
  echo "[hb3v3] $objdir/$pdir rc=$? $(date -u +%H:%M:%SZ)" | tee -a $DLOG
}
port=65100
for pair in "frac_beyond_optimal fracb" "max_util mlu" "joint_latency_bulk_download prio"; do
  set -- $pair; obj=$1; objdir=$2
  run_q $obj $objdir fixed fixed no_mc SCULPTOR_ABLATION_FIXED_BUDGET=1 $port 1 &
  port=$((port+40)); sleep 45
  run_q $obj $objdir sched scheduled no_mc,no_memory "" $port 3 &
  port=$((port+40)); sleep 45
  run_q $obj $objdir smart smart no_memory,no_direction,full "" $port 4 &
  port=$((port+40)); sleep 45
done
wait

nj=$(find cache/ablation/hardB3v2/fracb cache/ablation/hardB3v2/mlu cache/ablation/hardB3v2/prio -name "seed_*_*.json" 2>/dev/null | wc -l)
echo "[hb3v3] jsons=$nj/540" | tee -a $DLOG
if [ "$nj" != "540" ]; then
  echo "AUDIT FAILED" >> $DLOG
  exit 1
fi
echo "HB3V3 COMPLETE" | tee -a $DLOG
