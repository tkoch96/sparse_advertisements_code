#!/bin/bash
# v3 FULL GRID (2026-08-16, awaiting Tom's GO): the WHAT/WHEN ladder x
# all four objectives under post-fix semantics (lambduh=0, p5 cutoff).
# 840 cells: {classic g=.1, fracb(hinge), mlu(standalone max_util v2),
# prio} x {L1..L7} x seeds 1-5 x N{1,2,5,10,20,50}, 100 iters,
# deployment-major across ALL lanes (seed-1 block completes first).
# ONE governed manifest pool (~/queue_governor.json, RAM target 90%).
# Run: setsid nohup bash chain_v3.sh > /dev/null 2>&1 &
# Verdicts -> logs/v3_driver.log: V3 COMPLETE / AUDIT FAILED.
set -u
REPO=/home/ubuntu/sparse_advertisements_code
PY=/home/ubuntu/venv/bin/python
DLOG=$REPO/logs/v3_driver.log
cd $REPO
BASE="SCULPTOR_LAT_MODEL=geo SCULPTOR_PREF_MODEL=random SCULPTOR_LB_CACHE=0 SCULPTOR_MC_NUM=1 SCULPTOR_MC_NUM_EXPLORE=5 SCULPTOR_ABLATION_KEEP_RUNS=1"

nq=$(pgrep -fc "run_n_sweep_queu[e]" || true)
if [ "${nq:-0}" -ne 0 ]; then
  echo "[v3] queues already running - refusing to double-launch" | tee -a $DLOG
  exit 1
fi

echo "[v3] SMOKE (2 iters, one classic + one hard cell) $(date -u +%H:%M:%SZ)" | tee -a $DLOG
rm -rf cache/ablation/v3_smoke_a cache/ablation/v3_smoke_b /home/ubuntu/v3f_ws_smk*
cat > /home/ubuntu/v3_gate_manifest.json <<'EOF'
[
 {"label": "smkA", "out_root": "cache/ablation/v3_smoke_a",
  "probe_mode": "scheduled", "rungs": "full", "seeds": "1", "n_values": "5",
  "gamma": "0.1", "env": {"SCULPTOR_ABLATION_PROBE_TARGET": "current"}},
 {"label": "smkB", "out_root": "cache/ablation/v3_smoke_b",
  "probe_mode": "smart", "rungs": "full", "seeds": "1", "n_values": "5",
  "gamma": "0", "env": {"SCULPTOR_XOBJS": "1",
      "SCULPTOR_ABLATION_OBJECTIVE": "max_util",
      "SCULPTOR_ABLATION_PROBE_TARGET": "maxinfo",
      "SCULPTOR_ABLATION_SMART_MINGAP_FRAC": "0.7"}}
]
EOF
env $BASE $PY -m experiments.ablation.run_n_sweep_queue \
  --manifest /home/ubuntu/v3_gate_manifest.json \
  --init-src cache/ablation/nsweep_v2_inits_georand \
  --ws-root /home/ubuntu/v3f_ws_smk --max-iter 2 --slots 2 \
  --workers-per-run 1 --port0 61000 --launch-stagger 5 \
  --py $PY >> logs/v3_chain.log 2>&1
n=$(find cache/ablation/v3_smoke_a cache/ablation/v3_smoke_b -name "seed_1_full.json" 2>/dev/null | wc -l)
echo "[v3] smoke jsons=$n/2" | tee -a $DLOG
if [ "$n" != "2" ]; then
  echo "[v3] SMOKE FAILED - stopping" | tee -a $DLOG
  echo "AUDIT FAILED" >> $DLOG
  exit 1
fi

echo "[v3] DENSE 840 cells (4 objectives x L1-L7 x 5 seeds x 6 N, 100 iters) $(date -u +%H:%M:%SZ)" | tee -a $DLOG
env $BASE $PY -m experiments.ablation.run_n_sweep_queue \
  --manifest /home/ubuntu/v3_full_manifest.json \
  --init-src cache/ablation/nsweep_v2_inits_georand \
  --ws-root /home/ubuntu/v3f_ws --max-iter 100 --slots 28 \
  --workers-per-run 1 --port0 61100 --launch-stagger 30 \
  --py $PY >> logs/v3_chain.log 2>&1
echo "[v3] manifest rc=$? $(date -u +%H:%M:%SZ)" | tee -a $DLOG

nc=$(find cache/ablation/policy_ladder_v3 -name "seed_*_*.json" 2>/dev/null | wc -l)
nh=$(find cache/ablation/hardobj_v3 -name "seed_*_*.json" 2>/dev/null | wc -l)
echo "[v3] classic=$nc/210 hard=$nh/630" | tee -a $DLOG
if [ "$nc" != "210" ] || [ "$nh" != "630" ]; then
  echo "AUDIT FAILED" >> $DLOG
  exit 1
fi
echo "V3 COMPLETE" | tee -a $DLOG
