#!/bin/bash
# Watcher for the v3 era (2026-08-16): WHAT/WHEN ladder grids under
# post-fix semantics. IP re-resolved from the cluster alert JSON every
# beat. Targets: classic policy_ladder_v3 (210) + hardobj_v3 (630).
# Verdicts from logs/v3_driver.log (V3 COMPLETE / AUDIT FAILED).
IP=$(python3 -c "import json;print(json.load(open('$HOME/.sculptor_cluster_alert/active_cluster.json'))['head']['public_ip'])" 2>/dev/null)
SSH="ssh -i $HOME/.ssh/ray-autoscaler_us-east-1.pem -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new ubuntu@$IP"
OUT=$($SSH '
  cd sparse_advertisements_code
  C=$(find cache/ablation/policy_ladder_v3 -name "seed_*_*.json" 2>/dev/null | wc -l)
  H=$(find cache/ablation/hardobj_v3 -name "seed_*_*.json" 2>/dev/null | wc -l)
  D=$(df -h / | tail -1 | awk "{print \$4}")
  M=$(free -g | awk "/^Mem:/{print \$7}")
  R=$(pgrep -fc "run_fork_ladde[r]" || echo 0)
  V=$(grep -hE "V3 COMPLETE|AUDIT FAILED|SMOKE FAILED" logs/v3_driver.log 2>/dev/null | tail -1)
  G=$(grep -h "\[governor\]" logs/v3_chain.log 2>/dev/null | tail -1 | grep -oE "active=[0-9]+ .*admit=[A-Za-z]+")
  echo "v3: classic=$C/210 hard=$H/630 (${V:-running}) | gov: ${G:-n/a} | disk=$D memfree=${M}G | runners=$R"
' 2>&1)
echo "$(date -u +%H:%MZ) $OUT"
case "$OUT" in
  *ssh:*|*timed\ out*|*refused*|*"Connection closed"*)
    echo "WATCH_VERDICT: SSH_ERROR" ;;
  *"V3 COMPLETE"*) echo "WATCH_VERDICT: COMPLETE" ;;
  *"AUDIT FAILED"*|*"SMOKE FAILED"*) echo "WATCH_VERDICT: FAILED" ;;
  *) echo "WATCH_VERDICT: RUNNING" ;;
esac
