#!/bin/bash
# Runbook for the actual-10 ablation on the AWS head node. Stages are
# idempotent; run pieces by hand or the whole thing. Driven from the Mac.
#
#   ./experiments/ablation/vm_runbook.sh start|sync|prebuild|smoke|full|pull|stop
#
# Uses the session-10 head (EBS has repo + venv + 4.5GB latency CSV).
set -euo pipefail

HEAD_ID=i-0428c395787bc3ca0
KEY=~/.ssh/ray-autoscaler_us-east-1.pem
AWS=~/Documents/venv312/bin/aws
REMOTE_REPO=/home/ubuntu/sparse_advertisements_code
REMOTE_PY=/home/ubuntu/venv/bin/python
N_SEEDS=${N_SEEDS:-12}
MAX_ITER=${MAX_ITER:-200}
JOBS=${JOBS:-6}
DPSIZE=${DPSIZE:-actual-10}

ip() { $AWS ec2 describe-instances --instance-ids $HEAD_ID \
      --query 'Reservations[].Instances[].PublicIpAddress' --output text; }
SSH() { ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@$(ip) "$@"; }

case "${1:-}" in
start)
  $AWS ec2 start-instances --instance-ids $HEAD_ID
  $AWS ec2 wait instance-running --instance-ids $HEAD_ID
  echo "head running at $(ip); waiting for SSH..."
  until SSH true 2>/dev/null; do sleep 10; done
  echo "SSH up"
  ;;
sync)
  rsync -avz -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      experiments/ubuntu@$(ip):$REMOTE_REPO/experiments/ \
      --include='ablation/***' --include='__init__.py' --exclude='*' || \
  rsync -avz -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      experiments/ablation ubuntu@$(ip):$REMOTE_REPO/experiments/
  SSH "ls $REMOTE_REPO/experiments/ablation/"
  ;;
prebuild)
  # Build deployment caches sequentially (avoids N concurrent 4.5GB CSV parses)
  SSH "cd $REMOTE_REPO && for s in \$(seq 1 $N_SEEDS); do \
      nohup $REMOTE_PY -c \"import sys; sys.path.insert(0,'.'); \
from experiments.ablation.common import Problem; \
p=Problem(\$s, dpsize='$DPSIZE'); \
print('seed', \$s, 'n_ug', p.n_ug, 'n_popp', p.n_popp, 'n_pref', p.n_prefixes)\" ; done" \
      2>&1 | grep -E "^seed|Error|Traceback" || true
  ;;
smoke)
  SSH "cd $REMOTE_REPO && $REMOTE_PY -m experiments.ablation.run_ablation \
      --dpsize $DPSIZE --n-seeds 1 --max-iter 20 --jobs 1 \
      --out-dir cache/ablation/${DPSIZE}_smoke 2>&1 | tail -12"
  ;;
full)
  SSH "cd $REMOTE_REPO && mkdir -p logs && nohup $REMOTE_PY -m experiments.ablation.run_ablation \
      --dpsize $DPSIZE --n-seeds $N_SEEDS --max-iter $MAX_ITER --jobs $JOBS \
      --out-dir cache/ablation/${DPSIZE}_full > logs/ablation_${DPSIZE}.log 2>&1 & \
      echo remote pid \$!"
  ;;
status)
  SSH "tail -5 $REMOTE_REPO/logs/ablation_${DPSIZE}.log; \
       ls $REMOTE_REPO/cache/ablation/${DPSIZE}_full/ 2>/dev/null | wc -l"
  ;;
pull)
  mkdir -p cache/ablation/${DPSIZE}_full
  rsync -avz -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      ubuntu@$(ip):$REMOTE_REPO/cache/ablation/${DPSIZE}_full/ cache/ablation/${DPSIZE}_full/
  ;;
stop)
  $AWS ec2 stop-instances --instance-ids $HEAD_ID
  echo "head stopping"
  ;;
*) echo "usage: $0 start|sync|prebuild|smoke|full|status|pull|stop"; exit 1;;
esac
