#!/bin/bash
# THE dashboard update path (Tom 2026-08-25: "this should all just be
# tied to one 'update dashboard' script call that runs every 30 seconds").
# One shot = harvest live runs -> timing plots -> site build -> push to
# BOTH dash hosts. --loop N repeats forever every N seconds (flock'd so
# cycles never overlap). Everything the old manual chain did, in order,
# in one place.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY=/Users/tomkoch/Documents/venv312/bin/python
LOCK=/tmp/sculptor_update_dashboard.lockdir
LOG=$HOME/sculptor_dashboard/update_dashboard.log
DASH_INSTANCES="i-09a6ff2823b0bb304 i-0428c395787bc3ca0"

one_shot() {
  cd "$REPO"
  T0=$(date +%s)
  $PY -m cluster.harvest_all --quiet >/dev/null 2>&1
  # timing plots for every non-terminal run (manifest without a verdict)
  for RID in $($PY - <<'PYEOF'
import glob, json, os
for fn in glob.glob('cache/cluster_runs/*/manifest.json'):
    try:
        m = json.load(open(fn))
    except Exception:
        continue
    if not m.get('verdict') and not m.get('terminal'):
        print(m.get('run_id') or os.path.basename(os.path.dirname(fn)))
PYEOF
  ); do
    $PY -m dashboard.plot_cluster_timing "$RID" >/dev/null 2>&1
  done
  $PY -m dashboard.generate >/dev/null 2>&1
  for INST in $DASH_INSTANCES; do
    IP=$($PY -m cluster.vmctl ip "$INST" 2>/dev/null | tail -1 | tr -d ' ')
    if [[ "$IP" =~ ^[0-9.]+$ ]]; then
      rsync -azL --delete --exclude eods25_tail.txt --exclude worker_flame.svg \
        --exclude eods25_convergence.pdf --timeout=120 \
        -e "ssh -i $HOME/.ssh/ray-autoscaler_us-east-1.pem -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes" \
        "$REPO/dashboard_site/" "ubuntu@$IP:/var/www/dash/" >/dev/null 2>&1
      echo "$(date -u +%H:%M:%SZ) inst=$INST ip=$IP rc=$? took=$(( $(date +%s) - T0 ))s" >> "$LOG"
    else
      echo "$(date -u +%H:%M:%SZ) inst=$INST no-ip" >> "$LOG"
    fi
  done
  tail -400 "$LOG" > "$LOG.t" 2>/dev/null && mv "$LOG.t" "$LOG"
}

run_locked() {
  # mkdir is atomic and portable (macOS ships no flock); a stale lock
  # older than 30 min is reclaimed
  if mkdir "$LOCK" 2>/dev/null; then
    trap 'rmdir "$LOCK" 2>/dev/null' EXIT
    one_shot
    rmdir "$LOCK" 2>/dev/null; trap - EXIT
  else
    if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +30 2>/dev/null)" ]; then
      rmdir "$LOCK" 2>/dev/null
    fi
  fi
}

if [ "${1:-}" = "--loop" ]; then
  N="${2:-30}"
  while true; do
    run_locked
    sleep "$N"
  done
else
  run_locked
fi
