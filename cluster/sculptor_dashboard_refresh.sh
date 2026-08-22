#!/usr/bin/env bash
# Pull SCULPTOR cluster logs, ingest timing stats into the local SQLite DB, and
# regenerate the dashboard plots + index.html. Intended for cron (~every 10 min)
# while a sweep runs on AWS. All state lives under ~/sculptor_dashboard/.
#
# Cron entry (added by session 10):
#   */10 * * * * /Users/tomkoch/Documents/sparse_advertisements_code/cluster/sculptor_dashboard_refresh.sh
#
# View: open ~/sculptor_dashboard/index.html (auto-refreshes every 60s).
set -uo pipefail
PY="$HOME/Documents/venv312/bin/python"
REPO="$HOME/Documents/sparse_advertisements_code"
OUT="$HOME/sculptor_dashboard"
mkdir -p "$OUT"
echo "===== refresh $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
"$PY" "$REPO/cluster/cluster_dashboard.py"
