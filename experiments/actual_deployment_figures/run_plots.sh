#!/bin/bash
# Regenerate the real-deployment paper figures locally from pickles pulled (read-only) from the old VM,
# then copy the ones the paper includes into the paper repo's figures/ (dynamic reload by the paper's cron build).
# usage: ./run_plots.sh [dpsize]   (default actual_third_prototype, the paper's deployment)
#        PAPER_DIR=... to override the paper checkout (default ~/Documents/resilient_advertisements_paper)
cd "$(dirname "$0")"
PYTHONPATH=vm_src MPLBACKEND=Agg /Users/tomkoch/Documents/venv312/bin/python make_actual_deployment_plots.py --dpsize "${1:-actual_third_prototype}" || exit 1
PAPER_DIR="${PAPER_DIR:-$HOME/Documents/resilient_advertisements_paper}"
if [ -d "$PAPER_DIR/figures" ]; then
  n=0
  for f in figures/paper/*_actual_deployment.pdf figures/paper/uncertainty_path_measures_over_iterations.pdf; do
    [ -f "$f" ] && cp "$f" "$PAPER_DIR/figures/" && n=$((n+1))
  done
  echo "[run_plots] copied $n figures into $PAPER_DIR/figures/"
else
  echo "[run_plots] paper dir $PAPER_DIR not found; figures left in figures/paper/"
fi
