#!/usr/bin/env bash
set -euo pipefail

log=/shared/Code/Sign-AI/outputs/video_token_decoder/paper_run.log
mkdir -p "$(dirname "$log")"
echo "[$(date -Is)] restarting with live tmux output" | tee -a "$log"

/home/nakato/miniconda3/bin/conda run --no-capture-output -n Sign-env \
  bash /shared/Code/Sign-AI/Sign-chris/scripts/train/run_ar_paper_15h.sh \
  2>&1 | tee -a "$log"
