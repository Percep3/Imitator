#!/usr/bin/env bash
set -euo pipefail

ROOT=/shared/Code/Sign-AI/Sign-chris
PY=/home/nakato/miniconda3/envs/Sign-env/bin/python
OUTPUT_ROOT=/shared/Code/Sign-AI/outputs/video_token_decoder

cd "$ROOT"
export PYTHONPATH=.

for fold in 7 8; do
  checkpoint="$OUTPUT_ROOT/fold${fold}_seed23/e1_pe_vocab121/checkpoint_closed.pt"
  robustness="$OUTPUT_ROOT/fold${fold}_seed23/e1_pe_vocab121/outer_test_robustness.json"
  if [[ ! -f "$checkpoint" || ! -f "$robustness" ]]; then
    echo "confirmation artifacts incomplete for fold ${fold}; post-analysis aborted" >&2
    exit 1
  fi
done

echo "[$(date -Is)] phase4 clip-balanced per-frame identity probes"
for fold in 7 8; do
  "$PY" scripts/eval/probe_frame_identity.py --fold "$fold" --seed 23 \
    --checkpoint "$OUTPUT_ROOT/fold${fold}_seed23/e1_pe_vocab121/checkpoint_closed.pt"
done

echo "[$(date -Is)] phase4 master JSON and paper tables"
"$PY" scripts/eval/summarize_imitator_paper.py
echo "[$(date -Is)] E1_POST_ANALYSIS_DONE"
