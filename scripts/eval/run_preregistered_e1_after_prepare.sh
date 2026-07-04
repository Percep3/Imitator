#!/usr/bin/env bash
set -euo pipefail

ROOT=/shared/Code/Sign-AI/Sign-chris
PY=/home/nakato/miniconda3/envs/Sign-env/bin/python
OUTPUT_ROOT=/shared/Code/Sign-AI/outputs/video_token_decoder
LOSO_ROOT=/shared/Code/Sign-AI/outputs/loso_clean

cd "$ROOT"
export PYTHONPATH=.

for fold in 7 8; do
  checkpoint="$LOSO_ROOT/diag_fold${fold}_promotion/checkpoint_best.pt"
  done_record="$LOSO_ROOT/diag_fold${fold}_promotion/stage_done.json"
  if [[ ! -f "$checkpoint" || ! -f "$done_record" ]]; then
    echo "missing prepared fold ${fold} checkpoint or attestation" >&2
    exit 1
  fi
  if [[ -e "$LOSO_ROOT/fold${fold}_test_eval.json" ]]; then
    echo "guardrail violation: outer test eval already exists for fold ${fold}" >&2
    exit 1
  fi
done

echo "[$(date -Is)] paired CIF comparators (outputs remain blinded)"
for fold in 7 8; do
  "$PY" scripts/train/train_video_token_decoder.py cif-comparator --fold "$fold" --seed 23 \
    --output "$OUTPUT_ROOT/fold${fold}_cif_comparator.json"
done

echo "[$(date -Is)] phase3 train and close both E1 checkpoints before outer evaluation"
for fold in 7 8; do
  "$PY" scripts/train/train_video_token_decoder.py train --fold "$fold" --seed 23 \
    --encoder-pe --restricted-vocab --label-smoothing 0 --select edit --epochs 30 \
    --run-tag e1_pe_vocab121
done

echo "[$(date -Is)] phase3 outer evaluation after both checkpoints are closed"
for fold in 7 8; do
  "$PY" scripts/train/train_video_token_decoder.py evaluate --fold "$fold" --seed 23 \
    --checkpoint "$OUTPUT_ROOT/fold${fold}_seed23/e1_pe_vocab121/checkpoint_closed.pt" \
    --cif-comparator "$OUTPUT_ROOT/fold${fold}_cif_comparator.json"
  "$PY" scripts/eval/robustness_video_token_decoder.py --fold "$fold" --seed 23 \
    --checkpoint "$OUTPUT_ROOT/fold${fold}_seed23/e1_pe_vocab121/checkpoint_closed.pt" \
    --cif-comparator "$OUTPUT_ROOT/fold${fold}_cif_comparator.json" \
    --conditions clean segment_permutation
done

"$PY" scripts/eval/summarize_imitator_paper.py
echo "[$(date -Is)] PREREGISTERED_E1_CONFIRMATION_DONE"
