#!/usr/bin/env bash
set -euo pipefail

ROOT=/shared/Code/Sign-AI/Sign-chris
PY=/home/nakato/miniconda3/envs/Sign-env/bin/python
OUTPUT_ROOT=/shared/Code/Sign-AI/outputs/video_token_decoder

cd "$ROOT"
export PYTHONPATH=.

echo "[$(date -Is)] exploratory E3 completion on development folds 5-6"
for fold in 5 6; do
  source_checkpoint="/shared/Code/Sign-AI/outputs/loso_clean/diag_fold${fold}_promotion/checkpoint_best.pt"
  comparator="$OUTPUT_ROOT/fold${fold}_cif_comparator.json"
  if [[ ! -f "$source_checkpoint" || ! -f "$comparator" ]]; then
    echo "missing source checkpoint or CIF comparator for development fold ${fold}" >&2
    exit 1
  fi
  "$PY" scripts/train/train_video_token_decoder.py train --fold "$fold" --seed 23 \
    --encoder-pe --restricted-vocab --label-smoothing 0 --select edit --epochs 30 \
    --unfreeze-stgcn-epoch 10 --run-tag e3_unfreeze_stgcn
done

for fold in 5 6; do
  "$PY" scripts/train/train_video_token_decoder.py evaluate --fold "$fold" --seed 23 \
    --checkpoint "$OUTPUT_ROOT/fold${fold}_seed23/e3_unfreeze_stgcn/checkpoint_closed.pt" \
    --cif-comparator "$OUTPUT_ROOT/fold${fold}_cif_comparator.json"
done

"$PY" scripts/eval/summarize_imitator_paper.py
echo "[$(date -Is)] E3_DEV_COMPLETION_DONE"
