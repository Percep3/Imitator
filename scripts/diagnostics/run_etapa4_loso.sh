#!/usr/bin/env bash
# Etapa 4 LOSO: one A3 fold per held-out signer (dataset1 has 10 signers).
set -euo pipefail

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="../outputs/v126_temporal"
RESUME_CKPT="${OUT_ROOT}/diag_A3_etapa3_scheduled_pred_alpha_suave_v2_20260627_223500/checkpoint_best.pt"
PY=/home/nakato/miniconda3/envs/Sign-env/bin/python

echo "RUN_TAG=${RUN_TAG}"

for SIGNER in 1 2 3 4 5 6 7 8 9 10; do
  RUN_NAME="A4_loso_signer${SIGNER}_${RUN_TAG}"
  echo "=== fold start signer=${SIGNER} run=${RUN_NAME} $(date -Iseconds) ==="

  PYTHONPATH=. "$PY" scripts/train/train_temporal_v126.py \
    --heldout-signer "$SIGNER" \
    --phase learned_cif \
    --resume "$RESUME_CKPT" \
    --resume-weights-only \
    --epochs 15 \
    --min-clips 1 --max-clips 1 \
    --min-neutral-frames 0 --max-neutral-frames 8 \
    --alpha-schedule target_only \
    --diag-alpha-loss logit_l1 \
    --diag-freeze target_only_stage1 \
    --stgcn-lr-scale 0.1 \
    --prediction-alpha-mode pred_rescaled_to_pred_len \
    --seed 23 \
    --run-name "$RUN_NAME"

  PYTHONPATH=. "$PY" scripts/diagnostics/analyze_imitator_a2.py \
    --checkpoint "${OUT_ROOT}/diag_${RUN_NAME}/checkpoint_latest.pt" \
    --heldout-signer "$SIGNER" \
    --output "${OUT_ROOT}/diag_${RUN_NAME}_audit.json" \
    --seed 23

  echo "=== fold done signer=${SIGNER} run=${RUN_NAME} $(date -Iseconds) ==="
done

echo "ALL_FOLDS_DONE RUN_TAG=${RUN_TAG}"
