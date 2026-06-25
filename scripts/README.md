# Scripts

Entradas activas del repo actual:

- `train/train_imitator_dataset1_tokens.py`: prototipo Imitator real, dataset1 aislado -> token IDs Gemma -> prompt de correccion v125.
- `train/train_temporal_v126.py`: entrenamiento temporal sintetico v126/v126b.
- `train/train_isolated_staged.py`: runner configurable para clasificacion aislada v121-v124.
- `train/train_cls_v120.py` y `train/train_cls_v121.py`: compatibilidad para experimentos aislados previos.
- `train/train_ctc_v119.py` y `train/train_contrastive_v118.py`: runners historicos aun utiles para reproducir reportes v118-v119.
- `train/train.py`: baseline configurable con `config/training/train_config.toml` y `MSLM_EXPERIMENT_CONFIG=experiments/<familia>/<archivo>.toml`.

Datos y artefactos:

- `data/build_dataset1_h5.py`, `data/build_dataset1_v122_h5.py`: construccion de dataset1 aislado.
- `data/build_dataset2_h5.py`, `data/add_token_ids_h5.py`, `data/backfill_dataset2_metadata.py`, `data/build_dataset2_manifest.py`, `data/build_dataset2_split.py`, `data/rebuild_dataset2_alignment_targets_v125.py`: pipeline de dataset2/v125.
- `data/build_adjacency.py`: matriz estructural del esqueleto.
- `data/data_eda.py`: auditoria EDA reproducible de datos crudos.

Diagnosticos y auditorias:

- `diagnostics/diag_v118_train_val_gap.py`, `diagnostics/diag_text_discriminability.py`: diagnosticos v118.
- `diagnostics/diagnose_ctc_posteriors.py`, `diagnostics/diagnose_ctc_activations.py`: diagnosticos v119.
- `audits/run_anisotropy_gate_v125.py`, `audits/run_gemma_oracle_v125.py`, `audits/run_gemma_oracle_prompt_sweep_v125.py`, `audits/freeze_v125_artifacts.py`, `audits/smoke_gemma_inputs_embeds.py`: auditorias v125/Gemma.

Inferencia:

- `inference/gemma_correct_imitator_predictions.py`: toma `predictions.jsonl`
  de Imitator y escribe `gemma_corrected_text` usando GemmaBridge/v125.

Comandos canonicos:

```bash
PYTHONPATH=. python scripts/train/train_imitator_dataset1_tokens.py
PYTHONPATH=. python scripts/train/train_temporal_v126.py
PYTHONPATH=. python scripts/train/train_isolated_staged.py --config experiments/v121_v124_isolated_staged/cls_v121.toml
MSLM_EXPERIMENT_CONFIG=experiments/v119_ctc/ctc_v119.toml PYTHONPATH=. python scripts/train/train_ctc_v119.py
```

Para Level B, usar el mismo trainer temporal con secuencias sinteticas:

```bash
PYTHONPATH=. python scripts/train/train_temporal_v126.py \
  --phase teacher_forced \
  --min-clips 2 \
  --max-clips 8 \
  --prediction-alpha-mode teacher_alpha
```

Ambos caminos escriben `predictions.jsonl` con `target_token_ids`,
`predicted_token_ids`, texto decodificado y `gemma_prompt`. La clasificacion
64-way queda solo como baseline visual auxiliar, no como salida Imitator.

Para ejecutar la correccion final con Gemma:

```bash
PYTHONPATH=. python scripts/inference/gemma_correct_imitator_predictions.py \
  ../outputs/v126_temporal/imitator_dataset1_tokens/predictions.jsonl
```

Los scripts legacy de finetuning Llama 2025, Optuna study antiguo, profiling hardcodeado y gRPC/DDP experimental fueron retirados porque ya no estaban referenciados por el pipeline actual y varios dependian de APIs/rutas obsoletas.
