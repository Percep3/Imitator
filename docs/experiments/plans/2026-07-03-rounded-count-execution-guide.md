# Guía de ejecución — modo `pred_rescaled_to_rounded_count` (H1)

Plan completo con criterios de aceptación y ramas de decisión:
`/home/nakato/.claude/plans/genere-el-plan-detallado-indexed-quokka.md`.
Fase 0 ya ejecutada: `reports/report_2026-07-03_fase0_rounded_count_prediction.md`
(resultado PARCIAL: redondeo puro ~0.53 count_match esperado; con bias +0.2 → ~0.66).

**Qué se implementó (sin ejecutar nada más que la Fase 0):**

| Archivo | Cambio |
|---|---|
| `src/mslm/models/temporal_sign_prompt.py` | Helper `rescale_alphas_to_rounded_count(alphas, *, max_len, min_len=1, bias=0.0)` — reutiliza `rescale_alphas_to_predicted_lengths`/`clamp_predicted_lengths` |
| `scripts/train/train_temporal_v126.py` | Modo `pred_rescaled_to_rounded_count` en `evaluate_alpha_mode` (kwarg `rounded_count_bias`) y en `write_prediction_report`; `rescale_length_mae` también para el modo nuevo |
| `scripts/diagnostics/analyze_imitator_a2.py` | Modo agregado al loop de audit + flag CLI `--rounded-count-bias` (default 0.0, queda registrado en el JSON de salida) |
| `tests/models/test_v126_synthetic_temporal.py` | 2 tests nuevos del helper (round/clamp y efecto del bias) |
| `scripts/diagnostics/predict_rounded_count_from_audits.py` | Script de la Fase 0 (reproducible, solo lee JSONs) |

Todo pasa `py_compile`. **Los tests NO se corrieron** (instrucción: solo Fase 0).

---

## Paso 1 — Tests (obligatorio antes de nada)

```bash
cd /shared/Code/Sign-AI/Sign-chris
PYTHONPATH=. /home/nakato/miniconda3/envs/Sign-env/bin/python -m pytest \
  tests/models/test_v126_synthetic_temporal.py -q
```

Esperado: todos pasan (25 previos + 2 nuevos). Si falla algo del helper nuevo,
no seguir.

## Paso 2 — Sanity signer-dependent (gate de Fase 1)

Contra el checkpoint oficial A3 (signer-dependent, donde `pred_len_mae`=0.053):

```bash
PYTHONPATH=. /home/nakato/miniconda3/envs/Sign-env/bin/python \
  scripts/diagnostics/analyze_imitator_a2.py \
  --checkpoint "../outputs/v126_temporal/diag_A3_etapa3_scheduled_pred_alpha_suave_v2_20260627_223500/checkpoint_best.pt" \
  --output ../outputs/v126_temporal/sanity_rounded_count_official_a3.json
```

**Gate:** en el `.md` de salida, `exact` de `pred_rescaled_to_rounded_count` ≥
`pred_rescaled_to_pred_len` − 0.03. Si no cumple → bug de implementación,
debuggear antes de tocar folds (corte: 1 día).

## Paso 3 — Fase 2: re-evaluación de los 6 folds LOSO limpios (b=0)

```bash
for N in 1 2 3 4 5 6; do
  PYTHONPATH=. /home/nakato/miniconda3/envs/Sign-env/bin/python \
    scripts/diagnostics/analyze_imitator_a2.py \
    --checkpoint /shared/Code/Sign-AI/outputs/loso_clean/diag_fold${N}_promotion/checkpoint_best.pt \
    --heldout-signer ${N} \
    --output /shared/Code/Sign-AI/outputs/loso_clean/fold${N}_test_eval_rounded.json
done
```

**Control de reproducibilidad:** los modos viejos (`pred_raw`,
`pred_rescaled_to_pred_len`, oracle) deben reproducir los números de
`fold{N}_test_eval.json` originales. Si no reproducen, hay problema de carga
de checkpoint y los números nuevos no valen.

## Paso 4 — Bias calibrado sin contaminación (adelanto de B2, justificado por Fase 0)

Por fold: ajustar `b` contra un signante de TRAIN del fold (usar el
`inner_val_signer = N%10+1` del manifiesto, que nunca es el held-out), luego
aplicar ese `b` congelado al held-out real:

```bash
# ejemplo fold 1: inner val = signer 2
PYTHONPATH=. /home/nakato/miniconda3/envs/Sign-env/bin/python \
  scripts/diagnostics/analyze_imitator_a2.py \
  --checkpoint /shared/Code/Sign-AI/outputs/loso_clean/diag_fold1_promotion/checkpoint_best.pt \
  --heldout-signer 2 \
  --output /shared/Code/Sign-AI/outputs/loso_clean/fold1_biasfit_inner.json
# leer hist_quantity_minus_target_len de ese JSON con
# scripts/diagnostics/predict_rounded_count_from_audits.py --audits <ese json>
# elegir b* y re-correr el paso 3 con --rounded-count-bias <b*>
```

(Fase 0 sugiere que b* ≈ 0.2 será casi el mismo en todos los folds.)

## Paso 5 — Decisión (criterios del plan, media de 6 folds, modo rounded)

| Resultado | Umbral | Siguiente acción |
|---|---|---|
| ÉXITO | `exact ≥ 0.40` y `count_match ≥ 0.75` | Rama A: barrido de regla (máx. 4 variantes, 1 día) → folds 7–10 (~44 h GPU) → demo e2e |
| PARCIAL | `exact` 0.25–0.40 | Rama B2: calibración afín `a·q+b` fit en train signers (corte: 2 días, sin reentrenar) |
| FALLO | `exact < 0.25` con count_match alto | Rama B3: piloto normalización de esqueleto, 3 folds (~33 h GPU, gate: +10pp oracle-exact) |
| FALLO total | B2 y B3 fallan | B4: punto de abandono CIF — decisión explícita del usuario |

Referencia de techo: `exact` oracle-length ≈ 0.59–0.60 en estos mismos folds.

## Paso 6 — Cierre documental (solo en rama A)

- Reporte `reports/report_YYYY-MM-DD_rounded_count_loso.md` con tabla 10 folds
  todos los modos vs el report del 2026-07-03.
- Actualizar `ROADMAP_A3_CIF_LENGTH_CONDITIONED.md` (Etapa 4 re-cerrada;
  `length_head` documentado como muerto para inferencia).
- Hardening: loguear la métrica deployable en la consola de entrenamiento
  (`train_temporal_v126.py` ~línea 1215 imprime `val_pred_raw['mae_len']` — por
  eso el colapso del length_head fue invisible durante el entrenamiento).
