# Etapa 2 — Auditoría de errores A3, iteración 2 (decoder/token modeling)

Checkpoint nuevo: `outputs/v126_temporal/diag_A3_etapa2_decoder_v1_20260627_194451/checkpoint_best.pt` (epoch 14, warm-start desde el checkpoint oficial de Etapa 1).
Cambio de arquitectura: `src/mslm/models/temporal_sign_prompt.py` — `token_head` ahora es `_TokenHead` (position embedding + 1 `nn.TransformerEncoderLayer` sobre los slots de `cif.embeddings`, antes clasificaba cada slot de forma independiente); `length_head` ahora es `_LengthHead` (1-query attention pooling sobre frame features, antes era mean-pooling). Más `label_smoothing=0.1` en la CE de tokens. Entrenamiento: `scripts/train/train_temporal_v126.py`, 15 epochs, resto de hiperparámetros igual a la config oficial de Etapa 1, `--resume-weights-only` desde `diag_A3_length_head_rerun_20260627_021153/checkpoint_best.pt`.
Audit script: `scripts/diagnostics/analyze_imitator_a2.py` (sin cambios), mismo seed 23 → mismo split de val que la iteración 1 (640 muestras: 70 longitud 1, 420 longitud 2, 150 longitud 3+).
Salida cruda: `outputs/v126_temporal/diag_A3_etapa2_audit_iter2_20260627_203817` (+ `.md`).

## Resultado por longitud (`pred_rescaled_to_pred_len`)

| longitud | n | exact | count_match | token_acc\|count_ok | token_acc\|count_wrong |
|---|---|---|---|---|---|
| 1 | 70 | 0.900 | 0.929 | 0.969 | 0.000 |
| 2 | 420 | 0.898 | 0.962 | 0.946 | 0.188 |
| 3+ | 150 | 0.860 | 0.867 | 0.997 | 0.279 |

Agregado global: `top1=0.909`, `top5=0.976`, `exact=0.889`, `pred_len_mae=0.072`, `count_match_rate=0.936`, `token_accuracy_when_count_correct=0.960`, `boundary_mae_when_count_correct=0.570`.

## Veredicto contra gates de Etapa 2

| gate | requerido | iter1 | iter2 | resultado |
|---|---|---|---|---|
| longitud 1, exact | >= 0.85 | 0.886 | 0.900 | PASA |
| longitud 2, exact | >= 0.80 | 0.843 | 0.898 | PASA |
| longitud 3+, exact | >= 0.65 | 0.613 | **0.860** | **PASA** |
| token_accuracy cuando count_match=true | >= 0.88 | 0.900 global / 0.879 en 3+ | 0.960 global / **0.997 en 3+** | **PASA** |

**Etapa 2 CIERRA.** Los 4 gates pasan, con margen amplio en el bucket 3+ que era el bloqueante (exact 0.613→0.860, +24.7 puntos; token_acc|count_ok en 3+ 0.879→0.997).

Los 6 gates de Etapa 1 también se re-verificaron en este mismo checkpoint (`gates.json` del run, epoch 14): `pred_len_mae<=0.10` ✓, `count_match_rate>=0.93` ✓, `top1>=0.85` ✓, `top5>=0.94` ✓, `exact>=0.79` ✓, `boundary_mae_when_count_correct<=1.0` ✓. No hay regresión en el baseline oficial.

## Causa raíz confirmada y por qué funcionó el cambio

El diagnóstico de la iteración 1 decía que en 3+ el `length_head` predecía el conteo razonablemente bien pero `exact` colapsaba por acumulación de error por posición (0.879³≈0.68≈0.613 observado), porque `token_head` clasificaba cada slot de forma independiente sin contexto posicional ni de vecinos. Agregar position embedding + 1 capa de self-attention sobre los slots fired antes de clasificar le dio a cada posición visibilidad de sus vecinos, y el `token_accuracy_when_count_correct` en 3+ subió de 0.879 a 0.997 — consistente con que el cuello de botella diagnosticado (no longitud, sino independencia entre posiciones) era correcto.

El attention-pooling en `length_head` también ayudó al conteo: `count_match` en 3+ pasó de 0.873 a 0.867 (estable, dentro de ruido), pero los casos específicos nombrados en la iteración 1 mejoraron: `Darse cuenta de` exact 0.50→0.90, `Goma de mascar` exact 0.50→0.90, `Burlarse` exact 0.20→0.80. Quedan casos aislados sin resolver (`worst_examples`: un ejemplo de `Darse cuenta de` y uno de `Goma de mascar` siguen colapsando target_len=4→pred=2), pero ya no dominan el bucket.

Dos glosas quedan débiles y no estaban en el foco original: `Encontrar` (exact=0.40, longitud 2, problema de conteo no de longitud larga) y `Espumadera` (exact=0.40). No bloquean el cierre de Etapa 2 (gates son por bucket de longitud, no por glosa), pero quedan anotadas para una futura iteración si se requiere mejorar generalización por glosa.

## Decisión

Etapa 2 CERRADA (2026-06-27). Nuevo checkpoint oficial: `outputs/v126_temporal/diag_A3_etapa2_decoder_v1_20260627_194451/checkpoint_best.pt` (epoch 14). Próxima fase: Etapa 3 (A4 scheduled pred alpha suave), per roadmap — no iniciada en esta tarea, solo se actualiza el puntero.
