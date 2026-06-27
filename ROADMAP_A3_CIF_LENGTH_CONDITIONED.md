# Roadmap A3 CIF Length-Conditioned y Decisión de Arquitectura

## Estado Actual
- **Etapa 1: CERRADA (2026-06-27).** A3 official congelado como baseline deployable.
- **Etapa 2: EN PROGRESO, NO CIERRA (auditoría 2026-06-27).** Ver `reports/report_2026-06-27_etapa2_a3_audit.md` y `outputs/v126_temporal/diag_A3_etapa2_audit_20260627_175042.json`. Resultado por longitud (`pred_rescaled_to_pred_len`, re-eval standalone, 640 muestras): longitud `1` exact=0.886 (gate >=0.85 PASA), longitud `2` exact=0.843 (gate >=0.80 PASA), longitud `3+` exact=0.613 (gate >=0.65 **NO PASA**), `token_accuracy_when_count_correct` global=0.900 pero en bucket `3+`=0.879 (gate >=0.88 **NO PASA**, marginal). No se cumple el criterio de abandono de CIF (`3+` exact 0.613 > 0.60). Diagnóstico: el `length_head` predice bien el conteo en `3+` (`count_match_rate=0.873`), el fallo de `exact` es por acumulación de error de token-accuracy por posición (~0.88³≈0.68 ≈ exact observado), no por longitud mal calibrada. Casos peores: frases de 4+ tokens donde `length_head` subestima a la mitad (ej. "Darse cuenta de", "Goma de mascar").
- **Fase actual recomendada:** seguir en Etapa 2 — mejorar decoder/token modeling para `3+` (revisar longitudes >=4 dentro del bucket) antes de reintentar el audit. No avanzar a Etapa 3 (A4) todavía.
- **Baseline candidato deployable actual:** `pred_rescaled_to_pred_len`.
- **Checkpoint oficial A3:** `../outputs/v126_temporal/diag_A3_length_head_rerun_20260627_021153/checkpoint_best.pt` (epoch 11, seleccionado por `val_pred_rescaled_to_pred_len.(exact, top1)`).
- **Resultado oficial A3 (`val_pred_rescaled_to_pred_len`, epoch 11, métricas in-training):** `top1=0.874`, `top5=0.967`, `exact=0.820`, `pred_len_mae=0.0547`, `count_match_rate=0.949`, `boundary_mae_when_count_correct=0.637`. Los 6 gates de Etapa 1 pasan (`gates.json` del run). Nota: la re-evaluación standalone de Etapa 2 (mismo checkpoint, mismo split) da números ~1.5-2 puntos más bajos incluso en `teacher_alpha` — diferencia preexistente in-training vs. standalone, no invalida el cierre de Etapa 1 (certificado por `gates.json` del propio entrenamiento).
- **Bug encontrado y corregido durante el cierre de Etapa 1:** `checkpoint_best.pt` se seleccionaba por `val_pred_raw.top1` (no por el criterio oficial), y `gates.json` solo reflejaba el último epoch en vez del epoch del checkpoint seleccionado. Ambos se corrigieron en `scripts/train/train_temporal_v126.py` (selección por tupla `(exact, top1)` de `pred_rescaled_to_pred_len`; `gates.json` se escribe junto con `checkpoint_best.pt`, no al final del loop).
- **Objetivo inmediato:** mejorar decoder/token modeling para multi-token (`3+`, especialmente longitudes >=4) y re-correr la auditoría de Etapa 2 hasta que los 4 gates pasen.
- **Objetivo después de eso:** validar generalización antes de decidir si CIF sigue o si conviene migrar a una arquitectura con token queries.

## Summary
A3 demostró que el `length_head` resuelve casi todo el gap de A2: el checkpoint oficial (epoch 11 de `diag_A3_length_head_rerun_20260627_021153`) llega a `pred_rescaled_to_pred_len.top1=0.874`, `top5=0.967`, `exact=0.820`, `pred_len_mae=0.0547`. La siguiente línea principal debe ser **A3 length-conditioned CIF** como baseline deployable, no CIF raw puro. Solo se abandona CIF si A3/A4 falla en generalización, multi-token real, o si la calidad depende demasiado de rescale externo y no escala a secuencias más largas.

## Próximos Pasos Inmediatos
1. ~~Reejecutar o confirmar A3 con selección de `checkpoint_best.pt` por `val_pred_rescaled_to_pred_len.exact`, desempate por `top1`.~~ Hecho 2026-06-27: rerun `diag_A3_length_head_rerun_20260627_021153`, checkpoint oficial en epoch 11, 6/6 gates pasan.
2. ~~Extender `scripts/diagnostics/analyze_imitator_a2.py` para incluir el modo `pred_rescaled_to_pred_len` (+ corte `by_gloss` + `token_accuracy_when_count_correct/wrong` por bucket).~~ Hecho 2026-06-27.
3. ~~Generar los cortes de Etapa 2 (longitud `1`/`2`/`3+`, top errores por glosa/token) y validarlos contra los thresholds de Etapa 2.~~ Hecho 2026-06-27: ver `reports/report_2026-06-27_etapa2_a3_audit.md`. Resultado: **3 de 4 gates pasan, falla `longitud 3+ exact (0.613 < 0.65)`** y marginalmente `token_accuracy_when_count_correct` en bucket `3+` (0.879 < 0.88). No se cumple criterio de abandono de CIF.
4. **Siguiente paso real:** mejorar decoder/token modeling para secuencias `3+` (priorizar longitudes >=4, donde `length_head` colapsa el conteo a la mitad en los peores casos) y re-correr el audit de Etapa 2. Solo si esos 4 gates pasan, correr A4 scheduled pred alpha suave y luego LOSO (Etapa 3 y 4 siguen bloqueadas hasta entonces).

## Etapas Clave

### Etapa 1: Congelar A3 Como Baseline Correcto — CERRADA (2026-06-27)
- Rerun A3 con selección de `checkpoint_best.pt` por `val_pred_rescaled_to_pred_len.exact`, desempate por `top1`.
- Mantener reportes separados para:
  - `teacher_alpha`: techo oracle.
  - `pred_rescaled_to_target_len`: oracle de longitud.
  - `pred_rescaled_to_pred_len`: deployable A3 oficial.
  - `pred_raw`: diagnóstico CIF puro, no métrica principal.
- Acceptance (los 6 pasan en el checkpoint oficial, ver **Estado Actual**):
  - `pred_len_mae <= 0.10`
  - `count_match_rate >= 0.93`
  - `top1 >= 0.85`
  - `top5 >= 0.94`
  - `exact >= 0.79`
  - `boundary_mae_when_count_correct <= 1.0`
- Si falla: Realizar un studio de Optuna para encontrar hiperparametros optimos. si vuelve a fallar, revisar entrenamiento del `length_head` antes de avanzar. (No fue necesario: el modelo ya cumplía el bar; el bloqueo real era un bug de selección de checkpoint, ver **Estado Actual**.)

### Etapa 2: Auditoría De Errores A3
- Generar cortes por longitud target: `1`, `2`, `3+`.
- Generar top errores por glosa/token, con foco en multi-token: `víveres`, `aceptar`, plurales, tokens repetidos y palabras que comparten prefijo/subtoken.
- Reportar para cada corte:
  - `pred_len_mae`
  - `count_match_rate`
  - `token_accuracy_when_count_correct`
  - `token_accuracy_when_count_wrong`
  - `exact`
- Acceptance:
  - Longitud `1`: `exact >= 0.85`
  - Longitud `2`: `exact >= 0.80`
  - Longitud `3+`: `exact >= 0.65`
  - Cuando `count_match=true`, `token_accuracy >= 0.88`
- Si `3+` queda bajo pero longitud está bien: mejorar decoder/token modeling, no abandonar CIF todavía.

### Etapa 3: A4 Scheduled Pred Alpha Suave
- Partir del mejor A3.
- Entrenar con exposición gradual a alpha predicho:
  - `w_pred=0.02 -> 0.10`
  - no usar `0.50` en esta fase.
- Evaluar siempre el modo oficial `pred_rescaled_to_pred_len`.
- Stop temprano si:
  - `teacher_top1` cae más de 5 puntos contra A3.
  - `pred_len_mae` empeora por encima de `0.15`.
  - `count_match_rate` cae por debajo de `0.90`.
- Acceptance:
  - Igualar o mejorar A3: `exact >= 0.818` o `top1 >= 0.869`.
  - `pred_raw.top1 >= 0.76` sin degradar `pred_rescaled_to_pred_len`.
- Si A4 mejora raw pero empeora A3 oficial: conservar A3 y descartar A4.

### Etapa 4: Generalización y LOSO
- Ejecutar A3 oficial en signer-independent/LOSO.
- Reportar por signante:
  - `top1`, `top5`, `exact`
  - `pred_len_mae`
  - `count_match_rate`
  - worst glosses
- Acceptance mínima para seguir con CIF:
  - LOSO reportado para todos los signantes disponibles.
  - `pred_len_mae <= 0.25` promedio.
  - `count_match_rate >= 0.80` promedio.
  - `exact >= 0.45` promedio.
- Si LOSO cae fuerte pero teacher/rescaled oracle sigue alto: priorizar generalización visual.
- Si LOSO cae incluso con teacher/rescaled oracle: el problema no es CIF; revisar encoder visual/dataset split.

## Criterios Para Abandonar A3/CIF
Abandonar A3 length-conditioned CIF y pasar a decoder non-CIF con token queries si se cumple cualquiera:

- **Dependencia excesiva de oracle/longitud**: `pred_rescaled_to_pred_len` queda más de 10 puntos debajo de `pred_rescaled_to_target_len` después de dos runs A3.
- **Multi-token no recupera**: longitud `3+` queda con `exact < 0.60` aunque `pred_len_mae <= 0.10`.
- **Boundary no mejora donde count es correcto**: `boundary_mae_when_count_correct > 3.0` de forma estable y los tokens fallan por posición, no por longitud.
- **LOSO no viable**: `exact < 0.40` promedio o `count_match_rate < 0.75` promedio en signer-independent, mientras una arquitectura query-decoder smoke supera esos números.
- **A4 degrada repetidamente**: dos runs A4 empeoran `pred_rescaled_to_pred_len.exact` más de 3 puntos o bajan `teacher_top1` más de 5 puntos.

No abandonar CIF solo porque `pred_raw` sea peor. Desde ahora `pred_raw` es diagnóstico; el candidato deployable es `pred_rescaled_to_pred_len`.

## Próxima Arquitectura Si Se Abandona CIF
- Implementar baseline `visual encoder + length_head + learned token query slots + Transformer decoder`.
- Usar `predicted_len` para seleccionar/mascarar slots.
- Comparar contra A3 oficial, no contra A2 raw.
- Acceptance para reemplazar CIF:
  - `exact >= A3_exact + 0.03` en stratified.
  - `exact >= A3_LOSO_exact + 0.05` en LOSO.
  - Sin usar `target_lengths` en métricas deployables.

## Test Plan
- Unit tests:
  - `length_head` shape `[B, max_len_class + 1]`.
  - predicted length clamp `[1, max_len_class]`.
  - deployable inference path no recibe `target_lengths`.
  - checkpoint selection puede usar `pred_rescaled_to_pred_len`.
- Acceptance tests:
  - Reportes incluyen los cuatro modos: teacher, target-rescaled, pred-len-rescaled, raw.
  - A3 official supera `top1>=0.85`, `top5>=0.94`, `exact>=0.79`.
  - Error report incluye count-correct vs count-wrong.
- Paper gate:
  - No reclamar Q1-ready hasta tener A3 official + LOSO reportado.
  - A1/A2 oracle se presenta solo como techo, no como deployable.

## Assumptions
- La métrica principal de A3 será `val_pred_rescaled_to_pred_len`, porque no usa longitud target.
- `pred_raw` queda como métrica de salud/calibración CIF, no como criterio principal de aceptación.
- El objetivo inmediato es publicar un baseline deployable robusto, no cambiar arquitectura prematuramente.

## Nota Operativa Para Claude y Codex
- Tratar `pred_rescaled_to_pred_len` como la métrica oficial de A3.
- Tratar `pred_raw` solo como señal diagnóstica, no como criterio para abandonar CIF.
- No proponer cambio de arquitectura antes de cerrar:
  - ~~Etapa 1: A3 official.~~ Cerrada 2026-06-27.
  - Etapa 2: auditoría multi-token. **EN PROGRESO, no cierra (audit 2026-06-27, falla bucket `3+`). (siguiente paso: mejorar decoder/token modeling para 3+)**
  - Etapa 4: LOSO o signer-independent.
- Si se actualizan resultados, sobrescribir este documento manteniendo la sección **Estado Actual** y **Próximos Pasos Inmediatos** siempre al día.
