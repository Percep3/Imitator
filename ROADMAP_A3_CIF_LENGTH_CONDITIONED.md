# Roadmap A3 CIF Length-Conditioned y Decisión de Arquitectura

## Estado Actual
- **Etapa 1: CERRADA (2026-06-27).** A3 official congelado como baseline deployable.
- **Etapa 2: CERRADA (2026-06-27, iteración 2; conclusión causal corregida 2026-06-28).** Ver `reports/report_2026-06-27_etapa2_a3_audit_iter2.md` (resultado del run original) y `reports/report_2026-06-28_etapa2_ablation_causal.md` (ablación factorial que corrige la atribución causal). Cambio aplicado en el run original: `token_head` pasó de Linear por slot a incluir position embedding + 1 `nn.TransformerEncoderLayer`; `length_head` pasó de mean-pooling a 1-query attention pooling; + `label_smoothing=0.1` (ver `src/mslm/models/temporal_sign_prompt.py`, clases `_TokenHead`/`_LengthHead`). Resultado: 4/4 gates de Etapa 2 pasan (`longitud 3+ exact` 0.613→0.860, `token_accuracy_when_count_correct` en `3+` 0.879→0.997). **Conclusión causal corregida tras la ablación factorial (`2×2×2` × 3 semillas, 24 runs, `artifacts/v126_closeout/ablation_summary.json`):** solo `token_head=contextual` es causa confirmada del salto en `3+ exact` (positivo en las 3 semillas, marginal y al retirarlo del modelo completo); `length_head=attention` no tiene efecto causal confirmado en ninguna métrica (mixto, a veces negativo); `label_smoothing=0.1` solo tiene efecto causal confirmado (pequeño) en `exact` global, no en `3+`. El checkpoint oficial no cambia — no hay evidencia de que retroceder `length_head`/`label_smoothing` mejore nada. Detalle completo en el reporte de ablación.
- **Etapa 3: CERRADA (2026-06-27) con resultado NEGATIVO para la hipótesis de scheduled pred-alpha mixing.** Ver `reports/report_2026-06-27_etapa3_a4_scheduled_alpha.md`. Se corrió dos veces: un primer intento (`--mix-start-epoch 0`) quedó invalidado porque para `epoch<3` el loss excluye `token_loss`/`emb_loss` (`train_temporal_v126.py:978-983`), los únicos términos que dependen de `w_pred` — así que la mezcla no tuvo efecto ahí. Un re-run corregido (`--mix-start-epoch 5`, después de estabilizado el shock de desbloqueo de epoch 3) confirmó el diagnóstico: epochs 0-4 dieron números idénticos al intento 1, y ningún epoch de la ventana 5-14 (donde `w_pred` sí afecta el gradiente) superó al checkpoint pre-mezcla en la métrica oficial. Con `w_pred=0.10` sostenido (epoch 14) el oficial queda igual a A3 (`exact=0.8891` vs `0.889`) pero `count_match_rate` cae a `0.9141`, rompiendo el gate de Etapa 1 (`>=0.93`). Conclusión: **la mezcla no mejora el modo deployable** en este rango de `w_pred`; sí mejora mucho `pred_raw` (diagnóstico, `count_match_rate` 0.458→0.722, `top1` 0.735→0.857) — queda anotado como pista no bloqueante para una futura iteración si se necesita depender menos del rescale a `pred_len`, pero no es prioridad ahora.
- **Checkpoint promovido a oficial (no por el mecanismo de Etapa 3, ver nota abajo):** el checkpoint pre-mezcla (epoch 0 de ambos runs, idénticos) es equivalente a 1 epoch extra de fine-tuning de `cif.alpha`/`length_head` sobre Etapa 2 con el mismo régimen `--alpha-schedule target_only` — una mejora real y reproducible, no atribuible a `w_pred`.
- **Etapa 4: CERRADA (2026-06-28) con resultado POSITIVO.** Ver `reports/report_2026-06-28_etapa4_loso.md` y `../outputs/v126_temporal/diag_A4_etapa4_loso_summary_20260627_234427.md`. 10 folds leave-one-signer-out (`dataset1` tiene `signer_id` 1..10, 320 clips cada uno — no 0..9 ni 5138 como se estimó al explorar el código), cada uno 15 epochs partiendo del checkpoint oficial de Etapa 3 (`--resume-weights-only`), auditado con `analyze_imitator_a2.py --heldout-signer N` (flag agregado para esta etapa) contra `checkpoint_latest.pt` (no `checkpoint_best.pt`, para no seleccionar el checkpoint espiando la métrica del propio signante held-out). Promedio de los 10 folds en `pred_rescaled_to_pred_len`: `exact=0.9053` (gate `>=0.45` **PASA**), `count_match_rate=0.9566` (gate `>=0.80` **PASA**), `pred_len_mae=0.0528` (gate `<=0.25` **PASA**), `top1=0.9286`, `top5=0.9811`. Los 10 signantes individuales pasan también (`exact` entre 0.8406 y 0.9531), sin outliers; ninguna señal de abandono se activó. Hallazgo lateral no bloqueante: las glosas `Opaco`, `Enemigo`, `Verde`, `Caramelo`, `Espumadera`, `Rojo` se repiten como peor entrada en 4-6 de los 10 folds — debilidad del modelo en esas glosas puntuales, no de un signante particular. **Decisión: CIF sigue, no migrar a arquitectura con token queries** — la pregunta abierta del roadmap queda resuelta por evidencia LOSO, no por default.
- **Fase actual recomendada:** Etapa 1-4 cerradas. No queda iteración interna de arquitectura pendiente en este roadmap; el siguiente paso es de producto/paper (declarar A3 length-conditioned CIF baseline final reportable) o, opcionalmente, perseguir el hallazgo lateral de glosas repetidas como pulido adicional.
- **Baseline candidato deployable actual:** `pred_rescaled_to_pred_len`.
- **Checkpoint oficial:** `../outputs/v126_temporal/diag_A3_etapa3_scheduled_pred_alpha_suave_v2_20260627_223500/checkpoint_best.pt` (epoch 0).
- **Resultado oficial actualizado (`val_pred_rescaled_to_pred_len`, re-audit standalone, 640 muestras):** `top1=0.9165`, `top5=0.9782`, `exact=0.9000`, `pred_len_mae=0.0641`, `count_match_rate=0.9422`, `token_accuracy_when_count_correct=0.9601`, `boundary_mae_when_count_correct=0.5742`. Checkpoint anterior (epoch 14, `diag_A3_etapa2_decoder_v1_20260627_194451`) queda como referencia histórica de Etapa 2.
- **Bug encontrado y corregido durante el cierre de Etapa 1:** `checkpoint_best.pt` se seleccionaba por `val_pred_raw.top1` (no por el criterio oficial), y `gates.json` solo reflejaba el último epoch en vez del epoch del checkpoint seleccionado. Ambos se corrigieron en `scripts/train/train_temporal_v126.py` (selección por tupla `(exact, top1)` de `pred_rescaled_to_pred_len`; `gates.json` se escribe junto con `checkpoint_best.pt`, no al final del loop).

## Summary
La ablación factorial de Etapa 2 demostró que `token_head=contextual` es la causa confirmada del salto frente a A2 en secuencias de tokens `3+`; `length_head=attention` no tiene un efecto causal confirmado en esa mejora y `label_smoothing=0.1` solo aporta una mejora pequeña y consistente en `exact` global. El checkpoint oficial mantiene el paquete completo (`token_head=contextual` + `length_head=attention` + `label_smoothing=0.1`) porque no hay evidencia de que retirar los otros cambios mejore el resultado global. La línea principal sigue siendo **A3 length-conditioned CIF** como baseline deployable, no CIF raw puro. Solo se abandona CIF si A3/A4 falla en generalización, multi-token real, o si la calidad depende demasiado de rescale externo y no escala a secuencias más largas.

## Próximos Pasos Inmediatos
1. ~~Reejecutar o confirmar A3 con selección de `checkpoint_best.pt` por `val_pred_rescaled_to_pred_len.exact`, desempate por `top1`.~~ Hecho 2026-06-27: rerun `diag_A3_length_head_rerun_20260627_021153`, checkpoint oficial en epoch 11, 6/6 gates pasan.
2. ~~Extender `scripts/diagnostics/analyze_imitator_a2.py` para incluir el modo `pred_rescaled_to_pred_len` (+ corte `by_gloss` + `token_accuracy_when_count_correct/wrong` por bucket).~~ Hecho 2026-06-27.
3. ~~Generar los cortes de Etapa 2 (longitud `1`/`2`/`3+`, top errores por glosa/token) y validarlos contra los thresholds de Etapa 2.~~ Hecho 2026-06-27: ver `reports/report_2026-06-27_etapa2_a3_audit.md`. Resultado iteración 1: **3 de 4 gates pasan, falla `longitud 3+ exact (0.613 < 0.65)`** y marginalmente `token_accuracy_when_count_correct` en bucket `3+` (0.879 < 0.88). No se cumple criterio de abandono de CIF.
4. ~~Mejorar decoder/token modeling para secuencias `3+` (positional context en `token_head` + attention pooling en `length_head`) y re-correr el audit de Etapa 2.~~ Hecho 2026-06-27: ver `reports/report_2026-06-27_etapa2_a3_audit_iter2.md`. Resultado: **4/4 gates pasan** (`longitud 3+ exact=0.860`, `token_accuracy_when_count_correct` en `3+`=0.997). Etapa 2 CERRADA.
5. ~~Correr A4 scheduled pred alpha suave (Etapa 3) partiendo del checkpoint A3.~~ Hecho 2026-06-27: primer intento (`--mix-start-epoch 0`) quedó invalidado por un confound (`w_pred` no afecta el gradiente en epochs 0-2); ver paso 6 para el re-run corregido.
6. ~~Re-run corregido de Etapa 3 (`--mix-start-epoch 5`) y auditoría.~~ Hecho 2026-06-27: ver `reports/report_2026-06-27_etapa3_a4_scheduled_alpha.md`. Confirmó el confound del intento 1 (epochs 0-4 idénticos) y que ningún epoch con `w_pred` activo supera al checkpoint pre-mezcla en el oficial; `w_pred=0.10` sostenido rompe el gate `count_match_rate>=0.93` de Etapa 1. **Etapa 3 CERRADA con resultado negativo para el mecanismo**, pero se promueve el checkpoint pre-mezcla (mejora real por entrenamiento adicional, no por la mezcla).
7. ~~LOSO/generalización (Etapa 4) partiendo del checkpoint promovido.~~ Hecho 2026-06-28: ver `reports/report_2026-06-28_etapa4_loso.md`. 10/10 folds pasan los 3 gates, promedio `exact=0.9053`, `count_match_rate=0.9566`, `pred_len_mae=0.0528`. **Etapa 4 CERRADA con resultado positivo.**
8. ~~Ablación factorial causal de Etapa 2 (`token_head`×`length_head`×`label_smoothing`, 3 semillas, 24 runs).~~ Hecho 2026-06-28: ver `reports/report_2026-06-28_etapa2_ablation_causal.md`. Corrige la atribución causal del cierre original de Etapa 2 — solo `token_head=contextual` es causa confirmada del salto en `3+ exact`; `length_head`/`label_smoothing` no lo son (ver **Estado Actual**, entrada de Etapa 2).
9. **Siguiente paso real:** ya no queda iteración de arquitectura ni de atribución causal pendiente en este roadmap. Decisión de producto/paper sobre A3 length-conditioned CIF como baseline final, o pulido opcional de las glosas repetidamente problemáticas (`Opaco`, `Enemigo`, `Verde`, `Caramelo`, `Espumadera`, `Rojo`).

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

### Etapa 2: Auditoría De Errores A3 — CERRADA (2026-06-27, iteración 2)
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

### Etapa 3: A4 Scheduled Pred Alpha Suave — CERRADA (2026-06-27, resultado negativo para el mecanismo)
- Partir del mejor A3.
- Entrenar con exposición gradual a alpha predicho:
  - `w_pred=0.02 -> 0.10`
  - no usar `0.50` en esta fase.
- Evaluar siempre el modo oficial `pred_rescaled_to_pred_len`.
- Stop temprano si:
  - `teacher_top1` cae más de 5 puntos contra A3.
  - `pred_len_mae` empeora por encima de `0.15`.
  - `count_match_rate` cae por debajo de `0.90`.
- Acceptance: igualar o mejorar el A3 oficial vigente al momento de correr esta etapa (ver `exact`/`top1` en **Estado Actual** → "Resultado oficial actualizado"), no un umbral absoluto fijo — el oficial cambia entre etapas y un umbral congelado queda obsoleto. `pred_raw.top1` debe mejorar sin degradar `pred_rescaled_to_pred_len`.
- Si A4 mejora raw pero empeora A3 oficial: conservar A3 y descartar A4.

### Etapa 4: Generalización y LOSO — CERRADA (2026-06-28, resultado positivo)
- Ver `reports/report_2026-06-28_etapa4_loso.md`. 10 folds leave-one-signer-out (`signer_id` 1..10), 15 epochs cada uno desde el checkpoint oficial de Etapa 3. Promedio: `exact=0.9053`, `count_match_rate=0.9566`, `pred_len_mae=0.0528`, `top1=0.9286`, `top5=0.9811`. Los 3 gates pasan con margen amplio; los 10 signantes individuales pasan también. **CIF sigue**, no se migra a arquitectura con token queries.
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
  - ~~Etapa 2: auditoría multi-token.~~ Cerrada 2026-06-27 (iteración 2, 4/4 gates pasan); atribución causal corregida 2026-06-28 por ablación factorial — solo `token_head=contextual` es causa confirmada, no `length_head`/`label_smoothing` (ver `reports/report_2026-06-28_etapa2_ablation_causal.md`).
  - ~~Etapa 3: A4 scheduled pred alpha suave.~~ Cerrada 2026-06-27 con resultado negativo para el mecanismo de mezcla (no mejora el oficial, rompe un gate de Etapa 1 a `w_pred` alto); checkpoint promovido por una mejora no relacionada (1 epoch extra de entrenamiento).
  - ~~Etapa 4: LOSO o signer-independent.~~ Cerrada 2026-06-28 con resultado positivo (10/10 folds pasan los 3 gates, sin outliers por signante). CIF sigue, no se migra de arquitectura.
- Si se actualizan resultados, sobrescribir este documento manteniendo la sección **Estado Actual** y **Próximos Pasos Inmediatos** siempre al día.
