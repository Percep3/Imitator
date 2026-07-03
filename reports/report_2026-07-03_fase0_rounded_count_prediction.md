# Fase 0 — Predicción del modo rounded-count desde los audits LOSO existentes

**Fecha:** 2026-07-03
**Costo:** 0 GPU-h (solo lectura de los `fold{1..6}_test_eval.json` ya escritos)
**Script reproducible:** `scripts/diagnostics/predict_rounded_count_from_audits.py`
**Plan padre:** `/home/nakato/.claude/plans/genere-el-plan-detallado-indexed-quokka.md`

## Pregunta

Antes de gastar GPU en re-evaluar los 6 folds del LOSO limpio con el modo nuevo
`pred_rescaled_to_rounded_count` (conteo = `round(alpha.sum())` en vez del
`length_head` clasificador discreto que colapsó — ver
`report_2026-07-03_etapa4_loso_clean_correction.md`): ¿qué `count_match_rate`
cabe esperar? Los histogramas `hist_quantity_minus_target_len` (bins de 0.1)
guardados en los audits de `pred_raw` permiten calcularlo exactamente, porque
`P(round(quantity) == target_len) = P(|quantity − target_len| < 0.5)`.

## Resultado 1 — redondeo puro (bias=0): PARCIAL

| fold | P(round correcto) | sesgo medio de quantity | count_match del length_head (referencia) |
|---|---|---|---|
| 1 | 0.550 | −0.398 | 0.028 |
| 2 | 0.545 | −0.392 | 0.009 |
| 3 | 0.573 | −0.220 | 0.006 |
| 4 | 0.487 | −0.586 | 0.000 |
| 5 | 0.502 | −0.516 | 0.009 |
| 6 | 0.519 | −0.481 | 0.003 |
| **media** | **0.529** | −0.432 | 0.009 |

Contra el criterio de Fase 0 del plan (≥0.60 en ≥4/6 folds): **0/6 folds cumplen**
→ banda PARCIAL (0.40–0.60). El redondeo puro ya multiplica ×56 el count_match
del clasificador discreto, pero no alcanza el gate de éxito de Fase 2
(`count_match ≥ 0.75`) por sí solo.

## Resultado 2 — el déficit es un sesgo sistemático corregible con UN escalar

El CIF **sub-cuenta en los 6 folds** (sesgo medio −0.22 a −0.59, nunca positivo).
Con una corrección global `count = round(quantity + b)`:

| b | f1 | f2 | f3 | f4 | f5 | f6 | media |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.491* | 0.497* | 0.516* | 0.447* | 0.459* | 0.456* | 0.478* |
| **0.2** | 0.672 | 0.659 | 0.656 | 0.641 | 0.662 | 0.675 | **0.661** |
| 0.3 | 0.622 | 0.637 | 0.603 | 0.628 | 0.641 | 0.647 | 0.630 |
| 0.4 | 0.631 | 0.637 | 0.588 | 0.666 | 0.650 | 0.672 | 0.641 |

(\*fila b=0.0 con criterio estricto `|δ+b|<0.5` sobre bins; difiere ~2pp de la
tabla de arriba por el tratamiento del bin borde ±0.5 — no cambia la conclusión.)

Dos observaciones que hacen esto prometedor:

1. **b=0.2 es casi óptimo para TODOS los folds a la vez**: el techo con sesgo
   óptimo por-fold (que espía al held-out, solo referencia) es 0.656–0.725 —
   un único b global captura casi todo ese margen. El escalar transfiere entre
   signantes; no hace falta calibración por signante.
2. Estimación de `exact` esperado: count_match ~0.66 × token-accuracy-dado-conteo
   (exact oracle-length ~0.59–0.60 en estos folds) ≈ **0.39–0.45** — justo en el
   umbral de éxito del plan (`exact ≥ 0.40`). La re-evaluación real (Fase 2)
   decide de qué lado cae.

## Decisión

- Se implementó el modo con el bias como parámetro desde el inicio
  (`--rounded-count-bias`, default 0.0), adelantando la palanca B2 del plan que
  esta evidencia ya justifica.
- La Fase 2 debe correrse con **dos** valores por fold: `b=0` (hipótesis H1 pura)
  y `b` ajustado SOLO con signantes de train del fold (protocolo limpio; el b*
  por held-out de arriba es solo techo de referencia, NO deployable).
- Cómo ajustar b sin contaminar: correr el mismo audit del fold contra un
  `--heldout-signer` de TRAIN de ese fold (p.ej. el `inner_val_signer` del
  manifiesto) y elegir el b que maximice count_match ahí; ese b se congela y se
  aplica al held-out real.

## Estimación honesta de riesgo

Si la re-evaluación real cae en `exact` 0.25–0.40 (banda PARCIAL de Fase 2), el
plan sigue por B2 (calibración afín `a·quantity + b`, no solo aditiva) antes de
tocar entrenamiento. El histograma sugiere que la pendiente `a` podría importar:
el sesgo por fold correlaciona con la longitud media del target (folds 4–6
peores), lo que una calibración solo-aditiva no captura del todo.
