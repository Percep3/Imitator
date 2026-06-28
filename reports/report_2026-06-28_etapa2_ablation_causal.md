# Etapa 2 — Ablación factorial causal (2x2x2, 3 semillas)

## Contexto

El cierre original de Etapa 2 (`reports/report_2026-06-27_etapa2_a3_audit_iter2.md`) cambió simultáneamente tres cosas respecto al baseline de Etapa 1 — `token_head` (linear → contextual, con position embedding + `nn.TransformerEncoderLayer`), `length_head` (mean-pooling → 1-query attention pooling) y `label_smoothing` (0.0 → 0.1) — y atribuyó la mejora de `longitud 3+ exact` (0.613 → 0.860) al conjunto, sin aislar cuál de los tres cambios fue la causa real. Esta ablación factorial completa (`2×2×2` combinaciones × 3 semillas = 24 runs, warm-start desde el checkpoint oficial de Etapa 1, `--split-seed 23` fijo en los 24 runs, 15 epochs, mismos hiperparámetros que el run original) resuelve esa atribución.

Plan: `docs/superpowers/plans/2026-06-28-v126-etapa2-ablation-closeout.md`. Evidencia completa (config/metrics/gates/audit por run + resumen + manifiesto con SHA-256): `artifacts/v126_closeout/` (`ablation_summary.json` es la fuente exacta de los números abajo; `manifest.json` permite verificar que no se alteraron).

## Regla de atribución causal usada

Un componente solo se reporta como causa confirmada de una mejora si su efecto es positivo en las 3 semillas (`23, 42, 101`) **tanto** como efecto marginal (promediado sobre las 4 combinaciones de los otros dos factores) **como** al retirarlo del modelo completo (`contextual`+`attention`+`ls=0.1` vs la misma combinación con un factor apagado). Si una sola semilla discrepa, o si el marginal y el modelo-completo discrepan, el veredicto es "efecto mixto" — nunca "causa confirmada".

## Resultado por variante (media ± desviación sobre 3 semillas)

| token_head | length_head | label_smoothing | 3+ exact | tacc_count_ok | count_match_rate | exact (global) | top1 | top5 | pred_len_mae |
|---|---|---|---|---|---|---|---|---|---|
| contextual | attention | 0.0 | 0.8711 ± 0.0204 | 0.9544 | 0.9432 | 0.8870 | 0.9111 | 0.9703 | 0.0641 |
| contextual | attention | 0.1 | 0.8667 ± 0.0176 | 0.9565 | 0.9406 | 0.8948 | 0.9143 | 0.9757 | 0.0651 |
| contextual | mean | 0.0 | 0.8800 ± 0.0291 | 0.9414 | 0.9490 | 0.8844 | 0.9072 | 0.9724 | 0.0563 |
| contextual | mean | 0.1 | 0.8533 ± 0.0533 | 0.9546 | 0.9328 | 0.8859 | 0.9076 | 0.9699 | 0.0734 |
| linear | attention | 0.0 | 0.5889 ± 0.0168 | 0.8957 | 0.9375 | 0.7833 | 0.8437 | 0.9548 | 0.0688 |
| linear | attention | 0.1 | 0.6111 ± 0.0329 | 0.8995 | 0.9380 | 0.7818 | 0.8482 | 0.9606 | 0.0677 |
| linear | mean | 0.0 | 0.5911 ± 0.0482 | 0.8989 | 0.9396 | 0.7833 | 0.8480 | 0.9574 | 0.0656 |
| linear | mean | 0.1 | 0.6444 ± 0.0038 | 0.8978 | 0.9552 | 0.8000 | 0.8604 | 0.9647 | 0.0505 |

## Veredictos causales (`causal_report` en `ablation_summary.json`)

| factor | 3+ exact | token_accuracy_when_count_correct | count_match_rate | exact (global) |
|---|---|---|---|---|
| `token_head=contextual` | **causa respaldada** | **causa respaldada** | efecto mixto | **causa respaldada** |
| `length_head=attention` | efecto mixto | efecto mixto | efecto mixto | efecto mixto |
| `token_label_smoothing=0.1` | efecto mixto | efecto mixto | efecto mixto | **causa respaldada** |

## Conclusión

- **`token_head=contextual` es la causa confirmada** del salto de Etapa 2 en `longitud 3+`: efecto marginal de +0.25 a +0.28 puntos de `exact` en las 3 semillas (0.59 → 0.87 promedio), consistentemente positivo tanto marginal como al retirarlo del modelo completo. También causa confirmada en `token_accuracy_when_count_correct` y en `exact` global. Esto es lo que da contexto entre posiciones al decodificar tokens — el diagnóstico original ("acumulación de error por posición") apuntaba al mecanismo correcto, pero el cambio que realmente lo resuelve es solo `token_head`, no el paquete de tres cambios.
- **`length_head=attention` no tiene efecto causal confirmado** en ninguna de las 4 métricas — el efecto es mixto (a veces negativo) entre semillas, tanto marginal como al retirarlo del modelo completo. El cambio de mean-pooling a attention pooling no fue, por sí solo, responsable de la mejora atribuida en el cierre original.
- **`token_label_smoothing=0.1` tiene un efecto causal confirmado pero pequeño**, solo en `exact` global (+0.3 a +1.1 puntos en las 3 semillas) — no en `3+ exact` específicamente, donde el efecto es mixto.
- El mejor variante observado para `3+ exact` (`contextual`+`mean`+`ls=0.0`, 0.880 ± 0.029) no incluye `length_head=attention`; el oficial de Etapa 2 (`contextual`+`attention`+`ls=0.1`, 0.867 ± 0.018) está dentro del ruido entre semillas de las variantes `contextual`+*+*. Esto es consistente con que `length_head`/`label_smoothing` no mueven la aguja en `3+ exact` una vez que `token_head=contextual` está presente.

**Implicación para el roadmap:** el checkpoint oficial actual (que usa `contextual`+`attention`+`ls=0.1`) se mantiene como candidato deployable — no hay evidencia de que retroceder `length_head`/`label_smoothing` mejore nada, y sí hay evidencia (causa confirmada) de que `token_head=contextual` es indispensable. No se justifica re-entrenar el oficial; solo se corrige la explicación causal.
