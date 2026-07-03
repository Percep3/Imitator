# Etapa 4 (corrección) — LOSO Limpio, Sin Contaminación En Ninguna Etapa

**Resultado: el cierre POSITIVO de Etapa 4 (`report_2026-06-28_etapa4_loso.md`, `exact=0.9053`) queda ANULADO.**
Ese resultado era un artefacto de contaminación: solo los últimos 15 epochs excluían al signante held-out; A1→A3
seguían entrenando con split estratificado que incluía a todos los signantes (limitación ya reconocida en ese mismo
reporte). Con el signante de test excluido de **toda** la cadena (`v121 → A1 → A2 → A3_pre_decoder → decoder_final →
promotion`), el resultado en la métrica deployable colapsa a casi cero en los 6 folds completados hasta ahora.

## Método

- Manifiesto limpio de 10 folds (`scripts/loso/build_clean_loso_manifest.py`): fold `i` usa `outer_test_signer=i`
  (excluido de train y de val en todas las etapas), `inner_val_signer=i%10+1` (solo usado para early-stop/selección de
  checkpoint), resto en train.
- Orquestador resumible con hash de linaje por checkpoint (`scripts/loso/run_clean_loso_orchestrator.py`), que re-entrena
  la cadena completa desde cero por fold en vez de partir de un checkpoint pre-entrenado con todos los signantes.
- Evaluación final con `analyze_imitator_a2.py --heldout-signer <outer_test_signer>` contra `checkpoint_best.pt` de la
  etapa `promotion`.
- **Nota sobre la métrica `exact` de `analyze_imitator_a2.py`:** compara token-a-token tras forzar el re-alineado del
  logit sequence al largo del target (`align_time_to_targets`), no es un chequeo estricto de
  `pred_count==target_len` + contenido. `count_match_rate` sí mide con exactitud si el conteo disparado por CIF
  coincide con el target. Ambas métricas colapsan igual en los folds abajo, así que la conclusión no depende de esta
  distinción — pero queda anotado para no sobre-interpretar `exact` como el "strict_exact" definido en
  `src/mslm/utils/sequence_metrics.py` (ese solo se integró en el training loop de `train_temporal_v126.py`, no en
  este script de auditoría).

## Corte de entrenamiento en fold 6/10

El entrenamiento no se detuvo por un mecanismo de early-stop del orquestador ni por "falta de mejora" automática:
un watcher externo (`stop_after_fold6.log`) esperó a que apareciera `fold6_test_eval.json` y mató el process group
completo a propósito, para poder revisar resultados antes de comprometer las ~44h de GPU restantes (folds 7-10). El
patrón de los 6 folds completados es lo bastante estable (ver tabla) como para no requerir los 4 folds restantes para
la conclusión cualitativa; quedan pendientes solo si se quiere cobertura completa de los 10 signantes para el reporte
final.

## Resultado — métrica oficial deployable `pred_rescaled_to_pred_len`

| fold (outer test signer) | exact | count_match_rate | top1 | quantity_mae (señas) |
|---|---|---|---|---|
| 1 | 0.0125 | 0.0281 | 0.532 | 2.294 |
| 2 | 0.0000 | 0.0094 | 0.513 | 2.300 |
| 3 | 0.0000 | 0.0063 | 0.389 | 2.594 |
| 4 | 0.0000 | 0.0000 | 0.364 | 2.981 |
| 5 | 0.0094 | 0.0094 | 0.466 | 2.628 |
| 6 | 0.0000 | 0.0031 | 0.428 | 2.966 |
| **media (6 folds)** | **0.0037** | **0.0094** | **0.449** | **2.627** |

Comparado contra el gate de Etapa 4 (`exact>=0.45`, `count_match_rate>=0.80`) y contra el criterio de abandono del
roadmap (`exact<0.40` o `count_match_rate<0.75` promedio): **ambos criterios de abandono se cumplen con margen
extremo**, de forma consistente en los 6 folds (varianza mínima: `exact` 0.0–1.25%, `count_match_rate` 0.0–2.8%).

## Diagnóstico: el colapso no es del token classifier ni del encoder visual

Comparando modos en los mismos 6 folds:

| modo | exact (media) | count_match_rate (media) | qué usa |
|---|---|---|---|
| `teacher_alpha` (oracle) | 0.60 | 1.00 | boundary real, no depende del modelo |
| `pred_rescaled_to_target_len` (oracle de longitud) | 0.59 | 1.00 | longitud real, alpha predicho |
| `pred_raw` (sin length_head) | 0.10 | 0.19 | alpha crudo, sin clasificador ni rescale |
| `pred_rescaled_to_pred_len` (oficial deployable) | 0.004 | 0.009 | `length_head` clasificador discreto + rescale |

Dos hallazgos separados:

1. **Bajo longitud oracle, el modelo generaliza razonablemente** (`exact` ~0.44–0.70 por fold, `top1` ~0.59–0.77) —
   no es un colapso del encoder visual ni del token classifier per se.
2. **El `length_head` (clasificador discreto sobre buckets `[0..max_len_class]`, `predict_length_logits`) es el
   componente que falla en signantes no vistos**, no la señal de conteo continua del CIF. Prueba: se re-evaluó el
   checkpoint del fold 6 contra `pred_raw` (que ignora el clasificador y usa directamente `alpha_pred.sum()` como
   conteo) en los 6 folds — `quantity_mae` queda en **0.50–0.63 señas** (menos de una seña de error en promedio),
   muy por debajo de los ~2.3–3.0 señas de error de `pred_rescaled_to_pred_len`. Esto también coincide con el
   `pred_mae_len` ~0.7–0.8 que se veía en los logs de entrenamiento en vivo — ese valor impreso es
   `val_pred_raw['mae_len']` (`train_temporal_v126.py:1211`, `val_predicted = val_pred_raw` cuando
   `--diag-eval-force-count` no está seteado), **no** la métrica del modo deployable — por eso el entrenamiento nunca
   mostró señal visible de este colapso en consola.

Conclusión: la señal continua de conteo del CIF (integración de alpha) sí generaliza a signantes no vistos; lo que
falla es específicamente el mecanismo de rescale-a-longitud-predicha vía el clasificador discreto `length_head`. Esto
es un objetivo de arreglo mucho más acotado que "CIF no generaliza" — por ejemplo, redondear/clampar el conteo
continuo (`pred_raw`) en vez de usar `predict_length_logits().argmax()` para decidir cuántos tokens disparar, es una
hipótesis de arreglo barata que no se probó todavía.

## Estado y próximos pasos (solo dejado documentado, sin acción tomada)

- Folds 7-10 no corridos; el patrón de los 6 folds ya es lo bastante consistente para no depender de ellos para la
  conclusión cualitativa.
- No se abandona CIF todavía pese a que el criterio formal del roadmap se cumple — el diagnóstico de arriba muestra
  que el mecanismo específico que falla (`length_head` clasificador + rescale) es más angosto que "arquitectura CIF
  entera", y no se investigó ni se intentó ningún arreglo en esta sesión.
- No se tocó `ROADMAP_A3_CIF_LENGTH_CONDITIONED.md` más allá de anotar esta corrección — la decisión de migrar de
  arquitectura o de intentar arreglar `length_head` queda pendiente de decisión explícita.
