# Etapa 3 — A4 Scheduled Pred Alpha Suave

**Resultado final: la hipótesis de Etapa 3 (exponer al modelo a su propia alpha predicha durante entrenamiento mejora el
modo deployable) NO se confirma.** Se corrió dos veces (un intento inválido por confound + un re-run corregido) y en ambos
el mejor checkpoint quedó fuera de la ventana donde la mezcla realmente actúa. Sí hay un hallazgo real y aprovechable: 1
epoch extra de fine-tuning de `cif.alpha`/`length_head` sobre el checkpoint de Etapa 2 (sin nada que ver con la mezcla)
mejora el oficial de forma reproducible. Etapa 3 **CIERRA** con ese checkpoint promovido a oficial; la mezcla `w_pred` queda
documentada como pista para una futura iteración (mejora mucho `pred_raw` pero no el deployable, y a niveles altos rompe un
gate de Etapa 1).

## Intento 1 (INVÁLIDO) — confound de freeze schedule

Run: `outputs/v126_temporal/diag_A3_etapa3_scheduled_pred_alpha_suave_20260627_212649`, `--alpha-schedule linear_pred_mix
--mix-start-epoch 0 --mix-ramp-epochs 10 --mix-w-pred-start 0.02 --mix-w-pred-end 0.10`, warm-start desde Etapa 2, 15 epochs.

`checkpoint_best.pt` quedó fijo en epoch 0. Revisando `train_temporal_v126.py:978-983`: para `epoch < 3` el loss es solo
`alpha_loss + qty_loss + length_loss` — **excluye `token_loss`/`emb_loss`**, los únicos términos que dependen de
`blended_alpha` (la mezcla que usa `w_pred`). `alpha_loss`/`qty_loss`/`length_loss` se calculan directo de
`alpha_logits`/`alpha_pred`/`length_logits`, sin pasar por `blended_alpha`. Conclusión: **`w_pred` no tuvo ningún efecto en
el gradiente durante epochs 0-2**, donde vivía toda la rampa de este intento. El checkpoint ganador es matemáticamente
equivalente a continuar entrenando con `--alpha-schedule target_only` (lo de Etapa 2) un epoch más — no prueba nada sobre
exposición a alpha predicha.

## Intento 2 (válido) — re-run con `--mix-start-epoch 5`

Run: `outputs/v126_temporal/diag_A3_etapa3_scheduled_pred_alpha_suave_v2_20260627_223500`, mismos hiperparámetros salvo
`--mix-start-epoch 5 --mix-ramp-epochs 8` (deja que se estabilice el shock de desbloqueo de parámetros en epoch 3 antes de
empezar a mezclar, así `w_pred` sí entra en epochs donde `token_loss`/`emb_loss` ya están en el loss).

**Verificación empírica del confound:** epochs 0-4 de este re-run (`w_pred=0` real) dieron números **idénticos** a los del
intento 1 (`loss=1.2233`, `teacher_top1=0.952`, etc. en epoch 0) — confirma que el intento 1 nunca probó la mezcla.

`checkpoint_best.pt` volvió a quedar en epoch 0 (`select_metric=(0.9199, 0.9279)`, idéntico al intento 1). Auditoría
standalone (`outputs/v126_temporal/diag_A3_etapa3_v2_audit_best_20260627_230500.json`) confirma que es el mismo modelo:

| métrica (`pred_rescaled_to_pred_len`) | A3 (Etapa 2) | epoch 0 (pre-mezcla) |
|---|---|---|
| top1 | 0.909 | 0.9165 |
| exact | 0.889 | 0.9000 |
| pred_len_mae | 0.072 | 0.0641 |
| count_match_rate | 0.936 | 0.9422 |

Para ver qué pasa cuando la mezcla sí actúa, se auditó también `checkpoint_latest.pt` (epoch 14, fin del ramp,
`w_pred=0.10` sostenido 5 epochs): `outputs/v126_temporal/diag_A3_etapa3_v2_audit_epoch14_20260627_230600.json`.

| métrica | A3 (Etapa 2) | epoch 0 (pre-mezcla) | epoch 14 (mezcla completa) |
|---|---|---|---|
| oficial `exact` | 0.889 | 0.9000 | 0.8891 (≈ A3, no mejora) |
| oficial `top1` | 0.909 | 0.9165 | 0.9146 (≈ A3, no mejora) |
| oficial `count_match_rate` | 0.936 | 0.9422 | **0.9141 — rompe gate Etapa 1 (>=0.93)** |
| `pred_raw.top1` (diagnóstico) | 0.7351 | 0.8065 | **0.8566** |
| `pred_raw.exact` | 0.5797 | 0.6969 | **0.7891** |
| `pred_raw.count_match_rate` | 0.4578 | 0.6734 | **0.7219** |

Ningún epoch de la ventana 5-14 (donde `w_pred` sí afecta el gradiente) superó a epoch 0 en `(exact, top1)` — la selección
automática hizo lo correcto evitándolos. Revisión epoch por epoch de los 3 criterios de stop temprano del roadmap (no se
violó ninguno en todo el run): `teacher_top1` mínimo 0.9247 (umbral 0.8941), `pred_len_mae` máximo 0.1035 en epoch 3
(umbral 0.15), `count_match_rate` mínimo 0.9141 en epoch 14 (umbral 0.90).

## Veredicto

| gate Etapa 3 | requerido | epoch 0 | epoch 14 |
|---|---|---|---|
| Igualar/mejorar A3 | `exact>=0.818` o `top1>=0.869` | PASA (0.900/0.9165) | PASA (0.8891/0.9146) |
| `pred_raw.top1` sin degradar oficial | `>=0.76`, oficial no degrada | PASA (0.8065, oficial mejora) | PASA en pred_raw (0.8566) pero degrada `count_match_rate` oficial bajo el gate de Etapa 1 |

El gate de Etapa 3 tal como está escrito (comparado solo contra A3) técnicamente pasa en ambos checkpoints, pero **epoch 14
rompe un gate de Etapa 1 que debe seguir cumpliéndose** (`count_match_rate>=0.93` → 0.9141). Por eso no se promueve ningún
checkpoint de la ventana de mezcla activa.

## Decisión

**Etapa 3 CIERRA (2026-06-27) con resultado negativo para la hipótesis de scheduled pred-alpha mixing en el modo deployable.**
No se aprueba ningún checkpoint que dependa de `w_pred>0` para producción. Sí se promueve el checkpoint de epoch 0
(`outputs/v126_temporal/diag_A3_etapa3_scheduled_pred_alpha_suave_v2_20260627_223500/checkpoint_best.pt`) a oficial, porque
es una mejora real y reproducible sobre Etapa 2 — pero **no por el mecanismo de Etapa 3**, sino porque es equivalente a un
epoch extra de fine-tuning de `cif.alpha`/`length_head` con el régimen de Etapa 2 (`--alpha-schedule target_only`).

**Pista para el futuro (no bloqueante, no es la siguiente prioridad):** la mezcla sí mejora mucho `pred_raw` (la calibración
cruda del CIF sin el crutch de rescale) — `count_match_rate` 0.458→0.722, `top1` 0.735→0.857 con `w_pred=0.10` sostenido. Si
en algún momento se quiere depender menos del rescale a `pred_len` (por ejemplo para escalar a oraciones más largas o señas
no vistas), vale la pena retomar esta línea con un objetivo distinto: optimizar explícitamente para `pred_raw` en vez de
para el modo rescalado, posiblemente con un `w_pred` más alto o un loss que penalice directamente el `count_match_rate` sin
rescale. No es la siguiente prioridad del roadmap (Etapa 4 LOSO lo es), pero queda anotado.

Próxima fase: Etapa 4 (LOSO/generalización), partiendo del checkpoint promovido.
