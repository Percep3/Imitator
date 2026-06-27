# Etapa 2 — Auditoría de errores A3 (longitud / multi-token)

Checkpoint oficial: `outputs/v126_temporal/diag_A3_length_head_rerun_20260627_021153/checkpoint_best.pt` (epoch 11).
Audit script: `scripts/diagnostics/analyze_imitator_a2.py` (extendido con modo `pred_rescaled_to_pred_len`, corte `by_gloss`, y `token_accuracy_when_count_correct/wrong` por bucket de longitud).
Salida cruda: `outputs/v126_temporal/diag_A3_etapa2_audit_20260627_175042.json` (+ `.md`).
Muestras evaluadas: 640 (val split completo: 70 longitud 1, 420 longitud 2, 150 longitud 3+).

## Resultado por longitud (`pred_rescaled_to_pred_len`)

| longitud | n | exact | count_match | token_acc\|count_ok | token_acc\|count_wrong |
|---|---|---|---|---|---|
| 1 | 70 | 0.886 | 0.871 | 0.967 | 0.333 |
| 2 | 420 | 0.843 | 0.976 | 0.899 | 0.000 |
| 3+ | 150 | 0.613 | 0.873 | 0.879 | 0.263 |

Agregado global: `top1=0.853`, `top5=0.957`, `exact=0.794`, `pred_len_mae=0.066`, `count_match_rate=0.941`, `token_accuracy_when_count_correct=0.900`, `boundary_mae_when_count_correct=0.646`.

**Nota de calibración:** esta re-evaluación standalone da números ~1.5-2 puntos por debajo de los registrados en `metrics.jsonl` para epoch 11 (`top1=0.874`, `exact=0.820`), y la brecha aparece igual en `teacher_alpha` (oracle, no usa `length_head`), así que no es un bug introducido por la extensión del script — es una diferencia preexistente entre la validación in-training y la re-evaluación standalone (mismo split, mismo checkpoint, `state_load missing=[]/unexpected=[]`). Los gates de Etapa 1 ya están certificados por `gates.json` del propio entrenamiento; este audit usa sus propios números de forma consistente entre los 4 modos y los cortes (la suma ponderada de los 3 buckets reproduce el `exact` global de este mismo run: 0.794).

## Veredicto contra gates de Etapa 2

| gate | requerido | obtenido | resultado |
|---|---|---|---|
| longitud 1, exact | >= 0.85 | 0.886 | PASA |
| longitud 2, exact | >= 0.80 | 0.843 | PASA |
| longitud 3+, exact | >= 0.65 | 0.613 | **NO PASA** |
| token_accuracy cuando count_match=true | >= 0.88 | 0.900 global / 0.879 en bucket 3+ | **NO PASA en 3+** (marginal, -0.001) |

**Etapa 2 no cierra todavía.** Solo falla el bucket de longitud `3+`, y por margen relativamente chico.

## Diagnóstico: ¿problema de longitud o de decoder?

Para `3+`: `count_match_rate=0.873` y `quantity_abs_error≈0.15` — el `length_head` predice razonablemente bien incluso en secuencias largas. El fallo de `exact` es dominado por **acumulación de error por token**: cuando el conteo es correcto, `token_accuracy≈0.879` por posición, pero `exact` exige que las 3+ posiciones acierten simultáneamente (`0.879³≈0.68`, consistente con el `exact=0.613` observado). Esto coincide con el criterio del roadmap: *"si 3+ queda bajo pero longitud está bien: mejorar decoder/token modeling, no abandonar CIF todavía."*

No se cumple el criterio de abandono de CIF (`longitud 3+ exact < 0.60` con `pred_len_mae <= 0.10`): aquí `exact=0.613 > 0.60`, aunque `pred_len_mae` global (0.066) ya está dentro del límite. Está cerca del borde — si una próxima iteración degrada en vez de mejorar, sí se acercaría al criterio de abandono.

## Corte por glosa (foco multi-token)

Glosas nombradas en el roadmap:
- `Víveres`: exact=0.600, count_match=0.800, token_acc=0.800 (n=10)
- `Aceptar`: exact=1.000 (n=10) — sin problema

Peores 10 glosas por `exact` (n=10 cada una): `Burlarse` (0.20), `Espumadera` (0.30), `Encontrar` (0.40), `Hambriento` (0.40), `Paciencia` (0.40), `Darse cuenta de` (0.50), `Enemigo` (0.50), `Goma de mascar` (0.50), `Lejos` (0.50), `Opaco` (0.50).

Patrón notable en `worst_examples`: las fallas más severas (`token_accuracy=0`) son casos donde el `length_head` se equivoca en frases compuestas multi-palabra largas, ej. `Darse cuenta de` (target_len=4 → predicho 2) y `Goma de mascar` (target_len=4 → predicho 2) — el conteo colapsa a la mitad. Esto sugiere que el `length_head` subestima longitud específicamente en frases de 4+ tokens (fuera del bucket `3+` que aquí incluye todo `>=3`), más que un problema homogéneo de decoder.

## Decisión

No avanzar a Etapa 3 (A4) todavía. Seguir en Etapa 2: mejorar decoder/token modeling para `3+` (y revisar específicamente longitudes >=4 dentro del bucket, que parecen concentrar los peores casos de `length_head`) antes de reintentar el audit. No hay evidencia para abandonar CIF.
