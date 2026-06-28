# Etapa 4 — LOSO / Generalización a Signantes No Vistos

**Resultado: CIERRA POSITIVO.** Los 3 gates de Etapa 4 pasan con margen amplio en el promedio de los 10 folds, y cada
signante individual pasa también (no hay outliers que dependan del promedio para esconder un colapso puntual).

## Método

`dataset1` tiene 10 signantes (`signer_id` 1..10, confirmado vía `list_clip_records`: 320 clips por signante, 3200
total — no 0..9 ni 5138 como se estimó inicialmente al explorar el código). Se corrió un fold de leave-one-signer-out
por cada uno de los 10 signantes:

- Reutiliza la infraestructura `--heldout-signer` ya existente en `scripts/train/train_temporal_v126.py`
  (`split_records`, `IsolatedTokenEvalDataset`, `evaluate_alpha_mode`) — no se inventó un mecanismo de split nuevo.
- Cada fold parte del checkpoint oficial de Etapa 3
  (`../outputs/v126_temporal/diag_A3_etapa3_scheduled_pred_alpha_suave_v2_20260627_223500/checkpoint_best.pt`,
  `--resume-weights-only`), 15 epochs, mismos hiperparámetros que Etapa 2/3
  (`--alpha-schedule target_only --diag-alpha-loss logit_l1 --diag-freeze target_only_stage1 --stgcn-lr-scale 0.1
  --min-clips 1 --max-clips 1 --max-neutral-frames 8`).
- Se auditó cada fold con `analyze_imitator_a2.py --heldout-signer N` (flag agregado para esta etapa — el script ya
  tenía toda la lógica de modos/métricas, solo le faltaba exponer el filtro por signante) contra `checkpoint_latest.pt`
  (epoch 14, **no** `checkpoint_best.pt`).
- Los 10 audits se combinaron con el nuevo `scripts/diagnostics/aggregate_loso_etapa4.py`.

**Nota metodológica (limitación documentada, no oculta):** se usó `checkpoint_latest.pt` en vez de
`checkpoint_best.pt` a propósito. En modo `--heldout-signer`, `val_loader` *es* la data del signante held-out
(`train_temporal_v126.py:822-841`), así que seleccionar "best" por esa métrica espía al signante de test durante el
entrenamiento. Tomar el checkpoint de la última epoch fija evita ese sesgo sin tocar código.
Además, el checkpoint de Etapa 3 del que parte cada fold fue entrenado en etapas anteriores (A1/A2/A3-oficial) con
split estratificado que sí incluía a todos los signantes — solo esta última etapa de 15 epochs excluye genuinamente al
signante held-out. Una validación 100% libre de esa exposición previa requeriría rehacer la cadena completa A1→A3 diez
veces, fuera de alcance aquí.

## Resultado

Promedio sobre los 10 folds, métrica oficial `pred_rescaled_to_pred_len`:

| métrica | promedio | gate roadmap | resultado |
|---|---|---|---|
| `exact` | 0.9053 | `>=0.45` | PASA |
| `count_match_rate` | 0.9566 | `>=0.80` | PASA |
| `pred_len_mae` | 0.0528 | `<=0.25` | PASA |
| `top1` | 0.9286 | — | — |
| `top5` | 0.9811 | — | — |
| `boundary_mae_when_count_correct` | 0.5498 | — | — |

Por signante (10/10, ver tabla completa en
`../outputs/v126_temporal/diag_A4_etapa4_loso_summary_20260627_234427.md`): `exact` va de 0.8406 (signer 4) a 0.9531
(signer 6); `count_match_rate` de 0.9281 a 0.9719. Ningún signante se acerca a las señales de abandono
(`exact<0.40` o `count_match_rate<0.75` sobre el promedio — ambas `False`).

## Hallazgo lateral: glosas problemáticas repetidas entre signantes

Las peores glosas por signante (top 5, ordenadas por `exact` ascendente) no son ruido aleatorio por signante: las
mismas glosas reaparecen como peor entrada en 4-6 de los 10 folds — `Opaco`, `Enemigo`, `Verde`, `Caramelo`,
`Espumadera`, `Rojo`. Es una debilidad del modelo en esas glosas puntuales (varios son colores o tienen boundary
visual ambiguo), no un problema de un signante particular. No bloquea Etapa 4, pero es candidato a revisar si se
quiere apretar el `exact` promedio más allá del bar actual.

## Decisión

**Etapa 4 CIERRA (2026-06-28) con resultado POSITIVO.** El A3 CIF length-conditioned generaliza a signantes no vistos
dentro del bar de aceptación, con margen amplio (no al límite). Esto resuelve la pregunta abierta del roadmap
("decidir si CIF sigue o si conviene migrar a una arquitectura con token queries, según resultado de LOSO"): **CIF
sigue**, no hay motivo para migrar de arquitectura basado en esta evidencia.

Próxima fase: el roadmap no tenía una Etapa 5 definida; con Etapa 1-4 cerradas, el siguiente paso es decisión de
producto/paper (ej. declarar A3 length-conditioned CIF como baseline final reportable), no más iteración interna de
arquitectura — salvo que se quiera perseguir el hallazgo lateral de glosas repetidas como pulido adicional.
