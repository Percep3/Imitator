# HANDOFF — decoder AR video→tokens Gemma (Imitator)

Última actualización: 2026-07-04. Estado: **causa raíz encontrada, corregida y confirmada en folds 4–6.** Rama: `v125-temporal-prompt-alignment`.

## 1. Qué es esto en una frase

Imitator convierte secuencias de keypoints de lengua de señas en **secuencias de token IDs de Gemma** (NO clasifica glosas — esa ruta está descartada por decisión de producto). El decoder autoregresivo original daba strict_exact ~2%; tras corregir la causa raíz da **17–20%** y gana a CIF en los tres folds.

## 2. El problema que estaba roto y por qué

El decoder AR emitía una **bolsa de señas**: identificaba las glosas presentes pero en **orden aleatorio** (0.53, azar=0.5). Causa raíz demostrada por intervención:

- El encoder visual (`STGCNTemporalFrameEncoder`, `src/mslm/models/temporal_sign_prompt.py`) no tenía **positional encoding**. Un `TransformerEncoder` sin PE es permutación-equivariante ⇒ la memoria que ve el decoder es un *conjunto* de frames, no una secuencia.
- Diagnóstico decisivo: permutar los segmentos de video del input dejaba el **86.3% de las predicciones idénticas** y no cambiaba las métricas.
- Refutado (NO eran la causa): la init tied desde `token_head.classifier.weight` (geometría sana), y el vocabulario de 262 400 (el 100% de los tokens predichos ya caía dentro de los 118 IDs efectivos).

Informe completo con evidencia línea a línea: **`AUDIT_REPORT.md`** (13 secciones + addendum de resultados). Léelo antes de tocar nada.

## 3. La corrección que funcionó (variante "E1", confirmada)

Tres cambios, todos en código ya integrado:

1. **PE sinusoidal** en el encoder y en la memoria del decoder (`--encoder-pe`). Esta es la corrección de la causa raíz.
2. **Vocabulario de salida restringido a los 121 IDs Gemma efectivos** (118 glosas + BOS/EOS/PAD) con mapeo biyectivo denso↔Gemma guardado en el checkpoint (`--restricted-vocab`). La salida sigue siendo IDs Gemma; cuando el dataset crezca solo crece la tabla.
3. **Sin label smoothing** (`--label-smoothing 0`) + **selección por edit_sim** (`--select edit`) + **30 epochs** (el original nunca convergía en 15).

ST-GCN y `linear_hidden` quedan **congelados**; TCN/transformer lr 3e-5 desde epoch 5; decoder lr 3e-4.

### Comando exacto para reproducir un fold

```bash
cd /shared/Code/Sign-AI/Sign-chris
PY=/home/nakato/miniconda3/envs/Sign-env/bin/python
$PY scripts/train/train_video_token_decoder.py train --fold 4 \
  --encoder-pe --restricted-vocab --label-smoothing 0 --select edit --epochs 30 \
  --run-tag e1_pe_vocab121
$PY scripts/train/train_video_token_decoder.py evaluate --fold 4 \
  --checkpoint ../outputs/video_token_decoder/fold4_seed23/e1_pe_vocab121/checkpoint_closed.pt \
  --cif-comparator ../outputs/video_token_decoder/fold4_cif_comparator.json
```

Coste: ~1.2 h train + ~10 min eval por fold en 1× RTX 4060 Ti. Memoria <2 GB.

## 4. Resultados confirmados (C1, outer-test pareado, seed 23)

| fold | CIF | AR baseline | **E1** | edit_sim | vs CIF (W/L/T) |
|---|---|---|---|---|---|
| 4 | 0.023 | 0.022 | **0.181** | 0.630 | 144/3/749 |
| 5 | 0.049 | 0.023 | **0.198** | 0.619 | 140/7/749 |
| 6 | 0.026 | 0.018 | **0.169** | 0.585 | 131/3/762 |

Total pareado n=2688: **415W / 13L / 2260T**, sign test bilateral **p ≈ 6·10⁻¹⁰⁵**. Orden entre glosas: 0.53 → **~0.99**.

Los JSON con estos números están en `../outputs/video_token_decoder/fold{4,5,6}_seed23/e1_pe_vocab121/outer_test.json` (métricas + `predictions` fila a fila + bloque `paired_vs_affine_cif`). **Los checkpoints `.pt` fueron borrados a propósito** para no acumular peso; se reentrenan con el comando de arriba (deterministas por seed).

## 5. Experimentos descartados (no repetir sin motivo)

- **E2 = JTA con CTC token-level auxiliar (λ=0.3)** (`--ctc-weight 0.3`): empata con E1, el CTC no aporta y su decodificación colapsa a blank. Descartado.
- **E3 = unfreeze del ST-GCN desde epoch 10** (`--unfreeze-stgcn-epoch 10`): mejoró en outer (19.5%, 154W/0L) pero **NO en inner-val** (edit 0.606 < 0.615 de E1). NO se adoptó porque seleccionar por outer contaminaría la confirmación. **Es la palanca exploratoria más prometedora** (ataca el techo de identidad), pero necesita su propia etapa pre-registrada evaluada contra folds 7–10.

## 6. Dónde está cada cosa

- **Modelo**: `src/mslm/models/video_token_decoder.py` — `VideoTokenDecoder` (PE, vocab restringido, cabeza CTC opcional), `set_decoder_training_stage` (freeze schedule + `unfreeze_stgcn_epoch`), `initialize_from_cif_checkpoint`, `validate_checkpoint_lineage`.
- **Encoder**: `src/mslm/models/temporal_sign_prompt.py` — `STGCNTemporalFrameEncoder` (aquí vive el PE nuevo) y el `TemporalSignPromptModel`/CIF de referencia.
- **Entrenamiento/eval**: `scripts/train/train_video_token_decoder.py` — subcomandos `train`, `evaluate`, `cif-comparator`. `optimizer_for` tiene ahora 3 grupos (decoder 3e-4, visual 3e-5, stgcn 1e-5).
- **Datos sintéticos**: `src/mslm/dataloader/synthetic_temporal.py` — concatena 2–8 clips aislados en secuencias continuas.
- **Tests**: `tests/models/test_video_token_decoder.py`, `tests/models/test_video_token_decoder_jta.py`, `tests/scripts/test_train_video_token_decoder.py`. **26/26 en verde.** Corre: `Sign-env/bin/python -m pytest tests/models/ tests/scripts/test_train_video_token_decoder.py -q`.
- **Comparadores CIF** (precomputados, pareados por seed): `../outputs/video_token_decoder/fold{4,5,6}_cif_comparator.json`.
- **Checkpoints fuente CIF** (init del encoder, lineage validado, NO tocar): `../outputs/loso_clean/diag_fold{1..6}_promotion/checkpoint_best.pt`.

## 7. Restricciones e invariantes (no romper)

- **Salida = tokens Gemma siempre.** Nada de clasificación de glosas ni predicción de las 64 clases, en ningún escenario.
- **Folds 7–10 vírgenes**: jamás entrenados ni evaluados; reservados como held-out final para claims de paper.
- **Selección de variante SOLO en inner-val.** Usar el outer-test para elegir entre variantes quema la confirmación (así se rechazó E3).
- **Lineage**: el init del encoder viene de un checkpoint CIF que nunca vio al signer outer; `validate_checkpoint_lineage` lo verifica. Mantener.
- Entrenamientos largos: lanzar en `tmux` + observar con Monitor (preferencia del usuario). Entorno Python: `/home/nakato/miniconda3/envs/Sign-env/bin/python`.

## 8. Siguiente paso recomendado

El techo restante es la **identidad de glosa entre signers (~70%**, medido con un linear probe sobre las features congeladas). La inversión legítima es una **etapa nueva y pre-registrada** de unfreeze/augmentación del encoder (partiendo de E3), con criterio de éxito fijado *antes* de mirar resultados, evaluada por primera vez contra folds 7–10. No re-optimizar sobre folds 4–6 (ya usados para desarrollo — sesgo documentado en `AUDIT_REPORT.md` §12).
