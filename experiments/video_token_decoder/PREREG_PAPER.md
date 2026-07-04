# Pre-registro prospectivo de resultados confirmatorios — Imitator E1

**Fecha de congelación:** 2026-07-04 (UTC)

**Protocolo:** decoder AR video→tokens Gemma, variante E1

**Folds confirmatorios:** 7 y 8, seed 23

**Reserva no tocada por este estudio:** folds 9 y 10

Este documento fija hipótesis, configuración, métricas y reglas de decisión antes de
computar u observar resultados outer-test de los signers 7–8 bajo este protocolo.
Un resultado negativo se conservará y reportará sin cambiar los umbrales.

## 1. Alcance preciso de la prospectividad

La configuración E1 fue seleccionada exclusivamente con inner-val durante el desarrollo
en folds 4–6; sus outer-tests y los números usados para fijar H2 ya son conocidos. Por
tanto, E1 y los umbrales derivados de desarrollo no son descubrimientos prospectivos.
El claim prospectivo es más estrecho:

> Antes del commit de este documento no se computó ni observó ningún resultado
> outer-test de los signers 7–10 bajo este protocolo E1.

Los signers 7–10 sí aparecieron como train/inner-val en otras rotaciones LOSO y en
protocolos antiguos v117–v124 con splits u objetivos distintos. «Held-out» o «virgen»
en el paper significará únicamente *outer-test no observado para E1 en esta rotación*.

El 2026-07-01 a las 23:54:52 UTC el orquestador inició por error la etapa v121 del fold 7
y un watcher preconfigurado la terminó nueve segundos después, antes de cualquier señal
del fold. Solo quedó `outputs/checkpoints/121/9007/label_to_idx.json`, un mapeo
determinista sin pesos, métricas ni evaluación. Ese directorio se elimina antes del
relanzamiento y fold 7 se reconstruye desde cero después del commit.

La rotación hace que signer 8 sea inner-val de fold 7 y outer-test de fold 8. No existe
fuga entre modelos: cada fold parte de cero y excluye a su outer signer. Para eliminar
grados de libertad del investigador, la configuración y los criterios quedan congelados
aquí, se entrenarán ambos folds antes de evaluar cualquiera de sus outer-tests, y no se
ajustará nada tras observar fold 7.

## 2. Claim permitido

Los claims se restringen a **transcripción token-level ordenada de streams sintéticos
continuos construidos concatenando 2–8 clips de señas aisladas, LOSO sobre 10 signers y
64 glosas**. La salida son IDs de Gemma. El orden de glosas del generador es aleatorio:
ningún modelo puede explotar un prior lingüístico, por lo que la evidencia de orden
procede del video. No se harán claims sobre lengua de señas continua real, coarticulación
o texto libre.

Folds 1–6 se etiquetan como desarrollo; folds 7–8 como confirmación; folds 9–10 se
mantienen intactos para trabajo futuro. E3/unfreeze se presenta únicamente como hallazgo
exploratorio de folds 4–6 y no se ejecuta aquí.

## 3. Variante E1 congelada

- Seed: 23.
- Init independiente por fold desde
  `../outputs/loso_clean/diag_fold{FOLD}_promotion/checkpoint_best.pt`, con lineage y hash
  validados y sin exposición del encoder al outer signer del fold.
- PE sinusoidal en el encoder (`--encoder-pe`).
- Vocabulario de salida restringido al mapeo biyectivo de 121 IDs Gemma efectivos
  (`--restricted-vocab`).
- Label smoothing 0; selección de checkpoint por edit similarity de inner-val; 30 epochs.
- 2048 ejemplos sintéticos de train por epoch; 512 ejemplos fijos de inner-val.
- ST-GCN y `linear_hidden` siempre congelados; TCN/transformer se habilitan desde epoch 5
  a lr 3e-5; decoder a lr 3e-4. Sin CTC, rescate ni unfreeze.
- Outer-test: exactamente 896 muestras, seed `23 + 200000 = 200023`, batch 2.

Comandos congelados, sustituyendo `FOLD` por 7 y 8:

```bash
PY=/home/nakato/miniconda3/envs/Sign-env/bin/python

$PY scripts/train/train_video_token_decoder.py train --fold FOLD --seed 23 \
  --encoder-pe --restricted-vocab --label-smoothing 0 --select edit --epochs 30 \
  --run-tag e1_pe_vocab121

$PY scripts/train/train_video_token_decoder.py evaluate --fold FOLD --seed 23 \
  --checkpoint ../outputs/video_token_decoder/foldFOLD_seed23/e1_pe_vocab121/checkpoint_closed.pt \
  --cif-comparator ../outputs/video_token_decoder/foldFOLD_cif_comparator.json

$PY scripts/eval/robustness_video_token_decoder.py --fold FOLD --seed 23 \
  --checkpoint ../outputs/video_token_decoder/foldFOLD_seed23/e1_pe_vocab121/checkpoint_closed.pt \
  --cif-comparator ../outputs/video_token_decoder/foldFOLD_cif_comparator.json
```

El script de robustez rechaza checkpoints que no coincidan con esta configuración.

## 4. Hipótesis y reglas de decisión

Todas las condiciones son **conjuntivas por fold**: una media de folds nunca puede
ocultar el colapso de un signer. El estudio satisface una hipótesis solo si la satisface
por separado en fold 7 y fold 8.

### H1 — primaria, superioridad frente a CIF affine

En cada fold, E1 debe tener más wins que losses de `strict_exact` frente a CIF affine y
un sign test exacto bilateral con `p < 0.01`. Cada una de las 896 secuencias es un par;
los ties se excluyen del tamaño binomial discordante. CIF usa las mismas muestras y seed,
y su calibración affine se ajusta únicamente en inner-val.

### H2 — generalización mínima por signer

En cada fold, `strict_exact >= 0.119`. Es una cota inferior unilateral: cualquier valor
superior cuenta como éxito, incluido uno por encima del rango dev.

La cota es el mínimo dev observado (16.9%) menos 5 puntos porcentuales. Vista desde el
centro dev (~18.3%), el margen hasta 11.9% cubre el spread completo entre signers de
desarrollo (2.9 pp) y aproximadamente 2.7 errores estándar muestrales adicionales
(SE ~1.3 pp con n=896 y p~0.18). Es deliberadamente conservadora, pero se exige en cada
signer y no sobre una media que pueda ocultar un fallo.

### H3 — uso causal del orden temporal

En cada fold deben cumplirse ambas condiciones:

1. exactitud de orden pareado entre glosas `>= 0.90` en la condición limpia;
2. caída absoluta media de edit similarity `clean - segment_permutation >= 0.15`.

La segunda cantidad es pareada por muestra y luego promediada dentro del fold. No se
transforma a porcentaje relativo.

La métrica de orden está versionada en
`src/mslm/utils/sequence_metrics.py::pairwise_gloss_order_counts` y su agregación en
`gloss_sequence_diagnostics`. Reproduce el diagnóstico Fase2b: los IDs se parsean a
glosas mediante longest-match; solo son elegibles las glosas que aparecen exactamente
una vez tanto en target como en predicción; se evalúan todos sus pares no ordenados y se
micro-promedian pares concordantes. Una muestra con menos de dos glosas elegibles aporta
cero pares y queda fuera del denominador. Un prefijo parseable de una predicción con
suffix no parseable puede contribuir, igual que en Fase2b; la parseabilidad se reporta
por separado. En dev, esta implementación reproduce 0.9856/0.9893/0.9862 en folds 4–6.

La intervención usa
`src/mslm/dataloader/synthetic_temporal.py::permute_video_segments` con boundaries gold,
mantiene fijo el target, conserva frames y longitud, y obliga una permutación no identidad.
Se aplica una permutación a cada una de las mismas 896 muestras con seed por muestra
`7800023 + sample_index`.

### Regla global

- **Confirmación completa:** H1, H2 y H3 pasan en ambos folds.
- **Confirmación parcial o negativa:** se informa qué hipótesis/fold falló, sin redefinir
  métricas, umbrales, subconjuntos o configuración.

## 5. Batería secundaria de robustez

Todas las perturbaciones actúan después de la normalización y nunca cambian el target:

- jitter gaussiano iid sobre todos los keypoints de frames válidos, sigma
  `{0.01, 0.02, 0.05}`, seed por muestra `7800024 + sample_index`;
- frame dropout por eliminación Bernoulli independiente, p `{0.1, 0.2}`, reteniendo al
  menos un frame, seed por muestra `7800025 + sample_index`;
- velocidad temporal `{0.75x, 1.25x}` mediante interpolación lineal, donde
  `T_out = round(T_in / speed)`; 0.75x alarga y 1.25x acorta.

Se reportan curvas de `strict_exact` y edit similarity, sus deltas pareadas frente a
clean, diagnóstico de glosa y estratificación por número de clips (2–8), cuartiles de
longitud en frames y cuartiles de longitud target en tokens. Los bordes de cuartil se
calculan por fold usando solo longitudes —nunca outcomes— y se reutilizan dentro de sus
condiciones. Estas pruebas son descriptivas y no tienen umbrales de éxito adicionales.

## 6. Incertidumbre y outputs

Para `strict_exact` y edit similarity se calculan intervalos bootstrap percentiles 95%
no paramétricos a nivel de muestra, 10 000 réplicas, seed base 7800026. Para cada
perturbación se bootstrappea además el delta pareado clean−perturbación. Los CIs son
descriptivos; las decisiones H1–H3 usan los estimadores puntuales y el sign test fijado.

Cada fold produce `outer_test.json` y `outer_test_robustness.json` con predicciones fila a
fila, hashes de checkpoint/comparador, seeds, métricas, CIs, estratos y evaluación
automática de H1–H3. Los checkpoints de folds 7–8 se conservan hasta cerrar el paper.

## 7. Orden operativo y cegamiento

1. Commit de código, tests y este documento.
2. Borrar el residuo autorizado `../outputs/checkpoints/121/9007/`.
3. Preparar ambos inits con el orquestador en `--prepare-only`; este modo no ejecuta
   `run_test_signer_eval`.
4. Generar comparadores CIF pareados sin inspeccionar sus métricas.
5. Entrenar y cerrar E1 en fold 7 y fold 8 antes de evaluar cualquier outer-test.
6. Evaluar ambos folds y ejecutar la batería sin modificar código/configuración entre
   resultados.
7. Aplicar literalmente la regla global y reportar también resultados negativos.

## 8. Limitaciones fijadas

- Una sola seed de entrenamiento: la varianza entre seeds no está estimada.
- Solo dos folds confirmatorios: incertidumbre de heterogeneidad entre signers limitada.
- Datos sintéticos derivados de clips aislados, con gaps artificiales y sin
  coarticulación real.
- E1 fue seleccionada y H2 calibrada en folds 4–6 de desarrollo.
- La relación inner-val/outer entre rotaciones permite que un signer sea inner de un fold
  antes de ser outer de otro; la configuración congelada y el entrenamiento previo a
  toda evaluación outer eliminan decisiones adaptativas, pero no cambian esa estructura.
