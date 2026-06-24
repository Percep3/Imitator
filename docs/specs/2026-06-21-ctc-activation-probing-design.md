# Diagnóstico de probing de activaciones por capa — CTCEncoder (v119)

**Fecha:** 2026-06-21
**Contexto:** ver `report.md`, sección v119, "Diagnóstico de interpretabilidad". El
diagnóstico de posteriors (`scripts/diagnostics/diagnose_ctc_posteriors.py`) ya mostró que el
colapso a blank no es ruido localizado: P(blank) domina TODO el interior de la
secuencia, y la única señal no-blank aparece en los bordes (primer/último frame),
nunca por contenido visual intermedio. Eso apunta la causa raíz río arriba del
BiLSTM, pero no distingue entre dos hipótesis:

- **(a)** El BiLSTM no logra propagar información discriminativa de frames
  intermedios (la señal existe a la entrada del BiLSTM pero se pierde dentro de él).
- **(b)** La señal que llega desde GCN+TCN/TLP a la entrada del BiLSTM ya viene
  plana por frame (el BiLSTM solo refleja sus propios estados de borde porque no
  hay nada que propagar).

Este documento diseña el probing de activaciones por capa para diferenciar (a) de (b).

## Objetivo

Medir, en 4 puntos del forward de `CTCEncoder`, cuánta variación temporal frame-a-frame
sobrevive, sobre clips reales de val, en dos checkpoints ya entrenados. Identificar en
qué etapa colapsa la actividad temporal.

## Checkpoints a sondear

- `outputs/checkpoints/119/1/best_wer/checkpoint.pth` (ep2, WER 99.7%, con palancas
  data-oriented activas — mismo checkpoint que ya usó `diagnose_ctc_posteriors.py`)
- `outputs/checkpoints/119/1_baseline_no_data_levers/best_wer/checkpoint.pth` (ep15,
  WER 99.25%, arquitectura nueva sin palancas data-oriented)

Sondear ambos confirma si el patrón de colapso es independiente de esa variable (ya
descartada en el report como causa raíz) o no.

## Puntos de sondeo

Usando los submódulos públicos de `CTCEncoder` (sin modificar `forward()`), replicando
sus pasos en el script de diagnóstico:

1. `gcn` — tras GCN (`stgcn_layers`/`stgcn_motion_layers`) + `linear_hidden` + mean-pool
   sobre nodos. Forma `[B, hidden, T]`.
2. `tcn1` — tras `tcn_conv1` + `tlp1`.
3. `tcn2_short` — tras `tcn_conv2` + `tlp2` (= `feats_short`, entrada al BiLSTM, salida
   Y_s del paper).
4. `bilstm_long` — tras `bilstm` (= `feats_long`, salida Y_l).

## Métrica: actividad temporal normalizada

Por canal `c`, sobre el eje temporal post-downsampling de esa etapa:

```
activity_c = var_t(x_c) / (mean_t(x_c²) + ε)
```

Promediada sobre canales → un score escalar de "actividad temporal" por etapa y por
clip. Normalizar por la potencia (`mean(x²)`) en vez de usar varianza cruda permite
comparar etapas con escalas de activación distintas. Score cercano a 0 = la
representación es casi constante a lo largo del tiempo (plana); score más alto = más
variación frame a frame.

## Refactor necesario en `diagnose_ctc_posteriors.py`

`build_val(tag)` deriva la ruta del checkpoint solo de `run_id` (int) leído del config
(`checkpoints/{version}/{run_id}/{tag}/checkpoint.pth`), por lo que no puede apuntar a
`1_baseline_no_data_levers`. Se le agrega un parámetro opcional `run_dir: str | None`
que, si se pasa, reemplaza `str(run_id)` en la ruta de checkpoint y de vocab. Sin el
parámetro, comportamiento idéntico al actual (compatibilidad con el script existente).

## Script nuevo: `scripts/diagnostics/diagnose_ctc_activations.py`

Mismo patrón de uso que `diagnose_ctc_posteriors.py` (variable de entorno
`MSLM_EXPERIMENT_CONFIG`, debe correr desde la raíz del repo con `PYTHONPATH=.`).

Para cada checkpoint (los dos de la sección anterior) y para ~20 clips de val (mismo
`val_dl` de `build_val`, batch_size=1, sin padding):

1. Forward manual replicando `CTCEncoder.forward()` hasta cada uno de los 4 puntos de
   sondeo, capturando el tensor `[1, hidden, T_etapa]` en cada uno (sin necesidad de
   forward hooks: los submódulos son atributos públicos del modelo).
2. Calcular `activity_c` por canal y promediar → un score por etapa.
3. Acumular por clip → matriz `[n_clips, 4_etapas]`.

### Salida por checkpoint (en `outputs/diagnostics/{version}/{run_dir}/activations/`)

- Un plot de líneas: eje X = las 4 etapas (`gcn`, `tcn1`, `tcn2_short`,
  `bilstm_long`), eje Y = score de actividad, una línea por clip (semi-transparente)
  + una línea más gruesa con la media sobre clips. Permite ver visualmente en qué
  etapa cae la actividad, y si es consistente entre clips.
- `summary.txt`: media ± std del score por etapa, sobre los clips evaluados.

### Salida final (consola)

Tabla comparando media±std por etapa entre los dos checkpoints, para responder
directamente: ¿la caída ocurre en la misma etapa en ambos?

## Interpretación esperada

- Si la actividad cae fuerte entre `tcn2_short` → `bilstm_long`: confirma hipótesis
  (a), el BiLSTM lava la señal.
- Si ya cae en `gcn` o `tcn1` (antes de llegar a `tcn2_short`): confirma hipótesis (b),
  el problema es anterior al BiLSTM.
- Si no cae en ninguna etapa de forma clara: el colapso a blank observado en posteriors
  no se explica por pérdida de varianza temporal cruda, y habría que medir otra cosa
  (p.ej. si la varianza existe pero no está alineada con las clases correctas).

## Fuera de alcance

- No se reentrena nada ni se modifica el modelo productivo (`CTCEncoder`) — el
  probing es enteramente reconstrucción manual del forward en el script de
  diagnóstico, usando los submódulos ya públicos.
- No se agregan tests unitarios — mismo criterio que `diagnose_ctc_posteriors.py`
  (script de diagnóstico/throwaway, no código de producción).
- No se decide la solución (recortar BiLSTM, agrandar GCN, etc.) en este documento;
  solo se diseña la medición que permitirá decidir.
