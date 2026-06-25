# Imitator A1 token-ID robustness - 2026-06-25

## Resumen

Se verifico el run `imitator_A1_alpha1_20260625_163403`, entrenado como
Imitator real dataset1 -> token IDs de Gemma con alpha oracle
(`teacher_alpha`). El entrenamiento termino, paso tests y produjo el commit
`959bbfd Add dataset1 Gemma token Imitator prototype`.

Resultado del mejor checkpoint segun el val sintetico del trainer:

| epoch | token top1 | token top5 | exact sequence | val loss |
|---:|---:|---:|---:|---:|
| 28 | 0.901 | 0.980 | 0.850 | 0.761 |

Para robustez se uso un protocolo deterministico sobre los 640 clips del split
validation estratificado de dataset1. Por eso el baseline diagnostico es algo
mas conservador que el val sintetico del trainer.

## Ablaciones ejecutadas

Artefacto completo:

- `../outputs/v126_temporal/imitator_A1_alpha1_20260625_163403/robustness.json`

### 1. Velocidad / escalado temporal

`temporal_scale < 1` acorta clips; `> 1` alarga clips.

| scale | token top1 | exact | chrF | BLEU |
|---:|---:|---:|---:|---:|
| 0.50 | 0.871 | 0.820 | 86.39 | 82.60 |
| 0.75 | 0.874 | 0.819 | 86.70 | 82.60 |
| 1.00 | 0.876 | 0.817 | 86.65 | 82.48 |
| 1.25 | 0.875 | 0.816 | 86.46 | 82.32 |
| 1.50 | 0.871 | 0.811 | 85.99 | 81.85 |
| 2.00 | 0.861 | 0.795 | 84.96 | 80.43 |

Conclusion: A1 es razonablemente robusto a velocidad. La caida fuerte no esta
en temporal scaling moderado; incluso 2x solo baja exact de 0.817 a 0.795.

### 2. Ruido espacial en keypoints

| noise std | token top1 | exact | chrF | BLEU |
|---:|---:|---:|---:|---:|
| 0.00 | 0.876 | 0.817 | 86.65 | 82.48 |
| 0.01 | 0.874 | 0.816 | 86.52 | 82.32 |
| 0.03 | 0.866 | 0.806 | 85.74 | 81.44 |
| 0.05 | 0.860 | 0.798 | 85.02 | 80.65 |

Conclusion: tolera jitter continuo moderado. Esto sugiere que la normalizacion y
ST-GCN no son fragiles ante pequeno ruido de coordenadas.

### 3. Dropout de keypoints completos

| drop rate | token top1 | exact | chrF | BLEU |
|---:|---:|---:|---:|---:|
| 0.00 | 0.876 | 0.817 | 86.65 | 82.48 |
| 0.05 | 0.748 | 0.656 | 72.66 | 66.58 |
| 0.10 | 0.594 | 0.491 | 58.93 | 50.19 |
| 0.20 | 0.303 | 0.223 | 32.13 | 23.25 |

Conclusion: la mayor fragilidad no es jitter sino ausencia de articulaciones.
A1 depende mucho de trayectorias especificas; conviene entrenar con keypoint
dropout estructurado y/o masking por articulacion.

### 4. Desglose por signante

Peores signantes en baseline scale=1.0:

| signer | samples | exact | chrF | BLEU |
|---:|---:|---:|---:|---:|
| 3 | 61 | 0.607 | 69.90 | 61.56 |
| 1 | 66 | 0.788 | 85.82 | 80.15 |
| 5 | 63 | 0.794 | 83.44 | 79.85 |
| 2 | 62 | 0.823 | 86.91 | 82.70 |
| 9 | 52 | 0.827 | 86.77 | 82.69 |

Mejores signantes:

| signer | samples | exact | chrF | BLEU |
|---:|---:|---:|---:|---:|
| 8 | 65 | 0.908 | 93.87 | 91.30 |
| 4 | 79 | 0.886 | 92.17 | 89.94 |
| 10 | 62 | 0.855 | 90.45 | 87.26 |

Conclusion: la limitacion principal observada es variabilidad por signante. El
signante 3 es el caso critico y debe convertirse en el primer LOSO oficial.

## Lectura metodologica

La evaluacion de robustez en SLR/SLT se suele separar en:

- signer-dependent vs signer-independent vs unseen-sentence;
- perturbaciones/augmentations temporales;
- robustez ante ruido o cambios de pose/keypoints;
- metricas de secuencia/texto, como exactitud token-level, BLEU/chrF o WER.

Esto coincide con los benchmarks CSLR recientes, que resaltan que signer-indep
es la prueba real de generalizacion a usuarios no vistos. Tambien coincide con
trabajos de signer-diversity augmentation, donde la diversidad de signantes y
perturbaciones controladas se usan para mejorar generalizacion.

## Proximos pasos recomendados

1. Ejecutar `Imitator A1-LOSO-signer3`: entrenar excluyendo signante 3 y evaluar
   solo signante 3. Es la prueba mas importante antes de paper.
2. Agregar entrenamiento con `keypoint_drop_rate` estructurado 0.05 y 0.10,
   idealmente evitando borrar simultaneamente ambas manos completas.
3. Entrenar A1 con augment temporal explicito aunque el modelo ya es bastante
   estable a velocidad; se puede mantener como ablation secundaria.
4. Para el paper, reportar dos tablas separadas:
   - teacher-alpha token imitation en split estandar;
   - robustness/LOSO, dejando claro que learned-CIF es todavia experimental.
