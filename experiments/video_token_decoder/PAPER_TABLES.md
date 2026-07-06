# Imitator E1 — tablas maestras

Generado solo desde artefactos versionados/hasheados. `—` significa que la corrida o el artefacto no existe; no se imputa.

## Comparación principal

| Fold | Rol | CIF exact % | AR-base exact % | E1 exact % | E1 edit | Orden | W/L/T vs CIF | p bilateral |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | development | — | — | — | — | — | — | — |
| 2 | development | — | — | — | — | — | — | — |
| 3 | development | — | — | — | — | — | — | — |
| 4 | development | 2.3 | 2.2 | 18.1 | 63.0 | 98.6 | 144/3/749 | 5.94e-39 |
| 5 | development | 4.9 | 2.3 | 19.8 | 61.8 | 98.9 | 140/7/749 | 3e-33 |
| 6 | development | 2.6 | 1.8 | 16.9 | 58.5 | 98.6 | 131/3/762 | 3.68e-35 |
| 7 | confirmatory | 1.8 | — | 12.5 | 53.0 | 98.2 | 99/3/794 | 6.98e-26 |
| 8 | confirmatory | 2.7 | — | 14.1 | 58.7 | 98.9 | 108/6/782 | 2.72e-25 |

## Decisión confirmatoria

- H1: PASA — folds observados: [True, True]
- H2: PASA — folds observados: [True, True]
- H3: PASA — folds observados: [True, True]

**Confirmación completa:** PASA

## Ablations de desarrollo

| Fold | Variante | Estado | strict_exact % | edit_sim % |
|---:|---|---|---:|---:|
| 4 | E1 | available | 18.1 | 63.0 |
| 4 | E2_CTC | available | 17.9 | 61.4 |
| 4 | E3_unfreeze | available | 19.5 | 64.8 |
| 5 | E1 | available | 19.8 | 61.8 |
| 5 | E2_CTC | not_run_or_artifact_missing | — | — |
| 5 | E3_unfreeze | not_run_or_artifact_missing | — | — |
| 6 | E1 | available | 16.9 | 58.5 |
| 6 | E2_CTC | not_run_or_artifact_missing | — | — |
| 6 | E3_unfreeze | not_run_or_artifact_missing | — | — |
