# EDA de datos crudos

- Raiz analizada: `/shared/Code/Sign-AI/data/raw`
- Archivos totales: **13,666** (54.77 GiB).
- Archivos con extension de video: **13,107**.
- Sidecars AppleDouble `._*` descartados: **262**.
- Videos candidatos reales: **12,845**.
- Videos con duracion valida: **12,845**; fallos de lectura: **0**.
- Duracion total: **39:18:09.95**.

## Resumen por dataset

| Dataset | Videos | Filas meta | Tipo de texto | Idioma | Mediana video (s) | P95 video (s) | Max video (s) | Outliers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset1 | 3200 | 3200 | glosas/signos aislados | espanol | 2.035 | 3.036 | 4.037 | 59 |
| dataset2 | 8459 | 8459 | frases | espanol | 7.407 | 24.228 | 89.833 | 427 |
| dataset3 | 38 | 38 | glosas/signos aislados | espanol | 1.467 | 3.78 | 4.238 | 0 |
| dataset4 | 4 | 1 | narraciones/transcripciones largas | espanol | 46.046 | 98.699 | 104.104 | 0 |
| dataset5 | 359 | 354 | glosas/signos aislados | espanol | 0.549 | 1.548 | 167.767 | 28 |
| dataset6 | 259 | 259 | narraciones/transcripciones largas | espanol con ruido multilingue | 143.658 | 598.672 | 4108.415 | 22 |
| dataset7 | 526 | 526 | glosas/signos aislados | espanol | 3.904 | 8.275 | 54.421 | 14 |

## Texto y correspondencia

| Dataset | Mediana palabras | P95 palabras | Max palabras | Labels unicos | Match exacto | Match normalizado |
| --- | --- | --- | --- | --- | --- | --- |
| dataset1 | 1.0 | 2.0 | 3.0 | 64 | 3200/3200 | 3200/3200 |
| dataset2 | 13.0 | 37.0 | 153.0 | 8016 | 8459/8459 | 8459/8459 |
| dataset3 | 1.0 | 1.0 | 1.0 | 38 | 37/38 | 38/38 |
| dataset4 | 205.0 | 205.0 | 205.0 | 1 | 0/1 | 1/1 |
| dataset5 | 1.0 | 3.0 | 5.0 | 214 | 354/354 | 354/354 |
| dataset6 | 220.0 | 693.1 | 2590.0 | 252 | 0/259 | 259/259 |
| dataset7 | 1.0 | 1.0 | 2.0 | 144 | 517/526 | 526/526 |

## Notas de interpretacion

- `dataset1`, `dataset3`, `dataset5` y `dataset7` contienen principalmente glosas o signos aislados.
- `dataset2` contiene frases segmentadas; `dataset4` y `dataset6` contienen narraciones o transcripciones largas.
- El idioma dominante de las anotaciones es espanol. La etiqueta `espanol con ruido multilingue` indica caracteres de otros alfabetos, probablemente errores de transcripcion automatica.
- Los outliers se calculan dentro de cada dataset con la regla IQR (fuera de Q1 - 1.5*IQR o Q3 + 1.5*IQR).
- Los matches normalizados ignoran mayusculas, tildes, espacios y signos para detectar inconsistencias de nombres sin confundirlas con faltantes.
- Revisar `duration_outliers.csv`, `label_outliers.csv` y `metadata_unmatched.csv` antes de entrenar.
