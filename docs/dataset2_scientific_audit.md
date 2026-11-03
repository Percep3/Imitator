# Auditoría científica de dataset2 (LSA → español)

- Fecha: 2026-10-19. Repo `Sign-chris` (Imitator-E1), rama `v125-temporal-prompt-alignment`, HEAD `a7db233a86eff014459bedecb470e1a7c4141add`.
- Alcance: decidir si dataset2 aporta supervisión lo bastante fiable para entrenar traducción continua LSA → español. No se entrenó nada, no hubo inferencia en GPU ni descargas, y no se modificó ningún dato (HDF5 abierto con `h5py.File(p, "r")`).
- Etiquetas de evidencia:
  - **VERIFIED**: inspección directa o cómputo reproducible.
  - **DOCUMENTED**: consta en docs, configs, memoria o reportes, sin re-ejecutar.
  - **INFERRED**: interpretación plausible.
  - **UNKNOWN**: no determinable con lo disponible.
- Notas de trabajo fuente (todas bajo `artifacts/dataset2_audit/_work/`): `01_provenance.md`, `02_visual.md`, `03_text_split.md`, `04_experiments_impact.md`, `05_sampling.md`, más `artifacts/dataset2_audit/human_review_rubric.md`. En adelante se citan como `01:§`, `02:§`, etc., o como `archivo:líneas`.
- Advertencia central: **"pasa los chequeos computacionales" no equivale a "certificado lingüísticamente"**. Ningún clip ha sido revisado por anotadores competentes en LSA: 0 de 5.600.

> Uso de IA: este documento fue redactado con asistencia de IA generativa (Claude) a partir de notas de auditoría generadas también con IA. Antes de usarlo en decisiones o publicaciones, hay que validarlo con expertos en LSA y en el dominio. Los transcripts de dataset2 provienen de un canal público (CN Sordos), pero los videos muestran personas identificables. No se copiaron videos fuera de su ubicación, y las previsualizaciones privadas están en `artifacts/dataset2_audit/_private_previews/`.

---

## 1. Resumen ejecutivo

1. dataset2, tal como lo usan v116–v125, es un HDF5 (`dataset_v6_unsloth.hdf5`) con **5.600 clips**:
   - keypoints RTMPose 2D de forma (T, 134, 2), sin scores de confianza;
   - una etiqueta en español en minúsculas;
   - token IDs y embeddings de Gemma-3n, y `text_ctx`;
   - `video_id` y `source_group`.

   Los clips provienen de **8.459 clips MP4 sin audio** cortados de **64 videos padre** de CN Sordos, el noticiero semanal en LSA más una serie de ecología y piezas sueltas (conteos VERIFIED, `01:§I,§P1`; `01_h5_provenance.json`; canal CN Sordos DOCUMENTED, `contrastive_v118f.toml:1-8`).
2. El filtro de integridad v125 conserva **5.138** clips y excluye 462: 413 truncados, 29 por min_tokens, 16 por frames_per_token y 4 por min_frames. El split agrupado por video padre es **4.112/513/513**, con **0 grupos compartidos entre splits**. Ambos resultados se recomputaron y coinciden exactamente con los artefactos congelados (VERIFIED, `02:§Integrity filter`; `03:§4.1-4.2`).
3. **Correspondencia temporal (nivel B): UNKNOWN.** El `meta.csv` original tenía `start`, `end` y `splits=(texto, start, end)` por clip. Se borró el 2026-09-10 y solo sobreviven 10 filas en un notebook de git (VERIFIED, `01:§P2`). No existe ningún SRT/VTT de dataset2 y los clips no tienen audio. La cifra histórica de "3.088 clips desalineados >1 s" **carece de soporte**: es un comentario en un TOML, sin procedimiento, sin denominador y sin lista de clips (`01:§V`).
4. **Correspondencia semántica (nivel C): UNKNOWN.** Hay 0 clips anotados. El texto es español escrito de estilo subtítulo, con números, siglas, paréntesis y marcas de hablante como "ambos:" (VERIFIED, `03:§3.3`). Que no corresponde uno a uno con lo señado es INFERRED.
5. Hallazgo visual principal: el extractor agrupa las detecciones por índice de detección y no por identidad rastreada (`build_dataset2_h5.py:52-63`). Esto produce:
   - **saltos de identidad sospechosos en 1.728 clips**, 1.461 de ellos dentro del conjunto kept (heurística no calibrada, `02:§Identity switches`);
   - **frames descartados en silencio**: 198 clips kept tienen una razón de frames entre 0,9 y 1,0.

   El filtro v125 no detecta ninguno de los dos problemas (VERIFIED).
6. Los experimentos v116–v119 usaron splits aleatorios **a nivel de clip**: el 100% de los clips de val compartía video padre con train (VERIFIED, `04:§0.2`). Además, **v117 es bit-idéntico a v116 durante el entrenamiento**: InfoNCE = 0 en 25/25 épocas (VERIFIED, `04:§0.1`). La fuga infla las métricas, así que no explica los fracasos. v118 tampoco ajusta su propio train (R@1 de train 0,73%), lo que deja abierta una causa de modelo u optimización (INFERRED, `04:§0.6`).
7. Veredicto: con la evidencia actual **no se puede justificar todavía un experimento de traducción continua como prueba de capacidad SLT**. Sí se justifica un experimento mínimo de **separación datos-vs-modelo** (§12, E1) sobre el split v125, en paralelo con la revisión humana del piloto (§9).

### Preguntas de cierre

| # | Pregunta | Respuesta corta | Tag | § |
|---|---|---|---|---|
| 1 | ¿Qué contiene realmente dataset2? | 5.600 pares (keypoints 2D 134 pts, sin confianza; etiqueta en español en minúsculas) de 5.600 clips únicos de 8.459, de 64 videos padre de CN Sordos | VERIFIED (canal: DOCUMENTED) | 3, 4 |
| 2 | ¿Procedencia, cómo se construyeron los pares? | Clips cortados externamente en 2023 a partir de cues de subtítulos (start/end/splits en `meta.csv`, hoy borrado). Etiqueta = concatenación de cues (INFERRED). Origen del subtítulo (manual, auto o ASR) UNKNOWN. Selección: shuffle seed 23 y primeros n | VERIFIED / INFERRED / UNKNOWN | 2, 3 |
| 3 | ¿Ejemplos usables tras los filtros computacionales? | 5.138 kept; 3.590 kept sin advertencias de nivel A (2.740/417/433 por split). Ninguno está certificado lingüísticamente | VERIFIED | 6 |
| 4 | ¿Qué se sabe de la correspondencia temporal? | Solo que existían timestamps de cues y que los cues vecinos distan ~0,06–0,08 s en las filas visibles. Desfase por clip UNKNOWN. "3.088 >1 s" sin soporte | UNKNOWN | 7 |
| 5 | ¿Y de la correspondencia semántica? | Nada medido: 0 anotaciones. Hay indicios textuales de añadidos no señados (números, paréntesis, "ambos:") | UNKNOWN | 7 |
| 6 | ¿Riesgos de fuga train/val/test? | v125: 0 grupos y 0 video_ids compartidos; texto exacto compartido 2,5% val / 1,2% test (fórmulas). Riesgo residual: mismos intérpretes y serie entre splits, y reuso de metraje entre grupos (dentro de train hoy). v116–v119: fuga total por episodio | VERIFIED / INFERRED | 8 |
| 7 | ¿Problemas de datos que pudieron afectar a v116–v119? | Truncado a 250 (v116–v118 runs 1–3), extracción parcial (413 clips sin filtrar en v118f/v119), saltos de identidad, split por clip, objetivo CTC monótono frente a texto de subtítulo. Ninguno explica por sí solo los resultados | VERIFIED / INFERRED | 10 |
| 8 | ¿Qué evidencia necesita la revisión humana LSA? | Muestra de 80 clips probabilística + 42 intencional; rúbrica de 6 campos más desfase; piloto de 20 clips con doble revisor y α ≥ 0,67 | VERIFIED (diseño) | 9 |
| 9 | ¿Se justifica un experimento nuevo de traducción continua? | Todavía no como prueba de SLT. Sí un control de separación datos-vs-modelo (E1). La traducción queda condicionada a E1 y al piloto humano | INFERRED | 11, 12 |
| 10 | ¿Siguiente experimento mínimo, controlado y reproducible? | E1: sobreajuste a 64 pares + pares reales vs barajados + control solo-longitud sobre el split v125 kept, con encoder existente y contraste con B > 1 y gradiente verificado | proposal | 12 |

---

## 2. Procedencia y entorno

| Ítem | Valor | Tag | Evidencia |
|---|---|---|---|
| Fecha / HEAD | 2026-10-19; `a7db233` en `v125-temporal-prompt-alignment` | VERIFIED | `01:§E` |
| Entorno conda `Sign-env` | Vacío (solo `conda-meta/`), vaciado con `conda remove --all --keep-env` el 2026-08-08 | VERIFIED | `01:§E1` |
| Intérprete usado | `/shared/Code/nakato/projects/VQ_RL/.venv/bin/python` (py3.13, h5py 3.16, numpy 2.3.1, torch 2.13 solo en CPU), en modo lectura; no se instaló nada | VERIFIED | `AUDIT_SPEC.md:30-32`; torch en `04_experiments_split_stats.json:torch` |
| Sondeo de video | `ffprobe` 6.1.1 en lugar de cv2 (ninguna env tiene cv2). `nb_frames` coincide con `-count_packets` en 5.600/5.600 | VERIFIED | `02:§Environment` |
| Código frente a artefactos | La lógica en HEAD de todos los scripts v125 de datos es igual a la del freeze `c4113c3` (solo cambian rutas). La extensión de 5.000 clips se corrió con `build_dataset2_h5.py`, que difiere de HEAD solo en el LLM por defecto y en la fase text_ctx | VERIFIED | `01:§C` |
| Hashes v125 | Los 7 artefactos congelados coinciden con `v125_artifacts.json` (HDF5 `24c9aa40…1d38`, manifest `04c3a7c1…bad`, split `6669635a…f9`, tokenizer `b6c35ee6…a5d`) | VERIFIED | `01:§I`; `03:§1` |
| Línea de comando de la extensión | No quedó registrada | UNKNOWN | `01:§C` |

---

## 3. Reconstrucción de dataset2

1. **Fuente.** 8.459 clips MP4 (h264, sin audio) con mtimes de 2023-07, cortados por un pipeline externo que no está en ningún repo accesible (VERIFIED: ausencia; origen UNKNOWN, `01:§P1-P2`). El nombre sigue el patrón `<slug-padre>_<n>.mp4`, que da 64 padres, de 3 a 439 clips cada uno (VERIFIED). El canal "CN Sordos" está DOCUMENTED (`contrastive_v118f.toml:1-8`). La lengua es LSA, DOCUMENTED (`docs/audits/v125_language_audit.md:5-9`) pero no certificada.
2. **Metadatos originales.** `meta.csv` tenía 8.459 filas y 14 columnas: `id, label, video, playlist, start, end, duration, splits, prev_delta, post_delta, signers_amount, infered_signer, infered_signer_confidence, movement_per_signer`. Se recuperó de la salida del notebook en `git show fef07a3:Sign-Dataset.ipynb`; solo son visibles 10 filas (VERIFIED, `01:§P2`). De esas filas:
   - `splits` es una lista de cues `(texto, start, end)`. Que la etiqueta concatene varios cues es INFERRED.
   - `duration` ≠ `end − start` en las filas con varios cues; su significado es UNKNOWN.
   - `infered_signer` alterna entre clips consecutivos y `signers_amount`=18 aparece en una fila, es decir, hay ruido del detector (VERIFIED en las filas visibles).
3. **Selección de 5.600.** `select_clips` (`build_dataset2_h5.py:33-43`) descarta etiquetas vacías, exige que exista `<id>.mp4`, pasa la etiqueta a minúsculas (`:41`), hace `random.Random(23).shuffle` y toma los primeros n. Hoy no es reproducible porque `meta.csv` fue borrado (VERIFIED).
4. **Keypoints.** RTMPose wholebody (rtmlib `to_openpose`, 134 pts) hasta `--max-frames`. Los scores se descartan (`:59`). Las detecciones se agrupan por índice `p` de detección (`:52-63`) y se elige el índice con más movimiento de muñeca. Los frames en que ese índice no aparece se omiten (VERIFIED, código). Que esto produzca saltos de identidad y compresión temporal está VERIFIED estadísticamente y es INFERRED como mecanismo (§6).
5. **Texto y objetivos.** `rebuild_dataset2_alignment_targets_v125.py:42-69` regenera `token_ids` = Gemma(`normalize_alignment_text(label)`, sin tokens especiales) y `embeddings` = `table[ids]`. La etiqueta original se conserva (VERIFIED, `03:§2`).
6. **Backfill.** `backfill_dataset2_metadata.py` añade `video_id`, `source_group` (columna `video` de meta), `frame_count`, `token_count` y `word_count`. Además lanza `ValueError` si alguna etiqueta almacenada difiere del orden de `select_clips(5600, 23)` (`:59-64`). Que el backfill terminara bien es INFERRED, y queda respaldado por la consistencia interna verificada en §4.
7. **Historia del archivo** (mutado in situ, `04:§1`):
   1. 600 clips con tope de 250 frames;
   2. 228 clips re-extraídos con tope 1400 el 2026-06-17 (DOCUMENTED);
   3. extensión con *resumable append* a 5.600 clips hacia el 2026-06-19;
   4. backfill v125 (mtime 2026-06-23 10:49).

   No hay hash anterior a v125, así que la identidad byte a byte de los keypoints vistos por v118f/v119 es INFERRED.
8. **Claves 0–599 = el subconjunto histórico de 600.** La evidencia:
   - entre las claves 0–599, exactamente 228 tienen T ≥ 250, con media 467,3 (468,3 para los 227 con T > 250) y máximo 1.312, lo que coincide con el registro del fix v118 (228, ~469, 1.312);
   - el shuffle con prefijo determinista y el chequeo de etiquetas del backfill son consistentes con esa identidad.

   Estado: INFERRED (fuerte), `01:§HDF5`; `04:§0.3`. Esto **contradice** el comentario de `scripts/train/train_ctc_v119.py:204-209` y la cabecera de `experiments/v119_ctc/ctc_v119_subset1k.toml:2-6` ("las claves 0..599 NO corresponden"), que se basaban en una comparación de medias de frames. Ver §14, tabla de contradicciones.

---

## 4. Inventario real de archivos

| Ruta | Existe | Tamaño | sha256 | Nota | Tag |
|---|---|---|---|---|---|
| `/shared/Code/Sign-AI/data/processed/dataset_v6_unsloth.hdf5` | sí | 2.244.045.489 B | `24c9aa401b44…1d38` | 10 grupos × 5.600 claves | VERIFIED |
| `…/dataset2_manifest_v125.json` | sí | 439.755 B | `04c3a7c1…bad` | | VERIFIED |
| `…/dataset2_split_v125.json` | sí | 60.749 B | `6669635a…f9` | | VERIFIED |
| `…/gemma3n_embed_table.pt` | sí | 1.074.792.061 B | `20437971…d7` | | VERIFIED |
| `…/v125_gemma_oracle.json` | sí | 58.147 B | `b037bfca…a75be` | 100 etiquetas, iguales a las del H5 en 100/100 | VERIFIED |
| `/shared/Code/Sign-AI/data/raw/dataset2/videos/` | sí | 28,58 GB | huella del listado `3062dde4…` | 8.459 `.mp4` + 1 `._*` (AppleDouble) | VERIFIED |
| `…/raw/dataset2/meta.csv` | **no** | | | borrado 2026-09-10 | VERIFIED |
| `…/raw/dataset2/keypoints.h5` (MediaPipe legado) | **no** | | | borrado 2026-09-10 | VERIFIED |
| `…/dataset_v6_unsloth.hdf5.bak_pre250fix` | **no** | | | backup del fix v118, perdido | VERIFIED |
| `artifacts/data_eda/video_metadata.csv` | sí | 2.815.816 B | `67689c12…a18` | ffprobe por video, 2026-06-08 | VERIFIED |

- Tabla completa: `_work/01_inventory.tsv`.
- Búsquedas negativas: en `/shared/Code/Sign-AI` y `/home/nakato` no existe ningún SRT/VTT/JSON de subtítulos para dataset2. Las 27 SRT presentes son de dataset5 (VERIFIED, `01:§I`).

Contenido del HDF5 (VERIFIED, `01_h5_provenance.json`; `03:§3.1`):
- Claves contiguas `0..5599`, iguales en los 10 grupos e iguales a las `clip_ids` del manifest.
- `keypoints` con forma (T, 134, 2) para todos los clips, y `frame_count == T` en 5.600/5.600.
- `video_id`: 5.600 únicos, y los 5.600 existen en `videos/`.
- `source_group` es igual al padre del `video_id` en 5.600/5.600: hay 64 grupos, los mismos 64 padres del directorio raw.
- 0 fallos en las 9 comprobaciones de consistencia texto↔token↔embedding (`text_split_stats.json:consistency_failures`).

---

## 5. Estadísticas visuales y textuales

Fuente: `artifacts/dataset2_audit/dataset_statistics.json` (fusiona `visual_stats.json`, `text_split_stats.json`, `01_h5_provenance.json`, `04_experiments_split_stats.json` y la muestra). Todo VERIFIED salvo que se indique otra cosa.

| Magnitud | p5 | p50 | p95 | máx | media |
|---|---|---|---|---|---|
| T en el H5 (frames), 5.600 clips | 65 | 215 | 711 | 1.400 | 278,7 |
| Duración del video (s), 5.600 clips | 2,47 | 7,44 | 24,1 | 89,8 | 9,59 |
| Palabras (`vocab.tokenize`), todos / kept | 3 / 3 | 13 / 13 | 37 / 38 | 153 / 111 | 16,1 / 16,3 |
| Tokens Gemma, texto normalizado | 4 / 4 | 16 / 16 | 46 / 46 | 169 / 122 | 19,5 / 19,7 |
| Tokens Gemma, etiqueta cruda | 6 / 6 | 18 / 18 | 50 / 50 | 196 / 137 | 22,1 / 22,3 |

Otros resultados:
- fps: 29,97 en 5.136 clips, 30 en 442 y 24 en 22. Resolución mayoritaria 1920×1080. **has_audio = False en los 8.459 clips**, de modo que no es posible re-derivar el texto por ASR desde los clips (`01:§I`).
- Frames por palabra (5.600 clips): mediana 16,7 (~0,56 s), p5 8,45; 234 clips con menos de 8 y 56 con menos de 4 (`04:§3`).
- Texto:
  - 0 mayúsculas, porque el pipeline pasa todo a minúsculas (`build_dataset2_h5.py:41`), así que se pierden nombres propios y siglas.
  - 844 clips con dígitos, 506 con comillas o paréntesis y 11 con "ambos:".
  - 253 clips dan secuencias de palabras distintas según se normalicen con `vocab.tokenize` o con `normalize_alignment_text`, una inconsistencia entre la unidad de WER y la de alineamiento (`03:§3.3`).
- Duplicados (kept, texto normalizado): 4.925 textos distintos y 213 clips redundantes. Son sobre todo fórmulas del programa (`03:§3.4`).
- Léxico (5.600, split aleatorio de v119): 53,6% de los tipos de train son singletons; el top-20 acumula el 35,9% de los tokens; OOV de val 7,2% (`04:§3`).

---

## 6. Auditoría de integridad (nivel A, computacional)

**Filtro v125.** Reaplicando `integrity_filter.py` sin cambios, con ffprobe en lugar de cv2, se obtienen kept 5.138 y excluidos 462 (413 truncados con razón < 0,9, 29 min_tokens, 16 frames_per_token, 4 min_frames). Hay **0 discrepancias por clip** con el manifest almacenado (VERIFIED, `02:§Integrity filter`). El filtro solo revisa NaN/Inf, longitud, tokens y truncado. **No revisa saltos de identidad, frames descartados en medio del clip ni manos fuera de cuadro.**

| Defecto (heurísticas de `02`, no calibradas salvo indicación) | Todos | Kept | Tag |
|---|---|---|---|
| NaN / Inf / puntos exactamente cero | 0 / 0 / 0 | 0 | VERIFIED |
| Razón T/frames del video < 0,9 | 434 | 0 (excluidos: 413 con causa primera truncated + 21 con otra causa primera) | VERIFIED |
| Razón en [0,9, 1,0), frames descartados en silencio | — | 198 | VERIFIED |
| `identity_switch_suspect` (fracción de saltos > 5% y ≥ ½ con retorno A-B-A) | 1.728 | 1.461 (28,4%) | VERIFIED (cómputo); identidad INFERRED |
| Saltos de cuello > 0,5 de ancho de hombros | 3.187 clips, 59.025 saltos, 81,4% con retorno en ≤ 30 frames | 2.822 | VERIFIED |
| Manos > 20% fuera de cuadro | 37 | 32 | VERIFIED |
| Hombros degenerados (< 1 px) | 25 | 20 | VERIFIED |
| Cola cortada en 1.400 (T = 1.400 y video > 1.400 frames) | 12 | 5 | VERIFIED |

Notas sobre la tabla:
- **Revisión visual.** Solo un clip se revisó visualmente (clip 430, dos intérpretes; el esqueleto salta de persona). La precisión del flag de identidad es UNKNOWN y se calibrará con el bucket intencional de §9.
- **Distribución por split.** Los clips kept con `identity_switch_suspect` o con otra advertencia de nivel A se reparten así: train 1.372/4.112, val 96/513, test 80/513 (VERIFIED, `integrity_report.csv`). Que train concentre más casos coincide con su mayor proporción de noticiero con dos intérpretes en pantalla (INFERRED).
- **Capa de 111 puntos asimétrica** (`data_augmentation.py:142-151`). Se descarta la raíz de la muñeca izquierda (92) y la punta del meñique derecho (133). `LH_WRIST=93` (`build_dataset2_h5.py:30`) y `wrist_idx=[71,91]` (`data_augmentation.py:173`) apuntan a un CMC del pulgar y no a la muñeca. El orden de los índices está INFERRED a partir de las convenciones de rtmlib y está VERIFIED geométricamente en 300 clips (`02:§Keypoint layout`). Esto supersede el INFERRED de `01:§V`.

**Estado por clip.** `artifacts/dataset2_audit/integrity_report.csv` tiene 5.600 filas, generadas por `scripts/merge_integrity_report.py`:
- `level_A_status`:
  - `excluded_v125:<causa>`: 462;
  - `kept_warn`: 1.548 (al menos una de las advertencias `identity_switch_suspect`, `hand_offframe_gt20pct`, `degenerate_shoulders`, `tail_cut_at_1400` o `frames_dropped_ratio_lt1`);
  - `kept_no_warn`: 3.590.
- `level_B_status` = `unknown_no_timestamps` para todos los clips.
- `level_C_status` = `not_annotated` para todos los clips.
- `body_jump` y `scale_jump` no cuentan como advertencia porque afectan a más de la mitad de los clips y no discriminan. Siguen visibles en `visual_flags`.

---

## 7. Evaluación de alineamiento disponible

| Nivel | Qué mide | Estado | Tag |
|---|---|---|---|
| **A, computacional** | Integridad de arrays, consistencia texto↔token, cobertura de frames, plausibilidad del esqueleto | Hecho (§6). 5.138 kept; 3.590 sin advertencias | VERIFIED |
| **B, temporal** | Si el tramo señado que corresponde al texto cae dentro de los límites del clip, y con qué desfase | **No evaluable con los datos disponibles.** No hay timestamps por clip (meta.csv borrado), ni SRT/VTT, ni audio. Solo hay 10 filas históricas, con huecos entre cues vecinos de 0,06–0,08 s, de modo que cualquier desfase de la interpretación respecto del subtítulo empuja señas al clip vecino (aritmética VERIFIED en las filas visibles; riesgo INFERRED, `01:§P2`) | UNKNOWN |
| **C, semántico** | Si el español es una traducción o paráfrasis fiel de lo señado | **0 clips anotados.** Indicios textuales de añadidos no señados (números, teléfonos, expansiones de siglas, "ambos:") | UNKNOWN |

- **La cifra "3.088 clips desalineados > 1 s" no tiene soporte.** Solo aparece como comentario en `experiments/v115_ce_vocab/sigreg.toml:18-19` (introducido en `867c563`). La clave `check_alignment` no la consume ningún código. Tampoco se encontraron procedimiento, denominador ni lista de clips en git (`git log --all -S3088`), en los docs, en los outputs ni en la memoria (`01:§V`). **No debe citarse como hecho.**
- Que los clips "pasen el nivel A" no dice nada sobre los niveles B y C.

---

## 8. Riesgo de fuga

| Riesgo | Split v125 (agrupado) | Splits v116–v119 (aleatorios por clip) | Tag |
|---|---|---|---|
| Mismo video padre en train y eval | 0 grupos compartidos (40/12/12 grupos). El split se regenera de forma idéntica con `make_grouped_split(seed=23)` | 100% de val comparte padre con train (5.600 clips: 1.119/1.119 de val, 62 de 64 grupos en val; 600 clips: 119/119, 49 de 62 grupos) | VERIFIED |
| Mismo `video_id` o mismo array de keypoints | 0 / 0 (sha1, hash cuantizado de movimiento y firma longitud+media) | — | VERIFIED |
| Mismo texto | Exacto: val 13/513 (2,53%), test 6/513 (1,17%). Normalizado: 3,31% / 1,56%. Son fórmulas de saludo | 66/1.119 exactos (5,9%) a 5.600 clips; 75/1.108 normalizados en v119 levers | VERIFIED |
| Metraje casi duplicado | 2 pares con cos ≥ 0,95, ninguno cruza splits. Hay reuso entre grupos distintos (p. ej. clips 2407↔3827, cos 0,957: "programa especial" recopila segmentos), por ahora todo dentro de train | no medido | VERIFIED (cómputo) / INFERRED (interpretación) |
| Mismo intérprete en train y eval | Probable: los presentadores ("lucía", "lautaro") aparecen en más de un split. No hay metadatos de intérprete | seguro, a nivel de episodio | INFERRED / UNKNOWN |
| Continuidad temática o temporal | Plausible: las ediciones semanales se intercalan en el tiempo entre splits | — | INFERRED |
| Cambio de dominio | Test: 48% serie ecología frente a 1,5% en train. No es fuga, pero sesga test | — | VERIFIED |

- Conclusión: el split v125 es **independiente por episodio pero dependiente de intérprete y de serie**. Mide generalización a episodios nuevos de los mismos productores e intérpretes, no a intérpretes nuevos.
- Recomendaciones (no se modificó ningún split, `03:§5`):
  - reportar las métricas de test por serie;
  - fusionar en una sola unidad de independencia los grupos con reuso demostrado de metraje;
  - considerar un holdout temporal para el noticiero.

---

## 9. Diseño de la revisión humana

Fuentes: `artifacts/dataset2_audit/sample_manifest.csv` (122 filas, sha256 `5ba98a52…de13`), `human_review_template.csv` (`ec3b3c05…63f0`), `human_review_rubric.md` y `_work/05_sampling.md`. Generador `scripts/build_review_sample.py` (seed 23, determinismo VERIFIED con `sha256sum -c`).

**Componente probabilístico (n = 80).** MAS estratificado sin reemplazo sobre los 5.138 kept.
- Estratos: split × tercil de duración, con cortes en 5,572 s y 10,544 s.
- Train va aproximadamente proporcional; val y test tienen un mínimo de 4 por estrato.
- Pesos N_h/n_h, con Σw = 5.138 y Kish n_eff = 76,2.
- Piloto de 20 clips (los primeros k_h sorteados en cada estrato): 8 de train, 6 de val y 6 de test.

**Componente intencional (n = 42).** 12 buckets: identidad, frames descartados, umbrales del filtro, colas de fpt, transcripts muy cortos o muy largos, saludos duplicados, reuso de metraje, rechazados por v125, serie eco en test y manos o hombros. **Nunca se mezcla con el probabilístico** y solo sirve para calibrar flags.

**Rúbrica**: `video_complete`, `signer_visible`, `temporal_alignment`, `semantic_correspondence`, `translation_completeness`, `annotation_confidence` y `estimated_offset_seconds`. El desfase se rellena solo con evidencia, nunca con 0 por defecto. Los revisores deben ser competentes en LSA: personas sordas usuarias de LSA o intérpretes certificados.
- En el piloto, 2 revisores independientes. Se reportan κ, Krippendorff α y bootstrap por clips, con umbral de α ≥ 0,67 en semántica y temporal antes de escalar.
- A los revisores se les ocultan los flags y el motivo del muestreo.

**Comprobación del protocolo piloto (VERIFIED).** Los 20 clips decodifican completos. En 19/20, T del keypoint coincide con los frames decodificados. En PR022 (clip 2565) T es 253 frente a 257 frames decodificados, un caso del mecanismo de descarte silencioso.

**Evidencia adicional que necesita la revisión.**
- Revisores reales y medición del tiempo por clip en el piloto (hoy UNKNOWN).
- Una decisión previa sobre cómo cuentan `unknown` y `not_assessable`.
- Para el nivel B, contexto del episodio alrededor del clip, lo que exige localizar el video padre. Descargarlo está fuera del alcance de esta auditoría y requiere aprobación del usuario.
- Opcional: etiquetas de identidad del presentador por grupo, para poder plantear una evaluación independiente del intérprete.

**Estado actual: 0 clips anotados**, así que la calidad semántica es UNKNOWN.

---

## 10. Implicaciones para v116–v119

Resumen de `04:§2-3`:

| Exp. | Resultado | Defecto de datos presente | ¿Lo explica? | Tag |
|---|---|---|---|---|
| v116 (CE-AR, prefijo suave → Gemma congelado) | chrF máx. 20,67; colapso del prefijo (cos 0,31 → 0,987); retr@1 = 0/119 (DOCUMENTED vía v117) | truncado a 250 en 228/600 (38%, las frases largas); split por clip | No: el 62% de los pares sin truncar tampoco muestra grounding | VERIFIED / DOCUMENTED / INFERRED |
| v117 (+InfoNCE) | **Bit-idéntico a v116** en 350/354 escalares; `train_infonce = 0` en 25/25 épocas (`sub_batch_size=1` produce una matriz 1×1 con CE = 0) | igual que v116 | La pregunta no aplica: el contraste nunca actuó | VERIFIED |
| v118 runs 1–3 (CLIP/VICReg) | R@1 ≤ 2,52%/119 | truncado a 250 | No: run 4 sin truncar sube solo a 2,94% (1 semilla) | VERIFIED |
| v118 run 4 / e run 7 | R@1 2,94% / 2,52%; **R@1 de train 0,73%/481** | desalineamiento UNKNOWN; pooling global | No ajusta su propio train, lo que apunta a una causa de optimización o arquitectura al menos tan plausible como los datos | VERIFIED / INFERRED |
| v118f run 8 (5.600) | R@1 0,268%/1.119 (3,0× azar) | incluye los 413 "truncados"; 100% de grupos de val en train | No: ~7,7% de clips parciales; la fuga infla | VERIFIED / INFERRED |
| v119a (STGCN+RoPE, CTC) | memoriza train (loss 32,3 → 1,26); WER de val 99,7–100% | CTC monótono frente a español de subtítulo; 53,6% singletons; 35,9% palabras función | Falla la generalización, no el ajuste | VERIFIED / INFERRED |
| v119 reemplazo (TCN+BiLSTM, CTC dual) | WER 99,25% → 100,23%; 98,2% de frames con argmax blank (DOCUMENTED) | igual | Colapsar a blank también es el óptimo con objetivos no alineables, así que "la culpa es del BiLSTM" no está probado | VERIFIED / INFERRED |
| v126 / A1–A4 / E1 | usan **dataset1**, no dataset2 | ninguno de dataset2 | — | VERIFIED |

Conclusiones:
- Ningún defecto de datos medido explica por sí solo los fracasos.
- Cualquier resultado positivo futuro debe re-medirse en el split v125 agrupado.
- El defecto "encoder sin PE ⇒ bolsa de señas" pertenece a VideoTokenDecoder sobre dataset1. **No aplica a v116–v119**, que usan RoPE, TCN o BiLSTM (VERIFIED, `04:§6`).
- La afirmación de la memoria "InfoNCE retrasó el colapso ~9 épocas" (`v117-no-grounding-verdict.md`) queda **refutada** (VERIFIED, `04:§5`).

---

## 11. Viabilidad SLT: escenarios A/B/C

**Estado actual: la calidad semántica y temporal es UNKNOWN** (0 clips anotados). Los escenarios siguientes son una **regla de decisión que se debe pre-registrar** antes de ver las anotaciones. No son un resultado.

**Variable primaria.** U = proporción ponderada (estimador de §5 de la rúbrica) de clips kept con:
- `semantic_correspondence ∈ {translation, paraphrase}`, y
- `temporal_alignment = aligned`.

Como análisis de sensibilidad se cuentan `unknown` y `not_assessable` de dos maneras: como fallo y como excluidos.

**Umbrales propuestos.** τ_C = 0,5 y τ_A = 0,8.
- τ_C = 0,5: por debajo, la mayoría de la supervisión es incorrecta o parcial, y un resultado positivo entrenado sobre todo el corpus no sería interpretable.
- τ_A = 0,8 es una convención (ruido ≤ 20%), no un valor derivado de literatura. **Deben fijarlo el equipo y los expertos LSA** antes de anotar.
- La parte no arbitraria es la siguiente. La decisión se toma sobre los **límites del IC 95%**, no sobre el punto. El n se elige para que la semiamplitud del IC sea menor que la mitad de la distancia entre umbrales (15 pp), de modo que el IC solo pueda cruzar un umbral a la vez.

**Semiamplitud Wilson al 95% (pp).** VERIFIED: calculada con `scripts/build_dataset_statistics.py`, `dataset_statistics.json:derived.wilson95_half_width_pp`.

| n | p = 0,5 | p = 0,7 | p = 0,8 | p = 0,9 | Comentario |
|---|---|---|---|---|---|
| 12 (val o test por separado) | ±24,6 | ±23,1 | ±21,0 | ±17,7 | no decide nada por split |
| 20 (piloto) | ±20,1 | ±18,7 | ±16,8 | ±13,7 | solo acuerdo y rúbrica |
| 76 (n_eff del diseño) | ±11,0 | ±10,1 | ±8,9 | ±6,9 | **muestra actual: separa escenarios distantes 30 pp** |
| 150 | ±7,9 | ±7,3 | ±6,4 | ±4,8 | |
| 200 | ±6,9 | ±6,3 | ±5,5 | ±4,2 | |
| 385 | ±5,0 | ±4,6 | ±4,0 | ±3,0 | necesario para separar umbrales a 10 pp |

**Coste de anotación.**
- El tiempo de revisor por clip es UNKNOWN y se medirá en el piloto. El coste es n_revisiones × t_clip.
- Muestra actual: 80 + 42 = 122 clips, más 20 segundas revisiones del piloto, es decir, 142 revisiones. Son 752 s de video probabilístico más 381 s intencional, unos 19 minutos de visionado bruto (VERIFIED, `dataset_statistics.json:derived.review_sample`).
- A título ilustrativo y no medido: con t_clip = 5 min serían unas 12 h de revisor; con 10 min, unas 24 h.
- Pasar a ±5 pp exigiría unos 305 clips probabilísticos más, unas 4,8 veces la muestra actual.

| Escenario | Criterio (IC 95% de U, ponderado por diseño) | Consecuencia | Datos usables implicados (sobre 5.138 kept) |
|---|---|---|---|
| **A, corpus usable tal cual** | límite inferior ≥ 0,8 | Entrenar sobre todo el kept v125 está justificado como supervisión. Un fracaso posterior apuntaría a modelo u objetivo | ≥ ~4.110 pares |
| **B, usable solo filtrado** | IC entre 0,5 y 0,8, o que cruce 0,8 | Entrenar solo sobre un subconjunto filtrado o certificado; ampliar la anotación (n ≈ 200, ±7 pp) para estimar U por estrato y calibrar filtros (flag de identidad, duración, fpt); evaluar en un subconjunto certificado | ~2.570–4.110 pares, sin identificar cuáles sin más anotación |
| **C, corpus no apto para SLT continua** | límite superior < 0,5 | No entrenar traducción continua sobre dataset2. Re-segmentar con los timestamps originales (hay que recuperar los subtítulos de los videos padre, lo que requiere aprobación) o cambiar de corpus. dataset2 queda como mucho para preentrenamiento auto-supervisado del encoder | < ~2.570 pares |

- Con n_eff ≈ 76, un IC que cruce τ_C o τ_A (p. ej. p̂ ∈ [0,4, 0,6]) **no decide**: obliga a ampliar la muestra antes de elegir escenario.
- Condición adicional, válida para todos los escenarios: si en el bucket intencional `identity_switch_suspect` (n = 5) la mayoría de los clips muestran cambio de persona, el nivel A de 1.461 clips kept queda en entredicho con independencia de U (§12, E2).

---

## 12. Experimentos recomendados (máximo 3; no se ejecuta ninguno en esta auditoría)

Prioridad: separar datos de modelo antes de cualquier arquitectura nueva. Se consideran los componentes existentes (encoder Imitator/`PrefixImitator` con RoPE, cabezales de v118, Gemma-3n congelado) y mT5 sin presuponer cuál es mejor. Hiperbólico/Geo-Sign queda como opción posterior, **solo si existe antes una línea base fiable** que supere sus controles.

### E1. Control datos-vs-modelo en el split v125 (prioridad 1)

| Campo | Contenido |
|---|---|
| Hipótesis | H1a: el pipeline puede memorizar 64 pares reales. Si no puede, el cuello de botella es de modelo u optimización. H1b: en val v125, pares reales superan a pares barajados y a un predictor solo-longitud. Si no los superan, no hay señal explotable a este N, o la supervisión es inválida |
| Datos | `dataset_v6_unsloth.hdf5`, kept v125, split `dataset2_split_v125.json` (4.112/513/513). Subconjuntos fijos (seed 23) de 32 y 64 pares de train para sobreajuste. Variante `kept_no_warn` (3.590 clips, `integrity_report.csv`). Test sellado y reportado por serie |
| Arquitectura | Encoder existente (`PrefixImitator`/Imitator con RoPE, `src/mslm/models/imitator.py`) más el cabezal de recuperación de v118 (`contrastive.py`). Sin componentes nuevos |
| Baseline | Azar = 1/|pool|; pares barajados con el mismo presupuesto; solo-longitud (ranking por \|log T − log tokens\| frente al texto); v118 run 4 como referencia histórica (no comparable, era split por clip) |
| Loss | InfoNCE simétrica con B > 1 real (≥ 32), con norma de gradiente de la pérdida contrastiva registrada y distinta de cero en cada época (evita repetir v117); VICReg de v118 como brazo secundario |
| Métricas | R@1/5/10 y rango mediano con IC bootstrap por clips; R@1 sobre el propio train (ajuste); rango efectivo y coseno por pares del embedding (colapso) |
| Controles | (i) pares barajados, (ii) solo-longitud o solo-texto, (iii) fuga: solo split v125 y reporte por serie, (iv) ≥ 3 semillas, (v) `kept` frente a `kept_no_warn` |
| Criterio de éxito | Sobreajuste: R@1 de train ≥ 0,9 en 64 pares. Las épocas que tarda en memorizar los pares barajados sirven de referencia: los pares reales deberían memorizarse antes. Señal: R@1 de val real > barajado y > solo-longitud con IC 95% disjuntos en ≥ 3 semillas. Umbrales a pre-registrar |
| Riesgo | Con 513 clips por split la potencia es baja, porque el azar a R@1 ronda 0,2%. Un éxito en recuperación no implica traducción. La dependencia de intérprete puede inflar val. Que falle el sobreajuste apunta al modelo, no a los datos, pero no dice cuál de los dos factores domina en val |

### E2. Re-extracción con seguimiento de persona (prioridad 2; requiere aprobación del usuario y GPU)

| Campo | Contenido |
|---|---|
| Hipótesis | Una parte sustancial del ruido visual (1.461 kept con `identity_switch_suspect`, 198 con frames descartados) viene del agrupamiento por índice de detección. Un tracker de identidad, sin descartar frames y conservando los scores, la reduce y mejora la métrica de E1 |
| Datos | Primero los 122 clips de la muestra de §9 más los 1.461 kept con flag de identidad y un control aleatorio del mismo tamaño sin flag. Después, todo el kept si el piloto mejora. Se escribe a un **HDF5 nuevo**, nunca sobre `dataset_v6_unsloth.hdf5` |
| Arquitectura | Extracción: RTMPose (rtmlib) más un tracker de identidad (IoU o ByteTrack-like), con selección del firmante por persistencia y movimiento, scores guardados y frames vacíos rellenados con máscara. Corregir el índice de la mano izquierda (92:113) en una capa nueva, sin cambiar la histórica. Modelo de evaluación: el de E1 sin cambios |
| Baseline | Keypoints actuales del mismo subconjunto |
| Loss | La de E1 |
| Métricas | Tasa de `identity_switch_suspect` y de saltos de cuello; razón T/frames; precisión del flag frente al bucket humano; ΔR@1 de E1 entre keypoints nuevos y actuales sobre el mismo split |
| Controles | Mismo split, semillas y presupuesto; control aleatorio sin flag; revisión visual de una muestra de clips re-extraídos |
| Criterio de éxito | Caída de la tasa de saltos con retorno A-B-A en ≥ 50% sobre los clips con flag, sin pérdida de cobertura de frames, y mejora de R@1 de E1 con IC disjunto. Si la tasa cae y R@1 no cambia, el ruido visual no es el factor limitante |
| Riesgo | Los pesos `local_models/Keypoint-Models` fueron retirados y habría que volver a obtenerlos. La heurística de selección de firmante sigue siendo frágil con dos intérpretes. El coste de cómputo es real |

### E3. Subconjunto certificado por humanos frente a subconjunto aleatorio (prioridad 3; depende de §9)

| Campo | Contenido |
|---|---|
| Hipótesis | Si el desalineamiento semántico o temporal limita el aprendizaje, el modelo de E1 rinde más sobre clips certificados `aligned` y `translation/paraphrase` que sobre clips de igual tamaño y duración no certificados o con fallo |
| Datos | Clips de val y test anotados según la rúbrica, ampliados si hace falta (la muestra actual solo tiene 24 clips probabilísticos de val+test), estratificados por etiqueta humana. Contraste: subconjunto aleatorio del mismo tamaño y distribución de duración |
| Arquitectura | El modelo de E1 entrenado en train v125 sin cambios; solo cambia el subconjunto de evaluación. Si E1 pasa, se añade un brazo generativo comparando prefijo → Gemma-3n congelado (la ruta v116, ahora en el split v125) con fine-tuning de mT5, ambos con el mismo encoder |
| Baseline | Puntuación del mismo modelo en el subconjunto aleatorio; pares barajados |
| Loss | Recuperación: la de E1. Generación: CE con teacher forcing sobre tokens del decodificador elegido |
| Métricas | R@k estratificado por etiqueta humana (aligned frente a no aligned; translation/paraphrase frente a partial/unrelated); chrF y BLEU solo en el brazo generativo, con un oráculo de texto como techo |
| Controles | Barajado; solo-texto (decodificador sin video) para el brazo generativo; fuga: solo split v125 y reporte por serie |
| Criterio de éxito | Diferencia certificado − aleatorio con IC 95% que excluye 0. Si no hay diferencia y E1 fue positivo, la calidad de las etiquetas no es el factor dominante. Brazo generativo: chrF con video > solo-texto con IC disjunto |
| Riesgo | Muestra certificada pequeña y costosa (tiempo por clip UNKNOWN). Sesgo del anotador. El brazo generativo solo es interpretable si E1 pasa. Hiperbólico/Geo-Sign no se plantea hasta tener una línea base que supere estos controles |

---

## 13. Evidencia, rutas, hashes y comandos

- Comandos exactos y hashes esperados: `artifacts/dataset2_audit/reproduction_commands.md`. Métodos y umbrales: `artifacts/dataset2_audit/audit_methodology.md`.
- Artefactos de esta auditoría (sha256 al 2026-10-19, HEAD `a7db233`):

| Archivo | sha256 |
|---|---|
| `artifacts/dataset2_audit/integrity_report.csv` | `8b72d8a4caaf60693313451af4836913156bbb1ecf4847771d3dd2136a7a5e49` |
| `artifacts/dataset2_audit/dataset_statistics.json` | `6b73fe6d1a255b259f87305e91f52834fcf243a052c88ba03a2d513ad5eeae27` |
| `artifacts/dataset2_audit/sample_manifest.csv` | `5ba98a5243b29a301fb0ce112acdbc14daac3dbaa68955d072b362816931de13` |
| `artifacts/dataset2_audit/human_review_template.csv` | `89a5d9fa03f1aa2fc96c0bd3fcc8c7ae6bf7e0223b11609a2cb12afa69ab21ef` |
| `_work/visual_integrity_per_clip.csv` | `38040b42c668bfd6b10a99b84dc374cce5ca22b5b943b436c2a40fe93a5887d0` |
| `_work/text_split_per_clip.csv` | `c6f16fce50b9c6bca15eb6736e61d360d9b4dfd4ea50811a2b2decd52f07b248` |
| `_work/visual_stats.json` | `036ae3522f7533dbd4ab02ea40bd83f2c723e4653b206c3fee0acb1260077ab7` |
| `_work/text_split_stats.json` | `f6ab42bf651d184c7088283887ed24320857a0ce58c08ffead3b28ef2c47782a` |
| `_work/01_h5_provenance.json` | `1349e854cdc30525f8ff87dca17bd4dfa22a34a9fc4416fc970d3765458f965c` |
| `_work/04_experiments_split_stats.json` | `5e80ce6f8d68e3a9a4a874cf87ada94ccf6c48d9f75f0b8e565110dced0c0cdd` |

- Entradas de datos: ver §4. Los hashes completos están en `_work/01_inventory.tsv` y en `dataset_statistics.json:provenance`.
- Scripts (todos bajo `artifacts/dataset2_audit/scripts/` e independientes del pipeline de entrenamiento): `inventory.py`, `h5_provenance.py`, `visual_integrity.py`, `text_split_audit.py`, `experiment_split_leakage.py`, `tb_scalars.py`, `build_review_sample.py`, `merge_integrity_report.py` y `build_dataset_statistics.py`.

**Hallazgo lateral de seguridad.** `/shared/Code/Sign-AI/data/scripts/DataverseDownload.ipynb` (dataset5/PUCP, no dataset2; propietario `giorgio6846`, mayo de 2025) contiene en texto plano una **API key de Dataverse** (variable `API_KEY`, cabecera `X-Dataverse-key`). La clave no se reproduce aquí. Se recomienda revocarla y rotarla, y sacarla del notebook y de cualquier historial git que la incluya (VERIFIED: presencia del patrón por `grep -n -o`, sin imprimir el valor).

---

## 14. Limitaciones y preguntas abiertas

### Contradicciones entre notas, reconciliadas

| Contradicción | Resolución | Tag |
|---|---|---|
| `02` dice "no hay tope de 250 en el build actual"; `01` y `04` dicen que la extensión se corrió con `--max-frames 1400` y que 12 clips están en el tope | **Ambas cosas son ciertas y se refieren a 12 clips distintos.** En `visual_integrity_per_clip.csv`: (a) 12 clips con T = 250, de los cuales 11 tienen un video de exactamente 250 frames y 1 (clip 5285) de 270 frames (razón 0,926, descarte, no tope), así que son longitudes naturales y el flag `hits_cap_250` de `02` está mal nombrado; (b) 12 clips con T = 1.400, y los 12 videos con > 1.400 frames tienen T = 1.400 (0 excepciones), lo que demuestra empíricamente un tope de 1.400. De esos 12, 7 están excluidos como truncados y **5 están kept con la cola cortada** (627, 3642, 3846, 4467, 4768; razón 0,93–0,999). El tope está VERIFIED por los datos; la línea de comando sigue UNKNOWN | VERIFIED |
| "3.088 clips desalineados > 1 s" (auditoría previa) | Sin soporte: es un comentario sin procedimiento, denominador ni lista de clips. No se usa en ninguna parte de este informe | VERIFIED (ausencia) |
| Memoria: "InfoNCE retrasó el colapso ~9 épocas" en v117 | Refutada: v117 ≡ v116 en 350/354 escalares, con InfoNCE = 0 | VERIFIED |
| "Las claves 0..599 no son los 600 originales" (`train_ctc_v119.py:204-209`, `ctc_v119_subset1k.toml:2-6`, `research_audit_imitator_e1.md:260`; repetido en `03:§4.6`) frente a la reconstrucción (228 clips ≥ 250, máx. 1.312, más el chequeo del backfill) | Gana la identidad. Esos comentarios comparaban la media de todos los frames y no la de los 228 re-extraídos (467,3 de T frente a ~469 frames de video registrados) | INFERRED (fuerte) |
| Familias por split: `01` da 417 clips de noticiero en val + 2 de "último momento"; `03` da 407 de noticiero + 12 de "otros" | Diferencia de definición. `cnsordos-se-renueva-noticias-en-lsa-tu-idioma` (10 clips, val) cuenta como noticiero en `01` (subcadena "noticias") y como "otros" en `03`. La conclusión (test ~48% eco) no cambia | VERIFIED |
| Memoria: truncados "25 palabras frente a 16" | El recómputo da medias de 25,2 frente a 10,8. Se usa el recómputo | VERIFIED |
| Índices de muñeca: `01` INFERRED frente a `02` VERIFIED | La comprobación geométrica de `02` (300 clips) supersede a `01` | VERIFIED |
| Mediana de la razón de los truncados: 0,63 (`02`) frente a "0,55–0,63" (`04`) | `04` cita un rango sobre conjuntos distintos (413 frente a 434). Se usa 0,63 para los 413 | VERIFIED |
| Posteriores de v119: "300 clips, 98,2% blank" frente al resumen actual de 5 clips al 99,8% | El resumen se sobrescribió; la cifra de 300 queda como DOCUMENTED | DOCUMENTED |

### Limitaciones

- No hay evaluación de los niveles B y C: 0 anotaciones y ningún timestamp.
- `meta.csv` fue borrado. Las etiquetas, `video_id` y `source_group` solo se verifican por consistencia interna, no contra la fuente.
- Las heurísticas de identidad y de casi duplicados no están calibradas. Solo se revisó un clip visualmente.
- No hay scores de confianza, de modo que las tasas de "punto ausente" son proxies.
- ffprobe sustituye a cv2. Solo los 20 clips del piloto se decodificaron completos.
- La reconstrucción de los splits históricos supone que `torch.randperm` se comporta igual en 2.8 y 2.13. Los tamaños coinciden, pero la permutación por índice es INFERRED.
- `text_ctx` no se recomputó.
- No hay identidad de intérprete.
- Las afirmaciones lingüísticas (palabras función sin seña manual, orden LSA frente a español) son INFERRED y requieren un experto.

### Preguntas abiertas

1. ¿Existe una copia externa de `meta.csv` (con `start`/`end`/`splits`) o de los subtítulos de los videos padre? Su recuperación desbloquearía el nivel B.
2. ¿Los subtítulos de CN Sordos son manuales, automáticos o ASR?
3. ¿Qué precisión tiene `identity_switch_suspect`? Se responde con el bucket intencional de §9.
4. ¿Cuánto tarda un revisor LSA por clip? Determina el coste real de los escenarios B y C.
5. ¿Qué umbrales τ_A y τ_C pre-registran el equipo y los expertos?

_Documento generado con asistencia de IA (Claude). Validar con expertos en LSA y en el dominio antes de usarlo en decisiones._
