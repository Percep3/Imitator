# Metodología de la auditoría de dataset2

Complementa a `docs/dataset2_scientific_audit.md`. Los comandos están en `reproduction_commands.md`.

## 1. Reglas

- Sin entrenamiento, sin inferencia en GPU y sin descargas.
- El HDF5 se abre con `h5py.File(p, "r")`. No se modificaron videos, splits, checkpoints ni outputs previos.
- El código auxiliar vive solo en `artifacts/dataset2_audit/scripts/`, es independiente del pipeline de entrenamiento y solo usa stdlib, numpy, h5py y tokenizers.
- Las funciones de texto y de split del pipeline se importaron **por ruta de archivo**: `alignment_text.py`, `vocab.py`, `grouped_split.py` e `integrity_filter.py`. Así se reproducen exactamente sin importar el paquete, que importa torch.
- No se fabricaron timestamps, correspondencias palabra-frame ni porcentajes semánticos.

## 2. Categorías de evidencia

| Tag | Definición |
|---|---|
| VERIFIED | Inspección directa o cómputo reproducible con el comando citado |
| DOCUMENTED | Consta en docs, configs, reportes, memoria o historial git, pero no se re-ejecutó |
| INFERRED | Interpretación plausible de evidencia VERIFIED o DOCUMENTED |
| UNKNOWN | No determinable con los datos disponibles |

Las cifras de auditorías previas se trataron como hipótesis. "3.088 desalineados" quedó sin soporte.

## 3. Niveles de evaluación del alineamiento

- **A, computacional.** Integridad de arrays, consistencia texto↔token↔embedding, cobertura de frames y plausibilidad del esqueleto. Es el único nivel evaluado.
- **B, temporal.** Si el tramo señado correspondiente cae dentro de los límites del clip. No es evaluable: no hay timestamps (meta.csv borrado), ni SRT, ni audio.
- **C, semántico.** Si el texto es una traducción fiel de lo señado. Requiere revisores competentes en LSA; hay 0 anotados.

## 4. Chequeos y umbrales (nivel A)

| Chequeo | Definición | Umbral | Calibrado | Fuente |
|---|---|---|---|---|
| Truncado (filtro v125) | T / `nb_frames` del video (ffprobe en lugar de cv2) | < 0,9 | No (umbral del pipeline) | `build_dataset2_manifest.py:31,49` |
| min_frames / min_tokens / frames_per_token | Filtro v125 de primera coincidencia | los del pipeline | — | `integrity_filter.py:31-54` |
| Frames descartados | razón en [0,9, 1,0) entre los kept | < 1,0 | — | `02_visual.md` |
| Salto corporal | desplazamiento del cuello entre frames, en anchos de hombro (sw) | > 0,5 sw | **No** | `visual_integrity.py` |
| Retorno A-B-A | el cuello vuelve a ≤ 0,2 sw de la posición previa al salto en ≤ 30 frames | 0,2 sw / 30 frames | **No** | ídem |
| `identity_switch_suspect` | fracción de frames con salto > 5% y ≥ ½ de los saltos con retorno | 5%, ½ | **No** (1 clip revisado a ojo) | ídem |
| Mano fuera de cuadro | fracción de puntos de mano fuera de [0,W]×[0,H] | > 20% | No | ídem |
| Hombros degenerados | ancho de hombros < 1 px | 1 px | — | ídem |
| Cola cortada a 1.400 | T == 1.400 y video > 1.400 frames | — | VERIFIED: los 12 videos > 1.400 tienen T = 1.400 | `merge_integrity_report.py` |
| `hits_cap_250` (flag de `02`) | T == 250 | — | **Mal nombrado**: son longitudes naturales (11 videos de 250 frames y 1 de 270) | `visual_integrity.py:40,152` |
| Duplicado exacto | sha1 de los arrays de keypoints | igualdad | — | `text_split_audit.py` |
| Casi duplicado | firma de movimiento: normalizada por escala, sin media temporal por articulación, re-muestreada a 32 frames, L2; vecino más cercano por coseno, búsqueda exhaustiva | ≥ 0,95 | **No** | ídem |
| Fuga de texto | etiqueta exacta y normalizada (`normalize_alignment_text`) en train | igualdad | — | ídem |

`level_A_status` en `integrity_report.csv`:
- `excluded_v125:<causa>`;
- `kept_warn`: kept con al menos una de `identity_switch_suspect`, `hand_offframe_gt20pct`, `degenerate_shoulders`, `tail_cut_at_1400`, `frames_dropped_ratio_lt1`;
- `kept_no_warn`.

`body_jump` y `scale_jump` no cuentan como advertencia porque afectan a más de la mitad de los clips. Ningún estado implica certificación lingüística.

## 5. Splits históricos

Los splits por clip de v116–v119 se reconstruyeron con `random_split([0.8, 0.2], Generator().manual_seed(42))` (`keypoint_dataset.py:111-112`) sobre los `valid_index` replicados. Los tamaños (481/119, 4.481/1.119, 4.436/1.108) coinciden con los logs. La identidad de la permutación entre torch 2.8 y 2.13 es INFERRED.

## 6. Diseño muestral para la revisión humana

- **Marco:** los 5.138 clips kept v125. Los 462 rechazados quedan fuera del componente probabilístico.
- **Probabilístico, n = 80:** MAS estratificado sin reemplazo.
  - Estratos: split × tercil de duración del video, con cortes en 5,572 s y 10,544 s sobre los kept.
  - Asignación: train aproximadamente proporcional (19/19/18); val y test con un mínimo de 4 por estrato.
  - Semillas: `random.Random("23:<estrato>")`.
- **Piloto:** los primeros k_h sorteados en cada estrato, 20 en total, fijados antes de ver contenido y sin reemplazo.
- **Intencional, n = 42:** 12 buckets definidos por flags, disjuntos del probabilístico. **Excluido de toda estimación poblacional.** Solo sirve para estimar la precisión de un flag dentro de su bucket.

## 7. Estimadores

- Proporción ponderada: p̂ = Σ wᵢyᵢ / Σ wᵢ, con wᵢ = N_h/n_h (Σw = 5.138).
- Tamaño efectivo de Kish: n_eff = (Σw)²/Σw² = 76,2.
- IC 95%:
  - Wilson o Clopper-Pearson con n_eff y x_eff = p̂·n_eff. Clopper-Pearson cuando p̂ está cerca de 0 o 1.
  - Alternativa: varianza estratificada Σ W_h² p_h(1−p_h)/(n_h−1), solo si n_h ≥ 2 y 0 < p_h < 1.
- `unknown` y `not_assessable` se reportan como categorías propias, con una cota de sensibilidad: contados como fallo y como excluidos.
- Acuerdo entre anotadores (piloto): κ de Cohen y α de Krippendorff (nominal, y ordinal para confianza y completitud), más porcentaje bruto y bootstrap por clips con 2.000 réplicas. Criterio para escalar: α ≥ 0,67 en `semantic_correspondence` y `temporal_alignment`.
- Semiamplitud Wilson 95%: z = 1,959964, h = z·√(p(1−p)/n + z²/4n²)/(1 + z²/n). La tabla para n ∈ {12, 20, 76, 80, 150, 200, 300, 385} está en `dataset_statistics.json:derived.wilson95_half_width_pp`. Por ejemplo, n = 80 y p = 0,5 dan ±10,7 pp.
- Limitación: el diseño no agrupa por `source_group`, de modo que los IC pueden ser algo optimistas para propiedades que varían por episodio.

## 8. Limitaciones del método

- No hay scores de confianza, así que las "ausencias" son proxies.
- ffprobe sustituye a cv2. Solo los 20 clips del piloto se decodificaron completos.
- Las heurísticas de identidad y de casi duplicados no están calibradas.
- `meta.csv` fue borrado, de modo que solo se verifica la consistencia interna.

_Generado con asistencia de IA (Claude); validar con expertos._
