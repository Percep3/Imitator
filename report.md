# Reporte de experimentos: v115 → v118c
**Fecha:** 2026-06-18  
**Tarea:** Traducción de lengua de señas (LSP) — keypoints → texto en español  
**Dataset:** dataset2, 600 vídeos (481 train / 119 val)  
**Hardware:** RTX 4060 Ti 16 GB  

---

## Contexto del problema

El sistema convierte keypoints de manos/cuerpo (111 puntos) en texto español usando un Imitator (STGCN) que produce embeddings que guían a un LLM (Gemma-3n E2B 4-bit) congelado. La pregunta central de esta serie de experimentos es: **¿cómo hacer que el LLM use la señal visual del encoder en lugar de su prior lingüístico?**

---

## v115 — CE-vocab (baseline)

**Arquitectura:** Imitator → embedding → cosine similarity contra tabla de embeddings de Gemma → CE vocabulario  
**Idea:** Clasificar directamente el token de salida en cada posición usando la tabla de embeddings congelada del LLM.

### Resultados
| Métrica | Valor |
|---------|-------|
| Top-1 acc (teacher-forced) | ~10.9% |
| Top-5 acc (teacher-forced) | ~23.2% |

### Problema detectado
El modelo sufrió **runaway de confianza**: la norma L2 de los embeddings predichos creció de 16 → 33 con coseno → 0.998. El modelo colapsó hacia embeddings de alta confianza en pocos tokens, lo que hace la distribución de salida degenerada.

---

## v115.1 — CE-vocab + temperatura aprendible + label smoothing

**Cambios sobre v115:**
- Temperatura aprendible (init log_temp = 2.659, temp ≈ 14.3 — igual que CLIP)
- Label smoothing ε=0.1
- Datasets secundarios (dataset6)

### Resultados
Controlaba el runaway pero los números de generación real **no se midieron** — la métrica era CE teacher-forced, insensible a la calidad generativa. Esta fue la limitación que motivó el cambio de arquitectura.

### Lección
CE-vocab con teacher forcing evalúa "¿predice el token correcto dado los anteriores?", no "¿genera frases coherentes desde el prefix visual?". Las dos métricas se desacoplan cuando el modelo colapsa.

---

## v116.0 — CE-AR con soft prefix (Gemma-3n congelado)

**Arquitectura:** Imitator + PrefixAdapter → prefix [B, K=20, 2048] → Gemma (frozen) → CE autoregresiva  
**74.31 M parámetros entrenables** (Gemma congelado)  
**Config:** `ce_ar_v116.toml`

### Idea
En lugar de predecir tokens directamente, el Imitator produce 20 tokens de "soft prefix" que se inyectan en Gemma antes de los tokens de texto. Gemma predice el siguiente token dado el prefix + tokens anteriores (teacher forcing). La pérdida es CE sobre los tokens de texto.

### Bugs corregidos en v116
- **Bug-1:** El último token del prefix (posición K-1) no supervisaba el primer token de texto → corregido con `build_ar_labels`.
- **Bug-3:** Los tokens de padding asistían en la atención → añadida máscara de padding.

### Resultados
| Métrica | Valor |
|---------|-------|
| Val Top-1 acc (teacher-forced) | ~28–30% |
| Val Top-5 acc (teacher-forced) | ~52% |
| chrF pico (generativo) | **20.7** (época 10) |
| chrF final (época ~25) | ~13 |
| effrank época 0 | 25 |
| effrank época 10 | ~4 |

### Problemas descubiertos

**1. Colapso del prefix.**  
A partir de la época 10, el effrank del prefix cae de 25 a ~4 (cos-sim ~0.97 → todos los vídeos producen casi el mismo prefix). Con teacher forcing el LLM predice desde los tokens GT anteriores y **no necesita el prefix**. Cuando el prefix colapsa, la generación real se degrada (chrF 20 → 13) mientras la CE teacher-forced permanece plana.

**2. Selección de checkpoint con la métrica equivocada.**  
`early_stopping` usaba `val_ce_ar` (CE teacher-forced), que es plana desde ep12 e insensible al colapso. El mejor checkpoint real (época 10, chrF 20.7) no se conservó de forma robusta. La eval generativa corría solo cada 5 épocas con 4 batches (ruidoso).

---

## v116.1 — CE-AR + SIGReg λ=0.1

**Cambio sobre v116.0:** Añade regularizador SIGReg para empujar el prefix hacia una gaussiana isotrópica.

### Resultado
**Completamente inútil.** La contribución de SIGReg a la loss total fue:

```
λ · nce / CE = 0.1 · 0.029 / 4 ≈ 0.07%
```

Los runs v116.0 y v116.1 son indistinguibles en TensorBoard. El regularizador es demasiado débil para competir con la CE-AR que domina el gradiente.

### Lección
λ=0.1 sobre una loss auxiliar de magnitud 0.029 frente a una CE de ~4 es efectivamente λ_efectivo ≈ 0.001. Para que SIGReg tuviera efecto requeriría λ ≥ 10, lo que distorsionaría la optimización principal.

---

## v117 — CE-AR + InfoNCE contrastivo (λ=0.1, τ=0.07)

**Cambios sobre v116.0:**
1. **InfoNCE video↔texto** (NT-Xent simétrico) — penaliza el colapso del prefix
2. **Gen eval cada época** con todos los batches de val (999 → todos ~4 batches)
3. **Checkpoint por chrF** (`best_chrf/`) en lugar de por val_ce_ar
4. **Diagnóstico de shuffle prefix** — mide si el prefix aporta información real
5. **Logging de norma de gradiente** del Imitator
6. **SIGReg desactivado**

**Config:** `ce_ar_v117.toml`  
**Duración:** 2h 46min (25 épocas, ~397s/época)

### Resultados por época

| Ép | Train CE | Val CE | TokenAcc | Top5 | chrF | effrank | cos | gap shuffle |
|----|---------|--------|----------|------|------|---------|-----|-------------|
| 0 | 13.42 | 13.75 | 4.5% | 10.5% | 0.12 | 25.1 | 0.31 | -0.04 |
| 2 | 12.80 | 12.31 | 4.4% | 10.9% | 2.01 | 24.7 | 0.38 | +0.02 |
| 4 | 7.31 | 6.28 | 14.8% | 30.8% | 0.96 | 21.1 | 0.78 | -0.18 |
| 5 | — | — | — | — | 13.37 | 22.5 | 0.91 | **+0.22** |
| 6 | 4.51 | 4.58 | 24.0% | 44.9% | 10.09 | 23.5 | 0.96 | +0.15 |
| 8 | 4.09 | 4.41 | 26.4% | 46.9% | 0.09 | 22.9 | 0.98 | -0.12 |
| **10** | 3.85 | 4.17 | 28.2% | 49.9% | **20.60 ★** | 19.9 | 0.99 | -0.26 |
| 12 | 3.77 | 4.08 | 28.7% | 50.6% | 9.03 | 16.0 | 0.99 | +0.05 |
| 14 | 3.66 | 4.29 | 27.7% | 48.9% | 10.92 | 13.0 | 0.98 | +0.15 |
| 16 | 3.62 | 4.15 | 28.6% | 51.7% | 5.96 | 10.2 | 0.98 | -0.67 |
| 18 | 3.47 | 4.32 | 28.0% | 48.6% | 15.14 | 7.0 | 0.96 | +0.51 |
| 20 | 3.42 | 4.08 | 29.9% | 52.1% | 20.54 | 5.8 | 0.98 | -0.20 |
| 22 | 3.53 | 4.14 | 28.4% | 50.5% | 17.24 | 4.6 | 0.97 | -0.45 |
| 23 | — | — | — | — | 17.84 | 4.9 | 0.98 | **-0.90** |
| 24 | 3.40 | 4.14 | 28.2% | 51.7% | 18.03 | 5.1 | 0.97 | -0.42 |

### Hallazgos clave

#### 1. Pico de chrF idéntico a v116: 20.60 vs 20.7

El InfoNCE no mejoró el techo generativo. El mejor checkpoint (`best_chrf/`) se guardó en la época 10 con chrF=20.60 — exactamente el mismo pico que v116 en la misma época. La hipótesis de que el InfoNCE elevaría el pico no se cumplió.

#### 2. El InfoNCE retrasó el colapso ~9 épocas

| | effrank en ep 10 | effrank mínimo | época del mínimo |
|---|---|---|---|
| v116.0 | ~4 | ~4 | ep 10 |
| v117 | 19.9 | 4.6 | ep 22 |

El prefix mantuvo diversidad hasta la época ~19. Este es el único beneficio estructural medible del InfoNCE: compra tiempo antes del colapso, pero no lo evita.

#### 3. El gap shuffle se vuelve definitivamente negativo desde la época 10

```
gap = chrF(prefix_real) - chrF(prefix_aleatorio)
```

A partir de la época 10, el gap es mayoritariamente negativo (−0.02 a −0.90), lo que significa que generar con un prefix aleatorio produce resultados tan buenos o mejores que con el prefix real. **El LLM aprendió a ignorar el prefix y usa solo su prior lingüístico.**

Épocas con gap positivo (prefix aporta): 1, 2, 5, 6, 12, 13, 14, 18  
Épocas con gap negativo (prefix ignorado): 0, 3, 4, 7–11, 15–17, 19–24

#### 4. chrF tardío más alto que v116 (~17–18 vs ~13)

Las épocas 20–24 mantienen chrF entre 17 y 20, mientras que v116 caía a ~13. La estabilización tardía probablemente refleja que el effrank se mantiene ligeramente más alto (4.6–5.9 vs ~4 de v116) y que el prior lingüístico de Gemma simplemente mejora con más entrenamiento de la cabeza AR.

#### 5. Val CE estable, sin divergencia

La val CE oscila entre 4.08 y 4.32 sin tendencia creciente. El early_stopping no se activó. El modelo no sobreajusta en CE — el cuello de botella está en la calidad generativa, no en la capacidad del modelo.

#### 6. chrF extremadamente volátil entre épocas

La varianza entre épocas consecutivas es enorme (0.09, 0.33, 20.60, 15.91, 9.03...). Con solo 119 muestras de validación y 4 batches de tamaño 32, la generación greedy es muy ruidosa. Cualquier evaluación generativa puntual es poco fiable; hay que mirar la tendencia.

---

## Comparación global

| Versión | Loss | chrF pico | chrF tardío | effrank ep10 | Checkpoint por |
|---------|------|-----------|-------------|--------------|----------------|
| v115 | CE-vocab | — (no medido) | — | — | val_ce |
| v115.1 | CE-vocab + temp | — (no medido) | — | — | val_ce |
| v116.0 | CE-AR | **20.7** (ep10) | ~13 | ~4 | val_ce_ar ❌ |
| v116.1 | CE-AR + SIGReg | ~20.7 (igual) | ~13 | ~4 | val_ce_ar ❌ |
| v117 | CE-AR + InfoNCE | **20.60** (ep10) | ~17–18 | 19.9 | chrF ✓ |

El salto de calidad real fue **v115.1 → v116.0** (de ~10.9% top-1 a ~28% top-1, y generación coherente). Las versiones v116.1 y v117 refinaron el pipeline pero no movieron el techo de chrF.

---

## Diagnóstico final: por qué el prefix es ignorado

Con teacher forcing durante el entrenamiento:

```
P(token_t | prefix, token_0, token_1, ..., token_{t-1})
```

El LLM tiene acceso a todos los tokens GT anteriores y puede predecir `token_t` desde ellos con alta confianza sin consultar el prefix. El prefix solo necesitaría ser útil en el primer token (`token_0`), pero incluso ahí el prior de Gemma sobre primeros tokens de frases en español es fuerte. El modelo aprende a ignorar el prefix porque **no recibe presión de entrenamiento que lo obligue a usarlo**.

El gap shuffle negativo es la evidencia directa: el prefix no aporta información por encima del prior del LLM.

---

## Plan original tras v117: Scheduled Sampling (descartado)

**Hipótesis:** Si durante el entrenamiento se reemplazan gradualmente los tokens GT por las propias predicciones del modelo, el LLM ya no puede depender de los tokens GT anteriores y debe usar el prefix para producir tokens coherentes.

**Diseño propuesto:**
- Probabilidad de usar el token predicho (en lugar de GT): `p_sched = min(epoch/total_epochs, 0.5)`
- Implementar en `_ce_ar_criterion`: dado el logit en posición `t`, samplear si usar `argmax(logit_t)` o `token_ids[:, t]` para construir el `inputs_embeds` de la posición siguiente
- Criterio de éxito: gap `chrF - chrF_shuffled > 2` puntos de forma consistente durante 5+ épocas

Antes de implementarlo se hizo una re-evaluación de grounding sobre los checkpoints de v117 ya entrenados (sección siguiente), que mostró que el prefix no porta información del vídeo en absoluto. Scheduled sampling solo combate la dependencia de los tokens GT durante el entrenamiento, pero no obliga al encoder a producir un prefix discriminativo — no ataca la causa raíz, así que se descartó sin implementar.

---

## Re-evaluación de v117 por grounding (retrieval@1)

**Motivación:** el gap shuffle (chrF real − chrF aleatorio) de v117 es ruidoso época a época pero mayoritariamente negativo desde ep10. Antes de diseñar v118 hacía falta una métrica menos ruidosa y más directa: ¿el prefix de cada vídeo permite **identificar ese vídeo** frente a los demás, o es intercambiable?

**Método** (`/tmp/reeval_v117_gap.py`):
1. Cargar los checkpoints `117/1` en las épocas 1, 5, 10, 15, 20, 24.
2. Generar (greedy, `@torch.no_grad()`) sobre los 119 vídeos de val completos, en sub-batches de 1 para evitar OOM.
3. Construir la matriz cruzada de chrF(hyp_i, ref_j) para todo i, j (chrF inline: n-gramas 1-6, β=2, sin dependencia de `sacrebleu` que no está instalado en Sign-env).
4. Test de Wilcoxon pareado: ¿chrF(hyp_i, ref_i) es consistentemente mayor que la mejor alternativa chrF(hyp_i, ref_j≠i)?
5. Retrieval@1: por cada hyp_i, ¿la fila de la matriz tiene su máximo en la columna i?

### Resultado

| Época | Retrieval@1 | Wilcoxon p-valor |
|-------|-------------|-------------------|
| 1, 5, 10, 15, 20, 24 (todas) | **0%** | 0.20 – 0.93 (no significativo) |

Azar esperado: 1/119 ≈ 0.84%. El checkpoint `best_chrf` (ep10, chrF=20.60) tiene retrieval@1 = 0% — **peor que el azar en la práctica**, y el test de Wilcoxon no encuentra diferencia significativa entre el hyp emparejado con su propio ref y con cualquier otro. El chrF≈20 reportado en la sección anterior es enteramente el prior de español de Gemma, no traducción.

### Conclusión

La línea CE-AR con soft-prefix (v116/v117) **no traduce**. El prefix está vacío de información específica del vídeo, no parcialmente degradado. Esto descarta scheduled sampling (que solo ataca la dependencia de tokens GT, no la falta de señal en el encoder) y motiva separar "aprender a codificar seña" de "aprender a generar texto".

---

## Giro a v118: pre-entrenamiento contrastivo (CLIP-style, sin LLM)

**Idea (GFSLT-VLP):** en vez de entrenar el encoder y el LLM juntos con CE autoregresiva (donde el LLM puede ignorar el prefix), pre-entrenar el encoder solo, alineando su embedding con el embedding de la frase mediante InfoNCE simétrico — el colapso es imposible por construcción (si todos los vídeos colapsan al mismo vector, la pérdida alcanza su máximo en vez de un mínimo gratis). El LLM no participa en esta etapa; el lado texto usa los embeddings de entrada de Gemma ya cacheados en el HDF5. Selección de checkpoint por **retrieval@1** sobre val — la métrica que demostró el fallo de v117.

**Arquitectura (`ContrastiveAligner`):** el mismo `PrefixImitator` de v116/v117 como torre de vídeo (pesos transferibles a una etapa generativa futura) + cabezas de proyección (`LayerNorm→Linear→GELU→Linear`) por modalidad hacia un espacio compartido de 256 dims.

### v118 — InfoNCE simétrico (NT-Xent), batch=16

| Métrica | Valor |
|---------|-------|
| Mejor R@1 | 2.5% (ep17, azar=0.84%) |
| val/train loss al final | ~4.2× |

El val loss diverge fuertemente del train loss porque `logit_scale` (temperatura aprendible) crece sin freno y amplifica el error en pares incorrectos de val.

### v118b — InfoNCE + augmentations (temporal crop 70-100%, ruido gaussiano), batch=32

| Métrica | Valor |
|---------|-------|
| Mejor R@1 | 1.7% (ep11) — **peor** que v118 |
| val/train loss al final | ~3.9× |

Las augmentaciones no resolvieron la divergencia ni mejoraron el techo.

### v118c — VICReg en vez de InfoNCE

**Motivación:** la divergencia val/train de v118/v118b es síntoma de la temperatura, no de los datos. VICReg (Bardes et al.) no usa temperatura ni negativos explícitos: un término de invariancia (MSE) alinea cada par vídeo-texto, mientras varianza + covarianza previenen el colapso por construcción.

| Run | Datos | Mejor R@1 | val/train final |
|-----|-------|-----------|------------------|
| v118c run_id=3 | truncados (ver bug abajo) | 2.5% (ep3) | ~1.3× |

VICReg resolvió la divergencia val/train (1.3× vs 4.2-3.9× de InfoNCE) pero no movió el techo de R@1 — confirmando que la pérdida no era la causa del estancamiento.

#### Bug encontrado: 38% de los clips truncados a mitad de oración

`scripts/build_dataset2_h5.py` extrae keypoints en vivo desde los `.mp4` con `--max-frames 250` por defecto (≈8.3s a 30fps). Esto truncó **228 de 600 clips (38%)** a mitad de la seña, mientras el label seguía siendo la oración completa. Verificado reproduciendo `select_clips(600, seed=23)` y comparando contra el frame count real del vídeo: media real 469 frames (hasta 1312) vs cap de 250 — en el peor caso se perdía el 78% del vídeo real. Los clips truncados eran justo las oraciones más largas (25 palabras de media vs 16 del resto), así que el sesgo no era aleatorio.

**Fix:** se borraron las 228 entradas truncadas del HDF5 (`data/processed/dataset_v6_unsloth.hdf5`, backup en `.bak_pre250fix`) y se re-extrajeron con `--max-frames 1400` (cubre el máximo real de 1312). Resultado: 0 clips quedan truncados.

### v118c run_id=4 — VICReg sobre datos corregidos

| Métrica | Techo previo (datos truncados) | Con fix |
|---------|----------------------------------|---------|
| R@1 | 2.5% | **2.9%** (ep24, early-stopped ep49) |
| R@10 | ~12.6% | **14.3%** |
| median_rank | 51–69 | **44–46** |

**Veredicto:** el bug de truncado sí contribuía al estancamiento — mejora real y consistente en R@5/R@10/median_rank (no solo ruido de R@1, que tiene varianza alta con solo 119 muestras de val). Pero **no era la causa dominante**: con datos limpios el R@1 sigue en el rango 2-3× azar, no rompe a dos cifras. El cuello de botella ahora apunta con más fuerza a **modelo/cantidad de datos** (70M parámetros entrenables vs 481 clips de train) que a la calidad de los pares.

---

## Comparación global (extendida)

| Versión | Loss | Métrica de selección | Pico/Mejor | Diagnóstico |
|---------|------|------------------------|------------|-------------|
| v116.0 | CE-AR | val_ce_ar ❌ | chrF 20.7 (ep10) | prefix colapsa ep10, no traduce |
| v117 | CE-AR + InfoNCE | chrF ✓ | chrF 20.60 (ep10) | **retrieval@1 = 0%**, prior de Gemma puro |
| v118 | InfoNCE puro, sin LLM | retrieval@1 | R@1 2.5% (ep17) | val/train diverge 4.2× |
| v118b | InfoNCE + aug | retrieval@1 | R@1 1.7% (ep11) | aug no ayuda |
| v118c (run 3) | VICReg | retrieval@1 | R@1 2.5% (ep3) | resuelve divergencia, no el techo |
| v118c (run 4) | VICReg + datos corregidos | retrieval@1 | R@1 2.9% (ep24) | techo real ≈ modelo/datos, no pipeline |

---

## Próximo paso

Con el bug de truncado corregido y tres formulaciones de pérdida (InfoNCE, InfoNCE+aug, VICReg) convergiendo al mismo techo (~2-3× azar), el cuello de botella deja de ser "qué pérdida usar" y pasa a ser una decisión entre:

1. **Reducir la capacidad del encoder** — 70M parámetros para 481 clips de train es sobreparametrización severa; recortar el STGCN/Transformer puede mejorar generalización sin tocar datos ni pérdida.
2. **Pretexto más fácil que retrieval exacto** — si hay estructura de gloss/sintaxis compartida entre clips, una tarea de clasificación discreta podría dar señal más alcanzable y diagnosticar si el problema es la dificultad de la tarea en sí.
3. **Aceptar la señal débil (2.9%, 3.5× azar) y pasar a v119** — congelar este encoder y probar si aporta algo medible en una etapa generativa con QLoRA, en vez de seguir optimizando la fase contrastiva en aislamiento.
4. **Revisar el resto del pipeline de datos** (dataset1, dataset3-7) por el mismo patrón de truncado antes de escalar arquitectura — aunque la evidencia de v118c run_id=4 sugiere que esto por sí solo no destraba el problema.

---

## Artefactos

| Archivo | Descripción |
|---------|-------------|
| `config/experiment/ce_ar_v117.toml` | Config completa v117 |
| `src/mslm/training/loss_infonce.py` | InfoNCE NT-Xent simétrico (v117, prefix↔texto dentro del CE-AR) |
| `tests/test_loss_infonce.py` | 5 tests CPU (todos pasan) |
| `../outputs/checkpoints/117/1/best_chrf/` | Checkpoint época 10, chrF=20.60 (retrieval@1=0%, no traduce) |
| `src/mslm/training/loss_contrastive.py` | InfoNCE simétrico + VICReg + métricas de retrieval (v118) |
| `tests/test_loss_contrastive.py` | 10 tests CPU (todos pasan) |
| `src/mslm/models/contrastive.py` | `ContrastiveAligner`: torres vídeo/texto + flag `normalize` (VICReg) |
| `tests/test_contrastive_aligner.py` | 4 tests CPU (todos pasan) |
| `src/mslm/dataloader/augmentations.py` | Temporal crop + ruido gaussiano para keypoints (v118b) |
| `scripts/train_contrastive_v118.py` | Loop de entrenamiento contrastivo (InfoNCE/VICReg, selección por R@1) |
| `config/experiment/contrastive_v118.toml` / `v118b.toml` / `v118c.toml` | Configs de las tres variantes |
| `scripts/build_dataset2_h5.py` | Extractor de keypoints (bug de truncado a 250 frames documentado y corregido) |
| `data/processed/dataset_v6_unsloth.hdf5.bak_pre250fix` | Backup del HDF5 antes del fix de truncado |
| `../outputs/checkpoints/118/4/best_r1/` | Mejor checkpoint v118c sobre datos corregidos (R@1=2.9%, ep24) |

---

## v119 — CTC sobre secuencia (reemplaza el objetivo contrastivo global)

**Diagnóstico que motivó el giro:** `outputs/diag_v118_train_val_gap.json` mostró
que v118c (VICReg) tiene **train R@1 ≈ val R@1** (0.73% vs 2.94%, ambos ~3.5×
azar, gap_factor≈1.0) — el encoder ni siquiera ajusta el train, lo que descarta
sobreajuste y señala que el cuello de botella es la **formulación de la tarea**
(comprimir la frase a un vector y rankearla), no la pérdida ni la cantidad de
datos per se. La literatura cargada (LiftSign, CVPRW 2026, único paper
skeleton-based del KB) formula CSLR como secuencia con CTC, no como retrieval
global. v119 introduce `CTCEncoder` (clasifica por-frame, sin el cuello de
botella de K=20 tokens fijos de `Imitator`) + vocabulario propio word-level
(no el BPE de Gemma) construido solo con labels de train.

Cambia el redireccionamiento documentado en `contrastive.py`: v119 ya **no** es
"PrefixImitator + QLoRA" (plan original) sino CTC sin LLM en el loop, por el
mismo motivo del diagnóstico anterior.

### A1 sobre subconjunto de 1000 clips (chequeo rápido, `ctc_v119_subset1k.toml`)

Antes de correr la ablation completa (A1/A2/A3) sobre el dataset completo, se
hizo un chequeo rápido en background sobre 1000 clips muestreados al azar
(seed=23) de dataset2 (800 train / 200 val) — **no** son los 600 clips
originales documentados arriba: se verificó que las claves `"0".."599"` del
HDF5 actual no corresponden a ese subconjunto (su media de frames no coincide:
266 vs los 469 documentados), v118f mezcló los 5000 clips nuevos bajo el mismo
índice secuencial sin preservar esa identidad.

**Resultado (single-stream, sin TLP, 41 épocas, early-stopped):**

| Métrica | ep 0 | ep 10 | ep 20 | ep 41 |
|---|---|---|---|---|
| train CTC loss | 32.30 | 5.08 | 3.35 | 1.26 |
| val CTC loss | 10.01 | 16.30 | 16.06 | 18.63 |
| val WER | 100.0% | 99.9% | 100.0% | 100.0% |

Train loss cae 25× mientras val loss casi se duplica (sobreajuste de manual) y
el WER de val queda clavado en ~100% durante las 41 épocas — un patrón
completamente distinto al de v118 (ahí train y val eran igual de malos).

**Diagnóstico post-mortem** (cargando `outputs/checkpoints/119/10/best_wer/`):

1. **Colapso a blank, no solo overfitting.** El argmax por-frame predice
   blank en 98.3-100% de los frames **incluso en muestras de TRAIN**
   (`frac_argmax_es_blank` entre 0.983 y 1.000, ≤1 clase no-blank gana el
   argmax en toda la secuencia). El train loss bajando no implica que el
   greedy-decode recupere palabras: CTC puede minimizar su loss colocando una
   probabilidad mínima sobre la palabra correcta en algún frame (suficiente
   para el forward-backward) sin que esa probabilidad supere nunca a blank en
   el argmax.
2. **Vocabulario demasiado esparso para el tamaño de la muestra.** 3505
   palabras desde solo 801 frases de train: la mayoría de palabras aparece 1-2
   veces, insuficiente para que el modelo desarrolle confianza por encima de
   blank. OOV en val: 20.6% de las palabras (684/3323), pero el colapso a
   blank ocurre también en frases sin ninguna palabra OOV, así que el OOV no es
   la causa dominante — es la escasez de ejemplos por palabra.

**Conclusión:** este resultado **no refuta** la hipótesis CTC — está confundido
por el tamaño de muestra (800 train) combinado con un vocabulario de cola larga
(3505 clases). Lanzar A2/A3 (motion stream + TLP) sobre el mismo subconjunto
solo reproduciría el mismo colapso. Se relanza A1 sobre el **dataset completo**
(4481 train / 1119 val, 5.6× más ejemplos por palabra en promedio) antes de
sacar conclusiones sobre la formulación, usando el config original
`ctc_v119.toml` (sin `max_samples`).

### Artefactos

| Archivo | Descripción |
|---------|-------------|
| `src/mslm/models/ctc_encoder.py` | `CTCEncoder`: STGCN+Transformer reusados de `Imitator`, sin el cuello de botella de K tokens fijos; flags `use_motion_stream`/`use_tlp` |
| `src/mslm/dataloader/vocab.py` | `Vocab` word-level (blank=0, unk=1) construido solo con labels de train |
| `src/mslm/models/components/tlp.py` | Temporal Lift Pooling (LiftSign §3.2.2), para A3 |
| `src/mslm/training/loss_ctc.py`, `src/mslm/utils/wer.py` | `nn.CTCLoss` wrapper + greedy decode + WER (ecuación 9 de LiftSign) |
| `scripts/train_ctc_v119.py` | Loop de entrenamiento CTC, selección de checkpoint por menor WER, codifica cada vídeo del batch por separado (gradient checkpointing) para evitar OOM en clips largos — mismo problema y misma solución que `encode_video_batch` en v118 |
| `config/experiment/ctc_v119.toml` / `v119b.toml` / `v119c.toml` | A1 (single-stream) / A2 (+motion) / A3 (+TLP) sobre dataset completo |
| `config/experiment/ctc_v119_subset1k.toml` | Chequeo rápido sobre 1000 clips (resultado: colapso a blank, ver arriba) |
| `../outputs/checkpoints/119/10/best_wer/` | Checkpoint del chequeo rápido sobre subset (WER≈100%, colapso a blank documentado) |
| `reports/cleanup_2026-06-20.md` | Limpieza de checkpoints v113-v117 (43.7G) para liberar espacio antes de este run |
