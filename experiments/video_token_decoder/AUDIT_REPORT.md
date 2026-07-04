# Auditoría del decoder AR video→tokens Gemma y diseño de Imitator

Fecha: 2026-07-03 · Auditor: Claude (investigación multimodal) · Alcance: folds 4–6 confirmatorios, código en `v125-temporal-prompt-alignment`.

Convención de etiquetas: **[HECHO]** = verificado en código, logs o diagnóstico reproducible; **[INFERENCIA]** = conclusión que se sigue de hechos; **[HIPÓTESIS]** = plausible, no verificado.

Scripts de diagnóstico (reproducibles, read-only): `analyze_predictions.py`, `gloss_level_analysis.py`, `gpu_diagnostics.py` en el scratchpad de la sesión; sus salidas se citan como *Diag-A…F*.

---

## 1. Resumen ejecutivo

El decoder autoregresivo no falla por bugs de plumbing (máscaras, teacher forcing, greedy, lineage y métricas son correctos, y el test de overfit de un minibatch pasa), sino por un **defecto arquitectural del encoder visual: su salida no contiene información de orden temporal global**. El `TransformerEncoder` del encoder no tiene positional encoding y el TCN solo aporta fase local (±2 frames), de modo que la memoria que ve la cross-attention es un *conjunto* de features, no una secuencia. Diagnóstico causal: **permutar los segmentos de video deja el 86.3% de las predicciones idénticas** y no cambia las métricas (Diag-A). Consecuencia: el modelo emite una *bolsa de señas* — 92–96% de las predicciones parsean a glosas bien formadas, precision de glosa ~55%, recall ~42%, y **orden entre glosas = azar exacto (0.47–0.53)** (Diag-Fase2b). strict_exact (~2%) es el producto de identidad imperfecta × orden aleatorio × conteo corto, no un solo error.

Las hipótesis "geometría del embedding tied" y "el vocabulario de 262 400 impide aprender" quedan **refutadas**: el 100% de los tokens predichos cae dentro de los 118 IDs efectivos, sin colapso a frecuentes, y la init desde el clasificador CIF es geométricamente sana (cos init→final 0.963).

Recomendación: **continuar la ruta token-level con correcciones profundas** ("Imitator-JTA": PE en la memoria visual + pérdida CTC auxiliar *a nivel de token* + vocabulario de salida reparametrizado a los 121 IDs Gemma efectivos con mapeo biyectivo). Restricción de producto (2026-07-03): Imitator debe predecir tokens Gemma directamente; la clasificación de glosas queda descartada en cualquier escenario. Plan de 15h abajo.

---

## 2. Errores confirmados

| # | Error | Evidencia |
|---|-------|-----------|
| E1 | **[HECHO]** Encoder visual sin positional encoding global: `STGCNTemporalFrameEncoder.forward` pasa TCN→`TransformerEncoder` sin PE (`src/mslm/models/temporal_sign_prompt.py:429-438`, capas definidas en `:400-422`). Un transformer sin PE es permutación-equivariante; el TCN (2 capas, k=3, `:404`) solo inyecta fase local ±2 frames. La memoria de cross-attention (`src/mslm/models/video_token_decoder.py:132-140`) tampoco recibe PE — solo el target (`:131`). | Diag-A: permutar segmentos de video → 86.3% de predicciones idénticas; exact 0.0352→0.0352; edit_sim 0.264→0.272. Orden de glosas en predicciones = 0.471–0.529 (azar=0.5). |
| E2 | **[HECHO]** Selección de checkpoint sobre ruido muestral: inner-val de 512 muestras con exact 1–3% ⇒ error estándar ≈ 0.7 pp; las diferencias entre epochs 4–14 (p. ej. fold4: 0.8–2.5%) están dentro del ruido. `scripts/train/train_video_token_decoder.py:404,412` selecciona por `(strict_exact, edit_sim)` con exact como criterio primario. | `metrics.jsonl` folds 4–6: best epochs 10/13/10 con métricas indistinguibles de vecinas. |
| E3 | **[HECHO]** Entrenamiento truncado antes de converger: el train loss sigue cayendo en el epoch 14 en los tres folds (fold4: 2.28→2.24 entre e13 y e14) y edit_sim de inner-val aún oscila al alza. 15 epochs × 2048 muestras (batch 2, accum 2 = 7680 updates) es insuficiente para un decoder entrenado desde cero. | `fold*/base/metrics.jsonl`. |
| E4 | **[HECHO]** Label smoothing 0.1 sobre 262 400 clases (`train_video_token_decoder.py:300`): reparte 0.1 de masa entre 262 399 clases (3.8e-7 cada una) cuando el vocabulario efectivo es de 118+EOS tokens. Impone un suelo de loss de ~1.3 nats que enmascara la lectura de las curvas y desperdicia gradiente. No es la causa raíz (ver §3), pero es objetivamente inapropiado aquí. | Aritmética + Diag-Fase2 (100% in-vocab pese a ello). |
| E5 | **[HECHO]** Frames neutros = ceros crudos idénticos al padding: `src/mslm/dataloader/synthetic_temporal.py:169` inserta `kp.new_zeros(...)` *después* de normalizar cada clip (`:132-134`), y `pad_sequence` (`:233`) también rellena con ceros. El TCN (convolución con padding=1) mezcla ±2 frames a través de las fronteras gap/clip y clip/padding. Gap y padding son indistinguibles para el modelo. | Lectura de código; impacto acotado (los gaps son 0–8 frames). |

No se encontraron bugs en: máscara causal (`video_token_decoder.py:30-34`), construcción BOS/EOS del teacher forcing (`:37-65`), greedy decode y manejo de EOS/no-EOS (`:152-189`, prohibe EOS en paso 0 en `:173`), `strict_exact`/TER/edit_sim (`src/mslm/utils/sequence_metrics.py`), acumulación y clipping (`train_video_token_decoder.py:291-309`), freeze efectivo (`video_token_decoder.py:192-217`; verificado por grupos del optimizador `:274-288`), lineage y separación signer (`:85-99,355-356`; manifiesto y hashes verificados), correspondencia tokenizer↔IDs (BOS=2=`<bos>`, EOS=106=`<end_of_turn>`, PAD=0=`<pad>` verificados contra el tokenizer Gemma real), y paridad de seeds AR/CIF en la evaluación pareada (`:471` y `:609-611` usan `seed+200_000` idéntico).

---

## 3. Incongruencias y riesgos probables (no causa raíz)

- **[INFERENCIA]** Techo de identidad ~70%: un probe lineal sobre features congeladas clasifica las 64 glosas al 70.3% en signer held-out (99.8% en signers de train) (Diag-D). El ST-GCN + `linear_hidden` congelados (entrenados para CIF) fijan ese techo; la brecha train/heldout indica que el problema restante es *generalización entre signers*, no capacidad.
- **[HECHO]** Exposure bias real pero secundario: teacher-forcing token acc = 63.2% vs greedy edit_sim 0.26 (Diag-B). La acc por posición con prefijo gold: pos0=0.19, pos1=0.91, resto≈0.6 — el fallo se concentra en el *primer token de cada seña* (elegir qué seña sigue), exactamente lo que el orden ausente no permite resolver.
- **[HECHO]** Subconteo sistemático: predice 3.5–4.0 glosas cuando el target tiene 4.88 (del=0.23–0.34 por token; ins=0.05–0.10). corr(len_pred, len_target)=0.72–0.77 y EOS acc con prefijo gold = 93% ⇒ EOS aprende duración razonablemente; el subconteo viene de omitir señas enteras, no de EOS prematuro aleatorio.
- **[HIPÓTESIS]** El promedio espacial (`temporal_sign_prompt.py:433`, `.mean(dim=-1)` sobre nodos) pierde detalle de configuración manual; contribuiría al techo de 70% pero no hay medición separada.
- **[HECHO]** Riesgo latente en `optimizer_for` (`train_video_token_decoder.py:274-288`): los parámetros congelados de ST-GCN/`linear_hidden` quedan en el grupo "decoder" a lr 3e-4; si alguien los des-congela sin tocar el optimizador, se entrenarían 10× más rápido de lo previsto.
- **[HECHO]** Ambigüedad de código de glosas mínima: solo 1 par prefijo ('Leche' ⊂ 'Leche dulce'); 0/2688 targets fallan el parseo. La falta de separador **no** es la causa del fallo actual, aunque un separador simplificaría la segmentación.
- **[HECHO]** `label_tokens` asume ≤4 tokens por glosa (`train_video_token_decoder.py:140-143`) y `build_teacher_forcing` lanza error >32 tokens: 8 clips × 4 = 32 justo en el límite — frágil ante cualquier cambio de tokenizer.

## 4. Evidencia extraída (logs, código, checkpoints)

**Curvas y selección [HECHO]** (`fold*/base/metrics.jsonl`): loss monótono 5.2→2.24 sin converger; exact inner-val plano en 1–3% desde e04; eos_rate=1.00 desde e01; best epochs 10/13/10.

**Predicciones outer-test [HECHO]** (896 pares/fold, `outer_test.json` + `fold*_cif_comparator.json`):

| Métrica | fold4 AR | fold4 CIF | fold5 AR | fold5 CIF | fold6 AR | fold6 CIF |
|---|---|---|---|---|---|---|
| strict_exact | 0.022 | 0.023 | 0.023 | 0.049 | 0.018 | 0.026 |
| edit_sim | 0.258 | 0.480 | 0.239 | 0.558 | 0.213 | 0.407 |
| sub/ins/del por token | .37/.06/.34 | .35/.01/.16 | .45/.08/.26 | .33/.07/.05 | .48/.10/.23 | .44/.15/.02 |
| exact-length rate | 0.137 | 0.172 | 0.166 | 0.307 | 0.182 | 0.217 |

- 100% de los tokens predichos ∈ vocabulario efectivo (118 IDs); masa top-5 pred≈target (0.11 vs 0.10) ⇒ **sin colapso a frecuentes**.
- Nivel glosa: 92–96% de predicciones parsean a glosas completas; precision .52–.56, recall .40–.45; multiset-exact (glosas correctas ignorando orden) = 4.6–4.9% ≈ 2× strict_exact; **orden pareado entre glosas comunes = 0.47–0.53**.
- Pareado AR vs CIF: fold4 14W/15L/867T, fold5 12W/35L/849T, fold6 12W/19L/865T — CIF domina en edit_sim por su ventaja estructural en *orden* (integración monótona) y *conteo* (del=0.02–0.16), no en identidad (sub CIF 0.33–0.44 ≈ AR).
- Acc posicional AR decae 0.21→0.05 de pos0 a pos9 (drift autoregresivo).

**Diagnósticos GPU sobre `fold4/checkpoint_closed.pt` [HECHO]** (Diag A–F, 64 s):
- (A) permutar segmentos: métricas invariantes, 86.3% predicciones idénticas;
- (B) TF acc 63.2%, entropía media 2.34 nats (max 12.5), margen top1-top2 = 3.10 — el modelo está *confiado*, no difuso;
- (C) single-sign exact = 25.0% (vs ~2% multi-seña);
- (D) probe glosas 70.3% heldout / 99.8% train;
- (E) probe de posición intra-clip R²=0.577 (fase local del movimiento; no sirve para ordenar segmentos);
- (F) embedding: cos(init CIF, final) = 0.963 en los 118 efectivos, normas efectivos 1.92→2.10 vs resto 0.80→1.15, ||EOS|| 0.80→1.40, cosenos entre efectivos ≈ 0 (max 0.62) — geometría sana; el entrenamiento *refinó* la init, no la destruyó.

**Tests [HECHO]**: existe test de overfit (`tests/models/test_video_token_decoder.py:189`) que fuerza strict_exact=1 en un minibatch ⇒ el loop de entrenamiento funciona. Ningún test comprueba sensibilidad al orden temporal, tasa in-vocab, ni parseabilidad a glosas — por eso el fallo fue silencioso.

## 5. Causa raíz más probable

**[INFERENCIA a partir de HECHOS E1 + Diag-A + orden=0.5]** Cadena causal:

1. La salida del encoder es (contenido local, fase local) por frame, **sin posición global** — arquitectónicamente imposible recuperar el orden de los segmentos (E1).
2. La cross-attention del decoder solo puede recuperar una *bolsa de señas*; identifica ~55% de las señas presentes (limitado además por el techo de identidad de 70% del encoder congelado, Diag-D) y las emite en orden arbitrario (0.5) con subconteo (~3.7 de 4.9).
3. strict_exact ≈ P(identidad)^k × P(orden) × P(conteo) colapsa a ~2% para k≈4.9.
4. El exposure bias (TF 63% vs greedy 26%) amplifica el daño en greedy: cada seña mal elegida arrastra las siguientes posiciones.
5. CIF supera al AR en edit_sim (0.41–0.56 vs 0.21–0.26) precisamente porque su integración monótona le da orden y conteo estructurales — consistente con que la única diferencia de resultados esté en orden/conteo y no en identidad.

Por qué el decoder AR "no mejora CIF": no es que CIF sea bueno (identidad 55–65%), es que el AR compite sin la única ventaja estructural que CIF sí tiene.

## 6. Arquitectura recomendada para Imitator

**Restricción de producto (2026-07-03, decisión del investigador principal):** Imitator predice tokens Gemma directamente; la clasificación a nivel de glosa queda descartada en cualquier escenario. Todo lo que sigue mantiene el objetivo token-level. Nota clave: el "vocabulario restringido" NO viola esta restricción — es una reparametrización de la capa de salida en la que cada una de las 121 filas *es* un ID Gemma concreto (mapeo biyectivo denso↔Gemma almacenado en el checkpoint); el modelo sigue emitiendo secuencias de IDs Gemma, y cuando el dataset crezca solo crece la tabla, no la arquitectura. El propio protocolo original lo listaba como opción legítima.

### Propuesta principal: **"Imitator-JTA"** — decoder AR token-level + memoria posicional + CTC token-level auxiliar

El diseño ataca cada fallo *medido* sin cambiar el objetivo. Es el esquema joint CTC/attention estándar de ASR (Kim/Watanabe et al. 2017), aplicado con tokens Gemma como unidades:

```
keypoints [B,T,111,2]
  → ST-GCN congelado ep0-4 (ft lr 1e-5 desde ep10, E3) → linear_hidden + mean espacial → [B,128,T]
  → TCN (2 capas) → +PE sinusoidal → TransformerEncoder → memoria M [B,T,128]   ← PE NUEVO (fix E1)
  ├─ Cabeza CTC token-level: Linear 128→122 (121 IDs Gemma efectivos + blank) → [B,T,122]
  │    CTC loss contra la secuencia de tokens Gemma (remapeada, L≤33)
  └─ TransformerDecoder causal (2 capas, 4 heads, FFN 512, como hoy)
       target: embedding 121×128 (tied con salida) + PE aprendido (como hoy)
       memoria: M + PE  →  logits [B,L,121]  →  CE sin label smoothing
Loss total = 0.7·CE_AR + 0.3·CTC        (λ fijo, sin sweep)
Inferencia: greedy AR idéntico al actual → remapeo a IDs Gemma
```

- **Formas**: M [B,T,128] con T≈100–1500; logits AR [B,≤33,121]; logits CTC [B,T,122]; targets = los mismos `token_ids` actuales remapeados por la tabla (creada con `label_tokens` sobre los signers de train; los 118 IDs son globales al dataset, verificado).
- **Objetivo**: CE (teacher forcing, `build_teacher_forcing` sin cambios) + `F.ctc_loss(zero_infinity=True)`. T≥L garantizado (≥12 frames por token en el peor caso).
- **Init**: encoder desde `diag_foldN_promotion/checkpoint_best.pt` vía `initialize_from_cif_checkpoint`; embedding/salida AR = las 121 filas correspondientes de `token_head.classifier.weight` (geometría sana, Diag-F); cabeza CTC aleatoria.
- **Freeze**: idéntico al actual (decoder+cabezas ep0–4; TCN/transformer lr 3e-5 desde ep5) + unfreeze ST-GCN lr 1e-5 desde ep10 solo si E3 lo valida. 30 epochs (E3 de §2: la loss nunca convergió con 15).
- **Decodificación**: greedy AR como hoy (EOS prohibido en paso 0). Opcional barato si el subconteo persiste: usar el conteo de la cabeza CTC (nº de emisiones no-blank colapsadas) para vetar EOS hasta haber emitido ≥ conteo−2 tokens — heurística de inferencia, sin tocar el entrenamiento.
- **Manejo de longitud**: EOS (como hoy) + presión estructural de CTC: la pérdida CTC solo baja si *todos* los tokens del target encuentran frames — ataca directamente las omisiones (del=0.23–0.34, el mayor término de error tras las sustituciones).
- **Pérdidas auxiliares**: solo la CTC. Nada de supervisión con `boundaries` en la principal (no existirá en datos continuos reales).
- **Parámetros**: encoder ~1.1M + decoder ~0.53M + embedding 15K + CTC head 16K ≈ **1.7M** (vs 34.6M actuales; el 95% del modelo actual era la tabla de 262 400 filas de la que solo se usaban 121). Memoria <2 GB con batch 8 ⇒ 30 epochs cuestan ≈ lo que 15 costaban.
- **Mecanismo por el que debe superar a CIF y al AR actual** (cada pieza ligada a evidencia):
  1. PE en memoria → el decoder *puede* ordenar (Diag-A demostró que hoy es imposible: 86.3% invariante a permutación; orden 0.5).
  2. CTC auxiliar → obliga a la memoria a ser decodificable monótonamente frame→token: identidad y orden quedan anclados en el encoder aunque la cross-attention tarde en aprender — exactamente la ventaja estructural por la que CIF gana hoy en edit_sim (0.48 vs 0.26), trasladada a tokens.
  3. Vocab 121 + sin smoothing → gradiente concentrado (E4); el 100% in-vocab del run actual demuestra que las 262 279 filas restantes solo costaban memoria.
  4. TF acc ya es 63% con memoria sin orden (Diag-B); el gap TF→greedy se cierra desde ambos lados (mejor memoria ⇒ mejor pos0, que hoy es 0.19; CTC ⇒ menos omisiones que desalinean el prefijo).
  - Orden de magnitud esperado: con identidad por seña ~0.75 y orden/conteo estructurales, strict_exact ≈ 0.75^4.9×0.8 ≈ 15–20% (vs 2%).
- **Riesgos**: (1) λ=0.3 subóptimo — se acepta, sin sweep; (2) CTC con vocab de sub-palabras puede alinear tokens de una seña en frames desplazados — impacto solo en la auxiliar, no en la decodificación; (3) el techo de identidad 70% del encoder congelado limita todo — por eso E3 existe; (4) T hasta ~1500: CTC O(T·L) ≈ 50K celdas, trivial.

### Alternativa A: **CTC token-level puro** (decodificación sin AR)

*Gratis desde el mismo checkpoint JTA*: colapsar la cabeza CTC (argmax por frame, quitar repeticiones y blanks) → secuencia de IDs Gemma. Sin exposure bias, orden y conteo estructurales. Sirve de ablation decisiva: si supera al greedy AR del mismo modelo, el decoder AR no está aportando y la interfaz token-level se sostiene sola sobre CTC; si pierde, el AR aporta modelado intra-glosa (pos1 acc 0.91 sugiere que sí). Riesgo: independencia condicional entre frames — mitigada por el prior nulo entre señas (orden aleatorio por construcción) pero real dentro de cada glosa. Costo de evaluarla: un flag en el script de eval.

### Alternativa B: **atención cross guiada por centros de token** ("guided attention", solo si JTA no rompe el techo)

Añadir a JTA una pérdida KL entre los pesos de cross-attention de la última capa del decoder (promedio de heads) y una gaussiana centrada en el centro temporal de cada token objetivo — los centros ya existen: `CIFAggregator.token_centers` (`src/mslm/models/temporal_sign_prompt.py:48-85`) los computa desde `boundaries`/`token_spans` sintéticos. Ataca el binding token↔tiempo con supervisión directa. Riesgo principal y por lo que es alternativa: esa supervisión solo existe en datos sintéticos; un modelo que dependa de ella puede no transferir a señas continuas reales. Anular con warmup→decay (peso 0.5→0 en ep0–10) si se usa.

Descartadas: **clasificador de glosas / CTC-glosa / segment-then-classify** (violan la restricción de producto), RNN-T (mismo prior monotónico que la CTC auxiliar con mucho más costo de implementación y memoria O(T·L·V)), Perceiver/resampler (comprime la memoria pero no añade orden — ortogonal a la causa raíz), monotonic attention dura (MoChA: complejidad alta; la CTC auxiliar da el mismo prior con 16K params), embeddings reales de Gemma como init (sub-palabras de español sin relación con la señal; la init actual ya demostró no ser el problema, Diag-F).

## 7. Alternativas (máximo dos)

Ver §6: **CTC token-level puro** (ablation gratis del propio JTA, se evalúa siempre) y **guided attention por token_centers** (contingencia si JTA no supera a CIF; no transfiere a real, usar solo como warmup). Ambas mantienen la salida en tokens Gemma.

## 8. Plan experimental de 15 horas (1× RTX 4060 Ti 16 GB, 55 GB libres)

Referencia de costo medido: pipeline actual completo por fold (comparator+train 15 ep+eval) ≈ 33 min (`paper_run.log`). CTC con 1.2M params y sin softmax 262K será ≤ igual; presupuesto con margen 2×.

| # | Qué | Criterio de éxito / abandono | Est. |
|---|-----|------------------------------|------|
| S0 | Smoke: unit tests nuevos (§9) + forward/backward JTA en CPU con fixture (PE, remapeo 121, CTC) | todo verde | 0.5 h |
| S1 | **Overfit deliberado**: JTA, 32 secuencias fijas, 200 steps | strict_exact=1.0 (greedy AR) en esas 32; si no → bug, parar | 0.5 h |
| S2 | Diagnóstico pre-entrenamiento barato: probe *por-frame* de identidad (no mean-pool) sobre encoder congelado fold4 | fija el techo de identidad esperable para la cabeza CTC; solo informativo | 0.5 h |
| E1 | **Dev AR-fix** fold4 (solo PE memoria + vocab 121 + sin smoothing, 30 epochs, sin CTC) — falsación de la causa raíz y baseline token-level | ÉXITO mínimo: orden pareado (script Fase2b) > 0.6 a 15 epochs — si sigue ≈0.5, el diagnóstico E1 era incompleto y se re-audita antes de seguir; ABANDONAR la variante si además edit_sim inner < 0.30 | 3 h |
| E2 | **Dev Imitator-JTA** fold4 (E1 + cabeza CTC λ=0.3, 30 epochs). Del mismo checkpoint se evalúan las 2 decodificaciones: greedy AR y CTC-collapse (Alternativa A, gratis) | ABANDONAR si no supera a E1 en edit_sim inner Y ninguna decodificación alcanza edit_sim ≥ 0.48 (CIF fold4); ÉXITO si strict_exact inner ≥ 10% | 3 h |
| E3 | Mejor de {E1,E2} + des-freeze ST-GCN lr 1e-5 desde ep10, fold4 (ataca el techo de identidad 70%, Diag-D) | adoptar solo si mejora strict_exact inner ≥ +3 pp absolutos | 2 h |
| C1 | **Confirmación**: variante ganadora, folds 4–6, 1 seed (23), eval outer con los MISMOS seeds (`seed+200_000`) y comparación pareada vs `fold*_cif_comparator.json` y vs `fold*/base/outer_test.json` (test binomial/McNemar sobre W/L pareados) | ganar a CIF en strict_exact en ≥2 de 3 folds con W−L positivo significativo | 3 h |
| — | Buffer (re-runs, análisis) | — | 2 h |

Total: 14.5 h. Métricas primarias: strict_exact y token_edit_similarity outer-test sobre IDs Gemma (idéntica definición actual, `sequence_metrics` — comparable 1:1 con los runs AR y CIF existentes). Secundarias: orden pareado entre glosas comunes, precision/recall de glosa (como *diagnóstico* de las predicciones token-level, no como objetivo), exact-length rate, gap TF↔greedy. Prohibido: folds 7–10, sweeps de hiperparámetros (λ fijo 0.3), más de 1 seed en dev. Los folds 4–6 se reutilizan como confirmación pareada (mismas 896 muestras por fold); no hay redacción ni generación libre.

## 9. Tests que deben añadirse

1. **Sensibilidad al orden** (el que habría detectado todo esto): modelo entrenado (o fixture con PE) debe *degradar* edit_sim al permutar segmentos con `permute_video_segments`; para el modelo actual, asertar lo contrario documenta el bug.
2. **Tasa in-vocab y parseabilidad**: predicciones de eval deben reportar % tokens en vocabulario efectivo y % parseable a glosas (guard-rail de colapso).
3. **Round-trip del mapeo de vocabulario**: `to_gemma(to_dense(ids)) == ids` para los 121 IDs; error explícito si un target contiene un ID fuera de la tabla; la tabla guardada en el checkpoint debe reconstruir exactamente el vocabulario efectivo del manifiesto.
4. **CTC auxiliar**: collapse de repeticiones/blanks correcto en casos borde (tokens repetidos consecutivos — p. ej. dos señas 'Agua' seguidas, blank inicial/final); `ctc_loss` finito con T mínimo y L=33.
5. **Brecha TF vs greedy**: registrar ambas en `metrics.jsonl` cada epoch; alerta si TF_acc − greedy_edit_sim > 0.3 (exposure bias/orden).
6. **Gap ≠ padding**: si se adopta el fix de E5, test de que frames neutros y padding producen máscaras distintas.
7. **Grupos del optimizador**: asertar que ningún parámetro con `requires_grad=False` está en un grupo con lr>0 destinado a entrenarse tras un unfreeze (riesgo señalado en §3).

## 10. Cambios concretos por archivo (parches propuestos — NO aplicados, esperan aprobación)

1. `src/mslm/models/temporal_sign_prompt.py` (encoder, para ambas rutas): añadir PE sinusoidal antes del transformer:
```python
# en STGCNTemporalFrameEncoder.__init__:
self.register_buffer("pe", sinusoidal_pe(4096, hidden_size), persistent=False)
# en forward, antes de self.transformer(...):
x = x + self.pe[: x.size(1)].unsqueeze(0)
```
2. `src/mslm/models/video_token_decoder.py` (núcleo del JTA, ~60 líneas de diff): (a) sumar PE sinusoidal a `frame_features` en `decode_features` (línea 132) — mismo buffer que el del encoder; (b) parámetro `vocab_map: list[int]` (121 IDs Gemma) con `to_dense`/`to_gemma` y embedding/salida de 121 filas; `initialize_from_cif_checkpoint` copia solo esas 121 filas del clasificador (línea 284-286); `greedy_decode` devuelve IDs Gemma vía `to_gemma`; (c) cabeza CTC opcional `self.ctc_head = nn.Linear(128, 122)` y método `ctc_logits(features)`; (d) exponer `label_smoothing` con defecto 0.0.
3. `scripts/train/train_video_token_decoder.py`: (a) en `train_epoch`, loss = `0.7*CE + 0.3*F.ctc_loss(...)` sobre los mismos `token_ids` remapeados; (b) construir `vocab_map` con `label_tokens` + BOS/EOS/PAD y guardarlo en el checkpoint/`config.json`; (c) `--epochs 30`; (d) flag de eval `--decode {ar,ctc}` para la ablation de la Alternativa A (misma `sequence_metrics` sobre IDs Gemma ⇒ comparable 1:1 con los runs existentes).
5. `scripts/train/train_video_token_decoder.py`: selección por `(token_edit_similarity, strict_exact)` (invertir la tupla de la línea 404) o subir `--val-samples` a ≥2048 (E2 §2).
6. `src/mslm/dataloader/synthetic_temporal.py` (opcional, E5): generar frames neutros como pose de reposo con ruido (`kp[-1] + ε`) en vez de ceros, para distinguirlos del padding.

## 11. Tabla de decisiones

| Componente | Decisión | Motivo |
|---|---|---|
| Protocolo lineage/manifiesto/hashes/closed-checkpoint | **Mantener** | Correcto y valioso (§2, sin hallazgos) |
| Dataset sintético + collate + seeds pareados | **Mantener** (fix opcional de gaps) | Sin bugs; paridad AR/CIF verificada |
| Métricas `sequence_metrics` + eval pareada CIF | **Mantener** | Correctas; añadir métricas de glosa/orden (§9.2) |
| Encoder ST-GCN+TCN+Transformer | **Modificar** | Añadir PE (E1, causa raíz); considerar unfreeze tardío (techo 70%) |
| Tabla de embedding de 262 400 filas | **Eliminar** | 33.6M params sin función; 100% in-vocab demuestra que 121 filas (IDs Gemma reales, mapeo reversible) bastan |
| Init desde `token_head.classifier.weight` | **Mantener** | Geometría sana (Diag-F); copiar solo las 121 filas del vocab efectivo |
| Label smoothing 0.1 | **Eliminar** | E4 |
| Selección por `(strict_exact, edit_sim)` @512 | **Modificar** | E2: primaria edit_sim o val≥2048 |
| Freeze schedule 0–4/5–14 | **Mantener** | Funciona; extender con unfreeze ST-GCN solo vía E3 |
| Decoder AR token-level como línea principal | **Mantener con correcciones** | PE en memoria (causa raíz E1) + CTC token-level auxiliar + vocab 121 = Imitator-JTA (§6); la salida sigue siendo tokens Gemma |

## 12. Riesgos para la validez científica

1. **Folds 4–6 ya observados**: se usaron para confirmar el AR y ahora guiarán el pivote ⇒ la confirmación C1 es *pareada y honesta* pero no es un test en datos vírgenes; cualquier claim del paper debe declarar que folds 7–10 quedan como held-out final jamás tocado.
2. **Eval sobre datos sintéticos**: train y eval provienen del mismo generador de concatenaciones (solo cambia el signer). El strict_exact reportado no habla de lengua de señas continua real (co-articulación, sin gaps de ceros). El orden aleatorio de glosas del generador también implica que *ningún* modelo puede usar prior lingüístico — realista para este dataset, pero hay que decirlo.
3. **Selección con exact≈ruido (E2)**: los checkpoints "best" de los runs publicados son efectivamente aleatorios entre epochs 4–14; las cifras AR confirmatorias son robustas de todos modos porque todas las epochs rinden igual de mal.
4. **Comparación AR vs CIF asimétrica**: CIF usa calibración affine ajustada en inner-val (`fit_affine`) y el AR no tiene ajuste equivalente — sesgo pro-CIF leve y documentado; JTA tampoco usa calibración, así que si gana lo hace con desventaja (más honesto). La heurística opcional EOS-por-conteo-CTC sería el análogo de esa calibración y debe declararse si se activa.
5. **Probe de 70% con mean-pool**: es cota *inferior* del contenido de las features; no probar per-frame podría subestimar el techo (se corrige con S2).
6. **Una sola seed**: todo el protocolo corre con seed 23; las diferencias pareadas W−L con n=896×3 mitigan, pero la varianza entre seeds queda sin estimar.

## Addendum — Resultados E1 (2026-07-03, post-informe)

**[HECHO]** E1 (AR-fix: PE en encoder + vocab 121 + sin label smoothing + selección por edit_sim, 30 epochs, fold 4, `fold4_seed23/e1_pe_vocab121/`) confirma la causa raíz **por intervención**:

| Métrica outer-test fold4 | AR baseline | CIF affine | **E1** |
|---|---|---|---|
| strict_exact | 0.022 | 0.023 | **0.181** |
| token_edit_similarity | 0.258 | 0.480 | **0.630** |
| length_mae | 3.46 | 1.71 | **1.23** |
| orden pareado entre glosas | 0.529 (azar) | — | **0.986** |
| gloss precision / recall | .56 / .40 | — | **.70 / .64** |

- Pareado vs CIF (mismas 896 muestras): **144 wins / 3 losses / 749 ties** — significancia aplastante (binomial p≈1e-38).
- multiset-exact (18.19%) ≈ strict_exact (18.08%): el error de *orden* desapareció como modo de fallo; lo que queda es identidad (~70% precision de glosa) y omisiones residuales.
- Inner-val: mejor epoch por edit_sim ~0.60; meseta desde ~epoch 20 — 30 epochs son suficientes para esta variante.
- S0/S1 previos: 24/24 tests (incluido el test de regresión de PE) y overfit deliberado exact=1.0 en 20 pasadas.

**[HECHO] E2 (JTA con CTC λ=0.3, mismo fold4):** empate estadístico con E1 (outer exact 17.86%, edit 0.614; pareado vs CIF 141W/2L/753T). La decodificación CTC-collapse quedó colapsada a blank en inner-val (Alternativa A muerta). **El CTC auxiliar no aporta y se descarta** — la ganancia entera viene del PE + vocab restringido de E1.

**[HECHO] E3 (E1 + unfreeze ST-GCN lr 1e-5 desde ep10):** en outer fue mejor (exact 19.53%, 154W/0L/742T) pero en **inner-val quedó por debajo de E1** (edit 0.6058 < 0.6146; exact 0.1836 < 0.1914). Como la selección de variante debe hacerse en inner-val (usar outer sería seleccionar sobre el test y quemar la confirmación), **E3 NO se adopta**; queda como hallazgo exploratorio prometedor (descongelar el encoder ataca el techo de identidad de 70%, Diag-D) pendiente de su propia confirmación pre-registrada.

### Confirmación C1 (folds 4–6, E1 entrenado desde cero por fold, seed 23, outer-test pareado)

Variante confirmada: **E1 = PE en encoder + vocabulario Gemma restringido a 121 IDs (mapeo biyectivo) + sin label smoothing + 30 epochs + selección por edit_sim.** ST-GCN+linear_hidden congelados; TCN/transformer lr 3e-5 desde ep5; decoder lr 3e-4.

| fold | CIF exact | AR baseline exact | **E1 exact** | E1 edit_sim | E1 len_mae | pareado vs CIF (W/L/T) |
|---|---|---|---|---|---|---|
| 4 | 0.023 | 0.022 | **0.181** | 0.630 | 1.23 | 144/3/749 |
| 5 | 0.049 | 0.023 | **0.198** | 0.619 | 1.03 | 140/7/749 |
| 6 | 0.026 | 0.018 | **0.169** | 0.585 | 1.09 | 131/3/762 |

**Total pareado n=2688: 415 wins / 13 losses / 2260 ties** — sign test bilateral sobre los 428 pares discordantes **p ≈ 6.4e-105**. E1 mejora la exactitud absoluta ~8× sobre el decoder AR original y ~4–7× sobre CIF affine, en los tres folds, con orden entre glosas ~0.99. Confirmación robusta.

**Estado**: causa raíz confirmada por intervención y variante ganadora confirmada en folds 4–6. Folds 7–10 intactos como held-out final. Checkpoints `.pt` de folds 4–6 eliminados por petición (se conservan métricas, predicciones y configs); código de la variante integrado en `train_video_token_decoder.py` (flags `--encoder-pe --restricted-vocab --label-smoothing 0 --select edit`) y `set_decoder_training_stage`/`optimizer_for` (con `--unfreeze-stgcn-epoch` para reproducir E3). 26/26 tests en verde.

## 13. Recomendación final

**Continuar la ruta token-level: la corrección funciona y está confirmada.** El pivote a glosas queda descartado por decisión de producto y ya no hace falta — la interfaz token-level (salida = IDs Gemma) supera a CIF con la única corrección de la causa raíz. Veredicto empírico (C1, folds 4–6):

1. **La causa raíz era el orden temporal ausente en la memoria visual** (E1 sin PE ≡ bag-of-signs). Añadir PE + restringir el vocabulario a los 121 IDs Gemma efectivos + quitar label smoothing lleva el strict_exact de ~2% a **16.9–19.8%** y el orden entre glosas de 0.53 (azar) a **~0.99**, ganando a CIF en 415 de 428 pares discordantes (p≈6e-105). Confirmado en los 3 folds.
2. **El CTC auxiliar (JTA) no aporta** — empate con E1; descartado por parsimonia.
3. **Descongelar el ST-GCN (E3) es la siguiente palanca prometedora** pero no se adopta todavía: mejoró en outer sin mejorar en inner-val, y adoptarlo sobre el outer contaminaría la confirmación. El techo restante es la identidad de glosa entre signers (~70%, Diag-D); la próxima inversión legítima es un experimento pre-registrado de unfreeze/augmentación del encoder, evaluado contra held-out fresco (folds 7–10), manteniendo intacta la salida token-level.

**Recomendación operativa**: adoptar E1 como nueva línea base de Imitator; abrir una etapa nueva y pre-registrada para el unfreeze del encoder antes de tocar folds 7–10.
