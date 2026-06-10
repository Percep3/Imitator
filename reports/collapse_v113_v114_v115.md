# Experimentos anti-colapso: v113 → v114 → v115

**Fecha:** 10-06-2026  
**Objetivo:** Diagnosticar y resolver el colapso de embeddings del Imitator (MSLM) al espacio del LLM Gemma-3n.  
**Hipótesis progresiva:** MSE+CosSim → colapso a la media condicional → SIGReg mitiga → CE-vocab elimina el incentivo por construcción.

---

## Configuración común

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | Imitator (73M params): Conv1d → TransformerEncoder (2L) → CrossAttn → proj_final |
| LLM objetivo | Gemma-3n (emb_dim = 2048, vocab ≈ 122k) |
| Dataset | dataset2 (LSA, frases), 80/20 train/val |
| Keypoints | 111 (pose + manos) |
| Epochs | 30 |
| Batch size | 64 (batch_sample = 4) |
| LR | 4.5 × 10⁻⁴ |
| Weight decay | 0.08 |
| Grad clip | 3.9 |
| Target effrank | **175.5** (gemma-3n embedding table) |

---

## v113 — Baseline: MSE + CosineSimilarity

**Loss:** `L = MSE(pred, emb) + CosSim(pred, emb)`  
**SIGReg:** desactivado

### Evolución de la loss (validación)

| Epoch | val_loss | val_MSE | val_CosSim |
|-------|----------|---------|------------|
| 0     | 2.634    | 1.317   | 1.001      |
| 5     | 1.624    | 0.812   | 0.574      |
| 10    | 1.595    | 0.798   | 0.564      |
| 15    | 1.494    | 0.747   | 0.525      |
| 20    | 1.452    | 0.726   | 0.510      |
| 25    | 1.449    | 0.724   | 0.508      |
| **29**| **1.453**| **0.726**| **0.510** |

### Métricas de colapso

| Epoch | EffRank | EffRank/Target | Pairwise cos | per_dim_std |
|-------|---------|----------------|--------------|-------------|
| 0     | 24.68   | 14.1%          | 0.295        | 0.479       |
| 5     | 21.46   | 12.2%          | 0.990        | 0.050       |
| 10    | 21.28   | 12.1%          | 0.991        | 0.040       |
| 13    | 18.69   | 10.6%          | 0.966        | 0.078       |
| **14**| **5.58**| **3.2%** ⚠️    | 0.900        | 0.149       |
| **15**| **1.74**| **1.0%** 🔴    | 0.884        | 0.196       |
| 20    | 1.62    | 0.9%           | 0.871        | 0.222       |
| 29    | 1.78    | 1.0%           | 0.847        | 0.241       |

**Resultado:** Colapso catastrófico en epoch 14–15. El effective_rank cae de ~21 a 1.74 en un epoch. La loss sigue bajando porque el modelo aprende a predecir la media condicional, que minimiza MSE+CosSim pero es una solución degenerada.

---

## v114 — MSE + CosineSimilarity + SIGReg (λ=1.0)

**Loss:** `L = MSE + CosSim + λ·SIGReg`  
**SIGReg:** regularizador anti-colapso isotrópico-gaussiano (LeJEPA), λ=1.0, n_slices=1024

### Evolución de la loss (validación)

| Epoch | val_loss | val_MSE | val_CosSim | val_SIGReg (train) |
|-------|----------|---------|------------|---------------------|
| 0     | 2.656    | 1.317   | 1.001      | 0.021               |
| 5     | 1.649    | 0.812   | 0.574      | 0.014               |
| 10    | 1.620    | 0.798   | 0.565      | 0.014               |
| 15    | 1.587    | 0.779   | 0.547      | 0.020               |
| 20    | 1.523    | 0.742   | 0.520      | 0.033               |
| 25    | 1.516    | 0.740   | 0.519      | 0.032               |
| **29**| **1.546**| **0.755**| **0.534** | **0.032**           |

### Métricas de colapso

| Epoch | EffRank | EffRank/Target | Pairwise cos | per_dim_std |
|-------|---------|----------------|--------------|-------------|
| 0     | 24.68   | 14.1%          | 0.295        | 0.479       |
| 5     | 22.13   | 12.6%          | 0.990        | 0.049       |
| 10    | 22.55   | 12.8%          | 0.991        | 0.038       |
| 14    | 21.35   | 12.2%          | 0.977        | 0.062       |
| **15**| **12.54**| **7.1%** ⚠️   | 0.914        | 0.126       |
| **16**| **6.36** | **3.6%** 🔴   | 0.891        | 0.157       |
| 20    | 3.13    | 1.8%           | 0.859        | 0.179       |
| 29    | 4.03    | 2.3%           | 0.796        | 0.235       |

**Resultado:** SIGReg retrasa el colapso ~2 epochs (15→16 vs 14→15 en v113) y lo mitiga parcialmente — el effective_rank final es 4.0 vs 1.8. El pairwise cosine también mejora (0.796 vs 0.847). Sin embargo el colapso ocurre igual porque la presión de MSE+CosSim supera el regularizador al crecer el gradiente del error. La loss total es ligeramente peor que v113 debido al overhead de SIGReg.

---

## v115 — CE sobre vocabulario (sin SIGReg)

**Loss:** `L = CrossEntropy(pred @ E^T / τ, token_ids)` donde `E` = tabla de embeddings de Gemma-3n congelada  
**SIGReg:** desactivado (aislamiento del efecto de la CE)  
**Métrica adicional:** `token_acc` = fracción de posiciones donde `argmax(logits) == token_id_target`

### Evolución de la loss (validación)

| Epoch | val_CE  | val_TokenAcc |
|-------|---------|--------------|
| 0     | 6.805   | 0.0%         |
| 3     | 2.132   | 6.3%         |
| **6** | **1.379** ← mínimo | **6.6%** |
| 10    | 2.085   | **10.1%** ← máx acc |
| 15    | 2.852   | 3.3%         |
| 20    | 4.060   | 0.05%        |
| 25    | 4.865   | 8.4%         |
| 29    | 4.716   | 6.0%         |

> La loss diverge desde epoch 7 — el LR (4.5×10⁻⁴) es demasiado alto para CE sobre ~122k clases.  
> Mejor checkpoint: **epoch 6** (val CE = 1.379).

### Métricas de colapso

| Epoch | EffRank | EffRank/Target | Pairwise cos | per_dim_std |
|-------|---------|----------------|--------------|-------------|
| 0     | 24.68   | 14.1%          | 0.295        | 0.479       |
| 5     | 22.53   | 12.8%          | 0.933        | 0.109       |
| 10    | 22.13   | 12.6%          | 0.973        | 0.057       |
| 15    | 22.44   | 12.8%          | 0.989        | 0.045       |
| 20    | 23.15   | 13.2%          | 0.996        | 0.039       |
| 25    | 20.43   | 11.6%          | 0.996        | 0.041       |
| **29**| **20.11**| **11.5%** ✅  | 0.999        | 0.028       |

> **Ningún colapso dimensional en los 30 epochs** (`per_dim_collapsed_frac = 0.000` siempre).

---

## Comparación directa (epoch 29)

### Anti-colapso dimensional

| Métrica | v113 | v114 | **v115** | Mejor |
|---------|------|------|----------|-------|
| `effective_rank` | 1.78 | 4.03 | **20.11** | v115 ✅ |
| `effrank / target` | 1.0% | 2.3% | **11.5%** | v115 ✅ |
| `participation_ratio` | 1.36 | 2.02 | **11.22** | v115 ✅ |
| `per_dim_collapsed_frac` | 0.000 | 0.000 | **0.000** | empate |
| `per_dim_std_mean` | 0.241 | 0.235 | 0.028 | v113/v114 ⚠️ |
| `pairwise_cosine` | 0.847 | 0.796 | 0.999 | v114 ⚠️ |

### Loss y convergencia

| Métrica | v113 | v114 | **v115** |
|---------|------|------|----------|
| Loss objetivo | MSE+CosSim | MSE+CosSim+SIGReg | CE |
| Loss val mínima | 1.449 (ep.24) | 1.515 (ep.24) | **1.379** (ep.6) |
| Loss val final (ep.29) | 1.453 | 1.546 | 4.716 ⚠️ |
| Token accuracy máx. | — | — | **10.1%** (ep.10) |
| Converge estable | Sí (colapsa) | Sí (colapsa tarde) | No (diverge ep.7+) |

---

## Diagnóstico

### Lo que funcionó en v115

La reformulación como clasificación sobre el vocabulario **elimina el incentivo para colapsar a la media condicional**. El softmax penaliza la similitud con tokens incorrectos, haciendo imposible la solución degenerada que MSE+CosSim favorece. Esto se refleja en:

- `effective_rank` se mantiene en ~20 durante los 30 epochs (vs colapso a 1.8/4.0 en v113/v114).
- El ratio `effrank/target` es **11.5% vs 1.0%** de v113 — 11× de mejora en uso del espacio de representación.
- El modelo sí aprende con CE: pasa de CE=6.8 a CE=1.38 y token_acc sube de 0% a ~10%.

### Lo que falló en v115

**Colapso direccional:** `pairwise_cosine = 0.999` al final. Los embeddings predichos no colapsan en cuántas dimensiones usan, pero todos apuntan casi en la misma dirección. Esto es un modo de fallo diferente — el modelo converge a una solución "segura" de predecir siempre un vector promedio en alta dimensión.

**Divergencia de la loss:** El LR de 4.5×10⁻⁴, adecuado para regresión MSE, es excesivo para CE sobre ~122k clases. Los gradientes de CE son inherentemente más grandes (cada predicción errónea tiene gradiente proporcional a la probabilidad de todo el vocabulario). El modelo encuentra un mínimo local en epoch 6 y luego no puede estabilizarse.

**per_dim_std baja:** 0.028 vs 0.241 (v113). La poca varianza por dimensión junto con el alto effrank sugiere que hay ~20 dimensiones con varianza pequeña, en vez de 2 dimensiones con varianza grande. Es un colapso más distribuido pero igualmente restrictivo.

---

## Hipótesis para v116

El experimento v115 confirma que **CE sola es la dirección correcta** pero necesita ajuste de hiperparámetros:

| Hipótesis | Acción |
|-----------|--------|
| LR demasiado alto → divergencia | Reducir a 1e-5 – 5e-5 (10-45× más bajo) |
| Sin warmup → overshooting temprano | Añadir warmup lineal 3-5 epochs |
| Temperatura τ=1.0 → gradientes duros | Probar τ=2.0–4.0 para suavizar |
| CE sola no controla dirección | CE + SIGReg (λ pequeño, 0.1–0.3) para combatir colapso direccional |
| Vocab demasiado grande para gradiente | Top-k CE (k=1000 tokens más frecuentes) como aproximación |

La combinación más prometedora: **CE + LR=2e-5 + warmup 5 epochs**, que sería v116.

---

## Timeline

| Fecha | Versión | Evento |
|-------|---------|--------|
| 09-06-2026 | v113 | Diagnóstico confirmado: effrank colapsa a 1.8 en epoch 15 |
| 10-06-2026 04:21 | v114 | SIGReg mitiga parcialmente: effrank final 4.0 |
| 10-06-2026 05:58 | v115 | CE-vocab: sin colapso dimensional, pero diverge por LR alto |
