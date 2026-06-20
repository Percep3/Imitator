---
name: design-experiment
description: Use when asked to design an experiment, propose a research direction, or evaluate a hypothesis against the literature loaded under knowledge/ (papers in knowledge/papers/, structured KB in knowledge/kb/ and knowledge/graph.json) — designs an experiment (hipótesis/baseline/métrica/ablation) citing only data that actually exists in that knowledge base, never inventing metrics, datasets, or baselines.
---

# Design Experiment from Literature

Diseña experimentos citando exclusivamente la literatura ya estructurada en
`knowledge/`. Nunca inventes una métrica, dataset o baseline — si el dato no está
en el knowledge base cargado, decilo explícitamente en la respuesta en vez de
aproximarlo.

## Navegación jerárquica del knowledge base

El KB tiene 4 niveles, de más barato/resumido a más caro/detallado. Empezá siempre
en el nivel 0 y subí de nivel SOLO si el nivel actual no tiene el dato que
necesitás. No leas `knowledge/markdown/` por defecto — eso infla el contexto
innecesariamente para preguntas que las tablas ya resuelven.

| Nivel | Archivo(s) | Cuándo leerlo |
|---|---|---|
| 0 | `knowledge/kb/papers_metadata.json` | Siempre primero — panorama del corpus: qué papers hay, autores/año/venue/doi |
| 1 | `knowledge/kb/extraction_table.json` + `knowledge/kb/research_gaps.md` | Siempre segundo — métricas/baselines ya tabulados (`own_result`/`baseline_comparison`) y gaps/limitaciones en texto. Suficiente para la mayoría de hipótesis |
| 2 | `knowledge/graph.json` | Solo si la pregunta necesita relaciones que las tablas planas no capturan — ej. edges `semantically_similar_to` entre métodos de papers distintos, o `addresses_gap` |
| 3 | `knowledge/markdown/<paper>.md` | Solo si necesitás una cita textual exacta o un detalle de implementación que no esté en los niveles 0-2. Al usar este nivel, citá `source_file` en la respuesta |

Si terminás escalando a nivel 2 o 3, mencioná brevemente por qué los niveles
anteriores no bastaban (una frase) — eso es lo que hace auditable la navegación.

## Constraints anti-alucinación

- Solo usá métricas, datasets, métodos o baselines que existan en el KB cargado.
  Nunca inventes un número o una comparación que no esté documentada.
- Toda afirmación cuantitativa debe citar su fuente: paper + archivo + métrica
  (ej. "STNet, IEEE Access 2025, WER=19.3 en PHOENIX14 dev
  [kb/extraction_table.json]").
- Si dos papers evalúan en datasets distintos, no los mezcles como si fueran
  comparables — decilo explícitamente (ej. "LiftSign evalúa en Isharah-2000, no
  en PHOENIX14; no hay comparación directa documentada").
- Si el KB no tiene el dato que la pregunta requiere, decilo: "el KB cargado no
  tiene este dato" — no rellenes el hueco con una estimación.

## Template de salida

Respondé siempre con estas 6 secciones, en este orden:

```
## Hipótesis
## Baseline (de la literatura, citado)
## Métrica objetivo (de la literatura, citada)
## Plan de ablation
## Gaps que aborda (si alguno, citado de research_gaps.md)
## Fuentes citadas
```

- **Hipótesis**: una frase, lo que se espera demostrar.
- **Baseline**: el método/resultado de la literatura contra el que se compara,
  con su valor de métrica y fuente.
- **Métrica objetivo**: la métrica usada en la literatura cargada para este tipo
  de tarea (no inventes una métrica nueva si la literatura usa otra).
- **Plan de ablation**: 2-4 pasos concretos y verificables.
- **Gaps que aborda**: solo si el experimento conecta con un gap de
  `research_gaps.md`; si no aplica, escribir "Ninguno de los gaps documentados".
- **Fuentes citadas**: lista plana de cada paper/archivo del KB usado en la
  respuesta.

No agregues secciones fuera de este template. Esta skill solo diseña el
experimento — no genera código ni verifica automáticamente las citas (eso es
trabajo futuro separado).
