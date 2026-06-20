# Skill `design-experiment` — Fase 3 del pipeline knowledge/

## Contexto

El pipeline `knowledge/` convierte papers (PDF → markdown vía LlamaParse, Fase 1 parcial)
y los estructura en un knowledge base (Fase 2, ya implementada):

- `knowledge/graph.json` — grafo completo (nodos paper/method/dataset/gap, edges con
  relation/confidence/confidence_score/metric/metric_value), generado por
  `extract_graph.py` vía DeepSeek.
- `knowledge/kb/papers_metadata.json` — autores/año/venue/doi por paper.
- `knowledge/kb/extraction_table.json` — resultados propios (`own_result`) y
  comparaciones contra baselines (`baseline_comparison`) con métrica/valor.
- `knowledge/kb/research_gaps.md` — gaps/limitaciones en texto legible + puentes
  `semantically_similar_to` entre métodos de papers distintos.
- `knowledge/markdown/*.md` — texto completo de cada paper (fuente de verdad original).

Se decidió **no construir RAG/vector DB** (Fase 1 completa) por ahora: el corpus es de
solo 2 papers y el KB estructurado ya cabe cómodo en contexto. Esta spec cubre la
Fase 3 del diagrama de arquitectura: el system prompt/skill del agente que diseña
experimentos citando esta literatura, con constraints anti-alucinación.

## Objetivo

Una skill de Claude Code (también usable desde Codex) que, invocada con una pregunta
de investigación o hipótesis, diseña un experimento (hipótesis → baseline → métrica →
ablation plan) citando exclusivamente datos que existen en el KB cargado — nunca
inventa métricas, datasets o baselines.

## Decisiones de diseño

1. **Entregable: skill de proyecto**, no un script standalone ni un prompt estático
   para copiar/pegar. Vive en `.claude/skills/design-experiment/SKILL.md`, scoped a
   este repo porque depende de archivos específicos de `knowledge/`.
2. **Sin llamadas a API propias** — la skill corre dentro de la sesión de Claude
   Code/Codex que la invoca; usa las herramientas de lectura de archivos del propio
   agente (Read/Grep), no hace requests HTTP a Anthropic/DeepSeek por su cuenta.
3. **Navegación jerárquica del KB** (no inyección plana de todo el contexto):

   | Nivel | Archivo(s) | Cuándo leerlo |
   |---|---|---|
   | 0 | `kb/papers_metadata.json` | Siempre primero — barato, da el panorama del corpus |
   | 1 | `kb/extraction_table.json` + `kb/research_gaps.md` | Siempre segundo — suficiente para la mayoría de hipótesis |
   | 2 | `graph.json` | Solo si la pregunta requiere relaciones que las tablas planas no capturan (ej. `semantically_similar_to`, `addresses_gap`) |
   | 3 | `markdown/<paper>.md` | Solo si necesita una cita textual exacta o un detalle no capturado en niveles 0-2; debe citar `source_file` al usar este nivel |

   Regla explícita: no leer `markdown/` por defecto. Solo escalar de nivel cuando el
   nivel actual no tiene el dato necesario, para no inflar el contexto con texto
   completo en preguntas simples.
4. **Constraints anti-alucinación** (texto literal en la skill):
   - Solo usar métricas/datasets/baselines que existan en el KB cargado.
   - Toda afirmación cuantitativa debe citar su fuente (paper, archivo, métrica).
   - Si el KB no tiene el dato, decirlo explícitamente en vez de aproximar o inventar.
5. **Template de salida** (siempre estas secciones, en este orden):
   ```
   ## Hipótesis
   ## Baseline (de la literatura, citado)
   ## Métrica objetivo (de la literatura, citada)
   ## Plan de ablation
   ## Gaps que aborda (si alguno, citado de research_gaps.md)
   ## Fuentes citadas
   ```
6. **Output:** solo respuesta conversacional. No escribe archivos nuevos (el usuario
   decide después si quiere persistir un diseño).

## Fuera de alcance (explícitamente, para esta spec)

- Fase 1 (chunking semántico, embeddings, vector DB) — pendiente, se evaluará si el
  corpus crece más allá de unos pocos papers.
- Fase 4 (checker automático que valida que cada métrica citada exista en el KB,
  generación de código del experimento) — sigue como trabajo futuro separado. Esta
  skill es responsable de citar correctamente, pero no incluye un verificador
  automatizado independiente del propio LLM que sigue la skill.
- Múltiples skills/triggers (ej. comando separado para "explicar un paper") — solo se
  construye `design-experiment` en esta iteración.

## Criterios de aceptación

- Invocar `/design-experiment <pregunta>` produce una respuesta con las 6 secciones
  del template, en orden.
- Toda métrica/baseline/dataset mencionado existe verificablemente en
  `kb/extraction_table.json` o `kb/papers_metadata.json`.
- La skill no lee `markdown/*.md` salvo que la pregunta explícitamente requiera un
  detalle no presente en los niveles 0-2 (verificable leyendo la transcripción: el
  agente debe justificar por qué escaló de nivel).
- Si el usuario pregunta algo no cubierto por el KB cargado (ej. un dataset que no
  está en ningún paper), la skill lo dice explícitamente en vez de inventar un valor.
