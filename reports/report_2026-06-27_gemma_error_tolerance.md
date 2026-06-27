# Tolerancia a error de token vía reconstrucción Gemma (exploración descartada)

**Fecha:** 2026-06-27
**Estado:** experimento descartado, no se incorpora al roadmap. Este documento solo deja constancia de qué se probó y qué se encontró, por si se retoma con otro modelo/prompt más adelante.
**Modelo:** `unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit`
**Salidas crudas (JSON), fuera del repo:** `outputs/v126_temporal/diag_gemma_error_tolerance_20260627.json` (v1) y `diag_gemma_error_tolerance_v2_20260627.json` (v2, metodología corregida).

## Pregunta original

El roadmap fija gates de Etapa 2 (`exact>=0.65/0.80/0.85` por bucket, `token_accuracy_when_count_correct>=0.88`) sobre la salida cruda del Imitator/CIF, sin medir cuánto error tolera el pipeline completo una vez que Gemma corrige el texto aguas abajo. La idea era corromper sintéticamente secuencias de tokens gold a tasas crecientes (`0.0`-`0.50`), pasarlas por una corrección de Gemma, y usar la curva calidad-vs-error para confirmar o relajar esos gates.

## Iteración 1: corrupción cruda + prompt de puntuación v125

Primer intento: pool de sustitución = todos los subtokens que aparecen en alguna gloss del dataset, prompt = `build_gemma_correction_prompt` de producción (few-shot de puntuación, `src/mslm/inference/imitator_tokens.py`).

Resultado: lift negativo en todos los niveles de error. Pero el setup tenía dos problemas reales, señalados en revisión:
1. **Corrupción demasiado agresiva e irrealista.** El pool incluía fragmentos BPE de continuación (piezas que solo tienen sentido pegadas a otro subtoken), así que sustituir una gloss de 1 token producía basura no-lingüística (`foto`→`tram`, `dar`→`perf`) que ni un humano podría revertir.
2. **El prompt nunca pedía corrección semántica.** El few-shot v125 solo demuestra agregar puntuación — nunca corregir una palabra mal reconocida. Era injusto esperar que Gemma "arreglara" algo que el prompt no le pedía arreglar.

## Iteración 2: corrupción realista + prompt de corrección explícito

Se corrigieron ambos problemas:
- Pool de sustitución restringido a palabras completas reales (otras glosas de 1 token del propio dataset), no fragmentos sueltos.
- Prompt nuevo, con instrucción explícita de corregir un error de reconocimiento (few-shot sintético, sin contaminar con las respuestas reales del dataset).
- Se excluyeron las glosas de 1 token: corromper la única palabra de una secuencia de 1 token destruye el 100% de la información, ni un humano podría adivinar el gold — no es una medida útil de tolerancia a error.

Con esa corrección se descubrió además que la mayoría de las glosas "de 2+ tokens" en `dataset1` (64 labels en total) **no son frases multi-palabra**: son palabras únicas que el tokenizer BPE parte en varios subtokens (`dónde`→2 tokens, `hambriento`→4 tokens, `cajón`→3 tokens). Solo 5 de 64 labels son frases reales con más de una palabra (`A tierra`, `Azul claro`, `Darse cuenta de`, `Goma de mascar`, `Leche dulce`). Para el resto, corromper cualquier subtoken interno sigue sin dejar contexto real que ayude a la corrección — es esencialmente el mismo problema que con 1 token, solo que repartido en más piezas.

### Resultado (90→120 filas, labels `>=2` tokens, pool realista, prompt de corrección)

| error_rate | n | exact_before | exact_after | chrf_before | chrf_after | lift_chrf |
|---|---|---|---|---|---|---|
| 0.00 | 120 | 1.000 | 0.950 | 100.00 | 99.16 | -0.84 |
| 0.10 | 120 | 0.000 | 0.000 | 36.67 | 34.58 | -2.10 |
| 0.20 | 120 | 0.000 | 0.000 | 36.67 | 34.58 | -2.10 |
| 0.30 | 120 | 0.000 | 0.000 | 36.67 | 34.58 | -2.10 |
| 0.40 | 120 | 0.000 | 0.000 | 34.29 | 32.35 | -1.94 |
| 0.50 | 120 | 0.000 | 0.000 | 23.30 | 21.85 | -1.45 |

El control en `error_rate=0.0` ahora es limpio (`exact_after=0.95`, lift ~0 — el prompt nuevo no rompe el exact-match agregando puntuación espuria como hacía el v125). Pero en **todo nivel con error real (`>=0.10`) el lift sigue siendo negativo**: ni con sustituciones por palabras reales ni con un prompt que pide explícitamente corregir, Gemma-3n-E2B recupera nada del error introducido.

Ejemplos cualitativos confirmaron el mecanismo: frente a una gloss aislada corrompida (`foto`→`verde`), el modelo no tiene contexto de oración para desambiguar y o bien copia el ruido sin cambios, o ancla su respuesta al patrón superficial del few-shot (contestaba `"nombre"` repetidamente porque era la respuesta de uno de los ejemplos del prompt, no por razonamiento real).

## Por qué se descarta el experimento

No se trata de un bug de implementación corregible con más iteración de prompt — es una limitación estructural de **este dataset + esta tarea**: `dataset1` son glosas aisladas (palabras sueltas o fragmentos de palabra, casi sin frases multi-palabra reales), así que no hay contexto de oración para que un LLM infiera la palabra correcta a partir de una mal reconocida. Eso convierte "calibrar gates vía tolerancia de Gemma" en una pregunta mal planteada para este dataset — no hay manera realista de que Gemma aporte una red de seguridad aquí, independientemente del modelo o prompt usado.

Se investigó además si el cuello de botella podía ser el tokenizer de Gemma 3n (vocabulario de 262,400, `GemmaTokenizer`/SentencePiece, compartido con la familia Gemma 3) — no lo es; sigue tokens de palabra limpios. Quedó pendiente, sin probar, si un modelo de mayor capacidad (Gemma 4, lanzado 2026-04-02, mismo naming E2B/E4B/12B/26B/31B pero tokenizer reentrenado de tamaño similar) cambiaría el resultado — la hipótesis es que el límite es capacidad de razonamiento del modelo, no codificación, pero no se descargó ni probó ningún modelo nuevo.

**No se modifica el roadmap ni los gates de Etapa 2 en base a este experimento.** Si se retoma en el futuro, haría falta o bien un dataset con contexto de oración real, o aceptar que la "corrección downstream" no es una vía válida para relajar gates upstream en este pipeline de glosas aisladas.
