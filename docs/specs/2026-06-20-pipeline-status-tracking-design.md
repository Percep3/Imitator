# Tracking incremental del pipeline de papers (knowledge/)

## Problema

El pipeline `knowledge/papers/` → `markdown/` → `graph.json` → `kb/` reprocesa
de más cuando se agregan papers nuevos. En particular, `extract_graph.py`
(la única etapa que llama a un LLM, DeepSeek) es todo-o-nada: si `graph.json`
existe, no procesa nada salvo `--force`, y `--force` reprocesa **todos** los
papers, no solo los nuevos. Esto gasta tokens innecesariamente cada vez que
se agrega un paper a una colección que ya creció.

## Solución

Un archivo `knowledge/pipeline_status.json`, versionado en git (igual que
`graph.json`), que registra por paper si ya fue convertido a markdown y si
ya fue extraído al grafo, junto con un hash del PDF fuente para detectar
reemplazos.

### Esquema

```json
{
  "papers": {
    "SM4465": {
      "source_hash": "sha256:abc123...",
      "converted": true,
      "extracted": true
    }
  },
  "kb_built_at": "2026-06-20T18:30:00Z"
}
```

- Clave = `stem` del archivo (nombre sin extensión), igual que usan
  `convert_papers.py` y `extract_graph.py` hoy para nombrar `.md`.
- `source_hash`: sha256 del PDF en `papers/`, formato `"sha256:<hex>"`.
- `converted` / `extracted`: booleanos de estado por etapa.
- `kb_built_at`: timestamp ISO de la última corrida de `build_kb.py` (solo
  informativo, no gatea nada).

Se implementa un módulo chico `knowledge/pipeline_status.py` con
`load_status()`, `save_status(status)` y `sha256_file(path)`, usado por
`convert_papers.py` y `extract_graph.py`.

### `convert_papers.py`

Para cada PDF en `papers/`:
- Calcula `current_hash`.
- `needs_convert = force or not dst.exists() or entry is None or entry["source_hash"] != current_hash`.
- Si no hace falta: `skip <name> (hash sin cambios)`.
- Si hace falta: convierte, escribe el `.md`, y actualiza el status:
  - `source_hash = current_hash`, `converted = True`.
  - `extracted`: se resetea a `False` **solo si el hash cambió o la entrada no existía** (un nuevo contenido invalida la extracción previa). Si fue un `--force` sobre un PDF sin cambios de hash, se preserva el valor anterior de `extracted`.
- Guarda el status después de cada paper procesado (progreso persistente si se interrumpe a mitad de un batch).

### `extract_graph.py`

- Se elimina el gate todo-o-nada (`if OUTPUT_PATH.exists() and not force: return`).
- Sin `--force`: carga `graph.json` existente como punto de partida (si no existe, arranca vacío). Con `--force`: arranca de cero (mismo comportamiento que hoy), ignorando tanto el `graph.json` existente como los flags `extracted` del status.
- Recorre los `.md` en orden; por cada uno, `needs_extract = force or not status["papers"].get(stem, {}).get("extracted", False)`.
- Si no hace falta: `skip <name> (ya extraído)`.
- Si hace falta: llama al LLM, mergea `nodes` (overwrite por id, comportamiento ya existente) y `edges` (se agregan a la lista, comportamiento ya existente) en las estructuras en memoria, y marca `extracted = True` para ese stem.
- Guarda `graph.json` y `pipeline_status.json` después de cada paper procesado (no al final del loop completo), para no perder progreso ni tokens ya gastados si se interrumpe.

### `build_kb.py`

Sin cambios funcionales — sigue siendo determinístico y gratis (no llama
LLM), así que se reconstruye completo desde `graph.json` en cada corrida.
Único agregado: al final, actualiza `kb_built_at` en el status.

### Caso: PDF modificado (mismo nombre, contenido distinto)

El hash cambia → se detecta como "pendiente" automáticamente y se
reconvierte/re-extrae sin intervención manual. **No** se limpian los nodos/edges
viejos de esa versión anterior del paper en `graph.json` antes de mergear los
nuevos — los nodos se sobreescriben por id (sin duplicar), pero los edges
viejos pueden quedar duplicados u obsoletos junto a los nuevos. Decisión
explícita: mantener la implementación simple. Para una limpieza completa,
el usuario borra la entrada del stem en `pipeline_status.json` y/o corre con
`--force` para regenerar `graph.json` desde cero.

## Fuera de alcance

- No se agrega un flag para forzar el reprocesamiento de un solo paper
  específico — el fallback manual (editar `pipeline_status.json` o usar
  `--force` global) es suficiente para el caso raro de modificar un paper
  existente.
- No se versiona el contenido completo de los PDFs ni se hace diff de texto;
  el hash sha256 del archivo es la única señal de cambio.
