"""Extrae un grafo de conocimiento (nodos Paper/Method/Dataset/Gap + edges) desde
los markdown en knowledge/markdown/ usando la API de DeepSeek (cliente compatible OpenAI).

Uso:
    knowledge/.venv/bin/python knowledge/extract_graph.py [--force]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from pipeline_status import bootstrap_if_missing, load_status, save_status

if TYPE_CHECKING:
    from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

MARKDOWN_DIR = Path(__file__).parent / "markdown"
PAPERS_DIR = Path(__file__).parent / "papers"
OUTPUT_PATH = Path(__file__).parent / "graph.json"
MODEL = "deepseek-chat"

SYSTEM_PROMPT = """Eres un extractor de grafos de conocimiento para papers académicos de \
reconocimiento de lenguaje de señas (CSLR/SLR). A partir del texto markdown de un paper, \
extraés nodos y relaciones siguiendo este esquema, y respondés ÚNICAMENTE con un objeto JSON \
válido (sin texto adicional, sin markdown fences).

NODOS (campo node_type):
- "paper": el paper en sí. Uno por documento. label = título completo. Además de los \
  campos comunes, el nodo paper lleva: authors (lista de strings, nombres de autores tal \
  como aparecen), year (entero, año de publicación si se menciona, si no null), venue \
  (string, ej. "IEEE Access", "CVPRW 2026", null si no se menciona), doi (string si el \
  texto lo da explícitamente, si no null).
- "method": un método/modelo propio o un baseline citado y comparado en resultados \
  (ej. "STNet", "CorrNet", "Temporal Lift Pooling"). NO crear un nodo method por cada \
  módulo interno menor — solo por el método/arquitectura nombrado que se propone o compara.
- "dataset": un dataset de evaluación citado (ej. "PHOENIX14", "CSL-Daily", "Isharah-2000").
- "gap": una limitación o brecha de investigación que el paper identifica explícitamente \
  (ej. "no probaron transfer learning entre datasets"). Solo crear un nodo gap si el paper \
  lo menciona como limitación/trabajo futuro, no inventar gaps.

Cada nodo tiene: id, label, node_type, source_file (y los nodos paper además authors, \
year, venue, doi). Reglas de id: lowercase, solo [a-z0-9_], formato {entidad_normalizada}. \
Determinístico: la misma entidad (ej. "PHOENIX14") debe producir siempre el mismo id sin \
importar qué paper la mencione, para que los nodos se compartan entre papers (ej. \
dataset_phoenix14).

EDGES (campo relation):
- "proposes": paper -> method (el método que el paper propone como contribución propia).
- "evaluated_on": paper -> dataset, con metric/metric_value si el texto da un número \
  (ej. metric="WER", metric_value="19.2").
- "compares_against": paper -> method (un baseline contra el que se compara), con \
  metric/metric_value del baseline si está disponible.
- "identifies_gap": paper -> gap.
- "addresses_gap": method -> gap, SOLO si el método de ESTE MISMO paper resuelve \
  explícitamente esa limitación (raro, normalmente se omite).
- "semantically_similar_to": method -> method, marcada SIEMPRE como INFERRED, cuando dos \
  métodos (de papers distintos) resuelven el mismo problema con enfoques distintos sin \
  relación estructural directa. Usar con moderación, solo cuando la similitud es genuina.

Cada edge tiene: source, target, relation, confidence, confidence_score, y opcionalmente \
metric/metric_value (omitirlos si el texto no da un número).

confidence y confidence_score (REQUERIDO en cada edge, nunca 0.5 por defecto):
- EXTRACTED: la relación está explícita en el texto (texto lo dice directamente). \
  confidence_score = 1.0 siempre.
- INFERRED: inferencia razonable no dicha explícitamente. Elegir EXACTAMENTE uno: \
  0.95 (evidencia estructural directa), 0.85 (inferencia fuerte), 0.75 (inferencia \
  razonable que requiere interpretación), 0.65 (inferencia débil), 0.55 (especulativo \
  pero plausible). Si no calza ninguno, marcar AMBIGUOUS en vez de usar un valor bajo.
- AMBIGUOUS: incierto, score 0.1-0.3.

No inventes relaciones. Responde con este shape JSON exacto:
{"nodes": [
  {"id": "...", "label": "...", "node_type": "paper", "source_file": "...", "authors": ["..."], "year": 2025, "venue": "...", "doi": null},
  {"id": "...", "label": "...", "node_type": "method|dataset|gap", "source_file": "..."}
 ],
 "edges": [{"source": "...", "target": "...", "relation": "...", "confidence": "EXTRACTED|INFERRED|AMBIGUOUS", "confidence_score": 0.0, "metric": "...", "metric_value": "..."}]}"""


def extract_paper(client: OpenAI, md_path: Path) -> dict:
    text = md_path.read_text(encoding="utf-8")
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=8000,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Archivo: {md_path.name}\n\n{text}"},
        ],
    )
    usage = response.usage
    cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0)
    cache_miss = getattr(usage, "prompt_cache_miss_tokens", 0)
    print(
        f"  usage: prompt={usage.prompt_tokens} cache_hit={cache_hit} "
        f"cache_miss={cache_miss} completion={usage.completion_tokens}"
    )
    return json.loads(response.choices[0].message.content)


def needs_extract(entry: dict | None, force: bool) -> bool:
    if force:
        return True
    if entry is None:
        return True
    return not entry.get("extracted", False)


def merge_graph(existing: dict, new_nodes: list[dict], new_edges: list[dict]) -> dict:
    nodes_by_id = {n["id"]: n for n in existing.get("nodes", [])}
    for n in new_nodes:
        nodes_by_id[n["id"]] = n
    edges = list(existing.get("edges", [])) + list(new_edges)
    return {"nodes": list(nodes_by_id.values()), "edges": edges}


def main(force: bool) -> None:
    from openai import OpenAI

    status = load_status()
    bootstrap_if_missing(status, PAPERS_DIR, MARKDOWN_DIR, OUTPUT_PATH)
    papers_status = status["papers"]

    if force or not OUTPUT_PATH.exists():
        graph = {"nodes": [], "edges": []}
    else:
        graph = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))

    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url="https://api.deepseek.com")

    processed = 0
    for md_path in sorted(MARKDOWN_DIR.glob("*.md")):
        stem = md_path.stem
        entry = papers_status.get(stem)
        if not needs_extract(entry, force):
            print(f"skip  {md_path.name} (ya extraido)")
            continue

        print(f"extrayendo {md_path.name}...")
        result = extract_paper(client, md_path)
        graph = merge_graph(graph, result["nodes"], result["edges"])
        papers_status[stem] = {**(entry or {}), "extracted": True}
        processed += 1

        OUTPUT_PATH.write_text(
            json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        save_status(status)
        print(f"  -> {len(result['nodes'])} nodes, {len(result['edges'])} edges")

    if processed == 0:
        print("Nada para extraer, todos los papers ya estaban procesados (usa --force para regenerar todo)")
    else:
        print(f"Grafo guardado en {OUTPUT_PATH} ({len(graph['nodes'])} nodes, {len(graph['edges'])} edges)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
