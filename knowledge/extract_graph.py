"""Extrae un grafo de conocimiento (nodos Paper/Method/Dataset/Gap + edges) desde
los markdown en knowledge/markdown/ usando la API de DeepSeek (cliente compatible OpenAI).

Uso:
    knowledge/.venv/bin/python knowledge/extract_graph.py [--force]
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

MARKDOWN_DIR = Path(__file__).parent / "markdown"
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


def main(force: bool) -> None:
    if OUTPUT_PATH.exists() and not force:
        print(f"{OUTPUT_PATH.name} ya existe, usa --force para regenerar")
        return

    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url="https://api.deepseek.com")

    all_nodes: dict[str, dict] = {}
    all_edges: list[dict] = []

    for md_path in sorted(MARKDOWN_DIR.glob("*.md")):
        print(f"extrayendo {md_path.name}...")
        result = extract_paper(client, md_path)
        for node in result["nodes"]:
            all_nodes[node["id"]] = node
        all_edges.extend(result["edges"])
        print(f"  -> {len(result['nodes'])} nodes, {len(result['edges'])} edges")

    OUTPUT_PATH.write_text(
        json.dumps({"nodes": list(all_nodes.values()), "edges": all_edges}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Grafo guardado en {OUTPUT_PATH} ({len(all_nodes)} nodes, {len(all_edges)} edges)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
