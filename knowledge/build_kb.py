"""Deriva las 3 piezas planas de la Fase 2 (KB estructurado) desde graph.json:
- kb/papers_metadata.json: autores, año, venue, doi por paper
- kb/extraction_table.json: resultados propios y comparaciones contra baselines (métrica/valor)
- kb/research_gaps.md: limitaciones/gaps identificados, en texto legible

graph.json es la fuente de verdad (generado por extract_graph.py); este script solo
reorganiza esos mismos hechos en las formas que consume la Fase 3 (system prompt del
agente). No vuelve a llamar al LLM.

Uso:
    knowledge/.venv/bin/python knowledge/build_kb.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline_status import load_status, save_status

GRAPH_PATH = Path(__file__).parent / "graph.json"
KB_DIR = Path(__file__).parent / "kb"


def mark_kb_built(status: dict, built_at: str) -> dict:
    status["kb_built_at"] = built_at
    return status


def build_papers_metadata(nodes_by_id: dict) -> list[dict]:
    out = []
    for n in nodes_by_id.values():
        if n["node_type"] != "paper":
            continue
        out.append({
            "id": n["id"],
            "title": n["label"],
            "authors": n.get("authors", []),
            "year": n.get("year"),
            "venue": n.get("venue"),
            "doi": n.get("doi"),
            "source_file": n["source_file"],
        })
    return out


def build_extraction_table(nodes_by_id: dict, edges: list[dict]) -> list[dict]:
    rows = []
    for e in edges:
        src, tgt = nodes_by_id.get(e["source"]), nodes_by_id.get(e["target"])
        if not src or not tgt:
            continue
        if e["relation"] == "evaluated_on" and src["node_type"] == "paper":
            rows.append({
                "type": "own_result",
                "paper": src["label"],
                "dataset": tgt["label"],
                "metric": e.get("metric"),
                "value": e.get("metric_value"),
                "confidence": e["confidence"],
            })
        elif e["relation"] == "compares_against" and src["node_type"] == "paper":
            rows.append({
                "type": "baseline_comparison",
                "paper": src["label"],
                "baseline_method": tgt["label"],
                "metric": e.get("metric"),
                "value": e.get("metric_value"),
                "confidence": e["confidence"],
            })
    return rows


def build_research_gaps_md(nodes_by_id: dict, edges: list[dict]) -> str:
    lines = ["# Research gaps identificados en la literatura cargada", ""]
    papers = [n for n in nodes_by_id.values() if n["node_type"] == "paper"]
    for paper in papers:
        gap_edges = [e for e in edges if e["source"] == paper["id"] and e["relation"] == "identifies_gap"]
        if not gap_edges:
            continue
        lines.append(f"## {paper['label']}")
        for e in gap_edges:
            gap = nodes_by_id.get(e["target"])
            if gap:
                lines.append(f"- {gap['label']} (`{gap['id']}`, {e['confidence']})")
        lines.append("")

    bridges = [e for e in edges if e["relation"] == "semantically_similar_to"]
    if bridges:
        lines.append("## Métodos relacionados entre papers (puentes inferidos)")
        for e in bridges:
            src, tgt = nodes_by_id.get(e["source"]), nodes_by_id.get(e["target"])
            if src and tgt:
                lines.append(f"- {src['label']} ~ {tgt['label']} (confidence_score={e['confidence_score']})")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]

    KB_DIR.mkdir(exist_ok=True)

    (KB_DIR / "papers_metadata.json").write_text(
        json.dumps(build_papers_metadata(nodes_by_id), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (KB_DIR / "extraction_table.json").write_text(
        json.dumps(build_extraction_table(nodes_by_id, edges), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (KB_DIR / "research_gaps.md").write_text(
        build_research_gaps_md(nodes_by_id, edges), encoding="utf-8"
    )

    status = load_status()
    save_status(mark_kb_built(status, datetime.now(timezone.utc).isoformat()))

    print(f"KB escrito en {KB_DIR}/: papers_metadata.json, extraction_table.json, research_gaps.md")


if __name__ == "__main__":
    main()
