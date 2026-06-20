"""Tracking de estado del pipeline de papers (papers/ -> markdown/ -> graph.json -> kb/).

Permite que convert_papers.py y extract_graph.py sepan qué papers ya fueron
procesados, para no reconvertir PDFs sin cambios ni volver a llamar al LLM
sobre papers ya extraídos.
"""
import hashlib
import json
from pathlib import Path

STATUS_PATH = Path(__file__).parent / "pipeline_status.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def load_status(path: Path = STATUS_PATH) -> dict:
    if not path.exists():
        return {"papers": {}, "kb_built_at": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_status(status: dict, path: Path = STATUS_PATH) -> None:
    path.write_text(
        json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def bootstrap_if_missing(
    status: dict, papers_dir: Path, markdown_dir: Path, graph_path: Path
) -> dict:
    """Completa entradas faltantes en status['papers'] a partir del estado en
    disco (markdown/ y graph.json), para no reprocesar papers que ya estaban
    convertidos/extraídos antes de que este tracking existiera. No toca
    entradas que ya estén registradas.
    """
    papers_status = status.setdefault("papers", {})

    graph_nodes = []
    if graph_path.exists():
        graph_nodes = json.loads(graph_path.read_text(encoding="utf-8")).get("nodes", [])
    extracted_source_files = {n.get("source_file") for n in graph_nodes}

    if not papers_dir.exists():
        return status

    for pdf in papers_dir.iterdir():
        if not pdf.is_file():
            continue
        stem = pdf.stem
        if stem in papers_status:
            continue
        md_path = markdown_dir / f"{stem}.md"
        if not md_path.exists():
            continue
        papers_status[stem] = {
            "source_hash": sha256_file(pdf),
            "converted": True,
            "extracted": md_path.name in extracted_source_files,
        }
    return status
