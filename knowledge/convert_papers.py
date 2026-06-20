"""Convierte los papers de knowledge/papers/ a Markdown en knowledge/markdown/.

Soporta dos parsers:
- markitdown (default, local, gratis, pero rompe el layout de 2 columnas en tablas)
- llamaparse (cloud, requiere LLAMA_PARSER_KEY en knowledge/.env, preserva mejor el orden de lectura)

Uso:
    knowledge/.venv/bin/python knowledge/convert_papers.py [--force] [--parser markitdown|llamaparse] [--tier fast|cost_effective|agentic|agentic_plus]
"""
import argparse
import os
from pathlib import Path

PAPERS_DIR = Path(__file__).parent / "papers"
MARKDOWN_DIR = Path(__file__).parent / "markdown"


def convert_with_markitdown(src: Path) -> str:
    from markitdown import MarkItDown

    md = MarkItDown()
    return md.convert(str(src)).text_content


def convert_with_llamaparse(src: Path, tier: str) -> str:
    from dotenv import load_dotenv
    from llama_cloud import LlamaCloud

    load_dotenv(Path(__file__).parent / ".env")
    api_key = os.environ["LLAMA_PARSER_KEY"]
    client = LlamaCloud(api_key=api_key)

    with open(src, "rb") as f:
        result = client.parsing.parse(
            upload_file=f,
            tier=tier,
            version="latest",
            expand=["markdown_full"],
        )
    return result.markdown_full or ""


def convert_all(force: bool, parser_name: str, tier: str) -> None:
    MARKDOWN_DIR.mkdir(exist_ok=True)

    sources = sorted(p for p in PAPERS_DIR.iterdir() if p.is_file())
    if not sources:
        print(f"No se encontraron archivos en {PAPERS_DIR}")
        return

    for src in sources:
        dst = MARKDOWN_DIR / f"{src.stem}.md"
        if dst.exists() and not force:
            print(f"skip  {dst.name} (ya existe, usa --force para regenerar)")
            continue
        try:
            if parser_name == "llamaparse":
                text = convert_with_llamaparse(src, tier)
            else:
                text = convert_with_markitdown(src)
        except Exception as e:
            print(f"error {src.name}: {e}")
            continue
        dst.write_text(text, encoding="utf-8")
        print(f"ok    {src.name} -> {dst.name} ({parser_name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Regenerar incluso si el .md ya existe")
    parser.add_argument("--parser", choices=["markitdown", "llamaparse"], default="markitdown")
    parser.add_argument("--tier", choices=["fast", "cost_effective", "agentic", "agentic_plus"], default="cost_effective",
                         help="Solo aplica con --parser llamaparse")
    args = parser.parse_args()
    convert_all(force=args.force, parser_name=args.parser, tier=args.tier)
