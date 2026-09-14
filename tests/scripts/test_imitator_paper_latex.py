import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
FULL = PAPER / "content/full"


def manuscript_tex() -> str:
    """Read the universal full-paper content plus its Springer wrapper."""
    paths = [
        PAPER / "formats/springer-lncs/full.tex",
        FULL / "abstract.tex",
        FULL / "body.tex",
        *sorted((FULL / "sections").glob("*.tex")),
    ]
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_every_citation_key_is_defined_and_every_bib_entry_is_used():
    tex = manuscript_tex()
    bib = (PAPER / "shared/references.bib").read_text(encoding="utf-8")
    defined = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set()
    for block in re.findall(r"\\cite\w*\{([^}]+)\}", tex):
        cited.update(key.strip() for key in block.split(","))
    assert cited - defined == set()
    assert defined - cited == set()


def test_every_local_knowledge_markdown_has_a_citation_mapping():
    markdown_names = {
        path.name for path in (ROOT / "knowledge/markdown").glob("*.md")
    }
    mapping = (PAPER / "shared/KNOWLEDGE_CITATION_MAP.md").read_text(encoding="utf-8")
    missing = {name for name in markdown_names if f"`{name}`" not in mapping}
    assert missing == set()


def test_confirmatory_result_macros_are_defined_in_placeholder_file():
    tex = manuscript_tex()
    macros = (PAPER / "shared/results_macros.tex").read_text(encoding="utf-8")
    referenced = set(re.findall(r"\\(Fold(?:Seven|Eight)\w+|H(?:One|Two|Three)Outcome|OverallOutcome)", tex))
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", macros))
    assert referenced <= defined


def test_full_body_includes_every_modular_section_exactly_once():
    root = (FULL / "body.tex").read_text(encoding="utf-8")
    section_names = {path.name for path in (FULL / "sections").glob("*.tex")}
    included = re.findall(r"\\input\{\.\./\.\./content/full/sections/([^}]+)\}", root)
    assert set(included) == section_names
    assert len(included) == len(set(included))
