import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "experiments/video_token_decoder/paper"


def test_every_citation_key_is_defined_and_every_bib_entry_is_used():
    tex = (PAPER / "imitator_e1.tex").read_text(encoding="utf-8")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
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
    mapping = (PAPER / "KNOWLEDGE_CITATION_MAP.md").read_text(encoding="utf-8")
    missing = {name for name in markdown_names if f"`{name}`" not in mapping}
    assert missing == set()


def test_confirmatory_result_macros_are_defined_in_placeholder_file():
    tex = (PAPER / "imitator_e1.tex").read_text(encoding="utf-8")
    macros = (PAPER / "results_macros.tex").read_text(encoding="utf-8")
    referenced = set(re.findall(r"\\(Fold(?:Seven|Eight)\w+|H(?:One|Two|Three)Outcome|OverallOutcome)", tex))
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", macros))
    assert referenced <= defined
