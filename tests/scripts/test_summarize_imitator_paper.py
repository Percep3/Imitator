import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "summarize_imitator_paper_test_module",
    ROOT / "scripts/eval/summarize_imitator_paper.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_render_markdown_keeps_missing_cells_explicit():
    results = {
        "folds": [
            {
                "fold": 1,
                "role": "development",
                "cif_affine": None,
                "ar_baseline": None,
                "e1": None,
                "paired_e1_vs_cif": None,
            }
        ],
        "confirmatory_hypotheses": {
            "H1": {"fold_passes": [], "pass": None},
            "H2": {"fold_passes": [], "pass": None},
            "H3": {"fold_passes": [], "pass": None},
        },
        "full_confirmation": None,
        "development_ablations": [],
    }
    rendered = MODULE.render_markdown(results)
    assert "| 1 | development | — | — | — | — | — | — | — |" in rendered
    assert "Confirmación completa:** pendiente" in rendered


def test_percent_formats_proportions_as_percent_values():
    assert MODULE.percent(0.181) == "18.1"
    assert MODULE.percent(None) == "—"
