#!/usr/bin/env python3
"""Build the auditable master JSON/Markdown tables for the Imitator E1 paper."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from scripts.eval.robustness_video_token_decoder import (
    BOOTSTRAP_SEED,
    bootstrap_mean_ci,
    paired_cif_result,
)
from scripts.train import train_video_token_decoder as protocol
from src.mslm.dataloader.isolated_keypoint_dataset import list_clip_records
from src.mslm.utils.sequence_metrics import gloss_sequence_diagnostics


DEFAULT_OUTPUT_ROOT = ROOT.parent / "outputs/video_token_decoder"
DEFAULT_JSON = ROOT / "experiments/video_token_decoder/paper_master_results.json"
DEFAULT_MARKDOWN = ROOT / "experiments/video_token_decoder/PAPER_TABLES.md"
DEFAULT_LATEX_MACROS = ROOT / "experiments/video_token_decoder/paper/results_macros.tex"


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def compact_metrics(payload: dict | None, *keys: str) -> dict | None:
    if payload is None:
        return None
    current = payload
    for key in keys:
        current = current[key]
    return {
        name: current.get(name)
        for name in ("strict_exact", "token_edit_similarity", "length_mae")
    }


def e1_summary(payload: dict | None, tokens: dict, fold: int) -> dict | None:
    if payload is None:
        return None
    rows = payload["predictions"]
    diagnostics = gloss_sequence_diagnostics(
        [row["pred"] for row in rows], [row["target"] for row in rows], tokens
    )
    cis = {
        metric: bootstrap_mean_ci(
            [row[metric] for row in rows], seed=BOOTSTRAP_SEED + fold * 10 + index
        )
        for index, metric in enumerate(("strict_exact", "token_edit_similarity"))
    }
    return {
        "metrics": payload["metrics"],
        "bootstrap_95_ci": cis,
        "gloss_diagnostics": diagnostics,
        "checkpoint": payload.get("checkpoint"),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
    }


def ablation_rows(output_root: Path) -> list[dict]:
    variants = {
        "E1": "e1_pe_vocab121",
        "E2_CTC": "e2_jta_ctc03",
        "E3_unfreeze": "e3_unfreeze_stgcn",
    }
    rows = []
    for fold in (4, 5, 6):
        for variant, directory in variants.items():
            payload = read_json(
                output_root / f"fold{fold}_seed23/{directory}/outer_test.json"
            )
            rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "status": "available" if payload else "not_run_or_artifact_missing",
                    "metrics": payload.get("metrics") if payload else None,
                    "paired_vs_affine_cif": (
                        payload.get("paired_vs_affine_cif") if payload else None
                    ),
                }
            )
    return rows


def build_results(output_root: Path) -> dict:
    records = list_clip_records(protocol.DEFAULT_H5, "dataset1")
    tokens = protocol.label_tokens(records, protocol.load_tokenizer(protocol.DEFAULT_TOKENIZER))
    folds = []
    for fold in range(1, 9):
        comparator_path = output_root / f"fold{fold}_cif_comparator.json"
        baseline_path = output_root / f"fold{fold}_seed23/base/outer_test.json"
        e1_path = output_root / f"fold{fold}_seed23/e1_pe_vocab121/outer_test.json"
        robustness_path = output_root / (
            f"fold{fold}_seed23/e1_pe_vocab121/outer_test_robustness.json"
        )
        probe_path = output_root / f"fold{fold}_seed23/e1_pe_vocab121/frame_identity_probe.json"
        comparator = read_json(comparator_path)
        baseline = read_json(baseline_path)
        e1 = read_json(e1_path)
        robustness = read_json(robustness_path)
        paired = paired_cif_result(e1["predictions"], comparator, fold) if e1 and comparator else None
        folds.append(
            {
                "fold": fold,
                "role": "development" if fold <= 6 else "confirmatory",
                "cif_affine": compact_metrics(comparator, "outer_test", "sequence"),
                "ar_baseline": compact_metrics(baseline, "metrics"),
                "e1": e1_summary(e1, tokens, fold),
                "paired_e1_vs_cif": paired,
                "pre_registered_decisions": (
                    robustness.get("pre_registered_decisions_for_this_fold")
                    if robustness
                    else None
                ),
                "robustness_curves": (
                    {
                        condition: {
                            "metrics": values["metrics"],
                            "paired_clean_minus_condition_bootstrap_95_ci": values.get(
                                "paired_clean_minus_condition_bootstrap_95_ci"
                            ),
                        }
                        for condition, values in robustness["conditions"].items()
                    }
                    if robustness
                    else None
                ),
                "frame_identity_probe": read_json(probe_path),
                "artifacts": {
                    "cif": str(comparator_path) if comparator else None,
                    "ar_baseline": str(baseline_path) if baseline else None,
                    "e1": str(e1_path) if e1 else None,
                    "robustness": str(robustness_path) if robustness else None,
                    "probe": str(probe_path) if probe_path.exists() else None,
                },
            }
        )

    confirmatory = [row for row in folds if row["role"] == "confirmatory"]
    hypothesis_status = {}
    for hypothesis in ("H1", "H2", "H3"):
        observed = [
            row["pre_registered_decisions"][hypothesis]["pass"]
            for row in confirmatory
            if row["pre_registered_decisions"] is not None
        ]
        hypothesis_status[hypothesis] = {
            "fold_passes": observed,
            "pass": all(observed) if len(observed) == 2 else None,
        }
    return {
        "schema_version": 1,
        "protocol": "Imitator E1, seed 23",
        "preregistration_commit": "67641eba6a28daee2b19e68d466a46f79eff7334",
        "folds": folds,
        "confirmatory_hypotheses": hypothesis_status,
        "full_confirmation": (
            all(status["pass"] for status in hypothesis_status.values())
            if all(status["pass"] is not None for status in hypothesis_status.values())
            else None
        ),
        "development_ablations": ablation_rows(output_root),
        "limitations": [
            "one training seed; between-seed variance is not estimated",
            "two confirmatory outer signers",
            "synthetic concatenations of isolated signs, not real continuous signing",
            "missing cells are reported as not run/artifact missing and are never imputed",
        ],
    }


def percent(value) -> str:
    return "—" if value is None else f"{100 * value:.1f}"


def render_markdown(results: dict) -> str:
    lines = [
        "# Imitator E1 — tablas maestras",
        "",
        "Generado solo desde artefactos versionados/hasheados. `—` significa que la corrida o el artefacto no existe; no se imputa.",
        "",
        "## Comparación principal",
        "",
        "| Fold | Rol | CIF exact % | AR-base exact % | E1 exact % | E1 edit | Orden | W/L/T vs CIF | p bilateral |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["folds"]:
        cif = row["cif_affine"] or {}
        baseline = row["ar_baseline"] or {}
        e1 = row["e1"] or {}
        metrics = e1.get("metrics", {})
        order = e1.get("gloss_diagnostics", {}).get("pairwise_order_accuracy")
        paired = row["paired_e1_vs_cif"] or {}
        wlt = (
            f"{paired['wins']}/{paired['losses']}/{paired['ties']}" if paired else "—"
        )
        p = f"{paired['two_sided_sign_test_p']:.3g}" if paired else "—"
        lines.append(
            f"| {row['fold']} | {row['role']} | {percent(cif.get('strict_exact'))} | "
            f"{percent(baseline.get('strict_exact'))} | {percent(metrics.get('strict_exact'))} | "
            f"{percent(metrics.get('token_edit_similarity'))} | {percent(order)} | {wlt} | {p} |"
        )

    lines += ["", "## Decisión confirmatoria", ""]
    for hypothesis, status in results["confirmatory_hypotheses"].items():
        value = "pendiente" if status["pass"] is None else ("PASA" if status["pass"] else "FALLA")
        lines.append(f"- {hypothesis}: {value} — folds observados: {status['fold_passes']}")
    overall = results["full_confirmation"]
    overall_text = "pendiente" if overall is None else ("PASA" if overall else "FALLA")
    lines += ["", f"**Confirmación completa:** {overall_text}", "", "## Ablations de desarrollo", ""]
    lines += [
        "| Fold | Variante | Estado | strict_exact % | edit_sim % |",
        "|---:|---|---|---:|---:|",
    ]
    for row in results["development_ablations"]:
        metrics = row["metrics"] or {}
        lines.append(
            f"| {row['fold']} | {row['variant']} | {row['status']} | "
            f"{percent(metrics.get('strict_exact'))} | {percent(metrics.get('token_edit_similarity'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_latex_macros(results: dict) -> str:
    """Render confirmatory values without permitting manual result transcription."""
    by_fold = {row["fold"]: row for row in results["folds"]}

    def value_or_pending(value, formatter):
        return r"\pending" if value is None else formatter(value)

    def fold_commands(fold: int, word: str) -> list[str]:
        row = by_fold.get(fold, {})
        cif = row.get("cif_affine") or {}
        e1 = row.get("e1") or {}
        metrics = e1.get("metrics", {})
        diagnostics = e1.get("gloss_diagnostics", {})
        paired = row.get("paired_e1_vs_cif") or {}
        curves = row.get("robustness_curves") or {}
        permutation = curves.get("segment_permutation") or {}
        delta_block = permutation.get("paired_clean_minus_condition_bootstrap_95_ci") or {}
        edit_delta = (delta_block.get("token_edit_similarity") or {}).get("estimate")
        wlt = None
        if paired:
            wlt = f"{paired['wins']}/{paired['losses']}/{paired['ties']}"
        return [
            rf"\newcommand{{\Fold{word}CIFExact}}{{{value_or_pending(cif.get('strict_exact'), lambda x: f'{100*x:.1f}')}}}",
            rf"\newcommand{{\Fold{word}Exact}}{{{value_or_pending(metrics.get('strict_exact'), lambda x: f'{100*x:.1f}')}}}",
            rf"\newcommand{{\Fold{word}Edit}}{{{value_or_pending(metrics.get('token_edit_similarity'), lambda x: f'{100*x:.1f}')}}}",
            rf"\newcommand{{\Fold{word}Order}}{{{value_or_pending(diagnostics.get('pairwise_order_accuracy'), lambda x: f'{100*x:.1f}')}}}",
            rf"\newcommand{{\Fold{word}PermutationDrop}}{{{value_or_pending(edit_delta, lambda x: f'{x:.3f}')}}}",
            rf"\newcommand{{\Fold{word}WLT}}{{{value_or_pending(wlt, str)}}}",
            rf"\newcommand{{\Fold{word}PValue}}{{{value_or_pending(paired.get('two_sided_sign_test_p'), lambda x: f'{x:.3g}')}}}",
        ]

    commands = [
        "% Auto-generated from paper_master_results.json; do not edit result values manually.",
        *fold_commands(7, "Seven"),
        *fold_commands(8, "Eight"),
    ]
    for hypothesis, macro in (("H1", "HOneOutcome"), ("H2", "HTwoOutcome"), ("H3", "HThreeOutcome")):
        status = results["confirmatory_hypotheses"][hypothesis]["pass"]
        text = r"\pending" if status is None else ("passes" if status else "fails")
        commands.append(rf"\newcommand{{\{macro}}}{{{text}}}")
    overall = results["full_confirmation"]
    overall_text = r"\pending" if overall is None else ("passes" if overall else "fails")
    commands.append(rf"\newcommand{{\OverallOutcome}}{{{overall_text}}}")
    commands.append("")
    return "\n".join(commands)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--latex-macros", type=Path, default=DEFAULT_LATEX_MACROS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results = build_results(args.output_root)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.latex_macros.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown.write_text(render_markdown(results), encoding="utf-8")
    args.latex_macros.write_text(render_latex_macros(results), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(args.json),
                "markdown": str(args.markdown),
                "latex_macros": str(args.latex_macros),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
