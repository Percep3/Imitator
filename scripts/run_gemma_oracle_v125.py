"""Oracle Gemma v125: token IDs vs embeddings exactos vs espacio seleccionado."""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

# Debe definirse antes de importar Unsloth/Gemma. Evita una incompatibilidad
# conocida del entorno torch 2.8 + triton 3.6 al intentar compilar RMSNorm.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.mslm.utils  # noqa: E402,F401
from src.mslm.models.gemma_bridge import GemmaBridge  # noqa: E402
from src.mslm.utils.embedding_space import (  # noqa: E402
    EmbeddingTransform,
    StandardizedEmbeddingTransform,
)
from src.mslm.utils.text_metrics import (  # noqa: E402
    bleu_score,
    chrf_score,
    content_word_f1,
    rouge_l_f1,
)

PROMPT_PRE = "Traduce la siguiente representación de lengua de señas a español natural.\n"
PROMPT_POST = "\nRespuesta:"
MODEL_ID = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"


@dataclass
class OracleSample:
    clip_id: str
    label: str
    token_ids: np.ndarray
    embeddings: np.ndarray


def load_transform(
    path: Path,
) -> tuple[str, EmbeddingTransform | StandardizedEmbeddingTransform]:
    blob = torch.load(path, map_location="cpu")
    mode = blob.get("mode", "pca_whitening")
    if mode == "pca_whitening":
        return mode, EmbeddingTransform(
            mean=blob["mean"].numpy(),
            components=blob["components"].numpy(),
            singular_values=blob["singular_values"].numpy(),
            n_samples=blob["n_samples"],
            epsilon=blob["epsilon"],
        )
    if mode == "full_standardized":
        return mode, StandardizedEmbeddingTransform(
            mean=blob["mean"].numpy(),
            scale=blob["scale"].numpy(),
            epsilon=blob["epsilon"],
        )
    raise ValueError(f"modo de transformación desconocido: {mode}")


def load_samples(
    h5_path: Path, split_path: Path, n_samples: int, seed: int
) -> list[OracleSample]:
    split = json.loads(split_path.read_text())
    rng = random.Random(seed)
    selected = rng.sample(split["val"], min(n_samples, len(split["val"])))
    samples = []
    with h5py.File(h5_path, "r") as file:
        dataset = file["dataset2"]
        for clip_id in selected:
            samples.append(
                OracleSample(
                    clip_id=clip_id,
                    label=dataset["labels"][clip_id][0].decode(),
                    token_ids=dataset["token_ids"][clip_id][:].astype(np.int64),
                    embeddings=dataset["embeddings"][clip_id][:].astype(np.float32),
                )
            )
    return samples


def prompt_token_ids(bridge: GemmaBridge, sign_ids: torch.Tensor) -> torch.Tensor:
    pre = bridge.tokenizer(
        PROMPT_PRE, return_tensors="pt", add_special_tokens=True
    ).input_ids.to(sign_ids.device)
    post = bridge.tokenizer(
        PROMPT_POST, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(sign_ids.device)
    return torch.cat([pre, sign_ids.unsqueeze(0), post], dim=1)


def prompt_embeddings(
    bridge: GemmaBridge, sign_embeddings: torch.Tensor
) -> torch.Tensor:
    device = sign_embeddings.device
    pre_ids = bridge.tokenizer(
        PROMPT_PRE, return_tensors="pt", add_special_tokens=True
    ).input_ids.to(device)
    post_ids = bridge.tokenizer(
        PROMPT_POST, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    return torch.cat(
        [
            bridge.embed_tokens(pre_ids),
            sign_embeddings.unsqueeze(0),
            bridge.embed_tokens(post_ids),
        ],
        dim=1,
    )


def prompt_per_layer_inputs(bridge: GemmaBridge, sign_ids: torch.Tensor) -> torch.Tensor:
    return bridge.get_per_layer_inputs(prompt_token_ids(bridge, sign_ids))


def run_sample(
    bridge: GemmaBridge,
    sample: OracleSample,
    transform: EmbeddingTransform | StandardizedEmbeddingTransform,
    max_new_tokens: int,
    reuse_exact_for_transformed: bool = False,
) -> dict:
    device = next(bridge.lm.parameters()).device
    sign_ids = torch.from_numpy(sample.token_ids).to(device=device, dtype=torch.long)
    stored = torch.from_numpy(sample.embeddings).to(
        device=device, dtype=bridge.embed_tokens(sign_ids[:1].unsqueeze(0)).dtype
    )

    live = bridge.embed_tokens(sign_ids.unsqueeze(0))[0]
    table_max_abs_error = float((stored - live).abs().max().item())
    if table_max_abs_error >= 1e-2:
        raise ValueError(
            f"clip {sample.clip_id}: embeddings H5 no corresponden al modelo "
            f"(max_abs_error={table_max_abs_error:.6f})"
        )

    transformed = transform.transform(sample.embeddings)
    reconstructed = transform.inverse_transform(transformed)
    reconstructed_tensor = torch.from_numpy(reconstructed).to(
        device=device, dtype=stored.dtype
    )

    output_ids = bridge.generate_from_ids(
        prompt_token_ids(bridge, sign_ids), max_new_tokens=max_new_tokens
    )
    per_layer_inputs = prompt_per_layer_inputs(bridge, sign_ids)
    output_exact = bridge.generate(
        prompt_embeddings(bridge, stored),
        max_new_tokens=max_new_tokens,
        per_layer_inputs=per_layer_inputs,
    )
    if reuse_exact_for_transformed:
        output_transformed = output_exact
    else:
        output_transformed = bridge.generate(
            prompt_embeddings(bridge, reconstructed_tensor),
            max_new_tokens=max_new_tokens,
            per_layer_inputs=per_layer_inputs,
        )
    return {
        "clip_id": sample.clip_id,
        "label": sample.label,
        "token_ids": bridge.decode(output_ids)[0].strip(),
        "exact": bridge.decode(output_exact)[0].strip(),
        "transformed": bridge.decode(output_transformed)[0].strip(),
        "token_count": int(len(sample.token_ids)),
        "table_max_abs_error": table_max_abs_error,
    }


def summarize(rows: list[dict], space_mode: str) -> dict:
    def average(metric, key):
        return sum(metric(row[key], row["label"]) for row in rows) / max(len(rows), 1)

    summary = {
        "n_samples": len(rows),
        "space_mode": space_mode,
        "bridge_mode": "inputs_embeds_with_gemma3n_per_layer_inputs",
    }
    for key in ("token_ids", "exact", "transformed"):
        summary[key] = {
            "chrf": average(chrf_score, key),
            "bleu": average(bleu_score, key),
            "rouge_l": average(rouge_l_f1, key),
            "content_word_f1": average(content_word_f1, key),
        }
    summary["exact_chrf_drop"] = (
        summary["token_ids"]["chrf"] - summary["exact"]["chrf"]
    )
    summary["transformed_extra_chrf_drop"] = (
        summary["exact"]["chrf"] - summary["transformed"]["chrf"]
    )
    summary["gate_oracle_chrf_ge_70"] = summary["token_ids"]["chrf"] >= 70.0
    summary["gate_exact_drop_le_2"] = summary["exact_chrf_drop"] <= 2.0
    summary["gate_transformed_extra_drop_le_2"] = (
        summary["transformed_extra_chrf_drop"] <= 2.0
    )
    summary["all_gates_pass"] = all(
        summary[key]
        for key in (
            "gate_oracle_chrf_ge_70",
            "gate_exact_drop_le_2",
            "gate_transformed_extra_drop_le_2",
        )
    )
    return summary


def main(
    h5_path: Path,
    split_path: Path,
    transform_path: Path,
    out_path: Path,
    model_id: str,
    n_samples: int,
    seed: int,
    max_new_tokens: int,
) -> None:
    samples = load_samples(h5_path, split_path, n_samples, seed)
    if not samples:
        raise ValueError("el split de validación no contiene muestras")

    space_mode, transform = load_transform(transform_path)
    bridge = GemmaBridge(model_id)

    special_ids = {
        bridge.tokenizer.bos_token_id,
        bridge.tokenizer.eos_token_id,
        bridge.tokenizer.pad_token_id,
    }
    for sample in samples:
        found = special_ids.intersection(int(token) for token in sample.token_ids)
        if found:
            raise ValueError(
                f"clip {sample.clip_id}: special tokens visuales no permitidos: {found}"
            )

    rows = []
    for index, sample in enumerate(samples, start=1):
        rows.append(
            run_sample(
                bridge,
                sample,
                transform,
                max_new_tokens,
                reuse_exact_for_transformed=space_mode == "full_standardized",
            )
        )
        print(f"[oracle] {index}/{len(samples)} clip={sample.clip_id}")

    summary = summarize(rows, space_mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5", type=Path, default=Path("data/processed/dataset_v6_unsloth.hdf5")
    )
    parser.add_argument(
        "--split", type=Path, default=Path("data/processed/dataset2_split_v125.json")
    )
    parser.add_argument(
        "--transform",
        type=Path,
        default=Path("data/processed/v125_anisotropy_gate.transform.pt"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/v125_gemma_oracle.json")
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1].parent
    main(
        args.h5 if args.h5.is_absolute() else root / args.h5,
        args.split if args.split.is_absolute() else root / args.split,
        args.transform if args.transform.is_absolute() else root / args.transform,
        args.out if args.out.is_absolute() else root / args.out,
        args.model,
        args.n_samples,
        args.seed,
        args.max_new_tokens,
    )
