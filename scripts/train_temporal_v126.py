"""Train v126 synthetic temporal pretraining model."""
import argparse
import json
import os
import random
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", "/shared/Code/Sign-AI/.hf_cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from settings import initialize
from src.mslm.dataloader.isolated_keypoint_dataset import list_clip_records
from src.mslm.dataloader.synthetic_temporal import (
    SyntheticTemporalSignDataset,
    synthetic_temporal_collate,
)
from src.mslm.models.temporal_sign_prompt import (
    CIFAggregator,
    STGCNTemporalFrameEncoder,
    TemporalSignPromptModel,
    load_visual_low_level_weights,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--max-clips", type=int, default=8)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path("/shared/Code/Sign-AI/data/processed/dataset1_isolated_v122.hdf5"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(
            "/shared/Code/Sign-AI/.hf_cache/hub/models--unsloth--gemma-3n-E2B-it-unsloth-bnb-4bit/"
            "snapshots/3d26ffdd2276698f582562bf01511aa625bc6f30"
        ),
    )
    parser.add_argument(
        "--embedding-table",
        type=Path,
        default=Path("/shared/Code/Sign-AI/data/processed/gemma3n_embed_table.pt"),
    )
    parser.add_argument(
        "--checkpoint-v121",
        type=Path,
        default=Path("../outputs/checkpoints/121/23/best_top1/checkpoint.pth"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../outputs/v126_temporal"),
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).strip().lower()
    return " ".join(text.split())


def stratified_clip_split(records, seed: int, n_val_per_class: int = 10):
    by_label = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)
    rng = random.Random(seed)
    train, val = [], []
    for rows in by_label.values():
        rows = rows[:]
        rng.shuffle(rows)
        val.extend(rows[:n_val_per_class])
        train.extend(rows[n_val_per_class:])
    return train, val


def make_label_tokens(records, tokenizer):
    labels = sorted({record["label"] for record in records})
    return {
        label: tokenizer(normalize_text(label), add_special_tokens=False).input_ids
        for label in labels
    }


def load_embedding_rows(table_path: Path, token_ids: set[int], embedding_dim: int) -> torch.Tensor:
    table = torch.load(table_path, map_location="cpu")
    if isinstance(table, dict):
        for key in ("weight", "embeddings", "embedding_table"):
            if key in table:
                table = table[key]
                break
    rows = torch.zeros(max(token_ids) + 1, embedding_dim, dtype=table.dtype)
    ids = torch.tensor(sorted(token_ids), dtype=torch.long)
    rows[ids] = table[ids]
    return rows


def gather_logits(logits, targets):
    if logits.size(1) < targets.size(1):
        pad = logits.new_zeros(logits.size(0), targets.size(1) - logits.size(1), logits.size(2))
        logits = torch.cat([logits, pad], dim=1)
    elif logits.size(1) > targets.size(1):
        logits = logits[:, : targets.size(1)]
    mask = targets.ne(-100)
    return logits[mask], targets[mask]


def evaluate(model, loader, device):
    model.eval()
    totals = defaultdict(float)
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            alpha_target = CIFAggregator.boundary_targets(
                batch["boundaries"],
                batch["keypoints"].size(1),
                token_spans=batch["token_spans"],
            )
            out = model(
                batch["keypoints"],
                batch["frame_lengths"],
                alphas=alpha_target,
                target_lengths=batch["token_lengths"],
            )
            token_logits, token_targets = gather_logits(out["token_logits"], batch["token_ids"])
            ce = F.cross_entropy(token_logits, token_targets)
            pred = token_logits.argmax(dim=-1)
            top1 = pred.eq(token_targets).float().mean()
            top5 = token_logits.topk(5, dim=-1).indices.eq(token_targets.unsqueeze(1)).any(dim=1).float().mean()
            exact = sequence_accuracy(out["token_logits"], batch["token_ids"], batch["token_lengths"])
            mae = (out["cif"].quantity - batch["token_lengths"].float()).abs().mean()
            totals["loss"] += ce.item()
            totals["top1"] += top1.item()
            totals["top5"] += top5.item()
            totals["exact"] += exact
            totals["mae_len"] += mae.item()
            batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def sequence_accuracy(logits, targets, lengths):
    if logits.size(1) < targets.size(1):
        pad = logits.new_zeros(logits.size(0), targets.size(1) - logits.size(1), logits.size(2))
        logits = torch.cat([logits, pad], dim=1)
    elif logits.size(1) > targets.size(1):
        logits = logits[:, : targets.size(1)]
    pred = logits.argmax(dim=-1)
    hits = 0
    for i, length in enumerate(lengths.tolist()):
        hits += bool(torch.equal(pred[i, :length].cpu(), targets[i, :length].cpu()))
    return hits / max(1, targets.size(0))


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def main():
    args = parse_args()
    initialize(seed=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    metrics_path = out_dir / "metrics.jsonl"
    ckpt_path = out_dir / "checkpoint_latest.pt"
    best_path = out_dir / "checkpoint_best.pt"

    records = list_clip_records(args.h5, "dataset1")
    train_records, val_records = stratified_clip_split(records, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    token_ids_by_label = make_label_tokens(records, tokenizer)
    all_token_ids = {tid for ids in token_ids_by_label.values() for tid in ids}
    embedding_rows = load_embedding_rows(args.embedding_table, all_token_ids, args.embedding_dim)

    def make_ds(rows, samples, seed):
        return SyntheticTemporalSignDataset(
            args.h5,
            [r["clip_id"] for r in rows],
            {r["clip_id"]: r["label"] for r in rows},
            token_ids_by_label,
            min_clips=2,
            max_clips=args.max_clips,
            min_neutral_frames=0,
            max_neutral_frames=8,
            samples_per_epoch=samples,
            seed=seed,
            embedding_table=embedding_rows,
            apply_remove_keypoints=True,
            normalize=True,
            n_keypoints=111,
        )

    train_loader = DataLoader(
        make_ds(train_records, args.samples_per_epoch, args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=synthetic_temporal_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        make_ds(val_records, args.val_samples, args.seed + 100_000),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=synthetic_temporal_collate,
        pin_memory=torch.cuda.is_available(),
    )

    A = np.load("/shared/Code/Sign-AI/data/processed/adjacency_matrix.npy", allow_pickle=True)
    encoder = STGCNTemporalFrameEncoder(A, hidden_size=args.hidden_size)
    load_info = load_visual_low_level_weights(encoder, args.checkpoint_v121)
    model = TemporalSignPromptModel(
        encoder,
        hidden_size=args.hidden_size,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    best_top1 = -1.0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_top1 = state.get("best_top1", best_top1)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "load_info": load_info, "device": device}, f, indent=2, default=str)
    print(f"[v126] out={out_dir} device={device} load={load_info}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals = defaultdict(float)
        steps = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            alpha_target = CIFAggregator.boundary_targets(
                batch["boundaries"],
                batch["keypoints"].size(1),
                token_spans=batch["token_spans"],
            )
            out = model(
                batch["keypoints"],
                batch["frame_lengths"],
                alphas=alpha_target,
                target_lengths=batch["token_lengths"],
            )
            token_logits, token_targets = gather_logits(out["token_logits"], batch["token_ids"])
            token_loss = F.cross_entropy(token_logits, token_targets)
            emb_mask = batch["token_ids"].ne(-100)
            emb_loss = F.smooth_l1_loss(out["embeddings"][emb_mask], batch["target_embeddings"][emb_mask])
            qty_loss = F.l1_loss(out["cif"].quantity, batch["token_lengths"].float())
            loss = token_loss + 0.05 * emb_loss + 0.1 * qty_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += loss.item()
            totals["token_loss"] += token_loss.item()
            totals["emb_loss"] += emb_loss.item()
            totals["qty_loss"] += qty_loss.item()
            steps += 1

        train_metrics = {key: value / max(1, steps) for key, value in totals.items()}
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"ep{epoch:03d} loss={train_metrics['loss']:.4f} "
            f"val_top1={val_metrics['top1']:.3f} val_top5={val_metrics['top5']:.3f} "
            f"exact={val_metrics['exact']:.3f} mae_len={val_metrics['mae_len']:.3f}",
            flush=True,
        )
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_top1": best_top1,
            "token_ids_by_label": token_ids_by_label,
            "load_info": load_info,
        }
        torch.save(state, ckpt_path)
        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            state["best_top1"] = best_top1
            torch.save(state, best_path)

    print(f"[v126] done best_top1={best_top1:.3f} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
