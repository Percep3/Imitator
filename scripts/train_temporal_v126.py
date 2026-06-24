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
    permute_video_segments,
    synthetic_temporal_collate,
)
from src.mslm.models.temporal_sign_prompt import (
    CIFAggregator,
    STGCNTemporalFrameEncoder,
    TemporalSignPromptModel,
    alpha_diagnostics,
    alpha_schedule_weights,
    boundary_error_mae,
    length_mask_from_lengths,
    load_visual_low_level_weights,
    rescale_alphas_to_target_lengths,
    set_cif_diagnostic_freeze,
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
        "--phase",
        choices=["teacher_forced", "learned_cif"],
        default="teacher_forced",
        help="teacher_forced: v126 baseline (alpha=alpha_target always). "
        "learned_cif: v126b, freeze/teacher-forcing curriculum to train CIF.alpha.",
    )
    parser.add_argument("--stgcn-lr-scale", type=float, default=0.1)
    parser.add_argument(
        "--diag-alpha-loss",
        choices=["current", "qty_only", "kl_quantity", "logit_l1"],
        default="current",
        help="Diagnostic CIF-alpha objective. Default preserves the current v126b loss.",
    )
    parser.add_argument(
        "--diag-freeze",
        choices=["alpha_only", "target_only_stage1", "heads_only", "tcn_heads", "full_current"],
        default="full_current",
        help="Diagnostic freezing regime. Default preserves the current v126b schedule.",
    )
    parser.add_argument(
        "--diag-eval-force-count",
        action="store_true",
        help="Use target-length-rescaled predicted CIF for val_predicted/select_metric.",
    )
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
    logits = align_time_to_targets(logits, targets.size(1))
    mask = targets.ne(-100)
    return logits[mask], targets[mask]


def align_time_to_targets(values, target_steps: int):
    if values.size(1) < target_steps:
        pad_shape = (values.size(0), target_steps - values.size(1), *values.shape[2:])
        pad = values.new_zeros(pad_shape)
        values = torch.cat([values, pad], dim=1)
    elif values.size(1) > target_steps:
        values = values[:, :target_steps]
    return values


def gather_embeddings(embeddings, targets, target_embeddings):
    embeddings = align_time_to_targets(embeddings, targets.size(1))
    mask = targets.ne(-100)
    return embeddings[mask], target_embeddings[mask]


def forward_from_cif(model, frame_features, frame_lengths, **cif_kwargs):
    cif_out = model.cif(frame_features, frame_lengths, **cif_kwargs)
    return {
        "cif": cif_out,
        "token_logits": model.token_head(cif_out.embeddings),
        "embeddings": model.embedding_head(cif_out.embeddings),
    }


def compute_alpha_loss(mode, alpha_logits, alpha_pred, alpha_target, frame_mask):
    if mode == "qty_only":
        return alpha_pred.new_zeros(())

    target = alpha_target[frame_mask]
    if mode == "current":
        frame_loss = F.smooth_l1_loss(
            alpha_pred[frame_mask],
            target,
            reduction="none",
        )
        positive_weight = 1.0 + 20.0 * (target > 0).float()
        return (frame_loss * positive_weight).mean()

    if mode == "kl_quantity":
        eps = 1e-8
        pred_dist = alpha_pred / alpha_pred.sum(dim=1, keepdim=True).clamp(min=eps)
        target_dist = alpha_target / alpha_target.sum(dim=1, keepdim=True).clamp(min=eps)
        return F.kl_div(
            (pred_dist[frame_mask] + eps).log(),
            target_dist[frame_mask],
            reduction="batchmean",
        )

    if mode == "logit_l1":
        eps = 1e-4
        target_logits = torch.logit(alpha_target.clamp(min=eps, max=1.0 - eps))
        frame_loss = F.smooth_l1_loss(
            alpha_logits[frame_mask],
            target_logits[frame_mask],
            reduction="none",
        )
        positive_weight = 1.0 + 20.0 * (alpha_target[frame_mask] > 0).float()
        return (frame_loss * positive_weight).mean()

    raise ValueError(f"unknown diagnostic alpha loss: {mode}")


def _module_grad_norm(module):
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().float().norm(2).item()
        total += grad_norm * grad_norm
    return total ** 0.5


def compute_grad_norms(model):
    encoder = model.frame_encoder
    groups = {
        "stgcn": encoder.stgcn_layers,
        "linear_hidden": encoder.linear_hidden,
        "tcn": encoder.tcn,
        "transformer": encoder.transformer,
        "cif": model.cif,
        "token_head": model.token_head,
        "embedding_head": model.embedding_head,
        "all": model,
    }
    return {name: _module_grad_norm(module) for name, module in groups.items()}


def _classification_metrics(out, batch):
    token_logits, token_targets = gather_logits(out["token_logits"], batch["token_ids"])
    ce = F.cross_entropy(token_logits, token_targets)
    pred = token_logits.argmax(dim=-1)
    top1 = pred.eq(token_targets).float().mean()
    top5 = token_logits.topk(5, dim=-1).indices.eq(token_targets.unsqueeze(1)).any(dim=1).float().mean()
    exact = sequence_accuracy(out["token_logits"], batch["token_ids"], batch["token_lengths"])
    mae = (out["cif"].quantity - batch["token_lengths"].float()).abs().mean()
    count_mae = (out["cif"].counts.float() - batch["token_lengths"].float()).abs().mean()
    return {
        "loss": ce.item(),
        "top1": top1.item(),
        "top5": top5.item(),
        "exact": exact,
        "mae_len": mae.item(),
        "count_mae": count_mae.item(),
        "pred_count_mean": out["cif"].counts.float().mean().item(),
        "quantity_mean": out["cif"].quantity.float().mean().item(),
        "target_len_mean": batch["token_lengths"].float().mean().item(),
    }


def evaluate_alpha_mode(model, loader, device, mode, w_target=1.0, w_pred=0.0):
    """Evaluate a concrete CIF alpha source."""
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
            frame_features = model.frame_encoder(batch["keypoints"], batch["frame_lengths"])
            alpha_logits = model.cif.predict_alpha_logits(frame_features, batch["frame_lengths"])
            alpha_pred = torch.sigmoid(alpha_logits).masked_fill(
                ~length_mask_from_lengths(batch["frame_lengths"], alpha_logits.size(1)),
                0.0,
            )
            alpha_rescaled = rescale_alphas_to_target_lengths(alpha_pred, batch["token_lengths"])
            if mode == "teacher_alpha":
                alphas = alpha_target
            elif mode == "pred_raw":
                alphas = alpha_pred
            elif mode == "pred_rescaled_to_target_len":
                alphas = alpha_rescaled
            elif mode == "blended_alpha":
                alphas = w_target * alpha_target + w_pred * alpha_rescaled
            else:
                raise ValueError(f"unknown eval alpha mode: {mode}")

            out = forward_from_cif(model, frame_features, batch["frame_lengths"], alphas=alphas)
            for key, value in _classification_metrics(out, batch).items():
                totals[key] += value
            totals["alpha_eval_sum_mean"] += alphas.sum(dim=1).mean().item()
            totals["target_alpha_sum_mean"] += alpha_target.sum(dim=1).mean().item()
            if mode != "teacher_alpha":
                for key, value in alpha_diagnostics(
                    frame_features,
                    alpha_logits,
                    alpha_pred,
                    batch["frame_lengths"],
                ).items():
                    totals[key] += value
            centers = CIFAggregator.token_centers(batch["boundaries"], batch["token_spans"]).to(device)
            totals["boundary_mae"] += boundary_error_mae(
                out["cif"].fire_positions, out["cif"].counts, centers, batch["token_lengths"]
            )
            batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def evaluate_teacher(model, loader, device):
    """Forward with alpha=alpha_target (perfect injected boundaries). Comparable ceiling."""
    return evaluate_alpha_mode(model, loader, device, "teacher_alpha")


def evaluate_predicted(model, loader, device):
    """Forward using only CIF.alpha (no injected boundaries). Official v126b gate."""
    return evaluate_alpha_mode(model, loader, device, "pred_raw")


def evaluate_permuted(model, loader, device, seed: int):
    """Predicted-mode top1 after shuffling whole gloss segments; targets stay unpermuted."""
    model.eval()
    totals = defaultdict(float)
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            permuted = batch["keypoints"].clone()
            for i in range(permuted.size(0)):
                length = int(batch["frame_lengths"][i].item())
                rng = random.Random(seed + i)
                permuted[i, :length] = permute_video_segments(
                    batch["keypoints"][i, :length].cpu(),
                    batch["boundaries"][i].cpu(),
                    length,
                    rng,
                ).to(device)
            out = model(permuted, batch["frame_lengths"])
            token_logits, token_targets = gather_logits(out["token_logits"], batch["token_ids"])
            top1 = token_logits.argmax(dim=-1).eq(token_targets).float().mean()
            totals["top1"] += top1.item()
            batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def sequence_accuracy(logits, targets, lengths):
    logits = align_time_to_targets(logits, targets.size(1))
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

    diagnostic_mode = (
        args.diag_alpha_loss != "current"
        or args.diag_freeze != "full_current"
        or args.diag_eval_force_count
    )
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    if diagnostic_mode and not run_name.startswith("diag_"):
        run_name = f"diag_{run_name}"
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
    if args.phase == "learned_cif":
        stgcn_params = list(encoder.stgcn_layers.parameters()) + list(encoder.linear_hidden.parameters())
        stgcn_ids = {id(p) for p in stgcn_params}
        other_params = [p for p in model.parameters() if id(p) not in stgcn_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": stgcn_params, "lr": args.lr * args.stgcn_lr_scale},
                {"params": other_params, "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    best_top1 = -1.0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        same_stage = state.get("phase", "teacher_forced") == args.phase
        if same_stage:
            try:
                optimizer.load_state_dict(state["optimizer"])
            except ValueError:
                same_stage = False
        if same_stage:
            start_epoch = state["epoch"] + 1
            best_top1 = state.get("best_top1", best_top1)
        else:
            print(
                f"[v126] stage transition: checkpoint phase={state.get('phase', 'teacher_forced')!r} "
                f"-> {args.phase!r}; loaded model weights only, starting fresh epoch/optimizer",
                flush=True,
            )

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "load_info": load_info, "device": device}, f, indent=2, default=str)
    print(f"[v126] out={out_dir} device={device} load={load_info}", flush=True)

    final_gates = None
    for epoch in range(start_epoch, args.epochs):
        if args.phase == "learned_cif":
            phase_state = set_cif_diagnostic_freeze(model, epoch, args.diag_freeze)
            w_target, w_pred = alpha_schedule_weights(epoch)
            if args.diag_freeze == "target_only_stage1":
                w_target, w_pred = 1.0, 0.0
        else:
            phase_state, w_target, w_pred = None, 1.0, 0.0

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

            if args.phase == "learned_cif":
                frame_features = model.frame_encoder(batch["keypoints"], batch["frame_lengths"])
                frame_mask = length_mask_from_lengths(batch["frame_lengths"], frame_features.size(1))
                alpha_logits = model.cif.predict_alpha_logits(frame_features, batch["frame_lengths"])
                alpha_pred = torch.sigmoid(alpha_logits).masked_fill(~frame_mask, 0.0)
                pred_quantity = alpha_pred.sum(dim=1)
                pred_scale = (
                    batch["token_lengths"].float()
                    / pred_quantity.detach().clamp(min=1e-6)
                ).unsqueeze(1)
                alpha_pred_for_tokens = alpha_pred * pred_scale
                blended_alpha = w_target * alpha_target + w_pred * alpha_pred_for_tokens
                out = forward_from_cif(
                    model,
                    frame_features,
                    batch["frame_lengths"],
                    alphas=blended_alpha,
                )
            else:
                out = model(
                    batch["keypoints"],
                    batch["frame_lengths"],
                    alphas=alpha_target,
                    target_lengths=batch["token_lengths"],
                )

            token_logits, token_targets = gather_logits(out["token_logits"], batch["token_ids"])
            token_loss = F.cross_entropy(token_logits, token_targets)
            pred_emb, target_emb = gather_embeddings(
                out["embeddings"], batch["token_ids"], batch["target_embeddings"]
            )
            emb_loss = F.smooth_l1_loss(pred_emb, target_emb)

            if args.phase == "learned_cif":
                alpha_loss = compute_alpha_loss(
                    args.diag_alpha_loss,
                    alpha_logits,
                    alpha_pred,
                    alpha_target,
                    frame_mask,
                )
                qty_loss = F.l1_loss(pred_quantity, batch["token_lengths"].float())
                if epoch < 3 or args.diag_freeze == "alpha_only":
                    loss = 5.0 * alpha_loss + qty_loss
                else:
                    loss = token_loss + 0.05 * emb_loss + 5.0 * alpha_loss + qty_loss
            else:
                alpha_loss = token_loss.new_zeros(())
                qty_loss = F.l1_loss(out["cif"].quantity, batch["token_lengths"].float())
                loss = token_loss + 0.05 * emb_loss + 0.1 * qty_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norms = compute_grad_norms(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += loss.item()
            totals["token_loss"] += token_loss.item()
            totals["emb_loss"] += emb_loss.item()
            totals["qty_loss"] += qty_loss.item()
            totals["alpha_loss"] += alpha_loss.item()
            if args.phase == "learned_cif":
                for key, value in alpha_diagnostics(
                    frame_features,
                    alpha_logits,
                    alpha_pred,
                    batch["frame_lengths"],
                ).items():
                    totals[key] += value
            for key, value in grad_norms.items():
                totals[f"grad_norm_{key}"] += value
            steps += 1

        train_metrics = {key: value / max(1, steps) for key, value in totals.items()}

        if args.phase == "learned_cif":
            val_teacher = evaluate_teacher(model, val_loader, device)
            val_pred_raw = evaluate_predicted(model, val_loader, device)
            val_pred_rescaled = evaluate_alpha_mode(
                model,
                val_loader,
                device,
                "pred_rescaled_to_target_len",
            )
            val_blended = evaluate_alpha_mode(
                model,
                val_loader,
                device,
                "blended_alpha",
                w_target=w_target,
                w_pred=w_pred,
            )
            val_predicted = val_pred_rescaled if args.diag_eval_force_count else val_pred_raw
            val_permuted = evaluate_permuted(model, val_loader, device, seed=args.seed + epoch)
            permuted_drop = val_pred_raw["top1"] - val_permuted["top1"]
            row = {
                "epoch": epoch,
                "phase_state": phase_state,
                "tf_weights": [w_target, w_pred],
                "diagnostics": {
                    "alpha_loss": args.diag_alpha_loss,
                    "freeze": args.diag_freeze,
                    "eval_force_count": args.diag_eval_force_count,
                },
                "train": train_metrics,
                "val_teacher": val_teacher,
                "val_pred_raw": val_pred_raw,
                "val_pred_rescaled_to_target_len": val_pred_rescaled,
                "val_blended_alpha": val_blended,
                "val_predicted": val_predicted,
                "val_permuted_top1": val_permuted["top1"],
                "permuted_top1_drop": permuted_drop,
            }
            select_metric = val_predicted["top1"]
            print(
                f"ep{epoch:03d} loss={train_metrics['loss']:.4f} tf=({w_target:.2f},{w_pred:.2f}) "
                f"teacher_top1={val_teacher['top1']:.3f} pred_raw_top1={val_pred_raw['top1']:.3f} "
                f"pred_rescaled_top1={val_pred_rescaled['top1']:.3f} pred_exact={val_predicted['exact']:.3f} "
                f"pred_mae_len={val_predicted['mae_len']:.3f} pred_boundary_mae={val_predicted['boundary_mae']:.3f} "
                f"permuted_drop={permuted_drop:.3f}",
                flush=True,
            )
            final_gates = {
                "predicted_mae_length<=0.5": val_predicted["mae_len"] <= 0.5,
                "predicted_token_top1>=0.75": val_predicted["top1"] >= 0.75,
                "predicted_token_top5>=0.90": val_predicted["top5"] >= 0.90,
                "predicted_exact_sequence_accuracy>=0.55": val_predicted["exact"] >= 0.55,
                "predicted_boundary_error_mae<=5": val_predicted["boundary_mae"] <= 5.0,
                "permuted_video_top1_drop>=0.30": permuted_drop >= 0.30,
            }
        else:
            val_metrics = evaluate_teacher(model, val_loader, device)
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            select_metric = val_metrics["top1"]
            print(
                f"ep{epoch:03d} loss={train_metrics['loss']:.4f} "
                f"val_top1={val_metrics['top1']:.3f} val_top5={val_metrics['top5']:.3f} "
                f"exact={val_metrics['exact']:.3f} mae_len={val_metrics['mae_len']:.3f}",
                flush=True,
            )

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_top1": best_top1,
            "token_ids_by_label": token_ids_by_label,
            "load_info": load_info,
            "phase": args.phase,
        }
        torch.save(state, ckpt_path)
        if select_metric > best_top1:
            best_top1 = select_metric
            state["best_top1"] = best_top1
            torch.save(state, best_path)

    if final_gates is not None:
        with open(out_dir / "gates.json", "w", encoding="utf-8") as f:
            json.dump(final_gates, f, indent=2)
        print(f"[v126b] gates={final_gates}", flush=True)

    print(f"[v126] done best_top1={best_top1:.3f} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
