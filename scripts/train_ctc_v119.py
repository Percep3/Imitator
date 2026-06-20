"""v119 — CTC sobre secuencia (reemplaza el objetivo contrastivo global de v118).

Tres formulaciones contrastivas (InfoNCE, InfoNCE+aug, VICReg) convergieron al
mismo techo (R@1 ≈ 2.5-2.9%, train≈val -> no es sobreajuste, ver
outputs/diag_v118_train_val_gap.json). La causa no es la pérdida ni los datos,
sino comprimir la frase entera en un vector y rankearla. Aquí el encoder emite
una distribución por paso temporal y CTC aprende el alineamiento implícito
contra la secuencia de palabras -- igual que LiftSign (CVPRW 2026) formula CSLR.

Vocabulario propio (word-level, NO el BPE de Gemma): se construye desde las
labels de TRAIN únicamente (evita fuga val->vocab), blank=0 (convención CTC).

Ablation (mismo script, flags en [model] del toml):
    ctc_v119.toml  (A1): use_motion_stream=false, use_tlp=false
    ctc_v119b.toml (A2): use_motion_stream=true,  use_tlp=false
    ctc_v119c.toml (A3): use_motion_stream=true,  use_tlp=true

Uso:
    MSLM_EXPERIMENT_CONFIG=config/experiment/ctc_v119.toml \
        PYTHONPATH=. python scripts/train_ctc_v119.py
"""
import os

os.environ.setdefault("MSLM_EXPERIMENT_CONFIG", "config/experiment/ctc_v119.toml")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from settings import initialize

initialize()

import functools
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Orden de imports importante: utils.setup_train ANTES de models.ctc_encoder.
# src/mslm/models/__init__.py <-> src/mslm/utils/__init__.py tienen un import
# circular real (models importa utils.early_stopping, utils importa setup_train,
# que importa `from src.mslm.models import Imitator`); cargar utils primero deja
# que `models` termine de inicializarse antes de que setup_train lo necesite. Es
# el mismo orden que ya usa scripts/train_contrastive_v118.py.
from src.mslm.utils.setup_train import setup_paths
from src.mslm.utils.config_loader import cfg
from src.mslm.utils.wer import greedy_ctc_decode, word_error_rate
from src.mslm.models.ctc_encoder import CTCEncoder
from src.mslm.dataloader import KeypointDataset, BatchSampler
from src.mslm.dataloader.vocab import Vocab, collect_labels, tokenize
from src.mslm.training.loss_ctc import ctc_loss
from src.mslm.checkpoint.manager import CheckpointManager

torch.manual_seed(23)
random.seed(23)


def _cfg(section, key, default):
    s = getattr(cfg, section, {}) or {}
    return s.get(key, default)


def _truncate(keypoint, frames_mask, max_frames):
    if max_frames and keypoint.size(1) > max_frames:
        keypoint = keypoint[:, :max_frames]
        frames_mask = frames_mask[:, :max_frames]
    return keypoint, frames_mask


def ctc_collate_fn(batch, vocab: Vocab):
    """Pad-ea keypoints igual que components.collate_fn y arma los targets
    concatenados que pide nn.CTCLoss a partir de las labels crudas."""
    keypoints_list = [item[0] for item in batch]
    labels = [item[2] for item in batch]

    frame_lengths = torch.tensor([kp.size(0) for kp in keypoints_list], dtype=torch.long)
    keypoints_padded = pad_sequence(keypoints_list, batch_first=True, padding_value=0.0).float()

    B, T_max = keypoints_padded.shape[:2]
    arange_frames = torch.arange(T_max).unsqueeze(0).expand(B, -1)
    frames_mask = arange_frames >= frame_lengths.unsqueeze(1)

    target_ids = [torch.tensor(vocab.encode(lbl), dtype=torch.long) for lbl in labels]
    target_lengths = torch.tensor([len(t) for t in target_ids], dtype=torch.long)
    targets_concat = torch.cat(target_ids)

    return keypoints_padded, frames_mask.bool(), targets_concat, target_lengths, labels


def _encode_batch(model, keypoint, frames_mask, grad_ckpt: bool):
    """Codifica cada vídeo del batch por separado (con gradient checkpointing),
    igual que encode_video_batch en train_contrastive_v118.py: un forward batcheado
    del STGCN sobre frames de padding (T_max compartido por todo el batch) agota la
    VRAM con vídeos largos -- mismo problema que motivó ese patrón en v118. Procesar
    cada muestra a su longitud REAL (sin padding) evita ese desperdicio de memoria.
    """
    B = keypoint.size(0)
    lengths = (~frames_mask).sum(dim=1)
    log_probs_list, seq_lengths_list, aux_list = [], [], []
    for i in range(B):
        L = int(lengths[i].item())
        kp = keypoint[i : i + 1, :L]
        fm = torch.zeros(1, L, dtype=torch.bool, device=keypoint.device)
        if grad_ckpt and model.training:
            lp, sl, aux = checkpoint(model, kp, fm, use_reentrant=False)
        else:
            lp, sl, aux = model(kp, fm)
        log_probs_list.append(lp)
        seq_lengths_list.append(sl)
        aux_list.append(aux)

    max_t = max(lp.size(1) for lp in log_probs_list)
    padded = [F.pad(lp, (0, 0, 0, max_t - lp.size(1))) for lp in log_probs_list]
    log_probs = torch.cat(padded, dim=0)
    seq_lengths = torch.cat(seq_lengths_list)
    aux = {}
    if aux_list and aux_list[0]:
        aux = {k: sum(a[k] for a in aux_list) / len(aux_list) for k in aux_list[0]}
    return log_probs, seq_lengths, aux


def _forward_batch(model, keypoint, frames_mask, targets, target_lengths, device, max_frames,
                    grad_ckpt: bool = True):
    keypoint, frames_mask = keypoint.to(device), frames_mask.to(device)
    keypoint, frames_mask = _truncate(keypoint, frames_mask, max_frames)
    log_probs, seq_lengths, aux = _encode_batch(model, keypoint, frames_mask, grad_ckpt)
    loss = ctc_loss(log_probs, targets.to(device), seq_lengths, target_lengths.to(device))
    return log_probs, seq_lengths, aux, loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, h5_file = setup_paths()
    from src.mslm.utils.paths import path_vars

    adj_path = path_vars.data_path / "processed" / "adjacency_matrix.npy"
    A = np.load(adj_path, allow_pickle=True)

    model_cfg = dict(cfg.model)
    use_motion_stream = bool(model_cfg.get("use_motion_stream", False))
    use_tlp = bool(model_cfg.get("use_tlp", False))
    lambda_u = float(_cfg("loss", "lambda_u", 0.1))
    lambda_p = float(_cfg("loss", "lambda_p", 0.1))

    epochs = int(_cfg("training", "epochs", 60))
    batch_size = int(_cfg("training", "batch_size", 16))
    lr = float(_cfg("training", "learning_rate", 3e-4))
    wd = float(_cfg("training", "weight_decay", 0.05))
    warmup = int(_cfg("training", "warmup_epochs", 5))
    max_frames = int(_cfg("training", "max_frames", 1024))
    version = int(_cfg("training", "model_version", 119))
    run_id = int(_cfg("training", "run_id", 1))
    patience = int(_cfg("training", "early_stopping_patience", 25))

    include = _cfg("data", "primary_datasets", ["dataset2"])
    train_ratio = float(_cfg("data", "train_ratio", 0.8))
    text_group = _cfg("data", "text_group", "text_ctx")
    n_keypoints = int(_cfg("data", "n_keypoints", model_cfg.get("input_size", 111)))
    max_samples = _cfg("data", "max_samples", None)

    ds = KeypointDataset(
        h5Path=h5_file, n_keypoints=n_keypoints, return_label=True,
        text_group=text_group, include_datasets=include,
        data_augmentation=False, max_length=4000,
    )

    if max_samples and int(max_samples) < len(ds.valid_index):
        # Muestra aleatoria determinista (seed=23): dataset2 mezcla los 600 clips
        # originales con los 5000 añadidos en v118f bajo el mismo índice secuencial
        # (verificado: las claves "0".."599" NO corresponden a los 600 originales,
        # su media de frames no coincide con la documentada en report.md), así que
        # no hay forma de recuperar esa identidad por id -- se usa un subconjunto
        # aleatorio del tamaño pedido en vez de asumir una identidad incorrecta.
        rng = random.Random(23)
        idx = list(range(len(ds.valid_index)))
        rng.shuffle(idx)
        keep = sorted(idx[: int(max_samples)])
        ds.valid_index = [ds.valid_index[i] for i in keep]
        ds.video_lengths = [ds.video_lengths[i] for i in keep]
        ds.dataset_length = len(ds.valid_index)
        print(f"[v119] Subset aleatorio: {ds.dataset_length} clips (max_samples={max_samples}).")

    train_subset, val_subset, _, _ = ds.split_dataset(train_ratio)

    # Vocab SOLO con labels de train (evita fuga val->vocab).
    train_clip_ids = [ds.valid_index[i][1] for i in train_subset.indices]
    train_labels = collect_labels(h5_file, include[0], train_clip_ids)
    vocab = Vocab.build_from_labels(train_labels)
    print(f"[v119] Vocab: {len(vocab)} tokens (incl. blank+unk) desde {len(train_labels)} labels de train.")

    collate = functools.partial(ctc_collate_fn, vocab=vocab)
    train_dl = DataLoader(
        train_subset, num_workers=8, pin_memory=True, persistent_workers=True,
        collate_fn=collate, batch_sampler=BatchSampler(train_subset, batch_size),
    )
    val_dl = DataLoader(
        val_subset, num_workers=8, pin_memory=True, persistent_workers=True,
        collate_fn=collate, batch_sampler=BatchSampler(val_subset, batch_size),
    )

    # CTCEncoder.vocab_size = clases SIN contar blank (suma +1 internamente para
    # blank=0); Vocab ya reserva el id 0 para blank, así que restamos 1 para que
    # el índice 0 del clasificador siga correspondiendo a blank.
    model = CTCEncoder(
        A=A, input_size=model_cfg.get("input_size", n_keypoints),
        hidden_size=model_cfg.get("hidden_size", 1024),
        nhead=model_cfg.get("nhead", 16),
        ff_dim=model_cfg.get("ff_dim", 2816),
        n_layers=model_cfg.get("n_layers", 6),
        encoder_dropout=model_cfg.get("encoder_dropout", 0.4),
        multihead_dropout=model_cfg.get("multihead_dropout", 0.1),
        vocab_size=len(vocab) - 1,
        use_motion_stream=use_motion_stream,
        use_tlp=use_tlp,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[v119] CTCEncoder: {n_params:.2f} M params entrenables | "
          f"use_motion_stream={use_motion_stream} use_tlp={use_tlp} "
          f"batch={batch_size} max_frames={max_frames}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, len(train_dl))

    def lr_at(step):
        warm = warmup * steps_per_epoch
        total = epochs * steps_per_epoch
        if step < warm:
            return step / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    writer = SummaryWriter(
        f"../outputs/reports/{version}/{run_id}/{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
    ckpt = CheckpointManager("../outputs/checkpoints", version, run_id)
    vocab.save(Path(ckpt._path()) / "vocab.json")

    best_wer, since_improve, gstep = float("inf"), 0, 0
    for epoch in range(epochs):
        # ---- train ----
        model.train()
        tot, n_steps = 0.0, 0
        for keypoint, frames_mask, targets, target_lengths, _ in tqdm(
            train_dl, desc=f"ep{epoch} train", leave=False, mininterval=10.0
        ):
            _, _, aux, loss = _forward_batch(
                model, keypoint, frames_mask, targets, target_lengths, device, max_frames)
            if use_tlp:
                loss = loss + lambda_u * aux["L_u"] + lambda_p * aux["L_p"]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            gstep += 1
            tot += loss.item()
            n_steps += 1
        train_loss = tot / max(1, n_steps)

        # ---- val: WER sobre TODO el val (greedy decode) ----
        model.eval()
        val_tot, vb = 0.0, 0
        hyp_words_all, ref_words_all = [], []
        with torch.no_grad():
            for keypoint, frames_mask, targets, target_lengths, labels in val_dl:
                log_probs, seq_lengths, aux, loss = _forward_batch(
                    model, keypoint, frames_mask, targets, target_lengths, device, max_frames)
                if use_tlp:
                    loss = loss + lambda_u * aux["L_u"] + lambda_p * aux["L_p"]
                val_tot += loss.item()
                vb += 1

                decoded_ids = greedy_ctc_decode(log_probs.cpu(), seq_lengths.cpu(), blank=vocab.blank_id)
                for ids, label in zip(decoded_ids, labels):
                    hyp_words_all.append(vocab.decode(ids))
                    ref_words_all.append(tokenize(label))
        val_loss = val_tot / max(1, vb)
        wers = [word_error_rate(h, r) for h, r in zip(hyp_words_all, ref_words_all) if r]
        val_wer = sum(wers) / max(1, len(wers))

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("WER/val", val_wer, epoch)
        tqdm.write(f"ep{epoch:>3} | train {train_loss:.3f} | val {val_loss:.3f} | val WER {val_wer:.1%}")

        improved = val_wer < best_wer
        if improved:
            best_wer, since_improve = val_wer, 0
            ckpt.save_checkpoint(model, epoch, opt, sched, tag="best_wer")
            tqdm.write(f"  ↓ best WER: {best_wer:.1%} (ep {epoch})")
        else:
            since_improve += 1
        if (epoch + 1) % int(_cfg("training", "checkpoint_interval", 10)) == 0:
            ckpt.save_checkpoint(model, epoch, opt, sched)
        if since_improve >= patience:
            tqdm.write(f"Early stopping: sin mejora de WER en {patience} épocas.")
            break

    writer.close()
    print(f"[v119] FIN. Mejor WER val = {best_wer:.1%}.")


if __name__ == "__main__":
    main()
