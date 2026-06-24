"""TemporalSignPromptModel building blocks for v126."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.stgcn import STGCNBlock, make_channel_norm_2d, partition_adjacency


@dataclass
class CIFOutput:
    embeddings: torch.Tensor
    padding_mask: torch.Tensor
    counts: torch.Tensor
    quantity: torch.Tensor
    fire_positions: torch.Tensor


class CIFAggregator(nn.Module):
    """Continuous integrate-and-fire aggregation over frame features.

    ``forward`` accepts frame features and optional externally supplied alphas.
    Supplying alphas makes the module easy to unit-test and lets v126 supervise
    the alignment from known synthetic boundaries; omitting them uses a learned
    sigmoid predictor.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.alpha = nn.Linear(hidden_size, 1)

    @staticmethod
    def _length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        steps = torch.arange(max_len, device=lengths.device)
        return steps.unsqueeze(0) >= lengths.unsqueeze(1)

    def predict_alpha(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        alphas = torch.sigmoid(self.predict_alpha_logits(features, lengths))
        return alphas.masked_fill(self._length_mask(lengths, features.size(1)), 0.0)

    def predict_alpha_logits(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        logits = self.alpha(features).squeeze(-1)
        return logits.masked_fill(self._length_mask(lengths, features.size(1)), 0.0)

    @staticmethod
    def token_centers(
        boundaries: torch.Tensor,
        token_spans: torch.Tensor,
    ) -> torch.Tensor:
        """Uniformly spaced target frame positions for each target token.

        A sign spanning frames ``[start, end)`` and producing tokens
        ``token_spans[b, s] = (t0, t1)`` places its ``k = t1 - t0`` token
        centers at ``start + (i + 0.5) * (end - start) / k`` for ``i in [0, k)``.
        Padding entries (``boundaries`` or ``token_spans`` with a negative
        start) are skipped. Output is padded with ``-1`` up to the maximum
        number of tokens across the batch.
        """
        if boundaries.dim() == 2:
            boundaries = boundaries.unsqueeze(0)
            token_spans = token_spans.unsqueeze(0)
        B, S, _ = boundaries.shape

        per_sample_centers: list[list[float]] = [[] for _ in range(B)]
        for b in range(B):
            for s in range(S):
                start, end = boundaries[b, s].tolist()
                t0, t1 = token_spans[b, s].tolist()
                if start < 0 or end <= start or t0 < 0 or t1 <= t0:
                    continue
                k = t1 - t0
                span = float(end - start)
                for i in range(k):
                    per_sample_centers[b].append(start + (i + 0.5) * span / k)

        max_tokens = max((len(c) for c in per_sample_centers), default=0)
        max_tokens = max(max_tokens, 1)
        out = boundaries.new_full((B, max_tokens), -1.0, dtype=torch.float32)
        for b, centers in enumerate(per_sample_centers):
            if centers:
                out[b, : len(centers)] = torch.tensor(centers, dtype=torch.float32)
        return out

    @staticmethod
    def boundary_targets(
        boundaries: torch.Tensor,
        frame_count: int,
        *,
        token_spans: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build alpha targets whose per-sign mass equals its token count.

        Without token spans each sign receives mass 1. With token spans, a sign
        spanning ``k`` target tokens receives total mass ``k`` spread uniformly
        across its frames.
        """
        if boundaries.dim() == 2:
            boundaries = boundaries.unsqueeze(0)
        B, S, _ = boundaries.shape
        out = torch.zeros(B, frame_count, dtype=torch.float32, device=boundaries.device)
        for b in range(B):
            for s in range(S):
                start, end = boundaries[b, s].tolist()
                if start < 0 or end <= start:
                    continue
                mass = 1.0
                if token_spans is not None:
                    t0, t1 = token_spans[b, s].tolist()
                    mass = max(0, t1 - t0)
                out[b, start:end] = float(mass) / float(end - start)
        return out

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        *,
        alphas: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
    ) -> CIFOutput:
        if features.dim() != 3:
            raise ValueError("features must have shape [B, T, H]")
        B, T, H = features.shape
        if alphas is None:
            alphas = self.predict_alpha(features, lengths)
        else:
            alphas = alphas.to(device=features.device, dtype=features.dtype)
            alphas = alphas.masked_fill(self._length_mask(lengths, T), 0.0)

        quantity = alphas.sum(dim=1)
        if target_lengths is not None:
            scale = target_lengths.to(features).clamp(min=0) / quantity.clamp(min=1e-6)
            alphas = alphas * scale.unsqueeze(1)
            quantity = alphas.sum(dim=1)

        sample_outputs = []
        sample_positions = []
        counts = []
        for b in range(B):
            outputs, positions = self._integrate_one(features[b], alphas[b], int(lengths[b].item()))
            sample_outputs.append(outputs)
            sample_positions.append(positions)
            counts.append(outputs.size(0))

        max_count = max(max(counts), 1)
        padded = features.new_zeros((B, max_count, H))
        pos_padded = features.new_full((B, max_count), -1.0)
        padding_mask = torch.ones((B, max_count), dtype=torch.bool, device=features.device)
        for b, (outputs, positions) in enumerate(zip(sample_outputs, sample_positions)):
            if outputs.numel() == 0:
                continue
            n = outputs.size(0)
            padded[b, :n] = outputs
            pos_padded[b, :n] = positions
            padding_mask[b, :n] = False

        return CIFOutput(
            embeddings=padded,
            padding_mask=padding_mask,
            counts=torch.tensor(counts, dtype=torch.long, device=features.device),
            quantity=quantity,
            fire_positions=pos_padded,
        )

    @staticmethod
    def _integrate_one(features: torch.Tensor, alphas: torch.Tensor, length: int):
        acc = features.new_zeros(())
        state = features.new_zeros(features.size(-1))
        pos_state = features.new_zeros(())
        outputs = []
        positions = []

        for t in range(length):
            remaining = alphas[t]
            while bool((acc + remaining) >= 1.0 - 1e-4):
                need = 1.0 - acc
                outputs.append(state + need * features[t])
                positions.append(pos_state + need * features.new_tensor(float(t)))
                remaining = remaining - need
                acc = features.new_zeros(())
                state = features.new_zeros(features.size(-1))
                pos_state = features.new_zeros(())
            if bool(remaining > 0):
                acc = acc + remaining
                state = state + remaining * features[t]
                pos_state = pos_state + remaining * features.new_tensor(float(t))

        if bool(acc >= 1.0 - 1e-4):
            outputs.append(state)
            positions.append(pos_state)

        if outputs:
            return torch.stack(outputs), torch.stack(positions)
        return features.new_zeros((0, features.size(-1))), features.new_zeros((0,))


class TemporalSignPromptModel(nn.Module):
    """Thin v126 scaffold: frame encoder -> CIF -> token/embedding heads.

    The frame encoder is intentionally injected so tests and the training script
    can evolve independently from the synthetic-data contract.
    """

    def __init__(
        self,
        frame_encoder: nn.Module,
        hidden_size: int,
        vocab_size: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.cif = CIFAggregator(hidden_size)
        self.token_head = nn.Linear(hidden_size, vocab_size)
        self.embedding_head = nn.Linear(hidden_size, embedding_dim)

    def forward(self, keypoints: torch.Tensor, frame_lengths: torch.Tensor, **cif_kwargs):
        frame_features = self.frame_encoder(keypoints, frame_lengths)
        cif = self.cif(frame_features, frame_lengths, **cif_kwargs)
        return {
            "cif": cif,
            "token_logits": self.token_head(cif.embeddings),
            "embeddings": self.embedding_head(cif.embeddings),
        }


class STGCNTemporalFrameEncoder(nn.Module):
    """ST-GCN + TCN + Transformer frame encoder for v126 synthetic sequences."""

    def __init__(
        self,
        A,
        *,
        gcn_channels=(32, 64, 128),
        hidden_size: int = 128,
        norm_type: str = "group",
        norm_groups: int = 16,
        tcn_layers: int = 2,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        A_part = partition_adjacency(A)
        layers = nn.ModuleList()
        c_in = 2
        for c_out in gcn_channels:
            layers.append(
                STGCNBlock(
                    c_in,
                    c_out,
                    A_part,
                    kernel_size=3,
                    norm_type=norm_type,
                    norm_groups=norm_groups,
                )
            )
            c_in = 3 * c_out
        self.stgcn_layers = layers
        self.linear_hidden = nn.Sequential(
            nn.Conv2d(c_in, hidden_size, kernel_size=1),
            make_channel_norm_2d(hidden_size, norm_type, norm_groups),
            nn.ReLU(),
        )
        tcn = []
        for _ in range(tcn_layers):
            tcn.extend(
                [
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                    nn.GroupNorm(1, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.tcn = nn.Sequential(*tcn)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=transformer_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

    @staticmethod
    def _length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        steps = torch.arange(max_len, device=lengths.device)
        return steps.unsqueeze(0) >= lengths.unsqueeze(1)

    def forward(self, keypoints: torch.Tensor, frame_lengths: torch.Tensor) -> torch.Tensor:
        x = keypoints.permute(0, 3, 1, 2)
        for layer in self.stgcn_layers:
            x = layer(x)
        x = self.linear_hidden(x).mean(dim=-1)
        x = self.tcn(x).permute(0, 2, 1).contiguous()
        return self.transformer(
            x,
            src_key_padding_mask=self._length_mask(frame_lengths, x.size(1)),
        )


def alpha_schedule_weights(epoch: int) -> tuple[float, float]:
    """Teacher-forcing blend ``(w_target, w_pred)`` for the learned-CIF curriculum."""
    if epoch < 3:
        return (0.75, 0.25)
    if epoch < 6:
        return (0.50, 0.50)
    if epoch < 10:
        return (0.25, 0.75)
    return (0.0, 1.0)


def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad = value


def module_trainable_state(model: "TemporalSignPromptModel") -> dict[str, bool]:
    encoder = model.frame_encoder
    state = {
        "frame_encoder": any(p.requires_grad for p in encoder.parameters()),
        "cif": any(p.requires_grad for p in model.cif.parameters()),
        "token_head": any(p.requires_grad for p in model.token_head.parameters()),
        "embedding_head": any(p.requires_grad for p in model.embedding_head.parameters()),
    }
    for name in ("stgcn_layers", "linear_hidden", "tcn", "transformer"):
        if hasattr(encoder, name):
            module = getattr(encoder, name)
            state[f"frame_encoder.{name}"] = any(p.requires_grad for p in module.parameters())
    return state


def set_cif_phase(model: "TemporalSignPromptModel", epoch: int) -> dict[str, bool]:
    """Apply the v126b freezing schedule for the given epoch.

    - epoch 0-2: only ``cif.alpha`` trains; ST-GCN, TCN/Transformer and heads frozen.
    - epoch 3-9: ``cif``, TCN/Transformer and heads train; ST-GCN stays frozen.
    - epoch 10+: everything trains.
    """
    encoder = model.frame_encoder
    if epoch < 3:
        _set_requires_grad(encoder, False)
        _set_requires_grad(model.token_head, False)
        _set_requires_grad(model.embedding_head, False)
        _set_requires_grad(model.cif, True)
    elif epoch < 10:
        _set_requires_grad(encoder.stgcn_layers, False)
        _set_requires_grad(encoder.linear_hidden, False)
        _set_requires_grad(encoder.tcn, True)
        _set_requires_grad(encoder.transformer, True)
        _set_requires_grad(model.cif, True)
        _set_requires_grad(model.token_head, True)
        _set_requires_grad(model.embedding_head, True)
    else:
        _set_requires_grad(model, True)
    return module_trainable_state(model)


def set_cif_diagnostic_freeze(
    model: "TemporalSignPromptModel",
    epoch: int,
    mode: str,
) -> dict[str, bool]:
    """Apply diagnostic freezing regimes without changing the default schedule."""
    if mode == "full_current":
        return set_cif_phase(model, epoch)

    encoder = model.frame_encoder
    _set_requires_grad(model, False)

    if mode == "alpha_only":
        _set_requires_grad(model.cif, True)
    elif mode == "target_only_stage1":
        if epoch < 3:
            _set_requires_grad(model.cif, True)
        else:
            _set_requires_grad(encoder.tcn, True)
            _set_requires_grad(encoder.transformer, True)
            _set_requires_grad(model.cif, True)
            _set_requires_grad(model.token_head, True)
            _set_requires_grad(model.embedding_head, True)
    elif mode == "heads_only":
        _set_requires_grad(model.token_head, True)
        _set_requires_grad(model.embedding_head, True)
    elif mode == "tcn_heads":
        _set_requires_grad(encoder.tcn, True)
        _set_requires_grad(encoder.transformer, True)
        _set_requires_grad(model.token_head, True)
        _set_requires_grad(model.embedding_head, True)
    else:
        raise ValueError(f"unknown diagnostic freeze mode: {mode}")

    return module_trainable_state(model)


def length_mask_from_lengths(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    steps = torch.arange(max_len, device=lengths.device)
    return steps.unsqueeze(0) < lengths.unsqueeze(1)


def rescale_alphas_to_target_lengths(
    alphas: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    scale = target_lengths.to(alphas).clamp(min=0) / alphas.sum(dim=1).clamp(min=1e-6)
    return alphas * scale.unsqueeze(1)


def alpha_diagnostics(
    features: torch.Tensor,
    alpha_logits: torch.Tensor,
    alphas: torch.Tensor,
    lengths: torch.Tensor,
) -> dict[str, float]:
    """Finite summary stats for CIF calibration probes."""
    mask = length_mask_from_lengths(lengths, alphas.size(1))
    valid_logits = alpha_logits[mask]
    valid_alphas = alphas[mask]
    valid_features = features[mask]
    if valid_alphas.numel() == 0:
        return {
            "alpha_logit_mean": 0.0,
            "alpha_logit_p01": 0.0,
            "alpha_logit_p50": 0.0,
            "alpha_logit_p99": 0.0,
            "alpha_mean": 0.0,
            "alpha_sum_mean": 0.0,
            "feature_norm_mean": 0.0,
            "feature_norm_max": 0.0,
        }
    feature_norm = valid_features.norm(dim=-1)
    return {
        "alpha_logit_mean": valid_logits.mean().item(),
        "alpha_logit_p01": valid_logits.quantile(0.01).item(),
        "alpha_logit_p50": valid_logits.median().item(),
        "alpha_logit_p99": valid_logits.quantile(0.99).item(),
        "alpha_mean": valid_alphas.mean().item(),
        "alpha_sum_mean": alphas.sum(dim=1).mean().item(),
        "feature_norm_mean": feature_norm.mean().item(),
        "feature_norm_max": feature_norm.max().item(),
    }


def boundary_error_mae(
    fire_positions: torch.Tensor,
    counts: torch.Tensor,
    centers: torch.Tensor,
    token_lengths: torch.Tensor,
    missing_penalty: float = 1000.0,
) -> float:
    """Mean absolute error between fired CIF positions and token-level centers.

    Matched entries compare positions directly. Missing or extra fired tokens
    receive a large penalty so a collapsed CIF cannot pass the boundary gate.
    """
    diffs = []
    for b in range(fire_positions.size(0)):
        count = int(counts[b].item())
        target = int(token_lengths[b].item())
        n = min(count, target)
        if n > 0:
            diffs.append((fire_positions[b, :n] - centers[b, :n]).abs())
        missing_or_extra = abs(count - target)
        if missing_or_extra:
            diffs.append(
                fire_positions.new_full((missing_or_extra,), float(missing_penalty))
            )
    if not diffs:
        return float(missing_penalty)
    return torch.cat(diffs).mean().item()


def load_visual_low_level_weights(model: nn.Module, checkpoint_path) -> dict[str, int]:
    """Load v121 visual low-level weights, excluding classifier-like heads."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint.get("model_state", checkpoint)
    target = model.state_dict()
    loadable = {}
    skipped = 0
    for name, value in source.items():
        if name.startswith("classifier.") or "temporal_attention" in name:
            skipped += 1
            continue
        if name in target and target[name].shape == value.shape:
            loadable[name] = value
        else:
            skipped += 1
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    return {
        "loaded": len(loadable),
        "skipped": skipped,
        "missing": len(missing),
        "unexpected": len(unexpected),
    }
