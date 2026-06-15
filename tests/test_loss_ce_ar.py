"""Unit tests for the AR CE loss (v116). Runs without GPU or model."""
import importlib.util
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "loss_ce_ar", _ROOT / "src/mslm/training/loss_ce_ar.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
imitator_ar_loss = _mod.imitator_ar_loss

VOCAB = 50
B, K, L = 2, 4, 6  # batch, prefix tokens, text tokens


def _logits(b=B, seq=K + L - 1, v=VOCAB, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, seq, v, generator=g)


def _labels(b=B, k=K, l=L):
    """Labels: -100 for prefix positions, random token IDs for the rest."""
    g = torch.Generator().manual_seed(42)
    text = torch.randint(0, VOCAB, (b, l - 1), generator=g)
    prefix = torch.full((b, k), -100, dtype=torch.long)
    return torch.cat([prefix, text], dim=1)  # [B, K+L-1]


def test_returns_4_tuple():
    logits = _logits()
    labels = _labels()
    result = imitator_ar_loss(logits, labels, k_prefix=K)
    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}"


def test_prefix_positions_ignored():
    """Changing logits at prefix positions (-100 labels) must not change loss."""
    logits = _logits()
    labels = _labels()

    # Completely corrupt the prefix logit positions
    logits_corrupt = logits.clone()
    logits_corrupt[:, :K, :] = 1e6  # huge garbage values at prefix positions

    ce_orig, _, _, _ = imitator_ar_loss(logits, labels, k_prefix=K)
    ce_corrupt, _, _, _ = imitator_ar_loss(logits_corrupt, labels, k_prefix=K)
    assert torch.allclose(ce_orig, ce_corrupt, atol=1e-5), (
        f"Prefix positions affected loss: {ce_orig.item():.6f} vs {ce_corrupt.item():.6f}"
    )


def test_padding_ignored():
    """Padding positions (labels=-100) don't affect CE or accuracy."""
    logits = _logits()
    labels = _labels()

    # Add padding at the end of text (set last 2 text positions to -100)
    labels_padded = labels.clone()
    labels_padded[:, -2:] = -100

    # Corrupt logits at those padding positions
    logits_corrupt_pad = logits.clone()
    logits_corrupt_pad[:, -2:, :] = -1e6

    ce_a, _, acc1_a, acc5_a = imitator_ar_loss(logits, labels_padded, k_prefix=K)
    ce_b, _, acc1_b, acc5_b = imitator_ar_loss(logits_corrupt_pad, labels_padded, k_prefix=K)

    assert torch.allclose(ce_a, ce_b, atol=1e-5), (
        f"Padding positions affected loss: {ce_a.item():.6f} vs {ce_b.item():.6f}"
    )
    assert torch.allclose(acc1_a, acc1_b, atol=1e-5)
    assert torch.allclose(acc5_a, acc5_b, atol=1e-5)


def test_top5_geq_top1():
    """Top-5 accuracy is always >= top-1 accuracy."""
    logits = _logits()
    labels = _labels()
    _, _, top1, top5 = imitator_ar_loss(logits, labels, k_prefix=K)
    assert top5.item() >= top1.item(), (
        f"top5 ({top5.item():.4f}) < top1 ({top1.item():.4f})"
    )


def test_gradient_flows_to_logits_not_to_labels():
    """Gradient flows through logits; labels are long so no grad needed."""
    logits = torch.randn(B, K + L - 1, VOCAB, requires_grad=True)
    labels = _labels()
    ce, _, _, _ = imitator_ar_loss(logits, labels, k_prefix=K)
    ce.backward()
    assert logits.grad is not None, "logits.grad is None — gradient did not flow"
    assert logits.grad.abs().sum().item() > 0, "logits.grad is all zeros"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK {name}")
    print("All tests passed.")
