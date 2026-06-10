"""Unit tests para la pérdida CE-vocab (v115). Corren sin datos ni GPU."""
import importlib.util
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]

# Carga por ruta de archivo (solo importa torch) para no disparar el import circular
# latente training ↔ utils.setup_train (ver tests/test_sigreg.py).
_spec = importlib.util.spec_from_file_location(
    "loss_ce_vocab", _ROOT / "src/mslm/training/loss_ce_vocab.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
imitator_ce_loss = _mod.imitator_ce_loss
IGNORE_INDEX = _mod.IGNORE_INDEX

VOCAB, DIM = 50, 16


def _table():
    g = torch.Generator().manual_seed(7)
    return torch.randn(VOCAB, DIM, generator=g)


def test_perfect_prediction_high_acc():
    table = _table()
    ids = torch.tensor([[3, 11, 42, 7]])
    pred = table[ids] * 5.0  # misma dirección, mayor norma → logit propio domina
    loss, ce, acc = imitator_ce_loss(pred, ids, table)
    assert acc.item() == 1.0
    assert ce.item() < 1.0


def test_padding_ignored():
    table = _table()
    ids = torch.tensor([[3, 11, IGNORE_INDEX, IGNORE_INDEX]])
    pred_good = table[torch.tensor([[3, 11, 0, 0]])] * 5.0
    # Las posiciones de padding no deben afectar ni a la CE ni a la accuracy.
    loss_a, _, acc_a = imitator_ce_loss(pred_good, ids, table)
    pred_bad_pad = pred_good.clone()
    pred_bad_pad[:, 2:] = -pred_bad_pad[:, 2:]
    loss_b, _, acc_b = imitator_ce_loss(pred_bad_pad, ids, table)
    assert torch.allclose(loss_a, loss_b)
    assert acc_a.item() == acc_b.item() == 1.0


def test_length_alignment():
    table = _table()
    ids = torch.tensor([[3, 11]])
    pred = table[torch.tensor([[3, 11, 5, 9]])]  # el modelo emite más tokens que el target
    loss, _, acc = imitator_ce_loss(pred, ids, table)
    assert acc.item() == 1.0


def test_gradient_flows_to_pred_not_table():
    table = _table()
    ids = torch.tensor([[3, 11, 42, 7]])
    pred = torch.randn(1, 4, DIM, requires_grad=True)
    loss, _, _ = imitator_ce_loss(pred, ids, table)
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
    assert table.grad is None


def test_collapsed_prediction_is_penalized():
    """La predicción degenerada (misma salida para todo) debe tener CE peor que la correcta."""
    table = _table()
    ids = torch.tensor([[3, 11, 42, 7], [29, 1, 14, 33]])
    mean_pred = table[ids].mean(dim=(0, 1), keepdim=True).expand(2, 4, DIM)
    correct_pred = table[ids] * 5.0
    loss_mean, _, _ = imitator_ce_loss(mean_pred, ids, table)
    loss_correct, _, _ = imitator_ce_loss(correct_pred, ids, table)
    assert loss_correct.item() < loss_mean.item()


def test_collate_fn_includes_token_ids():
    _cspec = importlib.util.spec_from_file_location(
        "components", _ROOT / "src/mslm/dataloader/components.py")
    _c = importlib.util.module_from_spec(_cspec)
    _cspec.loader.exec_module(_c)

    batch = [
        (torch.randn(5, 4, 2), torch.randn(3, DIM), torch.tensor([3, 11, 42])),
        (torch.randn(7, 4, 2), torch.randn(2, DIM), torch.tensor([29, 1])),
    ]
    out = _c.collate_fn(batch)
    assert len(out) == 5
    kp, fmask, emb, emask, ids = out
    assert ids.shape == (2, 3)
    assert ids[1, 2].item() == IGNORE_INDEX
    assert ids[0].tolist() == [3, 11, 42]

    # Sin token_ids (tercer slot None) el contrato anterior se mantiene.
    batch_legacy = [
        (torch.randn(5, 4, 2), torch.randn(3, DIM), None),
        (torch.randn(7, 4, 2), torch.randn(2, DIM), None),
    ]
    assert len(_c.collate_fn(batch_legacy)) == 4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK {name}")
    print("Todos los tests pasaron.")
