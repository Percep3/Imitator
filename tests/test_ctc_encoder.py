"""Tests para CTCEncoder (v119 — reemplaza el cuello de botella de K tokens fijos
de Imitator por una clasificación por-frame apta para CTC).

Carga directa de archivos (igual que test_contrastive_aligner.py) para evitar el
import circular real de src/mslm/models/__init__.py <-> src/mslm/utils/__init__.py
(models/__init__ importa utils.early_stopping, utils/__init__ importa setup_train,
que importa `from src.mslm.models import Imitator` -- si `models` se carga primero
ese símbolo todavía no existe). No es un problema introducido por v119; es el mismo
workaround que ya usan test_contrastive_aligner.py y test_loss_infonce.py.
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, file_path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_package(name, dir_path):
    spec = importlib.util.spec_from_file_location(
        name, dir_path / "__init__.py", submodule_search_locations=[str(dir_path)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.modules.setdefault("src", types.ModuleType("src"))
sys.modules.setdefault("src.mslm", types.ModuleType("src.mslm"))
models_stub = types.ModuleType("src.mslm.models")
models_stub.__path__ = [str(_ROOT / "src/mslm/models")]
sys.modules["src.mslm.models"] = models_stub

_load_package("src.mslm.models.components", _ROOT / "src/mslm/models/components")
_load_module("src.mslm.models.components.stgcn", _ROOT / "src/mslm/models/components/stgcn.py")
_load_module("src.mslm.models.components.tlp", _ROOT / "src/mslm/models/components/tlp.py")
_ctc_mod = _load_module("src.mslm.models.ctc_encoder", _ROOT / "src/mslm/models/ctc_encoder.py")
CTCEncoder = _ctc_mod.CTCEncoder


N = 4  # nodos de juguete (no 111, para que el test sea rápido)
A = np.array(
    [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=np.float32,
)
B, T, V_SIZE, HIDDEN = 2, 16, 5, 8


def _make_encoder(**kwargs):
    return CTCEncoder(
        A=A, input_size=2, hidden_size=HIDDEN, nhead=2, ff_dim=16, n_layers=1,
        vocab_size=V_SIZE, encoder_dropout=0.0, multihead_dropout=0.0, **kwargs,
    )


def _make_batch(lengths=(16, 10)):
    x = torch.randn(B, T, N, 2)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, length:] = True
    return x, mask, torch.tensor(lengths)


def test_forward_single_stream_shape():
    model = _make_encoder(use_motion_stream=False, use_tlp=False)
    x, mask, lengths = _make_batch()
    log_probs, seq_lengths, aux = model(x, mask)
    assert log_probs.shape == (B, T, V_SIZE + 1)
    assert torch.equal(seq_lengths, lengths)
    assert aux == {}


def test_forward_log_probs_sum_to_one_after_exp():
    model = _make_encoder()
    x, mask, _ = _make_batch()
    log_probs, _, _ = model(x, mask)
    probs_sum = log_probs.exp().sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4)


def test_forward_with_motion_stream_shape():
    model = _make_encoder(use_motion_stream=True, use_tlp=False)
    x, mask, lengths = _make_batch()
    log_probs, seq_lengths, aux = model(x, mask)
    assert log_probs.shape == (B, T, V_SIZE + 1)
    assert torch.equal(seq_lengths, lengths)


def test_forward_with_tlp_reduces_length():
    model = _make_encoder(use_motion_stream=False, use_tlp=True)
    x, mask, lengths = _make_batch(lengths=(16, 10))
    log_probs, seq_lengths, aux = model(x, mask)
    expected = ((lengths + 1) // 2 + 1) // 2  # dos TLP en cascada
    assert torch.equal(seq_lengths, expected)
    assert log_probs.size(1) == 4  # T=16 -> 8 (tlp1) -> 4 (tlp2)
    assert "L_u" in aux and "L_p" in aux


def test_forward_with_motion_and_tlp_combined():
    model = _make_encoder(use_motion_stream=True, use_tlp=True)
    x, mask, lengths = _make_batch(lengths=(16, 10))
    log_probs, seq_lengths, aux = model(x, mask)
    expected = ((lengths + 1) // 2 + 1) // 2
    assert torch.equal(seq_lengths, expected)
    assert "L_u" in aux and "L_p" in aux


def test_gradients_flow_through_full_model():
    model = _make_encoder(use_motion_stream=True, use_tlp=False)
    x, mask, _ = _make_batch()
    log_probs, _, _ = model(x, mask)
    log_probs.sum().backward()
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()
    assert model.stgcn_static.gconv.weight.grad is not None
    assert model.stgcn_motion.gconv.weight.grad is not None
