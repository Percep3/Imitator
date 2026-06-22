"""Tests para IsolatedSignClassifier + IsolatedKeypointDataset (v120 —
reconocimiento de señas aislado sobre dataset1, 64 glosas x 50 ejemplos c/u).

Carga directa de archivos (mismo patrón que test_ctc_encoder.py) para evitar
el import circular real de src/mslm/models/__init__.py <-> src/mslm/utils/__init__.py.
"""
import importlib.util
import sys
import types
from pathlib import Path

import h5py
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


models_stub = types.ModuleType("src.mslm.models")
models_stub.__path__ = [str(_ROOT / "src/mslm/models")]
sys.modules["src.mslm.models"] = models_stub

_load_package("src.mslm.models.components", _ROOT / "src/mslm/models/components")
_load_module("src.mslm.models.components.stgcn", _ROOT / "src/mslm/models/components/stgcn.py")
_cls_mod = _load_module("src.mslm.models.isolated_classifier", _ROOT / "src/mslm/models/isolated_classifier.py")
IsolatedSignClassifier = _cls_mod.IsolatedSignClassifier

sys.modules.setdefault("src", types.ModuleType("src"))
sys.modules.setdefault("src.mslm", types.ModuleType("src.mslm"))
sys.modules.setdefault("src.mslm.dataloader", types.ModuleType("src.mslm.dataloader"))
_load_module("src.mslm.dataloader.data_augmentation", _ROOT / "src/mslm/dataloader/data_augmentation.py")
_ds_mod = _load_module(
    "src.mslm.dataloader.isolated_keypoint_dataset", _ROOT / "src/mslm/dataloader/isolated_keypoint_dataset.py"
)
IsolatedKeypointDataset = _ds_mod.IsolatedKeypointDataset
isolated_collate_fn = _ds_mod.isolated_collate_fn
list_clips = _ds_mod.list_clips


N = 4  # nodos de juguete
A = np.array(
    [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=np.float32,
)
B, T, NUM_CLASSES, HIDDEN = 2, 20, 5, 16


def _make_model(**kwargs):
    return IsolatedSignClassifier(
        A=A, input_size=2, gcn_channels=(8, 8, 8), hidden_size=HIDDEN, num_classes=NUM_CLASSES, **kwargs,
    )


# --- 1. Modelo: forward shapes ----------------------------------------------

def test_forward_shape_no_padding():
    model = _make_model()
    x = torch.randn(B, T, N, 2)
    logits = model(x)
    assert logits.shape == (B, NUM_CLASSES)


def test_forward_handles_variable_length_clips_independently():
    """Cada clip se procesa con su propia longitud real (B=1); no debe
    requerir que todos los clips de un batch compartan T."""
    model = _make_model()
    x_short = torch.randn(1, 10, N, 2)
    x_long = torch.randn(1, 30, N, 2)
    assert model(x_short).shape == (1, NUM_CLASSES)
    assert model(x_long).shape == (1, NUM_CLASSES)


def test_gradients_flow_to_gcn():
    """grad is not None NO basta: un BatchNorm2d como última capa de
    linear_hidden seguido de mean-pool sobre las MISMAS dims que esa BN
    normaliza a media cero anula el feature pooled para cualquier input,
    dando grad EXACTAMENTE 0.0 (no None) en todo el backbone y en
    classifier.weight -- bug real que pasó un test que solo chequeaba
    `is not None`. Por eso este test exige norma > 0."""
    model = _make_model()
    x = torch.randn(B, T, N, 2)
    logits = model(x)
    logits.sum().backward()
    assert model.stgcn_layers[0].gconv.weight.grad is not None
    assert torch.isfinite(model.stgcn_layers[0].gconv.weight.grad).all()
    assert model.stgcn_layers[0].gconv.weight.grad.norm().item() > 0
    assert model.classifier.weight.grad is not None
    assert model.classifier.weight.grad.norm().item() > 0


def test_pooled_features_are_not_exactly_zero_in_train_mode():
    """Firma exacta del bug: linear_hidden terminando en BatchNorm2d, seguido
    de mean-pool sobre las MISMAS dims (T,N) que esa BN normaliza a media
    cero con batch=1, da pooled features EXACTAMENTE 0.0 (no solo chicos) en
    modo train -- el modelo solo podía aprender un sesgo por clase."""
    model = _make_model()  # train() es el modo por default de nn.Module
    feats = {}

    def hook(_m, inp, _out):
        feats["val"] = inp[0].detach().clone()

    model.classifier.register_forward_hook(hook)
    model(torch.randn(1, T, N, 2))
    assert feats["val"].abs().max().item() > 1e-6


# --- 2. Dataset slim ---------------------------------------------------------

def _write_clip(group, clip_id, n_frames, label, n_keypoints=133, n_channels=2):
    rng = np.random.default_rng(0)
    group["keypoints"].create_dataset(clip_id, data=rng.random((n_frames, n_keypoints, n_channels), dtype=np.float32))
    group["labels"].create_dataset(clip_id, data=np.array([label.encode()]))


def _make_hdf5(tmp_path, clips, dataset_name="dataset1"):
    h5_path = tmp_path / "fixture.hdf5"
    with h5py.File(h5_path, "w") as f:
        g = f.require_group(dataset_name)
        g.require_group("keypoints")
        g.require_group("labels")
        for clip_id, n_frames, label in clips:
            _write_clip(g, clip_id, n_frames, label)
    return h5_path


def test_list_clips_reads_keypoints_and_labels(tmp_path):
    clips = [("0", 30, "rojo"), ("1", 40, "verde")]
    h5_path = _make_hdf5(tmp_path, clips)

    clip_ids, labels = list_clips(h5_path, dataset_name="dataset1")
    assert clip_ids == ["0", "1"]
    assert labels == ["rojo", "verde"]


def test_dataset_returns_keypoint_and_label_with_expected_shape(tmp_path):
    clips = [("0", 30, "rojo")]
    h5_path = _make_hdf5(tmp_path, clips)
    clip_ids, labels = list_clips(h5_path, dataset_name="dataset1")

    ds = IsolatedKeypointDataset(h5_path, clip_ids, labels, dataset_name="dataset1", n_keypoints=111, augment=False)
    keypoint, label = ds[0]
    assert keypoint.shape == (30, 111, 2)
    assert label == "rojo"


def test_dataset_augmentation_does_not_break_shapes(tmp_path):
    clips = [("0", 30, "rojo")] * 1
    h5_path = _make_hdf5(tmp_path, clips)
    clip_ids, labels = list_clips(h5_path, dataset_name="dataset1")

    ds = IsolatedKeypointDataset(h5_path, clip_ids, labels, dataset_name="dataset1", n_keypoints=111, augment=True)
    for _ in range(10):  # cubre las 5 augmentations posibles (elegidas al azar)
        keypoint, label = ds[0]
        assert keypoint.shape[1:] == (111, 2)
        assert keypoint.shape[0] > 0
        assert label == "rojo"


def test_collate_fn_keeps_variable_lengths_and_maps_labels(tmp_path):
    """NO debe pad-ear (ver docstring de isolated_collate_fn): devuelve la
    lista de keypoints a su longitud real, sin stack-ear en un tensor único."""
    clips = [("0", 20, "rojo"), ("1", 35, "verde")]
    h5_path = _make_hdf5(tmp_path, clips)
    clip_ids, labels = list_clips(h5_path, dataset_name="dataset1")
    ds = IsolatedKeypointDataset(h5_path, clip_ids, labels, dataset_name="dataset1", n_keypoints=111, augment=False)

    label_to_idx = {"rojo": 0, "verde": 1}
    batch = [ds[0], ds[1]]
    keypoints, target = isolated_collate_fn(batch, label_to_idx)

    assert [kp.shape[0] for kp in keypoints] == [20, 35]
    assert all(kp.shape[1:] == (111, 2) for kp in keypoints)
    assert torch.equal(target, torch.tensor([0, 1]))
