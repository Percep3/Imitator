import importlib.util
import sys
import types
from pathlib import Path

import h5py
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.modules.setdefault("src", types.ModuleType("src"))
sys.modules.setdefault("src.mslm", types.ModuleType("src.mslm"))
sys.modules.setdefault("src.mslm.dataloader", types.ModuleType("src.mslm.dataloader"))
models_stub = types.ModuleType("src.mslm.models")
models_stub.__path__ = [str(_ROOT / "src/mslm/models")]
sys.modules["src.mslm.models"] = models_stub
_load("src.mslm.dataloader.data_augmentation", "src/mslm/dataloader/data_augmentation.py")
_load("src.mslm.models.components", "src/mslm/models/components/__init__.py")
_load("src.mslm.models.components.stgcn", "src/mslm/models/components/stgcn.py")
_data_mod = _load("src.mslm.dataloader.synthetic_temporal", "src/mslm/dataloader/synthetic_temporal.py")
_model_mod = _load("src.mslm.models.temporal_sign_prompt", "src/mslm/models/temporal_sign_prompt.py")

SyntheticTemporalSignDataset = _data_mod.SyntheticTemporalSignDataset
synthetic_temporal_collate = _data_mod.synthetic_temporal_collate
CIFAggregator = _model_mod.CIFAggregator


def _fixture_h5(tmp_path):
    path = tmp_path / "dataset1.hdf5"
    with h5py.File(path, "w") as f:
        g = f.create_group("dataset1")
        kg = g.create_group("keypoints")
        kg.create_dataset("0", data=torch.ones(3, 2, 2).numpy())
        kg.create_dataset("1", data=(2 * torch.ones(4, 2, 2)).numpy())
        kg.create_dataset("2", data=(3 * torch.ones(5, 2, 2)).numpy())
    return path


def test_synthetic_dataset_concatenates_clips_tokens_and_boundaries(tmp_path):
    h5_path = _fixture_h5(tmp_path)
    ds = SyntheticTemporalSignDataset(
        h5_path,
        ["0", "1", "2"],
        {"0": "hola", "1": "mundo", "2": "si"},
        {"hola": [10, 11], "mundo": [12], "si": [13, 14, 15]},
        min_clips=3,
        max_clips=3,
        min_neutral_frames=2,
        max_neutral_frames=2,
        samples_per_epoch=2,
        seed=5,
    )

    sample = ds[0]

    assert sample.boundaries.shape == (3, 2)
    for start, end in sample.boundaries.tolist():
        assert end > start
        assert torch.count_nonzero(sample.keypoints[start:end]).item() > 0
    assert sample.boundaries[1, 0].item() - sample.boundaries[0, 1].item() == 2
    assert sample.boundaries[2, 0].item() - sample.boundaries[1, 1].item() == 2

    expected = []
    for gloss in sample.glosses:
        expected.extend(ds.token_ids_by_label[gloss])
    assert sample.token_ids.tolist() == expected
    assert sample.token_spans[-1, 1].item() == len(expected)


def test_synthetic_collate_pads_frames_tokens_and_sign_metadata(tmp_path):
    h5_path = _fixture_h5(tmp_path)
    ds = SyntheticTemporalSignDataset(
        h5_path,
        ["0", "1", "2"],
        {"0": "hola", "1": "mundo", "2": "si"},
        {"hola": [10], "mundo": [12, 13], "si": [14]},
        min_clips=2,
        max_clips=3,
        min_neutral_frames=0,
        max_neutral_frames=1,
        samples_per_epoch=4,
        seed=11,
        embedding_table=torch.arange(40, dtype=torch.float32).view(20, 2),
    )

    batch = synthetic_temporal_collate([ds[0], ds[1]])

    assert batch["keypoints"].shape[0] == 2
    assert batch["frame_lengths"].tolist() == [
        len(ds[0].keypoints),
        len(ds[1].keypoints),
    ]
    assert batch["token_ids"].shape[0] == 2
    assert (batch["token_ids"] == -100).any()
    assert batch["boundaries"].shape[:2] == (2, batch["sign_counts"].max().item())
    assert batch["target_embeddings"].shape[:2] == batch["token_ids"].shape


def test_cif_boundary_targets_use_token_span_mass():
    boundaries = torch.tensor([[[0, 2], [4, 8]]])
    spans = torch.tensor([[[0, 1], [1, 3]]])

    target = CIFAggregator.boundary_targets(boundaries, frame_count=8, token_spans=spans)

    assert torch.allclose(target[0, :2], torch.tensor([0.5, 0.5]))
    assert torch.allclose(target[0, 2:4], torch.zeros(2))
    assert torch.allclose(target[0, 4:8], torch.full((4,), 0.5))
    assert torch.isclose(target.sum(), torch.tensor(3.0))


def test_cif_integrates_known_alpha_spikes_and_reports_positions():
    cif = CIFAggregator(hidden_size=2)
    features = torch.tensor(
        [
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 0.0],
                [0.0, 3.0],
                [0.0, 4.0],
            ]
        ]
    )
    alphas = torch.tensor([[0.5, 0.5, 0.0, 0.25, 0.75]])

    out = cif(features, torch.tensor([5]), alphas=alphas)

    assert out.counts.tolist() == [2]
    assert torch.allclose(out.embeddings[0, 0], torch.tensor([1.5, 0.0]))
    assert torch.allclose(out.embeddings[0, 1], torch.tensor([0.0, 3.75]))
    assert torch.allclose(out.fire_positions[0, :2], torch.tensor([0.5, 3.75]))
    assert out.padding_mask.tolist() == [[False, False]]


def test_cif_scales_alpha_to_target_lengths():
    cif = CIFAggregator(hidden_size=1)
    features = torch.ones(1, 4, 1)
    alphas = torch.full((1, 4), 0.25)

    out = cif(features, torch.tensor([4]), alphas=alphas, target_lengths=torch.tensor([2]))

    assert out.counts.tolist() == [2]
    assert torch.allclose(out.quantity, torch.tensor([2.0]))
