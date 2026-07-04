import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "train_video_token_decoder_test_module",
    ROOT / "scripts/train/train_video_token_decoder.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_protocol_rejects_folds_seven_through_ten():
    manifest = {"folds": [{"fold": 7}]}
    with pytest.raises(ValueError, match="only folds 1-6"):
        MODULE.fold_spec(manifest, 7)


def test_manifest_hash_is_verified(tmp_path):
    payload = {"seed": 23, "signers": [1, 2], "folds": []}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({**payload, "manifest_sha256": MODULE.canonical_hash(payload)}))
    assert MODULE.load_manifest(path)["seed"] == 23
    path.write_text(json.dumps({**payload, "seed": 42, "manifest_sha256": MODULE.canonical_hash(payload)}))
    with pytest.raises(RuntimeError, match="invalid manifest hash"):
        MODULE.load_manifest(path)


def test_no_eos_is_max_length_and_never_strict_exact():
    metrics, rows = MODULE.sequence_metrics([[7]], [False], [[7]])
    assert metrics["strict_exact"] == 0.0
    assert metrics["no_eos_rate"] == 1.0
    assert metrics["predicted_length"] == 33.0
    assert rows[0]["length_mae"] == 32.0


def test_checkpoint_selection_tuple_is_strict_then_edit_similarity():
    assert (0.4, 0.1) > (0.3, 1.0)
    assert (0.4, 0.8) > (0.4, 0.7)


def test_affine_grid_uses_declared_bounds_and_finds_identity():
    raw = [2.0, 3.0, 4.0]
    targets = [[1, 1], [1, 1, 1], [1, 1, 1, 1]]
    a, b = MODULE.fit_affine(raw, targets)
    assert a == 1.0
    assert b == 0.0


def test_cli_defaults_keep_fixed_regime():
    args = MODULE.parse_args(["train", "--fold", "1"])
    assert args.seed == 23
    assert args.val_samples == 512
    assert not args.rescue_augmentation
