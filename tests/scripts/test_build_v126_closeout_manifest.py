import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audits" / "build_v126_closeout_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_v126_closeout_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _make_artifact_dir(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "v126_closeout"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "ablation_summary.json").write_text(json.dumps({"n_runs": 24}), encoding="utf-8")
    (artifact_dir / "config.json").write_text(json.dumps({"seeds": [23, 42, 101]}), encoding="utf-8")
    return artifact_dir


def _make_checkpoints_and_inputs(tmp_path):
    ckpt_dir = tmp_path / "outputs_external" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    checkpoint_paths = []
    for i in range(2):  # 2 stand-ins for the real 24-checkpoint list
        path = ckpt_dir / f"checkpoint_{i}.pt"
        path.write_bytes(f"weights-{i}".encode())
        checkpoint_paths.append(path)
    dataset_path = tmp_path / "outputs_external" / "dataset.hdf5"
    dataset_path.write_bytes(b"fake-dataset-bytes")
    external_inputs = {"dataset_h5": dataset_path}
    return checkpoint_paths, external_inputs


def test_build_manifest_hashes_artifact_files_checkpoints_and_external_inputs(tmp_path):
    artifact_dir = _make_artifact_dir(tmp_path)
    checkpoint_paths, external_inputs = _make_checkpoints_and_inputs(tmp_path)
    manifest = MODULE.build_manifest(
        tmp_path, artifact_dir, checkpoint_paths=checkpoint_paths,
        external_inputs=external_inputs, commit_hash="abc123",
    )
    assert manifest["schema_version"] == 1
    assert manifest["commit_hash"] == "abc123"
    paths = {entry["path"] for entry in manifest["files"]}
    assert "artifacts/v126_closeout/ablation_summary.json" in paths
    assert len(manifest["checkpoints"]) == 2
    assert manifest["external_inputs"]["dataset_h5"]["sha256"] == hashlib.sha256(
        external_inputs["dataset_h5"].read_bytes()
    ).hexdigest()
    for entry in manifest["files"] + manifest["checkpoints"]:
        full_path = tmp_path / entry["path"]
        assert entry["sha256"] == hashlib.sha256(full_path.read_bytes()).hexdigest()
        assert entry["bytes"] == full_path.stat().st_size


def test_check_manifest_passes_when_nothing_changed(tmp_path):
    artifact_dir = _make_artifact_dir(tmp_path)
    checkpoint_paths, external_inputs = _make_checkpoints_and_inputs(tmp_path)
    manifest = MODULE.build_manifest(
        tmp_path, artifact_dir, checkpoint_paths=checkpoint_paths,
        external_inputs=external_inputs, commit_hash="abc123",
    )
    assert MODULE.check_manifest(manifest, tmp_path) == []


def test_check_manifest_fails_on_missing_file(tmp_path):
    artifact_dir = _make_artifact_dir(tmp_path)
    checkpoint_paths, external_inputs = _make_checkpoints_and_inputs(tmp_path)
    manifest = MODULE.build_manifest(
        tmp_path, artifact_dir, checkpoint_paths=checkpoint_paths,
        external_inputs=external_inputs, commit_hash="abc123",
    )
    (artifact_dir / "config.json").unlink()
    problems = MODULE.check_manifest(manifest, tmp_path)
    assert any("missing" in p for p in problems)


def test_check_manifest_fails_on_missing_checkpoint(tmp_path):
    artifact_dir = _make_artifact_dir(tmp_path)
    checkpoint_paths, external_inputs = _make_checkpoints_and_inputs(tmp_path)
    manifest = MODULE.build_manifest(
        tmp_path, artifact_dir, checkpoint_paths=checkpoint_paths,
        external_inputs=external_inputs, commit_hash="abc123",
    )
    checkpoint_paths[0].unlink()
    problems = MODULE.check_manifest(manifest, tmp_path)
    assert any("missing" in p for p in problems)


def test_check_manifest_fails_on_hash_mismatch(tmp_path):
    artifact_dir = _make_artifact_dir(tmp_path)
    checkpoint_paths, external_inputs = _make_checkpoints_and_inputs(tmp_path)
    manifest = MODULE.build_manifest(
        tmp_path, artifact_dir, checkpoint_paths=checkpoint_paths,
        external_inputs=external_inputs, commit_hash="abc123",
    )
    (artifact_dir / "config.json").write_text(json.dumps({"seeds": [1]}), encoding="utf-8")
    problems = MODULE.check_manifest(manifest, tmp_path)
    assert any("hash mismatch" in p for p in problems)


def test_build_manifest_and_check_manifest_handle_paths_outside_root(tmp_path):
    # Checkpoints/external inputs live in a sibling dir of the repo root in
    # production (../outputs/..., /shared/Code/Sign-AI/data/...), not nested
    # under it. relative_to() would raise ValueError for these; build_manifest
    # must not crash, and check_manifest must still detect OK vs hash-mismatch.
    root = tmp_path / "repo"
    artifact_dir = root / "artifacts" / "v126_closeout"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "config.json").write_text(json.dumps({"seeds": [23]}), encoding="utf-8")

    sibling_dir = tmp_path / "outside_root" / "checkpoints"  # NOT under root
    sibling_dir.mkdir(parents=True)
    out_of_root_ckpt = sibling_dir / "checkpoint_best.pt"
    out_of_root_ckpt.write_bytes(b"weights-out-of-root")

    manifest = MODULE.build_manifest(
        root, artifact_dir, checkpoint_paths=[out_of_root_ckpt],
        external_inputs={}, commit_hash="abc123",
    )
    # Out-of-root path is recorded as an absolute string, not relative_to'd.
    assert manifest["checkpoints"][0]["path"] == str(out_of_root_ckpt)
    assert manifest["checkpoints"][0]["path"].startswith("/")

    # OK case: nothing changed.
    assert MODULE.check_manifest(manifest, root) == []

    # Hash-mismatch case: tamper with the out-of-root checkpoint.
    out_of_root_ckpt.write_bytes(b"weights-out-of-root-TAMPERED")
    problems = MODULE.check_manifest(manifest, root)
    assert any("hash mismatch" in p for p in problems)


def test_build_manifest_raises_when_artifact_dir_empty(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "v126_closeout"
    artifact_dir.mkdir(parents=True)
    checkpoint_paths, external_inputs = _make_checkpoints_and_inputs(tmp_path)
    import pytest
    with pytest.raises(FileNotFoundError):
        MODULE.build_manifest(
            tmp_path, artifact_dir, checkpoint_paths=checkpoint_paths,
            external_inputs=external_inputs, commit_hash="abc123",
        )
