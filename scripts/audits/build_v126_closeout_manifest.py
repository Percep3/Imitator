"""Build/check a SHA-256 manifest of artifacts/v126_closeout/ for the Etapa 2 ablation closeout."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_hash(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _hash_entry(path: Path, root: Path) -> dict:
    try:
        path_str = str(path.relative_to(root))
    except ValueError:
        # Checkpoints and external inputs live in a sibling directory of the repo
        # root (e.g. ../outputs/...), not nested under it — record the absolute
        # path instead. `root / entry["path"]` still resolves correctly later
        # because pathlib's `/` discards the left side for an absolute RHS.
        path_str = str(path)
    return {
        "path": path_str,
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
    }


def build_manifest(
    root: Path,
    artifact_dir: Path,
    checkpoint_paths: list[Path],
    external_inputs: dict[str, Path],
    commit_hash: str | None = None,
) -> dict:
    files = sorted(p for p in artifact_dir.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no files found under {artifact_dir}")
    return {
        "schema_version": 1,
        "commit_hash": commit_hash if commit_hash is not None else git_commit_hash(root),
        "artifact_dir": str(artifact_dir.relative_to(root)),
        "files": [_hash_entry(path, root) for path in files],
        "checkpoints": [_hash_entry(path, root) for path in checkpoint_paths],
        "external_inputs": {
            name: _hash_entry(path, root) for name, path in external_inputs.items()
        },
    }


def _check_entries(entries: list[dict], root: Path) -> list[str]:
    problems = []
    for entry in entries:
        path = root / entry["path"]
        if not path.is_file():
            problems.append(f"missing: {entry['path']}")
            continue
        actual_hash = sha256_of(path)
        if actual_hash != entry["sha256"]:
            problems.append(f"hash mismatch: {entry['path']} (expected {entry['sha256']}, got {actual_hash})")
        if path.stat().st_size != entry["bytes"]:
            problems.append(f"size mismatch: {entry['path']}")
    return problems


def check_manifest(manifest: dict, root: Path) -> list[str]:
    problems = _check_entries(manifest["files"], root)
    problems += _check_entries(manifest["checkpoints"], root)
    problems += _check_entries(list(manifest["external_inputs"].values()), root)
    return problems


# Checkpoints (one per ablation run) and external inputs live outside artifact_dir —
# the global constraint excludes heavy checkpoints from the versioned artifact bundle,
# but the spec still requires their hashes recorded for reproducibility.
def default_checkpoint_paths(registry_dir: Path, out_root: Path) -> list[Path]:
    paths = []
    for registry_path in sorted(registry_dir.glob("*.json")):
        run_name = json.loads(registry_path.read_text(encoding="utf-8"))["run_name"]
        paths.append(out_root / f"diag_{run_name}" / "checkpoint_best.pt")
    return paths


# Matches the --h5/--embedding-table argparse defaults in scripts/train/train_temporal_v126.py
# so the manifest hashes the actual files the training script reads from.
DEFAULT_EXTERNAL_INPUTS = {
    "dataset_h5": Path("/shared/Code/Sign-AI/data/processed/dataset1_isolated_v122.hdf5"),
    "embedding_table": Path("/shared/Code/Sign-AI/data/processed/gemma3n_embed_table.pt"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "check"])
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/v126_closeout"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/v126_closeout/manifest.json"))
    parser.add_argument("--registry-dir", type=Path, default=Path("artifacts/v126_closeout/ablation_etapa2/registry"))
    parser.add_argument("--out-root", type=Path, default=Path("../outputs/v126_temporal"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else root / args.artifact_dir
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest

    if args.command == "build":
        checkpoint_paths = default_checkpoint_paths(
            args.registry_dir if args.registry_dir.is_absolute() else root / args.registry_dir,
            args.out_root if args.out_root.is_absolute() else root / args.out_root,
        )
        external_inputs = {
            name: (path if path.is_absolute() else root / path)
            for name, path in DEFAULT_EXTERNAL_INPUTS.items()
        }
        manifest = build_manifest(root, artifact_dir, checkpoint_paths, external_inputs)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"wrote": str(manifest_path), "files": len(manifest["files"])}, indent=2))
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        problems = check_manifest(manifest, root)
        if problems:
            print(json.dumps({"ok": False, "problems": problems}, indent=2))
            raise SystemExit(1)
        print(json.dumps({"ok": True, "files_checked": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
