"""Read-only physical inventory of dataset2 artifacts (audit helper, stdlib only).

Usage: python artifacts/dataset2_audit/scripts/inventory.py > artifacts/dataset2_audit/_work/01_inventory.tsv
Files <2 GiB: full sha256. Larger: full sha256 too only if --full-big, else sha256 of first 64 MiB (labelled).
Directories: file count, total bytes and a listing fingerprint = sha256 of sorted "name\tsize\n" lines
(content of the files is NOT hashed; label 'listing_fp').
"""
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

D = "/shared/Code/Sign-AI/data"
R = "/shared/Code/Sign-AI/Sign-chris"
FILES = [
    (f"{D}/processed/dataset_v6_unsloth.hdf5", "build_dataset2_h5.py (+add_token_ids_h5.py, backfill, rebuild_alignment_targets_v125)"),
    (f"{D}/processed/dataset2_manifest_v125.json", "build_dataset2_manifest.py"),
    (f"{D}/processed/dataset2_split_v125.json", "build_dataset2_split.py"),
    (f"{D}/processed/v125_artifacts.json", "v125 artifact freeze (c4113c3)"),
    (f"{D}/processed/gemma3n_embed_table.pt", "Gemma-3n input embedding table"),
    (f"{D}/processed/v125_gemma_oracle.json", "v125 oracle"),
    (f"{D}/processed/v125_anisotropy_gate.json", "v125 PCA gate"),
    (f"{D}/processed/v125_anisotropy_gate.transform.pt", "v125 PCA gate"),
    (f"{D}/processed/ds2_smoke.hdf5", "unknown (smoke build, 2026-06-09)"),
    (f"{D}/raw/dataset2/meta.csv", "original metadata (deleted 2026-09-10 per DATA_CLEANUP)"),
    (f"{D}/raw/dataset2/keypoints.h5", "legacy keypoints (deleted 2026-09-10 per DATA_CLEANUP)"),
    (f"{D}/processed/dataset_v6_unsloth.hdf5.bak_pre250fix", "backup before v118 250-cap fix"),
    (f"{R}/artifacts/data_eda/video_metadata.csv", "scripts/data/data_eda.py (e68c7be)"),
    (f"{R}/artifacts/data_eda/dataset_summary.csv", "scripts/data/data_eda.py (e68c7be)"),
    (f"{R}/artifacts/data_eda/eda_summary.json", "scripts/data/data_eda.py (e68c7be)"),
    (f"{R}/artifacts/data_eda/srt_summary.csv", "scripts/data/data_eda.py (e68c7be)"),
]
DIRS = [(f"{D}/raw/dataset2/videos", "original segmented clips (provenance pipeline not in repo)")]
TWO_GIB, HEAD = 2 * 1024**3, 64 * 1024**2


def sha(path, limit=None):
    h, left = hashlib.sha256(), limit
    with open(path, "rb") as f:
        while chunk := f.read(min(1 << 20, left) if left else 1 << 20):
            h.update(chunk)
            if left:
                left -= len(chunk)
                if left <= 0:
                    break
    return h.hexdigest()


def mtime(p):
    return datetime.fromtimestamp(os.stat(p).st_mtime, timezone.utc).astimezone().isoformat(timespec="seconds")


full_big = "--full-big" in sys.argv
print("path\texists\tsize_bytes\tmtime\tsha256\thash_kind\tproduced_by\tn_files")
for p, by in FILES:
    if not os.path.exists(p):
        print(f"{p}\tFalse\t\t\t\t\t{by}\t")
        continue
    size = os.path.getsize(p)
    if size < TWO_GIB or full_big:
        digest, kind = sha(p), "full"
    else:
        digest, kind = sha(p, HEAD), "first64MiB"
    print(f"{p}\tTrue\t{size}\t{mtime(p)}\t{digest}\t{kind}\t{by}\t")
for p, by in DIRS:
    if not os.path.isdir(p):
        print(f"{p}\tFalse\t\t\t\t\t{by}\t")
        continue
    entries = sorted((e.name, e.stat().st_size) for e in os.scandir(p) if e.is_file())
    fp = hashlib.sha256("".join(f"{n}\t{s}\n" for n, s in entries).encode()).hexdigest()
    print(f"{p}\tTrue\t{sum(s for _, s in entries)}\t{mtime(p)}\t{fp}\tlisting_fp\t{by}\t{len(entries)}")
