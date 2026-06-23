"""Construye el manifiesto de integridad de dataset2 sobre el HDF5 real (v125).

Requiere que `backfill_dataset2_metadata.py` ya corrió (lee video_id,
source_group, frame_count, token_count, que no vienen del build original).
El truncamiento se detecta comparando frame_count contra max_frames=250 (el
límite duro de `build_dataset2_h5.py --max-frames`, ver memoria
"v118 truncated video bug"): un clip truncado es uno que llegó exactamente a
ese tope, señal de que el video original era más largo.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mslm.dataloader.integrity_filter import ClipRecord, filter_clip_records  # noqa: E402

TRUNCATION_FRAME_CAP = 250


def build_records(h5_path: Path) -> list[ClipRecord]:
    records = []
    with h5py.File(h5_path, "r") as f:
        g = f["dataset2"]
        clip_ids = sorted(g["keypoints"].keys(), key=int)
        for key in clip_ids:
            has_kp = key in g["keypoints"]
            has_label = key in g["labels"]
            has_emb = key in g["embeddings"]
            has_ids = "token_ids" in g and key in g["token_ids"]
            frame_count = int(g["frame_count"][key][0]) if "frame_count" in g and key in g["frame_count"] else 0
            token_count = int(g["token_count"][key][0]) if "token_count" in g and key in g["token_count"] else 0
            embedding_rows = g["embeddings"][key].shape[0] if has_emb else 0
            kp_arr = g["keypoints"][key][:] if has_kp else np.array([])
            emb_arr = g["embeddings"][key][:] if has_emb else np.array([])
            records.append(
                ClipRecord(
                    clip_id=key,
                    has_keypoints=has_kp,
                    has_label=has_label,
                    has_embeddings=has_emb,
                    has_token_ids=has_ids,
                    frame_count=frame_count,
                    token_count=token_count,
                    embedding_rows=embedding_rows,
                    truncated=frame_count >= TRUNCATION_FRAME_CAP,
                    keypoints_have_nan_inf=bool(kp_arr.size and not np.isfinite(kp_arr).all()),
                    embeddings_have_nan_inf=bool(emb_arr.size and not np.isfinite(emb_arr).all()),
                )
            )
    return records


def main(h5_path: Path, out_path: Path) -> None:
    records = build_records(h5_path)
    kept, manifest = filter_clip_records(records)
    causes = Counter(m["cause"] for m in manifest if not m["kept"])
    summary = {"total": len(manifest), "kept": len(kept), "excluded": len(manifest) - len(kept), "by_cause": dict(causes)}
    out_path.write_text(json.dumps({"summary": summary, "manifest": manifest}, indent=2, ensure_ascii=False))
    print(f"[manifest] {summary}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, default=Path("data/processed/dataset_v6_unsloth.hdf5"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/dataset2_manifest_v125.json"))
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1].parent
    h5 = args.h5 if args.h5.is_absolute() else root / args.h5
    out = args.out if args.out.is_absolute() else root / args.out
    main(h5, out)
