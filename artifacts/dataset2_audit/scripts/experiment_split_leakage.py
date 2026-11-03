"""Reconstruct the clip-level random splits used by v116-v119 and measure
data conditions relevant to each experiment (read-only, CPU, no training).

Reproduces KeypointDataset.processData ordering (h5py key iteration of
dataset2/embeddings) + random_split([0.8, 0.2], torch.Generator().manual_seed(42))
as in src/mslm/dataloader/keypoint_dataset.py:111-112.

CAVEATS (see 04_experiments_impact.md):
- The current HDF5 is the 5600-clip v125 state. The 600-clip file used by
  v116-v118e no longer exists; the "600" scenario assumes keys 0..599 are the
  same clips (contested: ctc_v119_subset1k.toml header says otherwise).
- torch here is 2.13 (VQ_RL venv), historical runs used torch 2.8; CPU randperm
  with a fixed seed is expected but not guaranteed to be identical.

Run: /shared/Code/nakato/projects/VQ_RL/.venv/bin/python \
       artifacts/dataset2_audit/scripts/experiment_split_leakage.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import random_split

H5 = "/shared/Code/Sign-AI/data/processed/dataset_v6_unsloth.hdf5"
OUT = Path(__file__).resolve().parents[1] / "_work" / "04_experiments_split_stats.json"
_PUNCT_RE = re.compile(r"[^\w\sáéíóúñü]", re.UNICODE)  # copy of vocab.tokenize


def tokenize(t):
    return _PUNCT_RE.sub("", t.lower()).split()


def load():
    rows = []
    with h5py.File(H5, "r") as f:
        g = f["dataset2"]
        groups = sorted(g.keys())
        sg = g["source_group"] if "source_group" in g else None
        vid = g["video_id"] if "video_id" in g else None
        for k in g["embeddings"].keys():  # same iteration as processData
            if k not in g["keypoints"]:
                continue
            lab = g["labels"][k][0]
            lab = lab.decode() if isinstance(lab, bytes) else str(lab)
            rows.append(dict(
                key=k, frames=int(g["keypoints"][k].shape[0]), label=lab,
                words=len(tokenize(lab)),
                group=(sg[k][0].decode() if sg is not None and k in sg else None),
                video=(vid[k][0].decode() if vid is not None and k in vid else None)))
    return groups, rows


def split(n):
    tr, va = random_split(range(n), [0.8, 1 - 0.8],  # exact expr of split_dataset(train_ratio)
                           generator=torch.Generator().manual_seed(42))
    return list(tr.indices), list(va.indices)


def scenario(rows, name):
    tr_i, va_i = split(len(rows))
    tr = [rows[i] for i in tr_i]; va = [rows[i] for i in va_i]
    tr_lab = Counter(r["label"].strip().lower() for r in tr)
    va_lab = Counter(r["label"].strip().lower() for r in va)
    dup_val_in_train = sum(1 for r in va if r["label"].strip().lower() in tr_lab)
    dup_within_val = sum(c for c in va_lab.values() if c > 1)
    vocab = set(w for r in tr for w in tokenize(r["label"]))
    va_tok = [w for r in va for w in tokenize(r["label"])]
    oov = sum(1 for w in va_tok if w not in vocab)
    tr_g = set(r["group"] for r in tr if r["group"]); va_g = [r["group"] for r in va if r["group"]]
    fr = np.array([r["frames"] for r in rows]); wd = np.array([max(r["words"], 1) for r in rows])
    fpw = fr / wd
    return {
        "scenario": name, "n": len(rows), "n_train": len(tr), "n_val": len(va),
        "val_exact_label_in_train": dup_val_in_train,
        "val_clips_with_label_repeated_within_val": dup_within_val,
        "val_tokens": len(va_tok), "val_oov_tokens_vs_train_vocab": oov,
        "val_oov_rate": round(oov / max(len(va_tok), 1), 4), "train_vocab_size": len(vocab),
        "groups_known": bool(va_g), "n_groups_total": len(set(r["group"] for r in rows if r["group"])),
        "n_frames_gt_250": int(sum(1 for r in rows if r["frames"] > 250)),
        "frames_mean": round(float(np.mean([r["frames"] for r in rows])), 1),
        "frames_mean_of_gt_250": round(float(np.mean([r["frames"] for r in rows if r["frames"] > 250] or [0])), 1),
        "words_mean_gt_250_vs_rest": [round(float(np.mean([r["words"] for r in rows if r["frames"] > 250] or [0])), 1),
                                      round(float(np.mean([r["words"] for r in rows if r["frames"] <= 250] or [0])), 1)],
        "val_clips_whose_source_group_in_train": sum(1 for g in va_g if g in tr_g) if va_g else None,
        "n_val_groups": len(set(va_g)) if va_g else None,
        "frames": {"median": float(np.median(fr)), "p95": float(np.percentile(fr, 95)), "max": int(fr.max()),
                   "n_eq_250": int((fr == 250).sum()), "n_gt_768": int((fr > 768).sum()),
                   "n_gt_1024": int((fr > 1024).sum())},
        "words": {"median": float(np.median(wd)), "p95": float(np.percentile(wd, 95)), "max": int(wd.max())},
        "frames_per_word": {"median": round(float(np.median(fpw)), 2), "p5": round(float(np.percentile(fpw, 5)), 2),
                            "n_lt_4": int((fpw < 4).sum()), "n_lt_8": int((fpw < 8).sum())},
    }


def main():
    groups, rows = load()
    by_key = {r["key"]: r for r in rows}
    res = {"h5": H5, "hdf5_groups": groups, "n_rows": len(rows), "torch": torch.__version__}
    res["full_5600_v118f_v119_baseline"] = scenario(rows, "5600 no filter (v118f, v119 baseline)")
    filt = [r for r in rows if r["frames"] >= 16 and r["words"] > 0 and r["frames"] // 4 >= r["words"]]
    res["filtered_v119_levers"] = scenario(filt, "v119 min_frames=16 + frames//4>=words")
    sub600 = [r for r in rows if r["key"].isdigit() and int(r["key"]) < 600]
    res["keys_0_599_assumed_v116_v118e"] = scenario(sub600, "keys 0..599 (assumed = historical 600)")
    s = res["keys_0_599_assumed_v116_v118e"]
    s["caveat"] = "frame counts are CURRENT (post-250fix re-extraction); historical v116/v117/v118 runs1-3 saw <=250"
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
