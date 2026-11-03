"""Read-only HDF5 provenance/grouping checks for dataset2 (audit helper).

Usage: <python-with-h5py> artifacts/dataset2_audit/scripts/h5_provenance.py
Prints JSON to stdout. Opens the HDF5 with mode "r" only.
"""
import json
import os
import re
from collections import Counter, defaultdict

import h5py

H5 = "/shared/Code/Sign-AI/data/processed/dataset_v6_unsloth.hdf5"
MAN = "/shared/Code/Sign-AI/data/processed/dataset2_manifest_v125.json"
SPL = "/shared/Code/Sign-AI/data/processed/dataset2_split_v125.json"
VID = "/shared/Code/Sign-AI/data/raw/dataset2/videos"


def s(ds):
    v = ds[0]
    return v.decode() if isinstance(v, bytes) else str(v)


out = {}
with h5py.File(H5, "r") as f:
    g = f["dataset2"]
    out["groups"] = {k: len(g[k]) for k in g.keys()}
    keys = {k: set(g[k].keys()) for k in g.keys()}
    allk = set().union(*keys.values())
    out["all_keys_contiguous_0_5599"] = allk == {str(i) for i in range(5600)}
    out["groups_missing_keys"] = {k: len(allk - v) for k, v in keys.items()}
    vid = {k: s(g["video_id"][k]) for k in g["video_id"]}
    grp = {k: s(g["source_group"][k]) for k in g["source_group"]}
    lab = {k: s(g["labels"][k]) for k in g["labels"]}
    fc = {k: int(g["frame_count"][k][0]) for k in g["frame_count"]}
    kp_t = {k: g["keypoints"][k].shape for k in g["keypoints"]}
    out["kp_shape_joint_dims"] = dict(Counter(str(v[1:]) for v in kp_t.values()))
    out["frame_count_eq_kp_T"] = sum(fc[k] == kp_t[k][0] for k in fc)
    out["frame_count_ge_250"] = {"keys<600": sum(fc[k] >= 250 for k in fc if int(k) < 600),
                                 "keys>=600": sum(fc[k] >= 250 for k in fc if int(k) >= 600),
                                 "eq250_keys<600": sum(fc[k] == 250 for k in fc if int(k) < 600),
                                 "max_keys<600": max(fc[k] for k in fc if int(k) < 600),
                                 "max_keys>=600": max(fc[k] for k in fc if int(k) >= 600)}

stems = {n[:-4] for n in os.listdir(VID) if n.endswith(".mp4") and not n.startswith("._")}
parent = lambda st: re.sub(r"_[0-9]+$", "", st)
out["video_id_unique"] = len(set(vid.values()))
out["video_id_duplicates"] = [v for v, c in Counter(vid.values()).items() if c > 1][:10]
out["video_id_in_raw"] = sum(v in stems for v in vid.values())
out["source_group_eq_parent(video_id)"] = sum(grp[k] == parent(vid[k]) for k in grp)
out["source_group_mismatch_examples"] = [(k, vid[k], grp[k]) for k in grp if grp[k] != parent(vid[k])][:5]
out["n_source_groups"] = len(set(grp.values()))
out["n_parents_raw"] = len({parent(x) for x in stems})
out["labels_empty"] = sum(not x.strip() for x in lab.values())

man = json.load(open(MAN))["manifest"]
out["manifest_keys_eq_h5_keys"] = {m["clip_id"] for m in man} == allk
spl = json.load(open(SPL))
part = {cid: p for p in ("train", "val", "test") for cid in spl[p]}
by_group = defaultdict(lambda: Counter())
for cid, p in part.items():
    by_group[grp[cid]][p] += 1
out["groups_per_split"] = {p: sum(1 for c in by_group.values() if p in c) for p in ("train", "val", "test")}
out["groups_spanning_splits"] = {g_: dict(c) for g_, c in by_group.items() if len(c) > 1}
out["test_groups"] = {g_: dict(c) for g_, c in by_group.items() if "test" in c}
out["val_groups"] = {g_: dict(c) for g_, c in by_group.items() if "val" in c}
# excluded clips: groups only present among excluded
kept = set(part)
out["groups_only_in_excluded"] = sorted({grp[k] for k in grp} - {grp[k] for k in kept})
print(json.dumps(out, indent=1, ensure_ascii=False))
