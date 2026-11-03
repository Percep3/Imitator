"""Dataset2 audit — Level-A visual/computational integrity (CPU only, read-only).

Reads dataset_v6_unsloth.hdf5 with h5py mode "r" and probes raw MP4s with ffprobe
(container/stream metadata + packet count; no frame decoding). Writes:
  _work/visual_integrity_per_clip.csv, _work/visual_stats.json

Run (no h5py in Sign-env on this host; VQ_RL venv has h5py 3.16 / numpy 2.3):
  /shared/Code/nakato/projects/VQ_RL/.venv/bin/python \
      artifacts/dataset2_audit/scripts/visual_integrity.py

Layouts (see 02_visual.md):
  134 = rtmlib to_openpose(COCO-WholeBody 133 + synthetic neck):
        body18 0-17 | feet 18-23 | face68 24-91 | lhand21 92-112 | rhand21 113-133
  111 = src/mslm/dataloader/data_augmentation.py:142-151 remove_keypoints:
        134[0..6] + 134[28:92] + 134[93:113] + 134[113:133]
"""
import csv
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[3]
# load integrity_filter.py by path: the package __init__ imports torch (absent in this venv)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("integrity_filter", REPO / "src/mslm/dataloader/integrity_filter.py")
_mod = importlib.util.module_from_spec(_spec); sys.modules["integrity_filter"] = _mod; _spec.loader.exec_module(_mod)
ClipRecord, filter_clip_records = _mod.ClipRecord, _mod.filter_clip_records

H5 = Path("/shared/Code/Sign-AI/data/processed/dataset_v6_unsloth.hdf5")
MANIFEST = Path("/shared/Code/Sign-AI/data/processed/dataset2_manifest_v125.json")
VIDEOS = Path("/shared/Code/Sign-AI/data/raw/dataset2/videos")
WORK = REPO / "artifacts/dataset2_audit/_work"
TRUNC_RATIO = 0.9  # scripts/data/build_dataset2_manifest.py:31
CAP = 250          # scripts/data/build_dataset2_h5.py:271 default --max-frames

# index map of the 111 layout back into the 134 layout
IDX111 = list(range(7)) + list(range(28, 92)) + list(range(93, 113)) + list(range(113, 133))
assert len(IDX111) == 111
REGIONS111 = {"pose": slice(0, 7), "face": slice(7, 71), "lhand": slice(71, 91), "rhand": slice(91, 111)}
REGIONS134 = {"body": slice(0, 18), "feet": slice(18, 24), "face": slice(24, 92),
              "lhand": slice(92, 113), "rhand": slice(113, 134)}
NECK, RSHO, LSHO = 1, 2, 5  # same indices in 134 and 111

# thresholds (in shoulder widths, per consecutive-frame step) — heuristic, documented in 02_visual.md
JUMP_BODY = 0.5   # neck moving >0.5 shoulder widths in one frame
SCALE_JUMP = 0.3  # shoulder width changing >30% in one frame
RETURN_TOL, RETURN_WIN = 0.2, 30
SWITCH_FRAC = 0.05  # >5% of steps are body jumps -> identity-switch suspect


def ffprobe(vid):
    p = VIDEOS / f"{vid}.mp4"
    if not p.exists():
        return vid, {"exists": False}
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
           "stream=width,height,nb_frames,nb_read_packets,avg_frame_rate,r_frame_rate,duration:format=duration",
           "-of", "json", str(p)]
    try:
        j = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout)
        s, fmt = j["streams"][0], j.get("format", {})
        num, den = (s.get("avg_frame_rate") or "0/1").split("/")
        return vid, {"exists": True, "width": int(s["width"]), "height": int(s["height"]),
                     "nb_frames": int(s.get("nb_frames", 0) or 0),
                     "nb_packets": int(s.get("nb_read_packets", 0) or 0),
                     "fps": float(num) / float(den) if float(den) else float("nan"),
                     "r_frame_rate": s.get("r_frame_rate"),
                     "duration": float(s.get("duration") or fmt.get("duration") or "nan")}
    except Exception as e:  # unreadable video is itself a finding
        return vid, {"exists": True, "probe_error": repr(e)}


def kp_stats(kp, w, h):
    """kp: (T,134,2) raw pixels."""
    T = kp.shape[0]
    finite = np.isfinite(kp).all(-1)                       # (T,134)
    zero = (kp == 0).all(-1)
    out = np.zeros_like(zero)
    if w and h:
        out = (kp[..., 0] < 0) | (kp[..., 0] >= w) | (kp[..., 1] < 0) | (kp[..., 1] >= h)
    bad = ~finite | zero | out                              # "missing" proxy, no confidence stored
    r = {"nan": int(np.isnan(kp).sum()), "inf": int(np.isinf(kp).sum()),
         "zero_points": int(zero.sum()), "frames_all_zero": int(zero.all(1).sum()),
         "frames_all_bad": int(bad.all(1).sum())}
    b111 = bad[:, IDX111]
    for name, sl in REGIONS111.items():
        r[f"miss111_{name}"] = float(b111[:, sl].mean())
    for name, sl in REGIONS134.items():
        r[f"out134_{name}"] = float(out[:, sl].mean())
    k = kp[:, IDX111].astype(np.float64)
    sw = np.linalg.norm(k[:, RSHO] - k[:, LSHO], axis=-1)
    r["shoulder_w_median_px"] = float(np.median(sw))
    r["frames_degenerate_shoulders"] = int((sw < 1.0).sum())
    if T < 2:
        return r | {"disp_max": np.nan, "disp_p95": np.nan, "disp_p99": np.nan,
                    "neck_jump_max": np.nan, "n_body_jumps": 0, "n_scale_jumps": 0,
                    "body_jump_frac": 0.0, "n_jumps_returning": 0}
    scale = max(float(np.median(sw)), 1.0)
    d = np.linalg.norm(np.diff(k, axis=0), axis=-1) / scale        # (T-1,111) in shoulder widths
    per_frame = np.nanmedian(d, axis=1)                             # robust whole-skeleton motion
    neck = d[:, NECK]
    ds = np.abs(np.diff(sw)) / np.maximum(sw[:-1], 1.0)
    # A-B-A test: after a jump at step t, does the neck come back within RETURN_TOL of its
    # pre-jump position in the next RETURN_WIN frames? (oscillation between two people)
    neck_xy = k[:, NECK]
    jumps = np.where(neck > JUMP_BODY)[0]
    ret = sum(bool((np.linalg.norm(neck_xy[t + 2:t + 2 + RETURN_WIN] - neck_xy[t], axis=-1) / scale < RETURN_TOL).any())
              for t in jumps)
    return r | {"disp_max": float(np.nanmax(per_frame)), "disp_p95": float(np.nanpercentile(per_frame, 95)),
                "disp_p99": float(np.nanpercentile(per_frame, 99)), "neck_jump_max": float(np.nanmax(neck)),
                "n_body_jumps": int((neck > JUMP_BODY).sum()), "n_scale_jumps": int((ds > SCALE_JUMP).sum()),
                "body_jump_frac": float(len(jumps) / len(neck)), "n_jumps_returning": int(ret)}


def pct(a, qs=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], dtype=float)
    return {f"p{q}": float(np.percentile(a, q)) for q in qs} | {"n": int(a.size), "mean": float(a.mean())} if a.size else {"n": 0}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    with h5py.File(H5, "r") as f:
        g = f["dataset2"]
        keys = sorted(g["keypoints"].keys(), key=int)
        vids = {k: g["video_id"][k][0].decode() for k in keys}
        print(f"[h5] {len(keys)} clips; probing {len(set(vids.values()))} videos", flush=True)
        with ThreadPoolExecutor(10) as ex:
            probes = dict(ex.map(ffprobe, sorted(set(vids.values()))))
        rows, records = [], []
        for i, k in enumerate(keys):
            kp = g["keypoints"][k][:]
            emb = g["embeddings"][k][:] if k in g["embeddings"] else np.array([])
            pr = probes[vids[k]]
            T = kp.shape[0]
            fc = int(g["frame_count"][k][0]) if k in g["frame_count"] else 0
            tc = int(g["token_count"][k][0]) if k in g["token_count"] else 0
            ref = pr.get("nb_frames") or 0  # same header field cv2 CAP_PROP_FRAME_COUNT reads for MP4
            ratio = fc / ref if ref > 0 else float("nan")
            trunc = (not ref) or ratio < TRUNC_RATIO  # fail-safe as build_dataset2_manifest.py:37-49
            st = kp_stats(kp, pr.get("width"), pr.get("height"))
            flags = []
            if kp.ndim != 3 or kp.shape[1:] != (134, 2): flags.append("bad_shape")
            if fc != T: flags.append("frame_count_attr_mismatch")
            if st["nan"] or st["inf"]: flags.append("nan_inf")
            if st["frames_all_bad"]: flags.append("frames_no_valid_points")
            if st["frames_degenerate_shoulders"]: flags.append("degenerate_shoulders")
            if T == CAP: flags.append("hits_cap_250")
            if not pr.get("exists"): flags.append("video_missing")
            if "probe_error" in pr: flags.append("probe_error")
            if ref and ratio < TRUNC_RATIO: flags.append("frames_lt_90pct_video")
            if ref and fc > ref: flags.append("frames_gt_video")
            if pr.get("nb_packets") and ref and pr["nb_packets"] != ref: flags.append("header_vs_packets_mismatch")
            if st["n_body_jumps"]: flags.append("body_jump")
            if st["n_scale_jumps"]: flags.append("scale_jump")
            if st["body_jump_frac"] > SWITCH_FRAC and st["n_jumps_returning"] >= 0.5 * st["n_body_jumps"]:
                flags.append("identity_switch_suspect")
            if max(st["miss111_lhand"], st["miss111_rhand"]) > 0.2: flags.append("hand_offframe_gt20pct")
            rec = ClipRecord(clip_id=k, has_keypoints=True, has_label=k in g["labels"],
                             has_embeddings=k in g["embeddings"], has_token_ids=k in g["token_ids"],
                             frame_count=fc, token_count=tc, embedding_rows=emb.shape[0] if emb.size else 0,
                             truncated=bool(trunc), keypoints_have_nan_inf=bool(st["nan"] or st["inf"]),
                             embeddings_have_nan_inf=bool(emb.size and not np.isfinite(emb).all()))
            records.append(rec)
            rows.append({"clip_key": k, "video_id": vids[k], "source_group": g["source_group"][k][0].decode(),
                         "kp_shape": "x".join(map(str, kp.shape)), "frame_count_kp": T, "frame_count_attr": fc,
                         "token_count": tc, "video_nb_frames": ref, "video_nb_packets": pr.get("nb_packets"),
                         "video_fps": pr.get("fps"), "video_duration_s": pr.get("duration"),
                         "video_w": pr.get("width"), "video_h": pr.get("height"),
                         "frames_ratio_h5_over_video": ratio, **st, "flags": ";".join(flags)})
            if i % 500 == 0:
                print(f"[clip] {i}", flush=True)
    # re-apply the v125 filter (imported unchanged) and compare with stored manifest
    kept, man = filter_clip_records(records)
    stored = {m["clip_id"]: m for m in json.loads(MANIFEST.read_text())["manifest"]}
    cause = {m["clip_id"]: m["cause"] for m in man}
    for r in rows:
        r["filter_cause_recomputed"] = cause[r["clip_key"]] or ""
        r["filter_cause_stored"] = stored.get(r["clip_key"], {}).get("cause") or ""
    with open(WORK / "visual_integrity_per_clip.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    col = lambda c: [r[c] for r in rows]
    flagc = Counter(fl for r in rows for fl in r["flags"].split(";") if fl)
    disagree = [r["clip_key"] for r in rows if r["filter_cause_recomputed"] != r["filter_cause_stored"]]
    T = np.array(col("frame_count_kp"))
    stats = {
        "_note": "observed = direct read of HDF5/ffprobe; derived = computed from observed values",
        "inputs": {"h5": str(H5), "manifest": str(MANIFEST), "videos_dir": str(VIDEOS)},
        "n_clips_h5": {"observed": len(rows)},
        "n_unique_video_ids": {"observed": len(set(col("video_id")))},
        "n_unique_source_groups": {"observed": len(set(col("source_group")))},
        "n_mp4_in_videos_dir": {"observed": sum(1 for _ in VIDEOS.glob("*.mp4"))},
        "kp_shapes": {"observed": dict(Counter(col("kp_shape")).most_common(3)) | {"distinct": len(set(r["kp_shape"].split("x", 1)[1] for r in rows))}},
        "length_frames_h5": {"derived": pct(T) | {"n_eq_250": int((T == CAP).sum()), "n_ge_250": int((T >= CAP).sum()), "n_lt_8": int((T < 8).sum())}},
        "video_fps": {"observed": dict(Counter(round(x, 3) for x in col("video_fps") if x is not None).most_common(10))},
        "video_duration_s": {"derived": pct(col("video_duration_s"))},
        "video_nb_frames": {"derived": pct(col("video_nb_frames"))},
        "frames_ratio_h5_over_video": {"derived": pct(col("frames_ratio_h5_over_video"), (0, 1, 5, 10, 25, 50, 75, 90, 100)),
                                       "n_lt_0.9": int(sum(1 for x in col("frames_ratio_h5_over_video") if x < TRUNC_RATIO)),
                                       "n_lt_0.5": int(sum(1 for x in col("frames_ratio_h5_over_video") if x < 0.5)),
                                       "n_eq_1": int(sum(1 for r in rows if r["frame_count_kp"] == r["video_nb_frames"]))},
        "nan_total": {"observed": int(sum(col("nan")))}, "inf_total": {"observed": int(sum(col("inf")))},
        "zero_points_total": {"observed": int(sum(col("zero_points")))},
        "frames_all_bad_total": {"observed": int(sum(col("frames_all_bad")))},
        "missing_proxy_per_region_111": {"derived": {n: pct(col(f"miss111_{n}"), (50, 95, 99, 100)) for n in REGIONS111}},
        "offframe_per_region_134": {"derived": {n: float(np.mean(col(f"out134_{n}"))) for n in REGIONS134}},
        "disp_per_frame_median_sw": {"derived": {"max": pct(col("disp_max")), "p99": pct(col("disp_p99"))}},
        "neck_jump_max_sw": {"derived": pct(col("neck_jump_max"))},
        "clips_with_body_jump": {"derived": int(sum(1 for x in col("n_body_jumps") if x))},
        "clips_with_scale_jump": {"derived": int(sum(1 for x in col("n_scale_jumps") if x))},
        "body_jump_frac": {"derived": pct(col("body_jump_frac"), (50, 75, 90, 95, 99, 100))},
        "body_jumps_returning_ABA": {"derived": {"jumps": int(sum(col("n_body_jumps"))), "returning": int(sum(col("n_jumps_returning")))}},
        "identity_switch_suspect": {"derived": {"all": flagc["identity_switch_suspect"],
            "among_kept_v125": int(sum(1 for r in rows if "identity_switch_suspect" in r["flags"] and not r["filter_cause_recomputed"]))}},
        "kept_v125_flag_counts": {"derived": dict(Counter(fl for r in rows if not r["filter_cause_recomputed"] for fl in r["flags"].split(";") if fl))},
        "flag_counts": {"derived": dict(flagc)},
        "filter_recomputed": {"derived": {"kept": len(kept), "excluded": len(rows) - len(kept),
                                          "by_cause": dict(Counter(c for c in cause.values() if c))}},
        "filter_stored_manifest": {"observed": json.loads(MANIFEST.read_text())["summary"]},
        "filter_disagreements": {"derived": {"n": len(disagree), "clip_keys_first50": disagree[:50]}},
        "confidence_scores": {"observed": "not stored in HDF5 (only (T,134,2) xy); build_dataset2_h5.py:57 discards scores"},
    }
    (WORK / "visual_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps({k: stats[k] for k in ("filter_recomputed", "filter_stored_manifest", "filter_disagreements", "flag_counts")}, indent=1))


if __name__ == "__main__":
    main()
