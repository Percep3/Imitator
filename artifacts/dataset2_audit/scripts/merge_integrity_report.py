"""Merge per-clip visual and text/split audit CSVs into integrity_report.csv (stdlib only).

Inputs : _work/visual_integrity_per_clip.csv, _work/text_split_per_clip.csv
Output : integrity_report.csv (5,600 rows, one per HDF5 clip)

level_A_status is a COMPUTATIONAL status only (v125 integrity filter + uncalibrated visual
heuristics from 02_visual.md). It is not linguistic certification:
  excluded_v125:<cause>  clip rejected by src/mslm/dataloader/integrity_filter.py (stored manifest cause)
  kept_warn              kept by v125 but has >=1 entry in level_A_warnings
  kept_no_warn           kept by v125 and no warning below
level_B_status is always unknown_no_timestamps (meta.csv with start/end/splits was deleted 2026-09-10).
level_C_status is always not_annotated (0 clips have human LSA review as of 2026-10-19).

Usage: /shared/Code/nakato/projects/VQ_RL/.venv/bin/python artifacts/dataset2_audit/scripts/merge_integrity_report.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIS = ROOT / "_work/visual_integrity_per_clip.csv"
TXT = ROOT / "_work/text_split_per_clip.csv"
OUT = ROOT / "integrity_report.csv"

CAP_EXT = 1400  # inferred --max-frames of the 5,000-clip extension (12 clips at T==1400, all with video > 1400 frames)
WARN_FLAGS = ("identity_switch_suspect", "hand_offframe_gt20pct", "degenerate_shoulders")

COLS = ["clip_id", "video_id", "source_group", "split", "v125_kept", "v125_exclusion_cause",
        "frame_count", "video_nb_frames", "frames_ratio_kp_over_video", "word_count", "token_count",
        "frames_per_token", "visual_flags", "text_flags", "level_A_warnings",
        "level_A_status", "level_B_status", "level_C_status"]


def warnings(v):
    flags = [f for f in v["flags"].split(";") if f]
    w = [f for f in flags if f in WARN_FLAGS]
    T, nb = int(v["frame_count_kp"]), int(v["video_nb_frames"])
    if T == CAP_EXT and nb > CAP_EXT:
        w.append("tail_cut_at_1400")
    ratio = float(v["frames_ratio_h5_over_video"])
    if ratio < 1.0:
        w.append("frames_dropped_ratio_lt1")
    return w


def row(v, t):
    assert v["video_id"] == t["video_id"] and v["source_group"] == t["source_group"], v["clip_key"]
    assert int(v["frame_count_kp"]) == int(t["frame_count"]), v["clip_key"]
    kept = t["kept_v125"] == "True"
    cause = t["manifest_cause"]
    assert kept == (cause == ""), v["clip_key"]
    w = warnings(v)
    status = f"excluded_v125:{cause}" if not kept else ("kept_warn" if w else "kept_no_warn")
    T, tok = int(t["frame_count"]), int(t["token_count"])
    return {
        "clip_id": int(v["clip_key"]), "video_id": v["video_id"], "source_group": v["source_group"],
        "split": t["split"], "v125_kept": kept, "v125_exclusion_cause": cause,
        "frame_count": T, "video_nb_frames": int(v["video_nb_frames"]),
        "frames_ratio_kp_over_video": round(float(v["frames_ratio_h5_over_video"]), 4),
        "word_count": int(t["word_count"]), "token_count": tok,
        "frames_per_token": round(T / tok, 4) if tok else "",
        "visual_flags": v["flags"].replace(";", "|"), "text_flags": t["text_flags"],
        "level_A_warnings": "|".join(w), "level_A_status": status,
        "level_B_status": "unknown_no_timestamps", "level_C_status": "not_annotated",
    }


def main():
    vis = {r["clip_key"]: r for r in csv.DictReader(open(VIS, newline="", encoding="utf-8"))}
    txt = {r["clip_key"]: r for r in csv.DictReader(open(TXT, newline="", encoding="utf-8"))}
    assert vis.keys() == txt.keys() and len(vis) == 5600
    rows = sorted((row(vis[k], txt[k]) for k in vis), key=lambda r: r["clip_id"])
    assert sum(r["v125_kept"] for r in rows) == 5138
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    from collections import Counter
    print(OUT, len(rows), dict(Counter(r["level_A_status"] for r in rows)))


def _selfcheck():
    base = {"flags": "", "frame_count_kp": "1400", "video_nb_frames": "1441", "frames_ratio_h5_over_video": "0.97"}
    assert warnings(base) == ["tail_cut_at_1400", "frames_dropped_ratio_lt1"]
    assert warnings(base | {"frame_count_kp": "250", "video_nb_frames": "250", "frames_ratio_h5_over_video": "1.0",
                            "flags": "hits_cap_250;body_jump"}) == []
    assert warnings(base | {"frame_count_kp": "10", "video_nb_frames": "10", "frames_ratio_h5_over_video": "1.0",
                            "flags": "identity_switch_suspect"}) == ["identity_switch_suspect"]


if __name__ == "__main__":
    _selfcheck()
    main()
