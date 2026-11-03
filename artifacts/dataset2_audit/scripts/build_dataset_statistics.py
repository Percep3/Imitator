"""Merge the audit's stats files into dataset_statistics.json (stdlib only, read-only on inputs).

Top-level keys: provenance / observed / derived / pending_verification.
  observed = direct reads (HDF5 via the sub-audits, ffprobe, split/manifest JSON)
  derived  = computed from observed values (percentiles, flags, filter recomputation, CIs)
  pending_verification = quantities that cannot be computed from the available data

Usage (from repo root):
  /shared/Code/nakato/projects/VQ_RL/.venv/bin/python artifacts/dataset2_audit/scripts/build_dataset_statistics.py
Run AFTER merge_integrity_report.py (it reads integrity_report.csv).
"""
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

A = Path(__file__).resolve().parents[1]
W = A / "_work"
REPO = A.parents[1]
DATA = Path("/shared/Code/Sign-AI/data/processed")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wilson_half(p, n, z=1.959964):
    """Half-width of the Wilson 95% interval (z = 1.959964)."""
    d = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d


def strip_examples(x):
    return {k: strip_examples(v) for k, v in x.items() if k != "examples"} if isinstance(x, dict) else x


def load(p):
    return json.load(open(p, encoding="utf-8"))


def main():
    vis, txt = load(W / "visual_stats.json"), load(W / "text_split_stats.json")
    h5p, exp = load(W / "01_h5_provenance.json"), load(W / "04_experiments_split_stats.json")
    ir = list(csv.DictReader(open(A / "integrity_report.csv", encoding="utf-8")))
    sm = list(csv.DictReader(open(A / "sample_manifest.csv", encoding="utf-8")))
    prob = [r for r in sm if r["sampling_component"] == "probability"]
    w = [float(r["design_weight"]) for r in prob]
    n_eff = sum(w) ** 2 / sum(x * x for x in w)

    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    hashed = [W / f for f in ("visual_stats.json", "text_split_stats.json", "01_h5_provenance.json",
                              "04_experiments_split_stats.json", "visual_integrity_per_clip.csv",
                              "text_split_per_clip.csv", "01_inventory.tsv")]
    hashed += [A / "integrity_report.csv", A / "sample_manifest.csv", A / "human_review_template.csv"]
    # big inputs: reference hashes from 01_inventory.tsv (full sha256 computed there; 2.2 GB HDF5 not re-hashed here)
    inv = {r["path"]: r["sha256"] for r in csv.DictReader(open(W / "01_inventory.tsv"), delimiter="\t") if r["sha256"]}

    kept = [r for r in ir if r["v125_kept"] == "True"]
    out = {
        "provenance": {
            "repo_commit": commit,
            "branch_at_audit": "v125-temporal-prompt-alignment",
            "audit_date": "2026-10-19",
            "python": "/shared/Code/nakato/projects/VQ_RL/.venv/bin/python (Sign-env is empty)",
            "input_artifacts_sha256_from_01_inventory": inv,
            "audit_artifacts_sha256": {str(p.relative_to(A)): sha(p) for p in hashed},
            "ai_disclosure": "Generated with AI assistance (Claude); validate with subject-matter experts.",
        },
        "observed": {
            "raw_videos": {"mp4_files": 8459, "appledouble_sidecars": 1, "parent_videos": 64,
                           "has_audio_true": 0, "source": "01_provenance.md §I; artifacts/data_eda/video_metadata.csv"},
            "meta_csv": {"exists": False, "deleted": "2026-09-10 (DATA_CLEANUP_2026-09-10.md)",
                         "historical_rows": 8459, "historical_columns": 14,
                         "had_timestamps": "start,end,splits=(text,start,end) [git show fef07a3:Sign-Dataset.ipynb]"},
            "hdf5": {"path": str(DATA / "dataset_v6_unsloth.hdf5"), "groups": h5p["groups"],
                     "keys_contiguous_0_5599": h5p["all_keys_contiguous_0_5599"],
                     "kp_shape_joint_dims": h5p["kp_shape_joint_dims"],
                     "video_id_unique": h5p["video_id_unique"], "video_id_in_raw": h5p["video_id_in_raw"],
                     "n_source_groups": h5p["n_source_groups"], "labels_empty": h5p["labels_empty"],
                     "confidence_scores": vis["confidence_scores"]["observed"],
                     "nan": vis["nan_total"]["observed"], "inf": vis["inf_total"]["observed"]},
            "video_fps": vis["video_fps"]["observed"],
            "manifest_v125": vis["filter_stored_manifest"]["observed"],
            "split_v125": txt["split_verification"],
            "groups_per_split": txt["groups_per_split"],
            "consistency_failures": txt["consistency_failures"],
        },
        "derived": {
            "length_frames_h5": vis["length_frames_h5"]["derived"],
            "video_duration_s": vis["video_duration_s"]["derived"],
            "frames_ratio_h5_over_video": vis["frames_ratio_h5_over_video"],
            "frame_count_ge_250_by_key_range": h5p["frame_count_ge_250"],
            "cap_reconciliation": {
                "T_eq_250": 12, "T_eq_250_note": "11 have video nb_frames==250, 1 has 270 (clip 5285, ratio 0.926): "
                "natural lengths, not a cap; visual flag 'hits_cap_250' is a misnomer",
                "T_eq_1400": 12, "videos_gt_1400_frames": 12, "videos_gt_1400_with_T_lt_1400": 0,
                "T_eq_1400_excluded_truncated": 7, "T_eq_1400_kept_tail_cut": 5,
                "conclusion": "5,600 build has no 250 cap; the extension behaved as --max-frames 1400 (command not recorded)"},
            "visual_flag_counts_all": vis["flag_counts"]["derived"],
            "visual_flag_counts_kept": vis["kept_v125_flag_counts"]["derived"],
            "identity_switch": {"jumps": vis["body_jumps_returning_ABA"]["derived"],
                                "suspect": vis["identity_switch_suspect"]["derived"]},
            "filter_recomputed": vis["filter_recomputed"]["derived"],
            "filter_disagreements": vis["filter_disagreements"]["derived"]["n"],
            "text_all": txt["text_all"], "text_kept": txt["text_kept"],
            "duplicates_all": txt["dups_all"], "duplicates_kept": txt["dups_kept"],
            "series_groups_per_split": txt["series_per_split"],
            "text_leakage_cross_split": strip_examples(txt["text_leakage"]),
            "id_leakage": txt["leakage_id"],
            "near_dup_motion": {k: v for k, v in txt["near_dup_motion"].items() if not isinstance(v, list)},
            "historical_clip_level_splits": exp,
            "integrity_report_level_A": dict(Counter(r["level_A_status"] for r in ir)),
            "integrity_report_level_A_warnings_kept": dict(Counter(
                x for r in kept for x in r["level_A_warnings"].split("|") if x)),
            "integrity_report_kept_no_warn_by_split": dict(Counter(r["split"] for r in kept if r["level_A_status"] == "kept_no_warn")),
            "review_sample": {
                "probability_n": len(prob), "purposive_n": sum(r["sampling_component"] == "purposive" for r in sm),
                "pilot_n": sum(r["pilot"] == "1" for r in sm), "kish_n_eff": round(n_eff, 1),
                "sum_design_weights": round(sum(w), 1),
                "video_seconds": {c: round(sum(float(r["duration_seconds"]) for r in sm if r["sampling_component"] == c), 1)
                                  for c in ("probability", "purposive")},
            },
            "wilson95_half_width_pp": {f"n={n}": {f"p={p}": round(100 * wilson_half(p, n), 1) for p in (0.5, 0.7, 0.8, 0.9)}
                                       for n in (12, 20, 76, 80, 150, 200, 300, 385)},
        },
        "pending_verification": {
            "level_B_temporal_alignment": "UNKNOWN: no per-clip timestamps (meta.csv deleted); 0 clips annotated",
            "level_C_semantic_correspondence": "UNKNOWN: 0 of 5,600 clips reviewed by LSA-competent annotators",
            "subtitle_source": "UNKNOWN (manual captions vs auto captions vs ASR)",
            "signer_identity": "UNKNOWN (no signer metadata in HDF5)",
            "identity_switch_flag_precision": "UNKNOWN (heuristic; 1 clip visually checked)",
            "misaligned_3088_claim": "UNSUPPORTED: config comment only, no procedure/denominator/clip list",
            "extension_command_line": "UNKNOWN (--max-frames 1400 inferred from data)",
            "text_ctx_recomputation": "not recomputed (needs model forward pass)",
        },
    }
    p = A / "dataset_statistics.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(p)


if __name__ == "__main__":
    assert abs(wilson_half(0.5, 80) - 0.1069) < 1e-3  # self-check
    main()
