"""Dataset2 audit, section 6: reproducible sample for human LSA review (READ-ONLY on data).

Builds two DISTINCT components that must never be pooled as one simple random sample:
  (a) probability  -- stratified random sample of the 5,138 v125-kept clips,
                      strata = split (train/val/test) x duration tercile (short/medium/long).
                      Records N_h, n_h, inclusion probability n_h/N_h, design weight N_h/n_h.
                      The pilot (pilot=1, 20 clips) is the first-drawn clips of each stratum,
                      fixed before any content is seen; pilot clips are never replaced.
  (b) purposive    -- oversample of suspicious / edge cases (buckets below), random within
                      bucket, disjoint from (a). NOT for prevalence estimation.

Inputs (all read-only):
  _work/visual_integrity_per_clip.csv, _work/text_split_per_clip.csv (join on clip_key),
  HDF5 dataset2/labels (transcript), _work/text_split_stats.json (footage re-use pairs),
  raw mp4s under /shared/Code/Sign-AI/data/raw/dataset2/videos/, artifacts/data_eda/video_metadata.csv.
Outputs:
  artifacts/dataset2_audit/sample_manifest.csv, human_review_template.csv,
  _private_previews/pilot/*.jpg (3 small frames per pilot clip; no video copies).

Run:
  /shared/Code/nakato/projects/VQ_RL/.venv/bin/python artifacts/dataset2_audit/scripts/build_review_sample.py
Deterministic: seed=23 (one random.Random per stratum / bucket, keyed by "23:<name>").
"""
import csv
import importlib.util
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import h5py

SEED = 23
REPO = Path(__file__).resolve().parents[3]
AUD = REPO / "artifacts/dataset2_audit"
WORK = AUD / "_work"
H5 = Path("/shared/Code/Sign-AI/data/processed/dataset_v6_unsloth.hdf5")
VIDEOS = Path("/shared/Code/Sign-AI/data/raw/dataset2/videos")
VIDEO_META = REPO / "artifacts/data_eda/video_metadata.csv"
PREVIEWS = AUD / "_private_previews/pilot"

# Probability allocation: train strata ~proportional, val/test strata get a floor of 4 so
# every split is represented (unequal weights are handled by design weights).
ALLOC = {"train": (19, 19, 18), "val": (4, 4, 4), "test": (4, 4, 4)}       # sum 80
PILOT = {"train": (3, 3, 2), "val": (2, 2, 2), "test": (2, 2, 2)}          # sum 20, subset of ALLOC
TERCILES = ("short", "medium", "long")

# Purposive buckets: (name, n, reason). Order matters: a clip goes to the first bucket that draws it.
BUCKETS = [
    ("identity_switch_suspect", 5, "visual audit flag identity_switch_suspect (A-B-A neck jumps); kept clip"),
    ("dropped_frames", 4, "kept clip with keypoint/video frame ratio in [0.9,1.0): frames silently dropped"),
    ("near_filter_threshold", 4, "kept clip near a v125 threshold: ratio<0.92, fpt<2.5, tokens==2 or frames<=12"),
    ("fpt_low_tail", 3, "kept clip in bottom 2% of frames_per_token"),
    ("fpt_high_tail", 3, "kept clip in top 2% of frames_per_token"),
    ("very_short_transcript", 3, "kept clip with <=2 words"),
    ("very_long_transcript", 3, "kept clip with >40 words"),
    ("duplicated_greeting_text", 3, "kept clip whose normalized text occurs >=2 times among kept clips"),
    ("footage_reuse_cross_group", 4, "same text in different source_groups with motion cosine >=0.8 (top-2 pairs, both clips)"),
    ("v125_rejected", 4, "v125-excluded clip, one per exclusion cause"),
    ("eco_series_test", 4, "test clip from CN Sordos ecologia 'episodio' series (48% of test)"),
    ("hand_offframe_or_degenerate", 2, "kept clip with >20% hand points off-frame or degenerate shoulders"),
]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


normalize_alignment_text = _load("alignment_text", "src/mslm/dataloader/alignment_text.py").normalize_alignment_text


def rng(name):
    return random.Random(f"{SEED}:{name}")


def terciles(values):
    s = sorted(values)
    return s[len(s) // 3], s[2 * len(s) // 3]


def tercile_of(x, cuts):
    return "short" if x < cuts[0] else "medium" if x < cuts[1] else "long"


def load_clips():
    vis = {r["clip_key"]: r for r in csv.DictReader(open(WORK / "visual_integrity_per_clip.csv"))}
    txt = {r["clip_key"]: r for r in csv.DictReader(open(WORK / "text_split_per_clip.csv"))}
    assert vis.keys() == txt.keys() and len(vis) == 5600
    with h5py.File(H5, "r") as f:
        labels = {k: f["dataset2/labels"][k][0].decode() for k in vis}
    clips = {}
    for k in sorted(vis, key=int):
        v, t = vis[k], txt[k]
        assert v["video_id"] == t["video_id"] and v["source_group"] == t["source_group"]
        fc, tc = int(t["frame_count"]), int(t["token_count"])
        flags = [x for x in v["flags"].split(";") if x] + [x for x in t["text_flags"].split("|") if x]
        if t["manifest_cause"]:
            flags.append("v125_excluded:" + t["manifest_cause"])
        clips[k] = dict(
            clip_id=k, video_id=t["video_id"], source_group=t["source_group"], split=t["split"],
            kept=t["kept_v125"] == "True", cause=t["manifest_cause"],
            duration_seconds=float(v["video_duration_s"]), frame_count=fc, token_count=tc,
            word_count=int(t["word_count"]), frames_per_token=fc / tc if tc else float("inf"),
            ratio=float(v["frames_ratio_h5_over_video"]), vflags=set(v["flags"].split(";")),
            integrity_flags=";".join(flags), transcript=labels[k], norm=normalize_alignment_text(labels[k]),
            source_video_path=str(VIDEOS / f"{t['video_id']}.mp4"))
    return clips


def probability_sample(clips):
    kept = [c for c in clips.values() if c["kept"]]
    cuts = terciles([c["duration_seconds"] for c in kept])   # terciles over the kept frame
    strata = defaultdict(list)
    for c in kept:
        strata[(c["split"], tercile_of(c["duration_seconds"], cuts))].append(c)
    rows = []
    for split in ("train", "val", "test"):
        for i, ter in enumerate(TERCILES):
            pop = sorted(strata[(split, ter)], key=lambda c: int(c["clip_id"]))
            n, k = ALLOC[split][i], PILOT[split][i]
            drawn = rng(f"prob:{split}:{ter}").sample(pop, n)   # draw order = random order
            for j, c in enumerate(drawn):
                rows.append(dict(c, sampling_component="probability", sampling_stratum=f"{split}|{ter}",
                                 stratum_population=len(pop), stratum_sample_size=n,
                                 inclusion_probability=n / len(pop), design_weight=len(pop) / n,
                                 pilot=int(j < k), sampling_reason=f"stratified SRS, draw #{j + 1} in stratum"))
    return rows, cuts, {s: len(v) for s, v in strata.items()}


def purposive_sample(clips, taken, stats):
    kept = [c for c in clips.values() if c["kept"]]
    fpts = sorted(c["frames_per_token"] for c in kept)
    lo, hi = fpts[int(0.02 * len(fpts))], fpts[int(0.98 * len(fpts))]
    norm_counts = Counter(c["norm"] for c in kept)
    reuse_pairs = [p for p in stats["same_text_cross_group_motion_cos"]["top"] if p[0] >= 0.8][:2]
    reuse_ids = {x for p in reuse_pairs for x in (p[1], p[2])}
    pred = {
        "identity_switch_suspect": lambda c: c["kept"] and "identity_switch_suspect" in c["vflags"],
        "dropped_frames": lambda c: c["kept"] and 0.9 <= c["ratio"] < 1.0,
        "near_filter_threshold": lambda c: c["kept"] and (c["ratio"] < 0.92 or c["frames_per_token"] < 2.5
                                                          or c["token_count"] == 2 or c["frame_count"] <= 12),
        "fpt_low_tail": lambda c: c["kept"] and c["frames_per_token"] <= lo,
        "fpt_high_tail": lambda c: c["kept"] and c["frames_per_token"] >= hi,
        "very_short_transcript": lambda c: c["kept"] and c["word_count"] <= 2,
        "very_long_transcript": lambda c: c["kept"] and c["word_count"] > 40,
        "duplicated_greeting_text": lambda c: c["kept"] and norm_counts[c["norm"]] >= 2,
        "footage_reuse_cross_group": lambda c: c["clip_id"] in reuse_ids,
        "eco_series_test": lambda c: c["split"] == "test" and "episodio" in c["source_group"],
        "hand_offframe_or_degenerate": lambda c: c["kept"] and bool(
            {"hand_offframe_gt20pct", "degenerate_shoulders"} & c["vflags"]),
    }
    rows, overlap = [], {}
    for name, n, reason in BUCKETS:
        if name == "v125_rejected":
            by_cause = defaultdict(list)
            for c in clips.values():
                if not c["kept"] and c["clip_id"] not in taken:
                    by_cause[c["cause"]].append(c)
            picks = [rng(f"purp:{name}:{cause}").choice(by_cause[cause]) for cause in sorted(by_cause)][:n]
        else:
            elig_all = [c for c in clips.values() if pred[name](c)]
            elig = [c for c in elig_all if c["clip_id"] not in taken]
            overlap[name] = (len(elig_all), len(elig_all) - len(elig))
            if name == "footage_reuse_cross_group":
                picks = sorted(elig, key=lambda c: int(c["clip_id"]))  # take the whole top-2 pairs
            else:
                picks = rng(f"purp:{name}").sample(sorted(elig, key=lambda c: int(c["clip_id"])), min(n, len(elig)))
        for c in picks:
            taken.add(c["clip_id"])
            also = [b for b, _, _ in BUCKETS if b in pred and b != name and pred[b](c)]
            r = reason + (f" (cause={c['cause']})" if name == "v125_rejected" else "")
            if also:
                r += "; also matches: " + ",".join(also)
            rows.append(dict(c, sampling_component="purposive", sampling_stratum=name, stratum_population="",
                             stratum_sample_size="", inclusion_probability="", design_weight="", pilot=0,
                             sampling_reason=r))
    return rows, overlap, (lo, hi)


def ffprobe_decode(path):
    """Full decode count (-count_frames) + container duration/fps."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=nb_read_frames,nb_frames,r_frame_rate:format=duration",
                          "-of", "json", path], capture_output=True, text=True, timeout=300)
    if out.returncode:
        return None, out.stderr.strip()[:200]
    j = json.loads(out.stdout)
    s = j["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return dict(duration=float(j["format"]["duration"]), fps=num / den, nb_frames=int(s.get("nb_frames", -1)),
                decoded=int(s["nb_read_frames"])), ""


def pilot_check(row, meta_paths):
    p = Path(row["source_video_path"])
    probs = []
    if not p.is_file():
        return "MISSING_VIDEO"
    if f"dataset2/videos/{p.name}" not in meta_paths:
        probs.append("not_in_video_metadata_csv")
    info, err = ffprobe_decode(str(p))
    if info is None:
        return "DECODE_ERROR:" + err
    r = info["decoded"] and row["frame_count"] / info["decoded"]
    if info["decoded"] != info["nb_frames"]:
        probs.append("decoded_ne_header")
    if r < 0.9:
        probs.append("kp_frames_lt_90pct_decoded")
    elif r < 1.0:
        probs.append("kp_frames_lt_decoded")
    elif r > 1.0:
        probs.append("kp_frames_gt_decoded")
    if abs(info["duration"] - row["duration_seconds"]) > 0.1:
        probs.append("duration_mismatch_vs_visual_csv")
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    for q in (0.25, 0.5, 0.75):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{info['duration'] * q:.2f}", "-i", str(p),
                        "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "5",
                        str(PREVIEWS / f"{row['sample_id']}_clip{row['clip_id']}_{int(q * 100)}.jpg")], check=False)
    return (f"ok={not probs};dur={info['duration']:.3f}s;fps={info['fps']:.3f};decoded={info['decoded']};"
            f"header={info['nb_frames']};kp={row['frame_count']};kp/decoded={r:.3f}"
            + (";problems=" + "|".join(probs) if probs else ""))


COLS = ["sample_id", "clip_id", "video_id", "source_group", "split", "duration_seconds", "frame_count",
        "token_count", "word_count", "frames_per_token", "integrity_flags", "sampling_component",
        "sampling_stratum", "stratum_population", "stratum_sample_size", "inclusion_probability",
        "design_weight", "pilot", "sampling_reason", "source_video_path", "transcript", "review_status",
        "pilot_check"]
RUBRIC = ["video_complete", "signer_visible", "temporal_alignment", "semantic_correspondence",
          "translation_completeness", "annotation_confidence", "estimated_offset_seconds",
          "reviewer_id", "review_date", "reviewer_notes"]


def main():
    clips = load_clips()
    stats = json.load(open(WORK / "text_split_stats.json"))
    prob, cuts, strata_N = probability_sample(clips)
    taken = {r["clip_id"] for r in prob}
    purp, overlap, fpt_tails = purposive_sample(clips, set(taken), stats)
    rows = prob + purp
    assert len({r["clip_id"] for r in rows}) == len(rows), "components must be disjoint"
    for i, r in enumerate(prob, 1):
        r["sample_id"] = f"PR{i:03d}"
    for i, r in enumerate(purp, 1):
        r["sample_id"] = f"PU{i:03d}"
    meta_paths = {r["path"] for r in csv.DictReader(open(VIDEO_META))}
    for r in rows:
        r["review_status"] = "pending"
        r["frames_per_token"] = round(r["frames_per_token"], 3)
        r["duration_seconds"] = round(r["duration_seconds"], 3)
        if r["inclusion_probability"] != "":
            r["inclusion_probability"] = round(r["inclusion_probability"], 6)
            r["design_weight"] = round(r["design_weight"], 4)
        r["pilot_check"] = pilot_check(r, meta_paths) if r["pilot"] else ""
    with open(AUD / "sample_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(AUD / "human_review_template.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        # blind: no sampling_component/reason in the reviewer copy
        w.writerow(["sample_id", "clip_id", "pilot", "source_video_path", "transcript"] + RUBRIC)
        for r in rows:
            w.writerow([r["sample_id"], r["clip_id"], r["pilot"],
                        r["source_video_path"], r["transcript"]] + [""] * len(RUBRIC))
    # summary for the notes
    groups_all = {c["source_group"] for c in clips.values()}
    summary = dict(
        duration_tercile_cuts_s=cuts, fpt_tails=fpt_tails,
        strata={f"{s}|{t}": dict(N=strata_N[(s, t)], n=ALLOC[s][i], pilot=PILOT[s][i])
                for s in ("train", "val", "test") for i, t in enumerate(TERCILES)},
        purposive_buckets=Counter(r["sampling_stratum"] for r in purp),
        purposive_overlap_with_probability={k: dict(eligible=a, already_in_probability=b) for k, (a, b) in overlap.items()},
        groups_total=len(groups_all),
        groups_probability=len({r["source_group"] for r in prob}),
        groups_purposive=len({r["source_group"] for r in purp}),
        groups_any=len({r["source_group"] for r in rows}),
        groups_pilot=len({r["source_group"] for r in prob if r["pilot"]}),
        splits_probability=Counter(r["split"] for r in prob),
        pilot_checks={r["sample_id"]: r["pilot_check"] for r in rows if r["pilot"]},
    )
    print(json.dumps(summary, indent=1, default=str))


def _selfcheck():
    assert terciles([1, 2, 3, 4, 5, 6]) == (3, 5)
    assert tercile_of(2, (3, 5)) == "short" and tercile_of(3, (3, 5)) == "medium" and tercile_of(5, (3, 5)) == "long"
    assert sum(map(sum, ALLOC.values())) == 80 and sum(map(sum, PILOT.values())) == 20
    assert all(p <= a for s in ALLOC for p, a in zip(PILOT[s], ALLOC[s]))
    assert rng("x").random() == rng("x").random()


if __name__ == "__main__":
    _selfcheck()
    main()
