# 02 — Level-A visual/computational integrity of dataset2

Scope: spec section 4 ("Integridad visual"). CPU only, HDF5 opened `"r"`, videos only probed (ffprobe metadata + packet count, no decoding), apart from one 2-frame preview.

## Environment (VERIFIED)
- `conda env Sign-env` at `/home/nakato/miniconda3/envs/Sign-env` holds only `conda-meta/`, with no python binary. No env on the host has `cv2`. The only env with `h5py` is `/shared/Code/nakato/projects/VQ_RL/.venv` (python 3.13, h5py 3.16.0, numpy 2.3.1), and I used it without installing anything.
- `ffprobe` 6.1.1 (`/usr/bin/ffprobe`) stands in for cv2. For MP4, `nb_frames` comes from the same stream header that `cv2.CAP_PROP_FRAME_COUNT` reads (INFERRED). I also compared it with `-count_packets`: the two agree for all 5,600 clips (VERIFIED, no `header_vs_packets_mismatch` flag).

## Command
```
/shared/Code/nakato/projects/VQ_RL/.venv/bin/python artifacts/dataset2_audit/scripts/visual_integrity.py   # ~65 s, 10 ffprobe threads
```
Outputs: `_work/visual_integrity_per_clip.csv` (5,600 rows) and `_work/visual_stats.json`. All 5,600 videos were probed, so no subsampling was needed.

## Inputs located (VERIFIED)
- HDF5: `/shared/Code/Sign-AI/data/processed/dataset_v6_unsloth.hdf5` (2,244,045,489 B, mtime 2026-06-23 10:49). Group `dataset2/{keypoints,labels,embeddings,token_ids,text_ctx,video_id,source_group,frame_count,token_count,word_count}`, 5,600 keys each.
- Manifest: `/shared/Code/Sign-AI/data/processed/dataset2_manifest_v125.json` (mtime 2026-06-23 10:53).
- Raw videos: `/shared/Code/Sign-AI/data/raw/dataset2/videos/` holds 8,460 entries. These are 8,459 real MP4s plus one AppleDouble file, `._lsa-noticias-...-05092021_56.mp4`. This matches the prior claim of ~8,459 videos.
- `meta.csv` is missing. `build_dataset2_h5.py:35` and `backfill_dataset2_metadata.py:38` need it. It was removed on 2026-09-10 (`/shared/Code/Sign-AI/DATA_CLEANUP_2026-09-10.md`, "Retirado": everything in `data/raw/dataset2` except `videos/`). Consequence: `select_clips` cannot be re-run, so the 5,600-clip selection cannot be reproduced from source. The H5 still stores `video_id` per clip, so the clip-to-video mapping can be checked (VERIFIED).
- The 5,600 `video_id` values are unique and every one has a video on disk. They come from 64 `source_group` values (VERIFIED).

## Keypoint layout: 134 to 111 (VERIFIED by code plus geometry)
- Stored shape: (T, 134, 2) float32 for every clip. Values are raw pixel xy. **No confidence scores are stored**: `build_dataset2_h5.py:59` drops the score array (`keypoints, _ = model(frame_rgb)`).
- The docstrings say 137 points (`build_dataset2_h5.py:4-5,52,59,79`), which is stale. The code filters on `N_OPENPOSE = 134` (`:30,61`).
- 134 = rtmlib `to_openpose`: COCO-WholeBody 133 plus a synthetic neck, reordered into OpenPose-18 body order:
  - body18 0-17 (0 nose, 1 neck, 2 RSho, 3 RElb, 4 RWri, 5 LSho, 6 LElb, 7 LWri, 8-13 hips/knees/ankles, 14-17 eyes/ears)
  - feet 18-23
  - face68 24-91
  - lhand21 92-112
  - rhand21 113-133

  The rtmlib source is not installed, so the ordering is INFERRED from rtmlib. I checked it empirically on 300 clips. Median distances in shoulder widths: kp92 to body LWri(7) = 0.038; kp113 to body RWri(4) = 0.047; kp93 to 7 = 0.15; kp133 to 4 = 0.31. So 92 and 113 are the hand-wrist roots (VERIFIED).
- 111 = `src/mslm/dataloader/data_augmentation.py:142-151` `remove_keypoints`:
  - `[0..6]` gives 7 pose points: nose, neck, RSho, RElb, RWri, LSho, LElb
  - `[28:92]` gives 64 face points: face68 minus jaw points 0-3
  - `[93:113]` gives 20 left-hand points: **the left-hand wrist root (92) is dropped**
  - `[113:133]` gives 20 right-hand points: **keeps the right wrist (113) but drops the right pinky tip (133)**
  - 7 + 64 + 20 + 20 = 111.

  **Problem: the reduction is asymmetric.** The body left wrist (7) is also excluded while the body right wrist (4) is kept. The left hand loses its wrist point while the right hand keeps its wrist and loses a fingertip.
  - The comment "era 113:134=21 -> off-by-one" was added in `867c563` (2026-06-09). Before that commit the layout was 7+64+20+21 = 112 (`df1c7d1`, 2025-08-01). The real off-by-one is in the left-hand start index (should be 92:113).
  - Downstream, `trim_active_interval` uses `wrist_idx=[71,91]` (`data_augmentation.py:173`), which maps to 134-index 93 (left thumb CMC, not a wrist) and 113 (right wrist).
  - Signer selection in the build script uses `LH_WRIST=93` (`build_dataset2_h5.py:31`), which has the same issue.
  - The 133-point (dataset1) remark in `synthetic_temporal.py:42` is out of scope here.

## Results (all VERIFIED unless noted; numbers from `visual_stats.json`)
| Item | Value |
|---|---|
| Clips / keypoint shape | 5,600 / (T,134,2) for all; `frame_count` attr == T for all |
| NaN / Inf / exact-zero points | 0 / 0 / 0 |
| Frames with no valid point | 0. No-detection frames are not stored at all; see the frame drops below |
| Length T (frames) | p0 3, p5 65, p25 129, p50 215, p75 359, p95 711, p99 1051, max 1400; mean 278.7 |
| Video fps | 29.97: 5,136; 30: 442; 24: 22 |
| Video duration (s) | p0 1.0, p50 7.44, p95 24.1, max 89.8 |
| T == 250 | 12 clips, all with video `nb_frames` 250 (11) or 270 (1). There is **no 250 cap** in this 5,600 build: 2,373 clips have T ≥ 250. The v118 truncation bug belongs to the old 600-clip build |
| T / video frames | 4,967 clips == 1.0; 434 < 0.9; 145 < 0.5; min 0.037 |
| Off-frame proxy (111 regions, mean) | pose 0.30%, face 0.29%, lhand 0.80%, rhand 0.52%; 37 clips have >20% hand points off-frame |
| Degenerate frames (shoulder width < 1 px) | 25 clips |

### Mechanism behind T < video frames (VERIFIED by code)
`build_dataset2_h5.py:50-80` groups detections by the detector's per-frame index `p` (`per_person.setdefault(p, [])`), not by a tracked identity. It then keeps the "person" with the largest wrist motion. Two consequences:
1. Any frame where fewer than p+1 people are detected is silently skipped, which compresses time. Truncation is therefore not only at the tail.
2. Index `p` can refer to a different person in each frame, which produces identity switches.

### Identity switches / tracking failures (main new finding)
- Neck jumps > 0.5 shoulder width in one frame: 3,187 clips; 59,025 jumps in total (VERIFIED).
- 81.4% of those jumps (48,038) return within 0.2 shoulder width of the pre-jump position within 30 frames. This A-B-A oscillation looks like switching between people (INFERRED, strongly supported by mechanism 2 above).
- Jump-frame fraction per clip: p50 1.0%, p75 7.4%, p90 14%, max 42%.
- Flag `identity_switch_suspect` (jump fraction > 5% and at least half the jumps returning): **1,728 clips, of which 1,461 are in the v125 KEPT set**.
- 61 of the 64 source groups are affected.
- Visual check (VERIFIED, one clip): clip 430 (`leyfederalsa-...-patagonia_114`, 1280×720) shows two signers side by side (`_private_previews/clip430_frames1-2.jpg`). In frame 1 the neck is at x=351 with shoulder width 178 px (the left signer). In frame 2 the neck is at x=623 with shoulder width 717 px, a skeleton wider than either person's panel, i.e. a cross-person or spurious skeleton. The woman, who is visibly signing, is not the one tracked in frame 1.
- Thresholds (0.5 sw, 0.2 sw, 30 frames, 5%) are heuristics, not calibrated. I reviewed one clip visually; the rest are flagged statistically and need a human/visual sample (INFERRED that most of them are real).

### Integrity filter re-application (VERIFIED)
- I imported `src/mslm/dataloader/integrity_filter.py` unchanged by path, because the package `__init__` imports torch.
- I rebuilt `ClipRecord` exactly as `scripts/data/build_dataset2_manifest.py:56-112` does. Truncation = `frame_count / video_frames < 0.9` (`:31,49`), with ffprobe `nb_frames` standing in for cv2.
- Result: **kept 5,138 / excluded 462**. Breakdown: truncated 413, min_tokens 29, frames_per_token 16, min_frames 4. This is identical to the stored manifest, with **0 per-clip cause disagreements**.
- 21 clips have ratio < 0.9 but are excluded for an earlier cause first (the filter is first-match, `integrity_filter.py:31-54`). Truncated clips have a median ratio of 0.63.
- 198 KEPT clips have a ratio in [0.9, 1.0), meaning up to 10% of frames were silently dropped.
- The filter checks NaN, Inf, length and tokens only. It has **no check for identity switches, jumps or off-frame hands**.
- Note: the HEAD `integrity_filter.py` was last touched by `7de8f18` (2026-06-24, refactor), after the manifest's mtime. The logic is evidently equivalent because the recomputation matches exactly.

## Limitations
- There are no confidence scores, so "missing point" fractions are only proxies (off-frame or zero). Low-confidence in-frame guesses cannot be detected.
- ffprobe stands in for cv2. Packet count matched the header count, but I did not do decoded frame counts.
- The jump and identity heuristics are uncalibrated, and only one visual check was done.
- The 5,600-clip selection cannot be reproduced because `meta.csv` was deleted.
- The 134-point ordering is inferred from rtmlib conventions and backed by a geometric check, not by reading the rtmlib source.

_Generated with AI assistance (Claude); validate with a subject-matter expert._
