# 05 — Reproducible sample for human LSA review (spec section 6)

Scope: sample design, manifest, review template, rubric, and a pilot protocol check.
- No semantic judgement was made by the agent. Every semantic field is left for LSA reviewers.
- Data was used read-only: HDF5 opened `"r"`, videos only probed or decoded by ffprobe, 3 small JPG frames per pilot clip.

## Command (VERIFIED, ~24 s on CPU)
```
cd /shared/Code/Sign-AI/Sign-chris
/shared/Code/nakato/projects/VQ_RL/.venv/bin/python artifacts/dataset2_audit/scripts/build_review_sample.py
```
- The script runs `_selfcheck()` (asserts on the tercile logic, allocation sums, pilot ⊆ allocation and RNG determinism) before `main()`.
- Determinism (VERIFIED): a second run produced byte-identical `sample_manifest.csv` and `human_review_template.csv` (`sha256sum -c` OK).
- Randomness: `random.Random("23:<stratum-or-bucket>")`, so each stratum and bucket gets its own seed-23 stream and adding a bucket does not shift the others.

Outputs:
- `artifacts/dataset2_audit/sample_manifest.csv`: 122 rows (80 probability + 42 purposive). The specified columns, plus `pilot_check`.
- `artifacts/dataset2_audit/human_review_template.csv`: 122 rows, rubric fields empty.
- `artifacts/dataset2_audit/human_review_rubric.md`: field definitions, reviewer protocol, IAA, estimation, and what not to infer.
- `artifacts/dataset2_audit/_private_previews/pilot/`: 60 JPGs, 480 px wide, frames at 25/50/75% of each pilot clip, 1.6 MB in total. No video copies.

Inputs:
- `_work/visual_integrity_per_clip.csv` ⋈ `_work/text_split_per_clip.csv` on `clip_key`. All 5,600 keys match, and `video_id`/`source_group` agree (asserted).
- HDF5 `dataset2/labels` (the transcript).
- `_work/text_split_stats.json` (footage re-use pairs).
- `artifacts/data_eda/video_metadata.csv` (cross-check of the path listing).

## Population frame
- 5,600 HDF5 clips in total.
- **Main frame:** the 5,138 v125-kept clips (train 4,112 / val 513 / test 513).
- The 462 rejected clips are outside the probability frame. Only 4 of them appear, all in the purposive component, in bucket `v125_rejected`, with `integrity_flags` containing `v125_excluded:<cause>`.

## (a) Probability component: stratified SRS without replacement, n=80
- Strata: split × duration tercile. Duration is the video duration from ffprobe (`video_duration_s`).
- Tercile cuts are computed over the 5,138 kept clips: **5.572 s and 10.544 s**.
- Why duration: it is known for every clip before review, it drives review effort, and it plausibly relates to the alignment and completeness outcomes. Why split: it gives estimates for each split.
- Allocation: train ≈ proportional; val/test have a floor of 4 per stratum so that each split is covered. The unequal weights are corrected by the design weights.

| stratum | N_h | n_h | π = n_h/N_h | weight N_h/n_h | pilot k_h |
|---|---|---|---|---|---|
| train/short | 1,413 | 19 | 0.01345 | 74.37 | 3 |
| train/medium | 1,383 | 19 | 0.01374 | 72.79 | 3 |
| train/long | 1,316 | 18 | 0.01368 | 73.11 | 2 |
| val/short | 157 | 4 | 0.02548 | 39.25 | 2 |
| val/medium | 160 | 4 | 0.02500 | 40.00 | 2 |
| val/long | 196 | 4 | 0.02041 | 49.00 | 2 |
| test/short | 136 | 4 | 0.02941 | 34.00 | 2 |
| test/medium | 174 | 4 | 0.02299 | 43.50 | 2 |
| test/long | 203 | 4 | 0.01970 | 50.75 | 2 |
| **total** | **5,138** | **80** | | Σw = 5,138 | **20** |

- Kish effective sample size: n_eff = 76.2 (VERIFIED).
- **Pilot:** the first k_h clips drawn in each stratum (`pilot=1`), 20 clips in total (train 8 / val 6 / test 6). They were fixed by the draw order before any content was viewed, so the pilot is a stratified random subsample of (a). If the pilot is analysed alone, use weights N_h/k_h.
- The pilot is never replaced. A failed clip stays in and is coded `not_assessable`.
- Incidental composition, not a design target: 15 of the 80 probability clips (2 of the 20 pilot clips) carry `identity_switch_suspect`. Across the kept population the rate is 1,461/5,138 = 28.4%. The rate of 15/80 = 19% is within sampling noise.

## (b) Purposive component: 42 clips, disjoint from (a)
- Random within each bucket (seed 23), after removing clips that are already in (a) or in an earlier bucket.
- `sampling_reason` lists any other buckets the clip also matches.
- These clips must never be pooled with (a) or weighted.

| bucket | n | eligible (all) | of which already in (a) | definition |
|---|---|---|---|---|
| identity_switch_suspect | 5 | 1,461 | 15 | kept, visual flag (02_visual.md) |
| dropped_frames | 4 | 198 | 3 | kept, kp/video frame ratio in [0.9, 1.0) |
| near_filter_threshold | 4 | 82 | 1 | kept, with ratio < 0.92, fpt < 2.5, token_count == 2, or frame_count ≤ 12 |
| fpt_low_tail | 3 | 104 | 1 | kept, fpt ≤ 7.0 (2nd percentile) |
| fpt_high_tail | 3 | 104 | 4 | kept, fpt ≥ 36.0 (98th percentile) |
| very_short_transcript | 3 | 131 | 6 | kept, ≤ 2 words |
| very_long_transcript | 3 | 183 | 0 | kept, > 40 words |
| duplicated_greeting_text | 3 | 376 | 9 | kept, normalized text occurring ≥ 2 times among kept clips |
| footage_reuse_cross_group | 4 | 4 | 0 | both clips of the top-2 same-text cross-group pairs (cos 0.957: 2407↔3827; 0.908: 1529↔5324; 03_text_split.md §4.4). Taken whole, deterministically, not random |
| v125_rejected | 4 | 462 | n/a | one random clip per exclusion cause (frames_per_token, min_frames, min_tokens, truncated) |
| eco_series_test | 4 | 247 | 8 | test clips from the "…episodio-N…" CN Sordos ecology series |
| hand_offframe_or_degenerate | 2 | 52 | 2 | kept, `hand_offframe_gt20pct` or `degenerate_shoulders` |

Note: the 4 `near_filter_threshold` picks all came from the `token_count == 2` arm. This branch dominates the eligible set. No clip near the 0.9 truncation ratio was drawn there, but the `dropped_frames` bucket covers that region.

## Coverage (VERIFIED)
| | source_groups covered (of 64) |
|---|---|
| probability (80) | 43 |
| pilot (20) | 17 |
| purposive (42) | 24 |
| either | 46 |

- 18 groups are not represented. Most are small one-off or eco groups.
- Coverage is a by-product of the design and was not forced, because forcing it would distort the inclusion probabilities.
- If every group must be seen, add a separate third component, for example one random clip per uncovered group. It must stay outside the estimation.

## Pilot protocol check (VERIFIED, `pilot_check` column)
- For each of the 20 pilot clips:
  - The mp4 exists.
  - It is listed in `video_metadata.csv`.
  - It was **fully decoded** with `ffprobe -count_frames`.
  - Decoded frames were compared with the header `nb_frames` and with the keypoint T.
  - Container duration was compared with the visual CSV.
- Results:
  - 20/20 mp4s exist and decode.
  - The decoded frame count equals the header count in 20/20.
  - The duration matches within 0.1 s in 20/20.
  - fps: 13 clips at 29.97 and 7 at 30.000. No 24 fps clip is in the pilot.
  - **19/20: keypoint T == decoded frames.**
  - **PR022 (clip 2565, train/medium, kept):** T = 253 vs 257 decoded (ratio 0.984). This is the known silent frame-dropping mechanism (02_visual.md); 4 frames are missing. It is recorded and not replaced.
- Previews: 60 JPGs in `_private_previews/pilot/` (`<sample_id>_clip<id>_<pct>.jpg`). They are private and pilot-only.

## Limitations
- Only 4 clips per val/test stratum. Per-split CIs will be roughly ±25 pp. Global estimates rest mostly on train (56/80 clips; 80% of the population weight).
- Duration terciles are a proxy stratifier. Strata on source series (news / eco / other) or signer were considered and rejected. Series is nested in split and is very imbalanced (eco is 1.5% of train). Signer is UNKNOWN (no metadata).
- One clip per video_id, so the clips are effectively independent within a stratum. Clips within the same source_group (episode, signer) are correlated, and the design does not cluster on that. CIs may therefore be somewhat optimistic for properties that vary by episode (INFERRED).
- The purposive bucket definitions use the uncalibrated heuristics of 02/03 (for example, the identity-switch thresholds). The review of these buckets is meant precisely to calibrate them.
- `meta.csv` was deleted, so `source_video_path` relies on the HDF5 `video_id`, which was verified to exist on disk for all 5,600 clips (01/02).

_Generated with AI assistance (Claude); validate with a subject-matter expert._
