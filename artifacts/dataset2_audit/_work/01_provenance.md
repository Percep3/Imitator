# 01 — Provenance, environment and physical inventory of dataset2

Agent scope: AUDIT_SPEC sections 2 (Provenance & environment) and 3 (Dataset2 reconstruction) + inventory input for section 4.
All operations were read-only. No HDF5 was opened with h5py (none available, see E1). No commits.
Tags: VERIFIED / DOCUMENTED / INFERRED / UNKNOWN.

---

## E. Environment

| # | Item | Value | Tag | Evidence |
|---|---|---|---|---|
| E0 | Date | 2026-10-19T23:21-05:00 | VERIFIED | `date -Is` |
| E0 | Branch / HEAD | `v125-temporal-prompt-alignment` / `a7db233a86eff014459bedecb470e1a7c4141add` | VERIFIED | `git branch --show-current; git rev-parse HEAD` |
| E0 | `git status --short` | `M .beads/issues.jsonl`; ` D paper/revisions/Sign-chris-1tl/README.md`; `?? academic_writing_avoid_ai_style_signals.md`; `?? artifacts/dataset2_audit/`; `?? docs/research_audit_imitator_e1.md` | VERIFIED | `git status --short` |
| E1 | **Sign-env is EMPTY** | `/home/nakato/miniconda3/envs/Sign-env/` contains only `conda-meta/`; no `bin/python`. History: created 2026-06-09 (`conda env create -n Sign-env -f /tmp/sign-env.yml`), then **2026-08-08 23:33 `conda remove -n Sign-env --all --keep-env -y`**. | VERIFIED | `ls -la .../envs/Sign-env`; `grep -E '^==>|^# cmd' .../Sign-env/conda-meta/history` |
| E1 | Other pythons | `~/miniconda3/bin/python` 3.13.13, numpy 2.5.1, **no h5py**; `knowledge/.venv` 3.13.13 numpy 2.4.6, no h5py; `/usr/bin/python3` 3.12.3, no numpy; `envs/tex` has no python. `find / -xdev -name h5py -type d` → nothing; no `h5dump`/`h5ls`. | VERIFIED | commands in §R |
| E1 | Consequence | **BLOCKER for all HDF5-content checks** (shapes, labels, video_id, source_group, frame_count). Per HARD RULES no pip install was done. Options for the coordinator: (a) user-approved isolated venv in scratchpad with `h5py` only (does not touch Sign-env), or (b) restore Sign-env from `environment.lock.yml`. The prior audit hit the same block (`docs/research_audit_imitator_e1.md:268,471`). | VERIFIED | — |
| E2 | Original Sign-env (historical) | python 3.12.9, torch 2.8.0+cu128, transformers 5.10.2, rtmlib, onnxruntime-gpu | DOCUMENTED | `~/.claude/.../memory/sign-env-setup.md`; `environment.lock.yml` |

---

## C. Code lineage (HEAD vs code that produced the artifacts)

All dataset2 scripts lived in `scripts/` until the refactor `7de8f18` (2026-06-24 13:18 -05) moved them to `scripts/data/`. Command: `git log --follow --format='%h %ad %s' -- <file>`; diff vs freeze commit: `git diff d405c7d:scripts/<f>.py HEAD:scripts/data/<f>.py`.

| File (HEAD path) | Commits (local time -05) | Diff vs v125 freeze `c4113c3`/`d405c7d` (2026-06-23 11:11) | Tag |
|---|---|---|---|
| `scripts/data/build_dataset2_h5.py` | 867c563 06-09 (created, N_OPENPOSE=134, default `--max-frames 250`, `--n 600`, seed 23); 05bca63 06-10 (LLM default → gemma-3n bnb-4bit); 35a1d89 06-18 (text_ctx phase); 7de8f18 06-24 (path only) | 2 lines: usage docstring path + `parents[2]` | VERIFIED |
| `scripts/data/build_dataset2_manifest.py` | 74a7ea9 06-23 00:33; 0cc6cdf 00:44 (truncation vs source video, ratio 0.9); c4113c3 11:11; 7de8f18 | 2 lines (paths) | VERIFIED |
| `scripts/data/build_dataset2_split.py` | 1c5206f 06-23 01:05; 7de8f18 | 2 lines (paths) | VERIFIED |
| `scripts/data/backfill_dataset2_metadata.py` | 422d285 06-23 00:15; c4113c3; 7de8f18 | 3 lines (paths) | VERIFIED |
| `scripts/data/rebuild_dataset2_alignment_targets_v125.py` | c4113c3 06-23 11:11; 7de8f18 | 2 lines (paths) | VERIFIED |
| `src/mslm/dataloader/integrity_filter.py` | b41997b 06-23 00:05 only | none | VERIFIED |
| `src/mslm/dataloader/grouped_split.py` | 3a738d1 06-23 00:09 only | none | VERIFIED |
| `scripts/data/data_eda.py` | e68c7be 06-08/09; 7de8f18 | path only | VERIFIED |
| `scripts/data/add_token_ids_h5.py` | 2a70917 06-10; 7de8f18 | — | VERIFIED |

Artifact mtimes vs commits (`ls -l --time-style=full-iso`, git dates in local time):
- `dataset_v6_unsloth.hdf5` mtime 2026-06-23 10:49:51; manifest 10:53:17; split 10:53:20; anisotropy gate 10:53:32; then commit c4113c3 at 11:11 ("reparar oracle Gemma y congelar artefactos"); `v125_artifacts.json` 15:02:22. Order is consistent with: rebuild_alignment_targets (writes H5) → manifest → split → gate → commit. **VERIFIED (timestamps); causal order INFERRED.**
- HEAD logic of every v125 data script equals the freeze-time logic (only path edits). **VERIFIED.** The 600-clip build (06-09/10) and the 5,000-clip extension (≈06-18/19) were run with `build_dataset2_h5.py` whose logic at that time differs from HEAD only by the LLM default and the added text_ctx phase. **VERIFIED via `git log -p --follow`.**
- The exact command lines (and `--max-frames`) of the 5,000-clip extension are **not recorded** anywhere (searched `git log --grep`, reports, configs, `.claude/settings.json`). **UNKNOWN.** Only `contrastive_v118f.toml:1-8` documents "600 originales + 5000 nuevos extraídos del mismo pipeline RTMPose, mismo canal CN Sordos" (DOCUMENTED), and `build_dataset2_manifest.py:9-12` states "el build actual de 5600 clips no usó ese cap [250]" (DOCUMENTED).
- The backfill was run as `PYTHONPATH=. .../Sign-env/bin/python scripts/data/backfill_dataset2_metadata.py --h5 /shared/Code/Sign-AI/data/processed/dataset_v6_unsloth.hdf5 --n 5600 --seed 23` (`.claude/settings.json:94`, permission allow-list entry). **DOCUMENTED.**

---

## I. Physical inventory

Helper: `artifacts/dataset2_audit/scripts/inventory.py` (stdlib only). Output TSV: `artifacts/dataset2_audit/_work/01_inventory.tsv`. Run with `--full-big` (the 2.24 GB HDF5 was hashed in full because a full reference hash exists; first-64MiB hash also given).

| path | exists | size (B) | mtime (-05) | sha256 | produced_by | notes |
|---|---|---|---|---|---|---|
| /shared/Code/Sign-AI/data/processed/dataset_v6_unsloth.hdf5 | yes | 2,244,045,489 | 2026-06-23 10:49:51 | 24c9aa401b44b14283ea71851b12ccb44b3c224ec7d7b85119b534389c3b1d38 (full); first 64 MiB: 36d98a45673dee6bae901f4f915adac8ba95628e580c23f99f0cf096b90faa68 | build_dataset2_h5.py → add_token_ids_h5.py → backfill → rebuild_alignment_targets_v125 | **matches v125_artifacts.json** (VERIFIED). Content not inspectable (E1). |
| /shared/Code/Sign-AI/data/processed/dataset2_manifest_v125.json | yes | 439,755 | 2026-06-23 10:53:17 | 04c3a7c1836483bc7fcd749b32386339b5e407be34d9483bb4e1ec3276598bad | build_dataset2_manifest.py | matches freeze |
| /shared/Code/Sign-AI/data/processed/dataset2_split_v125.json | yes | 60,749 | 2026-06-23 10:53:20 | 6669635ad97950c4f48c90816f8bf99e5902bd107412cc049670bec6f388dbf9 | build_dataset2_split.py | matches freeze |
| /shared/Code/Sign-AI/data/processed/v125_artifacts.json | yes | 2,122 | 2026-06-23 15:02:22 | 9d71868cdfbbedc60b6c4c3e7530afc37a65679569ec457dc32e6be24785614e | v125 freeze | reference hashes; tokenizer revision 3d26ffdd… |
| /shared/Code/Sign-AI/data/processed/gemma3n_embed_table.pt | yes | 1,074,792,061 | 2026-06-10 00:50:10 | 2043797146e6d6ff94d107f25715fe602d73429adf74ab79471818be2763a6d7 | add_token_ids_h5.py --phase table | matches freeze |
| /shared/Code/Sign-AI/data/processed/v125_gemma_oracle.json | yes | 58,147 | 2026-06-23 15:02:09 | b037bfca2b539ecef2923925feca88a126a4ed9595976701d9f3ea09148a75be | v125 oracle | matches freeze; 100 rows with clip_id+label (only surviving label text outside H5) |
| /shared/Code/Sign-AI/data/processed/v125_anisotropy_gate.json | yes | 258 | 2026-06-23 10:53:32 | b7fa2139bca8e112b40579039489e27a5c2c3e2067a0940e9a29ec81497b8a10 | v125 gate | matches freeze |
| /shared/Code/Sign-AI/data/processed/v125_anisotropy_gate.transform.pt | yes | 18,589 | 2026-06-23 10:53:32 | fbf55997b32fc512a35a62d06c18253048a0f50bb40f42adc3e0fb7b46e11bc9 | v125 gate | matches freeze |
| /shared/Code/Sign-AI/data/processed/ds2_smoke.hdf5 | yes | 419,556 | 2026-06-09 15:18:14 | b54430e35000c2e580d12b82fb817ed44fcf1579610c37734d5a63c850840970 | UNKNOWN (smoke run of build_dataset2_h5) | not used by v116+ configs (INFERRED) |
| /shared/Code/Sign-AI/data/raw/dataset2/videos/ (dir) | yes | 28,581,651,780 total | dir 2025-08-09; files 2023-07 | listing fingerprint 3062dde404c54ef2130ae3f7ddec38282e4d1eaf2f6921859753c51e92b18b44 (sha256 of sorted `name\tsize`; contents NOT hashed) | external segmentation pipeline (not in repo) | 8,460 files = **8,459 .mp4 + 1 AppleDouble `._*`**; 8,459 file mtimes in 2023-07 |
| /shared/Code/Sign-AI/data/raw/dataset2/meta.csv | **no** | — | — | — | external | deleted 2026-09-10 (DATA_CLEANUP) |
| /shared/Code/Sign-AI/data/raw/dataset2/keypoints.h5 | **no** | (≈20.4 GB per `artifacts/data_eda/file_inventory.csv`) | — | — | legacy MediaPipe pipeline | deleted 2026-09-10 |
| /shared/Code/Sign-AI/data/processed/dataset_v6_unsloth.hdf5.bak_pre250fix | **no** | — | — | — | v118 fix backup | referenced in `report.md:274`; gone |
| Sign-chris/artifacts/data_eda/video_metadata.csv | yes | 2,815,816 | 2026-06-08 23:31:11 | 67689c12d4b5dcd1ebe34127b0169f1e7326c9a9d0c0ed85dad28f8dabb2da18 | data_eda.py | per-video ffprobe: duration, WxH, fps, codec, has_audio |
| Sign-chris/artifacts/data_eda/dataset_summary.csv | yes | 1,938 | 2026-06-08 23:31:12 | 994ef8e89841e1d157ea2225822153d90de823e484bbb81954e7be2d046bd347 | data_eda.py | ds2 meta stats while meta.csv still existed |
| Sign-chris/artifacts/data_eda/eda_summary.json | yes | 7,915 | 2026-06-08 23:31:12 | a2e984531a3b77b2f116574308ef8533f4f699dbd36143167d6d155111af7a4f | data_eda.py | |
| Sign-chris/artifacts/data_eda/srt_summary.csv | yes | 1,754 | 2026-06-08 23:31:12 | 6fb35e6d14e38d41c57dd78c4af3d72546c8a8d58f0d47537302c00ff489669c | data_eda.py | **27 SRTs, all dataset5; none for dataset2** |

Searches for other dataset2 sources (all negative): `find /shared/Code/Sign-AI /home/nakato -xdev \( -name meta.csv -o -name '*dataset2*' -o -name '*.srt' -o -name '*.vtt' -o -name '*.bak_pre250fix' \)` → dataset2 has no meta.csv, no .srt/.vtt/.json subtitles; only dataset1/3/4/5 meta.csv and dataset5 SRTs exist. **VERIFIED.**
`/shared/Code/Sign-AI/outputs` contains no dataset2 data files (v126/LOSO/decoder outputs only). **VERIFIED** (`find -maxdepth 3`).

Raw video properties (from `video_metadata.csv`, ffprobe at 2026-06-08): 8,459 h264 clips; fps 29.97 (7,735) / 30.0 (686) / 24.0 (38); 1920x1080 (8,109), 1920x1078 (256), 1280x720 (79), 640x360 (15); **has_audio = False for all 8,459**; duration min 1.001 s, median 7.407 s, max 89.833 s; 176 clips > 30 s; 20 clips > 46.67 s (=1400 frames @30fps). **VERIFIED** (python over the CSV, §R). Consequence: clip audio cannot be used to re-derive or check Spanish text (no ASR possible from clips). **VERIFIED.**

---

## P. Provenance of (X_i, Y_i)

### P1. Source
- 8,459 clips from **64 parent videos** (filename = `<parent-slug>_<n>.mp4`; groups by stripping `_<n>`), e.g. `noticias-en-lengua-de-senas-argentina-resumen-semanal-*` (CN Sordos weekly news), `leyfederalsa-resumen-de-la-mesa-de-trabajo-region-patagonia`, `agua-como-tomar-conciencia-...-episodio-3-...-lsa`, `ultimo-momento-en-lsa-sismo-san-juan-18-enero-2021`. Largest group 439 clips, smallest 3. **VERIFIED** (filename parse, `ls | sed -E 's/_[0-9]+\.mp4$//' | sort | uniq -c`). That the parent slug equals the original YouTube/CN Sordos video id is **INFERRED** (slug style; channel "CN Sordos" DOCUMENTED in `contrastive_v118f.toml:3` and label "buenas noches. bienvenidos a cn sordos argent…" in the historical meta preview below).
- Language LSA (Argentina), not LSP. **DOCUMENTED** (`docs/audits/v125_language_audit.md:5-9`; program names). Not linguistically certified.
- Clip suffix numbering has gaps (e.g. `leyfederalsa…` has 79 files with suffixes up to ≥120). **VERIFIED**; whether gaps are dropped segments is **INFERRED**.

### P2. Original metadata schema — recovered from git history (new finding)
Commit `fef07a3` (2025-03-25, Giorgio6846) added `Sign-Dataset.ipynb` (removed in `03f4066`, 2025-06-19) whose saved output prints `pd.read_csv("../data/dataset2/meta.csv")`: **8459 rows x 14 columns**:
`id, label, video, playlist, start, end, duration, splits, prev_delta, post_delta, signers_amount, infered_signer, infered_signer_confidence, movement_per_signer`.
Command: `git show fef07a3:Sign-Dataset.ipynb | python -c '<print outputs>'`. **VERIFIED** (only head 5 + tail 5 rows are visible).

Visible rows (verbatim values, truncated by pandas):

| row | video (truncated) | playlist | start | end | duration | splits (truncated) | prev_delta | post_delta | signers_amount | infered_signer | conf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | noticias-…-resumen-… | resumen_semanal | 3.90 | 14.64 | 9.698 | `[('para dejar en el camino tanto inglés, para …` | NaN | 0.08 | 18 | signer_0 | 0.656 |
| 1 | idem | resumen_semanal | 14.72 | 37.12 | 26.420 | `[(' mientras trabajábamos para brindarte toda …` | 0.08 | 0.06 | 2 | signer_1 | 0.918 |
| 2 | idem | resumen_semanal | 37.18 | 43.54 | 6.360 | `[('será recordado por siempre. este programa e…` | 0.06 | 33.27 | 2 | signer_1 | 0.902 |
| 3 | idem | resumen_semanal | 76.81 | 82.66 | 4.730 | `[(' buenas noches. bienvenidos a cn sordos arg…` | 33.27 | 0.18 | 2 | signer_1 | 0.845 |
| 4 | idem | resumen_semanal | 82.84 | 84.75 | 1.910 | `[('mi nombre es lucía.', 82.84, 84.75)]` | 0.18 | 32.00 | 2 | signer_0 | 0.671 |
| 8454 | leyfederalsa-… | ley_federal_lsa | 497.36 | 499.09 | 1.730 | `[('sí, ahora.', 497.36, 499.09)]` | 0.06 | 0.06 | 2 | signer_0 | 0.538 |
| 8457 | idem | ley_federal_lsa | 510.57 | 532.60 | 20.690 | `[('deborah es docente de la unco, …` | 4.33 | 0.00 | 2 | signer_0 | 0.958 |
| 8458 | idem | ley_federal_lsa | 532.66 | 536.75 | 4.090 | NaN | 0.00 | NaN | 2 | signer_0 | 0.979 |

Interpretation:
- **Explicit timestamps existed**: `start`/`end` (seconds in the parent video) and `splits` = list of `(text, start, end)` sub-segments, i.e. caption cues. **VERIFIED that the columns existed; the file is deleted, so per-clip timestamps are NOT recoverable** (only these 8 rows). Do not invent the rest. **VERIFIED.**
- A clip's label appears to be the concatenation of one or more caption cues (`splits` with several tuples, label starting with a leading space like `' mientras…'`), and the clip span = first cue start → last cue end. **INFERRED** (from row 0/1/3 strings and leading spaces; cannot check beyond 10 rows).
- `prev_delta`/`post_delta` = gap to neighbouring segment (rows 2→3: post 33.27 = prev 33.27; 3→4: 0.18). Many gaps are 0.06–0.08 s, i.e. consecutive cues nearly abut → interpretation spilling across clip boundaries is plausible. **VERIFIED arithmetic on visible rows; spillover risk INFERRED.**
- `duration` ≠ `end − start` for multi-cue rows (row 0: 10.74 vs 9.698; row 1: 22.40 vs 26.42; row 3: 5.85 vs 4.73; row 8457: 22.03 vs 20.69) but equal for single-cue rows (2, 4, 8454). Meaning of `duration` **UNKNOWN**; this inconsistency means clip-file length vs caption span cannot be assumed equal. **VERIFIED (arithmetic); meaning UNKNOWN.**
- Whether subtitles are **manual captions, auto-generated captions or ASR** is **UNKNOWN**. No subtitle file, download script or ASR script for dataset2 exists in the repo, data dirs or git history. The only ASR code found (`data/scripts/transcripcion.ipynb`, Whisper large-v3 `language="es"`) and the SRT→clip splitter (`data/scripts/srt_videos.ipynb`, moviepy `subclip(start,end)`, ids `<name>-<index>`) target **dataset5** (outputs reference `.../data/dataset5/`; dataset2 ids use `_<n>` not `-<n>`). **VERIFIED that they are not dataset2's pipeline (path/id format); dataset2 text origin UNKNOWN.**
- The segmentation code that cut dataset2 clips (file mtimes 2023-07) is not in any accessible repo. **VERIFIED absence / origin UNKNOWN.**
- Interpreter changes: `infered_signer` (signer_0/signer_1) alternates between consecutive clips; `signers_amount` = 18 on row 0 (detector noise). So ≥2 people visible and the "signer" varies per clip. **VERIFIED on visible rows.** Prevalence across 8,459 **UNKNOWN**. `v125_language_audit.md:37-40` says inferred signer is not a stable identity (DOCUMENTED).
- Multiple sentences per clip: visible labels contain several sentences (row 3 "buenas noches. bienvenidos a cn sordos…", row 8456 "chau. chau.", row 8455 "perfecto, muchas gracias por venir. gracias a…"). **VERIFIED on visible rows.**
- Could caption start/end fall outside the clip? Clip files are presumably cut at [start,end] of the cues (INFERRED). Sign-language interpretation typically lags captions; with 0.06 s gaps to neighbours any lag pushes signs into the adjacent clip. Not measured. **UNKNOWN (plausible risk).**

### P3. Legacy keypoints (not used by v116+)
Old pipeline: `data/raw/dataset2/keypoints.h5` grouped `clip/signer/keypoints`, 543 joints x 4 (MediaPipe-holistic-like), loaded by `Classes/KeypointDataset.py` (`git show e3b600a:Classes/KeypointDataset.py`) filtering `infered_signer==signer & duration<30` → 5,412 items (`fef07a3` notebook output). **VERIFIED from history; file deleted 2026-09-10.**

### P4. Current HDF5 pipeline (v116–v125)
`build_dataset2_h5.py` (HEAD, line refs):
- `select_clips` (`:35-46`): read `meta.csv`, drop empty labels, keep rows whose `<id>.mp4` exists, `label.lower()`, `random.Random(23).shuffle`, take first n. **VERIFIED.**
- Keypoints (`:50-80, 83-119`): RTMPose wholebody via rtmlib `Custom(to_openpose=True)`, cv2 decode, frames up to `--max-frames`; keeps only frames where `kp.shape[1]==134` (`N_OPENPOSE=134`, `:30`); groups detections by **detection index p, not by tracked identity** (`:60-63`), picks the index with max wrist movement (`:71-78`), concatenates its frames **dropping frames where that index was absent** (`:78`). **VERIFIED (code).** Consequence: silent temporal gaps and possible identity switches inside one clip **INFERRED** (not measured). Supports why 50 of the first 600 clips are still "truncated" (<0.9 of source frames) after the v118 re-extraction (see V5). **INFERRED.**
- Labels: `dataset2/labels/<idx>` = lower-cased `label` from meta.csv (`:112-113`). Embeddings: gemma-3n input-embedding lookup of tokenized label (`:139-164`), later rebuilt by `rebuild_dataset2_alignment_targets_v125.py:33-80` with NFC/lowercase/no-punctuation text, `add_special_tokens=False`; original label kept. **VERIFIED.**
- Backfill (`backfill_dataset2_metadata.py:28-102`): `video_id = Path(fname).stem`, `source_group = meta["video"]` for that id; it **raises ValueError if any stored label differs from `select_clips(5600,23)` order** (`:59-64`). Therefore if it completed, the key→video mapping of all 5,600 clips was checked against meta.csv at that time. **Code VERIFIED; successful completion INFERRED** (note: the manifest maps a missing `video_id` to `truncated`, `build_dataset2_manifest.py:88-93`, so the manifest alone cannot prove the backfill covered all clips).
- `random.Random(23).shuffle` on the same list gives the same permutation, so `select_clips(5600,23)[:600] == select_clips(600,23)` provided meta.csv and the video dir were unchanged → keys 0–599 are the original v116–v118e subset. **INFERRED** (contradicts the prior audit's statement `research_audit_imitator_e1.md:260` that numeric keys did not preserve identity; the backfill check above supports identity).
- `source_group` = meta.csv `video` column = parent video slug (test fixture `tests/scripts/test_backfill_dataset2_metadata.py:8-17`: `…resumen-semanal-13062021_227` → `…resumen-semanal-13062021`). **DOCUMENTED.**
- Reconstructing `select_clips` now is **BLOCKED**: meta.csv row order is unknown and the file is deleted.

Heuristic (read-only `strings -n 12` over the HDF5): 5,520 distinct strings exactly equal to raw-video stems were found (each twice; HDF5 does not reclaim freed space after `del`, so stale copies are possible). This supports that `video_id` entries reference real raw files but is **NOT a count** — **INFERRED**, needs h5py.

---

## V. Verification of prior numbers

| Claim | Result | Tag | Evidence |
|---|---|---|---|
| ~8,459 original videos | 8,459 `.mp4` + 1 AppleDouble sidecar in `data/raw/dataset2/videos`; meta.csv had 8,459 rows (2025-03 notebook output and 2026-06-08 EDA `dataset_summary.csv`: metadata_rows 8459, exact id matches 8459, 8,016 unique labels, 443 duplicate-label rows, 0 empty labels) | VERIFIED (files) / DOCUMENTED (meta rows) | `ls | wc -l`; `artifacts/data_eda/dataset_summary.csv` |
| 5,600 clips in HDF5 | manifest `summary.total=5600`, clip_ids exactly 0..5599 contiguous, none missing | VERIFIED (manifest); H5 itself not opened | python over manifest (§R) |
| 5,138 kept by v125 filter | kept 5,138; excluded 462 = truncated 413 + min_tokens 29 + frames_per_token 16 + min_frames 4 (first-cause only) | VERIFIED | manifest |
| truncated by origin | 50 truncated among keys 0–599, 363 among 600–5599 | VERIFIED (counts) | manifest |
| split 4,112/513/513 | seed 23, ratios 0.8/0.1/0.1; pairwise disjoint; union == kept | VERIFIED | split JSON |
| split grouped by source_group | algorithm groups by `source_group` (`grouped_split.py:13-44`); group disjointness of the actual JSON **cannot be re-verified** without H5 `source_group` | code VERIFIED / data UNKNOWN (blocked) | — |
| 111 keypoints | `remove_keypoints` (`data_augmentation.py:142-151`): 7 pose + 64 face (28:92) + 20 left hand (93:113) + 20 right hand (113:133) = 111; 2D (x,y), no confidence | VERIFIED (code) | — |
| 133/134/137 layout history | 250kp (datasetv1) → 133kp (dataset_v2/v3) → 113kp (v7) per `data/readme.MD`; 543x4 legacy MediaPipe keypoints.h5 (2025); RTMPose builder uses 134 (`N_OPENPOSE=134`) while its docstring and inline comments say 137; right-hand slice was 113:134 (21 pts → 112 total) until `867c563` (2026-06-09) changed it to 113:133 | VERIFIED (files/history) | `git log -L142,152:src/mslm/dataloader/data_augmentation.py` |
| Hand index semantics | In rtmlib's 134-pt openpose layout (body 18, feet 6, face 68 at 24–91, L-hand 92–112, R-hand 113–133) the slices drop the **left wrist (92)** but the **right last fingertip (133)** — asymmetric; and `LH_WRIST=93` in the builder would be a thumb joint, not the wrist | INFERRED (depends on rtmlib layout, not checked against rtmlib source here) | `build_dataset2_h5.py:31`, `data_augmentation.py:147-148` |
| "3,088 clips misaligned >1 s" | Appears **only** as a comment `check_alignment = true  # QA EDA: 3088 clips desalineados >1s` and `alignment_tolerance_s = 1.0` in the sigreg config, introduced in `867c563` (2026-06-09, `config/experiment/sigreg.toml`, now `experiments/v115_ce_vocab/sigreg.toml:18-19`, copy in Sign-chris-public). `check_alignment` is consumed by **no code** in `src/` or `scripts/` (grep). Not in `artifacts/data_eda/*` (grep). No procedure, denominator or clip list found in git (`git log --all -S3088`, `-S check_alignment`, `-S desalineados`), docs, reports, outputs or Claude memory/transcripts. | DOCUMENTED (comment only); procedure UNKNOWN — **do not use as fact** | commands §R |

---

## B. Reconstructable only from logs/git vs blocked

Reconstructable from git/logs (partial):
1. meta.csv schema (14 columns) and 10 sample rows — `git show fef07a3:Sign-Dataset.ipynb`.
2. Legacy loader filters and old keypoint format — `e3b600a`, `d557e9e^`.
3. 600→5600 history and v118 truncation fix — `report.md:272-274`, `contrastive_v118c.toml:9-12`, `contrastive_v118f.toml:1-8`, memory `v118-truncated-video-bug.md`.
4. Code used for v125 artifacts == HEAD logic (path changes only).
5. Artifact integrity: all 7 v125 artifacts match `v125_artifacts.json` hashes.

Blocked:
1. **All HDF5 content checks** (labels, video_id, source_group, frame_count, keypoint shapes, group disjointness of split, layout 134 in stored arrays) — no h5py anywhere; Sign-env emptied 2026-08-08. Needs user decision (E1).
2. **Per-clip timestamps / subtitle cues / original caption source** — meta.csv deleted 2026-09-10; no SRT/VTT for dataset2 anywhere; clip videos have no audio. Only restorable from an external copy or by re-downloading parent videos+captions (download forbidden in this audit).
3. Re-running `select_clips`, backfill, manifest truncation check — require meta.csv.
4. Re-extracting keypoints — `local_models/Keypoint-Models` removed (DATA_CLEANUP) and no GPU inference allowed.
5. Exact command/`--max-frames` used for the 5,000-clip extension — never recorded.
6. Origin of the 3,088 figure.

Side notes:
- `data/scripts/DataverseDownload.ipynb` (dataset5/PUCP, not dataset2) contains a plaintext Dataverse API key — security hygiene, outside scope; flag to the user.
- Prior-audit statement "keys did not preserve identity" (`research_audit_imitator_e1.md:260`) is weakened by P4 (shuffle-prefix property + backfill label check).

---

## R. Exact commands used

```bash
cd /shared/Code/Sign-AI/Sign-chris
git branch --show-current; git rev-parse HEAD; date -Is; git status --short
ls -la /home/nakato/miniconda3/envs/Sign-env/; grep -E '^==>|^# cmd' /home/nakato/miniconda3/envs/Sign-env/conda-meta/history
for p in ~/miniconda3/bin/python knowledge/.venv/bin/python /usr/bin/python3; do $p -c "import sys,numpy,h5py"; done
find / -xdev -name h5py -type d; which h5dump h5ls
for f in <key files>; do git log --follow --format='%h %ad %s' --date=iso -- $f; done
git diff d405c7d:scripts/<f>.py HEAD:scripts/data/<f>.py
git log --all --format='%h %ad %s' --date=format-local:'%Y-%m-%d %H:%M' --since=2026-06-22 --until=2026-06-25
cat /shared/Code/Sign-AI/DATA_CLEANUP_2026-09-10.md /shared/Code/Sign-AI/data/readme.MD /shared/Code/Sign-AI/data/scripts/{note.md,script.sh}
ls /shared/Code/Sign-AI/data/raw/dataset2/videos | wc -l; du -sb .../videos; find .../videos -type f -printf '%TY-%Tm\n' | sort | uniq -c
ls .../videos | sed -E 's/_[0-9]+\.mp4$//' | sort | uniq -c | sort -rn      # 64 parent groups
python (json) over data/scripts/*.ipynb                                      # dataset5 SRT/Whisper pipelines
git log --all -S'playlist' / -S'infered_signer' / -S'prev_delta'; git show fef07a3:Sign-Dataset.ipynb | python (print outputs)
git show e3b600a:Classes/KeypointDataset.py
git log --all -S3088; git log --all -S'check_alignment'; git log --all -S'desalineados'
grep -rn -I '3088\|3,088' . ; grep -rn check_alignment src scripts
git log -L142,152:src/mslm/dataloader/data_augmentation.py
python3 artifacts/dataset2_audit/scripts/inventory.py --full-big > artifacts/dataset2_audit/_work/01_inventory.tsv
head -c 67108864 /shared/Code/Sign-AI/data/processed/dataset_v6_unsloth.hdf5 | sha256sum
python3 - <<EOF  # manifest/split: contiguity, counts, disjointness, union==kept, truncated by key range
strings -n 12 dataset_v6_unsloth.hdf5 | sort | uniq -c   # heuristic only, read-only
python3 over artifacts/data_eda/video_metadata.csv      # fps, has_audio, durations
find /shared/Code/Sign-AI /home/nakato -xdev \( -name meta.csv -o -name '*dataset2*' -o -name '*.srt' -o -name '*.vtt' -o -name '*.bak_pre250fix' \)
```

---

## HDF5 verification (resumed)

Unblocked by coordinator: `/shared/Code/nakato/projects/VQ_RL/.venv/bin/python` (h5py 3.16, numpy 2.3.1), used read-only (`h5py.File(p, "r")`), nothing installed.
Helper: `artifacts/dataset2_audit/scripts/h5_provenance.py` → `artifacts/dataset2_audit/_work/01_h5_provenance.json`. Extra one-off checks were run inline with the same interpreter (commands below). All items in this section are **VERIFIED** unless tagged otherwise. This supersedes the "blocked" items B1/B3 (partly) and the string-scan heuristic in P4.

| Check | Result |
|---|---|
| Groups under `dataset2/` | embeddings, frame_count, keypoints, labels, source_group, text_ctx, token_count, token_ids, video_id, word_count; **5,600 keys each, none missing in any group** |
| Keys | exactly `"0".."5599"`; **identical to manifest clip_ids** |
| Keypoint arrays | all 5,600 have shape (T, **134**, 2) → stored layout is 134 (not 133/137); `frame_count == keypoints.shape[0]` for 5,600/5,600 |
| Labels | 5,600 present, 0 empty; the 100 `v125_gemma_oracle.json` labels equal the H5 labels 100/100 |
| video_id | 5,600 present, **5,600 unique**, **5,600/5,600 exist as `data/raw/dataset2/videos/<video_id>.mp4`** |
| source_group | 5,600 present; `source_group == video_id` with the trailing `_<n>` removed for 5,600/5,600 → source_group = parent video |
| Parent videos | **64 source_groups in H5 = all 64 parents in raw videos**; every parent has ≥1 kept clip |
| Parent-level leakage | **0 source_groups span more than one split.** Train 40 parents (4,112 clips), val 12 (513), test 12 (513). Full group→split table is in `01_h5_provenance.json` (`test_groups`, `val_groups`) |
| Keys 0–599 = old 600-clip subset | Strong evidence: exactly **228** clips with frame_count ≥250 among keys 0–599 (1 exactly 250), **max 1,312**. That matches the v118 fix record (228 truncated clips re-extracted with cap 1400, real max 1312; `report.md:272-274`, memory `v118-truncated-video-bug.md`). Keys 0–599 cover 62 parents. Strongly **INFERRED** (the old 600-id list is not saved anywhere, so it can't be compared directly). Together with the shuffle-prefix property and the backfill label check (P4), the prior audit's "keys did not preserve identity" claim should be dropped. |
| `--max-frames` of the 5,000-clip extension | keys ≥600: max frame_count = **1,400**, 12 clips exactly 1,400 (none among keys <600) → extension was run with `--max-frames 1400` and **12 clips are hard-capped**. **INFERRED** (from the cap value; the command line was never recorded) |

Observations relevant to other sections (provenance/grouping only; text duplicates are left to 03_text_split.md):
- Program-family mix differs by split (families by slug substring, INFERRED): train = 4,005 weekly-news + 60 "episodio" series + 47 `leyfederalsa`; val = 417 news + 94 episodio + 2 último-momento; **test = 256 news + 247 episodio + 10 último-momento**. Test is ~48% the educational "episodio" series vs ~1.5% in train → a **domain/genre shift**, not leakage.
- Grouping is by parent video only. Weekly-news episodes from the same program and season sit in different splits, and likely share interpreters, studio and recurring phrases. So the split is **not signer-independent**: parent-level leakage is zero, but interpreter overlap across splits is likely. **INFERRED** (signer identity isn't stored in the H5; `infered_signer` was only in the deleted meta.csv).
- Still not recoverable from the H5: per-clip start/end timestamps, caption cues, original caption source (none of these were ever written to the H5).

Commands:
```bash
PY=/shared/Code/nakato/projects/VQ_RL/.venv/bin/python
$PY artifacts/dataset2_audit/scripts/h5_provenance.py > artifacts/dataset2_audit/_work/01_h5_provenance.json
$PY - <<'X'   # inline: count frame_count==1400; oracle label==H5 label; family counts per split; parents in keys<600
X
```
