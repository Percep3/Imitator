# 03 — Textual integrity and partition/leakage integrity of dataset2

Scope: spec section 4 ("Integridad textual", "Integridad de particiones") and report section 8 (leakage).
Date: 2026-10-19. Everything was read-only. No training, no GPU, no downloads, no installs.

## 1. Method and commands

- Script: `artifacts/dataset2_audit/scripts/text_split_audit.py` (standalone; includes a `_selfcheck()` with asserts, which runs before `main()`).
- Python env: **Sign-env is unusable on this host.** `~/miniconda3/envs/Sign-env/` contains only `conda-meta/` (VERIFIED, `ls`). No env with h5py existed except `/shared/Code/nakato/projects/VQ_RL/.venv` (Python 3.13.13, h5py 3.16.0, numpy 2.3.1, torch 2.13.0+cu130 used on CPU only, tokenizers 0.22.2). I used that env read-only and installed nothing.
- Command (runtime ~75 s on CPU):
  ```
  cd /shared/Code/Sign-AI/Sign-chris
  /shared/Code/nakato/projects/VQ_RL/.venv/bin/python artifacts/dataset2_audit/scripts/text_split_audit.py
  ```
- Outputs: `_work/text_split_per_clip.csv` (5,600 rows), `_work/text_split_stats.json`.
- Inputs and hashes. Each hash was recomputed and matches `data/processed/v125_artifacts.json` (VERIFIED):
  - `dataset_v6_unsloth.hdf5` sha256 `24c9aa40…1d38` (mtime 2026-06-23 10:49)
  - `dataset2_manifest_v125.json` sha256 `04c3a7c1…bad`
  - `dataset2_split_v125.json` sha256 `6669635a…f9` (mtime 2026-06-23 10:53)
  - `gemma3n_embed_table.pt` sha256 `20437971…d7` (fp16, 262400×2048)
  - Tokenizer: local HF cache `models--unsloth--gemma-3n-E2B-it-unsloth-bnb-4bit/snapshots/3d26ffdd…/tokenizer.json`, sha256 `b6c35ee6…a5d`, which equals the frozen hash. Loaded with `tokenizers.Tokenizer.from_file`. No network access.
- Pipeline text functions were imported **by file path** so the audit uses the exact same normalization: `src/mslm/dataloader/alignment_text.py:9-13` (`normalize_alignment_text`), `src/mslm/dataloader/vocab.py:17,20-30` (`tokenize`), `src/mslm/dataloader/grouped_split.py:13-44`.

## 2. How the text, token and split arrays were produced (code provenance)

| Step | Code | Notes |
|---|---|---|
| Clip selection plus label | `scripts/data/build_dataset2_h5.py:33-43` | Reads `raw/dataset2/meta.csv`, drops empty labels (l.35), **lowercases the label (l.41)**, shuffles with seed 23 and takes the first n (5,600). Labels are written at l.111-112. |
| Original embeddings | `build_dataset2_h5.py:158` | `tok(label)`, with special tokens. **Superseded** by the v125 rebuild. |
| v125 targets rebuild | `scripts/data/rebuild_dataset2_alignment_targets_v125.py:42-69` | `token_ids` = Gemma tokenization of `normalize_alignment_text(label)` with `add_special_tokens=False`. `embeddings` = `table[ids]`. `token_count` = len(ids). The label keeps its original form. |
| Metadata backfill | `scripts/data/backfill_dataset2_metadata.py:157-164, 184-225` | `video_id` = meta `id`; `source_group` = meta column `video`; `word_count` = `vocab.tokenize(label)`. |
| Manifest | `scripts/data/build_dataset2_manifest.py:54-118` + `src/mslm/dataloader/integrity_filter.py:272-295` | kept 5,138 / 5,600. Causes: truncated 413, min_tokens 29, frames_per_token 16, min_frames 4. Re-verifying the causes is outside this scope. |
| v125 split | `scripts/data/build_dataset2_split.py:14-24` → `grouped_split.py:13-44` | Whole `source_group`s are assigned to train/val/test by a greedy bin-packing. Groups are sorted by size, largest first, and each group goes to the split with the largest absolute deficit (l.40). The comment at l.39 says "relativo", but the code computes an absolute deficit. Seed 23, ratios 0.8/0.1/0.1. |

**Provenance gap (VERIFIED):** `raw/dataset2/meta.csv` no longer exists. Only `videos/` (8,459 files) remains. `/shared/Code/Sign-AI/DATA_CLEANUP_2026-09-10.md` records that everything in `raw/dataset2` except `videos/` was removed. As a result, `source_group`, `video_id` and the labels **cannot be re-derived from the source metadata today**. They can only be checked for internal consistency inside the HDF5. The EDA run before the cleanup (`artifacts/data_eda/REPORT.md:16,28`) reported 8,459 meta rows, 8,016 unique labels and 8,459/8,459 id matches (DOCUMENTED).

## 3. Text integrity results

### 3.1 Presence, keys and shapes (VERIFIED, `consistency_failures` in the stats JSON)
- The HDF5 has only the `dataset2` group, with 10 subgroups (`embeddings, frame_count, keypoints, labels, source_group, text_ctx, token_count, token_ids, video_id, word_count`). Each has **5,600 keys**. The key sets are identical across subgroups and contiguous `0..5599`.
- 0 missing labels and 0 empty labels.
- The following checks each have **0 failures over 5,600 clips**:
  - `token_ids` == re-tokenization of `normalize_alignment_text(label)` with the cached tokenizer
  - `token_count` == len(`token_ids`)
  - `embeddings.shape[0]` == `token_count`, and the dimension is 2048
  - `embeddings` == `table[token_ids]` (atol 1e-3)
  - `text_ctx` has shape (1, 2048)
  - `frame_count` == keypoints T
  - keypoints have shape (T, 134, 2) (not 111; the reduction to 111 happens in the dataloader)
  - `word_count` == `vocab.tokenize(label)`
  - `video_id` prefix == `source_group`
- Caveat: `text_ctx` was computed from the raw label with `unsloth/gemma-3n-E2B-it` (`build_dataset2_h5.py:171, 240`). I did not recompute it because that would need a forward pass of the model.

### 3.2 Length distributions (VERIFIED). Columns are all 5,600 clips / the 5,138 kept clips.
| | p1 | p5 | p50 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| words (`vocab.tokenize`) | 1 / 2 | 3 / 3 | 13 / 13 | 37 / 38 | 52 / 52.6 | 153 / 111 | 16.1 / 16.3 |
| Gemma tokens, normalized text (= stored `token_ids`) | 2 / 2 | 4 / 4 | 16 / 16 | 46 / 46 | 64 / 63.6 | 169 / 122 | 19.5 / 19.7 |
| Gemma tokens, raw label (generative target) | 3 / 4 | 6 / 6 | 18 / 18 | 50 / 50 | 69 / 69 | 196 / 137 | 22.1 / 22.3 |

- Very short labels (≤2 words): 167 clips in total, 131 kept. Examples: "¡gracias!", "chau.", "comencemos".
- Very long labels (>40 words): 194 clips in total, 183 kept. The longest has 153 words (clip 4235, excluded).

### 3.3 Normalization and quality flags (VERIFIED counts, all / kept)
- **Casing information is lost.** No label contains an uppercase letter (0/5,600) because the build lowercased everything (`build_dataset2_h5.py:41`). Proper nouns and acronyms are lowercased, for example "cn sordos", "auh", "gcba".
- Mojibake: 0. Non-NFC strings: 0. Control characters: 0. Non-Latin letters: 0.
- Digits: 844 / 806 clips, 1,237 numeric tokens in total, including phone numbers ("11-5050-0147"), percentages and decimals. The two normalizers handle these differently:
  - `vocab.tokenize` removes the punctuation and glues the pieces ("1,5"→"15", "12.7%"→"127", "covid-19"→"covid19").
  - `normalize_alignment_text` replaces punctuation with a space and splits ("1 5", "covid 19").
  - **253 clips** give a different word sequence under the two normalizers. This is a WER/vocabulary vs. alignment-unit inconsistency.
- Quotes or brackets: 506 / 477. Parenthesized expansions of acronyms are present ("(auh)", "(reprocann)", "(open forum infectious diseases)").
- Speaker or meta annotations inside the text: for example "ambos: chau". The string "ambos:" occurs in 11 clips. These are not signed content (INFERRED).
- Leading or trailing whitespace: 29 / 20. Internal double spaces: 9 / 8. Unbalanced ¿?/¡!: 25 / 17. Ellipsis: 40 / 38. No final punctuation: 54 / 44. Hyphenated compounds: 34 / 32.
- INFERRED: the register is subtitle-style written Spanish (numbers, acronyms, parentheticals). It is not a gloss transcription. Many tokens (phone numbers, parenthesized expansions) are unlikely to be signed one-to-one. Confirming this requires human LSA review.

### 3.4 Duplicate transcripts (VERIFIED)
| | distinct texts | texts appearing more than once | clips sharing them | redundant clips |
|---|---|---|---|---|
| all, exact | 5,382 | 167 | 385 | 218 |
| all, normalized | 5,348 | 174 | 426 | 252 |
| kept, exact | 4,949 | 155 | 344 | 189 |
| kept, normalized | 4,925 | 163 | 376 | 213 |

Most duplicates are program formulae: "empezamos empezamos" (7), "comencemos" (7), "hola cómo estás" (6), "mi nombre es lucía/lautaro", "nos vemos el próximo domingo ambos chau". The dataset as a whole is ~96% unique sentences. A text-memorizing decoder therefore gets little benefit from train/test text overlap (see §4.3).

## 4. Partition and leakage results

### 4.1 Split sizes and how they are produced (VERIFIED)
- The sizes are **4,112 / 513 / 513**, which confirms the prior claim. The JSON contains `seed=23` and `ratios=[0.8,0.1,0.1]`.
- There are 0 shared clip ids between any pair of splits and no duplicate ids inside any split list.
- The union of the three splits equals the 5,138 manifest-kept ids exactly.
- **Re-running `make_grouped_split(kept_ids, source_groups, seed=23)` in memory reproduces the stored split exactly.**
- There are 64 source groups: 40 in train, 12 in val, 12 in test.

### 4.2 Identity leakage (VERIFIED)
- `source_group`: 0 groups span more than one split.
- `video_id`: 5,138 distinct ids, each one clip, 0 spanning splits.
- Exact duplicate keypoint arrays (sha1 of the raw arrays): **0** duplicate groups.
- Quantized motion-signature hash: 0 duplicate groups. Length + mean-position signature: 0 collisions.

### 4.3 Text leakage across splits (VERIFIED)
- Exact label also in train: val 13/513 (2.53%), test 6/513 (1.17%).
- Normalized label also in train: val 17/513 (3.31%), test 8/513 (1.56%). Val and test share 5 normalized texts.
- Cross-split duplicates with 6 or more words: 10 clips (6 val, 4 test). All are program formulae ("hola bienvenidos a una nueva edición de cn sordos ecología", "vieron toda la edición les gustó", "nos vemos el próximo domingo ambos chau").
- Risk: low. A decoder that memorizes train sentences gains at most ~3% exact hits on val and ~1.5% on test, and these are greetings and openings.

### 4.4 Near-duplicate footage (VERIFIED computation; the interpretation is INFERRED)
- Method: per clip, the keypoints are scale-normalized, the per-joint temporal mean is removed (so the signature encodes motion, not framing), the sequence is resampled to 32 frames and L2-normalized. I then searched exhaustively for each clip's nearest neighbour by cosine.
- Nearest-neighbour cosine: p50 0.41, p99 0.87, max 0.957. Random pairs: p50 0.00, p99 0.26.
- Pairs at or above 0.95: 2 pairs, **0 crossing splits**.
- Pairs with the same normalized text in different splits (52 pairs): max cosine 0.716, 0 at or above 0.95. These look like re-signed formulae, not re-used footage.
- **Pairs with the same text in different source groups (427 pairs)**: 5 at or above 0.8, all inside train. For example, clip 2407 ↔ 3827, "había asesores también.", cosine 0.957, from `leyfederalsa-resumen-de-la-mesa-de-trabajo-region-patagonia` ↔ `lsa-noticias-…-programa-especial-20062021`. Similarly, 1529 ↔ 5324 (programa-especial ↔ resumen-semanal-semana-santa-04042021) has cosine 0.908. **Footage re-use across source groups does exist** (a "programa especial" compiles earlier segments). In this split it happens to stay within train.
- Limits of the method:
  - Re-encoded or re-cropped re-broadcasts change the keypoints, so exact hashes would miss them.
  - The cosine signature misses temporal sub-clip overlap (one clip contained inside a longer one) and shifts in where the clip boundaries fall.
  - The 0.95 threshold is heuristic. It is not calibrated against labelled duplicates.
  - Absence of evidence is not proof that no near-duplicates exist.

### 4.5 Is `source_group` an adequate independence unit? (INFERRED, partly VERIFIED)
- What each group is (VERIFIED from the names): one YouTube-style program or video. The groups are 49 "resumen semanal / programa especial" CN Sordos news editions (2020-11-29 … 2021-12-19), 9 "CN Sordos ecología" episodes, and 6 one-off items.
- **No group spans splits, but the same program series, studio and presenters span all three splits.**
  - Weekly news editions are interleaved in time. For example, val has 2020-12-20 and 2021-02-21, while train has 2020-12-13, 2020-12-27, 2021-02-14 and 2021-03-07.
  - The eco series is split 2 train / 2 val / 5 test episodes.
  - Presenter self-introductions ("mi nombre es lucía", "lautaro", "luciana") occur in more than one split. So the **same signers very likely appear in train and test** (INFERRED).
  - Signer identity is **UNKNOWN**. There is no signer metadata, and `meta.csv` has been deleted.
- Topical and temporal leakage is plausible: news stories continue across consecutive weeks, and the one-off item "último momento sismo San Juan 18-01-2021" is in val while the 2021-01-17 and 2021-01-24 weekly editions are in train. This was not quantified beyond the text and motion checks.
- So the v125 split is **signer-dependent and series-dependent, but episode-independent**. It measures generalization to new episodes from the same producers and interpreters. It does not measure generalization to new signers.
- **Domain imbalance (VERIFIED composition; the cause is INFERRED to be the size-sorted greedy):**

  | split | news clips | eco clips | other clips |
  |---|---|---|---|
  | train | 4,005 | 60 | 47 |
  | val | 407 | 94 | 12 |
  | test | 256 | 247 | 10 |

  Eco episodes are 1.5% of train but 48% of test, so test scores mix in a domain shift. The algorithm places the large news groups first, and they go to train; the small eco groups are left for val and test.

### 4.6 Historical v116–v119 splits (VERIFIED code; the replication is INFERRED)
- v116–v119 did **not** use the grouped split. `scripts/train/train.py:71` → `src/mslm/utils/setup_train.py:38-39` and `scripts/train/train_ctc_v119.py:193-218` all call `KeypointDataset.split_dataset` (`src/mslm/dataloader/keypoint_dataset.py:111-112`), which is a **clip-level `random_split([0.8,0.2], seed 42)` with no test split**.
- Replication of v119's `valid_index` (min_frames=16, `filter_invalid_labels`, key order at `keypoint_dataset.py:90-103`): it gives **5,544 clips, which matches `outputs/logs/ctc_v119_*.log` ("Videos: 5544")**.
- Running `random_split` on those clips gives train 4,436 / val 1,108. In that split, **63 of 64 source groups span train and val**, all 1,108 val clips come from a group that is also in train, and 75 val clips (6.8%) have a normalized text that also appears in train.
- Caveat: this assumes `torch.randperm` with seed 42 behaves the same in torch 2.8 and 2.13 (INFERRED).
- v116–v118 used a different, smaller HDF5 state (for example, `diag_v118_train_val_gap.json` implies 481 train / 119 val). The comment at `train_ctc_v119.py:204-209` states that keys 0..599 are not the original 600 clips, so I could not reconstruct those splits. Even so, the code path is the same clip-level random split, so the same kind of episode-level leakage applies to them (INFERRED).
- **Validation metrics reported for v116–v119 are within-episode (and within-signer) estimates.** They are not comparable to the v125 grouped split.

## 5. Proposal (no split was modified)
The v125 split is internally correct and reproducible, and it is episode-disjoint. I do not recommend replacing it. If a stricter evaluation is wanted, these are additional proposals only:
1. **Stratified group split.** Assign groups per series (news / eco / other) in proportion, so test is not 48% eco. Or report test metrics separately per series.
2. **Temporal holdout for news.** For example, all editions dated 2021-10 or later become test. This reduces the risk of story continuation across splits.
3. **Merge groups with demonstrated footage re-use.** `programa-especial-20062021`, `leyfederalsa-…-patagonia` and `resumen-semanal-semana-santa-04042021` showed re-use (§4.4), so they should be treated as one independence unit. They are all in train today, so the current split is unaffected.
4. **Signer-independent evaluation.** This needs signer labels, which do not exist. They would have to be annotated by humans (at least presenter identity per group).

## 6. Limitations
- `meta.csv` has been deleted, so labels, `video_id` and `source_group` are verified only for internal consistency against the HDF5, not against the source metadata.
- Signer identity is UNKNOWN. Story-level (topic) leakage was not measured.
- The near-duplicate detection is heuristic (§4.4).
- `text_ctx` was not recomputed.
- The env is not Sign-env (it is gutted). Numerical results do not depend on library versions, except the torch `randperm` replication in §4.6.
- None of these checks assess whether the Spanish text is a correct translation of the signing. That requires human LSA review.
