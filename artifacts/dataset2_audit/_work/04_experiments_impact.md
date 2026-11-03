# 04 — Data conditions of v116–v119 (and context v125/v126/E1) and possible dataset2 impact

Audit spec section 7 (feeds final report §10). Date 2026-10-19. Branch `v125-temporal-prompt-alignment`, HEAD `a7db233`.
Read-only: no training, no inference, HDF5 opened with `h5py.File(p, "r")`. Tags: **VERIFIED** (I inspected or recomputed it), **DOCUMENTED** (stated in docs, configs, memory or reports and not re-run), **INFERRED** (my interpretation), **UNKNOWN**.

Helper code written for this section (under `artifacts/dataset2_audit/scripts/`):
- `tb_scalars.py`: a stdlib-only TensorBoard scalar reader (no CRC check).
- `experiment_split_leakage.py`: rebuilds the historical clip-level splits and writes `_work/04_experiments_split_stats.json`.

Interpreter: `/shared/Code/nakato/projects/VQ_RL/.venv/bin/python` (h5py 3.16, torch 2.13 CPU). The Sign-env directory only contains `conda-meta/`, so it has no usable python (VERIFIED, `ls ~/miniconda3/envs/Sign-env`).

---

## 0. Key findings (summary)

1. **VERIFIED, new: the v117 run is bit-identical to v116 during training.** I compared the TensorBoard logs of `116/1/15-06-2026-19-58-08` and `117/1/16-06-2026-05-42-59` at every common (tag, step):
   - 350 of 354 scalars are exactly equal: Loss/train, Loss/val, val_token_acc, effective_rank and pairwise_cosine at every epoch.
   - The only differences are in `Gen/val_chrf`. v116 evaluated generation on 4 batches; v117 used all of val (`gen_val_batches` 4 vs 999 in the configs).
   - In v117, `Loss/train_infonce = 0` in all 25 of 25 epochs.
   - Conclusion: InfoNCE added neither loss nor gradient. v117 is v116 with a different checkpoint-selection and evaluation protocol. It is not a contrastive experiment, so no conclusion about contrast, and no conclusion about how dataset2 affects contrast, can be drawn from v117. Command: §7.1.
2. **VERIFIED: v116 to v119 all used a random clip-level split (80/20), not a split grouped by source.**
   - The split is `random_split(self, [train_ratio, 1-train_ratio], Generator().manual_seed(42))` (`src/mslm/dataloader/keypoint_dataset.py:111-112`).
   - Rebuilding it gives exactly the historical sizes: 481/119 (600 clips) and 4481/1119 (5600 clips). The val pool of 1119 also matches the logged chance level of 1/1119 = 0.000894 in TensorBoard v118/8.
   - Under these splits, **100% of val clips share their `source_group` with train**: 64 groups at 5600, 62 at 600. Also, 66/1119 val clips (5.9%) at 5600 have a transcript that appears exactly in train (2/119 at 600).
   - Leakage of this kind inflates val metrics, so it **cannot explain the near-chance results**. It does invalidate any future positive result that is not re-tested on the grouped split (v125).
3. **VERIFIED (strong reconstruction): keys `0..599` of the current HDF5 are the historical 600-clip subset.**
   - 227 of them have more than 250 frames, plus 1 with exactly 250, for 228 in total.
   - Their mean frame count is 468.3 and the maximum is 1312.
   - This matches the memory note "228/600, mean 469, max 1312" (`memory/v118-truncated-video-bug.md`).
   - It also agrees with `backfill_dataset2_metadata.py:54-62`, which aborts if any stored label differs from `select_clips(5600, 23)`.
   - It **contradicts** the header of `experiments/v119_ctc/ctc_v119_subset1k.toml:2-6` ("keys 0..599 no longer correspond"). Status of that header claim: DOCUMENTED but contradicted.
4. **v116, v117 and v118 runs 1–3 were trained on clips truncated at 250 frames** (DOCUMENTED; the run dates come before the fix). That affected 228/600 clips (38%), and they were the longest sentences: mean 25.2 words vs 10.8 for the rest (VERIFIED on the current HDF5). Truncation is a real defect in the data supervision. It **cannot be the sole cause** of the failures: after the fix (v118 run 4), retrieval stayed at 2.94% R@1 (3.5× chance). §3.
5. **The "encoder without PE ⇒ bag-of-signs" bug belongs to VideoTokenDecoder/AR-base on dataset1. It does not apply to v116–v119.** Their encoders carry temporal order through RoPE, TCN or BiLSTM (VERIFIED in code, §6). It is not a candidate explanation for v116–v119.
6. **v118 does not fit even its own train set.** After training, train R@1 is 0.73% on 481 clips (3.5× chance), the same relative-to-chance level as val (`outputs/diag_v118_train_val_gap.json`, VERIFIED). Label noise or misalignment typically hurts generalization; it does not stop a 70 M-parameter model from memorizing 481 pairs. INFERRED: an optimization, architecture or global-pooling bottleneck is at least as plausible as a data cause. Isolating test: an overfit test on 32–64 pairs and on deliberately shuffled pairs.
7. **CTC (v119) uses Spanish word targets, which conflict with the data.** The objective assumes monotonic alignment between video and Spanish words. Several data properties work against that:
   - 53.6% of train word types appear only once (VERIFIED).
   - The 20 most frequent words, mostly function words (`de, la, que, el, en, y, a, los, se, es`), make up 35.9% of target tokens (VERIFIED). INFERRED: many probably have no manual sign in LSA (needs LSA review).
   - LSA and Spanish word order may differ (INFERRED).
   - The OOV floor is 7.2% of val tokens (VERIFIED).
   - None of these is isolated; the BiLSTM-collapse diagnosis does not separate them from data issues either.

---

## 1. Data actually used, per experiment

**File history.** One file, `/shared/Code/Sign-AI/data/processed/dataset_v6_unsloth.hdf5` (`src/mslm/utils/paths.py:48`), was mutated in place:
- **600 clips with a 250-frame cap.** Built by `build_dataset2_h5.py --n 600 --max-frames 250`. The 250 default is still at `scripts/data/build_dataset2_h5.py:256` in HEAD.
- **600 clips with 228 re-extracted at a 1400 cap**, on 2026-06-17 (DOCUMENTED memory; the backup `.bak_pre250fix` no longer exists in `data/processed/` — VERIFIED `ls`).
- **5600 clips** for v118f, built about 2026-06-19 by resumable append: keys already present are skipped (`build_dataset2_h5.py:101-104`). The new clips were extracted with a 1400 cap: 12 clips have exactly 1400 keypoint frames, all from videos longer than 1400 frames (VERIFIED, `_work/visual_integrity_per_clip.csv`).
- **v125 metadata backfill** (mtime 2026-06-23 10:49, VERIFIED `ls -la`). It adds `video_id`, `source_group`, `frame_count`, `token_count` and `word_count` (current groups, VERIFIED), and it does not rewrite keypoints (code, `backfill_dataset2_metadata.py:64-90`).

There is no hash of the file before v125, so byte-identity of the keypoints between v118f/v119 (June 19–21) and today is INFERRED from the append/skip code, not VERIFIED.

| Exp. | Dates (TB dir / commit) | HDF5 state | N / split | Video frames seen | Text target | Filter |
|---|---|---|---|---|---|---|
| v116 | TB 2026-06-15 (main run `19-58-08`); commits `1470d4a`, `be8e9f2` (06-15) | 600 clips, **pre-fix (≤250 frames)**: DOCUMENTED | 481/119 random clip split, seed 42 (VERIFIED by reconstruction) | ≤250 for 228/600 clips; trainer limit 4000 (`KeypointDataset max_length=4000`, `keypoint_dataset.py:42`) | Gemma-3n token IDs of the full Spanish label (CE-AR, frozen 4-bit Gemma, K=20 soft prefix) (`ce_ar_v116.toml`) | none |
| v117 | TB 2026-06-16 05:42; commit `eb97677` is dated 06-18, so the run used **uncommitted code** at the time (INFERRED from the date order) | same as v116 (VERIFIED: identical trajectories, §0.1) | 481/119 | same | same + InfoNCE prefix↔mean text embedding (inactive) | none |
| v118 run1 (InfoNCE) / run2 (+aug) / run3 (VICReg) | TB 2026-06-17 04:23 / 05:25 / 15:45; commit `07ec4cf` dated 06-18 | 600, **pre-fix** (DOCUMENTED; dates before the fix) | 481/119 | min(≤250, `max_frames` 768 or 1024) (`train_contrastive_v118.py:49-54`) | mean-pool of Gemma **input embeddings** (group `embeddings`) | none |
| v118 run4 (VICReg) | TB 2026-06-17 18:05 | 600 **post-fix** (DOCUMENTED, `contrastive_v118c.toml:9-12`) | 481/119 | first 1024 frames (4/600 clips are longer: VERIFIED) | same | none |
| v118e run7 | TB 2026-06-18 03:23; commit `35a1d89` (06-19) | 600 post-fix | 481/119 | first 1024 | **text_ctx**: last hidden state of bf16 Gemma-3n (`build_dataset2_h5.py:206-245`) | none |
| v118f run8 | TB 2026-06-19 15:37 | 5600 | 4481/1119 (VERIFIED) | first 1024 (63/5600 longer) | text_ctx | none: **includes the 413 later marked "truncated" by v125** |
| v119a subset1k (run 10) | TB 2026-06-20 03:41; commit `a48b9fe` is dated 06-20 20:38, after the run | 5600, `max_samples=1000` | 801/199 or 800/200 (contradictory per the prior audit [E08]) | first 1024 | Spanish **words** (train-only word vocab, blank=0, unk=1; `src/mslm/dataloader/vocab.py`) | UNKNOWN for that run. The HEAD toml was edited afterwards (header says first architecture, body has the replacement architecture's parameters) |
| v119a full | DOCUMENTED (`reports/report_2026-06-21.md:101-107`) | 5600 | 4481/1119 | first 1024 | words | none |
| v119 replacement baseline (`1_baseline_no_data_levers`) | TB 2026-06-20 23:02 | 5600 | 4481/1119 | first 1024 | words, dual CTC | none |
| v119 levers (run 1) | TB 2026-06-21 15:47 and 21:14 | 5600 → 5544 | 4436/1108 (VERIFIED by reconstruction) | first 1024 | words | `min_frames=16`, `frames//4 ≥ n_words`; excludes 56 (VERIFIED: 56 with frames-per-word < 4) |
| v125 | commits `d8923cf`…`5977e40` (06-23) | 5600 → 5138 kept | 4112/513/513 grouped by `source_group` (`dataset2_split_v125.json`) | n/a, no visual model | oracle using GT IDs | integrity filter |
| v126 / A1–A4 / E1 | 06-24 → 07-04 | **dataset1** `dataset1_isolated_v122.hdf5` (VERIFIED in `outputs/v126_temporal/*/config.json` and all 10 `outputs/video_token_decoder/*/*/config.json`) | stratified / LOSO | n/a | Spanish gloss tokens from dataset1 / restricted Gemma IDs | n/a |

---

## 2. Per-experiment impact table

| Exp. | Objective | Result (tag) | Known bugs (tag) | dataset2 issues that COULD matter | Issues that CANNOT explain the result | Isolating experiment |
|---|---|---|---|---|---|---|
| **v116** | CE-AR: soft prefix (K=20) → frozen Gemma-3n, teacher forcing on Gemma tokens | chrF max 20.67 at ep10 (5 evals); val token acc max 0.299 at ep20; prefix collapse with pairwise cosine 0.31 → 0.987 (ep10) and effective rank 25 → 5.1 (VERIFIED, TB `116/1/15-06-2026-19-58-08`). No grounding: retrieval@1 = 0/119 (DOCUMENTED for v117 checkpoints, script `/tmp/reeval_v117_gap.py` lost; it transfers to v116 because the trajectories are identical, INFERRED) | 250-frame truncation of 228/600 (DOCUMENTED); earlier CE-AR bugs fixed in `be8e9f2` (DOCUMENTED); 4-bit init, PLE not equivalent to v125 (DOCUMENTED [E05, E16]) | truncation (38% of pairs, biased toward long sentences); semantic misalignment and subtitle offset (UNKNOWN magnitude); small N (481) against a 262k-token output space; 22.7% val-token OOV relative to train words (VERIFIED, 600 scenario) limits what can be learned from 481 sentences | Duplicates and leakage: they would inflate val, and only 2/119 val labels are in train (VERIFIED). Missing PE: the encoder uses RoPE (VERIFIED). Truncation alone: 62% of pairs were untruncated and still showed no grounding (INFERRED) | Same model on untruncated clips with a short-sentence subset, plus a per-sample shuffled-pairs control with the same budget. If real ≈ shuffled ⇒ the data/objective carries no exploitable signal at this N. An overfit test on 32 pairs separates interface from data |
| **v117** | v116 + InfoNCE λ=0.1, τ=0.07 | chrF 20.60 at ep10, chrF with shuffled prefix 20.86 (≥ real) (VERIFIED TB); retr@1 = 0% (DOCUMENTED) | **InfoNCE inactive**: computed per sub-batch with `sub_batch_size=1` (`ce_ar_v117.toml:39`; `eb97677:src/mslm/training/trainer.py:165-170`); a 1×1 matrix gives CE = 0 and gradient 0; TB `train_infonce = 0` in 25/25. **Training identical to v116** (VERIFIED §7.1) | Same as v116 | **No dataset2 issue can be invoked as an explanation of "InfoNCE did not help"**: the objective never acted. Any data effect is the v116 effect | No new experiment needed to explain v117. To test contrast: InfoNCE with real B > 1, gradient logged non-zero, on the v125 grouped split |
| **v118 (runs 1–3)** | Global CLIP-style InfoNCE / +aug / VICReg, no LM; text = mean of Gemma input embeddings | val R@1 max 2.52% / 1.68% / 2.52% out of 119 (chance 0.84%); R@10 ≤ 12.6% (VERIFIED TB `118/{1,2,3}`) | 250-frame truncation (DOCUMENTED); the text space is nearly isotropic (mean cosine between sentences 0.80, nearest neighbour 0.90; DOCUMENTED `build_dataset2_h5.py:211-218`, not recomputed); retrieval rank uses strict `>` so ties get rank 1 (DOCUMENTED [E07]) | truncation; weakly discriminative text target; misalignment and offset | Leakage (it inflates). Duplicates in the val pool: 0 repeated labels inside the 119 (VERIFIED) | See run 4 (it already isolated truncation, with n = 1 run) |
| **v118 run4** | VICReg, untruncated data, max_frames 1024 | val R@1 2.94% (ep24), R@10 14.3%, median rank 44–46 (VERIFIED TB `118/4`); train R@1 0.73%/481 (VERIFIED diag JSON) | none new; one seed | misalignment, offset, semantic omissions; global pooling (`contrastive.py:66` `prefix.mean(dim=1)`) washes out local content | **Truncation as the dominant cause**: fixing it moved R@1 only from 2.52% to 2.94%, one seed, no CI (VERIFIED/INFERRED). Label noise alone does not explain that train itself is not fitted (INFERRED) | (a) Overfit test on 32/64 pairs. (b) True pairs vs pairs shuffled within the batch, same seeds. (c) Repeat on a human-certified subset (§9 of the spec) against a random subset of the same size |
| **v118e run7** | VICReg + text_ctx (contextual Gemma) | R@1 2.52% at ep6, R@10 15.1% at ep16, median rank 37–39 (VERIFIED TB `118/7`) | — | same as run 4 | **Text anisotropy as the sole cause**: changing the text target did not break the ceiling (VERIFIED/INFERRED) | Same as run 4 |
| **v118f run8** | Same as v118e, 5600 clips | R@1 max 0.268%/1119 (3.0× chance), R@10 1.88% (2.1× chance), median rank ~373–501 (VERIFIED TB `118/8`); 16 epochs logged, 20 configured | includes v125-"truncated" clips (keypoint/video frame ratio < 0.9: 56 among keys < 600 and 378 among ≥ 600 per `_work/visual_integrity_per_clip.csv`, recomputed by the sibling agent; 413 stored as first cause) | the ~7.7% partially extracted clips; misalignment and offset; 27 val clips with repeated labels inside the pool create false negatives (VERIFIED) | False negatives from 27 duplicates: at most 2.4% of the pool, they cannot push R@1 down to the observed level (INFERRED). Leakage of 66 exact labels and 100% groups: it inflates | v118e on the v125 kept/grouped split; report R@k against the chance of the pool; deduplicate labels in the pool |
| **v119a (first: STGCN/RoPE, one CTC head)** | CTC on Spanish words | subset1k: train loss 32.3 → 1.26 (memorizes), val WER 99.7–100% (VERIFIED TB `119/10`). Full 5600: train 26 → 1.87, WER ~99.5% (DOCUMENTED `reports/report_2026-06-21.md:101-107`) | HEAD toml for run 10 is not the executed config (VERIFIED by contradiction in the header); split counts contradictory (DOCUMENTED [E08]) | monotonicity Spanish ↔ LSA (INFERRED); function words with no manual sign (INFERRED; 35.9% of tokens in the top-20, VERIFIED); 53.6% singleton types (VERIFIED); OOV 7.2% (VERIFIED); offset at clip edges; truncation at 1024 (63 clips, 1.1%) | Memorizing train while WER ≈ 100% rules out "the data contains no usable signal for fitting". The failure is generalization (INFERRED). OOV alone sets a 7.2% floor, not 99% | CTC on dataset1 synthetic sequences (monotonic by construction) with the same encoder, against dataset2. Also a "content-words-only" target, and a model whose objective does not need monotonicity (seq2seq) on the same certified split |
| **v119 replacement (GCN → TCN/TLP → BiLSTM, dual CTC)** | dual CTC + TLP auxiliaries | best val WER 99.25% at ep15 → 100.23% at ep39; train loss 33.9 → 9.9 (does not memorize) (VERIFIED TB `119/1_baseline_no_data_levers`). Posteriors: 98.2% of frames argmax = blank over 300 clips (DOCUMENTED); the current summary covers 5 clips at 99.8% (VERIFIED `outputs/diagnostics/119/1/summary.txt`). Temporal activity drops at the BiLSTM, 0.54 → 0.21/0.27 (VERIFIED summaries) | `BatchSampler` length grouping is a no-op (DOCUMENTED `report_2026-06-21.md:194-199`); vocab.json of the baseline overwritten (DOCUMENTED); 2 bugs fixed in the levers branch (DOCUMENTED memory) | same as v119a | "BiLSTM flattens ⇒ the cause is not the data" is not proven: a collapse to blank is also the optimum under targets that cannot be aligned (INFERRED). Neither is "5.6× data refutes scarcity" (scaling the same collection does not test target quality) | Same as v119a, plus training on a synthetic dataset where target alignment is guaranteed. If it learns there and not on dataset2, the data/target is implicated |
| **v119 levers** | + filter + aug ×5 (+ temporal drop) | WER 99.72% at ep2 → 104.9% at ep12, stopped manually (VERIFIED TB `119/1/21-06-2026-15-47-49`); second run 5 epochs, val_long min 99.69% (VERIFIED) | stopped at ep12 | same | The filter only removes CTC infeasibility (frames//4 < n_words); it does not test semantic alignment. "Filters refute noise" is not supported (INFERRED) | — |
| **v125 (context)** | integrity + grouped split + text oracle | 5138/5600 kept; oracle chrF 92.86 with GT text (DOCUMENTED/VERIFIED by the prior audit [E15, E16]) | — | — | It certifies neither semantics nor sync, and makes no visual claim | — |
| **v126 / A1–A4 / E1 (context)** | CIF / AR on **dataset1** (synthetic concatenations) | E1 strict exact 12.5 / 14.06% on folds 7/8 (VERIFIED by the prior audit [E12]) | encoder without PE ⇒ bag-of-signs in AR-base (DOCUMENTED `memory/ar-decoder-order-blind-root-cause.md`); contaminated LOSO annulled (DOCUMENTED) | **None from dataset2**: they do not use it (VERIFIED configs) | Any dataset2 issue | n/a. Their success does not transfer to dataset2 either: synthetic targets are monotonic and gloss-aligned by construction |

---

## 3. Analysis by data issue

| dataset2 issue | Evidence of the issue (tag) | Retrieval (v118) needs | Generation (v116/117) needs | CTC (v119) needs | Assessment |
|---|---|---|---|---|---|
| Truncation at 250 (historical) | 228/600; 25.2 vs 10.8 mean words (VERIFIED on the reconstructed subset) | partial global match; tolerates partial video | full content to generate the full sentence | the end of the target has no video ⇒ infeasible alignment | Affects v116, v117 and v118 runs 1–3. Fixing it did not unlock v118 (run 4). Not sufficient on its own (INFERRED) |
| Current partial extraction (keypoint/video ratio < 0.9) | 413 stored / ~434 recomputed; median ratio 0.55–0.63 (VERIFIED from the sibling's CSV); cause outside this scope (see 02_visual) | moderate | high | high | Present in v118f and v119 (no v125 filter). ~7.7% of pairs: cannot explain WER ≈ 100% or chance-level R@1 on the remaining 92% (INFERRED) |
| Tail cut by `max_frames=1024` in the trainer | 63/5600 clips > 1024 (VERIFIED); `_truncate` keeps the first N frames (`train_ctc_v119.py:77-81`, `train_contrastive_v118.py:49-54`) | low | medium | high for those 63 | Marginal (1.1%) |
| Temporal offset of subtitles | **Not measured.** The "3,088 > 1 s" figure has no recoverable procedure (spec) | tolerable if overlap is the majority | medium | **critical** at clip edges: signs from the neighbouring sentence | UNKNOWN magnitude. Isolate with human annotation of offset (Level B) and re-evaluation by offset stratum |
| Semantic misalignment (interpreter summarizes or omits) | UNKNOWN (needs LSA review) | degrades positives | degrades | degrades | Plausible for all; not quantified |
| Duplicate transcripts and leakage | 5600 split: 66/1119 val labels in train; 27 val clips with a repeated label; 100% of val groups present in train (VERIFIED) | inflates (leakage) / false negatives (duplicates in the pool) | inflates | inflates | **Cannot explain failures**; it biases upward. Requirement: grouped split (v125) for any new result |
| Short clips vs long sentences (frames per word) | median 16.7 frames/word (~0.56 s at 29.97 fps), p5 8.45; 234 clips < 8 frames/word; 56 < 4 (VERIFIED) | irrelevant | relevant for prefix capacity (K=20 fixed against p95 37 words; Gemma tokens per word not computed here, UNKNOWN) | after //4: median ~4.2 steps per word; 56 infeasible (removed by the filter) | Feasible in most clips; tight for ~4% (INFERRED) |
| CTC monotonicity vs LSA ↔ Spanish order | linguistic: LSA does not follow Spanish order (INFERRED; needs an expert); 35.9% of tokens are top-20 function words (VERIFIED) | not required | not required (seq2seq reorders) | **required** | Structural risk specific to v119. Isolate: CTC on content words, or on LSA glosses annotated on a sample (glosses only as a diagnostic, given the product restriction in `memory/ar-decoder-order-blind-root-cause.md`) |
| Lexical sparsity | 9,878 types, 53.6% singleton; top-20 = 35.9% of tokens; val OOV 7.2% (VERIFIED) | not applicable | Gemma BPE mitigates | per-word classes with 1–2 examples | Plausible for v119 generalization; does not imply that SLT is impossible (prior audit §7.4.5) |

---

## 4. What would isolate it (minimum, no new architecture)

1. **Overfit / shuffled-pairs control (v118, v119 replacement):** 32–64 train pairs, true vs shuffled, same seed and budget.
   - If neither true nor shuffled pairs fit, the problem is model or optimization.
   - If both fit, generalization depends on data.
   - If only the true pairs generalize to val, the signal exists.
2. **Same pipeline on the v125 grouped split and kept set:** removes leakage and partial extraction. It does not test semantics.
3. **Human-certified subset (spec §9) against a random subset of the same size, same split:** isolates semantic misalignment and offset. It is the only test that answers "data vs model" for dataset2.
4. **Monotonic synthetic CTC (dataset1) with the v119 encoder:** if CTC learns there, v119's failure points to the target or alignment of dataset2 (monotonicity, offset, function words), not to the BiLSTM alone.
5. **Contrast with real B > 1 and gradient logged non-zero:** the first valid test of what v117 claimed to test.

---

## 5. Contradictions found

| Sources | Resolution |
|---|---|
| `ctc_v119_subset1k.toml:2-6` ("keys 0..599 no longer = the 600") vs reconstruction (228 with >250 frames, mean 468, max 1312) + `backfill` that aborts on mismatch | The evidence favours identity (VERIFIED counts; identity INFERRED, strong) |
| Memory: truncated clips "25 words vs 16 for the rest" vs computation: 25.2 vs 10.8 (mean) | Probably a different statistic (median?). Use the recomputed value |
| `report_2026-06-21.md:188` "~480–580 train clips" vs the run with 4481 train | That figure belongs to v118 at 600; do not use it for v119 |
| Posteriors "300 clips, 98.2% blank" (report) vs current summary of 5 clips at 99.8% | The summary was overwritten; the 300 figure is DOCUMENTED |
| v117 "InfoNCE delayed collapse by ~9 epochs" (memory `v117-no-grounding-verdict.md`) vs bit-identical trajectory to v116 | **Refuted** (VERIFIED §7.1) |
| v117 date (TB 06-16) vs commit `eb97677` (06-18); v119 run 10 (06-20 03:41) vs `a48b9fe` (06-20 20:38) | The code that ran is not exactly identifiable with a commit (INFERRED) |

## 6. Encoder positional information (rules out "bag-of-signs" for v116–v119)

- v116/v117/v118 use `PrefixImitator`, which is the Imitator with `TransformerEncoderLayerRoPE` (`src/mslm/models/imitator.py:4,53-54`; `contrastive.py:5,58`). v118 projects `prefix.mean(dim=1)` (`contrastive.py:66`): global pooling, which is order-agnostic by design for retrieval.
- v119a: STGCN + `TransformerEncoderLayerRoPE` (`a48b9fe:src/mslm/models/ctc_encoder.py`).
- v119 replacement: TCN + BiLSTM (`src/mslm/models/ctc_encoder.py:68-75`). Both are order-aware.
- The no-PE defect was in `STGCNTemporalFrameEncoder` → VideoTokenDecoder on dataset1 (DOCUMENTED memory and `experiments/video_token_decoder/AUDIT_REPORT.md`).

## 7. Reproduction commands

### 7.1 v116 ≡ v117 (TensorBoard)

```bash
cd /shared/Code/Sign-AI/outputs/reports
python3 - <<'EOF'
import sys; sys.path.insert(0,'/shared/Code/Sign-AI/Sign-chris/artifacts/dataset2_audit/scripts')
from tb_scalars import scalars
a={(t,s):x for t,s,x in scalars('116/1/15-06-2026-19-58-08/events.out.tfevents.1781553488.fischl-ubuntusv.3096757.0')}
b={(t,s):x for t,s,x in scalars('117/1/16-06-2026-05-42-59/events.out.tfevents.1781588579.fischl-ubuntusv.120820.0')}
c=[k for k in a if k in b]; print(len(c), sum(a[k]==b[k] for k in c), {k[0] for k in c if a[k]!=b[k]})
EOF
# -> 354 350 {'Gen/val_chrf'}
python3 /shared/Code/Sign-AI/Sign-chris/artifacts/dataset2_audit/scripts/tb_scalars.py 117 infonce
# -> Loss/train_infonce: n=25 ... max=0
```

### 7.2 Splits, leakage, frames per word

```bash
cd /shared/Code/Sign-AI/Sign-chris
HDF5_USE_FILE_LOCKING=FALSE /shared/Code/nakato/projects/VQ_RL/.venv/bin/python \
  artifacts/dataset2_audit/scripts/experiment_split_leakage.py   # -> _work/04_experiments_split_stats.json
```

Caveat: torch 2.13 here vs 2.8 historically. The exact match of the split sizes (481/119, 4481/1119) supports the reconstruction, but does not prove that the per-index permutation is identical (INFERRED).

### 7.3 Other metrics

```bash
python3 artifacts/dataset2_audit/scripts/tb_scalars.py /shared/Code/Sign-AI/outputs/reports/118 R@1 R@10 median
python3 artifacts/dataset2_audit/scripts/tb_scalars.py /shared/Code/Sign-AI/outputs/reports/119
```

## 8. Limitations

- I did not re-run v117's grounding re-eval (script lost, no checkpoints): retr@1 = 0% stays DOCUMENTED.
- The 600-clip file (pre-fix and post-fix) no longer exists. Statistics for the "600" scenario use the current frame counts (post-fix), not those the model saw in v116, v117 and v118 runs 1–3.
- The linguistic claims (function words without a manual sign, LSA order) are INFERRED and need an LSA expert.
- No causal claim in this document has an isolating experiment behind it. §4 lists the ones that would.
