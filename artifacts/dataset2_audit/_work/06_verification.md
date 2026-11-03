# 06 Independent verification (2026-10-19)

Verifier: independent agent (Claude, AI-generated; validate with domain experts). Read-only on all data. Interpreter: `/shared/Code/nakato/projects/VQ_RL/.venv/bin/python`; TB events parsed with a separately written stdlib TFRecord parser (not `tb_scalars.py`).

## A. Numeric claims (recomputed from HDF5 / manifest / split / CSVs / TB / diag JSON)

| # | Claim | Location (report) | Recomputed value | Verdict |
|---|---|---|---|---|
| 1 | 5,600 clips, keypoints (T,134,2), 10 groups | §1.1, §4 | 5,600 in all 10 groups; shapes (134,2) 5,600/5,600 | OK |
| 2 | kept 5,138 / excluded 462 = 413 truncated + 29 min_tokens + 16 fpt + 4 min_frames | §1.2, §6 | manifest causes identical | OK |
| 3 | split 4,112/513/513, 40/12/12 groups, 0 shared, union = kept | §1.2, §8 | 4112/513/513; 40/12/12; 0/0/0 shared; union == kept | OK |
| 4 | 64 source_groups; 8,459 raw mp4 + 1 AppleDouble; 3–439 clips/parent | §3.1, §4 | 64; 8,459 + `._…05092021_56.mp4`; min 3, max 439 | OK |
| 5 | has_audio False in 8,459 | §5 | `video_metadata.csv`: 8,459/8,459 False | OK |
| 6 | T p5/p50/p95/max/mean = 65/215/711/1,400/278.7 | §5 | 65 / 215 / 711.05 / 1400 / 278.7 | OK |
| 7 | 12 clips T=1,400; 12 videos >1,400 frames all T=1,400; 7 excluded truncated, 5 kept (627,3642,3846,4467,4768) | §6, §14 | identical; ratios 0.93–0.999 | OK |
| 8 | 12 clips T=250 (11 nb_frames 250, clip 5285 = 270) | §14 | identical | OK |
| 9 | keys 0–599: 228 with T≥250, mean 468.3, max 1,312 | §3.8, §14 | 228 with T≥250 have mean **467.3**; 468.3 is the mean of the 227 with T>250 (JSON `frames_mean_of_gt_250`, n=227) | FIXED |
| 10 | identity_switch_suspect 1,728 all / 1,461 kept (28.4%) | §1.5, §6 | 1,728 / 1,461 (28.44%) | OK |
| 11 | ratio<0.9: 434 all, 0 kept; [0.9,1) kept 198 | §6 | 434 / 0 / 198 | OK |
| 12 | body_jump 3,187 / 2,822 kept; hands offframe 37/32; degenerate shoulders 25/20 | §6 | identical | OK |
| 13 | level_A: excluded 462, kept_warn 1,548, kept_no_warn 3,590 (2,740/417/433); warn by split 1,372/96/80; B/C status uniform | §1 Q3, §6 | identical | OK |
| 14 | v125 exact text shared val 13/513 (2.53%), test 6/513 (1.17%); normalized 3.31%/1.56% | §8 | 13 / 6; normalized 17 (3.31%) / 8 (1.56%) | OK |
| 15 | kept normalized: 4,925 distinct, 213 redundant | §5 | 4,925 / 213 | OK |
| 16 | 844 digits, 506 quotes/parens, 11 "ambos:", 0 uppercase | §5 | 844 / 506 / 11 / 0 | OK |
| 17 | v116–119 random split: 100% val shares parent (5,600: 64 groups; 600: 62) | §8 | Reconstructed independently (`random_split`, seed 42, `[0.8, 1-0.8]`): 1,119/1,119 and 119/119 val clips share a parent; val spans 62/64 and 49/62 groups. Original parenthetical ambiguous | FIXED (clarified) |
| 18 | 66/1,119 exact labels val∈train at 5,600 | §8 | 66 with `.strip().lower()` (script); 65 without strip | OK (note: "exact" = after strip) |
| 19 | v117 ≡ v116 in 350/354 common scalars, only `Gen/val_chrf` differs; `train_infonce`=0 in 25/25 | §1.6, §10 | independent parser: 354 common, 350 equal, diff {Gen/val_chrf}; infonce n=25, min=max=0 | OK |
| 20 | v116 chrF max 20.67 | §10 | 20.6707 | OK |
| 21 | v118 run4 val R@1 2.94%, train R@1 0.73%/481 | §10 | TB 118/4 max 0.029412; `Sign-chris/outputs/diag_v118_train_val_gap.json` train R@1 0.007277 | OK |
| 22 | v118f run8 R@1 0.268%/1,119 | §10 | TB 118/8 max 0.002681, n=16 epochs | OK |
| 23 | Wilson 95% half-widths table (n=12…385); n=80, p=0.5 → ±10.7 | §11 | recomputed all cells identical; n=80 → 10.7, n=76 → 11.0 | OK |
| 24 | sample: 80 prob + 42 purposive, Σw = 5,138, Kish n_eff 76.2, pilot 20 = 8/6/6, 12 buckets, 24 prob val+test, 752 s + 381 s | §9, §11 | Σw 5,137.9999; n_eff 76.16; pilot 8/6/6 all probability; 12 buckets; purposive weights empty; 0 overlap; 24; 752.2 / 381.5 s | OK |
| 25 | Artifact sha256 table in §13 | §13 | all 8 listed files match | OK |
| 26 | Data files unchanged | §2, §4 | sha256 HDF5 / manifest / split = `v125_artifacts.json`; mtimes 2026-06-23 | OK |

## B. file:line citations

| Citation | Claimed | Actual | Verdict |
|---|---|---|---|
| `build_dataset2_h5.py:35-46` (select_clips) | select_clips | function spans **33-43** | FIXED |
| `build_dataset2_h5.py:41` | lowercases label | `str(r["label"]).lower()` | OK |
| `build_dataset2_h5.py:52-63` | groups by detection index p | `per_person.setdefault(p, …)` | OK |
| `build_dataset2_h5.py:59` | scores discarded | `keypoints, _ = model(...)` | OK |
| `build_dataset2_h5.py:30` | LH_WRIST=93 | yes | OK |
| `rebuild_dataset2_alignment_targets_v125.py:42-69` | normalized-label Gemma ids, no special tokens, table[ids] | yes | OK |
| `backfill_dataset2_metadata.py:59-64` | ValueError on label mismatch | yes | OK |
| `data_augmentation.py:142-151` | drops 92 and 133 | face 28:92, lh 93:113, rh 113:133 | OK |
| `data_augmentation.py:173` | wrist_idx=[71,91] | yes | OK |
| `experiments/v115_ce_vocab/sigreg.toml:18-19` | "3088" comment only | line 18 comment; 19 tolerance | OK |
| `train_ctc_v119.py:204-209`, `ctc_v119_subset1k.toml:2-6` | "keys 0..599 do NOT correspond" | yes | OK |
| `research_audit_imitator_e1.md:260` | same claim | yes | OK |
| `contrastive_v118f.toml:1-8` | CN Sordos | yes | OK |
| `docs/audits/v125_language_audit.md:5-9` | LSA | yes | OK |
| `keypoint_dataset.py:112` (script docstring 111-112) | random_split seed 42 | line 112 | OK |
| `AUDIT_SPEC.md:30-32` | env (py/h5py/numpy) | yes; torch 2.13 not there | FIXED (added `04_experiments_split_stats.json:torch`) |

## C. Evidence tags
- No semantic or temporal correspondence claim is tagged VERIFIED; levels B/C are UNKNOWN everywhere; no semantic-quality % is computed (U, scenarios are pre-registration rules).
- Line 25 / Q1 row tagged the CN Sordos channel as VERIFIED though §3.1 calls it DOCUMENTED → FIXED (counts VERIFIED, channel DOCUMENTED).

## D. Sample manifest / reproducibility
- `build_review_sample.py` writes in place (`AUD/…`). Re-ran it in a scratch mirror tree (scratchpad `rep/`, symlinked inputs). Output sha256 = `5ba98a52…de13` / `ec3b3c05…63f0`, identical to originals; originals unchanged (`sha256sum -c` OK). Seed 23 reproducible: OK.
- Probability vs purposive not pooled (purposive weights empty, components disjoint): OK.
- WRONG-UNFIXED (design caveat, not regenerated to keep hashes): §9 says reviewers are blind to flags and sampling reason, but `human_review_template.csv` exposes `sampling_component` (probability/purposive), which partly reveals that purposive clips are flagged. Recommend dropping that column (and `pilot`) from the copy given to reviewers.

## E. Rubric
- All 6 categorical fields include `unknown` and `not_assessable`. `estimated_offset_seconds` is numeric/empty → FIXED: rubric now states empty = unknown/not_assessable, with reason in notes.

## F. Structure
- 14 sections present; 10 completion questions answered (§1 table); 3 experiments, each with hypothesis, data, architecture, baseline, loss, metrics, controls (shuffle, length/text-only, leakage), success criterion, risk: OK.

## G. Secrets
- Dataverse key (36 chars) extracted into a shell variable only; `grep -rF` of full key and its 12-char prefix over `docs/` and `artifacts/`: no matches. Long-token scan shows only hashes and identifiers: OK.

## Edits made
1. `docs/dataset2_scientific_audit.md` §2 row "Intérprete usado": added torch evidence source.
2. §3.3: `build_dataset2_h5.py:35-46` → `:33-43`.
3. §3.8: mean 468.3 → 467.3 (468.3 for the 227 with T>250).
4. §8 row 1: clarified v116–119 group parenthetical (1,119/1,119, 62/64; 119/119, 49/62).
5. §14 contradictions row: "468,3 ≈ 469" → "467,3 de T frente a ~469 frames de video registrados".
6. §1 item 1 and Q1 row: channel tag DOCUMENTED.
7. `human_review_rubric.md`: offset field empty = unknown/not_assessable.
- 2026-10-19 coordinator: removed sampling_component from human_review_template.csv (blind reviewers); script updated; hash updated in reproduction_commands.md
