# Dataset2 human LSA review: rubric and protocol

Files: `sample_manifest.csv` (what to review, with the stored Spanish label in `transcript`) and `human_review_template.csv` (one row per `sample_id`, rubric fields empty). Sample design: `_work/05_sampling.md`. Generator: `scripts/build_review_sample.py` (seed 23).

## 1. Reviewers

- Reviewers must be competent in LSA (Lengua de Señas Argentina) and Spanish: Deaf LSA users, or certified LSA–Spanish interpreters. Engineers on the project may not fill semantic fields.
- **Pilot (20 clips, `pilot=1`):** two independent reviewers. Neither sees the other's answers until both have finished. The pilot is used to compute agreement (section 4) and to refine the wording of this rubric. **Clips are never replaced.** A broken or hard clip is coded `not_assessable` and still counts.
- **Rest of the sample:** one reviewer is acceptable once pilot agreement is adequate. A second reviewer on at least 20% of clips is recommended.
- Review order: shuffle the order within each reviewer's sheet. Do **not** show `sampling_component`, `sampling_stratum`, `sampling_reason` or `integrity_flags` to reviewers, so that knowing a clip was "suspicious" cannot bias their judgement. The template's `sampling_component` and `pilot` columns can be hidden.
- Watch the raw video at `source_video_path`, with sound off. You may re-watch the clip and play it slowed down. Do not look up the surrounding episode unless you are filling `temporal_alignment` (see below), and note in `reviewer_notes` if you did.

## 2. Fields and admitted values

Every categorical field also admits:
- `unknown`: the clip can be assessed, but you cannot decide.
- `not_assessable`: the clip cannot be assessed (does not play, signer not visible, and so on).

Never leave a field blank. Blank means "not yet reviewed".

| Field | Admitted values | Criteria |
|---|---|---|
| `video_complete` | `yes`, `truncated_start`, `truncated_end`, `truncated_both`, `corrupt`, `unknown`, `not_assessable` | Use `yes` if the clip plays start to end without freezes or corruption and the signing does not appear cut by the clip boundaries. Use `truncated_*` if a sign or utterance is visibly cut at that edge. Use `corrupt` for decoding artefacts, frozen frames or black video. |
| `signer_visible` | `single_signer_full`, `single_signer_partial`, `multiple_signers`, `signer_changes`, `no_signer`, `unknown`, `not_assessable` | `full` means the hands, face and upper body are visible throughout. Use `partial` if the hands or face leave the frame or are occluded for a noticeable part of the clip. Use `multiple_signers` if two or more people sign or appear in the frame. Use `signer_changes` if the shot cuts from one signer to another. |
| `temporal_alignment` | `aligned`, `starts_early`, `starts_late`, `ends_early`, `ends_late`, `cut_mid_sign`, `multiple_units`, `unknown`, `not_assessable` | This compares the clip boundaries with the signed utterance that corresponds to the transcript. `aligned`: the corresponding signing starts and ends within the clip, with no large unrelated signing before or after it. `starts_early`/`ends_late`: the clip contains extra signing that is not in the transcript. `starts_late`/`ends_early`: part of the transcript's signing is missing. `cut_mid_sign`: a boundary falls inside a sign. `multiple_units`: the clip contains several separate utterances and only some of them match. If several values apply, give the main one and list the others in `reviewer_notes`. |
| `semantic_correspondence` | `translation`, `paraphrase`, `partial`, `approximate_subtitle`, `unrelated`, `unknown`, `not_assessable` | `translation`: the Spanish faithfully renders what is signed. `paraphrase`: the same meaning in a different wording or structure. `partial`: only part of the signed content is in the text, or the reverse. `approximate_subtitle`: a loosely related subtitle-style text (for example a summary, or speech-track wording not signed one to one). `unrelated`: the text does not match the signing. |
| `translation_completeness` | `complete`, `minor_omissions`, `major_omissions`, `additions_not_signed`, `omissions_and_additions`, `unknown`, `not_assessable` | Compare the content units. Typical additions to look for are numbers, phone numbers, acronym expansions in parentheses, and speaker labels such as "ambos:". `minor` means at most about 1 content word or a detail. `major` means a clause or more. |
| `annotation_confidence` | `high`, `medium`, `low`, `unknown`, `not_assessable` | How confident you are in your own answers for this clip. |
| `estimated_offset_seconds` | a signed number in seconds, or empty (empty = `unknown` or `not_assessable`; say which in `reviewer_notes`) | Fill this **only with evidence**. You must be able to point to where the transcript's signing actually starts (or ends) relative to the clip boundary. Negative means the signing starts before the clip; positive means it starts after the clip start. In `reviewer_notes`, state which boundary you measured and how (for example "start; player timestamp"). Leave it empty when you are guessing. Never write 0 by default. |
| `reviewer_id` | pseudonymous code (R1, R2, …) | No names or e-mails. |
| `review_date` | ISO date (YYYY-MM-DD) | |
| `reviewer_notes` | free text | Note secondary codes, signs you could not identify, lexical or regional variants, and anything about the signer or framing. Do not record personal data about the people in the video. |

## 3. What the transcript is

The stored label is lowercased written Spanish, as delivered by the source pipeline (`build_dataset2_h5.py:41` lowercases it). Casing therefore carries no information. Judge meaning and coverage, not punctuation.

## 4. Inter-annotator agreement (pilot)

- Per categorical field, report Cohen's κ for 2 reviewers, and Krippendorff's α (nominal; ordinal for `annotation_confidence` and `translation_completeness`), which also handles more than 2 reviewers or missing ratings. Report raw percent agreement and the confusion table as well, because κ is unstable with n=20 and skewed marginals.
- Treat `unknown` and `not_assessable` as their own categories in a first analysis. In a second analysis, treat them as missing; α handles missing values.
- With 20 clips the agreement confidence intervals are wide. Report a bootstrap CI over clips (resample clips, 2,000 replicates). If α < 0.67 on `semantic_correspondence` or `temporal_alignment`, revise the rubric and re-train the reviewers before scaling up. Do **not** drop the disagreeing clips.
- For `estimated_offset_seconds`, report the number of clips where both reviewers filled it, and the mean absolute difference.

## 5. Estimating proportions (probability component only)

Use only rows with `sampling_component=probability`. The 80 clips are a stratified sample with **unequal** inclusion probabilities: val/test strata are over-represented on purpose. The target population is the 5,138 v125-kept clips.

- Weighted proportion: p̂ = Σ w_i·y_i / Σ w_i, with `w_i = design_weight` (= N_h/n_h; the weights sum to 5,138), where y_i = 1 if the clip has the property (for example `semantic_correspondence ∈ {translation, paraphrase}`).
- Decide in advance how `unknown`/`not_assessable` enter the calculation. Report them as their own category, and also give a sensitivity bound (counted as failures versus excluded).
- CI: Wilson or Clopper–Pearson, applied with the **Kish effective sample size** n_eff = (Σw)²/Σw² (≈ 76 for this design) and x_eff = p̂·n_eff. Alternatively, use the stratified variance Σ_h W_h² p_h(1−p_h)/(n_h−1) with W_h = N_h/N, but only when every stratum has n_h ≥ 2 and its estimate is not 0 or 1. Prefer Clopper–Pearson when p̂ is near 0 or 1.
- Per-split estimates (train, val, test) are also valid. They are computed within each split's 3 strata. Val and test have only 12 clips each, so their CIs will be very wide (roughly ±25 pp); say so.
- The 20-clip pilot is also stratified. If it is analysed alone, use weights N_h/k_h with k_h = the number of pilot clips in the stratum (see `05_sampling.md`). Report it as a pilot, not as the final estimate.
- The probability sample does not contain the 462 rejected clips. Estimates describe the kept set, not the 5,600 clips, and not the 8,459 raw clips.

## 6. What NOT to infer from the purposive component

The purposive rows (`PU*`) were chosen *because* they look suspicious.
- **Never** pool them with the probability rows. Never compute a dataset-wide rate from them, and never weight them.
- Valid uses: per-bucket qualitative evidence, such as "do `identity_switch_suspect` clips really show a person switch?". This gives counts out of n per bucket, describes **those clips only**, and is useful for calibrating automatic flags (precision of a flag in its bucket).
- A failure rate in a bucket is not the failure rate of all flagged clips. Each bucket holds 2 to 5 clips, so it is useful for detecting a problem, not for sizing it.
- An absence of problems in a bucket does not show that the problem is absent from the dataset.

## 7. Data handling

- Videos stay where they are. Do not copy, upload or redistribute clips. The previews in `_private_previews/` are private and pilot-only.
- The template contains no personal data. Reviewers are identified by pseudonym only.

_Prepared with AI assistance (Claude). The rubric categories and thresholds are proposals and should be validated by LSA subject-matter experts before use._
