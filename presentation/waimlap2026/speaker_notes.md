# WAIMLAp 2026 — Imitator talk: slide cards and Q&A prep

Deck: `imitator_waimlap2026.pptx` (rebuild: `~/miniconda3/envs/deck-tools/bin/python build_deck.py`).
Source: the camera-ready paper `springer_submission/imitator_e1.pdf`. Every number comes from
`shared/results_macros.tex` or `sections/07_results.tex`.
Format: 10 min talk + 5 min questions. 11 slides plus 1 backup. The word-for-word script is in each slide's PowerPoint notes.

**Goal:** by the end, the audience understands that predicting exact Gemma token IDs makes the sign→LLM
interface measurable. The evidence is held-out-signer accuracy that passes the preregistered criteria. The
audience also learns that LLM correction is still untested because the data has no sentence context.
**Structure:** first the context (the sign→LLM interface problem and our earlier Imitator), then the turning point and the pipeline, then three pieces of evidence, then the honest limit.

| # | Claim (title) | Evidence on slide | Look first at | Transition | ~time |
|---|---|---|---|---|---|
| 1 | — (title) | — | — | "How should a small sign model talk to an LLM?" | 0:20 |
| 2 | Sign-to-text needs a bridge between vision and language | See → interface? → write; prior bridges (Sec. 1–2) | Dashed "Interface ?" box | "Our own first answer was this" | 1:00 |
| 3 | Our first Imitator imitated the LLM's input embeddings | Original pipeline; loss ≈ 8×10⁻⁴ (Sec. 1, 3) | Purple "Vectors ≈ embeddings" box | "Does the LLM read them as the intended words?" | 0:50 |
| 4 | Close to the right embedding is not the right token | Limits vs this work (grey vs teal card) | Teal card | "Let me show the pipeline" | 0:50 |
| 5 | Imitator writes exact Gemma IDs; Gemma only reads text | Pipeline diagram, scoring brackets | Purple "Exact Gemma IDs" box | "How we evaluated it" | 1:00 |
| 6 | Test signers are unseen; confirmation was preregistered | LSA64 facts, 10-fold strip, H1–H3 | Purple folds 7–8 | "First, development" | 1:00 |
| 7 | Token IDs raise strict accuracy ~10× on dev signers | Bars folds 4–6: 2.2/2.3/1.8 → 18.1/19.8/16.9 | Purple bars | "Does it hold on new signers?" | 0:50 |
| 8 | Both held-out signers pass all three criteria | Bars folds 7–8 vs CIF + 11.9% line; H1–H3 | Purple bars above dashed line | "Can Gemma fix the remaining errors?" | 1:10 |
| 9 | Without sentence context, Gemma cannot repair substitutions | Exact-after bars + chrF before/after (Table 2) | 0% bars | "To summarise" | 1:10 |
| 10 | Exact IDs make the bridge measurable; correction needs context | 3 takeaways and 3 limits/next steps | Teal card | — | 1:00 |
| 11 | Thank you · Questions? | — | — | — | 0:10 |
| B | Backup: left hand drives central subtokens (one example) | Paper Fig. 3 | — | only if asked | — |

Visual choices:
- **Context slides:** slides 2–4 use the same box-and-arrow language as the pipeline slide, so the new design reads as a change to one box.
- **Colour:** purple marks our model and grey marks comparisons. Teal marks thresholds and labels.
- **Accuracy scale:** slides 7 and 8 share the same 0–25% axis. Bars start at 0, and every bar has its value label.
- **Gemma audit:** shown as two small panels (exact match, then chrF). This avoids a dual-axis chart.

## Fill in before the talk
- Session and Paper ID are filled in: Oral Session 10 · Paper ID 58.
- Contact email or repository on slide 11, if you want them shown.

## Likely questions (answer only from the paper)
- **Why is 12–14% strict accuracy useful?** Strict accuracy means the whole 2–8-sign sequence is exact. Edit similarity is 53.0 / 58.7% on signers 7 / 8. The paper's claim is "viable, not solved".
- **Why not fine-tune Gemma jointly?** We chose not to, on purpose. Scoring the transcript before the LLM keeps fluent output from hiding visual errors.
- **Is it gloss-free?** No. The targets come from the 64 LSA64 labels. The claim is only that inference needs no 64-class gloss classifier.
- **Why 121 output rows?** The task uses 118 Gemma content IDs, plus BOS, EOS and PAD. The rows map one-to-one back to the original IDs, so this is a reparameterization, not a new vocabulary. It must be expanded and revalidated for sentence data.
- **What is CIF?** CIF is our earlier non-autoregressive model. It uses the same encoder design, with a length calibration that was frozen on inner validation. It is a historical comparator, not a state-of-the-art baseline.
- **Did you try other seeds?** No. We used a single seed (23). Variance between seeds is unknown, and the bootstrap CIs cover only variation across examples.
- **Natural signing / continuous SLT?** Not evaluated. The streams are synthetic, with no coarticulation and no LSA grammar.
- **Does this show LLMs can't correct sign errors?** No. It shows that for isolated labels with random order and one model and prompt, there is no context to recover from. The next step is a preregistered sentence corpus.
- **Why did the order become correct?** Sinusoidal positional encoding. Without it, permuting segments left 86.3% of predictions unchanged. With it, shuffling drops edit similarity by 0.34–0.38.
- If a question goes beyond this: say "we have not evaluated that".

## Animations (click order)
Everything uses a 0.5 s fade-in. Titles and charts are visible from the start; only the supporting elements appear on click.
On slides 2, 3 and 5 the slide opens with the title centred; click ① glides it up to its normal place (1 s, ease in/out) where it stays, then the first content appears. The glide plays in PowerPoint, Keynote and LibreOffice. Google Slides has no motion effect, so there the title snaps to its place on that click (no overlap, no exit). Present from PowerPoint to get the movement.
- **2 Context:** ① "See the signs" ② arrow + "Interface ?" ③ arrow + "Write the language" ④ prior bridges.
- **3 History:** ① keypoints → visual model ② vectors ≈ embeddings ③ frozen LLM ④ premise + loss ⑤ open question.
- **4 Turning point:** ① "This work" card.
- **5 Pipeline:** ① the five boxes left to right ② scoring brackets ③ bullets.
- **6 Protocol:** ① fold strip ② preregistered criteria.
- **7 Development:** ① key result ② positional encoding ③ "viable, not solved".
- **8 Confirmation:** ① H1 ② H2 ③ H3.
- **9 Gemma audit:** ① clean input ② any substitution → 0% ③ chrF + "photo → green".
- **10 Conclusions:** ① "Limits → next step" card.
- Slides 1, 11 and the backup have no animations.
