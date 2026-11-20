# WAIMLAp 2026 — Imitator talk: slide cards and Q&A prep

Deck: `imitator_waimlap2026.pptx` (rebuild: `~/miniconda3/envs/deck-tools/bin/python build_deck.py`).
Source: the draft in `paper/content/full/`. Isolated-sign metrics come from the retrained folds 7–8.
Format: 10 min talk + 5 min questions. 12 slides plus 1 backup. The word-for-word script is in each slide's PowerPoint notes.

**Goal:** explain how exact Gemma IDs let us measure the visual bridge. Start with one isolated sign,
then test ordered transcription on synthetic streams of 2–8 signs as a step toward sentence translation.
End with a separate exploratory audit of Gemma's ability to recover corrupted isolated labels.

| # | Claim (title) | Evidence on slide | Look first at | Transition | ~time |
|---|---|---|---|---|---|
| 1 | — (title) | — | — | "How should a small sign model talk to an LLM?" | 0:20 |
| 2 | Sign-to-text needs a bridge between vision and language | See → interface? → write; prior bridges (Sec. 1–2) | Dashed "Interface ?" box | "Our own first answer was this" | 1:00 |
| 3 | Our first Imitator imitated the LLM's input embeddings | Original pipeline; loss ≈ 8×10⁻⁴ (Sec. 1, 3) | Purple "Vectors ≈ embeddings" box | "Does the LLM read them as the intended words?" | 0:50 |
| 4 | Close to the right embedding is not the right token | Limits vs this work (grey vs teal card) | Teal card | "Let me show the pipeline" | 0:50 |
| 5 | Imitator writes exact Gemma IDs; Gemma only reads text | Pipeline diagram, scoring brackets | Purple "Exact Gemma IDs" box | "How we evaluated it" | 1:00 |
| 6 | From one isolated sign to ordered sign sequences | LSA64, signer splits, sequence criteria | Purple folds 7–8 | "Start with one sign" | 0:50 |
| 7 | Can Imitator recover the exact IDs of one sign? | 13.75% and 10.94% strict accuracy; 77.81% and 79.06% emit more tokens than the reference | Folds 7–8 | "Next, two to eight signs" | 0:30 |
| 8 | On 2–8 signs, strict accuracy rises about tenfold | Bars folds 4–6: 2.2/2.3/1.8 → 18.1/19.8/16.9 | Purple bars | "Does it hold on new signers?" | 0:50 |
| 9 | Both held-out signers pass all three criteria | Bars folds 7–8 vs CIF + 11.9% line; H1–H3 | Purple bars above dashed line | "Could Gemma recover a corrupted label?" | 1:10 |
| 10 | Gemma recovered no exact label at tested nonzero rates | Synthetic token substitutions; exact match and chrF | 0% bars | "To summarise" | 1:00 |
| 11 | Exact IDs make the visual bridge measurable | 3 takeaways and 3 limits/next steps | Teal card | — | 1:00 |
| 12 | Thank you · Questions? | — | — | — | 0:10 |
| B | Backup: left hand drives central subtokens (one example) | Paper Fig. 3 | — | only if asked | — |

Visual choices:
- **Context slides:** slides 2–4 use the same box-and-arrow language as the pipeline slide, so the new design reads as a change to one box.
- **Colour:** purple marks our model and grey marks comparisons. Teal marks thresholds and labels.
- **Accuracy scale:** slides 8 and 9 share the same 0–25% axis. Bars start at 0, and every bar has its value label.
- **Gemma audit:** shown as two small panels (exact match, then chrF). This avoids a dual-axis chart.

## Fill in before the talk
- Session and Paper ID are filled in: Oral Session 10 · Paper ID 58.
- Contact email or repository on slide 12, if you want them shown.

## Likely questions (answer only from the paper)
- **How is one sign scored?** We require the complete token ID sequence for its Spanish label plus EOS. We evaluate 320 clips per held-out signer with retrained models. This is a separate exploratory evaluation from the original 2–8-sign confirmation.
- **Why is strict accuracy low for one sign?** Outputs are longer than the reference on 249/320 clips in fold 7 and 253/320 in fold 8. E1 was trained on streams of two to eight signs.
- **Why is 12–14% strict accuracy useful?** Strict accuracy means the whole 2–8-sign sequence is exact. Edit similarity is 53.0 / 58.7% on signers 7 / 8. The paper's claim is "viable, not solved".
- **Why not fine-tune Gemma jointly?** We chose not to, on purpose. Scoring the transcript before the LLM keeps fluent output from hiding visual errors.
- **Is it gloss-free?** No. The targets come from the 64 LSA64 labels. The claim is only that inference needs no 64-class gloss classifier.
- **Why 121 output rows?** The task uses 118 Gemma content IDs, plus BOS, EOS and PAD. The rows map one-to-one back to the original IDs, so this is a reparameterization, not a new vocabulary. It must be expanded and revalidated for sentence data.
- **What is CIF?** CIF is our earlier non-autoregressive model. It uses the same encoder design, with a length calibration that was frozen on inner validation. It is a historical comparator, not a state-of-the-art baseline.
- **Did you try other seeds?** No. We used a single seed (23). Variance between seeds is unknown, and the bootstrap CIs cover only variation across examples.
- **Natural signing / continuous SLT?** Not evaluated. The streams are synthetic, with no coarticulation and no LSA grammar.
- **Does this show LLMs can't correct sign errors?** The evidence covers one model and prompt on synthetic token substitutions in isolated labels. The schematic perfume example motivates the audit. Actual visual decoder errors and sentence-level context require separate evaluation.
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
- **8 Development:** ① key result ② positional encoding ③ remaining sequence errors.
- **9 Confirmation:** ① H1 ② H2 ③ H3.
- **10 Gemma audit:** ① clean input ② tested nonzero rates ③ chrF and the schematic example.
- **11 Conclusions:** ① "Limits → next step" card.
- Slides 1, 7, 12 and the backup have no animations.
