# WAIMLAp 2026 — Imitator talk: slide cards and Q&A prep

Deck: `imitator_waimlap2026.pptx` (rebuild: `~/miniconda3/envs/deck-tools/bin/python build_deck.py`).
Source: the draft in `paper/content/full/`. Isolated-sign metrics come from the retrained folds 7–8.
Format: 10 min talk + 5 min questions. 13 slides plus 1 backup. The word-for-word script is in each slide's PowerPoint notes.

**Story:** open with a short sign clip the audience cannot read. The LLM knows language but not sign language, so do we need a translator from scratch, or just a small adapter? One slide of previous work (embeddings), then the talk is ours: methodology, results, conclusions, future work. A small tag above each title names the section.

| # | Section | Claim (title) | Evidence on slide | Transition | ~time |
|---|---|---|---|---|---|
| 1 | — | (title) | — | "Let me show you a short video" | 0:10 |
| 2 | Motivation | What did this person just say? | LSA64 clip of the sign "Perfume" (signer 7) | "Its input is text, and this is movement" | 0:40 |
| 3 | Motivation | The LLM knows language. It does not know sign language. | Signer … LLM, "train a translator from scratch?", then the small adapter | "What should the adapter hand over?" | 0:50 |
| 4 | Previous work | Before: imitate embeddings. Now: write token IDs. | LLM input path; first Imitator (loss ≈ 8×10⁻⁴) vs. this work | "Here is the model" | 1:00 |
| 5 | Methodology · model | Imitator now writes Gemma’s own token IDs | Video → encoder → decoder → Gemma IDs → Gemma 3n; scored before the LLM | "Start with the easiest case: one sign" | 0:55 |
| 6 | Methodology · evaluation | One sign first. Then several signs in order. | LSA64, the two tasks, folds 1–8 | "Does Imitator write the right label?" | 0:50 |
| 7 | Results · one sign | On one sign, does the output start with the right label? | Correct prefix on 229/320 and 220/320 clips, 95% intervals | "What if the video has several signs?" | 0:45 |
| 8 | Results · several signs, development | On 2–8 signs, strict accuracy rises about tenfold | Folds 4–6: 2.2/2.3/1.8 → 18.1/19.8/16.9 | "Does it hold on new signers?" | 0:35 |
| 9 | Results · several signs, unseen signers | Both held-out signers pass all three preregistered criteria | Folds 7–8 vs CIF + 11.9% line; H1–H3 | "Could Gemma recover a corrupted label?" | 0:50 |
| 10 | Results · can the LLM repair a draft? | Gemma recovered no exact label at tested nonzero rates | Synthetic token substitutions; exact match and chrF | "To conclude" | 0:45 |
| 11 | Conclusions | A small adapter can write a draft in Gemma’s own token IDs | What we showed · Limits | "So what comes next" | 0:45 |
| 12 | Future work | Next: real signed sentences | Three directions + the goal | Return to the person in the clip | 0:35 |
| 13 | — | Thank you · Questions? | — | — | 0:10 |
| B | Backup | Left hand drives central subtokens (one example) | Paper Fig. 3 | only if asked | — |

Visual choices:
- **Context slides:** slides 3–5 use the same box-and-arrow language as the pipeline slide, so the new design reads as a change to one box.
- **Colour:** purple marks our model and grey marks comparisons. Teal marks thresholds and labels.
- **Accuracy scale:** slides 8 and 9 share the same 0–25% axis. Bars start at 0, and every bar has its value label.
- **Gemma audit:** shown as two small panels (exact match, then chrF). This avoids a dual-axis chart.

## Fill in before the talk
- Session and Paper ID are filled in: Oral Session 10 · Paper ID 58.
- Contact email or repository on slide 13, if you want them shown.

## Likely questions (answer only from the paper)
- **How is one sign scored?** We require the complete token ID sequence for its Spanish label plus EOS. We evaluate 320 clips per held-out signer with retrained models. This is a separate exploratory evaluation from the original 2–8-sign confirmation.
- **What do the bars and whiskers measure?** A correct prefix matches every reference ID at the start of the output. The rates are 71.56% (95% interval 60.94–81.26%) on fold 7 and 68.75% (58.75–78.75%) on fold 8. Percentile intervals use 10,000 bootstrap resamples of the 64 glosses within each signer, keeping each gloss's five clips together (seed 23). They do not estimate variability across signers or training runs.
- **Why is strict accuracy low for one sign?** Outputs are longer than the reference on 249/320 clips in fold 7 and 253/320 in fold 8. E1 was trained on streams of two to eight signs.
- **Why is 12–14% strict accuracy useful?** Strict accuracy means the whole 2–8-sign sequence is exact. Edit similarity is 53.0 / 58.7% on signers 7 / 8. The paper's claim is "viable, not solved".
- **Why not fine-tune Gemma jointly?** We chose not to, on purpose. Scoring the transcript before the LLM keeps fluent output from hiding visual errors.
- **Is it gloss-free?** No. The targets come from the 64 LSA64 labels. The claim is only that inference needs no 64-class gloss classifier.
- **Why 121 output rows?** The task uses 118 Gemma content IDs, plus BOS, EOS and PAD. The rows map one-to-one back to the original IDs, so this is a reparameterization, not a new vocabulary. It must be expanded and revalidated for sentence data.
- **What is CIF?** CIF is our earlier non-autoregressive model. It uses the same encoder design, with a length calibration that was frozen on inner validation. It is a historical comparator, not a state-of-the-art baseline.
- **Did you try other seeds?** No. We used a single seed (23). Variance between seeds is unknown, and the bootstrap CIs cover only variation across examples.
- **Natural signing / continuous SLT?** Not evaluated. The streams are synthetic, with no coarticulation and no LSA grammar.
- **Does this show LLMs can't correct sign errors?** The evidence covers one model and prompt on synthetic token substitutions in isolated labels. The schematic perfume example motivates the audit. Actual visual decoder errors and sentence-level context require separate evaluation.
- **Why does the fold strip stop at 8?** Folds 9–10 are reserved in the protocol and produced no result. Their signers are still training data for the other folds.
- **Did embeddings fail?** The paper does not claim that. It says a low distance does not guarantee the intended token, there was no exact check, and the earlier splits were not signer-exclusive. There is no head-to-head comparison of embeddings and token IDs.
- **Did you give Gemma the video, or test another video model on signs?** Not evaluated. "The LLM does not know sign language" is our starting premise, not a measured result.
- **Does Gemma read Imitator's output?** Not in this paper. Imitator is scored on its own. The Gemma audit used reference labels with synthetic substitutions.
- **Is Imitator trained from scratch? How big is it?** The encoder's ST-GCN and projection are initialised from our earlier CIF model and kept frozen; there is no external pretraining. The paper does not state parameter counts.
- **Which Gemma, and what is the end marker?** Gemma 3n E2B, frozen. The end marker is implemented with Gemma's end-of-turn ID.
- **What prompt did the audit use?** A fixed few-shot correction prompt in Spanish. The slides no longer quote an instruction.
- **If Imitator already writes text, what does Gemma add? Why not classify 64 signs?** In this paper Gemma contributes its vocabulary and the separate audit; it did not repair the tested errors. Token IDs are the interface we want for sentences, where the LLM would have context. That is future work, and the 121-row output must be expanded and revalidated for it.
- **Why are the baselines different on the development and confirmation slides?** Development compares against our initial decoder, which lacked positional encoding. Confirmation compares against CIF because the preregistration named it.
- **Why is one sign no better than several?** Different checkpoints (replicas vs. originals), and the replicas run long on single clips: they were trained on streams of two or more signs.
- **Development folds 1–3?** E1 development results exist for folds 4–6 only.
- **Why did the order become correct?** Sinusoidal positional encoding. Without it, permuting segments left 86.3% of predictions unchanged. With it, shuffling drops edit similarity by 0.34–0.38.
- If a question goes beyond this: say "we have not evaluated that".

## Animations (click order)
Supporting elements use a 0.5 s fade-in. On slides 3, 4 and 5, a single centred title is visible at first. Click ① moves that title straight up to the header over 1 second, and the first group of content appears once it arrives. Present from PowerPoint to play the motion path.
- **2 Hook:** click the clip itself to play it (it loops). Then ① "Could it understand this one?" ② the answer, "Perfume".
- **3 Motivation:** ① "A signer moves" ② "The LLM writes" ③ "train a sign translator from scratch?" ④ arrows + "A small adapter?".
- **4 Previous work:** ① the LLM's input path ② previous work card ③ this work card.
- **5 Model:** ① the five boxes left to right ② scoring brackets ③ why we score before the LLM.
- **6 Evaluation:** ① fold strip ② the three rules.
- **8 Development:** ① key result ② positional encoding ③ remaining sequence errors.
- **9 Confirmation:** ① H1 ② H2 ③ H3.
- **10 Gemma audit:** ① clean input ② tested nonzero rates ③ chrF and the schematic example.
- **11 Conclusions:** ① "Limits" card.
- **12 Future work:** ① real sentences ② correction in context ③ richer drafts ④ the goal.
- Slides 1, 7, 13 and the backup have no animations.
