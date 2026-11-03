# Locked development protocol — WSP Imitator E1

The authoritative machine-readable contract is `protocol.json`, authenticated by
`protocol.sha256`. Source file hashes bind the existing model/evaluator and this
harness. The source base is commit 095789d18381f138b1507f2c70fb594f0a66baff.
No real-model inference preceded this lock. This is a preflight protocol, not a
claim that a candidate or final outer-evaluation algorithm has been frozen.

The task is closed-vocabulary transcription of randomly ordered synthetic
concatenations of 2–8 isolated LSA64 signs. Gemma realization is outside all scores.
Only checkpoint values within the declared decoder matrix surface may change;
architecture, input, normalization, tokenizer, mapping and greedy decoding remain
those of the repository. The initial editing surface is deliberately narrower
than the task's maximum permissions. Expanding it requires causal evidence and a
preregistered amendment before testing, never a silent change to the lock.

Development uses folds 4–6. Discovery draws 128 fixed examples from each fold's
training-signers pool (seed 300023); confirmation draws 128 fixed examples from
its inner-validation signer (seed 400023). These pools are clip- and signer-disjoint
within the fold. They are NOT blind outer evaluation. Checkpoint reproduction
uses the original 512 inner-validation sequences, seed 100023 and batch size 2,
requiring both recorded selection metrics to agree within 1e-6 before any edits.
Any difference must be investigated and recorded before experimentation resumes.

The rotating LOSO manifest permits a person to have different roles in different
folds. In particular, signers 9/10 occur in permitted training pools for folds
4–6; this does not authorize any fold-9/10 outer examples or results. This harness
rejects folds outside 4–6. It reads signer membership before any permitted clip's
label or keypoints. Gold IDs/lengths/boundaries may be used only in diagnostics;
free-running generation receives only keypoints and frame lengths.

Strict exact means complete content-ID equality AND emitted EOS. Edit similarity
is mean per-example max(0, 1 − Levenshtein/reference length); TER is the unclipped
mean per-example ratio. Text exact remains separate. The evaluator's no-EOS
length is 33; the actual greedy loop can output 33 non-EOS IDs. This is retained.
Early EOS means emitted EOS with content shorter than reference: an operational
proxy, not proof of an EOS circuit defect. Alignment tie-breaks are diagonal,
deletion, insertion. Frame count represents duration; no unverified FPS is used.
Pairwise sign order uses the repository's unique-common-gloss eligibility rule.

Every inference reserves one journaled pass BEFORE sample materialization/model
forward, with IDs/hash, original/patch hash, seed and output path. A paired pass
has at most 256 examples; a single/probe pass at most 512. Failed reservations
remain spent; failures before inference or data reveal are recorded without a
reservation. The hard maximum is 50. Unit tests use artificial fixtures and
separate temporary ledgers. They are software tests, never E1 measurements.

A single candidate family may be preregistered after discovery demonstrates a
localized mechanism and a competing explanation has been tested. Directions and
indices must cite permitted discovery evidence. Strengths are 0.5, 1, 2, with a
no-op and a same-support/same-norm random control. All replacements start from
immutable original tensors. The cap is floor(0.005 × unique parameter entries),
counted bitwise after dtype conversion; buffers are outside the denominator.
Every patch records exact changed flat indices, original/replacement hashes,
norms, original tensors and replacement values. No optimizer, gradient, supervised
fitting, global rescaling, vocabulary-row change, bias or final-norm edit is allowed.

Selection maximizes pooled development strict exact, then edit similarity.
Direction/magnitude are frozen before confirmation. Acceptance requires a paired
strict-exact CI lower bound above zero, same-sign fold deltas, superiority to the
matched random control, and no sign-count/robustness stratum with strict or edit
similarity regression exceeding 0.02. The complete repository jitter, dropout and
speed battery is required for a supported robustness claim. Sample CIs use 10,000
paired percentile resamples with seed 7800026; strict discordances use the exact
two-sided sign test. Sample bootstrap does not establish population-level signer
uncertainty, and three development folds share training data. One candidate
family limits selection multiplicity; an exploratory sweep CI is not confirmation.

No positive finding is required. Insufficient localization, non-significant
confirmation, random-control parity, regressions or incomplete coverage means no
accepted repair. If the budget cannot cover the evidence, report incompleteness.
Reserve eight passes for any eventual 896-example paired evaluation on each of
folds 9/10; do not reduce that endpoint to accommodate search. No final outer
runner is enabled here. A separate algorithm/code/strength lock, accessible valid
closed checkpoints, audited exposure history, and access separation must precede
its implementation and single reveal. Folds 7/8 remain previously reported.
