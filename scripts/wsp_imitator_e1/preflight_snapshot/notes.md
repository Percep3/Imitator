# Session recovery

Status: blocked before baseline. Branch research/wsp-imitator-e1; bead Sign-chris-v58.
Protocol is locked; preserve its SHA and source hashes. Research passes used 0/50.
All synthetic test ledgers live in pytest temporary directories and are not research.
No candidate, negative mechanistic result, selected patch, or edited E1 checkpoint.
No outer predictions/results opened. Original checkpoints could not be found.
Source/dataset metadata and recorded hashes are in checkpoint_manifest/environment.
See final_report.md and BLOCKED.md for exact findings and restoration requirements.

Tests: 22 passed (harness, sequence metrics, existing robustness tests); no gradients.
Paper rebuilt: 11 pages. The old Sign-env executable and NVIDIA driver are unavailable;
.venv-wsp is an isolated CPU validation environment, not a reproduction of training.
Failures recorded: initial bwrap loopback startup failed (permissions later changed),
historical Python executable missing, unrelated torchtune import resolved by package
initializer bypass, pdfinfo missing (used pypdf), actual checkpoint CLI fails closed.
None involved model inference or an evaluation data reveal.

Harness deliberately refuses folds outside 4–6. No final evaluator has been enabled.
Pending scientific stages must remain pending until original artifact restoration.
Do not interpret historical or stored outer metrics as fresh development evidence.

Restoration blocker: Sign-chris-s21; dependency added to Sign-chris-v58.
