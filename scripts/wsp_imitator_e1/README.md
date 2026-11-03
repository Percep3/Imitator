# Imitator E1 WSP: development-only harness

## Recovery training authorized on 2026-11-03

The user subsequently authorized retraining when the historical checkpoint search
failed. `recover.py` now runs the existing v121 → CIF → E1 training chain, starting
with fold 4, then folds 5/6, in `outputs/wsp_imitator_e1/recovery/`. This recovery
uses gradients and is explicitly separate from the gradient-free WSP arm and its
50-pass ledger. The original training/model sources and WSP harness are unchanged.
No outer evaluation is launched. All resulting weights will be new checkpoints.

```bash
uv pip sync --python .venv-wsp/bin/python scripts/wsp_imitator_e1/recovery_requirements.lock
.venv-wsp/bin/python -u -m scripts.wsp_imitator_e1.recover chain --folds 4 5 6 --threads 8
```

**Do not launch a duplicate.** Inspect `recovery/job.json`, verify its actual PID,
and check the referenced log first. Completed stages are skipped only after hash
verification; incomplete logs cause refusal rather than silently restarting.
The current live job was launched by this session with a detached coordinator.
The historical preflight snapshot below is preserved, not a current training ban.
The current scientific status is in `outputs/wsp_imitator_e1/final_report.md`.

## Original preflight

The experiment is **blocked before baseline**: original E1 fold-4/5/6 closed
checkpoints and their initializer chains are missing. Sidecars, dataset, adjacency
and the matching tokenizer revision exist. No trained weights changed and no gain
was measured. Research budget: 0/50 passes. See the local report in
`outputs/wsp_imitator_e1/final_report.md` and exact hashes in `BLOCKED.md`.

All additions are isolated from the original model, training and scoring code.
`preflight.py` records evidence and locks a new output directory exactly once.
`harness.py` provides Weights, Ledger and Study; `run.py` exposes development
reproduction/evaluation/probes. `diagnostics.py` only describes predictions.
No training, autograd, optimizer or gold-conditioned greedy decoding is used.

## Setup and checks

```bash
uv venv .venv-wsp --python 3.12
uv pip sync --python .venv-wsp/bin/python scripts/wsp_imitator_e1/requirements.lock
.venv-wsp/bin/python -m pytest -q scripts/wsp_imitator_e1/test_harness.py \
  tests/utils/test_sequence_metrics.py tests/scripts/test_robustness_video_token_decoder.py
```

The recorded run passed 22 tests. Test weights/HDF5 files are artificial fixtures,
not substitute E1 artifacts. The roundtrip test checks the original closed loader.
Avoid running training-oriented legacy tests as part of WSP: some call optimizers.
The harness bypasses only eager legacy model-package initializers (torchtune),
loading actual unmodified E1/ST-GCN implementation modules.

## Preflight and resume

```bash
.venv-wsp/bin/python -m scripts.wsp_imitator_e1.preflight \
  --tokenizer /path/to/original/local/tokenizer/snapshot
```

On this machine, the matching snapshot is
`/home/nakato/.cache/huggingface/hub/models--unsloth--gemma-3n-E2B-it-unsloth-bnb-4bit/snapshots/3d26ffdd2276698f582562bf01511aa625bc6f30`.
The lock cannot be overwritten; do not restart output directories to reset the
budget. Retain the existing study on resume. Restore original checkpoint bytes
from an authorized archive, together with every recorded source ancestor.
Never train or invent replacements. The manifest records expected SHA-256 values.

```bash
.venv-wsp/bin/python -m scripts.wsp_imitator_e1.run --fold 4 \
  --checkpoint /path/to/original/checkpoint_closed.pt \
  --tokenizer /path/to/original/local/tokenizer/snapshot --role reproduction
```

Repeat reproduction for folds 5/6 before experiments. It consumes one single-model
pass of 512 original inner-validation examples per fold; mismatch against recorded
selection metrics at 1e-6 tolerance blocks editing. Discovery/confirmation use fixed
128-example sets from training/inner-validation signer pools respectively. Generation
always calls original `greedy_decode` with only keypoints and frame lengths.
Each perturbation or probe is separately metered. Folds 7–10 are rejected.

Use `Study.weights.load_original`, `save_edit`, `drop_edit`, `list_edits`,
`save_patch`, and `export` only after a preregistered causal case. Patch files retain
both exact original and replacement tensors, changed indices/norms and checksums.
Export preserves the original payload metadata and refuses overwrite. Every edited
entry counts against 0.5% of unique parameter entries. Only localized decoder FFN
and attention-output matrices are initially eligible. No candidate is provided
because the actual model has not been measured.

`evaluate_original`, `evaluate_edited`, `evaluate_paired`, `teacher_forced_probe`
and `activation_intervention` share the journal. Outputs contain repository metrics,
paired CIs/tests, aligned error counts, sign counts/frame durations and gloss order.
Probes explicitly identify gold-prefix diagnostics, never free-running accuracy.
The initial baseline gate and provenance checks are fail-closed. No prospective
outer runner exists; it requires a later algorithm lock and exposure-history audit.

The report and lock are local ignored artifacts. The small `preflight_snapshot/`
archive preserves this blocked session's evidence in Git; it contains no keypoints,
labels, outer predictions, fabricated metrics or trained weights. Append new evidence
to the local study; do not rewrite the archived initial observation.
