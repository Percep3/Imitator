# Sign-chris-1tl review record

This folder contains the scoped manuscript diff for the reviewer request about
training settings and keypoint normalization. The main paper change is in
`paper/content/full/sections/05_data_protocol.tex`; the patch isolates that
paragraph from other pre-existing edits in the working tree. This patch is the
minimal addition against the manuscript version in `HEAD`, leaving unrelated
local wording changes out of the archived diff.

## Audited evidence

- Final E1 run configs for folds 4–8 (paths relative to the repository root):
  `../outputs/video_token_decoder/fold{4,5,6,7,8}_seed23/e1_pe_vocab121/config.json`.
  Each records seed 23, 30 epochs, physical batch size 2, and gradient
  accumulation 2. The corresponding closed and outer-test result JSON files are
  present; no new training run was needed for this documentation correction.
- `scripts/train/train_video_token_decoder.py`: `optimizer_for` defines AdamW
  with weight decay `1e-4`, decoder LR `3e-4`, and visual TCN/Transformer LR
  `3e-5`. E1 configs set `unfreeze_stgcn_epoch` to null.
- `src/mslm/models/video_token_decoder.py`: the visual TCN and Transformer are
  enabled starting at zero-indexed epoch 5, after five decoder-only epochs;
  ST-GCN layers and spatial projection stay frozen for E1.
- `src/mslm/dataloader/synthetic_temporal.py` and
  `src/mslm/dataloader/data_augmentation.py`: each clip is reduced from 133 to
  111 landmarks and normalized before clips are concatenated and neutral gaps
  inserted. Normalization takes absolute coordinates, excludes near-zero points
  from per-axis extrema, min--max scales valid points, and preserves invalid
  markers.

The checkpoint binaries referenced by the frozen manifest are not available in
this workspace; the run configs and their recorded closed/outer-test results are
available and were used for this audit.
