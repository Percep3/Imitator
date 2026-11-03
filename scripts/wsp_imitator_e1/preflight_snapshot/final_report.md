# WSP Imitator E1 — blocked at preflight

No conclusion about efficacy is possible: the original trained E1 checkpoint
bytes are absent. No research baseline, teacher-forced probe, causal experiment,
weight edit, control comparison, or outer evaluation ran. Zero of 50 research
passes were consumed. This is a blocked experiment, **not a negative efficacy
result** and not evidence that any mechanism resisted intervention.

## Artifact findings

The local checkout is Percep3/Imitator, base commit
`095789d18381f138b1507f2c70fb594f0a66baff`, rather than the requested remote
nakato156/Imitator-E1. Work is isolated on `research/wsp-imitator-e1`.
No remote was changed. Source equivalence to the public fork was not established.

`dataset1_isolated_v122.hdf5` exists (332,819,822 bytes; SHA-256
`5a2766d61d659c96b1108dcaa668cf13c940c40ed2884c92243473c4282ed5dc`). Its
schema contains 3,200 keypoint datasets plus labels, signer IDs, video IDs and
repetitions. Only schema and streamed byte hashes were read, not clip arrays or
labels. Adjacency is float32, 111×111, SHA-256
`a2c08f2ee0415a95ffcbd05a8a820d691effdad632cb5099871c6cf4f3da2230`.
The historical tokenizer path is absent; a local cache with the SAME recorded
revision `3d26ffdd2276698f582562bf01511aa625bc6f30` loads successfully with
262,400 vocabulary entries. Its individual file hashes are recorded; checkpoint
map equivalence remains unverified without the checkpoint.

All three development directories contain config/closure sidecars but lack the
closed checkpoint. Recorded SHA-256 values (not recomputed checkpoint hashes):

| Fold | Recorded closed-checkpoint SHA-256 |
|---|---|
| 4 | `1e10ee8cb5f865d35c2c170a0d468e5ab4f3a97788eb7269ae0c46c8f73661e3` |
| 5 | `2e12cab3d4eaf6502ee3543b86626fc3d22749252a199a34da56a8680f834a47` |
| 6 | `5f0e4d7a58c7a4cfb0d1935d8b6f978fadaffc8e1667c508338ea74c3b648368` |

The source initializers are also absent. `BLOCKED.md` identifies their recorded
paths and hashes. Filename searches covered the local Sign-AI tree, including
ignored files. Sidecar signer assignments agree with the hashed LOSO manifest,
but do not prove the contents, root ancestry or leakage status of missing weights.
No valid fold-9/10 closed E1 checkpoint or untouched exposure history was certified.
Existing outer result files were not opened, including development outer files.

The historical Sign-env Python executable is absent. Base Python lacks Torch and
HDF5. A new isolated `.venv-wsp` installs CPU packages solely for harness checks;
exact versions are in `environment.json` and the requirements lock. NVIDIA-SMI
cannot communicate with a driver; CUDA is unavailable. CPU execution is supported,
but no trained-model speed or baseline equivalence was measured in this environment.
No dataset, pretrained model or replacement checkpoint was downloaded.

## Source contract versus unverified checkpoint

The inspected autoregressive source defines the ST-GCN/TCN/temporal Transformer
encoder and two-layer, four-head 128/512 decoder. The output projection directly
reuses `token_embedding.weight`; there is no independent output-projection tensor.
E1 sidecars record positional encoding, restricted vocabulary, 30 epochs, seed 23,
zero CTC/smoothing, and edit-similarity checkpoint selection. The historical
README describes an older full-vocabulary 15-epoch protocol; it is not the E1
artifact authority.

One code/documentation mismatch matters: the configured content maximum is 32,
but the exhausted greedy loop can return 33 non-EOS content IDs; the scorer then
assigns length 33. The harness leaves this unchanged. The low-level utility
`strict_exact` checks IDs/length only; the training/evaluation wrapper additionally
requires EOS. That wrapper is reused. Edit similarity is clipped after TER is
normalized by reference length, separately for every example.

Because no actual trained state_dict could be loaded, no actual parameter inventory,
module tree, tied-storage audit, tensor mapping, checkpoint hash verification or
baseline reproduction is claimed. No fake `parameter_inventory.csv` was produced.
The harness emits an actual per-fold inventory only after loading a verified artifact.

## What was implemented and verified

`scripts/wsp_imitator_e1/` provides an artifact preflight, immutable source/protocol
hash checks, a concurrency-safe persistent 50-pass journal, signer-filtered dataset
construction, baseline reproduction gate, original/edited/paired greedy evaluation,
repository robustness transforms, paired bootstrap/sign tests, descriptive token
alignment taxonomy, gold-prefix probes and reversible diagnostic hooks. It also
provides exact replacement-tensor patches, a 0.5% changed-entry guard, forbidden
parameter protection, original-byte preservation and checkpoint export. Any edit
proposal must be preregistered separately; no repair formula is selected by this code.

The initial safe surface is decoder self/cross-attention output matrices and FFN
matrices only, with localized edits; even small whole-tensor edits are rejected.
The original model/evaluation modules remain unchanged. To avoid unrelated legacy
RoPE/torchtune imports, the harness bypasses only two package initializers, following
the existing synthetic-temporal test pattern. Actual E1/ST-GCN classes and evaluation
functions are imported from their original source files, not reimplemented.

22 tests passed: synthetic patch roundtrip, source immutability, forbidden tensors,
dtype/numerics, entry cap, concurrent pass reservation/exhaustion, failed-pass
accounting, signer filtering with deliberately absent outer payloads, paired identity
checks, EOS scoring, alignment and repository robustness tests. A randomly initialized
fixture exported through the patch utility loads under the untouched repository
`load_closed_decoder`; this is compatibility evidence, not a trained E1 result.
A separate CLI invocation failed on the missing fold-4 checkpoint before any
inference, as required. No tests called gradients or optimizers.

The current manuscript was rebuilt with Tectonic 0.16.9 and its new PDF counted
with pypdf: **11 pages**. PDF SHA-256:
`6f93238da4ea61da76612acccba58111f68ce6b2ff889a0b974d19b1ed9896d7`.
No manuscript source was edited. Preexisting user edits and stashes were preserved.

## Required scientific answers

1. **Original failures, quantitatively:** not measured; N=0 research examples.
   Historical paper percentages are not fresh baselines and are not reused here.
2. **Circuit and causal evidence:** none identified. Substitution, alignment,
   visual-information and EOS explanations remain untested alternatives.
3. **Exact tensors changed:** zero trained-checkpoint tensors/entries. No chosen
   operation, patch or edited E1 checkpoint exists.
4. **Gains and controls:** no effect estimate, CI, wins/losses, random control or
   robustness result exists. Software fixture outcomes are excluded from research.
5. **Unseen-signer improvement:** unknown; neither development nor blind evidence
   of improvement was obtained. No prospective final claim is made.
6. **What did not improve:** efficacy, regressions and non-significance cannot be
   assessed. Missing evidence is not a measured zero effect or a failed edit.
7. **What this does not establish:** no conclusion about a repair, a training/data
   bottleneck, general signer performance, or natural sentence-level sign-language
   translation. LSA64 random concatenations do not demonstrate natural SLT.

## Resume conditions and limits

Restore the exact recorded closed checkpoints and initializer ancestry from an
authorized archive; do not train replacements. Run the recorded inner-validation
reproduction first. Inspect the actual inventories/mapping, compare them with the
protocol, and resolve any discrepancy before creating candidates. Baseline mismatch
fails closed rather than silently accepting a new environment's score.

The current harness is development-only. Matched case selection, mechanism-specific
direction estimation, norm/support-matched controls, candidate freeze and a separate
access-controlled prospective runner still require actual evidence. It would be
scientifically invalid to invent them now. Real-data end-to-end inference remains
untested until artifacts return. The 50-pass budget can constrain complete
cross-fold sweeps/robustness and final evaluation; incomplete evidence must be
reported as such, with eight passes reserved if final evaluation becomes possible.
