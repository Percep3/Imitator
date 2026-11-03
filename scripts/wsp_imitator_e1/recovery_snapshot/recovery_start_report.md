# Imitator E1 recovery training and WSP — in progress

The user explicitly authorized recovery training on 2026-11-03. This supersedes
the earlier prohibition on training replacements, **only for recovery training**.
The subsequent WSP treatment remains gradient-free and limited to 50 passes.
The original preflight report is preserved in `recovery/pre_recovery_report.md`
and the committed `scripts/wsp_imitator_e1/preflight_snapshot/` archive.

## Current execution

No original E1 checkpoint was recovered. A bounded search covered /shared,
/home/nakato, /tmp, /mnt and /media (84,706 directories; 6.46 seconds), excluding
package caches and virtual environments; relevant Git histories and LFS locations
were also checked. No E1 candidate existed to verify against historical hashes.
The 159 MB Git object store has no checkpoint-path history; the public clone has
no checkpoint history either. Git LFS is not installed and no local LFS object
store or LFS attributes were found. Unrelated model files were not loaded.
Search scope/exclusions and all candidate filenames are in recovery/bounded_search.json.

A direct CUDA driver call returned 100 / CUDA_ERROR_NO_DEVICE. PCI exposes a
QEMU display, with no compute GPU, NVIDIA device nodes, ROCm device or DRM render
node. PyTorch reports no CUDA/MPS/XPU acceleration. CPU is a virtualized Ryzen
7600X with 11 available logical processors and approximately 5.9 GiB initially
available RAM. No paid service or system driver change was made.

Recovery uses the existing `.venv-wsp` CPU environment, adding tensorboard and
sacrebleu dependencies. A bounded training-only timing run used permitted fold-4
training inputs. Three E1 minibatches took 15.07 / 6.05 / 5.01 seconds with
1 / 4 / 8 threads; a v121 batch of 16 took 6.66 seconds at one thread. These
are resource measurements on disposable initial weights, **not transcription
baselines or WSP probes**. The tested synthetic lengths ranged from 268 to 1116
frames; vocabulary construction from permitted training labels yielded 121 IDs.

The original reconstruction chain is now running, prioritizing fold 4:
v121 (up to 80 epochs, original early stopping) → A1 (30) → A2 (40) →
A3_pre_decoder (30) → decoder_final (15) → promotion (1) → E1 (30).
The coordinator then proceeds to folds 5 and 6. This will require substantial
CPU time; the bounded benchmark does not establish a precise completion time.
Check `recovery/job.json`, its actual PID and the referenced stage log for current
state. A stale job.json alone is not evidence that training remains live.

All trainer/model sources remain unchanged. The recovery wrapper redirects
outputs, uses the local tokenizer revision, supplies the correct fold-specific
v121 initializer to every CIF stage and filters signer metadata BEFORE reading
labels or clips. It records A1's v121 parent explicitly, fixing a provenance gap
in the original metadata without changing the model or training computation.
New checkpoints are labeled retrained; historical identity is never asserted.

Fold 4 uses 2,560 training clips and 320 signer-5 inner-validation clips; signer 4
is excluded. The rotating manifest's training members may include people 9/10;
no fold-9/10 experiment, outer input or result is consulted. Folds 7/8 outer
results are likewise not consulted. No outer evaluation is launched by recovery.

## Experimental evidence not yet available

There is not yet a closed recovered E1 checkpoint. Consequently there are no
fresh E1 strict-exact/edit-similarity/TER/EOS baselines, no diagnosed E1 circuit,
no WSP tensor edits, no paired improvement or regression estimates, and no random
control comparison. WSP budget remains **0/50**. Initializer classification metrics
must not be presented as E1 transcription scores. No model improvement is claimed.

After E1 closes, record its hash and complete lineage, explicitly amend the
historical-checkpoint admission contract for the new artifact without resetting
the ledger, reproduce its inner baseline, then conduct the original causal WSP
procedure. No architecture, scoring, token mapping or greedy-decoding changes
are authorized as WSP gains. The goal remains active until actual paired evidence
is obtained, or a material external blocker is established.

These experiments concern controlled LSA64 sign concatenations, not natural
sentence-level sign-language translation. The paper was not rebuilt and the
prior 22-test harness suite was not rerun during recovery setup.
