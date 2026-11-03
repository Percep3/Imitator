# Dataset2 Scientific Audit — shared spec for all sub-agents

Repo: /shared/Code/Sign-AI/Sign-chris (Imitator-E1). Branch v125-temporal-prompt-alignment. Date 2026-10-19.
Goal: decide whether dataset2 gives reliable enough supervision to train continuous LSA (Lengua de Señas Argentina) -> Spanish translation.

## HARD RULES (all agents)
- NO training, NO GPU inference, NO model/dataset downloads.
- NEVER modify/delete HDF5, videos, checkpoints, splits, or prior experiment outputs. Open HDF5 read-only (`h5py.File(p, "r")`).
- No commits, no pushes, no repo restructuring, no indiscriminate pip installs.
  Python env: conda env `Sign-env` (try `conda run -n Sign-env python ...` or find its python under ~/miniconda*/envs or /opt/conda/envs). Do NOT pip install into Sign-env (it breaks torch). If a lib is missing, look for another existing env before declaring a block.
- Helper code ONLY under artifacts/dataset2_audit/scripts/ (independent of training pipeline). Working notes under artifacts/dataset2_audit/_work/.
- Never copy videos to public places. Lightweight previews only under artifacts/dataset2_audit/_private_previews/ (if at all).
- Never fabricate data, timestamps, word-frame correspondences, or semantic-quality percentages. Semantic correctness requires human LSA review.
- Do not assume HEAD code == code that produced historical artifacts; check git log / file mtimes.
- Evidence tags for every important claim: VERIFIED (direct inspection / reproducible computation), DOCUMENTED (docs/historical results), INFERRED (plausible interpretation), UNKNOWN.
- Every key claim must cite file:line ranges, commands, artifact paths or computed stats.
- Prior-audit numbers are hypotheses to verify, not facts: ~8,459 original videos; 5,600 clips in HDF5; 5,138 kept by v125 filter; split 4,112/513/513; 111 2D keypoints (RTMPose); historical "3,088 clips misaligned >1s" has NO recovered procedure — do not treat as fact.

## Key files to start from
docs/research_audit_imitator_e1.md; scripts/data/build_dataset2_h5.py; scripts/data/build_dataset2_manifest.py; scripts/data/build_dataset2_split.py; scripts/data/backfill_dataset2_metadata.py; scripts/data/rebuild_dataset2_alignment_targets_v125.py; src/mslm/dataloader/integrity_filter.py; src/mslm/dataloader/grouped_split.py; scripts/data/data_eda.py; configs/diagnostics for v116–v119, v125. Data dirs: /shared/Code/Sign-AI/data/processed, /shared/Code/Sign-AI/outputs (and whatever the scripts reference).

## Final deliverables (assembled at the end)
docs/dataset2_scientific_audit.md and artifacts/dataset2_audit/{dataset_statistics.json, integrity_report.csv, sample_manifest.csv, human_review_template.csv, audit_methodology.md, reproduction_commands.md}.
Report sections: 1 Exec summary; 2 Provenance & environment; 3 Dataset2 reconstruction; 4 Real file inventory; 5 Visual & text stats; 6 Integrity audit; 7 Available alignment evaluation (Level A computational / B temporal / C semantic); 8 Leakage risk; 9 Human review design; 10 Implications for v116–v119; 11 SLT viability (scenarios A/B/C); 12 Max 3 recommended experiments (hypothesis, data, architecture reused, baseline, loss, metrics, controls incl. shuffle/text-only/leakage, success criterion, risk; consider existing encoder, Gemma, mT5 without presupposing; hyperbolic/Geo-Sign only as later option); 13 Evidence, paths, hashes, reproducible commands; 14 Limitations & open questions.

## Completion questions the audit must answer
1 What dataset2 really contains. 2 Provenance / how video-text pairs were built. 3 Usable examples after computational filters. 4 Known/unknown on temporal correspondence. 5 Known/unknown on semantic correspondence. 6 Train/val/test leakage risks. 7 Data problems that could affect v116–v119. 8 What extra evidence human LSA review needs. 9 Can a new continuous-translation experiment be justified? 10 Next minimal controlled reproducible experiment.
Do not confuse "passes computational checks" with "linguistically certified".

## Shared env note (added by coordinator)
Sign-env is EMPTY (no python). Use the existing venv with h5py (read-only use, do not install into it):
/shared/Code/nakato/projects/VQ_RL/.venv/bin/python  (py3.13, h5py 3.16, numpy 2.3.1). No env has cv2; use ffprobe for video probing.
Main data: HDF5 /shared/Code/Sign-AI/data/processed/dataset_v6_unsloth.hdf5 ; manifest dataset2_manifest_v125.json ; split dataset2_split_v125.json ; raw videos /shared/Code/Sign-AI/data/raw/dataset2/videos/ ; per-video stats artifacts/data_eda/video_metadata.csv
