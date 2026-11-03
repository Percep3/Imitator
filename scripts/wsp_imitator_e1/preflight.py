"""Non-inference preflight and immutable protocol lock; never opens outer results."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys

from scripts.wsp_imitator_e1.harness import ROOT, file_hash, write_json, repo, torch, np, h5py


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/wsp_imitator_e1")
    parser.add_argument("--artifact-root", type=Path, default=ROOT.parent)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError("protocol already locked: refusing to overwrite this study")
    manifest = repo.load_manifest(repo.DEFAULT_MANIFEST)
    development, blockers = [], []
    for fold in (4, 5, 6):
        folder = args.artifact_root / "outputs/video_token_decoder" / f"fold{fold}_seed23/e1_pe_vocab121"
        closed = json.loads((folder / "closed.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        checkpoint = Path(closed["checkpoint"])
        spec = repo.fold_spec(manifest, fold)
        provenance = config["provenance"]
        agreement = all(provenance[k] == spec[k] for k in ("fold", "train_signers", "inner_val_signer", "outer_test_signer"))
        row = dict(fold=fold, checkpoint=str(checkpoint), exists=checkpoint.is_file(),
                   recorded_sha256=closed["checkpoint_sha256"],
                   actual_sha256=file_hash(checkpoint) if checkpoint.is_file() else None,
                   config_path=str(folder / "config.json"), config_sha256=file_hash(folder / "config.json"),
                   closed_record_sha256=file_hash(folder / "closed.json"),
                   recorded_config=config, manifest_agreement=agreement,
                   provenance_status="sidecar evidence only; tensor payload and ancestry unverified")
        development.append(row)
        if not checkpoint.is_file():
            blockers.append(f"Fold {fold} closed E1 checkpoint missing: `{checkpoint}`; expected SHA-256 `{closed['checkpoint_sha256']}`.")
        source = Path(provenance["source_checkpoint"])
        if not source.is_file():
            blockers.append(f"Fold {fold} initializer missing: `{source}`; expected SHA-256 `{provenance['checkpoint_sha256']}`; restore its recorded parent chain too.")
    write_json(out / "checkpoint_manifest.json", dict(development=development,
        reserved_folds="9-10 not opened; existence/history of valid closed E1 artifacts not certified",
        searched_roots=[str(args.artifact_root)],
        search_scope="filenames only for checkpoint_closed.pt/checkpoint_best.pt; no outer result content read"))
    artifacts = {}
    for key, path in (("h5", repo.DEFAULT_H5), ("adjacency", repo.DEFAULT_ADJACENCY)):
        artifacts[key] = dict(path=str(path), exists=path.is_file(),
                              sha256=file_hash(path) if path.is_file() else None)
        if not path.is_file():
            blockers.append(f"Missing {key}: {path}")
    if repo.DEFAULT_H5.is_file():
        with h5py.File(repo.DEFAULT_H5, "r") as handle:
            # Schema only. No keypoint arrays, labels, or outer example metadata loaded.
            artifacts["h5"]["dataset1_groups"] = list(handle["dataset1"].keys())
            artifacts["h5"]["keypoint_datasets"] = len(handle["dataset1/keypoints"])
    if repo.DEFAULT_ADJACENCY.is_file():
        adjacency = np.load(repo.DEFAULT_ADJACENCY, allow_pickle=False)
        artifacts["adjacency"].update(shape=list(adjacency.shape), dtype=str(adjacency.dtype))
    tokenizer_files = {p.name:file_hash(p) for p in sorted(args.tokenizer.glob("*"))
                       if p.is_file() and ("token" in p.name or p.name == "special_tokens_map.json")}
    artifacts["tokenizer"] = dict(path=str(args.tokenizer), files=tokenizer_files)
    try:
        tokenizer = repo.load_tokenizer(args.tokenizer)
        artifacts["tokenizer"]["vocab_size"] = tokenizer.vocab_size
    except Exception as exc:
        blockers.append("Local tokenizer load failed: " + repr(exc))
    sources = ["src/mslm/models/video_token_decoder.py", "src/mslm/models/temporal_sign_prompt.py",
               "src/mslm/models/components/stgcn.py", "scripts/train/train_video_token_decoder.py",
               "scripts/eval/robustness_video_token_decoder.py", "src/mslm/dataloader/synthetic_temporal.py",
               "src/mslm/dataloader/data_augmentation.py", "src/mslm/utils/sequence_metrics.py",
               "experiments/a3_etapa4_clean_loso/manifest.json"]
    sources += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "scripts/wsp_imitator_e1").glob("*.py"))]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    protocol = dict(schema_version=1, status="locked_preflight_blocked" if blockers else "locked_preflight",
        source_commit=commit, source_hashes={s:file_hash(ROOT / s) for s in sources},
        pass_limit=50, max_single=512, max_paired=256, seed=23,
        artifacts=artifacts, manifest_sha256=manifest["manifest_sha256"],
        development_folds=[4,5,6], splits={str(f):repo.fold_spec(manifest,f) for f in (4,5,6)},
        discovery=dict(role="train_signers", seed=300023, indices=list(range(128))),
        confirmation=dict(role="inner_val_signer", seed=400023, indices=list(range(128))),
        reproduction=dict(role="inner_val_signer", seed=100023, indices=list(range(512)), tolerance=1e-6),
        decoding=dict(implementation="unchanged VideoTokenDecoder.greedy_decode", batch_size=2,
                      max_content_tokens_config=32, exhausted_loop_may_return_33=True, eos_forbidden_step_zero=True),
        metrics=dict(primary="strict_exact: all content IDs equal AND EOS emitted",
                     secondary="mean(max(0,1-Levenshtein(pred,target)/len(target)))",
                     ter="mean per-example Levenshtein(pred,target)/len(target), not clipped",
                     other=["text_exact", "eos_rate", "no_eos_rate", "premature_eos", "length_mae",
                            "alignment operations/token/position/sign_count/frame_duration/fold", "pairwise gloss order"],
                     premature_eos="EOS and predicted content shorter than reference; descriptive proxy",
                     ci="paired percentile bootstrap, 10000 replicates, seed 7800026; descriptive sample uncertainty",
                     test="exact two-sided paired sign test on strict-exact discordances"),
        edit_rules=dict(max_fraction=.005, denominator="unique named parameter entries; no buffers",
                        allowed=["decoder.layers.*.self_attn.out_proj.weight", "decoder.layers.*.multihead_attn.out_proj.weight",
                                 "decoder.layers.*.linear1.weight", "decoder.layers.*.linear2.weight"],
                        no_entire_tensor=True, no_training=True, no_optimizer=True,
                        immutable=["token_embedding", "final_norm", "output_bias", "position_embedding",
                                   "frame_encoder", "config", "provenance", "tokenizer", "scoring", "decoding"]),
        candidate_policy=dict(max_families=1, strengths=[.5,1.,2.],
            preregister_before_inference=True, no_op=True, same_norm_same_support_random=True,
            evidence="matched discovery examples plus localized forward causal interventions required before any patch"),
        selection=dict(primary="maximize pooled development strict_exact", secondary="token_edit_similarity",
            accept="paired strict CI lower > 0; same-sign strict deltas on all three folds; beat matched random control; no sign-count stratum or robustness condition with strict delta < -0.02 or edit-sim delta < -0.02",
            confirmation="direction and magnitude fixed before inspecting confirmation; one family only; no further tuning after confirmation",
            negative="no accepted candidate is an acceptable outcome"),
        robustness=dict(sigmas=[.01,.02,.05], dropout=[.1,.2], speeds=[.75,1.25],
                        implementation="scripts/eval/robustness_video_token_decoder.py"),
        budget_policy="Reserve before inference or sample materialization; failed reservations consume; no external unmetered inference. Stop rather than exceed 50. Full coverage may be infeasible within budget: report incomplete, never declare success.",
        blind_plan=dict(enabled=False, folds=[9,10], previously_reported=[7,8],
            requirements=["valid closed checkpoints and audited unseen outer history", "all algorithm/code/strength hashes frozen",
                          "separate access-controlled outer runner; one reveal per fold; no later edits",
                          "8 remaining passes for 896 paired rows per fold in blocks of <=256; preserve batch=2 and global sample seeds"],
            status="not authorized by this development runner; no claim of prospective validity"))
    write_json(out / "protocol.json", protocol)
    (out / "protocol.sha256").write_text(file_hash(out / "protocol.json") + "\n")
    env = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
               packages={p:importlib.metadata.version(p) for p in ("torch","numpy","h5py","transformers","pytest")},
               cuda_available=torch.cuda.is_available(), cuda_build=torch.version.cuda,
               cpu_count=__import__("os").cpu_count(), memory=Path("/proc/meminfo").read_text(),
               git_commit=commit, branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
               remote=subprocess.check_output(["git","remote","get-url","origin"],text=True).strip(),
               requested_repository="https://github.com/nakato156/Imitator-E1",
               artifacts=artifacts)
    write_json(out / "environment.json", env)
    with (out / "pass_ledger.jsonl").open("x") as log:
        log.write(json.dumps(dict(event="preflight", inference=False, passes_consumed=0,
                                 blockers=blockers, protocol_sha256=file_hash(out / "protocol.json"))) + "\n")
    if blockers:
        (out / "BLOCKED.md").write_text("# Blocked before baseline inference\n\n" + "\n\n".join(blockers) +
            "\n\nRestore original bytes from an authorized archive; do not train replacements. Sidecars alone cannot verify tensors, vocabulary mapping, tied storage, or ancestry. No benchmark predictions or patch exists.\n")
    print(json.dumps(dict(output=str(out), blocked=bool(blockers), missing=len(blockers), passes=0)))


if __name__ == "__main__":
    main()
