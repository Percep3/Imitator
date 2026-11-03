"""Audited development-only WSP interface. No training; repository inference only."""
from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import sys
import types

import h5py
import numpy as np
import torch
from torch.utils.data import Subset

# These package initializers eagerly import unrelated legacy RoPE/torchtune models.
# Follow the repository's synthetic-temporal tests: bypass only package __init__,
# loading the actual unchanged E1, STGCN, dataset, and evaluation source modules.
ROOT = Path(__file__).resolve().parents[2]
for package, relative in (("src.mslm.models", "src/mslm/models"),
                          ("src.mslm.models.components", "src/mslm/models/components")):
    if package not in sys.modules:
        namespace = types.ModuleType(package)
        namespace.__path__ = [str(ROOT / relative)]
        sys.modules[package] = namespace

from scripts.train import train_video_token_decoder as repo
from scripts.eval import robustness_video_token_decoder as robustness
from src.mslm.models.video_token_decoder import build_teacher_forcing, validate_checkpoint_lineage
from src.mslm.utils.sequence_metrics import gloss_sequence_diagnostics
from scripts.wsp_imitator_e1.diagnostics import error_taxonomy

SEED = 23


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return repo.sha256_file(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")


def append(path, value):
    with Path(path).open("a") as out:
        fcntl.flock(out, fcntl.LOCK_EX)
        out.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        out.flush()
        os.fsync(out.fileno())


class Ledger:
    """One locked append-only journal; reservations survive errors and process death."""
    def __init__(self, output):
        self.output = Path(output)
        self.protocol = json.loads((self.output / "protocol.json").read_text())
        lock = (self.output / "protocol.sha256").read_text().strip()
        if lock != file_hash(self.output / "protocol.json"):
            raise ValueError("protocol hash changed")
        for name, sha in self.protocol["source_hashes"].items():
            if file_hash(ROOT / name) != sha:
                raise ValueError(f"locked source changed: {name}")
        if self.protocol["pass_limit"] != 50:
            raise ValueError("WSP limit must be 50")
        self.sha = lock
        self.path = self.output / "pass_ledger.jsonl"

    @contextlib.contextmanager
    def pass_(self, kind, sample_ids, checkpoint_hash, patch_hash, **details):
        paired = kind == "paired"
        limit = 256 if paired else 512
        if not sample_ids or len(sample_ids) > limit or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("invalid or oversized sample batch")
        with self.path.open("a+") as log:
            fcntl.flock(log, fcntl.LOCK_EX)
            log.seek(0)
            events = [json.loads(line) for line in log if line.strip()]
            used = sum(event["event"] == "reserved" for event in events)
            if used >= 50:
                raise RuntimeError("50-pass budget exhausted")
            number = used + 1
            target = self.output / "runs" / f"pass_{number:02d}.json"
            event = dict(event="reserved", pass_id=number, kind=kind,
                         sample_hash=digest(sample_ids), sample_ids=sample_ids,
                         checkpoint_sha256=checkpoint_hash, patch_sha256=patch_hash,
                         protocol_sha256=self.sha, seed=SEED, output=str(target),
                         time_ns=time.time_ns(), **details)
            log.write(json.dumps(event, sort_keys=True) + "\n")
            log.flush()
            os.fsync(log.fileno())
        try:
            yield target
        except BaseException as exc:
            append(self.path, dict(event="failed_consumed", pass_id=number,
                                   error=type(exc).__name__ + ": " + str(exc)))
            raise
        else:
            append(self.path, dict(event="completed", pass_id=number,
                                   output_sha256=file_hash(target)))


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def different(a, b):
    """Entrywise byte comparison, including signed zero, after dtype validation."""
    return (a.contiguous().view(torch.uint8).reshape(a.numel(), -1) !=
            b.contiguous().view(torch.uint8).reshape(b.numel(), -1)).any(1)


class Weights:
    """Replacement tensors always relative to the immutable original, never compounded."""
    def __init__(self, model, audit_path):
        self.model = model.eval().requires_grad_(False)
        self.original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.parameters = dict(model.named_parameters())
        self.total = sum(p.numel() for p in self.parameters.values())
        self.edits = {}
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def load_original(self, name):
        return self.original[name].clone()

    def load_edited(self, name):
        return self.edits.get(name, self.original[name]).clone()

    def list_parameters(self):
        return [dict(name=k, shape=list(v.shape), dtype=str(v.dtype), entries=v.numel(),
                     frozen=not self.parameters[k].requires_grad,
                     tied_output=k == "token_embedding.weight",
                     storage=self.parameters[k].untyped_storage().data_ptr())
                for k, v in self.original.items() if k in self.parameters]

    def list_edits(self):
        result = []
        for name, edited in sorted(self.edits.items()):
            original = self.original[name]
            mask = different(original, edited)
            delta = edited.double() - original.double()
            result.append(dict(name=name, changed=int(mask.sum()), fraction=int(mask.sum())/self.total,
                               indices=mask.nonzero().flatten().tolist(),
                               original_sha256=tensor_hash(original), edited_sha256=tensor_hash(edited),
                               delta_l2=float(delta.norm()), delta_max=float(delta.abs().max())))
        return result

    def save_edit(self, name, replacement):
        # Deliberately narrower than the permitted maximum: decoder matrices only.
        allowed = (name.startswith("decoder.layers.") and name.endswith((
            ".self_attn.out_proj.weight", ".multihead_attn.out_proj.weight",
            ".linear1.weight", ".linear2.weight")))
        if not allowed or name not in self.parameters:
            raise ValueError("tensor outside initial decoder editing surface")
        original = self.original[name]
        if replacement.shape != original.shape or replacement.dtype != original.dtype:
            raise ValueError("shape/dtype must match original")
        replacement = replacement.detach().cpu().clone()
        if not torch.isfinite(replacement).all():
            raise ValueError("nonfinite replacement")
        trial = {**self.edits, name: replacement}
        count = sum(int(different(self.original[k], v).sum()) for k, v in trial.items())
        # No entire-tensor edits even when a small tensor fits the global cap.
        if int(different(original, replacement).sum()) == original.numel():
            raise ValueError("whole-tensor edits prohibited")
        if count > math.floor(self.total * .005):
            raise ValueError("0.5% changed-entry budget exceeded")
        self.edits = trial
        append(self.audit_path, dict(event="save_edit", edits=self.list_edits()))

    def drop_edit(self, name):
        self.edits.pop(name, None)
        append(self.audit_path, dict(event="drop_edit", name=name))

    def apply(self, edited):
        with torch.inference_mode():
            self.model.load_state_dict({k: self.edits.get(k, v) if edited else v
                                        for k, v in self.original.items()}, strict=True)
        for k, v in self.model.state_dict().items():
            expected = self.edits.get(k, self.original[k]) if edited else self.original[k]
            if tensor_hash(v) != tensor_hash(expected):
                raise RuntimeError(f"unexpected state mutation: {k}")

    def save_patch(self, path, checkpoint_sha256, preregistration):
        required = {"hypothesis", "evidence", "operation", "indices", "strength",
                    "expected_effect", "falsifier", "random_control", "direction_sample_hash"}
        if not required.issubset(preregistration):
            raise ValueError("candidate preregistration incomplete")
        payload = dict(checkpoint_sha256=checkpoint_sha256, replacements=self.edits,
                       originals={k: self.original[k] for k in self.edits},
                       manifest=self.list_edits(), preregistration=preregistration)
        with Path(path).open("xb") as out:
            torch.save(payload, out)
        append(self.audit_path, dict(event="patch_saved", path=str(path), sha256=file_hash(path),
                                     preregistration=preregistration, edits=self.list_edits()))

    def load_patch(self, path, checkpoint_sha256):
        patch = torch.load(path, map_location="cpu", weights_only=True)
        if patch["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("patch source mismatch")
        previous = self.edits
        self.edits = {}
        try:
            for name, value in patch["replacements"].items():
                if tensor_hash(patch["originals"][name]) != tensor_hash(self.original[name]):
                    raise ValueError("patch original differs")
                self.save_edit(name, value)
            if self.list_edits() != patch["manifest"]:
                raise ValueError("patch manifest mismatch")
        except BaseException:
            self.edits = previous
            raise

    def export(self, original_path, destination):
        if Path(original_path).resolve() == Path(destination).resolve():
            raise ValueError("original checkpoint is immutable")
        payload = torch.load(original_path, map_location="cpu", weights_only=True)
        if any(tensor_hash(payload["model"][k]) != tensor_hash(v) for k, v in self.original.items()):
            raise ValueError("source checkpoint no longer matches original")
        payload["model"] = {k: self.edits.get(k, v) for k, v in self.original.items()}
        with Path(destination).open("xb") as out:
            torch.save(payload, out)
        check = torch.load(destination, map_location="cpu", weights_only=True)
        for name, value in check["model"].items():
            if tensor_hash(value) != tensor_hash(self.edits.get(name, self.original[name])):
                raise RuntimeError("export verification failed")
        # All other payload fields, including provenance and config, are preserved.


def permitted_records(h5_path, spec, role):
    """Read signer membership first; never read excluded labels or keypoints."""
    if int(spec["fold"]) not in (4, 5, 6) or role not in ("discovery", "confirmation", "reproduction"):
        raise ValueError("only development folds 4-6; outer access disabled")
    allowed = set(spec["train_signers"] if role == "discovery" else [spec["inner_val_signer"]])
    if spec["outer_test_signer"] in allowed:
        raise ValueError("outer signer contamination")
    records = []
    with h5py.File(h5_path, "r") as handle:
        group = handle["dataset1"]
        if "signer_id" not in group:
            raise ValueError("explicit signer metadata required")
        for cid in sorted(group["signer_id"], key=int):
            signer = int(group["signer_id"][cid][0])
            if signer not in allowed:
                continue
            video = group["video_id"][cid][0].decode()
            # LSA64 video_id is class_signer_repetition.mp4.
            if int(Path(video).stem.split("_")[1]) != signer:
                raise ValueError("clip/video signer disagreement")
            records.append(dict(clip_id=cid, signer_id=signer,
                                label=group["labels"][cid][0].decode(), video_id=video))
    if not records:
        raise ValueError("no permitted clips")
    return records


def paired_metrics(original, edited):
    if not original or len(original) != len(edited):
        raise ValueError("unequal or empty paired rows")
    for a, b in zip(original, edited):
        if a["sample_id"] != b["sample_id"] or a["target"] != b["target"]:
            raise ValueError("pair identity/target mismatch")
    result = {}
    for index, name in enumerate(("strict_exact", "token_edit_similarity", "token_error_rate")):
        delta = [b[name] - a[name] for a, b in zip(original, edited)]
        result[name] = robustness.bootstrap_mean_ci(delta, seed=7_800_026 + index)
    differences = [b["strict_exact"] - a["strict_exact"] for a, b in zip(original, edited)]
    wins, losses = sum(d > 0 for d in differences), sum(d < 0 for d in differences)
    result.update(wins=wins, losses=losses, ties=len(differences)-wins-losses,
                  sign_test_p=robustness.two_sided_sign_test(wins, losses))
    return result


class Study:
    """Requires artifact hashes and lineage verification before constructing any dataset."""
    def __init__(self, args):
        self.args = args
        self.ledger = Ledger(args.output)
        if args.fold not in (4, 5, 6):
            raise ValueError("outer evaluation unavailable until a separate final lock")
        self.spec = repo.fold_spec(repo.load_manifest(args.manifest), args.fold)
        records = json.loads((Path(args.output) / "checkpoint_manifest.json").read_text())
        entry = next(x for x in records["development"] if x["fold"] == args.fold)
        self.checkpoint_hash = file_hash(args.checkpoint)
        if self.checkpoint_hash != entry["recorded_sha256"]:
            raise ValueError("closed checkpoint checksum differs from recorded original")
        # Inspect only weight/config/provenance payloads, never any outer result files.
        self.state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        robustness.validate_e1_checkpoint(self.state, args.fold, SEED)
        provenance = self.state["provenance"]
        for key in ("train_signers", "inner_val_signer", "outer_test_signer", "fold"):
            if provenance[key] != self.spec[key]:
                raise ValueError(f"provenance mismatch: {key}")
        if provenance["manifest_sha256"] != repo.load_manifest(args.manifest)["manifest_sha256"]:
            raise ValueError("manifest lineage mismatch")
        validate_checkpoint_lineage({"lineage": provenance["lineage"]}, self.spec["outer_test_signer"])
        # Verify every accessible parent, refusing to infer lineage from a missing ancestor.
        source, expected = Path(provenance["source_checkpoint"]), provenance["checkpoint_sha256"]
        visited = set()
        while True:
            if source in visited or file_hash(source) != expected:
                raise ValueError("source ancestry hash/cycle failure")
            visited.add(source)
            ancestor = torch.load(source, map_location="cpu", weights_only=True)
            lineage = validate_checkpoint_lineage(ancestor, self.spec["outer_test_signer"])
            if not lineage.get("train_signers") or not lineage.get("val_signers"):
                raise ValueError("incomplete ancestor signer evidence")
            if not lineage.get("parent_checkpoint"):
                break
            source = Path(lineage["parent_checkpoint"])
            expected = lineage["parent_checkpoint_sha256"]
        cfg = self.state["config"]
        expected_cfg = dict(hidden_size=128, decoder_layers=2, heads=4, ffn_size=512,
                            vocab_size=121, bos_id=2, eos_id=106, pad_id=0, max_content_tokens=32)
        if any(cfg.get(k) != v for k, v in expected_cfg.items()):
            raise ValueError("architecture differs: audit and amend before inference")
        for field in ("h5", "adjacency"):
            if file_hash(getattr(args, field)) != self.ledger.protocol["artifacts"][field]["sha256"]:
                raise ValueError(f"artifact hash mismatch: {field}")
        for name, sha in self.ledger.protocol["artifacts"]["tokenizer"]["files"].items():
            if file_hash(Path(args.tokenizer) / name) != sha:
                raise ValueError("tokenizer hash mismatch")
        self.device = torch.device(args.device)
        self.model, _ = repo.load_closed_decoder(args, self.spec, self.device)
        self.weights = Weights(self.model, Path(args.output) / "candidates/candidate_manifest.jsonl")
        self.tokenizer = repo.load_tokenizer(args.tokenizer)
        self.tokens = repo.label_tokens(permitted_records(args.h5, self.spec, "discovery"), self.tokenizer)
        if repo.make_vocab_map(self.tokens) != cfg["vocab_map"]:
            raise ValueError("actual token map differs from permitted training-label tokenizer map")
        inventory = Path(args.output) / f"parameter_inventory_fold{args.fold}.csv"
        if not inventory.exists():
            rows = self.weights.list_parameters()
            with inventory.open("x") as out:
                writer = csv.DictWriter(out, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            write_json(inventory.with_suffix("modules.json"),
                       {n: repr(m) for n, m in self.model.named_modules()})

    def _dataset(self, role, indices):
        if role not in ("discovery", "confirmation", "reproduction"):
            raise ValueError("outer access prohibited")
        expected = list(range(512)) if role == "reproduction" else list(range(128))
        if indices != expected:
            raise ValueError("only precommitted sample IDs allowed")
        seed = 100_023 if role == "reproduction" else (300_023 if role == "discovery" else 400_023)
        records = permitted_records(self.args.h5, self.spec, role)
        ds = repo.make_dataset(self.args, records, self.tokens, samples=max(indices)+1, seed=seed)
        ids = [f"f{self.args.fold}:{role}:{seed}:{i}" for i in indices]
        return Subset(ds, indices), ids, seed

    def _reset(self, edited):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        self.weights.apply(edited)
        self.model.eval()

    def evaluate_original(self, role="discovery", condition="clean"):
        return self.evaluate(role, condition, "original")

    def evaluate_edited(self, role="discovery", condition="clean"):
        return self.evaluate(role, condition, "edited")

    def evaluate_paired(self, role="confirmation", condition="clean"):
        return self.evaluate(role, condition, "paired")

    def evaluate(self, role, condition, arm):
        if arm not in ("paired", "original", "edited"):
            raise ValueError("unknown arm")
        indices = list(range(512 if role == "reproduction" else 128))
        dataset, ids, seed = self._dataset(role, indices)
        transforms = robustness.condition_transforms()
        if condition not in transforms:
            raise ValueError("unknown perturbation")
        if role == "reproduction" and (arm != "original" or condition != "clean"):
            raise ValueError("reproduction is original clean only")
        gate = Path(self.args.output) / f"baseline/reproduced_fold{self.args.fold}.json"
        if role != "reproduction" and not gate.exists():
            raise ValueError("reproduce the original inner-validation baseline before experimentation")
        with self.ledger.pass_("paired" if arm == "paired" else "generation", ids,
                              self.checkpoint_hash, digest(self.weights.list_edits()),
                              role=role, condition=condition, dataset_seed=seed) as path:
            results = {}
            with torch.inference_mode():
                for edited in ([False, True] if arm == "paired" else [arm == "edited"]):
                    self._reset(edited)
                    start = time.perf_counter()
                    metrics, rows = robustness.evaluate_condition(
                        self.model, repo.make_loader(dataset, 2, False), self.device,
                        self.tokenizer, transforms[condition])
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    for row, sample_id in zip(rows, ids):
                        row["sample_id"] = sample_id
                        row["fold"] = self.args.fold
                        row["premature_eos"] = row["emitted_eos"] and len(row["pred"]) < len(row["target"])
                    results["edited" if edited else "original"] = dict(
                        metrics=metrics, rows=rows, elapsed_seconds=time.perf_counter()-start,
                        error_taxonomy=error_taxonomy(rows),
                        gloss_diagnostics=gloss_sequence_diagnostics(
                            [r["pred"] for r in rows], [r["target"] for r in rows], self.tokens))
            if arm == "paired":
                results["paired"] = paired_metrics(results["original"]["rows"], results["edited"]["rows"])
            write_json(path, results)
        if role == "reproduction":
            actual = results["original"]["metrics"]
            expected = self.state["best_select_metric"]
            if any(abs(actual[k] - v) > 1e-6 for k, v in zip(
                    ("token_edit_similarity", "strict_exact"), expected)):
                raise RuntimeError("fresh baseline differs from recorded inner-val; no editing allowed")
            write_json(gate, dict(pass_output=str(path), checkpoint_sha256=self.checkpoint_hash,
                                 metrics=actual, tolerance=1e-6))
        return results

    def teacher_forced_probe(self, layers, intervention=None):
        """Diagnostic only: bounded module summaries and gold-prefix margins; no gradients."""
        if not (Path(self.args.output) / f"baseline/reproduced_fold{self.args.fold}.json").exists():
            raise ValueError("baseline reproduction required")
        dataset, ids, seed = self._dataset("discovery", list(range(128)))
        modules = dict(self.model.named_modules())
        if not layers or any(n not in modules or not n.startswith("decoder.layers.") for n in layers):
            raise ValueError("explicit decoder hook locations required")
        if intervention and (intervention["layer"] not in layers or
                             intervention["scale"] not in (0., .5, 1.)):
            raise ValueError("intervention must target a recorded layer with fixed scale")
        with self.ledger.pass_("teacher_forced", ids, self.checkpoint_hash,
                              digest(self.weights.list_edits()), dataset_seed=seed,
                              layers=layers, intervention=intervention) as path:
            self._reset(False)
            traces, decisions, handles = [], [], []
            def hook(name):
                def apply(module, inputs, output):
                    value = output[0] if isinstance(output, tuple) else output
                    traces.append(dict(layer=name, shape=list(value.shape), norm=float(value.norm())))
                    if intervention and name == intervention["layer"]:
                        changed = value.clone()
                        changed[..., intervention["indices"]] *= intervention["scale"]
                        return (changed, *output[1:]) if isinstance(output, tuple) else changed
                return apply
            try:
                for name in layers:
                    handles.append(modules[name].register_forward_hook(hook(name)))
                with torch.inference_mode():
                    for batch in repo.make_loader(dataset, 2, False):
                        x, lengths, tokens = repo.move_core_batch(batch, self.device)
                        inputs, labels = build_teacher_forcing(self.model.to_dense(tokens),
                            bos_id=self.model.bos_id, eos_id=self.model.eos_id, pad_id=self.model.pad_id)
                        logits = self.model(x, lengths, inputs)
                        competitors = logits.clone().scatter_(-1, labels.unsqueeze(-1), -torch.inf)
                        margins = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)-competitors.max(-1).values
                        for i, row in enumerate(labels):
                            valid = row != self.model.pad_id
                            decisions.append(dict(sample_id=ids[len(decisions)],
                                gold_margin=margins[i, valid].cpu().tolist(),
                                gold_log_probability=logits.log_softmax(-1)[i, valid].gather(
                                    -1, row[valid, None]).squeeze(-1).cpu().tolist(),
                                predicted_dense=logits[i, valid].argmax(-1).cpu().tolist(),
                                target_dense=row[valid].cpu().tolist()))
            finally:
                for handle in handles:
                    handle.remove()
            result = dict(diagnostic_only=True, gold_prefix=True, decisions=decisions, traces=traces)
            write_json(path, result)
        return result

    def activation_intervention(self, layer, indices, scale):
        return self.teacher_forced_probe([layer], dict(layer=layer, indices=indices, scale=scale))
