"""Artificial fixtures only; no trained E1 checkpoint or LSA64 sample is used."""
import json
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.wsp_imitator_e1.harness import (
    Weights, Ledger, permitted_records, paired_metrics, tensor_hash, file_hash,
    write_json, torch, np, h5py, repo, robustness,
)
from scripts.wsp_imitator_e1.diagnostics import alignment, error_taxonomy
from src.mslm.models.video_token_decoder import VideoTokenDecoder


@pytest.fixture(autouse=True)
def no_grad():
    torch.set_num_threads(1)
    with torch.inference_mode():
        yield


class Frames(torch.nn.Module):
    def forward(self, x, lengths):
        return x


def tiny():
    torch.manual_seed(23)
    return VideoTokenDecoder(Frames(), hidden_size=8, vocab_size=16, num_layers=2,
                             num_heads=2, ffn_size=32, dropout=0., max_content_tokens=4,
                             bos_id=2, eos_id=3, pad_id=0).eval().requires_grad_(False)


def locked(tmp_path):
    write_json(tmp_path / "protocol.json", dict(pass_limit=50, source_hashes={}))
    (tmp_path / "protocol.sha256").write_text(file_hash(tmp_path / "protocol.json"))
    return Ledger(tmp_path)


def test_reversible_patch_budget_export_and_forbidden_targets(tmp_path):
    model = tiny()
    weights = Weights(model, tmp_path / "audit.jsonl")
    name = "decoder.layers.0.linear2.weight"
    original = weights.load_original(name)
    changed = original.clone()
    changed[0,0] += .01
    weights.save_edit(name, changed)
    assert weights.list_edits()[0]["changed"] == 1
    assert torch.equal(weights.load_original(name), original)
    changed.zero_()
    assert not torch.equal(weights.load_edited(name), changed)
    weights.apply(True)
    assert not torch.equal(model.state_dict()[name], original)
    weights.apply(False)
    assert all(tensor_hash(v) == tensor_hash(weights.original[k]) for k,v in model.state_dict().items())
    for forbidden in ("token_embedding.weight", "final_norm.weight", "output_bias", "position_embedding.weight"):
        with pytest.raises(ValueError, match="surface"):
            weights.save_edit(forbidden, weights.load_original(forbidden))
    with pytest.raises(ValueError, match="dtype"):
        weights.save_edit(name, original.double())
    with pytest.raises(ValueError, match="nonfinite"):
        weights.save_edit(name, original * float("nan"))
    too_many = original.clone()
    too_many.flatten()[:int(weights.total*.005)+1] += 1
    with pytest.raises(ValueError, match="budget"):
        weights.save_edit(name, too_many)
    prereg = {k:"fixture only" for k in ("hypothesis", "evidence", "operation", "indices", "strength",
                "expected_effect", "falsifier", "random_control", "direction_sample_hash")}
    weights.save_patch(tmp_path / "patch.pt", "source", prereg)
    fresh = Weights(tiny(), tmp_path / "fresh.jsonl")
    fresh.load_patch(tmp_path / "patch.pt", "source")
    assert fresh.list_edits() == weights.list_edits()
    with pytest.raises(ValueError, match="source"):
        fresh.load_patch(tmp_path / "patch.pt", "wrong")
    source = tmp_path / "original.pt"
    torch.save(dict(model=weights.original, config={"unchanged":True}, provenance={"fixture":True}), source)
    before = file_hash(source)
    weights.export(source, tmp_path / "edited.pt")
    assert file_hash(source) == before
    restored = torch.load(tmp_path / "edited.pt", weights_only=True)
    assert restored["config"] == {"unchanged":True}
    assert restored["provenance"] == {"fixture":True}
    weights.drop_edit(name)
    weights.apply(True)
    assert not weights.list_edits()


def test_ledger_concurrency_failure_and_51st_refusal(tmp_path):
    ledger = locked(tmp_path)
    def reserve(i):
        try:
            with ledger.pass_("generation", [f"fixture:{i}"], "untrained-fixture", "noop") as output:
                write_json(output, {"fixture":True})
            return True
        except RuntimeError:
            return False
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert sum(workers.map(reserve, range(55))) == 50
    assert len({json.loads(s)["pass_id"] for s in ledger.path.read_text().splitlines()
                if json.loads(s)["event"] == "reserved"}) == 50
    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        Ledger(tmp_path)


def test_failures_consume_and_oversized_batches_do_not_start(tmp_path):
    ledger = locked(tmp_path)
    with pytest.raises(RuntimeError, match="fixture crash"):
        with ledger.pass_("generation", ["x"], "fixture", "noop"):
            raise RuntimeError("fixture crash")
    assert "failed_consumed" in ledger.path.read_text()
    with pytest.raises(ValueError, match="oversized"):
        with ledger.pass_("paired", list(map(str,range(257))), "fixture", "noop"):
            pytest.fail("should not enter")
    with pytest.raises(ValueError):
        with ledger.pass_("generation", ["duplicate"]*2, "fixture", "noop"):
            pytest.fail("should not enter")


def test_signer_guard_filters_before_label_or_keypoint_reads(tmp_path):
    path = tmp_path / "artificial.h5"
    with h5py.File(path, "w") as h:
        group = h.create_group("dataset1")
        # Outer clip has deliberately missing label/video/keypoint payloads.
        for cid, signer in (("1",1),("2",4),("3",5)):
            group.create_dataset("signer_id/"+cid, data=[signer])
            if signer != 4:
                group.create_dataset("labels/"+cid, data=[b"fixture"])
                group.create_dataset("video_id/"+cid, data=[f"001_{signer:03d}_001.mp4".encode()])
    spec = dict(fold=4, outer_test_signer=4, inner_val_signer=5, train_signers=[1])
    assert [r["clip_id"] for r in permitted_records(path,spec,"discovery")] == ["1"]
    assert [r["clip_id"] for r in permitted_records(path,spec,"confirmation")] == ["3"]
    for fold in (7,8,9,10):
        with pytest.raises(ValueError):
            permitted_records(path, {**spec,"fold":fold}, "discovery")
    with pytest.raises(ValueError):
        permitted_records(path,spec,"outer")


def test_paired_metrics_eos_alignment_and_taxonomy():
    _, original = repo.sequence_metrics([[7],[7],[8]], [False,True,True], [[7],[7],[7]])
    _, edited = repo.sequence_metrics([[7],[8],[7]], [True,True,True], [[7],[7],[7]])
    for rows in (original,edited):
        for i,row in enumerate(rows):
            row.update(sample_id=str(i),sign_count=2,frame_length=10,fold=4)
    assert original[0]["strict_exact"] == 0 and original[0]["length_mae"] == 32
    metrics = paired_metrics(original,edited)
    assert (metrics["wins"],metrics["losses"],metrics["ties"]) == (2,1,0)
    assert metrics["strict_exact"]["estimate"] == pytest.approx(1/3)
    edited[0]["sample_id"] = "bad"
    with pytest.raises(ValueError,match="identity"):
        paired_metrics(original,edited)
    assert [e["op"] for e in alignment([1,3,4],[1,2])] == ["insertion","substitution"]
    taxonomy = error_taxonomy(original)
    assert taxonomy["operations"]["missing_eos_samples"] == 1


def test_metered_real_decoder_fixture_noop_and_hook_cleanup(tmp_path):
    ledger = locked(tmp_path)
    model = tiny()
    x, lengths = torch.randn(2,3,8), torch.tensor([3,2])
    with ledger.pass_("paired", ["fixture:0","fixture:1"], "untrained-fixture", "noop") as output:
        assert not torch.is_grad_enabled()
        first = model.greedy_decode(x,lengths)
        second = model.greedy_decode(x,lengths)
        assert first.token_ids == second.token_ids
        assert torch.equal(first.emitted_eos, second.emitted_eos)
        write_json(output, {"fixture_only":True,"passed":True})
    with ledger.pass_("teacher_forced", ["fixture:0","fixture:1"], "untrained-fixture", "hook") as output:
        handle = model.decoder.layers[0].linear2.register_forward_hook(lambda m,i,o: o*0)
        try:
            logits = model(x,lengths,torch.tensor([[2,7],[2,8]]))
            assert torch.isfinite(logits).all() and not logits.requires_grad
        finally:
            handle.remove()
        assert not model.decoder.layers[0].linear2._forward_hooks
        write_json(output, {"fixture_only":True,"passed":True})
    assert all(p.grad is None for p in model.parameters())


def test_export_loads_unchanged_repo_closed_loader(tmp_path):
    adjacency = tmp_path / "adj.npy"
    np.save(adjacency, np.eye(111,dtype=np.float32))
    vocabulary = [0,2,106,*range(1000,1118)]
    model = repo.build_model(adjacency, encoder_pe=True, vocab_map=vocabulary)
    weights = Weights(model,tmp_path / "audit.jsonl")
    name = "decoder.layers.0.multihead_attn.out_proj.weight"
    replacement = weights.load_original(name)
    replacement[0,0] += .01
    weights.save_edit(name,replacement)
    source = tmp_path / "original.pt"
    torch.save(dict(model=weights.original, closed=True, provenance=dict(fold=4,outer_test_signer=4),
                    config=dict(encoder_pe=True,vocab_map=vocabulary,ctc_weight=0.)),source)
    weights.export(source,tmp_path / "edited.pt")
    loaded,_ = repo.load_closed_decoder(SimpleNamespace(checkpoint=tmp_path / "edited.pt",fold=4,
                                     adjacency=adjacency), dict(outer_test_signer=4),torch.device("cpu"))
    for k,v in loaded.state_dict().items():
        assert tensor_hash(v) == tensor_hash(weights.load_edited(k))
