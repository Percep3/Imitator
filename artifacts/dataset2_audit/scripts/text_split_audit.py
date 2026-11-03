"""Dataset2 audit: textual integrity + partition/leakage integrity (READ-ONLY).

Opens the HDF5 / manifest / split / embedding table read-only, writes only under
artifacts/dataset2_audit/_work/. No training, no GPU, no downloads.

Run (Sign-env is gutted on this host -- only conda-meta/ remains -- so an existing
venv with h5py/torch/tokenizers is used read-only):
    /shared/Code/nakato/projects/VQ_RL/.venv/bin/python \
        artifacts/dataset2_audit/scripts/text_split_audit.py
"""
import csv
import hashlib
import importlib.util
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = Path("/shared/Code/Sign-AI/data/processed")
H5 = DATA / "dataset_v6_unsloth.hdf5"
MANIFEST = DATA / "dataset2_manifest_v125.json"
SPLIT = DATA / "dataset2_split_v125.json"
TABLE = DATA / "gemma3n_embed_table.pt"
ARTS = DATA / "v125_artifacts.json"
TOK_JSON = Path.home() / (
    ".cache/huggingface/hub/models--unsloth--gemma-3n-E2B-it-unsloth-bnb-4bit/"
    "snapshots/3d26ffdd2276698f582562bf01511aa625bc6f30/tokenizer.json"
)
OUT = REPO / "artifacts/dataset2_audit/_work"
RESAMPLE_T = 32          # frames per near-dup signature
NEAR_DUP_COS = 0.95      # motion-signature cosine threshold for "near-duplicate candidate"
LONG_WORDS, SHORT_WORDS = 40, 2


def _load(name, rel):
    """Import a pure repo helper by file path (no package side effects)."""
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


normalize_alignment_text = _load("alignment_text", "src/mslm/dataloader/alignment_text.py").normalize_alignment_text
vocab_tokenize = _load("vocab", "src/mslm/dataloader/vocab.py").tokenize
make_grouped_split = _load("grouped_split", "src/mslm/dataloader/grouped_split.py").make_grouped_split

MOJIBAKE = re.compile(r"Ã.|Â.|â€|�")
DATE_RE = re.compile(r"(\d{1,2})(\d{2})(\d{4})$")  # some names use 7 digits (e.g. 7112021)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def text_flags(label):
    f = []
    s = label.strip()
    if not s:
        f.append("empty")
    if label != s:
        f.append("lead_trail_ws")
    if "  " in label or "\t" in label or "\n" in label:
        f.append("internal_ws")
    if MOJIBAKE.search(label):
        f.append("mojibake")
    if unicodedata.normalize("NFC", label) != label:
        f.append("non_nfc")
    if any(unicodedata.category(c) == "Cc" for c in label):
        f.append("control_char")
    if any(c.isupper() for c in label):
        f.append("uppercase")
    if re.search(r"\d", label):
        f.append("digits")
    if any(ord(c) > 0x24F and unicodedata.category(c)[0] == "L" for c in label):
        f.append("non_latin_letter")
    if label.count("¿") != label.count("?") or label.count("¡") != label.count("!"):
        f.append("unbalanced_inv_punct")
    if re.search(r"[\"“”«»()\[\]]", label):
        f.append("quotes_brackets")
    if "..." in label or "…" in label:
        f.append("ellipsis")
    if s and not re.search(r"[.!?…]$", s):
        f.append("no_final_punct")
    if re.search(r"\b[a-záéíóúñü]+-[a-záéíóúñü]+\b", label):
        f.append("hyphen_compound")  # vocab.tokenize glues it, alignment norm splits it
    return f


def motion_signature(kp):
    """(T,J,2) -> L2-normalized motion signature of RESAMPLE_T frames.

    Per-clip scale normalization, then per-joint temporal mean removed so the
    signature encodes movement, not studio framing/posture.
    """
    kp = np.nan_to_num(kp.astype(np.float32))
    idx = np.linspace(0, len(kp) - 1, RESAMPLE_T).round().astype(int)
    x = kp[idx]
    x = (x - x.mean()) / (x.std() + 1e-6)
    x = x - x.mean(axis=0, keepdims=True)
    v = x.reshape(-1)
    return v / (np.linalg.norm(v) + 1e-6)


def pct(a, qs=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    a = np.asarray(a)
    return {f"p{q}": float(np.percentile(a, q)) for q in qs} | {"mean": float(a.mean()), "n": int(a.size)}


def main():
    import torch
    from tokenizers import Tokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    arts = json.loads(ARTS.read_text())
    tok_sha = sha256(TOK_JSON)
    tok_sha_expected = arts["model_files"]["tokenizer.json"]["sha256"]
    tok = Tokenizer.from_file(str(TOK_JSON))
    table = torch.load(TABLE, map_location="cpu", mmap=True)

    manifest = json.loads(MANIFEST.read_text())
    kept = {m["clip_id"] for m in manifest["manifest"] if m["kept"]}
    cause = {m["clip_id"]: m["cause"] for m in manifest["manifest"]}
    split = json.loads(SPLIT.read_text())
    split_of = {cid: s for s in ("train", "val", "test") for cid in split[s]}

    rows, sigs, sig_ids = [], [], []
    consistency = Counter()
    exact_kp_hash, quant_kp_hash, lenmean_sig = defaultdict(list), defaultdict(list), defaultdict(list)

    with h5py.File(H5, "r") as f:
        g = f["dataset2"]
        groups = {k: set(g[k].keys()) for k in g}
        all_keys = sorted(set().union(*groups.values()), key=int)
        key_sets_equal = all(v == set(all_keys) for v in groups.values())
        h5_key_order = list(g["embeddings"].keys())  # order KeypointDataset iterates

        for i, k in enumerate(all_keys):
            label = g["labels"][k][0].decode()
            kp = g["keypoints"][k][:]
            ids = g["token_ids"][k][:]
            emb = g["embeddings"][k]
            tc = int(g["token_count"][k][0])
            wc = int(g["word_count"][k][0])
            fc = int(g["frame_count"][k][0])
            vid = g["video_id"][k][0].decode()
            sg = g["source_group"][k][0].decode()
            ctx_shape = g["text_ctx"][k].shape

            norm = normalize_alignment_text(label)
            re_ids = tok.encode(norm, add_special_tokens=False).ids
            raw_ids = tok.encode(label, add_special_tokens=False).ids
            vt = vocab_tokenize(label)
            flags = text_flags(label)
            if len(vt) <= SHORT_WORDS:
                flags.append("very_short")
            if len(vt) > LONG_WORDS:
                flags.append("very_long")

            # --- consistency checks (each counter = number of FAILING clips) ---
            consistency["token_ids_ne_retokenized_norm_label"] += list(map(int, ids)) != re_ids
            consistency["token_count_ne_len_token_ids"] += tc != len(ids)
            consistency["emb_rows_ne_token_count"] += emb.shape[0] != tc
            consistency["emb_dim_ne_2048"] += emb.shape[1] != 2048
            consistency["frame_count_ne_kp_T"] += fc != kp.shape[0]
            consistency["word_count_ne_vocab_tokenize"] += wc != len(vt)
            consistency["text_ctx_shape_ne_1x2048"] += tuple(ctx_shape) != (1, 2048)
            consistency["kp_shape_not_Tx134x2"] += kp.shape[1:] != (134, 2)
            consistency["video_id_prefix_ne_source_group"] += vid.rsplit("_", 1)[0] != sg
            consistency["vocab_tokenize_ne_alignment_norm_split"] += vt != norm.split()
            if len(ids):
                ref = table[torch.as_tensor(ids, dtype=torch.long)].float().numpy()
                consistency["emb_ne_table_lookup_atol1e-3"] += not np.allclose(emb[:], ref, atol=1e-3)

            exact_kp_hash[hashlib.sha1(kp.tobytes()).hexdigest()].append(k)
            s = motion_signature(kp)
            quant_kp_hash[hashlib.sha1(np.round(s * 100).astype(np.int16).tobytes()).hexdigest()].append(k)
            lenmean_sig[(kp.shape[0], *np.round(kp.reshape(-1, 2).mean(0), 0).astype(int).tolist())].append(k)
            sigs.append(s)
            sig_ids.append(k)

            rows.append({
                "clip_key": k, "kept_v125": k in kept, "manifest_cause": cause.get(k) or "",
                "split": split_of.get(k, "excluded"), "source_group": sg, "video_id": vid,
                "frame_count": fc, "word_count": len(vt), "word_count_stored": wc,
                "token_count": tc, "token_count_raw_label": len(raw_ids),
                "chars": len(label), "text_flags": "|".join(flags),
                "label": label, "label_norm": norm,
            })
            if (i + 1) % 500 == 0:
                print(f"[audit] {i + 1}/{len(all_keys)}", flush=True)

    by_key = {r["clip_key"]: r for r in rows}

    # ---------------- text stats ----------------
    def text_block(rs):
        flags = Counter(fl for r in rs for fl in r["text_flags"].split("|") if fl)
        return {
            "n": len(rs),
            "word_count": pct([r["word_count"] for r in rs]),
            "token_count_norm_gemma": pct([r["token_count"] for r in rs]),
            "token_count_raw_label_gemma": pct([r["token_count_raw_label"] for r in rs]),
            "chars": pct([r["chars"] for r in rs]),
            "flags": dict(flags.most_common()),
        }

    kept_rows = [r for r in rows if r["kept_v125"]]

    def dup_block(rs, field):
        c = Counter(r[field] for r in rs)
        dups = {t: n for t, n in c.items() if n > 1}
        return {
            "distinct": len(c), "dup_texts": len(dups), "clips_in_dup_texts": sum(dups.values()),
            "redundant_clips": sum(n - 1 for n in dups.values()),
            "top": [[t, n] for t, n in Counter(dups).most_common(15)],
        }

    # ---------------- split verification ----------------
    sv = {}
    sets = {s: set(split[s]) for s in ("train", "val", "test")}
    sv["sizes"] = {s: len(v) for s, v in sets.items()}
    sv["ratios_field"] = split.get("ratios")
    sv["seed"] = split.get("seed")
    sv["pairwise_overlap_clip_ids"] = {
        "train&val": len(sets["train"] & sets["val"]), "train&test": len(sets["train"] & sets["test"]),
        "val&test": len(sets["val"] & sets["test"]),
    }
    union = sets["train"] | sets["val"] | sets["test"]
    sv["union_eq_manifest_kept"] = union == kept
    sv["n_union"], sv["n_kept"] = len(union), len(kept)
    sv["duplicate_ids_within_split_lists"] = {s: len(split[s]) - len(sets[s]) for s in sets}
    kept_sorted = [m["clip_id"] for m in manifest["manifest"] if m["kept"]]
    regen = make_grouped_split(kept_sorted, [by_key[c]["source_group"] for c in kept_sorted], seed=23, ratios=(0.8, 0.1, 0.1))
    sv["regenerated_split_identical"] = all(regen[s] == split[s] for s in sets)

    def leakage(field, splitmap):
        vals = defaultdict(set)
        for cid, s in splitmap.items():
            vals[by_key[cid][field]].add(s)
        spanning = {v: sorted(ss) for v, ss in vals.items() if len(ss) > 1}
        return {"distinct": len(vals), "spanning_splits": len(spanning),
                "clips_in_spanning": sum(1 for cid in splitmap if by_key[cid][field] in spanning)}

    leak = {f: leakage(f, split_of) for f in ("source_group", "video_id")}

    # group composition per split
    grp_split = defaultdict(lambda: Counter())
    for cid, s in split_of.items():
        grp_split[by_key[cid]["source_group"]][s] += 1
    split_groups = {s: sorted((g for g, c in grp_split.items() if s in c), key=lambda g: -grp_split[g][s]) for s in sets}
    group_table = {s: [[g, grp_split[g][s]] for g in split_groups[s]] for s in sets}

    def series(gname):
        m = DATE_RE.search(gname)
        return ("weekly_news" if "resumen-semanal" in gname or "programa-especial" in gname
                else "eco_episodes" if "episodio" in gname else "other"), (m.group(0) if m else None)

    series_per_split = {s: dict(Counter(series(g)[0] for g in split_groups[s])) for s in sets}
    dated = {s: sorted(f"{d[-4:]}-{d[-6:-4]}-{d[:-6].zfill(2)}" for g in split_groups[s] if (d := series(g)[1])) for s in sets}

    # text leakage across splits
    def cross_text(field):
        train_txt = Counter(by_key[c][field] for c in split["train"])
        out = {}
        for s in ("val", "test"):
            hit = [c for c in split[s] if by_key[c][field] in train_txt]
            out[s] = {"clips_with_text_in_train": len(hit), "of": len(split[s]),
                      "pct": round(100 * len(hit) / len(split[s]), 2),
                      "examples": Counter(by_key[c][field] for c in hit).most_common(10)}
        vt = {by_key[c][field] for c in split["val"]}
        out["val&test_shared_texts"] = len(vt & {by_key[c][field] for c in split["test"]})
        return out

    text_leak = {"exact_label": cross_text("label"), "normalized": cross_text("label_norm")}
    # long (>=6 words) cross-split duplicates: unlikely to be generic phrases -> likely re-used footage/story
    train_norm = {by_key[c]["label_norm"] for c in split["train"]}
    long_cross = [c for s in ("val", "test") for c in split[s]
                  if by_key[c]["word_count"] >= 6 and by_key[c]["label_norm"] in train_norm]

    # ---------------- keypoint duplicate checks ----------------
    def dup_hash(d):
        groups_ = [v for v in d.values() if len(v) > 1]
        cross = sum(1 for v in groups_ if len({split_of.get(c, "excluded") for c in v} - {"excluded"}) > 1)
        return {"dup_groups": len(groups_), "clips_in_dup_groups": sum(map(len, groups_)),
                "groups_spanning_splits": cross, "examples": groups_[:10]}

    S = np.stack(sigs).astype(np.float32)
    nn_cos, nn_idx = np.empty(len(S), np.float32), np.empty(len(S), np.int64)
    for a in range(0, len(S), 512):
        sim = S[a:a + 512] @ S.T
        sim[np.arange(sim.shape[0]), np.arange(a, a + sim.shape[0])] = -2
        nn_idx[a:a + 512] = sim.argmax(1)
        nn_cos[a:a + 512] = sim.max(1)
    pairs = set()
    for i in np.where(nn_cos >= NEAR_DUP_COS)[0]:
        pairs.add(tuple(sorted((sig_ids[i], sig_ids[nn_idx[i]]), key=int)))
    near = []
    for a, b in sorted(pairs, key=lambda p: int(p[0])):
        ra, rb = by_key[a], by_key[b]
        near.append({"a": a, "b": b, "cos": float(S[sig_ids.index(a)] @ S[sig_ids.index(b)]),
                     "split_a": ra["split"], "split_b": rb["split"],
                     "same_group": ra["source_group"] == rb["source_group"], "groups": [ra["source_group"], rb["source_group"]],
                     "same_norm_text": ra["label_norm"] == rb["label_norm"],
                     "frames": [ra["frame_count"], rb["frame_count"]],
                     "text_a": ra["label"][:80], "text_b": rb["label"][:80]})
    near_cross = [p for p in near if "excluded" not in (p["split_a"], p["split_b"]) and p["split_a"] != p["split_b"]]

    # NN cosine for same-text cross-split pairs (is cross-split same text the same footage?)
    idx_of = {k: i for i, k in enumerate(sig_ids)}
    by_norm = defaultdict(list)
    for c in split_of:
        by_norm[by_key[c]["label_norm"]].append(c)
    same_text_cross_cos = []
    for t, cs in by_norm.items():
        if len({split_of[c] for c in cs}) < 2:
            continue
        for x in cs:
            for y in cs:
                if int(x) < int(y) and split_of[x] != split_of[y]:
                    same_text_cross_cos.append((float(S[idx_of[x]] @ S[idx_of[y]]), x, y, len(t.split())))
    # same normalized text, DIFFERENT source_group (any split incl. excluded): footage re-use across groups?
    by_norm_all = defaultdict(list)
    for r in rows:
        by_norm_all[r["label_norm"]].append(r["clip_key"])
    same_text_cross_group = []
    for t, cs in by_norm_all.items():
        for x in cs:
            for y in cs:
                if int(x) < int(y) and by_key[x]["source_group"] != by_key[y]["source_group"]:
                    same_text_cross_group.append((float(S[idx_of[x]] @ S[idx_of[y]]), x, y, by_key[x]["split"], by_key[y]["split"],
                                                  by_key[x]["source_group"], by_key[y]["source_group"], t[:80]))
    rng = random.Random(0)
    kept_list = sorted(split_of, key=int)
    rand_cos = [float(S[idx_of[a]] @ S[idx_of[b]]) for a, b in (rng.sample(kept_list, 2) for _ in range(20000))]

    # ---------------- historical v119 split replication (clip-level random_split) ----------------
    hist = {}
    try:
        from torch.utils.data import random_split
        valid = []
        for k in h5_key_order:  # KeypointDataset.processData order (keypoint_dataset.py:83-104)
            r = by_key[k]
            if r["frame_count"] < 4000 and r["frame_count"] >= 16:
                n_tok = r["word_count"]
                if n_tok == 0 or r["frame_count"] // 4 < n_tok:
                    continue
                valid.append(k)
        tr, va = random_split(valid, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
        trk, vak = [valid[i] for i in tr.indices], [valid[i] for i in va.indices]
        hm = {**{c: "train" for c in trk}, **{c: "val" for c in vak}}
        trg = {by_key[c]["source_group"] for c in trk}
        trn = Counter(by_key[c]["label_norm"] for c in trk)
        hist = {
            "n_valid_replicated": len(valid), "n_valid_logged": 5544,
            "train": len(trk), "val": len(vak),
            "source_groups_spanning": leakage("source_group", hm),
            "val_clips_whose_group_in_train": sum(by_key[c]["source_group"] in trg for c in vak),
            "val_clips_norm_text_in_train": sum(by_key[c]["label_norm"] in trn for c in vak),
        }
    except Exception as e:  # pragma: no cover
        hist = {"error": repr(e)}

    stats = {
        "inputs": {"h5": str(H5), "manifest": str(MANIFEST), "split": str(SPLIT), "table": str(TABLE),
                   "manifest_sha256": sha256(MANIFEST), "split_sha256": sha256(SPLIT),
                   "split_sha256_frozen": arts["artifacts"]["split"]["sha256"],
                   "manifest_sha256_frozen": arts["artifacts"]["manifest"]["sha256"],
                   "tokenizer_json": str(TOK_JSON), "tokenizer_sha256": tok_sha,
                   "tokenizer_sha256_frozen": tok_sha_expected,
                   "tokenizer_revision": arts["model_revision"], "tokenizer_model_id": arts["tokenizer_model_id"]},
        "h5": {"n_clips": len(all_keys), "groups": {k: len(v) for k, v in groups.items()},
               "key_sets_equal_across_groups": key_sets_equal,
               "keys_contiguous_0_to_n": all_keys == [str(i) for i in range(len(all_keys))]},
        "consistency_failures": dict(consistency),
        "text_all": text_block(rows), "text_kept": text_block(kept_rows),
        "dups_all": {"exact": dup_block(rows, "label"), "normalized": dup_block(rows, "label_norm")},
        "dups_kept": {"exact": dup_block(kept_rows, "label"), "normalized": dup_block(kept_rows, "label_norm")},
        "split_verification": sv,
        "leakage_id": leak,
        "groups_per_split": {s: len(v) for s, v in split_groups.items()},
        "group_table": group_table, "series_per_split": series_per_split, "group_dates_per_split": dated,
        "text_leakage": text_leak,
        "long_cross_split_text_dups_ge6w": {"n": len(long_cross),
                                             "examples": [[c, by_key[c]["split"], by_key[c]["label"][:100]] for c in long_cross[:15]]},
        "kp_exact_hash": dup_hash(exact_kp_hash),
        "kp_quantized_motion_hash": dup_hash(quant_kp_hash),
        "kp_len_mean_signature": dup_hash(lenmean_sig),
        "near_dup_motion": {"resample_T": RESAMPLE_T, "threshold_cos": NEAR_DUP_COS,
                            "nn_cos_all": pct(nn_cos), "random_pair_cos": pct(rand_cos),
                            "n_pairs": len(near), "n_cross_split_pairs": len(near_cross),
                            "pairs": near[:50], "cross_split_pairs": near_cross[:50]},
        "same_text_cross_split_motion_cos": {
            "n_pairs": len(same_text_cross_cos),
            "cos": pct([c for c, *_ in same_text_cross_cos]) if same_text_cross_cos else None,
            "n_ge_threshold": sum(c >= NEAR_DUP_COS for c, *_ in same_text_cross_cos),
            "top": sorted(same_text_cross_cos, reverse=True)[:15]},
        "same_text_cross_group_motion_cos": {
            "n_pairs": len(same_text_cross_group),
            "cos": pct([c for c, *_ in same_text_cross_group]) if same_text_cross_group else None,
            "n_ge_0.8": sum(c >= 0.8 for c, *_ in same_text_cross_group),
            "top": sorted(same_text_cross_group, reverse=True)[:15]},
        "historical_v119_random_split_replication": hist,
    }
    (OUT / "text_split_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str))

    cols = ["clip_key", "split", "kept_v125", "manifest_cause", "source_group", "video_id", "frame_count",
            "word_count", "word_count_stored", "token_count", "token_count_raw_label", "chars", "text_flags"]
    with open(OUT / "text_split_per_clip.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: stats[k] for k in ("h5", "consistency_failures", "split_verification", "leakage_id")}, indent=1, default=str))


def _selfcheck():
    assert "mojibake" in text_flags("canciÃ³n")
    assert "unbalanced_inv_punct" in text_flags("¿qué pasa.")
    assert "hyphen_compound" in text_flags("socio-ambiental.")
    assert text_flags("hola, ¿qué tal?") == []
    a = np.random.rand(50, 134, 2).astype(np.float32)
    assert abs(float(motion_signature(a) @ motion_signature(a * 3 + 7)) - 1) < 1e-4  # scale/shift invariant


if __name__ == "__main__":
    _selfcheck()
    if "--selfcheck" not in sys.argv:
        main()
