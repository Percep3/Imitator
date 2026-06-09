"""Fase 0 (subconjunto): construye el HDF5 de entrenamiento para dataset2.

Para cada clip seleccionado:
  - keypoints: extraídos con RTMPose (rtmlib, to_openpose -> 137 puntos), decodificando con cv2.
    Se elige el "signer" por mayor movimiento de muñecas. Formato (T, 137, 2), el que espera
    `remove_keypoints` del dataloader (reduce a 111).
  - embeddings objetivo: embeddings de token de entrada de un LLM (Llama-3.2-1B, hidden=2048,
    = output_size del Imitator). Sustituye a gemma-3n, que NO carga con transformers 4.51.3.
  - labels: la transcripción (col `label` de meta.csv).

Escribe en data/processed/<out> bajo el grupo `dataset2/{keypoints,embeddings,labels}/<idx>`,
con claves enteras alineadas (formato que lee KeypointDataset). Es resumible (salta claves ya
presentes) y procesa por fases (keypoints -> libera RTMPose -> embeddings) para no saturar GPU.

Uso (lanzar en tmux, ver tiempos largos):
    PYTHONPATH=. python scripts/build_dataset2_h5.py --n 600 --max-frames 250
"""
import argparse
import os
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

RAW = Path("/shared/Code/Sign-AI/data/raw/dataset2")
N_OPENPOSE = 134  # rtmlib to_openpose -> 134 puntos (layout que asume remove_keypoints del dataloader)
LH_WRIST, RH_WRIST = 93, 113  # muñecas: hand_l empieza en 93, hand_r en 113 (ver remove_keypoints)


def select_clips(n, seed):
    meta = pd.read_csv(RAW / "meta.csv")
    meta = meta[meta["label"].notna() & (meta["label"].astype(str).str.strip() != "")]
    videos = set(os.listdir(RAW / "videos"))
    rows = []
    for _, r in meta.iterrows():
        fname = f"{r['id']}.mp4"
        if fname in videos:
            rows.append((fname, str(r["label"]).lower()))
    random.Random(seed).shuffle(rows)
    return rows[:n]


# ----------------------------- Keypoints (RTMPose) -----------------------------
def extract_keypoints(video_path, model, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    per_person = {}  # person_idx -> list[(frame_idx, kpts(137,2))]
    fi = 0
    while fi < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        keypoints, _ = model(frame_rgb)          # (P, 137, 2), (P, 137)
        kp = np.asarray(keypoints, dtype=np.float32)
        if kp.ndim == 3 and kp.shape[1] == N_OPENPOSE:
            for p in range(kp.shape[0]):
                per_person.setdefault(p, []).append((fi, kp[p]))
        fi += 1
    cap.release()

    if not per_person:
        return None

    # signer = persona con mayor movimiento total de muñecas
    def movement(seq):
        tot = 0.0
        for (_, a), (_, b) in zip(seq[:-1], seq[1:]):
            tot += np.linalg.norm(a[LH_WRIST] - b[LH_WRIST])
            tot += np.linalg.norm(a[RH_WRIST] - b[RH_WRIST])
        return tot

    best = max(per_person.values(), key=movement)
    frames = np.stack([k for _, k in sorted(best, key=lambda t: t[0])])  # (T, 137, 2)
    return frames


def phase_keypoints(f, clips, max_frames):
    from rtmlib import Custom
    model = Custom(
        to_openpose=True,
        det_class="RTMDet",
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
        det_input_size=(640, 640),
        pose_class="RTMPose",
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip",
        pose_input_size=(288, 384),
        backend="onnxruntime",
        device="cuda",
    )
    g_kp = f["dataset2"].require_group("keypoints")
    g_lb = f["dataset2"].require_group("labels")
    dt = h5py.string_dtype(encoding="utf-8")

    done = skipped = 0
    for idx, (fname, label) in enumerate(clips):
        key = str(idx)
        if key in g_kp:
            continue
        kp = extract_keypoints(RAW / "videos" / fname, model, max_frames)
        if kp is None or kp.shape[0] < 2:
            skipped += 1
            print(f"[kp] {idx} SKIP ({fname})")
            continue
        g_kp.create_dataset(key, data=kp, compression="gzip", compression_opts=4)
        if key not in g_lb:
            g_lb.create_dataset(key, data=[label], dtype=dt, compression="gzip")
        done += 1
        if done % 25 == 0:
            f.flush()
            print(f"[kp] {done} hechos | último {kp.shape} ({fname})")
    f.flush()
    print(f"[kp] FASE keypoints lista: {done} hechos, {skipped} saltados")


# ----------------------------- Embeddings (LLM) -----------------------------
def phase_embeddings(f, clips, llm_model):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                             bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(llm_model)
    model = AutoModelForCausalLM.from_pretrained(llm_model, quantization_config=bnb)
    emb = model.get_input_embeddings().to("cuda")
    dim = emb.weight.shape[1]
    print(f"[emb] {llm_model} | dim embeddings = {dim}  (output_size del Imitator debe ser {dim})")

    g_emb = f["dataset2"].require_group("embeddings")
    g_kp = f["dataset2"]["keypoints"]

    done = 0
    for idx, (fname, label) in enumerate(clips):
        key = str(idx)
        if key not in g_kp:       # solo clips con keypoints válidos
            continue
        if key in g_emb:
            continue
        with torch.no_grad():
            ids = tok(label, return_tensors="pt").input_ids.to("cuda")
            vecs = emb(ids[0]).detach().cpu().float().numpy()   # (n_tokens, dim)
        g_emb.create_dataset(key, data=vecs, compression="gzip", compression_opts=4)
        done += 1
        if done % 50 == 0:
            f.flush()
            print(f"[emb] {done} hechos")
    f.flush()
    print(f"[emb] FASE embeddings lista: {done} hechos | dim={dim}")
    return dim


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="nº de clips del subconjunto")
    ap.add_argument("--max-frames", type=int, default=250, help="frames máx por video")
    ap.add_argument("--out", type=Path, default=Path("data/processed/dataset_v6_unsloth.hdf5"))
    ap.add_argument("--llm-model", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--phase", choices=["all", "keypoints", "embeddings"], default="all")
    args = ap.parse_args()

    out = args.out
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1].parent / out
    out.parent.mkdir(parents=True, exist_ok=True)

    clips = select_clips(args.n, args.seed)
    print(f"Subconjunto: {len(clips)} clips | salida: {out}")

    with h5py.File(out, "a") as f:
        f.require_group("dataset2")
        if args.phase in ("all", "keypoints"):
            phase_keypoints(f, clips, args.max_frames)
        if args.phase in ("all", "embeddings"):
            phase_embeddings(f, clips, args.llm_model)
        nk = len(f["dataset2"]["keypoints"]) if "keypoints" in f["dataset2"] else 0
        ne = len(f["dataset2"]["embeddings"]) if "embeddings" in f["dataset2"] else 0
        print(f"HDF5 listo: dataset2 con {nk} keypoints y {ne} embeddings")
