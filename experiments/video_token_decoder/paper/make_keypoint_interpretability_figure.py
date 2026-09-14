"""Create a compact paper figure from the fold-7 keypoint saliency artifact."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
artifact = ROOT / "interpretability/e1_fold7_paper/e1/saliency.npz"
if not artifact.exists():
    artifact = HERE / "keypoint_saliency.npz"
p = np.load(artifact)

frames = int(p["frame_lengths"][0])
tokens = int(p["token_lengths"][0])
raw = p["reference_normalized"][: tokens * frames].reshape(tokens, frames, 111)
bounds = p["region_bounds"]
saliency = np.stack([raw[:, :, start:end].sum(axis=(1, 2)) for start, end in bounds], axis=1)
saliency = 100 * saliency / saliency.sum(axis=1, keepdims=True)
ablation = p["reference_ablation"][:tokens]

token_labels = ["a", "tierra", "perf", "ume"]
region_labels = ["Pose", "Face", "Left hand", "Right hand"]
plt.rcParams.update({"font.family": "serif", "font.size": 8})
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.15))

panels = [
    (saliency, "Input × Gradient attribution", "Share of attribution (%)", "magma", ".1f"),
    (ablation, "Region ablation", "Target log-probability drop", "OrRd", ".2f"),
]
for ax, (values, title, cbar_label, cmap, fmt) in zip(axes, panels):
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_title(title, fontweight="bold", pad=5)
    ax.set_xticks(range(4), region_labels, rotation=25, ha="right")
    ax.set_yticks(range(4), token_labels)
    ax.set_ylabel("Gemma target subtoken")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            rgba = image.cmap(image.norm(values[row, col]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "black" if luminance > 0.58 else "white"
            ax.text(col, row, format(values[row, col], fmt), ha="center", va="center",
                    fontsize=6.5, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

fig.tight_layout(w_pad=1.7)
fig.savefig(HERE / "keypoint_interpretability.pdf", bbox_inches="tight")
fig.savefig(HERE / "keypoint_interpretability.png", dpi=220, bbox_inches="tight")
