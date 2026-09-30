"""Render the confirmatory paper figure from the frozen master results."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[2]
data = json.loads((PAPER / "data/frozen-paper-results/paper_master_results.json").read_text())
folds = {row["fold"]: row for row in data["folds"]}

labels = ["Signer 7", "Signer 8"]
cif = [100 * folds[f]["cif_affine"]["strict_exact"] for f in (7, 8)]
e1 = [100 * folds[f]["e1"]["metrics"]["strict_exact"] for f in (7, 8)]
clean = [100 * folds[f]["e1"]["metrics"]["token_edit_similarity"] for f in (7, 8)]
permuted = [
    100 * folds[f]["robustness_curves"]["segment_permutation"]["metrics"]["token_edit_similarity"]
    for f in (7, 8)
]

plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.05))
x = np.arange(2)
w = 0.34
colors = ("#8B8B8B", "#2878B5")

for ax, left, right, names, title, ylabel, ymax in (
    (axes[0], cif, e1, ("CIF", "imitator_e1"), "Strict sequence accuracy", "Accuracy (%)", 18),
    (axes[1], permuted, clean, ("Permuted", "Clean"), "Causal order intervention", "Token edit similarity (%)", 70),
):
    bars1 = ax.bar(x - w / 2, left, w, label=names[0], color=colors[0])
    bars2 = ax.bar(x + w / 2, right, w, label=names[1], color=colors[1])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=4)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    legend_location = "upper right" if title == "Causal order intervention" else "upper left"
    ax.legend(frameon=False, fontsize=7, loc=legend_location)
    ax.bar_label(bars1, fmt="%.1f", padding=2, fontsize=7)
    if title == "Causal order intervention":
        ax.bar_label(bars2, fmt="%.1f", label_type="center", color="white", fontsize=7)
    else:
        ax.bar_label(bars2, fmt="%.1f", padding=2, fontsize=7)

fig.tight_layout(w_pad=2.0)
fig.savefig(PAPER / "figures/confirmation_results.pdf", bbox_inches="tight")
fig.savefig(PAPER / "figures/confirmation_results.png", dpi=220, bbox_inches="tight")
