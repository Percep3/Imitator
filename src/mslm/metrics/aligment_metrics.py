# We'll paste the alignment_metrics.py content (as provided) and run a quick synthetic test.
from __future__ import annotations
import os
import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for file saving
matplotlib.use("Agg")

# ---------------------------
# Helpers (normalization etc.)
# ---------------------------

def _center(z: torch.Tensor) -> torch.Tensor:
    return z - z.mean(dim=0, keepdim=True)

def _unit(z: torch.Tensor) -> torch.Tensor:
    z = _center(z)
    return F.normalize(z, p=2, dim=1)

def _as_bool_mask(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts bool mask or {0,1} or {False,True} shaped [B,L].
    Returns bool mask where True = VALID (keep).
    """
    if x.dtype == torch.bool:
        return x
    return x != 0


# ---------------------------
# Core metrics
# ---------------------------

@torch.no_grad()
def alignment_mse(X: torch.Tensor, Y: torch.Tensor, pairs: Optional[torch.Tensor]=None) -> float:
    """
    E[||x - y||^2] on positives. If pairs is None, assume index-aligned.
    X, Y: [N,d] (will be unit-normalized and centered)
    """
    Xz, Yz = _unit(X.float()), _unit(Y.float())
    if pairs is None:
        dif = Xz - Yz
    else:
        dif = Xz[pairs[:,0]] - Yz[pairs[:,1]]
    return torch.mean(torch.sum(dif**2, dim=1)).item()

@torch.no_grad()
def uniformity(Z: torch.Tensor, t: float = 2.0, max_pairs: int = 50000) -> float:
    """
    log E_{u != v} exp(-t ||u - v||^2). Lower is better (more uniform).
    Z: [N,d] (will be unit-normalized)
    """
    Z = _unit(Z.float())
    N = Z.size(0)
    num = min(max_pairs, max(1, N*(N-1)//2))
    a = torch.randint(0, N, (num,), device=Z.device)
    b = torch.randint(0, N, (num,), device=Z.device)
    mask = a != b
    a, b = a[mask], b[mask]
    D2 = torch.sum((Z[a] - Z[b])**2, dim=1)
    return torch.log(torch.mean(torch.exp(-t * D2))).item()

@torch.no_grad()
def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Centered linear CKA between X and Y.
    """
    Xc = _center(X.float())
    Yc = _center(Y.float())
    K = Xc @ Xc.t()
    L = Yc @ Yc.t()
    HSIC = (K * L).mean()
    varK = (K * K).mean().sqrt()
    varL = (L * L).mean().sqrt()
    return (HSIC / (varK * varL + 1e-8)).item()

@torch.no_grad()
def procrustes_error(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Procrustes con escala: min_{s,R} || s * X R - Y ||_F / ||Y||_F
    - L2-normaliza por fila + centra columnas (ambos espacios).
    - Usa factor de escala óptimo s = trace(S) / ||Xc||_F^2.
    """
    # Normalización por fila para quitar magnitud por muestra
    Xn = F.normalize(X.float(), dim=1)
    Yn = F.normalize(Y.float(), dim=1)

    # Centrado por característica
    Xc = Xn - Xn.mean(0, keepdim=True)
    Yc = Yn - Yn.mean(0, keepdim=True)

    # SVD de la covarianza
    C = Xc.t() @ Yc
    U, S, Vt = torch.linalg.svd(C, full_matrices=False)
    R = U @ Vt

    trS = S.sum()
    denom = (torch.linalg.norm(Xc, ord='fro') ** 2) + 1e-8
    s = (trS / denom).clamp_min(1e-8)

    num = torch.linalg.norm(s * (Xc @ R) - Yc, ord='fro')
    den = torch.linalg.norm(Yc, ord='fro') + 1e-8
    return (num / den).item()

@torch.no_grad()
def recall_at_k(
    X: torch.Tensor,
    Y: torch.Tensor,
    pairs: Optional[torch.Tensor] = None,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Cosine retrieval X->Y and Y->X with ground-truth pairs (index-aligned if None).
    Returns dict: {'R@1_x2y':..., 'R@1_y2x':..., ...}
    """
    Xn, Yn = F.normalize(X.float(), dim=1), F.normalize(Y.float(), dim=1)
    S = Xn @ Yn.t()  # [Nx, Ny]

    N = X.size(0)
    if pairs is None:
        gt_x = torch.arange(N, device=X.device)
        gt_y = torch.arange(N, device=Y.device)
    else:
        gt_x = torch.full((N,), -1, dtype=torch.long, device=X.device)
        gt_y = torch.full((Y.size(0),), -1, dtype=torch.long, device=Y.device)
        gt_x[pairs[:,0]] = pairs[:,1]
        gt_y[pairs[:,1]] = pairs[:,0]

    out = {}
    # X->Y
    ranks = torch.argsort(-S, dim=1)
    for k in ks:
        topk = ranks[:, :k]
        hits = (topk == gt_x[:, None]).any(dim=1)
        out[f'R@{k}_x2y'] = hits.float().mean().item()

    # Y->X
    ranksT = torch.argsort(-S.t(), dim=1)
    for k in ks:
        topk = ranksT[:, :k]
        hits = (topk == gt_y[:, None]).any(dim=1)
        out[f'R@{k}_y2x'] = hits.float().mean().item()

    return out

@torch.no_grad()
def attention_entropy(attn: torch.Tensor, token_mask: Optional[torch.Tensor]=None) -> float:
    """
    attn: [B, T_y, T_x], rows sum to 1 over T_x (tokens->frames).
    token_mask: [B, T_y] True=valid (optional).
    """
    eps = 1e-12
    A = attn.float().clamp_min(eps)
    H = -torch.sum(A * torch.log(A), dim=-1)  # [B, T_y]
    if token_mask is not None:
        m = _as_bool_mask(token_mask)
        H = H[m]
    return H.mean().item() if H.numel() > 0 else float('nan')

@torch.no_grad()
def attention_monotonicity(attn: torch.Tensor, token_mask: Optional[torch.Tensor]=None) -> float:
    """
    Pearson corr entre índice de query (tokens) y argmax(frame-index).
    """
    A = attn.float()
    idx_q = torch.arange(A.size(1), device=A.device)  # [T_y]
    arg = torch.argmax(A, dim=-1)                     # [B, T_y]
    if token_mask is not None:
        m = _as_bool_mask(token_mask)                 # [B, T_y]
        arg = arg[m]
        iq = idx_q.repeat(A.size(0), 1)[m]
    else:
        iq = idx_q.repeat(A.size(0), 1).reshape(-1)
        arg = arg.reshape(-1)

    if arg.numel() < 2:
        return float('nan')

    x = iq.float()
    y = arg.float()
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return torch.mean(x * y).item()


# ------------------------------------------------
# Batch-level evaluator and epoch-level aggregator
# ------------------------------------------------

class AlignmentMetrics:
    """
    Collects and logs alignment metrics for one batch (or micro-batch).
    Call accumulate(...) over batches, then summary() at the end.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.reset()
    
    def reset(self):
        self._sum = {}
        self._count = 0

    @torch.no_grad()
    def accumulate(
        self,
        pred: torch.Tensor,              # [B, d]
        target: torch.Tensor,            # [B, d]
        token_mask: Optional[torch.Tensor] = None,   # [B, T_y] True=valid
        frame_mask: Optional[torch.Tensor] = None,   # [B, T_x] True=valid or 1=valid
        attn: Optional[torch.Tensor] = None,         # [B, T_y, T_x] (opcional)
        pairs: Optional[torch.Tensor] = None,
        compute_cka: bool = True,
        compute_procrustes: bool = True,
    ):
        X = pred.detach()
        Y = target.detach()

        scalars = {}
        scalars["alignment_mse"] = alignment_mse(X, Y, pairs)
        scalars["uniformity_x"] = uniformity(X)
        scalars["uniformity_y"] = uniformity(Y)
        if compute_cka:
            scalars["cka_xy"] = linear_cka(X, Y)
        if compute_procrustes:
            scalars["procrustes_err"] = procrustes_error(X, Y)

        r = recall_at_k(X, Y, pairs, ks=(1,5,10))
        scalars.update(r)

        if attn is not None:
            scalars["attn_entropy"] = attention_entropy(attn, token_mask)
            scalars["attn_monotonicity"] = attention_monotonicity(attn, token_mask)

        self._count += 1
        for k, v in scalars.items():
            self._sum[k] = self._sum.get(k, 0.0) + (float(v) if math.isfinite(v) else 0.0)

    def summary(self) -> Dict[str, float]:
        if self._count == 0:
            return {}
        return {k: v / self._count for k, v in self._sum.items()}


class MetricsLogger:
    """
    Logger a TensorBoard y guardado de un PNG con múltiples subplots.
    """

    def __init__(self, save_dir: str = "/mnt/data/metrics_out"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.history: Dict[str, list] = {}

    def log_epoch(self, writer, epoch: int, metrics: Dict[str, float], prefix: str = "Align"):
        for k, v in metrics.items():
            tag = f"{prefix}/{k}"
            # writer may be a dummy; call if present
            try:
                writer.add_scalar(tag, v, epoch)
            except Exception:
                pass
            self.history.setdefault(k, []).append((epoch, v))

    def _plot_panel(self, ax, keys: Sequence[str], title: str):
        present = False
        for k in keys:
            xs = [e for e, _ in self.history.get(k, [])]
            ys = [v for _, v in self.history.get(k, [])]
            if xs:
                ax.plot(xs, ys, label=k)
                present = True
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        if present:
            ax.legend(loc="best")

    def save_curves(self, keys: Optional[Sequence[str]] = None, fname: str = "metrics_curves.png"):
        """
        Crea un único PNG con subplots:
          - Panel 1 (arriba): Retrieval (R@K X->Y y Y->X)
          - Panel 2 (medio): Geometría (alignment_mse, cka_xy, procrustes_err, uniformity_x/y)
          - Panel 3 (abajo): Atención (attn_entropy, attn_monotonicity)
        """
        # Define grupos
        retrieval_keys = [
            "R@1_x2y", "R@5_x2y", "R@10_x2y",
            "R@1_y2x", "R@5_y2x", "R@10_y2x",
        ]
        geom_keys = [
            "alignment_mse", "cka_xy", "procrustes_err",
            "uniformity_x", "uniformity_y",
        ]
        attn_keys = [
            "attn_entropy", "attn_monotonicity",
        ]

        # Si el usuario pasa 'keys', usa solo las que aparezcan en algún grupo
        if keys is not None:
            retrieval_keys = [k for k in retrieval_keys if k in keys]
            geom_keys      = [k for k in geom_keys      if k in keys]
            attn_keys      = [k for k in attn_keys      if k in keys]

        # Crear figura con 3 subplots
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))
        fig.suptitle("Alignment Metrics", fontsize=14)

        self._plot_panel(axes[0], retrieval_keys, "Retrieval (R@K)")
        self._plot_panel(axes[1], geom_keys, "Geometry (Alignment/CKA/Procrustes/Uniformity)")
        self._plot_panel(axes[2], attn_keys, "Attention (Entropy/Monotonicity)")

        # Ajustes y guardado
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        out = os.path.join(self.save_dir, fname)
        plt.savefig(out, dpi=160)
        plt.close(fig)
        return out