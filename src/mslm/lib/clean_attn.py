import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

def _fit_frames_mask_to_Lk(frames_pad_mask: torch.Tensor, Lk: int) -> torch.Tensor:
    """
    Acepta [T_any] o [B, T_any]. Devuelve [Lk] o [B, Lk] manteniendo dimensionalidad.
    Convención: True=PAD (ignorar), False=válido.
    """
    squeeze_back = False
    if frames_pad_mask.dim() == 1:
        frames_pad_mask = frames_pad_mask.unsqueeze(0)  # [1, T_any]
        squeeze_back = True
    elif frames_pad_mask.dim() != 2:
        raise ValueError(f"frames_pad_mask debe ser 1D o 2D, recibido {frames_pad_mask.shape}")

    B, T_any = frames_pad_mask.shape
    if T_any == Lk:
        pad_new = frames_pad_mask  # [B, Lk]
    else:
        valid = (~frames_pad_mask).float().unsqueeze(1)    # [B, 1, T_any]
        if T_any > Lk:
            pooled = F.adaptive_max_pool1d(valid, Lk)      # [B,1,Lk]
            valid_new = pooled > 0
        else:
            up = F.interpolate(valid, size=Lk, mode="nearest")  # [B,1,Lk]
            valid_new = up > 0.5
        pad_new = (~valid_new.squeeze(1)).contiguous()     # [B, Lk]  True=PAD

    # Garantiza al menos una posición válida por batch
    all_pad = pad_new.all(dim=1)
    if all_pad.any():
        idx = all_pad.nonzero(as_tuple=True)[0]
        pad_new[idx, 0] = False

    if squeeze_back:
        return pad_new.squeeze(0)                          # [Lk]
    return pad_new                                         # [B, Lk]

def _fit_tokens_mask_to_Lq(tokens_pad_mask: torch.Tensor, Lq: int) -> torch.Tensor:
    """
    Acepta [N_max] o [B, N_max]. Devuelve [Lq] o [B, Lq] con True=PAD para i >= longitud_real.
    """
    squeeze_back = False
    if tokens_pad_mask.dim() == 1:
        tokens_pad_mask = tokens_pad_mask.unsqueeze(0)  # [1, N_max]
        squeeze_back = True
    elif tokens_pad_mask.dim() != 2:
        raise ValueError(f"tokens_pad_mask debe ser 1D o 2D, recibido {tokens_pad_mask.shape}")

    B, N_max = tokens_pad_mask.shape
    token_lengths = (~tokens_pad_mask).sum(dim=1)                        # [B]
    ar = torch.arange(Lq, device=tokens_pad_mask.device)[None, :]        # [1, Lq]
    q_pad = ar >= token_lengths[:, None]                                  # [B, Lq]
    return q_pad.squeeze(0) if squeeze_back else q_pad

def make_queries_pad_mask_from_tokens_mask(tokens_pad_mask: torch.Tensor, Lq: int) -> torch.Tensor:
    B, N_max = tokens_pad_mask.shape
    token_lengths = (~tokens_pad_mask).sum(dim=1)  # [B]
    ar = torch.arange(Lq, device=tokens_pad_mask.device)[None, :]  # [1, Lq]
    return ar >= token_lengths[:, None]  # [B, Lq]

def clean_attn(
    attn_w: torch.Tensor,                 # [B, H, Lq, Lk]
    frames_pad_mask: torch.Tensor,        # [B, ?] True=PAD frame
    tokens_pad_mask: torch.Tensor | None  # [B, N_max] True=PAD token (opcional)
) -> torch.Tensor:
    """
    Limpia y renormaliza pesos de atención por cabeza.
    - Ajusta automáticamente la máscara de frames a Lk si hiciera falta.
    - Opcionalmente anula filas de queries (tokens PAD) derivando longitud real.
    Devuelve: [B, H, Lq, Lk]
    """
    A = attn_w.detach().clone()
    B, H, Lq, Lk = A.shape

    # 0) Ajusta la máscara de frames a Lk
    frames_mask_kv = _fit_frames_mask_to_Lk(frames_pad_mask, Lk=Lk)  # [B, Lk]

    # 1) Anula columnas PAD (frames)
    A = A.masked_fill(frames_mask_kv[:, None, None, :], 0.0)

    # 2) Anula filas PAD (queries) si se provee tokens_pad_mask
    if tokens_pad_mask is not None:
        queries_pad_mask = make_queries_pad_mask_from_tokens_mask(tokens_pad_mask, Lq=Lq)  # [B, Lq]
        A = A.masked_fill(queries_pad_mask[:, None, :, None], 0.0)

    # 3) Renormaliza por fila (sobre Lk)
    denom = A.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    A = A / denom

    # 4) Filas degeneradas → relleno uniforme en frames válidos
    bad = (denom <= 1e-6)  # [B, H, Lq, 1] (broadcast)
    if bad.any():
        filler = (~frames_mask_kv).float()                            # [B, Lk]
        filler = filler / filler.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        filler = filler[:, None, None, :]                             # [B,1,1,Lk]
        A = torch.where(bad, filler, A)

    return A  # [B, H, Lq, Lk]

@torch.no_grad()
def clean_and_mean_attn(attn_w, frames_pad_mask, tokens_pad_mask=None):
    """
    attn_w: [B, H, Lq, Lk] (probabilidades ya softmax si average_attn_weights=False)
    frames_pad_mask: [T_any] o [B, T_any]  (True=PAD)
    tokens_pad_mask: [N_max] o [B, N_max]  (True=PAD)
    return: [B, Lq, Lk]
    """
    A = attn_w.detach().clone()                      # [B,H,Lq,Lk]
    B, H, Lq, Lk = A.shape
    device = A.device

    # --- Frames mask → ajustar longitud y garantizar [B, Lk] ---
    frames_mask_kv = _fit_frames_mask_to_Lk(frames_pad_mask.to(device), Lk=Lk)  # [Lk] o [B,Lk]
    if frames_mask_kv.dim() == 1:
        frames_mask_kv = frames_mask_kv.unsqueeze(0)            # [1, Lk]
    if frames_mask_kv.size(0) == 1 and B > 1:
        frames_mask_kv = frames_mask_kv.expand(B, -1)           # [B, Lk]
    elif frames_mask_kv.size(0) != B:
        # Caso raro: longitudes batch no coinciden
        frames_mask_kv = frames_mask_kv[:1].expand(B, -1)       # fuerza [B, Lk]

    # Anular columnas PAD (frames)
    A = A.masked_fill(frames_mask_kv[:, None, None, :], 0.0)

    # --- Tokens mask → derivar queries_pad_mask y garantizar [B, Lq] ---
    if tokens_pad_mask is not None:
        q_pad = _fit_tokens_mask_to_Lq(tokens_pad_mask.to(device), Lq=Lq)  # [Lq] o [B,Lq]
        if q_pad.dim() == 1:
            q_pad = q_pad.unsqueeze(0)                                     # [1, Lq]
        if q_pad.size(0) == 1 and B > 1:
            q_pad = q_pad.expand(B, -1)                                    # [B, Lq]
        elif q_pad.size(0) != B:
            q_pad = q_pad[:1].expand(B, -1)                                # [B, Lq]

        # Anular filas PAD (queries)
        A = A.masked_fill(q_pad[:, None, :, None], 0.0)

    # Renormalizar por fila (sobre Lk)
    denom = A.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    A = A / denom

    # Promedio por cabezas
    A_mean = A.mean(dim=1)  # [B,Lq,Lk]
    return A_mean

def metrics_from_attn(A_2d):
    """
    A_2d: [Lq, Lk] filas suman 1 (numpy o torch)
    Devuelve: rho_spearman, entropy_mean, coverage
    """
    if isinstance(A_2d, torch.Tensor):
        A = A_2d.detach().cpu().numpy()
    else:
        A = A_2d
    Lq, Lk = A.shape

    # a) argmax por token (fila)
    argmax_frames = A.argmax(axis=1)  # [Lq]

    # b) Spearman rho ~ correlación de rangos entre i y argmax_frames
    i = np.arange(Lq)
    # rangos de argmax_frames (sin empates, típico con argmax)
    ranks = np.empty_like(argmax_frames)
    ranks[argmax_frames.argsort()] = np.arange(Lq)
    rho = np.corrcoef(i, ranks)[0, 1]  # equivalente a Spearman sin empates

    # c) entropía media por token (normalizada a [0,1])
    P = np.clip(A, 1e-12, 1.0)
    H = -(P * np.log(P)).sum(axis=1) / np.log(Lk)
    H_mean = float(H.mean())

    # d) cobertura temporal: % de frames que son "ganadores" de algún token
    covered = np.unique(argmax_frames).size
    cov = covered / float(Lk)

    return float(rho), H_mean, float(cov)

def plot_attn_with_metrics(A_mean_b, frames_pad_mask_b, tokens_pad_mask_b=None,
                           epoch=None, title_prefix="Cross-attention (tokens->frames)",
                           savepath=None):
    """
    A_mean_b: [Lq, Lk] para un batch index ya promediado por cabezas
    frames_pad_mask_b: [T_any]
    tokens_pad_mask_b: [N_max] (opcional)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    Lq, Lk = (A_mean_b.shape if isinstance(A_mean_b, torch.Tensor) else A_mean_b.shape)

    # Ajusta máscara de frames al Lk de la matriz
    if isinstance(A_mean_b, torch.Tensor):
        frames_mask_kv = _fit_frames_mask_to_Lk(frames_pad_mask_b.to(A_mean_b.device), Lk=Lk)
        A_np = A_mean_b.detach().cpu().numpy()
        frames_mask_kv_np = frames_mask_kv.cpu().numpy()
    else:
        # versión numpy (simple: si difiere, recorta o padd con True al final)
        fm = frames_pad_mask_b
        if fm.shape[0] != Lk:
            if fm.shape[0] > Lk:
                fm = fm[:Lk]
            else:
                fm = np.concatenate([fm, np.ones(Lk - fm.shape[0], dtype=bool)], axis=0)
        frames_mask_kv_np = fm
        A_np = A_mean_b

    # longitudes efectivas
    Lk_eff = int((~frames_mask_kv_np).sum())
    if tokens_pad_mask_b is not None:
        Lq_eff = int((tokens_pad_mask_b).sum()) # modified
    else:
        Lq_eff = Lq

    Lq_eff = min(Lq_eff, Lq)
    Lk_eff = min(Lk_eff, Lk)

    A_crop = A_np[:Lq_eff, :Lk_eff]

    # métricas
    rho, H_mean, cov = metrics_from_attn(A_crop)

    # argmax para la línea
    align = A_crop.argmax(axis=1)

    plt.figure(figsize=(7,5))
    plt.imshow(A_crop, aspect='auto', origin='lower', interpolation='nearest')
    plt.plot(align, np.arange(Lq_eff), linewidth=1.5)

    t_epoch = f" Epoch {epoch}" if epoch is not None else ""
    plt.title(f"{title_prefix}{t_epoch}\nρ={rho:.2f} · H={H_mean:.2f} · cov={cov*100:.1f}%")
    plt.xlabel("Frames (tiempo)")
    plt.ylabel("Tokens (posición)")
    cbar = plt.colorbar()
    cbar.set_label("Attention weight")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close()