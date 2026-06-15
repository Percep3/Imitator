import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def imitator_ar_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k_prefix: int,
) -> tuple:
    """Autoregressive CE loss for the v116 soft-prefix pipeline.

    Args:
        logits:   [B, K+L-1, V] — logits from Gemma forward (prefix + text positions)
        labels:   [B, K+L-1]   — -100 for prefix positions and padding; token IDs elsewhere
        k_prefix: int           — number of prefix tokens (for documentation only;
                                  the labels tensor already masks them with -100)

    Returns:
        (ce, ce_detached, top1_acc, top5_acc) — same 4-tuple contract as imitator_ce_loss
    """
    V = logits.size(-1)

    ce = F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1).long(),
        ignore_index=IGNORE_INDEX,
    )

    with torch.no_grad():
        valid = labels != IGNORE_INDEX
        if valid.any():
            flat_logits = logits[valid]          # (N_valid, V)
            flat_ids    = labels[valid].long()   # (N_valid,)
            top1 = (flat_logits.argmax(dim=-1) == flat_ids).float().mean()
            top5 = (flat_logits.topk(5, dim=-1).indices == flat_ids.unsqueeze(1)).any(1).float().mean()
        else:
            top1 = top5 = torch.zeros((), device=logits.device)

    return ce, ce.detach(), top1, top5
