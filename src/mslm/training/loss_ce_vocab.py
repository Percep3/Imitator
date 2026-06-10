import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def imitator_ce_loss(
    pred_embs: torch.Tensor,
    token_ids: torch.Tensor,
    embed_table: torch.Tensor,
    logit_temp: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy sobre el vocabulario con cabeza de salida atada a la tabla del LLM.

    En vez de regresar el embedding objetivo punto a punto (MSE+coseno, que se minimiza
    en la media condicional y colapsa), clasifica cada posición contra TODO el vocabulario:
    logits = pred @ E^T. El denominador del softmax empuja la predicción lejos de los
    tokens incorrectos, lo que hace el colapso a la media imposible por construcción.

    Args:
        pred_embs: (batch_size, seq_len, emb_dim) embeddings predichos por el Imitator.
        token_ids: (batch_size, seq_len) IDs objetivo, padding = IGNORE_INDEX (-100).
        embed_table: (vocab_size, emb_dim) tabla de embeddings de entrada del LLM, congelada.
        logit_temp: temperatura de los logits (divide el producto punto).
    Returns:
        (loss, ce, token_acc) — loss == ce (el triple mantiene el contrato del Trainer;
        token_acc es métrica, no participa del gradiente).
    """
    L_common = min(pred_embs.size(1), token_ids.size(1))
    pred = pred_embs[:, :L_common]
    ids = token_ids[:, :L_common]

    logits = pred @ embed_table.to(pred.dtype).T / logit_temp   # (B, L, V)

    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        ids.reshape(-1).long(),
        ignore_index=IGNORE_INDEX,
    )

    with torch.no_grad():
        valid = ids != IGNORE_INDEX
        if valid.any():
            token_acc = (logits.argmax(dim=-1)[valid] == ids[valid]).float().mean()
        else:
            token_acc = torch.zeros((), device=pred.device)

    return ce, ce.detach(), token_acc
