import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    batch: List of tuples (keypoints, embeddings)
      - keypoints: [T_i, K, D]
      - embeddings: [N_i, E]
    returns:
      keypoints_padded  → [B, T_max, K, D]
      frames_mask       → [B, T_max]     (True = padding)
      embeddings_padded → [B, N_max, E]
      embeddings_mask   → [B, N_max]     (True = padding)
    """
    keypoints_list  = [item[0] for item in batch]
    embeddings_list = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    has_dataset_tags = any(isinstance(item, tuple) and len(item) > 3 for item in batch)
    dataset_tags = None
    if has_dataset_tags:
        dataset_tags = [item[3] if isinstance(item, tuple) and len(item) > 3 else None for item in batch]

    keypoints_device   = keypoints_list[0].device
    embeddings_device  = embeddings_list[0].device

    # Normaliza embeddings a [N, E]
    for i in range(len(embeddings_list)):
        emb = embeddings_list[i]
        if emb.dim() == 3 and emb.size(0) == 1:
            embeddings_list[i] = emb.squeeze(0)
        elif emb.dim() != 2:
            raise ValueError(f"Embedding at index {i} has invalid shape {emb.shape}. Expected [N, E] or [1, N, E].")

    frame_lengths = torch.tensor([kp.size(0) for kp in keypoints_list],  dtype=torch.long, device=keypoints_device)
    token_lengths = torch.tensor([emb.size(0) for emb in embeddings_list], dtype=torch.long, device=embeddings_device)

    keypoints_padded  = pad_sequence(keypoints_list,  batch_first=True, padding_value=0.0)  # [B, T_max, K, D]
    embeddings_padded = pad_sequence(embeddings_list, batch_first=True, padding_value=0.0)  # [B, N_max, E]

    B, T_max, K, D = keypoints_padded.shape
    _, N_max, E    = embeddings_padded.shape

    arange_frames = torch.arange(T_max, device=keypoints_padded.device).unsqueeze(0).expand(B, -1)  # [B, T_max]
    arange_tokens = torch.arange(N_max, device=embeddings_padded.device).unsqueeze(0).expand(B, -1)  # [B, N_max]

    frames_mask     = arange_frames >= frame_lengths.unsqueeze(1)  # True = padding
    embeddings_mask = arange_tokens >= token_lengths.unsqueeze(1)

    return (
        keypoints_padded.to(torch.float32),
        frames_mask.to(torch.bool),
        embeddings_padded.to(torch.float32),
    embeddings_mask.to(torch.bool),
    labels,
    dataset_tags,
    )
