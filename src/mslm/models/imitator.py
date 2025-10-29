import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import typing as t

from .components import TransformerEncoderLayerRoPE


class TokenPositionalEncoding(nn.Module):
    """
    Aprendible positional embeddings for the token queries.
    Returns [L, D] which is added to the base token queries.
    """
    def __init__(self, max_len: int, dim: int, dropout: float = 0.0):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(max_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.drop = nn.Dropout(dropout)

    def forward(self, length: int) -> torch.Tensor:
        pe = self.pos_embed[:length, :]  # [L, D]
        return self.drop(pe)


class Imitator(nn.Module):
    def __init__(
        self,
        input_size: int = 111 * 2,
        hidden_size: int = 512,
        output_size: int = 3072,
        nhead: int = 8,
        ff_dim: int = 1024,
        n_layers: int = 2,
        max_seq_length: int = 20,
        encoder_dropout: float = 0.4,
        cross_attention_dropout: float = 0.4,
        pe_dropout: float = 0.0,
        pool_dim: int = 256,
        pool_strategy: t.Literal["cls", "mean"] = "mean",
    ):
        super().__init__()

        self.cfg = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "nhead": nhead,
            "ff_dim": ff_dim,
            "n_layers": n_layers,
            "max_seq_length": max_seq_length,
            "encoder_dropout": encoder_dropout,
            "cross_attention_dropout": cross_attention_dropout,
            "pool_dim": pool_dim,
            "pool_strategy": pool_strategy,
        }
        print(self.cfg)

        self.max_seq_length = max_seq_length
        self.pool_strategy = pool_strategy

        # --- Input projection block ---
        # (Per-frame MLP-ish projection to hidden//2)
        self.linear_feat = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
        )

        # --- Temporal/channel mixing block ---
        # We keep temporal length T. Conv1d acts over time dimension.
        # After transpose, shape is [B, C, T], where C = hidden_size//2 initially.
        self.conv1 = nn.Conv1d(hidden_size // 2, pool_dim, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm(pool_dim)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(pool_dim, pool_dim, kernel_size=1)
        self.ln2 = nn.LayerNorm(pool_dim)
        self.act2 = nn.GELU()

        # Project channel dimension pool_dim -> hidden_size (for transformer)
        self.linear_hidden = nn.Linear(pool_dim, hidden_size)

        # --- Transformer encoder over the temporal dimension ---
        encoder_layer = TransformerEncoderLayerRoPE(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=encoder_dropout,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Project each timestep to output_size (the "token space")
        self.proj = nn.Linear(hidden_size, output_size)

        # Learnable query tokens + learnable positional encodings for them
        self.token_queries = nn.Parameter(
            torch.randn(max_seq_length, output_size)
        )  # [max_seq_length, output_size]
        self.token_posenc = TokenPositionalEncoding(
            max_len=max_seq_length, dim=output_size, dropout=pe_dropout
        )

        # Cross-attention:
        #   Q = learned token queries  (length <= max_seq_length)
        #   K,V = encoded frames       (length = T', truncated if needed)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_size,
            num_heads=nhead,
            dropout=cross_attention_dropout,
            batch_first=True,
        )

        # LayerNorm before feedforward in the cross-attn block
        self.norm_attn = nn.LayerNorm(output_size)

        # Feedforward after attention (token-wise MLP)
        self.proj_final = nn.Sequential(
            nn.Linear(output_size, output_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_size * 2, output_size),
        )

    def _pool_tokens(
        self, tokens: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        tokens: [B, L, D]
        mask:   [B, L] with True = PAD (ignore)
        returns: [B, D] pooled summary
        """
        if self.pool_strategy == "mean":
            if mask is not None:
                # weight valid positions
                valid = (~mask).unsqueeze(-1).to(tokens.dtype)  # [B, L, 1]
                num = (tokens * valid).sum(dim=1)               # [B, D]
                den = valid.sum(dim=1).clamp_min(1e-6)          # [B, 1]
                v = num / den
            else:
                v = tokens.mean(dim=1)                          # [B, D]
        else:  # 'cls'
            v = tokens[:, 0, :]                                 # [B, D]

        return F.normalize(v, dim=-1)

    # @torch.compile(dynamic=True)
    def forward(
        self, x: torch.Tensor, frames_padding_mask: torch.Tensor | None = None
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        x: Tensor of frames, shape [B, T, D, K]
           (we'll view it as [B, T, D*K] = [B, T, input_size])

        frames_padding_mask: Bool Tensor [B, T]
            True = padding (to ignore)
            False = valid frame

        Returns:
            tokens_out: [B, L, output_size]  (L <= max_seq_length)
            pooled:     [B, output_size]     pooled summary
        """

        B, T, D, K = x.shape                   # x: [B, T, D, K]
        x = x.view(B, T, D * K)                # [B, T, input_size]

        # Per-frame projection down to hidden//2
        x = self.linear_feat(x)                # [B, T, hidden//2]

        # Conv1d expects [B, C, T]
        x = x.transpose(1, 2)                  # [B, hidden//2, T]

        # conv1: (hidden//2 -> pool_dim), keeps temporal length T
        x = self.conv1(x)                      # [B, pool_dim, T]

        # Back to [B, T, pool_dim] for LayerNorm over last dim
        x = x.transpose(1, 2)                  # [B, T, pool_dim]
        x = self.ln1(x)                        # [B, T, pool_dim]
        x = self.act1(x)                       # [B, T, pool_dim]

        # conv2: (pool_dim -> pool_dim), still length T
        x = x.transpose(1, 2)                  # [B, pool_dim, T]
        x = self.conv2(x)                      # [B, pool_dim, T]
        x = x.transpose(1, 2)                  # [B, T, pool_dim]
        x = self.ln2(x)                        # [B, T, pool_dim]
        x = self.act2(x)                       # [B, T, pool_dim]

        # Project channel dim pool_dim -> hidden_size for the Transformer
        x = self.linear_hidden(x)              # [B, T, hidden_size]

        # Build/clean the padding mask for frames
        pad = None
        if frames_padding_mask is not None:
            pad = frames_padding_mask.bool().clone()  # [B, T], True = ignore
            # Guarantee at least one valid timestep per sample
            valid_counts = (~pad).sum(dim=1)          # [B]
            if (valid_counts == 0).any():
                idx = (valid_counts == 0).nonzero(as_tuple=True)[0]
                pad[idx, 0] = False
            pad = pad.contiguous()                    # [B, T]

        # Run Transformer encoder over time
        def enc(z):
            return self.transformer(z, src_key_padding_mask=pad)

        if self.training:
            x = checkpoint.checkpoint(enc, x, use_reentrant=True)
        else:
            x = enc(x)
        # x is now [B, T, hidden_size]

        # Project per-timestep hidden -> output_size
        M = self.proj(x).contiguous()          # [B, T, output_size]
        M = torch.nan_to_num(M)

        # We'll now create query tokens and do cross-attention
        T_k = M.size(1)                        # actual temporal length
        T_q = min(self.max_seq_length, T_k)    # number of query tokens to use

        # Truncate memory and mask consistently to T_q
        M = M[:, :T_q, :]                      # [B, T_q, output_size]

        pad_eff = None
        if pad is not None:
            pad_eff = pad[:, :T_q]             # [B, T_q]

        # Build query tokens Q of length T_q
        Q_base = self.token_queries[:T_q, :]   # [T_q, output_size]
        Q_pos  = self.token_posenc(T_q)        # [T_q, output_size]
        Q = (Q_base + Q_pos).unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, T_q, output_size]

        # Cross-attention:
        #   Q attends to M (same length T_q after truncation)
        attn_out, _ = self.cross_attn(
            query=Q,
            key=M,
            value=M,
            key_padding_mask=pad_eff,          # [B, T_q]
        )
        attn_out = torch.nan_to_num(attn_out)  # [B, T_q, output_size]

        # Residual + norm + FFN in a norm-first style
        y = self.norm_attn(Q + attn_out)       # [B, T_q, output_size]
        y = y + self.proj_final(y)             # [B, T_q, output_size]
        y = torch.nan_to_num(y)

        # Pool tokens into a single embedding
        pooled = self._pool_tokens(y, mask=pad_eff)  # [B, output_size]

        # y:     per-token embeddings after cross-attn
        # pooled: global embedding
        return y, pooled
