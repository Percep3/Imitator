import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import MultiheadAttentionRoPE, TransformerEncoderLayerRoPE

def _align_kpm(kpm: torch.Tensor | None, target_len: int) -> torch.Tensor | None:
    """
    Alinea/crea una mascara de padding para frames a longitud target_len.
    Convencion: True = PAD (ignorar), False = valido (usar).
    - Garantiza al menos 1 posicion valida por batch.
    """
    if kpm is None:
        return None

    kpm = kpm.bool()  # [B, T0]
    valid_counts = (~kpm).sum(dim=1)
    if (valid_counts == 0).any():
        idx = (valid_counts == 0).nonzero(as_tuple=True)[0]
        kpm[idx, 0] = False  # destapar la primera posicion

    B, L = kpm.shape
    if L < target_len:
        pad = torch.ones(B, target_len - L, dtype=torch.bool, device=kpm.device)
        kpm = torch.cat([kpm, pad], dim=1)
    elif L > target_len:
        kpm = kpm[:, :target_len]

    return kpm.contiguous()


def _adaptive_pool_mask(pad: torch.Tensor, t_cap: int) -> torch.Tensor:
    """
    Reduce una mascara de padding [B, T] a [B, t_cap] usando pooling contiguo.
    Un bin resultante es valido si algun frame del bin fue valido.
    - True = PAD (ignorar), False = valido
    - Garantiza al menos un bin valido por batch.
    """
    # pad: True=pad -> valido = (~pad)
    valid = (~pad).float().unsqueeze(1)                        # [B, 1, T]
    pooled = F.adaptive_max_pool1d(valid, t_cap)               # [B, 1, t_cap], 1 si hay algun valido
    valid_down = pooled > 0                                    # bool
    pad_down = ~valid_down.squeeze(1)                          # True si el bin quedo vacio (todo pad)

    # Garantizar al menos 1 valido por batch
    all_pad = pad_down.all(dim=1)
    if all_pad.any():
        idx = all_pad.nonzero(as_tuple=True)[0]
        pad_down[idx, 0] = False                               # destapar primer bin

    return pad_down.contiguous()


class TokenPositionalEncoding(nn.Module):
    """
    Embeddings posicionales aprendibles para las token queries:
    devuelve un tensor [L, D] que se suma a las queries base.
    """
    def __init__(self, max_len: int, dim: int, dropout: float = 0.0):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(max_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # init estable
        self.drop = nn.Dropout(dropout)

    def forward(self, length: int) -> torch.Tensor:
        pe = self.pos_embed[:length, :]  # [L, D]
        return self.drop(pe)


class Imitator(nn.Module):
    """
    Mapea secuencias de keypoints de video (lengua de señas) a
    una secuencia fija de embeddings en el espacio del LLM.
    """

    def __init__(
        self,
        input_size: int = 133 * 2,     # D keypoints * K coords (x,y)
        hidden_size: int = 512,        # H del Transformer
        output_size: int = 3072,
        nhead: int = 8,
        ff_dim: int = 1024,
        n_layers: int = 2,
        max_seq_length: int = 128,     # L tokens de salida (fijo max)
        encoder_dropout: float = 0.1,
        cross_attention_dropout: float = 0.1,
        pe_dropout: float = 0.0,       
        conv_channels: int = 256,
        time_pool_len: int | None = None  # (si None, se calcula ~2×L)
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
            "pe_dropout": pe_dropout,
            "conv_channels": conv_channels,
        }
        self.max_seq_length = max_seq_length
        self.time_pool_len = time_pool_len or max(2 * max_seq_length, max_seq_length)

        # MLP por frame (no cambia T)   [B, T, D*K] -> [B, T, H/2]
        self.linear_feat = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
        )

        # modelado temporal + mezcla de canales (no cambia T)
        # se trabaja con formato [B, C, T] para Conv1d
        # [B, T, H/2] -> (T<->C) -> Conv1d(k=3,s=1) -> (T<->C) -> 1x1 -> (T<->C)
        # output: [B, T, C]
        self.conv1 = nn.Conv1d(hidden_size // 2, conv_channels, kernel_size=3, stride=1, padding=1)
        self.ln1   = nn.LayerNorm(conv_channels)
        self.act1  = nn.GELU()

        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=1, stride=1)
        self.ln2   = nn.LayerNorm(conv_channels)
        self.act2  = nn.GELU()

        # Proyeccion a H para el Transformer
        # [B, T, C] -> [B, T, H]
        self.linear_hidden = nn.Linear(conv_channels, hidden_size)

        # Encoder Transformer con RoPE
        # [B, T', H] -> [B, T', H]
        encoder_layer = TransformerEncoderLayerRoPE(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=encoder_dropout,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Proj al espacio del LLM por frame (K/V de cross-attn)
        # [B, T', H] -> [B, T', D_out]
        self.proj = nn.Linear(hidden_size, output_size)

        # Token Queries + Positional Encoding aprendible
        # Q_base: [L, D_out], PE_tokens: [L, D_out]
        self.token_queries = nn.Parameter(torch.randn(max_seq_length, output_size) * 0.02)
        self.token_posenc  = TokenPositionalEncoding(max_len=max_seq_length, dim=output_size, dropout=pe_dropout)

        # Cross-Attention with RoPE
        # Q: [B, L, D_out]
        # K/V: [B, T', D_out]
        self.cross_attn = MultiheadAttentionRoPE(
            embed_dim=output_size,
            num_heads=nhead,
            dropout=cross_attention_dropout,
            batch_first=True,
            use_rotary=True,
        )
        self.norm_attn = nn.LayerNorm(output_size)

        # MLP pos-attn
        self.head_mlp = nn.Sequential(
            nn.Linear(output_size, output_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_size * 2, output_size),
        )
        self.norm_out = nn.LayerNorm(output_size)

    def forward(
        self,
        x: torch.Tensor,                                 # [B, T, D, K]
        frames_padding_mask: torch.Tensor | None = None, # [B, T], True=pad
        return_attn: bool = False
    ) -> t.Tuple[torch.Tensor, torch.Tensor | None]:
        B, T, D, K = x.shape

        
        x = x.view(B, T, D * K)                     # [B, T, D*K]
        x = self.linear_feat(x)                    # [B, T, D*K] -> [B, T, H/2]

        x = x.transpose(1, 2)          # [B, H/2, T]
        x = self.conv1(x)              # [B, C, T]
        x = x.transpose(1, 2)          # [B, T, C]
        x = self.ln1(x)
        x = self.act1(x)

        x = x.transpose(1, 2)          # [B, C, T]
        x = self.conv2(x)              # [B, C, T]
        x = x.transpose(1, 2)          # [B, T, C]
        x = self.ln2(x)
        x = self.act2(x)

        # Pooling adaptativo temporal (contiguo)
        T_cap = self.time_pool_len
        x = x.transpose(1, 2)                               # [B, C, T]
        x = F.adaptive_avg_pool1d(x, T_cap)                 # [B, C, T']
        x = x.transpose(1, 2)                               # [B, T', C]
        T_prime = x.size(1)

        # Preparar mascara de padding para T'
        pad = _align_kpm(frames_padding_mask, T)            # [B, T] o None
        if pad is not None:
            pad = _adaptive_pool_mask(pad, T_prime)         # [B, T']

        # Proyeccion a H para Transformer
        x = self.linear_hidden(x)                           # [B, T', C] -> [B, T', H]

        # Transformer Encoder con mask de padding
        x = self.transformer(x, src_key_padding_mask=pad)   # [B, T', H] -> [B, T', H]

        M = self.proj(x)                                    # [B, T', H] -> [B, T', D_out]
        T_k = M.size(1)

        # Queries: base + pos-enc aprendible
        # L = min(max_seq_length, T') (opcion segura para PoC)
        # Q_base: [L, D_out];
        # PE_tokens: [L, D_out]
        # Expand a batch: [B, L, D_out]
        T_q = min(self.max_seq_length, T_k)
        Q_base = self.token_queries[:T_q, :]               # [L, D_out]
        Q_pos  = self.token_posenc(T_q)                    # [L, D_out]
        Q = (Q_base + Q_pos).unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, L, D_out]

        attn_out, attn_w = self.cross_attn(
            query=Q,
            key=M,
            value=M,
            key_padding_mask=pad,               # [B, T'] o None
            average_attn_weights=False
        )                                       # [B, L, D_out]
        x = self.norm_attn(Q + attn_out)        # residual + norm

        # MLP pos-attn + norm
        x = x + self.head_mlp(x)
        x = self.norm_out(x)

        x = F.normalize(x, p=2, dim=-1)         # [B, L, D_out]

        if return_attn:
            return x, attn_w
        
        return x, None
