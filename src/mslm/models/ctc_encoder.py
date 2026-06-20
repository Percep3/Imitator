"""CTCEncoder — encoder de vídeo para v119 (CTC sobre secuencia).

A diferencia de Imitator (imitator.py), que cuella la secuencia a K tokens fijos
vía cross-attention, este encoder clasifica directamente sobre la salida por-frame
del Transformer: la longitud de salida escala con la entrada (cientos de frames,
o T/4 con TLP), siempre >> la longitud de las frases (16-25 palabras), así que la
restricción de CTC (input_length >= target_length) se cumple con margen amplio.

Reusa los mismos bloques que Imitator (STGCNBlock, partition_adjacency,
TransformerEncoderLayerRoPE). Streams estático/movimiento y TLP son flags
opcionales (ablations A2/A3 de v119), no reescrituras del encoder base (A1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .components import TransformerEncoderLayerRoPE
from .components.stgcn import STGCNBlock, partition_adjacency
from .components.tlp import TemporalLiftPooling


class CTCEncoder(nn.Module):
    def __init__(
        self,
        A,
        input_size: int,
        hidden_size: int = 512,
        nhead: int = 8,
        ff_dim: int = 1024,
        n_layers: int = 2,
        vocab_size: int = 100,
        encoder_dropout: float = 0.4,
        multihead_dropout: float = 0.1,
        use_motion_stream: bool = False,
        use_tlp: bool = False,
    ):
        super().__init__()
        self.use_motion_stream = use_motion_stream
        self.use_tlp = use_tlp

        A_part = partition_adjacency(A)
        stream_channels = hidden_size // 2
        self.stgcn_static = STGCNBlock(2, stream_channels, A_part, kernel_size=3)
        if use_motion_stream:
            self.stgcn_motion = STGCNBlock(2, stream_channels, A_part, kernel_size=3)
            fuse_in = 2 * 3 * stream_channels
        else:
            fuse_in = 3 * stream_channels

        self.linear_hidden = nn.Sequential(
            nn.Conv2d(fuse_in, hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size),
        )

        encoder_layer = TransformerEncoderLayerRoPE(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=encoder_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        if use_tlp:
            self.tlp1 = TemporalLiftPooling(hidden_size)
            self.tlp2 = TemporalLiftPooling(hidden_size)

        self.classifier = nn.Linear(hidden_size, vocab_size + 1)  # +1 = blank (índice 0)

    @staticmethod
    def _motion_stream(x: torch.Tensor) -> torch.Tensor:
        """Diferencias frame-a-frame (CoSign/LiftSign §3.2.1); primer frame con padding cero."""
        diff = x[:, 1:] - x[:, :-1]
        pad = torch.zeros_like(x[:, :1])
        return torch.cat([pad, diff], dim=1)

    def forward(self, x: torch.Tensor, frames_padding_mask: torch.Tensor):
        """x: [B, T, N, 2]; frames_padding_mask: [B, T] (True = padding).
        Devuelve (log_probs [B, T', V+1], seq_lengths [B], aux_losses dict)."""
        lengths = (~frames_padding_mask).sum(dim=1)

        static_in = x.permute(0, 3, 1, 2)  # [B, 2, T, N]
        feats = self.stgcn_static(static_in)
        if self.use_motion_stream:
            motion_in = self._motion_stream(x).permute(0, 3, 1, 2)
            feats_m = self.stgcn_motion(motion_in)
            feats = torch.cat([feats, feats_m], dim=1)

        feats = self.linear_hidden(feats)             # [B, hidden, T, N]
        feats = feats.mean(dim=-1)                     # [B, hidden, T]
        feats = feats.permute(0, 2, 1).contiguous()     # [B, T, hidden]

        def transformer_checkpoint(t):
            return self.transformer(t, src_key_padding_mask=frames_padding_mask)

        if self.training:
            feats = checkpoint(transformer_checkpoint, feats, use_reentrant=False)
        else:
            feats = transformer_checkpoint(feats)

        aux_losses = {}
        if self.use_tlp:
            feats, lengths, aux1 = self.tlp1(feats, lengths)
            feats, lengths, aux2 = self.tlp2(feats, lengths)
            aux_losses = {"L_u": aux1["L_u"] + aux2["L_u"], "L_p": aux1["L_p"] + aux2["L_p"]}

        logits = self.classifier(feats)                 # [B, T', V+1]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, lengths, aux_losses
