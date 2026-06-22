"""IsolatedSignClassifier — clasificador de señas aisladas (v120, dataset1: 64
glosas x 50 ejemplos c/u).

Reusa el backbone GCN ya portado para CTCEncoder (`STGCNBlock` +
`partition_adjacency`, ver components/stgcn.py): mismo stack de 3 capas que
`CTCEncoder.make_stack` (gcn_channels=[32,64,128]). No hay BiLSTM ni CTC: con
1 palabra (glosa) por clip no hay secuencia que alinear, solo una clase por
clip -- mean-pool sobre joints y tiempo + clasificador lineal.

Modelo chico a propósito: "Less is More" (Huamani-malca & Bejarano, PUCP/LSP)
encuentra que modelos chicos generalizan mejor en datasets chicos de la misma
lengua y régimen de datos (3200 clips, 64 clases).
"""
import torch
import torch.nn as nn

from .components.stgcn import STGCNBlock, partition_adjacency


class IsolatedSignClassifier(nn.Module):
    def __init__(
        self,
        A,
        input_size: int = 111,
        gcn_channels=(32, 64, 128),
        hidden_size: int = 128,
        num_classes: int = 64,
    ):
        super().__init__()
        A_part = partition_adjacency(A)

        layers = nn.ModuleList()
        c_in = 2
        for c_out in gcn_channels:
            layers.append(STGCNBlock(c_in, c_out, A_part, kernel_size=3))
            c_in = 3 * c_out
        self.stgcn_layers = layers

        # Conv -> BN -> ReLU (NO Conv -> ReLU -> BN, que usa CTCEncoder.linear_hidden):
        # ahí se promedia solo sobre joints, dejando T para el TCN/BiLSTM
        # posterior, así que el promedio global de BatchNorm2d (media cero
        # sobre B,T,N) nunca coincide con ese pooling parcial. Aquí se
        # promedia sobre (T,N) A LA VEZ -- exactamente las dims que
        # BatchNorm2d normaliza a media cero -- así que terminar en BN
        # anula matemáticamente el pooled feature para cualquier input
        # (verificado: gradiente de classifier.weight y de todo el GCN
        # backbone exactamente 0.0, el modelo solo podía aprender un sesgo
        # constante por clase). ReLU después de BN rompe esa cancelación.
        self.linear_hidden = nn.Sequential(
            nn.Conv2d(c_in, hidden_size, kernel_size=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, N, 2], SIN padding -- cada clip se procesa a su longitud
        real (mismo motivo que `_encode_batch` en train_ctc_v119.py: batchear
        con padding+máscara filtra entre samples porque el cero de padding deja
        de ser exactamente cero tras la primera capa STGCN, contaminando el
        frame límite en las capas siguientes; verificado numéricamente incluso
        en float64). El training loop llama una vez por clip (B=1) y concatena
        logits. Devuelve logits [B, num_classes]."""
        feats = x.permute(0, 3, 1, 2)  # [B, 2, T, N]
        for layer in self.stgcn_layers:
            feats = layer(feats)
        feats = self.linear_hidden(feats)         # [B, hidden, T, N]
        feats = feats.mean(dim=(2, 3))             # mean-pool sobre tiempo y joints -> [B, hidden]
        return self.classifier(feats)
