import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_attn_heatmap(A_one, valid_Lq, valid_Lk, title="Cross-attention (tokens -> frames)", savepath=None):
    """
    A_one: [Lq, Lk] (ya promediada por cabezas y recortada)
    valid_Lq, valid_Lk: longitudes válidas (int)
    """
    A_crop = A_one[:valid_Lq, :valid_Lk].detach().cpu().numpy()
    # argmax por token (fila)
    align = A_crop.argmax(axis=1)  # [Lq]

    fig = plt.figure(figsize=(7, 5))
    plt.imshow(A_crop, aspect='auto', origin='lower', interpolation='nearest')
    plt.plot(align, np.arange(valid_Lq), linewidth=1.5)  # línea de alineación
    plt.colorbar(label='Attention weight')
    plt.xlabel('Frames (tiempo)')
    plt.ylabel('Tokens (posición)')
    plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    # plt.show()  # descomenta en notebook
