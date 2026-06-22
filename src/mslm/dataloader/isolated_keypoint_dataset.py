"""IsolatedKeypointDataset — dataset slim para clasificación de señas aisladas
(v120, dataset1: 64 glosas x 50 ejemplos c/u).

No reusa `KeypointDataset` porque esa clase exige un grupo `embeddings` en el
h5 (`processData` enumera clips desde `f[dataset]["embeddings"].keys()`) que
dataset1 (clasificación, sin embeddings de Gemma) no tiene. Sí reusa
`remove_keypoints` y `normalize_augment_data` (incluido `temporal_drop`,
agregado en v119) para no duplicar esa lógica.
"""
import random

import h5py
import torch
from torch.utils.data import Dataset

from .data_augmentation import normalize_augment_data, remove_keypoints

AUGMENTATIONS = ["Gaussian_jitter", "Rotation_2D", "Scaling", "Temporal_drop", "Length_variance"]


def list_clips(h5_path, dataset_name="dataset1"):
    """Enumera (clip_id, label) de un grupo del h5 que solo tiene
    keypoints/labels (sin embeddings), ordenado por clip_id numérico."""
    with h5py.File(h5_path, "r") as f:
        g = f[dataset_name]
        clip_ids = sorted(g["keypoints"].keys(), key=int)
        labels = [g["labels"][cid][:][0].decode() for cid in clip_ids]
    return clip_ids, labels


class IsolatedKeypointDataset(Dataset):
    def __init__(self, h5_path, clip_ids, labels, dataset_name="dataset1", n_keypoints=111, augment=False):
        self.h5_path = h5_path
        self.dataset_name = dataset_name
        self.clip_ids = clip_ids
        self.labels = labels
        self.n_keypoints = n_keypoints
        self.augment = augment

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        with h5py.File(self.h5_path, "r") as f:
            keypoint = f[self.dataset_name]["keypoints"][clip_id][:]

        keypoint = remove_keypoints(keypoint)
        transform = random.choice(AUGMENTATIONS) if self.augment else "Original"
        keypoint = normalize_augment_data(keypoint, transform, self.n_keypoints)
        if not isinstance(keypoint, torch.Tensor):
            keypoint = torch.as_tensor(keypoint)
        return keypoint.float(), self.labels[idx]


def isolated_collate_fn(batch, label_to_idx):
    """NO pad-ea: cada clip dura distinto y un STGCN multi-capa con padding+
    máscara filtra entre samples (ver docstring de
    IsolatedSignClassifier.forward). Devuelve la lista de keypoints a su
    longitud real; el training loop llama al modelo una vez por clip (B=1)."""
    keypoints = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    target = torch.tensor([label_to_idx[lbl] for lbl in labels], dtype=torch.long)
    return keypoints, target
