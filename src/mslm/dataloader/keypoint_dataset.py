import json
import h5py
import numpy as np
import torch
import os
from typing import Optional, List, Dict
from torch.utils.data import random_split, Dataset, Subset, ConcatDataset
from .data_augmentation import normalize_augment_data, remove_keypoints
from collections import defaultdict
import random

class TransformedSubset(Dataset):
    def __init__(
        self,
        subset: Subset,
        transform_fn: str,
        return_label: bool = False,
        return_dataset: bool = False,
        video_lengths: Optional[List[int]] = None,
        n_keypoints: int = 133,
    ) -> None:
        self.subset = subset
        self.transform = transform_fn
        self.return_label = return_label
        self.return_dataset = return_dataset
        self.video_lengths = list(video_lengths) if video_lengths is not None else []
        self.n_keypoints = n_keypoints

        if self.transform == "Length_variance":
            self.video_lengths = [int(round(0.8 * video)) for video in self.video_lengths]

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        item = self.subset[idx]

        keypoint = item
        embedding = None
        label = None
        dataset_tag = None

        if isinstance(item, tuple):
            if len(item) >= 4:
                keypoint, embedding, label, dataset_tag = item[0], item[1], item[2], item[3]
            elif len(item) == 3:
                keypoint, embedding, label = item
            elif len(item) == 2:
                keypoint, embedding = item

        keypoint = normalize_augment_data(keypoint, self.transform, self.n_keypoints)

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding)

        label_value = label if self.return_label else None

        if self.return_dataset:
            return keypoint, embedding, label_value, dataset_tag

        return keypoint, embedding, label_value

class KeypointDataset(Dataset):
    def __init__(
        self,
        h5Path,
        n_keypoints: int = 111,
        transform=None,
        return_label: bool = False,
        max_length: int = 4000,
        data_augmentation: bool = True,
        labels_vocab_path: Optional[str] = None,
        allowed_datasets: Optional[List[str]] = None,
        return_dataset: bool = False,
    ):
        self.h5Path = h5Path
        self.n_keypoints = n_keypoints
        self.transform = transform
        self.return_label = return_label
        self.max_length = max_length
        self.video_lengths = []
        self.data_augmentation = data_augmentation
        self.allowed_datasets = set(allowed_datasets) if allowed_datasets is not None else None
        self.return_dataset = return_dataset
        self.dataset_to_id: Dict[str, int] = {}
        self.id_to_dataset: List[str] = []
    
        self.data_augmentation_dict = {
            0: "Length_variance",
            1: "Gaussian_jitter",
            2: "Rotation_2D",
            4: "Scaling"
        }

        self.dataset_length = 0
        self.processData()

        self.labels_vocab_path = labels_vocab_path or (os.path.splitext(h5Path)[0] + "_labels_vocab.json")
        self.label_to_id = {}
        self.id_to_label = []
        
        if self.return_label:
            self._build_or_load_label_vocab()
    
    def _build_or_load_label_vocab(self):
        # Si ya existe, cargar
        if os.path.exists(self.labels_vocab_path):
            print("cargando vocabulario de etiquetas desde disco...")
            with open(self.labels_vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.id_to_label = data["id_to_label"]
            self.label_to_id = {s: i for i, s in enumerate(self.id_to_label)}
            return

        # Si no existe, recorrer solo los índices válidos y recolectar labels
        labels_set = set()
        with h5py.File(self.h5Path, 'r') as f:
            for (dataset, clip) in self.valid_index:
                s = f[dataset]["labels"][clip][:][0].decode()
                labels_set.add(s)

        # Vocab ordenado para estabilidad
        self.id_to_label = sorted(labels_set)
        self.label_to_id = {s: i for i, s in enumerate(self.id_to_label)}

        # Guardar a disco (recomendado)
        with open(self.labels_vocab_path, "w", encoding="utf-8") as f:
            json.dump({"id_to_label": self.id_to_label}, f, ensure_ascii=False, indent=2)
            
    def processData(self):
        with h5py.File(self.h5Path, 'r') as f:
            datasets = sorted(list(f.keys()))

            self.valid_index = []
            self.original_videos = []
            observed_datasets = set()
            allowed = self.allowed_datasets

            for dataset in datasets:
                if allowed is not None and dataset not in allowed:
                    continue

                observed_datasets.add(dataset)

                if dataset not in self.dataset_to_id:
                    self.dataset_to_id[dataset] = len(self.id_to_dataset)
                    self.id_to_dataset.append(dataset)

                clip_ids = list(f[dataset]["embeddings"].keys())

                max_len = 0
                for clip in clip_ids:
                    try:
                        shape = f[dataset]["keypoints"][clip].shape[0]

                        label_str = f[dataset]["labels"][clip][:][0].decode()
                        if "-() " in label_str:
                            continue

                        if shape > max_len:
                            max_len = shape

                        if shape < self.max_length:
                            self.valid_index.append((dataset, clip))
                            self.video_lengths.append(shape)
                    except KeyError:
                        print(f"KeyError for {dataset}/{clip}, skipping...")
                        continue
                print(f"Dataset: {dataset}, max video length: {max_len}")

            if allowed is not None:
                missing = sorted(list(allowed - observed_datasets))
                if missing:
                    print(f"Warning: requested datasets not found in file: {', '.join(missing)}")

            self.dataset_length = len(self.valid_index)

    def split_dataset(self, train_ratio):
        train_dataset, validation_dataset = random_split(self, [train_ratio, 1 - train_ratio], generator=torch.Generator().manual_seed(42))
        val_length = [self.video_lengths[i] for i in validation_dataset.indices] 
        
        if self.data_augmentation:
            train_length = [self.video_lengths[i] 
                            for i in train_dataset.indices]
            train_subset = Subset(self, train_dataset.indices)
            aug_subsets = [
                TransformedSubset(train_subset, 
                                  transform_fn=tf,
                                  return_label=self.return_label,
                                  return_dataset=self.return_dataset,
                                  video_lengths=train_length,
                                  n_keypoints=self.n_keypoints
                                  )
                for tf in self.data_augmentation_dict.values()
            ]

            trains_subset_length = [ length
                for subset in aug_subsets
                for length in subset.video_lengths
            ]
            
            train_lengths = train_length + trains_subset_length 
            train_dataset = ConcatDataset([train_subset, *aug_subsets])
            
            self.dataset_length = len(val_length) + len(train_length)
        else:
            train_lengths = [self.video_lengths[i] for i in train_dataset.indices]

        print("Videos: ", self.dataset_length)
        return train_dataset, validation_dataset, train_lengths, val_length

    def get_video_lengths(self):
        return self.dataset_length 
    
    def __len__(self):
        return len(self.valid_index)

    def __getitem__(self, idx):
        """
        Recupera una muestra individual del conjunto de datos.
        Este método recupera los puntos clave, la matriz de adyacencia, los embeddings y opcionalmente las etiquetas
        del archivo HDF5 para el índice dado. Los puntos clave se procesan eliminando
        puntos específicos y normalizando/aumentando los datos.
        Args:
            idx (int): Índice de la muestra a recuperar.
        Returns:
            Tupla que contiene:
            - keypoint (torch.Tensor): Datos de puntos clave procesados.
            - embedding (torch.Tensor): Vector de embedding para la muestra.
            - label (Optional[str]): Cadena de etiqueta si return_label es True, None en caso contrario.
        """
        
        mapped_idx = self.valid_index[idx]
        dataset_name = mapped_idx[0]

        label_id = None
        with h5py.File(self.h5Path, 'r') as f:
            keypoint = f[mapped_idx[0]]["keypoints"][mapped_idx[1]][:]
            embedding = f[mapped_idx[0]]["embeddings"][mapped_idx[1]][:]
    
            if self.return_label:
                label_str = f[mapped_idx[0]]["labels"][mapped_idx[1]][:][0].decode()
                label_id = self.label_to_id[label_str]
        
        # print("Before deletion:", keypoint.shape)
        keypoint = remove_keypoints(keypoint)
 
        # print("After deletion:", keypoint.shape)
        keypoint = normalize_augment_data(keypoint, "Original", self.n_keypoints)

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding)

        label_value = label_id if self.return_label else None

        if self.return_dataset:
            return keypoint, embedding, label_value, dataset_name

        return keypoint, embedding, label_value
    

class ContrastiveDataset(Dataset):
    """
    Envuelve un KeypointDataset para producir dos vistas por índice.
    - pos_mode='instance': positivo = misma muestra con otra augmentación
    - pos_mode='class': positivo = otra muestra con misma etiqueta (requiere return_label=True en base)
    """
    def __init__(
        self,
        base_dataset,                    # instancia de KeypointDataset
        n_keypoints: int = 111,
        pos_mode: str = "instance",      # 'instance' | 'class'
        aug_pool=None,                   # lista de nombres de augmentaciones a muestrear
        include_original_in_pool: bool = True,
        return_label: bool = False
    ):
        self.base = base_dataset
        self.n_keypoints = n_keypoints
        self.pos_mode = pos_mode
        self.return_label = return_label

        # Pool de augs a muestrear en cada vista
        default_pool = ["Length_variance", "Gaussian_jitter", "Rotation_2D", "Scaling"]
        self.aug_pool = list(default_pool if aug_pool is None else aug_pool)
        if include_original_in_pool and "Original" not in self.aug_pool:
            self.aug_pool.append("Original")

        # Si pedimos positivos por clase, construimos índice etiqueta→índices
        if self.pos_mode == "class":
            if not getattr(self.base, "return_label", False):
                raise ValueError("pos_mode='class' requiere KeypointDataset(return_label=True).")
            self.label_to_indices = defaultdict(list)
            with h5py.File(self.base.h5Path, "r") as f:
                for i, (ds, clip) in enumerate(self.base.valid_index):
                    label = f[ds]["labels"][clip][:][0].decode()
                    self.label_to_indices[label].append(i)

    def __len__(self):
        return len(self.base)

    def _load_item(self, idx):
        """
        Carga una muestra base (normalizada con 'Original' en tu pipeline).
        Devuelve (keypoints_norm, embedding, label|None).
        """
        # KeypointDataset ya aplica:
        #   keypoint = remove_keypoints(...) y normalize_augment_data(..., "Original", ...)
        item = self.base[idx]

        if isinstance(item, tuple):
            if len(item) >= 3:
                keypoint, embedding, label = item[0], item[1], item[2]
            else:
                raise ValueError("Unexpected item structure in base dataset.")
        else:
            raise TypeError("Base dataset item must be a tuple.")

        return keypoint, embedding, label

    def _random_aug(self, keypoint):
        """Aplica una augmentación elegida al azar del pool (apoyado en tu normalize_augment_data)."""
        tf = random.choice(self.aug_pool)
        # NOTA: asumimos que normalize_augment_data soporta inputs ya normalizados;
        # si no, mueve la normalización al final dentro de ese helper.
        return normalize_augment_data(keypoint, tf, self.n_keypoints), tf

    def __getitem__(self, idx):
        # Cargamos anchor
        kp_anchor, emb_anchor, label_anchor = self._load_item(idx)

        # Elegimos el índice del positivo
        if self.pos_mode == "instance":
            idx_pos = idx
        else:  # 'class'
            same_pool = self.label_to_indices[label_anchor]
            if len(same_pool) == 1:
                # No hay otro en la clase; caemos a instance para no romper
                idx_pos = idx
            else:
                # Elegimos otro índice diferente dentro de la misma clase
                while True:
                    idx_pos = random.choice(same_pool)
                    if idx_pos != idx:
                        break

        # Cargamos (o reutilizamos) la muestra para el positivo
        if idx_pos == idx:
            kp_pos_base, emb_pos, label_pos = kp_anchor, emb_anchor, label_anchor
        else:
            kp_pos_base, emb_pos, label_pos = self._load_item(idx_pos)

        # Generamos DOS vistas con augmentaciones (pueden incluir "Original")
        kp_q, tf_q = self._random_aug(kp_anchor)
        kp_k, tf_k = self._random_aug(kp_pos_base)

        # Devolvemos ambas vistas y metadatos útiles para debugging
        if self.return_label or getattr(self.base, "return_label", False):
            return (kp_q, emb_anchor, kp_k, emb_pos, label_anchor, label_pos, idx, idx_pos, tf_q, tf_k)
        else:
            return (kp_q, emb_anchor, kp_k, emb_pos, None, None, idx, idx_pos, tf_q, tf_k)