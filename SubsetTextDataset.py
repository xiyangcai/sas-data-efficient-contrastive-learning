from abc import ABC
from typing import Dict, List, Optional
import math

import numpy as np
import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset

from sas.submodular_maximization import lazy_greedy
from sas.subset_dataset import BaseSubsetDataset, SubsetSelectionObjective
from tqdm import tqdm

from data_proc.NLPDataLoader import IMDbDataLoader
from data_proc.dataset import IMDbDataset



class SASSubsetTextDataset(BaseSubsetDataset):
    def __init__(
            self,
            dataset: Dataset,
            subset_fraction: float,
            num_downstream_classes: int,
            device: torch.device,
            approx_latent_class_partition: Dict[int, int],
            proxy_model: Optional[nn.Module] = None,
            augmentation_distance: Optional[Dict[int, np.array]] = None,
            num_runs=1,
            pairwise_distance_block_size: int = 1024,
            threshold: float = 0.0,
            verbose: bool = False
    ):
        """
        dataset: Dataset
            Original dataset for contrastive learning. Assumes that dataset[i] returns a list of augmented views of the original example i.

        subset_fraction: float
            Fractional size of subset.

        num_downstream_classes: int
            Number of downstream classes (can be an estimate).

        proxy_model: nn.Module
            Proxy model to calculate the augmentation distance (and kmeans clustering if the avoid clip option is chosen).

        augmentation_distance: Dict[int, np.array]
            Pass a precomputed dictionary containing augmentation distance for each latent class.

        num_augmentations: int
            Number of augmentations to consider while approximating the augmentation distance.

        pairwise_distance_block_size: int
            Block size for calculating pairwise distance. This is just to optimize GPU usage while calculating pairwise distance and will not affect the subset created in any way.

        verbose: boolean
            Verbosity of the output.
        """
        super().__init__(
            dataset=dataset,
            subset_fraction=subset_fraction,
            verbose=verbose
        )
        self.device = device
        self.num_downstream_classes = num_downstream_classes
        self.proxy_model = proxy_model
        self.partition = approx_latent_class_partition
        self.augmentation_distance = augmentation_distance
        self.num_runs = num_runs
        self.pairwise_distance_block_size = pairwise_distance_block_size

        if self.augmentation_distance is None:
            self.augmentation_distance = self.approximate_augmentation_distance()

        class_wise_idx = {}
        for latent_class in tqdm(self.partition.keys(), desc="Subset Selection:", disable=not verbose):
            F = SubsetSelectionObjective(self.augmentation_distance[latent_class].copy(), threshold=threshold)
            class_wise_idx[latent_class] = lazy_greedy(F, range(len(self.augmentation_distance[latent_class])),
                                                       len(self.augmentation_distance[latent_class]))
            class_wise_idx[latent_class] = [self.partition[latent_class][i] for i in class_wise_idx[latent_class]]

        self.subset_indices = []
        for latent_class in class_wise_idx.keys():
            l = len(class_wise_idx[latent_class])
            self.subset_indices.extend(class_wise_idx[latent_class][:int(self.subset_fraction * l)])

        self.initialization_complete()

    def approximate_augmentation_distance(self):
        self.proxy_model = self.proxy_model.to(self.device)

        # Initialize augmentation distance with all 0s
        augmentation_distance = {}
        Z = self.encode_trainset()
        for latent_class in self.partition.keys():
            Z_partition = Z[self.partition[latent_class]]
            pairwise_distance = SASSubsetTextDataset.pairwise_distance(Z_partition, Z_partition)
            augmentation_distance[latent_class] = pairwise_distance.copy()
        return augmentation_distance

    def encode_trainset(self):
        trainloader = IMDbDataLoader(self.dataset, batch_size=self.pairwise_distance_block_size)

        with torch.no_grad():
            Z = []
            for input in trainloader:
                text, text_lengths = input.text
                Z.append(self.proxy_model(text.to(self.device), text_lengths.to(self.device)))
        return torch.cat(Z, dim=0)

    def encode_augmented_trainset(self, num_positives=1):
        trainloader = IMDbDataLoader(self.dataset, batch_size=self.pairwise_distance_block_size)

        with torch.no_grad():
            Z = []

            for input in trainloader:
                text, text_lengths = input.text
                Z.append(self.proxy_model(text.to(self.device), text_lengths.to(self.device)))
            Z = torch.stack(Z)

            aug_z = []
            idxs = torch.arange(0, len(Z), num_positives)
            for i in range(num_positives):
                aug_z.append(Z[idxs + i])

            Z = torch.cat(aug_z, dim=0)
        return Z

    @staticmethod
    def pairwise_distance(Z1: torch.tensor, Z2: torch.tensor, block_size: int = 1024):
        similarity_matrices = []
        for i in range(Z1.shape[0] // block_size + 1):
            similarity_matrices_i = []
            e = Z1[i * block_size:(i + 1) * block_size]
            for j in range(Z2.shape[0] // block_size + 1):
                e_t = Z2[j * block_size:(j + 1) * block_size].t()
                similarity_matrices_i.append(
                    np.array(
                        torch.cosine_similarity(e[:, :, None], e_t[None, :, :]).detach().cpu()
                    )
                )
            similarity_matrices.append(similarity_matrices_i)
        similarity_matrix = np.block(similarity_matrices)

        return similarity_matrix