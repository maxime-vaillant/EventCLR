from typing import Optional, Callable, List

import numpy as np
from tonic.datasets import CIFAR10DVS as CIFAR10DVSBase


class CIFAR10DVS(CIFAR10DVSBase):
    """`CIFAR10-DVS with train/test split support.

    This class extends CIFAR10-DVS to support train/test splits, loading only
    the data for the current split.

    Parameters:
        save_to (string): Location to save files to on disk.
        train (bool): If True, creates dataset from training set, otherwise from test set.
        split_ratio (float): Ratio of data to use for training (default: 0.9).
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and labels.
    """

    _all_data = None
    _all_targets = None
    _train_indices: List[int] = None
    _test_indices: List[int] = None
    _split_created = False

    def __init__(
            self,
            save_to: str,
            train: bool = True,
            split_ratio: float = 0.9,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            seed: int = 42,
    ):
        # Initialize parent WITHOUT preloading to just get file paths
        super().__init__(
            save_to=save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

        # Store all data paths at class level (first time only)
        if not CIFAR10DVS._split_created:
            CIFAR10DVS._all_data = self.data.copy()
            CIFAR10DVS._all_targets = self.targets.copy()
            self._create_split(split_ratio, seed)
            CIFAR10DVS._split_created = True

        # Select only the data for the current split
        self.train = train
        if train:
            indices = CIFAR10DVS._train_indices
        else:
            indices = CIFAR10DVS._test_indices

        self.data = [CIFAR10DVS._all_data[i] for i in indices]
        self.targets = [CIFAR10DVS._all_targets[i] for i in indices]

    def _create_split(self, split_ratio: float, seed: int):
        """Create train/test split indices, stratified by class."""
        train_indices = []
        test_indices = []

        np.random.seed(seed)  # For reproducibility

        for class_idx in self.classes.values():
            class_samples = [i for i, target in enumerate(CIFAR10DVS._all_targets) if target == class_idx]

            n_train = int(len(class_samples) * split_ratio)
            np.random.shuffle(class_samples)

            train_indices.extend(class_samples[:n_train])
            test_indices.extend(class_samples[n_train:])

        CIFAR10DVS._train_indices = sorted(train_indices)
        CIFAR10DVS._test_indices = sorted(test_indices)
