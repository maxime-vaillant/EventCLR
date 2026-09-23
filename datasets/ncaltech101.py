from typing import Optional, Callable

import numpy as np
from tonic.datasets import NCALTECH101 as NCALTECH101Base


class NCALTECH101(NCALTECH101Base):
    """`NCALTECH101 with train/test split support.

    This class extends NCALTECH101 to support train/test splits, loading only
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
    _train_indices = None
    _test_indices = None
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
        if not NCALTECH101._split_created:
            NCALTECH101._all_data = self.data.copy()
            NCALTECH101._all_targets = self.targets.copy()
            self._create_split(split_ratio, seed)
            NCALTECH101._split_created = True

        # Select only the data for the current split
        self.train = train
        if train:
            indices = NCALTECH101._train_indices
        else:
            indices = NCALTECH101._test_indices

        self.data = [NCALTECH101._all_data[i] for i in indices]
        self.targets = [NCALTECH101._all_targets[i] for i in indices]

    def _create_split(self, split_ratio: float, seed: int):
        """Create train/test split indices, stratified by class."""
        train_indices = []
        test_indices = []

        np.random.seed(seed)  # For reproducibility

        # Get unique classes and create a mapping if targets are strings
        unique_classes = sorted(set(NCALTECH101._all_targets))

        for class_label in unique_classes:
            class_samples = [i for i, target in enumerate(NCALTECH101._all_targets) if target == class_label]

            n_train = int(len(class_samples) * split_ratio)
            np.random.shuffle(class_samples)

            train_indices.extend(class_samples[:n_train])
            test_indices.extend(class_samples[n_train:])

        # Convert all targets to indices
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        NCALTECH101._all_targets = [class_to_idx[target] for target in NCALTECH101._all_targets]

        NCALTECH101._train_indices = sorted(train_indices)
        NCALTECH101._test_indices = sorted(test_indices)

