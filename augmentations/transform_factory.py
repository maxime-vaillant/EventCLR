from enum import Enum
from typing import Tuple

import torch
import torchvision
from tonic import transforms

from augmentations.event_transforms import RandomCropTime, RandomApply, Rolling
from augmentations.policies import AugmentEventCLR, NeuromorphicDataAugmentation, EventDrop
from augmentations.provider import DataTransform
from augmentations.representations import ToFrame, ToVoxelGrid, ToTensor
from utils.config import ExperimentConfig


class AugmentationSetup(str, Enum):
    """Available augmentation setups for pretraining / evaluation."""
    EVENTCLR = "eventclr"
    NDA = "nda"
    EVENTDROP = "eventdrop"


class TransformFactory:
    """Factory for creating data transformations."""

    @staticmethod
    def create_representation(
            sensor_size: Tuple[int, int],
            n_time_bins: int,
            resize_size: Tuple[int, int],
            representation: str = 'frame',
            normalize: bool = True
    ):
        """Create representation transformation.

        Args:
            representation: 'frame' for ToFrame, 'voxel' for ToVoxelGrid.
            normalize: whether to normalize the resulting tensor to [0, 1].
        """
        if representation == 'voxel':
            rep_fn = ToVoxelGrid(sensor_size=sensor_size, n_time_bins=n_time_bins)
        else:
            rep_fn = ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins)

        return torchvision.transforms.Compose([
            rep_fn,
            ToTensor(normalize=normalize),
            torchvision.transforms.Lambda(lambda x: torch.nn.functional.adaptive_max_pool2d(x, resize_size)),
        ])

    # ------------------------------------------------------------------
    # Setup-aware builders
    # ------------------------------------------------------------------

    @classmethod
    def _build_pre_aug(
            cls,
            setup: AugmentationSetup,
            sensor_size: Tuple[int, int],
            families=None,
    ):
        """Return the pre-augmentation transform for the given setup."""
        if families is None:
            families = frozenset(['temporal', 'spatial', 'polarity'])
        families = frozenset(families)

        if setup == AugmentationSetup.EVENTCLR:
            pre_transforms = []
            if 'temporal' in families:
                pre_transforms.append(RandomCropTime(sensor_size=sensor_size, crop_ratio=(0.1, 1.0)))
            if 'spatial' in families:
                pre_transforms.append(RandomApply(transforms.Compose([Rolling(sensor_size=sensor_size, max_shift=0.5)]), p=0.5))
            return transforms.Compose(pre_transforms)
        elif setup == AugmentationSetup.NDA:
            return NeuromorphicDataAugmentation(sensor_size=sensor_size, n=2, m=3)
        elif setup == AugmentationSetup.EVENTDROP:
            return EventDrop(sensor_size=sensor_size)
        else:
            raise ValueError(f"Unknown augmentation setup: {setup!r}")

    @classmethod
    def _build_post_aug(
            cls,
            setup: AugmentationSetup,
            sensor_size: Tuple[int, int],
            resize_size: Tuple[int, int],
            families=None,
    ):
        """Return the post-augmentation transform for the given setup."""
        if families is None:
            families = frozenset(['temporal', 'spatial', 'polarity'])

        if setup == AugmentationSetup.EVENTCLR:
            return torchvision.transforms.Compose([
                AugmentEventCLR(sensor_size=sensor_size, target_size=resize_size, strength='strong', families=families),
            ])
        elif setup in (AugmentationSetup.NDA, AugmentationSetup.EVENTDROP):
            # NDA and EventDrop handle all augmentation in the event domain (pre-rep),
            # so no post-representation augmentation is needed.
            return torchvision.transforms.Compose([])
        else:
            raise ValueError(f"Unknown augmentation setup: {setup!r}")

    @classmethod
    def create_pretrain_transforms(
            cls,
            sensor_size: Tuple[int, int],
            resize_size: Tuple[int, int],
            n_time_bins: int = 4,
            representation: str = 'frame',
            normalize: bool = True,
            setup: AugmentationSetup = AugmentationSetup.EVENTCLR,
            families=None,
    ) -> DataTransform:
        """Create transformations for pretraining.

        Args:
            setup: Which augmentation setup to use (eventclr, nda, eventdrop).
            families: Augmentation families to enable ('temporal', 'spatial', 'polarity').
                      Defaults to all three. Pass a subset for ablation studies.
        """
        if families is None:
            families = frozenset(['temporal', 'spatial', 'polarity'])
        families = frozenset(families)

        representation_fn = cls.create_representation(sensor_size, n_time_bins, resize_size, representation, normalize)

        return DataTransform(
            pre_augmentation_fn=cls._build_pre_aug(setup, sensor_size, families=families),
            representation_fn=representation_fn,
            post_augmentation_fn=cls._build_post_aug(setup, sensor_size, resize_size, families=families),
        )

    @classmethod
    def create_eval_transforms(
            cls,
            sensor_size: Tuple[int, int],
            resize_size: Tuple[int, int],
            n_time_bins: int = 1,
            representation: str = 'frame',
            normalize: bool = True,
    ) -> DataTransform:
        """Create transformations for evaluation (light, fixed pipeline)."""
        representation_fn = cls.create_representation(sensor_size, n_time_bins, resize_size, representation, normalize)

        return DataTransform(
            pre_augmentation_fn=transforms.Compose([
                RandomCropTime(sensor_size=sensor_size, crop_ratio=(0.5, 1.0)),
                RandomApply(transforms.Compose([Rolling(sensor_size=sensor_size, max_shift=0.2)]), p=0.5),
            ]),
            representation_fn=representation_fn,
            post_augmentation_fn=torchvision.transforms.Compose([]),
        )

    # ------------------------------------------------------------------
    # Convenience constructors that read directly from an ExperimentConfig
    # ------------------------------------------------------------------

    @classmethod
    def pretrain_from_config(
            cls,
            config: ExperimentConfig,
            sensor_size=None,
            setup: AugmentationSetup = AugmentationSetup.EVENTCLR,
    ) -> DataTransform:
        """Create pretrain transforms from config (uses config.sensor_size by default)."""
        return cls.create_pretrain_transforms(
            sensor_size or config.sensor_size,
            config.resize_size,
            config.n_time_bins,
            config.representation,
            config.normalize,
            setup=setup,
            families=config.aug_families,
        )

    @classmethod
    def eval_from_config(cls, config: ExperimentConfig) -> DataTransform:
        """Create eval transforms from config."""
        return cls.create_eval_transforms(
            config.sensor_size,
            config.resize_size,
            config.n_time_bins,
            config.representation,
            config.normalize,
        )

