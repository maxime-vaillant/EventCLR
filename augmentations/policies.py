from enum import Enum
from typing import Tuple

import numpy as np
from tonic.transforms import DropEventByTime, DropEvent
from torchvision.transforms import InterpolationMode, transforms

from augmentations.event_transforms import RandomFlipLR, Rolling, DropEventByArea, Rotate, ShearX, RandomApply
from augmentations.frame_transforms import PolarityJitter, RandomPolarityAverage

class AugmentEventCLR:
    def __init__(self, sensor_size, target_size=(48, 48), strength='light', families=None):
        self.sensor_size = sensor_size
        self.target_size = target_size
        self.strength = strength

        if families is None:
            families = frozenset(['temporal', 'spatial', 'polarity'])
        self.families = frozenset(families)

        if strength == 'light':
            aug_list = []
            if 'spatial' in self.families:
                aug_list.append(transforms.RandomResizedCrop(size=target_size, scale=(0.5, 1.0), interpolation=InterpolationMode.NEAREST_EXACT))
                aug_list.append(transforms.RandomHorizontalFlip(p=0.5))
            if 'polarity' in self.families:
                aug_list.append(RandomApply(PolarityJitter(contrast=0.2, brightness=0.2), p=0.5))
            self.augmentations = RandomApply(transforms.Compose(aug_list), p=0.1) if aug_list else (lambda x: x)

        elif strength == 'strong':
            aug_list = []
            if 'spatial' in self.families:
                aug_list.append(transforms.RandomResizedCrop(size=target_size, scale=(0.1, 1.0), interpolation=InterpolationMode.NEAREST_EXACT))
                aug_list.append(transforms.RandomHorizontalFlip(p=0.5))
            if 'polarity' in self.families:
                aug_list.append(RandomApply(PolarityJitter(contrast=0.5, brightness=0.5), p=0.8))
                aug_list.append(RandomPolarityAverage(p=0.2))
            self.augmentations = transforms.Compose(aug_list) if aug_list else (lambda x: x)

    def __call__(self, events):
        return self.augmentations(events)


class AugmentationType(Enum):
    """Enumeration of available augmentation types."""
    IDENTITY = "identity"
    FLIP = "flip"
    ROLL = "roll"
    ROTATE = "rotate"
    SHEAR = "shear"
    CUTOUT = "cutout"


class NeuromorphicDataAugmentation:
    """
    Applies random augmentations to neuromorphic event data.

    This class provides a collection of data augmentation techniques specifically
    designed for event-based neuromorphic datasets. It randomly selects and applies
    one augmentation from a set of geometric and spatial transformations to enhance
    model robustness and generalization.

    Methods from paper https://arxiv.org/pdf/2203.06145

    Attributes:
        sensor_size (tuple): The dimensions of the event sensor (height, width) or
            (height, width, polarity).
        flip (tonic.transforms.RandomFlipLR): Random horizontal flip transformation.
        roll (Rolling): Spatial rolling/shifting transformation along x and y axes.
        cutout (tonic.transforms.DropEventByArea): Random event dropout by spatial area.
        rotate (Rotate): Rotation transformation within specified angle range.
        shear (ShearX): Horizontal shear transformation.

    Args:
        sensor_size (tuple): The dimensions of the event sensor used to configure
            all augmentation transforms.

    Example:
        nda = NeuromorphicDataAugmentation(sensor_size=(128, 128, 2))
        augmented_events = nda(events)

    Note:
        The class automatically applies one randomly selected augmentation per call,
        including the possibility of no augmentation (identity transformation).
    """

    def __init__(self, sensor_size, n, m):
        self.sensor_size = sensor_size
        self._init_transforms(n)

        self.m = m
        self.n = n

    def _init_transforms(self, n):
        """Initialize all augmentation transforms."""
        self.flip = RandomFlipLR(
            sensor_size=self.sensor_size,
            p=0.5
        )
        self.roll = Rolling(
            sensor_size=self.sensor_size,
            max_shift=(1 + 2*n) / float(max(self.sensor_size[0], self.sensor_size[1]))
        )
        self.cutout = DropEventByArea(
            sensor_size=self.sensor_size,
            area_ratio=8*n / float(max(self.sensor_size[0], self.sensor_size[1]))
        )
        self.rotate = Rotate(
            sensor_size=self.sensor_size,
            angle=(-15*n, 15*n)
        )
        self.shear = ShearX(
            sensor_size=self.sensor_size,
            shear_range=(-0.15*n, 0.15*n)
        )

        self._augmentation_map = {
            AugmentationType.ROLL: self.roll,
            AugmentationType.ROTATE: self.rotate,
            AugmentationType.SHEAR: self.shear,
            AugmentationType.CUTOUT: self.cutout,
        }

    def __repr__(self):
        return (f"{self.__class__.__name__}(sensor_size={self.sensor_size}, "
                f"m={self.m}, n={self.n})")

    def __call__(self, events):
        """Apply a randomly selected augmentation to events."""
        events = self.flip(events)

        # Choose m transforms directly from the augmentation map
        chosen_transforms = list(np.random.choice(
            list(self._augmentation_map.values()),
            size=self.m,
            replace=False
        ))

        for transform in chosen_transforms:
            events = transform(events)

        return events


class EventDrop:
    """Applies EventDrop transformation from the paper "EventDrop: Data Augmentation for Event-based Learning".
        Applies one of the 4 drops of event strategies between:
            1. Identity (do nothing)
            2. Drop events by time
            3. Drop events by area
            4. Drop events randomly

        For each strategy, the ratio of dropped events are determined in the paper.

    Args:
        sensor_size (Tuple): size of the sensor that was used [W,H,P]
    """

    def __init__(self, sensor_size):
        self.sensor_size = sensor_size

        self.drop_by_time = DropEventByTime(duration_ratio=(0.1, 0.9))
        self.drop_by_area = DropEventByArea(sensor_size=sensor_size, area_ratio=(0.05, 0.24))
        self.drop_event = DropEvent(p=(0.1, 0.9))

        self.identity = lambda x: x.copy()

    def __call__(self, events):
        choice = np.random.randint(0, 4)
        if choice == 0:
            return self.identity(events)
        if choice == 1:
            return self.drop_by_time(events)
        if choice == 2:
            return self.drop_by_area(events)
        if choice == 3:
            return self.drop_event(events)
        return None
