import os

import tonic

from datasets.cifar10dvs import CIFAR10DVS
from datasets.ncaltech101 import NCALTECH101
from datasets.ncars import NCARS
from datasets.nimagenet import NImageNet

DEFAULT_PATH = os.path.expanduser('~/data')


class DatasetFactory:
    """Factory for creating different datasets."""

    DATASETS = {
        'cifar10dvs': {
            'class': CIFAR10DVS,
            'sensor_size': tonic.datasets.CIFAR10DVS.sensor_size,
            'num_classes': 10
        },
        'ncaltech101': {
            'class': NCALTECH101,
            'sensor_size': (240, 180, 2),
            'num_classes': 101
        },
        'nmnist': {
            'class': tonic.datasets.NMNIST,
            'sensor_size': tonic.datasets.NMNIST.sensor_size,
            'num_classes': 10
        },
        'dvsgesture': {
            'class': tonic.datasets.DVSGesture,
            'sensor_size': tonic.datasets.DVSGesture.sensor_size,
            'num_classes': 11
        },
        'ncars': {
            'class': NCARS,
            'sensor_size': NCARS.sensor_size,
            'num_classes': 2,
        },
        'nimagenet': {
            'class': NImageNet,
            'sensor_size': NImageNet.sensor_size,
            'num_classes': 1000
        }
    }

    @classmethod
    def create(cls, dataset_name, train=True, transform=None, path=DEFAULT_PATH, **kwargs):
        """Create a dataset instance with proper train/test handling."""
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = cls.DATASETS[dataset_name]
        params = {
            'save_to': path,
            'transform': transform,
            'train': train
        }
        params.update(kwargs)

        return config['class'](**params)

    @classmethod
    def get_sensor_size(cls, dataset_name):
        """Get sensor size for a dataset."""
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return cls.DATASETS[dataset_name]['sensor_size']

    @classmethod
    def get_num_classes(cls, dataset_name):
        """Get number of classes for a dataset."""
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return cls.DATASETS[dataset_name]['num_classes']
