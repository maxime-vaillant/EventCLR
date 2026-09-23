import os
import tempfile

import mlflow
import torch
from pytorch_lightning import Trainer
from torch.utils.data import ConcatDataset, DataLoader

from augmentations.transform_factory import TransformFactory, AugmentationSetup
from datasets.dataset_factory import DatasetFactory
from datasets.subset import create_subset
from models.backbones.backbone_factory import BackboneFactory
from models.pretraining import EventCLR
from modules.self_supervised import LitSelfSupervised
from pipelines.utils import print_section
from utils.config import ExperimentConfig
from utils.seed import set_seed


class PretrainPipeline:
    """Pipeline for self-supervised pretraining."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self):
        """Pretrain the backbone using SimCLR on multiple datasets."""
        print_section(
            f"PHASE: Self-Supervised Pretraining with SimCLR\n"
            f"Datasets: {', '.join(self.config.ssl_datasets)}"
        )

        # Build (possibly concatenated) training dataset
        datasets = []
        for dataset_name in self.config.ssl_datasets:
            sensor_size = DatasetFactory.get_sensor_size(dataset_name)
            data_transform = TransformFactory.create_pretrain_transforms(
                sensor_size,
                self.config.resize_size,
                self.config.n_time_bins,
                self.config.representation,
                self.config.normalize,
                setup=AugmentationSetup(self.config.aug_setup),
                families=self.config.aug_families,
            )
            dataset = DatasetFactory.create(
                dataset_name,
                train=True,
                transform=data_transform.pretrain_transform,
            )
            if self.config.pretrain_samples_percentage < 1.0:
                dataset = create_subset(dataset, self.config.pretrain_samples_percentage, seed=self.config.seed)
                print(
                    f"Loaded {dataset_name}: {len(dataset)} samples "
                    f"({self.config.pretrain_samples_percentage * 100}% of dataset)"
                )
            else:
                print(f"Loaded {dataset_name}: {len(dataset)} samples")
            datasets.append(dataset)

        train_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        if len(datasets) > 1:
            print(f"\nTotal pretraining samples: {len(train_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.pretrain_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Create model
        backbone, n_features = BackboneFactory.create(
            self.config.backbone_name,
            in_channels=2,
            width=self.config.backbone_width,
            neuron_type=self.config.neuron_type,
            cnf=self.config.cnf,
            **self.config.neuron_kwargs,
        )

        eventclr = EventCLR(
            backbone=backbone,
            n_features=n_features,
            projection_dim=self.config.projection_dim,
            neuron_type=self.config.neuron_type,
            **self.config.neuron_kwargs,
        )
        lit_model = LitSelfSupervised(
            model=eventclr,
            lr=self.config.pretrain_lr,
            weight_decay=self.config.pretrain_weight_decay,
            max_epochs=self.config.pretrain_epochs,
            strategy=self.config.pretrain_loss_strategy,
        )

        trainer = Trainer(
            max_epochs=self.config.pretrain_epochs,
            accelerator='auto',
            devices=[self.config.device],
            enable_checkpointing=False,
            logger=False,
            deterministic="warn",
        )
        set_seed(self.config.seed)
        trainer.fit(lit_model, train_loader)

        # Persist backbone weights to MLflow
        with tempfile.TemporaryDirectory() as tmp_dir:
            backbone_path = os.path.join(tmp_dir, "backbone_weights.pth")
            torch.save(eventclr.backbone.state_dict(), backbone_path)
            mlflow.log_artifact(backbone_path, artifact_path="pretrained_weights")
            print("\n✓ Saved backbone weights to MLflow")

        mlflow.log_param("n_features", n_features)
        mlflow.log_param("backbone_architecture", self.config.backbone_name)

        return eventclr.backbone, n_features

