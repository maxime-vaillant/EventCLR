from typing import Optional

import mlflow
import torch
from torch.utils.data import DataLoader

from augmentations.transform_factory import TransformFactory, AugmentationSetup
from datasets.dataset_factory import DatasetFactory
from datasets.subset import create_subset
from models.backbones.backbone_factory import BackboneFactory
from models.finetuning import FinetuningModule
from models.linear_probing import LinearProbingModule
from modules.supervised import LitSupervised
from pytorch_lightning import Trainer
from utils.config import ExperimentConfig
from utils.seed import set_seed


class EvaluationPipeline:
    """Pipeline for evaluating models."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_backbone(self):
        """Instantiate a fresh backbone from config."""
        backbone, n_features = BackboneFactory.create(
            self.config.backbone_name,
            in_channels=2,
            width=self.config.backbone_width,
            neuron_type=self.config.neuron_type,
            cnf=self.config.cnf,
            **self.config.neuron_kwargs,
        )
        return backbone, n_features

    def _create_dataloaders(
            self,
            samples_per_class_percentage: float,
            subset_seed: Optional[int] = None,
    ):
        """Create train and validation dataloaders for the target dataset."""
        data_transform = TransformFactory.eval_from_config(self.config)

        full_train_dataset = DatasetFactory.create(
            self.config.target_dataset,
            train=True,
            transform=data_transform.train_transform,
        )
        train_dataset = create_subset(
            full_train_dataset,
            samples_per_class_percentage,
            seed=subset_seed,
            samples_mode=self.config.samples_mode,
        )
        if hasattr(train_dataset, 'indices') and len(train_dataset.indices) > 0:
            mlflow.log_param("subset_first_idx", str(train_dataset.indices[:5]))

        val_dataset = DatasetFactory.create(
            self.config.target_dataset,
            train=False,
            transform=data_transform.val_transform,
        )

        loader_kwargs = dict(
            batch_size=self.config.eval_batch_size,
            num_workers=self.config.num_workers,
            prefetch_factor=2,
            pin_memory=True,
        )
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

        return train_loader, val_loader

    def _train_model(self, model, train_loader, val_loader, lr: float, seed: Optional[int] = None) -> float:
        """Train a model and return best validation accuracy."""
        lit_model = LitSupervised(
            model=model,
            lr=lr,
            weight_decay=self.config.eval_weight_decay,
            max_epochs=self.config.eval_epochs,
            use_cutmix=self.config.use_cutmix,
            cutmix_prob=self.config.cutmix_prob,
            strategy=self.config.supervised_loss_strategy
        )

        trainer = Trainer(
            max_epochs=self.config.eval_epochs,
            accelerator='auto',
            devices=[self.config.device],
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
            deterministic="warn",
        )

        set_seed(seed if seed is not None else self.config.seed)
        trainer.fit(lit_model, train_loader, val_loader)
        return lit_model.best_val_acc

    # ------------------------------------------------------------------
    # Public evaluation methods
    # ------------------------------------------------------------------

    def evaluate_linear_probing(
            self,
            backbone,
            n_features: int,
            samples_per_class: float,
            subset_seed: Optional[int] = None,
    ) -> float:
        """Evaluate with linear probing (frozen backbone)."""
        train_loader, val_loader = self._create_dataloaders(samples_per_class, subset_seed=subset_seed)

        backbone_copy, _ = self._create_backbone()
        backbone_copy.load_state_dict(backbone.state_dict())

        model = LinearProbingModule(
            backbone=backbone_copy,
            n_features=n_features,
            n_classes=self.config.num_classes,
        )
        return self._train_model(model, train_loader, val_loader, self.config.eval_lr_linear, seed=subset_seed)

    def evaluate_finetuning(
            self,
            backbone,
            n_features: int,
            samples_per_class: float,
            subset_seed: Optional[int] = None,
    ) -> float:
        """Evaluate with full backbone finetuning."""
        train_loader, val_loader = self._create_dataloaders(samples_per_class, subset_seed=subset_seed)

        backbone_copy, _ = self._create_backbone()
        backbone_copy.load_state_dict(backbone.state_dict())

        model = FinetuningModule(
            backbone=backbone_copy,
            n_features=n_features,
            n_classes=self.config.num_classes,
        )
        return self._train_model(model, train_loader, val_loader, self.config.eval_lr_finetune, seed=subset_seed)

    def evaluate_supervised(
            self,
            n_features: int,
            samples_per_class: float,
            subset_seed: Optional[int] = None,
    ) -> float:
        """Evaluate supervised learning from scratch (no pretrained backbone)."""
        train_loader, val_loader = self._create_dataloaders(samples_per_class, subset_seed=subset_seed)

        backbone, _ = self._create_backbone()
        model = FinetuningModule(
            backbone=backbone,
            n_features=n_features,
            n_classes=self.config.num_classes,
        )
        return self._train_model(model, train_loader, val_loader, self.config.eval_lr_linear, seed=subset_seed)

    def evaluate_supervised_with_aug(
            self,
            n_features: int,
            samples_per_class: float,
            subset_seed: Optional[int] = None,
    ) -> float:
        """Supervised from scratch with the full EventCLR augmentation pipeline.

        Disentangles whether SSL gains come from the pretraining objective or
        from augmentation alone. Uses the EventCLR augmentation as a single-view
        train transform (not contrastive pairs).
        """
        # Build train transform using EventCLR full augmentation (single view)
        aug_transform = TransformFactory.create_pretrain_transforms(
            self.config.sensor_size,
            self.config.resize_size,
            self.config.n_time_bins,
            self.config.representation,
            self.config.normalize,
            setup=AugmentationSetup(self.config.aug_setup),
            families=self.config.aug_families,
        )
        eval_transform = TransformFactory.eval_from_config(self.config)

        full_train = DatasetFactory.create(
            self.config.target_dataset,
            train=True,
            transform=aug_transform.train_transform,  # single-view augmented
        )
        train_dataset = create_subset(
            full_train, samples_per_class, seed=subset_seed, samples_mode=self.config.samples_mode
        )
        val_dataset = DatasetFactory.create(
            self.config.target_dataset,
            train=False,
            transform=eval_transform.val_transform,
        )

        loader_kwargs = dict(
            batch_size=self.config.eval_batch_size,
            num_workers=self.config.num_workers,
            prefetch_factor=2,
            pin_memory=True,
        )
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

        backbone, _ = self._create_backbone()
        model = FinetuningModule(
            backbone=backbone,
            n_features=n_features,
            n_classes=self.config.num_classes,
        )
        return self._train_model(model, train_loader, val_loader, self.config.eval_lr_linear, seed=subset_seed)

