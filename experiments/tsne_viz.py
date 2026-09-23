"""t-SNE visualization of pretrained EventCLR features vs supervised features.

For a given labeled subset (e.g. 9-shot or 45-shot), extracts features from:
  1. A EventCLR pretrained encoder (frozen backbone)
  2. A supervised-from-scratch encoder (trained in the same few-shot setting)

Then runs t-SNE (sklearn) and saves PNG plots colored by class label.

Usage
-----
python experiments/tsne_viz.py \
    --pretrained-run-id <MLFLOW_RUN_ID> \
    --target-dataset cifar10dvs \
    --samples-per-class 9 --samples-mode count \
    --device 0 --output-dir figures/tsne
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from spikingjelly.activation_based import functional, surrogate
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import sys

# Allow direct script execution from the debug/ folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augmentations.transform_factory import TransformFactory
from datasets.dataset_factory import DatasetFactory
from datasets.subset import create_subset
from models.backbones.backbone_factory import BackboneFactory
from models.finetuning import FinetuningModule
from modules.supervised import LitSupervised
from utils.config import ExperimentConfig
from utils.seed import set_seed
from utils.setup_mlflow import setup_mlflow


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features_snn(backbone: nn.Module, loader: DataLoader, device: torch.device):
    """Extract features from an SNN backbone. Returns (features, labels)."""
    backbone.eval().to(device)
    all_features, all_labels = [], []
    for x, y in loader:
        x = x.permute(1, 0, 2, 3, 4).to(device)  # (T, B, C, H, W)
        h = backbone(x)                            # (T, B, n_features)
        functional.reset_net(backbone)
        h = h.mean(dim=0)                          # (B, n_features)
        all_features.append(h.cpu())
        all_labels.append(y)
    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, title: str, save_path: str,
              class_names=None) -> str:
    n_classes = len(np.unique(labels))
    cmap = plt.cm.get_cmap("tab10", n_classes)

    fig, ax = plt.subplots(figsize=(8, 8))
    for cls in range(n_classes):
        idx = labels == cls
        lbl = class_names[cls] if class_names else str(cls)
        ax.scatter(embeddings[idx, 0], embeddings[idx, 1], c=[cmap(cls)],
                   label=lbl, alpha=0.7, s=18, linewidths=0)
        pts = embeddings[idx]
        centroid = pts.mean(axis=0)
        ax.scatter(*centroid, c=[cmap(cls)], s=180, marker="*",
                   edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(lbl, centroid, fontsize=7, fontweight="bold",
                    ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")
        if len(pts) > 2:
            cov = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * np.sqrt(vals)
            ellipse = Ellipse(centroid, w, h, angle=angle,
                              edgecolor=cmap(cls), facecolor="none",
                              linewidth=1.2, linestyle="--", zorder=4)
            ax.add_patch(ellipse)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=8, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization: pretrained vs supervised features")
    parser.add_argument("--pretrained-run-id", required=True)
    parser.add_argument("--target-dataset", default="cifar10dvs",
                        choices=list(DatasetFactory.DATASETS.keys()))
    parser.add_argument("--samples-per-class", type=float, default=9)
    parser.add_argument("--samples-mode", default="count", choices=["percent", "count"])
    parser.add_argument("--n-time-bins", type=int, default=4)
    parser.add_argument("--resize-size", type=int, nargs=2, default=[48, 48], metavar=("H", "W"))
    parser.add_argument("--representation", default="frame", choices=["frame", "voxel"])
    parser.add_argument("--eval-epochs", type=int, default=150,
                        help="Epochs for training the supervised baseline")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-lr", type=float, default=1e-3)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-n-iter", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone-name", default="resnet18", choices=["resnet18", "resnet18sep", "vgg9"])
    parser.add_argument("--output-dir", default="figures/tsne")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    setup_mlflow("EventCLR")

    with mlflow.start_run():

        config = ExperimentConfig(
            target_dataset=args.target_dataset,
            backbone_name=args.backbone_name,
            n_time_bins=args.n_time_bins,
            resize_size=tuple(args.resize_size),
            representation=args.representation,
            eval_epochs=args.eval_epochs,
            eval_batch_size=args.eval_batch_size,
            eval_lr_linear=args.eval_lr,
            eval_lr_finetune=args.eval_lr,
            samples_mode=args.samples_mode,
            device=args.device,
            num_workers=args.num_workers,
            seed=args.seed,
            decay_input=False,
            surrogate_function=surrogate.ATan(),
            backend='cupy',
        )

        # Build transforms
        data_transform = TransformFactory.eval_from_config(config)

        # Full validation set for feature extraction
        val_dataset = DatasetFactory.create(
            args.target_dataset, train=False, transform=data_transform.val_transform
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        # Labeled subset for supervised baseline training
        full_train = DatasetFactory.create(
            args.target_dataset, train=True, transform=data_transform.train_transform
        )
        train_subset = create_subset(
            full_train, args.samples_per_class, seed=args.seed, samples_mode=args.samples_mode
        )
        train_loader = DataLoader(
            train_subset, batch_size=args.eval_batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )

        # ---- 1. EventCLR pretrained features ----------------------------------------
        print("\n[1/2] Extracting EventCLR pretrained features...")
        pretrained_backbone, n_features = BackboneFactory.load_pretrained_from_mlflow(
            args.pretrained_run_id, config
        )
        pretrained_feats, labels = extract_features_snn(pretrained_backbone, val_loader, device)
        print(f"  Feature matrix: {pretrained_feats.shape}")

        # ---- 2. Supervised-from-scratch features ------------------------------------
        print("\n[2/2] Training supervised baseline and extracting features...")
        sup_backbone, _ = BackboneFactory.create(
            config.backbone_name, in_channels=2, width=config.backbone_width,
            neuron_type=config.neuron_type, cnf=config.cnf, **config.neuron_kwargs,
        )
        sup_model = FinetuningModule(
            backbone=sup_backbone, n_features=n_features, n_classes=config.num_classes
        )
        lit_sup = LitSupervised(
            model=sup_model, lr=args.eval_lr, weight_decay=1e-4,
            max_epochs=args.eval_epochs, use_cutmix=False,
        )
        trainer = Trainer(
            max_epochs=args.eval_epochs, accelerator='auto', devices=[args.device],
            enable_checkpointing=False, logger=False, num_sanity_val_steps=0,
            deterministic="warn",
        )
        set_seed(args.seed)
        trainer.fit(lit_sup, train_loader, val_loader)

        sup_feats, sup_labels = extract_features_snn(sup_model.backbone, val_loader, device)
        print(f"  Feature matrix: {sup_feats.shape}")

        # ---- t-SNE -------------------------------------------------------------------
        print("\nRunning t-SNE...")
        tsne = TSNE(
            n_components=2,
            perplexity=args.tsne_perplexity,
            max_iter=args.tsne_n_iter,
            random_state=args.seed,
            init='pca',
        )

        pretrained_2d = tsne.fit_transform(pretrained_feats)
        sup_2d = tsne.fit_transform(sup_feats)

        shot_label = (
            f"{int(args.samples_per_class)}-shot"
            if args.samples_mode == 'count'
            else f"{args.samples_per_class * 100:.0f}%-shot"
        )

        run_name = f"tsne_{args.target_dataset}_{shot_label}"
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.set_tag("pretrained_run_id", args.pretrained_run_id)
            mlflow.log_params({
                "target_dataset": args.target_dataset,
                "samples_per_class": args.samples_per_class,
                "samples_mode": args.samples_mode,
                "tsne_perplexity": args.tsne_perplexity,
                "tsne_n_iter": args.tsne_n_iter,
                "seed": args.seed,
                "backbone_name": args.backbone_name,
            })

            raw_classes = getattr(val_dataset, "classes", None)
            if isinstance(raw_classes, dict):
                # {name: idx} — reorder by index value
                class_names = [name for name, _ in sorted(raw_classes.items(), key=lambda x: x[1])]
            else:
                class_names = raw_classes  # list or None

            path_pretrained = plot_tsne(
                pretrained_2d, labels,
                title=f"EventCLR Pretrained — {args.target_dataset} ({shot_label} labeled subset)",
                save_path=os.path.join(args.output_dir, f"tsne_eventclr_{args.target_dataset}_{shot_label}.png"),
                class_names=class_names,
            )
            path_supervised = plot_tsne(
                sup_2d, sup_labels,
                title=f"Supervised Baseline — {args.target_dataset} ({shot_label} labeled subset)",
                save_path=os.path.join(args.output_dir, f"tsne_supervised_{args.target_dataset}_{shot_label}.png"),
                class_names=class_names,
            )

            mlflow.log_artifact(path_pretrained, artifact_path="tsne")
            mlflow.log_artifact(path_supervised, artifact_path="tsne")

        print(f"\n✓ t-SNE plots saved to {args.output_dir}/ and logged to MLflow run '{run_name}'")


if __name__ == "__main__":
    main()