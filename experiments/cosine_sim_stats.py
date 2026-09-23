"""Cosine similarity statistics for a pretrained EventCLR model.

Computes and reports, over the full training set:
  - Mean cosine similarity of positive pairs (two augmented views of the same sample)
  - Mean cosine similarity of negative pairs (cross-pair cosine sim within each batch)
  - Gap and ratio between the two

Usage
-----
python experiments/cosine_sim_stats.py \
    --pretrained-run-id <MLFLOW_RUN_ID> \
    --target-dataset cifar10dvs \
    --n-time-bins 4 \
    --device 0
"""
import argparse
import copy
import os
import sys

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional, surrogate
from torch.utils.data import DataLoader

# Allow direct script execution from the debug/ folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augmentations.transform_factory import TransformFactory, AugmentationSetup
from datasets.dataset_factory import DatasetFactory
from models.backbones.backbone_factory import BackboneFactory
from models.pretraining import EventCLR
from utils.config import ExperimentConfig
from utils.seed import set_seed
from utils.setup_mlflow import setup_mlflow


@torch.no_grad()
def compute_cosine_similarity_stats(
        model: EventCLR,
        loader: DataLoader,
        device: torch.device,
        n_batches: int = 50,
) -> dict:
    """Run forward passes and collect positive/negative cosine similarities.

    Args:
        model: Pretrained EventCLR (backbone + projection head).
        loader: DataLoader yielding ((x_i, x_j), target) pairs.
        device: Compute device.
        n_batches: Maximum number of batches to process (None = all).

    Returns:
        dict with keys: pos_mean, pos_std, neg_mean, neg_std, gap, ratio
    """
    model.eval()
    model.to(device)

    pos_sims: list[float] = []
    neg_sims: list[float] = []

    for batch_idx, ((x_i, x_j), _) in enumerate(loader):
        if n_batches is not None and batch_idx >= n_batches:
            break

        # (B, T, C, H, W) → (T, B, C, H, W)
        x_i = x_i.permute(1, 0, 2, 3, 4).to(device)
        x_j = x_j.permute(1, 0, 2, 3, 4).to(device)

        z_i = model(x_i)  # (T, B, proj_dim)
        functional.reset_net(model)
        z_j = model(x_j)
        functional.reset_net(model)

        # Average over time dimension → (B, proj_dim)
        z_i = z_i.mean(dim=0)
        z_j = z_j.mean(dim=0)

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        B = z_i.shape[0]

        # Positive similarity: diagonal of (z_i · z_j^T) per sample
        pos = (z_i * z_j).sum(dim=1)  # (B,)
        pos_sims.extend(pos.cpu().tolist())

        # Negative similarity: all off-diagonal pairs within the batch
        # Build the full (2B, 2B) similarity matrix and extract off-diagonal
        representations = torch.cat([z_i, z_j], dim=0)  # (2B, proj_dim)
        sim_matrix = representations @ representations.T  # (2B, 2B)

        # Positive-pair indices: (i, i+B) and (i+B, i)
        pos_mask = torch.zeros(2 * B, 2 * B, dtype=torch.bool, device=device)
        for i in range(B):
            pos_mask[i, i + B] = True
            pos_mask[i + B, i] = True
        diag_mask = torch.eye(2 * B, dtype=torch.bool, device=device)
        neg_mask = ~pos_mask & ~diag_mask

        neg = sim_matrix[neg_mask]  # flatten negative pairs
        neg_sims.extend(neg.cpu().tolist())

    pos_arr = np.array(pos_sims)
    neg_arr = np.array(neg_sims)

    stats = {
        "pos_mean": float(pos_arr.mean()),
        "pos_std": float(pos_arr.std()),
        "neg_mean": float(neg_arr.mean()),
        "neg_std": float(neg_arr.std()),
        "gap": float(pos_arr.mean() - neg_arr.mean()),
        "ratio": float(pos_arr.mean() / (abs(neg_arr.mean()) + 1e-8)),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Cosine similarity statistics for a pretrained EventCLR model")
    parser.add_argument("--pretrained-run-id", required=True, help="MLflow run ID of the pretrained backbone")
    parser.add_argument("--target-dataset", default="cifar10dvs", choices=list(DatasetFactory.DATASETS.keys()))
    parser.add_argument("--n-time-bins", type=int, default=4)
    parser.add_argument("--resize-size", type=int, nargs=2, default=[48, 48], metavar=("H", "W"))
    parser.add_argument("--representation", default="frame", choices=["frame", "voxel"])
    parser.add_argument("--aug-setup", default="eventclr", choices=["eventclr", "nda", "eventdrop"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--n-batches", type=int, default=50,
                        help="Number of batches to process (None = all)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone-name", default="resnet18", choices=["resnet18", "resnet18sep", "vgg9"])
    parser.add_argument("--projection-dim", type=int, default=128)
    args = parser.parse_args()

    set_seed(args.seed)
    setup_mlflow("EventCLR")
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    config = ExperimentConfig(
        target_dataset=args.target_dataset,
        backbone_name=args.backbone_name,
        n_time_bins=args.n_time_bins,
        resize_size=tuple(args.resize_size),
        representation=args.representation,
        projection_dim=args.projection_dim,
        aug_setup=args.aug_setup,
        decay_input=False,
        surrogate_function=surrogate.ATan(),
        backend='cupy',
    )

    # Load pretrained backbone
    backbone, n_features = BackboneFactory.load_pretrained_from_mlflow(args.pretrained_run_id, config)

    # Rebuild full EventCLR model (backbone + projection head)
    eventclr = EventCLR(
        backbone=backbone,
        n_features=n_features,
        projection_dim=args.projection_dim,
        neuron_type='LIF',
        decay_input=False,
        surrogate_function=surrogate.ATan(),
        backend='cupy',
    )

    # Build pretrain-style dataloader (returns pairs)
    sensor_size = DatasetFactory.get_sensor_size(args.target_dataset)
    data_transform = TransformFactory.create_pretrain_transforms(
        sensor_size,
        tuple(args.resize_size),
        args.n_time_bins,
        args.representation,
        normalize=True,
        setup=AugmentationSetup(args.aug_setup),
    )
    dataset = DatasetFactory.create(
        args.target_dataset, train=True, transform=data_transform.pretrain_transform
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    print(f"\nComputing cosine similarity stats on {args.target_dataset} "
          f"({min(args.n_batches, len(loader))} batches of {args.batch_size})...\n")

    stats = compute_cosine_similarity_stats(eventclr, loader, device, n_batches=args.n_batches)

    print("=" * 50)
    print("NT-Xent Cosine Similarity Statistics")
    print("=" * 50)
    print(f"  Positive pairs — mean: {stats['pos_mean']:.4f}  std: {stats['pos_std']:.4f}")
    print(f"  Negative pairs — mean: {stats['neg_mean']:.4f}  std: {stats['neg_std']:.4f}")
    print(f"  Gap  (pos - neg): {stats['gap']:.4f}")
    print(f"  Ratio (pos / |neg|): {stats['ratio']:.4f}")
    print("=" * 50)

    # Log to the parent MLflow run if one is active
    with mlflow.start_run(run_name="cosine_sim_stats"):
        mlflow.log_param("pretrained_run_id", args.pretrained_run_id)
        mlflow.log_param("dataset", args.target_dataset)
        mlflow.log_param("n_batches", args.n_batches)
        mlflow.log_metrics(stats)
        print("\n✓ Results logged to MLflow.")


if __name__ == "__main__":
    main()
