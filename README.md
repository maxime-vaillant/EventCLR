# EventCLR

Self-supervised contrastive learning framework for spiking neural networks (SNNs) on neuromorphic (event-based) data. EventCLR adapts the SimCLR contrastive learning paradigm to the spiking domain, using event-specific augmentations and temporal-aware loss functions to learn rich spike-based representations without labels.

## Overview

EventCLR operates in two phases:

1. **Self-Supervised Pretraining** — A spiking backbone is trained using a contrastive objective (NT-Xent loss) on augmented pairs of event streams. Two augmented views of the same event recording are passed through a shared spiking encoder and projection head, and the model learns to maximise agreement between views of the same sample.

2. **Downstream Evaluation** — The pretrained backbone is evaluated on a target dataset via:
   - **Linear Probing (LP):** A linear classifier is trained on top of the frozen backbone.
   - **Finetuning (FT):** The full model (backbone + classifier) is finetuned end-to-end.
   - **Supervised Baseline (SUP):** A randomly initialised backbone is trained from scratch for comparison.

Evaluations are run across varying data fractions (e.g. 1 %, 5 %, 10 %, …, 100 % of labelled data per class) and multiple random seeds to produce robust accuracy estimates.

## Features

- **Spiking Backbones:** SEW-ResNet-18 (with ADD shortcut connections), SEW-ResNet-18Sep, and Spiking VGG-9, built on [SpikingJelly](https://github.com/fangwei123456/spikingjelly).
- **Event Augmentations:** A rich augmentation pipeline operating in both the event domain (temporal crop, spatial rolling, rotation, shear, area dropout) and the frame/tensor domain (random resized crop, horizontal flip, polarity jitter, polarity averaging).
- **Multiple Augmentation Policies:** `eventclr` (default), `nda` ([Neuromorphic Data Augmentation](https://arxiv.org/abs/2203.06145)), and `eventdrop`.
- **Temporal Loss Strategies:** `naive` (average embeddings over time, then compute loss) and `temporal` (compute loss at each time-step, then average).
- **Event Representations:** Frame-based (`ToFrame`) and voxel grid (`ToVoxelGrid`) representations via [Tonic](https://tonic.readthedocs.io/).
- **CutMix** augmentation during supervised/evaluation training.
- **Experiment Tracking:** Full integration with [MLflow](https://mlflow.org/) for parameter logging, metric tracking, and model artifact storage.
- **Configurable Data Subsets:** Evaluate label efficiency by training on controlled fractions of the labelled data.

## Supported Datasets

| Dataset | Classes | Sensor Size |
|---|---|---|
| CIFAR-10 DVS | 10 | 128 × 128 × 2 |
| N-Caltech101 | 101 | 240 × 180 × 2 |
| N-MNIST | 10 | 34 × 34 × 2 |
| DVS Gesture | 11 | 128 × 128 × 2 |
| N-ImageNet | 1000 | 640 × 480 × 2 |

Datasets are fetched and managed via [Tonic](https://tonic.readthedocs.io/).

## Installation

```bash
# Clone the repository
git clone https://github.com/maxime-vaillant/EventCLR.git
cd EventCLR

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** SpikingJelly is installed from a pinned Git commit. A CUDA-capable GPU with [cupy](https://cupy.dev/) is required to use the default `cupy` backend.

## Environment Variables

EventCLR uses a `.env` file at the project root for configuration (loaded via `python-dotenv`). Copy the example file and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | No | MLflow tracking server URI. Defaults to `local` (filesystem-based tracking) if unset. Set to a remote URI (e.g. `https://mlflow.example.com`) for centralized tracking. |
| `MLFLOW_TRACKING_USERNAME` | No | Username for authenticated MLflow tracking servers. |
| `MLFLOW_TRACKING_PASSWORD` | No | Password for authenticated MLflow tracking servers. |

Example `.env` for a remote MLflow server:

```dotenv
MLFLOW_TRACKING_URI=https://mlflow.example.com
MLFLOW_TRACKING_USERNAME=user
MLFLOW_TRACKING_PASSWORD=secret
```

For local-only tracking you can leave the file empty or omit it entirely — MLflow will log runs to the local `mlruns/` directory.

## Usage

All experiments are launched through `main.py`. The CLI exposes every configurable hyperparameter.

### Quickstart — Full Pipeline

```bash
# Pretrain on CIFAR-10 DVS, then evaluate with LP, FT, and supervised baselines
python main.py --target-dataset cifar10dvs --backbone resnet18 --pretrain-epochs 500 --eval-epochs 150
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--target-dataset` | `cifar10dvs` | Target evaluation dataset (`cifar10dvs`, `ncaltech101`, `nmnist`, `dvsgesture`, `nimagenet`) |
| `--ssl-datasets` | *(same as target)* | Datasets used for SSL pretraining (supports multiple) |
| `--backbone` | `resnet18` | Backbone architecture (`resnet18`, `resnet18sep`, `vgg9`) |
| `--backbone-width` | `1.0` | Width multiplier for backbone channels |
| `--projection-dim` | `128` | Output dimension of the projection head |
| `--pretrain-epochs` | `500` | Number of pretraining epochs |
| `--eval-epochs` | `150` | Number of evaluation epochs |
| `--pretrain-batch-size` | `256` | Batch size for pretraining |
| `--eval-batch-size` | `64` | Batch size for evaluation |
| `--pretrain-lr` | `1e-3` | Pretraining learning rate |
| `--eval-lr-linear` | `1e-3` | Learning rate for linear probing |
| `--eval-lr-finetune` | `1e-3` | Learning rate for finetuning |
| `--n-time-bins` | `4` | Number of temporal bins for the event representation |
| `--resize-size` | `48 48` | Spatial resize dimensions (H W) |
| `--representation` | `frame` | Event representation (`frame` or `voxel`) |
| `--normalize` / `--no-normalize` | `True` | Normalize representation to [0, 1] |
| `--samples-per-class` | `0 0.01 0.05 0.1 0.25 0.5 1.0` | Labelled data fractions to evaluate |
| `--samples-mode` | `percent` | Interpret samples-per-class as `percent` or `count` |
| `--n-subset-runs` | `3` | Number of random subset runs per data fraction |
| `--pretrain-samples` | `1.0` | Fraction of pretraining data to use |
| `--eval-types` | `lp ft sup` | Evaluation protocols to run |
| `--pretrain-loss-strategy` | `naive` | NT-Xent temporal strategy (`naive` or `temporal`) |
| `--supervised-loss-strategy` | `naive` | Cross-entropy temporal strategy (`naive` or `temporal`) |
| `--use-cutmix` / `--no-use-cutmix` | `True` | Enable CutMix augmentation during evaluation |
| `--cutmix-prob` | `0.5` | CutMix application probability |
| `--device` | `0` | GPU index |
| `--num-workers` | `32` | DataLoader worker count |
| `--seed` | `42` | Random seed |
| `--pretrained-run-id` | `None` | MLflow run ID to load a pretrained backbone (skips pretraining) |

### Skip Pretraining (Resume from MLflow)

If you have a pretrained backbone saved in an MLflow run, you can skip the pretraining phase:

```bash
python main.py --pretrained-run-id <MLFLOW_RUN_ID> --target-dataset cifar10dvs
```

### Run Only Specific Evaluation Types

```bash
# Only linear probing and finetuning (no supervised baseline)
python main.py --eval-types lp ft
```

### N-ImageNet LMDB Backend (Optional, Faster I/O)

`datasets/nimagenet.py` expects `<save_to>/NImageNet/training/<class>/*.npz` and
`<save_to>/NImageNet/validation/extracted_val/<class>/*.npz`, and supports
`backend="auto"|"npz"|"lmdb"`:

- `auto` (default): use LMDB if present, otherwise `.npz`
- `npz`: force file-per-sample loading
- `lmdb`: force LMDB loading

Convert `.npz` to LMDB once:

```bash
python debug/convert_nimagenet_to_lmdb.py --path ~/data --split both --overwrite
```

Sanity check and compare both backends:

```bash
python debug/read_nimagenet.py --path ~/data --split both --backend lmdb --compare-backends 16
```

## Project Structure

```
EventCLR/
├── main.py                        # CLI entry point
├── augmentations/
│   ├── provider.py                # DataTransform: train / val / pretrain views
│   ├── transform_factory.py       # Builds representation + augmentation pipelines
│   ├── policies.py                # EventCLR, NDA, and EventDrop augmentation policies
│   ├── event_transforms.py        # Event-domain transforms (crop, roll, flip, …)
│   ├── frame_transforms.py        # Tensor-domain transforms (polarity jitter, …)
│   ├── batch_transforms.py        # Batch-level transforms (CutMix)
│   └── representations.py         # ToFrame, ToVoxelGrid, ToTensor wrappers
├── datasets/
│   ├── dataset_factory.py         # Unified dataset creation interface
│   ├── cifar10dvs.py              # CIFAR-10 DVS dataset wrapper
│   ├── ncaltech101.py             # N-Caltech101 dataset wrapper
│   ├── nimagenet.py               # N-ImageNet dataset wrapper (npz / LMDB backends)
│   └── subset.py                  # Label-efficient subset sampling
├── losses/
│   ├── ntxent.py                  # NT-Xent (contrastive) loss with temporal variants
│   └── tet.py                     # Temporal Efficient Training (TET) loss
├── models/
│   ├── pretraining.py             # EventCLR model (backbone + projection head)
│   ├── linear_probing.py          # Frozen-backbone linear evaluation module
│   ├── finetuning.py              # Full finetuning evaluation module
│   └── backbones/
│       ├── backbone_factory.py    # Backbone creation and MLflow loading
│       ├── sew_resnet.py          # SEW-ResNet-18 / SEW-ResNet-18Sep
│       └── spiking_vgg.py         # Spiking VGG-9
├── modules/
│   ├── self_supervised.py         # PyTorch Lightning module for SSL pretraining
│   └── supervised.py              # PyTorch Lightning module for supervised training
├── pipelines/
│   ├── experiment.py              # Top-level experiment orchestration
│   ├── pretrain.py                # Pretraining pipeline
│   ├── evaluation.py              # Evaluation pipeline (LP / FT / supervised)
│   └── utils.py                   # Pipeline utilities
├── utils/
│   ├── config.py                  # ExperimentConfig dataclass
│   ├── metrics.py                 # Evaluation metrics (Top-K accuracy)
│   ├── seed.py                    # Reproducibility utilities
│   ├── setup_mlflow.py            # MLflow setup
│   └── spiking_neuron.py          # Spiking neuron factory (LIF, IF, …)
└── debug/
    ├── read_nimagenet.py          # N-ImageNet loader sanity check / backend comparison
    └── convert_nimagenet_to_lmdb.py  # Converts N-ImageNet .npz splits to LMDB
```

## Experiment Tracking

EventCLR uses **MLflow** for experiment tracking. By default, experiments are logged under the `"EventCLR"` experiment name. To view results:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser. Each run logs:

- All hyperparameters
- Per-epoch pretraining loss
- Per-epoch train/val loss and accuracy for each evaluation type
- Aggregated mean ± std accuracy across subset runs
- Pretrained backbone weights as artifacts

## Requirements

- Python ≥ 3.12
- PyTorch ≥ 2.9
- PyTorch Lightning ≥ 2.6
- SpikingJelly (activation-based)
- Tonic ≥ 1.6
- MLflow ≥ 3.10
- CUDA-capable GPU with cupy (for the default SpikingJelly backend)

See `requirements.txt` for the full pinned dependency list.

