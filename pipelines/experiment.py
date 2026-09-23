from typing import Dict, List, Tuple

import mlflow
import numpy as np

from models.backbones.backbone_factory import BackboneFactory
from pipelines.evaluation import EvaluationPipeline
from pipelines.pretrain import PretrainPipeline
from pipelines.utils import print_section
from utils.config import ExperimentConfig
from utils.seed import set_seed
from utils.setup_mlflow import setup_mlflow


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_eval_subset(
        pipeline: EvaluationPipeline,
        backbone,
        n_features: int,
        samples_per_class: float,
        subset_seed: int,
        run_idx: int,
) -> Dict[str, float]:
    """Run one subset evaluation (LP + FT + Supervised) and log to MLflow."""
    eval_types = pipeline.config.eval_types
    results: Dict[str, float] = {}

    with mlflow.start_run(run_name=f"eval_{samples_per_class}_run{run_idx}", nested=True):
        mlflow.log_param("subset_seed", subset_seed)
        mlflow.set_tag("phase", "evaluation")
        mlflow.set_tag("samples_value", str(samples_per_class))

        if "lp" in eval_types:
            with mlflow.start_run(run_name="lp", nested=True):
                mlflow.set_tag("eval_type", "linear_probing")
                lp_acc = pipeline.evaluate_linear_probing(backbone, n_features, samples_per_class, subset_seed)
            mlflow.log_metric("lp_accuracy", lp_acc)
            results["lp"] = lp_acc

        if "ft" in eval_types:
            with mlflow.start_run(run_name="finetune", nested=True):
                mlflow.set_tag("eval_type", "finetuning")
                ft_acc = pipeline.evaluate_finetuning(backbone, n_features, samples_per_class, subset_seed)
            mlflow.log_metric("ft_accuracy", ft_acc)
            results["ft"] = ft_acc

        if "sup" in eval_types:
            with mlflow.start_run(run_name="supervised", nested=True):
                mlflow.set_tag("eval_type", "supervised")
                sup_acc = pipeline.evaluate_supervised(n_features, samples_per_class, subset_seed)
            mlflow.log_metric("sup_accuracy", sup_acc)
            results["sup"] = sup_acc

        if "sup_aug" in eval_types:
            with mlflow.start_run(run_name="supervised_aug", nested=True):
                mlflow.set_tag("eval_type", "supervised_aug")
                sup_aug_acc = pipeline.evaluate_supervised_with_aug(n_features, samples_per_class, subset_seed)
            mlflow.log_metric("sup_aug_accuracy", sup_aug_acc)
            results["sup_aug"] = sup_aug_acc

    return results


def _aggregate_and_log(accs: Dict[str, List[float]], step: int) -> Dict[str, Tuple[float, float]]:
    """Compute mean/std for each eval type and log aggregated metrics to MLflow."""
    stats = {}
    for key, values in accs.items():
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1) if len(values) > 1 else 0.0)
        stats[key] = (mean, std)
        mlflow.log_metric(f"{key}_acc_mean", mean, step=step)
        mlflow.log_metric(f"{key}_acc_std", std, step=step)
    return stats


# ---------------------------------------------------------------------------
# Top-level experiment entry point
# ---------------------------------------------------------------------------

def run_self_supervised_experiment(config: ExperimentConfig) -> None:
    """Run self-supervised pretraining followed by evaluation on the target dataset."""
    set_seed(config.seed)
    setup_mlflow("EventCLR")

    with mlflow.start_run(run_name=config.run_name or f"self_supervised_{config.target_dataset}"):
        mlflow.log_params(vars(config))

        # ---- Phase 1: load or pretrain backbone ----
        if config.pretrained_run_id:
            print_section(
                f"Loading Pretrained Backbone from MLflow\n"
                f"Run ID: {config.pretrained_run_id}"
            )
            pretrained_backbone, n_features = BackboneFactory.load_pretrained_from_mlflow(
                config.pretrained_run_id, config
            )
            mlflow.set_tag("using_pretrained_weights", "True")
        else:
            with mlflow.start_run(run_name="pretraining", nested=True):
                mlflow.set_tag("phase", "pretraining")
                pretrained_backbone, n_features = PretrainPipeline(config).run()
            mlflow.set_tag("using_pretrained_weights", "False")

        # ---- Phase 2: evaluation ----
        print_section(f"Evaluation on Target Dataset: {config.target_dataset}")
        eval_pipeline = EvaluationPipeline(config)

        for samples_per_class in config.samples_per_class_list:
            step = (
                int(samples_per_class * 100)
                if config.samples_mode == 'percent'
                else int(samples_per_class)
            )
            label = (
                f"{samples_per_class * 100:.1f}% samples per class"
                if config.samples_mode == 'percent'
                else f"{int(samples_per_class)} samples per class"
            )
            print(f"\n--- Evaluating with {label} ---")

            accs: Dict[str, List[float]] = {t: [] for t in config.eval_types}

            for run_idx in range(config.n_subset_runs):
                print(f"\n  Subset run {run_idx + 1}/{config.n_subset_runs}")
                subset_seed = config.seed + run_idx * 1000
                result = _run_eval_subset(
                    eval_pipeline, pretrained_backbone, n_features,
                    samples_per_class, subset_seed, run_idx,
                )
                for key in accs:
                    accs[key].append(result[key])

            stats = _aggregate_and_log(accs, step)

            print(f"\n  Results for {label} (mean ± std):")
            label_map = {"lp": "Linear Probing", "ft": "Finetuning", "sup": "Supervised"}
            for key, (mean, std) in stats.items():
                print(f"    {label_map.get(key, key)}: {mean:.4f} ± {std:.4f}")

    print_section("Experiment Complete!")

