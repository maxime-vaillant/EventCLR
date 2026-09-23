import logging

import torch
import torch.nn.functional as F


def encoder_representation_metrics(H_i: torch.Tensor, H_j: torch.Tensor, epoch: int = None) -> dict:
    """Representation-quality metrics for a pair of encoder outputs (e.g. two augmented views).

    H_i, H_j: [N, D] encoder outputs (already time-averaged if the encoder is spiking).
    Returns a flat dict of metric_name -> float, ready to be logged as-is.
    Rank metrics (stable/effective rank) are omitted if the SVD fails.
    """
    H_i_n = F.normalize(H_i, dim=1)
    H_j_n = F.normalize(H_j, dim=1)

    # Positive pair cosine similarity
    pos_sim = (H_i_n * H_j_n).sum(dim=1)  # [N]

    # Negative cosine similarity (off-diagonal of cross-similarity matrix)
    sim_matrix = H_i_n @ H_j_n.T  # [N, N]
    neg_mask = ~torch.eye(H_i.size(0), dtype=torch.bool, device=H_i.device)
    neg_sim = sim_matrix[neg_mask]

    metrics = {
        "cos_sim_pos_mean": pos_sim.mean().item(),
        "cos_sim_pos_std": pos_sim.std().item(),
        "cos_sim_neg_mean": neg_sim.mean().item(),
    }

    # Representation rank on the full set of encoder outputs
    H_cat = torch.cat([H_i, H_j], dim=0)  # [2N, D]
    H_cat = H_cat - H_cat.mean(dim=0)  # center to remove positive-orthant bias
    H_all = F.normalize(H_cat, dim=1)  # [2N, D]
    try:
        # SVD on CPU: cuSOLVER's handle/workspace alloc can fail here since
        # GPU memory is already saturated by training state at epoch end.
        S = torch.linalg.svdvals(H_all.cpu())
        stable_rank = (S ** 2).sum() / (S[0] ** 2)
        p = (S ** 2) / (S ** 2).sum()
        effective_rank = torch.exp(-(p * torch.log(p + 1e-8)).sum())
        metrics["stable_rank"] = stable_rank.item()
        metrics["effective_rank"] = effective_rank.item()
    except Exception as e:
        logging.warning(f"Skipping rank metrics{f' at epoch {epoch}' if epoch is not None else ''}: {e}")

    # Firing rate per neuron (H values are rate-coded: avg spikes over T)
    H_raw = torch.cat([H_i, H_j], dim=0)  # [2N, D]
    neuron_rates = H_raw.mean(dim=0)  # [D] mean rate per neuron
    metrics["firing_rate_mean"] = neuron_rates.mean().item()
    metrics["firing_rate_std"] = neuron_rates.std().item()
    metrics["dead_neuron_frac"] = (neuron_rates < 0.05).float().mean().item()
    metrics["saturated_neuron_frac"] = (neuron_rates > 0.95).float().mean().item()

    return metrics


class TopKAccuracy:
    def __init__(self, k: int = 1):
        self.k = k

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        top_k = torch.topk(y_pred, self.k, dim=1).indices
        correct = top_k.eq(y_true.unsqueeze(1)).any(dim=1)

        return correct.float().mean().item()

    def __str__(self):
        return f'top_{self.k}_acc'