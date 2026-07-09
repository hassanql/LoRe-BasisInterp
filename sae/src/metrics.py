"""SAE reconstruction and LoRe preservation metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_mse(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


def explained_variance(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    residual_var = torch.var(x - x_hat)
    total_var = torch.var(x)
    return 1.0 - residual_var / total_var.clamp_min(1e-12)


def active_feature_counts(z: torch.Tensor) -> torch.Tensor:
    return (z != 0).sum(dim=-1)


def pearson_corr_by_column(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    numerator = (x_centered * y_centered).sum(dim=0)
    denominator = torch.linalg.norm(x_centered, dim=0) * torch.linalg.norm(y_centered, dim=0)
    return numerator / denominator.clamp_min(1e-12)


def rank_columns(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x, dim=0)
    ranks = torch.empty_like(order, dtype=torch.float32)
    base = torch.arange(x.shape[0], device=x.device, dtype=torch.float32).unsqueeze(1).expand_as(ranks)
    ranks.scatter_(0, order, base)
    return ranks


def spearman_corr_by_column(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return pearson_corr_by_column(rank_columns(x), rank_columns(y))
