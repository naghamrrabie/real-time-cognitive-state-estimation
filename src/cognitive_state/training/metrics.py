"""Evaluation metrics with synthetic-label warnings (minimal stub for T054)."""

from __future__ import annotations

import torch
from torch import Tensor

SMOKE_METRIC_WARNING: str = (
    "These metrics use synthetic or placeholder labels and are not "
    "evidence of real cognitive-state predictive accuracy."
)


def compute_mae(predictions: Tensor, targets: Tensor) -> Tensor:
    """Mean Absolute Error per score column.

    Args:
        predictions: Float tensor of shape ``(n_samples, 4)``.
        targets: Float tensor of shape ``(n_samples, 4)``.

    Returns:
        Tensor of shape ``(4,)`` with per-score MAE.
    """
    return (predictions - targets).abs().mean(dim=0)


def compute_rmse(predictions: Tensor, targets: Tensor) -> Tensor:
    """Root Mean Squared Error per score column.

    Args:
        predictions: Float tensor of shape ``(n_samples, 4)``.
        targets: Float tensor of shape ``(n_samples, 4)``.

    Returns:
        Tensor of shape ``(4,)`` with per-score RMSE.
    """
    return ((predictions - targets) ** 2).mean(dim=0).sqrt()


def compute_r2(predictions: Tensor, targets: Tensor) -> Tensor:
    """Coefficient of determination (R²) per score column.

    Returns 1.0 for columns where the target variance is zero (degenerate
    case — constant targets cannot be predicted away from their mean).

    Args:
        predictions: Float tensor of shape ``(n_samples, 4)``.
        targets: Float tensor of shape ``(n_samples, 4)``.

    Returns:
        Tensor of shape ``(4,)`` with per-score R².
    """
    ss_res = ((targets - predictions) ** 2).sum(dim=0)
    ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum(dim=0)
    return torch.where(
        ss_tot == 0,
        torch.ones_like(ss_tot),
        1.0 - ss_res / ss_tot,
    )


def compute_pearson(predictions: Tensor, targets: Tensor) -> Tensor:
    """Pearson correlation coefficient per score column.

    Returns 0.0 for columns with zero variance in either predictions or
    targets.  Returns NaN when fewer than two samples are provided.

    Args:
        predictions: Float tensor of shape ``(n_samples, 4)``.
        targets: Float tensor of shape ``(n_samples, 4)``.

    Returns:
        Tensor of shape ``(4,)`` with per-score Pearson r.
    """
    n = predictions.shape[0]
    if n < 2:
        return torch.full((predictions.shape[1],), float("nan"))

    pred_c = predictions - predictions.mean(dim=0)
    tgt_c = targets - targets.mean(dim=0)
    numerator = (pred_c * tgt_c).sum(dim=0)
    denominator = pred_c.pow(2).sum(dim=0).sqrt() * tgt_c.pow(2).sum(dim=0).sqrt()
    return torch.where(
        denominator == 0,
        torch.zeros_like(numerator),
        numerator / denominator,
    )
