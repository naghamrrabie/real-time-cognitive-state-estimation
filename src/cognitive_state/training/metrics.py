"""Evaluation metrics with synthetic-label warnings (T054)."""

from __future__ import annotations

import torch
from torch import Tensor

from cognitive_state.data.constants import SCORE_NAMES

SMOKE_METRIC_WARNING: str = (
    "These metrics use synthetic or placeholder labels and are not "
    "evidence of real cognitive-state predictive accuracy."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_metric_inputs(predictions: Tensor, targets: Tensor) -> None:
    """Raise ``ValueError`` for incompatible prediction/target tensors."""
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions shape {tuple(predictions.shape)} does not match "
            f"targets shape {tuple(targets.shape)}."
        )
    if predictions.ndim != 2 or predictions.shape[1] != 4:
        raise ValueError(
            f"Expected tensors of shape (n_samples, 4); "
            f"got {tuple(predictions.shape)}."
        )


# ---------------------------------------------------------------------------
# Per-score scalar metrics
# ---------------------------------------------------------------------------

def compute_mae(predictions: Tensor, targets: Tensor) -> Tensor:
    """Mean Absolute Error per score column.

    Args:
        predictions: Float tensor of shape ``(n_samples, 4)``.
        targets: Float tensor of shape ``(n_samples, 4)``.

    Returns:
        Tensor of shape ``(4,)`` — per-score MAE in order
        ``fatigue, attention, stress, engagement``.

    Raises:
        ValueError: If shapes are incompatible or not ``(n_samples, 4)``.
    """
    _validate_metric_inputs(predictions, targets)
    return (predictions - targets).abs().mean(dim=0)


def compute_rmse(predictions: Tensor, targets: Tensor) -> Tensor:
    """Root Mean Squared Error per score column.

    Args:
        predictions: Float tensor of shape ``(n_samples, 4)``.
        targets: Float tensor of shape ``(n_samples, 4)``.

    Returns:
        Tensor of shape ``(4,)`` — per-score RMSE.

    Raises:
        ValueError: If shapes are incompatible or not ``(n_samples, 4)``.
    """
    _validate_metric_inputs(predictions, targets)
    return ((predictions - targets) ** 2).mean(dim=0).sqrt()


def compute_r2(predictions: Tensor, targets: Tensor) -> Tensor:
    """Coefficient of determination (R²) per score column.

    Returns ``1.0`` for columns where target variance is zero (degenerate
    case — constant targets cannot be predicted away from their mean).

    Args:
        predictions: Float tensor of shape ``(n_samples, 4)``.
        targets: Float tensor of shape ``(n_samples, 4)``.

    Returns:
        Tensor of shape ``(4,)`` — per-score R².

    Raises:
        ValueError: If shapes are incompatible or not ``(n_samples, 4)``.
    """
    _validate_metric_inputs(predictions, targets)
    ss_res = ((targets - predictions) ** 2).sum(dim=0)
    ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum(dim=0)
    return torch.where(
        ss_tot == 0,
        torch.ones_like(ss_tot),
        1.0 - ss_res / ss_tot,
    )


def compute_pearson(predictions: Tensor, targets: Tensor) -> Tensor:
    """Pearson correlation coefficient per score column.

    Returns ``0.0`` for columns with zero variance in either predictions or
    targets.  Returns NaN when fewer than two samples are provided.

    Args:
        predictions: Float tensor of shape ``(n_samples, 4)``.
        targets: Float tensor of shape ``(n_samples, 4)``.

    Returns:
        Tensor of shape ``(4,)`` — per-score Pearson r.

    Raises:
        ValueError: If shapes are incompatible or not ``(n_samples, 4)``.
    """
    _validate_metric_inputs(predictions, targets)
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


# ---------------------------------------------------------------------------
# Per-score structured report
# ---------------------------------------------------------------------------

def compute_per_score_report(
    predictions: Tensor,
    targets: Tensor,
) -> dict[str, dict[str, float]]:
    """Compute MAE, RMSE, and R² for each of the four cognitive scores.

    Args:
        predictions: Float tensor of shape ``(n_samples, 4)``.
        targets: Float tensor of shape ``(n_samples, 4)``.

    Returns:
        Nested dict ``{score_name: {"mae": float, "rmse": float, "r2": float}}``
        in the canonical order ``fatigue, attention, stress, engagement``.

    Raises:
        ValueError: If shapes are incompatible or not ``(n_samples, 4)``.
    """
    mae = compute_mae(predictions, targets)
    rmse = compute_rmse(predictions, targets)
    r2 = compute_r2(predictions, targets)
    return {
        name: {
            "mae": mae[i].item(),
            "rmse": rmse[i].item(),
            "r2": r2[i].item(),
        }
        for i, name in enumerate(SCORE_NAMES)
    }
