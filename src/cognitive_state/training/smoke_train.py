"""Smoke-training loop for Temporal Transformer regression (T053)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from cognitive_state.data.dataset import WindowDataset
from cognitive_state.data.synthetic_labels import (
    PLACEHOLDER_LABEL_SOURCE,
    SYNTHETIC_LABEL_SOURCE,
)
from cognitive_state.data.windowing import WINDOW_FEATURE_COLUMNS
from cognitive_state.models.transformer import TemporalTransformer
from cognitive_state.training.batching import collate_batch
from cognitive_state.training.metrics import (
    SMOKE_METRIC_WARNING,
    compute_mae,
    compute_rmse,
)

if TYPE_CHECKING:
    pass

_SMOKE_LABEL_SOURCES: frozenset[str] = frozenset(
    {SYNTHETIC_LABEL_SOURCE, PLACEHOLDER_LABEL_SOURCE}
)


@dataclass
class SmokeTrainResult:
    """Structured outcome of a smoke-training run.

    Attributes:
        final_loss: MSE loss value recorded on the last optimisation step.
        epochs: Number of epochs completed.
        n_samples: Number of training windows.
        label_source: Source tag of the labels used (e.g. ``"synthetic"``).
        warning: Non-empty provenance disclaimer when synthetic/placeholder
            labels are used; empty string otherwise.
        mae: Per-score MAE tensor of shape ``(4,)`` evaluated after training.
        rmse: Per-score RMSE tensor of shape ``(4,)`` evaluated after training.
        input_shape: Shape of the feature batch tensor ``(n, window_size, feature_dim)``.
        output_shape: Shape of the prediction tensor ``(n, 4)``.
        model: The trained ``TemporalTransformer`` (excluded from repr).
    """

    final_loss: float
    epochs: int
    n_samples: int
    label_source: str
    warning: str
    mae: Tensor
    rmse: Tensor
    input_shape: tuple[int, ...] = field(default_factory=tuple)
    output_shape: tuple[int, ...] = field(default_factory=tuple)
    model: nn.Module | None = field(default=None, repr=False)


def run_smoke_train(
    dataset: WindowDataset,
    *,
    input_features: int | None = None,
    epochs: int = 1,
    lr: float = 1e-3,
    seed: int = 0,
) -> SmokeTrainResult:
    """Run a minimal smoke-training pass to validate gradient flow and shapes.

    The entire dataset is treated as a single batch (CPU-friendly).  Real
    backpropagation is performed so that weight updates and loss reduction are
    exercisable, but no accuracy claim should be drawn from results.

    Args:
        dataset: Feature windows paired with labels.
        input_features: Feature dimension per timestep.  Defaults to
            ``len(WINDOW_FEATURE_COLUMNS)`` (the standard 28-column set).
        epochs: Number of gradient-descent steps (default 1).
        lr: Adam learning rate (default 1e-3).
        seed: Manual seed for reproducibility.

    Returns:
        :class:`SmokeTrainResult` with loss, epoch count, shapes,
        per-score MAE/RMSE, and the trained model.
    """
    torch.manual_seed(seed)

    if input_features is None:
        input_features = len(WINDOW_FEATURE_COLUMNS)

    samples = list(dataset)
    x, y = collate_batch(samples)  # (n, window_size, features), (n, 4)

    model = TemporalTransformer(input_features=input_features)
    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    final_loss: float = float("nan")
    last_pred: Tensor | None = None

    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        last_pred = pred.detach()

    model.eval()
    if last_pred is None:
        with torch.no_grad():
            last_pred = model(x)

    mae = compute_mae(last_pred, y)
    rmse = compute_rmse(last_pred, y)

    warning = (
        SMOKE_METRIC_WARNING
        if dataset.label_source in _SMOKE_LABEL_SOURCES
        else ""
    )

    return SmokeTrainResult(
        final_loss=final_loss,
        epochs=epochs,
        n_samples=len(dataset),
        label_source=dataset.label_source,
        warning=warning,
        mae=mae,
        rmse=rmse,
        input_shape=tuple(x.shape),
        output_shape=tuple(last_pred.shape),
        model=model,
    )
