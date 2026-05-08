"""Training scaffold package."""

from cognitive_state.training.batching import collate_batch, collate_fn
from cognitive_state.training.checkpoints import load_checkpoint, save_checkpoint
from cognitive_state.training.metrics import (
    SMOKE_METRIC_WARNING,
    compute_mae,
    compute_pearson,
    compute_per_score_report,
    compute_r2,
    compute_rmse,
)
from cognitive_state.training.smoke_train import SmokeTrainResult, run_smoke_train

__all__ = [
    "SMOKE_METRIC_WARNING",
    "SmokeTrainResult",
    "collate_batch",
    "collate_fn",
    "compute_mae",
    "compute_pearson",
    "compute_per_score_report",
    "compute_r2",
    "compute_rmse",
    "load_checkpoint",
    "run_smoke_train",
    "save_checkpoint",
]
