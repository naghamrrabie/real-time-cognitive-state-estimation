"""End-to-end inference pipeline: feature rows → windows → model → predictions."""

from __future__ import annotations

from pathlib import Path

import torch

from cognitive_state.data.feature_csv import read_feature_csv
from cognitive_state.data.schemas import ModelPrediction
from cognitive_state.data.windowing import FeatureWindowBatch, build_windows
from cognitive_state.inference.scoring import predictions_from_batch
from cognitive_state.models.transformer import TemporalTransformer

DEFAULT_WINDOW_SIZE: int = 30
DEFAULT_STRIDE: int = 15


def load_model(
    input_features: int,
    checkpoint_path: str | Path | None = None,
    **model_kwargs: object,
) -> TemporalTransformer:
    """Create or restore a TemporalTransformer for inference.

    Args:
        input_features: Number of feature columns per timestep.  Must match
            the feature dimension of the windows that will be passed to the
            model.
        checkpoint_path: Optional path to a ``state_dict`` checkpoint saved
            with ``torch.save(model.state_dict(), path)``.  When ``None`` a
            fresh model with random weights is returned.
        **model_kwargs: Extra keyword arguments forwarded to
            ``TemporalTransformer`` (e.g. ``d_model``, ``nhead``,
            ``num_layers``).

    Returns:
        A ``TemporalTransformer`` in eval mode.
    """
    model = TemporalTransformer(input_features=input_features, **model_kwargs)
    if checkpoint_path is not None:
        data = torch.load(Path(checkpoint_path), map_location="cpu")
        # Support both wrapped format {"model_state_dict": ...} and bare state_dict
        if isinstance(data, dict) and "model_state_dict" in data:
            data = data["model_state_dict"]
        model.load_state_dict(data)
    model.eval()
    return model


def run_inference(
    rows: list[dict[str, float | int]],
    model: TemporalTransformer,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    model_id: str = "",
) -> list[ModelPrediction]:
    """Run inference on an ordered sequence of per-frame feature rows.

    Args:
        rows: Per-frame feature records from ``extract_frame_features`` or
              ``read_feature_csv``.
        model: A ``TemporalTransformer`` instance.  Automatically switched to
               eval mode during inference.
        window_size: Number of frames per temporal window.
        stride: Frame advance between consecutive windows.
        model_id: Optional identifier written to each ``ModelPrediction``.

    Returns:
        List of ``ModelPrediction`` objects, one per temporal window, in order.
        Returns an empty list when the rows cannot fill a single window.

    Raises:
        WindowShapeError: If ``window_size < 1`` or ``stride < 1``.
        ValidationError: If required feature columns are absent from the rows.
    """
    batch: FeatureWindowBatch = build_windows(
        rows, window_size=window_size, stride=stride
    )
    if batch.n_windows == 0:
        return []

    features_tensor = torch.from_numpy(batch.features)  # (n_windows, ws, dim)
    model.eval()
    with torch.no_grad():
        raw_output = model(features_tensor)  # (n_windows, 4)

    return predictions_from_batch(
        raw_output,
        batch.metadata,
        model_id=model_id,
        total_windows=batch.n_windows,
    )


def run_inference_from_csv(
    csv_path: str | Path,
    model: TemporalTransformer,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    model_id: str = "",
) -> list[ModelPrediction]:
    """Run inference on feature rows loaded from a CSV file.

    Args:
        csv_path: Path to a feature CSV written by ``write_feature_csv``.
        model: A ``TemporalTransformer`` instance.
        window_size: Number of frames per temporal window.
        stride: Frame advance between consecutive windows.
        model_id: Optional identifier written to each ``ModelPrediction``.

    Returns:
        List of ``ModelPrediction`` objects, one per temporal window.
    """
    rows = read_feature_csv(csv_path)
    return run_inference(
        rows, model, window_size=window_size, stride=stride, model_id=model_id
    )
