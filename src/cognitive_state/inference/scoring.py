"""Score bounding and prediction formatting helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from cognitive_state.data.constants import SCORE_MAX, SCORE_MIN
from cognitive_state.data.schemas import CognitiveScoreVector, ModelPrediction
from cognitive_state.data.windowing import WindowMetadata


def bound_scores(raw: Tensor) -> Tensor:
    """Clamp all values to the valid score range [SCORE_MIN, SCORE_MAX].

    Provides a safety guard even when the model already applies Sigmoid,
    protecting against rare edge cases and future model replacements.
    """
    return torch.clamp(raw, SCORE_MIN, SCORE_MAX)


def scores_to_vector(scores: Tensor) -> CognitiveScoreVector:
    """Convert a 1-D four-element float tensor to a CognitiveScoreVector.

    The tensor must already be bounded to [0.0, 1.0]; use ``bound_scores``
    first.  Order matches SCORE_NAMES: fatigue, attention, stress, engagement.
    """
    values = scores.tolist()
    return CognitiveScoreVector(
        fatigue=float(values[0]),
        attention=float(values[1]),
        stress=float(values[2]),
        engagement=float(values[3]),
    )


def predictions_from_batch(
    raw_output: Tensor,
    metadata: list[WindowMetadata],
    *,
    model_id: str = "",
    total_windows: int | None = None,
) -> list[ModelPrediction]:
    """Convert batch model output and window metadata to ModelPrediction objects.

    Args:
        raw_output: Float tensor of shape ``(n_windows, 4)`` from a forward pass.
        metadata: Per-window provenance records in window index order.
        model_id: Optional model or checkpoint identifier stored on each result.
        total_windows: Total window count for source_progress formatting.
            Defaults to ``len(metadata)``.

    Returns:
        List of ``ModelPrediction``, one per window, in order.
    """
    bounded = bound_scores(raw_output)
    total = total_windows if total_windows is not None else len(metadata)

    predictions: list[ModelPrediction] = []
    for i, meta in enumerate(metadata):
        scores = scores_to_vector(bounded[i])
        predictions.append(
            ModelPrediction(
                window_id=f"w{meta.window_index}",
                timestamp_seconds=meta.end_time,
                scores=scores,
                source_progress=f"{meta.window_index + 1}/{total}",
                model_id=model_id,
            )
        )
    return predictions
