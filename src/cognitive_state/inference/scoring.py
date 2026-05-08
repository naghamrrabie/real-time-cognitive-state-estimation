"""Score bounding and prediction formatting helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from cognitive_state.data.constants import SCORE_MAX, SCORE_MIN, SCORE_NAMES
from cognitive_state.data.schemas import CognitiveScoreVector


def bound_scores(raw: Tensor) -> Tensor:
    """Clamp all values to the valid score range [SCORE_MIN, SCORE_MAX]."""
    return torch.clamp(raw, SCORE_MIN, SCORE_MAX)


def scores_to_vector(scores: Tensor) -> CognitiveScoreVector:
    """Convert a 1-D four-element float tensor to a CognitiveScoreVector.

    The tensor must already be bounded to [0.0, 1.0]; use bound_scores first.
    Order matches SCORE_NAMES: fatigue, attention, stress, engagement.
    """
    values = scores.tolist()
    return CognitiveScoreVector(
        fatigue=float(values[0]),
        attention=float(values[1]),
        stress=float(values[2]),
        engagement=float(values[3]),
    )
