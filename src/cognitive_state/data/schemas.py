"""Cognitive score and prediction dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

from cognitive_state.data.constants import SCORE_NAMES
from cognitive_state.data.validation import validate_score_vector


@dataclass(frozen=True)
class CognitiveScoreVector:
    """Four bounded continuous cognitive state scores, each in [0.0, 1.0]."""

    fatigue: float
    attention: float
    stress: float
    engagement: float

    def __post_init__(self) -> None:
        validate_score_vector(self)

    def as_dict(self) -> dict[str, float]:
        """Return all four scores keyed by canonical score name."""
        return {name: getattr(self, name) for name in SCORE_NAMES}


@dataclass
class ModelPrediction:
    """One model output for one feature window."""

    window_id: str
    timestamp_seconds: float
    scores: CognitiveScoreVector
    source_progress: str = field(default="")
    model_id: str = field(default="")
