"""Validation helpers for finite values, score ranges, and modality names."""

from __future__ import annotations

import math
from typing import Any

from cognitive_state.data.constants import MODALITY_NAMES, SCORE_MAX, SCORE_MIN, SCORE_NAMES
from cognitive_state.data.exceptions import (
    FiniteValueError,
    ModalityError,
    ScoreRangeError,
    ValidationError,
    WindowShapeError,
)


def validate_finite(name: str, value: float) -> None:
    """Raise FiniteValueError if value is NaN or infinite."""
    if not math.isfinite(value):
        raise FiniteValueError(
            f"{name!r} must be a finite number; got {value!r}"
        )


def validate_score_range(name: str, value: float) -> None:
    """Raise ScoreRangeError if value is outside [SCORE_MIN, SCORE_MAX]."""
    if value < SCORE_MIN or value > SCORE_MAX:
        raise ScoreRangeError(
            f"{name!r} must be in [{SCORE_MIN}, {SCORE_MAX}]; got {value!r}"
        )


def validate_score_value(name: str, value: float) -> None:
    """Validate that a single named score is finite and in range."""
    validate_finite(name, value)
    validate_score_range(name, value)


def validate_score_vector(vector: Any) -> None:
    """Validate all four required scores on a CognitiveScoreVector-like object."""
    for name in SCORE_NAMES:
        value = getattr(vector, name)
        validate_score_value(name, value)


def validate_score_names(names: tuple[str, ...] | list[str]) -> None:
    """Raise ValidationError if any required score name is absent from names."""
    missing = [n for n in SCORE_NAMES if n not in names]
    if missing:
        raise ValidationError(
            f"Missing required score names: {missing}"
        )


def validate_modality_name(name: str) -> None:
    """Raise ModalityError if name is not a recognised modality."""
    if name not in MODALITY_NAMES:
        raise ModalityError(
            f"Unknown modality {name!r}; expected one of {MODALITY_NAMES}"
        )


def validate_window_shape(shape: tuple[int, ...], expected_features: int) -> None:
    """Raise WindowShapeError if the 2-D window shape does not match expectations."""
    if len(shape) != 2:
        raise WindowShapeError(
            f"Feature window must be 2-D (timesteps, features); got shape {shape}"
        )
    timesteps, features = shape
    if timesteps < 1:
        raise WindowShapeError(
            f"Feature window must have at least 1 timestep; got {timesteps}"
        )
    if features != expected_features:
        raise WindowShapeError(
            f"Feature window expected {expected_features} features per step; got {features}"
        )
