"""Domain-specific exceptions for validation and data contract errors."""

from __future__ import annotations


class CognitiveStateError(Exception):
    """Base exception for all cognitive state estimation errors."""


class ValidationError(CognitiveStateError):
    """Raised when a value fails a validation check."""


class ScoreRangeError(ValidationError):
    """Raised when a score value is outside the valid [0.0, 1.0] range."""


class FiniteValueError(ValidationError):
    """Raised when a value is not finite (NaN or infinity)."""


class ModalityError(CognitiveStateError):
    """Raised when a modality name is invalid or not recognised."""


class WindowShapeError(CognitiveStateError):
    """Raised when a feature window has an invalid shape."""


class LabelError(CognitiveStateError):
    """Raised when a dataset label is missing, mismatched, or out of range."""
