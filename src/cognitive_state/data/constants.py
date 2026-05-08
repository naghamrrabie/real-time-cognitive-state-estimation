"""Package-wide shared constants for score names, modality names, and CSV columns."""

from __future__ import annotations

SCORE_NAMES: tuple[str, ...] = ("fatigue", "attention", "stress", "engagement")
SCORE_MIN: float = 0.0
SCORE_MAX: float = 1.0

MODALITY_NAMES: tuple[str, ...] = ("eyes", "face", "head_pose", "posture")

LABEL_SOURCES: tuple[str, ...] = ("synthetic", "placeholder", "real", "missing")
DATASET_SPLITS: tuple[str, ...] = ("train", "validation", "test")

SCORE_CSV_COLUMNS: tuple[str, ...] = (
    "timestamp_seconds",
    "source_progress",
    "fatigue",
    "attention",
    "stress",
    "engagement",
)
