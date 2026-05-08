"""Synthetic test data helpers for frames, landmarks, feature windows, and labels.

Used by windowing, dataset, model shape, and training tests. All helpers return
deterministic, minimal objects suitable for unit and integration tests.
"""

from __future__ import annotations

import numpy as np

from cognitive_state.data.constants import MODALITY_NAMES, SCORE_NAMES
from cognitive_state.data.schemas import CognitiveScoreVector
from cognitive_state.features.landmarks import LandmarkSet


def make_rgb_frame(
    *,
    height: int = 4,
    width: int = 4,
    channels: int = 3,
    value: int = 0,
) -> np.ndarray:
    """Return a small solid-colour uint8 RGB array."""
    return np.full((height, width, channels), fill_value=value, dtype=np.uint8)


def make_landmark_set(
    *,
    frame_index: int = 0,
    timestamp_seconds: float = 0.0,
    face_detected: bool = True,
    pose_detected: bool = True,
) -> LandmarkSet:
    """Return a LandmarkSet with no actual landmark coordinates."""
    return LandmarkSet(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        face_landmarks=None,
        pose_landmarks=None,
        face_detected=face_detected,
        pose_detected=pose_detected,
    )


def make_score_vector(
    *,
    fatigue: float = 0.5,
    attention: float = 0.5,
    stress: float = 0.5,
    engagement: float = 0.5,
) -> CognitiveScoreVector:
    """Return a CognitiveScoreVector with default mid-range scores."""
    return CognitiveScoreVector(
        fatigue=fatigue,
        attention=attention,
        stress=stress,
        engagement=engagement,
    )


def make_feature_window(
    *,
    timesteps: int = 30,
    features_per_step: int = 8,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Return a synthetic modality feature window keyed by modality name.

    Each value is a float32 array of shape (timesteps, features_per_step).
    Suitable for windowing, dataset, and model shape tests.
    """
    rng = np.random.default_rng(seed=seed)
    return {
        modality: rng.random((timesteps, features_per_step)).astype(np.float32)
        for modality in MODALITY_NAMES
    }


def make_label_sequence(
    *,
    count: int = 8,
    seed: int = 0,
) -> list[CognitiveScoreVector]:
    """Return a list of random CognitiveScoreVectors with values in [0.0, 1.0]."""
    rng = np.random.default_rng(seed=seed)
    result: list[CognitiveScoreVector] = []
    for _ in range(count):
        values = rng.random(len(SCORE_NAMES)).tolist()
        result.append(
            CognitiveScoreVector(
                fatigue=values[0],
                attention=values[1],
                stress=values[2],
                engagement=values[3],
            )
        )
    return result
