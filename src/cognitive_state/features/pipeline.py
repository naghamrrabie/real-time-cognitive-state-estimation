"""Per-frame multimodal feature dictionary assembly."""

from __future__ import annotations

from cognitive_state.features.eyes import EYE_FEATURE_NAMES, extract_eye_features
from cognitive_state.features.face import FACE_FEATURE_NAMES, extract_face_features
from cognitive_state.features.landmarks import LandmarkSet
from cognitive_state.features.posture import (
    HEAD_POSTURE_FEATURE_NAMES,
    extract_head_posture_features,
)

FRAME_METADATA_NAMES = ("frame_index", "timestamp_seconds")
FRAME_FEATURE_NAMES = (
    *FRAME_METADATA_NAMES,
    *EYE_FEATURE_NAMES,
    *FACE_FEATURE_NAMES,
    *HEAD_POSTURE_FEATURE_NAMES,
)


def extract_frame_features(
    landmarks: LandmarkSet,
    *,
    previous_landmarks: LandmarkSet | None = None,
) -> dict[str, float | int]:
    """Return one stable numeric feature dictionary for a frame."""

    features: dict[str, float | int] = {
        "frame_index": landmarks.frame_index,
        "timestamp_seconds": float(landmarks.timestamp_seconds),
    }
    features.update(extract_eye_features(landmarks))
    features.update(extract_face_features(landmarks))
    features.update(
        extract_head_posture_features(
            landmarks,
            previous_landmarks=previous_landmarks,
        )
    )
    return features
