"""Eye feature calculations from Face Mesh landmarks.

Feature names are stable MVP keys. Missing landmark values are emitted as
``0.0`` with ``eye_missing`` set to ``1.0`` so downstream arrays stay numeric.
"""

from __future__ import annotations

from math import dist

from cognitive_state.features.landmarks import LandmarkPoint, LandmarkSet

LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)
EAR_BLINK_THRESHOLD = 0.25

EYE_FEATURE_NAMES = (
    "eye_left_ear",
    "eye_right_ear",
    "eye_mean_ear",
    "eye_ear_asymmetry",
    "eye_blink_proxy",
    "eye_missing",
)


def extract_eye_features(landmarks: LandmarkSet) -> dict[str, float]:
    """Compute per-frame eye features from face landmarks."""

    face_landmarks = landmarks.face_landmarks
    if not _has_indices(face_landmarks, LEFT_EYE_INDICES + RIGHT_EYE_INDICES):
        return _missing_eye_features()

    left_ear = _eye_aspect_ratio(face_landmarks, LEFT_EYE_INDICES)
    right_ear = _eye_aspect_ratio(face_landmarks, RIGHT_EYE_INDICES)
    mean_ear = (left_ear + right_ear) / 2.0
    blink_proxy = max(0.0, (EAR_BLINK_THRESHOLD - mean_ear) / EAR_BLINK_THRESHOLD)

    return {
        "eye_left_ear": _rounded(left_ear),
        "eye_right_ear": _rounded(right_ear),
        "eye_mean_ear": _rounded(mean_ear),
        "eye_ear_asymmetry": _rounded(abs(left_ear - right_ear)),
        "eye_blink_proxy": _rounded(blink_proxy),
        "eye_missing": 0.0,
    }


def _missing_eye_features() -> dict[str, float]:
    return {
        "eye_left_ear": 0.0,
        "eye_right_ear": 0.0,
        "eye_mean_ear": 0.0,
        "eye_ear_asymmetry": 0.0,
        "eye_blink_proxy": 0.0,
        "eye_missing": 1.0,
    }


def _eye_aspect_ratio(
    landmarks: tuple[LandmarkPoint, ...],
    indices: tuple[int, int, int, int, int, int],
) -> float:
    p1, p2, p3, p4, p5, p6 = (landmarks[index] for index in indices)
    horizontal = _distance_2d(p1, p4)
    if horizontal == 0:
        return 0.0
    vertical = _distance_2d(p2, p6) + _distance_2d(p3, p5)
    return vertical / (2.0 * horizontal)


def _distance_2d(a: LandmarkPoint, b: LandmarkPoint) -> float:
    return dist((a.x, a.y), (b.x, b.y))


def _has_indices(
    landmarks: tuple[LandmarkPoint, ...] | None,
    indices: tuple[int, ...],
) -> bool:
    if landmarks is None:
        return False
    return all(index < len(landmarks) for index in indices)


def _rounded(value: float) -> float:
    return round(float(value), 6)
