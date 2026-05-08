"""Face geometry features from Face Mesh landmarks.

The MVP uses interpretable landmark-distance proxies only. Missing values are
numeric ``0.0`` with ``face_missing`` set to ``1.0``.
"""

from __future__ import annotations

from math import dist

from cognitive_state.features.landmarks import LandmarkPoint, LandmarkSet

FACE_FEATURE_NAMES = (
    "face_mouth_openness",
    "face_mouth_width",
    "face_jaw_width",
    "face_nose_to_mouth",
    "face_left_brow_eye_distance",
    "face_right_brow_eye_distance",
    "face_missing",
)

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
JAW_LEFT = 234
JAW_RIGHT = 454
NOSE_TIP = 1
LEFT_BROW = 105
LEFT_UPPER_EYE = 159
RIGHT_BROW = 334
RIGHT_UPPER_EYE = 386


def extract_face_features(landmarks: LandmarkSet) -> dict[str, float]:
    """Compute per-frame face expression proxy features."""

    face_landmarks = landmarks.face_landmarks
    required = (
        MOUTH_TOP,
        MOUTH_BOTTOM,
        MOUTH_LEFT,
        MOUTH_RIGHT,
        JAW_LEFT,
        JAW_RIGHT,
        NOSE_TIP,
        LEFT_BROW,
        LEFT_UPPER_EYE,
        RIGHT_BROW,
        RIGHT_UPPER_EYE,
    )
    if not _has_indices(face_landmarks, required):
        return _missing_face_features()

    mouth_width = _distance_2d(face_landmarks[MOUTH_LEFT], face_landmarks[MOUTH_RIGHT])
    mouth_gap = _distance_2d(face_landmarks[MOUTH_TOP], face_landmarks[MOUTH_BOTTOM])
    mouth_openness = _ratio(mouth_gap, mouth_width)

    return {
        "face_mouth_openness": _rounded(mouth_openness),
        "face_mouth_width": _rounded(mouth_width),
        "face_jaw_width": _rounded(_distance_2d(face_landmarks[JAW_LEFT], face_landmarks[JAW_RIGHT])),
        "face_nose_to_mouth": _rounded(_distance_2d(face_landmarks[NOSE_TIP], face_landmarks[MOUTH_TOP])),
        "face_left_brow_eye_distance": _rounded(
            _distance_2d(face_landmarks[LEFT_BROW], face_landmarks[LEFT_UPPER_EYE])
        ),
        "face_right_brow_eye_distance": _rounded(
            _distance_2d(face_landmarks[RIGHT_BROW], face_landmarks[RIGHT_UPPER_EYE])
        ),
        "face_missing": 0.0,
    }


def _missing_face_features() -> dict[str, float]:
    return {
        "face_mouth_openness": 0.0,
        "face_mouth_width": 0.0,
        "face_jaw_width": 0.0,
        "face_nose_to_mouth": 0.0,
        "face_left_brow_eye_distance": 0.0,
        "face_right_brow_eye_distance": 0.0,
        "face_missing": 1.0,
    }


def _distance_2d(a: LandmarkPoint, b: LandmarkPoint) -> float:
    return dist((a.x, a.y), (b.x, b.y))


def _has_indices(
    landmarks: tuple[LandmarkPoint, ...] | None,
    indices: tuple[int, ...],
) -> bool:
    if landmarks is None:
        return False
    return all(index < len(landmarks) for index in indices)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _rounded(value: float) -> float:
    return round(float(value), 6)
