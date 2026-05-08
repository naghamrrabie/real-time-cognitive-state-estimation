"""Head and posture proxy features from face and pose landmarks.

Single-frame head center values are always clean numeric values when face
landmarks exist. Movement and micro-jitter are non-zero only when a previous
frame's landmarks are supplied.
"""

from __future__ import annotations

from math import dist

from cognitive_state.features.landmarks import LandmarkPoint, LandmarkSet

HEAD_CENTER_INDICES = (1, 33, 263, 61, 291)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

HEAD_POSTURE_FEATURE_NAMES = (
    "head_center_x",
    "head_center_y",
    "head_center_z",
    "head_movement",
    "head_micro_jitter",
    "head_missing",
    "posture_shoulder_drop",
    "posture_shoulder_slope",
    "posture_shoulder_width",
    "posture_torso_center_x",
    "posture_torso_center_y",
    "posture_torso_length",
    "posture_missing",
)


def extract_head_posture_features(
    landmarks: LandmarkSet,
    *,
    previous_landmarks: LandmarkSet | None = None,
) -> dict[str, float]:
    """Compute per-frame head and posture proxy features."""

    features = {}
    features.update(_extract_head_features(landmarks, previous_landmarks=previous_landmarks))
    features.update(_extract_posture_features(landmarks))
    return features


def _extract_head_features(
    landmarks: LandmarkSet,
    *,
    previous_landmarks: LandmarkSet | None,
) -> dict[str, float]:
    center = _head_center(landmarks.face_landmarks)
    if center is None:
        return {
            "head_center_x": 0.0,
            "head_center_y": 0.0,
            "head_center_z": 0.0,
            "head_movement": 0.0,
            "head_micro_jitter": 0.0,
            "head_missing": 1.0,
        }

    previous_center = None
    if previous_landmarks is not None:
        previous_center = _head_center(previous_landmarks.face_landmarks)
    movement = _distance_3d(center, previous_center) if previous_center else 0.0

    return {
        "head_center_x": _rounded(center[0]),
        "head_center_y": _rounded(center[1]),
        "head_center_z": _rounded(center[2]),
        "head_movement": _rounded(movement),
        "head_micro_jitter": _rounded(movement),
        "head_missing": 0.0,
    }


def _extract_posture_features(landmarks: LandmarkSet) -> dict[str, float]:
    pose_landmarks = landmarks.pose_landmarks
    required = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    if not _has_indices(pose_landmarks, required):
        return {
            "posture_shoulder_drop": 0.0,
            "posture_shoulder_slope": 0.0,
            "posture_shoulder_width": 0.0,
            "posture_torso_center_x": 0.0,
            "posture_torso_center_y": 0.0,
            "posture_torso_length": 0.0,
            "posture_missing": 1.0,
        }

    left_shoulder = pose_landmarks[LEFT_SHOULDER]
    right_shoulder = pose_landmarks[RIGHT_SHOULDER]
    left_hip = pose_landmarks[LEFT_HIP]
    right_hip = pose_landmarks[RIGHT_HIP]
    shoulder_midpoint = _midpoint(left_shoulder, right_shoulder)
    hip_midpoint = _midpoint(left_hip, right_hip)
    shoulder_slope = right_shoulder.y - left_shoulder.y

    return {
        "posture_shoulder_drop": _rounded(abs(shoulder_slope)),
        "posture_shoulder_slope": _rounded(shoulder_slope),
        "posture_shoulder_width": _rounded(_distance_2d(left_shoulder, right_shoulder)),
        "posture_torso_center_x": _rounded((shoulder_midpoint[0] + hip_midpoint[0]) / 2.0),
        "posture_torso_center_y": _rounded((shoulder_midpoint[1] + hip_midpoint[1]) / 2.0),
        "posture_torso_length": _rounded(dist(shoulder_midpoint, hip_midpoint)),
        "posture_missing": 0.0,
    }


def _head_center(
    face_landmarks: tuple[LandmarkPoint, ...] | None,
) -> tuple[float, float, float] | None:
    if not _has_indices(face_landmarks, HEAD_CENTER_INDICES):
        return None
    points = [face_landmarks[index] for index in HEAD_CENTER_INDICES]
    return (
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
        sum(point.z for point in points) / len(points),
    )


def _midpoint(a: LandmarkPoint, b: LandmarkPoint) -> tuple[float, float]:
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def _distance_2d(a: LandmarkPoint, b: LandmarkPoint) -> float:
    return dist((a.x, a.y), (b.x, b.y))


def _distance_3d(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    return dist(a, b)


def _has_indices(
    landmarks: tuple[LandmarkPoint, ...] | None,
    indices: tuple[int, ...],
) -> bool:
    if landmarks is None:
        return False
    return all(index < len(landmarks) for index in indices)


def _rounded(value: float) -> float:
    return round(float(value), 6)
