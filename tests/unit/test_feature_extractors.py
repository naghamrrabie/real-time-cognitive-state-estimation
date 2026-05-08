from cognitive_state.features import (
    FRAME_FEATURE_NAMES,
    LandmarkPoint,
    LandmarkSet,
    extract_eye_features,
    extract_face_features,
    extract_frame_features,
    extract_head_posture_features,
)


def _points(size: int, values: dict[int, tuple[float, float, float]]) -> tuple[LandmarkPoint, ...]:
    points = [LandmarkPoint(0.0, 0.0, 0.0) for _ in range(size)]
    for index, value in values.items():
        points[index] = LandmarkPoint(*value)
    return tuple(points)


def _landmark_set(
    *,
    face_landmarks: tuple[LandmarkPoint, ...] | None = None,
    pose_landmarks: tuple[LandmarkPoint, ...] | None = None,
    frame_index: int = 3,
    timestamp_seconds: float = 1.5,
) -> LandmarkSet:
    return LandmarkSet(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        face_landmarks=face_landmarks,
        pose_landmarks=pose_landmarks,
        face_detected=face_landmarks is not None,
        pose_detected=pose_landmarks is not None,
    )


def _face_points() -> tuple[LandmarkPoint, ...]:
    return _points(
        468,
        {
            1: (0.50, 0.40, 0.00),
            13: (0.50, 0.62, 0.00),
            14: (0.50, 0.70, 0.00),
            33: (0.20, 0.30, 0.00),
            61: (0.40, 0.66, 0.00),
            105: (0.24, 0.22, 0.00),
            133: (0.40, 0.30, 0.00),
            144: (0.25, 0.34, 0.00),
            153: (0.35, 0.34, 0.00),
            158: (0.35, 0.26, 0.00),
            160: (0.25, 0.26, 0.00),
            234: (0.15, 0.55, 0.00),
            263: (0.80, 0.30, 0.00),
            291: (0.60, 0.66, 0.00),
            334: (0.76, 0.22, 0.00),
            362: (0.60, 0.30, 0.00),
            373: (0.75, 0.34, 0.00),
            380: (0.65, 0.34, 0.00),
            385: (0.65, 0.26, 0.00),
            386: (0.74, 0.26, 0.00),
            387: (0.75, 0.26, 0.00),
            454: (0.85, 0.55, 0.00),
        },
    )


def _pose_points() -> tuple[LandmarkPoint, ...]:
    return _points(
        33,
        {
            11: (0.35, 0.40, 0.00, 0.95),
            12: (0.65, 0.46, 0.00, 0.95),
            23: (0.40, 0.78, 0.00, 0.90),
            24: (0.60, 0.80, 0.00, 0.90),
        },
    )


def test_eye_features_compute_ear_and_blink_proxy() -> None:
    features = extract_eye_features(_landmark_set(face_landmarks=_face_points()))

    assert features["eye_left_ear"] == 0.4
    assert features["eye_right_ear"] == 0.4
    assert features["eye_mean_ear"] == 0.4
    assert features["eye_ear_asymmetry"] == 0.0
    assert features["eye_blink_proxy"] == 0.0
    assert features["eye_missing"] == 0.0


def test_eye_features_mark_missing_face_landmarks() -> None:
    features = extract_eye_features(_landmark_set())

    assert features["eye_missing"] == 1.0
    assert features["eye_mean_ear"] == 0.0
    assert features["eye_blink_proxy"] == 0.0


def test_face_features_compute_mouth_and_distance_proxies() -> None:
    features = extract_face_features(_landmark_set(face_landmarks=_face_points()))

    assert features["face_mouth_openness"] == 0.4
    assert features["face_mouth_width"] == 0.2
    assert features["face_jaw_width"] == 0.7
    assert features["face_nose_to_mouth"] == 0.22
    assert features["face_missing"] == 0.0


def test_head_and_posture_features_use_pose_and_previous_head_center() -> None:
    previous = _landmark_set(
        face_landmarks=_points(
            468,
            {
                1: (0.45, 0.40, 0.00),
                33: (0.15, 0.30, 0.00),
                61: (0.35, 0.66, 0.00),
                263: (0.75, 0.30, 0.00),
                291: (0.55, 0.66, 0.00),
            },
        ),
        pose_landmarks=_pose_points(),
    )
    current = _landmark_set(
        face_landmarks=_face_points(),
        pose_landmarks=_pose_points(),
    )

    features = extract_head_posture_features(current, previous_landmarks=previous)

    assert features["head_center_x"] == 0.5
    assert features["head_center_y"] == 0.464
    assert features["head_movement"] == 0.05
    assert features["head_micro_jitter"] == 0.05
    assert features["posture_shoulder_drop"] == 0.06
    assert features["posture_shoulder_width"] == 0.305941
    assert features["posture_torso_length"] == 0.36
    assert features["posture_missing"] == 0.0


def test_frame_feature_dictionary_contains_stable_numeric_missing_safe_values() -> None:
    features = extract_frame_features(_landmark_set())

    assert features["frame_index"] == 3
    assert features["timestamp_seconds"] == 1.5
    assert features["eye_missing"] == 1.0
    assert features["face_missing"] == 1.0
    assert features["head_missing"] == 1.0
    assert features["posture_missing"] == 1.0
    assert all(isinstance(value, (float, int)) for value in features.values())
    assert tuple(features) == FRAME_FEATURE_NAMES
