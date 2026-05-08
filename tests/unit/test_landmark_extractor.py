from types import SimpleNamespace

import numpy as np

from cognitive_state.features import (
    LandmarkPoint,
    MediaPipeLandmarkExtractor,
)
from cognitive_state.video import VideoFrame


class FakeSolution:
    def __init__(self, result: object) -> None:
        self.result = result
        self.processed_images: list[np.ndarray] = []
        self.closed = False

    def process(self, image: np.ndarray) -> object:
        self.processed_images.append(image.copy())
        return self.result

    def close(self) -> None:
        self.closed = True


def _landmark(
    x: float,
    y: float,
    z: float = 0.0,
    visibility: float | None = None,
    presence: float | None = None,
) -> SimpleNamespace:
    values: dict[str, float] = {"x": x, "y": y, "z": z}
    if visibility is not None:
        values["visibility"] = visibility
    if presence is not None:
        values["presence"] = presence
    return SimpleNamespace(**values)


def _landmark_list(*points: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(landmark=list(points))


def test_extractor_returns_face_landmarks_when_detected() -> None:
    face_result = SimpleNamespace(
        multi_face_landmarks=[
            _landmark_list(
                _landmark(0.1, 0.2, 0.3),
                _landmark(0.4, 0.5, 0.6),
            )
        ]
    )
    pose_result = SimpleNamespace(pose_landmarks=None)
    face_solution = FakeSolution(face_result)
    pose_solution = FakeSolution(pose_result)
    extractor = MediaPipeLandmarkExtractor(
        face_mesh=face_solution,
        pose=pose_solution,
    )
    frame = VideoFrame(
        frame_index=7,
        timestamp_seconds=1.25,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
    )

    result = extractor.extract(frame)

    assert result.frame_index == 7
    assert result.timestamp_seconds == 1.25
    assert result.face_detected is True
    assert result.face_available is True
    assert result.pose_detected is False
    assert result.pose_available is False
    assert result.face_landmarks == (
        LandmarkPoint(x=0.1, y=0.2, z=0.3),
        LandmarkPoint(x=0.4, y=0.5, z=0.6),
    )
    assert result.pose_landmarks is None
    assert result.missing_reason == "pose_not_detected"


def test_extractor_returns_pose_landmarks_when_detected() -> None:
    face_result = SimpleNamespace(multi_face_landmarks=[])
    pose_result = SimpleNamespace(
        pose_landmarks=_landmark_list(
            _landmark(0.1, 0.2, visibility=0.9, presence=0.8),
        )
    )
    extractor = MediaPipeLandmarkExtractor(
        face_mesh=FakeSolution(face_result),
        pose=FakeSolution(pose_result),
    )
    frame = VideoFrame(
        frame_index=1,
        timestamp_seconds=0.1,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
    )

    result = extractor.extract(frame)

    assert result.face_detected is False
    assert result.pose_detected is True
    assert result.face_landmarks is None
    assert result.pose_landmarks == (
        LandmarkPoint(
            x=0.1,
            y=0.2,
            z=0.0,
            visibility=0.9,
            presence=0.8,
        ),
    )
    assert result.missing_reason == "face_not_detected"


def test_extractor_handles_missing_face_and_pose_without_crashing() -> None:
    extractor = MediaPipeLandmarkExtractor(
        face_mesh=FakeSolution(SimpleNamespace(multi_face_landmarks=None)),
        pose=FakeSolution(SimpleNamespace(pose_landmarks=None)),
    )
    frame = VideoFrame(
        frame_index=2,
        timestamp_seconds=0.2,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
    )

    result = extractor.extract(frame)

    assert result.face_detected is False
    assert result.pose_detected is False
    assert result.face_landmarks is None
    assert result.pose_landmarks is None
    assert result.missing_reason == "face_not_detected;pose_not_detected"


def test_extractor_converts_bgr_frame_to_rgb_for_mediapipe() -> None:
    face_solution = FakeSolution(SimpleNamespace(multi_face_landmarks=None))
    pose_solution = FakeSolution(SimpleNamespace(pose_landmarks=None))
    extractor = MediaPipeLandmarkExtractor(
        face_mesh=face_solution,
        pose=pose_solution,
    )
    image = np.array([[[1, 2, 3]]], dtype=np.uint8)
    frame = VideoFrame(frame_index=0, timestamp_seconds=0.0, image=image)

    extractor.extract(frame)

    assert face_solution.processed_images[0].tolist() == [[[3, 2, 1]]]
    assert pose_solution.processed_images[0].tolist() == [[[3, 2, 1]]]


def test_extractor_closes_owned_solutions() -> None:
    face_solution = FakeSolution(SimpleNamespace(multi_face_landmarks=None))
    pose_solution = FakeSolution(SimpleNamespace(pose_landmarks=None))
    extractor = MediaPipeLandmarkExtractor(
        face_mesh=face_solution,
        pose=pose_solution,
    )

    extractor.close()

    assert face_solution.closed is True
    assert pose_solution.closed is True
