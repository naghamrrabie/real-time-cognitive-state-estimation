"""MediaPipe face and pose landmark extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from cognitive_state.video import VideoFrame


@dataclass(frozen=True, slots=True)
class LandmarkPoint:
    """One normalized MediaPipe landmark point."""

    x: float
    y: float
    z: float
    visibility: float | None = None
    presence: float | None = None


@dataclass(frozen=True, slots=True)
class LandmarkSet:
    """Face and pose landmark observations for one video frame."""

    frame_index: int
    timestamp_seconds: float
    face_landmarks: tuple[LandmarkPoint, ...] | None
    pose_landmarks: tuple[LandmarkPoint, ...] | None
    face_detected: bool
    pose_detected: bool
    missing_reason: str | None = None

    @property
    def face_available(self) -> bool:
        """Whether face landmarks are available for this frame."""

        return self.face_detected

    @property
    def pose_available(self) -> bool:
        """Whether pose landmarks are available for this frame."""

        return self.pose_detected


class MediaPipeLandmarkExtractor:
    """Extract MediaPipe Face Mesh and Pose landmarks from video frames."""

    def __init__(
        self,
        *,
        face_mesh: Any | None = None,
        pose: Any | None = None,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._face_mesh = face_mesh
        self._pose = pose

        if self._face_mesh is None or self._pose is None:
            import mediapipe as mp

            if self._face_mesh is None:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=max_num_faces,
                    refine_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
            if self._pose is None:
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )

    def extract(self, frame: VideoFrame) -> LandmarkSet:
        """Extract face and pose landmarks from one OpenCV BGR frame."""

        rgb_image = _bgr_to_rgb(frame.image)
        face_results = self._face_mesh.process(rgb_image)
        pose_results = self._pose.process(rgb_image)

        face_landmarks = _extract_first_face_landmarks(face_results)
        pose_landmarks = _extract_pose_landmarks(pose_results)
        missing_reasons = []
        if face_landmarks is None:
            missing_reasons.append("face_not_detected")
        if pose_landmarks is None:
            missing_reasons.append("pose_not_detected")

        return LandmarkSet(
            frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds,
            face_landmarks=face_landmarks,
            pose_landmarks=pose_landmarks,
            face_detected=face_landmarks is not None,
            pose_detected=pose_landmarks is not None,
            missing_reason=";".join(missing_reasons) or None,
        )

    def close(self) -> None:
        """Close configured MediaPipe resources."""

        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._pose is not None:
            self._pose.close()

    def __enter__(self) -> "MediaPipeLandmarkExtractor":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("MediaPipe landmark extraction requires a 3-channel frame")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _extract_first_face_landmarks(results: Any) -> tuple[LandmarkPoint, ...] | None:
    face_results = getattr(results, "multi_face_landmarks", None)
    if not face_results:
        return None
    return _convert_landmarks(face_results[0])


def _extract_pose_landmarks(results: Any) -> tuple[LandmarkPoint, ...] | None:
    pose_landmarks = getattr(results, "pose_landmarks", None)
    if pose_landmarks is None:
        return None
    return _convert_landmarks(pose_landmarks)


def _convert_landmarks(landmark_list: Any) -> tuple[LandmarkPoint, ...]:
    return tuple(
        LandmarkPoint(
            x=float(point.x),
            y=float(point.y),
            z=float(getattr(point, "z", 0.0)),
            visibility=_optional_float(getattr(point, "visibility", None)),
            presence=_optional_float(getattr(point, "presence", None)),
        )
        for point in getattr(landmark_list, "landmark", ())
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
