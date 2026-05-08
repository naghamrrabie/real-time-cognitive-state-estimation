"""Unit tests for live webcam preview inference."""

from __future__ import annotations

import numpy as np

from cognitive_state.data.windowing import WINDOW_FEATURE_COLUMNS
from cognitive_state.features.landmarks import LandmarkSet
from cognitive_state.inference.live_preview import run_webcam_preview
from cognitive_state.inference.pipeline import load_model
from cognitive_state.video import VideoFrame


class FakeExtractor:
    def __init__(self) -> None:
        self.closed = False

    def extract(self, frame: VideoFrame) -> LandmarkSet:
        return LandmarkSet(
            frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds,
            face_landmarks=None,
            pose_landmarks=None,
            face_detected=False,
            pose_detected=False,
        )

    def close(self) -> None:
        self.closed = True


def _fake_frames(_index: int):
    for i in range(3):
        yield VideoFrame(
            frame_index=i,
            timestamp_seconds=i / 30.0,
            image=np.zeros((32, 32, 3), dtype=np.uint8),
        )


def test_run_webcam_preview_returns_predictions_without_real_window() -> None:
    shown_frames = []
    extractor = FakeExtractor()
    model = load_model(input_features=len(WINDOW_FEATURE_COLUMNS))

    def fake_show(window_name: str, frame: np.ndarray) -> int:
        shown_frames.append((window_name, frame.shape))
        return -1

    result = run_webcam_preview(
        0,
        model,
        window_size=2,
        stride=1,
        target_windows=1,
        frame_reader=_fake_frames,
        extractor_factory=lambda: extractor,
        show_frame=fake_show,
        close_windows=lambda: None,
    )

    assert len(result.rows) == 2
    assert len(result.predictions) == 1
    assert result.predictions[0].source_progress == "1/1"
    assert shown_frames
    assert extractor.closed
