from pathlib import Path

import cv2
import numpy as np
import pytest

from cognitive_state.video import (
    SourceMode,
    SourceType,
    VideoOpenError,
    VideoSource,
    describe_open_source,
    iter_video_file,
    iter_webcam,
)


class FakeCapture:
    def __init__(
        self,
        *,
        opened: bool = True,
        frames: list[np.ndarray] | None = None,
        props: dict[int, float] | None = None,
    ) -> None:
        self.opened = opened
        self.frames = list(frames or [])
        self.props = props or {}
        self.released = False
        self._index = 0

    def isOpened(self) -> bool:
        return self.opened

    def get(self, prop_id: int) -> float:
        return self.props.get(prop_id, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self.frames):
            return False, None
        frame = self.frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def test_video_file_source_uses_offline_mode(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    source = VideoSource.video_file(video_path)

    assert source.source_type == SourceType.VIDEO_FILE
    assert source.mode == SourceMode.OFFLINE_BATCH
    assert source.uri == str(video_path)


def test_webcam_source_uses_live_mode() -> None:
    source = VideoSource.webcam(1)

    assert source.source_type == SourceType.WEBCAM
    assert source.mode == SourceMode.LIVE_PACED
    assert source.uri == 1


def test_video_file_reader_yields_index_timestamp_and_frame(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"placeholder")
    frames = [
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.ones((2, 3, 3), dtype=np.uint8),
    ]
    capture = FakeCapture(
        frames=frames,
        props={
            cv2.CAP_PROP_FPS: 2.0,
            cv2.CAP_PROP_FRAME_WIDTH: 3.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 2.0,
            cv2.CAP_PROP_FRAME_COUNT: 2.0,
        },
    )

    records = list(iter_video_file(video_path, capture_factory=lambda _: capture))

    assert [record.frame_index for record in records] == [0, 1]
    assert [record.timestamp_seconds for record in records] == [0.0, 0.5]
    assert np.array_equal(records[1].image, frames[1])
    assert capture.released


def test_invalid_video_path_raises_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.mp4"

    with pytest.raises(VideoOpenError, match="Video file does not exist"):
        list(iter_video_file(missing_path))


def test_describe_open_source_returns_capture_metadata(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"placeholder")
    capture = FakeCapture(
        props={
            cv2.CAP_PROP_FPS: 4.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_FRAME_COUNT: 8.0,
        },
    )

    source = describe_open_source(
        VideoSource.video_file(video_path),
        capture_factory=lambda _: capture,
    )

    assert source.opened is True
    assert source.fps == 4.0
    assert source.width == 640
    assert source.height == 480
    assert source.duration_seconds == 2.0
    assert capture.released


def test_webcam_reader_yields_elapsed_live_timestamps() -> None:
    frames = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2, 3), dtype=np.uint8),
    ]
    capture = FakeCapture(opened=True, frames=frames)
    clock_values = iter([10.0, 10.25, 10.75])

    records = list(
        iter_webcam(
            0,
            capture_factory=lambda _: capture,
            clock=lambda: next(clock_values),
        )
    )

    assert [record.frame_index for record in records] == [0, 1]
    assert [record.timestamp_seconds for record in records] == [0.25, 0.75]
    assert capture.released


def test_webcam_open_failure_raises_clear_error() -> None:
    capture = FakeCapture(opened=False)

    with pytest.raises(VideoOpenError, match="webcam device 7"):
        list(iter_webcam(7, capture_factory=lambda _: capture))
