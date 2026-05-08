"""Video source metadata and source-level errors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path


class VideoSourceError(RuntimeError):
    """Base error for video source failures."""


class VideoOpenError(VideoSourceError):
    """Raised when a video file or webcam cannot be opened."""


class SourceType(str, Enum):
    """Supported video source kinds."""

    VIDEO_FILE = "video_file"
    WEBCAM = "webcam"


class SourceMode(str, Enum):
    """Supported source timing modes."""

    OFFLINE_BATCH = "offline_batch"
    LIVE_PACED = "live_paced"


@dataclass(frozen=True, slots=True)
class VideoSource:
    """Metadata for a configured or opened video source."""

    source_id: str
    source_type: SourceType
    uri: str | int
    mode: SourceMode
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    opened: bool = False
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        source_type = SourceType(self.source_type)
        mode = SourceMode(self.mode)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "mode", mode)

        if source_type is SourceType.WEBCAM and mode is not SourceMode.LIVE_PACED:
            raise ValueError("Webcam sources must use live_paced mode")
        if source_type is SourceType.VIDEO_FILE and mode is not SourceMode.OFFLINE_BATCH:
            raise ValueError("Video file sources must use offline_batch mode")
        if source_type is SourceType.WEBCAM and int(self.uri) < 0:
            raise ValueError("Webcam device index must be non-negative")

    @classmethod
    def video_file(cls, path: str | Path) -> "VideoSource":
        """Create metadata for a recorded video file source."""

        normalized_path = Path(path)
        return cls(
            source_id=f"video_file:{normalized_path}",
            source_type=SourceType.VIDEO_FILE,
            uri=str(normalized_path),
            mode=SourceMode.OFFLINE_BATCH,
        )

    @classmethod
    def webcam(cls, index: int = 0) -> "VideoSource":
        """Create metadata for a webcam source."""

        return cls(
            source_id=f"webcam:{index}",
            source_type=SourceType.WEBCAM,
            uri=int(index),
            mode=SourceMode.LIVE_PACED,
        )

    def with_open_metadata(
        self,
        *,
        fps: float | None,
        width: int | None,
        height: int | None,
        duration_seconds: float | None,
        warnings: tuple[str, ...] = (),
    ) -> "VideoSource":
        """Return a copy marked as opened with capture metadata attached."""

        return replace(
            self,
            fps=fps,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            opened=True,
            warnings=warnings,
        )
