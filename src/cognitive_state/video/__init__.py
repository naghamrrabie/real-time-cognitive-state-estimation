"""Video input scaffold package."""
"""Video input abstractions for webcam and video-file frame reading."""

from cognitive_state.video.readers import (
    VideoFrame,
    describe_open_source,
    iter_video_file,
    iter_webcam,
    read_frames,
)
from cognitive_state.video.sources import (
    SourceMode,
    SourceType,
    VideoOpenError,
    VideoSource,
    VideoSourceError,
)
from cognitive_state.video.timing import (
    duration_from_frame_count,
    elapsed_timestamp_seconds,
    format_source_progress,
    normalize_fps,
    timestamp_from_frame_index,
    timestamp_from_position,
)

__all__ = [
    "SourceMode",
    "SourceType",
    "VideoFrame",
    "VideoOpenError",
    "VideoSource",
    "VideoSourceError",
    "describe_open_source",
    "duration_from_frame_count",
    "elapsed_timestamp_seconds",
    "format_source_progress",
    "iter_video_file",
    "iter_webcam",
    "normalize_fps",
    "read_frames",
    "timestamp_from_frame_index",
    "timestamp_from_position",
]
