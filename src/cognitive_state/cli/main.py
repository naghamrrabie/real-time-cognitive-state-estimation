"""Command-line interface for the cognitive state MVP scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from cognitive_state import __version__
from cognitive_state.features.export import (
    FeatureExportError,
    FeatureExportSummary,
    extract_features_from_video,
)
from cognitive_state.video import VideoSourceError


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        prog="cognitive-state",
        description="Scaffold CLI for cognitive state estimation.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser(
        "extract-features",
        help="Extract per-frame landmark-derived features from a video file.",
    )
    extract_parser.add_argument(
        "--video",
        required=True,
        help="Path to a recorded RGB video file.",
    )
    extract_parser.add_argument(
        "--output",
        "--features-csv",
        dest="output",
        required=True,
        help="CSV output path for per-frame features.",
    )
    extract_parser.add_argument(
        "--max-frames",
        "--max-windows",
        dest="max_frames",
        type=int,
        default=None,
        help="Optional frame limit for smoke runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "extract-features":
        return _run_extract_features(args)
    return 0


def _run_extract_features(args: argparse.Namespace) -> int:
    try:
        summary = extract_features_from_video(
            Path(args.video),
            Path(args.output),
            max_frames=args.max_frames,
        )
    except (FeatureExportError, VideoSourceError) as exc:
        print(f"video: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"output: {exc}", file=sys.stderr)
        return 2

    _print_extract_summary(summary)
    return 0


def _print_extract_summary(summary: FeatureExportSummary) -> None:
    print(
        "Extracted "
        f"{summary.frames_processed} frames to {summary.output_csv} "
        f"(rows: {summary.rows_written}, "
        f"face detections: {summary.face_detected_frames}, "
        f"pose detections: {summary.pose_detected_frames})"
    )


if __name__ == "__main__":
    raise SystemExit(main())
