"""Command-line interface for the cognitive state MVP."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from cognitive_state import __version__
from cognitive_state.data.dataset import WindowDataset
from cognitive_state.data.exceptions import LabelError, ValidationError, WindowShapeError
from cognitive_state.data.feature_csv import read_feature_csv, write_feature_csv
from cognitive_state.data.score_csv import write_score_csv
from cognitive_state.data.synthetic_labels import (
    SYNTHETIC_LABEL_WARNING,
    generate_synthetic_labels,
)
from cognitive_state.data.windowing import WINDOW_FEATURE_COLUMNS, build_windows
from cognitive_state.features.export import (
    FeatureExportError,
    FeatureExportSummary,
    extract_features_from_video,
    iter_feature_rows_from_video,
    iter_feature_rows_from_webcam,
)
from cognitive_state.inference import (
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    format_score_table,
    load_model,
    run_inference,
)
from cognitive_state.training.checkpoints import save_checkpoint
from cognitive_state.training.metrics import SMOKE_METRIC_WARNING
from cognitive_state.training.smoke_train import run_smoke_train
from cognitive_state.video import VideoSourceError

_ASSUME_FPS: float = 30.0
_WEBCAM_DEFAULT_WINDOWS: int = 10


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

    # ------------------------------------------------------------------
    # infer subcommand
    # ------------------------------------------------------------------
    infer_parser = subparsers.add_parser(
        "infer",
        help="Estimate cognitive state scores from a video file, webcam, or feature CSV.",
    )
    source_group = infer_parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--video",
        metavar="PATH",
        help="Path to a recorded RGB video file.",
    )
    source_group.add_argument(
        "--webcam",
        type=int,
        metavar="INDEX",
        help="Webcam device index (e.g. 0 for the default camera).",
    )
    infer_parser.add_argument(
        "--features-csv",
        dest="features_csv",
        metavar="PATH",
        help=(
            "Feature CSV path. Used as the input source when --video and "
            "--webcam are not provided; used as an output path to save "
            "extracted features when --video or --webcam is the source."
        ),
    )
    infer_parser.add_argument(
        "--checkpoint",
        metavar="PATH",
        help="Path to a model checkpoint (state_dict). Random weights if omitted.",
    )
    infer_parser.add_argument(
        "--scores-csv",
        dest="scores_csv",
        metavar="PATH",
        help="Optional path to save per-window score predictions as CSV.",
    )
    infer_parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a score-over-time plot after inference (requires --plot-out).",
    )
    infer_parser.add_argument(
        "--plot-out",
        dest="plot_out",
        metavar="PATH",
        help="Path to save the score-over-time plot as a PNG image.",
    )
    infer_parser.add_argument(
        "--window-seconds",
        dest="window_seconds",
        type=float,
        default=None,
        metavar="N",
        help="Temporal window duration in seconds (default: 30 frames).",
    )
    infer_parser.add_argument(
        "--stride-seconds",
        dest="stride_seconds",
        type=float,
        default=None,
        metavar="N",
        help="Window stride in seconds (default: 15 frames).",
    )
    infer_parser.add_argument(
        "--max-windows",
        dest="max_windows",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of windows to process.",
    )

    # ------------------------------------------------------------------
    # smoke-train subcommand
    # ------------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "smoke-train",
        help="Run a minimal smoke-training flow to validate data and model contracts.",
    )
    train_parser.add_argument(
        "--features-csv",
        dest="features_csv",
        required=True,
        metavar="PATH",
        help="Feature CSV produced by extract-features.",
    )
    label_group = train_parser.add_mutually_exclusive_group()
    label_group.add_argument(
        "--synthetic-labels",
        dest="synthetic_labels",
        action="store_true",
        help="Generate synthetic/demo labels for smoke-training (not scientifically valid).",
    )
    label_group.add_argument(
        "--labels-csv",
        dest="labels_csv",
        metavar="PATH",
        help="[reserved] Load real or prepared window-level labels from CSV.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="Number of smoke-training epochs (default: 1).",
    )
    train_parser.add_argument(
        "--checkpoint-out",
        dest="checkpoint_out",
        metavar="PATH",
        help="Optional path to save the smoke-training checkpoint.",
    )

    # ------------------------------------------------------------------
    # plot-scores subcommand
    # ------------------------------------------------------------------
    plot_parser = subparsers.add_parser(
        "plot-scores",
        help="Generate a score-over-time plot from a previously saved scores CSV.",
    )
    plot_parser.add_argument(
        "--scores-csv",
        dest="scores_csv",
        required=True,
        metavar="PATH",
        help="Scores CSV produced by `cognitive-state infer --scores-csv`.",
    )
    plot_parser.add_argument(
        "--output",
        dest="output",
        required=True,
        metavar="PATH",
        help="Output PNG path for the score-over-time plot.",
    )
    plot_parser.add_argument(
        "--title",
        dest="title",
        default="Cognitive State Scores Over Time",
        metavar="TEXT",
        help="Plot title (default: 'Cognitive State Scores Over Time').",
    )

    # ------------------------------------------------------------------
    # extract-features subcommand
    # ------------------------------------------------------------------
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
    if args.command == "infer":
        return _run_infer(args)
    if args.command == "smoke-train":
        return _run_smoke_train(args)
    if args.command == "extract-features":
        return _run_extract_features(args)
    if args.command == "plot-scores":
        return _run_plot_scores(args)
    return 0


# ------------------------------------------------------------------
# infer helpers
# ------------------------------------------------------------------

def _run_infer(args: argparse.Namespace) -> int:
    has_video = bool(args.video)
    has_webcam = args.webcam is not None
    has_features_csv = bool(args.features_csv)

    if not has_video and not has_webcam and not has_features_csv:
        print(
            "error: provide --video, --webcam, or --features-csv as the input source.",
            file=sys.stderr,
        )
        return 2

    if args.plot and not getattr(args, "plot_out", None):
        print(
            "warning: --plot requires --plot-out PATH; no plot will be generated.",
            file=sys.stderr,
        )

    if not args.checkpoint:
        print(
            "warning: no --checkpoint provided; using random model weights.",
            file=sys.stderr,
        )

    # --- load feature rows ---
    rows: list[dict[str, float | int]] = []
    try:
        if has_video:
            rows = list(iter_feature_rows_from_video(args.video))
        elif has_webcam:
            max_frames = _webcam_max_frames(args)
            rows = list(iter_feature_rows_from_webcam(args.webcam, max_frames=max_frames))
        else:
            rows = read_feature_csv(args.features_csv)
    except (FeatureExportError, VideoSourceError) as exc:
        print(f"source: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"file: {exc}", file=sys.stderr)
        return 2

    if not rows:
        print("error: no feature rows found.", file=sys.stderr)
        return 2

    # optionally save extracted features when video/webcam was the source
    if (has_video or has_webcam) and has_features_csv:
        try:
            write_feature_csv(rows, Path(args.features_csv))
        except OSError as exc:
            print(f"features-csv: {exc}", file=sys.stderr)
            return 2

    # --- compute window / stride sizes ---
    fps = _estimate_fps(rows)
    window_size = (
        max(1, round(args.window_seconds * fps))
        if args.window_seconds is not None
        else DEFAULT_WINDOW_SIZE
    )
    stride = (
        max(1, round(args.stride_seconds * fps))
        if args.stride_seconds is not None
        else DEFAULT_STRIDE
    )

    # trim rows to honour --max-windows
    if args.max_windows is not None:
        max_rows = (args.max_windows - 1) * stride + window_size
        rows = rows[:max_rows]

    # --- load model ---
    try:
        model = load_model(
            input_features=len(WINDOW_FEATURE_COLUMNS),
            checkpoint_path=args.checkpoint if args.checkpoint else None,
        )
    except (OSError, RuntimeError) as exc:
        print(f"checkpoint: {exc}", file=sys.stderr)
        return 2

    # --- run inference ---
    try:
        predictions = run_inference(rows, model, window_size=window_size, stride=stride)
    except (WindowShapeError, ValidationError) as exc:
        print(f"inference: {exc}", file=sys.stderr)
        return 2

    print(format_score_table(predictions))

    # --- optional scores CSV ---
    if args.scores_csv:
        try:
            write_score_csv(predictions, Path(args.scores_csv))
        except OSError as exc:
            print(f"scores-csv: {exc}", file=sys.stderr)
            return 2

    # --- optional plot ---
    if args.plot and getattr(args, "plot_out", None):
        try:
            from cognitive_state.inference.plotting import plot_scores
            saved = plot_scores(predictions, args.plot_out)
            print(f"plot saved to {saved}")
        except Exception as exc:
            print(f"plot: {exc}", file=sys.stderr)
            return 2

    return 0


def _webcam_max_frames(args: argparse.Namespace) -> int:
    """Compute max frames to read from webcam before inference."""
    fps = _ASSUME_FPS
    ws = (
        max(1, round(args.window_seconds * fps))
        if args.window_seconds is not None
        else DEFAULT_WINDOW_SIZE
    )
    st = (
        max(1, round(args.stride_seconds * fps))
        if args.stride_seconds is not None
        else DEFAULT_STRIDE
    )
    n = args.max_windows if args.max_windows is not None else _WEBCAM_DEFAULT_WINDOWS
    return (n - 1) * st + ws


def _estimate_fps(rows: list[dict[str, float | int]]) -> float:
    if len(rows) < 2:
        return _ASSUME_FPS
    t0 = float(rows[0].get("timestamp_seconds", 0.0))
    t1 = float(rows[-1].get("timestamp_seconds", 0.0))
    duration = t1 - t0
    if duration <= 0.0:
        return _ASSUME_FPS
    return (len(rows) - 1) / duration


# ------------------------------------------------------------------
# smoke-train helpers
# ------------------------------------------------------------------

def _run_smoke_train(args: argparse.Namespace) -> int:
    if not args.synthetic_labels and not args.labels_csv:
        print(
            "error: provide --synthetic-labels or --labels-csv.",
            file=sys.stderr,
        )
        return 2

    # --- load feature rows ---
    try:
        rows = read_feature_csv(args.features_csv)
    except OSError as exc:
        print(f"features-csv: {exc}", file=sys.stderr)
        return 2

    if not rows:
        print("error: no feature rows found in CSV.", file=sys.stderr)
        return 2

    # --- build windows ---
    try:
        batch = build_windows(
            rows,
            window_size=DEFAULT_WINDOW_SIZE,
            stride=DEFAULT_STRIDE,
        )
    except (WindowShapeError, ValidationError) as exc:
        print(f"windows: {exc}", file=sys.stderr)
        return 2

    n_windows = batch.n_windows

    # --- labels ---
    if args.synthetic_labels:
        print(f"warning: {SYNTHETIC_LABEL_WARNING}", file=sys.stderr)
        labels = generate_synthetic_labels(n_windows, seed=0)
        label_source = "synthetic"
    else:
        print("error: --labels-csv is not yet supported.", file=sys.stderr)
        return 2

    try:
        dataset = WindowDataset(batch.features, labels, label_source=label_source)
    except (LabelError, ValidationError) as exc:
        print(f"dataset: {exc}", file=sys.stderr)
        return 2

    # --- run smoke training with real backpropagation ---
    result = run_smoke_train(dataset, epochs=args.epochs)

    print("smoke-train complete")
    print(f"  samples:      {result.n_samples}")
    print(f"  input shape:  {result.input_shape}")
    print(f"  output shape: {result.output_shape}")
    print(f"  final loss:   {result.final_loss:.6f}")
    print(
        "  MAE   (fatigue/attention/stress/engagement): "
        + "  ".join(f"{v:.4f}" for v in result.mae.tolist())
    )
    print(
        "  RMSE  (fatigue/attention/stress/engagement): "
        + "  ".join(f"{v:.4f}" for v in result.rmse.tolist())
    )
    print(f"note: {SMOKE_METRIC_WARNING}")

    # --- optional checkpoint ---
    if args.checkpoint_out and result.model is not None:
        try:
            saved = save_checkpoint(
                result.model,
                args.checkpoint_out,
                feature_dim=len(WINDOW_FEATURE_COLUMNS),
                label_source=label_source,
                window_size=DEFAULT_WINDOW_SIZE,
            )
            print(f"checkpoint saved to {saved}")
        except OSError as exc:
            print(f"checkpoint-out: {exc}", file=sys.stderr)
            return 2

    return 0


# ------------------------------------------------------------------
# plot-scores helpers
# ------------------------------------------------------------------

def _run_plot_scores(args: argparse.Namespace) -> int:
    from cognitive_state.data.score_csv import read_score_csv
    from cognitive_state.data.schemas import CognitiveScoreVector, ModelPrediction
    from cognitive_state.inference.plotting import plot_scores

    scores_path = Path(args.scores_csv)
    if not scores_path.exists():
        print(f"scores-csv: file not found: {scores_path}", file=sys.stderr)
        return 2

    try:
        rows = read_score_csv(scores_path)
    except Exception as exc:
        print(f"scores-csv: {exc}", file=sys.stderr)
        return 2

    predictions = [
        ModelPrediction(
            window_id=f"w{i}",
            timestamp_seconds=float(row["timestamp_seconds"]),
            scores=CognitiveScoreVector(
                fatigue=float(row["fatigue"]),
                attention=float(row["attention"]),
                stress=float(row["stress"]),
                engagement=float(row["engagement"]),
            ),
            source_progress=str(row.get("source_progress", "")),
        )
        for i, row in enumerate(rows)
    ]

    try:
        saved = plot_scores(predictions, args.output, title=args.title)
    except Exception as exc:
        print(f"plot: {exc}", file=sys.stderr)
        return 2

    print(f"plot saved to {saved}")
    return 0


# ------------------------------------------------------------------
# extract-features helpers
# ------------------------------------------------------------------

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
