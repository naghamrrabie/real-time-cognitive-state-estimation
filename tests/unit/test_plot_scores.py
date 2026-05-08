"""Score plot generation tests (T058)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cognitive_state.data.schemas import CognitiveScoreVector, ModelPrediction
from cognitive_state.inference.plotting import plot_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_predictions(n: int = 5) -> list[ModelPrediction]:
    return [
        ModelPrediction(
            window_id=f"w{i}",
            timestamp_seconds=float(i),
            scores=CognitiveScoreVector(
                fatigue=0.2,
                attention=0.4,
                stress=0.6,
                engagement=0.8,
            ),
            source_progress=f"{i}/{n}",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# plot_scores — output file
# ---------------------------------------------------------------------------

def test_plot_scores_creates_output_file(tmp_path: Path) -> None:
    out = tmp_path / "scores.png"
    plot_scores(_make_predictions(), out)
    assert out.exists()


def test_plot_scores_returns_path(tmp_path: Path) -> None:
    out = tmp_path / "scores.png"
    result = plot_scores(_make_predictions(), out)
    assert isinstance(result, Path)


def test_plot_scores_output_path_matches_requested(tmp_path: Path) -> None:
    out = tmp_path / "scores.png"
    result = plot_scores(_make_predictions(), out)
    assert result.resolve() == out.resolve()


def test_plot_scores_creates_parent_directories(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "subdir" / "scores.png"
    plot_scores(_make_predictions(), out)
    assert out.exists()


def test_plot_scores_file_is_non_empty(tmp_path: Path) -> None:
    out = tmp_path / "scores.png"
    plot_scores(_make_predictions(), out)
    assert out.stat().st_size > 0


def test_plot_scores_accepts_str_path(tmp_path: Path) -> None:
    out = str(tmp_path / "scores.png")
    plot_scores(_make_predictions(), out)
    assert Path(out).exists()


# ---------------------------------------------------------------------------
# plot_scores — minimal predictions
# ---------------------------------------------------------------------------

def test_plot_scores_single_prediction_does_not_raise(tmp_path: Path) -> None:
    plot_scores(_make_predictions(1), tmp_path / "single.png")


def test_plot_scores_empty_predictions_does_not_raise(tmp_path: Path) -> None:
    plot_scores([], tmp_path / "empty.png")


# ---------------------------------------------------------------------------
# plot_scores — custom title
# ---------------------------------------------------------------------------

def test_plot_scores_accepts_custom_title(tmp_path: Path) -> None:
    out = tmp_path / "scores.png"
    plot_scores(_make_predictions(), out, title="My Custom Title")
    assert out.exists()
