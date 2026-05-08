"""Score bounding and console output format contract tests (T018)."""

import pytest
import torch

from cognitive_state.data.constants import SCORE_CSV_COLUMNS, SCORE_NAMES
from cognitive_state.data.schemas import CognitiveScoreVector, ModelPrediction
from cognitive_state.inference.output import format_score_line, format_score_row
from cognitive_state.inference.scoring import bound_scores, scores_to_vector


def _prediction(
    *,
    timestamp: float = 1.5,
    progress: str = "10/30",
    fatigue: float = 0.5,
    attention: float = 0.4,
    stress: float = 0.6,
    engagement: float = 0.7,
) -> ModelPrediction:
    return ModelPrediction(
        window_id="w0",
        timestamp_seconds=timestamp,
        scores=CognitiveScoreVector(
            fatigue=fatigue,
            attention=attention,
            stress=stress,
            engagement=engagement,
        ),
        source_progress=progress,
    )


# ---------------------------------------------------------------------------
# bound_scores
# ---------------------------------------------------------------------------

def test_bound_scores_clamps_value_above_one_to_one() -> None:
    raw = torch.tensor([[1.5, 0.5, 0.5, 0.5]])
    bounded = bound_scores(raw)
    assert bounded[0, 0].item() == pytest.approx(1.0)


def test_bound_scores_clamps_value_below_zero_to_zero() -> None:
    raw = torch.tensor([[-0.3, 0.5, 0.5, 0.5]])
    bounded = bound_scores(raw)
    assert bounded[0, 0].item() == pytest.approx(0.0)


def test_bound_scores_leaves_valid_values_unchanged() -> None:
    raw = torch.tensor([[0.2, 0.8, 0.0, 1.0]])
    bounded = bound_scores(raw)
    assert bounded[0].tolist() == pytest.approx([0.2, 0.8, 0.0, 1.0])


def test_bound_scores_preserves_output_shape() -> None:
    raw = torch.randn(4, 4)
    bounded = bound_scores(raw)
    assert bounded.shape == raw.shape


def test_bound_scores_all_values_within_range() -> None:
    raw = torch.randn(8, 4) * 5.0
    bounded = bound_scores(raw)
    assert bounded.min().item() >= 0.0
    assert bounded.max().item() <= 1.0


# ---------------------------------------------------------------------------
# scores_to_vector
# ---------------------------------------------------------------------------

def test_scores_to_vector_returns_cognitive_score_vector() -> None:
    bounded = torch.tensor([0.1, 0.2, 0.3, 0.4])
    vec = scores_to_vector(bounded)
    assert isinstance(vec, CognitiveScoreVector)


def test_scores_to_vector_maps_values_in_score_name_order() -> None:
    bounded = torch.tensor([0.1, 0.2, 0.3, 0.4])
    vec = scores_to_vector(bounded)
    assert vec.fatigue == pytest.approx(0.1)
    assert vec.attention == pytest.approx(0.2)
    assert vec.stress == pytest.approx(0.3)
    assert vec.engagement == pytest.approx(0.4)


def test_scores_to_vector_boundary_values_are_accepted() -> None:
    bounded = torch.tensor([0.0, 1.0, 0.0, 1.0])
    vec = scores_to_vector(bounded)
    assert vec.fatigue == pytest.approx(0.0)
    assert vec.attention == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# format_score_row
# ---------------------------------------------------------------------------

def test_format_score_row_contains_all_score_csv_columns() -> None:
    row = format_score_row(_prediction())
    for field in SCORE_CSV_COLUMNS:
        assert field in row, f"Missing field in score row: {field!r}"


def test_format_score_row_score_values_match_prediction() -> None:
    pred = _prediction(fatigue=0.3, attention=0.7, stress=0.2, engagement=0.9)
    row = format_score_row(pred)
    assert row["fatigue"] == pytest.approx(0.3)
    assert row["attention"] == pytest.approx(0.7)
    assert row["stress"] == pytest.approx(0.2)
    assert row["engagement"] == pytest.approx(0.9)


def test_format_score_row_timestamp_and_progress_match_prediction() -> None:
    pred = _prediction(timestamp=2.5, progress="15/30")
    row = format_score_row(pred)
    assert row["timestamp_seconds"] == pytest.approx(2.5)
    assert row["source_progress"] == "15/30"


def test_format_score_row_contains_no_extra_score_keys() -> None:
    row = format_score_row(_prediction())
    for name in SCORE_NAMES:
        assert name in row


# ---------------------------------------------------------------------------
# format_score_line
# ---------------------------------------------------------------------------

def test_format_score_line_returns_a_string() -> None:
    line = format_score_line(_prediction())
    assert isinstance(line, str)


def test_format_score_line_contains_all_four_score_names() -> None:
    line = format_score_line(_prediction())
    for name in SCORE_NAMES:
        assert name in line, f"Score name missing from output line: {name!r}"


def test_format_score_line_contains_timestamp() -> None:
    line = format_score_line(_prediction(timestamp=3.14))
    assert "3.14" in line


def test_format_score_line_contains_source_progress() -> None:
    line = format_score_line(_prediction(progress="5/20"))
    assert "5/20" in line
