"""Tests for CognitiveScoreVector: four bounded scores, validation, immutability."""

import pytest

from cognitive_state.data.constants import SCORE_MAX, SCORE_MIN, SCORE_NAMES
from cognitive_state.data.exceptions import FiniteValueError, ScoreRangeError
from cognitive_state.data.schemas import CognitiveScoreVector


def _valid(**overrides: float) -> CognitiveScoreVector:
    values: dict[str, float] = dict(fatigue=0.5, attention=0.3, stress=0.7, engagement=0.9)
    values.update(overrides)
    return CognitiveScoreVector(**values)


def test_valid_score_vector_stores_all_four_fields() -> None:
    vec = _valid()
    assert vec.fatigue == 0.5
    assert vec.attention == 0.3
    assert vec.stress == 0.7
    assert vec.engagement == 0.9


def test_score_names_constant_matches_four_canonical_names() -> None:
    assert SCORE_NAMES == ("fatigue", "attention", "stress", "engagement")


def test_all_four_fields_are_present_on_schema() -> None:
    vec = _valid()
    for name in SCORE_NAMES:
        assert hasattr(vec, name)


def test_boundary_value_zero_is_valid_for_all_scores() -> None:
    vec = CognitiveScoreVector(
        fatigue=SCORE_MIN,
        attention=SCORE_MIN,
        stress=SCORE_MIN,
        engagement=SCORE_MIN,
    )
    assert vec.fatigue == 0.0


def test_boundary_value_one_is_valid_for_all_scores() -> None:
    vec = CognitiveScoreVector(
        fatigue=SCORE_MAX,
        attention=SCORE_MAX,
        stress=SCORE_MAX,
        engagement=SCORE_MAX,
    )
    assert vec.fatigue == 1.0


def test_score_above_range_raises_score_range_error() -> None:
    with pytest.raises(ScoreRangeError, match="fatigue"):
        _valid(fatigue=1.01)


def test_score_below_range_raises_score_range_error() -> None:
    with pytest.raises(ScoreRangeError, match="stress"):
        _valid(stress=-0.01)


def test_nan_raises_finite_value_error() -> None:
    with pytest.raises(FiniteValueError, match="attention"):
        _valid(attention=float("nan"))


def test_positive_inf_raises_finite_value_error() -> None:
    with pytest.raises(FiniteValueError, match="engagement"):
        _valid(engagement=float("inf"))


def test_negative_inf_raises_finite_value_error() -> None:
    with pytest.raises(FiniteValueError, match="fatigue"):
        _valid(fatigue=float("-inf"))


def test_as_dict_returns_mapping_of_all_four_score_names() -> None:
    vec = _valid()
    result = vec.as_dict()
    assert set(result.keys()) == set(SCORE_NAMES)
    for name in SCORE_NAMES:
        assert result[name] == getattr(vec, name)


def test_cognitive_score_vector_is_immutable() -> None:
    vec = _valid()
    with pytest.raises(Exception):
        vec.fatigue = 0.99  # type: ignore[misc]
