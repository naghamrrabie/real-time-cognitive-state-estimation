"""Tests for the domain exception hierarchy."""

import pytest

from cognitive_state.data.exceptions import (
    CognitiveStateError,
    FiniteValueError,
    LabelError,
    ModalityError,
    ScoreRangeError,
    ValidationError,
    WindowShapeError,
)


def test_all_leaf_exceptions_are_subclasses_of_cognitive_state_error() -> None:
    for exc_cls in (
        ValidationError,
        ScoreRangeError,
        FiniteValueError,
        ModalityError,
        WindowShapeError,
        LabelError,
    ):
        assert issubclass(exc_cls, CognitiveStateError)


def test_score_range_and_finite_errors_are_validation_errors() -> None:
    assert issubclass(ScoreRangeError, ValidationError)
    assert issubclass(FiniteValueError, ValidationError)


def test_modality_error_is_not_a_validation_error() -> None:
    assert not issubclass(ModalityError, ValidationError)


def test_window_shape_error_is_not_a_validation_error() -> None:
    assert not issubclass(WindowShapeError, ValidationError)


def test_each_exception_can_be_raised_and_caught() -> None:
    for exc_cls in (
        CognitiveStateError,
        ValidationError,
        ScoreRangeError,
        FiniteValueError,
        ModalityError,
        WindowShapeError,
        LabelError,
    ):
        with pytest.raises(exc_cls):
            raise exc_cls("test message")


def test_cognitive_state_error_is_caught_as_plain_exception() -> None:
    with pytest.raises(Exception):
        raise CognitiveStateError("base error")


def test_score_range_error_message_is_preserved() -> None:
    msg = "fatigue must be in [0.0, 1.0]; got 1.5"
    exc = ScoreRangeError(msg)
    assert str(exc) == msg


def test_label_error_message_is_preserved() -> None:
    msg = "sample_42: label window mismatch"
    exc = LabelError(msg)
    assert str(exc) == msg
