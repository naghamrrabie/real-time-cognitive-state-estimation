"""Synthetic label generation tests (T046)."""

from __future__ import annotations

import pytest

from cognitive_state.data.schemas import CognitiveScoreVector
from cognitive_state.data.synthetic_labels import (
    SYNTHETIC_LABEL_SOURCE,
    SYNTHETIC_LABEL_WARNING,
    generate_synthetic_labels,
)


# ---------------------------------------------------------------------------
# generate_synthetic_labels — count and type
# ---------------------------------------------------------------------------

def test_generate_synthetic_labels_returns_requested_count() -> None:
    assert len(generate_synthetic_labels(10)) == 10


def test_generate_synthetic_labels_returns_cognitive_score_vectors() -> None:
    for label in generate_synthetic_labels(5, seed=0):
        assert isinstance(label, CognitiveScoreVector)


def test_generate_synthetic_labels_zero_count_returns_empty_list() -> None:
    assert generate_synthetic_labels(0) == []


# ---------------------------------------------------------------------------
# generate_synthetic_labels — value bounds
# ---------------------------------------------------------------------------

def test_generate_synthetic_labels_fatigue_in_unit_range() -> None:
    for label in generate_synthetic_labels(20, seed=1):
        assert 0.0 <= label.fatigue <= 1.0


def test_generate_synthetic_labels_attention_in_unit_range() -> None:
    for label in generate_synthetic_labels(20, seed=2):
        assert 0.0 <= label.attention <= 1.0


def test_generate_synthetic_labels_stress_in_unit_range() -> None:
    for label in generate_synthetic_labels(20, seed=3):
        assert 0.0 <= label.stress <= 1.0


def test_generate_synthetic_labels_engagement_in_unit_range() -> None:
    for label in generate_synthetic_labels(20, seed=4):
        assert 0.0 <= label.engagement <= 1.0


# ---------------------------------------------------------------------------
# generate_synthetic_labels — reproducibility
# ---------------------------------------------------------------------------

def test_generate_synthetic_labels_same_seed_is_deterministic() -> None:
    a = generate_synthetic_labels(8, seed=42)
    b = generate_synthetic_labels(8, seed=42)
    for la, lb in zip(a, b):
        assert la.fatigue == pytest.approx(lb.fatigue)
        assert la.attention == pytest.approx(lb.attention)
        assert la.stress == pytest.approx(lb.stress)
        assert la.engagement == pytest.approx(lb.engagement)


def test_generate_synthetic_labels_different_seeds_produce_different_values() -> None:
    a = generate_synthetic_labels(8, seed=0)
    b = generate_synthetic_labels(8, seed=99)
    any_differ = any(
        la.fatigue != lb.fatigue or la.stress != lb.stress
        for la, lb in zip(a, b)
    )
    assert any_differ


def test_generate_synthetic_labels_different_counts_share_prefix_with_same_seed() -> None:
    short = generate_synthetic_labels(3, seed=7)
    long_ = generate_synthetic_labels(10, seed=7)
    for i in range(3):
        assert short[i].fatigue == pytest.approx(long_[i].fatigue)


# ---------------------------------------------------------------------------
# generate_synthetic_labels — all four score names present
# ---------------------------------------------------------------------------

def test_generate_synthetic_labels_label_has_fatigue() -> None:
    label = generate_synthetic_labels(1, seed=0)[0]
    assert hasattr(label, "fatigue")


def test_generate_synthetic_labels_label_has_attention() -> None:
    label = generate_synthetic_labels(1, seed=0)[0]
    assert hasattr(label, "attention")


def test_generate_synthetic_labels_label_has_stress() -> None:
    label = generate_synthetic_labels(1, seed=0)[0]
    assert hasattr(label, "stress")


def test_generate_synthetic_labels_label_has_engagement() -> None:
    label = generate_synthetic_labels(1, seed=0)[0]
    assert hasattr(label, "engagement")


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

def test_synthetic_label_source_constant_equals_synthetic() -> None:
    assert SYNTHETIC_LABEL_SOURCE == "synthetic"


def test_synthetic_label_warning_is_non_empty_string() -> None:
    assert isinstance(SYNTHETIC_LABEL_WARNING, str)
    assert len(SYNTHETIC_LABEL_WARNING) > 0


def test_synthetic_label_warning_references_validation_or_pipeline() -> None:
    lower = SYNTHETIC_LABEL_WARNING.lower()
    assert "validation" in lower or "pipeline" in lower
