"""Training batch shape and dtype contract tests (T047)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cognitive_state.data.dataset import DatasetSample, WindowDataset
from cognitive_state.data.schemas import CognitiveScoreVector
from cognitive_state.data.synthetic_labels import generate_synthetic_labels
from cognitive_state.training.batching import collate_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_samples(
    n: int,
    window_size: int = 30,
    feature_dim: int = 8,
    seed: int = 42,
) -> list[DatasetSample]:
    rng = np.random.default_rng(seed)
    features = rng.random((n, window_size, feature_dim)).astype(np.float32)
    labels = generate_synthetic_labels(n, seed=seed)
    return list(WindowDataset(features, labels))


# ---------------------------------------------------------------------------
# collate_batch — return type
# ---------------------------------------------------------------------------

def test_collate_batch_returns_tuple() -> None:
    result = collate_batch(_make_samples(4))
    assert isinstance(result, tuple)


def test_collate_batch_returns_two_elements() -> None:
    result = collate_batch(_make_samples(4))
    assert len(result) == 2


def test_collate_batch_x_is_tensor() -> None:
    x, _ = collate_batch(_make_samples(4))
    assert isinstance(x, torch.Tensor)


def test_collate_batch_y_is_tensor() -> None:
    _, y = collate_batch(_make_samples(4))
    assert isinstance(y, torch.Tensor)


# ---------------------------------------------------------------------------
# collate_batch — X shape
# ---------------------------------------------------------------------------

def test_collate_batch_x_batch_dim_matches_sample_count() -> None:
    x, _ = collate_batch(_make_samples(6))
    assert x.shape[0] == 6


def test_collate_batch_x_window_size_dim_matches_features() -> None:
    x, _ = collate_batch(_make_samples(4, window_size=30))
    assert x.shape[1] == 30


def test_collate_batch_x_feature_dim_matches_features() -> None:
    x, _ = collate_batch(_make_samples(4, feature_dim=12))
    assert x.shape[2] == 12


def test_collate_batch_x_shape_is_batch_window_size_feature_dim() -> None:
    x, _ = collate_batch(_make_samples(4, window_size=30, feature_dim=8))
    assert x.shape == (4, 30, 8)


# ---------------------------------------------------------------------------
# collate_batch — y shape
# ---------------------------------------------------------------------------

def test_collate_batch_y_batch_dim_matches_sample_count() -> None:
    _, y = collate_batch(_make_samples(5))
    assert y.shape[0] == 5


def test_collate_batch_y_score_dim_is_four() -> None:
    _, y = collate_batch(_make_samples(5))
    assert y.shape[1] == 4


def test_collate_batch_y_shape_is_batch_four() -> None:
    _, y = collate_batch(_make_samples(4))
    assert y.shape == (4, 4)


# ---------------------------------------------------------------------------
# collate_batch — dtypes
# ---------------------------------------------------------------------------

def test_collate_batch_x_dtype_is_float32() -> None:
    x, _ = collate_batch(_make_samples(4))
    assert x.dtype == torch.float32


def test_collate_batch_y_dtype_is_float32() -> None:
    _, y = collate_batch(_make_samples(4))
    assert y.dtype == torch.float32


# ---------------------------------------------------------------------------
# collate_batch — value correctness
# ---------------------------------------------------------------------------

def test_collate_batch_y_values_are_bounded_in_unit_range() -> None:
    _, y = collate_batch(_make_samples(8))
    assert y.min().item() >= 0.0
    assert y.max().item() <= 1.0


def test_collate_batch_x_values_match_input_features() -> None:
    features = np.ones((1, 5, 3), dtype=np.float32) * 0.7
    label = CognitiveScoreVector(fatigue=0.5, attention=0.5, stress=0.5, engagement=0.5)
    sample = DatasetSample(window_id="w0", features=features[0], label=label)
    x, _ = collate_batch([sample])
    assert x[0].numpy() == pytest.approx(features[0], abs=1e-6)


def test_collate_batch_y_values_match_label_order() -> None:
    label = CognitiveScoreVector(fatigue=0.1, attention=0.2, stress=0.3, engagement=0.4)
    sample = DatasetSample(
        window_id="w0",
        features=np.zeros((5, 3), dtype=np.float32),
        label=label,
    )
    _, y = collate_batch([sample])
    assert y[0, 0].item() == pytest.approx(0.1)
    assert y[0, 1].item() == pytest.approx(0.2)
    assert y[0, 2].item() == pytest.approx(0.3)
    assert y[0, 3].item() == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# collate_batch — edge cases
# ---------------------------------------------------------------------------

def test_collate_batch_single_sample_produces_batch_size_one() -> None:
    x, y = collate_batch(_make_samples(1))
    assert x.shape[0] == 1
    assert y.shape[0] == 1
