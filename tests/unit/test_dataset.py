"""Dataset sample and label alignment tests (T045)."""

from __future__ import annotations

import numpy as np
import pytest

from cognitive_state.data.dataset import DatasetSample, WindowDataset
from cognitive_state.data.exceptions import LabelError, ValidationError
from cognitive_state.data.schemas import CognitiveScoreVector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(n: int, window_size: int = 30, feature_dim: int = 8) -> np.ndarray:
    return np.random.default_rng(42).random((n, window_size, feature_dim)).astype(np.float32)


def _uniform_label() -> CognitiveScoreVector:
    return CognitiveScoreVector(fatigue=0.5, attention=0.5, stress=0.5, engagement=0.5)


def _make_labels(n: int) -> list[CognitiveScoreVector]:
    return [_uniform_label() for _ in range(n)]


# ---------------------------------------------------------------------------
# DatasetSample — field contract
# ---------------------------------------------------------------------------

def test_dataset_sample_stores_features() -> None:
    features = np.zeros((30, 8), dtype=np.float32)
    sample = DatasetSample(window_id="w0", features=features, label=_uniform_label())
    assert sample.features is features


def test_dataset_sample_stores_label() -> None:
    label = CognitiveScoreVector(fatigue=0.1, attention=0.2, stress=0.3, engagement=0.4)
    sample = DatasetSample(window_id="w0", features=np.zeros((30, 8), dtype=np.float32), label=label)
    assert sample.label.fatigue == pytest.approx(0.1)
    assert sample.label.attention == pytest.approx(0.2)
    assert sample.label.stress == pytest.approx(0.3)
    assert sample.label.engagement == pytest.approx(0.4)


def test_dataset_sample_features_shape_is_window_size_feature_dim() -> None:
    features = np.zeros((30, 8), dtype=np.float32)
    sample = DatasetSample(window_id="w0", features=features, label=_uniform_label())
    assert sample.features.shape == (30, 8)


def test_dataset_sample_default_label_source_is_synthetic() -> None:
    sample = DatasetSample(
        window_id="w0",
        features=np.zeros((30, 8), dtype=np.float32),
        label=_uniform_label(),
    )
    assert sample.label_source == "synthetic"


def test_dataset_sample_window_id_is_preserved() -> None:
    sample = DatasetSample(
        window_id="w42",
        features=np.zeros((5, 3), dtype=np.float32),
        label=_uniform_label(),
    )
    assert sample.window_id == "w42"


# ---------------------------------------------------------------------------
# WindowDataset — length and iteration
# ---------------------------------------------------------------------------

def test_window_dataset_len_matches_window_count() -> None:
    ds = WindowDataset(_make_features(10), _make_labels(10))
    assert len(ds) == 10


def test_window_dataset_iter_yields_dataset_sample_objects() -> None:
    ds = WindowDataset(_make_features(5), _make_labels(5))
    for sample in ds:
        assert isinstance(sample, DatasetSample)


def test_window_dataset_iter_count_matches_len() -> None:
    ds = WindowDataset(_make_features(7), _make_labels(7))
    assert sum(1 for _ in ds) == len(ds)


# ---------------------------------------------------------------------------
# WindowDataset — getitem shape and values
# ---------------------------------------------------------------------------

def test_window_dataset_getitem_features_shape() -> None:
    ds = WindowDataset(_make_features(5, window_size=30, feature_dim=8), _make_labels(5))
    assert ds[0].features.shape == (30, 8)


def test_window_dataset_getitem_features_dtype_is_float32() -> None:
    ds = WindowDataset(_make_features(3), _make_labels(3))
    assert ds[0].features.dtype == np.float32


def test_window_dataset_getitem_label_matches_input() -> None:
    label = CognitiveScoreVector(fatigue=0.3, attention=0.7, stress=0.2, engagement=0.9)
    ds = WindowDataset(_make_features(3), [label] + _make_labels(2))
    assert ds[0].label.fatigue == pytest.approx(0.3)
    assert ds[0].label.engagement == pytest.approx(0.9)


def test_window_dataset_getitem_window_id_sequential_by_default() -> None:
    ds = WindowDataset(_make_features(4), _make_labels(4))
    for i in range(4):
        assert ds[i].window_id == f"w{i}"


def test_window_dataset_custom_window_ids_are_preserved() -> None:
    ids = ["win_a", "win_b", "win_c"]
    ds = WindowDataset(_make_features(3), _make_labels(3), window_ids=ids)
    for i, expected_id in enumerate(ids):
        assert ds[i].window_id == expected_id


# ---------------------------------------------------------------------------
# WindowDataset — label source
# ---------------------------------------------------------------------------

def test_window_dataset_default_label_source_is_synthetic() -> None:
    ds = WindowDataset(_make_features(3), _make_labels(3))
    assert ds[0].label_source == "synthetic"


def test_window_dataset_placeholder_label_source_is_preserved() -> None:
    ds = WindowDataset(_make_features(3), _make_labels(3), label_source="placeholder")
    assert ds[0].label_source == "placeholder"


# ---------------------------------------------------------------------------
# WindowDataset — error cases
# ---------------------------------------------------------------------------

def test_window_dataset_raises_label_error_on_count_mismatch() -> None:
    with pytest.raises(LabelError):
        WindowDataset(_make_features(5), _make_labels(3))


def test_window_dataset_raises_validation_error_on_unknown_label_source() -> None:
    with pytest.raises(ValidationError):
        WindowDataset(_make_features(3), _make_labels(3), label_source="unknown_source")
